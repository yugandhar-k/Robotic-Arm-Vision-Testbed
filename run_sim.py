"""
run_sim.py - Interactive simulation of a tabletop claw with an eye-in-hand camera.

Research framing
----------------
The standard workcell does object detection from a camera bolted above the workspace.
That camera is occluded by the arm at exactly the moment the arm reaches in, it needs a
hand-eye calibration that drifts, and it offers one viewpoint chosen by where it happens
to be mounted.

This module is the testbed for the alternative: put the camera on the claw, so the robot
sees what it picks, from the approach direction, every time. A 6-DoF arm with a
two-finger claw, a field of objects on a work table, and an RGB camera mounted on the
claw itself, looking down the approach axis.

The trade is that the observed viewpoint becomes a function of the robot's own
configuration, so scale, angle and occlusion all vary as the arm moves. Measuring that
cost - and generating the data to train against it - is what this environment is for.

The arm runs a continuous inspect-and-pick cycle: hover over an object, hold still while
the classification frame is taken, then close the claw, lift, and set it back down. The
frame is captured *before* the grasp, because that is the point at which a real system
has to commit to a decision.

Object colour is randomised independently of object class, so a downstream classifier
cannot shortcut the task by learning colour. Shape is the only class-carrying signal.

Interactive controls
--------------------
  Left drag          orbit          Right drag     pan
  Scroll             zoom           Shift + drag   rotate / pan horizontally
  SPACE              pause / resume the cycle
  G                  toggle pick-and-place  <->  inspect only
  M                  toggle AUTO  <->  MANUAL (arm holds still, claw under O)
  O                  open / close the claw by hand (MANUAL mode)
  R                  reshuffle the object layout
  C                  save the current claw-camera frame, auto-labelled
  + / -              speed the cycle up or down
  ESC                quit

The panel in the top-right corner is the claw camera's own view, rendered live.

Usage
-----
  python run_sim.py                  interactive viewer
  python run_sim.py --no-grasp       inspect only, do not pick objects up
  python run_sim.py --capture        also save a labelled frame at every inspection
  python run_sim.py --ui             MuJoCo's stock UI, with a slider per joint
  python run_sim.py --dataset 500    headless: generate 500 paired eye-in-hand /
                                      fixed-overhead frames
  python run_sim.py --selftest       headless check of IK, labelling and grasping

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import glfw
import mujoco
import mujoco.viewer
import numpy as np

# ======================================================================================
# Constants - no magic numbers below this block
# ======================================================================================

RANDOM_SEED: int = 42

TABLE_TOP_Z: float = 0.62           # height of the working surface above the floor [m]
TABLE_HALF_X: float = 0.46          # table depth  / 2 [m]
TABLE_HALF_Y: float = 0.52          # table width  / 2 [m]
TABLE_TOP_HALF_THICK: float = 0.02
TABLE_CENTER_X: float = 0.20

ROBOT_BASE_XY: Tuple[float, float] = (-0.16, 0.0)   # mount point on the table top

# Object field - a jittered grid inside the arm's comfortable workspace.
FIELD_X: Tuple[float, ...] = (0.20, 0.33, 0.46)
FIELD_Y: Tuple[float, ...] = (-0.20, 0.20)
FIELD_JITTER: float = 0.030
N_OBJECTS: int = len(FIELD_X) * len(FIELD_Y)

# Two-finger claw. All offsets are in the tool frame, whose +z points out of the
# flange (i.e. straight down at the table while the arm is inspecting).
PALM_Z: float = 0.028               # centre of the palm block [m]
FINGER_BASE_Z: float = 0.044        # where the pads start [m]
FINGER_HALF_LENGTH: float = 0.034   # pads are twice this long [m]
FINGER_HALF_THICK: float = 0.008
FINGER_HOME_X: float = 0.050        # lateral offset of a fully open finger [m]
FINGER_TRAVEL: float = 0.040        # closing stroke per finger [m]
GRASP_SITE_Z: float = 0.100         # grasp frame, 12 mm above the fingertips [m]
FINGERTIP_Z: float = FINGER_BASE_Z + 2.0 * FINGER_HALF_LENGTH

OPEN_GAP: float = 0.080             # commanded gap when the claw is open [m]
GRASP_SQUEEZE: float = 0.009        # close this much past the object width [m]
MIN_GRASP_GAP: float = 0.010
OBJECT_GEOM_GROUP: int = 1          # objects live in their own group so rays can
                                    # ignore the robot's own geometry
# (sliding, torsional, rolling). The default MuJoCo rolling friction is small enough
# that a nudged sphere or cylinder keeps rolling well past the object field and, over
# enough reshuffles, off the table edge entirely - confirmed by scanning a generated
# dataset for objects resting at floor height (z far below the table top). Raising
# rolling and torsional friction an order of magnitude stops that without visibly
# changing how the objects sit or how the claw grips them.
OBJECT_FRICTION: Tuple[float, float, float] = (1.0, 0.08, 0.02)

# Claw-mounted camera.
CAM_IMAGE_SIZE: int = 224           # square; matches the CNN input used downstream
CAM_FOV_DEG: float = 72.0
CAM_MOUNT_Y: float = -0.036         # lateral mount offset on the palm [m]
CAM_MOUNT_Z: float = 0.044          # just past the palm, so it is never occluded

# Scan behaviour. Heights are of the GRASP FRAME above the table surface.
HOVER_HEIGHT: float = 0.15          # inspection height of the grasp frame [m]
LIFT_HEIGHT: float = 0.20           # how high a grasped object is raised [m]

# The camera sits to one side of the palm, so its optical axis is tilted back towards
# the tool axis. Aim it at the top of a typical object at inspection height, which is
# what centres the object in frame - measured empirically by sweeping the tilt and
# tracking the segmented object's centroid.
NOMINAL_OBJECT_TOP: float = 0.055   # representative object height above the table [m]
CAM_TILT_RAD: float = math.atan2(
    -CAM_MOUNT_Y,
    HOVER_HEIGHT + (GRASP_SITE_Z - CAM_MOUNT_Z) - NOMINAL_OBJECT_TOP,
)

# Fixed-overhead camera: the conventional baseline the eye-in-hand design is measured
# against. Bolted above the workspace, pointed straight down, never moves. Centred over
# the object field (not the table) so every slot sits comfortably inside the frame.
OVERHEAD_CAM_X: float = sum(FIELD_X) / len(FIELD_X)
OVERHEAD_CAM_HEIGHT: float = 1.00           # above the table surface [m]
OVERHEAD_CAM_FOV_DEG: float = 48.0

# Fixed-angled camera: a second, equally fixed baseline, bolted to the side and tilted
# down at exactly 45 degrees rather than straight down - a more typical mounting for a
# real workspace camera (a corner-mounted security camera, not a ceiling-mounted one),
# and a second, independent test of whether "fixed" is the disadvantage or "straight
# down" is. Height above the table and horizontal pull-back from the object field are
# set equal, which by definition of elevation angle gives exactly 45 degrees looking at
# the field centroid.
ANGLED_CAM_X: float = OVERHEAD_CAM_X
ANGLED_CAM_HEIGHT: float = 0.75             # above the table surface [m]
ANGLED_CAM_SETBACK: float = ANGLED_CAM_HEIGHT   # equal to height => 45 deg elevation
ANGLED_CAM_FOV_DEG: float = 55.0            # wider than overhead: the oblique view
                                            # foreshortens depth, needing more margin

# Collection basket for the search-and-pick demo (see search_and_pick.py). Its own
# position is fixed, known infrastructure - the same way a real deployment would know
# where its own collection bin is bolted down - so that demo's state machine is allowed
# to target it directly; only the *objects'* positions are ever discovered by search.
#
# The classification benchmark's fixed cameras crop a window nearly as wide as the
# object field itself (see FIXED_CAMERA_CROP_HALF_SIDE), so no on-table position clears
# every crop by a comfortable margin - moving the basket around the table only trades
# which slot's crop it grazes. Instead the basket lives in its own MuJoCo geom group,
# invisible by default (`SimCamera` and `PictureInPictureViewer` both use a default
# `MjvOption`, which hides this group), so it never appears in a classification-benchmark
# frame or figure regardless of where it sits or whether the dataset is regenerated.
# `search_and_pick.py` explicitly re-enables the group on its own viewer so a human
# watching that demo can still see it.
BASKET_GEOM_GROUP: int = 3          # hidden by MjvOption's own default geomgroup
BASKET_WORLD_X: float = 0.55
BASKET_WORLD_Y: float = 0.42
BASKET_INNER_HALF: float = 0.060    # inner half-width of the square collection area [m]
BASKET_WALL_THICK: float = 0.006
BASKET_WALL_HEIGHT: float = 0.050
BASKET_FLOOR_HALF_THICK: float = 0.004
BASKET_DROP_HEIGHT: float = 0.09    # grasp-frame height above the table when releasing -
                                    # just enough to clear the basket walls (see
                                    # BASKET_WALL_HEIGHT); releasing from higher was
                                    # measured to let a dropped object bounce off an
                                    # already-full basket hard enough to leave the table
                                    # entirely (see search_and_pick.py's Limitations)

# Duration of each phase of the inspect / pick cycle, in seconds.
PICK_SEQUENCE: Tuple[Tuple[str, float], ...] = (
    ("approach", 1.7),      # traverse to the hover pose, claw open and yaw aligned
    ("settle", 0.85),       # hold still: this is where the classification frame is taken
    ("descend", 0.8),       # lower the open claw around the object
    ("close", 0.6),         # squeeze
    ("lift", 0.9),          # raise the object clear of the table
    ("hold", 0.8),          # show the grasped object off in the claw-camera panel
    ("lower", 0.9),         # put it back
    ("release", 0.5),       # open
    ("retreat", 0.6),       # withdraw to the hover pose
)
INSPECT_SEQUENCE: Tuple[Tuple[str, float], ...] = (
    ("approach", 1.7),
    ("settle", 0.85),
    ("hold", 1.1),
)

# Inverse kinematics (analytic seed + damped least squares).
IK_DAMPING: float = 0.06
IK_ITERATIONS: int = 120             # iterations per multi-start seed
IK_WARM_ITERATIONS: int = 14         # iterations when warm-starting from the last solve
IK_STEP_LIMIT: float = 0.15          # max joint delta per IK iteration [rad]
IK_TOLERANCE: float = 4e-4           # task-space error treated as converged; well
                                     # below the servo tracking error, so tightening
                                     # it further would only cost time
IK_ACCEPT_TOLERANCE: float = 2e-3    # warm start good enough to skip the multi-start
ORIENTATION_WEIGHT: float = 0.45     # rad-to-metre weighting in the task-space error

SIM_TIMESTEP: float = 0.002
CONTROL_DECIMATION: int = 8         # control update every N physics steps (~60 Hz)

PROJECT_ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = PROJECT_ROOT / "data" / "captures"
EYE_IN_HAND_DIR = CAPTURE_DIR / "eye_in_hand"
FIXED_OVERHEAD_DIR = CAPTURE_DIR / "fixed_overhead"
FIXED_ANGLED_DIR = CAPTURE_DIR / "fixed_angled"

# ======================================================================================
# Object taxonomy - the classification target
# ======================================================================================


@dataclass(frozen=True)
class ObjectClass:
    """One entry of the classification taxonomy."""

    class_id: int
    name: str
    geom_type: int
    size: Tuple[float, float, float]    # MuJoCo geom size semantics, per type
    rest_height: float                  # centre-of-geom height above the table [m]


OBJECT_CLASSES: Tuple[ObjectClass, ...] = (
    ObjectClass(0, "cube", int(mujoco.mjtGeom.mjGEOM_BOX),
                (0.030, 0.030, 0.030), 0.030),
    ObjectClass(1, "sphere", int(mujoco.mjtGeom.mjGEOM_SPHERE),
                (0.033, 0.033, 0.033), 0.033),
    ObjectClass(2, "cylinder", int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                (0.028, 0.040, 0.040), 0.040),
    ObjectClass(3, "cuboid", int(mujoco.mjtGeom.mjGEOM_BOX),
                (0.060, 0.026, 0.020), 0.020),
    # The four below exist to make the task hard enough to be worth studying: each is
    # a genuine near-miss for one of the four above under a bounding-box or silhouette
    # shortcut - a capsule reads as "cylinder with soft edges", an ellipsoid as "stretched
    # sphere", a disk as "cube from directly above", a rod as "cuboid seen edge-on". A
    # detector that only learns aspect ratio or convex-hull shape should struggle to
    # separate these pairs; one that has actually learned silhouette curvature and
    # proportion should not.
    ObjectClass(4, "capsule", int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                (0.026, 0.035, 0.000), 0.026),
    ObjectClass(5, "ellipsoid", int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
                (0.026, 0.026, 0.042), 0.026),
    ObjectClass(6, "disk", int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                (0.032, 0.010, 0.000), 0.010),
    ObjectClass(7, "rod", int(mujoco.mjtGeom.mjGEOM_BOX),
                (0.065, 0.012, 0.012), 0.012),
)

# Class ids whose natural rest pose (long axis horizontal) is *not* reliably reached by
# just dropping them with gravity. A flat-ended shape (box, or the original cylinder,
# whose ends are flat discs) is unstable standing on end and tips over on its own from
# the smallest numerical asymmetry. A *rounded*-ended shape resting on a single contact
# point does not have that instability to exploit - and the rolling friction raised in
# `OBJECT_FRICTION` to stop spheres/cylinders rolling off the table (see Data Generation
# in the README) damps out what little rocking there is before it can tip over, so a
# capsule or ellipsoid dropped standing up simply stays standing up. Confirmed by
# checking their settled orientation directly: both sat with their long axis exactly
# vertical across every seed tested. Spawning them already on their side sidesteps the
# instability question entirely rather than depending on it.
LYING_ON_SIDE_CLASS_IDS: frozenset = frozenset({4, 5})    # capsule, ellipsoid

# Colours are sampled independently of class on purpose (see module docstring).
PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.86, 0.22, 0.22),
    (0.20, 0.45, 0.88),
    (0.96, 0.74, 0.16),
    (0.24, 0.72, 0.36),
    (0.66, 0.36, 0.82),
    (0.18, 0.76, 0.78),
)


@dataclass
class SceneObject:
    """A spawned object together with its ground-truth label and MuJoCo handles."""

    slot: int
    object_class: ObjectClass
    colour: Tuple[float, float, float]
    body_id: int = -1
    geom_id: int = -1
    qpos_adr: int = -1
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _half_extents(object_class: ObjectClass) -> Tuple[float, float, float]:
    """Half-size of an object along its three body axes, in metres.

    MuJoCo's `geom_size` means different things per primitive, so this normalises them
    into a common form the grasp planner can reason about.
    """
    a, b, c = object_class.size
    if object_class.geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return object_class.size
    if object_class.geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return (a, a, a)
    if object_class.geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return (a, a, b)                    # radius, radius, half-length
    if object_class.geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return (a, a, a + b)                # radius, radius, half-length + rounded cap
    if object_class.geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return (a, b, c)                    # three semi-axes directly
    raise ValueError(f"unhandled geom type {object_class.geom_type}")


def _spawn_quaternion(yaw: float, lying_on_side: bool) -> Tuple[float, float, float, float]:
    """World-orientation quaternion (w, x, y, z) for a freshly-spawned object.

    `yaw` alone (rotation about world +z) is enough for anything that naturally rests on
    a flat face - a box, or a flat-ended cylinder, both settle correctly from an upright
    drop. `lying_on_side` additionally rotates the object's local z-axis (its long axis)
    down onto the table before the yaw is applied, for the rounded-ended classes that do
    not reliably tip over on their own - see `LYING_ON_SIDE_CLASS_IDS`.
    """
    cos_yaw, sin_yaw = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    if not lying_on_side:
        return (cos_yaw, 0.0, 0.0, sin_yaw)
    # Quaternion product q_yaw * q_tilt, where q_tilt is a 90 deg rotation about local x
    # (bringing local +z to horizontal) applied before the yaw spin about world z.
    half_root2 = math.sqrt(0.5)
    return (cos_yaw * half_root2, cos_yaw * half_root2,
           sin_yaw * half_root2, sin_yaw * half_root2)


def finger_position(claw_gap: float) -> float:
    """Convert a desired pad-to-pad opening into the finger slide position.

    Each finger travels inwards from `FINGER_HOME_X`, so the gap between the two pad
    faces is `2 * (FINGER_HOME_X - q - FINGER_HALF_THICK)`. Inverting that and clamping
    to the mechanical stroke gives the position command.
    """
    position = FINGER_HOME_X - FINGER_HALF_THICK - 0.5 * claw_gap
    return float(np.clip(position, 0.0, FINGER_TRAVEL))


def grasp_gap(object_width: float) -> float:
    """Commanded opening that squeezes an object of the given width."""
    return max(MIN_GRASP_GAP, object_width - GRASP_SQUEEZE)


# Both fixed baselines crop one window size for every class, not a size fitted to
# whichever object is there - a real fixed camera has no way to know the class before
# classifying it. The window is set once, from the taxonomy's largest object, so smaller
# classes are genuinely disadvantaged by under-filling it - exactly the "object scale
# dictated by where the camera is bolted" cost these baselines are meant to measure, not
# an artefact of a class-conditioned crop. Sized from the full 3-D bounding radius, not
# just the horizontal footprint, so it stays correct for the angled camera too, which
# (unlike a straight-down view) can foreshorten any of an object's three dimensions
# depending on its resting orientation.
FIXED_CAMERA_CROP_MARGIN: float = 1.3
FIXED_CAMERA_CROP_HALF_SIDE: float = 0.5 * FIXED_CAMERA_CROP_MARGIN * max(
    2.0 * math.sqrt(sum(h * h for h in half))
    for half in (_half_extents(object_class) for object_class in OBJECT_CLASSES)
)


# ======================================================================================
# Scene description (MJCF)
# ======================================================================================


def build_mjcf() -> str:
    """Return the MJCF description of floor, table, 6-DoF arm and object slots.

    The object slots are declared generically; their geom type, size, colour and pose are
    assigned at run time so the layout can be reshuffled without rebuilding the model.
    """
    object_slots = "\n".join(
        f"""
    <body name="object_{i}" pos="{0.30 + 0.001 * i} 0 {TABLE_TOP_Z + 0.05}">
      <freejoint name="object_{i}_free"/>
      <geom name="object_{i}_geom" type="box" size="0.03 0.03 0.03"
            group="{OBJECT_GEOM_GROUP}"
            rgba="0.7 0.7 0.7 1" density="450"
            friction="{OBJECT_FRICTION[0]} {OBJECT_FRICTION[1]} {OBJECT_FRICTION[2]}"
            solref="0.006 1"/>
    </body>"""
        for i in range(N_OBJECTS)
    )

    leg_x, leg_y = TABLE_HALF_X - 0.07, TABLE_HALF_Y - 0.07
    leg_half_h = (TABLE_TOP_Z - 2 * TABLE_TOP_HALF_THICK) / 2.0
    legs = "\n".join(
        f"""      <geom name="table_leg_{i}" type="box" material="steel"
            pos="{sx * leg_x} {sy * leg_y} {leg_half_h - TABLE_TOP_Z}"
            size="0.025 0.025 {leg_half_h}"/>"""
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    )

    return f"""
<mujoco model="tabletop_eye_in_hand">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{SIM_TIMESTEP}" integrator="implicitfast" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.28 0.28 0.30" diffuse="0.38 0.38 0.38" specular="0.08 0.08 0.08"/>
    <rgba haze="0.15 0.20 0.26 1"/>
    <global azimuth="135" elevation="-24" offwidth="1024" offheight="1024"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.01" zfar="30"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.26 0.34 0.45"
             rgb2="0.05 0.06 0.09" width="512" height="1024"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.18 0.21"
             rgb2="0.21 0.23 0.27" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="8 8" texuniform="true"
              reflectance="0.08"/>
    <material name="tabletop" rgba="0.46 0.42 0.37 1" specular="0.15" shininess="0.15"/>
    <material name="steel" rgba="0.42 0.44 0.48 1" specular="0.6" shininess="0.5"/>
    <material name="link" rgba="0.80 0.81 0.84 1" specular="0.5" shininess="0.4"/>
    <material name="actuator" rgba="0.15 0.16 0.18 1" specular="0.4" shininess="0.4"/>
    <material name="accent" rgba="0.95 0.45 0.10 1" specular="0.4" shininess="0.4"/>
    <material name="lens" rgba="0.08 0.09 0.12 1" specular="0.9" shininess="0.9"/>
  </asset>

  <default>
    <geom solref="0.005 1"/>
    <default class="link">
      <geom type="capsule" material="link" density="700" contype="1" conaffinity="0"/>
    </default>
    <default class="arm_joint">
      <joint type="hinge" damping="8" armature="0.12" frictionloss="0.2"/>
      <position kp="900" kv="90" forcerange="-200 200"/>
    </default>
    <default class="finger">
      <joint type="slide" range="0 {FINGER_TRAVEL}" damping="12" armature="0.008"/>
      <position kp="900" kv="35" forcerange="-90 90" ctrlrange="0 {FINGER_TRAVEL}"/>
    </default>
    <default class="finger_pad">
      <geom type="box" material="accent" density="900" condim="4"
            friction="1.8 0.10 0.002" solref="0.008 1" solimp="0.95 0.99 0.001"/>
    </default>
  </default>

  <worldbody>
    <light name="key" pos="0.6 -0.8 2.4" dir="-0.2 0.3 -1" directional="true"
           diffuse="0.42 0.42 0.42" specular="0.15 0.15 0.15" castshadow="true"/>
    <light name="fill" pos="-1.0 1.0 2.0" dir="0.4 -0.4 -1" directional="true"
           diffuse="0.25 0.25 0.28" castshadow="false"/>
    <geom name="floor" type="plane" size="6 6 0.05" material="floor"/>

    <body name="table" pos="{TABLE_CENTER_X} 0 {TABLE_TOP_Z}">
      <geom name="table_top" type="box" material="tabletop"
            pos="0 0 {-TABLE_TOP_HALF_THICK}"
            size="{TABLE_HALF_X} {TABLE_HALF_Y} {TABLE_TOP_HALF_THICK}"/>
{legs}
    </body>

    <!-- ============ 6-DoF manipulator, base-mounted on the table top ============ -->
    <body name="arm_base" pos="{ROBOT_BASE_XY[0]} {ROBOT_BASE_XY[1]} {TABLE_TOP_Z}">
      <geom name="pedestal" type="cylinder" material="actuator" size="0.075 0.030"
            pos="0 0 0.030"/>
      <geom name="pedestal_ring" type="cylinder" material="accent" size="0.078 0.006"
            pos="0 0 0.062"/>

      <body name="shoulder_yaw" pos="0 0 0.072">
        <joint name="j1_base_yaw" class="arm_joint" axis="0 0 1" range="-3.05 3.05"/>
        <geom name="shoulder_housing" class="link" type="cylinder" size="0.058 0.045"
              pos="0 0 0.045" material="link"/>

        <body name="shoulder_pitch" pos="0 0 0.098">
          <joint name="j2_shoulder" class="arm_joint" axis="0 1 0" range="-2.10 2.10"/>
          <geom name="shoulder_joint" class="link" type="cylinder" size="0.048 0.052"
                quat="0.7071 0.7071 0 0" material="actuator"/>
          <geom name="upper_arm" class="link" size="0.038" fromto="0 0 0 0 0 0.400"/>

          <body name="elbow" pos="0 0 0.400">
            <joint name="j3_elbow" class="arm_joint" axis="0 1 0" range="-2.60 2.60"/>
            <geom name="elbow_joint" class="link" type="cylinder" size="0.042 0.046"
                  quat="0.7071 0.7071 0 0" material="actuator"/>
            <geom name="forearm" class="link" size="0.032" fromto="0 0 0 0 0 0.360"/>

            <body name="wrist_roll" pos="0 0 0.360">
              <joint name="j4_wrist_roll" class="arm_joint" axis="0 0 1"
                     range="-3.05 3.05"/>
              <geom name="wrist_roll_geom" class="link" type="cylinder"
                    size="0.030 0.038" pos="0 0 0.038"/>

              <body name="wrist_pitch" pos="0 0 0.078">
                <joint name="j5_wrist_pitch" class="arm_joint" axis="0 1 0"
                       range="-2.10 2.10"/>
                <geom name="wrist_pitch_geom" class="link" type="cylinder"
                      size="0.030 0.034" quat="0.7071 0.7071 0 0" material="actuator"/>

                <body name="tool" pos="0 0 0.062">
                  <joint name="j6_tool_yaw" class="arm_joint" axis="0 0 1"
                         range="-3.20 3.20"/>
                  <geom name="tool_flange" class="link" type="cylinder"
                        size="0.034 0.010" pos="0 0 0.006"/>
                  <geom name="claw_palm" type="box" material="actuator"
                        size="0.052 0.020 0.012" pos="0 0 {PALM_Z}"/>
                  <geom name="claw_rail" type="box" material="accent"
                        size="0.052 0.005 0.004" pos="0 0.022 {PALM_Z}"
                        contype="0" conaffinity="0"/>

                  <!-- Claw-mounted camera: sits on the side of the palm and looks
                       down the approach axis, so the grasp point stays centred and
                       both fingers frame the object. MuJoCo cameras look along their
                       own -z, so the 180 deg roll about x aims it along the tool +z;
                       the small extra tilt compensates the lateral mount offset. -->
                  <geom name="camera_bracket" type="box" material="actuator"
                        size="0.012 0.014 0.014"
                        pos="0 {CAM_MOUNT_Y + 0.004} {CAM_MOUNT_Z - 0.020}"
                        contype="0" conaffinity="0"/>
                  <geom name="camera_lens" type="cylinder" material="lens"
                        size="0.009 0.003"
                        pos="0 {CAM_MOUNT_Y} {CAM_MOUNT_Z - 0.004}"
                        contype="0" conaffinity="0"/>
                  <camera name="claw" pos="0 {CAM_MOUNT_Y} {CAM_MOUNT_Z}"
                          euler="{math.pi - CAM_TILT_RAD} 0 0" fovy="{CAM_FOV_DEG}"/>

                  <!-- Grasp frame: the point the IK drives, mid-way along the pads. -->
                  <site name="tool_site" pos="0 0 {GRASP_SITE_Z}" size="0.006"
                        rgba="1 0.3 0.1 0"/>

                  <body name="finger_left" pos="{FINGER_HOME_X} 0 {FINGER_BASE_Z}">
                    <joint name="finger_left" class="finger" axis="-1 0 0"/>
                    <geom name="finger_left_geom" class="finger_pad"
                          size="{FINGER_HALF_THICK} 0.018 {FINGER_HALF_LENGTH}"
                          pos="0 0 {FINGER_HALF_LENGTH}"/>
                  </body>
                  <body name="finger_right" pos="{-FINGER_HOME_X} 0 {FINGER_BASE_Z}">
                    <joint name="finger_right" class="finger" axis="1 0 0"/>
                    <geom name="finger_right_geom" class="finger_pad"
                          size="{FINGER_HALF_THICK} 0.018 {FINGER_HALF_LENGTH}"
                          pos="0 0 {FINGER_HALF_LENGTH}"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ============ object field ============ -->
{object_slots}

    <camera name="overview" pos="1.45 -1.25 1.75" xyaxes="0.66 0.75 0 -0.36 0.32 0.88"/>
    <camera name="front" pos="1.70 0 1.20" xyaxes="0 1 0 -0.55 0 0.84"/>

    <!-- Fixed-overhead baseline camera: bolted above the workspace, pointed straight
         down, never moves. This is the conventional setup the eye-in-hand design is
         benchmarked against. -->
    <camera name="overhead" pos="{OVERHEAD_CAM_X} 0 {TABLE_TOP_Z + OVERHEAD_CAM_HEIGHT}"
            xyaxes="1 0 0 0 1 0" fovy="{OVERHEAD_CAM_FOV_DEG}"/>

    <!-- Fixed-angled baseline camera: bolted to the side, tilted down at exactly
         45 degrees at the object field's centroid, never moves. xyaxes keeps the image
         x-axis aligned with world x (matching "overhead"'s convention, so left/right in
         both fixed cameras' frames means the same thing); the y-axis is derived so the
         camera looks at (ANGLED_CAM_X, 0, TABLE_TOP_Z) from
         (ANGLED_CAM_X, -ANGLED_CAM_SETBACK, TABLE_TOP_Z + ANGLED_CAM_HEIGHT). -->
    <camera name="angled"
            pos="{ANGLED_CAM_X} {-ANGLED_CAM_SETBACK} {TABLE_TOP_Z + ANGLED_CAM_HEIGHT}"
            xyaxes="1 0 0 0 {math.sqrt(0.5)} {math.sqrt(0.5)}"
            fovy="{ANGLED_CAM_FOV_DEG}"/>

    <!-- Collection basket for the search-and-pick demo: an open-top box bolted to a
         fixed, known spot on the table. Lives in its own geom group, hidden from every
         renderer by default (see BASKET_GEOM_GROUP), so it never appears in a
         classification-benchmark frame or figure. -->
    <body name="basket" pos="{BASKET_WORLD_X} {BASKET_WORLD_Y} {TABLE_TOP_Z}">
      <geom name="basket_floor" type="box" material="accent" group="{BASKET_GEOM_GROUP}"
            pos="0 0 {BASKET_FLOOR_HALF_THICK}"
            size="{BASKET_INNER_HALF + BASKET_WALL_THICK}
                  {BASKET_INNER_HALF + BASKET_WALL_THICK} {BASKET_FLOOR_HALF_THICK}"/>
      <geom name="basket_wall_neg_x" type="box" material="accent"
            group="{BASKET_GEOM_GROUP}"
            pos="{-(BASKET_INNER_HALF + BASKET_WALL_THICK)} 0 {BASKET_WALL_HEIGHT / 2}"
            size="{BASKET_WALL_THICK} {BASKET_INNER_HALF + BASKET_WALL_THICK}
                  {BASKET_WALL_HEIGHT / 2}"/>
      <geom name="basket_wall_pos_x" type="box" material="accent"
            group="{BASKET_GEOM_GROUP}"
            pos="{BASKET_INNER_HALF + BASKET_WALL_THICK} 0 {BASKET_WALL_HEIGHT / 2}"
            size="{BASKET_WALL_THICK} {BASKET_INNER_HALF + BASKET_WALL_THICK}
                  {BASKET_WALL_HEIGHT / 2}"/>
      <geom name="basket_wall_neg_y" type="box" material="accent"
            group="{BASKET_GEOM_GROUP}"
            pos="0 {-(BASKET_INNER_HALF + BASKET_WALL_THICK)} {BASKET_WALL_HEIGHT / 2}"
            size="{BASKET_INNER_HALF} {BASKET_WALL_THICK} {BASKET_WALL_HEIGHT / 2}"/>
      <geom name="basket_wall_pos_y" type="box" material="accent"
            group="{BASKET_GEOM_GROUP}"
            pos="0 {BASKET_INNER_HALF + BASKET_WALL_THICK} {BASKET_WALL_HEIGHT / 2}"
            size="{BASKET_INNER_HALF} {BASKET_WALL_THICK} {BASKET_WALL_HEIGHT / 2}"/>
    </body>
  </worldbody>

  <actuator>
    <position class="arm_joint" name="a1" joint="j1_base_yaw"/>
    <position class="arm_joint" name="a2" joint="j2_shoulder"/>
    <position class="arm_joint" name="a3" joint="j3_elbow"/>
    <position class="arm_joint" name="a4" joint="j4_wrist_roll"/>
    <position class="arm_joint" name="a5" joint="j5_wrist_pitch"/>
    <position class="arm_joint" name="a6" joint="j6_tool_yaw"/>
    <position class="finger" name="claw_left" joint="finger_left"/>
    <position class="finger" name="claw_right" joint="finger_right"/>
  </actuator>
</mujoco>
"""


# ======================================================================================
# Kinematics
# ======================================================================================

# Desired tool orientation: optical axis (tool +z) pointing straight down at the table.
TOOL_DOWN_ROTATION: np.ndarray = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def tool_rotation(yaw: float = 0.0) -> np.ndarray:
    """Tool orientation pointing straight down, yawed by `yaw` about the world z axis.

    The claw closes along the tool x axis, so this yaw is what aligns the fingers
    across an object's narrow dimension.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    spin = np.array([[cos_yaw, -sin_yaw, 0.0],
                     [sin_yaw, cos_yaw, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
    return spin @ TOOL_DOWN_ROTATION


def _wrap(angle: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class ArmKinematics:
    """Damped-least-squares inverse kinematics for the 6-DoF arm."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.joint_ids: List[int] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("j1_base_yaw", "j2_shoulder", "j3_elbow",
                         "j4_wrist_roll", "j5_wrist_pitch", "j6_tool_yaw")
        ]
        self.dof_ids: np.ndarray = np.array(
            [model.jnt_dofadr[j] for j in self.joint_ids], dtype=np.int32
        )
        self.qpos_ids: np.ndarray = np.array(
            [model.jnt_qposadr[j] for j in self.joint_ids], dtype=np.int32
        )
        self.lower: np.ndarray = model.jnt_range[self.joint_ids, 0].copy()
        self.upper: np.ndarray = model.jnt_range[self.joint_ids, 1].copy()
        self.site_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "tool_site"
        )
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        # Scratch state so IK never perturbs the live simulation.
        self._scratch = mujoco.MjData(model)
        self._last_solution: Optional[np.ndarray] = None

        # Measure the planar link lengths from the model itself, at the zero
        # configuration where the arm stands straight up, so the closed-form seed stays
        # correct if the MJCF link geometry is ever retuned.
        probe = mujoco.MjData(model)
        probe.qpos[self.qpos_ids] = 0.0
        mujoco.mj_kinematics(model, probe)
        shoulder, elbow, wrist = (probe.xanchor[self.joint_ids[i]].copy()
                                  for i in (1, 2, 4))
        self.shoulder_origin: np.ndarray = shoulder
        self.link_upper: float = float(np.linalg.norm(elbow - shoulder))
        self.link_lower: float = float(np.linalg.norm(wrist - elbow))
        self.wrist_to_tool: float = float(
            np.linalg.norm(probe.site_xpos[self.site_id] - wrist)
        )

    def tool_pose(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """World position and 3x3 rotation matrix of the tool frame."""
        position = np.asarray(data.site_xpos[self.site_id], dtype=np.float64).copy()
        rotation = np.asarray(
            data.site_xmat[self.site_id], dtype=np.float64
        ).reshape(3, 3).copy()
        return position, rotation

    @staticmethod
    def _orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        """Axis-angle error that rotates `current` onto `desired`.

        Uses the standard sum-of-cross-products approximation, which is well behaved for
        the small per-iteration corrections produced by DLS.
        """
        return 0.5 * sum(
            np.cross(current[:, axis], desired[:, axis]) for axis in range(3)
        )

    def analytic_seeds(self, target_position: np.ndarray,
                       target_yaw: float = 0.0) -> List[np.ndarray]:
        """Closed-form starting configurations for a downward-pointing tool.

        With the optical axis constrained to vertical, the arm reduces to a planar 2R
        chain (shoulder, elbow) plus a wrist pitch that absorbs the remaining angle:

            j1                 = atan2(dy, dx)                  base yaw towards target
            j2 + j3            = planar 2R solution to the wrist point
            j2 + j3 + j5       = +/- pi                         tool axis points down

        Both elbow branches and both wrist branches are returned; the caller polishes
        each with DLS and keeps whichever lands closest. Seeding this way is what keeps
        the numerical solver out of the joint-limit local minima that a naive
        current-pose seed falls into.
        """
        delta = np.asarray(target_position, dtype=np.float64) - self.shoulder_origin
        # The wrist-pitch joint must sit WRIST_TO_TOOL above the tool target.
        wrist_point = delta + np.array([0.0, 0.0, self.wrist_to_tool])

        base_yaw = math.atan2(wrist_point[1], wrist_point[0])
        radial = math.hypot(wrist_point[0], wrist_point[1])
        vertical = wrist_point[2]

        reach_sq = radial ** 2 + vertical ** 2
        cos_elbow = (reach_sq - self.link_upper ** 2 - self.link_lower ** 2) / (
            2.0 * self.link_upper * self.link_lower
        )
        cos_elbow = float(np.clip(cos_elbow, -1.0, 1.0))

        seeds: List[np.ndarray] = []
        for elbow_sign in (1.0, -1.0):
            elbow = elbow_sign * math.acos(cos_elbow)
            shoulder = math.atan2(radial, vertical) - math.atan2(
                self.link_lower * math.sin(elbow),
                self.link_upper + self.link_lower * math.cos(elbow),
            )
            for tool_axis_angle in (math.pi, -math.pi):
                wrist_pitch = tool_axis_angle - shoulder - elbow
                # The tool yaw has to undo the base yaw and add the requested one; the
                # branch offset depends on which way the wrist flipped over.
                for tool_yaw in (_wrap(target_yaw - base_yaw + tool_axis_angle),
                                 _wrap(target_yaw - base_yaw)):
                    seeds.append(np.clip(
                        np.array([base_yaw, shoulder, elbow, 0.0, wrist_pitch, tool_yaw]),
                        self.lower, self.upper,
                    ))
        return seeds

    def _polish(
        self,
        initial_angles: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        iterations: int = IK_ITERATIONS,
    ) -> Tuple[np.ndarray, float]:
        """Damped-least-squares refinement from one starting configuration.

        Returns:
            (joint angles, final task-space error norm)
        """
        scratch = self._scratch
        angles = np.clip(initial_angles, self.lower, self.upper)
        scratch.qpos[self.qpos_ids] = angles
        mujoco.mj_kinematics(self.model, scratch)
        mujoco.mj_comPos(self.model, scratch)

        error_norm = float("inf")
        for _ in range(iterations):
            position, rotation = self.tool_pose(scratch)
            error = np.concatenate([
                target_position - position,
                ORIENTATION_WEIGHT * self._orientation_error(rotation, target_rotation),
            ])
            error_norm = float(np.linalg.norm(error))
            if error_norm < IK_TOLERANCE:
                break

            mujoco.mj_jacSite(self.model, scratch, self._jacp, self._jacr, self.site_id)
            jacobian = np.vstack([
                self._jacp[:, self.dof_ids],
                ORIENTATION_WEIGHT * self._jacr[:, self.dof_ids],
            ])
            hessian = jacobian @ jacobian.T + (IK_DAMPING ** 2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(hessian, error)
            delta = np.clip(delta, -IK_STEP_LIMIT, IK_STEP_LIMIT)

            angles = np.clip(angles + delta, self.lower, self.upper)
            scratch.qpos[self.qpos_ids] = angles
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)

        return angles, error_norm

    def solve(
        self,
        data: mujoco.MjData,
        target_position: np.ndarray,
        target_yaw: float = 0.0,
    ) -> np.ndarray:
        """Solve for the joint angles that place the tool at the given pose.

        Strategy: warm-start from the previous solution (cheap, and the scan target moves
        smoothly), and fall back to a multi-start over the closed-form seeds plus the
        current pose whenever the warm start fails to converge. The solve runs entirely
        on scratch state, so it never disturbs the live simulation.
        """
        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = tool_rotation(target_yaw)
        scratch = self._scratch
        scratch.qpos[:] = data.qpos

        if self._last_solution is not None:
            angles, error = self._polish(self._last_solution, target_position,
                                         target_rotation, iterations=IK_WARM_ITERATIONS)
            if error < IK_ACCEPT_TOLERANCE:
                self._last_solution = angles
                return angles
            # The short warm-start budget didn't quite converge. Give the *same* branch
            # the full iteration budget before considering a different one - most
            # control-tick-to-control-tick target changes just needed a few more DLS
            # steps, not a different elbow/wrist configuration, and jumping branches
            # for a target that moved only slightly is a discontinuous, physically fast
            # arm motion a continuous target never asked for.
            angles, error = self._polish(self._last_solution, target_position,
                                         target_rotation)
            if error < IK_ACCEPT_TOLERANCE:
                self._last_solution = angles
                return angles

        # Continuity-preserving candidates first (current pose, then the warm start
        # again alongside the scratch state it was polished from) so that whenever one
        # of them is capable of reaching the target at all, it wins over an analytic
        # seed that reaches a *different*, equally-valid but discontinuous branch.
        candidates = [np.asarray(data.qpos[self.qpos_ids], dtype=np.float64).copy()]
        if self._last_solution is not None:
            candidates.append(self._last_solution)
        candidates.extend(self.analytic_seeds(target_position, target_yaw))

        best_angles, best_error = None, float("inf")
        for seed in candidates:
            angles, error = self._polish(seed, target_position, target_rotation)
            if error < best_error:
                best_angles, best_error = angles, error
            if best_error < IK_TOLERANCE:
                break

        assert best_angles is not None
        self._last_solution = best_angles
        return best_angles


# ======================================================================================
# Scene management
# ======================================================================================


def inspection_target_for(position: np.ndarray, hover: float = HOVER_HEIGHT,
                          yaw: float = 0.0) -> np.ndarray:
    """Grasp-frame target that puts a point at world `position` on the camera's optical
    axis while hovering `hover` above the table, yawed by `yaw`.

    The camera is mounted to one side of the palm and tilted back, so its axis only
    crosses the tool axis at one particular depth. Hovering straight above a point
    therefore centres it at exactly one height. Solving for the lateral offset instead
    keeps it centred at *any* inspection height and for objects of any height - which is
    what makes both the varied-viewpoint dataset and the vision-driven search-and-pick
    demo (`search_and_pick.py`) usable. Takes a raw position rather than an object index
    so it works equally for a ground-truth object pose and a vision-estimated one.

    The offset is undone before the claw descends, so grasping is unaffected.
    """
    tool_z = TABLE_TOP_Z + hover
    depth = (tool_z - position[2]) + (GRASP_SITE_Z - CAM_MOUNT_Z)
    lateral = CAM_MOUNT_Y + depth * math.tan(CAM_TILT_RAD)
    tool_y_axis = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    return np.array([position[0], position[1], tool_z]) - lateral * tool_y_axis


class Workspace:
    """Owns the MuJoCo model/data pair and the randomised object layout."""

    HOME_POSE: Tuple[float, ...] = (0.0, 0.70, -1.35, 0.0, 0.65, 0.0)

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.model = mujoco.MjModel.from_xml_string(build_mjcf())
        self.data = mujoco.MjData(self.model)
        self.kinematics = ArmKinematics(self.model)
        self.objects: List[SceneObject] = []
        self.seed = seed

        self.arm_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a{i}")
            for i in range(1, 7)
        ], dtype=np.int32)
        self.finger_actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("claw_left", "claw_right")
        ], dtype=np.int32)
        self.finger_qpos_ids = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in ("finger_left", "finger_right")
        ], dtype=np.int32)

        self.reshuffle(seed)

    # -- layout ------------------------------------------------------------------
    def reshuffle(self, seed: Optional[int] = None) -> None:
        """Assign a new randomised class / colour / pose to every object slot."""
        if seed is not None:
            self.seed = seed
        rng = random.Random(self.seed)

        cells = [(x, y) for x in FIELD_X for y in FIELD_Y]
        rng.shuffle(cells)

        # Guarantee every class is present, then fill the remaining slots at random.
        class_plan = list(OBJECT_CLASSES)
        while len(class_plan) < len(cells):
            class_plan.append(rng.choice(OBJECT_CLASSES))
        rng.shuffle(class_plan)

        mujoco.mj_resetData(self.model, self.data)
        self.objects = []

        for slot, ((cell_x, cell_y), obj_class) in enumerate(zip(cells, class_plan)):
            colour = rng.choice(PALETTE)
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"object_{slot}"
            )
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"object_{slot}_geom"
            )
            qpos_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[body_id]])

            self.model.geom_type[geom_id] = obj_class.geom_type
            self.model.geom_size[geom_id] = np.asarray(obj_class.size)
            self.model.geom_rgba[geom_id] = (*colour, 1.0)

            x = cell_x + rng.uniform(-FIELD_JITTER, FIELD_JITTER)
            y = cell_y + rng.uniform(-FIELD_JITTER, FIELD_JITTER)
            z = TABLE_TOP_Z + obj_class.rest_height + 0.002
            yaw = rng.uniform(-math.pi, math.pi)

            self.data.qpos[qpos_adr: qpos_adr + 3] = (x, y, z)
            self.data.qpos[qpos_adr + 3: qpos_adr + 7] = _spawn_quaternion(
                yaw, lying_on_side=obj_class.class_id in LYING_ON_SIDE_CLASS_IDS
            )

            self.objects.append(
                SceneObject(slot, obj_class, colour, body_id, geom_id, qpos_adr,
                            (x, y, z))
            )

        self.data.qpos[self.kinematics.qpos_ids] = self.HOME_POSE
        self.command(np.asarray(self.HOME_POSE), OPEN_GAP)
        self.data.qpos[self.finger_qpos_ids] = finger_position(OPEN_GAP)
        mujoco.mj_forward(self.model, self.data)
        self.settle()

    def settle(self, steps: int = 300) -> None:
        """Let the objects come to rest on the table, then cache their true poses."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.refresh_object_poses()

    def refresh_object_poses(self) -> None:
        """Re-read every object's resting position from the simulation state."""
        for obj in self.objects:
            obj.position = tuple(
                float(v) for v in self.data.qpos[obj.qpos_adr: obj.qpos_adr + 3]
            )

    # -- control -------------------------------------------------------------------
    def command(self, arm_angles: np.ndarray, claw_gap: float) -> None:
        """Write one full control vector: six arm joints plus both claw fingers."""
        self.data.ctrl[self.arm_actuator_ids] = arm_angles
        self.data.ctrl[self.finger_actuator_ids] = finger_position(claw_gap)

    @property
    def claw_gap(self) -> float:
        """Current opening between the finger pads [m]."""
        return float(
            2.0 * (FINGER_HOME_X - FINGER_HALF_THICK
                   - np.mean(self.data.qpos[self.finger_qpos_ids]))
        )

    # -- convenience ---------------------------------------------------------------
    def object_column(self, index: int) -> np.ndarray:
        """Horizontal position of an object, i.e. where the claw must descend."""
        obj = self.objects[index % len(self.objects)]
        return np.array([obj.position[0], obj.position[1]])

    def inspection_target(self, index: int, hover: float = HOVER_HEIGHT,
                          yaw: float = 0.0) -> np.ndarray:
        """Grasp-frame target that puts an object on the camera's optical axis.

        See `inspection_target_for` - this just reads the object's own ground-truth
        position and defers to it.
        """
        obj = self.objects[index % len(self.objects)]
        return inspection_target_for(np.asarray(obj.position), hover, yaw)

    def grasp_plan(self, index: int) -> Tuple[float, float]:
        """Work out how to approach one object.

        The claw closes along its local x axis, so it must be yawed perpendicular to
        the object's longest horizontal axis. That axis is found by projecting the
        object's three body axes - each scaled by its half-size - onto the table plane
        and taking the longest projection.

        Returns:
            (claw yaw [rad], object width across the closing direction [m])
        """
        obj = self.objects[index % len(self.objects)]
        rotation = self.data.xmat[obj.body_id].reshape(3, 3)
        half_sizes = _half_extents(obj.object_class)

        projections = [
            (float(np.hypot(*(rotation[:2, axis] * half_sizes[axis]))), axis)
            for axis in range(3)
        ]
        projections.sort(reverse=True)
        longest_extent, longest_axis = projections[0]
        second_extent = projections[1][0]

        direction = rotation[:2, longest_axis]
        if longest_extent < 1e-6:                       # perfectly upright: any yaw
            yaw = 0.0
        else:
            yaw = math.atan2(direction[1], direction[0]) + math.pi / 2.0

        # Spheres and upright cylinders are rotationally symmetric; use their radius.
        width = 2.0 * max(second_extent, 0.5 * MIN_GRASP_GAP)
        return yaw, width

    def geom_to_object(self) -> Dict[int, SceneObject]:
        return {obj.geom_id: obj for obj in self.objects}


# ======================================================================================
# Eye-in-hand camera and automatic labelling
# ======================================================================================


class SimCamera:
    """Offscreen renderer for one named MuJoCo camera (RGB + segmentation).

    Used for both the claw-mounted eye-in-hand camera and the fixed-overhead baseline
    camera - same rendering path, different `camera_name`.
    """

    def __init__(self, workspace: Workspace, camera_name: str = "claw",
                image_size: int = CAM_IMAGE_SIZE) -> None:
        self.workspace = workspace
        self.image_size = image_size
        self.renderer = mujoco.Renderer(workspace.model, height=image_size,
                                        width=image_size)
        self.camera_id = mujoco.mj_name2id(
            workspace.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        self.focal_px = 0.5 * image_size / math.tan(
            math.radians(float(workspace.model.cam_fovy[self.camera_id])) / 2.0
        )

    def project(self, point_world: Sequence[float]) -> Tuple[float, float, float]:
        """Pinhole-project a world point into this camera's pixel coordinates.

        Returns (pixel_x, pixel_y, depth). Depth is along the camera's optical axis,
        positive in front of the lens. Valid for any point, visible or not - this is
        what lets a fixed camera be cropped around an object's *known* position even
        while the object is occluded.
        """
        cam_pos = self.workspace.data.cam_xpos[self.camera_id]
        rotation = self.workspace.data.cam_xmat[self.camera_id].reshape(3, 3)
        rel_local = rotation.T @ (np.asarray(point_world) - cam_pos)
        depth = float(-rel_local[2])
        half = self.image_size / 2.0
        pixel_x = half + self.focal_px * rel_local[0] / depth
        pixel_y = half - self.focal_px * rel_local[1] / depth
        return pixel_x, pixel_y, depth

    def unproject(self, pixel_x: float, pixel_y: float, depth: float) -> np.ndarray:
        """Back-project one pixel + depth into a world point - the exact inverse of
        `project`. This is what a real depth camera (e.g. a RealSense) gives a
        perception pipeline for free: a 3-D point from a pixel it actually observed,
        with no privileged access to any object's true position.
        """
        half = self.image_size / 2.0
        local = np.array([
            (pixel_x - half) * depth / self.focal_px,
            (half - pixel_y) * depth / self.focal_px,
            -depth,
        ])
        cam_pos = self.workspace.data.cam_xpos[self.camera_id]
        rotation = self.workspace.data.cam_xmat[self.camera_id].reshape(3, 3)
        return cam_pos + rotation @ local

    def render_rgb(self) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB frame from the claw camera."""
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.workspace.data, camera=self.camera_id)
        return self.renderer.render().copy()

    def render_segmentation(self) -> np.ndarray:
        """Return an (H, W) int32 map of MuJoCo geom ids (-1 where nothing is visible)."""
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.workspace.data, camera=self.camera_id)
        segmentation = self.renderer.render()[:, :, 0].astype(np.int32).copy()
        self.renderer.disable_segmentation_rendering()
        return segmentation

    def render_depth(self) -> np.ndarray:
        """Return an (H, W) float32 map of per-pixel depth along the optical axis, in
        metres - what a real RGB-D sensor (e.g. a wrist-mounted RealSense) would give
        a perception pipeline, standing in here for a full depth-camera simulation.
        """
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.workspace.data, camera=self.camera_id)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return depth

    def close(self) -> None:
        self.renderer.close()


def label_from_segmentation(
    segmentation: np.ndarray,
    workspace: Workspace,
    patch_fraction: float = 0.40,
    min_coverage: float = 0.02,
) -> Optional[SceneObject]:
    """Ground-truth label of whatever dominates the centre patch of the frame.

    Reading the label out of the renderer's segmentation buffer means every captured
    frame is annotated exactly and for free - no manual labelling, and no label noise.

    Args:
        segmentation: (H, W) map of geom ids from `SimCamera.render_segmentation`.
        workspace: scene whose objects provide the geom-id -> class mapping.
        patch_fraction: side length of the centre patch, as a fraction of the image.
        min_coverage: minimum share of the patch an object must fill to be reported.

    Returns:
        The dominant `SceneObject`, or None if the camera is looking at background.
    """
    height, width = segmentation.shape
    half = max(1, int(min(height, width) * patch_fraction / 2.0))
    centre_y, centre_x = height // 2, width // 2
    patch = segmentation[centre_y - half: centre_y + half,
                         centre_x - half: centre_x + half]

    lookup = workspace.geom_to_object()
    best_object: Optional[SceneObject] = None
    best_count = 0
    geom_ids, counts = np.unique(patch, return_counts=True)
    for geom_id, count in zip(geom_ids.tolist(), counts.tolist()):
        if geom_id in lookup and count > best_count:
            best_object, best_count = lookup[geom_id], count

    if best_object is None or best_count < min_coverage * patch.size:
        return None
    return best_object


# ======================================================================================
# Dataset capture
# ======================================================================================


def _centroid_offset(segmentation: np.ndarray,
                     obj: Optional[SceneObject]) -> Optional[List[float]]:
    """How far the labelled object's centroid sits from the image centre, in pixels.

    Recorded per frame so the dataset carries evidence of its own framing quality
    instead of it having to be taken on trust.
    """
    if obj is None:
        return None
    rows, cols = np.where(segmentation == obj.geom_id)
    if len(cols) == 0:
        return None
    height, width = segmentation.shape
    return [round(float(cols.mean() - width / 2.0), 1),
            round(float(rows.mean() - height / 2.0), 1)]


def _clamped_crop_box(shape: Tuple[int, int], centre_x: float, centre_y: float,
                      half_side: float) -> Tuple[int, int, int]:
    """Square crop box clamped to stay inside `shape`, as (left, top, side)."""
    height, width = shape
    half_side = max(4.0, min(half_side, min(height, width) / 2.0))
    side = int(2 * half_side)
    left = int(np.clip(centre_x - half_side, 0, width - side))
    top = int(np.clip(centre_y - half_side, 0, height - side))
    return left, top, side


def crop_square(rgb: np.ndarray, centre_x: float, centre_y: float,
                half_side: float, out_size: int) -> np.ndarray:
    """Crop a square window around (centre_x, centre_y) and resize to `out_size`.

    The window is clamped to stay inside the source image (never padded), which is
    always possible here because the overhead camera's field of view is chosen with
    enough margin that every object slot stays well clear of the image border.

    Args:
        rgb: (H, W, 3) uint8 source frame.
        centre_x, centre_y: crop centre, in source-image pixel coordinates.
        half_side: half the crop's side length, in source-image pixels.
        out_size: output edge length in pixels.

    Returns:
        (out_size, out_size, 3) uint8 array.
    """
    from PIL import Image

    left, top, side = _clamped_crop_box(rgb.shape[:2], centre_x, centre_y, half_side)
    box = (left, top, left + side, top + side)
    image = Image.fromarray(rgb).crop(box).resize((out_size, out_size), Image.LANCZOS)
    return np.asarray(image)


def occluded_fraction(segmentation: np.ndarray, obj: SceneObject,
                      centre_x: float, centre_y: float, half_side: float) -> float:
    """Share of the crop window's pixels that are *not* the target object.

    Computed from the un-cropped segmentation buffer over the same box `crop_square`
    would cut, so it measures exactly what the classifier's input is missing - whether
    that is the manipulator in the way, another object, or bare table.
    """
    left, top, side = _clamped_crop_box(segmentation.shape, centre_x, centre_y, half_side)
    box = segmentation[top: top + side, left: left + side]
    if box.size == 0:
        return 1.0
    visible = int(np.count_nonzero(box == obj.geom_id))
    return round(1.0 - visible / box.size, 4)


def save_capture(rgb: np.ndarray, obj: Optional[SceneObject], index: int,
                 seed: int, capture_dir: Path = CAPTURE_DIR,
                 resize_to: Optional[int] = None,
                 extra: Optional[Dict[str, object]] = None) -> Path:
    """Persist one labelled frame plus its JSONL record.

    Args:
        rgb: (H, W, 3) uint8 camera frame.
        obj: the object the camera is looking at, or None for background.
        index: frame number, used for the filename.
        seed: layout seed, recorded so the frame can be traced back to a scene.
        capture_dir: directory to write the frame and append the label record to.
        resize_to: optional square edge length, so frames grabbed from the on-screen
            panel land in the dataset at the same resolution as headless renders.
        extra: additional fields to merge into the JSONL record.

    Returns:
        Path of the written image.
    """
    capture_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{index:05d}"
    image_path = capture_dir / f"{stem}.png"

    try:
        from PIL import Image

        image = Image.fromarray(rgb)
        if resize_to is not None and image.size != (resize_to, resize_to):
            image = image.resize((resize_to, resize_to), Image.LANCZOS)
        rgb = np.asarray(image)
        image.save(image_path)
    except ImportError:                                     # pragma: no cover
        image_path = capture_dir / f"{stem}.npy"
        np.save(image_path, rgb)

    record = {
        "file": image_path.name,
        "class_id": obj.object_class.class_id if obj else -1,
        "class_name": obj.object_class.name if obj else "background",
        "colour_rgb": list(obj.colour) if obj else None,
        "object_xyz": list(obj.position) if obj else None,
        "image_size": list(rgb.shape[:2]),
        "layout_seed": seed,
    }
    if extra:
        record.update(extra)
    with open(capture_dir / "labels.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return image_path


# ======================================================================================
# Scan controller
# ======================================================================================


def _smoothstep(alpha: float) -> float:
    """Ease-in / ease-out interpolation factor for alpha in [0, 1]."""
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


@dataclass
class ClawCommand:
    """One control setpoint: where the grasp frame goes and how far the claw opens."""

    position: np.ndarray
    yaw: float
    gap: float


class ScanController:
    """Finite-state machine that inspects, and optionally picks up, each object.

    In pick mode one object takes the cycle
    `approach -> descend -> close -> lift -> hold -> lower -> release -> retreat`;
    in inspect mode it is just `approach -> hold`.

    The classification frame is taken the moment `approach` finishes, i.e. with the claw
    hovering over an undisturbed object. That is the viewpoint a real system would
    classify from - it has to decide what the object is *before* committing to a grasp -
    and it is far enough back to frame the whole object. Once the object is in the claw
    it sits a few centimetres from the lens and overflows the frame.
    """

    def __init__(self, workspace: Workspace, grasping: bool = True) -> None:
        self.workspace = workspace
        self.grasping = grasping
        self.index = 0
        self.phase_index = 0
        self.phase_start = 0.0
        self.anchor = np.zeros(3)               # pose held at the start of a phase
        self.anchor_yaw = 0.0
        self.target_yaw = 0.0
        self.object_width = OPEN_GAP
        self.grasp_height = TABLE_TOP_Z

    # -- lifecycle -----------------------------------------------------------------
    @property
    def sequence(self) -> Tuple[Tuple[str, float], ...]:
        return PICK_SEQUENCE if self.grasping else INSPECT_SEQUENCE

    @property
    def phase(self) -> str:
        return self.sequence[self.phase_index % len(self.sequence)][0]

    def begin(self, now: float, index: Optional[int] = None) -> None:
        position, _ = self.workspace.kinematics.tool_pose(self.workspace.data)
        self.anchor = position.copy()
        self.phase_index = 0
        self.phase_start = now
        if index is not None:
            self.index = index
        self._plan_current_object()

    def _plan_current_object(self) -> None:
        """Read the object's current pose and decide the yaw and opening to use."""
        obj = self.workspace.objects[self.index % len(self.workspace.objects)]
        self.target_yaw, self.object_width = self.workspace.grasp_plan(self.index)
        self.grasp_height = float(obj.position[2])

    # -- per-control-step ------------------------------------------------------------
    def update(self, now: float, speed: float = 1.0) -> Tuple[ClawCommand, bool]:
        """Advance the machine by wall-clock time.

        Returns:
            (command for this step, True on the step the claw reaches the inspection
            pose - the hover pose at the end of `approach`)
        """
        phase, duration = self.sequence[self.phase_index % len(self.sequence)]
        elapsed = (now - self.phase_start) * max(0.05, speed)
        alpha = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)

        hover = self.workspace.inspection_target(self.index, HOVER_HEIGHT,
                                                 self.target_yaw)
        column = self.workspace.object_column(self.index)
        grasp = np.array([column[0], column[1], self.grasp_height])
        lifted = np.array([column[0], column[1], TABLE_TOP_Z + LIFT_HEIGHT])
        closed_gap = grasp_gap(self.object_width)

        if phase == "approach":
            position = self.anchor + (hover - self.anchor) * _smoothstep(alpha)
            yaw = self.anchor_yaw + _wrap(self.target_yaw - self.anchor_yaw) * \
                _smoothstep(alpha)
            gap = OPEN_GAP
        elif phase == "settle":
            position, yaw, gap = hover, self.target_yaw, OPEN_GAP
        elif phase == "descend":
            position = hover + (grasp - hover) * _smoothstep(alpha)
            yaw, gap = self.target_yaw, OPEN_GAP
        elif phase == "close":
            position, yaw = grasp, self.target_yaw
            gap = OPEN_GAP + (closed_gap - OPEN_GAP) * _smoothstep(alpha)
        elif phase == "lift":
            position = grasp + (lifted - grasp) * _smoothstep(alpha)
            yaw, gap = self.target_yaw, closed_gap
        elif phase == "hold":
            position = lifted if self.grasping else hover
            yaw = self.target_yaw
            gap = closed_gap if self.grasping else OPEN_GAP
        elif phase == "lower":
            position = lifted + (grasp - lifted) * _smoothstep(alpha)
            yaw, gap = self.target_yaw, closed_gap
        elif phase == "release":
            position, yaw = grasp, self.target_yaw
            gap = closed_gap + (OPEN_GAP - closed_gap) * _smoothstep(alpha)
        else:                                                   # retreat
            position = grasp + (hover - grasp) * _smoothstep(alpha)
            yaw, gap = self.target_yaw, OPEN_GAP

        at_inspection_pose = False
        if alpha >= 1.0:
            at_inspection_pose = phase == "settle"
            self.anchor, self.anchor_yaw = position.copy(), yaw
            self.phase_index += 1
            self.phase_start = now
            if self.phase_index % len(self.sequence) == 0:
                # Cycle complete: the object may have shifted, so re-read the layout.
                self.workspace.refresh_object_poses()
                self.index = (self.index + 1) % len(self.workspace.objects)
                self._plan_current_object()

        return ClawCommand(position, yaw, gap), at_inspection_pose


# ======================================================================================
# Entry points
# ======================================================================================


# ======================================================================================
# Interactive viewer with a claw-POV panel
# ======================================================================================

WINDOW_WIDTH: int = 1500
WINDOW_HEIGHT: int = 880
PANEL_WIDTH: int = 400            # claw-POV panel, pinned to the right edge
PANEL_MARGIN: int = 18
PANEL_BORDER: int = 3


class PictureInPictureViewer:
    """A MuJoCo viewer that renders the scene and the claw's own camera side by side.

    MuJoCo's stock passive viewer shows a single camera at a time, so the claw view
    would have to replace the third-person view rather than sit next to it. Owning the
    GLFW window directly means both can be drawn each frame: the free camera fills the
    window, then the claw camera is rendered again into a panel on the right.
    """

    def __init__(self, workspace: Workspace, title: str) -> None:
        self.workspace = workspace
        self.model, self.data = workspace.model, workspace.data

        if not glfw.init():
            raise RuntimeError("failed to initialise GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("failed to create a window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        # Some window managers create GLFW windows behind whatever already has focus
        # (e.g. the terminal `python` was launched from) - force it to the front so it
        # is not mistaken for having failed to open at all.
        glfw.show_window(self.window)
        glfw.focus_window(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        self.options = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()

        self.free_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.free_camera)
        self.free_camera.lookat[:] = (TABLE_CENTER_X, 0.0, TABLE_TOP_Z + 0.08)
        self.free_camera.distance = 1.75
        self.free_camera.azimuth = 133.0
        self.free_camera.elevation = -21.0

        self.claw_camera = mujoco.MjvCamera()
        self.claw_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.panel_label = "CLAW CAMERA"
        self.set_panel_camera("claw", self.panel_label)

        self._last_cursor = (0.0, 0.0)
        self._buttons = {"left": False, "right": False, "middle": False}
        self.key_handler: Optional[Callable[[int], None]] = None

        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)

    # -- input ----------------------------------------------------------------------
    def _on_button(self, window, button, action, mods) -> None:
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._buttons["left"] = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._buttons["right"] = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._buttons["middle"] = pressed
        self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor(self, window, xpos, ypos) -> None:
        dx, dy = xpos - self._last_cursor[0], ypos - self._last_cursor[1]
        self._last_cursor = (xpos, ypos)
        if not any(self._buttons.values()):
            return

        _, height = glfw.get_window_size(window)
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._buttons["left"]:
            action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift
                      else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        elif self._buttons["right"]:
            action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift
                      else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height,
                              self.scene, self.free_camera)

    def _on_scroll(self, window, xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoffset, self.scene, self.free_camera)

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif self.key_handler is not None:
            self.key_handler(key)

    # -- rendering ------------------------------------------------------------------
    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def set_panel_camera(self, camera_name: str, label: str) -> None:
        """Point the picture-in-picture panel at a different named MuJoCo camera."""
        self.claw_camera.fixedcamid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        self.panel_label = label

    def _panel_rect(self, window_width: int, window_height: int) -> mujoco.MjrRect:
        size = min(PANEL_WIDTH, window_width // 3, window_height // 2)
        return mujoco.MjrRect(
            window_width - size - PANEL_MARGIN,
            window_height - size - PANEL_MARGIN,
            size, size,
        )

    def render(self, overlay_lines: Sequence[Tuple[str, str]] = (),
               grab: Optional[str] = None,
               crop_box: Optional[Tuple[float, float, float, int]] = None,
               ) -> Optional[np.ndarray]:
        """Draw one frame: scene, panel camera, then the text overlay.

        Args:
            overlay_lines: up to two (left, right) text blocks, drawn top and bottom.
            grab: optionally read the frame back before it is presented - "panel" for
                just the panel, "window" for the whole window. Reading has to happen
                before the buffer swap, which is why it lives here rather than in a
                separate method.
            crop_box: optional (centre_x, centre_y, half_side, source_size) in the
                panel camera's own source-image pixel coordinates (top-left origin,
                as `SimCamera.project` and `crop_square` use) - draws a highlighted
                outline over the region of the panel that a classifier would actually
                receive, e.g. the fixed-overhead camera's crop window. `source_size` is
                the pixel width/height that (centre_x, centre_y, half_side) are measured
                against (typically `CAM_IMAGE_SIZE`), used to scale into panel pixels.

        Returns:
            The grabbed RGB image, or None.
        """
        # Any offscreen `mujoco.Renderer` created elsewhere (e.g. `SimCamera`, used by
        # `demo.py` to feed a live classifier) owns its own GL context and makes it
        # current when it renders. If one of those runs between this window's frames,
        # this window's own context is no longer current, and every call below would
        # silently draw into the wrong (invisible) context - the window stays black
        # while the process keeps running normally. Reclaiming it here, every frame,
        # makes `render()` correct regardless of what else rendered in between.
        glfw.make_context_current(self.window)

        width, height = glfw.get_framebuffer_size(self.window)
        full = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(self.model, self.data, self.options, self.perturb,
                               self.free_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(full, self.scene, self.context)

        panel = self._panel_rect(width, height)
        border = mujoco.MjrRect(
            panel.left - PANEL_BORDER, panel.bottom - PANEL_BORDER,
            panel.width + 2 * PANEL_BORDER, panel.height + 2 * PANEL_BORDER,
        )
        mujoco.mjr_rectangle(border, 0.95, 0.45, 0.10, 1.0)

        mujoco.mjv_updateScene(self.model, self.data, self.options, self.perturb,
                               self.claw_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(panel, self.scene, self.context)

        # Read the panel back *before* any overlay is drawn, so saved frames are clean
        # camera images rather than screenshots of the UI.
        grabbed: Optional[np.ndarray] = None
        if grab == "panel":
            grabbed = self._read_rect(panel)

        if crop_box is not None:
            self._draw_crop_box(panel, crop_box)

        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, panel,
                           self.panel_label, "", self.context)

        for grid_pos, (left, right) in zip(
            (mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
            overlay_lines,
        ):
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, grid_pos, full,
                               left, right, self.context)

        if grab == "window":
            grabbed = self._read_rect(full)

        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return grabbed

    def _draw_crop_box(self, panel: "mujoco.MjrRect",
                       crop_box: Tuple[float, float, float, int]) -> None:
        """Outline, in bright green, the region of the panel a classifier would crop.

        `crop_box` is (centre_x, centre_y, half_side, source_size) in the panel
        camera's own rendered pixel space (top-left origin); this maps it into the
        panel's on-screen rectangle (bottom-left origin, arbitrary panel size).
        """
        centre_x, centre_y, half_side, source_size = crop_box
        scale = panel.width / float(source_size)

        left = panel.left + int((centre_x - half_side) * scale)
        right = panel.left + int((centre_x + half_side) * scale)
        # Image row 0 is the top; MjrRect y grows upward from the panel's bottom.
        top = panel.bottom + panel.height - int((centre_y - half_side) * scale)
        bottom = panel.bottom + panel.height - int((centre_y + half_side) * scale)
        left, right = sorted((max(left, panel.left), min(right, panel.left + panel.width)))
        bottom, top = sorted((max(bottom, panel.bottom), min(top, panel.bottom + panel.height)))

        thickness = 3
        colour = (0.15, 0.95, 0.35, 1.0)
        edges = [
            (left, top - thickness, right - left, thickness),          # top edge
            (left, bottom, right - left, thickness),                   # bottom edge
            (left, bottom, thickness, top - bottom),                   # left edge
            (right - thickness, bottom, thickness, top - bottom),      # right edge
        ]
        for x, y, w, h in edges:
            if w > 0 and h > 0:
                mujoco.mjr_rectangle(mujoco.MjrRect(int(x), int(y), int(w), int(h)),
                                     *colour)

    def _read_rect(self, rect: mujoco.MjrRect) -> np.ndarray:
        """Read one viewport rectangle out of the framebuffer, top-down RGB."""
        rgb = np.zeros((rect.height, rect.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.context)
        return np.flipud(rgb).copy()            # OpenGL hands back bottom-up rows

    def close(self) -> None:
        self.context.free()
        glfw.terminate()


def object_under_claw(workspace: Workspace) -> Optional[SceneObject]:
    """Which object the claw camera's optical axis is pointing at, by ray cast.

    Faster and more direct than reading a segmentation buffer, and it needs no render,
    so it can run every frame to drive the on-screen readout.
    """
    model, data = workspace.model, workspace.data
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "claw")
    origin = data.cam_xpos[camera_id].copy()
    # A MuJoCo camera looks along the -z axis of its own frame.
    direction = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2].copy()

    # Restrict the ray to the object geom group, otherwise a finger or the palm can
    # intercept it and every frame comes back as background.
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[OBJECT_GEOM_GROUP] = 1

    geom_id = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction,
                             geomgroup, 1, -1, geom_id)
    if distance < 0:
        return None
    return workspace.geom_to_object().get(int(geom_id[0]))


def run_viewer(seed: int, capture_on_hold: bool, grasping: bool) -> None:
    """Interactive session: third-person scene plus the claw's own camera panel."""
    workspace = Workspace(seed)
    kinematics = workspace.kinematics
    scan = ScanController(workspace, grasping=grasping)
    viewer = PictureInPictureViewer(workspace, "Eye-in-Hand Claw - Object Inspection")

    state = {
        "paused": False,
        "auto": True,
        "speed": 1.0,
        "captures": 0,
        "seed": seed,
        "manual_gap": OPEN_GAP,
    }
    requests = {"capture": False, "reshuffle": False}

    def on_key(key: int) -> None:
        if key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_G:
            scan.grasping = not scan.grasping
            scan.begin(time.time())
        elif key == glfw.KEY_M:
            state["auto"] = not state["auto"]
        elif key == glfw.KEY_R:
            requests["reshuffle"] = True
        elif key == glfw.KEY_C:
            requests["capture"] = True
        elif key == glfw.KEY_O:
            state["manual_gap"] = (MIN_GRASP_GAP
                                   if state["manual_gap"] > MIN_GRASP_GAP else OPEN_GAP)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["speed"] = min(4.0, state["speed"] * 1.25)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["speed"] = max(0.2, state["speed"] / 1.25)

    viewer.key_handler = on_key

    print(f"[sim] seed {state['seed']} | {len(workspace.objects)} objects | "
          f"classes: {', '.join(c.name for c in OBJECT_CLASSES)}")
    print("[sim] mouse: drag = orbit, right-drag = pan, wheel = zoom")
    print("[sim] keys: SPACE pause | G grasp on/off | M auto/manual | O open/close claw")
    print("[sim]       R reshuffle | C capture | +/- speed | ESC quit")

    scan.begin(time.time())
    step_index = 0
    inspection_ready = False
    last_seen: Optional[SceneObject] = None

    try:
        while viewer.is_running():
            frame_start = time.time()

            if requests["reshuffle"]:
                requests["reshuffle"] = False
                state["seed"] += 1
                workspace.reshuffle(state["seed"])
                kinematics._last_solution = None
                scan.begin(time.time(), index=0)

            if not state["paused"]:
                # Physics runs several steps per rendered frame to keep real time.
                for _ in range(int(round((1.0 / 60.0) / SIM_TIMESTEP))):
                    if step_index % CONTROL_DECIMATION == 0:
                        if state["auto"]:
                            command, ready = scan.update(time.time(), state["speed"])
                            inspection_ready = inspection_ready or ready
                            angles = kinematics.solve(workspace.data, command.position,
                                                      command.yaw)
                            workspace.command(angles, command.gap)
                        else:
                            workspace.data.ctrl[workspace.finger_actuator_ids] = \
                                finger_position(state["manual_gap"])
                    mujoco.mj_step(workspace.model, workspace.data)
                    step_index += 1

            last_seen = object_under_claw(workspace) or last_seen
            seen_text = last_seen.object_class.name if last_seen else "-"
            mode = "AUTO" if state["auto"] else "MANUAL"
            action = scan.phase.upper() if state["auto"] else "held"

            # A capture has to be read back before the buffer swap, so it is requested
            # as part of the render call rather than taken afterwards.
            auto_grab = capture_on_hold and inspection_ready
            inspection_ready = False

            frame = viewer.render(
                overlay_lines=(
                    (f"{mode} | {action}{'  [PAUSED]' if state['paused'] else ''}\n"
                     f"under claw: {seen_text}\n"
                     f"claw gap: {workspace.claw_gap * 1000:.0f} mm\n"
                     f"speed: {state['speed']:.2f}x   captures: {state['captures']}",
                     ""),
                    ("SPACE pause   G grasp   M auto/manual   O open/close\n"
                     "R reshuffle   C capture   +/- speed   ESC quit",
                     ""),
                ),
                grab="panel" if (requests["capture"] or auto_grab) else None,
            )

            if frame is not None:
                save_capture(frame, object_under_claw(workspace), state["captures"],
                             state["seed"], capture_dir=EYE_IN_HAND_DIR,
                             resize_to=CAM_IMAGE_SIZE)
                state["captures"] += 1
                if requests["capture"]:
                    print(f"[sim] captured frame {state['captures'] - 1} -> {seen_text}")
                requests["capture"] = False

            remaining = (1.0 / 60.0) - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()

    if state["captures"]:
        print(f"[sim] saved {state['captures']} labelled frames to {EYE_IN_HAND_DIR}")


def run_full_ui(seed: int) -> None:
    """Launch MuJoCo's stock UI, which exposes a control slider per joint and finger."""
    workspace = Workspace(seed)
    print("[sim] full UI: drive each joint from the 'Control' panel on the right; "
          "press '[' / ']' to switch to the claw camera.")
    mujoco.viewer.launch(workspace.model, workspace.data)


def _on_table(position: Tuple[float, float, float]) -> bool:
    """Whether a position is still resting on the table, not fallen to the floor.

    A safety net behind `OBJECT_FRICTION`: if a sphere or cylinder ever does roll off
    (physics is stochastic across enough reshuffles even with high rolling friction),
    this catches it before a frame gets captured of an object that is not there.
    """
    return abs(position[2] - TABLE_TOP_Z) < 0.15


def _capture_fixed_view(camera: SimCamera, capture_dir: Path, view_name: str,
                        target_obj: SceneObject, index: int, seed: int,
                        hover: float) -> None:
    """Crop `camera`'s current frame around `target_obj`'s known position and save it.

    Shared by both fixed cameras (overhead, angled): same fixed, class-independent
    crop size, same "label from known position, not from segmentation" reasoning (the
    label must stay correct even on the frames where the object is what is occluded),
    same occlusion bookkeeping. `SimCamera.project` and `FIXED_CAMERA_CROP_HALF_SIDE`
    are both viewpoint-agnostic, so nothing here assumes straight-down vs. angled.
    """
    rgb = camera.render_rgb()
    segmentation = camera.render_segmentation()
    pixel_x, pixel_y, depth = camera.project(target_obj.position)
    half_side_px = FIXED_CAMERA_CROP_HALF_SIDE * camera.focal_px / depth
    crop = crop_square(rgb, pixel_x, pixel_y, half_side_px, CAM_IMAGE_SIZE)
    occlusion = occluded_fraction(segmentation, target_obj, pixel_x, pixel_y, half_side_px)
    save_capture(crop, target_obj, index, seed, capture_dir=capture_dir, extra={
        "view": view_name,
        "hover_height_m": round(hover, 4),
        "occluded_fraction": occlusion,
    })


def generate_dataset(n_frames: int, seed: int, views_per_layout: int = 12) -> None:
    """Headless generation of a *paired*, three-camera dataset.

    Both fixed-camera frames are captured first, from wherever the arm happens to be
    left over from the previous inspection - exactly how a real fixed camera would be
    used: it looks at the scene while the arm is doing something else, not through its
    own gripper. The arm is then driven to hover over the target object and the claw
    camera captures it from there. All three crops are ground-truthed to the same
    object and written under the same frame index, so an occasional coincidence - the
    arm having been left parked over today's target - shows up as genuine, unforced
    occlusion in the fixed-camera sets rather than being manufactured every single
    frame.

    The fixed-camera crops are taken around the object's known 3-D position rather than
    from their own segmentation, because the label has to stay correct even on the
    frames where the object is exactly what is occluded.
    """
    workspace = Workspace(seed)
    eye_camera = SimCamera(workspace, "claw")
    overhead_camera = SimCamera(workspace, "overhead")
    angled_camera = SimCamera(workspace, "angled")
    rng = random.Random(seed)
    kinematics = workspace.kinematics

    counts: Dict[str, int] = {c.name: 0 for c in OBJECT_CLASSES}
    label_agreement = 0
    saved = 0
    layout_seed = seed

    while saved < n_frames:
        for _ in range(views_per_layout):
            if saved >= n_frames:
                break
            # Objects can be nudged as the arm sweeps between viewpoints, so re-read
            # their true poses rather than trusting the layout they were spawned with.
            workspace.refresh_object_poses()
            index = rng.randrange(len(workspace.objects))
            target_obj = workspace.objects[index]
            attempts = 0
            while not _on_table(target_obj.position) and attempts < len(workspace.objects):
                index = rng.randrange(len(workspace.objects))
                target_obj = workspace.objects[index]
                attempts += 1
            if not _on_table(target_obj.position):
                # every object in this layout has ended up off the table - reshuffle
                # rather than capture a frame of an object that is not there
                layout_seed += 1
                workspace.reshuffle(layout_seed)
                kinematics._last_solution = None
                continue
            hover = rng.uniform(0.11, 0.18)
            yaw, _ = workspace.grasp_plan(index)
            yaw += rng.uniform(-0.40, 0.40)
            # Vary the viewpoint, but around a pose that keeps the object centred.
            target = workspace.inspection_target(index, hover, yaw)
            target[0] += rng.uniform(-0.015, 0.015)
            target[1] += rng.uniform(-0.015, 0.015)

            # -- both fixed cameras: captured from wherever the arm is *right now*,
            # i.e. before it moves to inspect this target - see the docstring.
            _capture_fixed_view(overhead_camera, FIXED_OVERHEAD_DIR, "fixed_overhead",
                               target_obj, saved, layout_seed, hover)
            _capture_fixed_view(angled_camera, FIXED_ANGLED_DIR, "fixed_angled",
                               target_obj, saved, layout_seed, hover)

            # -- eye-in-hand: drive the arm to hover over the target, then capture the
            # claw camera's own frame, cross-checked against its own segmentation (the
            # "auto-label agreement" metric also reported by --selftest).
            workspace.command(kinematics.solve(workspace.data, target, yaw), OPEN_GAP)
            for _ in range(220):
                mujoco.mj_step(workspace.model, workspace.data)
            workspace.refresh_object_poses()

            eye_rgb = eye_camera.render_rgb()
            eye_segmentation = eye_camera.render_segmentation()
            eye_hit = label_from_segmentation(eye_segmentation, workspace)
            agrees = eye_hit is not None and eye_hit.slot == target_obj.slot
            label_agreement += int(agrees)
            save_capture(eye_rgb, target_obj, saved, layout_seed,
                        capture_dir=EYE_IN_HAND_DIR, extra={
                "view": "eye_in_hand",
                "hover_height_m": round(hover, 4),
                "centre_offset_px": _centroid_offset(eye_segmentation, target_obj),
                "auto_label_agrees": agrees,
            })

            counts[target_obj.object_class.name] += 1
            saved += 1
            if saved % 25 == 0:
                print(f"  {saved}/{n_frames} paired frames")

        layout_seed += 1
        workspace.reshuffle(layout_seed)
        kinematics._last_solution = None

    eye_camera.close()
    overhead_camera.close()
    angled_camera.close()
    print(f"\nsaved {saved} paired frames")
    print(f"  eye-in-hand     -> {EYE_IN_HAND_DIR}")
    print(f"  fixed-overhead  -> {FIXED_OVERHEAD_DIR}")
    print(f"  fixed-angled    -> {FIXED_ANGLED_DIR}")
    print(f"  eye-in-hand auto-label agreement with ground truth: "
          f"{label_agreement}/{saved} ({100.0 * label_agreement / max(1, saved):.1f} %)")
    print("class distribution:")
    for name, count in counts.items():
        print(f"  {name:<11} {count:>5}  ({100.0 * count / max(1, saved):5.1f} %)")


def selftest(seed: int) -> None:
    """Headless verification of the scene, IK accuracy, labelling and grasping."""
    print("building workspace ...")
    workspace = Workspace(seed)
    kinematics = workspace.kinematics
    model, data = workspace.model, workspace.data

    print(f"  model: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
    print(f"  claw: {OPEN_GAP * 1000:.0f} mm open, "
          f"{FINGER_TRAVEL * 1000:.0f} mm stroke per finger")
    print(f"  objects: {len(workspace.objects)}")

    camera = SimCamera(workspace, "claw")
    position_errors: List[float] = []
    axis_errors: List[float] = []
    agreements = 0
    lifted = 0

    print("\ninspect + pick cycle for each object ...")
    for index, obj in enumerate(workspace.objects):
        workspace.refresh_object_poses()
        yaw, width = workspace.grasp_plan(index)
        hover = workspace.inspection_target(index, HOVER_HEIGHT, yaw)
        column = workspace.object_column(index)
        grasp = np.array([column[0], column[1], obj.position[2]])
        start_height = obj.position[2]

        # 1. hover, claw open: this is the pose the classifier's frames come from
        workspace.command(kinematics.solve(data, hover, yaw), OPEN_GAP)
        for _ in range(500):
            mujoco.mj_step(model, data)

        reached, rotation = kinematics.tool_pose(data)
        position_errors.append(float(np.linalg.norm(reached - hover)))
        axis_errors.append(math.degrees(math.acos(float(
            np.clip(np.dot(rotation[:, 2], [0, 0, -1]), -1.0, 1.0)
        ))))

        hit = label_from_segmentation(camera.render_segmentation(), workspace)
        correct = hit is not None and hit.geom_id == obj.geom_id
        agreements += int(correct)

        # 2. descend, close, lift - did the object actually come up with the claw?
        workspace.command(kinematics.solve(data, grasp, yaw), OPEN_GAP)
        for _ in range(400):
            mujoco.mj_step(model, data)
        workspace.command(kinematics.solve(data, grasp, yaw), grasp_gap(width))
        for _ in range(300):
            mujoco.mj_step(model, data)
        raised = np.array([column[0], column[1], TABLE_TOP_Z + LIFT_HEIGHT])
        workspace.command(kinematics.solve(data, raised, yaw), grasp_gap(width))
        for _ in range(600):
            mujoco.mj_step(model, data)

        height_gain = float(data.qpos[obj.qpos_adr + 2]) - start_height
        grasped = height_gain > 0.04
        lifted += int(grasped)

        print(f"    {obj.object_class.name:<9} "
              f"pos err {position_errors[-1] * 1000:5.1f} mm | "
              f"axis {axis_errors[-1]:4.2f} deg | "
              f"sees {(hit.object_class.name if hit else 'background'):<10} "
              f"{'OK ' if correct else 'MISS'} | "
              f"grasp width {width * 1000:4.1f} mm, lifted {height_gain * 1000:+6.1f} mm "
              f"{'OK' if grasped else 'FAILED'}")

        # put it back down and release before moving on
        workspace.command(kinematics.solve(data, grasp, yaw), grasp_gap(width))
        for _ in range(400):
            mujoco.mj_step(model, data)
        workspace.command(kinematics.solve(data, grasp, yaw), OPEN_GAP)
        for _ in range(300):
            mujoco.mj_step(model, data)
        workspace.command(kinematics.solve(data, hover, yaw), OPEN_GAP)
        for _ in range(300):
            mujoco.mj_step(model, data)

    total = len(workspace.objects)
    print(f"\n  mean IK position error : {np.mean(position_errors) * 1000:.1f} mm")
    print(f"  mean optical-axis error: {np.mean(axis_errors):.2f} deg")
    print(f"  auto-label agreement   : {agreements}/{total}")
    print(f"  successful grasps      : {lifted}/{total}")
    print(f"  claw frame             : {camera.render_rgb().shape}")
    camera.close()
    print("\nselftest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive tabletop manipulator simulation: two-finger claw with "
                    "a claw-mounted camera for object classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="random seed for the object layout (default: 42)")
    parser.add_argument("--no-grasp", action="store_true",
                        help="inspect only, without picking objects up")
    parser.add_argument("--ui", action="store_true",
                        help="launch MuJoCo's stock UI with per-joint control sliders")
    parser.add_argument("--capture", action="store_true",
                        help="save a labelled frame at each inspection pose, before the grasp")
    parser.add_argument("--dataset", type=int, metavar="N", default=None,
                        help="headless: generate N paired eye-in-hand / "
                             "fixed-overhead frames")
    parser.add_argument("--selftest", action="store_true",
                        help="headless check of scene, IK, labelling and grasping")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.selftest:
        selftest(args.seed)
    elif args.dataset is not None:
        generate_dataset(args.dataset, args.seed)
    elif args.ui:
        run_full_ui(args.seed)
    else:
        run_viewer(args.seed, capture_on_hold=args.capture,
                   grasping=not args.no_grasp)


if __name__ == "__main__":
    main()
