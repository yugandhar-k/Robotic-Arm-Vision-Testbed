"""dual_arm.py - Two 6-DoF arms facing each other across a shared workspace.

This is the foundation for the bimanual cooperative-manipulation study (closing a lid,
cooperatively lifting an oversized payload) - tasks that physically require two hands
acting on the same object at the same time. It reuses the *exact* arm geometry and
inverse kinematics the single-arm classification benchmark is built on: the arm body and
its actuators come from `run_sim.arm_body_mjcf` / `arm_actuator_mjcf` (one shared source,
so the two tracks can never drift), and each arm is driven by a `run_sim.ArmKinematics`
instance made prefix- and base-yaw-aware for exactly this purpose.

The two arms are mounted on opposite edges of the table, yawed to face each other, so both
can reach a common central region - the setup every two-hand task needs.

Usage
-----
    python dual_arm.py --selftest     # headless: both arms reach the shared centre
    python dual_arm.py                # interactive viewer (added in a later phase)

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import mujoco
import numpy as np

import run_sim as sim

# ---------------------------------------------------------------------------
# Two-arm placement: arms on opposite table edges, yawed to face each other.
# ---------------------------------------------------------------------------

# Left arm keeps the single-arm benchmark's mount; the right arm mirrors it across the
# workspace centre and is yawed 180 deg so its "forward" (+x at zero base yaw) points back
# toward the shared centre instead of off the far edge of the table.
ARM_L_PREFIX: str = "l_"
ARM_R_PREFIX: str = "r_"
ARM_L_BASE: Tuple[float, float] = (-0.16, 0.0)
ARM_R_BASE: Tuple[float, float] = (0.56, 0.0)
ARM_L_BASE_YAW: float = 0.0
ARM_R_BASE_YAW: float = math.pi

# Shared workspace centre - the midpoint between the two mounts, on the table.
WORKSPACE_CENTRE: Tuple[float, float] = (
    0.5 * (ARM_L_BASE[0] + ARM_R_BASE[0]), 0.5 * (ARM_L_BASE[1] + ARM_R_BASE[1]))

# Cooperative-lift payload: a DUMBBELL lying along the x axis (the line joining the two
# arms). A wide central block that no single two-finger gripper can span (its width
# exceeds the 80 mm open gap), joined to a thin graspable rod at each end. So the only
# way to pick it up is two hands, one on each end rod - and because the mass sits in the
# middle, a single hand on one end rod just torques it out of the gripper. Half-extents
# are (x, y, z); the two end rods are what the arms grasp.
PAYLOAD_BLOCK_HALF: Tuple[float, float, float] = (0.080, 0.060, 0.022)
PAYLOAD_ROD_HALF: Tuple[float, float, float] = (0.040, 0.016, 0.016)
PAYLOAD_ROD_OFFSET: float = PAYLOAD_BLOCK_HALF[0] + PAYLOAD_ROD_HALF[0]   # end rod centre x
PAYLOAD_DENSITY: float = 400.0
PAYLOAD_REST_Z: float = sim.TABLE_TOP_Z + PAYLOAD_BLOCK_HALF[2] + 0.002
# Backwards-friendly alias: the tallest half-extent, used for grasp/lift heights.
PAYLOAD_HALF: Tuple[float, float, float] = PAYLOAD_BLOCK_HALF

HOME_POSE: Tuple[float, ...] = sim.Workspace.HOME_POSE

# Static MJCF blocks the arm bodies depend on (joint/finger/link classes and materials),
# copied from run_sim's single-arm scene so a dual-arm model compiles stand-alone. They
# are pure declarations (no geometry), so there is nothing here to keep in sync with the
# arm itself - that lives entirely in run_sim.arm_body_mjcf.
_SCENE_ASSETS = f"""
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
    <material name="payload" rgba="0.30 0.55 0.85 1" specular="0.3" shininess="0.3"/>
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
      <joint type="slide" range="0 {sim.FINGER_TRAVEL}" damping="12" armature="0.008"/>
      <position kp="900" kv="35" forcerange="-90 90" ctrlrange="0 {sim.FINGER_TRAVEL}"/>
    </default>
    <default class="finger_pad">
      <geom type="box" material="accent" density="900" condim="4"
            friction="1.8 0.10 0.002" solref="0.008 1" solimp="0.95 0.99 0.001"/>
    </default>
  </default>
"""


def dumbbell_objects_xml() -> str:
    """The cooperative-lift dumbbell (wide central block + two graspable end rods) as a
    free body at the workspace centre."""
    cx, cy = WORKSPACE_CENTRE
    return f"""    <!-- cooperative-lift payload: dumbbell along x (wide centre block,
         ungraspable by one gripper, + a thin graspable rod at each end) -->
    <body name="payload" pos="{cx} {cy} {PAYLOAD_REST_Z}">
      <freejoint name="payload_free"/>
      <geom name="payload_block" type="box" material="payload"
            size="{PAYLOAD_BLOCK_HALF[0]} {PAYLOAD_BLOCK_HALF[1]} {PAYLOAD_BLOCK_HALF[2]}"
            density="{PAYLOAD_DENSITY}" condim="4"
            friction="1.2 0.06 0.01" solref="0.006 1"/>
      <geom name="payload_rod_l" type="box" material="accent"
            pos="{-PAYLOAD_ROD_OFFSET} 0 0"
            size="{PAYLOAD_ROD_HALF[0]} {PAYLOAD_ROD_HALF[1]} {PAYLOAD_ROD_HALF[2]}"
            density="{PAYLOAD_DENSITY}" condim="4"
            friction="1.8 0.10 0.01" solref="0.006 1"/>
      <geom name="payload_rod_r" type="box" material="accent"
            pos="{PAYLOAD_ROD_OFFSET} 0 0"
            size="{PAYLOAD_ROD_HALF[0]} {PAYLOAD_ROD_HALF[1]} {PAYLOAD_ROD_HALF[2]}"
            density="{PAYLOAD_DENSITY}" condim="4"
            friction="1.8 0.10 0.01" solref="0.006 1"/>
    </body>"""


# ---------------------------------------------------------------------------
# Lid-close task objects: a free-standing can + a separate lid (disk with a
# graspable central stem). One arm holds the can steady while the other seats the lid.
# ---------------------------------------------------------------------------
CAN_X: float = WORKSPACE_CENTRE[0]
CAN_R: float = 0.030
CAN_HALF_H: float = 0.055
CAN_DENSITY: float = 150.0
CAN_TOP_Z: float = sim.TABLE_TOP_Z + 2 * CAN_HALF_H
CAN_GRASP_Z: float = sim.TABLE_TOP_Z + CAN_HALF_H          # grasp the body at mid-height

LID_X: float = WORKSPACE_CENTRE[0] + 0.16                  # lid starts beside the can
LID_R: float = 0.040
LID_DISK_HALF_H: float = 0.006
LID_STEM_HALF: Tuple[float, float, float] = (0.010, 0.010, 0.016)
LID_DENSITY: float = 150.0
LID_REST_Z: float = sim.TABLE_TOP_Z + LID_DISK_HALF_H
# World z of the stem centre, where the arm grasps, relative to the lid body origin.
LID_STEM_CENTRE_DZ: float = LID_DISK_HALF_H + LID_STEM_HALF[2]


def lid_objects_xml() -> str:
    return f"""    <!-- lid-close task: a free-standing can and a separate lid (disk + stem) -->
    <body name="can" pos="{CAN_X} 0 {sim.TABLE_TOP_Z + CAN_HALF_H}">
      <freejoint name="can_free"/>
      <geom name="can_geom" type="cylinder" material="payload"
            size="{CAN_R} {CAN_HALF_H}" density="{CAN_DENSITY}" condim="4"
            friction="0.9 0.05 0.01" solref="0.006 1"/>
    </body>
    <body name="lid" pos="{LID_X} 0 {LID_REST_Z}">
      <freejoint name="lid_free"/>
      <geom name="lid_disk" type="cylinder" material="accent"
            size="{LID_R} {LID_DISK_HALF_H}" density="{LID_DENSITY}" condim="4"
            friction="1.0 0.05 0.01" solref="0.006 1"/>
      <geom name="lid_stem" type="box" material="actuator"
            pos="0 0 {LID_DISK_HALF_H + LID_STEM_HALF[2]}"
            size="{LID_STEM_HALF[0]} {LID_STEM_HALF[1]} {LID_STEM_HALF[2]}"
            density="{LID_DENSITY}" condim="4" friction="1.8 0.10 0.01"/>
    </body>"""


# ---------------------------------------------------------------------------
# Vision-guided ball-balancing task: the two arms jointly hold a tray (a flat board with
# side rails and a graspable post at each end) with a free ball on it, and must keep the
# ball centred by tilting the tray in a coordinated way - the ball rolls off the open ends
# if the tray is not actively levelled. This is a bimanual ball-and-beam closed with
# vision from a fixed camera.
# ---------------------------------------------------------------------------
TRAY_HALF: Tuple[float, float, float] = (0.070, 0.060, 0.006)   # board half-extents
TRAY_RAIL_HALF_Z: float = 0.010                                 # low side walls (keep y)
TRAY_TAB_HALF: Tuple[float, float, float] = (0.012, 0.016, 0.022)  # graspable end posts
# End posts stick out 30 mm beyond the board edge, so the grips land at world x ~ 0.10 and
# 0.30 - a comfortable reach distance for both arms - and the gripper palms clear the board.
TRAY_GRIP_OFFSET_X: float = TRAY_HALF[0] + 0.030
TRAY_DENSITY: float = 300.0
TRAY_REST_Z: float = sim.TABLE_TOP_Z + TRAY_HALF[2]
TRAY_TAB_CENTRE_DZ: float = TRAY_HALF[2] + TRAY_TAB_HALF[2]     # post centre above origin
BALL_R: float = 0.020
BALL_DENSITY: float = 220.0
BALL_START_X: float = 0.022                                     # starts off-centre on tray
BALL_REST_Z: float = sim.TABLE_TOP_Z + 2 * TRAY_HALF[2] + BALL_R


def tray_ball_objects_xml() -> str:
    cx, cy = WORKSPACE_CENTRE
    return f"""    <!-- ball-balancing task: a two-arm-held tray with a free ball on it -->
    <body name="tray" pos="{cx} {cy} {TRAY_REST_Z}">
      <freejoint name="tray_free"/>
      <geom name="tray_board" type="box" material="payload"
            size="{TRAY_HALF[0]} {TRAY_HALF[1]} {TRAY_HALF[2]}"
            density="{TRAY_DENSITY}" condim="4" friction="0.6 0.03 0.002" solref="0.006 1"/>
      <geom name="tray_rail_py" type="box" material="payload"
            pos="0 {TRAY_HALF[1] - 0.003} {TRAY_HALF[2] + TRAY_RAIL_HALF_Z}"
            size="{TRAY_HALF[0]} 0.003 {TRAY_RAIL_HALF_Z}" density="{TRAY_DENSITY}"/>
      <geom name="tray_rail_ny" type="box" material="payload"
            pos="0 {-(TRAY_HALF[1] - 0.003)} {TRAY_HALF[2] + TRAY_RAIL_HALF_Z}"
            size="{TRAY_HALF[0]} 0.003 {TRAY_RAIL_HALF_Z}" density="{TRAY_DENSITY}"/>
      <geom name="tray_tab_l" type="box" material="accent"
            pos="{-TRAY_GRIP_OFFSET_X} 0 {TRAY_TAB_CENTRE_DZ}"
            size="{TRAY_TAB_HALF[0]} {TRAY_TAB_HALF[1]} {TRAY_TAB_HALF[2]}"
            density="{TRAY_DENSITY}" condim="4" friction="1.8 0.10 0.01"/>
      <geom name="tray_tab_r" type="box" material="accent"
            pos="{TRAY_GRIP_OFFSET_X} 0 {TRAY_TAB_CENTRE_DZ}"
            size="{TRAY_TAB_HALF[0]} {TRAY_TAB_HALF[1]} {TRAY_TAB_HALF[2]}"
            density="{TRAY_DENSITY}" condim="4" friction="1.8 0.10 0.01"/>
    </body>
    <body name="ball" pos="{cx + BALL_START_X} {cy} {BALL_REST_Z}">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" material="lens" size="{BALL_R}"
            density="{BALL_DENSITY}" condim="4" friction="0.6 0.05 0.015" solref="0.006 1"/>
    </body>"""



def build_dual_arm_mjcf(objects_xml: Optional[str] = None) -> str:
    """Return the MJCF for the two-arm facing-each-other scene.

    `objects_xml` is the task-specific manipulated-object block (a free body or bodies);
    it defaults to the cooperative-lift dumbbell. Different bimanual tasks (lift, lid
    close, ...) share the identical two-arm rig and only swap this block.
    """
    if objects_xml is None:
        objects_xml = dumbbell_objects_xml()
    leg_x, leg_y = sim.TABLE_HALF_X - 0.07, sim.TABLE_HALF_Y - 0.07
    leg_half_h = (sim.TABLE_TOP_Z - 2 * sim.TABLE_TOP_HALF_THICK) / 2.0
    legs = "\n".join(
        f"""      <geom name="table_leg_{i}" type="box" material="steel"
            pos="{sx * leg_x} {sy * leg_y} {leg_half_h - sim.TABLE_TOP_Z}"
            size="0.025 0.025 {leg_half_h}"/>"""
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    )

    arm_l = sim.arm_body_mjcf(ARM_L_PREFIX, ARM_L_BASE[0], ARM_L_BASE[1], ARM_L_BASE_YAW)
    arm_r = sim.arm_body_mjcf(ARM_R_PREFIX, ARM_R_BASE[0], ARM_R_BASE[1], ARM_R_BASE_YAW)

    cx, cy = WORKSPACE_CENTRE
    return f"""
<mujoco model="dual_arm_bimanual">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{sim.SIM_TIMESTEP}" integrator="implicitfast" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.28 0.28 0.30" diffuse="0.38 0.38 0.38" specular="0.08 0.08 0.08"/>
    <rgba haze="0.15 0.20 0.26 1"/>
    <global azimuth="135" elevation="-24" offwidth="1024" offheight="1024"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.01" zfar="30"/>
  </visual>
{_SCENE_ASSETS}
  <worldbody>
    <light name="key" pos="0.6 -0.8 2.4" dir="-0.2 0.3 -1" directional="true"
           diffuse="0.42 0.42 0.42" specular="0.15 0.15 0.15" castshadow="true"/>
    <light name="fill" pos="-1.0 1.0 2.0" dir="0.4 -0.4 -1" directional="true"
           diffuse="0.25 0.25 0.28" castshadow="false"/>
    <geom name="floor" type="plane" size="6 6 0.05" material="floor"/>

    <body name="table" pos="{sim.TABLE_CENTER_X} 0 {sim.TABLE_TOP_Z}">
      <geom name="table_top" type="box" material="tabletop"
            pos="0 0 {-sim.TABLE_TOP_HALF_THICK}"
            size="{sim.TABLE_HALF_X} {sim.TABLE_HALF_Y} {sim.TABLE_TOP_HALF_THICK}"/>
{legs}
    </body>

    <!-- ============ left arm ============ -->
{arm_l}

    <!-- ============ right arm ============ -->
{arm_r}

    <!-- ============ task-specific manipulated object(s) ============ -->
{objects_xml}

    <camera name="front" pos="{cx} -1.25 1.35" xyaxes="1 0 0 0 0.55 0.84"/>
    <camera name="top" pos="{cx} 0 {sim.TABLE_TOP_Z + 1.0}" xyaxes="1 0 0 0 1 0" fovy="55"/>
  </worldbody>

  <actuator>
{sim.arm_actuator_mjcf(ARM_L_PREFIX)}
{sim.arm_actuator_mjcf(ARM_R_PREFIX)}
  </actuator>
</mujoco>
"""


@dataclass
class ArmHandle:
    """Everything needed to sense and command one of the two arms."""

    prefix: str
    base_yaw: float
    kinematics: sim.ArmKinematics
    arm_actuator_ids: np.ndarray
    finger_actuator_ids: np.ndarray
    finger_qpos_ids: np.ndarray

    @classmethod
    def build(cls, model: mujoco.MjModel, prefix: str, base_yaw: float) -> "ArmHandle":
        arm_actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}a{i}")
            for i in range(1, 7)], dtype=np.int32)
        finger_actuator_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
            for name in ("claw_left", "claw_right")], dtype=np.int32)
        finger_qpos_ids = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                prefix + name)]
            for name in ("finger_left", "finger_right")], dtype=np.int32)
        return cls(prefix, base_yaw, sim.ArmKinematics(model, prefix, base_yaw),
                   arm_actuator_ids, finger_actuator_ids, finger_qpos_ids)

    def command(self, data: mujoco.MjData, arm_angles: np.ndarray, claw_gap: float) -> None:
        data.ctrl[self.arm_actuator_ids] = arm_angles
        data.ctrl[self.finger_actuator_ids] = sim.finger_position(claw_gap)

    def tool_position(self, data: mujoco.MjData) -> np.ndarray:
        return self.kinematics.tool_pose(data)[0]


class DualArm:
    """Owns the two-arm MuJoCo model/data pair and both arm handles."""

    def __init__(self, objects_xml: Optional[str] = None) -> None:
        self.model = mujoco.MjModel.from_xml_string(build_dual_arm_mjcf(objects_xml))
        self.data = mujoco.MjData(self.model)
        self.left = ArmHandle.build(self.model, ARM_L_PREFIX, ARM_L_BASE_YAW)
        self.right = ArmHandle.build(self.model, ARM_R_PREFIX, ARM_R_BASE_YAW)
        payload_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                        "payload_free")
        self.payload_qpos_adr = (int(self.model.jnt_qposadr[payload_jid])
                                 if payload_jid != -1 else -1)
        self.reset()

    def free_body_qpos_adr(self, free_joint_name: str) -> int:
        """qpos address of a named free joint - for task objects other than the dumbbell
        (e.g. the lid task's can and lid)."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, free_joint_name)
        return int(self.model.jnt_qposadr[jid])

    @property
    def arms(self) -> Tuple[ArmHandle, ArmHandle]:
        return (self.left, self.right)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for arm in self.arms:
            self.data.qpos[arm.kinematics.qpos_ids] = HOME_POSE
            arm.command(self.data, np.asarray(HOME_POSE), sim.OPEN_GAP)
            self.data.qpos[arm.finger_qpos_ids] = sim.finger_position(sim.OPEN_GAP)
        mujoco.mj_forward(self.model, self.data)

    def payload_position(self) -> np.ndarray:
        return self.data.qpos[self.payload_qpos_adr: self.payload_qpos_adr + 3].copy()

    def payload_rotation(self) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        return self.data.xmat[bid].reshape(3, 3).copy()

    def grasp_ends(self) -> Tuple[np.ndarray, np.ndarray]:
        """World positions of the two end-rod grasp points, from the payload's live pose -
        the left arm takes the -x rod, the right arm the +x rod. Uses the body's actual
        rotation so the grasp tracks the rod even if the payload has shifted or tilted."""
        centre = self.payload_position()
        R = self.payload_rotation()
        left_end = centre + R @ np.array([-PAYLOAD_ROD_OFFSET, 0.0, 0.0])
        right_end = centre + R @ np.array([PAYLOAD_ROD_OFFSET, 0.0, 0.0])
        return left_end, right_end


def _settle(dual: DualArm, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(dual.model, dual.data)


def _selftest() -> None:
    print("building dual-arm scene ...")
    dual = DualArm()
    m = dual.model
    print(f"  model: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, "
          f"{m.ngeom} geoms")
    print(f"  left arm  base {ARM_L_BASE} yaw {ARM_L_BASE_YAW:.2f} rad")
    print(f"  right arm base {ARM_R_BASE} yaw {ARM_R_BASE_YAW:.2f} rad")
    print(f"  shared workspace centre {WORKSPACE_CENTRE}")

    # Drive each arm's tool frame to a hover pose just inboard of the shared centre,
    # pointing straight down, at a height that clears the central payload. A pure reach
    # test: the arms converge on the common central region from opposite sides, proving
    # each independently covers it - the precondition for any two-hand task. (Grasping the
    # payload itself, where the two tools meet at its ends, comes in the next phase.)
    cx, cy = WORKSPACE_CENTRE
    hover = 0.22
    targets = {
        "left": np.array([cx - 0.10, cy, sim.TABLE_TOP_Z + hover]),
        "right": np.array([cx + 0.10, cy, sim.TABLE_TOP_Z + hover]),
    }
    arms = {"left": dual.left, "right": dual.right}

    print("\ndriving both arms to the shared centre ...")
    dt = 1.0 / 60.0
    step_index = 0
    for _ in range(int(2.5 / dt)):
        for _ in range(int(round(dt / sim.SIM_TIMESTEP))):
            if step_index % sim.CONTROL_DECIMATION == 0:
                for name, arm in arms.items():
                    angles = arm.kinematics.solve(dual.data, targets[name], 0.0)
                    arm.command(dual.data, angles, sim.OPEN_GAP)
            mujoco.mj_step(dual.model, dual.data)
            step_index += 1

    ok = True
    for name, arm in arms.items():
        reached = arm.tool_position(dual.data)
        err = float(np.linalg.norm(reached - targets[name]))
        status = "OK " if err < 0.02 else "FAR"
        if err >= 0.02:
            ok = False
        print(f"    {name:<5} arm  target {targets[name].round(3)}  "
              f"reached {reached.round(3)}  err {err * 1000:5.1f} mm  {status}")

    # Stability: nothing should have gone to NaN or flown off the table.
    qpos = dual.data.qpos
    stable = bool(np.all(np.isfinite(qpos)))
    payload_z = dual.payload_position()[2]
    payload_on_table = payload_z > sim.TABLE_TOP_Z - 0.05
    print(f"\n  simulation finite: {stable} | payload still on table: {payload_on_table}")
    print(f"  mean reach error: "
          f"{np.mean([np.linalg.norm(arms[n].tool_position(dual.data) - targets[n]) for n in arms]) * 1000:.1f} mm")
    print("\nselftest complete." if (ok and stable and payload_on_table)
          else "\nselftest FAILED.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="headless check that both arms reach the shared centre")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        import time
        import mujoco.viewer
        dual = DualArm()
        print("[dual_arm] opening viewer of the two-arm rig; close the window to quit.")
        with mujoco.viewer.launch_passive(dual.model, dual.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(dual.model, dual.data)
                viewer.sync()
                time.sleep(sim.SIM_TIMESTEP)


if __name__ == "__main__":
    main()
