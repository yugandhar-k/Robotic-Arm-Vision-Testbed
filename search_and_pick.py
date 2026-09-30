"""search_and_pick.py - Fully dynamic, vision-driven pick-and-place.

Unlike `run_sim.py`'s `ScanController` (used by the classification benchmark and
`demo.py`), which reads each object's true position directly off the simulator state to
decide where to move, this demo's arm is never told where anything is. The vision source
used to *find* a candidate is selectable (`--view`, or the `V` key), matching the same
three cameras the classification benchmark compares:

  - `eye_in_hand`: SEARCHES the table with the claw's own camera, sweeping a raster of
    waypoints over the known *workspace* extent (a fixed property of the hardware, like
    the table's own size - not per-object information) until it sees a candidate object.
  - `fixed_overhead` / `fixed_angled`: PARKS the arm clear of the object field, then
    looks *once* from the fixed, wide-angle camera, which sees the whole field in a
    single shot (`perception.detect_all`) - no sweep needed, at the cost of being
    self-occluded whenever the arm happens to be in the way (the same self-occlusion
    trade-off the classification benchmark studies, now showing up as a search-time cost
    rather than a classification-accuracy one).

Whichever camera found the rough candidate, the rest of the pipeline is identical:

  1. CONFIRMS the candidate by moving the claw directly above its rough position and
     looking again with the eye-in-hand camera, refining the estimate the same way the
     classification benchmark's own "settle" pose already does.
  2. Localises it in 3-D from a single RGB-D-style view (`perception.detect`/
     `detect_all`): a segmentation-plus-depth back-projection stands in for a real depth
     camera, and an image-moments principal axis stands in for a real grasp-orientation
     estimator.
  3. Reaches, grasps, and - checking the *gripper's own encoder position* to confirm
     something was actually caught, not vision again - lifts it.
  4. Carries it to a basket at a fixed, known location (infrastructure, not a pick
     target) and releases it.
  5. Resumes searching for the next object, until a full search turns up nothing.

Nothing here reads `SceneObject.position`, `SceneObject.object_class`, or any other
per-object ground truth - see perception.py's module docstring for exactly what
knowledge the perception step is and is not allowed to use.

Usage
-----
    python search_and_pick.py
    python search_and_pick.py --seed 7
    python search_and_pick.py --view fixed_overhead
    python search_and_pick.py --view fixed_angled

Keys
----
  SPACE   pause / resume
  V       switch search view: eye-in-hand -> fixed-overhead -> fixed-angled -> ...
  R       reshuffle the object layout and start over
  +/-     speed up / slow down
  ESC     quit

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import glfw
import mujoco
import numpy as np

import perception
import run_sim as sim
from src.utils import RANDOM_SEED

SEARCH_VIEWS: Tuple[str, ...] = ("eye_in_hand", "fixed_overhead", "fixed_angled")
# The MuJoCo camera each search view looks through when finding a rough candidate - see
# run_sim.build_mjcf. The claw camera is always used for the close-up confirm step
# regardless of search view.
SEARCH_VIEW_CAMERA_NAME: Dict[str, str] = {
    "eye_in_hand": "claw",
    "fixed_overhead": "overhead",
    "fixed_angled": "angled",
}

# ---------------------------------------------------------------------------
# Search sweep
# ---------------------------------------------------------------------------

# Bounds of the region the arm sweeps while searching - the reachable tabletop area in
# front of the base, known the same way a real deployment knows its own workcell extent.
# Deliberately not derived from `run_sim.FIELD_X`/`FIELD_Y` (the actual object-slot grid),
# so nothing here is implicitly keyed to where objects happen to spawn.
SEARCH_X_RANGE: Tuple[float, float] = (0.14, 0.54)
SEARCH_Y_RANGE: Tuple[float, float] = (-0.28, 0.28)
SEARCH_STEP: float = 0.10           # waypoint spacing [m]; less than the camera's
                                    # footprint at SEARCH_HOVER_HEIGHT, for overlap
SEARCH_HOVER_HEIGHT: float = 0.30   # well above the tallest resting object (~80 mm) plus
                                    # the claw/finger assembly's own extent below the
                                    # grasp frame - measured to matter: at HOVER_HEIGHT
                                    # (the confirm/grasp framing height, 150 mm) the arm
                                    # clipped resting objects while transiting between
                                    # search waypoints often enough to launch them off the
                                    # table (see search_and_pick.py's Limitations section)
CONFIRM_HOVER_HEIGHT: float = sim.HOVER_HEIGHT

MIN_CONTOUR_AREA_PX: float = 250.0    # for the claw camera's close-up sweep/confirm view:
                                     # stricter than perception's default, since a hit
                                     # this small at close range is more likely noise
                                     # than an object
FIXED_VIEW_MIN_CONTOUR_AREA_PX: float = 60.0   # for a fixed camera's whole-table view:
                                              # the same physical object is much farther
                                              # away and covers far fewer pixels (measured
                                              # 80-230 px2 from the angled camera) than the
                                              # claw camera's close-up sweep, so the 250 px2
                                              # threshold above filtered out real objects,
                                              # not just noise
GRASP_WIDTH_MIN: float = 0.012
GRASP_WIDTH_MAX: float = 0.075
GRASP_SQUEEZE_MARGIN: float = 0.004  # extra squeeze beyond perception.py's own estimate and
                                     # `run_sim.grasp_gap`'s own built-in 9 mm squeeze.
                                     # Kept small: an estimate with real error already gets
                                     # squeezed past its true width often enough that a
                                     # large extra margin mainly risks a violent contact
                                     # response (a launched object) rather than a firmer
                                     # grip - see README's Limitations section.
GRASP_CONFIRM_MARGIN: float = 0.004  # fingers must stop at least this far short of fully
                                     # closed to count as "something is between them" -
                                     # read from the gripper's own joint encoder
                                     # (`Workspace.claw_gap`), not vision

# Per-phase durations [s], in the same spirit as `run_sim.PICK_SEQUENCE`.
DURATION_SEARCH_MOVE: float = 0.9
DURATION_SEARCH_LOOK: float = 0.35
DURATION_CONFIRM_MOVE: float = 0.9
DURATION_CONFIRM_LOOK: float = 0.35
DURATION_APPROACH: float = 0.7
DURATION_DESCEND: float = 0.8
DURATION_CLOSE: float = 1.1        # slow, gentle squeeze - a fast close on an imprecise,
                                   # vision-estimated width risks a violent contact
                                   # response, especially on curved (e.g. ellipsoid)
                                   # geometry; see README's Limitations section
DURATION_LIFT: float = 0.9
DURATION_TRANSPORT: float = 1.4
DURATION_LOWER: float = 0.7
DURATION_RELEASE: float = 0.5
DURATION_VERIFY_SETTLE: float = 0.8  # let physics settle after release before checking
                                     # whether the object actually stayed in the basket -
                                     # checking immediately risks catching it mid-bounce
MIN_AREA_INCREASE_PX: float = 25.0   # minimum basket-footprint pixel-area increase (see
                                     # SearchAndPlaceController._basket_masked_area) to
                                     # count one release as a real addition, not noise -
                                     # comfortably below the smallest single-object area
                                     # measured (~39 px2) at the verification camera's
                                     # resolution and distance
DURATION_RETREAT: float = 0.8
DURATION_RISE: float = 0.5          # straight-up climb to search altitude before any
                                    # lateral search-sweep motion (see _go_to_next_waypoint)
DURATION_PARK_MOVE: float = 1.0     # return to the parked pose before a fixed-camera look
DURATION_PARK_LOOK: float = 0.35    # settle before the fixed camera's single-shot look

EYE_IN_HAND_EMPTY_SWEEPS_LIMIT: int = 2   # each "sweep" is a full 30-waypoint raster, so
                                          # two of them finding nothing is already ~60
                                          # separate looks - a reliable "table is clear"
FIXED_VIEW_EMPTY_LOOKS_LIMIT: int = 4     # each "look" here is a single snapshot, and the
                                          # parked arm's own body can transiently occlude
                                          # part of the field from a fixed vantage point in
                                          # a way an eye-in-hand sweep never has to contend
                                          # with (the self-occlusion cost this repository's
                                          # classification benchmark already studies,
                                          # showing up here as a search-time one instead) -
                                          # a higher retry count absorbs that transient cost
                                          # before concluding the table is actually clear


def _build_search_waypoints() -> List[Tuple[float, float]]:
    """A boustrophedon (serpentine) raster over the search region.

    Serpentine rather than row-major so consecutive waypoints are always adjacent - no
    long diagonal jumps back to the start of the next row.
    """
    xs = np.arange(SEARCH_X_RANGE[0], SEARCH_X_RANGE[1] + 1e-9, SEARCH_STEP)
    ys = np.arange(SEARCH_Y_RANGE[0], SEARCH_Y_RANGE[1] + 1e-9, SEARCH_STEP)
    waypoints: List[Tuple[float, float]] = []
    for row, x in enumerate(xs):
        column_ys = ys if row % 2 == 0 else ys[::-1]
        waypoints.extend((float(x), float(y)) for y in column_ys)
    return waypoints


BASKET_XY: Tuple[float, float] = (sim.BASKET_WORLD_X, sim.BASKET_WORLD_Y)
# Small offsets cycled through on successive drops, so items don't land exactly on top of
# each other and knock a previous item back out of the basket while it settles.
BASKET_DROP_OFFSETS: Tuple[Tuple[float, float], ...] = (
    (-0.025, -0.025), (0.025, -0.025), (-0.025, 0.025), (0.025, 0.025),
)
# Half-width of the square region around the basket treated as "already collected, not a
# search candidate" - the wall's own outer half-extent plus the drop offsets' own spread
# plus a small margin.
BASKET_MARGIN_HALF: float = sim.BASKET_INNER_HALF + sim.BASKET_WALL_THICK + 0.03

ClawCommand = sim.ClawCommand
_smoothstep = sim._smoothstep
_wrap = sim._wrap


@dataclass
class _Motion:
    """One smoothly-interpolated move: from `start` to `end` pose over `duration`."""

    start_pos: np.ndarray
    end_pos: np.ndarray
    start_yaw: float
    end_yaw: float
    start_gap: float
    end_gap: float
    duration: float

    def at(self, alpha: float) -> ClawCommand:
        alpha = _smoothstep(alpha)
        position = self.start_pos + (self.end_pos - self.start_pos) * alpha
        yaw = self.start_yaw + _wrap(self.end_yaw - self.start_yaw) * alpha
        gap = self.start_gap + (self.end_gap - self.start_gap) * alpha
        return ClawCommand(position, yaw, gap)


@dataclass
class SearchAndPlaceController:
    """State machine: search -> confirm -> approach -> descend -> close -> lift ->
    transport -> lower -> release -> retreat -> (back to search).

    Every target pose fed to `_Motion` comes from either a waypoint (a fixed, known
    sweep pattern) or a `perception.Detection` (a live vision estimate) - never from
    `workspace.objects`.
    """

    workspace: sim.Workspace
    claw_camera: sim.SimCamera
    verification_camera: sim.SimCamera      # always the fixed-angled camera, regardless
                                            # of search_view (the overhead camera's frame
                                            # cuts the basket off at the edge - see
                                            # run_sim.BASKET_WORLD_Y - the angled camera
                                            # sees the whole thing) - used only to check,
                                            # after
                                            # every release, whether the object actually
                                            # stayed in the basket (see "verify_placement")
    search_view: str = "eye_in_hand"        # eye_in_hand | fixed_overhead | fixed_angled
    fixed_camera: Optional[sim.SimCamera] = None   # required unless search_view is eye_in_hand
    phase: str = "search_move"
    phase_start: float = 0.0
    waypoint_index: int = 0
    waypoints: List[Tuple[float, float]] = field(default_factory=_build_search_waypoints)
    motion: Optional[_Motion] = None
    anchor_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    anchor_yaw: float = 0.0
    park_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rough_position: Optional[np.ndarray] = None
    rough_detection: Optional[perception.Detection] = None
    detection: Optional[perception.Detection] = None
    target_yaw: float = 0.0
    object_width: float = sim.OPEN_GAP
    grasp_height: float = sim.TABLE_TOP_Z
    objects_collected: int = 0  # incremented only when "verify_placement" measures a real
                                # pixel-area increase in the basket after a release (see
                                # _basket_masked_area) - not simply once the claw opens.
                                # Does not currently detect a *later* drop knocking an
                                # *earlier*, already-counted item back out - see README's
                                # Limitations.
    attempts: int = 0           # release attempts made, successful or not - drives the
                                # basket drop-slot cycling; independent of
                                # objects_collected so a failed/undone placement doesn't
                                # throw off which slot the next drop aims for
    placements_lost: int = 0    # released, but the basket's occupied area didn't measurably
                                # increase afterwards - not reflected in objects_collected
    last_drop_xy: Optional[Tuple[float, float]] = None
    basket_area_before_drop: int = 0
    sweeps_with_nothing_found: int = 0
    status: str = "starting up..."

    def __post_init__(self) -> None:
        assert (self.search_view == "eye_in_hand") == (self.fixed_camera is None), (
            "fixed_camera is required for, and only for, a fixed search_view"
        )
        self.candidate_geom_ids = perception.free_body_geom_ids(self.workspace.model)
        position, _ = self.workspace.kinematics.tool_pose(self.workspace.data)
        self.anchor_pos = position.copy()
        # Wherever the arm is when the controller is built (fresh construction, or right
        # after a reshuffle) is a safe, clear-of-the-field pose - used as the "go home"
        # target a fixed-camera search returns to between looks, so its own body doesn't
        # occlude the object field it is about to look at.
        self.park_pos = position.copy()
        self._resume_search(now=0.0)

    # -- helpers --------------------------------------------------------------------
    def _current_waypoint_xy(self) -> Tuple[float, float]:
        return self.waypoints[self.waypoint_index % len(self.waypoints)]

    def _start_motion(self, now: float, end_pos: np.ndarray, end_yaw: float,
                      end_gap: float, duration: float, phase: str) -> None:
        self.motion = _Motion(self.anchor_pos.copy(), end_pos, self.anchor_yaw, end_yaw,
                              self._last_gap, end_gap, duration)
        self.phase = phase
        self.phase_start = now

    def _resume_search(self, now: float) -> None:
        """Go back to however this controller's `search_view` looks for a candidate, and
        try again - the single dispatch point every phase that needs to keep searching
        (initial start-up, a failed grasp, a completed placement, a lost candidate) calls,
        so switching `search_view` live (the `V` key) takes effect on the very next look.
        """
        if self.search_view == "eye_in_hand":
            self._go_to_next_waypoint(now)
        else:
            self._go_to_park_and_look(now)

    def _go_to_park_and_look(self, now: float) -> None:
        # Same two-stage reasoning as `_go_to_next_waypoint`: rise straight up to a safe
        # altitude at the *current* (x, y) before moving laterally toward the park pose -
        # going there directly interpolates a diagonal path from wherever we just were
        # (often right next to the basket, at a low release height) that can clip an
        # object along the way.
        self.status = f"returning to park position - collected {self.objects_collected}"
        current_xy = self.anchor_pos[:2].copy()
        safe_hover = sim.inspection_target_for(
            np.array([current_xy[0], current_xy[1], sim.TABLE_TOP_Z]),
            SEARCH_HOVER_HEIGHT, 0.0)
        self._start_motion(now, safe_hover, 0.0, sim.OPEN_GAP, DURATION_RISE,
                          "rise_before_park")

    def _go_to_next_waypoint(self, now: float) -> None:
        # Rise straight up to search altitude *before* moving laterally, rather than
        # letting the lateral move's own interpolation climb gradually while still over
        # wherever we just were (the basket, or a just-missed grasp) - a diagonal climb
        # measured to clip a just-placed object still settling in the basket often enough
        # to launch it off the table (see the Limitations section).
        self.status = (f"searching ({self.waypoint_index % len(self.waypoints) + 1}/"
                       f"{len(self.waypoints)}) - collected {self.objects_collected}")
        current_xy = self.anchor_pos[:2].copy()
        safe_hover = sim.inspection_target_for(
            np.array([current_xy[0], current_xy[1], sim.TABLE_TOP_Z]),
            SEARCH_HOVER_HEIGHT, 0.0)
        self._start_motion(now, safe_hover, 0.0, sim.OPEN_GAP, DURATION_RISE,
                          "rise_to_search_height")

    def _go_to_confirm(self, now: float, detection: perception.Detection) -> None:
        # Two-stage, same reasoning as `_go_to_next_waypoint`: move laterally at a safe,
        # clear-of-every-object altitude first, then descend straight down over the
        # candidate - rather than interpolating straight from wherever we currently are
        # (which, coming back from a fixed-camera park position ~1.6 m up, or from a
        # previous grasp attempt elsewhere, is a large, diagonal move that could clip
        # something resting on the table along the way.
        #
        # The detection itself is kept, not just its position: for a fixed search view
        # the arm never re-observes with the eye-in-hand camera (see the phase handling
        # for "confirm_move") - the fixed camera's own reading is what plans the grasp,
        # so search, localisation *and* picking all come from that one camera, never the
        # claw's.
        self.rough_position = detection.position
        self.rough_detection = detection
        safe_hover = sim.inspection_target_for(detection.position, SEARCH_HOVER_HEIGHT, 0.0)
        self.status = "candidate spotted - moving in to confirm..."
        self._start_motion(now, safe_hover, 0.0, sim.OPEN_GAP, DURATION_CONFIRM_MOVE,
                          "confirm_transit")

    def _begin_grasp_sequence(self, now: float, detection: perception.Detection) -> None:
        self.detection = detection
        yaw = detection.yaw
        width = float(np.clip(detection.width_m, GRASP_WIDTH_MIN, GRASP_WIDTH_MAX))
        self.target_yaw = yaw
        self.object_width = width
        # The detected point is the object's visible *top* surface; its centre (what a
        # ground-truth-driven controller would descend to) sits roughly midway between
        # that and the table, using the table's own known height - infrastructure, not
        # per-object information.
        self.grasp_height = float(
            np.clip((detection.position[2] + sim.TABLE_TOP_Z) / 2.0,
                   sim.TABLE_TOP_Z + 0.008, sim.TABLE_TOP_Z + 0.05)
        )
        hover = sim.inspection_target_for(detection.position, CONFIRM_HOVER_HEIGHT, yaw)
        self.status = f"confirmed - approaching (est. width {width * 1000:.0f} mm)"
        self._start_motion(now, hover, yaw, sim.OPEN_GAP, DURATION_APPROACH, "approach")

    def _column(self) -> np.ndarray:
        assert self.detection is not None
        return self.detection.position[:2]

    @property
    def _last_gap(self) -> float:
        return self.motion.at(1.0).gap if self.motion is not None else sim.OPEN_GAP

    def _still_gripping(self) -> bool:
        """Read the gripper's own joint encoder to check something is still between the
        finger pads - a real sensor signal, not a vision estimate. Checked after every
        phase that could lose the object: a failed grasp, or a slip while lifting or
        carrying it.
        """
        return self.workspace.claw_gap > sim.MIN_GRASP_GAP + GRASP_CONFIRM_MARGIN

    def _basket_drop_xy(self) -> Tuple[float, float]:
        offset = BASKET_DROP_OFFSETS[self.attempts % len(BASKET_DROP_OFFSETS)]
        return BASKET_XY[0] + offset[0], BASKET_XY[1] + offset[1]

    def _basket_masked_area(self) -> int:
        """How many pixels of the basket's footprint, as seen by the verification
        camera right now, belong to some free body.

        Used as a before/after check around one release (see "verify_placement"): a
        pixel-area increase is a far more reliable "did this drop actually land and stay"
        signal than trying to count distinct objects in a pile, since two objects resting
        close together routinely merge into a single contour in a 2-D view - measured:
        four objects at the basket's own drop-offset spacing rendered as one blob, which
        would have undercounted every pick after the first.
        """
        center = np.array([BASKET_XY[0], BASKET_XY[1], sim.TABLE_TOP_Z])
        half_width = sim.BASKET_INNER_HALF + sim.BASKET_WALL_THICK
        return perception.masked_area_in_region(self.verification_camera,
                                                self.candidate_geom_ids, center, half_width)

    @staticmethod
    def _discard_basket_detections(
        detections: List[perception.Detection]) -> List[perception.Detection]:
        """Drop any detection that falls inside the basket's own footprint.

        A fixed, wide camera sees the whole table - including the basket - in one shot,
        so without this an already-collected item sitting in the basket is just as valid
        a "candidate" as one still on the table, wasting a cycle re-detecting (and
        possibly re-grasping) it. The basket's footprint is the same known, fixed
        infrastructure its own drop target already uses - not per-object information.
        """
        half = BASKET_MARGIN_HALF
        return [
            d for d in detections
            if not (abs(d.position[0] - BASKET_XY[0]) < half
                   and abs(d.position[1] - BASKET_XY[1]) < half)
        ]

    def _abandon_and_resume_search(self, now: float) -> None:
        """Retreat to a safe hover above wherever the claw currently is, open the
        gripper, and resume the search sweep - used whenever the gripper encoder shows
        the object was never captured, or was lost partway through the carry.
        """
        current_xy = self.anchor_pos[:2].copy()
        hover = sim.inspection_target_for(
            np.array([current_xy[0], current_xy[1], sim.TABLE_TOP_Z]),
            CONFIRM_HOVER_HEIGHT, 0.0)
        self.detection = None
        self._start_motion(now, hover, 0.0, sim.OPEN_GAP, DURATION_RETREAT,
                          "retreat_after_miss")

    # -- main update ------------------------------------------------------------------
    def update(self, now: float, speed: float = 1.0) -> ClawCommand:
        elapsed = (now - self.phase_start) * max(0.05, speed)
        duration = self.motion.duration if self.motion is not None else 0.0
        alpha = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
        command = self.motion.at(alpha) if self.motion is not None else ClawCommand(
            self.anchor_pos.copy(), self.anchor_yaw, sim.OPEN_GAP
        )

        if alpha < 1.0:
            return command

        self.anchor_pos, self.anchor_yaw = command.position.copy(), command.yaw

        if self.phase == "rise_to_search_height":
            x, y = self._current_waypoint_xy()
            hover = sim.inspection_target_for(np.array([x, y, sim.TABLE_TOP_Z]),
                                              SEARCH_HOVER_HEIGHT, 0.0)
            self._start_motion(now, hover, 0.0, sim.OPEN_GAP, DURATION_SEARCH_MOVE,
                              "search_move")
        elif self.phase == "search_move":
            self._start_motion(now, self.anchor_pos.copy(), self.anchor_yaw,
                              self._last_gap, DURATION_SEARCH_LOOK, "search_look")
        elif self.phase == "search_look":
            detection = perception.detect(self.claw_camera, self.candidate_geom_ids,
                                          min_area_px=MIN_CONTOUR_AREA_PX)
            if detection is not None:
                self.sweeps_with_nothing_found = 0
                self._go_to_confirm(now, detection)
            else:
                self.waypoint_index += 1
                if self.waypoint_index % len(self.waypoints) == 0:
                    self.sweeps_with_nothing_found += 1
                    self.status = ("table clear - idle" if self.sweeps_with_nothing_found
                                   >= EYE_IN_HAND_EMPTY_SWEEPS_LIMIT
                                   else "sweep complete, nothing found - re-scanning")
                self._go_to_next_waypoint(now)
        elif self.phase == "rise_before_park":
            self._start_motion(now, self.park_pos.copy(), 0.0, sim.OPEN_GAP,
                              DURATION_PARK_MOVE, "park_move")
        elif self.phase == "park_move":
            self._start_motion(now, self.anchor_pos.copy(), self.anchor_yaw,
                              self._last_gap, DURATION_PARK_LOOK, "park_look")
        elif self.phase == "park_look":
            if self.fixed_camera is None:
                # search_view was switched (the V key) to eye_in_hand while this look was
                # already in flight, so the fixed camera it was headed for no longer
                # applies. Re-route through _resume_search, which dispatches on the
                # *current* search_view - safe and correct regardless of what the view
                # was when this phase started.
                self._resume_search(now)
                return command
            detections = perception.detect_all(self.fixed_camera, self.candidate_geom_ids,
                                               min_area_px=FIXED_VIEW_MIN_CONTOUR_AREA_PX)
            detections = self._discard_basket_detections(detections)
            if detections:
                self.sweeps_with_nothing_found = 0
                self._go_to_confirm(now, detections[0])
            else:
                self.sweeps_with_nothing_found += 1
                self.status = ("table clear - idle" if self.sweeps_with_nothing_found
                               >= FIXED_VIEW_EMPTY_LOOKS_LIMIT
                               else "no objects seen - looking again")
                self._go_to_park_and_look(now)
        elif self.phase == "confirm_transit":
            column = self.rough_position[:2] if self.rough_position is not None \
                else self.anchor_pos[:2]
            hover = sim.inspection_target_for(
                np.array([column[0], column[1], sim.TABLE_TOP_Z]),
                CONFIRM_HOVER_HEIGHT, 0.0)
            self._start_motion(now, hover, 0.0, sim.OPEN_GAP, DURATION_CONFIRM_MOVE,
                              "confirm_move")
        elif self.phase == "confirm_move":
            if self.search_view == "eye_in_hand":
                self._start_motion(now, self.anchor_pos.copy(), self.anchor_yaw,
                                  self._last_gap, DURATION_CONFIRM_LOOK, "confirm_look")
            else:
                # A fixed camera's view doesn't change with the arm's position, so a
                # second look from the eye-in-hand camera here would mean detection,
                # localisation and grasp planning were no longer coming from the fixed
                # camera alone - the whole point of this search view. The reading from
                # `park_look` is used directly.
                assert self.rough_detection is not None
                self._begin_grasp_sequence(now, self.rough_detection)
        elif self.phase == "confirm_look":
            detection = perception.detect(self.claw_camera, self.candidate_geom_ids,
                                          min_area_px=MIN_CONTOUR_AREA_PX)
            if detection is not None:
                self._begin_grasp_sequence(now, detection)
            else:
                self.status = "candidate lost on closer look - resuming search"
                self.waypoint_index += 1
                self._resume_search(now)
        elif self.phase == "approach":
            column = self._column()
            grasp = np.array([column[0], column[1], self.grasp_height])
            self._start_motion(now, grasp, self.target_yaw, sim.OPEN_GAP,
                              DURATION_DESCEND, "descend")
        elif self.phase == "descend":
            closed_gap = sim.grasp_gap(self.object_width) - GRASP_SQUEEZE_MARGIN
            closed_gap = max(sim.MIN_GRASP_GAP, closed_gap)
            self._start_motion(now, self.anchor_pos.copy(), self.target_yaw, closed_gap,
                              DURATION_CLOSE, "close")
        elif self.phase == "close":
            if self._still_gripping():
                self.status = "grasp confirmed (gripper encoder) - lifting"
                column = self._column()
                lifted = np.array([column[0], column[1],
                                  sim.TABLE_TOP_Z + sim.LIFT_HEIGHT])
                self._start_motion(now, lifted, self.target_yaw, self._last_gap,
                                  DURATION_LIFT, "lift")
            else:
                self.status = "grasp missed (gripper closed on nothing) - resuming search"
                self._abandon_and_resume_search(now)
        elif self.phase == "lift":
            if self._still_gripping():
                self.last_drop_xy = self._basket_drop_xy()
                self.basket_area_before_drop = self._basket_masked_area()
                basket_hover = np.array([self.last_drop_xy[0], self.last_drop_xy[1],
                                        sim.TABLE_TOP_Z + sim.LIFT_HEIGHT])
                self.status = "carrying to basket..."
                self._start_motion(now, basket_hover, self.target_yaw, self._last_gap,
                                  DURATION_TRANSPORT, "transport")
            else:
                self.status = "object slipped while lifting - resuming search"
                self._abandon_and_resume_search(now)
        elif self.phase == "transport":
            if self._still_gripping():
                assert self.last_drop_xy is not None
                drop = np.array([self.last_drop_xy[0], self.last_drop_xy[1],
                                sim.TABLE_TOP_Z + sim.BASKET_DROP_HEIGHT])
                self._start_motion(now, drop, self.target_yaw, self._last_gap,
                                  DURATION_LOWER, "lower")
            else:
                self.status = "object slipped in transit - resuming search"
                self._abandon_and_resume_search(now)
        elif self.phase == "lower":
            self.status = "releasing into basket"
            self._start_motion(now, self.anchor_pos.copy(), self.target_yaw,
                              sim.OPEN_GAP, DURATION_RELEASE, "release")
        elif self.phase == "release":
            self.status = "checking the drop landed..."
            self._start_motion(now, self.anchor_pos.copy(), self.anchor_yaw,
                              self._last_gap, DURATION_VERIFY_SETTLE, "verify_placement")
        elif self.phase == "verify_placement":
            self.attempts += 1
            area_now = self._basket_masked_area()
            if area_now - self.basket_area_before_drop >= MIN_AREA_INCREASE_PX:
                self.objects_collected += 1
                self.status = f"collected {self.objects_collected} - resuming search"
            else:
                self.placements_lost += 1
                self.status = ("object did not stay in the basket (not counted) - "
                               "resuming search")
            self.sweeps_with_nothing_found = 0
            self.detection = None
            self._resume_search(now)
        elif self.phase == "retreat_after_miss":
            self.waypoint_index += 1
            self._resume_search(now)
        else:
            self._resume_search(now)

        return command


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------


VIEW_LABELS: Dict[str, str] = {
    "eye_in_hand": "EYE-IN-HAND",
    "fixed_overhead": "FIXED-OVERHEAD",
    "fixed_angled": "FIXED-ANGLED",
}
# Phases in which the fixed camera (not the claw) is the one doing the looking - the
# panel follows whichever camera is actually searching right now.
_FIXED_LOOK_PHASES = frozenset({"park_move", "park_look"})


def _make_controller(workspace: sim.Workspace, claw_camera: sim.SimCamera,
                    fixed_cameras: Dict[str, sim.SimCamera],
                    view: str) -> SearchAndPlaceController:
    fixed_camera = None if view == "eye_in_hand" else fixed_cameras[view]
    return SearchAndPlaceController(
        workspace, claw_camera, fixed_cameras["fixed_angled"],
        search_view=view, fixed_camera=fixed_camera,
    )


def run_demo(seed: int, view: str) -> None:
    workspace = sim.Workspace(seed)
    kinematics = workspace.kinematics
    claw_camera = sim.SimCamera(workspace, "claw")
    fixed_cameras = {
        "fixed_overhead": sim.SimCamera(workspace, "overhead"),
        "fixed_angled": sim.SimCamera(workspace, "angled"),
    }
    controller = _make_controller(workspace, claw_camera, fixed_cameras, view)
    viewer = sim.PictureInPictureViewer(
        workspace, "Search and Pick - Vision-Driven Pick-and-Place Demo"
    )
    # The basket lives in its own geom group, hidden by default so it never appears in a
    # classification-benchmark frame (see run_sim.BASKET_GEOM_GROUP) - re-enable it just
    # for this demo's own viewer so it's visible on screen.
    viewer.options.geomgroup[sim.BASKET_GEOM_GROUP] = 1
    print("[search_and_pick] window opened - titled 'Search and Pick - Vision-Driven "
         "Pick-and-Place Demo'. If you don't see it, check other monitors/desktops and "
         "the taskbar; alt-tab should find it.")
    print(f"[search_and_pick] seed {seed} | {len(workspace.objects)} objects on the "
         "table, positions never read by the controller")
    print(f"[search_and_pick] search view: {view} (V cycles through "
         f"{', '.join(SEARCH_VIEWS)})")
    print("[search_and_pick] keys: SPACE pause | V search view | R reshuffle | "
         "+/- speed | ESC quit")

    state = {"paused": False, "speed": 1.0, "seed": seed, "view_index": SEARCH_VIEWS.index(view)}

    def on_key(key: int) -> None:
        nonlocal controller
        if key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_V:
            state["view_index"] = (state["view_index"] + 1) % len(SEARCH_VIEWS)
            new_view = SEARCH_VIEWS[state["view_index"]]
            controller.search_view = new_view
            controller.fixed_camera = (None if new_view == "eye_in_hand"
                                       else fixed_cameras[new_view])
        elif key == glfw.KEY_R:
            state["seed"] += 1
            workspace.reshuffle(state["seed"])
            kinematics._last_solution = None
            controller = _make_controller(workspace, claw_camera, fixed_cameras,
                                          SEARCH_VIEWS[state["view_index"]])
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            state["speed"] = min(4.0, state["speed"] * 1.25)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            state["speed"] = max(0.2, state["speed"] / 1.25)

    viewer.key_handler = on_key

    step_index = 0
    try:
        while viewer.is_running():
            frame_start = time.time()

            if not state["paused"]:
                for _ in range(int(round((1.0 / 60.0) / sim.SIM_TIMESTEP))):
                    if step_index % sim.CONTROL_DECIMATION == 0:
                        command = controller.update(time.time(), state["speed"])
                        angles = kinematics.solve(workspace.data, command.position,
                                                  command.yaw)
                        workspace.command(angles, command.gap)
                    mujoco.mj_step(workspace.model, workspace.data)
                    step_index += 1

            if controller.search_view == "eye_in_hand":
                viewer.set_panel_camera("claw", "EYE-IN-HAND CAMERA (search / confirm / grasp)")
            else:
                # A fixed search view never uses the claw camera for anything - not even
                # to confirm before grasping (see SearchAndPlaceController's "confirm_move"
                # handling) - so the panel stays on the fixed camera for the whole pick
                # cycle, not just while it's actively looking. Switching to "claw" here
                # would visually suggest eye-in-hand vision was involved when it wasn't.
                camera_name = SEARCH_VIEW_CAMERA_NAME[controller.search_view]
                label = "searching" if controller.phase in _FIXED_LOOK_PHASES else "picking"
                viewer.set_panel_camera(
                    camera_name, f"{VIEW_LABELS[controller.search_view]} CAMERA ({label})"
                )

            viewer.render(overlay_lines=(
                (f"search view: {VIEW_LABELS[controller.search_view]}"
                 f"{'  [PAUSED]' if state['paused'] else ''}\n"
                 f"phase: {controller.phase.upper()}\n"
                 f"{controller.status}\n"
                 f"objects collected: {controller.objects_collected}"
                 f"{f'   (lost after release: {controller.placements_lost})' if controller.placements_lost else ''}",
                 ""),
                ("SPACE pause   V search view   R reshuffle   +/- speed   ESC quit", ""),
            ))

            remaining = (1.0 / 60.0) - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()
        claw_camera.close()
        for camera in fixed_cameras.values():
            camera.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="object layout seed")
    parser.add_argument("--view", choices=SEARCH_VIEWS, default="eye_in_hand",
                        help="which camera finds the rough candidate (default: "
                             "eye_in_hand); the close-up confirm always uses the claw "
                             "camera regardless of this choice")
    args = parser.parse_args()
    run_demo(args.seed, args.view)


if __name__ == "__main__":
    main()
