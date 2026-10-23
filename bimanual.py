"""bimanual.py - Two-hand cooperative-manipulation task controllers for the dual-arm rig.

Tasks here require the two arms to act on the *same* object at the *same* time, with tight
coordination - the kind of manipulation a single arm cannot do. This module implements the
controllers and headless measurement runs; `dual_arm.py` owns the simulation/scene.

Task 1 - Cooperative lift: both arms grasp opposite ends of a long bar and raise it in
lockstep. The two grippers plus the rigid bar form a closed kinematic chain, so the lift
only stays level (and only avoids an internal tug-of-war) if the arms move synchronously.
We measure that directly, and contrast a desynchronized lift as an ablation.

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

import dual_arm as da
import run_sim as sim

# Gripper closes across the bar's width, so the tool is yawed 90 deg (closing axis along
# world y while the bar lies along world x).
GRASP_YAW: float = math.pi / 2.0
GRASP_CONFIRM_MARGIN: float = 0.004       # gripper encoder: fingers stop this far short of
                                          # fully closed => something is held
LIFT_HEIGHT: float = 0.18                 # how high above the table the bar is raised [m]

# Synchronized phase durations [s].
DUR_APPROACH: float = 1.2
DUR_DESCEND: float = 0.8
DUR_CLOSE: float = 0.8
DUR_LIFT: float = 1.1
DUR_HOLD: float = 0.7
DUR_LOWER: float = 0.9
DUR_RELEASE: float = 0.4


def _smoothstep(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3.0 - 2.0 * a)


@dataclass
class _ArmTraj:
    """A per-arm linear tool-frame move used within one synchronized phase."""
    start: np.ndarray
    end: np.ndarray
    yaw: float
    start_gap: float
    end_gap: float

    def at(self, alpha: float) -> Tuple[np.ndarray, float, float]:
        s = _smoothstep(alpha)
        return (self.start + (self.end - self.start) * s, self.yaw,
                self.start_gap + (self.end_gap - self.start_gap) * s)


class CooperativeLiftController:
    """Synchronized two-arm controller: grasp a bar at both ends and lift it level.

    Both arms advance through the same phase sequence on one shared clock - the essence of
    the coordination. The lift phase only begins once *both* grippers have confirmed a
    grasp (a synchronization barrier), and within the lift both tool frames follow the
    identical vertical profile so the shared rigid bar stays level.
    """

    PHASES: Tuple[Tuple[str, float], ...] = (
        ("approach", DUR_APPROACH), ("descend", DUR_DESCEND), ("close", DUR_CLOSE),
        ("lift", DUR_LIFT), ("hold", DUR_HOLD), ("lower", DUR_LOWER),
        ("release", DUR_RELEASE),
    )

    def __init__(self, dual: da.DualArm, sync: bool = True,
                 lag_s: float = 0.0) -> None:
        self.dual = dual
        self.sync = sync
        self.lag_s = lag_s          # desync ablation: right arm lags the left by this much
        self.payload_bid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_BODY,
                                             "payload")
        self.phase_index = 0
        self.phase_start = 0.0
        self.trajectories: Dict[str, _ArmTraj] = {}
        self.column: Dict[str, np.ndarray] = {}     # locked (x,y) grasp column per arm
        self.closed_gap: float = sim.grasp_gap(2.0 * da.PAYLOAD_ROD_HALF[1])
        self.done = False
        self.peak_tilt = 0.0        # worst bar tilt from horizontal during the lift [deg]
        self.peak_end_dz = 0.0      # worst height gap between the two bar ends [m]
        self._anchor = {name: arm.tool_position(dual.data)
                        for name, arm in self._arms().items()}
        self._begin_phase(0.0)

    # -- helpers --------------------------------------------------------------------
    def _arms(self) -> Dict[str, da.ArmHandle]:
        return {"left": self.dual.left, "right": self.dual.right}

    def _phase(self) -> str:
        return self.PHASES[self.phase_index][0]

    def _grasp_targets(self) -> Dict[str, np.ndarray]:
        left_end, right_end = self.dual.grasp_ends()
        bar_z = self.dual.payload_position()[2]
        return {"left": np.array([left_end[0], left_end[1], bar_z]),
                "right": np.array([right_end[0], right_end[1], bar_z])}

    def _begin_phase(self, now: float) -> None:
        phase = self._phase()
        self.phase_start = now
        arms = self._arms()
        self.trajectories = {}
        if phase == "approach":
            tgt = self._grasp_targets()
            for name, arm in arms.items():
                above = tgt[name] + np.array([0.0, 0.0, LIFT_HEIGHT])
                self.trajectories[name] = _ArmTraj(self._anchor[name], above, GRASP_YAW,
                                                   sim.OPEN_GAP, sim.OPEN_GAP)
        elif phase == "descend":
            tgt = self._grasp_targets()
            for name, arm in arms.items():
                self.column[name] = tgt[name][:2].copy()
                self.trajectories[name] = _ArmTraj(self._anchor[name], tgt[name],
                                                   GRASP_YAW, sim.OPEN_GAP, sim.OPEN_GAP)
        elif phase == "close":
            for name in arms:
                here = self._anchor[name]
                self.trajectories[name] = _ArmTraj(here, here, GRASP_YAW,
                                                   sim.OPEN_GAP, self.closed_gap)
        elif phase == "lift":
            for name in arms:
                col = self.column[name]
                top = np.array([col[0], col[1], sim.TABLE_TOP_Z + da.PAYLOAD_HALF[2]
                                + LIFT_HEIGHT])
                self.trajectories[name] = _ArmTraj(self._anchor[name], top, GRASP_YAW,
                                                   self.closed_gap, self.closed_gap)
        elif phase == "hold":
            for name in arms:
                here = self._anchor[name]
                self.trajectories[name] = _ArmTraj(here, here, GRASP_YAW,
                                                   self.closed_gap, self.closed_gap)
        elif phase == "lower":
            for name in arms:
                col = self.column[name]
                down = np.array([col[0], col[1], sim.TABLE_TOP_Z + da.PAYLOAD_HALF[2]
                                 + 0.01])
                self.trajectories[name] = _ArmTraj(self._anchor[name], down, GRASP_YAW,
                                                   self.closed_gap, self.closed_gap)
        else:  # release
            for name in arms:
                here = self._anchor[name]
                self.trajectories[name] = _ArmTraj(here, here, GRASP_YAW,
                                                   self.closed_gap, sim.OPEN_GAP)

    def _grasped(self) -> Dict[str, bool]:
        out = {}
        for name, arm in self._arms().items():
            gap = sim.OPEN_GAP
            q = self.dual.data.qpos[arm.finger_qpos_ids]
            gap = float(2.0 * (sim.FINGER_HOME_X - sim.FINGER_HALF_THICK - np.mean(q)))
            out[name] = gap > sim.MIN_GRASP_GAP + GRASP_CONFIRM_MARGIN
        return out

    def _phase_alpha(self, name: str, now: float) -> float:
        dur = self.PHASES[self.phase_index][1]
        t = now - self.phase_start
        # Desync ablation: in the lift phase, the right arm starts its motion `lag_s` late.
        if not self.sync and self._phase() == "lift" and name == "right":
            t = t - self.lag_s
        return 1.0 if dur <= 0 else min(1.0, max(0.0, t / dur))

    # -- main tick ------------------------------------------------------------------
    def update(self, now: float) -> None:
        if self.done:
            return
        arms = self._arms()
        alphas = {name: self._phase_alpha(name, now) for name in arms}
        for name, arm in arms.items():
            pos, yaw, gap = self.trajectories[name].at(alphas[name])
            angles = arm.kinematics.solve(self.dual.data, pos, yaw)
            arm.command(self.dual.data, angles, gap)

        # Track coordination quality while the bar is held: how far it tilts from level,
        # and the height gap between its two ends. Perfectly synchronized arms keep both
        # near zero; a desynchronized pair see-saws the bar (one end rises first).
        if self._phase() in ("lift", "hold"):
            left_end, right_end = self._end_positions()
            self.peak_end_dz = max(self.peak_end_dz, abs(left_end[2] - right_end[2]))
            self.peak_tilt = max(self.peak_tilt, self._tilt_deg())

        now_done = min(alphas.values()) >= 1.0
        if now_done:
            self._anchor = {name: arm.tool_position(self.dual.data)
                            for name, arm in arms.items()}
            # Synchronization barrier before lifting: do not advance from `close` to
            # `lift` until BOTH grippers confirm a grasp.
            if self._phase() == "close" and not all(self._grasped().values()):
                self.phase_start = now        # keep holding the close, re-check next tick
                return
            self.phase_index += 1
            if self.phase_index >= len(self.PHASES):
                self.done = True
                return
            self._begin_phase(now)

    def _end_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dual.grasp_ends()

    def _tilt_deg(self) -> float:
        R = self.dual.data.xmat[self.payload_bid].reshape(3, 3)
        return math.degrees(math.asin(min(1.0, abs(R[:, 0][2]))))

    def bar_metrics(self) -> Dict[str, float]:
        height_gain = self.dual.payload_position()[2] - da.PAYLOAD_REST_Z
        return {"height_gain_mm": height_gain * 1000.0,
                "final_tilt_deg": self._tilt_deg(),
                "peak_tilt_deg": self.peak_tilt,
                "peak_end_dz_mm": self.peak_end_dz * 1000.0}


def run_cooperative_lift(sync: bool = True, lag_s: float = 0.0,
                         jitter_seed: Optional[int] = None,
                         verbose: bool = True) -> Dict[str, float]:
    dual = da.DualArm()
    if jitter_seed is not None:
        # Small random perturbation of the payload's resting pose (+/-5 mm in x/y, +/-3
        # deg yaw) so repeated trials are not identical - lets us report mean +/- std and
        # show the synchronized lift is reliably level, not a single lucky run.
        rng = np.random.default_rng(jitter_seed)
        adr = dual.payload_qpos_adr
        dual.data.qpos[adr] += rng.uniform(-0.005, 0.005)
        dual.data.qpos[adr + 1] += rng.uniform(-0.005, 0.005)
        yaw = rng.uniform(-0.052, 0.052)
        dual.data.qpos[adr + 3: adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(dual.model, dual.data)
    ctrl = CooperativeLiftController(dual, sync=sync, lag_s=lag_s)
    dt = 1.0 / 60.0
    step_index = 0
    peak_height = 0.0
    held_through_hold = True
    t = 0.0
    while not ctrl.done and t < 20.0:
        for _ in range(int(round(dt / sim.SIM_TIMESTEP))):
            if step_index % sim.CONTROL_DECIMATION == 0:
                ctrl.update(t)
            mujoco.mj_step(dual.model, dual.data)
            step_index += 1
        t += dt
        if not ctrl.done and ctrl._phase() in ("lift", "hold"):
            peak_height = max(peak_height, dual.payload_position()[2] - da.PAYLOAD_REST_Z)
            if ctrl._phase() == "hold" and not all(ctrl._grasped().values()):
                held_through_hold = False

    m = ctrl.bar_metrics()
    m["peak_height_mm"] = peak_height * 1000.0
    m["held_through_hold"] = float(held_through_hold)
    if verbose:
        tag = "SYNC  " if sync else f"DESYNC(lag={lag_s:.2f}s)"
        print(f"[{tag}] peak_height {m['peak_height_mm']:6.1f} mm | "
              f"peak_tilt {m['peak_tilt_deg']:5.1f} deg | "
              f"peak_end_gap {m['peak_end_dz_mm']:6.1f} mm | "
              f"held {bool(m['held_through_hold'])}")
    return m


def run_single_arm_lift(verbose: bool = True) -> Dict[str, float]:
    """Baseline: one arm (left) alone grasps the only point it can - the left end rod -
    and tries to lift the dumbbell. The right arm stays parked. The central block is
    wider than the gripper's open span, so there is no one-hand grasp of the middle; the
    end-rod grasp leaves the mass a lever-arm away, which is what a single hand cannot
    hold level. We measure peak tilt and whether it stays gripped."""
    dual = da.DualArm()
    left = dual.left
    bar_z0 = dual.payload_position()[2]
    grasp_x = da.WORKSPACE_CENTRE[0] - da.PAYLOAD_ROD_OFFSET   # the left end rod
    col = np.array([grasp_x, da.WORKSPACE_CENTRE[1]])
    closed = sim.grasp_gap(2.0 * da.PAYLOAD_ROD_HALF[1])
    payload_bid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    peak_tilt = 0.0

    phases = (("approach", DUR_APPROACH), ("descend", DUR_DESCEND), ("close", DUR_CLOSE),
              ("lift", DUR_LIFT), ("hold", DUR_HOLD))
    anchor = left.tool_position(dual.data)

    def seg(a, b, g0, g1, alpha):
        s = _smoothstep(alpha)
        return a + (b - a) * s, g0 + (g1 - g0) * s

    dt = 1.0 / 60.0
    step_index = 0
    peak_height = 0.0
    t0 = 0.0
    for phase, dur in phases:
        grasp_pt = np.array([col[0], col[1], dual.payload_position()[2]])
        above = grasp_pt + np.array([0, 0, LIFT_HEIGHT])
        top = np.array([col[0], col[1], sim.TABLE_TOP_Z + da.PAYLOAD_HALF[2] + LIFT_HEIGHT])
        n = int(dur / dt)
        for k in range(n):
            alpha = (k + 1) / n
            if phase == "approach":
                pos, gap = seg(anchor, above, sim.OPEN_GAP, sim.OPEN_GAP, alpha)
            elif phase == "descend":
                pos, gap = seg(above, grasp_pt, sim.OPEN_GAP, sim.OPEN_GAP, alpha)
            elif phase == "close":
                pos, gap = grasp_pt, (sim.OPEN_GAP + (closed - sim.OPEN_GAP) * _smoothstep(alpha))
            elif phase == "lift":
                pos, gap = seg(grasp_pt, top, closed, closed, alpha)
            else:
                pos, gap = top, closed
            for _ in range(int(round(dt / sim.SIM_TIMESTEP))):
                if step_index % sim.CONTROL_DECIMATION == 0:
                    left.command(dual.data, left.kinematics.solve(dual.data, pos, GRASP_YAW), gap)
                mujoco.mj_step(dual.model, dual.data)
                step_index += 1
            if phase in ("lift", "hold"):
                peak_height = max(peak_height, dual.payload_position()[2] - bar_z0)
                R = dual.data.xmat[payload_bid].reshape(3, 3)
                peak_tilt = max(peak_tilt, math.degrees(math.asin(min(1.0, abs(R[:, 0][2])))))
        anchor = left.tool_position(dual.data)

    R = dual.data.xmat[payload_bid].reshape(3, 3)
    tilt_deg = math.degrees(math.asin(min(1.0, abs(R[:, 0][2]))))
    q = dual.data.qpos[left.finger_qpos_ids]
    gap = float(2.0 * (sim.FINGER_HOME_X - sim.FINGER_HALF_THICK - np.mean(q)))
    held = gap > sim.MIN_GRASP_GAP + GRASP_CONFIRM_MARGIN
    m = {"peak_height_mm": peak_height * 1000.0, "final_tilt_deg": tilt_deg,
         "peak_tilt_deg": peak_tilt, "held": float(held)}
    if verbose:
        print(f"[SINGLE end-rod] peak_height {m['peak_height_mm']:6.1f} mm | "
              f"peak_tilt {m['peak_tilt_deg']:5.1f} deg | "
              f"final_tilt {m['final_tilt_deg']:5.1f} deg | held {bool(m['held'])}")
    return m


# ===========================================================================
# Task 2 - Lid close (stabilize + act): one arm holds the can, the other seats
# the lid. A role-differentiated two-hand task - without the holding arm, seating
# the lid displaces the free-standing can.
# ===========================================================================

Segment = Tuple[np.ndarray, float, float, float]   # (end_pos, yaw, end_gap, duration)


def _play_two_arm(dual: da.DualArm, scripts: Dict[str, List[Segment]],
                  settle_after: float = 0.6) -> None:
    """Play per-arm timed tool-frame segment scripts on a shared clock.

    Arms absent from `scripts` are held at the home pose (parked clear of the workspace).
    Each arm's first segment starts from its current tool pose; later segments start where
    the previous one ended. This is the shared substrate every scripted bimanual task
    runs on.
    """
    arms = {"left": dual.left, "right": dual.right}
    timelines: Dict[str, List[tuple]] = {}
    total = 0.0
    for name, segs in scripts.items():
        arm = arms[name]
        t0 = 0.0
        start_pos = arm.tool_position(dual.data)
        start_gap = sim.OPEN_GAP
        tl = []
        for end_pos, yaw, end_gap, dur in segs:
            end_pos = np.asarray(end_pos, dtype=float)
            tl.append((t0, t0 + dur, np.asarray(start_pos, float), end_pos, yaw,
                       start_gap, end_gap))
            t0 += dur
            start_pos, start_gap = end_pos, end_gap
        timelines[name] = tl
        total = max(total, t0)

    dt = 1.0 / 60.0
    step_index = 0
    t = 0.0
    while t < total + settle_after:
        for _ in range(int(round(dt / sim.SIM_TIMESTEP))):
            if step_index % sim.CONTROL_DECIMATION == 0:
                for name, arm in arms.items():
                    if name not in timelines:
                        arm.command(dual.data, np.asarray(da.HOME_POSE), sim.OPEN_GAP)
                        continue
                    seg = timelines[name][-1]
                    for s in timelines[name]:
                        if t <= s[1]:
                            seg = s
                            break
                    ts, te, sp, ep, yaw, sg, eg = seg
                    a = 1.0 if te <= ts else min(1.0, max(0.0, (t - ts) / (te - ts)))
                    s = _smoothstep(a)
                    pos = sp + (ep - sp) * s
                    gap = sg + (eg - sg) * s
                    arm.command(dual.data, arm.kinematics.solve(dual.data, pos, yaw), gap)
            mujoco.mj_step(dual.model, dual.data)
            step_index += 1
        t += dt


def run_lid_close(two_handed: bool = True, verbose: bool = True) -> Dict[str, float]:
    dual = da.DualArm(objects_xml=da.lid_objects_xml())
    can_bid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_BODY, "can")
    lid_bid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_BODY, "lid")
    can_adr = dual.free_body_qpos_adr("can_free")
    can_xy0 = dual.data.qpos[can_adr: can_adr + 2].copy()

    z = sim.TABLE_TOP_Z
    can_grasp = np.array([da.CAN_X, 0.0, da.CAN_GRASP_Z])
    stem_z = da.LID_REST_Z + da.LID_STEM_CENTRE_DZ
    seated_stem_z = da.CAN_TOP_Z + da.LID_DISK_HALF_H + da.LID_STEM_CENTRE_DZ
    lid_grasp = np.array([da.LID_X, 0.0, stem_z])
    over_can = np.array([da.CAN_X, 0.0, seated_stem_z + 0.10])
    seat = np.array([da.CAN_X, 0.0, seated_stem_z])
    can_closed = sim.grasp_gap(2.0 * da.CAN_R)
    lid_closed = sim.grasp_gap(2.0 * da.LID_STEM_HALF[0])

    scripts: Dict[str, List[Segment]] = {
        "right": [
            (lid_grasp + [0, 0, 0.12], 0.0, sim.OPEN_GAP, 1.2),
            (lid_grasp, 0.0, sim.OPEN_GAP, 0.8),
            (lid_grasp, 0.0, lid_closed, 0.8),
            (lid_grasp + [0, 0, 0.12], 0.0, lid_closed, 0.6),
            (over_can, 0.0, lid_closed, 1.1),
            (seat, 0.0, lid_closed, 0.9),
            (seat, 0.0, sim.OPEN_GAP, 0.4),
            (seat + [0, 0, 0.12], 0.0, sim.OPEN_GAP, 0.6),
        ],
    }
    if two_handed:
        scripts["left"] = [
            (can_grasp + [0, 0, 0.12], 0.0, sim.OPEN_GAP, 1.2),
            (can_grasp, 0.0, sim.OPEN_GAP, 0.8),
            (can_grasp, 0.0, can_closed, 0.8),
            (can_grasp, 0.0, can_closed, 4.0),   # hold through the seating
        ]

    _play_two_arm(dual, scripts)

    # Measure outcome.
    can_xy = dual.data.qpos[can_adr: can_adr + 2]
    can_disp = float(np.linalg.norm(can_xy - can_xy0))
    can_R = dual.data.xmat[can_bid].reshape(3, 3)
    can_tilt = math.degrees(math.acos(min(1.0, abs(can_R[2, 2]))))
    lid_pos = dual.data.xpos[lid_bid].copy()
    lid_centre_err = float(np.linalg.norm(lid_pos[:2] - can_xy))
    seated_lid_z = da.CAN_TOP_Z + da.LID_DISK_HALF_H
    lid_z_err = float(abs(lid_pos[2] - seated_lid_z))

    seated = lid_centre_err < 0.015 and lid_z_err < 0.012
    can_ok = can_disp < 0.02 and can_tilt < 15.0
    success = bool(seated and can_ok)
    m = {"success": float(success), "lid_centre_err_mm": lid_centre_err * 1000,
         "lid_z_err_mm": lid_z_err * 1000, "can_disp_mm": can_disp * 1000,
         "can_tilt_deg": can_tilt}
    if verbose:
        tag = "TWO-HAND" if two_handed else "ONE-HAND"
        print(f"[{tag}] success {str(success):5} | lid_offset {m['lid_centre_err_mm']:5.1f} mm "
              f"| lid_z_err {m['lid_z_err_mm']:5.1f} mm | can_moved {m['can_disp_mm']:5.1f} mm "
              f"| can_tilt {m['can_tilt_deg']:4.1f} deg")
    return m


def view_cooperative_lift(sync: bool = True, lag_s: float = 0.0) -> None:
    """Open a live viewer and loop the cooperative lift (watchable demo)."""
    import time
    import mujoco.viewer
    dual = da.DualArm()
    print("[bimanual] opening viewer - two arms lifting the dumbbell cooperatively; "
          "close the window to quit.")
    with mujoco.viewer.launch_passive(dual.model, dual.data) as viewer:
        while viewer.is_running():
            ctrl = CooperativeLiftController(dual, sync=sync, lag_s=lag_s)
            step_index = 0
            t = 0.0
            while not ctrl.done and viewer.is_running():
                frame_start = time.time()
                for _ in range(int(round((1.0 / 60.0) / sim.SIM_TIMESTEP))):
                    if step_index % sim.CONTROL_DECIMATION == 0:
                        ctrl.update(t)
                    mujoco.mj_step(dual.model, dual.data)
                    step_index += 1
                t += 1.0 / 60.0
                viewer.sync()
                remaining = (1.0 / 60.0) - (time.time() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
            for _ in range(90):
                mujoco.mj_step(dual.model, dual.data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
            dual.reset()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", action="store_true",
                    help="open a live viewer of the cooperative lift instead of the "
                         "headless benchmark")
    ap.add_argument("--desync", type=float, default=None, metavar="LAG_S",
                    help="with --view, desynchronize the arms by this lag to watch the "
                         "payload see-saw")
    args = ap.parse_args()
    if args.view:
        view_cooperative_lift(sync=(args.desync is None), lag_s=(args.desync or 0.0))
        return
    print("=== Task 1: cooperative lift ===")
    run_cooperative_lift(sync=True)
    run_cooperative_lift(sync=False, lag_s=0.4)
    run_single_arm_lift()
    print("\n=== Task 2: lid close (apparent two-hand task) ===")
    run_lid_close(two_handed=False)
    run_lid_close(two_handed=True)


if __name__ == "__main__":
    main()
