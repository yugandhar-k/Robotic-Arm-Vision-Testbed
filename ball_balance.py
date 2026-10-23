"""ball_balance.py - Vision-guided bimanual ball-balancing (the project's innovation).

Two arms jointly hold a short tray with a free ball on it and keep the ball centred by
tilting the tray in a coordinated way. This is a bimanual ball-and-beam: the control
output (tray tilt) is produced by the *two* arms moving their grips to different heights -
neither arm can hold the tray steady and tilt it alone - and the loop is closed with
vision: the ball's position is read from the fixed top-down camera each control tick, the
same fixed camera the single-arm classification benchmark studied. A PID law maps the
measured ball offset to a commanded tilt, realised as a differential height between the two
grips.

What makes it work (and what the write-up is honest about): two stiff, independently
position-controlled arms rigidly sharing one payload form an over-constrained closed chain.
Re-solving each arm's inverse kinematics to a freshly commanded pose every tick injects
energy at the grip contacts and the chain diverges within a few seconds. The fix is a
joint-space slew limit: the applied joint targets ramp toward the IK solution at a bounded
rate, so the arms never receive a sharp command and the grasp stays in its stick regime.
With that in place the controller runs indefinitely.

Three runs make the point:
  - closed loop: the inherent tray-tilt bias is cancelled and the ball is held at the
    centre (a few mm) indefinitely;
  - closed loop + disturbance: an unmodelled centre-of-mass shift (a torque pulse on the
    tray, as if a load slid to one side) is rejected and the ball re-centred;
  - open loop (grips held level, no feedback): the ball rolls to the downhill rail and
    stays there - levelling genuinely needs the coordinated, vision-driven correction.

Usage
-----
    python ball_balance.py                 # headless: the three runs, with metrics
    python ball_balance.py --view          # watch the closed-loop balancing live
    python ball_balance.py --view --push   # watch it reject a centre-of-mass shift
    python ball_balance.py --view --open-loop   # watch the ball roll to the rail

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

import dual_arm as da
import run_sim as sim
from bimanual import GRASP_YAW

# --- Controller gains -----------------------------------------------------------------
# PID mapping the measured ball offset (m) -> commanded differential grip height (m).
KP: float = 0.22
KI: float = 0.04
KD: float = 0.09
I_MAX: float = 0.06            # integral anti-windup clamp [m*s]
MAX_DZ: float = 0.012          # clamp on the differential grip height (tilt authority) [m]
VISION_LP: float = 0.5         # low-pass factor on the measured offset (0 = none)
# Joint-space slew limit [rad per control tick]. This is the stability-critical term: it
# bounds how fast the applied joint targets chase the IK solution, keeping the shared
# grasp in its stick regime so the over-constrained two-arm chain does not inject energy.
JOINT_SLEW: float = 0.004

HOLD_LIFT: float = 0.14        # how high above the grip the tray is held [m]
CONTROL_DT: float = sim.CONTROL_DECIMATION * sim.SIM_TIMESTEP

# --- Disturbance (an unmodelled centre-of-mass shift) ---------------------------------
DISTURB_TORQUE: float = 0.45   # pitch torque applied to the tray body [N*m]
DISTURB_DURATION: float = 0.6  # for how long [s]
RECENTRE_BAND: float = 0.012   # |offset| considered "re-centred" [m]


def _grip_columns() -> Tuple[np.ndarray, np.ndarray, float]:
    """World-frame grip points on the two tray posts, and their shared height."""
    cx, cy = da.WORKSPACE_CENTRE
    gz = da.TRAY_REST_Z + da.TRAY_TAB_CENTRE_DZ
    left = np.array([cx - da.TRAY_GRIP_OFFSET_X, cy, gz])
    right = np.array([cx + da.TRAY_GRIP_OFFSET_X, cy, gz])
    return left, right, gz


class VisionBallTracker:
    """Estimates the ball's world x from the fixed top-down camera: segment the ball, take
    its centroid, back-project through the rendered depth. The tray centre is known from
    the arms' own grip geometry (proprioception), so the offset the controller acts on is
    vision for the ball against a proprioceptive reference - never the ball's true state."""

    def __init__(self, dual: da.DualArm) -> None:
        self.cam = sim.SimCamera(dual, "top")
        self.ball_gid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_GEOM,
                                          "ball_geom")

    def ball_x(self) -> Optional[float]:
        seg = self.cam.render_segmentation()
        mask = seg == self.ball_gid
        if not mask.any():
            return None
        ys, xs = np.where(mask)
        px, py = float(xs.mean()), float(ys.mean())
        depth = float(self.cam.render_depth()[int(round(py)), int(round(px))])
        return float(self.cam.unproject(px, py, depth)[0])

    def close(self) -> None:
        self.cam.close()


class BimanualBalanceController:
    """Decentralised ball-and-beam PID with a low-passed measurement and anti-windup.

    The output is a single differential height `dz`: the two arms are commanded to
    `hold_z + dz` and `hold_z - dz`, so a positive error (ball toward +x) produces a
    negative `dz` that raises the +x grip and lowers the -x grip, tilting the ball back."""

    def __init__(self) -> None:
        self._offset_lp: Optional[float] = None
        self._offset_prev: Optional[float] = None
        self._integral: float = 0.0

    def reset(self) -> None:
        self._offset_lp = None
        self._offset_prev = None
        self._integral = 0.0

    def step(self, offset: float) -> float:
        """Map a measured ball offset (m) to a commanded differential grip height (m)."""
        self._offset_lp = (offset if self._offset_lp is None
                           else VISION_LP * self._offset_lp + (1.0 - VISION_LP) * offset)
        e = self._offset_lp
        e_dot = 0.0 if self._offset_prev is None else (e - self._offset_prev) / CONTROL_DT
        self._offset_prev = e
        self._integral = float(np.clip(self._integral + e * CONTROL_DT, -I_MAX, I_MAX))
        dz = -(KP * e + KI * self._integral + KD * e_dot)
        return float(np.clip(dz, -MAX_DZ, MAX_DZ))


class BalanceSession:
    """Grips the tray, lifts it, and then runs the vision loop while slewing the joint
    targets. Shared by the headless benchmark and the live viewer so both execute the
    exact same control."""

    def __init__(self, dual: da.DualArm) -> None:
        self.dual = dual
        self.closed_gap = sim.grasp_gap(2.0 * da.TRAY_TAB_HALF[1])
        self.ball_bid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.tray_bid = mujoco.mj_name2id(dual.model, mujoco.mjtObj.mjOBJ_BODY, "tray")
        self.left, self.right, gz = _grip_columns()
        self.hold_z = gz + HOLD_LIFT
        self.centre_x = 0.5 * (self.left[0] + self.right[0])
        self._q_left: Optional[np.ndarray] = None   # applied (slewed) joint targets
        self._q_right: Optional[np.ndarray] = None

    def _solve(self, dz: float) -> Tuple[np.ndarray, np.ndarray]:
        lt = np.array([self.left[0], self.left[1], self.hold_z + dz])
        rt = np.array([self.right[0], self.right[1], self.hold_z - dz])
        ql = self.dual.left.kinematics.solve(self.dual.data, lt, GRASP_YAW)
        qr = self.dual.right.kinematics.solve(self.dual.data, rt, GRASP_YAW)
        return ql, qr

    def grip_and_lift(self) -> None:
        """Approach the posts, close the grips, and lift the tray to the hold height."""
        gz = self.left[2]
        left0 = np.array([self.left[0], self.left[1], gz])
        right0 = np.array([self.right[0], self.right[1], gz])
        up = np.array([0.0, 0.0, HOLD_LIFT])

        def drive(lt, rt, gap, seconds):
            steps = int(seconds / sim.SIM_TIMESTEP)
            for i in range(steps):
                if i % sim.CONTROL_DECIMATION == 0:
                    ql = self.dual.left.kinematics.solve(self.dual.data, lt, GRASP_YAW)
                    qr = self.dual.right.kinematics.solve(self.dual.data, rt, GRASP_YAW)
                    self.dual.left.command(self.dual.data, ql, gap)
                    self.dual.right.command(self.dual.data, qr, gap)
                mujoco.mj_step(self.dual.model, self.dual.data)

        drive(left0 + up, right0 + up, sim.OPEN_GAP, 1.2)
        drive(left0, right0, sim.OPEN_GAP, 0.9)
        drive(left0, right0, self.closed_gap, 0.9)
        drive(left0 + up, right0 + up, self.closed_gap, 1.0)
        # Seed the slewed joint targets at the solution for the level hold pose.
        self._q_left, self._q_right = self._solve(0.0)

    def control_tick(self, dz: float) -> None:
        """Slew the applied joint targets toward the IK solution for tilt `dz`, command
        the arms, and advance the physics one control interval."""
        ql, qr = self._solve(dz)
        self._q_left = self._q_left + np.clip(ql - self._q_left, -JOINT_SLEW, JOINT_SLEW)
        self._q_right = self._q_right + np.clip(qr - self._q_right, -JOINT_SLEW, JOINT_SLEW)
        for _ in range(sim.CONTROL_DECIMATION):
            self.dual.left.command(self.dual.data, self._q_left, self.closed_gap)
            self.dual.right.command(self.dual.data, self._q_right, self.closed_gap)
            mujoco.mj_step(self.dual.model, self.dual.data)

    def true_offset(self) -> float:
        """Ground-truth ball-vs-tray x offset (for evaluation only, never for control)."""
        return float(self.dual.data.xpos[self.ball_bid][0]
                     - self.dual.data.xpos[self.tray_bid][0])

    def ball_height(self) -> float:
        return float(self.dual.data.xpos[self.ball_bid][2])

    def set_com_shift(self, active: bool) -> None:
        """Apply (or clear) the pitch torque standing in for a shifted payload load."""
        self.dual.data.xfrc_applied[self.tray_bid, 4] = DISTURB_TORQUE if active else 0.0

    def ball_fell(self) -> bool:
        return (abs(self.true_offset()) > da.TRAY_HALF[0] + 0.03
                or self.ball_height() < sim.TABLE_TOP_Z + 0.05)


def run_balance(closed_loop: bool = True, disturb_at: Optional[float] = None,
                duration: float = 9.0, verbose: bool = True,
                return_trace: bool = False) -> Dict[str, float]:
    """Run one balancing trial headless and return its metrics.

    Metrics are computed from the ground-truth ball-tray offset (the controller still only
    ever sees the vision estimate). `settle_abs_offset_mm` is the mean |offset| over the
    final second; `peak_abs_offset_mm` the worst excursion; for a disturbed run,
    `recovery_time_s` is the time from the disturbance onset back to within the re-centre
    band."""
    dual = da.DualArm(objects_xml=da.tray_ball_objects_xml())
    session = BalanceSession(dual)
    session.grip_and_lift()
    tracker = VisionBallTracker(dual)
    controller = BimanualBalanceController()

    offsets: List[Tuple[float, float]] = []
    rolled_off = False
    disturbing = False
    disturbed_done = False
    recovery_time: Optional[float] = None
    peak_after_disturb = 0.0

    n_ctrl = int(duration / CONTROL_DT)
    for k in range(n_ctrl):
        t = k * CONTROL_DT
        if disturb_at is not None:
            within = disturb_at <= t < disturb_at + DISTURB_DURATION
            if within and not disturbing:
                disturbing = True
            if disturbing and not within:
                disturbing = False
                disturbed_done = True
            session.set_com_shift(within)

        ball_x = tracker.ball_x()
        offset = 0.0 if ball_x is None else (ball_x - session.centre_x)
        dz = controller.step(offset) if closed_loop else 0.0
        session.control_tick(dz)

        true_off = session.true_offset()
        offsets.append((t, true_off))
        if disturb_at is not None and t >= disturb_at:
            peak_after_disturb = max(peak_after_disturb, abs(true_off))
            if (recovery_time is None and peak_after_disturb > 0.02
                    and t >= disturb_at + DISTURB_DURATION and abs(true_off) < RECENTRE_BAND):
                recovery_time = t - disturb_at
        if session.ball_fell():
            rolled_off = True
            break

    tracker.close()

    tail = [abs(o) for (tt, o) in offsets if tt >= offsets[-1][0] - 1.0]
    settle = float(np.mean(tail)) if tail else 1.0
    peak = max(abs(o) for _, o in offsets) if offsets else 0.0
    metrics = {
        "closed_loop": float(closed_loop),
        "disturbed": float(disturb_at is not None),
        "rolled_off": float(rolled_off),
        "settle_abs_offset_mm": settle * 1000.0,
        "peak_abs_offset_mm": peak * 1000.0,
        "duration_s": offsets[-1][0] if offsets else 0.0,
    }
    if disturb_at is not None:
        metrics["peak_after_disturb_mm"] = peak_after_disturb * 1000.0
        metrics["recovery_time_s"] = (-1.0 if recovery_time is None else recovery_time)
    if return_trace:
        metrics["trace_t"] = [t for t, _ in offsets]
        metrics["trace_offset_mm"] = [o * 1000.0 for _, o in offsets]

    if verbose:
        tag = ("OPEN-LOOP  " if not closed_loop else
               ("CLOSED+SHIFT" if disturb_at is not None else "CLOSED-LOOP "))
        if rolled_off:
            outcome = "ball ROLLED OFF"
        elif disturb_at is not None:
            rec = metrics["recovery_time_s"]
            rec_s = f"recovered in {rec:.1f}s" if rec >= 0 else "held"
            outcome = (f"peak {metrics['peak_after_disturb_mm']:.0f}mm, {rec_s}, "
                       f"settled {settle * 1000:.0f}mm")
        else:
            outcome = f"settled {settle * 1000:.0f}mm (peak {peak * 1000:.0f}mm)"
        print(f"[{tag}] {outcome}")
    return metrics


def view_balance(closed_loop: bool = True, disturb_at: Optional[float] = None) -> None:
    import time
    import mujoco.viewer
    dual = da.DualArm(objects_xml=da.tray_ball_objects_xml())
    print("[ball_balance] opening viewer; close the window to quit.")
    with mujoco.viewer.launch_passive(dual.model, dual.data) as viewer:
        while viewer.is_running():
            session = BalanceSession(dual)
            session.grip_and_lift()
            tracker = VisionBallTracker(dual)
            controller = BimanualBalanceController()
            for k in range(int(9.0 / CONTROL_DT)):
                if not viewer.is_running():
                    break
                t = k * CONTROL_DT
                if disturb_at is not None:
                    session.set_com_shift(disturb_at <= t < disturb_at + DISTURB_DURATION)
                ball_x = tracker.ball_x()
                offset = 0.0 if ball_x is None else (ball_x - session.centre_x)
                dz = controller.step(offset) if closed_loop else 0.0
                frame_start = time.time()
                session.control_tick(dz)
                viewer.sync()
                remaining = CONTROL_DT - (time.time() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
            tracker.close()
            dual.reset()


def save_scene_snapshot(path: str = "figures/bimanual_ball_balance_scene.png") -> None:
    """Render the two arms holding the tray with the ball centred (closed loop settled),
    from the front and top cameras, and save the figure the README cites."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dual = da.DualArm(objects_xml=da.tray_ball_objects_xml())
    session = BalanceSession(dual)
    session.grip_and_lift()
    tracker = VisionBallTracker(dual)
    controller = BimanualBalanceController()
    for _ in range(int(2.5 / CONTROL_DT)):
        ball_x = tracker.ball_x()
        offset = 0.0 if ball_x is None else (ball_x - session.centre_x)
        session.control_tick(controller.step(offset))
    tracker.close()

    renderer = mujoco.Renderer(dual.model, height=600, width=900)
    frames = []
    for cam in ["front", "top"]:
        renderer.update_scene(dual.data, camera=cam)
        frames.append((cam, renderer.render().copy()))
    renderer.close()
    fig, axes = plt.subplots(1, len(frames), figsize=(6.0 * len(frames), 4.0))
    for ax, (cam, img) in zip(axes, frames):
        ax.imshow(img)
        ax.set_title(f"{cam} camera")
        ax.axis("off")
    fig.suptitle("Two arms jointly holding the tray, ball centred by vision (closed loop)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path} (settled offset {session.true_offset() * 1000:.1f} mm)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", action="store_true", help="live viewer")
    ap.add_argument("--snapshot", action="store_true",
                    help="render the held-tray scene figure and exit")
    ap.add_argument("--open-loop", action="store_true",
                    help="disable the feedback (watch the ball roll to the rail)")
    ap.add_argument("--push", action="store_true",
                    help="apply a centre-of-mass shift mid-run")
    args = ap.parse_args()
    if args.snapshot:
        save_scene_snapshot()
        return
    if args.view:
        view_balance(closed_loop=not args.open_loop,
                     disturb_at=(3.0 if args.push else None))
        return
    print("=== Vision-guided bimanual ball-balancing ===")
    run_balance(closed_loop=True)
    run_balance(closed_loop=True, disturb_at=3.0)
    run_balance(closed_loop=False)


if __name__ == "__main__":
    main()
