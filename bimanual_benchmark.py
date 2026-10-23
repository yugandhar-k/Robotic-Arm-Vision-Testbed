"""bimanual_benchmark.py - Quantitative benchmark for the dual-arm bimanual study.

Runs the coordinated-manipulation experiments, aggregates them, writes
`results/bimanual_results.json`, and renders the figures the book chapter cites:

  1. Cooperative lift, synchronized, over several jittered payload poses -> mean +/- std
     tilt and end-height gap (the two arms carry a dumbbell level and reliably).
  2. A desynchronization sweep (right arm lagging the left by 0 .. 0.5 s) -> how the lift
     degrades as coordination is lost (the core coordination result).
  3. Single-arm baseline -> one hand cannot carry the (oversized) payload level.
  4. Lid close, one-hand vs two-hand -> an *apparent* two-hand task that one hand actually
     does better (an honest negative that sharpens what genuine two-handedness requires).

Usage
-----
    python bimanual_benchmark.py

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import ball_balance as bb
import bimanual as bm

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LAGS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_JITTER = 8


def _agg(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}


def run() -> Dict:
    print("1/4 synchronized cooperative lift over jittered poses ...")
    sync_tilt, sync_gap, sync_held = [], [], []
    for seed in range(N_JITTER):
        m = bm.run_cooperative_lift(sync=True, jitter_seed=seed, verbose=False)
        sync_tilt.append(m["peak_tilt_deg"])
        sync_gap.append(m["peak_end_dz_mm"])
        sync_held.append(m["held_through_hold"])
    print(f"    peak tilt {_agg(sync_tilt)['mean']:.2f} +/- {_agg(sync_tilt)['std']:.2f} deg | "
          f"held {int(sum(sync_held))}/{N_JITTER}")

    print("2/4 desynchronization sweep ...")
    sweep = []
    for lag in LAGS:
        m = bm.run_cooperative_lift(sync=(lag == 0.0), lag_s=lag, verbose=False)
        sweep.append({"lag_s": lag, "peak_tilt_deg": m["peak_tilt_deg"],
                      "peak_end_dz_mm": m["peak_end_dz_mm"]})
        print(f"    lag {lag:.1f}s -> tilt {m['peak_tilt_deg']:5.1f} deg | "
              f"end-gap {m['peak_end_dz_mm']:5.1f} mm")

    print("3/4 single-arm baseline ...")
    single = bm.run_single_arm_lift(verbose=True)

    print("4/5 lid close one-hand vs two-hand ...")
    lid_one = bm.run_lid_close(two_handed=False, verbose=True)
    lid_two = bm.run_lid_close(two_handed=True, verbose=True)

    print("5/5 vision-guided ball-balancing (closed / disturbed / open) ...")
    balance_closed = bb.run_balance(closed_loop=True, verbose=True, return_trace=True)
    balance_shift = bb.run_balance(closed_loop=True, disturb_at=3.0, verbose=True,
                                   return_trace=True)
    balance_open = bb.run_balance(closed_loop=False, verbose=True, return_trace=True)

    results = {
        "cooperative_lift_sync": {
            "n_trials": N_JITTER,
            "peak_tilt_deg": _agg(sync_tilt),
            "peak_end_dz_mm": _agg(sync_gap),
            "held_rate": float(sum(sync_held) / N_JITTER),
        },
        "desync_sweep": sweep,
        "single_arm_lift": single,
        "lid_close_one_hand": lid_one,
        "lid_close_two_hand": lid_two,
        "vision_balance": {
            "closed_loop": balance_closed,
            "closed_loop_disturbed": balance_shift,
            "open_loop": balance_open,
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    # The per-tick offset traces are kept in memory for the figure but stripped from the
    # JSON summary so it stays a compact results table.
    serialisable = json.loads(json.dumps(results))
    for run_key in serialisable["vision_balance"].values():
        run_key.pop("trace_t", None)
        run_key.pop("trace_offset_mm", None)
    (RESULTS_DIR / "bimanual_results.json").write_text(json.dumps(serialisable, indent=2))
    print(f"\nwrote {RESULTS_DIR / 'bimanual_results.json'}")
    return results


def make_figures(results: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(exist_ok=True)
    sweep = results["desync_sweep"]
    lags = [s["lag_s"] for s in sweep]
    tilt = [s["peak_tilt_deg"] for s in sweep]
    gap = [s["peak_end_dz_mm"] for s in sweep]

    # Figure 1: coordination degradation vs desync lag.
    fig, ax1 = plt.subplots(figsize=(6.2, 4.0))
    c1, c2 = "#2a78d6", "#eb6834"
    ax1.plot(lags, tilt, "o-", color=c1, label="payload tilt")
    ax1.set_xlabel("inter-arm lag during the lift (s)")
    ax1.set_ylabel("peak payload tilt (deg)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(lags, gap, "s--", color=c2, label="end-height gap")
    ax2.set_ylabel("peak end-height gap (mm)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax1.set_title("Cooperative lift degrades as the two arms desynchronize")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bimanual_desync_sweep.png", dpi=160)
    plt.close(fig)

    # Figure 2: single vs synchronized-dual vs desynchronized-dual tilt.
    sync = results["cooperative_lift_sync"]["peak_tilt_deg"]
    desync04 = next(s["peak_tilt_deg"] for s in sweep if abs(s["lag_s"] - 0.4) < 1e-9)
    single = results["single_arm_lift"]["peak_tilt_deg"]
    labels = ["single arm", "two arms\n(synchronized)", "two arms\n(0.4 s desync)"]
    vals = [single, sync["mean"], desync04]
    errs = [0, sync["std"], 0]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.bar(labels, vals, yerr=errs, capsize=5,
                  color=["#999999", "#2a78d6", "#eb6834"], alpha=0.9)
    ax.set_ylabel("peak payload tilt (deg)")
    ax.set_title("Carrying an oversized payload level needs two synchronized arms")
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f} deg",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bimanual_tilt_comparison.png", dpi=160)
    plt.close(fig)

    # Figure 3: vision-guided ball-balancing - ball offset vs time for the three runs.
    vb = results["vision_balance"]
    closed, shift, openl = (vb["closed_loop"], vb["closed_loop_disturbed"],
                            vb["open_loop"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhspan(-bb.RECENTRE_BAND * 1000, bb.RECENTRE_BAND * 1000, color="#2a78d6",
               alpha=0.08, label="re-centred band")
    ax.plot(openl["trace_t"], openl["trace_offset_mm"], color="#999999", lw=1.8,
            label="open loop (no feedback)")
    ax.plot(closed["trace_t"], closed["trace_offset_mm"], color="#2a78d6", lw=1.8,
            label="closed loop (vision)")
    ax.plot(shift["trace_t"], shift["trace_offset_mm"], color="#eb6834", lw=1.8,
            label="closed loop + CoM shift")
    ax.axvspan(3.0, 3.0 + bb.DISTURB_DURATION, color="#eb6834", alpha=0.15)
    ax.annotate("CoM shift", xy=(3.0 + bb.DISTURB_DURATION / 2, ax.get_ylim()[1]),
                xytext=(3.3, 52), color="#eb6834", fontsize=9)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ball offset from tray centre (mm)")
    ax.set_title("Vision closes the loop: the ball is held centred and a shift is rejected")
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bimanual_ball_balance.png", dpi=160)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'bimanual_desync_sweep.png'}, "
          f"{FIGURES_DIR / 'bimanual_tilt_comparison.png'} and "
          f"{FIGURES_DIR / 'bimanual_ball_balance.png'}")


if __name__ == "__main__":
    make_figures(run())
