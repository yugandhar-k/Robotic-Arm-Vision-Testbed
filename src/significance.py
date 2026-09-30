"""significance.py - Paired statistical tests for the eye-in-hand vs. each fixed-camera
accuracy gap.

Every view's held-out test set is the *same* held-out frame indices (the split is a
deterministic function of an identical class-label sequence for every view - see
`src.dataset.stratified_split`), so every instance has an eye-in-hand prediction and a
prediction from each fixed camera. That pairing licenses McNemar's test (is one view
wrong more often than the other, on the same items?) and a paired bootstrap on the
accuracy gap directly, both of which are more powerful than comparing two independent
accuracies.

Usage
-----
    python -m src.significance

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from src.utils import FIXED_VIEWS, MODEL_KINDS, RANDOM_SEED, RESULTS_DIR, save_json

N_BOOTSTRAP: int = 10000


def _load_correctness(view: str, model: str) -> Dict[str, bool]:
    path = RESULTS_DIR / f"{view}_{model}_test_predictions.json"
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {r["file"]: r["true_class_id"] == r["pred_class_id"] for r in payload["records"]}


def _paired_arrays(fixed_view: str, model: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    eye = _load_correctness("eye_in_hand", model)
    fixed = _load_correctness(fixed_view, model)
    files = sorted(eye.keys())
    assert set(eye.keys()) == set(fixed.keys()), (
        f"eye_in_hand and {fixed_view} test sets differ for {model} - "
        "the paired tests below assume the same frame indices in both."
    )
    eye_correct = np.array([eye[f] for f in files])
    fixed_correct = np.array([fixed[f] for f in files])
    return eye_correct, fixed_correct, files


def mcnemar_test(eye_correct: np.ndarray, fixed_correct: np.ndarray) -> Dict[str, object]:
    """Exact McNemar's test on the discordant pairs.

    b = eye-in-hand right, fixed camera wrong; c = the reverse. Under the null that the
    two views are equally likely to be the one that's right when they disagree, b is
    Binomial(b + c, 0.5); the exact test uses that distribution directly rather than the
    chi-squared approximation, which is unreliable when b + c is small.
    """
    b = int(np.sum(eye_correct & ~fixed_correct))
    c = int(np.sum(~eye_correct & fixed_correct))
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(min(b, c), n_discordant, 0.5).pvalue)
    return {
        "eye_right_fixed_wrong": b,
        "fixed_right_eye_wrong": c,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def paired_bootstrap_gap(eye_correct: np.ndarray, fixed_correct: np.ndarray,
                         seed: int = RANDOM_SEED) -> Dict[str, object]:
    """Bootstrap 95% CI on the accuracy gap (eye-in-hand - fixed camera).

    Resampling instance indices (not the two conditions independently) keeps the pairing:
    each bootstrap replicate asks "what would the gap have been on a different sample of
    instances," using the same instances for both views every time.
    """
    rng = np.random.default_rng(seed)
    n = len(eye_correct)
    observed_gap = float(eye_correct.mean() - fixed_correct.mean())
    gaps = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        gaps[i] = eye_correct[idx].mean() - fixed_correct[idx].mean()
    lower, upper = np.percentile(gaps, [2.5, 97.5])
    return {
        "observed_gap": observed_gap,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "n_bootstrap": N_BOOTSTRAP,
        "excludes_zero": bool(lower > 0 or upper < 0),
    }


def bootstrap_accuracy_ci(correct: np.ndarray, seed: int = RANDOM_SEED) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    accuracies = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        accuracies[i] = correct[idx].mean()
    lower, upper = np.percentile(accuracies, [2.5, 97.5])
    return {"accuracy": float(correct.mean()), "ci95_lower": float(lower),
           "ci95_upper": float(upper)}


def main() -> None:
    summary: Dict[str, object] = {}
    print(f"{'fixed view':<16} {'model':<5} {'eye acc (95% CI)':<24} "
          f"{'fixed acc (95% CI)':<24} {'gap (95% CI)':<22} {'McNemar p':<10}")
    for fixed_view in FIXED_VIEWS:
        summary[fixed_view] = {}
        for model in MODEL_KINDS:
            eye_correct, fixed_correct, files = _paired_arrays(fixed_view, model)

            eye_ci = bootstrap_accuracy_ci(eye_correct)
            fixed_ci = bootstrap_accuracy_ci(fixed_correct)
            gap = paired_bootstrap_gap(eye_correct, fixed_correct)
            mcnemar = mcnemar_test(eye_correct, fixed_correct)

            eye_text = f"{eye_ci['accuracy']:.4f} [{eye_ci['ci95_lower']:.4f}, {eye_ci['ci95_upper']:.4f}]"
            fixed_text = (f"{fixed_ci['accuracy']:.4f} "
                         f"[{fixed_ci['ci95_lower']:.4f}, {fixed_ci['ci95_upper']:.4f}]")
            gap_text = f"{gap['observed_gap']:+.4f} [{gap['ci95_lower']:+.4f}, {gap['ci95_upper']:+.4f}]"
            print(f"{fixed_view:<16} {model:<5} {eye_text:<24} {fixed_text:<24} "
                  f"{gap_text:<22} {mcnemar['p_value']:<10.4f}")

            summary[fixed_view][model] = {
                "n_test": len(files),
                "eye_in_hand_accuracy_ci95": eye_ci,
                f"{fixed_view}_accuracy_ci95": fixed_ci,
                "paired_bootstrap_gap": gap,
                "mcnemar": mcnemar,
            }

    save_json(summary, RESULTS_DIR / "significance.json")
    print(f"\nwritten to {RESULTS_DIR / 'significance.json'}")


if __name__ == "__main__":
    main()
