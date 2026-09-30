"""evaluate.py - Aggregate the six (view x model) result files that `src.train`
produces into the paper's central comparison, plus the results figures.

Run after all six `python -m src.train --view ... --model ...` calls have completed.

Usage
-----
    python -m src.evaluate

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import (
    CLASS_NAMES,
    FIGURES_DIR,
    FIXED_VIEWS,
    MODEL_KINDS,
    RESULTS_DIR,
    VIEWS,
    save_json,
)

VIEW_LABELS = {"eye_in_hand": "eye-in-hand", "fixed_overhead": "fixed-overhead",
              "fixed_angled": "fixed-angled"}


def _load(view: str, model: str) -> Dict[str, object]:
    path = RESULTS_DIR / f"{view}_{model}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} - run `python -m src.train --view {view} --model {model}` first"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_confusion_matrices(results: Dict[str, Dict[str, object]]) -> None:
    n_views = len(VIEWS)
    fig, axes = plt.subplots(len(MODEL_KINDS), n_views,
                             figsize=(4.4 * n_views, 4.2 * len(MODEL_KINDS)))
    for row, model in enumerate(MODEL_KINDS):
        for col, view in enumerate(VIEWS):
            ax = axes[row, col]
            matrix = np.array(results[(view, model)]["test_confusion_matrix"])
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalised = np.divide(matrix, row_sums, where=row_sums != 0)

            ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(CLASS_NAMES, fontsize=8)
            for i in range(len(CLASS_NAMES)):
                for j in range(len(CLASS_NAMES)):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                           color="white" if normalised[i, j] > 0.5 else "black",
                           fontsize=7)
            accuracy = results[(view, model)]["test_accuracy"]
            ax.set_title(f"{model.upper()} - {VIEW_LABELS[view]}\ntest accuracy {accuracy:.3f}",
                        fontsize=10)
            if row == len(MODEL_KINDS) - 1:
                ax.set_xlabel("predicted")
            if col == 0:
                ax.set_ylabel("true")

    fig.suptitle("Confusion matrices: eye-in-hand vs. both fixed cameras, CNN vs. SVM",
                fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _plot_accuracy_comparison(results: Dict[str, Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(VIEWS))
    width = 0.32
    colours = {"cnn": "#2a78d6", "svm": "#eb6834"}     # validated categorical slots 1/2

    test_label_used = False
    for offset, model in zip((-width / 2, width / 2), MODEL_KINDS):
        means = [results[(view, model)]["cv_accuracy_mean"] for view in VIEWS]
        stds = [results[(view, model)]["cv_accuracy_std"] for view in VIEWS]
        tests = [results[(view, model)]["test_accuracy"] for view in VIEWS]
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
              label=f"{model.upper()} (5-fold CV)", color=colours[model], alpha=0.85)
        ax.scatter(x + offset, tests, color="black", zorder=5, s=22, marker="D",
                  label=None if test_label_used else "held-out test")
        test_label_used = True

    ax.set_xticks(x)
    ax.set_xticklabels([VIEW_LABELS[v] for v in VIEWS])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Classification accuracy: viewpoint x model")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "results.png", dpi=160)
    plt.close(fig)


def _occlusion_bins(view: str) -> Optional[Tuple[List[float], List[float], List[int]]]:
    path = RESULTS_DIR / f"{view}_cnn_test_predictions.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    occlusions = np.array([r["occluded_fraction"] for r in payload["records"]
                           if r["occluded_fraction"] is not None])
    correct = np.array([r["true_class_id"] == r["pred_class_id"]
                        for r in payload["records"] if r["occluded_fraction"] is not None])
    if len(occlusions) == 0:
        return None

    bin_edges = np.linspace(occlusions.min(), occlusions.max(), 6)
    bin_centres, bin_accuracy, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (occlusions >= lo) & (occlusions <= hi)
        if mask.sum() == 0:
            continue
        bin_centres.append(0.5 * (lo + hi))
        bin_accuracy.append(correct[mask].mean())
        bin_counts.append(int(mask.sum()))
    return bin_centres, bin_accuracy, bin_counts


def _plot_occlusion_vs_correctness() -> None:
    """Do the fixed cameras fail *because* of occlusion?

    Uses the per-sample test predictions saved by `src.train` for each fixed camera's
    CNN, binned by how much of the crop the target object actually filled.
    """
    colours = {"fixed_overhead": "#2a78d6", "fixed_angled": "#eb6834"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = False
    for view in FIXED_VIEWS:
        bins = _occlusion_bins(view)
        if bins is None:
            continue
        bin_centres, bin_accuracy, bin_counts = bins
        ax.plot(bin_centres, bin_accuracy, marker="o", color=colours[view],
               label=VIEW_LABELS[view])
        for cx, acc, count in zip(bin_centres, bin_accuracy, bin_counts):
            ax.annotate(f"n={count}", (cx, acc), textcoords="offset points",
                       xytext=(0, 6), fontsize=7, ha="center", color=colours[view])
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("fraction of the crop that is not the target object")
    ax.set_ylabel("CNN test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Fixed-camera accuracy falls as occlusion rises")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "occlusion_vs_accuracy.png", dpi=160)
    plt.close(fig)


def main() -> None:
    results = {(view, model): _load(view, model) for view in VIEWS for model in MODEL_KINDS}

    print(f"{'view':<16} {'model':<5} {'cv acc (mean+/-std)':<22} {'test acc':<10} "
          f"{'test macro-F1':<14}")
    for view in VIEWS:
        for model in MODEL_KINDS:
            r = results[(view, model)]
            cv_text = f"{r['cv_accuracy_mean']:.4f} +/- {r['cv_accuracy_std']:.4f}"
            print(f"{view:<16} {model:<5} {cv_text:<22} {r['test_accuracy']:<10.4f} "
                  f"{r['test_macro_f1']:<14.4f}")

    gaps: Dict[str, float] = {}
    print()
    for fixed_view in FIXED_VIEWS:
        for model in MODEL_KINDS:
            gap = (results[("eye_in_hand", model)]["test_accuracy"]
                  - results[(fixed_view, model)]["test_accuracy"])
            gaps[f"accuracy_gap_{fixed_view}_{model}"] = gap
            print(f"accuracy gap (eye-in-hand - {VIEW_LABELS[fixed_view]}), "
                  f"{model.upper()}: {gap:+.4f}")

    summary = {
        **gaps,
        "results": {f"{view}_{model}": results[(view, model)]
                   for view in VIEWS for model in MODEL_KINDS},
    }
    save_json(summary, RESULTS_DIR / "comparison.json")

    _plot_confusion_matrices(results)
    _plot_accuracy_comparison(results)
    _plot_occlusion_vs_correctness()
    print(f"\nfigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
