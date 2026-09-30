"""train.py - Train one (view, model) combination: 5-fold cross-validation for a
robustness estimate, then a single fit on the full train+val split evaluated once on
the held-out test split.

Usage
-----
    python -m src.train --view eye_in_hand    --model cnn
    python -m src.train --view eye_in_hand    --model svm
    python -m src.train --view fixed_overhead --model cnn
    python -m src.train --view fixed_overhead --model svm
    python -m src.train --view fixed_angled   --model cnn
    python -m src.train --view fixed_angled   --model svm

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.dataset import (
    Record,
    SimImageDataset,
    build_feature_matrix,
    stratified_split,
)
from src.model import CompactCNN, build_svm_pipeline
from src.utils import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    N_FOLDS,
    NUM_CLASSES,
    RANDOM_SEED,
    RESULTS_DIR,
    VIEWS,
    load_records,
    save_json,
    set_seed,
)

CNN_EPOCHS_CV: int = 18
CNN_EPOCHS_FINAL: int = 25
CNN_BATCH_SIZE: int = 64
CNN_LEARNING_RATE: float = 1e-3
CNN_WEIGHT_DECAY: float = 1e-4


# ======================================================================================
# CNN training / evaluation
# ======================================================================================


def _fit_cnn(view: str, train_records: List[Record], epochs: int,
            device: torch.device, augment: bool = True) -> CompactCNN:
    """Fit one CNN from a fresh initialisation.

    A fixed 10-epoch schedule was tried first and was unstable - cross-validation fold
    accuracies ranged from 0.58 to 0.94 on identical hyperparameters, which is
    under-training, not a property of the data. A cosine-annealed learning rate over
    more epochs converges consistently across folds (see the README's methodology
    note for the before/after numbers).
    """
    model = CompactCNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE,
                                 weight_decay=CNN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        SimImageDataset(view, train_records, augment=augment),
        batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=0,
    )
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def _predict_cnn(model: CompactCNN, view: str, records: List[Record],
                 device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        SimImageDataset(view, records, augment=False),
        batch_size=CNN_BATCH_SIZE, shuffle=False, num_workers=0,
    )
    predictions = []
    for images, _ in loader:
        logits = model(images.to(device))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _cross_validate_cnn(view: str, trainval: List[Record], device: torch.device,
                        seed: int) -> List[Dict[str, float]]:
    labels = np.array([r["class_id"] for r in trainval])
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []
    for fold_index, (train_idx, test_idx) in enumerate(folds.split(trainval, labels)):
        fold_train = [trainval[i] for i in train_idx]
        fold_test = [trainval[i] for i in test_idx]
        model = _fit_cnn(view, fold_train, CNN_EPOCHS_CV, device)
        predictions = _predict_cnn(model, view, fold_test, device)
        truth = np.array([r["class_id"] for r in fold_test])
        accuracy = accuracy_score(truth, predictions)
        macro_f1 = f1_score(truth, predictions, average="macro")
        fold_scores.append({"fold": fold_index, "accuracy": accuracy, "macro_f1": macro_f1})
        print(f"    fold {fold_index}: accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}")
    return fold_scores


def run_cnn(view: str, seed: int) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(view)
    train_split, val_split, test_split = stratified_split(records, seed=seed)
    trainval = train_split.records + val_split.records

    print(f"  5-fold cross-validation over {len(trainval)} train+val frames "
          f"({device})...")
    fold_scores = _cross_validate_cnn(view, trainval, device, seed)

    print(f"  final fit on all {len(trainval)} train+val frames...")
    final_model = _fit_cnn(view, trainval, CNN_EPOCHS_FINAL, device)
    test_predictions = _predict_cnn(final_model, view, test_split.records, device)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), CHECKPOINT_DIR / f"{view}_cnn.pt")

    return _summarise(view, "cnn", seed, fold_scores, test_split.records,
                      test_predictions, len(trainval))


# ======================================================================================
# SVM training / evaluation
# ======================================================================================


def _cross_validate_svm(features: np.ndarray, labels: np.ndarray,
                        seed: int) -> List[Dict[str, float]]:
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_scores = []
    for fold_index, (train_idx, test_idx) in enumerate(folds.split(features, labels)):
        pipeline = clone(build_svm_pipeline(seed=seed))
        pipeline.fit(features[train_idx], labels[train_idx])
        predictions = pipeline.predict(features[test_idx])
        truth = labels[test_idx]
        accuracy = accuracy_score(truth, predictions)
        macro_f1 = f1_score(truth, predictions, average="macro")
        fold_scores.append({"fold": fold_index, "accuracy": accuracy, "macro_f1": macro_f1})
        print(f"    fold {fold_index}: accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}")
    return fold_scores


def run_svm(view: str, seed: int) -> Dict[str, object]:
    records = load_records(view)
    train_split, val_split, test_split = stratified_split(records, seed=seed)
    trainval = train_split.records + val_split.records

    print(f"  extracting HOG + Hu-moment features for {len(records)} frames...")
    trainval_features = build_feature_matrix(view, trainval)
    trainval_labels = np.array([r["class_id"] for r in trainval])
    test_features = build_feature_matrix(view, test_split.records)

    print(f"  5-fold cross-validation over {len(trainval)} train+val frames...")
    fold_scores = _cross_validate_svm(trainval_features, trainval_labels, seed)

    print(f"  final fit on all {len(trainval)} train+val frames...")
    final_pipeline = build_svm_pipeline(seed=seed)
    final_pipeline.fit(trainval_features, trainval_labels)
    test_predictions = final_pipeline.predict(test_features)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(final_pipeline, CHECKPOINT_DIR / f"{view}_svm.joblib")

    return _summarise(view, "svm", seed, fold_scores, test_split.records,
                      test_predictions, len(trainval))


# ======================================================================================
# Shared: metrics, persistence
# ======================================================================================


def _summarise(view: str, model_kind: str, seed: int, fold_scores: List[Dict[str, float]],
              test_records: List[Record], test_predictions: np.ndarray,
              n_trainval: int) -> Dict[str, object]:
    test_truth = np.array([r["class_id"] for r in test_records])
    accuracy = float(accuracy_score(test_truth, test_predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        test_truth, test_predictions, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    matrix = confusion_matrix(test_truth, test_predictions, labels=list(range(NUM_CLASSES)))

    cv_accuracies = [f["accuracy"] for f in fold_scores]
    cv_f1s = [f["macro_f1"] for f in fold_scores]

    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(NUM_CLASSES)
    }

    results = {
        "view": view,
        "model": model_kind,
        "seed": seed,
        "n_train_val": n_trainval,
        "n_test": len(test_records),
        "cv_folds": fold_scores,
        "cv_accuracy_mean": float(np.mean(cv_accuracies)),
        "cv_accuracy_std": float(np.std(cv_accuracies)),
        "cv_macro_f1_mean": float(np.mean(cv_f1s)),
        "cv_macro_f1_std": float(np.std(cv_f1s)),
        "test_accuracy": accuracy,
        "test_macro_f1": float(f1_score(test_truth, test_predictions, average="macro")),
        "test_per_class": per_class,
        "test_confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    save_json(results, RESULTS_DIR / f"{view}_{model_kind}.json")

    predictions_payload = {
        "class_names": CLASS_NAMES,
        "records": [
            {
                "file": r["file"],
                "true_class_id": int(r["class_id"]),
                "pred_class_id": int(p),
                "occluded_fraction": r.get("occluded_fraction"),
                "hover_height_m": r.get("hover_height_m"),
            }
            for r, p in zip(test_records, test_predictions)
        ],
    }
    save_json(predictions_payload, RESULTS_DIR / f"{view}_{model_kind}_test_predictions.json")

    print(f"\n  test accuracy: {accuracy:.4f}   test macro-F1: {results['test_macro_f1']:.4f}")
    print(f"  cv accuracy:   {results['cv_accuracy_mean']:.4f} +/- "
          f"{results['cv_accuracy_std']:.4f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", required=True, choices=VIEWS)
    parser.add_argument("--model", required=True, choices=["cnn", "svm"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"[train] view={args.view} model={args.model} seed={args.seed}")
    start = time.time()
    if args.model == "cnn":
        run_cnn(args.view, args.seed)
    else:
        run_svm(args.view, args.seed)
    print(f"[train] done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
