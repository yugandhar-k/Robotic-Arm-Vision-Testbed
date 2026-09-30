"""utils.py - Shared constants and small helpers for the classification benchmark.

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "captures"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_SEED: int = 42

CLASS_NAMES: List[str] = ["cube", "sphere", "cylinder", "cuboid",
                          "capsule", "ellipsoid", "disk", "rod"]
NUM_CLASSES: int = len(CLASS_NAMES)

VIEWS: List[str] = ["eye_in_hand", "fixed_overhead", "fixed_angled"]
FIXED_VIEWS: List[str] = ["fixed_overhead", "fixed_angled"]   # compared against eye_in_hand
MODEL_KINDS: List[str] = ["cnn", "svm"]

TRAIN_FRACTION: float = 0.70
VAL_FRACTION: float = 0.15
TEST_FRACTION: float = 0.15
N_FOLDS: int = 5

CNN_IMAGE_SIZE: int = 96          # CNN input edge length, downsampled from the 224 px
                                   # captures - plenty of resolution for four primitives,
                                   # four times fewer pixels to convolve over than 224.


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every source of randomness this project touches."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:                                       # pragma: no cover
        pass


def load_records(view: str) -> List[Dict[str, object]]:
    """Read every labelled record for one camera view.

    Args:
        view: "eye_in_hand", "fixed_overhead", or "fixed_angled".

    Returns:
        List of JSON records, each carrying at least `file` and `class_id`.
    """
    labels_path = DATA_ROOT / view / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"no dataset at {labels_path} - run "
            f"`python run_sim.py --dataset 2000` first"
        )
    records = []
    with open(labels_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def image_path(view: str, record: Dict[str, object]) -> Path:
    return DATA_ROOT / view / str(record["file"])


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
