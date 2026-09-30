"""dataset.py - Loading, splitting and feature extraction for the classification
benchmark.

Every record's ground-truth `class_id` comes from the simulator's own object state at
capture time (see `run_sim.generate_dataset`), not from a downstream detector - so the
split below is the only place where "training on the test set" could sneak in, and it is
done once, by object-independent stratified sampling, before any feature or model code
sees the data.

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils import (
    CNN_IMAGE_SIZE,
    RANDOM_SEED,
    TEST_FRACTION,
    VAL_FRACTION,
    image_path,
)

Record = Dict[str, object]


@dataclass(frozen=True)
class Split:
    """A named, stratified partition of the dataset."""

    name: str
    records: List[Record]

    @property
    def labels(self) -> np.ndarray:
        return np.array([r["class_id"] for r in self.records], dtype=np.int64)


def stratified_split(records: List[Record], seed: int = RANDOM_SEED,
                     val_fraction: float = VAL_FRACTION,
                     test_fraction: float = TEST_FRACTION,
                     ) -> Tuple[Split, Split, Split]:
    """70/15/15 (by default) stratified train/val/test split.

    Stratifying on `class_id` keeps the class balance of the full dataset in every
    partition, which matters here because the layout sampler does not guarantee an
    exactly uniform class distribution (see the README's class-distribution table).
    """
    labels = [r["class_id"] for r in records]
    trainval, test = train_test_split(
        records, test_size=test_fraction, random_state=seed, stratify=labels
    )
    trainval_labels = [r["class_id"] for r in trainval]
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        trainval, test_size=relative_val, random_state=seed, stratify=trainval_labels
    )
    return Split("train", train), Split("val", val), Split("test", test)


def load_rgb(view: str, record: Record, size: int = CNN_IMAGE_SIZE) -> np.ndarray:
    """Load one capture as an (size, size, 3) uint8 RGB array."""
    image = Image.open(image_path(view, record)).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    return np.asarray(image)


# ======================================================================================
# Classical shape-descriptor features, for the SVM baseline
# ======================================================================================

_HOG_WINDOW: int = 64
_HOG = cv2.HOGDescriptor(
    _winSize=(_HOG_WINDOW, _HOG_WINDOW),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def _silhouette_contour(rgb: np.ndarray) -> np.ndarray | None:
    """Largest foreground contour, isolated by HSV saturation.

    Every object in the scene is drawn from a saturated colour palette (see
    `run_sim.PALETTE`); the table and most of the arm are desaturated greys and steel
    tones by construction, so Otsu-thresholding the saturation channel mostly separates
    "an object" from "everything else" without any class-specific tuning. The claw's
    orange accent trim (fingers, pedestal ring) is the exception - it is saturated too,
    so a frame where a finger is prominent and the object is small or partly hidden can
    have the largest saturated contour be a finger, not the object (see the README's
    Limitations section).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def extract_shape_features(rgb: np.ndarray) -> np.ndarray:
    """HOG texture/edge histogram plus log-scaled Hu shape moments.

    HOG captures local edge orientation statistics (useful for corners vs. curves);
    Hu moments are translation-, scale- and rotation-invariant descriptors of the
    silhouette's shape. Concatenating both gives the SVM a texture-and-shape feature
    vector that does not depend on a learned representation, which is the point of a
    classical baseline.

    Args:
        rgb: (H, W, 3) uint8 image, any size.

    Returns:
        1-D float32 feature vector.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (_HOG_WINDOW, _HOG_WINDOW))
    hog_features = _HOG.compute(gray).flatten()

    contour = _silhouette_contour(rgb)
    if contour is not None:
        hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    else:
        hu = np.zeros(7, dtype=np.float64)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return np.concatenate([hog_features, hu_log]).astype(np.float32)


def build_feature_matrix(view: str, records: List[Record]) -> np.ndarray:
    """Extract shape features for every record, in order."""
    features = [extract_shape_features(load_rgb(view, r, size=_HOG_WINDOW * 2))
                for r in records]
    return np.stack(features)


# ======================================================================================
# PyTorch dataset, for the CNN
# ======================================================================================


class SimImageDataset(Dataset):
    """CHW float32 tensors in [0, 1], normalised per-channel, for `src.model.CompactCNN`."""

    MEAN = (0.5, 0.5, 0.5)
    STD = (0.22, 0.22, 0.22)

    def __init__(self, view: str, records: List[Record], augment: bool = False) -> None:
        self.view = view
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        rgb = load_rgb(self.view, record).astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1, :].copy()          # horizontal flip only - colour and
                                                  # exact orientation both carry no
                                                  # class signal by construction, but a
                                                  # vertical flip would invert "up",
                                                  # which the overhead crop still uses
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        return tensor, int(record["class_id"])
