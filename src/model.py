"""model.py - The two classifiers being compared: a compact CNN and a classical
HOG + Hu-moment shape descriptor pipeline feeding an RBF-kernel SVM.

Author: Yugandhar Kulkarni - DRDO Research Intern - DES Pune University
"""

from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import NUM_CLASSES, RANDOM_SEED


class CompactCNN(nn.Module):
    """A small 4-block CNN classifier.

    Four primitives distinguished mainly by silhouette and edge structure do not need
    a deep network; this is sized to train from scratch on a few thousand 96x96 images
    on a CPU in minutes, not a pretrained backbone fine-tuned on ImageNet statistics
    that have nothing to do with this scene.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 16),
            self._block(16, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def build_svm_pipeline(C: float = 4.0, gamma: str = "scale",
                       seed: int = RANDOM_SEED) -> Pipeline:
    """Standardised HOG+Hu features -> RBF-kernel SVM.

    `C` and `gamma` were chosen by a 5-fold grid search on the eye-in-hand train+val
    split only (C in {1, 2, 4, 8, 16}, gamma in {"scale", 0.01, 0.001}; best mean CV
    accuracy 0.9002 +/- 0.0042 at C=4, gamma="scale", on the 5,100-frame, 8-class
    train+val split) and then reused as-is for both fixed cameras, so neither one's
    result is the product of a baseline tuned against it.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(C=C, gamma=gamma, kernel="rbf", random_state=seed)),
    ])
