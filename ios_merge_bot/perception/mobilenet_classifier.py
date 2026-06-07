"""MobileNetV3 wrapper that classifies a unit crop into (unit_id, star_level).

The model has two classification heads sharing a MobileNetV3-Small backbone:

* ``unit_head`` -> unit ID (including an explicit EMPTY class at index 0).
* ``star_head`` -> star/level (1..max_star_level, plus a 0 = "no unit" class).

If no weights are provided the wrapper still constructs the network (randomly
initialized) and returns low-confidence stub predictions, so the rest of the
pipeline runs before you train and inject a real checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import ModelConfig

logger = logging.getLogger(__name__)

EMPTY_UNIT_ID = 0


@dataclass
class UnitPrediction:
    """A single classified cell."""

    unit_id: int
    star_level: int
    confidence: float

    @property
    def is_empty(self) -> bool:
        return self.unit_id == EMPTY_UNIT_ID


class _MobileNetTwoHead:
    """Lazily-built torch module; isolated so torch import stays optional-ish."""

    def __init__(self, num_units: int, num_stars: int) -> None:
        import torch.nn as nn
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.nn = nn
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        # Replace the final classifier with identity; attach two heads.
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone
        self.unit_head = nn.Linear(in_features, num_units)
        self.star_head = nn.Linear(in_features, num_stars)

    def to_module(self):
        import torch.nn as nn

        module = nn.Module()
        module.backbone = self.backbone
        module.unit_head = self.unit_head
        module.star_head = self.star_head

        def forward(x):
            feats = module.backbone(x)
            return module.unit_head(feats), module.star_head(feats)

        module.forward = forward  # type: ignore[assignment]
        return module


class UnitClassifier:
    """Classifies a board/shop cell crop into a unit ID and star level."""

    INPUT_SIZE: Tuple[int, int] = (96, 96)

    def __init__(self, config: ModelConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self._model = None
        self._torch = None
        self._available = False
        self._num_stars = config.max_star_level + 1  # +1 for "no unit" = 0
        self._try_build()

    def _try_build(self) -> None:
        try:
            import torch

            self._torch = torch
            builder = _MobileNetTwoHead(self.config.num_unit_classes, self._num_stars)
            model = builder.to_module().to(self.device)

            if self.config.classifier_weights:
                state = torch.load(self.config.classifier_weights, map_location=self.device)
                model.load_state_dict(state)
                logger.info("Loaded classifier weights from %s", self.config.classifier_weights)
            else:
                logger.warning("No classifier weights provided; using untrained stub model.")

            model.eval()
            self._model = model
            self._available = True
        except Exception as err:  # noqa: BLE001
            logger.warning("UnitClassifier unavailable (%s); returning EMPTY stubs.", err)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _preprocess(self, crop: np.ndarray):
        import cv2

        img = cv2.resize(crop, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalization (MobileNetV3 default weights expectation).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor = self._torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, crop: np.ndarray) -> UnitPrediction:
        """Classify a single BGR cell crop into a :class:`UnitPrediction`."""
        if not self._available or crop is None or crop.size == 0:
            return UnitPrediction(unit_id=EMPTY_UNIT_ID, star_level=0, confidence=0.0)

        torch = self._torch
        with torch.no_grad():
            tensor = self._preprocess(crop)
            unit_logits, star_logits = self._model(tensor)
            unit_probs = torch.softmax(unit_logits, dim=-1)
            star_probs = torch.softmax(star_logits, dim=-1)
            unit_conf, unit_id = torch.max(unit_probs, dim=-1)
            _, star_level = torch.max(star_probs, dim=-1)

        return UnitPrediction(
            unit_id=int(unit_id.item()),
            star_level=int(star_level.item()),
            confidence=float(unit_conf.item()),
        )

    def classify_batch(self, crops: list[np.ndarray]) -> list[UnitPrediction]:
        """Classify a list of crops (simple per-crop loop; batchable later)."""
        return [self.classify(c) for c in crops]
