"""OpenCV state extraction: frame -> structured game state.

Uses :class:`CoordinateMapper` to crop the 16 board cells, the 3 shop slots,
and the elixir region from a captured BGR frame (physical Retina pixels), then:

* runs a cheap template/occupancy check to decide if a cell is populated, and
* delegates populated cells to :class:`UnitClassifier` for unit ID + star level.

The output is a plain ``GameState`` dataclass that the Gymnasium environment
converts into its observation dictionary.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import BoardGeometry, ModelConfig
from ..io_layer.coordinates import CoordinateMapper
from .mobilenet_classifier import EMPTY_UNIT_ID, UnitClassifier, UnitPrediction

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Structured snapshot of the game extracted from a single frame."""

    board_units: np.ndarray  # int32 [rows, cols] of unit IDs (0 == empty)
    board_stars: np.ndarray  # int32 [rows, cols] of star levels (0 == empty)
    shop_units: np.ndarray  # int32 [shop_slots] of unit IDs (0 == empty)
    elixir: int
    raw_predictions: Dict[str, List[UnitPrediction]] = field(default_factory=dict)


class StateExtractor:
    """Extracts a :class:`GameState` from a captured frame."""

    def __init__(
        self,
        mapper: CoordinateMapper,
        classifier: UnitClassifier,
        model_cfg: ModelConfig,
        occupancy_threshold: float = 12.0,
    ) -> None:
        self.mapper = mapper
        self.classifier = classifier
        self.geometry: BoardGeometry = mapper.geometry
        self.model_cfg = model_cfg
        # Std-dev of pixel intensity below which a cell is treated as empty.
        self.occupancy_threshold = occupancy_threshold
        self._templates = self._load_templates(model_cfg.template_dir)

    # ------------------------------------------------------------------ #
    # Template loading (optional)
    # ------------------------------------------------------------------ #
    def _load_templates(self, template_dir: Optional[str]) -> Dict[str, np.ndarray]:
        if not template_dir or not os.path.isdir(template_dir):
            if template_dir:
                logger.warning("Template dir %s not found; occupancy via variance only.", template_dir)
            return {}
        import cv2

        templates: Dict[str, np.ndarray] = {}
        for name in os.listdir(template_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(template_dir, name)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    templates[os.path.splitext(name)[0]] = img
        logger.info("Loaded %d template images.", len(templates))
        return templates

    # ------------------------------------------------------------------ #
    # Cropping helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_frame, x + w), min(h_frame, y + h)
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y0:y1, x0:x1]

    def _is_occupied(self, crop: np.ndarray) -> bool:
        """Cheap occupancy test: an empty slot has near-uniform color."""
        if crop is None or crop.size == 0:
            return False
        return float(np.std(crop)) > self.occupancy_threshold

    # ------------------------------------------------------------------ #
    # Main extraction
    # ------------------------------------------------------------------ #
    def extract(self, frame: np.ndarray) -> GameState:
        rows, cols = self.geometry.rows, self.geometry.cols
        board_units = np.zeros((rows, cols), dtype=np.int32)
        board_stars = np.zeros((rows, cols), dtype=np.int32)
        board_preds: List[UnitPrediction] = []

        for r in range(rows):
            for c in range(cols):
                crop = self._crop(frame, self.mapper.cell_bbox_pixel(r, c))
                if self._is_occupied(crop):
                    pred = self.classifier.classify(crop)
                else:
                    pred = UnitPrediction(EMPTY_UNIT_ID, 0, 0.0)
                board_units[r, c] = pred.unit_id
                board_stars[r, c] = pred.star_level
                board_preds.append(pred)

        shop_units = np.zeros((self.geometry.shop_slots,), dtype=np.int32)
        shop_preds: List[UnitPrediction] = []
        for s in range(self.geometry.shop_slots):
            crop = self._crop(frame, self.mapper.shop_bbox_pixel(s))
            if self._is_occupied(crop):
                pred = self.classifier.classify(crop)
            else:
                pred = UnitPrediction(EMPTY_UNIT_ID, 0, 0.0)
            shop_units[s] = pred.unit_id
            shop_preds.append(pred)

        elixir = self._read_elixir(frame)

        return GameState(
            board_units=board_units,
            board_stars=board_stars,
            shop_units=shop_units,
            elixir=elixir,
            raw_predictions={"board": board_preds, "shop": shop_preds},
        )

    # ------------------------------------------------------------------ #
    # Elixir OCR (placeholder heuristic)
    # ------------------------------------------------------------------ #
    def _read_elixir(self, frame: np.ndarray) -> int:
        """Estimate the elixir count from its UI region.

        Placeholder: returns 0 until a digit classifier / OCR template set is
        injected. Inject 7-segment templates or a small CNN here later.
        """
        crop = self._crop(frame, self.mapper.elixir_bbox_pixel())
        if crop.size == 0:
            return 0
        # TODO(perception): replace with digit template matching / OCR.
        return 0
