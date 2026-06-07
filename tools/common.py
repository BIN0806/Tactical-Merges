"""Shared helpers for the calibration / data-collection tools.

Centralizes WDA connection + frame-source startup and the on-disk paths so
``data_collector.py`` and ``label_helper.py`` stay focused on their workflows.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from ios_merge_bot.config import BotConfig
from ios_merge_bot.io_layer.coordinates import CoordinateMapper
from ios_merge_bot.io_layer.screen_capture import FrameSource, build_frame_source
from ios_merge_bot.io_layer.wda_client import WDAClient

logger = logging.getLogger(__name__)

# Repository root == parent of the tools/ package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
RAW_CELLS_DIR = os.path.join(DATASET_DIR, "raw_cells")
RAW_SHOP_DIR = os.path.join(DATASET_DIR, "raw_shop")
LABELED_DIR = os.path.join(DATASET_DIR, "labeled")
CALIBRATION_PATH = os.path.join(DATASET_DIR, "calibration.json")


def ensure_dirs(*paths: str) -> None:
    """Create each directory (and parents) if it does not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


@dataclass
class LiveSession:
    """A connected WDA client plus a running frame source."""

    client: WDAClient
    frames: FrameSource

    @property
    def mapper(self) -> CoordinateMapper:
        return self.client.mapper

    def close(self) -> None:
        try:
            self.frames.stop()
        except Exception as err:  # noqa: BLE001
            logger.debug("Error stopping frame source: %s", err)
        self.client.close()

    def __enter__(self) -> "LiveSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def open_live_session(config: Optional[BotConfig] = None) -> LiveSession:
    """Connect to WDA and start the configured capture backend."""
    config = config or BotConfig()
    logger.info("Connecting to WebDriverAgent at %s ...", config.device.wda_url)
    client = WDAClient(config).connect()
    logger.info("Starting capture backend: %s", config.capture.backend.value)
    frames = build_frame_source(config, client.mjpeg_url).start()
    return LiveSession(client=client, frames=frames)


def fit_to_screen(width: int, height: int, max_w: int = 1100, max_h: int = 900) -> float:
    """Return a display scale factor that fits ``(width, height)`` on screen.

    Returns 1.0 when the frame already fits; otherwise a value < 1.0 so callers
    can map display-space clicks back to true frame pixels via ``click / scale``.
    """
    scale = min(max_w / width, max_h / height, 1.0)
    return scale


def display_to_frame(point: Tuple[int, int], display_scale: float) -> Tuple[int, int]:
    """Map a click in the displayed (possibly downscaled) image back to frame pixels."""
    x, y = point
    return int(round(x / display_scale)), int(round(y / display_scale))
