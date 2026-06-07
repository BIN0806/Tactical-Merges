"""Central configuration for the iOS Merge Tactics bot.

All device geometry, capture, and model paths live here so that real
template images / trained weights and per-device calibration can be
injected without touching the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class CaptureBackend(str, Enum):
    """Selects how frames are pulled from the iOS device."""

    WDA_MJPEG = "wda_mjpeg"  # WebDriverAgent MJPEG stream
    MSS_WINDOW = "mss_window"  # macOS native capture of a QuickTime mirror window


@dataclass
class BoardGeometry:
    """Geometry of the game board / shop in *logical* screen points.

    Logical points are the resolution-independent coordinate system used by
    WebDriverAgent. ``CoordinateMapper`` converts these to physical Retina
    pixels for the vision pipeline.

    The board is a ``rows x cols`` grid. ``origin`` is the top-left corner of
    cell (0, 0); ``cell_size`` is the width/height of a single cell in points.
    """

    rows: int = 4
    cols: int = 4
    origin: Tuple[float, float] = (40.0, 360.0)
    cell_size: Tuple[float, float] = (75.0, 75.0)

    # Shop row: a horizontal strip of ``shop_slots`` cells.
    shop_slots: int = 3
    shop_origin: Tuple[float, float] = (60.0, 760.0)
    shop_slot_size: Tuple[float, float] = (90.0, 110.0)

    # Region (x, y, w, h) in logical points where the elixir counter renders.
    elixir_region: Tuple[float, float, float, float] = (20.0, 700.0, 120.0, 40.0)


@dataclass
class DeviceConfig:
    """Connection settings for the iPhone under test."""

    # facebook-wda connection URL. usbmux: requires `iproxy 8100 8100`.
    wda_url: str = "http://localhost:8100"
    udid: Optional[str] = None

    # Logical screen size (points) reported by WDA, e.g. iPhone 14 -> (390, 844).
    # Left None to auto-detect from the live WDA session at startup.
    logical_size: Optional[Tuple[int, int]] = None

    # Retina scale factor (points -> physical pixels). iPhone 14 == 3.0.
    # Auto-detected from screenshot size / logical size when None.
    scale_factor: Optional[float] = None


@dataclass
class CaptureConfig:
    """Screen capture settings."""

    backend: CaptureBackend = CaptureBackend.WDA_MJPEG
    mjpeg_port: int = 9100
    target_fps: int = 10

    # For MSS_WINDOW: bounding box (left, top, width, height) of the mirrored
    # QuickTime window in *display* pixels. Calibrate per setup.
    window_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class ModelConfig:
    """Paths to injectable assets and model hyper-parameters."""

    classifier_weights: Optional[str] = None  # MobileNetV3 unit classifier .pt
    template_dir: Optional[str] = None  # directory of OpenCV template images
    num_unit_classes: int = 32  # distinct unit IDs the classifier can emit
    max_star_level: int = 3


@dataclass
class TimingConfig:
    """Latency / humanization timing knobs (seconds)."""

    game_latency: float = 0.8  # wait after an action before reading next state
    tap_delay_mean: float = 0.12
    tap_delay_std: float = 0.04
    drag_step_delay_mean: float = 0.015
    drag_step_delay_std: float = 0.006


@dataclass
class BotConfig:
    """Top-level config aggregating all sub-configs."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    board: BoardGeometry = field(default_factory=BoardGeometry)
    model: ModelConfig = field(default_factory=ModelConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)


DEFAULT_CONFIG = BotConfig()
