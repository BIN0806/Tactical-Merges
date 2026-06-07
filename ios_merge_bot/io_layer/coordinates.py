"""Coordinate mapping between the mathematical grid and physical iOS pixels.

There are three coordinate systems in play:

* **Grid coordinates** ``(row, col)`` - the mathematical 4x4 board indices and
  the 1-D shop slot index used by the RL agent.
* **Logical points** ``(x, y)`` - the resolution-independent coordinate system
  WebDriverAgent uses for taps and swipes (matches ``size`` returned by WDA).
* **Physical pixels** ``(px, py)`` - the actual Retina pixels in a captured
  screenshot/frame, related to logical points by ``scale_factor`` (e.g. 3.0).

All Retina scaling lives here so the rest of the codebase only ever speaks in
grid coordinates (for the agent) or logical points (for WDA input).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..config import BoardGeometry


@dataclass
class CoordinateMapper:
    """Convert between grid, logical-point, and physical-pixel coordinates."""

    geometry: BoardGeometry
    logical_size: Tuple[int, int]  # (width, height) in logical points
    scale_factor: float  # logical point -> physical pixel multiplier

    # ------------------------------------------------------------------ #
    # Logical points <-> physical pixels (Retina scaling)
    # ------------------------------------------------------------------ #
    def logical_to_pixel(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert a logical point to an integer physical pixel coordinate."""
        x, y = point
        return int(round(x * self.scale_factor)), int(round(y * self.scale_factor))

    def pixel_to_logical(self, pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Convert a physical pixel coordinate back to logical points."""
        px, py = pixel
        return px / self.scale_factor, py / self.scale_factor

    # ------------------------------------------------------------------ #
    # Board grid -> logical points
    # ------------------------------------------------------------------ #
    def cell_center_logical(self, row: int, col: int) -> Tuple[float, float]:
        """Center of board cell ``(row, col)`` in logical points (for WDA taps)."""
        self._check_cell(row, col)
        ox, oy = self.geometry.origin
        cw, ch = self.geometry.cell_size
        x = ox + (col + 0.5) * cw
        y = oy + (row + 0.5) * ch
        return x, y

    def cell_center_pixel(self, row: int, col: int) -> Tuple[int, int]:
        """Center of board cell ``(row, col)`` in physical pixels (for cropping)."""
        return self.logical_to_pixel(self.cell_center_logical(row, col))

    def cell_bbox_pixel(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Bounding box ``(x, y, w, h)`` of a board cell in physical pixels."""
        self._check_cell(row, col)
        ox, oy = self.geometry.origin
        cw, ch = self.geometry.cell_size
        left = ox + col * cw
        top = oy + row * ch
        x, y = self.logical_to_pixel((left, top))
        w = int(round(cw * self.scale_factor))
        h = int(round(ch * self.scale_factor))
        return x, y, w, h

    # ------------------------------------------------------------------ #
    # Shop slots -> logical points
    # ------------------------------------------------------------------ #
    def shop_center_logical(self, slot: int) -> Tuple[float, float]:
        """Center of shop ``slot`` in logical points."""
        self._check_shop(slot)
        ox, oy = self.geometry.shop_origin
        sw, sh = self.geometry.shop_slot_size
        x = ox + (slot + 0.5) * sw
        y = oy + 0.5 * sh
        return x, y

    def shop_center_pixel(self, slot: int) -> Tuple[int, int]:
        return self.logical_to_pixel(self.shop_center_logical(slot))

    def shop_bbox_pixel(self, slot: int) -> Tuple[int, int, int, int]:
        """Bounding box ``(x, y, w, h)`` of a shop slot in physical pixels."""
        self._check_shop(slot)
        ox, oy = self.geometry.shop_origin
        sw, sh = self.geometry.shop_slot_size
        left = ox + slot * sw
        x, y = self.logical_to_pixel((left, oy))
        w = int(round(sw * self.scale_factor))
        h = int(round(sh * self.scale_factor))
        return x, y, w, h

    def elixir_bbox_pixel(self) -> Tuple[int, int, int, int]:
        """Bounding box ``(x, y, w, h)`` of the elixir counter in physical pixels."""
        ex, ey, ew, eh = self.geometry.elixir_region
        x, y = self.logical_to_pixel((ex, ey))
        w = int(round(ew * self.scale_factor))
        h = int(round(eh * self.scale_factor))
        return x, y, w, h

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def index_to_cell(self, index: int) -> Tuple[int, int]:
        """Flat board index (row-major) -> ``(row, col)``."""
        if not 0 <= index < self.geometry.rows * self.geometry.cols:
            raise IndexError(f"board index {index} out of range")
        return divmod(index, self.geometry.cols)

    def cell_to_index(self, row: int, col: int) -> int:
        """``(row, col)`` -> flat board index (row-major)."""
        self._check_cell(row, col)
        return row * self.geometry.cols + col

    def _check_cell(self, row: int, col: int) -> None:
        if not (0 <= row < self.geometry.rows and 0 <= col < self.geometry.cols):
            raise IndexError(f"cell ({row}, {col}) out of {self.geometry.rows}x{self.geometry.cols} board")

    def _check_shop(self, slot: int) -> None:
        if not 0 <= slot < self.geometry.shop_slots:
            raise IndexError(f"shop slot {slot} out of range [0, {self.geometry.shop_slots})")
