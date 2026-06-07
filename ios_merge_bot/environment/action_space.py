"""Discretized action space for Merge Tactics.

The flat ``Discrete(N)`` action index encodes one of four intent kinds:

* ``NOOP``  - do nothing this step.
* ``REROLL`` - refresh the shop.
* ``BUY``   - purchase shop slot ``s`` (dragged onto the board / bench).
* ``MOVE``  - drag a unit from board cell ``src`` to cell ``dst``. When
  ``src == dst`` this is treated as a tap/select; when ``dst`` holds a matching
  unit it becomes a merge. The physical layer (HumanizedTouch) performs an
  identical drag for moves and merges; the game resolves the semantics.

Layout of the flat index (board has ``rows*cols == n_cells`` cells)::

    [ 0 ]                              NOOP
    [ 1 ]                              REROLL
    [ 2 .. 2 + shop_slots )            BUY slot s
    [ buy_end .. buy_end + n_cells^2 ) MOVE src->dst   (src*n_cells + dst)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

from ..config import BoardGeometry
from ..io_layer.coordinates import CoordinateMapper

logger = logging.getLogger(__name__)


class ActionKind(IntEnum):
    NOOP = 0
    REROLL = 1
    BUY = 2
    MOVE = 3


@dataclass
class ActionIntent:
    """Decoded, physically-actionable description of a discrete action."""

    kind: ActionKind
    shop_slot: Optional[int] = None
    src_cell: Optional[Tuple[int, int]] = None
    dst_cell: Optional[Tuple[int, int]] = None

    # Logical-point coordinates (filled by ActionSpace.decode for the input layer).
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None


class ActionSpace:
    """Builds and decodes the discrete action index for a given board geometry."""

    def __init__(self, geometry: BoardGeometry, mapper: Optional[CoordinateMapper] = None) -> None:
        self.geometry = geometry
        self.mapper = mapper
        self.n_cells = geometry.rows * geometry.cols
        self.shop_slots = geometry.shop_slots

        self._buy_start = 2
        self._buy_end = self._buy_start + self.shop_slots
        self._move_start = self._buy_end
        self._move_end = self._move_start + self.n_cells * self.n_cells
        self.size = self._move_end

    # ------------------------------------------------------------------ #
    # gymnasium space
    # ------------------------------------------------------------------ #
    def to_gym_space(self):
        from gymnasium import spaces

        return spaces.Discrete(self.size)

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode(self, action: int) -> ActionIntent:
        """Map a flat action index to a coordinate-resolved :class:`ActionIntent`."""
        if not 0 <= action < self.size:
            raise IndexError(f"action {action} out of range [0, {self.size})")

        if action == 0:
            return ActionIntent(kind=ActionKind.NOOP)

        if action == 1:
            intent = ActionIntent(kind=ActionKind.REROLL)
            return self._attach_reroll_point(intent)

        if self._buy_start <= action < self._buy_end:
            slot = action - self._buy_start
            intent = ActionIntent(kind=ActionKind.BUY, shop_slot=slot)
            return self._attach_buy_points(intent)

        # MOVE / MERGE
        offset = action - self._move_start
        src_idx, dst_idx = divmod(offset, self.n_cells)
        src = divmod(src_idx, self.geometry.cols)
        dst = divmod(dst_idx, self.geometry.cols)
        intent = ActionIntent(kind=ActionKind.MOVE, src_cell=src, dst_cell=dst)
        return self._attach_move_points(intent)

    # ------------------------------------------------------------------ #
    # coordinate attachment
    # ------------------------------------------------------------------ #
    def _attach_move_points(self, intent: ActionIntent) -> ActionIntent:
        if self.mapper is not None and intent.src_cell and intent.dst_cell:
            intent.start_point = self.mapper.cell_center_logical(*intent.src_cell)
            intent.end_point = self.mapper.cell_center_logical(*intent.dst_cell)
        return intent

    def _attach_buy_points(self, intent: ActionIntent) -> ActionIntent:
        if self.mapper is not None and intent.shop_slot is not None:
            intent.start_point = self.mapper.shop_center_logical(intent.shop_slot)
            # Default drop target: center of the board; refine with bench logic later.
            mid_r = self.geometry.rows // 2
            mid_c = self.geometry.cols // 2
            intent.end_point = self.mapper.cell_center_logical(mid_r, mid_c)
        return intent

    def _attach_reroll_point(self, intent: ActionIntent) -> ActionIntent:
        if self.mapper is not None and self.shop_slots > 0:
            # Reroll button assumed just left of the first shop slot; tap there.
            sx, sy = self.mapper.shop_center_logical(0)
            slot_w = self.geometry.shop_slot_size[0]
            intent.start_point = (sx - slot_w, sy)
        return intent
