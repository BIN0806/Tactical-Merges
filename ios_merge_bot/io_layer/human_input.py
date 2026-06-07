"""Humanized touch input for anti-cheat / behavioral-biometrics evasion.

Instead of an instantaneous ``session.swipe(x1, y1, x2, y2)``, drags are
decomposed into many micro-steps following a cubic Bezier curve, each executed
as a short ``swipe`` with Gaussian-noised sleep timings. Taps add positional
jitter and randomized pre/post delays so the input stream is not mechanically
periodic.

Coordinates are in *logical points* (the WebDriverAgent coordinate system).
"""

from __future__ import annotations

import logging
import random
import time
from typing import List, Optional, Sequence, Tuple

from ..config import TimingConfig

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def generate_bezier_path(
    start: Point,
    end: Point,
    control_points: Sequence[Point],
    steps: int,
) -> List[Point]:
    """Sample a cubic Bezier curve into ``steps`` points (inclusive of endpoints).

    Uses two control points (cubic Bezier). If fewer than two are supplied, the
    missing ones are synthesized from the start/end so the call never fails.

    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3,  t in [0, 1]
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")

    p0 = start
    p3 = end
    ctrl = list(control_points)
    if len(ctrl) >= 2:
        p1, p2 = ctrl[0], ctrl[1]
    elif len(ctrl) == 1:
        p1 = ctrl[0]
        p2 = ((ctrl[0][0] + end[0]) / 2.0, (ctrl[0][1] + end[1]) / 2.0)
    else:
        p1 = ((2 * start[0] + end[0]) / 3.0, (2 * start[1] + end[1]) / 3.0)
        p2 = ((start[0] + 2 * end[0]) / 3.0, (start[1] + 2 * end[1]) / 3.0)

    path: List[Point] = []
    for i in range(steps):
        t = i / (steps - 1)
        mt = 1.0 - t
        x = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
        y = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
        path.append((x, y))
    return path


def _gaussian_sleep(mean: float, std: float) -> None:
    """Sleep for a Gaussian-distributed, non-negative duration."""
    delay = max(0.0, random.gauss(mean, std))
    time.sleep(delay)


class HumanizedTouch:
    """Sends human-like taps and drags through a WebDriverAgent session."""

    def __init__(self, wda_session: object, timing: Optional[TimingConfig] = None) -> None:
        self.session = wda_session
        self.timing = timing or TimingConfig()

    # ------------------------------------------------------------------ #
    # Drag
    # ------------------------------------------------------------------ #
    def _random_control_points(self, start: Point, end: Point) -> List[Point]:
        """Generate two control points perpendicular-ish to the drag for a natural arc."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        # Perpendicular unit vector for lateral bow.
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        px, py = -dy / length, dx / length
        bow = random.uniform(-0.18, 0.18) * length

        def lerp(a: Point, b: Point, frac: float) -> Point:
            return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)

        c1 = lerp(start, end, random.uniform(0.2, 0.4))
        c2 = lerp(start, end, random.uniform(0.6, 0.8))
        c1 = (c1[0] + px * bow * 0.5, c1[1] + py * bow * 0.5)
        c2 = (c2[0] + px * bow, c2[1] + py * bow)
        return [c1, c2]

    def human_drag(
        self,
        wda_session: Optional[object] = None,
        start_coord: Point = (0.0, 0.0),
        end_coord: Point = (0.0, 0.0),
        steps: int = 24,
        duration: float = 0.45,
    ) -> None:
        """Execute a Bezier-path drag from ``start_coord`` to ``end_coord``.

        Each micro-segment is a tiny ``session.swipe`` and the per-step pause is
        drawn from a Gaussian centered on ``drag_step_delay_mean`` to defeat
        timing-based behavioral biometrics. ``wda_session`` may override the
        instance session for one call (matches the requested signature).
        """
        session = wda_session or self.session
        if session is None:
            raise RuntimeError("No WDA session available for human_drag")

        control = self._random_control_points(start_coord, end_coord)
        path = generate_bezier_path(start_coord, end_coord, control, steps)

        # Initial dwell before the gesture begins.
        _gaussian_sleep(self.timing.tap_delay_mean, self.timing.tap_delay_std)

        per_seg = max(0.005, duration / max(1, len(path) - 1))
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            try:
                session.swipe(x1, y1, x2, y2, per_seg)  # type: ignore[attr-defined]
            except Exception as err:  # noqa: BLE001
                logger.warning("swipe segment failed: %s", err)
            _gaussian_sleep(self.timing.drag_step_delay_mean, self.timing.drag_step_delay_std)

        logger.debug("human_drag %s -> %s over %d steps", start_coord, end_coord, steps)

    # ------------------------------------------------------------------ #
    # Tap
    # ------------------------------------------------------------------ #
    def human_tap(self, coord: Point, jitter: float = 3.0) -> None:
        """Tap ``coord`` with small positional jitter and randomized delays."""
        if self.session is None:
            raise RuntimeError("No WDA session available for human_tap")

        _gaussian_sleep(self.timing.tap_delay_mean, self.timing.tap_delay_std)
        x = coord[0] + random.gauss(0.0, jitter)
        y = coord[1] + random.gauss(0.0, jitter)
        try:
            self.session.tap(x, y)  # type: ignore[attr-defined]
        except Exception as err:  # noqa: BLE001
            logger.warning("tap failed: %s", err)
        _gaussian_sleep(self.timing.tap_delay_mean, self.timing.tap_delay_std)
        logger.debug("human_tap at (%.1f, %.1f)", x, y)
