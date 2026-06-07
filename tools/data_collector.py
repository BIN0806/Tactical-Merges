"""Interactive calibration + raw data-collection harness.

Two subcommands:

* ``calibrate`` - click the board / shop / elixir corners on a live frame to
  derive a real :class:`BoardGeometry` in logical points. Before printing the
  paste-ready snippet it performs **on-device verification taps** on the two
  computed board corners so you can confirm the Retina math is correct.

* ``collect`` - using the calibrated geometry, watch the live stream and dump
  crops of the 16 board cells + 3 shop slots whenever a *structural* change is
  detected, throttled by a hard cooldown so idle animations don't flood the set.

Run from the repo root::

    python -m tools.data_collector calibrate
    python -m tools.data_collector collect --interval 1.0 --cooldown 3.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ios_merge_bot.config import BoardGeometry, BotConfig
from ios_merge_bot.io_layer.coordinates import CoordinateMapper

from .common import (
    CALIBRATION_PATH,
    RAW_CELLS_DIR,
    RAW_SHOP_DIR,
    LiveSession,
    display_to_frame,
    ensure_dirs,
    fit_to_screen,
    open_live_session,
)

logger = logging.getLogger(__name__)

WINDOW = "Merge Tactics - data_collector"

# Ordered calibration steps: (key, on-screen prompt).
CALIB_STEPS: List[Tuple[str, str]] = [
    ("board_tl", "Click BOARD top-left corner"),
    ("board_br", "Click BOARD bottom-right corner"),
    ("shop_tl", "Click SHOP row top-left corner"),
    ("shop_br", "Click SHOP row bottom-right corner"),
    ("elixir_tl", "Click ELIXIR region top-left corner"),
    ("elixir_br", "Click ELIXIR region bottom-right corner"),
]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class _ClickState:
    """Mutable click buffer shared with the OpenCV mouse callback."""

    points: List[Tuple[int, int]]  # in display-space pixels
    display_scale: float


def _mouse_callback(event: int, x: int, y: int, flags: int, state: _ClickState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(state.points) < len(CALIB_STEPS):
        state.points.append((x, y))


def _render_calibration(frame: np.ndarray, state: _ClickState) -> np.ndarray:
    disp = cv2.resize(frame, None, fx=state.display_scale, fy=state.display_scale)
    for i, pt in enumerate(state.points):
        cv2.circle(disp, pt, 6, (0, 255, 0), 2)
        cv2.putText(disp, CALIB_STEPS[i][0], (pt[0] + 8, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    step_idx = len(state.points)
    if step_idx < len(CALIB_STEPS):
        prompt = f"[{step_idx + 1}/{len(CALIB_STEPS)}] {CALIB_STEPS[step_idx][1]}"
    else:
        prompt = "All points set. Enter=accept  r=redo last  q=quit"
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(disp, prompt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return disp


def _collect_clicks(session: LiveSession) -> Optional[dict]:
    """Run the interactive click loop; returns frame-pixel points or None on quit."""
    first = session.frames.latest()
    h, w = first.shape[:2]
    scale = fit_to_screen(w, h)
    state = _ClickState(points=[], display_scale=scale)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, _mouse_callback, state)
    try:
        while True:
            frame = session.frames.latest()
            cv2.imshow(WINDOW, _render_calibration(frame, state))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("r") and state.points:
                state.points.pop()
            if key in (13, 10) and len(state.points) == len(CALIB_STEPS):  # Enter
                break
    finally:
        cv2.destroyWindow(WINDOW)

    # Map display clicks -> true frame pixels and tag by step name.
    frame_pts = {
        CALIB_STEPS[i][0]: display_to_frame(pt, scale)
        for i, pt in enumerate(state.points)
    }
    return frame_pts


def _derive_geometry(
    frame_pts: dict, mapper: CoordinateMapper, base: BoardGeometry
) -> Tuple[BoardGeometry, dict]:
    """Convert frame-pixel corner clicks into a logical-point BoardGeometry."""

    def to_logical(name: str) -> Tuple[float, float]:
        lx, ly = mapper.pixel_to_logical(frame_pts[name])
        return round(lx, 2), round(ly, 2)

    b_tl, b_br = to_logical("board_tl"), to_logical("board_br")
    s_tl, s_br = to_logical("shop_tl"), to_logical("shop_br")
    e_tl, e_br = to_logical("elixir_tl"), to_logical("elixir_br")

    rows, cols, slots = base.rows, base.cols, base.shop_slots
    geometry = BoardGeometry(
        rows=rows,
        cols=cols,
        origin=b_tl,
        cell_size=(round((b_br[0] - b_tl[0]) / cols, 2), round((b_br[1] - b_tl[1]) / rows, 2)),
        shop_slots=slots,
        shop_origin=s_tl,
        shop_slot_size=(round((s_br[0] - s_tl[0]) / slots, 2), round(s_br[1] - s_tl[1], 2)),
        elixir_region=(e_tl[0], e_tl[1], round(e_br[0] - e_tl[0], 2), round(e_br[1] - e_tl[1], 2)),
    )
    record = {
        "logical": {
            "board_tl": b_tl, "board_br": b_br,
            "shop_tl": s_tl, "shop_br": s_br,
            "elixir_tl": e_tl, "elixir_br": e_br,
        },
        "pixels": {k: list(v) for k, v in frame_pts.items()},
        "scale_factor": mapper.scale_factor,
        "logical_size": list(mapper.logical_size),
    }
    return geometry, record


def _geometry_snippet(geometry: BoardGeometry) -> str:
    """A paste-ready BoardGeometry(...) snippet for config.py."""
    return (
        "BoardGeometry(\n"
        f"    rows={geometry.rows},\n"
        f"    cols={geometry.cols},\n"
        f"    origin={geometry.origin},\n"
        f"    cell_size={geometry.cell_size},\n"
        f"    shop_slots={geometry.shop_slots},\n"
        f"    shop_origin={geometry.shop_origin},\n"
        f"    shop_slot_size={geometry.shop_slot_size},\n"
        f"    elixir_region={geometry.elixir_region},\n"
        ")"
    )


def _verify_taps(session: LiveSession, geometry: BoardGeometry) -> bool:
    """Constraint 1: physically tap the computed board corners for confirmation."""
    verify_mapper = CoordinateMapper(
        geometry=geometry,
        logical_size=session.mapper.logical_size,
        scale_factor=session.mapper.scale_factor,
    )
    tl = verify_mapper.cell_center_logical(0, 0)
    br = verify_mapper.cell_center_logical(geometry.rows - 1, geometry.cols - 1)

    print("\n[verify] Tapping computed TOP-LEFT board cell center:", tl)
    session.client.session.tap(*tl)
    time.sleep(1.0)
    print("[verify] Tapping computed BOTTOM-RIGHT board cell center:", br)
    session.client.session.tap(*br)

    answer = input("\nDid BOTH taps land on the corner board cells? [y/N]: ").strip().lower()
    return answer == "y"


def run_calibrate(config: BotConfig) -> None:
    ensure_dirs(os.path.dirname(CALIBRATION_PATH))
    with open_live_session(config) as session:
        while True:
            frame_pts = _collect_clicks(session)
            if frame_pts is None:
                print("Calibration cancelled.")
                return

            geometry, record = _derive_geometry(frame_pts, session.mapper, config.board)

            if not _verify_taps(session, geometry):
                print("Verification failed - re-running calibration. (q to quit)\n")
                continue

            snippet = _geometry_snippet(geometry)
            with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
                json.dump({"geometry": _geometry_to_dict(geometry), **record}, fh, indent=2)

            print("\n" + "=" * 70)
            print("Calibration verified. Paste this into BoardGeometry in config.py:")
            print("=" * 70)
            print(snippet)
            print("=" * 70)
            print(f"Also saved to: {CALIBRATION_PATH}\n")
            return


def _geometry_to_dict(geometry: BoardGeometry) -> dict:
    return {
        "rows": geometry.rows,
        "cols": geometry.cols,
        "origin": list(geometry.origin),
        "cell_size": list(geometry.cell_size),
        "shop_slots": geometry.shop_slots,
        "shop_origin": list(geometry.shop_origin),
        "shop_slot_size": list(geometry.shop_slot_size),
        "elixir_region": list(geometry.elixir_region),
    }


def _geometry_from_dict(data: dict) -> BoardGeometry:
    return BoardGeometry(
        rows=int(data["rows"]),
        cols=int(data["cols"]),
        origin=tuple(data["origin"]),
        cell_size=tuple(data["cell_size"]),
        shop_slots=int(data["shop_slots"]),
        shop_origin=tuple(data["shop_origin"]),
        shop_slot_size=tuple(data["shop_slot_size"]),
        elixir_region=tuple(data["elixir_region"]),
    )


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def _load_calibrated_mapper(config: BotConfig, session: LiveSession) -> CoordinateMapper:
    """Build a mapper from calibration.json, falling back to config.board."""
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        geometry = _geometry_from_dict(data["geometry"])
        logger.info("Loaded calibrated geometry from %s", CALIBRATION_PATH)
    else:
        geometry = config.board
        logger.warning("No calibration.json found; using placeholder config.board geometry.")
    return CoordinateMapper(
        geometry=geometry,
        logical_size=session.mapper.logical_size,
        scale_factor=session.mapper.scale_factor,
    )


def _board_region_bbox(mapper: CoordinateMapper) -> Tuple[int, int, int, int]:
    """Pixel bounding box spanning the whole board (for change detection)."""
    geo = mapper.geometry
    x0, y0, _, _ = mapper.cell_bbox_pixel(0, 0)
    bx, by, bw, bh = mapper.cell_bbox_pixel(geo.rows - 1, geo.cols - 1)
    return x0, y0, (bx + bw) - x0, (by + bh) - y0


def _crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y0:y1, x0:x1]


def _changed_fraction(prev: np.ndarray, curr: np.ndarray, pixel_thresh: int = 25) -> float:
    """Fraction of structurally changed pixels between two BGR regions."""
    if prev is None or curr is None or prev.shape != curr.shape:
        return 1.0
    g_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g_prev, g_curr)
    mask = (diff > pixel_thresh).astype(np.uint8)
    return float(mask.mean())


def _save_crops(frame: np.ndarray, mapper: CoordinateMapper, keep_empty_prob: float,
                occupancy_thresh: float) -> int:
    """Crop and save all board cells + shop slots; returns number of files saved."""
    ensure_dirs(RAW_CELLS_DIR, RAW_SHOP_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
    geo = mapper.geometry
    saved = 0

    def maybe_save(crop: np.ndarray, path: str) -> None:
        nonlocal saved
        if crop.size == 0:
            return
        occupied = float(np.std(crop)) > occupancy_thresh
        if not occupied and np.random.random() > keep_empty_prob:
            return  # throttle empty-cell flooding
        cv2.imwrite(path, crop)
        saved += 1

    for r in range(geo.rows):
        for c in range(geo.cols):
            idx = mapper.cell_to_index(r, c)
            maybe_save(_crop(frame, mapper.cell_bbox_pixel(r, c)),
                       os.path.join(RAW_CELLS_DIR, f"{ts}_cell{idx:02d}.png"))
    for s in range(geo.shop_slots):
        maybe_save(_crop(frame, mapper.shop_bbox_pixel(s)),
                   os.path.join(RAW_SHOP_DIR, f"{ts}_shop{s}.png"))
    return saved


def run_collect(config: BotConfig, interval: float, cooldown: float,
                change_frac: float, keep_empty_prob: float, occupancy_thresh: float) -> None:
    ensure_dirs(RAW_CELLS_DIR, RAW_SHOP_DIR)
    with open_live_session(config) as session:
        mapper = _load_calibrated_mapper(config, session)
        region = _board_region_bbox(mapper)

        last_region: Optional[np.ndarray] = None
        last_save_time = 0.0
        consecutive = 0
        total_saved = 0

        logger.info(
            "Collecting. interval=%.2fs cooldown=%.1fs change_frac=%.3f. Ctrl-C to stop.",
            interval, cooldown, change_frac,
        )
        try:
            while True:
                frame = session.frames.latest()
                region_crop = _crop(frame, region)
                frac = _changed_fraction(last_region, region_crop)

                structural = frac >= change_frac
                consecutive = consecutive + 1 if structural else 0
                cooled_down = (time.time() - last_save_time) >= cooldown

                if structural and consecutive >= 2 and cooled_down:
                    n = _save_crops(frame, mapper, keep_empty_prob, occupancy_thresh)
                    total_saved += n
                    last_save_time = time.time()
                    last_region = region_crop.copy()
                    consecutive = 0
                    logger.info("Saved %d crops (frac=%.3f, total=%d)", n, frac, total_saved)
                elif last_region is None:
                    last_region = region_crop.copy()

                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stopped. Total crops saved: %d", total_saved)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibration + data-collection harness")
    parser.add_argument("--wda-url", default=None, help="Override WDA URL (default from config)")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("calibrate", help="Interactively calibrate board/shop/elixir geometry")

    collect = sub.add_parser("collect", help="Dump change-detected crops for labeling")
    collect.add_argument("--interval", type=float, default=1.0, help="Seconds between frame checks")
    collect.add_argument("--cooldown", type=float, default=3.0, help="Min seconds between saves")
    collect.add_argument("--change-frac", type=float, default=0.04,
                         help="Min fraction of changed pixels to trigger a save")
    collect.add_argument("--keep-empty-prob", type=float, default=0.1,
                         help="Probability of keeping an empty-looking crop")
    collect.add_argument("--occupancy-thresh", type=float, default=12.0,
                         help="Pixel std-dev above which a crop is treated as occupied")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    config = BotConfig()
    if args.wda_url:
        config.device.wda_url = args.wda_url

    if args.mode == "calibrate":
        run_calibrate(config)
    elif args.mode == "collect":
        run_collect(
            config,
            interval=args.interval,
            cooldown=args.cooldown,
            change_frac=args.change_frac,
            keep_empty_prob=args.keep_empty_prob,
            occupancy_thresh=args.occupancy_thresh,
        )


if __name__ == "__main__":
    main()
