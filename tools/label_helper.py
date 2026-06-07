"""Ergonomic two-stroke labeling pass for raw crops.

Walks every crop in ``dataset/raw_cells`` and ``dataset/raw_shop`` and shows it
one at a time. Labeling is a two-keystroke gesture (Constraint 3):

* **Stroke 1** - a letter mapped to a unit (see ``UNIT_HOTKEYS`` below).
* **Stroke 2** - a digit ``1``-``4`` for the star level.

Typing ``p`` then ``2`` moves the file into ``dataset/labeled/pekka_2/`` and
auto-advances. Other keys:

* ``space`` - trash/skip: move an empty cell into ``dataset/labeled/_empty/``.
* ``u``     - undo the last move.
* ``Esc``   - cancel a half-entered code.
* ``q``     - quit.

Add new units in one place by editing ``UNIT_HOTKEYS``.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2

from .common import LABELED_DIR, RAW_CELLS_DIR, RAW_SHOP_DIR, ensure_dirs

logger = logging.getLogger(__name__)

WINDOW = "Merge Tactics - label_helper"

# Single-letter hotkey -> unit name. Extend this as you discover units.
UNIT_HOTKEYS: Dict[str, str] = {
    "a": "archer",
    "k": "knight",
    "p": "pekka",
    "g": "goblin",
    "b": "barbarian",
    "w": "wizard",
    "m": "musketeer",
    "v": "valkyrie",
    "d": "dart_goblin",
    "r": "archer_queen",
}

EMPTY_LABEL = "_empty"
MAX_STAR = 4
DISPLAY_SIZE = 320  # upscale crops so small cells are visible


def _gather_images() -> List[str]:
    paths: List[str] = []
    for root in (RAW_CELLS_DIR, RAW_SHOP_DIR):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(glob.glob(os.path.join(root, ext)))
    return sorted(paths)


def _legend_lines() -> List[str]:
    items = [f"{k}={v}" for k, v in UNIT_HOTKEYS.items()]
    # Pack a few mappings per line for the overlay.
    lines, chunk = [], 3
    for i in range(0, len(items), chunk):
        lines.append("  ".join(items[i:i + chunk]))
    return lines


def _render(img, pending_unit: Optional[str], idx: int, total: int):
    canvas = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    header = f"{idx + 1}/{total}"
    if pending_unit:
        header += f"  pending: {pending_unit}_?  (press 1-{MAX_STAR})"
    else:
        header += "  press unit letter, then star.  space=empty u=undo q=quit"
    framed = cv2.copyMakeBorder(canvas, 28, 92, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(framed, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y = framed.shape[0] - 76
    for line in _legend_lines():
        cv2.putText(framed, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)
        y += 18
    return framed


def _move(src: str, label: str) -> Tuple[str, str]:
    """Move ``src`` into ``dataset/labeled/<label>/``; return (src, dst) for undo."""
    dst_dir = os.path.join(LABELED_DIR, label)
    ensure_dirs(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.move(src, dst)
    return src, dst


def run_labeler() -> None:
    ensure_dirs(LABELED_DIR)
    images = _gather_images()
    if not images:
        print(f"No images found in {RAW_CELLS_DIR} or {RAW_SHOP_DIR}.")
        return

    cv2.namedWindow(WINDOW)
    idx = 0
    pending_unit: Optional[str] = None
    history: List[Tuple[str, str]] = []  # (original_src, dst) for undo

    print(f"Labeling {len(images)} images. q to quit.")
    while 0 <= idx < len(images):
        path = images[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            idx += 1
            continue

        cv2.imshow(WINDOW, _render(img, pending_unit, idx, len(images)))
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == 27:  # Esc cancels a pending unit
            pending_unit = None
            continue
        if key == ord("u") and history:
            src, dst = history.pop()
            ensure_dirs(os.path.dirname(src))
            shutil.move(dst, src)
            idx = max(0, idx - 1)
            pending_unit = None
            logger.info("Undo -> %s", os.path.basename(src))
            continue
        if key == ord(" "):  # spacebar: trash/skip empty
            history.append(_move(path, EMPTY_LABEL))
            pending_unit = None
            idx += 1
            continue

        ch = chr(key) if 32 <= key < 127 else ""
        if pending_unit is None:
            if ch in UNIT_HOTKEYS:
                pending_unit = UNIT_HOTKEYS[ch]
            # ignore other keys until a valid unit letter is pressed
            continue

        # awaiting star digit
        if ch.isdigit() and 1 <= int(ch) <= MAX_STAR:
            label = f"{pending_unit}_{ch}"
            history.append(_move(path, label))
            logger.info("%s -> %s", os.path.basename(path), label)
            pending_unit = None
            idx += 1
        elif ch in UNIT_HOTKEYS:  # corrected unit choice
            pending_unit = UNIT_HOTKEYS[ch]
        else:
            pending_unit = None  # invalid; reset

    cv2.destroyAllWindows()
    print("Labeling session ended.")


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description="Two-stroke crop labeler").parse_args(argv)
    run_labeler()


if __name__ == "__main__":
    main()
