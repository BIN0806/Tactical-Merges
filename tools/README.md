# tools: calibration + data-collection harness

Step 1 of grounding the bot in reality: measure the real on-device geometry and
build a labeled image dataset for the MobileNet unit classifier. All scripts are
run as modules from the **repository root** so they can import `ios_merge_bot`.

## Prerequisites

- WebDriverAgent running and reachable (default `http://localhost:8100`; see the
  top-level `ios_merge_bot/README.md`).
- Dependencies installed: `pip install -r ios_merge_bot/requirements.txt`.
- A display available for the OpenCV windows (this is an interactive, on-Mac tool).

## Workflow

```
calibrate  ->  paste snippet into config.py  ->  collect  ->  label
```

### 1. Calibrate the geometry

```bash
python -m tools.data_collector calibrate
```

- A live frame opens. Click the six prompted corners in order: board top-left,
  board bottom-right, shop top-left, shop bottom-right, elixir top-left, elixir
  bottom-right. Use `r` to redo the last click, `Enter` to accept, `q` to quit.
- The tool converts your clicks (physical Retina pixels) to logical points via
  the existing `CoordinateMapper`, then derives a `BoardGeometry`.
- **Verification taps:** before printing anything it physically taps the computed
  top-left and bottom-right board cells on your phone and asks `y/N`. Answer `y`
  only if both taps landed correctly; `n` restarts calibration. This proves the
  Retina math is right before you commit it.
- On success it prints a paste-ready `BoardGeometry(...)` snippet and writes
  `dataset/calibration.json`.

Paste the printed snippet into `BoardGeometry(...)` in
[`ios_merge_bot/config.py`](../ios_merge_bot/config.py).

### 2. Collect raw crops

```bash
python -m tools.data_collector collect --interval 1.0 --cooldown 3.0 --change-frac 0.04
```

Watches the live stream and, when a **structural** change is detected, dumps
crops of the 16 board cells and 3 shop slots into `dataset/raw_cells/` and
`dataset/raw_shop/`. Two anti-flooding guards (so idle troop/aura animations
don't spam the dataset):

- `--cooldown` - hard minimum seconds between saves (default `3.0`).
- `--change-frac` - minimum fraction of structurally changed pixels (binarized
  diff over the board region) required to trigger a save (default `0.04`), held
  for two consecutive reads.

Other flags:

- `--interval` - seconds between frame checks (default `1.0`).
- `--keep-empty-prob` - probability of keeping an empty-looking crop (default
  `0.1`) so the `EMPTY` class isn't over-represented.
- `--occupancy-thresh` - pixel std-dev above which a crop counts as occupied.

Stop with `Ctrl-C`.

### 3. Label the crops

```bash
python -m tools.label_helper
```

Two-stroke hotkey labeling: press a **unit letter** then a **star digit** to file
the image into `dataset/labeled/<unit>_<star>/` and auto-advance.

- Example: `p` then `2` -> `dataset/labeled/pekka_2/`.
- `space` - trash/skip an empty cell (-> `dataset/labeled/_empty/`).
- `u` - undo the last move. `Esc` - cancel a half-entered code. `q` - quit.

Unit hotkeys live in the `UNIT_HOTKEYS` dict at the top of
[`label_helper.py`](label_helper.py); add new units there in one place.

The resulting `dataset/labeled/<unit>_<star>/` tree is the training set for the
`UnitClassifier` (`ios_merge_bot/perception/mobilenet_classifier.py`).

## Output layout

```
dataset/
├── calibration.json     # written by `calibrate`
├── raw_cells/           # raw board-cell crops (awaiting labeling)
├── raw_shop/            # raw shop-slot crops (awaiting labeling)
└── labeled/
    ├── pekka_2/
    ├── archer_1/
    ├── _empty/
    └── ...
```
