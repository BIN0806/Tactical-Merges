# ios_merge_bot

Autonomous deep reinforcement learning agent for an iOS synchronous auto-battler
(Clash Royale: Merge Tactics). The bot runs a closed perceive -> think -> act loop:

1. **Perceive** - capture the iOS screen and extract the 4x4 board, shop, and elixir
   using OpenCV template matching + a MobileNetV3 unit classifier.
2. **Think** - evaluate the state through a custom Gymnasium environment driven by a
   PyTorch Transformer policy.
3. **Act** - send human-like touch input back to the iPhone via WebDriverAgent (WDA).

Because the target is iOS, `adb`/`scrcpy` are **not** used. All device interaction
goes through WebDriverAgent (`facebook-wda`) and/or macOS native window capture.

## Architecture

```
ios_merge_bot/
├── main.py                      # Entry point orchestrating the loop
├── config.py                    # Central device/geometry/model configuration
├── io_layer/
│   ├── coordinates.py           # Retina scaling + grid <-> pixel mapping
│   ├── wda_client.py            # WebDriverAgent connection and session
│   ├── screen_capture.py        # Continuous frame ingestion (MJPEG or mss)
│   └── human_input.py           # Bezier-curve dragging + randomized tap delays
├── perception/
│   ├── state_extractor.py       # OpenCV grid mapping + template matching
│   └── mobilenet_classifier.py  # MobileNetV3 unit-ID / star-level classifier
├── environment/
│   ├── merge_tactics_env.py     # Gymnasium Env wrapping the live game
│   └── action_space.py          # Discretized moves, buys, merges
└── agent/
    ├── transformer_policy.py    # Multi-headed Transformer (policy + value)
    └── train_ppo.py             # Ray RLlib PPO config and training loop
```

## Prerequisites

- macOS with Xcode (to build and run WebDriverAgent on a physical iPhone).
- Python 3.11+.
- A jailed (non-jailbroken) iPhone running the game, connected via USB.

### 1. Build / run WebDriverAgent

Build WDA from [appium/WebDriverAgent](https://github.com/appium/WebDriverAgent) (or
via Appium) and run it on the device. Forward the port over USB:

```bash
iproxy 8100 8100      # from libimobiledevice
```

WDA should now be reachable at `http://localhost:8100`.

### 2. (Optional) QuickTime mirror for `mss` capture

If you prefer macOS native capture over the WDA MJPEG stream, mirror the iPhone in
QuickTime Player (File -> New Movie Recording -> select the device) and set
`CaptureConfig.backend = CaptureBackend.MSS_WINDOW` plus `window_bbox` in `config.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline smoke test

From the directory that contains the `ios_merge_bot` package:

```bash
python -m ios_merge_bot.main
```

This connects to WDA, starts capture + perception, and runs a 10-step loop taking
random actions to prove the full I/O pipeline functions end to end.

## Injecting your own assets

- **Template images**: point `ModelConfig.template_dir` at a folder of PNGs used by
  the OpenCV occupancy / shop matcher.
- **Classifier weights**: set `ModelConfig.classifier_weights` to a `.pt` checkpoint
  for the MobileNetV3 unit classifier.
- **Board calibration**: adjust `BoardGeometry` (origin, cell size, shop, elixir
  region) in `config.py` to match your device's layout in logical points.

All ML/vision components degrade gracefully (stub outputs) when assets are absent, so
the pipeline smoke test runs without any trained weights.

## Training

```bash
python -m ios_merge_bot.agent.train_ppo
```

Configures Ray RLlib PPO against `MergeTacticsEnv` with the custom Transformer model.
Note: live on-device training is slow; consider a simulator/offline dataset first.
