"""Entry point orchestrating the perceive -> think -> act loop.

Initializes the WebDriverAgent connection, the screen-capture frame source, the
perception models, and the Gymnasium environment, then runs a dummy 10-step loop
taking random actions to prove the full I/O pipeline functions end to end.

Run with::

    python -m ios_merge_bot.main
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .config import DEFAULT_CONFIG, BotConfig
from .environment.merge_tactics_env import MergeTacticsEnv
from .io_layer.human_input import HumanizedTouch
from .io_layer.screen_capture import build_frame_source
from .io_layer.wda_client import WDAClient
from .perception.mobilenet_classifier import UnitClassifier
from .perception.state_extractor import StateExtractor

logger = logging.getLogger(__name__)


def build_env(config: BotConfig) -> tuple[MergeTacticsEnv, WDAClient]:
    """Wire the full live stack and return the env + the WDA client (for cleanup)."""
    logger.info("Connecting to WebDriverAgent...")
    client = WDAClient(config).connect()

    logger.info("Starting screen capture (%s)...", config.capture.backend.value)
    frame_source = build_frame_source(config, client.mjpeg_url).start()

    logger.info("Loading perception models...")
    classifier = UnitClassifier(config.model)
    extractor = StateExtractor(client.mapper, classifier, config.model)

    touch = HumanizedTouch(client.session, config.timing)

    env = MergeTacticsEnv(
        config=config,
        mapper=client.mapper,
        frame_source=frame_source,
        touch=touch,
        state_extractor=extractor,
    )
    return env, client


def run_pipeline_smoke_test(config: BotConfig = DEFAULT_CONFIG, steps: int = 10) -> None:
    """Run a dummy ``steps``-action loop to validate the I/O pipeline."""
    env: Optional[MergeTacticsEnv] = None
    client: Optional[WDAClient] = None
    try:
        env, client = build_env(config)

        logger.info("Resetting environment / reading initial state...")
        obs, info = env.reset()
        logger.info(
            "Initial obs shapes: board_units=%s shop=%s elixir=%s",
            obs["board_units"].shape, obs["shop"].shape, obs["elixir"].shape,
        )

        for step in range(1, steps + 1):
            action = env.action_space.sample()
            tic = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            dt = time.time() - tic
            intent = info.get("intent")
            logger.info(
                "step %02d/%d  action=%d  kind=%s  reward=%.2f  latency=%.3fs  term=%s trunc=%s",
                step, steps, int(action),
                getattr(getattr(intent, "kind", None), "name", "?"),
                reward, dt, terminated, truncated,
            )
            if terminated or truncated:
                logger.info("Episode ended; resetting.")
                obs, info = env.reset()

        logger.info("Pipeline smoke test complete: %d steps executed.", steps)
    finally:
        if env is not None:
            env.close()
        if client is not None:
            client.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline_smoke_test(DEFAULT_CONFIG, steps=10)


if __name__ == "__main__":
    main()
