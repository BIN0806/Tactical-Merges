"""Gymnasium environment wrapping the live iOS game.

``MergeTacticsEnv`` closes the loop:

    action (Discrete) --decode--> HumanizedTouch gesture --wait--> capture frame
    --perception--> GameState --encode--> observation (Dict)

It is a *real-time, on-device* environment: ``step`` physically taps/drags the
phone, sleeps for game latency, grabs the next frame, and re-extracts state.
Reward shaping is intentionally a clearly-marked placeholder.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import BotConfig
from ..io_layer.coordinates import CoordinateMapper
from ..io_layer.human_input import HumanizedTouch
from ..io_layer.screen_capture import FrameSource
from ..perception.mobilenet_classifier import UnitClassifier
from ..perception.state_extractor import GameState, StateExtractor
from .action_space import ActionIntent, ActionKind, ActionSpace

logger = logging.getLogger(__name__)


class MergeTacticsEnv(gym.Env):
    """Real-time Gymnasium env driving the iOS auto-battler through WDA."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: BotConfig,
        mapper: CoordinateMapper,
        frame_source: FrameSource,
        touch: HumanizedTouch,
        state_extractor: Optional[StateExtractor] = None,
        max_steps: int = 200,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapper = mapper
        self.frame_source = frame_source
        self.touch = touch
        self.max_steps = max_steps
        self._step_count = 0
        self._last_frame: Optional[np.ndarray] = None

        geometry = config.board
        self.actions = ActionSpace(geometry, mapper)

        if state_extractor is None:
            classifier = UnitClassifier(config.model)
            state_extractor = StateExtractor(mapper, classifier, config.model)
        self.state_extractor = state_extractor

        # ---- spaces -------------------------------------------------- #
        self.rows, self.cols = geometry.rows, geometry.cols
        self.shop_slots = geometry.shop_slots
        n_units = config.model.num_unit_classes
        max_star = config.model.max_star_level

        self.observation_space = spaces.Dict(
            {
                "board_units": spaces.Box(0, n_units - 1, shape=(self.rows, self.cols), dtype=np.int32),
                "board_stars": spaces.Box(0, max_star, shape=(self.rows, self.cols), dtype=np.int32),
                "shop": spaces.Box(0, n_units - 1, shape=(self.shop_slots,), dtype=np.int32),
                "elixir": spaces.Box(0, 100, shape=(1,), dtype=np.int32),
            }
        )
        self.action_space = self.actions.to_gym_space()

    # ------------------------------------------------------------------ #
    # observation assembly
    # ------------------------------------------------------------------ #
    def _state_to_obs(self, state: GameState) -> Dict[str, np.ndarray]:
        return {
            "board_units": state.board_units.astype(np.int32),
            "board_stars": state.board_stars.astype(np.int32),
            "shop": state.shop_units.astype(np.int32),
            "elixir": np.array([state.elixir], dtype=np.int32),
        }

    def _read_state(self) -> Tuple[GameState, Dict[str, np.ndarray]]:
        frame = self.frame_source.latest()
        self._last_frame = frame
        state = self.state_extractor.extract(frame)
        return state, self._state_to_obs(state)

    # ------------------------------------------------------------------ #
    # gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        state, obs = self._read_state()
        info = {"game_state": state}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        intent = self.actions.decode(int(action))
        self._execute(intent)

        # Wait for the game to react before observing the result.
        time.sleep(self.config.timing.game_latency)

        state, obs = self._read_state()
        reward = self._compute_reward(state, intent)
        terminated = self._is_terminal(state)
        truncated = self._step_count >= self.max_steps
        info = {"game_state": state, "intent": intent}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # action execution
    # ------------------------------------------------------------------ #
    def _execute(self, intent: ActionIntent) -> None:
        """Translate a decoded intent into a physical HumanizedTouch command."""
        if intent.kind == ActionKind.NOOP:
            return

        if intent.kind == ActionKind.REROLL:
            if intent.start_point is not None:
                self.touch.human_tap(intent.start_point)
            return

        if intent.kind == ActionKind.BUY:
            if intent.start_point is not None and intent.end_point is not None:
                self.touch.human_drag(
                    start_coord=intent.start_point, end_coord=intent.end_point
                )
            return

        if intent.kind == ActionKind.MOVE:
            if intent.start_point is None or intent.end_point is None:
                return
            if intent.start_point == intent.end_point:
                self.touch.human_tap(intent.start_point)  # select / tap-in-place
            else:
                self.touch.human_drag(
                    start_coord=intent.start_point, end_coord=intent.end_point
                )

    # ------------------------------------------------------------------ #
    # reward / termination (placeholders)
    # ------------------------------------------------------------------ #
    def _compute_reward(self, state: GameState, intent: ActionIntent) -> float:
        """Placeholder reward.

        Replace with a signal derived from win/loss, HP deltas, board value, or
        round outcome detected by the perception layer.
        """
        return 0.0

    def _is_terminal(self, state: GameState) -> bool:
        """Placeholder terminal check.

        Replace with detection of the victory/defeat screen via template match.
        """
        return False

    def render(self) -> Optional[np.ndarray]:
        return self._last_frame

    def close(self) -> None:
        try:
            self.frame_source.stop()
        except Exception as err:  # noqa: BLE001
            logger.debug("Error stopping frame source: %s", err)
