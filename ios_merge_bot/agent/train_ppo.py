"""Ray RLlib PPO training for Merge Tactics.

Wraps :class:`MergeTacticsTransformer` in an RLlib ``TorchModelV2`` and registers
:class:`MergeTacticsEnv` so PPO can be trained against the live (or simulated)
game. Live on-device rollouts are slow; for serious training, plug in an offline
dataset or a fast simulator behind the same observation/action interface.

Run directly::

    python -m ios_merge_bot.agent.train_ppo
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from ..config import DEFAULT_CONFIG, BotConfig
from .transformer_policy import TransformerConfig

logger = logging.getLogger(__name__)

MODEL_NAME = "merge_tactics_transformer"
ENV_NAME = "MergeTactics-v0"


def _build_rllib_model_cls():
    """Construct the RLlib TorchModelV2 subclass (deferred so torch/ray stay optional)."""
    import torch
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

    from .transformer_policy import MergeTacticsTransformer

    class MergeTacticsRLModel(TorchModelV2, torch.nn.Module):
        """Adapts the Transformer to RLlib's (logits, state) + value_function API."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            torch.nn.Module.__init__(self)

            custom = model_config.get("custom_model_config", {})
            tcfg: TransformerConfig = custom["transformer_config"]
            tcfg.action_dim = int(num_outputs)
            self.net = MergeTacticsTransformer(tcfg)
            self._value_out = None

        def forward(
            self,
            input_dict: Dict[str, Any],
            state: list,
            seq_lens: Any,
        ) -> Tuple[Any, list]:
            obs = input_dict["obs"]  # RLlib restores the Dict structure for us.
            torch_obs = {
                "board_units": obs["board_units"].long(),
                "board_stars": obs["board_stars"].long(),
                "shop": obs["shop"].long(),
                "elixir": obs["elixir"].float(),
            }
            logits, value = self.net(torch_obs)
            self._value_out = value
            return logits, state

        def value_function(self):
            return self._value_out.reshape(-1)

    return MergeTacticsRLModel


def _make_env_creator(config: BotConfig):
    """Return an RLlib env creator. Builds the full live I/O stack per worker.

    Note: this instantiates real WDA/capture connections. For headless training,
    swap this for a simulator that exposes the same obs/action spaces.
    """

    def _creator(env_config: Dict[str, Any]):
        from ..io_layer.human_input import HumanizedTouch
        from ..io_layer.screen_capture import build_frame_source
        from ..io_layer.wda_client import WDAClient
        from ..environment.merge_tactics_env import MergeTacticsEnv

        client = WDAClient(config).connect()
        frame_source = build_frame_source(config, client.mjpeg_url).start()
        touch = HumanizedTouch(client.session, config.timing)
        return MergeTacticsEnv(config, client.mapper, frame_source, touch)

    return _creator


def build_ppo_config(config: BotConfig):
    """Assemble a Ray RLlib ``PPOConfig`` for the Merge Tactics env + model."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    ModelCatalog.register_custom_model(MODEL_NAME, _build_rllib_model_cls())
    register_env(ENV_NAME, _make_env_creator(config))

    geometry = config.board
    transformer_config = TransformerConfig(
        num_unit_classes=config.model.num_unit_classes,
        max_star_level=config.model.max_star_level,
        board_rows=geometry.rows,
        board_cols=geometry.cols,
        shop_slots=geometry.shop_slots,
    )

    return (
        PPOConfig()
        .environment(ENV_NAME)
        .framework("torch")
        # Live on-device env is single-instance; keep one rollout worker.
        .env_runners(num_env_runners=0)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=512,
            model={
                "custom_model": MODEL_NAME,
                "custom_model_config": {"transformer_config": transformer_config},
            },
        )
    )


def train(config: BotConfig = DEFAULT_CONFIG, iterations: int = 10) -> None:
    """Run a minimal PPO training loop."""
    import ray

    ray.init(ignore_reinit_error=True)
    try:
        algo = build_ppo_config(config).build()
        for i in range(iterations):
            result = algo.train()
            reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
            logger.info("iter %d/%d  episode_return_mean=%s", i + 1, iterations, reward)
        algo.save()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train()
