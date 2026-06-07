"""Transformer policy network for Merge Tactics.

The board is treated as a sequence of tokens:

* 16 board-cell tokens (4x4 grid, row-major), and
* 3 shop tokens,

for a total of 19 tokens. Each token's unit ID is embedded, and learned
positional + token-type embeddings encode *where* (which cell / which shop slot)
the token lives. A ``nn.TransformerEncoderLayer`` computes spatial attention so
the network reasons about unit adjacency and merge opportunities. A pooled
representation feeds two heads:

* ``policy_head`` -> action logits, and
* ``value_head``  -> a scalar state value (for actor-critic / PPO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    num_unit_classes: int = 32
    max_star_level: int = 3
    board_rows: int = 4
    board_cols: int = 4
    shop_slots: int = 3
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    action_dim: int = 261

    @property
    def num_board_tokens(self) -> int:
        return self.board_rows * self.board_cols

    @property
    def num_tokens(self) -> int:
        return self.num_board_tokens + self.shop_slots


class MergeTacticsTransformer(nn.Module):
    """Multi-headed Transformer mapping a board+shop observation to (logits, value)."""

    TOKEN_TYPE_BOARD = 0
    TOKEN_TYPE_SHOP = 1

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        n_tokens = config.num_tokens

        # Embeddings: unit ID, star level, positional, and token type.
        self.unit_embed = nn.Embedding(config.num_unit_classes, d)
        self.star_embed = nn.Embedding(config.max_star_level + 1, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d))
        self.type_embed = nn.Embedding(2, d)

        # Scalar elixir projected into the model dimension and prepended as a token.
        self.elixir_proj = nn.Linear(1, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.action_dim)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # ------------------------------------------------------------------ #
    def _build_token_types(self, batch: int, device: torch.device) -> torch.Tensor:
        board_n = self.config.num_board_tokens
        shop_n = self.config.shop_slots
        types = torch.cat(
            [
                torch.full((board_n,), self.TOKEN_TYPE_BOARD, dtype=torch.long),
                torch.full((shop_n,), self.TOKEN_TYPE_SHOP, dtype=torch.long),
            ]
        ).to(device)
        return types.unsqueeze(0).expand(batch, -1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action logits and state value from a batched observation dict.

        Expected keys (each batched along dim 0):
            board_units: long  [B, rows, cols]
            board_stars: long  [B, rows, cols]
            shop:        long  [B, shop_slots]
            elixir:      float [B, 1]
        """
        board_units = obs["board_units"].long()
        board_stars = obs["board_stars"].long()
        shop = obs["shop"].long()
        elixir = obs["elixir"].float()

        batch = board_units.shape[0]
        device = board_units.device

        board_units_flat = board_units.reshape(batch, -1)  # [B, 16]
        board_stars_flat = board_stars.reshape(batch, -1)  # [B, 16]

        # Unit token embeddings (board cells carry star info; shop slots do not).
        board_tok = self.unit_embed(board_units_flat) + self.star_embed(board_stars_flat)
        shop_tok = self.unit_embed(shop)  # [B, shop_slots, d]
        tokens = torch.cat([board_tok, shop_tok], dim=1)  # [B, n_tokens, d]

        tokens = tokens + self.pos_embed
        tokens = tokens + self.type_embed(self._build_token_types(batch, device))

        # Prepend an elixir-conditioned CLS token used for pooling.
        cls = self.cls_token.expand(batch, -1, -1) + self.elixir_proj(elixir).unsqueeze(1)
        seq = torch.cat([cls, tokens], dim=1)  # [B, 1 + n_tokens, d]

        encoded = self.encoder(seq)
        pooled = encoded[:, 0]  # CLS representation

        logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) an action; returns (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value


def build_default_model(
    num_unit_classes: int,
    action_dim: int,
    max_star_level: int = 3,
    board_rows: int = 4,
    board_cols: int = 4,
    shop_slots: int = 3,
) -> MergeTacticsTransformer:
    """Convenience constructor wiring config from environment dimensions."""
    cfg = TransformerConfig(
        num_unit_classes=num_unit_classes,
        max_star_level=max_star_level,
        board_rows=board_rows,
        board_cols=board_cols,
        shop_slots=shop_slots,
        action_dim=action_dim,
    )
    return MergeTacticsTransformer(cfg)
