"""value/policy ネットワーク（Phase 3・torch）.

観測特徴量（features.encode_observation）と合法手特徴量（encode_actions）を入力に、
value（手番プレイヤーが勝つ確率 [0,1]）と policy（各合法手の logits）を出す MLP。
GTX1060 でも軽い構成（Transformer ではなく MLP）。torch 依存のため Docker で動かす。
"""

from __future__ import annotations

import torch
from torch import nn

from features import ACTION_FEAT_LEN, OBS_FEAT_LEN


class PVNet(nn.Module):
    """policy + value ネット（MLP）.

    forward は単一サンプル（MCTS の葉評価）向け: state (state_dim,), actions (n, action_dim)
    を受け、(value: scalar, logits: (n,)) を返す。バッチ学習側は別途パディング/マスクで扱う。
    """

    def __init__(
        self,
        state_dim: int = OBS_FEAT_LEN,
        action_dim: int = ACTION_FEAT_LEN,
        hidden: int = 256,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def trunk_forward(self, state: torch.Tensor) -> torch.Tensor:
        """状態 → 埋め込み（バッチ可: (..., state_dim) -> (..., hidden)）."""
        return self.trunk(state)

    def value(self, embedding: torch.Tensor) -> torch.Tensor:
        """埋め込み → value [0,1]."""
        return torch.sigmoid(self.value_head(embedding)).squeeze(-1)

    def policy_logits(
        self, embedding: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """埋め込み(hidden,) と 合法手(n, action_dim) → 各手の logits (n,)."""
        n = actions.shape[0]
        rep = embedding.unsqueeze(0).expand(n, -1)
        return self.policy_head(torch.cat([rep, actions], dim=-1)).squeeze(-1)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """単一サンプル: (state, actions) -> (value scalar, logits (n,))."""
        emb = self.trunk_forward(state)
        return self.value(emb), self.policy_logits(emb, actions)
