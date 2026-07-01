"""value/policy ネットワーク（Phase 3・torch）.

観測特徴量（features.encode_observation）と合法手特徴量（encode_actions）を入力に、
value（手番プレイヤーが勝つ確率 [0,1]）と policy（各合法手の logits）を出す MLP。
GTX1060 でも軽い構成（Transformer ではなく MLP）。torch 依存のため Docker で動かす。

**容量の方針（2026-07-01 right-size）**: cardId Embedding(1300種) は蒸留データ(数デッキ・
~1万サンプル)に対して過大で、大半のカードの埋め込みが学習されず policy が一様に張り付いた
（診断: 教師再現度0.10≈偶然・max prior 0.16≈uniform）。旧・小net(hidden256/2層・埋め込み無し)は
同予算で ISMCTS 五分に到達していた。よって既定を **hidden256・trunk2層・埋め込み無し(card_emb=0)**
に戻す。汎化する効果カテゴリ/KO/弱点の float 特徴は features 側に残す（低次元で学習しやすい）。

特徴ベクトル末尾には整数 cardId が載る（features の N_STATE_ID / N_ACTION_ID）。card_emb>0 なら
それを Embedding して連結し、card_emb=0 なら **id 列は捨てる**（生の id を float として入れない）。
"""

from __future__ import annotations

import torch
from torch import nn

from features import ACTION_FEAT_LEN, N_ACTION_ID, N_STATE_ID, OBS_FEAT_LEN

# cardId 0(なし/pad)..1267。安全のため少し余裕を持たせる。
N_CARDS = 1300


class PVNet(nn.Module):
    """policy + value ネット（MLP）.

    forward は単一サンプル（MCTS の葉評価）向け: state (state_dim,), actions (n, action_dim)
    を受け、(value: scalar, logits: (n,)) を返す。state/actions の末尾 N_STATE_ID / N_ACTION_ID
    列は整数 cardId。card_emb>0 なら Embedding して連結、card_emb=0 なら捨てる。
    """

    def __init__(
        self,
        state_dim: int = OBS_FEAT_LEN,
        action_dim: int = ACTION_FEAT_LEN,
        hidden: int = 256,
        card_emb: int = 0,
        n_cards: int = N_CARDS,
    ):
        super().__init__()
        self.n_state_id = N_STATE_ID
        self.n_action_id = N_ACTION_ID
        self.state_float_dim = state_dim - N_STATE_ID
        self.action_float_dim = action_dim - N_ACTION_ID
        self._n_cards = n_cards
        self._card_emb = card_emb
        self._use_emb = card_emb > 0
        # card_emb=0 の時は埋め込みを作らず末尾 id 列を捨てる（データ量に見合わない過大容量を回避）
        if self._use_emb:
            # state/action 共有のカード埋め込み（0=pad は常に 0 ベクトル）
            self.card_emb = nn.Embedding(n_cards, card_emb, padding_idx=0)

        trunk_in = self.state_float_dim + (self.n_state_id * card_emb)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        pol_in = hidden + self.action_float_dim + (self.n_action_id * card_emb)
        self.policy_head = nn.Sequential(
            nn.Linear(pol_in, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """末尾 id 列(..., k) を Embedding し (..., k*card_emb) に平坦化する."""
        idx = ids.long().clamp(0, self._n_cards - 1)
        return self.card_emb(idx).flatten(-2)

    def trunk_forward(self, state: torch.Tensor) -> torch.Tensor:
        """状態 → 埋め込み（末尾 N_STATE_ID 列は埋め込み or 破棄）."""
        floats = state[..., : self.state_float_dim]
        if self._use_emb:
            emb = self._embed_ids(state[..., self.state_float_dim :])
            return self.trunk(torch.cat([floats, emb], dim=-1))
        return self.trunk(floats)

    def value(self, embedding: torch.Tensor) -> torch.Tensor:
        """埋め込み → value [0,1]."""
        return torch.sigmoid(self.value_head(embedding)).squeeze(-1)

    def policy_logits(
        self, embedding: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """埋め込み(hidden,) と 合法手(n, action_dim) → 各手の logits (n,).

        actions の末尾 N_ACTION_ID 列は card_emb>0 なら Embedding して連結、0 なら破棄する。
        """
        n = actions.shape[0]
        a_floats = actions[..., : self.action_float_dim]
        rep = embedding.unsqueeze(0).expand(n, -1)
        if self._use_emb:
            a_emb = self._embed_ids(actions[..., self.action_float_dim :])
            cat = torch.cat([rep, a_floats, a_emb], dim=-1)
        else:
            cat = torch.cat([rep, a_floats], dim=-1)
        return self.policy_head(cat).squeeze(-1)

    def forward(
        self, state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """単一サンプル: (state, actions) -> (value scalar, logits (n,))."""
        emb = self.trunk_forward(state)
        return self.value(emb), self.policy_logits(emb, actions)
