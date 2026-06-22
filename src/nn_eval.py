"""NN 評価器（features → net → (value, priors)）. torch 依存.

学習済み（または初期化済み）の PVNet を、nn_mcts が要求する評価器
`evaluator(obs) -> (value, priors)` に変換する。これを make_nn_mcts_agent(evaluator=...) に
渡すと NN 誘導 MCTS（AlphaZero 型）になる。
"""

from __future__ import annotations

import torch

from cards import CardMeta
from cg.api import Observation
from features import encode_actions, encode_observation
from nn_mcts import Evaluator
from net import PVNet


def make_net_evaluator(net: PVNet, meta: CardMeta, device: str = "cpu") -> Evaluator:
    """PVNet を nn_mcts 用の評価器に変換する."""
    dev = torch.device(device)
    net.to(dev)
    net.eval()

    def evaluator(obs: Observation) -> tuple[float, list[float]]:
        sel = obs.select
        n = len(sel.option) if sel else 0
        if sel is None or obs.current is None or n == 0:
            return 0.5, ([1.0 / n] * n if n > 0 else [])
        state = torch.from_numpy(encode_observation(obs, meta)).to(dev)
        actions = torch.from_numpy(encode_actions(obs, meta)).to(dev)
        with torch.no_grad():
            value, logits = net(state, actions)
            priors = torch.softmax(logits, dim=-1).cpu().tolist()
            return float(value.item()), priors

    return evaluator
