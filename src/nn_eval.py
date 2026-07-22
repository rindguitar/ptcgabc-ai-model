"""NN 評価器（features → net → (value, priors)）. torch 依存.

学習済み（または初期化済み）の PVNet を、nn_mcts が要求する評価器
`evaluator(obs) -> (value, priors)` に変換する。これを make_nn_mcts_agent(evaluator=...) に
渡すと NN 誘導 MCTS（AlphaZero 型）になる。
"""

from __future__ import annotations

import torch

from agents import ko_threat
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


def wrap_board_bonus(evaluator: Evaluator, alpha: float) -> Evaluator:
    """value に盤面資源の補正を注入するラッパー（board-blind の即効処置）.

    v' = clamp01(v + α × (自分の場の駒数 − 相手の場の駒数) / 5)。
    診断で net の value は盤面資源（ベンチ切れ＝実戦敗因の6〜7割）に盲目（感度≈0）と判明。
    注入テスト（40試合×4α）で **α=0.2 が最良（平均+0.075・vs ismcts は α に単調増加）**＝
    「盤面を読めば手が変わり勝てる」を実証。v2.3（盤面特徴＋学習）が完成したら置き換えて撤去する。
    """

    def ev(obs: Observation) -> tuple[float, list[float]]:
        v, priors = evaluator(obs)
        st = obs.current
        if st is not None:
            yi = st.yourIndex

            def board(p) -> int:
                return len([a for a in (p.active or []) if a]) + len(p.bench or [])

            diff = board(st.players[yi]) - board(st.players[1 - yi])
            v = min(1.0, max(0.0, v + alpha * diff / 5.0))
        return v, priors

    return ev


def wrap_threat_bonus(evaluator: Evaluator, meta: CardMeta, alpha: float) -> Evaluator:
    """value に KO 脅威の対称差を注入するラッパー（§48・受けと詰めの事前信号）.

    v' = clamp01(v − α × (相手→自分の脅威 − 自分→相手の脅威))。脅威は agents.ko_threat
    （有効打点が残 HP に届くワザを、エネ充足度 1.0/0.5/0・ベンチ×0.5 で重み付けした最大値）。
    「取られる前に受ける（逃げる/進化で耐える）」と「先に取る/育てて詰める」の両方向が
    value に入り、探索が受けの手を自然に選べるようになる。wrap_board_bonus と同じ
    注入様式＝α は eval-net のスイープで実測決定（0 で無効）。
    """

    def ev(obs: Observation) -> tuple[float, list[float]]:
        v, priors = evaluator(obs)
        st = obs.current
        if st is not None:
            yi = st.yourIndex
            me, opp = st.players[yi], st.players[1 - yi]
            diff = ko_threat(opp, me, meta) - ko_threat(me, opp, meta)
            v = min(1.0, max(0.0, v - alpha * diff))
        return v, priors

    return ev
