"""NN 誘導 MCTS（PUCT・AlphaZero 型）の骨格.

ISMCTS と同様に determinize ＋ cabt search API でフォワードシミュレーションするが、
葉を rollout ではなく**評価器** `evaluator(obs) -> (value, priors)` で評価し、行動選択を
**PUCT**（事前確率 priors で誘導）にする。これにより弱い rollout に頼らず、少ない探索で
良手へ絞れる（rollout washout の回避）。

evaluator は torch ネットでも、テスト用のダミー（サイド差 value＋一様 priors）でもよい
（torch 非依存でホストでテスト可能）。value は「手番プレイヤーが勝つ確率」[0,1]、
priors は obs.select.option に整列した確率リスト。

v1 の割り切りは ISMCTS と同様: 木の分岐は MAIN/ATTACK の pick-one に集中し、その他の選択は
fallback 方策（既定ヒューリスティック）で1手に畳む。決定化の相手デッキは引数で与える。
"""

from __future__ import annotations

import math
import random
from typing import Callable

from agents import Agent, make_heuristic_agent
from cards import CardMeta
from cg.api import (
    Observation,
    SearchState,
    SelectType,
    search_begin,
    search_end,
    search_step,
)
from determinize import determinize

# evaluator(obs) -> (value[0,1] 手番視点, priors: option に整列した確率)
Evaluator = Callable[[Observation], "tuple[float, list[float]]"]

_MCTS_SELECT_TYPES = (SelectType.MAIN, SelectType.ATTACK)


def _is_terminal(obs: Observation) -> bool:
    return obs.current is not None and obs.current.result != -1


def _terminal_value(obs: Observation, root_seat: int) -> float:
    """終局の root 視点価値（勝ち1・引分0.5・負け0）."""
    r = obs.current.result
    if r == root_seat:
        return 1.0
    if r == 2:
        return 0.5
    return 0.0


def make_prize_evaluator(meta: CardMeta) -> Evaluator:
    """テスト/ベースライン用ダミー評価器: サイド残数差で value、priors は一様."""

    def ev(obs: Observation) -> tuple[float, list[float]]:
        st = obs.current
        n = len(obs.select.option) if obs.select else 0
        priors = [1.0 / n] * n if n > 0 else []
        if st is None:
            return 0.5, priors
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        diff = len(opp.prize) - len(me.prize)
        value = max(0.0, min(1.0, 0.5 + 0.5 * diff / 6.0))
        return value, priors

    return ev


class _Node:
    """PUCT のノード（1 つの search 状態に対応）."""

    __slots__ = (
        "search_id",
        "obs",
        "root_seat",
        "terminal",
        "to_move",
        "actions",
        "priors",
        "children",
        "n",
        "w",
        "expanded",
    )

    def __init__(self, state: SearchState, root_seat: int):
        self.search_id = state.searchId
        self.obs = state.observation
        self.root_seat = root_seat
        self.terminal = _is_terminal(self.obs)
        self.to_move = (
            self.obs.current.yourIndex
            if (self.obs.current and not self.terminal)
            else None
        )
        self.actions: list[list[int]] = []
        self.priors: list[float] = []
        self.children: dict[tuple[int, ...], _Node] = {}
        self.n = 0
        self.w = 0.0  # root 視点の価値合計
        self.expanded = False


def _actions_and_priors(
    obs: Observation, raw_priors: list[float], rng: random.Random, fallback: Agent
) -> tuple[list[list[int]], list[float]]:
    """ノードの行動集合と事前確率を作る（pick-one は全列挙、それ以外は fallback の1手）."""
    sel = obs.select
    n = len(sel.option)
    if sel.maxCount <= 1 and sel.type in _MCTS_SELECT_TYPES:
        actions = [[i] for i in range(n)]
        priors = [raw_priors[i] if i < len(raw_priors) else 0.0 for i in range(n)]
        if sel.minCount == 0:
            actions.append([])
            priors.append(1e-3)
        total = sum(priors) or 1.0
        return actions, [p / total for p in priors]
    # 非 pick-one: fallback 方策の1手に畳む（分岐しない）
    return [fallback(obs, rng)], [1.0]


def _to_root(value_to_move: float, node: _Node) -> float:
    """手番視点の value を root 視点に変換."""
    return value_to_move if node.to_move == node.root_seat else 1.0 - value_to_move


def _expand_evaluate(
    node: _Node, evaluator: Evaluator, rng: random.Random, fallback: Agent
) -> float:
    """葉を評価器で展開し、root 視点の価値を返す."""
    value_tm, raw_priors = evaluator(node.obs)
    node.actions, node.priors = _actions_and_priors(node.obs, raw_priors, rng, fallback)
    node.expanded = True
    return _to_root(value_tm, node)


def _puct_action(node: _Node, c_puct: float) -> int:
    """PUCT で行動インデックスを選ぶ（手番視点で最大化）."""
    own = node.to_move == node.root_seat
    sqrt_n = math.sqrt(node.n)
    best_i, best_score = 0, -1.0
    for i, a in enumerate(node.actions):
        child = node.children.get(tuple(a))
        cn = child.n if child else 0
        q_root = (child.w / cn) if (child and cn > 0) else 0.5  # 楽観初期値
        q = q_root if own else 1.0 - q_root
        u = c_puct * node.priors[i] * sqrt_n / (1 + cn)
        score = q + u
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def _simulate(
    root: _Node,
    c_puct: float,
    evaluator: Evaluator,
    rng: random.Random,
    fallback: Agent,
) -> None:
    """PUCT を1反復: 選択→（葉で）展開＋評価→逆伝播."""
    path = [root]
    node = root
    while True:
        if node.terminal:
            value = _terminal_value(node.obs, node.root_seat)
            break
        if not node.expanded:
            value = _expand_evaluate(node, evaluator, rng, fallback)
            break
        i = _puct_action(node, c_puct)
        key = tuple(node.actions[i])
        child = node.children.get(key)
        if child is None:
            child = _Node(search_step(node.search_id, list(key)), node.root_seat)
            node.children[key] = child
        node = child
        path.append(node)
    for nd in path:
        nd.n += 1
        nd.w += value


def aggregate_visits(
    obs: Observation,
    my_deck: list[int],
    opp_deck: list[int],
    evaluator: Evaluator,
    rng: random.Random,
    n_simulations: int,
    n_determinizations: int,
    c_puct: float,
    fallback: Agent,
) -> dict[tuple[int, ...], int]:
    """determinization 横断で PUCT を回し、根の行動別訪問数を集計して返す.

    返り値のキーは行動（option index のタプル）。self-play の方策ターゲット π や
    エージェントの行動選択に使う。
    """
    visits: dict[tuple[int, ...], int] = {}
    for _ in range(n_determinizations):
        det = determinize(obs, my_deck, opp_deck, rng)
        try:
            root_state = search_begin(
                obs,
                det["your_deck"],
                det["your_prize"],
                det["opponent_deck"],
                det["opponent_prize"],
                det["opponent_hand"],
                det["opponent_active"],
                False,
            )
        except (ValueError, IndexError):
            continue
        root = _Node(root_state, obs.current.yourIndex)
        sims = max(1, n_simulations // n_determinizations)
        for _ in range(sims):
            _simulate(root, c_puct, evaluator, rng, fallback)
        for key, child in root.children.items():
            visits[key] = visits.get(key, 0) + child.n
        search_end()
    return visits


def make_nn_mcts_agent(
    meta: CardMeta,
    my_deck: list[int],
    opp_deck: list[int],
    evaluator: Evaluator | None = None,
    n_simulations: int = 100,
    n_determinizations: int = 4,
    c_puct: float = 1.5,
    fallback: Agent | None = None,
) -> Agent:
    """NN 誘導 MCTS（PUCT）エージェントを生成する.

    evaluator 未指定時はダミー（サイド差 value＋一様 priors）を使う。実運用では学習済み
    NN を渡す。MAIN/ATTACK の pick-one 決定でのみ探索し、他はヒューリスティック即決。
    """
    heuristic = make_heuristic_agent(meta)
    evaluator = evaluator or make_prize_evaluator(meta)
    fallback = fallback or heuristic

    def agent(obs: Observation, rng: random.Random) -> list[int]:
        sel = obs.select
        if sel is None or len(sel.option) <= 1 or sel.type not in _MCTS_SELECT_TYPES:
            return heuristic(obs, rng)
        visits = aggregate_visits(
            obs,
            my_deck,
            opp_deck,
            evaluator,
            rng,
            n_simulations,
            n_determinizations,
            c_puct,
            fallback,
        )
        if not visits:
            return heuristic(obs, rng)
        return list(max(visits, key=lambda k: visits[k]))

    return agent
