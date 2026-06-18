"""ISMCTS エージェント（determinized UCT / PIMC）.

不完全情報ゲーム向けに、隠れ情報を determinize（=1つの仮定にサンプリング）してから
エンジンの search API（search_begin/step/end）をフォワードモデルとして UCT 木探索する。
determinization を複数試して、ルートで「最も多く訪問された行動」を採用する（PIMC）。

v1 の割り切り:
- 木の分岐は MAIN / ATTACK の pick-one 決定に集中する（戦略的に重要なため）。
- 多選択（minCount..maxCount で複数選ぶ）やサブ選択は rollout 方策に委譲し分岐しない。
- 葉は rollout 方策（既定: ヒューリスティック）で終局までプレイして勝敗を価値にする。
  打ち切り時はサイド残数差から価値を推定する。

時間予算 `time_budget`（秒）はパラメータ。1手の時間制限が判明したら安全マージンを見て設定する。
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from time import perf_counter

from agents import Agent, make_heuristic_agent
from cards import CardMeta
from cg.api import (
    Observation,
    SearchState,
    SelectType,
    search_begin,
    search_end,
    search_release,
    search_step,
)
from determinize import determinize

# MCTS で深掘りする選択タイプ（戦略的な pick-one）。それ以外は rollout 方策に委譲。
_MCTS_SELECT_TYPES = (SelectType.MAIN, SelectType.ATTACK)


def _is_terminal(obs: Observation) -> bool:
    """observation が終局状態か."""
    return obs.current is not None and obs.current.result != -1


def _value(obs: Observation, root_seat: int) -> float:
    """root プレイヤー視点の価値 [0,1]（勝ち=1, 引分=0.5, 負け=0）.

    終局していない（rollout 打ち切り）場合はサイド残数差から推定する。
    """
    st = obs.current
    if st is None:
        return 0.5
    if st.result != -1:
        if st.result == root_seat:
            return 1.0
        if st.result == 2:  # 引き分け
            return 0.5
        return 0.0
    # 未終局: サイド残数が少ない方が勝ちに近い
    me = st.players[root_seat]
    opp = st.players[1 - root_seat]
    diff = len(opp.prize) - len(me.prize)  # 自分のサイドが少ないほど + （勝ちに近い）
    return max(0.0, min(1.0, 0.5 + 0.5 * diff / 6.0))


def _actions_for(
    obs: Observation, rng: random.Random, rollout_policy: Agent
) -> list[list[int]]:
    """ノードの合法手集合を返す。pick-one は全列挙、多選択は方策の1手に畳む."""
    sel = obs.select
    n = len(sel.option)
    if sel.maxCount <= 1:
        acts: list[list[int]] = [[i] for i in range(n)]
        if sel.minCount == 0:
            acts.append([])  # 「何も選ばない」も選択肢
        # 展開順の偏り（pop は末尾＝常に END から）を避けるためシャッフルする
        rng.shuffle(acts)
        return acts
    # 多選択は分岐しない（v1）。方策が選ぶ1手のみ。
    return [rollout_policy(obs, rng)]


class _Node:
    """determinized UCT のノード（1 つの search 状態に対応）."""

    __slots__ = (
        "search_id",
        "obs",
        "root_seat",
        "terminal",
        "to_move",
        "untried",
        "children",
        "n",
        "w",
    )

    def __init__(
        self,
        state: SearchState,
        root_seat: int,
        rng: random.Random,
        rollout_policy: Agent,
    ):
        self.search_id = state.searchId
        self.obs = state.observation
        self.root_seat = root_seat
        self.terminal = _is_terminal(self.obs)
        self.to_move = (
            self.obs.current.yourIndex
            if (self.obs.current and not self.terminal)
            else None
        )
        self.untried = (
            [] if self.terminal else _actions_for(self.obs, rng, rollout_policy)
        )
        self.children: dict[tuple[int, ...], _Node] = {}
        self.n = 0
        self.w = 0.0  # root 視点の価値合計

    def is_fully_expanded(self) -> bool:
        return not self.untried

    def expand(self, rng: random.Random, rollout_policy: Agent) -> "_Node":
        """未試行の行動を1つ展開して子ノードを返す."""
        action = self.untried.pop()
        child_state = search_step(self.search_id, action)
        child = _Node(child_state, self.root_seat, rng, rollout_policy)
        self.children[tuple(action)] = child
        return child

    def best_child(self, c: float) -> "_Node":
        """UCB1 で子を選ぶ（手番プレイヤー視点で価値を最大化）."""
        log_n = math.log(self.n)
        own_turn = self.to_move == self.root_seat

        def ucb(child: "_Node") -> float:
            q = child.w / child.n  # root 視点の平均価値
            q = q if own_turn else (1.0 - q)  # 相手手番なら相手が自分の価値を最小化
            return q + c * math.sqrt(log_n / child.n)

        return max(self.children.values(), key=ucb)


def _rollout(
    node: _Node, rollout_policy: Agent, rng: random.Random, max_depth: int
) -> float:
    """葉から rollout 方策で終局まで進め、root 視点の価値を返す."""
    if node.terminal:
        return _value(node.obs, node.root_seat)

    search_id = node.search_id
    obs = node.obs
    prev_rollout_id: int | None = None  # rollout で作った使い捨て search_id
    for _ in range(max_depth):
        if _is_terminal(obs) or obs.select is None:
            break
        action = rollout_policy(obs, rng)
        state = search_step(search_id, action)
        # 直前の rollout 用 search 状態を解放（木ノードの search_id は解放しない）
        if prev_rollout_id is not None:
            search_release(prev_rollout_id)
        prev_rollout_id = state.searchId
        search_id = state.searchId
        obs = state.observation

    value = _value(obs, node.root_seat)
    if prev_rollout_id is not None:
        search_release(prev_rollout_id)
    return value


def _simulate(
    root: _Node, c: float, rollout_policy: Agent, rng: random.Random, max_depth: int
) -> None:
    """UCT を1反復: 選択 → 展開 → rollout → 逆伝播."""
    node = root
    path = [root]
    # 選択
    while not node.terminal and node.is_fully_expanded() and node.children:
        node = node.best_child(c)
        path.append(node)
    # 展開
    if not node.terminal and not node.is_fully_expanded():
        node = node.expand(rng, rollout_policy)
        path.append(node)
    # rollout
    value = _rollout(node, rollout_policy, rng, max_depth)
    # 逆伝播
    for nd in path:
        nd.n += 1
        nd.w += value


def make_ismcts_agent(
    meta: CardMeta,
    my_deck: list[int],
    opp_deck: list[int],
    time_budget: float = 0.5,
    iters_per_det: int = 40,
    c: float = 1.4,
    max_rollout_depth: int = 300,
    min_visits: int = 15,
    select_margin: float = 0.05,
    rollout_policy: Agent | None = None,
) -> Agent:
    """ISMCTS エージェントを生成する.

    Args:
        meta: カードメタ。
        my_deck/opp_deck: 自分/相手のデッキ構成（determinize 用。自己対戦は同一）。
        time_budget: 1手あたりの思考時間（秒）。
        iters_per_det: determinization 1つあたりの UCT 反復数。
        c: UCB の探索係数。
        max_rollout_depth: rollout の最大手数（打ち切り時はサイド差で価値推定）。
        min_visits: 行動を信頼するのに必要な集計訪問数（これ未満は採用しない）。
        select_margin: ヒューリスティックの手を上回ったと見なす平均価値の差。
        rollout_policy: 葉の評価方策（既定: ヒューリスティック）。

    行動選択は**ヒューリスティックをアンカー**にし、MCTS が十分な訪問数かつ
    明確なマージンで上回ったときだけ逸脱する（信号が弱い低予算では heuristic に一致＝劣化しない）。
    """
    heuristic = make_heuristic_agent(meta)
    policy = rollout_policy or heuristic

    def ismcts_agent(obs: Observation, rng: random.Random) -> list[int]:
        sel = obs.select
        # 自明 / 非戦略的な選択はヒューリスティック即決（探索コストを温存）
        if sel is None or len(sel.option) <= 1 or sel.type not in _MCTS_SELECT_TYPES:
            return heuristic(obs, rng)

        # ルート行動の統計を determinization 横断で集計: action_key -> [visits, value_sum]
        agg: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
        deadline = perf_counter() + time_budget
        n_det = 0
        while perf_counter() < deadline:
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
                n_det += 1
                continue

            root = _Node(root_state, obs.current.yourIndex, rng, policy)
            for _ in range(iters_per_det):
                if perf_counter() >= deadline:
                    break
                _simulate(root, c, policy, rng, max_rollout_depth)

            for key, child in root.children.items():
                agg[key][0] += child.n
                agg[key][1] += child.w
            search_end()
            n_det += 1

        h_action = tuple(heuristic(obs, rng))  # アンカー（既定の手）
        if not agg:
            return list(h_action)

        # ヒューリスティックの手の平均価値を基準に、明確に上回る行動だけ採用する
        def mean_value(key: tuple[int, ...]) -> float:
            visits, value_sum = agg[key]
            return value_sum / visits if visits > 0 else 0.0

        best_key = h_action
        best_mean = mean_value(h_action) if h_action in agg else -1.0
        for key, (visits, value_sum) in agg.items():
            if visits < min_visits:
                continue
            m = value_sum / visits
            if m > best_mean + select_margin:
                best_key, best_mean = key, m
        return list(best_key)

    return ismcts_agent
