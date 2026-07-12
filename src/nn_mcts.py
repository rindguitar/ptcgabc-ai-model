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
import time
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
from determinize import determinize, pick_opponent_deck

# evaluator(obs) -> (value[0,1] 手番視点, priors: option に整列した確率)
Evaluator = Callable[[Observation], "tuple[float, list[float]]"]

_MCTS_SELECT_TYPES = (SelectType.MAIN, SelectType.ATTACK)

# 適応 sims（時間で考える）: 1試合あたりの自分の MCTS 決定数の見込み。replay 実測の
# 平均 steps（両者合計 80〜100・サブ選択込み）から pick-one 決定はこの程度。残り決定数の
# 床 8 は「終盤に一気に使い切って時間切れ」を防ぐ保守項。
_EXPECTED_DECISIONS = 40.0
_MIN_DECISIONS_LEFT = 8.0


def plan_search(
    base_sims: int,
    base_dets: int,
    remaining: float,
    moves_done: int,
    unit_cost: float | None,
    sims_cap: int = 512,
    dets_cap: int = 16,
) -> tuple[int, int]:
    """残り時間予算から今手の (sims, dets) を決める（§31・適応探索の中核）.

    unit_cost = 「1 simulation × 1 determinization」の実測秒（EMA・floor 込み）。未計測（初手）は
    (base_sims, base_dets)。1手予算 = 残り予算 ÷ 残り決定数の見込み、units = 予算÷単価を
    **決定化の幅（dets）優先**で配る: この net は sims 32≈64 で深さ飽和の実測（design-decisions）が
    あり、不完全情報では決定化平均（PIMC）の分散低減＝§26 ベイズ推定の多サンプル化の方が効くため。
    base を床（従来品質を下回らない）・cap を天井（1手の暴走・メモリの防止）。
    """
    if unit_cost is None or unit_cost <= 0 or remaining <= 0:
        return base_sims, base_dets
    left = max(_MIN_DECISIONS_LEFT, _EXPECTED_DECISIONS - moves_done)
    units = (remaining / left) / unit_cost  # この手に使える (sim×det) 数
    dets = max(base_dets, min(dets_cap, int(units // max(2 * base_sims, 128))))
    sims = max(base_sims, min(sims_cap, int(units // dets)))
    return sims, dets


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


def _apply_root_noise(
    root: _Node,
    eps: float,
    evaluator: Evaluator,
    rng: random.Random,
    fallback: Agent,
) -> None:
    """根の priors に Dirichlet ノイズを混ぜる（AlphaZero 標準・self-play 収集用）.

    P(s,a) ← (1−ε)·p + ε·η, η〜Dir(α)（α=10/手数 の経験則）。net の priors が平坦/偏って
    いても探索が毎回同じ手に固まらず、self-play データの多様性が保たれる（改善オペレータの
    「見落とし」を防ぐ）。推論/評価では使わない（eps=0）。
    """
    if root.terminal:
        return
    if not root.expanded:
        # 先に根を展開して priors を得る（この評価も1訪問として逆伝播しておく）
        value = _expand_evaluate(root, evaluator, rng, fallback)
        root.n += 1
        root.w += value
    k = len(root.priors)
    if k <= 1:
        return
    alpha = 10.0 / k
    noise = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(noise) or 1.0
    root.priors = [
        (1.0 - eps) * p + eps * (g / total) for p, g in zip(root.priors, noise)
    ]


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
    opp_pool: list[list[int]] | None = None,
    root_noise_eps: float = 0.0,
) -> dict[tuple[int, ...], int]:
    """determinization 横断で PUCT を回し、根の行動別訪問数を集計して返す.

    返り値のキーは行動（option index のタプル）。self-play の方策ターゲット π や
    エージェントの行動選択に使う。opp_pool 指定時は相手デッキを観測整合で推定する。
    root_noise_eps>0 で根に Dirichlet ノイズ（self-play 収集の多様性確保・推論では 0）。
    """
    visits: dict[tuple[int, ...], int] = {}
    for _ in range(n_determinizations):
        opp = pick_opponent_deck(obs, opp_pool, opp_deck, rng)
        det = determinize(obs, my_deck, opp, rng)
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
        if root_noise_eps > 0:
            _apply_root_noise(root, root_noise_eps, evaluator, rng, fallback)
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
    opp_pool: list[list[int]] | None = None,
    floor_rollouts: int = 0,
    floor_margin: float = 0.0,
    game_budget: float | None = None,
    max_simulations: int | None = None,
) -> Agent:
    """NN 誘導 MCTS（PUCT）エージェントを生成する.

    evaluator 未指定時はダミー（サイド差 value＋一様 priors）を使う。実運用では学習済み
    NN を渡す。MAIN/ATTACK の pick-one 決定でのみ探索し、他はヒューリスティック即決。

    floor_rollouts>0 で**接地安全弁**を有効化: MCTS が選んだ手と heuristic の手が異なるとき、
    両者を heuristic rollout で実地比較し（net 非依存）、**heuristic を接地評価で上回るときだけ
    NN 手を採用**する。net の value が誤っていても pilot が heuristic を下回らない保証。
    推論/評価/提出向け（rollout 分のレイテンシ増・自己対戦の収集では使わない）。

    game_budget>0 で**適応 sims**を有効化（§31・提出向け）: 自前時計と単価 EMA から
    plan_sims で今手の sims を決める（床=n_simulations・天井=max_simulations 既定8倍）。
    実測で NN 提出は 600 秒中 ~550 秒を残しており、推論時間は未使用の計算資源。
    eval/gate は game_budget を渡さない＝従来の固定 sims（測定時間が爆発しない）。
    """
    heuristic = make_heuristic_agent(meta)
    evaluator = evaluator or make_prize_evaluator(meta)
    fallback = fallback or heuristic
    sims_cap = max_simulations or n_simulations * 8
    # 適応 sims の内部状態: [消費秒, 決定数, 単価 EMA（1sim×1det の秒・floor 込み）]
    spent, moves, unit_ema = [0.0], [0], [None]

    def agent(obs: Observation, rng: random.Random) -> list[int]:
        sel = obs.select
        if sel is None or len(sel.option) <= 1 or sel.type not in _MCTS_SELECT_TYPES:
            return heuristic(obs, rng)
        if game_budget is not None:
            sims, dets = plan_search(
                n_simulations,
                n_determinizations,
                game_budget - spent[0],
                moves[0],
                unit_ema[0],
                sims_cap=sims_cap,
            )
            t0 = time.perf_counter()
            action = _search_and_decide(obs, rng, sims, dets)
            dt = time.perf_counter() - t0
            spent[0] += dt
            moves[0] += 1
            unit = dt / max(1, sims * dets)  # floor 分も単価に畳む（保守的）
            unit_ema[0] = (
                unit if unit_ema[0] is None else 0.7 * unit_ema[0] + 0.3 * unit
            )
            return action
        return _search_and_decide(obs, rng, n_simulations, n_determinizations)

    def _search_and_decide(
        obs: Observation, rng: random.Random, sims: int, dets: int
    ):
        visits = aggregate_visits(
            obs,
            my_deck,
            opp_deck,
            evaluator,
            rng,
            sims,
            dets,
            c_puct,
            fallback,
            opp_pool,
        )
        if not visits:
            return heuristic(obs, rng)
        nn_action = list(max(visits, key=lambda k: visits[k]))
        if floor_rollouts <= 0:
            return nn_action
        # 接地 floor: NN 手と heuristic 手が違うときだけ rollout で実地比較
        h_action = heuristic(obs, rng)
        if tuple(h_action) == tuple(nn_action):
            return nn_action
        from ismcts import evaluate_actions_by_rollout

        vals = evaluate_actions_by_rollout(
            meta,
            obs,
            my_deck,
            opp_deck,
            [nn_action, h_action],
            rng,
            n_rollouts=floor_rollouts,
            opp_pool=opp_pool,
        )
        nn_v = vals.get(tuple(nn_action))
        h_v = vals.get(tuple(h_action))
        # 接地評価で NN 手が heuristic 手を margin 超で上回るときだけ NN を採用（さもなくば heuristic）
        if nn_v is not None and h_v is not None and nn_v > h_v + floor_margin:
            return nn_action
        return list(h_action)

    return agent
