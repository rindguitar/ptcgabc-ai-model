"""ISMCTS 教師からの蒸留データ収集（Phase 3）.

ISMCTS（合法手＝特性/効果/トレーナーを探索して価値で選ぶ強い操縦）で対戦し、各 MAIN/ATTACK
決定で `(状態特徴量, 合法手特徴量, 方策ターゲット π, 価値ターゲット z)` を収集する。
π は温度 temp で one-hot↔訪問分布(soft)を1本化（既定 temp=0＝最多訪問手の one-hot）、
z は試合結果（その手番視点で勝1/分0.5/負0）。

狙い: 自己対戦は弱い net から始めると崩壊しうる。ISMCTS は劣化しない強い教師なので、
これを真似る蒸留で**安定して強い土台**を作る。出力 Sample は selfplay と同形式で、既存の
train() がそのまま使える。torch 非依存（features は numpy・ホストで収集/検証可）。
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from agents import make_heuristic_agent
from cards import CardMeta
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from features import encode_actions, encode_observation
from ismcts import ismcts_aggregate
from nn_mcts import _MCTS_SELECT_TYPES
from selfplay import Sample, as_deck_list

# この訪問数未満は探索信号が薄いので方策を記録せず heuristic で進める（make_ismcts_agent と同値）
_MIN_VISITS = 15


def _visits_to_pi(visits: list[float], temp: float) -> np.ndarray:
    """訪問回数 → 方策ターゲット π。temp で one-hot↔soft を1本化（π ∝ visits^(1/temp)）.

    temp<=0: 最多訪問手の **one-hot**（既定。浅い teacher では安全＝tie-break ノイズに過信しない範囲で確実）。
    temp>0 : 訪問分布を温度で鋭く/平坦に（temp→0 で one-hot、temp=1 で生の訪問分布=soft）。
    数値安定のため最大訪問で正規化してから累乗する（巨大値^大指数の overflow 回避）。
    """
    n = len(visits)
    if temp <= 0.0:
        pi = np.zeros(n, dtype=np.float32)
        pi[int(max(range(n), key=lambda i: visits[i]))] = 1.0
        return pi
    maxv = max(visits) or 1.0
    w = [(v / maxv) ** (1.0 / temp) if v > 0 else 0.0 for v in visits]
    s = sum(w) or 1.0
    return np.asarray([x / s for x in w], dtype=np.float32)


def play_ismcts_distill_game(
    meta: CardMeta,
    deck: list[int],
    rng: random.Random,
    time_budget: float = 0.1,
    temp: float = 0.0,
    max_steps: int = 100_000,
) -> list[Sample]:
    """ISMCTS で1試合（両席ミラー）し、ISMCTS の探索を教師に蒸留サンプル列を返す.

    各 MAIN/ATTACK 決定で方策ターゲット π を temp で作る（既定 temp=0＝最多訪問手の one-hot）。
    temp>0 で訪問分布(soft-π)を解放できる（teacher を深くした時に僅差情報を活かす）。実行手は
    最多訪問手（標準的な MCTS の着手）。訪問が薄い局面は記録せず heuristic で進める。
    """
    heuristic = make_heuristic_agent(meta)
    obs_dict, _ = battle_start(deck, deck)
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    result: int | None = None

    try:
        for _ in range(max_steps):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                result = obs.current.result
                break
            sel = obs.select
            if sel is None:
                break

            if sel.type in _MCTS_SELECT_TYPES and len(sel.option) > 1:
                agg = ismcts_aggregate(
                    meta, obs, deck, deck, rng, time_budget=time_budget
                )
                n = len(sel.option)
                visits = [agg.get((i,), (0.0, 0.0))[0] for i in range(n)]
                total = sum(visits)
                if total >= _MIN_VISITS:
                    pi = _visits_to_pi(visits, temp)
                    pending.append(
                        (
                            encode_observation(obs, meta),
                            encode_actions(obs, meta),
                            pi,
                            obs.current.yourIndex,
                        )
                    )
                    action = [int(max(range(n), key=lambda i: visits[i]))]
                else:
                    action = heuristic(obs, rng)
            else:
                action = heuristic(obs, rng)
            obs_dict = battle_select(action)
    finally:
        battle_finish()

    # 価値ターゲット（その手番が勝てば 1、引分 0.5、負け 0）
    samples = []
    for state, actions, pi, seat in pending:
        if result is None or result == 2:
            z = 0.5
        elif result == seat:
            z = 1.0
        else:
            z = 0.0
        samples.append(Sample(state, actions, pi, z))
    return samples


# --- 並列収集（CPU マルチコアで「1回あたりの試行回数」を増やす） ---------------
# ISMCTS 教師の対戦は CPU バウンドで各試合が独立。worker プロセスに分散すれば
# ほぼコア数倍のサンプルが同じ時間で集まる（GPU は学習側でのみ使う）。
# meta はプロセス毎に load_card_meta() で読み直す（pickle 不要・データ漏洩もなし）。
_W: dict = {}


def _worker_init(pool: list[list[int]], time_budget: float, temp: float) -> None:
    """worker プロセス初期化: meta を読み込みデッキプール/教師強度/温度を保持."""
    from cards import load_card_meta

    _W["meta"] = load_card_meta()
    _W["pool"] = pool
    _W["time_budget"] = time_budget
    _W["temp"] = temp


def _worker_game(seed: int) -> list[Sample]:
    """worker: 与えられた seed でデッキを選び1試合の蒸留サンプルを返す."""
    rng = random.Random(seed)
    deck = rng.choice(_W["pool"])
    return play_ismcts_distill_game(
        _W["meta"], deck, rng, time_budget=_W["time_budget"], temp=_W["temp"]
    )


def generate_ismcts_samples(
    meta: CardMeta,
    decks,
    n_games: int,
    rng: random.Random,
    time_budget: float = 0.1,
    n_workers: int = 1,
    temp: float = 0.0,
) -> list[Sample]:
    """n_games 回 ISMCTS 対戦し、蒸留サンプルをまとめて収集する.

    decks は単一デッキ(list[int])でも複数デッキ(list[list[int]])でも可。複数なら毎試合
    ランダムに1つ選ぶ（多様なデッキを操縦できる汎用 pilot に近づける）。

    temp は方策ターゲットの温度（既定 0=one-hot。>0 で訪問分布 soft-π を解放）。
    n_workers>1 で試合を複数プロセスに並列化（各試合は独立＝ほぼコア数倍に高速化）。
    """
    pool = as_deck_list(decks)
    if n_workers and n_workers > 1:
        # 再現性のため各試合の seed を親 rng から先に確定させる
        seeds = [rng.randrange(2**31) for _ in range(n_games)]
        samples: list[Sample] = []
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(pool, time_budget, temp),
        ) as ex:
            for game_samples in ex.map(_worker_game, seeds):
                samples.extend(game_samples)
        return samples
    # 逐次（n_workers<=1）
    samples = []
    for _ in range(n_games):
        deck = rng.choice(pool)
        samples.extend(
            play_ismcts_distill_game(
                meta, deck, rng, time_budget=time_budget, temp=temp
            )
        )
    return samples
