"""ISMCTS 教師からの蒸留データ収集（Phase 3）.

ISMCTS（合法手＝特性/効果/トレーナーを探索して価値で選ぶ強い操縦）で対戦し、各 MAIN/ATTACK
決定で `(状態特徴量, 合法手特徴量, 方策ターゲット π, 価値ターゲット z)` を収集する。
π は **ISMCTS が選んだ手の one-hot**（行動クローン）、z は試合結果（その手番視点で勝1/分0.5/負0）。

狙い: 自己対戦は弱い net から始めると崩壊しうる。ISMCTS は劣化しない強い教師なので、
これを真似る蒸留で**安定して強い土台**を作る。出力 Sample は selfplay と同形式で、既存の
train() がそのまま使える。torch 非依存（features は numpy・ホストで収集/検証可）。
"""

from __future__ import annotations

import random

import numpy as np

from cards import CardMeta
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from features import encode_actions, encode_observation
from ismcts import make_ismcts_agent
from nn_mcts import _MCTS_SELECT_TYPES
from selfplay import Sample, as_deck_list


def play_ismcts_distill_game(
    meta: CardMeta,
    deck: list[int],
    rng: random.Random,
    time_budget: float = 0.1,
    max_steps: int = 100_000,
) -> list[Sample]:
    """ISMCTS で1試合（両席ミラー）し、ISMCTS の選択を教師に蒸留サンプル列を返す."""
    ismcts = make_ismcts_agent(meta, deck, deck, time_budget=time_budget)
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

            action = ismcts(obs, rng)  # ISMCTS（教師）の選択
            # 単一選択の MAIN/ATTACK 等のみ記録（one-hot 方策ターゲットが曖昧にならないよう）
            if (
                sel.type in _MCTS_SELECT_TYPES
                and len(sel.option) > 1
                and len(action) == 1
            ):
                pi = np.zeros(len(sel.option), dtype=np.float32)
                pi[action[0]] = 1.0
                pending.append(
                    (
                        encode_observation(obs, meta),
                        encode_actions(obs, meta),
                        pi,
                        obs.current.yourIndex,
                    )
                )
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


def generate_ismcts_samples(
    meta: CardMeta,
    decks,
    n_games: int,
    rng: random.Random,
    time_budget: float = 0.1,
) -> list[Sample]:
    """n_games 回 ISMCTS 対戦し、蒸留サンプルをまとめて収集する.

    decks は単一デッキ(list[int])でも複数デッキ(list[list[int]])でも可。複数なら毎試合
    ランダムに1つ選ぶ（多様なデッキを操縦できる汎用 pilot に近づける）。
    """
    pool = as_deck_list(decks)
    samples: list[Sample] = []
    for _ in range(n_games):
        deck = rng.choice(pool)
        samples.extend(
            play_ismcts_distill_game(meta, deck, rng, time_budget=time_budget)
        )
    return samples
