"""self-play による学習データ収集（Phase 3）.

NN 誘導 MCTS（nn_mcts）で自己対戦し、各 MAIN/ATTACK 決定で
`(状態特徴量, 合法手特徴量, 方策ターゲット π, 価値ターゲット z)` を収集する。
π は根の訪問分布（option index 整列）、z は試合結果（その手番視点で勝ち1/引分0.5/負け0）。

評価器は注入式: ダミー（make_prize_evaluator）ならホストで検証可、学習済み NN なら Docker。
torch 非依存（features は numpy）。サンプルの state/actions は numpy 配列。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from agents import make_heuristic_agent
from cards import CardMeta
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from features import encode_actions, encode_observation
from nn_mcts import (
    Evaluator,
    _MCTS_SELECT_TYPES,
    aggregate_visits,
    make_prize_evaluator,
)


@dataclass
class Sample:
    """学習サンプル."""

    state: np.ndarray  # (OBS_FEAT_LEN,)
    actions: np.ndarray  # (n_options, ACTION_FEAT_LEN)
    pi: np.ndarray  # (n_options,) 方策ターゲット（訪問分布）
    z: float  # 価値ターゲット（手番視点の勝敗）


def _sample_index(pi: list[float], rng: random.Random) -> int:
    """訪問分布 π から行動 index を1つサンプリング（探索のため）."""
    return rng.choices(range(len(pi)), weights=pi, k=1)[0]


def play_selfplay_game(
    meta: CardMeta,
    deck: list[int],
    evaluator: Evaluator,
    rng: random.Random,
    n_simulations: int = 64,
    n_determinizations: int = 2,
    c_puct: float = 1.5,
    max_steps: int = 100_000,
    sample_actions: bool = True,
) -> list[Sample]:
    """1 試合を自己対戦し（両席とも同じ評価器）、学習サンプル列を返す."""
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
                visits = aggregate_visits(
                    obs,
                    deck,
                    deck,
                    evaluator,
                    rng,
                    n_simulations,
                    n_determinizations,
                    c_puct,
                    heuristic,
                )
                n = len(sel.option)
                counts = [visits.get((i,), 0) for i in range(n)]
                total = sum(counts)
                if total > 0:
                    pi = np.asarray([c / total for c in counts], dtype=np.float32)
                    pending.append(
                        (
                            encode_observation(obs, meta),
                            encode_actions(obs, meta),
                            pi,
                            obs.current.yourIndex,
                        )
                    )
                    idx = (
                        _sample_index(list(pi), rng)
                        if sample_actions
                        else int(pi.argmax())
                    )
                    action = [idx]
                else:
                    action = heuristic(obs, rng)
            else:
                action = heuristic(obs, rng)
            obs_dict = battle_select(action)
    finally:
        battle_finish()

    # 価値ターゲットを付与（その手番が勝てば 1、引分 0.5、負け 0）
    samples = []
    for state, actions, pi, seat in pending:
        if result is None:
            z = 0.5
        elif result == seat:
            z = 1.0
        elif result == 2:
            z = 0.5
        else:
            z = 0.0
        samples.append(Sample(state, actions, pi, z))
    return samples


def generate_samples(
    meta: CardMeta,
    deck: list[int],
    evaluator: Evaluator | None,
    n_games: int,
    rng: random.Random,
    **game_kwargs,
) -> list[Sample]:
    """n_games 回の自己対戦でサンプルをまとめて収集する."""
    evaluator = evaluator or make_prize_evaluator(meta)
    samples: list[Sample] = []
    for _ in range(n_games):
        samples.extend(play_selfplay_game(meta, deck, evaluator, rng, **game_kwargs))
    return samples
