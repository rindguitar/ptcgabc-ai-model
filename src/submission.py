"""Kaggle 提出形式へのアダプタ.

我々の方策 `(obs, rng) -> list[int]` を、公式の `agent(obs_dict) -> list[int]` に包む。
初手（obs.select is None）はデッキ60枚（カードID）を返し、以降は方策で選ぶ。

提出物としても、公式ビュアー（kaggle_environments の env.render(html)）でチャンピオンの
動きを観るためにも、この形式が必要。本大会の1試合600秒クロックに対応するため、ISMCTS は
game_budget（既定 540 = 安全マージン）で動かす。

相手デッキ opp_deck は実戦では未知のため、既定では自分のデッキを仮置きする
（determinize の相手隠れ札サンプリング用の事前分布。観測からの推定は今後の改良余地）。
"""

from __future__ import annotations

import os
import random
from typing import Callable

from cards import load_card_meta
from cg.api import to_observation_class

DECK_SIZE = 60


def _read_deck_csv(path: str) -> list[int]:
    """deck CSV を読む。ローカルに無ければ Kaggle 提出パスにフォールバック."""
    if not os.path.exists(path):
        alt = "/kaggle_simulations/agent/" + os.path.basename(path)
        if os.path.exists(alt):
            path = alt
    with open(path) as f:
        ids = [int(line) for line in f.read().splitlines() if line.strip()]
    if len(ids) != DECK_SIZE:
        raise ValueError(f"デッキは {DECK_SIZE} 枚必要（実際 {len(ids)} 枚）")
    return ids


def make_kaggle_agent(
    policy: str = "ismcts",
    deck: list[int] | None = None,
    deck_path: str = "deck.csv",
    opp_deck: list[int] | None = None,
    meta=None,
    seed: int = 0,
    game_budget: float | None = 540.0,
    time_budget: float = 0.5,
    **policy_kwargs,
) -> Callable[[dict], list[int]]:
    """公式形式 `agent(obs_dict) -> list[int]` を返す.

    Args:
        policy: "ismcts" / "heuristic" / "random"。
        deck: 自分のデッキ（None なら deck_path から読む）。
        opp_deck: 相手デッキの事前分布（None なら deck を流用）。
        game_budget/time_budget: ISMCTS の時間管理（game_budget 優先）。
    """
    meta = meta or load_card_meta()
    deck = list(deck) if deck is not None else _read_deck_csv(deck_path)
    opp_deck = list(opp_deck) if opp_deck is not None else list(deck)
    rng = random.Random(seed)

    if policy == "heuristic":
        from agents import make_heuristic_agent

        inner = make_heuristic_agent(meta)
    elif policy == "random":
        from agents import random_agent

        inner = random_agent
    elif policy == "ismcts":
        from ismcts import make_ismcts_agent

        inner = make_ismcts_agent(
            meta,
            deck,
            opp_deck,
            time_budget=time_budget,
            game_budget=game_budget,
            **policy_kwargs,
        )
    else:
        raise ValueError(f"未知の policy: {policy}")

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)  # 初手: デッキ60枚を返す
        return inner(obs, rng)

    return agent
