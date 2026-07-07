"""チャンピオンデッキの ISMCTS 再評価（Step 3）.

デッキ最適化（evolve/league）はヒューリスティック操縦で行うため、コンボを回せず
単純な攻撃デッキを過大評価する**操縦バイアス**がある。本モジュールは候補デッキを
**ISMCTS 操縦**で相手プールに対し再評価し、ヒューリスティック操縦の結果と比較する。
これで「ヒューリスティックで強いだけ」のデッキを見破り、本物の頑健さを確かめる。

操縦は席ごとに自分/相手デッキで determinize する必要があるため、
harness.evaluate_decks_with_factory（席ごとに操縦者を生成）を使う。
"""

from __future__ import annotations

import argparse
import random

from agents import make_heuristic_agent
from cards import CardMeta, load_card_meta
from deckopt import _load_pool, default_opponent_paths
from harness import evaluate_decks_with_factory
from ismcts import make_ismcts_agent


def heuristic_factory(meta: CardMeta):
    """席に依らず共通のヒューリスティック操縦者を返す factory."""
    agent = make_heuristic_agent(meta)
    return lambda my_deck, opp_deck: agent


def ismcts_factory(
    meta: CardMeta, time_budget: float = 0.3, game_budget: float | None = None
):
    """席ごとに自分/相手デッキで determinize する ISMCTS 操縦者を返す factory."""
    return lambda my_deck, opp_deck: make_ismcts_agent(
        meta, my_deck, opp_deck, time_budget=time_budget, game_budget=game_budget
    )


def deck_winrates(
    candidate: list[int],
    pool: list[list[int]],
    factory,
    rng: random.Random,
    games_per_opp: int,
) -> dict:
    """candidate の pool 各デッキ（自分自身は除外）への勝率と min/mean を返す."""
    opps = [p for p in pool if p != candidate]
    if not opps:
        return {"min": 1.0, "mean": 1.0, "per_opp": []}
    per_opp = [
        evaluate_decks_with_factory(candidate, opp, factory, rng, games_per_opp)[
            "win_rate_a"
        ]
        for opp in opps
    ]
    return {
        "min": min(per_opp),
        "mean": sum(per_opp) / len(per_opp),
        "per_opp": per_opp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="チャンピオンの ISMCTS 再評価")
    parser.add_argument(
        "--candidate",
        default="models/champion_deck.csv",
        help="再評価する候補デッキ CSV",
    )
    parser.add_argument(
        "--pool",
        nargs="+",
        default=None,
        help="相手プール CSV 群（未指定なら実メタ優先＝gauntlet/>replays/opp_decks/>data/*.csv）",
    )
    parser.add_argument("--games", type=int, default=6, help="相手1体あたりの対戦数")
    parser.add_argument(
        "--time-budget", type=float, default=0.3, help="ISMCTS の1手思考時間（秒）"
    )
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    args = parser.parse_args()

    candidate = _load_pool([args.candidate])[0]
    pool_paths = args.pool or default_opponent_paths()  # 未指定は実メタ優先
    pool = _load_pool(pool_paths)
    meta = load_card_meta()

    print(f"候補を相手プール {len(pool)} デッキに対し再評価（各 {args.games} 試合）\n")

    rng = random.Random(args.seed)
    heur = deck_winrates(candidate, pool, heuristic_factory(meta), rng, args.games)
    print(
        f"ヒューリスティック操縦: 最悪={heur['min']:.3f} 平均={heur['mean']:.3f} per_opp={[round(x, 2) for x in heur['per_opp']]}"
    )

    rng = random.Random(args.seed)  # 同条件で比較
    ismcts = deck_winrates(
        candidate,
        pool,
        ismcts_factory(meta, time_budget=args.time_budget),
        rng,
        args.games,
    )
    print(
        f"ISMCTS 操縦         : 最悪={ismcts['min']:.3f} 平均={ismcts['mean']:.3f} per_opp={[round(x, 2) for x in ismcts['per_opp']]}"
    )

    print(
        f"\n操縦による最悪ケース勝率の差: {ismcts['min'] - heur['min']:+.3f}"
        "（負なら操縦バイアスで過大評価されていた可能性）"
    )


if __name__ == "__main__":
    main()
