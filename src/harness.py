"""自己対戦ハーネス（エージェント評価用）.

cabt Engine（追跡外の提供物 `cg`）を直接ドライブし、2 つのエージェントを対戦させて
勝率を集計する。先手有利を打ち消すため、試合ごとに席（先手/後手）を入れ替えて評価する。

実行例（リポジトリ直下から）:
    python src/harness.py --a heuristic --b random --games 100 --seed 0
    python src/harness.py --a random --b random --games 20   # 純粋なエンジン疎通確認
"""

from __future__ import annotations

import argparse
import os
import random
import sys

# このファイルと同階層（src/）を import path に追加し、提供物 `cg` と自作モジュールを読めるようにする。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import Agent, make_heuristic_agent, random_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def read_deck(path: str) -> list[int]:
    """deck.csv（1 行 1 カード ID、60 行）を読み込んでカード ID のリストを返す."""
    with open(path) as f:
        ids = [int(line) for line in f.read().splitlines() if line.strip()]
    if len(ids) != 60:
        raise ValueError(f"デッキは 60 枚である必要があります（実際: {len(ids)} 枚）")
    return ids


def play_match(
    agent0: Agent,
    agent1: Agent,
    deck0: list[int],
    deck1: list[int],
    rng: random.Random,
    max_steps: int = 100_000,
) -> dict:
    """1 試合を agent0（席0）vs agent1（席1）で実行し、結果を返す.

    Returns:
        dict: result（勝者席 index 0/1, 引き分け 2, 不明 None）, turn, steps。
    """
    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(
            f"battle_start に失敗（errorPlayer={start.errorPlayer}, errorType={start.errorType}）"
        )

    agents = (agent0, agent1)
    try:
        for steps in range(1, max_steps + 1):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                return {
                    "result": obs.current.result,
                    "turn": obs.current.turn,
                    "steps": steps,
                }
            if obs.select is None:
                return {
                    "result": None,
                    "turn": getattr(obs.current, "turn", None),
                    "steps": steps,
                }
            who = obs.current.yourIndex if obs.current is not None else 0
            obs_dict = battle_select(agents[who](obs, rng))
        raise RuntimeError(f"max_steps={max_steps} に到達（無限ループの疑い）")
    finally:
        battle_finish()


def evaluate(
    agent_a: Agent,
    agent_b: Agent,
    deck: list[int],
    rng: random.Random,
    games: int,
    alternate: bool = True,
) -> dict:
    """agent_a と agent_b を `games` 試合対戦させ、A 視点の勝敗を集計する.

    alternate=True のとき、試合ごとに席を入れ替えて先手有利を打ち消す。
    """
    wins_a = wins_b = draws = 0
    for g in range(games):
        a_seat1 = alternate and (g % 2 == 1)  # 奇数試合は A を後手（席1）に
        if a_seat1:
            out = play_match(agent_b, agent_a, deck, deck, rng)
            winner_is_a = out["result"] == 1
        else:
            out = play_match(agent_a, agent_b, deck, deck, rng)
            winner_is_a = out["result"] == 0

        if out["result"] is None or out["result"] == 2:
            draws += 1
        elif winner_is_a:
            wins_a += 1
        else:
            wins_b += 1

    decided = wins_a + wins_b
    win_rate_a = wins_a / decided if decided else float("nan")
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": win_rate_a,
    }


def evaluate_decks(
    deck_a: list[int],
    deck_b: list[int],
    agent: Agent,
    rng: random.Random,
    games: int,
    alternate: bool = True,
) -> dict:
    """同一エージェントで deck_a と deck_b を対戦させ、deck_a 視点の勝敗を集計する.

    デッキ最適化の適応度に使う（操縦者は共通の方策。デッキの強さだけを比較）。
    alternate=True のとき席を入れ替えて先手有利を打ち消す。
    """
    wins_a = wins_b = draws = 0
    for g in range(games):
        a_seat1 = alternate and (g % 2 == 1)
        if a_seat1:
            out = play_match(agent, agent, deck_b, deck_a, rng)
            a_seat = 1
        else:
            out = play_match(agent, agent, deck_a, deck_b, rng)
            a_seat = 0

        if out["result"] is None or out["result"] == 2:
            draws += 1
        elif out["result"] == a_seat:
            wins_a += 1
        else:
            wins_b += 1

    decided = wins_a + wins_b
    win_rate_a = wins_a / decided if decided else float("nan")
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": win_rate_a,
    }


def _build_agent(
    name: str,
    meta,
    deck: list[int],
    time_budget: float,
    game_budget: float | None = None,
) -> Agent:
    """エージェント名から実体を生成する."""
    if name == "random":
        return random_agent
    if name == "heuristic":
        return make_heuristic_agent(meta)
    if name == "ismcts":
        from ismcts import make_ismcts_agent

        return make_ismcts_agent(
            meta, deck, deck, time_budget=time_budget, game_budget=game_budget
        )
    raise ValueError(f"未知のエージェント: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="cabt Engine 自己対戦ハーネス")
    parser.add_argument(
        "--deck", default="data/deck.csv", help="両プレイヤー共通のデッキ CSV"
    )
    agent_choices = ["heuristic", "random", "ismcts"]
    parser.add_argument(
        "--a", default="heuristic", choices=agent_choices, help="エージェント A"
    )
    parser.add_argument(
        "--b", default="random", choices=agent_choices, help="エージェント B"
    )
    parser.add_argument("--games", type=int, default=100, help="試合数")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=0.5,
        help="ISMCTS の1手あたり思考時間（秒・game-budget 未指定時）",
    )
    parser.add_argument(
        "--game-budget",
        type=float,
        default=None,
        help="ISMCTS の1試合の累積持ち時間（秒）。指定すると残り時間から動的配分（本大会は600）",
    )
    parser.add_argument(
        "--no-alternate", action="store_true", help="席の入れ替えを無効化"
    )
    args = parser.parse_args()

    deck = read_deck(args.deck)
    rng = random.Random(args.seed)
    meta = load_card_meta()

    agent_a = _build_agent(args.a, meta, deck, args.time_budget, args.game_budget)
    agent_b = _build_agent(args.b, meta, deck, args.time_budget, args.game_budget)

    res = evaluate(
        agent_a, agent_b, deck, rng, args.games, alternate=not args.no_alternate
    )

    print(f"A={args.a} vs B={args.b}  ({args.games} games, seed={args.seed})")
    print(f"  A wins: {res['wins_a']}  B wins: {res['wins_b']}  draws: {res['draws']}")
    print(f"  A win rate (decided games): {res['win_rate_a']:.3f}")


if __name__ == "__main__":
    main()
