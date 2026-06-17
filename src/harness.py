"""Phase 0 自己対戦ハーネス（スモークテスト用）.

cabt Engine（追跡外の提供物 `cg`）を直接ドライブし、2 つのデッキで 1 試合を
最後まで実行できるかを確認する。エージェントは「合法手からランダムに選ぶ」だけの
最小実装。観測・選択・終局判定の取り回しを固めるための足場。

実行例（リポジトリ直下から）:
    python src/harness.py --deck data/deck.csv --games 1 --seed 0
"""

from __future__ import annotations

import argparse
import os
import random
import sys

# このファイルと同階層（src/）を import path に追加し、提供物 `cg` を top-level で読めるようにする。
# （cabt の提供コードは `from cg.api import ...` の形を前提にしているため。）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_start, battle_select  # noqa: E402


def read_deck(path: str) -> list[int]:
    """deck.csv（1 行 1 カード ID、60 行）を読み込んでカード ID のリストを返す."""
    with open(path) as f:
        ids = [int(line) for line in f.read().splitlines() if line.strip()]
    if len(ids) != 60:
        raise ValueError(f"デッキは 60 枚である必要があります（実際: {len(ids)} 枚）")
    return ids


def random_agent(obs: Observation, rng: random.Random) -> list[int]:
    """合法手からランダムに選ぶだけの baseline エージェント.

    返すインデックスは select.option の範囲内で、minCount〜maxCount 個・重複なし。
    """
    sel = obs.select
    count = rng.randint(sel.minCount, sel.maxCount)
    if count == 0:
        return []
    return rng.sample(range(len(sel.option)), count)


def play_one_game(deck0: list[int], deck1: list[int], rng: random.Random,
                  max_steps: int = 100_000) -> dict:
    """1 試合をランダムエージェント同士で最後まで実行し、結果を返す."""
    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(
            f"battle_start に失敗（errorPlayer={start.errorPlayer}, errorType={start.errorType}）"
        )

    steps = 0
    try:
        for steps in range(1, max_steps + 1):
            obs = to_observation_class(obs_dict)
            # 終局判定: current.result は勝者 index（0/1）、引き分けは 2、未終了は -1。
            if obs.current is not None and obs.current.result != -1:
                return {
                    "result": obs.current.result,
                    "turn": obs.current.turn,
                    "steps": steps,
                }
            if obs.select is None:
                # battle_start 経由では発生しない想定。安全のためのガード。
                return {"result": None, "turn": getattr(obs.current, "turn", None), "steps": steps}
            obs_dict = battle_select(random_agent(obs, rng))
        raise RuntimeError(f"max_steps={max_steps} に到達（無限ループの疑い）")
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="cabt Engine 自己対戦スモークテスト")
    parser.add_argument("--deck", default="data/deck.csv", help="両プレイヤー共通のデッキ CSV")
    parser.add_argument("--games", type=int, default=1, help="実行する試合数")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    args = parser.parse_args()

    deck = read_deck(args.deck)
    rng = random.Random(args.seed)

    wins = {0: 0, 1: 0, 2: 0}  # 0/1=各プレイヤー勝利, 2=引き分け
    for g in range(args.games):
        out = play_one_game(deck, deck, rng)
        r = out["result"]
        if r in wins:
            wins[r] += 1
        print(f"game {g}: result={r} turn={out['turn']} steps={out['steps']}")

    print(f"\nsummary over {args.games} games: "
          f"p0_win={wins[0]} p1_win={wins[1]} draw={wins[2]}")


if __name__ == "__main__":
    main()
