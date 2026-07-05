"""提出エージェントの煙テスト＋時間計測（600 秒クロック検証・Phase 3）.

make_kaggle_agent を公式形式（agent(obs_dict)）のまま実際に対戦させ、
1 試合あたりの決定数・合計消費時間・1 手あたりの平均/最大時間を測る。
本大会は 1 プレイヤー 600 秒/試合の累積クロックなので、合計が game_budget(540s) に
収まるかを提出前に確認する（特に --policy nn は torch 推論＋floor rollout の実測が必要）。

実行（nn は torch が要るので Docker）:
    make exec CMD="python scripts/smoke_submission.py --policy nn --net models/pvnet_distill_best.pt"
    python scripts/smoke_submission.py --policy ismcts   # ismcts はホストでも可
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from deck import load_deck  # noqa: E402
from submission import make_kaggle_agent  # noqa: E402


def _default_deck() -> str:
    """champion_best（提出に使う最良）を優先し、無ければ champion_deck."""
    for p in ("models/champion_best.csv", "models/champion_deck.csv"):
        if os.path.exists(p):
            return p
    return "data/deck.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="提出エージェントの煙テスト＋時間計測")
    p.add_argument("--policy", choices=["ismcts", "nn"], default="nn")
    p.add_argument("--net", default=None, help="--policy nn の学習済み PVNet (.pt)")
    p.add_argument("--deck", default=None, help="デッキ CSV（既定 champion_best）")
    p.add_argument("--games", type=int, default=2, help="計測する試合数")
    p.add_argument("--floor-rollouts", type=int, default=8)
    p.add_argument(
        "--board-bonus", type=float, default=0.2, help="nn の盤面補正 α（提出と同じ）"
    )
    p.add_argument("--game-budget", type=float, default=540.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = load_card_meta()
    deck_path = args.deck or _default_deck()
    deck = load_deck(deck_path)
    kwargs = {}
    if args.policy == "nn":
        if not (args.net and os.path.exists(args.net)):
            raise SystemExit(f"--policy nn には --net が必要: {args.net}")
        kwargs = {
            "net_path": args.net,
            "floor_rollouts": args.floor_rollouts,
            "board_bonus": args.board_bonus,
        }
    agent = make_kaggle_agent(
        args.policy,
        deck=deck,
        meta=meta,
        seed=args.seed,
        game_budget=args.game_budget,
        **kwargs,
    )
    heuristic = make_heuristic_agent(meta)
    rng = random.Random(args.seed)

    print(
        f"=== 煙テスト: policy={args.policy}"
        + (f" net={os.path.basename(args.net)}" if args.net else "")
        + f" deck={os.path.basename(deck_path)} ==="
    )
    worst_total = 0.0
    for g in range(args.games):
        obs_dict, _ = battle_start(deck, deck)
        times: list[float] = []
        result = None
        try:
            for _ in range(100_000):
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    result = obs.current.result
                    break
                if obs.select is None:
                    break
                if obs.current.yourIndex == 0:
                    t0 = time.perf_counter()
                    action = agent(obs_dict)  # 公式形式（生 dict を渡す）
                    times.append(time.perf_counter() - t0)
                else:
                    action = heuristic(obs, rng)
                obs_dict = battle_select(action)
        finally:
            battle_finish()
        total = sum(times)
        worst_total = max(worst_total, total)
        mx = max(times) if times else 0.0
        mean = total / len(times) if times else 0.0
        print(
            f"game {g}: 決定 {len(times)} 手 / 合計 {total:.1f}s / "
            f"平均 {mean * 1000:.0f}ms / 最大 {mx * 1000:.0f}ms / result={result}"
        )
    verdict = "OK" if worst_total < args.game_budget else "NG（予算超過）"
    print(
        f"\n判定: 最悪試合 {worst_total:.1f}s < 予算 {args.game_budget:.0f}s → {verdict}"
    )
    print("  ※ 提出環境の CPU はローカルより遅い可能性あり。2倍程度の余裕を推奨。")


if __name__ == "__main__":
    main()
