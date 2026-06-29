"""チャンピオンの信頼ラチェット（keep-best）.

league の内部評価は ISMCTS が遅く小サンプル（6試合）で、最悪ケースのノイズにより
「より弱いデッキへ更新」してしまう（実測: 旧 worst 0.45 → 新 0.35）。本スクリプトは
league 出力（--new）と現状ベスト（--best）を**多めの試合数でメタに対し評価**し、
**新が確実に上回った時だけ best を更新**する。これを毎 league 後に挟めば、回し続けるほど
**単調に良くなる**（ノイズドリフトを止める）。

評価基準は league と同じ「最悪ケース（worst）」を主、平均（mean）を tie-break に使う。
best が無ければ new をそのまま採用（初回）。ホスト(CPU)で動く。例:
    python scripts/champion_gate.py --games 20
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from deckopt import _load_pool, default_opponent_paths  # noqa: E402
from eval_deck import eval_deck_vs_meta  # noqa: E402


def _better(new: dict, best: dict, margin: float) -> bool:
    """new が best より良いか。最悪ケース優先＋平均 tie-break。margin で僅差更新を防ぐ."""
    if new["worst"] > best["worst"] + margin:
        return True
    if new["worst"] < best["worst"] - margin:
        return False
    return new["mean"] > best["mean"] + margin  # 最悪が同程度なら平均で判定


def main() -> None:
    p = argparse.ArgumentParser(description="チャンピオンの信頼ラチェット（keep-best）")
    p.add_argument(
        "--new", default="models/champion_deck.csv", help="league 出力デッキ"
    )
    p.add_argument("--best", default="models/champion_best.csv", help="現状ベスト")
    p.add_argument("--meta", nargs="+", default=None, help="相手メタ群")
    p.add_argument("--games", type=int, default=20, help="相手1体あたりの試合数")
    p.add_argument(
        "--pilot",
        choices=["ismcts", "nn", "heuristic"],
        default="ismcts",
        help="判定操縦（ismcts=独立判定で頑健 / nn=蒸留NN-MCTS高速・要torch）",
    )
    p.add_argument("--time-budget", type=float, default=0.1, help="ismcts の1手秒")
    p.add_argument(
        "--net",
        default="models/pvnet_distill_best.pt",
        help="pilot=nn のとき使う訓練済みネット",
    )
    p.add_argument(
        "--nn-sims", type=int, default=64, help="pilot=nn の1手あたり MCTS 反復数"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="この差を超えないと更新しない（僅差の入れ替わり抑制）",
    )
    args = p.parse_args()

    new_deck = load_deck(args.new)
    # 初回（best 無し）は new を採用
    if not os.path.exists(args.best):
        shutil.copyfile(args.new, args.best)
        print(f"初回: {args.new} を {args.best} に採用（以後ここが基準）")
        return

    meta = load_card_meta()
    meta_paths = args.meta or default_opponent_paths()
    pool = _load_pool(meta_paths)
    best_deck = load_deck(args.best)
    # 公平比較のため同一相手・同一シードで両者を評価
    opps = [d for d in pool if d != new_deck and d != best_deck]
    if not opps:
        print("評価相手（メタ）が見つかりません")
        return

    new_res = eval_deck_vs_meta(
        new_deck,
        meta,
        opps,
        random.Random(args.seed),
        args.games,
        args.pilot,
        args.time_budget,
        net=args.net,
        nn_sims=args.nn_sims,
    )
    best_res = eval_deck_vs_meta(
        best_deck,
        meta,
        opps,
        random.Random(args.seed),
        args.games,
        args.pilot,
        args.time_budget,
        net=args.net,
        nn_sims=args.nn_sims,
    )
    print(f"new : 最悪={new_res['worst']:.3f} 平均={new_res['mean']:.3f}")
    print(f"best: 最悪={best_res['worst']:.3f} 平均={best_res['mean']:.3f}")

    if _better(new_res, best_res, args.margin):
        shutil.copyfile(args.new, args.best)
        print(f"→ 更新: new が上回ったので {args.best} を置換（ラチェット前進）")
    else:
        print("→ 据え置き: new は best を上回らず（ノイズドリフトを阻止）")


if __name__ == "__main__":
    main()
