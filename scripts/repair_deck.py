"""デッキ構成の機械修復（実メタ分布への整形）.

league が「エネ33・グッズ5(2種)」のような実メタ分布の外れ値に収束していた問題への
一発修復ツール。中核ロジックは [deck.repair_composition](../src/deck.py)（league の
候補射影と共用）。本スクリプトは**目標枚数ちょうど**（lo=hi）へ整形する点だけが違う
（league 側は許容帯 COMP_BOUNDS 内なら無変更＝探索の自由を残す）。

出力はカード ID の CSV（Pokémon Elements は扱わない）。gate で検証してから採用する:
    python scripts/repair_deck.py --deck models/champion_best.csv --out models/champion_repaired.csv
    make champion-gate GATE_ARGS="--new models/champion_repaired.csv"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from deck import card_category, repair_composition, save_deck  # noqa: E402

# 実メタ70デッキ平均を丸めた目標枚数（--targets で上書き可）
DEFAULT_TARGETS = {"poke": 15, "item": 14, "sup": 11, "stad": 2, "tool": 1, "ene": 17}


def main() -> None:
    p = argparse.ArgumentParser(description="デッキ構成を実メタ分布へ機械修復")
    p.add_argument("--deck", default="models/champion_best.csv")
    p.add_argument("--out", default="models/champion_repaired.csv")
    p.add_argument(
        "--staples-dir",
        default="data/replays/opp_decks",
        help="補充カードの頻度を取る実メタデッキ群",
    )
    p.add_argument(
        "--targets", default=None, help='目標枚数の上書き（例 "ene=16,item=15"）'
    )
    args = p.parse_args()

    meta = load_card_meta()
    targets = dict(DEFAULT_TARGETS)
    if args.targets:
        for kv in args.targets.split(","):
            k, v = kv.split("=")
            targets[k.strip()] = int(v)

    deck = [int(x) for x in open(args.deck).read().split()]
    staple_freq: Counter[int] = Counter()
    for path in glob.glob(os.path.join(args.staples_dir, "*.csv")):
        staple_freq.update({int(x) for x in open(path).read().split()})

    # 目標ちょうど（lo=hi）の帯として共通射影を使う
    bounds = {cat: (t, t) for cat, t in targets.items()}
    repaired = repair_composition(deck, meta, staple_freq, bounds=bounds)
    if repaired == deck:
        print("変更なし（既に目標構成／または射影失敗で安全退化）")
    save_deck(repaired, args.out)

    def comp(ids) -> str:
        cc = Counter(card_category(meta, c) for c in ids)
        return " ".join(
            f"{k}:{cc.get(k, 0)}"
            for k in ("poke", "item", "sup", "stad", "tool", "ene")
        )

    print(f"修復前: {comp(deck)}")
    print(f"修復後: {comp(repaired)} → {args.out}")
    print('検証: make champion-gate GATE_ARGS="--new ' + args.out + '"')


if __name__ == "__main__":
    main()
