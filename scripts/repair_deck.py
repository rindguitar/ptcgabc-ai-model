"""デッキ構成の機械修復（実メタ分布への整形）.

league が「エネ33・グッズ5(2種)」のような実メタ分布の外れ値に収束していた問題
（heuristic ロールアウトがトレーナーを使わない＝グッズの価値を系統的に過小評価する
近親交配の歪み）への対処。指定デッキをカテゴリ目標枚数（既定＝実メタ70デッキの平均に
丸め）へ整形する:

  1. 過剰カテゴリ（エネ・スタジアム等）を目標まで削る（同一 cardId の多い順に削減）
  2. 不足カテゴリを補充する。補充カードは **data/replays/opp_decks/ の採用頻度上位**
     （＝実際の環境で選ばれている汎用札）から、4枚制限・aceSpec 1枚制限を守って選ぶ。
     ポケモンは元デッキの既存たねの増量を優先（タイプ整合を壊さない）。

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
from cg.api import CardType  # noqa: E402
from deck import DECK_SIZE, is_legal, save_deck  # noqa: E402

# カテゴリ別の目標枚数（実メタ70デッキ平均を丸めた値・--targets で上書き可）
DEFAULT_TARGETS = {"poke": 15, "item": 14, "sup": 11, "stad": 2, "tool": 1, "ene": 17}


def _cat(meta, cid: int) -> str:
    t = meta.card_type.get(cid)
    if t == CardType.POKEMON:
        return "poke"
    if t == CardType.ITEM:
        return "item"
    if t == CardType.SUPPORTER:
        return "sup"
    if t == CardType.STADIUM:
        return "stad"
    if t == CardType.TOOL:
        return "tool"
    if t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        return "ene"
    return "other"


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
        "--targets",
        default=None,
        help='目標枚数の上書き（例 "ene=16,item=15"）',
    )
    args = p.parse_args()

    meta = load_card_meta()
    targets = dict(DEFAULT_TARGETS)
    if args.targets:
        for kv in args.targets.split(","):
            k, v = kv.split("=")
            targets[k.strip()] = int(v)

    deck = [int(x) for x in open(args.deck).read().split()]
    counts = Counter(deck)

    # 実メタの採用頻度（デッキ単位: 何デッキが採用しているか）＝汎用札の代理指標
    staple_freq: Counter[int] = Counter()
    for path in glob.glob(os.path.join(args.staples_dir, "*.csv")):
        ids = {int(x) for x in open(path).read().split()}
        staple_freq.update(ids)

    def cat_count(cat: str) -> int:
        return sum(n for c, n in counts.items() if _cat(meta, c) == cat)

    # --- 1. 過剰カテゴリを削る（同一 cardId の多いものから均しつつ削減） ---
    for cat, tgt in targets.items():
        while cat_count(cat) > tgt:
            # 最も枚数の多い cardId から 1 枚削る（種類は残し枚数バランスを崩す順）
            cands = [(n, c) for c, n in counts.items() if _cat(meta, c) == cat]
            _, drop = max(cands)
            counts[drop] -= 1
            if counts[drop] == 0:
                del counts[drop]

    # --- 2. 不足カテゴリを補充する ---
    def n_ace() -> int:  # aceSpec はデッキ1枚制限（トレーナーの is_special ≒ aceSpec）
        return sum(
            n
            for c, n in counts.items()
            if meta.is_special.get(c, False) and _cat(meta, c) != "poke"
        )

    def addable(cid: int) -> bool:
        cat = _cat(meta, cid)
        if cat == "ene" and cid in meta.basic_energy_id.values():
            return True  # 基本エネは枚数無制限
        if counts.get(cid, 0) >= 4:
            return False
        if cat != "poke" and meta.is_special.get(cid, False) and n_ace() >= 1:
            return False
        return True

    def fill(cat: str, tgt: int, pool: list[int]) -> None:
        i = 0
        while cat_count(cat) < tgt and sum(counts.values()) < DECK_SIZE:
            while i < len(pool) and not addable(pool[i]):
                i += 1
            if i >= len(pool):
                break
            counts[pool[i]] += 1

    # ポケモン: 元デッキの既存たねを増量（タイプ整合を守る）→足りなければ実メタ頻度上位のたね
    own_basics = sorted(
        (c for c in counts if _cat(meta, c) == "poke" and meta.is_basic_pokemon(c)),
        key=lambda c: -staple_freq.get(c, 0),
    )
    meta_basics = [
        c
        for c, _ in staple_freq.most_common()
        if _cat(meta, c) == "poke" and meta.is_basic_pokemon(c)
    ]
    fill("poke", targets["poke"], own_basics * 4 + meta_basics)

    # トレーナー系: 実メタ頻度上位から（グッズ→サポ→道具→スタジアム）
    for cat in ("item", "sup", "tool", "stad"):
        pool = [c for c, _ in staple_freq.most_common() if _cat(meta, c) == cat]
        fill(cat, targets[cat], pool)

    # エネ: 元デッキの主要エネ（基本エネ）で埋める
    own_ene = sorted(
        (c for c in counts if _cat(meta, c) == "ene"),
        key=lambda c: -counts[c],
    )
    if own_ene:
        while sum(counts.values()) < DECK_SIZE and cat_count("ene") < targets["ene"]:
            counts[own_ene[0]] += 1

    # 端数が残ったらグッズ頻度上位で 60 枚まで充填
    pool = [c for c, _ in staple_freq.most_common() if _cat(meta, c) in ("item", "sup")]
    i = 0
    while sum(counts.values()) < DECK_SIZE and i < len(pool):
        if addable(pool[i]):
            counts[pool[i]] += 1
        else:
            i += 1

    repaired = [c for c, n in counts.items() for _ in range(n)]
    if len(repaired) != DECK_SIZE:
        raise SystemExit(f"修復失敗: {len(repaired)} 枚（60 枚にできない）")
    if not is_legal(repaired, repaired):  # エンジンに実際に受理されるか（ミラーで検証）
        raise SystemExit("修復失敗: 合法デッキにならない（4枚制限等）")

    save_deck(repaired, args.out)

    # 修復前後の構成を数値で表示（カード名は出さない）
    def comp(ids) -> str:
        cc = Counter(_cat(meta, c) for c in ids)
        return " ".join(
            f"{k}:{cc.get(k, 0)}"
            for k in ("poke", "item", "sup", "stad", "tool", "ene")
        )

    print(f"修復前: {comp(deck)}")
    print(f"修復後: {comp(repaired)} → {args.out}")
    print('検証: make champion-gate GATE_ARGS="--new ' + args.out + '"')


if __name__ == "__main__":
    main()
