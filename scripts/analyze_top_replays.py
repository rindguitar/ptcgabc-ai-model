"""リーダーボード上位チームの replay 分析（チーム別の成績・デッキ・時間の使い方）.

リーダーボードから DL した**他チーム同士の episode JSON** を `data/replays/others/` に置いて実行。
（同フォルダは `make replays` の収穫対象でもある＝デッキは opp_decks/ にも溜まる）

出すもの（すべてカード ID・数値・カテゴリまで＝Pokémon Elements は出さない・規約遵守）:
  - チーム別: 試合数・勝率・使用デッキ（ハッシュ）・思考時間の使い方（overage 消費）
  - 上位チームのデッキ構成サマリ（poke/item/sup/…枚数・energy_accel 枚数・ユニーク種数）
  - そのデッキが opp_decks/ にあるパス（→ そのまま `make eval-deck EVAL_DECK=...` で検証可能）

    python scripts/analyze_top_replays.py            # make top-replays
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import EFFECT_CATEGORIES, load_card_meta  # noqa: E402
from deck import card_category  # noqa: E402

_ACCEL_BIT = 1 << EFFECT_CATEGORIES.index("energy_accel")


def _deck_of(ep: dict, seat: int) -> list[int] | None:
    for step in (ep.get("steps") or [])[:3]:
        a = step[seat].get("action")
        if isinstance(a, list) and len(a) == 60:
            return [int(x) for x in a]
    return None


def _deck_hash(deck: list[int]) -> str:
    return hashlib.md5(str(sorted(deck)).encode()).hexdigest()[:8]


def _min_overage(ep: dict, seat: int) -> float | None:
    vals = [
        (step[seat].get("observation") or {}).get("remainingOverageTime")
        for step in (ep.get("steps") or [])
    ]
    vals = [float(v) for v in vals if v is not None]
    return min(vals) if vals else None


def _result(ep: dict, seat: int) -> str | None:
    r = ep.get("rewards") or [None, None]
    a, b = r[seat], r[1 - seat]
    if a is None or b is None:
        return None
    return "win" if a > b else "loss" if a < b else "draw"


def _comp_summary(deck: list[int], meta) -> str:
    cat = Counter(card_category(meta, c) for c in deck)
    accel = sum(1 for c in deck if meta.ability_effect.get(c, 0) & _ACCEL_BIT)
    kinds = len(set(deck))
    cats = " ".join(f"{k}{cat.get(k, 0)}" for k in ("poke", "item", "sup", "stad", "tool", "ene"))
    return f"{cats} 加速{accel} 種{kinds}"


def main() -> None:
    p = argparse.ArgumentParser(description="上位チーム replay の分析")
    p.add_argument("--dir", default="data/replays/others", help="episode JSON の場所")
    p.add_argument("--top", type=int, default=8, help="表示するチーム数")
    p.add_argument("--min-games", type=int, default=3, help="集計に必要な最少試合数")
    p.add_argument("--decks-dir", default="data/replays/opp_decks")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    if not paths:
        raise SystemExit(f"JSON がありません: {args.dir}（リーダーボードから DL して配置）")

    meta = load_card_meta()
    rec = defaultdict(Counter)  # team -> {win/loss/draw}
    decks = defaultdict(Counter)  # team -> {deck_hash: n}
    deck_by_hash: dict[str, list[int]] = {}
    time_used = defaultdict(list)  # team -> [600-残overage]
    for pth in paths:
        with open(pth) as f:
            ep = json.load(f)
        names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
        for seat in (0, 1):
            t = names[seat]
            r = _result(ep, seat)
            if r:
                rec[t][r] += 1
            d = _deck_of(ep, seat)
            if d:
                h = _deck_hash(d)
                decks[t][h] += 1
                deck_by_hash.setdefault(h, d)
            ov = _min_overage(ep, seat)
            if ov is not None:
                time_used[t].append(600.0 - ov)

    ranked = sorted(
        (
            (t, c["win"] / max(1, sum(c.values())), sum(c.values()))
            for t, c in rec.items()
            if sum(c.values()) >= args.min_games
        ),
        key=lambda x: -x[1],
    )
    print(f"episode {len(paths)} 件・チーム {len(rec)}（{args.min_games}試合以上を勝率順）\n")
    for t, wr, n in ranked[: args.top]:
        used = sorted(time_used.get(t, []))
        med = used[len(used) // 2] if used else float("nan")
        print(f"=== {t}: 勝率 {wr:.3f}（{n}試合）思考時間中央値 ≈ {med:.0f}s ===")
        for h, cnt in decks[t].most_common(3):
            d = deck_by_hash[h]
            path = os.path.join(args.decks_dir, f"opp_{h}.csv")
            mark = path if os.path.exists(path) else "(opp_decks 未収穫 → make replays)"
            print(f"  デッキ {h} ×{cnt}: {_comp_summary(d, meta)}")
            print(f"    → {mark}")
    print(
        "\n検証: make eval-deck EVAL_DECK=<上記CSV> / 昇格: cp → champion_deck → make champion-gate"
    )


if __name__ == "__main__":
    main()
