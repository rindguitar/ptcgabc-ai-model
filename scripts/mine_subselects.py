"""指定チームの replay から**サブ選択のペース配分**をマイニングする（§39・汎用）.

§38 の診断: デッキ切れ負け（本人4% vs 我々52%）の真犯人は MAIN でなく**サブ選択の掘りすぎ**
（COUNT=常に最大・サーチ=常に maxCount という貪欲固定）。本スクリプトは教師チームの
①COUNT 選択（何枚引くか等）②山札サーチ（何枚取るか）を**残デッキ数の帯別**に集計し、
「本人は残り X 枚帯で許容量の何割を使うか」のペース表を出す（_generic_select 注入の設計材料）。

出力はすべて数値・ID まで（Pokémon Elements 非表示・規約遵守）。

    python scripts/mine_subselects.py --team <TeamName> [--dir data/replays/others]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cg.api import SelectType, to_observation_class  # noqa: E402

# 残デッキ数の帯（この帯別に「許容量の何割使うか」を出す）
_BUCKETS = [(0, 9), (10, 19), (20, 29), (30, 60)]


def _bucket(deck_count: int) -> str:
    for lo, hi in _BUCKETS:
        if lo <= deck_count <= hi:
            return f"残{lo:02d}-{hi:02d}"
    return "残??"


def mine(paths: list[str], team: str) -> dict:
    """episode 群から教師席のサブ選択を集計する.

    流れ: 1. status==ACTIVE の step の select を見る → 2. 応答は次 step の action（§33 検証済み）
    → 3. COUNT は「選んだ数値/最大数値」、サーチ系は「取った枚数/maxCount」を残デッキ帯別に集計。
    """
    count_ratio = defaultdict(list)  # 帯 -> [選択値/最大値]
    search_ratio = defaultdict(list)  # 帯 -> [取得枚数/maxCount]
    for path in paths:
        with open(path) as f:
            ep = json.load(f)
        names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
        steps = ep.get("steps") or []
        for seat in (0, 1):
            if names[seat] != team:
                continue
            for i, step in enumerate(steps):
                rec = step[seat]
                if rec.get("status") != "ACTIVE" or not rec.get("observation"):
                    continue
                if i + 1 >= len(steps):
                    continue
                act = steps[i + 1][seat].get("action")
                if not isinstance(act, list) or not act:
                    continue
                obs = to_observation_class(rec["observation"])
                st, sel = obs.current, obs.select
                if st is None or sel is None:
                    continue
                me = st.players[st.yourIndex]
                b = _bucket(me.deckCount or 0)
                if sel.type == SelectType.COUNT and len(sel.option) > 1:
                    # 数値選択: 選んだ数値 / 最大数値（引き数の抑制が見える）
                    nums = [o.number or 0 for o in sel.option]
                    mx = max(nums)
                    if 0 <= act[0] < len(nums) and mx > 0:
                        count_ratio[b].append(nums[act[0]] / mx)
                elif sel.deck is not None and sel.maxCount > 1:
                    # 山札サーチ（複数取れる時）: 取った枚数 / maxCount
                    search_ratio[b].append(len(act) / sel.maxCount)
    return {"count": count_ratio, "search": search_ratio}


def main() -> None:
    p = argparse.ArgumentParser(description="サブ選択のペース配分マイニング")
    p.add_argument("--team", required=True)
    p.add_argument("--dir", default="data/replays/others")
    args = p.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    if not paths:
        raise SystemExit(f"JSON がありません: {args.dir}")
    r = mine(paths, args.team)
    print(f"=== {args.team} のサブ選択ペース（episode {len(paths)} 件走査）===")
    for kind, label in (("count", "COUNT（引く数など/最大）"), ("search", "山札サーチ（取得/max）")):
        print(f"--- {label} ---")
        for b in sorted(r[kind]):
            v = r[kind][b]
            v.sort()
            med = v[len(v) // 2]
            full = sum(1 for x in v if x >= 0.999) / len(v)
            print(f"  {b}: n={len(v):4d} 使用率中央値 {med:.2f} / 全量使用率 {full:.0%}")
    print("\n※ 我々の _generic_select は常に 1.00（全量）＝この表との差が「掘りすぎ」の定量値")


if __name__ == "__main__":
    main()
