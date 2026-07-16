"""replay コーパスに登場する**全チーム**の操縦・デッキを一括スカウトする（§41・1パス集計）.

episode には両席の観測・行動が入っているので、DL した対局の相手側もすべて解析できる。
出すもの（チーム別・ID/数値/カテゴリのみ）:
  試合数・勝率・主デッキ（ハッシュ＋ロール構成）・思考時間・特性/試合・初攻撃ターン・
  リサイクル率・敗北時獲得サイド（変換力）
モード:
  --min-games N       : 集計に必要な最少試合数（既定5）
  --vs TEAM --result loss : 「TEAM が負けた試合の相手側」だけを集計（例: 1位の敗因研究）

    python scripts/scout_field.py                          # 全チーム表
    python scripts/scout_field.py --vs "TeamA" --result loss   # 1位を倒した者たち
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
from cg.api import OptionType, SelectType  # noqa: E402
from deck import card_category  # noqa: E402

_ACCEL_BIT = 1 << EFFECT_CATEGORIES.index("energy_accel")


def _med(v):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[len(v) // 2]


def _deck_hash(deck):
    return hashlib.md5(",".join(map(str, sorted(deck))).encode()).hexdigest()[:8]


def _comp(deck, meta):
    cat = Counter(card_category(meta, c) for c in deck)
    accel = sum(1 for c in deck if meta.ability_effect.get(c, 0) & _ACCEL_BIT)
    return (
        " ".join(f"{k}{cat.get(k, 0)}" for k in ("poke", "item", "sup", "stad", "tool", "ene"))
        + f" 加速{accel}"
    )


def scout(paths, meta, vs=None, vs_result=None):
    """1パスで全チームの行動指紋を集計する.

    流れ: 1. episode ごとに（--vs 指定時は対象試合だけに絞り）両席を走査
    → 2. 席ごとに 勝敗/デッキ/思考時間/特性回数/初攻撃/リサイクル/敗北時サイド を集計
    → 3. チーム名 -> 統計 dict を返す。
    """
    T = defaultdict(lambda: {
        "games": 0, "wins": 0, "decks": Counter(), "deck_ex": {},
        "think": [], "ability": 0, "first_atk": [], "recycle": 0,
        "loss_prize": [],
    })
    for path in paths:
        with open(path) as f:
            ep = json.load(f)
        names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
        rew = ep.get("rewards") or [None, None]
        if rew[0] is None or rew[1] is None or names[0] == names[1]:
            continue
        # --vs フィルタ: 指定チームが指定結果（loss 等）だった試合の**相手側**だけを見る
        if vs is not None:
            if vs not in names:
                continue
            vseat = names.index(vs)
            v_res = "win" if rew[vseat] > rew[1 - vseat] else "loss"
            if vs_result and v_res != vs_result:
                continue
        steps = ep.get("steps") or []
        for seat in (0, 1):
            team = names[seat]
            if vs is not None and team == vs:
                continue  # 対象チーム自身は除外（相手側だけ集計）
            st = T[team]
            st["games"] += 1
            won = rew[seat] > rew[1 - seat]
            if won:
                st["wins"] += 1
            fa = None
            prev_dc = None
            jumped = False
            min_ov = None
            last_me = None
            for i, step in enumerate(steps):
                rec = step[seat]
                obs_d = rec.get("observation") or {}
                ov = obs_d.get("remainingOverageTime")
                if ov is not None:
                    min_ov = ov if min_ov is None else min(min_ov, ov)
                cur = obs_d.get("current") or {}
                players = cur.get("players")
                if players:
                    me = players[cur.get("yourIndex", seat)]
                    last_me = me
                    dc = me.get("deckCount")
                    if dc is not None:
                        if prev_dc is not None and dc - prev_dc >= 5:
                            jumped = True
                        prev_dc = dc
                    # デッキ採取（初手 action は _deck 用に別扱い）
                if rec.get("status") != "ACTIVE" or i + 1 >= len(steps):
                    continue
                act = steps[i + 1][seat].get("action")
                sel = obs_d.get("select") or {}
                opts = sel.get("option") or []
                if (
                    isinstance(act, list)
                    and len(act) == 1
                    and sel.get("type") == SelectType.MAIN
                    and len(opts) > 1
                    and 0 <= act[0] < len(opts)
                ):
                    t = opts[act[0]].get("type")
                    if t == OptionType.ABILITY:
                        st["ability"] += 1
                    if t == OptionType.ATTACK and fa is None:
                        fa = cur.get("turn") or 0
            # デッキ（最初の60枚 action）
            for step in steps[:3]:
                a = step[seat].get("action")
                if isinstance(a, list) and len(a) == 60:
                    h = _deck_hash(a)
                    st["decks"][h] += 1
                    st["deck_ex"].setdefault(h, [int(x) for x in a])
                    break
            if fa:
                st["first_atk"].append(fa)
            if jumped:
                st["recycle"] += 1
            if min_ov is not None:
                st["think"].append(600.0 - float(min_ov))
            if not won and last_me is not None:
                st["loss_prize"].append(6 - len(last_me.get("prize") or []))
    return T


def main() -> None:
    p = argparse.ArgumentParser(description="全チーム一括スカウト")
    p.add_argument("--dir", default="data/replays/others")
    p.add_argument("--min-games", type=int, default=5)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--vs", default=None, help="このチームの対戦相手だけを集計")
    p.add_argument("--result", default=None, choices=[None, "win", "loss"],
                   help="--vs チームがこの結果だった試合に絞る（loss=そのチームを倒した側）")
    args = p.parse_args()
    meta = load_card_meta()
    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    if not paths:
        raise SystemExit(f"JSON がありません: {args.dir}")
    T = scout(paths, meta, vs=args.vs, vs_result=args.result)
    title = f"--vs {args.vs}({args.result})" if args.vs else "全チーム"
    print(f"=== フィールドスカウト {title}: {len(paths)} episode・{len(T)} チーム ===")
    rows = [(s["wins"] / s["games"], t, s) for t, s in T.items() if s["games"] >= args.min_games]
    for wr, t, s in sorted(rows, reverse=True)[: args.top]:
        g = s["games"]
        h, cnt = (s["decks"].most_common(1) or [("?", 0)])[0]
        comp = _comp(s["deck_ex"][h], meta) if h in s["deck_ex"] else "?"
        print(
            f"{t[:20]:20s} 勝率{wr:.2f}({g:3d}戦) 思考{_med(s['think']):4.0f}s "
            f"特性{s['ability']/g:4.1f}/戦 初攻T{_med(s['first_atk']):.0f} "
            f"リサイクル{s['recycle']/g:3.0%} 敗時サイド{_med(s['loss_prize']):.0f}"
        )
        print(f"{'':20s} └ デッキ {h}×{cnt}: {comp}")


if __name__ == "__main__":
    main()
