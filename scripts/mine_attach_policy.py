"""上位帯 replay からエネ付与先の方策 P(ベンチ付与 | アクティブの装着エネ枚数) を採掘する.

背景（行動差分の実測）: 上位帯はアーキタイプ不問でベンチへ 4.3〜5.0 回/戦エネを回すのに
我々は 1.2 回（アクティブに 8 割集中）。アクティブ KO と同時に全エネを失い後続が立たず、
敗時サイド 0〜1 のワンサイド負けを量産していた。この「付与先の切替点」を帯全体の実測から
数値として取り出し、heuristic のエネ付与判断（agents.py）へ注入する。

流れ: 1. 各 episode の MAIN 決定を走査（ACTIVE 席の select ← 次 step の action・§45）
     → 2. ATTACH の選択肢に active/bench 両方があった局面だけを対象に、
          「アクティブの装着エネ枚数（バケット 0..3+）→ 選んだ付与先」を集計
     → 3. バケット別 P(ベンチ) とチーム別内訳を JSON へ出力（検証可能性のため併記）

    python scripts/mine_attach_policy.py --dirs data/replays/others --out data/attach_priors.json
    python scripts/mine_attach_policy.py --dirs DIR1 DIR2 --teams "TeamC,TeamD"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cg.api import AreaType, OptionType, SelectType  # noqa: E402

# アクティブの装着エネ枚数バケットの上限（これ以上は同一視。上位帯でも 3 枚超は稀）
MAX_BUCKET = 3


def mine_episode(path: str, counts: dict, teams: set[str] | None) -> None:
    """1 episode を走査して counts[team][bucket] = [bench回数, 総回数] に加算する."""
    with open(path) as f:
        ep = json.load(f)
    names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
    steps = ep.get("steps") or []
    for seat in (0, 1):
        team = names[seat]
        if teams is not None and team not in teams:
            continue
        for i, step in enumerate(steps):
            rec = step[seat]
            if rec.get("status") != "ACTIVE" or i + 1 >= len(steps):
                continue
            obs_d = rec.get("observation") or {}
            sel = obs_d.get("select") or {}
            if sel.get("type") != SelectType.MAIN:
                continue
            opts = sel.get("option") or []
            act = steps[i + 1][seat].get("action")
            if not isinstance(act, list) or len(act) != 1:
                continue
            if not (0 <= act[0] < len(opts)):
                continue
            # 付与先の「選択」が存在した局面のみ対象（active/bench 両方が合法だった時）
            areas = {
                o.get("inPlayArea")
                for o in opts
                if o.get("type") == OptionType.ATTACH
            }
            if not (AreaType.ACTIVE in areas and AreaType.BENCH in areas):
                continue
            chosen = opts[act[0]]
            if chosen.get("type") != OptionType.ATTACH:
                continue
            # アクティブの装着エネ枚数 → バケット
            cur = obs_d.get("current") or {}
            players = cur.get("players") or []
            if not players:
                continue
            me = players[cur.get("yourIndex", seat)]
            active = (me.get("active") or [None])[0]
            act_e = len((active or {}).get("energyCards") or [])
            bucket = min(act_e, MAX_BUCKET)
            cell = counts[team][bucket]
            cell[1] += 1
            if chosen.get("inPlayArea") == AreaType.BENCH:
                cell[0] += 1


def main() -> None:
    p = argparse.ArgumentParser(description="エネ付与先方策の採掘（帯共通の普遍傾向）")
    p.add_argument("--dirs", nargs="+", default=["data/replays/others"])
    p.add_argument("--teams", default=None, help="対象チームのカンマ区切り（既定: 全席）")
    p.add_argument("--out", default="data/attach_priors.json")
    p.add_argument("--min-n", type=int, default=30, help="バケット採用に必要な標本数")
    args = p.parse_args()

    teams = set(t.strip() for t in args.teams.split(",")) if args.teams else None
    counts: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    n_ep = 0
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                mine_episode(path, counts, teams)
                n_ep += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    # 集約: 全対象席の合算で p_bench を出し、チーム別は検証用に併記
    total = defaultdict(lambda: [0, 0])
    for team_counts in counts.values():
        for b, (bench, n) in team_counts.items():
            total[b][0] += bench
            total[b][1] += n

    priors = {}
    print(f"=== attach 方策採掘: {n_ep} episode・{len(counts)} チーム ===")
    print(f"{'アクティブのエネ枚数':<12}{'P(ベンチ)':>10}{'n':>8}")
    for b in sorted(total):
        bench, n = total[b]
        label = f"{b}+" if b == MAX_BUCKET else str(b)
        mark = ""
        if n >= args.min_n:
            priors[str(b)] = round(bench / n, 3)
        else:
            mark = "（n不足・採用せず）"
        print(f"{label:<16}{bench / max(n, 1):>10.2f}{n:>8}{mark}")

    out = {
        "priors": priors,
        "n": {str(b): total[b][1] for b in sorted(total)},
        "teams": {
            t: {
                str(b): [c[0], c[1]]
                for b, c in sorted(tc.items())
            }
            for t, tc in sorted(counts.items())
            if sum(c[1] for c in tc.values()) >= 10
        },
        "source_dirs": args.dirs,
        "source_teams": sorted(teams) if teams else "all",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"→ {args.out} に保存（priors キー: {sorted(priors)}）")


if __name__ == "__main__":
    main()
