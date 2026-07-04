"""Kaggle episode replay (JSON) の分析と実メタ抽出.

data/replays/**/*.json（kaggle_environments 形式・Competition Data 扱い＝追跡外）を走査し:
  1. 勝敗・試合長・remainingOverageTime（時間ガード検証）をエピソード別に集計
  2. **相手デッキ（60 カード ID）を復元**して data/replays/opp_decks/ に CSV 保存
     → gauntlet / eval-deck の相手プールに使えば「実際の対戦相手」で較正できる（issue #3）

自分の agent は --team で指定するか、全エピソードの TeamNames の共通集合から自動推定する
（自分は全試合に出るが相手は入れ替わるため）。

出力は数値・ID のみ（カード名・効果文＝Pokémon Elements は扱わない）。ホストで実行可:
    python scripts/analyze_replays.py
    python scripts/analyze_replays.py --team "MyTeam" --dir data/replays
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

DECK_SIZE = 60


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _team_names(ep: dict) -> list[str]:
    return ep.get("info", {}).get("TeamNames") or ["?", "?"]


def _guess_my_team(episodes: list[dict]) -> str | None:
    """全エピソードに出現するチーム名＝自分（相手は入れ替わる前提）."""
    if not episodes:
        return None
    common = set(_team_names(episodes[0]))
    for ep in episodes[1:]:
        common &= set(_team_names(ep))
    return sorted(common)[0] if len(common) == 1 else None


def _deck_of(ep: dict, agent_idx: int) -> list[int] | None:
    """agent の初手 action（デッキ 60 枚の カード ID 列）を取り出す."""
    steps = ep.get("steps") or []
    for step in steps[:3]:  # デッキ提出は最初の数 step のどれか
        act = step[agent_idx].get("action")
        if isinstance(act, list) and len(act) == DECK_SIZE:
            return [int(x) for x in act]
    return None


def _last_overage(ep: dict, agent_idx: int) -> float | None:
    steps = ep.get("steps") or []
    for step in reversed(steps):
        obs = step[agent_idx].get("observation") or {}
        t = obs.get("remainingOverageTime")
        if t is not None:
            return float(t)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle replay の分析と実メタ抽出")
    p.add_argument("--dir", default="data/replays", help="replay JSON のルート")
    p.add_argument("--team", default=None, help="自分のチーム名（未指定なら自動推定）")
    p.add_argument(
        "--out-decks",
        default="data/replays/opp_decks",
        help="相手デッキ CSV の出力先（追跡外）",
    )
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    episodes = [(pth, _load(pth)) for pth in paths]
    if not episodes:
        raise SystemExit(f"replay が見つかりません: {args.dir}")

    my_team = args.team or _guess_my_team([ep for _, ep in episodes])
    if my_team is None:
        raise SystemExit(
            "自分のチームを推定できません（エピソードが少ない/共通名が複数）。--team で指定"
        )
    print(f"自分のチーム: {my_team}（{len(episodes)} エピソード）\n")

    os.makedirs(args.out_decks, exist_ok=True)
    rows = []
    opp_seen: Counter[str] = Counter()
    for pth, ep in episodes:
        names = _team_names(ep)
        if my_team not in names:
            print(f"  ! {os.path.basename(pth)}: 自チームが見つからずスキップ")
            continue
        me = names.index(my_team)
        opp = 1 - me
        rewards = ep.get("rewards") or [None, None]
        r = rewards[me]
        result = "win" if (r or 0) > (rewards[opp] or 0) else "loss"
        n_steps = len(ep.get("steps") or [])
        overage = _last_overage(ep, me)
        opp_name = names[opp]
        opp_seen[opp_name] += 1

        # 相手デッキを CSV 保存（ID のみ・追跡外）。同一相手の複数試合はエピソード ID で区別。
        deck = _deck_of(ep, opp)
        deck_file = ""
        if deck:
            eid = (
                ep.get("info", {}).get("EpisodeId")
                or os.path.splitext(os.path.basename(pth))[0]
            )
            deck_file = os.path.join(args.out_decks, f"opp_{eid}.csv")
            with open(deck_file, "w") as f:
                f.write("\n".join(str(c) for c in deck) + "\n")

        # どの提出か（ディレクトリ名 nn/ismcts を拾う）
        variant = os.path.basename(os.path.dirname(pth))
        rows.append((variant, result, n_steps, overage, opp_name, deck_file))

    print(f"{'提出':8} {'結果':5} {'steps':>5} {'残overage':>10}  相手 / 抽出デッキ")
    for variant, result, n_steps, overage, opp_name, deck_file in rows:
        ov = f"{overage:.1f}s" if overage is not None else "n/a"
        df = os.path.basename(deck_file) if deck_file else "(デッキ抽出失敗)"
        print(f"{variant:8} {result:5} {n_steps:>5} {ov:>10}  {opp_name} / {df}")

    wins = sum(1 for r in rows if r[1] == "win")
    print(f"\n合計: {wins}/{len(rows)} 勝  相手チーム数: {len(opp_seen)}")
    print(
        f"相手デッキ CSV: {args.out_decks}/（gauntlet の相手プールに追加して実メタ較正）"
    )
    min_ov = min((r[3] for r in rows if r[3] is not None), default=None)
    if min_ov is not None:
        print(
            f"remainingOverageTime 最小: {min_ov:.1f}s（600 起点・小さいほど時間を消費）"
        )


if __name__ == "__main__":
    main()
