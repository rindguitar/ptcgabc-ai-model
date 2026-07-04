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


def _end_state(ep: dict, agent_idx: int) -> dict | None:
    """自分視点の最終盤面（current）を取り出す（無ければ None）."""
    for step in reversed(ep.get("steps") or []):
        cur = (step[agent_idx].get("observation") or {}).get("current")
        if cur and cur.get("players"):
            return cur
    return None


def _classify_end(ep: dict, agent_idx: int, result: str) -> str:
    """終局理由を最終盤面から近似分類する.

    エンジンの敗北条件: ①サイドを全部取られる ②ターン開始時に山札0 ③場にポケモンがいない。
    RESULT ログは replay に残らないことがあるため、自分視点の最終 current から推定する
    （自分の観測は数手遅れることがあるので近似・グレーは「その他」）。
    """
    cur = _end_state(ep, agent_idx)
    if cur is None:
        return "不明"
    yi = cur.get("yourIndex", agent_idx)
    mine, opp = cur["players"][yi], cur["players"][1 - yi]

    def board(p) -> int:  # 場のポケモン数（active + bench）
        return len([a for a in (p.get("active") or []) if a]) + len(
            p.get("bench") or []
        )

    my_prize = len(mine.get("prize") or [])
    opp_prize = len(opp.get("prize") or [])
    if result == "loss":
        if my_prize == 0 or opp_prize == 0:
            return (
                "サイド負け"  # どちらかのサイドが尽きた終局（自分が負けなら取り切られ）
            )
        if (mine.get("deckCount") or 0) <= 1:
            return "デッキ切れ"
        if board(mine) <= 1:
            return "ベンチ切れ"  # 場の駒が尽きる直前の観測（残1→KOで場が空）
        return "その他"
    # win 側: どう勝ったか
    if my_prize == 0:
        return "サイド勝ち"
    if (opp.get("deckCount") or 0) <= 1:
        return "相手デッキ切れ"
    if board(opp) <= 1:
        return "相手ベンチ切れ"
    return "その他"


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
    deck_hashes: set[str] = set()
    n_self = 0
    for pth, ep in episodes:
        names = _team_names(ep)
        if my_team not in names:
            print(f"  ! {os.path.basename(pth)}: 自チームが見つからずスキップ")
            continue
        me = names.index(my_team)
        opp = 1 - me
        # セルフ検証マッチ（相手＝自分）は集計から除外（メタ情報ゼロ・勝敗も無意味）
        if names[opp] == my_team:
            n_self += 1
            continue
        rewards = ep.get("rewards") or [None, None]
        result = "win" if (rewards[me] or 0) > (rewards[opp] or 0) else "loss"
        n_steps = len(ep.get("steps") or [])
        overage = _last_overage(ep, me)
        opp_name = names[opp]
        opp_seen[opp_name] += 1
        cause = _classify_end(ep, me, result)

        # 相手デッキを CSV 保存（ID のみ・追跡外）。内容ハッシュ名＝同一デッキは1ファイルに集約。
        deck = _deck_of(ep, opp)
        if deck:
            import hashlib

            h = hashlib.md5(",".join(map(str, sorted(deck))).encode()).hexdigest()[:8]
            deck_hashes.add(h)
            deck_file = os.path.join(args.out_decks, f"opp_{h}.csv")
            if not os.path.exists(deck_file):
                with open(deck_file, "w") as f:
                    f.write("\n".join(str(c) for c in deck) + "\n")

        variant = os.path.basename(os.path.dirname(pth))  # nn / ismcts（置き場所から）
        rows.append((variant, result, n_steps, overage, opp_name, cause))

    if n_self:
        print(f"（セルフ検証マッチ {n_self} 件を除外）\n")

    # 提出（variant）別サマリ
    for variant in sorted({r[0] for r in rows}):
        vs = [r for r in rows if r[0] == variant]
        wins = sum(1 for r in vs if r[1] == "win")
        steps_avg = sum(r[2] for r in vs) / len(vs)
        min_ov = min((r[3] for r in vs if r[3] is not None), default=None)
        ov = f"{min_ov:.1f}s" if min_ov is not None else "n/a"
        print(
            f"=== {variant}: {wins}/{len(vs)} 勝（{wins / len(vs):.3f}）"
            f" 平均steps {steps_avg:.0f} / overage最小 {ov} ==="
        )
        for res in ("win", "loss"):
            causes = Counter(r[5] for r in vs if r[1] == res)
            if causes:
                detail = "  ".join(f"{c}:{n}" for c, n in causes.most_common())
                print(f"  {res:4}: {detail}")

    print(f"\n相手チーム数: {len(opp_seen)}  対戦数上位: ", end="")
    print("  ".join(f"{n}({c})" for n, c in opp_seen.most_common(5)))
    print(
        f"実メタデッキ（ユニーク）: {len(deck_hashes)} 件 → {args.out_decks}/"
        "（gauntlet の相手プールに追加して実メタ較正）"
    )


if __name__ == "__main__":
    main()
