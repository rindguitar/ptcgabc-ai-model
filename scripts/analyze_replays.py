"""Kaggle episode replay (JSON) の分析と実メタ抽出.

data/replays/**/*.json（kaggle_environments 形式・Competition Data 扱い＝追跡外）を走査し:
  1. 勝敗・敗因分類・試合長・remainingOverageTime（時間ガード検証）をエピソード別に集計
  2. **相手デッキ（60 カード ID）を復元**して data/replays/opp_decks/ に CSV 保存
     （内容ハッシュ名＝同一デッキは1ファイル）→ `make gauntlet-real` で判定プールへ
  3. 結果を **episodes_log.csv に永続化（追記・冪等）** → 集計済みの JSON は削除してよい
     （replay は大容量・保持期限も不明なため「毎日 DL → analyze → JSON 削除」の運用）

レート帯バイアス対策: 自分の対戦相手は「今の自分と同レート帯」に偏る。上位帯のデッキは
Kaggle leaderboard から**他チーム同士の episode** を DL して `data/replays/others/` に置けば、
両プレイヤーのデッキを収穫する（勝敗集計には入れない）。

自分の agent は --team で指定するか、全エピソードで最頻出のチーム名から自動推定する。
出力は数値・ID・チーム名のみ（カード名・効果文＝Pokémon Elements は扱わない）。ホストで実行可:
    python scripts/analyze_replays.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
from collections import Counter

DECK_SIZE = 60
LOG_FIELDS = [
    "episode_id",
    "variant",
    "result",
    "cause",
    "steps",
    "overage",
    "opp_name",
    "deck_hash",
]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _team_names(ep: dict) -> list[str]:
    return ep.get("info", {}).get("TeamNames") or ["?", "?"]


def _guess_my_team(episodes: list[dict]) -> str | None:
    """最頻出のチーム名＝自分（自分は自分の全試合に出る。others/ が混ざっても頑健）."""
    c: Counter[str] = Counter()
    for ep in episodes:
        c.update(set(_team_names(ep)))
    return c.most_common(1)[0][0] if c else None


def _deck_of(ep: dict, agent_idx: int) -> list[int] | None:
    """agent の初手 action（デッキ 60 枚のカード ID 列）を取り出す."""
    steps = ep.get("steps") or []
    for step in steps[:3]:  # デッキ提出は最初の数 step のどれか
        act = step[agent_idx].get("action")
        if isinstance(act, list) and len(act) == DECK_SIZE:
            return [int(x) for x in act]
    return None


def _last_overage(ep: dict, agent_idx: int) -> float | None:
    for step in reversed(ep.get("steps") or []):
        t = (step[agent_idx].get("observation") or {}).get("remainingOverageTime")
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
            return "サイド負け"
        if (mine.get("deckCount") or 0) <= 1:
            return "デッキ切れ"
        if board(mine) <= 1:
            return "ベンチ切れ"  # 場の駒が尽きる直前の観測（残1→KOで場が空）
        return "その他"
    if my_prize == 0:
        return "サイド勝ち"
    if (opp.get("deckCount") or 0) <= 1:
        return "相手デッキ切れ"
    if board(opp) <= 1:
        return "相手ベンチ切れ"
    return "その他"


def _save_deck(deck: list[int], out_dir: str) -> str:
    """デッキを内容ハッシュ名で保存し hash を返す（同一デッキは1ファイルに集約）."""
    h = hashlib.md5(",".join(map(str, sorted(deck))).encode()).hexdigest()[:8]
    path = os.path.join(out_dir, f"opp_{h}.csv")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("\n".join(str(c) for c in deck) + "\n")
    return h


def _read_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle replay の分析と実メタ抽出")
    p.add_argument("--dir", default="data/replays", help="replay JSON のルート")
    p.add_argument("--team", default=None, help="自分のチーム名（未指定なら自動推定）")
    p.add_argument(
        "--out-decks",
        default="data/replays/opp_decks",
        help="相手デッキ CSV の出力先（追跡外）",
    )
    p.add_argument(
        "--log",
        default="data/replays/episodes_log.csv",
        help="エピソード結果の永続ログ（追記・冪等。これがあれば JSON は削除可）",
    )
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    episodes = [(pth, _load(pth)) for pth in paths]

    log_rows = _read_log(args.log)
    done_ids = {r["episode_id"].split("#")[0] for r in log_rows}

    my_team = args.team or (
        _guess_my_team([ep for _, ep in episodes]) if episodes else None
    )
    if my_team is None and episodes:
        raise SystemExit("自分のチームを推定できません。--team で指定")
    if my_team:
        print(f"自分のチーム: {my_team}")
    print(f"JSON {len(episodes)} 件 / ログ既存 {len(log_rows)} 行\n")

    os.makedirs(args.out_decks, exist_ok=True)
    new_rows: list[dict] = []
    n_skip = n_self = n_harvest = 0
    for pth, ep in episodes:
        eid = str(
            ep.get("info", {}).get("EpisodeId")
            or os.path.splitext(os.path.basename(pth))[0]
        )
        if eid in done_ids:
            n_skip += 1
            continue
        done_ids.add(eid)
        names = _team_names(ep)
        variant = os.path.basename(os.path.dirname(pth))  # nn / ismcts / others

        if my_team not in names:
            # 他チーム同士の episode（レート帯バイアス対策の収穫）: 両デッキを保存のみ
            for ag in (0, 1):
                deck = _deck_of(ep, ag)
                if deck:
                    h = _save_deck(deck, args.out_decks)
                    new_rows.append(
                        {
                            "episode_id": f"{eid}#{ag}",
                            "variant": "others",
                            "result": "",
                            "cause": "",
                            "steps": "",
                            "overage": "",
                            "opp_name": names[ag],
                            "deck_hash": h,
                        }
                    )
                    n_harvest += 1
            continue

        me = names.index(my_team)
        opp = 1 - me
        if names[opp] == my_team:  # セルフ検証マッチ（メタ情報ゼロ）
            n_self += 1
            new_rows.append(
                {
                    "episode_id": eid,
                    "variant": variant,
                    "result": "self",
                    "cause": "",
                    "steps": "",
                    "overage": "",
                    "opp_name": my_team,
                    "deck_hash": "",
                }
            )
            continue

        rewards = ep.get("rewards") or [None, None]
        result = "win" if (rewards[me] or 0) > (rewards[opp] or 0) else "loss"
        overage = _last_overage(ep, me)
        deck = _deck_of(ep, opp)
        new_rows.append(
            {
                "episode_id": eid,
                "variant": variant,
                "result": result,
                "cause": _classify_end(ep, me, result),
                "steps": len(ep.get("steps") or []),
                "overage": f"{overage:.1f}" if overage is not None else "",
                "opp_name": names[opp],
                "deck_hash": _save_deck(deck, args.out_decks) if deck else "",
            }
        )

    # ログへ追記（冪等）。これで JSON は削除してよい。
    if new_rows:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        write_header = not os.path.exists(args.log)
        with open(args.log, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
    all_rows = log_rows + new_rows
    print(
        f"新規 {len(new_rows)} 行を記録（処理済みスキップ {n_skip}・セルフ {n_self}・"
        f"他チーム収穫デッキ {n_harvest}）→ {args.log}"
    )
    print("※ 集計済みの JSON ファイルは削除して構いません（ログとデッキ CSV が残る）\n")

    # ==== 累積サマリ（ログ全体から） ====
    games = [r for r in all_rows if r["result"] in ("win", "loss")]
    for variant in sorted({r["variant"] for r in games}):
        vs = [r for r in games if r["variant"] == variant]
        wins = sum(1 for r in vs if r["result"] == "win")
        steps = [int(r["steps"]) for r in vs if r["steps"]]
        ovs = [float(r["overage"]) for r in vs if r["overage"]]
        print(
            f"=== {variant}: {wins}/{len(vs)} 勝（{wins / len(vs):.3f}） "
            f"平均steps {sum(steps) / len(steps):.0f} / overage最小 "
            f"{min(ovs):.1f}s ==="
            if steps and ovs
            else f"=== {variant}: {wins}/{len(vs)} 勝 ==="
        )
        for res in ("win", "loss"):
            causes = Counter(r["cause"] for r in vs if r["result"] == res)
            if causes:
                print(
                    f"  {res:4}: "
                    + "  ".join(f"{c}:{n}" for c, n in causes.most_common())
                )

    opp_seen = Counter(r["opp_name"] for r in games)
    hashes = {r["deck_hash"] for r in all_rows if r["deck_hash"]}
    print(f"\n対戦相手チーム数: {len(opp_seen)}")
    print(
        f"実メタデッキ（ユニーク・others 含む）: {len(hashes)} 件 → {args.out_decks}/"
    )
    print("判定プールへの反映: make gauntlet-real（models/gauntlet/ を実メタで置換）")


if __name__ == "__main__":
    main()
