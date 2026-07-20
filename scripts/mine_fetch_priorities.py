"""教師チームの replay から**山札サーチの取得優先度**（fetch_priors・§47）をマイニングする.

§43/§45 の実勢分析: リサイクル強制手の発火は100%だが機会が希少（律速は札の可用性）。
教師（例: 1位チーム）はサーチでキーカード（例: id741）を優先取得して可用性を保つが、
我々の _generic_select はカテゴリ順（たね>エネ>その他）のみ＝キーカードが埋没する。
本スクリプトは教師席の山札サーチ局面を集計し、**カード ID 別の取得率
（取った回数 / 提示された回数）**を出力する。出力 JSON は build_submission.py の
--fetch-priors で提出物に同梱し、_generic_select のサーチ順位に注入する。

**冪等・永続**（episodes_log / value_samples と同じ運用）: 集計カウントと処理済み
episode+席 を --state に追記蓄積する。**絞ったら生 JSON は削除してよい**
（replay-format.md のライフサイクル参照）。ペアリングは steps[i+1] 正実装（§45）。

出力はすべて数値・ID まで（Pokémon Elements 非表示・規約遵守）。

--team は省略可（省略時は corpus 内の全チームを一括マイニングし team×deck_hash 別に
JSON を出す）。deck_hash は完全一致デッキの出所ラベルに過ぎず、実採用時は priors の
cardId が自分のデッキの検索対象と重なっていれば別デッキ由来でも部分的に使える
（アーキタイプが近い・キーカードが共通、等）。

    python scripts/mine_fetch_priorities.py            # 全チーム一括
    python scripts/mine_fetch_priorities.py --team <TeamName>  # 単一チーム
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cg.api import to_observation_class  # noqa: E402

DECK_SIZE = 60


def _deck_hash(deck: list[int]) -> str:
    """analyze_replays._save_deck と同一書式（opp_decks のファイル名と突合できる）."""
    return hashlib.md5(",".join(map(str, sorted(deck))).encode()).hexdigest()[:8]


def _deck_of(ep: dict, seat: int) -> list[int] | None:
    """席の初手 action（デッキ 60 枚）を取り出す（先頭数 step のどれか・位置非依存）."""
    for step in (ep.get("steps") or [])[:3]:
        a = step[seat].get("action")
        if isinstance(a, list) and len(a) == DECK_SIZE:
            return [int(x) for x in a]
    return None


def _search_card_id(sel, option_i: int) -> int:
    """山札検索 option の実カード ID を解決する.

    ⚠️ option は cardId を持たない（area/index/playerIndex/type のみ・実測確認済み）。
    実カードは sel.deck[option.index].id で解決する（agents._generic_select と同じ修正）。
    """
    idx = sel.option[option_i].index
    if idx is None or sel.deck is None or not (0 <= idx < len(sel.deck)):
        return 0
    return sel.deck[idx].id or 0


def _count_search(obs, act: list[int], counts: dict[int, list[int]]) -> bool:
    """1つの山札サーチ選択を counts（cardId -> [取得, 提示]）へ加算する.

    流れ: 1. select が山札からの選択（sel.deck あり）か判定 → 2. 提示された全 option の
    実カード ID（sel.deck[index].id）を「提示」に加算 → 3. act（応答 index 列）が指す
    カード ID を「取得」に加算。サーチでなければ何もせず False。
    """
    sel = obs.select
    if sel is None or sel.deck is None or sel.maxCount < 1 or not sel.option:
        return False
    for i in range(len(sel.option)):
        counts[_search_card_id(sel, i)][1] += 1
    for idx in act:
        if 0 <= idx < len(sel.option):
            counts[_search_card_id(sel, idx)][0] += 1
    return True


def mine_episode(ep: dict, seat: int, counts: dict[int, list[int]]) -> int:
    """1 episode の指定席から全サーチ局面を集計し、局面数を返す.

    ペアリングは §45 の正実装: obs[i] への応答は steps[i+1][seat].action。
    """
    steps = ep.get("steps") or []
    n = 0
    for i, step in enumerate(steps):
        rec = step[seat]
        if rec.get("status") != "ACTIVE" or not rec.get("observation"):
            continue
        if i + 1 >= len(steps):
            continue
        act = steps[i + 1][seat].get("action")
        if not isinstance(act, list):
            continue
        obs = to_observation_class(rec["observation"])
        if _count_search(obs, act, counts):
            n += 1
    return n


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"episodes": [], "keys": {}}
    with open(path) as f:
        return json.load(f)


def _sanitize(name: str) -> str:
    """チーム名をファイル名に使える形へ（非英数は _）."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name) or "team"


def main() -> None:
    p = argparse.ArgumentParser(description="山札サーチ取得優先度のマイニング（§47）")
    p.add_argument(
        "--team",
        default=None,
        help="教師チーム名（省略時は corpus 内の全チームを一括マイニング）",
    )
    p.add_argument("--dir", default="data/replays", help="episode JSON のルート")
    p.add_argument(
        "--state",
        default="data/fetch_priors/state.json",
        help="累積カウントの永続先（冪等・絞ったら生 JSON は削除可）",
    )
    p.add_argument("--out-dir", default="data/fetch_priors", help="priors JSON の出力先")
    p.add_argument(
        "--min-offered",
        type=int,
        default=5,
        help="priors に載せる最少提示回数（未満はノイズとして除外）",
    )
    args = p.parse_args()

    # 1. 永続 state を読む → 2. 未処理の (episode, 席) を集計 → 3. state 保存 →
    # 4. チーム×デッキ別に priors JSON を出力
    state = _load_state(args.state)
    done = set(state["episodes"])
    keys: dict[str, dict] = state["keys"]

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    n_new = 0
    for path in paths:
        with open(path) as f:
            ep = json.load(f)
        eid = str(
            ep.get("info", {}).get("EpisodeId")
            or os.path.splitext(os.path.basename(path))[0]
        )
        names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
        for seat in (0, 1):
            tag = f"{eid}#{seat}"
            # --team 省略時は corpus 内の全チームを対象にする（低勝率player も含め
            # 冪等・永続で溜め、実採用は後段で team/deck を選んで行う＝top-replays と同じ思想）
            if (args.team is not None and names[seat] != args.team) or tag in done:
                continue
            done.add(tag)
            deck = _deck_of(ep, seat)
            if deck is None:  # デッキ不明＝帰属できないので数えない（処理済みには記録）
                continue
            key = f"{names[seat]}|{_deck_hash(deck)}"
            entry = keys.setdefault(key, {"episodes": 0, "selects": 0, "counts": {}})
            counts = defaultdict(lambda: [0, 0])
            for cid, to in entry["counts"].items():
                counts[int(cid)] = list(to)
            entry["selects"] += mine_episode(ep, seat, counts)
            entry["counts"] = {str(c): to for c, to in counts.items()}
            entry["episodes"] += 1
            n_new += 1

    os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
    state = {"episodes": sorted(done), "keys": keys}
    with open(args.state, "w") as f:
        json.dump(state, f)

    # priors 出力: 取得率 = 取得/提示（提示 min-offered 回未満は除外）
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"新規 {n_new} 席分を集計（累計 {len(done)} 席）→ {args.state}")
    for key, entry in sorted(keys.items()):
        team, dh = key.split("|", 1)
        if args.team is not None and team != args.team:
            continue
        priors = {
            cid: t / o
            for cid, (t, o) in entry["counts"].items()
            if o >= args.min_offered
        }
        out = os.path.join(args.out_dir, f"{_sanitize(team)}_{dh}.json")
        with open(out, "w") as f:
            json.dump(
                {
                    "team": team,
                    "deck_hash": dh,
                    "episodes": entry["episodes"],
                    "selects": entry["selects"],
                    "min_offered": args.min_offered,
                    "priors": priors,
                    "counts": entry["counts"],
                },
                f,
                indent=1,
            )
        top = sorted(priors.items(), key=lambda kv: -kv[1])[:10]
        print(
            f"=== {team} × deck {dh}: episode {entry['episodes']}・"
            f"サーチ局面 {entry['selects']}・対象札 {len(priors)} 種 → {out}"
        )
        for cid, r in top:
            t, o = entry["counts"][cid]
            print(f"  id{cid}: 取得率 {r:.2f}（{t}/{o}）")
    print("※ 同梱: build_submission.py --fetch-priors <上記JSON>（提出デッキと同型のものを使う）")


if __name__ == "__main__":
    main()
