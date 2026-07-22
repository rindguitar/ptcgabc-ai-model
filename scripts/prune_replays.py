"""消費済み replay JSON の破棄（日次: DL → analyze → replay-extract → prune）.

生 JSON を必要とする「消費者」はいずれも走査した episode_id を状態ファイルに記録する:
  - analyze_replays.py        → data/replays/episodes_log.csv（敗因ログ＋相手デッキ収穫）
  - extract_replay_samples.py → data/replays/value_samples.npz の episodes（value サンプル）
  - mine_fetch_priorities.py  → data/fetch_priors/state.json（§47 cardId 単位マイニング）
  - mine_search_role_priors.py→ data/fetch_priors/role_state.json（§50/51 役割別マイニング）
したがって **episode_id が状態ファイルに載っている＝その消費者を通過済み** と判定できる。

このスクリプトは、必要な全消費者を通過済みの JSON だけを削除する:
  1. 消費者ごとの処理済み episode_id 集合を状態ファイルから読む
  2. ディスク上の各 JSON の episode_id（＝ファイル名 stem・EpisodeId と一致する前提）が
     必要な全集合に含まれるかを判定
  3. --apply 指定時のみ削除（既定は dry-run で削減量を表示するだけ）

keep-variants は既定で破棄対象から除外し温存する（--include-own で明示的に含める）。理由は2つ:
  - 自分の試合（alphago/ismcts/nn 等）: behavior_diff / scout_field が状態を残さず随時再読み込み
  - others/: 上位帯から意図的に DL した教師プール。analyze/value だけでは
    mine_fetch_priorities/mine_search_role_priors（§47/§50/51）が未追跡のまま破棄されうる
    （§54 の事故）。`make mine-teachers` で4消費者（analyze,value,fetch_priors,role_priors）を
    揃えて破棄する運用にすれば --keep-variants から others を外して呼べる（既定は独立温存）。
    scout_field/behavior_diff 等の状態を持たない一発レポートツールは対象外
    （--team 等が必須で全体一括に馴染まない・使うなら破棄前に手動実行）。

出力は数値・ID・variant 名のみ（カード名・効果文＝Pokémon Elements は扱わない）。ホストで実行可:
    python scripts/prune_replays.py            # dry-run（何が消えるか確認）
    python scripts/prune_replays.py --apply    # 実削除
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


def _analyze_ids(log_path: str) -> set[str]:
    """analyze_replays.py が記録した処理済み episode_id（episodes_log.csv）."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path) as f:
        return {r["episode_id"].split("#")[0] for r in csv.DictReader(f)}


def _value_ids(npz_path: str) -> set[str]:
    """extract_replay_samples.py が記録した処理済み episode_id（value_samples.npz）."""
    if not os.path.exists(npz_path):
        return set()
    d = np.load(npz_path, allow_pickle=True)
    return {str(x) for x in d["episodes"]}


def _seat_state_ids(state_path: str) -> set[str]:
    """mine_fetch_priorities.py / mine_search_role_priors.py の state（"eid#seat" タグ）を
    episode_id 集合へ変換する。**--team 無指定（全チーム一括）で実行した前提**＝両席とも
    必ずタグが付くため、片方の席タグが載っていれば episode 全体を消費済みとみなせる
    （--team 指定で片席しか処理していない state をそのまま使うと過大判定になるので注意）."""
    if not os.path.exists(state_path):
        return set()
    with open(state_path) as f:
        state = json.load(f)
    return {tag.split("#", 1)[0] for tag in state.get("episodes", [])}


def _fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def main() -> None:
    p = argparse.ArgumentParser(
        description="消費済み replay JSON の破棄（冪等・ホスト）"
    )
    p.add_argument("--dir", default="data/replays", help="replay JSON のルート（再帰）")
    p.add_argument(
        "--consumers",
        default="analyze,value",
        help="通過を必須とする消費者（カンマ区切り: analyze,value,fetch_priors,role_priors）",
    )
    p.add_argument(
        "--fetch-priors-state",
        default="data/fetch_priors/state.json",
        help="mine_fetch_priorities.py の state（--consumers に fetch_priors を含む場合）",
    )
    p.add_argument(
        "--role-priors-state",
        default="data/fetch_priors/role_state.json",
        help="mine_search_role_priors.py の state（--consumers に role_priors を含む場合）",
    )
    p.add_argument(
        "--keep-variants",
        default="alphago,ismcts,nn,others",
        help="温存する variant（自分の試合＝behavior 分析が再読み込み／others＝§47 教師プール・"
        "cardId マイニング未追跡のため独立ライフサイクル）。"
        "**前方一致**: alphago は alphago_v4 / alphago_v35 等の派生ディレクトリも温存する",
    )
    p.add_argument(
        "--include-own",
        action="store_true",
        help="keep-variants も破棄対象に含める（behavior 分析や§47教師プールを諦める場合）",
    )
    p.add_argument("--apply", action="store_true", help="実削除する（既定は dry-run）")
    args = p.parse_args()

    # 1. 必要な消費者の処理済み集合を読む（analyze / value / fetch_priors / role_priors）
    known = {"analyze", "value", "fetch_priors", "role_priors"}
    wanted = {c.strip() for c in args.consumers.split(",") if c.strip()}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"未知の消費者: {sorted(unknown)}（{sorted(known)} のみ）")
    consumed_sets: list[set[str]] = []
    if "analyze" in wanted:
        consumed_sets.append(_analyze_ids(os.path.join(args.dir, "episodes_log.csv")))
    if "value" in wanted:
        consumed_sets.append(_value_ids(os.path.join(args.dir, "value_samples.npz")))
    if "fetch_priors" in wanted:
        consumed_sets.append(_seat_state_ids(args.fetch_priors_state))
    if "role_priors" in wanted:
        consumed_sets.append(_seat_state_ids(args.role_priors_state))
    if not consumed_sets:
        raise SystemExit(f"--consumers が空です（{sorted(known)} を指定）")
    required = set.intersection(*consumed_sets)  # 全消費者を通過した episode_id

    keep_variants = {v.strip() for v in args.keep_variants.split(",") if v.strip()}

    # 2. ディスク上の JSON を走査し、破棄可否を variant 別に集計する
    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    to_delete: list[tuple[str, int]] = []  # (path, size)
    kept = n_pending = 0
    for pth in paths:
        variant = os.path.basename(os.path.dirname(pth))
        eid = os.path.splitext(os.path.basename(pth))[0]  # ファイル名 stem = EpisodeId
        # 温存判定は前方一致: A/B 用の派生ディレクトリ（例 alphago_v4 / alphago_v35）を
        # keep-variants に列挙し忘れて scout 前に消してしまう事故を防ぐ
        keep = any(variant.startswith(k) for k in keep_variants)
        if not args.include_own and keep:
            kept += 1
            continue
        if eid in required:
            to_delete.append((pth, os.path.getsize(pth)))
        else:
            n_pending += 1  # まだ消費者を通過していない（削除しない）

    # 3. 集計を表示し、--apply 時のみ削除する
    by_var: dict[str, tuple[int, int]] = {}
    for pth, sz in to_delete:
        v = os.path.basename(os.path.dirname(pth))
        n, s = by_var.get(v, (0, 0))
        by_var[v] = (n + 1, s + sz)
    total_bytes = sum(s for _, s in to_delete)

    print(
        f"JSON {len(paths)} 件 / 必須消費者 {sorted(wanted)} を通過済み {len(required)} episode\n"
        f"温存（keep-variants {sorted(keep_variants)}）: {kept} 件 / "
        f"未消費で保留: {n_pending} 件\n"
    )
    for v in sorted(by_var):
        n, s = by_var[v]
        print(f"  {v:14} 破棄対象 {n:5} 件 / {_fmt_gb(s)}")
    print(f"\n破棄対象 合計: {len(to_delete)} 件 / {_fmt_gb(total_bytes)}")

    if not args.apply:
        print("（dry-run）実削除するには --apply を付ける")
        return
    for pth, _ in to_delete:
        os.remove(pth)
    print(f"削除しました: {len(to_delete)} 件 / {_fmt_gb(total_bytes)} を解放")


if __name__ == "__main__":
    main()
