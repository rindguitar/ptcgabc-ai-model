"""実メタ（replay 抽出デッキ）で判定用ガントレットを置換する.

analyze_replays.py が溜めた episodes_log.csv（遭遇頻度）と data/replays/opp_decks/（デッキ CSV）
から、**実際の対戦環境を代表する N デッキ**を選んで models/gauntlet/ を置き換える。
champion-gate / eval-deck は models/gauntlet/*.csv があればそれを相手プールに使うので、
これ以降の ratchet 判定は「本物の敵」で較正される（issue #3）。

選抜: 遭遇頻度の高い順（実メタ分布を反映）。others（上位帯から収穫したデッキ）も頻度に
数えるため、上位帯 replay を足すほど「これから当たる敵」に判定が寄る。
レートが上がったら replay を取り直して再実行＝ローリング較正。

    python scripts/gauntlet_from_replays.py --n 16     # make gauntlet-real
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser(description="実メタデッキで判定ガントレットを置換")
    p.add_argument("--log", default="data/replays/episodes_log.csv")
    p.add_argument("--decks-dir", default="data/replays/opp_decks")
    p.add_argument("--out", default="models/gauntlet")
    p.add_argument("--n", type=int, default=16, help="採用するデッキ数")
    args = p.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"ログがありません: {args.log}（先に analyze_replays.py）")
    with open(args.log) as f:
        rows = list(csv.DictReader(f))
    freq: Counter[str] = Counter(r["deck_hash"] for r in rows if r["deck_hash"])
    if not freq:
        raise SystemExit("抽出デッキがログにありません")

    # 頻度順（同数はハッシュで安定ソート）に、ファイルが存在するものだけ採用
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    picks: list[tuple[str, int]] = []
    for h, n in ranked:
        path = os.path.join(args.decks_dir, f"opp_{h}.csv")
        if os.path.exists(path):
            picks.append((path, n))
        if len(picks) >= args.n:
            break
    if not picks:
        raise SystemExit(f"デッキ CSV が見つかりません: {args.decks_dir}")

    # 置換（旧ガントレットは消す＝判定プールを実メタへ完全切替）
    os.makedirs(args.out, exist_ok=True)
    for old in glob.glob(os.path.join(args.out, "*.csv")):
        os.remove(old)
    for i, (path, _) in enumerate(picks):
        shutil.copy(path, os.path.join(args.out, f"gauntlet_{i:02d}.csv"))

    total = sum(freq.values())
    covered = sum(n for _, n in picks)
    print(
        f"実メタガントレット {len(picks)} デッキを {args.out}/ に配置"
        f"（全遭遇 {total} 回中 {covered} 回分をカバー・ユニーク {len(freq)} 種から選抜）"
    )
    print("以後の champion-gate / eval-deck はこのプールで判定される（実メタ較正）")


if __name__ == "__main__":
    main()
