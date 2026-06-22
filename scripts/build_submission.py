"""提出パッケージ submission.tar.gz を組み立てる.

公式形式（サンプルノートブック準拠）に合わせ、tar のルート直下に以下を置く:
    main.py            … 公式形式 agent(obs_dict) を公開するエントリ
    cg/                … cabt エンジン（実行時に必要）
    deck.csv           … チャンピオンデッキ（60 カード ID）
    submission.py / ismcts.py / determinize.py / agents.py / cards.py … 我々の方策

注意:
- deck.csv（デッキ＝Pokémon Element）と cg（Competition Data）を含むため、出力 tar と
  ビルドディレクトリは models/ 配下（追跡対象外）。**コミットしない**。
- 実際の提出（Kaggle へのアップロード）はユーザーが行う。Kaggle 側の cg-lib データセットの
  cg を使いたい場合は、Kaggle ノートブックで cg を差し替えて再パッケージしてもよい。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import tarfile

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
ROOT = os.path.dirname(SRC)
# 我々のモジュール（依存の閉包）。一意 package ptcgbot/ に入れて名前衝突を避ける。
MODULE_NAMES = ["submission", "ismcts", "determinize", "agents", "cards"]
PACKAGE = "ptcgbot"

# モジュール間の import を package 名前空間へ書き換える（cg エンジンは対象外＝そのまま）。
_IMPORT_RE = re.compile(r"^(\s*)from (" + "|".join(MODULE_NAMES) + r")\b", re.MULTILINE)


def _namespace_imports(source: str) -> str:
    """`from agents import ...` 等を `from ptcgbot.agents import ...` に書き換える."""
    return _IMPORT_RE.sub(r"\1from " + PACKAGE + r".\2", source)


MAIN_PY = '''\
"""Kaggle 提出エントリ: 公式形式 agent(obs_dict) -> list[int] を公開する。"""

import sys

# kaggle_environments は main.py を exec で読み込むため __file__ が無い。
# 提出物の展開先（固定パス）と cwd を import path に追加して同梱物を読めるようにする。
for _p in ("/kaggle_simulations/agent", "."):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 我々のモジュールは package ptcgbot に入れてある（汎用名の衝突回避）。
from ptcgbot.submission import make_kaggle_agent

# opp_decks/ の候補デッキで相手デッキを観測整合に推定（無ければミラー仮定にフォールバック）。
# 1 試合 600 秒の累積クロック（remainingOverageTime）に安全マージンを見て 540 秒で運用する。
agent = make_kaggle_agent(
    "ismcts", deck_path="deck.csv", opp_pool_dir="opp_decks", game_budget=540.0
)
'''


def build(deck_path: str, out_tar: str) -> tuple[str, list[str]]:
    """提出パッケージを組み立てて (tar パス, 同梱物一覧) を返す."""
    build_dir = os.path.join(ROOT, "models", "submission")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    with open(os.path.join(build_dir, "main.py"), "w") as f:
        f.write(MAIN_PY)
    # 我々のモジュールは package ptcgbot/ に入れ、相互 import を名前空間化する
    pkg_dir = os.path.join(build_dir, PACKAGE)
    os.makedirs(pkg_dir)
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()
    for module in MODULE_NAMES:
        src = open(os.path.join(SRC, module + ".py")).read()
        with open(os.path.join(pkg_dir, module + ".py"), "w") as f:
            f.write(_namespace_imports(src))
    # cg エンジンは root 直下（top-level import のまま）
    shutil.copytree(
        os.path.join(SRC, "cg"),
        os.path.join(build_dir, "cg"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(deck_path, os.path.join(build_dir, "deck.csv"))

    # 相手候補デッキ群（観測整合の相手デッキ推定用）。中立名で同梱（Pokémon 名を避ける）。
    opp_dir = os.path.join(build_dir, "opp_decks")
    os.makedirs(opp_dir)
    idx = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "*.csv"))):
        try:
            d = [int(x) for x in open(p).read().splitlines() if x.strip()]
        except ValueError:
            continue
        if len(d) == 60:
            shutil.copy(p, os.path.join(opp_dir, f"opp_{idx}.csv"))
            idx += 1

    os.makedirs(os.path.dirname(out_tar) or ".", exist_ok=True)
    names = sorted(os.listdir(build_dir))
    with tarfile.open(out_tar, "w:gz") as tar:
        for name in names:
            tar.add(os.path.join(build_dir, name), arcname=name)
    return out_tar, names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="提出パッケージ submission.tar.gz を作成"
    )
    parser.add_argument(
        "--deck", default="models/champion_deck.csv", help="同梱するデッキ CSV"
    )
    parser.add_argument("--out", default="models/submission.tar.gz", help="出力 tar.gz")
    args = parser.parse_args()

    if not os.path.exists(args.deck):
        raise SystemExit(f"デッキが見つかりません: {args.deck}（先にリーグを実行）")

    out_tar, names = build(args.deck, args.out)
    size_mb = os.path.getsize(out_tar) / 1e6
    print(f"提出パッケージを作成: {out_tar} ({size_mb:.1f} MB)")
    print(f"  同梱物（root 直下）: {', '.join(names)}")
    print(
        "  ※ deck.csv と cg は Competition Data を含むため追跡外。Kaggle へはユーザーが提出。"
    )


if __name__ == "__main__":
    main()
