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
import os
import shutil
import tarfile

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
ROOT = os.path.dirname(SRC)
# main.py が import する我々のモジュール（依存の閉包）
MODULES = ["submission.py", "ismcts.py", "determinize.py", "agents.py", "cards.py"]

MAIN_PY = '''\
"""Kaggle 提出エントリ: 公式形式 agent(obs_dict) -> list[int] を公開する。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from submission import make_kaggle_agent

# 1 試合 600 秒の累積クロックに安全マージンを見て 540 秒で運用する。
agent = make_kaggle_agent("ismcts", deck_path="deck.csv", game_budget=540.0)
'''


def build(deck_path: str, out_tar: str) -> tuple[str, list[str]]:
    """提出パッケージを組み立てて (tar パス, 同梱物一覧) を返す."""
    build_dir = os.path.join(ROOT, "models", "submission")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    with open(os.path.join(build_dir, "main.py"), "w") as f:
        f.write(MAIN_PY)
    for module in MODULES:
        shutil.copy(os.path.join(SRC, module), os.path.join(build_dir, module))
    shutil.copytree(
        os.path.join(SRC, "cg"),
        os.path.join(build_dir, "cg"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(deck_path, os.path.join(build_dir, "deck.csv"))

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
