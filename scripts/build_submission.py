"""提出パッケージ submission.tar.gz を組み立てる.

公式形式（サンプルノートブック準拠）に合わせ、tar のルート直下に以下を置く:
    main.py            … 公式形式 agent(obs_dict) を公開するエントリ
    cg/                … cabt エンジン（実行時に必要）
    deck.csv           … チャンピオンデッキ（60 カード ID）
    ptcgbot/           … 我々の方策モジュール群（submission/ismcts/nn_mcts/... ＝依存の閉包）
    pvnet.pt           … --policy nn のときのみ（学習済み PVNet 重み）

注意:
- deck.csv（デッキ＝Pokémon Element）と cg（Competition Data）を含むため、出力 tar と
  ビルドディレクトリは models/ 配下（追跡対象外）。**コミットしない**。
- 実際の提出（Kaggle へのアップロード）はユーザーが行う。Kaggle 側の cg-lib データセットの
  cg を使いたい場合は、Kaggle ノートブックで cg を差し替えて再パッケージしてもよい。
- --policy nn は提出環境に torch が必要（公式サンプルに AlphaZero 型 MCTS がある＝利用可の見込み）。
  ismcts 提出は torch 不要のまま（NN 系モジュールは同梱されるが import されない）。
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
# NN 系（nn_mcts/nn_eval/net/features/train）は常に同梱する。ファイルがあるだけでは torch を
# import しない（policy="nn" の時のみ submission.py が遅延 import）＝ismcts 提出でも無害。
MODULE_NAMES = [
    "submission",
    "ismcts",
    "determinize",
    "agents",
    "cards",
    "nn_mcts",
    "nn_eval",
    "net",
    "features",
    "train",
]
PACKAGE = "ptcgbot"

# モジュール間の import を package 名前空間へ書き換える（cg エンジンは対象外＝そのまま）。
_IMPORT_RE = re.compile(r"^(\s*)from (" + "|".join(MODULE_NAMES) + r")\b", re.MULTILINE)


def _namespace_imports(source: str) -> str:
    """`from agents import ...` 等を `from ptcgbot.agents import ...` に書き換える."""
    return _IMPORT_RE.sub(r"\1from " + PACKAGE + r".\2", source)


# main.py テンプレート。{agent_call} に policy 別の make_kaggle_agent 呼び出しが入る。
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
    {agent_call}
)
'''

_AGENT_CALLS = {
    "ismcts": '"ismcts", deck_path="deck.csv", opp_pool_dir="opp_decks", game_budget=540.0',
    # floored NN-MCTS: pilot≥heuristic の接地保証＋時間ガード（超過時 heuristic 退避）
    # ＋盤面補正（board-blind な value への注入・α=0.2 実測 +0.075）。
    "nn": '"nn", deck_path="deck.csv", opp_pool_dir="opp_decks", game_budget=540.0,\n'
    '    net_path="pvnet.pt", floor_rollouts={floor_rollouts}, board_bonus={board_bonus},\n'
    "    leaf_rollouts={leaf_rollouts}, threat_bonus={threat_bonus}",
}


def build(
    deck_path: str,
    out_tar: str,
    policy: str = "ismcts",
    net_path: str | None = None,
    floor_rollouts: int = 8,
    board_bonus: float = 0.2,
    leaf_rollouts: int = 0,
    recycle_at: int | None = None,
    fetch_priors: str | None = None,
    threat_bonus: float = 0.0,
) -> tuple[str, list[str]]:
    """提出パッケージを組み立てて (tar パス, 同梱物一覧) を返す."""
    build_dir = os.path.join(ROOT, "models", "submission")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)

    agent_call = _AGENT_CALLS[policy].format(
        floor_rollouts=floor_rollouts,
        board_bonus=board_bonus,
        leaf_rollouts=leaf_rollouts,
        threat_bonus=threat_bonus,
    )
    if policy == "nn" and recycle_at is not None:
        # リサイクル強制手（§43）の発動閾値の上書き（未指定は agents._RECYCLE_AT）。
        # policy_kwargs 経由で make_nn_mcts_agent(recycle_at=...) へ素通しされる。
        agent_call += f",\n    recycle_at={recycle_at}"
    with open(os.path.join(build_dir, "main.py"), "w") as f:
        f.write(MAIN_PY.format(agent_call=agent_call))
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
    if policy == "nn":
        # 学習済み重みを root 直下 pvnet.pt に固定名で同梱（main.py の net_path と一致させる）
        shutil.copy(net_path, os.path.join(build_dir, "pvnet.pt"))
    if fetch_priors:
        # 山札サーチ優先度（§47）を固定名で同梱。make_kaggle_agent が既定パス
        # fetch_priors.json を自動で読む（無ければ従来挙動＝main.py の変更は不要）。
        shutil.copy(fetch_priors, os.path.join(build_dir, "fetch_priors.json"))

    # 相手候補デッキ群（観測整合の相手デッキ推定＝ベイズ事前分布）。中立名で同梱
    # （Pokémon 名を避ける）。実 replay から抽出した相手デッキ（data/replays/opp_decks/）を
    # 優先し、無ければ data/*.csv にフォールバック。実メタが多いほど推定が当たる＝データが
    # 増えるほど強くなるループ。同一構成は重複除去する。
    opp_dir = os.path.join(build_dir, "opp_decks")
    os.makedirs(opp_dir)
    real = sorted(
        glob.glob(os.path.join(ROOT, "data", "replays", "opp_decks", "*.csv"))
    )
    sources = real or sorted(glob.glob(os.path.join(ROOT, "data", "*.csv")))
    idx, seen = 0, set()
    for p in sources:
        try:
            d = [int(x) for x in open(p).read().splitlines() if x.strip()]
        except ValueError:
            continue
        key = tuple(sorted(d))
        if len(d) == 60 and key not in seen:
            seen.add(key)
            shutil.copy(p, os.path.join(opp_dir, f"opp_{idx}.csv"))
            idx += 1
    print(f"相手候補デッキ {idx} 個を同梱（source={'replays' if real else 'data'}）")

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
    parser.add_argument(
        "--policy",
        choices=["ismcts", "nn"],
        default="ismcts",
        help="提出操縦（nn=floored NN-MCTS・--net 必須・提出環境に torch が必要）",
    )
    parser.add_argument(
        "--net", default=None, help="--policy nn で同梱する学習済み PVNet (.pt)"
    )
    parser.add_argument(
        "--floor-rollouts",
        type=int,
        default=8,
        help="接地 floor の rollout 数（nn のみ・0 で floor 無効）",
    )
    parser.add_argument(
        "--board-bonus",
        type=float,
        default=0.2,
        help="value への盤面補正 α（nn のみ・注入テストで 0.2 が最良 +0.075・0 で無効）",
    )
    parser.add_argument(
        "--leaf-rollouts",
        type=int,
        default=0,
        help="AlphaGo 型（§38）: 葉の価値を net でなく接地ロールアウト N 本の平均に（nn のみ）",
    )
    parser.add_argument(
        "--recycle-at",
        type=int,
        default=None,
        help="リサイクル強制手（§43）の発動閾値（nn のみ・未指定=既定15・閾値 A/B 用。例 25）",
    )
    parser.add_argument(
        "--fetch-priors",
        default=None,
        help="山札サーチ優先度 JSON（§47・mine_fetch_priorities.py の出力）を同梱する",
    )
    parser.add_argument(
        "--threat-bonus",
        type=float,
        default=0.0,
        help="value への KO 脅威注入 α（§48・nn のみ・0 で無効・α は eval で実測して決める）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.deck):
        raise SystemExit(f"デッキが見つかりません: {args.deck}（先にリーグを実行）")
    if args.policy == "nn" and not (args.net and os.path.exists(args.net)):
        raise SystemExit(f"--policy nn には学習済みネットが必要: --net {args.net}")
    if args.fetch_priors and not os.path.exists(args.fetch_priors):
        raise SystemExit(f"fetch-priors が見つかりません: {args.fetch_priors}")

    out_tar, names = build(
        args.deck,
        args.out,
        policy=args.policy,
        net_path=args.net,
        floor_rollouts=args.floor_rollouts,
        board_bonus=args.board_bonus,
        leaf_rollouts=args.leaf_rollouts,
        recycle_at=args.recycle_at,
        fetch_priors=args.fetch_priors,
        threat_bonus=args.threat_bonus,
    )
    size_mb = os.path.getsize(out_tar) / 1e6
    print(f"提出パッケージを作成: {out_tar} ({size_mb:.1f} MB・policy={args.policy})")
    print(f"  同梱物（root 直下）: {', '.join(names)}")
    print(
        "  ※ deck.csv と cg は Competition Data を含むため追跡外。Kaggle へはユーザーが提出。"
    )


if __name__ == "__main__":
    main()
