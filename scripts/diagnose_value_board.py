"""value の「盤面資源（ベンチ切れ）感度」を2つの net で比較診断する.

実戦（Kaggle）の勝敗はベンチ切れが支配的（敗因の6〜7割）なのに、訓練環境が旧メタだと
value が「相手の場が薄い＝勝ちに近い」を学べていない疑いがある。本スクリプトは:
  - heuristic で対戦を進める（net 非依存＝**両 net に同一局面列**を見せる完全ペア比較）
  - 各 MAIN/ATTACK 局面で「相手/自分の場の駒数（active+bench）」と両 net の value を記録
  - 駒数バケツ別の平均 value と、**盤面感度**（相手が残1体のときの value − 3体以上のときの value）
    を net ごとに出す。感度が大きい net ほど「盤面を枯らす勝ち筋」を理解している。

実行（Docker）:
    make exec CMD="python scripts/diagnose_value_board.py \
        --net-a models/pvnet_seed.pt --net-b models/pvnet_improve_best.pt"
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import torch  # noqa: E402

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from deckopt import _load_pool  # noqa: E402
from nn_eval import make_net_evaluator  # noqa: E402
from nn_mcts import _MCTS_SELECT_TYPES  # noqa: E402
from train import load_net, load_net_warmstart  # noqa: E402


def _board(p) -> int:
    """場の駒数（active + bench）."""
    return len([a for a in (p.active or []) if a]) + len(p.bench or [])


def _bucket(n: int) -> str:
    return "1" if n <= 1 else ("2" if n == 2 else "3+")


def main() -> None:
    p = argparse.ArgumentParser(description="value の盤面資源感度を2 net で比較")
    p.add_argument("--net-a", default="models/pvnet_seed.pt", help="旧 net")
    p.add_argument("--net-b", default="models/pvnet_improve_best.pt", help="新 net")
    p.add_argument("--deck", nargs="+", default=None)
    p.add_argument("--decisions", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_card_meta()
    paths = (
        args.deck
        or sorted(glob.glob("models/gauntlet/*.csv"))
        or sorted(glob.glob("data/*.csv"))
    )
    decks = _load_pool(paths)[:4]
    rng = random.Random(args.seed)

    def _load(path: str):
        """旧特徴次元の net は warm-start（新列ゼロ埋め）で読む＝特徴拡張を跨いだ比較を可能に."""
        try:
            return load_net(path, device)
        except RuntimeError:
            return load_net_warmstart(path, device)

    evs = {
        "A(旧)": make_net_evaluator(_load(args.net_a), meta, device),
        "B(新)": make_net_evaluator(_load(args.net_b), meta, device),
    }
    heuristic = make_heuristic_agent(meta)

    # sums[net名][(相手駒バケツ)] = [value合計, 件数]（自分駒数も同様に）
    opp_sums: dict[str, dict[str, list[float]]] = {
        k: defaultdict(lambda: [0.0, 0]) for k in evs
    }
    my_sums: dict[str, dict[str, list[float]]] = {
        k: defaultdict(lambda: [0.0, 0]) for k in evs
    }

    n = 0
    deck_i = 0
    while n < args.decisions:
        deck = decks[deck_i % len(decks)]
        deck_i += 1
        obs_dict, _ = battle_start(deck, deck)
        try:
            for _ in range(100_000):
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    break
                sel = obs.select
                if sel is None:
                    break
                if sel.type in _MCTS_SELECT_TYPES and len(sel.option) > 1:
                    st = obs.current
                    yi = st.yourIndex
                    mine, opp = st.players[yi], st.players[1 - yi]
                    ob, mb = _bucket(_board(opp)), _bucket(_board(mine))
                    for name, ev in evs.items():
                        v, _pri = ev(obs)
                        opp_sums[name][ob][0] += v
                        opp_sums[name][ob][1] += 1
                        my_sums[name][mb][0] += v
                        my_sums[name][mb][1] += 1
                    n += 1
                    if n >= args.decisions:
                        break
                obs_dict = battle_select(heuristic(obs, rng))
        finally:
            battle_finish()

    print(f"=== value の盤面感度（{n} 局面・同一局面のペア比較） ===")
    print("相手の場の駒数別 平均value（高いほど「勝ちに近い」と評価）:")
    print(f"{'net':6} {'相手1体':>8} {'相手2体':>8} {'相手3+':>8}  盤面感度(1体−3+)")
    for name in evs:
        row = []
        for b in ("1", "2", "3+"):
            s, c = opp_sums[name][b]
            row.append(s / c if c else float("nan"))
        sens = row[0] - row[2]
        print(f"{name:6} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}  {sens:+.3f}")
    print("\n自分の場の駒数別 平均value（低いほど「自分が危ない」と評価）:")
    print(f"{'net':6} {'自分1体':>8} {'自分2体':>8} {'自分3+':>8}  危険感度(3+−1体)")
    for name in evs:
        row = []
        for b in ("1", "2", "3+"):
            s, c = my_sums[name][b]
            row.append(s / c if c else float("nan"))
        sens = row[2] - row[0]
        print(f"{name:6} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}  {sens:+.3f}")
    print(
        "\n読み方: 盤面感度/危険感度が大きい net ほど「盤面を枯らす/枯らされる」勝敗構造を理解。"
        "\n  B(新・混合プール) ≫ A(旧) なら混合訓練は成功（続ける価値あり）。差が無ければ打ち切り。"
    )


if __name__ == "__main__":
    main()
