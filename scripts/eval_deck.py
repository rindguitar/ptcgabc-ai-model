"""デッキの強さを「メタ群に対し ISMCTS 操縦で多めの試合」を回して確定判断する.

league 内部の評価は ISMCTS が遅く 6試合程度＝小サンプルで、最悪ケース(min-of-5)は
下振れバイアスも乗るため「本当に強くなったか」を判定できない。本スクリプトは試合数を
増やして per-opp / 平均 / 最悪を信頼できる精度で測る。チャンピオンと champions/ の
バックアップを比較し、league が実際にデッキを改善しているかを確認する用途。

ホスト(CPU)で動く。例:
    python scripts/eval_deck.py --deck models/champion_deck.csv --games 20
    python scripts/eval_deck.py --deck models/champions/champ_XXXX.csv --games 20
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from deckopt import _load_pool, default_opponent_paths  # noqa: E402
from harness import evaluate_decks, evaluate_decks_with_factory  # noqa: E402
from ismcts import make_ismcts_agent  # noqa: E402


def eval_deck_vs_meta(
    deck,
    meta,
    opps,
    rng,
    games,
    pilot="ismcts",
    time_budget=0.1,
    net="models/pvnet_distill_best.pt",
    nn_sims=64,
) -> dict:
    """deck を opps（メタ群）に対し評価し per_opp/最悪/平均を返す（gate からも使う）.

    pilot: ismcts（特性対応・遅い）/ nn（蒸留NN-MCTS・ISMCTS同等を高速・要 torch）/
    heuristic（速いが特性/効果を使わない）。
    """
    if pilot == "nn":
        # 蒸留 NN-MCTS：ISMCTS 同等の強さを ~1/4 時間で（torch/GPU・要 Docker）
        import torch

        from nn_eval import make_net_evaluator
        from nn_mcts import make_nn_mcts_agent
        from train import load_net

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _net = load_net(net, device)
        _ev = make_net_evaluator(_net, meta, device)

        def factory(my_deck, opp_deck):
            return make_nn_mcts_agent(
                meta,
                my_deck,
                opp_deck,
                evaluator=_ev,
                n_simulations=nn_sims,
                n_determinizations=2,
            )

        rates = [
            evaluate_decks_with_factory(deck, opp, factory, rng, games)["win_rate_a"]
            for opp in opps
        ]
    elif pilot == "ismcts":

        def factory(my_deck, opp_deck):
            return make_ismcts_agent(meta, my_deck, opp_deck, time_budget=time_budget)

        rates = [
            evaluate_decks_with_factory(deck, opp, factory, rng, games)["win_rate_a"]
            for opp in opps
        ]
    else:
        from agents import make_heuristic_agent

        agent = make_heuristic_agent(meta)
        rates = [
            evaluate_decks(deck, opp, agent, rng, games)["win_rate_a"] for opp in opps
        ]
    return {
        "per_opp": rates,
        "worst": min(rates),
        "mean": statistics.mean(rates),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="デッキ強さの確定評価（vs メタ・ISMCTS操縦）"
    )
    p.add_argument("--deck", default="models/champion_deck.csv", help="評価するデッキ")
    p.add_argument(
        "--meta",
        nargs="+",
        default=None,
        help="相手メタ群（未指定なら data/*.csv の60枚デッキ）",
    )
    p.add_argument("--games", type=int, default=20, help="相手1体あたりの試合数")
    p.add_argument(
        "--pilot",
        choices=["ismcts", "nn", "heuristic"],
        default="ismcts",
        help="操縦（ismcts=特性対応・遅い / nn=蒸留NN-MCTS高速・要torch / heuristic）",
    )
    p.add_argument("--time-budget", type=float, default=0.1, help="ismcts の1手秒")
    p.add_argument(
        "--net",
        default="models/pvnet_distill_best.pt",
        help="pilot=nn のとき使う訓練済みネット",
    )
    p.add_argument(
        "--nn-sims", type=int, default=64, help="pilot=nn の1手あたり MCTS 反復数"
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = load_card_meta()
    deck = load_deck(args.deck)
    meta_paths = args.meta or default_opponent_paths()
    opps = [d for d in _load_pool(meta_paths) if d != deck]
    if not opps:
        print("評価相手（メタ）が見つかりません")
        return
    rng = random.Random(args.seed)
    res = eval_deck_vs_meta(
        deck,
        meta,
        opps,
        rng,
        args.games,
        args.pilot,
        args.time_budget,
        net=args.net,
        nn_sims=args.nn_sims,
    )
    print(
        f"{args.deck} vs メタ{len(opps)}デッキ（{args.pilot}操縦・各{args.games}試合・席入替）:"
    )
    print(f"  per_opp = {[round(r, 3) for r in res['per_opp']]}")
    print(f"  最悪 = {res['worst']:.3f}  平均 = {res['mean']:.3f}")


if __name__ == "__main__":
    main()
