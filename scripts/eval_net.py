"""訓練済み NN 操縦の強さ評価（Phase 3・torch・Docker）.

同一デッキで NN-MCTS(--net で指定したネット) vs 相手（heuristic / ismcts）を対戦させ、勝率を測る。
これで「訓練で操縦が強くなったか」を確認する。席入替で先手有利を打ち消す。

実行（Docker）:
    make exec CMD="python scripts/eval_net.py --games 20 --sims 64"
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch  # noqa: E402

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from harness import evaluate  # noqa: E402
from nn_eval import make_net_evaluator  # noqa: E402
from nn_mcts import make_nn_mcts_agent  # noqa: E402
from train import load_net  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="訓練済み NN 操縦の強さ評価")
    p.add_argument("--net", default="models/pvnet.pt", help="訓練済みネット")
    p.add_argument(
        "--deck", default="models/champion_deck.csv", help="評価に使うデッキ"
    )
    p.add_argument(
        "--vs", default="heuristic", choices=["heuristic", "ismcts"], help="相手の操縦"
    )
    p.add_argument("--games", type=int, default=20, help="試合数")
    p.add_argument("--sims", type=int, default=64, help="NN-MCTS の1手反復")
    p.add_argument("--dets", type=int, default=2, help="determinization 数")
    p.add_argument("--time-budget", type=float, default=0.3, help="相手 ISMCTS の1手秒")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_card_meta()
    deck = load_deck(args.deck)

    net = load_net(args.net, device)
    evaluator = make_net_evaluator(net, meta, device)
    nn_agent = make_nn_mcts_agent(
        meta,
        deck,
        deck,
        evaluator=evaluator,
        n_simulations=args.sims,
        n_determinizations=args.dets,
    )

    if args.vs == "heuristic":
        opponent = make_heuristic_agent(meta)
    else:
        from ismcts import make_ismcts_agent

        opponent = make_ismcts_agent(meta, deck, deck, time_budget=args.time_budget)

    rng = random.Random(args.seed)
    res = evaluate(nn_agent, opponent, deck, rng, args.games)
    # どの .pt を測ったか曖昧にしないよう実際のネット名を表示する
    net_name = os.path.basename(args.net)
    print(
        f"NN-MCTS({net_name}) vs {args.vs}（同一デッキ・{args.games}試合・席入替, device={device}）: "
        f"NN 勝率 = {res['win_rate_a']:.3f}  ({res['wins_a']}-{res['wins_b']}, draws {res['draws']})"
    )
    print("  >0.5 なら NN 操縦が相手より強い（訓練が効いている）")


if __name__ == "__main__":
    main()
