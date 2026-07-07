"""訓練済み NN 操縦の強さ評価（Phase 3・torch・Docker）.

同一デッキで NN-MCTS(--net で指定したネット) vs 相手（heuristic / ismcts）を対戦させ、勝率を測る。
これで「訓練で操縦が強くなったか」を大まかに確認する。席入替で先手有利を打ち消す。

⚠️ **net A/B（どちらのネットが強いか）の確定判断にこれ（特に --vs ismcts のミラー）を使わないこと**。
同一デッキ・単一相手のミラーは非中立リファレンスで、注入等との相互作用で差を過大/過小評価する
（design-decisions §25 訂正: ミラーで operative 0.675>>replay 0.500 が実メタでは同点だった実例）。
net の比較は **eval_deck.py（eval-deck）で実メタ相手プールに対して**行う（外部基準・非ミラー）。

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
from nn_eval import make_net_evaluator, wrap_board_bonus  # noqa: E402
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
    p.add_argument(
        "--floor-rollouts",
        type=int,
        default=0,
        help="接地安全弁の rollout 数（>0 で NN 手 vs heuristic 手を実地比較し pilot≥heuristic 保証）",
    )
    p.add_argument(
        "--board-bonus",
        type=float,
        default=0.0,
        help="value への盤面補正の注入 α（v2.3 事前検証用・0 で無効。例 0.1）",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_card_meta()
    deck = load_deck(args.deck)

    net = load_net(args.net, device)
    evaluator = make_net_evaluator(net, meta, device)
    if args.board_bonus:
        evaluator = wrap_board_bonus(evaluator, args.board_bonus)
    nn_agent = make_nn_mcts_agent(
        meta,
        deck,
        deck,
        evaluator=evaluator,
        n_simulations=args.sims,
        n_determinizations=args.dets,
        floor_rollouts=args.floor_rollouts,
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
    bonus = f"+board{args.board_bonus}" if args.board_bonus else ""
    print(
        f"NN-MCTS({net_name}{bonus}) vs {args.vs}（同一デッキ・{args.games}試合・席入替, device={device}）: "
        f"NN 勝率 = {res['win_rate_a']:.3f}  ({res['wins_a']}-{res['wins_b']}, draws {res['draws']})"
    )
    print("  >0.5 なら NN 操縦が相手より強い（訓練が効いている）")


if __name__ == "__main__":
    main()
