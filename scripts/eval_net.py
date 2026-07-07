"""訓練済み NN 操縦の強さ評価（torch・Docker）.

**既定（--vs meta）**: NN-MCTS(--net) が --deck を操縦し、**実メタ相手プール（非ミラー）**に対して
勝率を測る＝net の確定判断に使える外部基準。相手プールは models/gauntlet/（実メタ抽出・
`make gauntlet-real`）> data/replays/opp_decks/（実 replay 抽出）> data/*.csv の順で選ぶ。
per_opp / 最悪 / 平均を返す（eval_deck_vs_meta を共用）。net の A/B は --net を差し替えて
同一 --deck・同一プールで2回回して比べる。

--vs heuristic / ismcts: 同一デッキのミラー対戦（**速い大まかな確認のみ**）。
⚠️ **net A/B の確定判断にミラー（特に --vs ismcts）を使わないこと**。同一デッキ・単一相手の
ミラーは非中立リファレンスで、注入等との相互作用で差を過大/過小評価する（design-decisions §25
訂正: ミラーで operative 0.675>>replay 0.500 が実メタでは同点だった実例）。だから既定を meta にした。

実行（Docker）:
    make eval-net EVAL_VS=meta EVAL_NET=models/pvnet_operative.pt
    make eval-net EVAL_VS=meta EVAL_NET=models/pvnet_replay.pt   # net A/B
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
        "--vs",
        default="meta",
        choices=["meta", "heuristic", "ismcts"],
        help="meta=実メタ相手プール（非ミラー・net 判定の外部基準・既定）／"
        "heuristic・ismcts=同一デッキのミラー（大まかな確認のみ）",
    )
    p.add_argument(
        "--opp-glob",
        default=None,
        help="meta 時の相手プール glob（既定: gauntlet/ > replays/opp_decks/ > data/*.csv）",
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
    rng = random.Random(args.seed)
    net_name = os.path.basename(args.net)
    bonus = f"+board{args.board_bonus}" if args.board_bonus else ""

    if args.vs == "meta":
        # 実メタ相手プール（非ミラー）に対する net 判定＝外部基準（design-decisions §25）。
        # eval_deck_vs_meta を共用（提出と同じ floored NN＋盤面補正で判定できる）。
        import glob

        from deckopt import _load_pool
        from evaluation import eval_deck_vs_meta

        if args.opp_glob:
            paths = sorted(glob.glob(args.opp_glob))
        else:  # 実メタ優先: gauntlet(実メタ抽出) > replay 生抽出 > 公式サンプル
            paths = next(
                (
                    ps
                    for pat in (
                        "models/gauntlet/*.csv",
                        "data/replays/opp_decks/*.csv",
                        "data/*.csv",
                    )
                    if (ps := sorted(glob.glob(pat)))
                ),
                [],
            )
        opps = _load_pool(paths)
        if not opps:
            raise SystemExit("相手プールが空（先に make gauntlet-real か replay 抽出）")
        res = eval_deck_vs_meta(
            deck,
            meta,
            opps,
            rng,
            args.games,
            pilot="nn",
            net=args.net,
            nn_sims=args.sims,
            floor_rollouts=args.floor_rollouts,
            board_bonus=args.board_bonus,
        )
        floor = f"+floor{args.floor_rollouts}" if args.floor_rollouts else ""
        print(
            f"NN-MCTS({net_name}{bonus}{floor}) vs 実メタ{len(opps)}デッキ"
            f"（非ミラー・各{args.games}試合・席入替, device={device}）:"
        )
        print(f"  per_opp = {[round(r, 3) for r in res['per_opp']]}")
        print(f"  最悪 = {res['worst']:.3f}  平均 = {res['mean']:.3f}")
        print("  net の A/B は --net を替えて同一 --deck・同一プールで2回回して比べる（§25）")
        return

    # --- ミラー（heuristic / ismcts）: 大まかな確認のみ・net A/B には使わない ---
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

    res = evaluate(nn_agent, opponent, deck, rng, args.games)
    print(
        f"NN-MCTS({net_name}{bonus}) vs {args.vs}（同一デッキ・ミラー・{args.games}試合・席入替, "
        f"device={device}）: "
        f"NN 勝率 = {res['win_rate_a']:.3f}  ({res['wins_a']}-{res['wins_b']}, draws {res['draws']})"
    )
    print("  ⚠️ ミラーは大まかな確認のみ。net A/B の確定判断は --vs meta で（§25）")


if __name__ == "__main__":
    main()
