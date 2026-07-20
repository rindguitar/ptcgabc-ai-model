"""デッキ × 操縦の評価ヘルパ（eval-deck / champion-gate 共用）.

以前は scripts/eval_deck.py にあったが、champion_gate からの import が「スクリプトが
スクリプトを import する」形になっていたため src/ へ移設（役割集中の解消）。

pilot は**提出と同じ構成**（floored NN＋盤面補正）で判定できる: floor_rollouts で接地 floor、
board_bonus で value への盤面補正（[nn_eval.wrap_board_bonus]）を注入する。
torch は pilot="nn" のときのみ遅延 import（ホストの ismcts/heuristic 評価では不要）。
"""

from __future__ import annotations

import statistics

from harness import evaluate_decks, evaluate_decks_with_factory
from ismcts import make_ismcts_agent


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
    floor_rollouts=0,
    board_bonus=0.0,
    threat_bonus=0.0,
) -> dict:
    """deck を opps（メタ群）に対し評価し per_opp/最悪/平均を返す（gate からも使う）.

    pilot: ismcts（特性対応・遅い）/ nn（蒸留NN-MCTS・ISMCTS同等を高速・要 torch）/
    heuristic（速いが特性/効果を使わない）。
    floor_rollouts>0 は nn に接地 floor、board_bonus>0 は盤面補正、threat_bonus>0 は
    KO 脅威注入（§48）＝**提出と同じ操縦で判定**する。
    """
    if pilot == "nn":
        # 蒸留 NN-MCTS：ISMCTS 同等の強さを ~1/4 時間で（torch/GPU・要 Docker）
        import torch

        from nn_eval import make_net_evaluator, wrap_board_bonus, wrap_threat_bonus
        from nn_mcts import make_nn_mcts_agent
        from train import load_net

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _net = load_net(net, device)
        _ev = make_net_evaluator(_net, meta, device)
        if board_bonus:
            _ev = wrap_board_bonus(_ev, board_bonus)
        if threat_bonus:
            _ev = wrap_threat_bonus(_ev, meta, threat_bonus)

        def factory(my_deck, opp_deck):
            return make_nn_mcts_agent(
                meta,
                my_deck,
                opp_deck,
                evaluator=_ev,
                n_simulations=nn_sims,
                n_determinizations=2,
                floor_rollouts=floor_rollouts,
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
