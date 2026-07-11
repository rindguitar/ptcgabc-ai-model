"""NN 誘導 MCTS（PUCT）骨格のテスト（ダミー評価器で機構を確認）."""

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DECK = os.path.join(ROOT, "data", "deck.csv")
sys.path.insert(0, SRC)

pytest.importorskip("cg.sim", reason="cabt Engine (cg) が見つからない")
if not os.path.exists(DECK):
    pytest.skip("data/deck.csv が無い", allow_module_level=True)

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import SelectType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from deck import load_deck  # noqa: E402
from nn_mcts import make_nn_mcts_agent, make_prize_evaluator  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def _advance_to_main(deck, rng, meta):
    heuristic = make_heuristic_agent(meta)
    obs_dict, _ = battle_start(deck, deck)
    for _ in range(2000):
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            return None
        if obs.select is None:
            return None
        if (
            obs.select.type == SelectType.MAIN
            and obs.current.turn >= 3
            and len(obs.select.option) > 1
        ):
            return obs
        obs_dict = battle_select(heuristic(obs, rng))
    return None


def test_prize_evaluator_shapes(meta):
    """ダミー評価器が value[0,1] と option 整列の priors を返す."""
    deck = load_deck(DECK)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        value, priors = make_prize_evaluator(meta)(obs)
        assert 0.0 <= value <= 1.0
        assert len(priors) == len(obs.select.option)
        assert abs(sum(priors) - 1.0) < 1e-6
    finally:
        battle_finish()


def test_nn_mcts_returns_legal_action(meta):
    """PUCT エージェント（ダミー評価器）が決定点で合法な選択を返す."""
    deck = load_deck(DECK)
    agent = make_nn_mcts_agent(meta, deck, deck, n_simulations=24, n_determinizations=2)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        sel = obs.select
        action = agent(obs, random.Random(0))
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
        assert len(set(action)) == len(action)
    finally:
        battle_finish()


def test_plan_sims_budget_scaling():
    """適応 sims（§31）: 未計測は床・予算が厚いほど増え・cap で頭打ち・枯渇で床."""
    from nn_mcts import plan_sims

    base, cap = 64, 512
    # 未計測（初手）は床
    assert plan_sims(base, cap, 500.0, 0, None, 2) == base
    # 単価 5ms/unit・残り 500s・序盤（残り決定 ~40）→ 1手 ~12.5s ÷ (0.005×2dets) = 1250 → cap
    assert plan_sims(base, cap, 500.0, 0, 0.005, 2) == cap
    # 単価が重い（50ms/unit）→ 125 sims（床と cap の間で予算に比例）
    assert plan_sims(base, cap, 500.0, 0, 0.05, 2) == 125
    # 予算枯渇 → 床（品質を従来から下げない）
    assert plan_sims(base, cap, 0.0, 30, 0.005, 2) == base
    assert plan_sims(base, cap, -5.0, 30, 0.005, 2) == base
    # 残り決定数の床: 終盤（moves≫40）でも1手に全残額を注がない
    late = plan_sims(base, cap, 80.0, 60, 0.05, 2)
    assert late == min(cap, int(80.0 / 8.0 / 0.1))  # 残り決定の床=8 で配分


def test_adaptive_agent_returns_legal_and_tracks(meta):
    """game_budget 付き agent が合法手を返す（ダミー評価器・時計/EMA 経路の煙）."""
    deck = load_deck(DECK)
    agent = make_nn_mcts_agent(
        meta,
        deck,
        deck,
        n_simulations=8,
        n_determinizations=1,
        game_budget=540.0,
    )
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        sel = obs.select
        for _ in range(2):  # 2手目で EMA 経路（unit_ema 有り）も通す
            action = agent(obs, random.Random(0))
            assert sel.minCount <= len(action) <= sel.maxCount
            assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()
