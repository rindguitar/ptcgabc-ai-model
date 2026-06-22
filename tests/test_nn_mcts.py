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
