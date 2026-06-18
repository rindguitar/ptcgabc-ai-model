"""ISMCTS エージェントのテスト（軽量・合法性のみ）.

勝率の検証は時間がかかるため CI では行わず、エージェントが決定点で
**合法な選択**を返すことだけを高速に確認する（強さは `make bench` で別途測る）。
"""

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
from harness import read_deck  # noqa: E402
from ismcts import make_ismcts_agent  # noqa: E402


def _advance_to_main(deck, rng, min_turn=3, max_steps=2000):
    """battle を進め、turn>=min_turn の最初の MAIN(>1択) 時点の observation を返す."""
    meta = load_card_meta()
    heuristic = make_heuristic_agent(meta)
    obs_dict, _ = battle_start(deck, deck)
    for _ in range(max_steps):
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            return None
        if obs.select is None:
            return None
        if (
            obs.select.type == SelectType.MAIN
            and obs.current
            and obs.current.turn >= min_turn
            and len(obs.select.option) > 1
        ):
            return obs
        obs_dict = battle_select(heuristic(obs, rng))
    return None


def test_ismcts_returns_legal_action():
    """ISMCTS が決定点で合法な選択（個数・範囲・重複なし）を返す."""
    deck = read_deck(DECK)
    rng = random.Random(0)
    meta = load_card_meta()
    agent = make_ismcts_agent(meta, deck, deck, time_budget=0.05)

    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None, "decision point に到達しなかった"
        sel = obs.select
        action = agent(obs, rng)
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
        assert len(set(action)) == len(action)
    finally:
        battle_finish()
