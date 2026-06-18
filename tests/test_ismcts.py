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
from ismcts import _move_budget, make_ismcts_agent  # noqa: E402


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


def test_move_budget_never_exceeds_remaining():
    """1手予算は (残り - reserve) を超えず、下限・上限に収まる."""
    lo, hi, reserve, frac = 0.05, 20.0, 5.0, 0.08
    for remaining in [600, 100, 10, 5, 3, 0]:
        b = _move_budget(remaining, frac, lo, hi, reserve)
        assert b <= max(0.0, remaining - reserve) + 1e-9  # 予備時間を必ず残す
        assert b <= hi
        if remaining - reserve >= lo:
            assert b >= lo
    # 残りが reserve 以下なら予算 0（即ヒューリスティック）
    assert _move_budget(4.0, frac, lo, hi, reserve) == 0.0


def test_ismcts_clock_mode_legal_action():
    """クロック管理モードでも合法な選択を返す."""
    deck = read_deck(DECK)
    rng = random.Random(0)
    meta = load_card_meta()
    agent = make_ismcts_agent(meta, deck, deck, game_budget=2.0, max_move_budget=0.1)

    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None
        sel = obs.select
        action = agent(obs, rng)
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()
