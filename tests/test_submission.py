"""Kaggle 提出アダプタのテスト."""

import os
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
from submission import make_kaggle_agent  # noqa: E402


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return load_deck(DECK)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_returns_deck_on_initial_selection(deck, meta):
    """初手（select is None）でデッキ60枚を返す."""
    agent = make_kaggle_agent("heuristic", deck=deck, meta=meta)
    out = agent({"select": None, "logs": [], "current": None})
    assert out == deck


def test_in_game_returns_legal_selection(deck, meta):
    """対戦中の obs_dict に対し合法な選択を返す（公式形式）."""
    import random

    agent = make_kaggle_agent("heuristic", deck=deck, meta=meta)
    heuristic = make_heuristic_agent(meta)
    rng = random.Random(0)

    obs_dict, _ = battle_start(deck, deck)
    try:
        for _ in range(2000):
            obs = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                break
            if obs.select is None:
                break
            if obs.select.type == SelectType.MAIN and len(obs.select.option) > 1:
                action = agent(obs_dict)  # 公式形式の呼び出し（生 dict を渡す）
                sel = obs.select
                assert sel.minCount <= len(action) <= sel.maxCount
                assert all(0 <= i < len(sel.option) for i in action)
                return
            obs_dict = battle_select(heuristic(obs, rng))
    finally:
        battle_finish()
