"""ISMCTS 再評価（Step 3）の軽量テスト."""

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

from cards import load_card_meta  # noqa: E402
from deck import load_deck, mutate  # noqa: E402
from reeval import deck_winrates, heuristic_factory, ismcts_factory  # noqa: E402


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return load_deck(DECK)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_deck_winrates_heuristic(deck, meta):
    """ヒューリスティック factory で min/mean/per_opp が妥当."""
    rng = random.Random(0)
    other = mutate(deck, deck, rng)
    res = deck_winrates(
        deck, [deck, other], heuristic_factory(meta), rng, games_per_opp=4
    )
    assert 0.0 <= res["min"] <= 1.0
    assert len(res["per_opp"]) == 1  # 自分自身は除外


def test_deck_winrates_ismcts(deck, meta):
    """ISMCTS factory（極小予算）でも勝率を返す."""
    rng = random.Random(0)
    other = mutate(deck, deck, rng)
    factory = ismcts_factory(meta, time_budget=0.02)
    res = deck_winrates(deck, [other], factory, rng, games_per_opp=2)
    assert 0.0 <= res["min"] <= 1.0
