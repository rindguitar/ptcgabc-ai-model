"""デッキ最適化（進化計算）の軽量テスト.

実行時間を抑えるため、小さな集団・世代・対戦数で「合法なデッキと妥当な適応度を返す」
ことだけを確認する（実際の最適化品質は別途オフラインで測る）。
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

from cards import load_card_meta  # noqa: E402
from deck import DECK_SIZE, is_legal, load_deck  # noqa: E402
from deckopt import evolve, fitness  # noqa: E402
from agents import make_heuristic_agent  # noqa: E402


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return load_deck(DECK)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_fitness_keys(deck, meta):
    """fitness が min/mean/per_opp を返し、値域が妥当."""
    agent = make_heuristic_agent(meta)
    rng = random.Random(0)
    f = fitness(deck, [deck], agent, rng, games_per_opp=6)
    assert set(f) == {"min", "mean", "per_opp"}
    assert 0.0 <= f["min"] <= 1.0
    assert len(f["per_opp"]) == 1


def test_evolve_returns_legal_deck(deck, meta):
    """evolve が 60枚・合法のデッキと適応度を返す（小規模）."""
    rng = random.Random(0)
    result = evolve(
        [deck],
        meta,
        rng=rng,
        pop_size=4,
        generations=2,
        games_per_opp=4,
        elite=2,
        mutations_per_child=1,
    )
    best = result["deck"]
    assert len(best) == DECK_SIZE
    assert is_legal(best, deck)
    assert "min" in result["fitness"]
    assert len(result["history"]) == 2
