"""デッキ基盤（合法性・変異・構成・デッキ対デッキ評価）のテスト."""

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
from deck import DECK_SIZE, composition, is_legal, load_deck, mutate  # noqa: E402
from harness import evaluate_decks  # noqa: E402


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return load_deck(DECK)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_sample_deck_is_legal(deck):
    """サンプルデッキは合法."""
    assert is_legal(deck, deck)


def test_illegal_decks(deck):
    """枚数違反・ポケモン不在のデッキは非合法."""
    assert not is_legal(deck[:59], deck)  # 59 枚
    assert not is_legal([1] * DECK_SIZE, deck)  # 基本エネ60枚（たね不在）


def test_mutate_keeps_legal(deck):
    """変異後も 60 枚で合法."""
    rng = random.Random(0)
    mutated = mutate(deck, deck, rng)
    assert len(mutated) == DECK_SIZE
    assert is_legal(mutated, deck)


def test_composition_sums_to_60(deck, meta):
    """構成カウントの種別合計が 60."""
    comp = composition(deck, meta)
    total = sum(comp[k] for k in comp if k != "unique")
    assert total == DECK_SIZE


def test_evaluate_decks_self(deck, meta):
    """同一デッキ同士の対戦は集計が整合する（勝率は概ね五分）."""
    rng = random.Random(0)
    agent = make_heuristic_agent(meta)
    res = evaluate_decks(deck, deck, agent, rng, games=10)
    assert res["wins_a"] + res["wins_b"] + res["draws"] == 10
    assert 0.0 <= res["win_rate_a"] <= 1.0
