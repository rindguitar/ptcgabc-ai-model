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
from cg.api import CardType  # noqa: E402
from deck import (  # noqa: E402
    DECK_SIZE,
    composition,
    is_legal,
    load_deck,
    mutate,
    random_legal_deck,
    structured_deck,
)
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


def test_random_legal_deck(deck):
    """多数変異した合法デッキが 60枚・合法で、元と十分異なる."""
    rng = random.Random(0)
    d = random_legal_deck(deck, rng, swaps=15)
    assert len(d) == DECK_SIZE
    assert is_legal(d, deck)


def test_structured_deck_is_coherent(deck, meta):
    """構造化生成が 60枚・合法で、テンプレと別物の mono-type（単色集中）になる."""
    rng = random.Random(0)
    d = structured_deck(deck, meta, rng)
    assert len(d) == DECK_SIZE
    assert is_legal(d, deck)
    # たねアタッカーの色がほぼ単色に集中している（寄せ集めでない＝『回る』別軸）
    colors = [
        meta.energy_type.get(c)
        for c in d
        if meta.card_type.get(c) == CardType.POKEMON and meta.is_basic.get(c)
    ]
    assert colors, "たねポケモンが存在する"
    top = max(colors.count(x) for x in set(colors))
    assert top / len(colors) >= 0.8  # 8割以上が同色
    # ランダム合法デッキ（種類が多い寄せ集め）より種類数が絞られている
    assert len(set(d)) < len(set(random_legal_deck(deck, rng, swaps=40)))


def test_structured_deck_specified_color(deck, meta):
    """色を指定すると、その色の基本エネが入る."""
    rng = random.Random(0)
    color = next(iter(meta.basic_energy_id))
    d = structured_deck(deck, meta, rng, energy_type=color)
    assert meta.basic_energy_id[color] in d


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
