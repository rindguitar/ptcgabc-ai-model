"""Phase 1 baseline のテスト.

cabt Engine（`cg`）と `data/deck.csv` はいずれも Competition Data（追跡外）。
これらが無い環境ではテストモジュールごとスキップする。
"""

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DECK = os.path.join(ROOT, "data", "deck.csv")
sys.path.insert(0, SRC)

# 提供物が無ければスキップ（import 時にネイティブ lib をロードするため importorskip を使う）。
pytest.importorskip("cg.sim", reason="cabt Engine (cg) が見つからない")
if not os.path.exists(DECK):
    pytest.skip("data/deck.csv が無い", allow_module_level=True)

from agents import make_heuristic_agent, random_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from harness import evaluate, play_match, read_deck  # noqa: E402


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return read_deck(DECK)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_deck_is_60(deck):
    """deck.csv が 60 枚として読める."""
    assert len(deck) == 60


def test_match_completes(deck):
    """1 試合がランダム同士で完走し、妥当な result を返す."""
    rng = random.Random(0)
    out = play_match(random_agent, random_agent, deck, deck, rng)
    assert out["result"] in (0, 1, 2)
    assert out["steps"] >= 1


def test_heuristic_beats_random(deck, meta):
    """ヒューリスティックがランダムに有意に勝ち越す（baseline 検証）."""
    rng = random.Random(0)
    heuristic = make_heuristic_agent(meta)
    res = evaluate(heuristic, random_agent, deck, rng, games=60, alternate=True)
    assert res["win_rate_a"] > 0.6
