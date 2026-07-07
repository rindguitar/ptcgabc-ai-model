"""相手デッキ推定（pick_opponent_deck のベイズ重み付け）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組み、重み付けロジックを直接検証する。
カード ID はダミー整数（規約: Pokémon Elements を持ち込まない）。
"""

import os
import random
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from determinize import pick_opponent_deck  # noqa: E402


def _obs(discard_ids: list[int], your_index: int = 0):
    """相手の捨て札だけを可視にした最小観測を作る（他ゾーンは空）."""
    opp = NS(
        hand=[],
        discard=[NS(id=c) for c in discard_ids],
        active=[],
        bench=[],
        prize=[],
    )
    me = NS(hand=[], discard=[], active=[], bench=[], prize=[])
    players = [None, None]
    players[your_index] = me
    players[1 - your_index] = opp
    return NS(current=NS(yourIndex=your_index, players=players))


# アーキタイプ A（カード 1,2 中心）と B（カード 3,4 中心）
_DECK_A = [1] * 30 + [2] * 30
_DECK_B = [3] * 30 + [4] * 30


def test_matching_deck_dominates():
    """相手可視札を完全に説明する候補が、しない候補より圧倒的に多く選ばれる."""
    rng = random.Random(0)
    obs = _obs([1, 1, 2])  # A なら misses=0 / B なら misses=3
    counts = {"A": 0, "B": 0}
    for _ in range(400):
        d = pick_opponent_deck(obs, [_DECK_A, _DECK_B], _DECK_A, rng)
        counts["A" if d is _DECK_A else "B"] += 1
    assert counts["A"] > counts["B"] * 20  # e^-6 差 → ほぼ A


def test_no_evidence_is_uniform():
    """可視札が無い（初手など）と事前分布のまま＝両候補とも現れる."""
    rng = random.Random(1)
    obs = _obs([])
    seen = set()
    for _ in range(50):
        d = pick_opponent_deck(obs, [_DECK_A, _DECK_B], _DECK_A, rng)
        seen.add("A" if d is _DECK_A else "B")
    assert seen == {"A", "B"}


def test_partial_match_beats_fallback():
    """完全一致候補が無くても、最も近い実デッキへ縮退しミラーへ崩れない."""
    rng = random.Random(2)
    # どの候補も 1 枚は説明できない（A は 5 を、B は 1,2 を持たない）が、A の方が近い
    obs = _obs([1, 1, 2, 5])  # A: misses=1（5 のみ）/ B: misses=3
    fallback = [9] * 60  # ミラー相当（無関係）
    counts = {"A": 0, "B": 0, "fallback": 0}
    for _ in range(400):
        d = pick_opponent_deck(obs, [_DECK_A, _DECK_B], fallback, rng)
        key = "A" if d is _DECK_A else "B" if d is _DECK_B else "fallback"
        counts[key] += 1
    assert counts["fallback"] == 0  # プールがある限りミラーには落ちない
    assert counts["A"] > counts["B"]  # より近いアーキタイプが優先


def test_no_candidates_returns_fallback():
    """候補プールが無ければ従来どおり fallback（ミラー仮定）を返す."""
    rng = random.Random(3)
    obs = _obs([1, 2])
    fallback = [7] * 60
    assert pick_opponent_deck(obs, None, fallback, rng) is fallback
    assert pick_opponent_deck(obs, [], fallback, rng) is fallback
