"""ISMCTS 蒸留データ収集のテスト（torch 非依存・ホストで実行可）."""

import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DECK = os.path.join(ROOT, "data", "deck.csv")
sys.path.insert(0, SRC)

pytest.importorskip("cg.sim", reason="cabt Engine (cg) が見つからない")
if not os.path.exists(DECK):
    pytest.skip("data/deck.csv が無い", allow_module_level=True)

from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from distill import generate_ismcts_samples  # noqa: E402
from features import ACTION_FEAT_LEN, OBS_FEAT_LEN  # noqa: E402


def test_ismcts_distill_samples_format():
    """蒸留サンプルが Sample 形式（train() 互換）で、π は教師選択の one-hot."""
    meta = load_card_meta()
    deck = load_deck(DECK)
    rng = random.Random(0)
    samples = generate_ismcts_samples(meta, deck, 1, rng, time_budget=0.02)
    assert samples, "サンプルが1つ以上収集される"
    for s in samples:
        assert s.state.shape == (OBS_FEAT_LEN,)
        assert s.actions.shape == (len(s.pi), ACTION_FEAT_LEN)
        # π は教師が選んだ手の one-hot（合計1・最大1）
        assert abs(float(s.pi.sum()) - 1.0) < 1e-5
        assert float(s.pi.max()) == pytest.approx(1.0)
        assert np.count_nonzero(s.pi) == 1
        assert s.z in (0.0, 0.5, 1.0)


def test_ismcts_distill_accepts_multiple_decks():
    """複数デッキ（list[list[int]]）でも単一デッキでも収集できる（汎用 pilot 化）."""
    meta = load_card_meta()
    deck = load_deck(DECK)
    rng = random.Random(0)
    # 単一デッキ（後方互換）と複数デッキの両方が Sample を返す
    one = generate_ismcts_samples(meta, deck, 1, rng, time_budget=0.02)
    multi = generate_ismcts_samples(meta, [deck, deck], 2, rng, time_budget=0.02)
    assert one and multi
    assert all(s.state.shape == (OBS_FEAT_LEN,) for s in multi)
