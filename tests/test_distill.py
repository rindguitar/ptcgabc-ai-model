"""ISMCTS 蒸留データ収集のテスト（torch 非依存・ホストで実行可）."""

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
from deck import load_deck  # noqa: E402
from distill import generate_ismcts_samples  # noqa: E402
from features import ACTION_FEAT_LEN, OBS_FEAT_LEN  # noqa: E402


def test_ismcts_distill_samples_format():
    """蒸留サンプルが Sample 形式（train() 互換）で、π は確率分布（既定 temp=0＝one-hot）."""
    meta = load_card_meta()
    deck = load_deck(DECK)
    rng = random.Random(0)
    # 十分な訪問数が要るので少し長めの予算で確実にサンプルを得る
    samples = generate_ismcts_samples(meta, deck, 1, rng, time_budget=0.1)
    assert samples, "サンプルが1つ以上収集される"
    for s in samples:
        assert s.state.shape == (OBS_FEAT_LEN,)
        assert s.actions.shape == (len(s.pi), ACTION_FEAT_LEN)
        # π は確率分布（合計1・各要素 [0,1]）。既定 temp=0 は最多訪問手の one-hot
        assert abs(float(s.pi.sum()) - 1.0) < 1e-5
        assert float(s.pi.min()) >= 0.0
        assert float(s.pi.max()) <= 1.0 + 1e-6
        assert int((s.pi > 0).sum()) == 1  # 既定は one-hot
        assert s.z in (0.0, 0.5, 1.0)


def test_distill_temperature_softens_pi():
    """temp>0 で π が訪問分布(soft)に、temp=0(既定)は one-hot になる（温度1本化）."""
    meta = load_card_meta()
    deck = load_deck(DECK)
    hard = generate_ismcts_samples(
        meta, deck, 1, random.Random(0), time_budget=0.1, temp=0.0
    )
    soft = generate_ismcts_samples(
        meta, deck, 1, random.Random(0), time_budget=0.1, temp=1.0
    )
    assert hard and soft
    assert all(int((s.pi > 0).sum()) == 1 for s in hard), "temp=0 は one-hot"
    assert any(int((s.pi > 0).sum()) > 1 for s in soft), "temp=1 は2手以上に分散"


def test_ismcts_distill_accepts_multiple_decks():
    """複数デッキ（list[list[int]]）でも単一デッキでも収集できる（汎用 pilot 化）."""
    meta = load_card_meta()
    deck = load_deck(DECK)
    rng = random.Random(0)
    # 単一デッキ（後方互換）と複数デッキの両方が Sample を返す
    one = generate_ismcts_samples(meta, deck, 1, rng, time_budget=0.1)
    multi = generate_ismcts_samples(meta, [deck, deck], 2, rng, time_budget=0.1)
    assert one and multi
    assert all(s.state.shape == (OBS_FEAT_LEN,) for s in multi)
