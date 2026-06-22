"""self-play データ収集のテスト（ダミー評価器・ホストで実行可）."""

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
from features import ACTION_FEAT_LEN, OBS_FEAT_LEN  # noqa: E402
from nn_mcts import make_prize_evaluator  # noqa: E402
from selfplay import generate_samples, play_selfplay_game  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_selfplay_samples_shapes(meta):
    """自己対戦で妥当な形状のサンプルが取れる."""
    deck = load_deck(DECK)
    evaluator = make_prize_evaluator(meta)
    samples = play_selfplay_game(
        meta,
        deck,
        evaluator,
        random.Random(0),
        n_simulations=8,
        n_determinizations=1,
    )
    assert len(samples) > 0
    for s in samples:
        assert s.state.shape == (OBS_FEAT_LEN,)
        assert s.actions.shape[1] == ACTION_FEAT_LEN
        assert s.actions.shape[0] == s.pi.shape[0]  # 合法手数と π が一致
        assert abs(float(s.pi.sum()) - 1.0) < 1e-5
        assert s.z in (0.0, 0.5, 1.0)
        assert np.all(np.isfinite(s.state))


def test_generate_samples_multiple_games(meta):
    """複数試合分のサンプルを連結して返す."""
    deck = load_deck(DECK)
    samples = generate_samples(
        meta,
        deck,
        None,
        n_games=2,
        rng=random.Random(1),
        n_simulations=8,
        n_determinizations=1,
    )
    assert len(samples) > 0
