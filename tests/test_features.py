"""観測特徴量エンコーダ（Phase 3 土台）のテスト."""

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

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import SelectType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from deck import load_deck  # noqa: E402
from features import (  # noqa: E402
    ACTION_FEAT_LEN,
    HAND_FEAT,
    OBS_FEAT_LEN,
    encode_actions,
    encode_observation,
    observation_feature_size,
)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def _advance_to_main(deck, rng, meta):
    """turn>=3 の最初の MAIN(>1択) 時点の observation を返す."""
    heuristic = make_heuristic_agent(meta)
    obs_dict, _ = battle_start(deck, deck)
    for _ in range(2000):
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            return None
        if obs.select is None:
            return None
        if (
            obs.select.type == SelectType.MAIN
            and obs.current.turn >= 3
            and len(obs.select.option) > 1
        ):
            return obs
        obs_dict = battle_select(heuristic(obs, rng))
    return None


def test_observation_encoding_shape_and_finite(meta):
    """encode_observation が固定長・有限値を返す."""
    deck = load_deck(DECK)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        vec = encode_observation(obs, meta)
        assert vec.shape == (OBS_FEAT_LEN,)
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec))
        assert observation_feature_size() == OBS_FEAT_LEN
    finally:
        battle_finish()


def test_action_encoding_matches_options(meta):
    """encode_actions が (手数 × ACTION_FEAT_LEN) を返す."""
    deck = load_deck(DECK)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        mat = encode_actions(obs, meta)
        assert mat.shape == (len(obs.select.option), ACTION_FEAT_LEN)
        assert np.all(np.isfinite(mat))
        # one-hot 部分は各行ちょうど1つ立つ（既知の OptionType の範囲なら）
        assert np.all(mat[:, :17].sum(axis=1) <= 1.0 + 1e-6)
    finally:
        battle_finish()


def test_hand_block_reflects_hand(meta):
    """末尾の手札ブロックが [0,1] の有限値で、手札がある局面では非ゼロを含む."""
    deck = load_deck(DECK)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        vec = encode_observation(obs, meta)
        hand_block = vec[-HAND_FEAT:]
        assert hand_block.shape == (HAND_FEAT,)
        assert np.all((hand_block >= 0.0) & (hand_block <= 1.0))
        # turn>=3 の MAIN なので手札があり、種別構成のどれかは立つはず
        assert hand_block.any()
    finally:
        battle_finish()


def test_encode_handles_none_state(meta):
    """current/select が None でも全 0 ベクトルを返す（デッキ選択時など）."""
    none_obs = to_observation_class({"select": None, "logs": [], "current": None})
    vec = encode_observation(none_obs, meta)
    assert vec.shape == (OBS_FEAT_LEN,)
    assert not vec.any()
    assert encode_actions(none_obs, meta).shape == (0, ACTION_FEAT_LEN)
