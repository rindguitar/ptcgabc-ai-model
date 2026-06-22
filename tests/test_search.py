"""search API + determinization のスモークテスト.

cabt Engine（`cg`）と `data/deck.csv` が無い環境ではモジュールごとスキップ。
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

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import (  # noqa: E402
    SelectType,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from determinize import determinize, pick_opponent_deck  # noqa: E402
from harness import read_deck  # noqa: E402

POOL_MAX = 1267  # 大会リーガルなカード ID 上限


def _advance_to_main(deck, rng, min_turn=3, max_steps=2000):
    """battle を進め、turn>=min_turn の最初の MAIN 選択時点の observation を返す."""
    meta = load_card_meta()
    heuristic = make_heuristic_agent(meta)
    obs_dict, _ = battle_start(deck, deck)
    for _ in range(max_steps):
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            return None
        if obs.select is None:
            return None
        if (
            obs.select.type == SelectType.MAIN
            and obs.current
            and obs.current.turn >= min_turn
        ):
            return obs
        obs_dict = battle_select(heuristic(obs, rng))
    return None


def test_determinize_counts():
    """determinize が各隠れゾーンを正しい枚数・有効 ID で返す."""
    deck = read_deck(DECK)
    rng = random.Random(0)
    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None, "decision point に到達しなかった"
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        det = determinize(obs, deck, deck, rng)

        assert len(det["your_deck"]) == me.deckCount
        assert len(det["your_prize"]) == len(me.prize)
        assert len(det["opponent_deck"]) == opp.deckCount
        assert len(det["opponent_prize"]) == len(opp.prize)
        assert len(det["opponent_hand"]) == opp.handCount

        all_ids = sum(det.values(), [])
        assert all(1 <= c <= POOL_MAX for c in all_ids)
    finally:
        battle_finish()


def test_pick_opponent_deck():
    """観測整合の候補が選ばれ、無ければ fallback になる."""
    deck = read_deck(DECK)
    rng = random.Random(0)
    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None
        wrong = [1] * 60  # 基本エネのみ（相手の場のポケモン等を含まない＝非整合）
        # 整合候補(deck)があれば deck が選ばれる
        assert pick_opponent_deck(obs, [deck, wrong], wrong, rng) == deck
        # 整合候補が無ければ fallback
        assert pick_opponent_deck(obs, [wrong], deck, rng) == deck
        # 候補なし → fallback
        assert pick_opponent_deck(obs, None, deck, rng) == deck
    finally:
        battle_finish()


def test_search_roundtrip():
    """determinization → search_begin → search_step → search_end が動く."""
    deck = read_deck(DECK)
    rng = random.Random(1)
    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None
        det = determinize(obs, deck, deck, rng)
        ss = search_begin(
            obs,
            det["your_deck"],
            det["your_prize"],
            det["opponent_deck"],
            det["opponent_prize"],
            det["opponent_hand"],
            det["opponent_active"],
            False,
        )
        assert ss.observation is not None
        sel = ss.observation.select
        assert sel is not None

        count = sel.minCount if sel.minCount > 0 else min(1, sel.maxCount)
        ss2 = search_step(ss.searchId, list(range(count)))
        assert ss2 is not None
        search_end()
    finally:
        battle_finish()
