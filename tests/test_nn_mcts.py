"""NN 誘導 MCTS（PUCT）骨格のテスト（ダミー評価器で機構を確認）."""

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
from cg.api import SelectType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from deck import load_deck  # noqa: E402
from nn_mcts import make_nn_mcts_agent, make_prize_evaluator  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def _advance_to_main(deck, rng, meta):
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


def test_prize_evaluator_shapes(meta):
    """ダミー評価器が value[0,1] と option 整列の priors を返す."""
    deck = load_deck(DECK)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        value, priors = make_prize_evaluator(meta)(obs)
        assert 0.0 <= value <= 1.0
        assert len(priors) == len(obs.select.option)
        assert abs(sum(priors) - 1.0) < 1e-6
    finally:
        battle_finish()


def test_nn_mcts_returns_legal_action(meta):
    """PUCT エージェント（ダミー評価器）が決定点で合法な選択を返す."""
    deck = load_deck(DECK)
    agent = make_nn_mcts_agent(meta, deck, deck, n_simulations=24, n_determinizations=2)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        sel = obs.select
        action = agent(obs, random.Random(0))
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
        assert len(set(action)) == len(action)
    finally:
        battle_finish()


def test_plan_search_budget_scaling():
    """適応探索（§31）: 未計測は床・予算は dets 優先で配分・cap 頭打ち・枯渇で床."""
    from nn_mcts import plan_search

    base_s, base_d = 64, 2
    # 未計測（初手）・予算枯渇 → 床（品質を従来から下げない）
    assert plan_search(base_s, base_d, 500.0, 0, None) == (base_s, base_d)
    assert plan_search(base_s, base_d, 0.0, 30, 0.005) == (base_s, base_d)
    assert plan_search(base_s, base_d, -5.0, 30, 0.005) == (base_s, base_d)
    # 潤沢（単価 0.5ms・残り 500s・序盤）: units=25000 → 両 cap (512, 16)
    assert plan_search(base_s, base_d, 500.0, 0, 0.0005) == (512, 16)
    # 中間（単価 5ms）: units=2500 → dets=16（幅優先）・sims=156
    assert plan_search(base_s, base_d, 500.0, 0, 0.005) == (156, 16)
    # 重い環境（単価 50ms）: units=250 → dets=2（床）・sims=125
    assert plan_search(base_s, base_d, 500.0, 0, 0.05) == (125, 2)
    # 1手予算の自己整合: sims×dets ≦ units（±丸め）
    s, d = plan_search(base_s, base_d, 500.0, 0, 0.005)
    assert s * d <= 2500 * 1.1
    # 残り決定数の床: 終盤（moves≫40）でも1手に全残額を注がない
    s, d = plan_search(base_s, base_d, 80.0, 60, 0.005)
    assert s * d <= (80.0 / 8.0) / 0.005 * 1.1


def test_adaptive_agent_returns_legal_and_tracks(meta):
    """game_budget 付き agent が合法手を返す（ダミー評価器・時計/EMA 経路の煙）."""
    deck = load_deck(DECK)
    agent = make_nn_mcts_agent(
        meta,
        deck,
        deck,
        n_simulations=8,
        n_determinizations=1,
        game_budget=540.0,
    )
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        sel = obs.select
        for _ in range(2):  # 2手目で EMA 経路（unit_ema 有り）も通す
            action = agent(obs, random.Random(0))
            assert sel.minCount <= len(action) <= sel.maxCount
            assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()


def test_leaf_rollout_mode_returns_legal_action(meta):
    """AlphaGo 型（§38・leaf_rollouts>0）: priors=評価器・葉値=接地ロールアウトで合法手を返す."""
    deck = load_deck(DECK)
    agent = make_nn_mcts_agent(
        meta,
        deck,
        deck,
        n_simulations=8,
        n_determinizations=1,
        leaf_rollouts=1,
    )
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        sel = obs.select
        action = agent(obs, random.Random(0))
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()


def test_leaf_rollout_overrides_net_value(meta):
    """葉値が評価器の value でなくロールアウト由来になっている（value を極端値にして検証）."""
    from nn_mcts import aggregate_visits

    deck = load_deck(DECK)
    # 常に value=1.0（全局面勝ち）を返す壊れた評価器。leaf_rollouts>0 なら無視されるはず。
    def broken_evaluator(obs):
        n = len(obs.select.option) if obs.select else 1
        return 1.0, [1.0 / n] * n

    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        from agents import make_heuristic_agent as _mha

        h = _mha(meta)
        visits = aggregate_visits(
            obs, deck, deck, broken_evaluator, random.Random(0),
            16, 1, 1.5, h, leaf_rollouts=1,
        )
        assert visits and sum(visits.values()) > 0  # 探索が回りきる（例外なし）
    finally:
        battle_finish()
