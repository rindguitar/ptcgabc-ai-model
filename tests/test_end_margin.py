"""end_margin（END への逸脱に要求する追加マージン）のテスト.

_select_action を訪問集計のフェイクで直接検証する。カード情報は不要
（option の type のみ・Pokémon Elements を持ち込まない）。
"""

import os
import random
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from cg.api import OptionType  # noqa: E402
from ismcts import _select_action, make_ismcts_agent  # noqa: E402

# option 列: 0=PLAY（展開手）・1=PLAY・2=END
OPTS = [NS(type=OptionType.PLAY), NS(type=OptionType.PLAY), NS(type=OptionType.END)]


def test_small_edge_end_deviation_blocked():
    """僅差の END 逸脱は end_margin で阻止され、heuristic の展開手を維持する."""
    agg = {(0,): [20, 10.0], (2,): [20, 11.6]}  # mean 0.50 vs 0.58（差 0.08）
    # 従来（end_margin なし）: select_margin 0.05 を超えるので END へ逸脱してしまう
    assert _select_action(agg, (0,), OPTS, 15, 0.05) == [2]
    # end_margin 0.15: 差 0.08 では不足 → 展開手を維持
    assert _select_action(agg, (0,), OPTS, 15, 0.05, end_margin=0.15) == [0]


def test_clear_end_advantage_still_deviates():
    """大差なら end_margin があっても END を採用する（本当に終えるべき局面は妨げない）."""
    agg = {(0,): [20, 10.0], (2,): [20, 16.0]}  # mean 0.50 vs 0.80
    assert _select_action(agg, (0,), OPTS, 15, 0.05, end_margin=0.15) == [2]


def test_non_end_deviation_uses_normal_margin():
    """END 以外への逸脱は従来の select_margin のまま（end_margin の影響を受けない）."""
    agg = {(0,): [20, 10.0], (1,): [20, 11.6]}  # PLAY→PLAY の僅差逸脱
    assert _select_action(agg, (0,), OPTS, 15, 0.05, end_margin=0.15) == [1]


def test_heuristic_end_not_gated():
    """heuristic 自身が END を提案している時は適用しない（正当な終了を妨げない）."""
    agg = {(2,): [20, 10.0], (0,): [20, 11.6]}  # END 案から展開手への逸脱は通常マージン
    assert _select_action(agg, (2,), OPTS, 15, 0.05, end_margin=0.15) == [0]


def test_min_visits_still_applies():
    """訪問数不足の候補は end_margin 以前に不採用（既存ルールの維持）."""
    agg = {(0,): [20, 10.0], (2,): [5, 4.5]}  # END は mean 0.9 だが 5 訪問のみ
    assert _select_action(agg, (0,), OPTS, 15, 0.05, end_margin=0.15) == [0]


def test_make_ismcts_agent_accepts_end_margin():
    """make_ismcts_agent が end_margin を受け取り、合法な選択を返す（engine 経由）."""
    from cards import load_card_meta  # noqa: E402
    from cg.game import battle_finish  # noqa: E402
    from harness import read_deck  # noqa: E402
    from test_ismcts import DECK, _advance_to_main  # noqa: E402

    deck = read_deck(DECK)
    rng = random.Random(0)
    meta = load_card_meta()
    agent = make_ismcts_agent(meta, deck, deck, time_budget=0.05, end_margin=0.15)
    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None, "decision point に到達しなかった"
        sel = obs.select
        action = agent(obs, rng)
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()
