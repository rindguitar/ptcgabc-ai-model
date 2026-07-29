"""dev_margin（展開手からの逸脱に要求する追加マージン・§60 の第二の逸脱）のテスト.

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

# option 列: 0=PLAY（展開手）・1=ATTACK・2=END・3=EVOLVE（展開手）
OPTS = [
    NS(type=OptionType.PLAY),
    NS(type=OptionType.ATTACK),
    NS(type=OptionType.END),
    NS(type=OptionType.EVOLVE),
]


def test_small_edge_departure_from_dev_blocked():
    """僅差の「展開手→攻撃」逸脱は dev_margin で阻止され、展開手を維持する."""
    agg = {(0,): [20, 10.0], (1,): [20, 11.6]}  # mean 0.50 vs 0.58（差 0.08）
    # 従来（dev_margin なし）: select_margin 0.05 を超えるので攻撃へ逸脱してしまう
    assert _select_action(agg, (0,), OPTS, 15, 0.05) == [1]
    # dev_margin 0.10: 差 0.08 では不足 → 展開手を維持
    assert _select_action(agg, (0,), OPTS, 15, 0.05, dev_margin=0.10) == [0]


def test_clear_advantage_still_departs():
    """大差なら dev_margin があっても逸脱する（リーサル級の即断攻撃を潰さない）."""
    agg = {(0,): [20, 10.0], (1,): [20, 16.0]}  # mean 0.50 vs 0.80
    assert _select_action(agg, (0,), OPTS, 15, 0.05, dev_margin=0.10) == [1]


def test_dev_to_dev_uses_normal_margin():
    """展開手どうしの乗り換え（PLAY→EVOLVE）は従来の select_margin のまま."""
    agg = {(0,): [20, 10.0], (3,): [20, 11.6]}
    assert _select_action(agg, (0,), OPTS, 15, 0.05, dev_margin=0.10) == [3]


def test_non_dev_anchor_not_gated():
    """heuristic の提案が展開手でなければ適用しない（攻撃案→他への逸脱は通常マージン）."""
    agg = {(1,): [20, 10.0], (0,): [20, 11.6]}
    assert _select_action(agg, (1,), OPTS, 15, 0.05, dev_margin=0.10) == [0]


def test_dev_and_end_margins_compose_with_max():
    """展開手→END は両方該当＝大きい方のマージンを要求する."""
    agg = {(0,): [20, 10.0], (2,): [20, 12.4]}  # mean 0.50 vs 0.62（差 0.12）
    # dev 0.10 のみなら通る
    assert _select_action(agg, (0,), OPTS, 15, 0.05, dev_margin=0.10) == [2]
    # end 0.20 が併用されると大きい方が効いて阻止される（順序に依らない）
    assert _select_action(
        agg, (0,), OPTS, 15, 0.05, end_margin=0.20, dev_margin=0.10
    ) == [0]


def test_min_visits_still_applies():
    """訪問数不足の候補は dev_margin 以前に不採用（既存ルールの維持）."""
    agg = {(0,): [20, 10.0], (1,): [5, 4.5]}  # 攻撃は mean 0.9 だが 5 訪問のみ
    assert _select_action(agg, (0,), OPTS, 15, 0.05, dev_margin=0.10) == [0]


def test_unset_dev_margin_keeps_legacy_behavior():
    """未指定なら挙動不変（end_margin 導入時と同じ後方互換の保証）."""
    agg = {(0,): [20, 10.0], (1,): [20, 11.6]}
    assert _select_action(agg, (0,), OPTS, 15, 0.05) == [1]
    assert _select_action(agg, (0,), OPTS, 15, 0.05, end_margin=0.15) == [1]


def test_make_ismcts_agent_accepts_dev_margin():
    """make_ismcts_agent が dev_margin を受け取り、合法な選択を返す（engine 経由）."""
    from cards import load_card_meta  # noqa: E402
    from cg.game import battle_finish  # noqa: E402
    from harness import read_deck  # noqa: E402
    from test_ismcts import DECK, _advance_to_main  # noqa: E402

    deck = read_deck(DECK)
    rng = random.Random(0)
    meta = load_card_meta()
    agent = make_ismcts_agent(
        meta, deck, deck, time_budget=0.05, end_margin=0.15, dev_margin=0.10
    )
    obs = _advance_to_main(deck, rng)
    try:
        assert obs is not None, "decision point に到達しなかった"
        sel = obs.select
        action = agent(obs, rng)
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()
