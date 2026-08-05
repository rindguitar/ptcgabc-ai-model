"""開幕アクティブの帯基準（§89・特性なし → サイド小 → HP大）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。
カード ID はダミー扱い（規約: Pokémon Elements を持ち込まない）。
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from agents import _generic_select  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import AreaType, OptionType, SelectContext, SelectType  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def _pick_ids(meta):
    """メタから「特性持ちのたね」「特性なし・サイド1のたね」「サイド2以上のたね」を拾う."""
    basics = [c for c in meta.card_type if meta.is_basic_pokemon(c)]
    ability = next(c for c in basics if meta.has_ability.get(c))
    plain = next(
        c
        for c in basics
        if not meta.has_ability.get(c) and meta.prize_value.get(c, 1) == 1
    )
    heavy = next(
        c
        for c in basics
        if not meta.has_ability.get(c) and meta.prize_value.get(c, 1) >= 2
    )
    return ability, plain, heavy


def _obs(hand_ids, context=SelectContext.SETUP_ACTIVE_POKEMON):
    """手札の候補から開幕アクティブを1枚選ぶフェイク観測."""
    hand = [NS(id=c) for c in hand_ids]
    opts = [
        NS(
            type=OptionType.CARD,
            index=i,
            playerIndex=0,
            area=AreaType.HAND,
            inPlayArea=None,
            inPlayIndex=None,
        )
        for i in range(len(hand_ids))
    ]
    me = NS(hand=hand, active=[], bench=[], discard=[], deckCount=53)
    opp = NS(hand=[], active=[], bench=[], discard=[], deckCount=53)
    st = NS(yourIndex=0, players=[me, opp], stadium=None, supporterPlayed=False, turn=0)
    sel = NS(
        type=SelectType.CARD,
        context=context,
        option=opts,
        minCount=1,
        maxCount=1,
        deck=None,
    )
    return NS(current=st, select=sel)


def test_default_takes_head(meta):
    """既定（off）は先頭取り＝挙動不変の回帰ガード."""
    ability, plain, _ = _pick_ids(meta)
    obs = _obs([ability, plain])
    assert _generic_select(obs, meta) == [0]


def test_avoids_ability_holder(meta):
    """特性持ちを開幕アクティブに置かない（帯基準の本体）."""
    ability, plain, _ = _pick_ids(meta)
    obs = _obs([ability, plain])
    assert _generic_select(obs, meta, setup_active_rule=True) == [1]


def test_prefers_low_prize(meta):
    """特性の有無が同じならサイド枚数の少ない札を選ぶ."""
    _, plain, heavy = _pick_ids(meta)
    obs = _obs([heavy, plain])
    assert _generic_select(obs, meta, setup_active_rule=True) == [1]


def test_tiebreaks_on_high_hp(meta):
    """特性・サイドが同値なら HP の高い方（耐える方）を選ぶ."""
    basics = [
        c
        for c in meta.card_type
        if meta.is_basic_pokemon(c)
        and not meta.has_ability.get(c)
        and meta.prize_value.get(c, 1) == 1
    ]
    by_hp = sorted(basics, key=lambda c: meta.hp.get(c, 0))
    low, high = by_hp[0], by_hp[-1]
    if meta.hp.get(low, 0) == meta.hp.get(high, 0):
        pytest.skip("HP が異なるたねが2種見つからない")
    obs = _obs([low, high])
    assert _generic_select(obs, meta, setup_active_rule=True) == [1]


def test_single_candidate_is_unchanged(meta):
    """候補が1枚しかなければどちらの実装でも同じ."""
    ability, _, _ = _pick_ids(meta)
    obs = _obs([ability])
    assert _generic_select(obs, meta) == _generic_select(
        obs, meta, setup_active_rule=True
    ) == [0]


def test_other_contexts_are_untouched(meta):
    """SETUP_BENCH_POKEMON など他の context には影響しない."""
    ability, plain, _ = _pick_ids(meta)
    obs = _obs([ability, plain], context=SelectContext.SETUP_BENCH_POKEMON)
    obs.select.maxCount = 2
    assert _generic_select(obs, meta, setup_active_rule=True) == [0, 1]
