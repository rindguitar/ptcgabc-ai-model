"""heuristic の打ち回し（にげる・昇格先選択・§31）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組み、判断ロジックを直接検証する。
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

from agents import _choose_main, _generic_select, find_forced_recycle  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import AreaType, OptionType, SelectContext, SelectType  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def _pk(energy_n: int, hp: int) -> NS:
    return NS(energyCards=[1] * energy_n, hp=hp, tools=[], preEvolution=[])


def _obs_main(options, active, bench, stadium=None):
    me = NS(hand=[], discard=[], active=active, bench=bench, prize=[])
    opp = NS(hand=[], discard=[], active=[], bench=[], prize=[])
    st = NS(yourIndex=0, players=[me, opp], stadium=stadium)
    sel = NS(type=SelectType.MAIN, option=options, minCount=1, maxCount=1, deck=None)
    return NS(current=st, select=sel)


def test_retreat_when_active_cannot_attack(meta):
    """攻撃不能＋ベンチにエネが乗っている → にげるを選ぶ（§31）."""
    opts = [
        NS(type=OptionType.RETREAT, index=None, inPlayArea=None, inPlayIndex=None),
        NS(type=OptionType.END, index=None, inPlayArea=None, inPlayIndex=None),
    ]
    obs = _obs_main(opts, active=[_pk(0, 80)], bench=[_pk(2, 100)])
    assert _choose_main(obs, meta, random.Random(0)) == 0  # RETREAT


def test_no_retreat_when_attack_available(meta):
    """攻撃できるなら（にげるより）攻撃する."""
    opts = [
        NS(type=OptionType.RETREAT, index=None, inPlayArea=None, inPlayIndex=None),
        NS(
            type=OptionType.ATTACK,
            attackId=0,
            index=None,
            inPlayArea=None,
            inPlayIndex=None,
        ),
    ]
    obs = _obs_main(opts, active=[_pk(1, 80)], bench=[_pk(3, 100)])
    assert _choose_main(obs, meta, random.Random(0)) == 1  # ATTACK


def test_no_retreat_when_bench_not_better(meta):
    """ベンチのエネが active 以下なら交代しない（にげるコストの浪費防止）."""
    opts = [
        NS(type=OptionType.RETREAT, index=None, inPlayArea=None, inPlayIndex=None),
        NS(type=OptionType.END, index=None, inPlayArea=None, inPlayIndex=None),
    ]
    obs = _obs_main(opts, active=[_pk(2, 80)], bench=[_pk(1, 100)])
    assert _choose_main(obs, meta, random.Random(0)) == 1  # END（交代しない）


def test_promotion_prefers_energy_loaded_bench(meta):
    """昇格先（TO_ACTIVE）はエネ最多→HP のベンチを選ぶ（旧実装は先頭固定・§31）.

    option の形はエンジン実測に合わせる: inPlayArea/inPlayIndex は None・`index` がベンチ位置。
    """
    bench = [_pk(0, 110), _pk(3, 80), _pk(1, 150)]
    me = NS(hand=[], discard=[], active=[], bench=bench, prize=[])
    st = NS(yourIndex=0, players=[me, NS()], stadium=None)
    opts = [
        NS(type=OptionType.CARD, inPlayArea=None, inPlayIndex=None, index=i)
        for i in range(3)
    ]
    sel = NS(
        type=SelectType.CARD,
        context=SelectContext.TO_ACTIVE,
        option=opts,
        minCount=1,
        maxCount=1,
        deck=None,
    )
    obs = NS(current=st, select=sel)
    assert _generic_select(obs, meta) == [1]  # エネ3個の子

    # 同エネなら HP が高い方
    bench[1] = _pk(1, 80)
    assert _generic_select(obs, meta) == [2]  # エネ1・HP150 > エネ1・HP80

    # inPlayIndex 形式（防御対応）でも解決できる
    opts2 = [
        NS(type=OptionType.CARD, inPlayArea=AreaType.BENCH, inPlayIndex=i, index=None)
        for i in range(3)
    ]
    sel2 = NS(
        type=SelectType.CARD,
        context=SelectContext.TO_ACTIVE,
        option=opts2,
        minCount=1,
        maxCount=1,
        deck=None,
    )
    assert _generic_select(NS(current=st, select=sel2), meta) == [2]


def test_recycle_when_deck_low(meta):
    """残デッキ≤閾値でリサイクル札（§39）を最優先プレイ・十分残っていれば打たない."""

    rid = next((c for c, v in meta.is_deck_recycle.items() if v), None)
    if rid is None:
        pytest.skip("メタにリサイクル札が無い")
    hand = [NS(id=rid)]
    me = NS(
        hand=hand, discard=[], active=[_pk(1, 80)], bench=[], prize=[], deckCount=10
    )
    st = NS(yourIndex=0, players=[me, NS()], stadium=None)
    opts = [
        NS(type=OptionType.PLAY, index=0, inPlayArea=None, inPlayIndex=None),
        NS(type=OptionType.END, index=None, inPlayArea=None, inPlayIndex=None),
    ]
    sel = NS(type=SelectType.MAIN, option=opts, minCount=1, maxCount=1, deck=None)
    obs = NS(current=st, select=sel)
    assert _choose_main(obs, meta, random.Random(0)) == 0  # リサイクルを打つ

    me.deckCount = 40  # 山が厚ければ温存（END へ）
    assert _choose_main(obs, meta, random.Random(0)) == 1


def test_find_forced_recycle_returns_index_or_none(meta):
    """find_forced_recycle（§43 共用ヘルパ）: 僅少時のみリサイクル PLAY の index を返す."""

    rid = next((c for c, v in meta.is_deck_recycle.items() if v), None)
    if rid is None:
        pytest.skip("メタにリサイクル札が無い")
    nid = next(c for c in meta.card_type if not meta.is_deck_recycle.get(c, False))
    hand = [NS(id=nid), NS(id=rid)]  # 先頭は非リサイクル札＝素通りの確認
    me = NS(
        hand=hand, discard=[], active=[_pk(1, 80)], bench=[], prize=[], deckCount=10
    )
    st = NS(yourIndex=0, players=[me, NS()], stadium=None)
    opts = [
        NS(type=OptionType.PLAY, index=0, inPlayArea=None, inPlayIndex=None),
        NS(type=OptionType.PLAY, index=1, inPlayArea=None, inPlayIndex=None),
        NS(type=OptionType.END, index=None, inPlayArea=None, inPlayIndex=None),
    ]
    sel = NS(type=SelectType.MAIN, option=opts, minCount=1, maxCount=1, deck=None)
    obs = NS(current=st, select=sel)
    assert find_forced_recycle(obs, meta) == 1  # リサイクル札の PLAY だけを拾う

    me.deckCount = 40  # 山が厚ければ発動しない
    assert find_forced_recycle(obs, meta) is None

    me.deckCount = 20  # 閾値の上書き（§43 A/B）: 既定15では発動せず・25なら発動
    assert find_forced_recycle(obs, meta) is None
    assert find_forced_recycle(obs, meta, recycle_at=25) == 1

    sel.type = SelectType.ATTACK  # MAIN 以外では発動しない
    me.deckCount = 10
    assert find_forced_recycle(obs, meta) is None
