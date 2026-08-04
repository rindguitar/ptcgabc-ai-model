"""引きずり出し（`SWITCH` の相手指定）の解決バグ修正（§79）のテスト.

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


@pytest.fixture(scope="module")
def ids(meta):
    """「サイド1の札」と「サイド2以上の札」の実在 ID を1つずつ拾う."""
    small = next(
        c
        for c in meta.card_type
        if meta.is_basic_pokemon(c) and meta.prize_value.get(c, 1) == 1
    )
    big = next(c for c, v in meta.prize_value.items() if v >= 2 and meta.hp.get(c))
    return {"small": small, "big": big}


def _pk(cid, n_energy=0, hp=100):
    return NS(
        id=cid,
        energyCards=[NS(id=0)] * n_energy,
        energies=[0] * n_energy,
        tools=[],
        hp=hp,
        maxHp=hp,
    )


def _obs(ids, *, player_index, my_bench, opp_bench, context=SelectContext.SWITCH):
    """SWITCH/TO_ACTIVE のフェイク観測.

    option は `index` がベンチ位置・`playerIndex` が「どちらの場か」を指す
    （エンジン実測の形）。player_index に相手席を渡すと引きずり出しの決定になる。
    """
    opts = [
        NS(
            type=OptionType.CARD,
            index=i,
            playerIndex=player_index,
            inPlayArea=AreaType.BENCH,
            inPlayIndex=None,
        )
        for i in range(len(opp_bench if player_index == 1 else my_bench))
    ]
    me = NS(hand=[], active=[_pk(ids["small"])], bench=my_bench, discard=[], deckCount=40)
    opp = NS(hand=[], active=[_pk(ids["small"])], bench=opp_bench, discard=[], deckCount=40)
    st = NS(yourIndex=0, players=[me, opp], stadium=None, supporterPlayed=False, turn=5)
    sel = NS(
        type=SelectType.CARD,
        context=context,
        option=opts,
        minCount=1,
        maxCount=1,
        deck=None,
    )
    return NS(current=st, select=sel)


def test_own_bench_promotion_is_unchanged(meta, ids):
    """自分の場の昇格は従来どおり「エネが乗った子」を選ぶ（フラグの有無で不変）."""
    my_bench = [_pk(ids["small"], n_energy=0), _pk(ids["small"], n_energy=2)]
    obs = _obs(ids, player_index=0, my_bench=my_bench, opp_bench=[])
    assert _generic_select(obs, meta) == [1]
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]


def test_opponent_target_defaults_to_head(meta, ids):
    """既定（off）では相手指定を自分のベンチとして解決＝解決失敗で先頭取りに退化する.

    自分のベンチを空にすると全候補が解決できず、旧実装は index 昇順＝先頭になる。
    """
    opp_bench = [_pk(ids["small"]), _pk(ids["big"])]
    obs = _obs(ids, player_index=1, my_bench=[], opp_bench=opp_bench)
    assert _generic_select(obs, meta) == [0]


def test_opponent_target_picks_max_prize(meta, ids):
    """修正後は相手の場から「取れるサイドが最大」の個体を引きずり出す（§79/§80）."""
    opp_bench = [_pk(ids["small"]), _pk(ids["big"]), _pk(ids["small"])]
    obs = _obs(ids, player_index=1, my_bench=[], opp_bench=opp_bench)
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]


def test_opponent_target_tiebreaks_on_low_hp(meta, ids):
    """サイド枚数が同じなら残 HP の少ない個体（倒し切りやすい方）を選ぶ."""
    opp_bench = [_pk(ids["small"], hp=100), _pk(ids["small"], hp=30)]
    obs = _obs(ids, player_index=1, my_bench=[], opp_bench=opp_bench)
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]


def test_opponent_target_ignores_my_bench_energy(meta, ids):
    """自分のベンチのエネ数は相手指定の順位付けに混入しない（バグの本体）.

    自分のベンチ 0 番に大量のエネを載せても、選ぶのは相手のサイド最大の個体（1 番）。
    """
    my_bench = [_pk(ids["small"], n_energy=3), _pk(ids["small"], n_energy=0)]
    opp_bench = [_pk(ids["small"]), _pk(ids["big"])]
    obs = _obs(ids, player_index=1, my_bench=my_bench, opp_bench=opp_bench)
    assert _generic_select(obs, meta) == [0]  # 旧: 自分のエネ最多＝先頭
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]


def test_inplay_index_form_is_supported(meta, ids):
    """option が inPlayIndex 形式でも相手の場を解決できる（防御的対応の回帰ガード）."""
    opp_bench = [_pk(ids["small"]), _pk(ids["big"])]
    obs = _obs(ids, player_index=1, my_bench=[], opp_bench=opp_bench)
    for i, o in enumerate(obs.select.option):
        o.inPlayIndex = i
        o.index = None
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]


def test_to_active_context_also_covered(meta, ids):
    """TO_ACTIVE でも同じ解決を通る（context 違いで抜けない）."""
    my_bench = [_pk(ids["small"], n_energy=0), _pk(ids["small"], n_energy=1)]
    obs = _obs(
        ids,
        player_index=0,
        my_bench=my_bench,
        opp_bench=[],
        context=SelectContext.TO_ACTIVE,
    )
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]


def test_missing_player_index_falls_back_to_own_side(meta, ids):
    """playerIndex が無い（None）候補は従来どおり自分の場として解決する."""
    my_bench = [_pk(ids["small"], n_energy=0), _pk(ids["small"], n_energy=2)]
    obs = _obs(ids, player_index=0, my_bench=my_bench, opp_bench=[])
    for o in obs.select.option:
        o.playerIndex = None
    assert _generic_select(obs, meta, fix_switch_target=True) == [1]
