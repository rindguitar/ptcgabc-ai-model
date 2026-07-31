"""bench_first（§72・たねベンチ展開をエネ付与より前に出す）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。
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

from agents import _choose_main, make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import AreaType, CardType, OptionType, SelectType  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


@pytest.fixture(scope="module")
def ids(meta):
    """メタから カテゴリ別の実在 ID を拾う（値はダミー扱い・名前等は使わない）."""
    basic = next(c for c in meta.card_type if meta.is_basic_pokemon(c))
    energy = next(
        c
        for c, t in meta.card_type.items()
        if t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
    )
    return {"basic": basic, "energy": energy}


def _main_obs(ids, with_attach=True, with_basic_play=True):
    """MAIN 選択のフェイク: ATTACH（アクティブ宛）と たねベンチ展開 PLAY を並べる.

    ⚠️ _is_basic_pokemon_play は opt.index で手札を引くので、手札の該当位置に
    たねポケモンを置く（実データ形状に合わせる）。
    """
    opts, hand = [], [NS(id=ids["basic"])]
    if with_attach:
        opts.append(
            NS(
                type=OptionType.ATTACH,
                index=0,
                inPlayArea=AreaType.ACTIVE,
                inPlayIndex=0,
            )
        )
    if with_basic_play:
        opts.append(
            NS(type=OptionType.PLAY, index=0, inPlayArea=None, inPlayIndex=None)
        )
    active = NS(id=ids["basic"], energyCards=[], tools=[], hp=60)
    me = NS(hand=hand, active=[active], bench=[], discard=[], deckCount=40)
    st = NS(
        yourIndex=0,
        players=[me, NS(hand=[], active=[active], bench=[], discard=[], deckCount=40)],
        stadium=None,
        supporterPlayed=False,
        turn=3,
    )
    sel = NS(
        type=SelectType.MAIN,
        context=None,
        option=opts,
        minCount=1,
        maxCount=1,
        deck=None,
    )
    return NS(current=st, select=sel)


def test_default_order_puts_attach_before_bench(meta, ids):
    """既定（bench_first=False）は従来順＝エネ付与が先（挙動不変の回帰ガード）."""
    obs = _main_obs(ids)
    got = _choose_main(obs, meta, random.Random(0))
    assert obs.select.option[got].type == OptionType.ATTACH


def test_bench_first_puts_bench_play_before_attach(meta, ids):
    """bench_first=True は たねベンチ展開を先に選ぶ."""
    obs = _main_obs(ids)
    got = _choose_main(obs, meta, random.Random(0), bench_first=True)
    assert obs.select.option[got].type == OptionType.PLAY


def test_bench_first_falls_back_to_attach_when_no_basic(meta, ids):
    """展開できるたねが無ければ bench_first でもエネ付与に落ちる（順序だけの変更）."""
    obs = _main_obs(ids, with_basic_play=False)
    got = _choose_main(obs, meta, random.Random(0), bench_first=True)
    assert obs.select.option[got].type == OptionType.ATTACH


def test_bench_first_without_attach_is_unchanged(meta, ids):
    """エネ付与が無い局面では bench_first の有無で結果が変わらない."""
    obs = _main_obs(ids, with_attach=False)
    a = _choose_main(obs, meta, random.Random(0))
    b = _choose_main(obs, meta, random.Random(0), bench_first=True)
    assert a == b
    assert obs.select.option[a].type == OptionType.PLAY


def test_heuristic_agent_passes_bench_first(meta, ids):
    """make_heuristic_agent 経由でも bench_first が MAIN 選択に届く."""
    obs = _main_obs(ids)
    base = make_heuristic_agent(meta)
    swapped = make_heuristic_agent(meta, bench_first=True)
    assert obs.select.option[base(obs, random.Random(0))[0]].type == OptionType.ATTACH
    assert obs.select.option[swapped(obs, random.Random(0))[0]].type == OptionType.PLAY
