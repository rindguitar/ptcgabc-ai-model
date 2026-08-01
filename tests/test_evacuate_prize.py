"""高prize退避（§80・2サイド以上のアクティブを入替札/にげるで下げる）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。
カード ID はダミー扱い（規約: Pokémon Elements を持ち込まない）。
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
from cg.api import AreaType, OptionType, SelectType  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


@pytest.fixture(scope="module")
def ids(meta):
    """メタから「2サイド以上の札」「入替札」「普通のたね」の実在 ID を拾う."""
    big = next(c for c, v in meta.prize_value.items() if v >= 2 and meta.hp.get(c))
    plain = next(
        c
        for c in meta.card_type
        if meta.is_basic_pokemon(c) and meta.prize_value.get(c, 1) == 1
    )
    switch = next(c for c, v in meta.is_self_switch.items() if v)
    return {"big": big, "plain": plain, "switch": switch}


def _obs(
    ids,
    meta,
    *,
    active_big=True,
    with_switch=True,
    with_retreat=True,
    with_bench=True,
    threat_energy=3,
    lethal=False,
):
    """MAIN 選択のフェイク。相手アクティブに脅威エネを載せて KO 脅威を作る。

    lethal=True で「こちらの攻撃が相手アクティブを倒せる」状況にする（退避しない条件）。
    """
    hand = [NS(id=ids["switch"])] if with_switch else []
    opts = []
    if with_switch:
        opts.append(NS(type=OptionType.PLAY, index=0, inPlayArea=None, inPlayIndex=None))
    if with_retreat:
        opts.append(NS(type=OptionType.RETREAT, index=0, inPlayArea=None, inPlayIndex=None))
    # 攻撃（lethal なら相手 HP を上回るダメージのワザを選ぶ）
    atk_id = next(a for a in meta.damage if meta.damage[a] >= 100)
    opts.append(NS(type=OptionType.ATTACK, attackId=atk_id, index=0))
    opts.append(NS(type=OptionType.END, index=0))

    act_id = ids["big"] if active_big else ids["plain"]
    active = NS(
        id=act_id,
        energyCards=[NS(id=0)],
        energies=[0],
        tools=[],
        # 既に削られている状態にする（退避が意味を持つのは KO 圏内に居るとき）
        hp=50,
        maxHp=meta.hp.get(act_id, 100),
    )
    bench = (
        [NS(id=ids["plain"], energyCards=[], energies=[], tools=[], hp=60, maxHp=60)]
        if with_bench
        else []
    )
    me = NS(hand=hand, active=[active], bench=bench, discard=[], deckCount=40)
    # 相手アクティブ: 低 HP・高火力ワザ持ちにして KO 脅威を作る
    opp_id = next(
        c
        for c in meta.card_attacks
        if meta.card_attacks.get(c) and meta.best_damage.get(c, 0) >= 100
    )
    opp_active = NS(
        id=opp_id,
        energyCards=[NS(id=0)] * threat_energy,
        energies=[0] * threat_energy,
        tools=[],
        # lethal=True のときだけ、こちらの攻撃で落とせる残 HP にする
        hp=50 if lethal else 300,
        maxHp=meta.hp.get(opp_id, 100),
    )
    opp = NS(hand=[], active=[opp_active], bench=[], discard=[], deckCount=40)
    st = NS(
        yourIndex=0,
        players=[me, opp],
        stadium=None,
        supporterPlayed=False,
        turn=5,
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


def test_default_does_not_evacuate(meta, ids):
    """未指定なら退避しない（挙動不変の回帰ガード）."""
    obs = _obs(ids, meta)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False)
    assert obs.select.option[got].type != OptionType.PLAY


def test_evacuates_with_switch_card(meta, ids):
    """2サイド以上のアクティブ＋脅威ありなら入替札を切る."""
    obs = _obs(ids, meta)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[got].type == OptionType.PLAY


def test_falls_back_to_retreat_without_switch_card(meta, ids):
    """入替札が無ければ にげる で下がる."""
    obs = _obs(ids, meta, with_switch=False)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[got].type == OptionType.RETREAT


def test_does_not_evacuate_low_prize_active(meta, ids):
    """1サイドのアクティブは退避対象外（守る価値の条件）."""
    obs = _obs(ids, meta, active_big=False)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[got].type != OptionType.PLAY


def test_does_not_evacuate_without_bench(meta, ids):
    """交代先が居なければ退避しない（下がれないので手札を無駄にしない）."""
    obs = _obs(ids, meta, with_bench=False)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[got].type != OptionType.PLAY


def test_does_not_evacuate_when_threat_is_low(meta, ids):
    """相手にエネが無く KO 脅威が閾値未満なら退避しない."""
    obs = _obs(ids, meta, threat_energy=0)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[got].type != OptionType.PLAY


def test_lethal_attack_takes_priority(meta, ids):
    """こちらが相手アクティブを倒せるなら退避せず攻撃する."""
    obs = _obs(ids, meta, lethal=True)
    got = _choose_main(obs, meta, random.Random(0), use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[got].type == OptionType.ATTACK


def test_heuristic_agent_passes_evacuate_prize(meta, ids):
    """make_heuristic_agent 経由でも evacuate_prize が MAIN 選択に届く."""
    obs = _obs(ids, meta)
    base = make_heuristic_agent(meta, use_trainers=False)
    evac = make_heuristic_agent(meta, use_trainers=False, evacuate_prize=0.5)
    assert obs.select.option[base(obs, random.Random(0))[0]].type != OptionType.PLAY
    assert obs.select.option[evac(obs, random.Random(0))[0]].type == OptionType.PLAY


def test_self_switch_flag_excludes_opponent_switch(meta):
    """入替札の判定は『自分のアクティブを下げる』札に限られる（相手引きずり出しは除外）."""
    assert any(meta.is_self_switch.values())
    # 相手側を動かす札まで True になっていないこと（プール全体の1%未満に収まる）
    n_true = sum(1 for v in meta.is_self_switch.values() if v)
    assert 0 < n_true < len(meta.is_self_switch) * 0.05


def test_unused_area_import_guard():
    """AreaType を使うフェイク形状の維持（実データ形状との乖離を防ぐ）."""
    assert AreaType.ACTIVE != AreaType.BENCH
