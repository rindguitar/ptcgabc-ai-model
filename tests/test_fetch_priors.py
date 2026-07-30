"""fetch_priors（§47・山札サーチの取得優先度注入）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。
カード ID はダミー整数（規約: Pokémon Elements を持ち込まない）。
"""

import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from agents import _generic_select, make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import CardType, SelectContext, SelectType  # noqa: E402
from mine_fetch_priorities import _count_search, mine_episode  # noqa: E402
from submission import _load_fetch_priors  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


@pytest.fixture(scope="module")
def ids(meta):
    """メタから カテゴリ別の実在 ID を1つずつ拾う（値はダミー扱い・名前等は使わない）."""
    basic = next(c for c in meta.card_type if meta.is_basic_pokemon(c))
    energy = next(
        c
        for c, t in meta.card_type.items()
        if t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
    )
    other = next(
        c
        for c, t in meta.card_type.items()
        if not meta.is_basic_pokemon(c)
        and t not in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
    )
    return {"basic": basic, "energy": energy, "other": other}


def _search_obs(
    card_ids, max_count=1, min_count=0, context=None, supporter_played=False
):
    """山札サーチ（sel.deck あり）の観測フェイクを組む.

    ⚠️ 実データ形状（replay JSON で実測確認済み）: option は cardId を持たず
    area/index/playerIndex/type のみ。実カードは sel.deck[option.index].id で解決する
    （option[i] は deck[i] を指す 1:1 対応にして単純化）。cardId=None を明示し、
    実装が誤って cardId を直読みしないことをテストで保証する。
    supporterPlayed は State の必須フィールド（§69 のカテゴリ順分岐が読む）。
    """
    deck = [NS(id=c) for c in card_ids]
    opts = [
        NS(index=i, area=None, playerIndex=0, type=0, cardId=None, number=None)
        for i in range(len(card_ids))
    ]
    sel = NS(
        type=SelectType.CARD,
        context=context,
        option=opts,
        minCount=min_count,
        maxCount=max_count,
        deck=deck,
    )
    current = NS(yourIndex=0, players=[NS(), NS()], supporterPlayed=supporter_played)
    return NS(current=current, select=sel)


def test_search_without_priors_prefers_energy(meta, ids):
    """priors 未指定は従来挙動: カテゴリ順（エネ > その他）.

    たね優先だった旧順は §50/§51 の上位帯 replay 実測（局面数136の型混在局面で
    たねの選好指数 lift=0.46＝明確に忌避）で撤回済み（§52）。
    """
    obs = _search_obs([ids["other"], ids["basic"], ids["energy"]])
    assert _generic_select(obs, meta) == [2]  # エネ


def test_search_with_priors_prefers_high_rate(meta, ids):
    """priors がある札は取得率の降順でカテゴリ順より前に出る."""
    obs = _search_obs([ids["other"], ids["energy"], ids["basic"]])
    # 「その他」枠のキーカード（例: リサイクル札）が最優先になる
    assert _generic_select(obs, meta, {ids["other"]: 0.9}) == [0]
    # priors 同士は率の降順
    obs2 = _search_obs([ids["other"], ids["basic"]], max_count=1)
    assert _generic_select(obs2, meta, {ids["other"]: 0.4, ids["basic"]: 0.8}) == [1]


def test_search_priors_take_multiple_in_rate_order(meta, ids):
    """maxCount>1: priors 札 → カテゴリ順の残りの順で埋める（返り値は index 昇順）."""
    obs = _search_obs([ids["energy"], ids["other"], ids["basic"]], max_count=2)
    got = _generic_select(obs, meta, {ids["other"]: 0.9})
    assert got == sorted([0, 1])  # priors 札(other) + エネ（たねは§52で撤回済み・落ちる）


def test_heuristic_agent_passes_priors(meta, ids):
    """make_heuristic_agent 経由でも priors が効く（サブ選択に委譲される）."""
    agent = make_heuristic_agent(meta, fetch_priors={ids["other"]: 0.9})
    obs = _search_obs([ids["basic"], ids["other"]])
    assert agent(obs, random.Random(0)) == [1]


def test_search_resolves_card_via_deck_index_not_option_cardid(meta, ids):
    """option.cardId が None でも sel.deck[option.index].id で正しく解決できる（実データ形状）.

    option の並びが deck の並びと一致しない（先頭 option が deck 位置1=その他 を指す）
    ケースでも、index 経由の解決なら正しくエネを優先できる。cardId 直読みの旧バグでは
    どちらも cid=0 に落ちて同tier→tie-break で先頭 option（その他）を誤って選ぶため、
    エネが後方の option を指す配置にして回帰ガードにしている。
    """
    deck = [NS(id=ids["energy"]), NS(id=ids["other"])]
    opts = [
        NS(index=1, area=None, playerIndex=0, type=0, cardId=None, number=None),
        NS(index=0, area=None, playerIndex=0, type=0, cardId=None, number=None),
    ]
    sel = NS(type=SelectType.CARD, context=None, option=opts, minCount=0, maxCount=1, deck=deck)
    current = NS(yourIndex=0, players=[NS(), NS()], supporterPlayed=False)
    obs = NS(current=current, select=sel)
    assert _generic_select(obs, meta) == [1]  # option[1]->deck[0]=エネ を選ぶ


def test_search_prior_zero_is_not_promoted(meta, ids):
    """取得率 0 の札は tier0 に入れない（§68/§69 で見つかった実害の是正）.

    教師に提示されたのに一度も取られなかった札（率 0.0）を「priors に載っている」
    だけで最優先していた旧実装では、エネが押しのけられていた。
    """
    obs = _search_obs([ids["other"], ids["energy"]])
    assert _generic_select(obs, meta, {ids["other"]: 0.0}) == [1]  # エネを取る
    # 率 > 0 なら従来どおり前に出る（回帰ガード）
    assert _generic_select(obs, meta, {ids["other"]: 0.01}) == [0]


def test_search_tohand_before_supporter_prefers_basic(meta, ids):
    """TO_HAND × サポート未使用: たね > エネ > その他（教師 lift 1.93・§69）."""
    obs = _search_obs(
        [ids["other"], ids["energy"], ids["basic"]],
        max_count=2,
        context=SelectContext.TO_HAND,
        supporter_played=False,
    )
    assert _generic_select(obs, meta) == sorted([2, 1])  # たね → エネ


def test_search_tohand_after_supporter_demotes_basic(meta, ids):
    """TO_HAND × サポート使用後: エネ > その他 > たね（教師 lift 0.46・§69）."""
    obs = _search_obs(
        [ids["basic"], ids["other"], ids["energy"]],
        max_count=2,
        context=SelectContext.TO_HAND,
        supporter_played=True,
    )
    assert _generic_select(obs, meta) == sorted([2, 1])  # エネ → その他（たねは最後）


def test_search_non_tohand_context_keeps_energy_first(meta, ids):
    """TO_HAND 以外は分岐しない（採掘バケットが TO_HAND 限定・構造的に無風）."""
    obs = _search_obs(
        [ids["other"], ids["energy"], ids["basic"]],
        context=SelectContext.TO_BENCH,
        supporter_played=False,
    )
    assert _generic_select(obs, meta) == [1]  # エネ


def test_count_search_accumulates_taken_and_offered():
    """_count_search: 提示は全 option・取得は応答 index の札に加算。非サーチは無視."""
    counts = defaultdict(lambda: [0, 0])
    obs = _search_obs([11, 22, 22], max_count=2)
    assert _count_search(obs, [0, 2], counts) is True
    assert counts[11] == [1, 1]
    assert counts[22] == [1, 2]  # 2枚提示・1枚取得

    non_search = _search_obs([11])
    non_search.select.deck = None
    assert _count_search(non_search, [0], counts) is False
    assert counts[11] == [1, 1]  # 変化なし


def test_mine_episode_pairs_action_with_next_step():
    """mine_episode: obs[i] への応答は steps[i+1].action（§45 の正実装ペアリング）."""
    obs_dict = {
        "current": None,
        "logs": [],
        "select": {
            "type": int(SelectType.CARD),
            # ⚠️ 実データ形状: option は cardId を持たず index で deck を指す（実測確認済み）
            "option": [
                {"type": 0, "index": 0},
                {"type": 0, "index": 1},
            ],
            "minCount": 0,
            "maxCount": 1,
            "context": 0,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "contextCard": None,
            "effect": None,
            "deck": [
                {"id": 11, "serial": 0, "playerIndex": 0},
                {"id": 22, "serial": 1, "playerIndex": 0},
            ],
        },
    }
    steps = [
        [{"status": "ACTIVE", "observation": obs_dict, "action": None}],
        [{"status": "ACTIVE", "observation": None, "action": [1]}],  # ↑への応答
    ]
    counts = defaultdict(lambda: [0, 0])
    assert mine_episode({"steps": steps}, 0, counts) == 1
    assert counts[22] == [1, 1]  # 同一 step でペアにすると [0] を数えてしまう
    assert counts[11] == [0, 1]


def test_load_fetch_priors_reads_json(tmp_path):
    """_load_fetch_priors: priors キー形式と素の辞書の両方を int キーで読む。無ければ None."""
    p = tmp_path / "fetch_priors.json"
    p.write_text(json.dumps({"priors": {"11": 0.9, "22": 0.5}, "team": "t"}))
    assert _load_fetch_priors(str(p)) == {11: 0.9, 22: 0.5}
    p2 = tmp_path / "raw.json"
    p2.write_text(json.dumps({"33": 0.7}))
    assert _load_fetch_priors(str(p2)) == {33: 0.7}
    assert _load_fetch_priors(str(tmp_path / "missing.json")) is None
