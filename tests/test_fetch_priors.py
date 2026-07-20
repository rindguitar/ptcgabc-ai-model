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
from cg.api import CardType, SelectType  # noqa: E402
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


def _search_obs(card_ids, max_count=1, min_count=0):
    """山札サーチ（sel.deck あり）の観測フェイクを組む."""
    opts = [NS(cardId=c, number=None, index=i) for i, c in enumerate(card_ids)]
    sel = NS(
        type=SelectType.CARD,
        context=None,
        option=opts,
        minCount=min_count,
        maxCount=max_count,
        deck=[1] * 10,
    )
    return NS(current=NS(yourIndex=0, players=[NS(), NS()]), select=sel)


def test_search_without_priors_prefers_basic(meta, ids):
    """priors 未指定は従来挙動: カテゴリ順（たね > エネ > その他）."""
    obs = _search_obs([ids["other"], ids["energy"], ids["basic"]])
    assert _generic_select(obs, meta) == [2]  # たね


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
    assert got == sorted([1, 2])  # priors 札 + たね（エネは落ちる）


def test_heuristic_agent_passes_priors(meta, ids):
    """make_heuristic_agent 経由でも priors が効く（サブ選択に委譲される）."""
    agent = make_heuristic_agent(meta, fetch_priors={ids["other"]: 0.9})
    obs = _search_obs([ids["basic"], ids["other"]])
    assert agent(obs, random.Random(0)) == [1]


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
            "option": [
                {"type": 0, "cardId": 11},
                {"type": 0, "cardId": 22},
            ],
            "minCount": 0,
            "maxCount": 1,
            "context": 0,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "contextCard": None,
            "effect": None,
            "deck": [{"id": 1, "serial": 0, "playerIndex": 0}],
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
