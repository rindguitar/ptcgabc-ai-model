"""mine_search_role_priors（§50・山札サーチの役割別優先度）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。実データ形状
（option は cardId を持たず index で sel.deck を指す・§49）に合わせる。
カード ID はダミー整数（規約: Pokémon Elements を持ち込まない）。
"""

import os
import sys
from collections import defaultdict
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from cards import load_card_meta  # noqa: E402
from cg.api import CardType, SelectType  # noqa: E402
from mine_search_role_priors import (  # noqa: E402
    _count_search_roles,
    card_role,
    mine_episode_roles,
)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


@pytest.fixture(scope="module")
def ids(meta):
    """メタから役割別の実在 ID を1つずつ拾う（値はダミー扱い・名前等は使わない）."""

    def find(pred):
        return next(c for c in meta.card_type if pred(c))

    basic = find(lambda c: meta.is_basic_pokemon(c))
    evo = find(
        lambda c: meta.card_type.get(c) == CardType.POKEMON and not meta.is_basic.get(c)
    )
    energy = find(lambda c: meta.card_type.get(c) == CardType.BASIC_ENERGY)
    recycle = find(lambda c: meta.is_deck_recycle.get(c, False))
    return {"basic": basic, "evo": evo, "energy": energy, "recycle": recycle}


def test_card_role_classifies_by_category(meta, ids):
    assert card_role(ids["basic"], meta) == "basic_pokemon"
    assert card_role(ids["evo"], meta) == "evolution_pokemon"
    assert card_role(ids["energy"], meta) == "basic_energy"
    assert card_role(ids["recycle"], meta) == "recycle"  # 他カテゴリより優先


def test_card_role_unknown_id_is_other(meta):
    assert card_role(-999, meta) == "other"


def _search_obs(card_ids, max_count=1, min_count=0):
    """実データ形状の山札サーチ観測フェイク（§49: option は index で deck を指す）."""
    deck = [NS(id=c) for c in card_ids]
    opts = [
        NS(index=i, area=None, playerIndex=0, type=0, cardId=None, number=None)
        for i in range(len(card_ids))
    ]
    sel = NS(type=SelectType.CARD, context=None, option=opts, minCount=min_count, maxCount=max_count, deck=deck)
    return NS(current=NS(yourIndex=0, players=[NS(), NS()]), select=sel)


def test_count_search_roles_accumulates(meta, ids):
    counts = defaultdict(lambda: [0, 0])
    obs = _search_obs([ids["basic"], ids["energy"], ids["basic"]], max_count=2)
    assert _count_search_roles(obs, [0, 1], meta, counts) is True
    assert counts["basic_pokemon"] == [1, 2]  # 2枚提示・1枚取得
    assert counts["basic_energy"] == [1, 1]

    non_search = _search_obs([ids["basic"]])
    non_search.select.deck = None
    assert _count_search_roles(non_search, [0], meta, counts) is False


def test_mine_episode_roles_pairs_action_with_next_step(meta, ids):
    """ペアリングは §45 の正実装: obs[i] への応答は steps[i+1].action."""
    obs_dict = {
        "current": None,
        "logs": [],
        "select": {
            "type": int(SelectType.CARD),
            "option": [{"type": 0, "index": 0}, {"type": 0, "index": 1}],
            "minCount": 0,
            "maxCount": 1,
            "context": 0,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "contextCard": None,
            "effect": None,
            "deck": [{"id": ids["basic"], "serial": 0, "playerIndex": 0},
                     {"id": ids["energy"], "serial": 1, "playerIndex": 0}],
        },
    }
    steps = [
        [{"status": "ACTIVE", "observation": obs_dict, "action": None}],
        [{"status": "ACTIVE", "observation": None, "action": [1]}],  # ↑への応答
    ]
    counts = defaultdict(lambda: [0, 0])
    assert mine_episode_roles({"steps": steps}, 0, meta, counts) == 1
    assert counts["basic_energy"] == [1, 1]  # 取得
    assert counts["basic_pokemon"] == [0, 1]  # 提示のみ
