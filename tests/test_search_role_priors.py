"""mine_search_role_priors（§50/§51・山札サーチ/捨て札コストの役割別優先度）のテスト.

観測は cabt を通さず軽量フェイク（SimpleNamespace）で組む。実データ形状
（option は cardId を持たず index で sel.deck/hand を指す・§49/§51）に合わせる。
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
from cg.api import CardType, SelectContext, SelectType  # noqa: E402
from mine_search_role_priors import (  # noqa: E402
    _discard_card_id,
    _record_discard,
    _record_lift,
    _record_search,
    card_role,
    is_search_cost_discard,
    mine_episode,
    tohand_bucket,
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


@pytest.fixture(scope="module")
def search_trainer_id(meta):
    """ability_effect に search ビットが立つ実在トレーナー ID（§51 の discard-cost 判定用）."""
    from cards import EFFECT_CATEGORIES

    bit = 1 << EFFECT_CATEGORIES.index("search")
    return next(
        c
        for c, eff in meta.ability_effect.items()
        if eff & bit and meta.card_type.get(c) in (CardType.ITEM, CardType.SUPPORTER)
    )


def test_card_role_classifies_by_category(meta, ids):
    assert card_role(ids["basic"], meta) == "basic_pokemon"
    assert card_role(ids["evo"], meta) == "evolution_pokemon"
    assert card_role(ids["energy"], meta) == "basic_energy"
    assert card_role(ids["recycle"], meta) == "recycle"  # 他カテゴリより優先


def test_card_role_unknown_id_is_other(meta):
    assert card_role(-999, meta) == "other"


def _search_obs(card_ids, max_count=1, min_count=0, context=None, supporter_played=False):
    """実データ形状の山札サーチ観測フェイク（§49: option は index で deck を指す）."""
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
        effect=None,
    )
    cur = NS(yourIndex=0, players=[NS(), NS()], supporterPlayed=supporter_played)
    return NS(current=cur, select=sel)


def _discard_obs(hand_card_ids, effect_cid=None):
    """実データ形状の DISCARD 観測フェイク（§51: deck は None・手札由来・area=HAND）."""
    opts = [
        NS(index=i, area=None, playerIndex=0, type=0, cardId=None, number=None)
        for i in range(len(hand_card_ids))
    ]
    sel = NS(
        type=SelectType.CARD,
        context=SelectContext.DISCARD,
        option=opts,
        minCount=0,
        maxCount=1,
        deck=None,
        effect=NS(id=effect_cid) if effect_cid is not None else None,
    )
    hand = [NS(id=c) for c in hand_card_ids]
    me = NS(hand=hand)
    cur = NS(yourIndex=0, players=[me, NS()], supporterPlayed=False)
    return NS(current=cur, select=sel)


def test_record_search_accumulates(meta, ids):
    counts = defaultdict(lambda: [0, 0])
    obs = _search_obs([ids["basic"], ids["energy"], ids["basic"]], max_count=2)
    _record_search(obs.select, [0, 1], meta, counts)
    assert counts["basic_pokemon"] == [1, 2]  # 2枚提示・1枚取得
    assert counts["basic_energy"] == [1, 1]


def test_tohand_bucket_non_tohand_context_is_none(meta, ids):
    obs = _search_obs([ids["basic"]], context=SelectContext.TO_BENCH)
    assert tohand_bucket(obs, obs.select, meta) is None


def test_tohand_bucket_single_type_vs_mixed(meta, ids):
    """§50: 1局面内の card_type が単一か複数混在かで single/mixed を分ける."""
    single = _search_obs(
        [ids["basic"], ids["basic"]], context=SelectContext.TO_HAND, supporter_played=False
    )
    assert tohand_bucket(single, single.select, meta) == "tohand_single_before"

    mixed = _search_obs(
        [ids["basic"], ids["energy"]], context=SelectContext.TO_HAND, supporter_played=True
    )
    assert tohand_bucket(mixed, mixed.select, meta) == "tohand_mixed_after"


def test_record_lift_uniform_choice_gives_lift_one(meta, ids):
    """§51: 候補数に比例した取得（一様選択相当）なら lift=1.0 になる.

    2枚提示・1枚取得を2局面（各役割から1回ずつ取る）繰り返すと、各役割の
    実際の取得数と期待値（一様選択なら 1枚/2枚 の期待）が一致する。
    """
    lift = defaultdict(lambda: [0.0, 0.0])
    obs1 = _search_obs([ids["basic"], ids["energy"]], max_count=1)
    _record_lift(obs1.select, [0], meta, lift)  # basic を取得
    obs2 = _search_obs([ids["basic"], ids["energy"]], max_count=1)
    _record_lift(obs2.select, [1], meta, lift)  # energy を取得
    # 各役割: offered=2局面, taken=1局面 → 期待値も 1局面(=k×offered/N=1×1/2 を2回=1.0)
    t_b, e_b = lift["basic_pokemon"]
    t_e, e_e = lift["basic_energy"]
    assert t_b / e_b == pytest.approx(1.0)
    assert t_e / e_e == pytest.approx(1.0)


def test_record_lift_over_represented_role_has_lift_above_one(meta, ids):
    """§51: 候補数に対し取り過ぎている役割は lift>1（優先されている代理指標）."""
    lift = defaultdict(lambda: [0.0, 0.0])
    # 3枚中1枚が basic、常に basic だけを取る → basic は候補数なりの期待(1/3)を上回る
    obs = _search_obs([ids["basic"], ids["energy"], ids["evo"]], max_count=1)
    _record_lift(obs.select, [0], meta, lift)
    t_b, e_b = lift["basic_pokemon"]
    assert t_b / e_b > 1.0
    t_e, e_e = lift["basic_energy"]
    assert t_e / e_e == 0.0  # 一度も取られていない


def test_is_search_cost_discard_checks_effect_search_bit(meta, ids, search_trainer_id):
    """§51: sel.effect.id の効果が search ビットを持つかで判定する."""
    obs = _discard_obs([ids["basic"]], effect_cid=search_trainer_id)
    assert is_search_cost_discard(obs.select, meta) is True

    obs_no_search = _discard_obs([ids["basic"]], effect_cid=ids["energy"])
    assert is_search_cost_discard(obs_no_search.select, meta) is False

    obs_no_effect = _discard_obs([ids["basic"]], effect_cid=None)
    assert is_search_cost_discard(obs_no_effect.select, meta) is False


def test_discard_card_id_resolves_via_hand_not_deck(meta, ids):
    """§51: DISCARD は sel.deck でなく me.hand[index].id で解決する（実データ形状）."""
    obs = _discard_obs([ids["energy"], ids["basic"]])
    assert _discard_card_id(obs, obs.select, 0) == ids["energy"]
    assert _discard_card_id(obs, obs.select, 1) == ids["basic"]


def test_record_discard_accumulates_with_taken_semantics(meta, ids):
    """§51: 取得率でなく『廃棄率』（選ばれた=捨てられた）として集計する."""
    counts = defaultdict(lambda: [0, 0])
    obs = _discard_obs([ids["basic"], ids["energy"]])
    _record_discard(obs, obs.select, [0], meta, counts)
    assert counts["basic_pokemon"] == [1, 1]  # 捨てられた
    assert counts["basic_energy"] == [0, 1]  # 提示のみ（温存された）


def test_mine_episode_pairs_action_with_next_step(meta, ids):
    """ペアリングは §45 の正実装: obs[i] への応答は steps[i+1].action。
    overall と tohand_* の両方に加算される（§50 の2軸バケット）."""
    obs_dict = {
        "current": {
            "turn": 3,
            "turnActionCount": 1,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": True,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 30,
                 "discard": [], "prize": [], "handCount": 0, "hand": [],
                 "poisoned": False, "burned": False, "asleep": False,
                 "paralyzed": False, "confused": False},
                {"active": [], "bench": [], "benchMax": 5, "deckCount": 30,
                 "discard": [], "prize": [], "handCount": 0, "hand": None,
                 "poisoned": False, "burned": False, "asleep": False,
                 "paralyzed": False, "confused": False},
            ],
        },
        "logs": [],
        "select": {
            "type": int(SelectType.CARD),
            "context": int(SelectContext.TO_HAND),
            "option": [{"type": 0, "index": 0}, {"type": 0, "index": 1}],
            "minCount": 0,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "contextCard": None,
            "effect": None,
            "deck": [
                {"id": ids["basic"], "serial": 0, "playerIndex": 0},
                {"id": ids["energy"], "serial": 1, "playerIndex": 0},
            ],
        },
    }
    steps = [
        [{"status": "ACTIVE", "observation": obs_dict, "action": None}],
        [{"status": "ACTIVE", "observation": None, "action": [1]}],  # ↑への応答
    ]
    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    bucket_n = defaultdict(int)
    lift = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    assert mine_episode({"steps": steps}, 0, meta, buckets, bucket_n, lift) == 1
    assert buckets["overall"]["basic_energy"] == [1, 1]
    assert buckets["overall"]["basic_pokemon"] == [0, 1]
    # 型混在（たね+エネ）× サポート使用後
    assert buckets["tohand_mixed_after"]["basic_energy"] == [1, 1]
    assert buckets["tohand_mixed_after"]["basic_pokemon"] == [0, 1]
    assert "tohand_single_before" not in buckets
    assert bucket_n == {"overall": 1, "tohand_mixed_after": 1}
    # mixed バケットなので lift も加算される
    assert "basic_energy" in lift["tohand_mixed_after"]
