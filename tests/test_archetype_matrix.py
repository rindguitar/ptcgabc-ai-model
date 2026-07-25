"""archetype_matrix（Layer 1・型マッチアップ行列）のテスト.

カード ID はダミー整数・meta は SimpleNamespace のフェイク（規約: Pokémon Elements
を持ち込まない）。cg 依存は CardType のみのため importorskip で緩和する。
"""

import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytest.importorskip("cg.api", reason="cabt Engine (cg) が見つからない")

from cg.api import CardType  # noqa: E402

from archetype_matrix import (  # noqa: E402
    build_matrix,
    cluster_decks,
    deck_features,
    deck_hash,
    episode_matchup,
    kmeans,
)

# フェイク meta: id 1-9=たねポケ・10-19=進化ポケ・20-29=エネ・他=トレーナー扱い
_TYPES = {i: CardType.POKEMON for i in range(1, 20)}
_TYPES.update({i: CardType.BASIC_ENERGY for i in range(20, 30)})
FAKE_META = NS(
    card_type=_TYPES,
    is_basic_pokemon=lambda cid: 1 <= cid < 10,
)


def test_deck_features_counts():
    # たね2 + 進化1 + エネ3 + トレーナー2（ユニーク6・ダミー枚数8）
    ids = [1, 1, 12, 20, 20, 21, 40, 41]
    poke, ene, basic, uniq = deck_features(ids, FAKE_META)
    assert (poke, ene, basic, uniq) == (3.0, 3.0, 2.0, 6.0)


def test_kmeans_separates_blobs_deterministically():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.1, size=(20, 2))
    b = rng.normal(5.0, 0.1, size=(20, 2))
    x = np.vstack([a, b])
    l1 = kmeans(x, k=2, seed=0)
    l2 = kmeans(x, k=2, seed=0)
    # 決定性 ＆ 2 つの塊が分離される（ラベル値は任意なので集合で比較）
    assert (l1 == l2).all()
    assert len(set(l1[:20])) == 1 and len(set(l1[20:])) == 1
    assert l1[0] != l1[20]


def test_cluster_decks_groups_similar_compositions():
    # ポケ多デッキ 2 種とエネ多デッキ 2 種 → k=2 で同型同士が同クラスタ
    poke_heavy = [1, 2, 3, 4, 5, 12, 13] * 8 + [40, 41, 42, 43]
    ene_heavy = [20, 21, 22, 23, 24] * 10 + [1, 2, 3, 40, 41, 42, 43, 44, 45, 46]
    decks = {
        "aaaa": poke_heavy[:60],
        "bbbb": (poke_heavy[1:] + [6])[:60],
        "cccc": ene_heavy[:60],
        "dddd": (ene_heavy[1:] + [25])[:60],
    }
    assign, centroids = cluster_decks(decks, FAKE_META, k=2, seed=0)
    assert assign["aaaa"] == assign["bbbb"]
    assert assign["cccc"] == assign["dddd"]
    assert assign["aaaa"] != assign["cccc"]
    assert centroids.shape == (2, 4)


def _fake_episode(deck0, deck1, rewards):
    # 初手 action にデッキ 60 枚が入る実データ形状（analyze_replays._deck_of 準拠）
    return {
        "steps": [[{"action": deck0}, {"action": deck1}]],
        "rewards": rewards,
    }


def test_episode_matchup_extracts_hashes_and_winner():
    d0 = list(range(1, 61))
    d1 = list(range(2, 62))
    m = episode_matchup(_fake_episode(d0, d1, [1, 0]))
    assert m == (deck_hash(d0), deck_hash(d1), 0)
    # 引分・欠損は None
    assert episode_matchup(_fake_episode(d0, d1, [0, 0])) is None
    assert episode_matchup(_fake_episode(d0, None, [1, 0])) is None


def test_build_matrix_accumulates_both_directions():
    assign = {"h0": 0, "h1": 1}
    matchups = [("h0", "h1", 0), ("h0", "h1", 0), ("h1", "h0", 0), ("h0", "xx", 0)]
    wins, games = build_matrix(matchups, assign, k=2)
    # F0 勝 2・F1 勝 1・計 3 試合（未知 hash "xx" の試合は捨てる）
    assert games[0, 1] == games[1, 0] == 3
    assert wins[0, 1] == 2 and wins[1, 0] == 1
