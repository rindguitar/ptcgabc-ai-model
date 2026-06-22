"""デッキリーグ（bounded double-oracle）の軽量テスト."""

import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DECK = os.path.join(ROOT, "data", "deck.csv")
sys.path.insert(0, SRC)

pytest.importorskip("cg.sim", reason="cabt Engine (cg) が見つからない")
if not os.path.exists(DECK):
    pytest.skip("data/deck.csv が無い", allow_module_level=True)

from cards import load_card_meta  # noqa: E402
from deck import DECK_SIZE, is_legal, load_deck, mutate  # noqa: E402
from league import (  # noqa: E402
    _most_redundant_index,
    deck_similarity,
    load_state,
    run_league,
)


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return load_deck(DECK)


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def test_deck_similarity_bounds(deck):
    """同一デッキの類似度は 1.0、別物ほど小さい."""
    assert deck_similarity(deck, deck) == 1.0
    other = [1] * DECK_SIZE  # 基本エネ60枚（重なり小）
    assert 0.0 <= deck_similarity(deck, other) < 1.0


def test_most_redundant_index():
    """最も冗長（重複が多い）デッキの添字を返す."""
    a = list(range(1, 61))
    b = list(range(1, 61))  # a と同一 → 冗長
    c = list(range(61, 121))  # 異質
    idx = _most_redundant_index([a, b, c])
    assert idx in (0, 1)  # a か b（互いに完全一致）が冗長


def test_run_league_returns_legal_champion(deck, meta):
    """リーグが 60枚・合法のチャンピオンを返す（小規模）."""
    rng = random.Random(0)
    deck2 = mutate(deck, deck, rng)
    result = run_league(
        [deck, deck2],
        meta,
        rng=rng,
        cap=3,
        iterations=2,
        games_per_opp=4,
        plateau=5,
        evolve_kwargs={
            "pop_size": 4,
            "generations": 2,
            "mutations_per_child": 1,
            "elite": 2,
        },
    )
    champ = result["champion"]
    assert len(champ) == DECK_SIZE
    assert is_legal(champ, deck)
    assert result["archive_size"] >= 2
    assert 0.0 <= result["champion_worstcase_vs_archive"] <= 1.0


def test_checkpoint_and_resume(deck, meta, tmp_path):
    """チェックポイント保存と --resume での続行（done_iters と archive が増える）."""
    cp = str(tmp_path / "league" / "state.json")
    kw = {
        "pop_size": 4,
        "generations": 2,
        "mutations_per_child": 1,
        "elite": 2,
    }
    deck2 = mutate(deck, deck, random.Random(1))

    # 1回目: 1反復 → チェックポイント保存
    run_league(
        [deck, deck2],
        meta,
        rng=random.Random(0),
        cap=3,
        iterations=1,
        games_per_opp=4,
        plateau=99,
        checkpoint_path=cp,
        evolve_kwargs=kw,
    )
    st1 = load_state(cp)
    assert st1["done_iters"] == 1

    # 2回目: resume で 1反復追加 → done_iters=2, archive が増える
    run_league(
        [deck, deck2],
        meta,
        rng=random.Random(0),
        cap=3,
        iterations=1,
        games_per_opp=4,
        plateau=99,
        checkpoint_path=cp,
        resume=True,
        evolve_kwargs=kw,
    )
    st2 = load_state(cp)
    assert st2["done_iters"] == 2
    assert len(st2["archive"]) > len(st1["archive"])
