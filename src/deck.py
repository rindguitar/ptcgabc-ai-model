"""デッキの正当性チェック・変異・構成統計.

デッキは**カード ID の 60 要素リスト**。カード名（Pokémon Elements）は扱わず、ID と数値メタのみ。
合法性の判定は**エンジンを正**とする（4枚制限・基本エネ無制限・たね必須・ACE SPEC 等の規則を
自前で再実装せず、battle_start が受理するかで判定する）。
"""

from __future__ import annotations

import random
from collections import Counter

from cards import CardMeta
from cg.api import CardType
from cg.game import battle_finish, battle_start

DECK_SIZE = 60
# 大会リーガルなカード ID（1..1267）。determinize/最適化の探索空間。
CARD_POOL = list(range(1, 1268))


def load_deck(path: str) -> list[int]:
    """deck CSV（1 行 1 カード ID）を読み込む."""
    with open(path) as f:
        return [int(line) for line in f.read().splitlines() if line.strip()]


def save_deck(deck: list[int], path: str) -> None:
    """deck をカード ID の CSV（1 行 1 枚）で書き出す."""
    with open(path, "w") as f:
        f.write("\n".join(str(c) for c in deck) + "\n")


def is_legal(deck: list[int], ref_deck: list[int]) -> bool:
    """deck が合法か。既知の合法 ref_deck と対戦開始できるかで判定する.

    エンジンは両デッキを検証し、不正なら battle_start が None を返す（ref は合法前提）。
    """
    if len(deck) != DECK_SIZE:
        return False
    obs, _ = battle_start(deck, ref_deck)
    if obs is not None:
        battle_finish()
        return True
    return False


def mutate(
    deck: list[int],
    ref_deck: list[int],
    rng: random.Random,
    pool: list[int] = CARD_POOL,
    max_tries: int = 50,
) -> list[int]:
    """1 枚を別のカードに差し替える合法性保存の変異。失敗時は元のデッキを返す."""
    for _ in range(max_tries):
        cand = list(deck)
        cand[rng.randrange(DECK_SIZE)] = rng.choice(pool)
        if is_legal(cand, ref_deck):
            return cand
    return list(deck)


def random_legal_deck(
    ref_deck: list[int],
    rng: random.Random,
    swaps: int = 40,
    pool: list[int] = CARD_POOL,
) -> list[int]:
    """既知の合法デッキから多数回変異して、別アーキタイプ寄りの合法デッキを作る.

    多様性注入（探索）用。ゼロから合法デッキを組むより堅実（各変異をエンジンで合法性検証）。
    """
    deck = list(ref_deck)
    for _ in range(swaps):
        deck = mutate(deck, ref_deck, rng, pool)
    return deck


def composition(deck: list[int], meta: CardMeta) -> dict[str, int]:
    """デッキの種別構成（ポケモン/グッズ/道具/サポート/スタジアム/エネ）を数える."""
    counts = Counter(meta.card_type.get(cid) for cid in deck)
    return {
        "pokemon": counts.get(CardType.POKEMON, 0),
        "item": counts.get(CardType.ITEM, 0),
        "tool": counts.get(CardType.TOOL, 0),
        "supporter": counts.get(CardType.SUPPORTER, 0),
        "stadium": counts.get(CardType.STADIUM, 0),
        "basic_energy": counts.get(CardType.BASIC_ENERGY, 0),
        "special_energy": counts.get(CardType.SPECIAL_ENERGY, 0),
        "unique": len(set(deck)),
    }
