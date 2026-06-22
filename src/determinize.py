"""ISMCTS 用の determinization（隠れ情報のサンプリング）.

観測（自分視点）から、`search_begin` に渡す各隠れゾーンのカード ID 列を構築する。
隠れ札 = 「そのプレイヤーのデッキ構成」−「観測で見えているカード」を multiset 差で求め、
そこから deck / prize / hand へ必要枚数を無作為に割り当てる（= 1 つの determinization）。

自分のデッキ構成は既知。相手のデッキは本来予測が要るが、当面は引数で与える
（自己対戦では同一デッキを渡す）。観測の取りこぼしに備え、枚数が不足する場合は
デッキから補填して**常に有効なカウント**を返す（防御的）。
"""

from __future__ import annotations

import random
from collections import Counter

from cg.api import Observation


def _zone_card_ids(player, include_hand: bool) -> list[int]:
    """PlayerState の可視ゾーンにあるカード ID を集める（裏向きサイドは含めない）."""
    ids: list[int] = []
    if include_hand and player.hand:
        ids += [c.id for c in player.hand]
    ids += [c.id for c in player.discard]
    for zone in (player.active, player.bench):
        for pk in zone:
            if pk is None:  # 裏向きアクティブ
                continue
            ids.append(pk.id)
            ids += [c.id for c in pk.energyCards]
            ids += [c.id for c in pk.tools]
            ids += [c.id for c in pk.preEvolution]
    ids += [c.id for c in player.prize if c is not None]  # 表向きサイドのみ
    return ids


def _draw(
    pool: list[int], k: int, deck: list[int], rng: random.Random
) -> tuple[list[int], list[int]]:
    """pool から k 枚を無作為抽出し、(抽出, 残り) を返す。足りなければ deck から補填する."""
    if k <= 0:
        return [], pool
    work = list(pool)
    if len(work) < k:
        work += rng.choices(deck, k=k - len(work))  # 観測欠落への保険
    picked = rng.sample(work, k)
    rem = Counter(work)
    for c in picked:
        rem[c] -= 1
    return picked, list(rem.elements())


def pick_opponent_deck(
    obs: Observation,
    candidate_decks: list[list[int]] | None,
    fallback_deck: list[int],
    rng: random.Random,
) -> list[int]:
    """相手デッキを推定する.

    観測で見えている相手のカード（捨て札・場）と**矛盾しない候補デッキ**を candidate_decks
    から選ぶ（複数あれば無作為）。一致候補が無い（未知デッキ）なら fallback_deck（通常は
    自分のデッキ＝ミラー仮定）を返す。→ 既知アーキタイプには当たり、未知でも従来どおり＝悪化しない。
    """
    st = obs.current
    if st is None or not candidate_decks:
        return fallback_deck
    opp = st.players[1 - st.yourIndex]
    seen = Counter(_zone_card_ids(opp, include_hand=False))
    consistent = [
        d
        for d in candidate_decks
        if all(Counter(d).get(c, 0) >= k for c, k in seen.items())
    ]
    return rng.choice(consistent) if consistent else fallback_deck


def determinize(
    obs: Observation,
    my_deck: list[int],
    opp_deck: list[int],
    rng: random.Random,
) -> dict[str, list[int]]:
    """観測から search_begin 用の determinization を 1 つサンプリングする.

    Returns:
        dict: search_begin に渡す your_deck/your_prize/opponent_deck/
              opponent_prize/opponent_hand/opponent_active。
    """
    st = obs.current
    yi = st.yourIndex
    me = st.players[yi]
    opp = st.players[1 - yi]

    # 自分の隠れ札 = 自デッキ − 自分の可視札（手札含む）
    my_hidden = list(
        (Counter(my_deck) - Counter(_zone_card_ids(me, include_hand=True))).elements()
    )
    your_deck, my_hidden = _draw(my_hidden, me.deckCount, my_deck, rng)
    your_prize, my_hidden = _draw(my_hidden, len(me.prize), my_deck, rng)

    # 相手の隠れ札 = 相手デッキ − 相手の可視札（手札は不可視）
    opp_hidden = list(
        (
            Counter(opp_deck) - Counter(_zone_card_ids(opp, include_hand=False))
        ).elements()
    )
    opponent_hand, opp_hidden = _draw(opp_hidden, opp.handCount, opp_deck, rng)
    opponent_deck, opp_hidden = _draw(opp_hidden, opp.deckCount, opp_deck, rng)
    opponent_prize, opp_hidden = _draw(opp_hidden, len(opp.prize), opp_deck, rng)

    # 相手アクティブが裏向きなら 1 枚予測（簡易: 予測デッキ先頭）。通常は自ターンで表向き。
    opponent_active: list[int] = []
    if len(opp.active) > 0 and opp.active[0] is None:
        opponent_active = [opponent_deck[0] if opponent_deck else opp_deck[0]]

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }
