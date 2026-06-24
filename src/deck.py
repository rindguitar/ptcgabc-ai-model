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


def structured_deck(
    template: list[int],
    meta: CardMeta,
    rng: random.Random,
    energy_type: int | None = None,
    pool: list[int] = CARD_POOL,
    top_k: int = 8,
) -> list[int]:
    """テンプレデッキの構造（トレーナー枠）を保ち、ポケモンと基本エネを単一の色 T に
    差し替えた**コヒーレントな mono-type デッキ**を作る.

    ランダムな寄せ集め（random_legal_deck）は「まとまりが無く」ほぼ全敗するため探索で淘汰される。
    本関数は「同色のたねアタッカー＋その色の基本エネ＋実績あるトレーナー群」という**回る別軸**を
    生成し、別アーキタイプの探索を実効化する。名称は扱わず energyType/最大ダメージ等の数値のみ使う。

    威力は「効率（ダメージ/エネコスト）」で評価し、回しにくい大コスト技を過大評価しない。
    さらに**特性持ちのたねポケモン**を候補に含める（特性＝デッキの核になりうるため。意味は読まず
    有無のみ使い、価値は操縦・対戦結果側で判断させる）。

    Args:
        template: 構造の雛形にする合法デッキ（メタデッキ等）。トレーナー枠を流用する。
        energy_type: 差し替える色（None なら基本エネのある色からランダム）。
        top_k: アタッカー候補の上位種数（この中から複数種を選び 4枚制限内で配分）。
    """
    BASIC_ENERGY = CardType.BASIC_ENERGY
    # 差し替える色 T を決める（基本エネが存在する色のみ）
    colors = sorted(meta.basic_energy_id)
    if energy_type is None or energy_type not in meta.basic_energy_id:
        energy_type = rng.choice(colors)
    energy_card = meta.basic_energy_id[energy_type]

    # 色 T のたねポケモン（攻撃役 or 特性持ち）を集める
    mons = [
        cid
        for cid in pool
        if meta.card_type.get(cid) == CardType.POKEMON
        and meta.is_basic.get(cid)
        and meta.energy_type.get(cid) == energy_type
        and (meta.best_damage.get(cid, 0) > 0 or meta.has_ability.get(cid, False))
    ]
    if not mons:  # 候補ゼロの色（理論上ありえないが保険）はテンプレを返す
        return list(template)
    # 攻撃役は威力効率の高い順に top_k、特性持ちは別枠で数種を必ず混ぜる
    attackers = sorted(
        mons, key=lambda c: meta.best_efficiency.get(c, 0.0), reverse=True
    )[:top_k]
    ability_mons = sorted(
        (c for c in mons if meta.has_ability.get(c, False)),
        key=lambda c: meta.best_efficiency.get(c, 0.0),
        reverse=True,
    )[:3]
    candidates = list(dict.fromkeys(attackers + ability_mons))  # 重複除去・順序保持
    rng.shuffle(candidates)

    # テンプレの枠を種別で数える
    n_poke = sum(1 for c in template if meta.card_type.get(c) == CardType.POKEMON)
    new: list[int] = []
    # ポケモン枠: 選んだ種を 4枚制限内で循環配分（最低 1種は使う）
    species = candidates[: max(1, (n_poke + 3) // 4)] or candidates[:1]
    # 特性持ちが候補にあるのに種に入らなければ 1種を必ず混ぜる（特性軸の探索を保証）
    if ability_mons and not any(meta.has_ability.get(s) for s in species):
        species[-1] = rng.choice(ability_mons)
    counts = dict.fromkeys(species, 0)
    for _ in range(n_poke):
        avail = [s for s in species if counts[s] < 4]
        if not avail:  # 種が尽きたら候補から補充
            extra = next((c for c in candidates if c not in counts), None)
            if extra is None:
                break
            species.append(extra)
            counts[extra] = 0
            avail = [extra]
        pick = rng.choice(avail)
        counts[pick] += 1
        new.append(pick)
    # 残り（トレーナー/道具/特殊エネ）と 基本エネ枠 を流用・差し替え
    for c in template:
        ct = meta.card_type.get(c)
        if ct == CardType.POKEMON:
            continue  # 上で再構築済み
        new.append(energy_card if ct == BASIC_ENERGY else c)

    # 60枚に整える（端数は基本エネで調整）。合法でなければテンプレにフォールバック
    new = new[:DECK_SIZE]
    while len(new) < DECK_SIZE:
        new.append(energy_card)
    return new if is_legal(new, template) else list(template)


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
