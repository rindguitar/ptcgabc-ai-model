"""デッキの正当性チェック・変異・構成統計.

デッキは**カード ID の 60 要素リスト**。カード名（Pokémon Elements）は扱わず、ID と数値メタのみ。
合法性の判定は**エンジンを正**とする（4枚制限・基本エネ無制限・たね必須・ACE SPEC 等の規則を
自前で再実装せず、battle_start が受理するかで判定する）。
"""

from __future__ import annotations

import random
from collections import Counter

from cards import EFFECT_CATEGORIES, CardMeta
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


# --- 構成の射影（実メタ分布の範囲へ矯正・2026-07-06） --------------------------
# league が「エネ33・グッズ5」のような実メタ外れ値に収束した事故（heuristic ロールアウトが
# トレーナー不使用＝グッズ価値を過小評価する近親交配の歪み）の再発防止。候補デッキを
# カテゴリ枚数の許容帯へ射影する。帯は実メタ70デッキの平均±の緩い範囲＝帯内は自由探索。
COMP_BOUNDS: dict[str, tuple[int, int]] = {
    "poke": (12, 18),
    "item": (10, 17),
    "sup": (8, 14),
    "stad": (0, 3),
    "tool": (0, 3),
    "ene": (12, 20),
}
# 機能ロールの下限（カテゴリ枚数と別の制約）。凍結nnはテンポ/展開を評価できない（board-blind・
# §24）ため、探索が「ドロー過剰・エネ加速ゼロ」の遅いデッキへ収束する（実測 §29）。加速枠を最低
# 数だけ**構成に注入**して強制する（value への盤面補正注入と同じ思想＝学べない知識は注入する）。
MIN_ACCEL = 2
_ACCEL_BIT = 1 << EFFECT_CATEGORIES.index("energy_accel")


def card_category(meta: CardMeta, cid: int) -> str:
    """cardId → 構成カテゴリ（poke/item/sup/stad/tool/ene/other）."""
    t = meta.card_type.get(cid)
    if t == CardType.POKEMON:
        return "poke"
    if t == CardType.ITEM:
        return "item"
    if t == CardType.SUPPORTER:
        return "sup"
    if t == CardType.STADIUM:
        return "stad"
    if t == CardType.TOOL:
        return "tool"
    if t in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        return "ene"
    return "other"


def repair_composition(
    deck: list[int],
    meta: CardMeta,
    staple_freq: Counter,
    bounds: dict[str, tuple[int, int]] | None = None,
    min_accel: int = MIN_ACCEL,
) -> list[int]:
    """デッキをカテゴリ枚数の許容帯＋機能ロール下限へ射影する（帯内なら無変更）.

    - 超過カテゴリ: 同一 cardId の多いものから削る（種類は温存）。
    - 不足カテゴリ: ポケモンは自前たねの増量優先（タイプ整合維持）→実メタ頻度上位のたね、
      トレーナー/エネは staple_freq（実メタ採用頻度）上位から。4枚制限・aceSpec 1枚を遵守。
    - **機能ロール**: エネ加速（展開）を最低 min_accel 枚。不足なら実メタ頻出の加速トレーナーを
      同カテゴリの冗長札と入れ替える（カテゴリ枚数は不変＝帯を崩さない・§29）。
    - 射影後にエンジン受理（is_legal）を確認し、失敗時は**元のデッキを返す**（安全退化）。
    """
    bounds = bounds or COMP_BOUNDS
    counts = Counter(deck)

    def cat_count(cat: str) -> int:
        return sum(n for c, n in counts.items() if card_category(meta, c) == cat)

    def n_ace() -> int:  # トレーナーの is_special ≒ aceSpec（デッキ1枚制限）
        return sum(
            n
            for c, n in counts.items()
            if meta.is_special.get(c, False) and card_category(meta, c) != "poke"
        )

    def addable(cid: int) -> bool:
        cat = card_category(meta, cid)
        if cat == "ene" and cid in meta.basic_energy_id.values():
            return True  # 基本エネは枚数無制限
        if counts.get(cid, 0) >= 4:
            return False
        if cat != "poke" and meta.is_special.get(cid, False) and n_ace() >= 1:
            return False
        return True

    changed = False
    # 1. 超過を上限まで削る
    for cat, (_, hi) in bounds.items():
        while cat_count(cat) > hi:
            cands = [(n, c) for c, n in counts.items() if card_category(meta, c) == cat]
            _, drop = max(cands)
            counts[drop] -= 1
            if counts[drop] == 0:
                del counts[drop]
            changed = True

    # 2. 不足を下限まで補充（60 枚の範囲で）
    def fill(cat: str, lo: int, pool: list[int]) -> None:
        nonlocal changed
        i = 0
        while cat_count(cat) < lo and sum(counts.values()) < DECK_SIZE:
            while i < len(pool) and not addable(pool[i]):
                i += 1
            if i >= len(pool):
                break
            counts[pool[i]] += 1
            changed = True

    own_basics = sorted(
        (
            c
            for c in counts
            if card_category(meta, c) == "poke"
            and meta.card_type.get(c) == CardType.POKEMON
            and meta.is_basic.get(c, False)
        ),
        key=lambda c: -staple_freq.get(c, 0),
    )
    staple_by_cat = {
        cat: [c for c, _ in staple_freq.most_common() if card_category(meta, c) == cat]
        for cat in bounds
    }
    fill(
        "poke",
        bounds["poke"][0],
        own_basics * 4
        + [c for c in staple_by_cat["poke"] if meta.is_basic.get(c, False)],
    )
    for cat in ("item", "sup", "tool", "stad", "ene"):
        fill(cat, bounds[cat][0], staple_by_cat[cat])

    # 2.5 機能ロール: エネ加速（展開）を最低 min_accel 枚。実メタ頻出の加速トレーナーを、
    #     同カテゴリの非加速・最も冗長なカードと1枚ずつ入れ替える（カテゴリ枚数は不変）。
    def accel_count() -> int:
        return sum(
            n for c, n in counts.items() if meta.ability_effect.get(c, 0) & _ACCEL_BIT
        )

    accel_pool = [
        c
        for c, _ in staple_freq.most_common()
        if (meta.ability_effect.get(c, 0) & _ACCEL_BIT)
        and card_category(meta, c) in ("item", "sup")
    ]
    ai = 0
    while accel_count() < min_accel and ai < len(accel_pool):
        a = accel_pool[ai]
        cat_a = card_category(meta, a)
        if not addable(a):
            ai += 1
            continue
        # 同カテゴリの非加速・最多枚数（冗長）カードを退避
        same = [
            (n, c)
            for c, n in counts.items()
            if card_category(meta, c) == cat_a
            and not (meta.ability_effect.get(c, 0) & _ACCEL_BIT)
        ]
        if not same:
            ai += 1
            continue
        _, victim = max(same)
        counts[victim] -= 1
        if counts[victim] == 0:
            del counts[victim]
        counts[a] += 1
        changed = True

    # 3. 60 枚に満たなければグッズ/サポ頻度上位で充填
    filler = [
        c
        for c, _ in staple_freq.most_common()
        if card_category(meta, c) in ("item", "sup")
    ]
    i = 0
    while sum(counts.values()) < DECK_SIZE and i < len(filler):
        if addable(filler[i]):
            counts[filler[i]] += 1
            changed = True
        else:
            i += 1

    if not changed:
        return deck
    repaired = [c for c, n in counts.items() for _ in range(n)]
    if len(repaired) != DECK_SIZE or not is_legal(repaired, repaired):
        return deck  # 安全退化: 射影に失敗したら元のまま（合法性を最優先）
    return repaired
