"""カードメタデータ（cabt Engine から取得）.

エンジンの `AllCard` / `AllAttack` から、対戦判断に必要な**数値メタ**を取り出す。
効果テキスト(`text`)は**ローカルで読み、汎用ゲーム機構のキーワードで数値カテゴリ
（draw/search/heal 等のビットフラグ）に変換**する。生のテキスト・カード名（＝Pokémon
Elements）は**保持・出力・コミットしない**（規約: 使用は許諾だが公開・再配布は禁止）。
派生した数値フラグは参加者の生成物（Pokémon Elements ではない）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cg.api import CardType
from cg.sim import lib

# 効果カテゴリ（汎用 PTCG 機構）。順序がビット位置。効果テキストを下のキーワードで分類して
# 数値ビットマスクにする。キーワードは一般的なゲーム用語のみ（カード名・固有名は含めない）。
EFFECT_CATEGORIES: tuple[str, ...] = (
    "draw",
    "search",
    "heal",
    "energy_accel",
    "energy_disrupt",
    "damage_counter",
    "status",
    "switch",
    "hand_disrupt",
    "prevent",
)
EFFECT_CATEGORY_COUNT = len(EFFECT_CATEGORIES)
_EFFECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "draw": ("draw",),
    "search": ("search your deck", "look at the top"),
    "heal": ("heal",),
    "energy_accel": (
        "attach a",
        "attach an",
        "attach 1",
        "attach energy",
        "attach the",
    ),
    "energy_disrupt": ("discard an energy", "discard energy", "discards an energy"),
    "damage_counter": ("damage counter",),
    "status": ("asleep", "poisoned", "burned", "paralyzed", "confused"),
    "switch": ("switch",),
    "hand_disrupt": ("opponent's hand", "shuffle their hand", "discards their hand"),
    "prevent": ("prevent", "reduced by", "does nothing"),
}


def _effect_bitmask(text: str | None) -> int:
    """効果テキストを効果カテゴリのビットマスクに変換する（数値のみ・テキストは保持しない）."""
    if not text:
        return 0
    t = text.lower()
    mask = 0
    for i, cat in enumerate(EFFECT_CATEGORIES):
        if any(kw in t for kw in _EFFECT_KEYWORDS[cat]):
            mask |= 1 << i
    return mask


@dataclass
class CardMeta:
    """ヒューリスティックが参照する最小限のカード数値メタ."""

    damage: dict[int, int]  # attackId -> ダメージ
    card_type: dict[int, int]  # cardId -> CardType 値
    is_basic: dict[int, bool]  # cardId -> たねポケモンか
    hp: dict[int, int]  # cardId -> HP（無ければ 0）
    energy_type: dict[int, int]  # cardId -> energyType（色コード 0..10）
    best_damage: dict[int, int]  # cardId -> その札の最大ワザダメージ（構成用の質代理）
    best_efficiency: dict[int, float]  # cardId -> 最良の威力効率（ダメージ/エネコスト）
    has_ability: dict[int, bool]  # cardId -> 特性を持つか（有無のみ）
    basic_energy_id: dict[int, int]  # energyType -> 基本エネの cardId（色→カード）
    # --- 効果カテゴリ（効果テキストを数値フラグ化・生テキストは保持しない）---
    attack_effect: dict[int, int]  # attackId -> 効果カテゴリのビットマスク
    ability_effect: dict[
        int, int
    ]  # cardId -> 特性効果カテゴリのビットマスク（全特性のOR）
    # --- 構造メタ（数値・KO/相性計算や質の代理に使う）---
    pokemon_type: dict[int, int]  # cardId -> ポケモンのタイプ（色コード）
    weakness: dict[int, int]  # cardId -> 弱点タイプ（無し=-1）
    resistance: dict[int, int]  # cardId -> 抵抗タイプ（無し=-1）
    retreat_cost: dict[int, int]  # cardId -> にげるコスト
    is_special: dict[
        int, bool
    ]  # cardId -> ex/megaEx/tera/aceSpec のいずれか（高性能札の代理）
    # KO 時に相手が取るサイド枚数（megaEx=3 / ex=2 / それ以外=1）。ex/megaEx フラグ由来なので
    # tera が ex でない場合も正しく 1 になる（prize と tera=ベンチ無敵を独立に扱う）。
    prize_value: dict[int, int]  # cardId -> 1/2/3
    is_tera: dict[
        int, bool
    ]  # cardId -> tera（ベンチにいる間はワザのダメージを受けない）
    # 山札リサイクル札（トラッシュ→山に戻す）。EFFECT_CATEGORIES には足さない
    # （特徴量次元が変わり既存 net が全滅するため）＝独立フィールド（§39）
    is_deck_recycle: dict[int, bool]

    def is_basic_pokemon(self, card_id: int) -> bool:
        """指定 cardId がたねポケモンか."""
        return self.card_type.get(card_id) == CardType.POKEMON and self.is_basic.get(
            card_id, False
        )

    def attack_damage(self, attack_id: int) -> int:
        """指定 attackId のダメージ（無ければ 0）."""
        return self.damage.get(attack_id, 0)


def load_card_meta() -> CardMeta:
    """エンジンの全カード/全ワザ情報から数値メタを構築する."""
    cards = json.loads(lib.AllCard().decode())
    attacks = json.loads(lib.AllAttack().decode())

    damage = {a["attackId"]: int(a.get("damage") or 0) for a in attacks}
    # ワザのエネルギーコスト数（威力効率＝ダメージ/コストの算出に使う）
    cost = {a["attackId"]: max(1, len(a.get("energies") or [])) for a in attacks}
    # ワザ効果テキストを効果カテゴリのビットマスクに（生テキストは保持しない）
    attack_effect = {a["attackId"]: _effect_bitmask(a.get("text")) for a in attacks}

    card_type: dict[int, int] = {}
    is_basic: dict[int, bool] = {}
    hp: dict[int, int] = {}
    energy_type: dict[int, int] = {}
    best_damage: dict[int, int] = {}
    best_efficiency: dict[int, float] = {}
    has_ability: dict[int, bool] = {}
    basic_energy_id: dict[int, int] = {}
    ability_effect: dict[int, int] = {}
    pokemon_type: dict[int, int] = {}
    weakness: dict[int, int] = {}
    resistance: dict[int, int] = {}
    retreat_cost: dict[int, int] = {}
    is_special: dict[int, bool] = {}
    prize_value: dict[int, int] = {}
    is_tera: dict[int, bool] = {}
    is_deck_recycle: dict[int, bool] = {}
    for c in cards:
        cid = c["cardId"]
        ctype = c.get("cardType")
        card_type[cid] = ctype
        is_basic[cid] = bool(c.get("basic"))
        hp[cid] = int(c.get("hp") or 0)
        energy_type[cid] = c.get("energyType")
        aids = c.get("attacks") or []
        # 札の最大ワザダメージ（構成生成で「強い攻撃役」を選ぶ際の質の代理指標）
        best_damage[cid] = max((damage.get(aid, 0) for aid in aids), default=0)
        # 威力効率の最良値（少エネで大ダメージ＝回しやすい攻撃役の代理）
        best_efficiency[cid] = max(
            (damage.get(aid, 0) / cost.get(aid, 1) for aid in aids), default=0.0
        )
        # 特性(skills)の有無＋効果カテゴリ（全特性の text を OR・生テキストは保持しない）
        skills = c.get("skills") or []
        has_ability[cid] = bool(skills)
        amask = 0
        texts_l = []
        for sk in skills:
            amask |= _effect_bitmask(sk.get("text"))
            if sk.get("text"):
                texts_l.append(str(sk["text"]).lower())
        ability_effect[cid] = amask
        # 山札リサイクル（トラッシュ→山に戻す）判定。EFFECT_CATEGORIES には**足さない**
        # （足すと特徴量次元が変わり既存 net が全滅する）ため独立フィールドで持つ（§39）。
        is_deck_recycle[cid] = any(
            "discard pile into your deck" in t for t in texts_l
        )
        # 構造メタ（KO/相性計算用・タイプは色コード int、無しは -1）
        pokemon_type[cid] = (
            c.get("pokemonType") if c.get("pokemonType") is not None else -1
        )
        weakness[cid] = c.get("weakness") if c.get("weakness") is not None else -1
        resistance[cid] = c.get("resistance") if c.get("resistance") is not None else -1
        retreat_cost[cid] = int(c.get("retreatCost") or 0)
        is_special[cid] = bool(
            c.get("ex") or c.get("megaEx") or c.get("tera") or c.get("aceSpec")
        )
        # KO 時に取られるサイド枚数（megaEx=3 / ex=2 / それ以外=1）。ex/megaEx フラグ由来。
        prize_value[cid] = 3 if c.get("megaEx") else (2 if c.get("ex") else 1)
        is_tera[cid] = bool(c.get("tera"))  # ベンチにいる間はワザのダメージを受けない
        # 色 -> 基本エネの cardId（基本エネは色ごとに1枚）
        if ctype == CardType.BASIC_ENERGY:
            basic_energy_id.setdefault(c.get("energyType"), cid)

    return CardMeta(
        damage=damage,
        card_type=card_type,
        is_basic=is_basic,
        hp=hp,
        energy_type=energy_type,
        best_damage=best_damage,
        best_efficiency=best_efficiency,
        has_ability=has_ability,
        basic_energy_id=basic_energy_id,
        attack_effect=attack_effect,
        ability_effect=ability_effect,
        pokemon_type=pokemon_type,
        weakness=weakness,
        resistance=resistance,
        retreat_cost=retreat_cost,
        is_special=is_special,
        prize_value=prize_value,
        is_tera=is_tera,
        is_deck_recycle=is_deck_recycle,
    )
