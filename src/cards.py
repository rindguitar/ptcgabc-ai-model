"""カードメタデータ（cabt Engine から取得）.

エンジンの `AllCard` / `AllAttack` から、ヒューリスティック判断に必要な
最小限の**数値メタ**（ダメージ・カード種別・たねフラグ・HP）だけを取り出す。
カード名・効果文などの Pokémon Elements は保持しない（規約遵守）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cg.api import CardType
from cg.sim import lib


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
    has_ability: dict[int, bool]  # cardId -> 特性を持つか（有無のみ・意味は読まない）
    basic_energy_id: dict[int, int]  # energyType -> 基本エネの cardId（色→カード）

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

    card_type: dict[int, int] = {}
    is_basic: dict[int, bool] = {}
    hp: dict[int, int] = {}
    energy_type: dict[int, int] = {}
    best_damage: dict[int, int] = {}
    best_efficiency: dict[int, float] = {}
    has_ability: dict[int, bool] = {}
    basic_energy_id: dict[int, int] = {}
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
        # 特性(skills)の有無のみ保持（テキスト＝Pokémon Element の意味は読まない）
        has_ability[cid] = bool(c.get("skills"))
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
    )
