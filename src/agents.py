"""Phase 1 baseline エージェント群.

各エージェントは `agent(obs, rng) -> list[int]` のシグネチャを持ち、
`obs.select.option` から選んだインデックスのリスト（minCount〜maxCount 個・重複なし）を返す。
デッキ提出（select is None）はハーネスが battle_start でデッキを直接渡すため、ここでは扱わない。
"""

from __future__ import annotations

import random
from typing import Callable

from cards import EFFECT_CATEGORIES, CardMeta
from cg.api import (
    AreaType,
    CardType,
    Observation,
    OptionType,
    SelectContext,
    SelectType,
)

# エージェントの型エイリアス。
Agent = Callable[[Observation, random.Random], list[int]]

# draw / search 効果のビット（トレーナー活用の判定に使う）
_DRAWSEARCH_MASK = (1 << EFFECT_CATEGORIES.index("draw")) | (
    1 << EFFECT_CATEGORIES.index("search")
)
# 展開＝エネ加速のビット。draw/search（たね確保）と合わせて「使うと得な発展手」を表す。
# 加速札をデッキに入れても操縦が打たなければ腐る（design §29）ので draw/search と同格で使う。
_ACCEL_MASK = 1 << EFFECT_CATEGORIES.index("energy_accel")
_DEVELOP_MASK = _DRAWSEARCH_MASK | _ACCEL_MASK


def random_agent(obs: Observation, rng: random.Random) -> list[int]:
    """合法手からランダムに選ぶ baseline."""
    sel = obs.select
    count = rng.randint(sel.minCount, sel.maxCount)
    if count == 0:
        return []
    return rng.sample(range(len(sel.option)), count)


def make_heuristic_agent(meta: CardMeta, use_trainers: bool = True) -> Agent:
    """貪欲ヒューリスティックエージェントを生成する.

    方針（MAIN 選択）: **draw/search トレーナーを使う** → 進化 → エネ付与（アクティブ優先）
    → たねをベンチ展開 → 最大ダメージで攻撃 → ターン終了。攻撃はターンを終えるため、
    整地系を先に消化する。ドローは同ターンの選択肢を増やすので最優先。

    use_trainers=False で旧挙動（トレーナー完全不使用）に戻せる（A/B 検証用）。
    旧挙動は「グッズの価値を系統的に過小評価→league がエネ過多デッキへ収束」の根因だった。
    draw/search 以外のトレーナー（妨害・回復等）は引き続き使わない（誤爆リスク回避）。
    """

    def heuristic_agent(obs: Observation, rng: random.Random) -> list[int]:
        sel = obs.select
        if sel.type == SelectType.MAIN:
            return [_choose_main(obs, meta, rng, use_trainers)]
        if sel.type == SelectType.ATTACK:
            return [_argmax_damage(sel.option, meta)]
        return _generic_select(obs, meta)

    return heuristic_agent


def _choose_main(
    obs: Observation, meta: CardMeta, rng: random.Random, use_trainers: bool = True
) -> int:
    """MAIN 選択での貪欲な行動選択."""
    opts = obs.select.option
    st = obs.current
    by_type: dict[int, list[int]] = {}
    for i, o in enumerate(opts):
        by_type.setdefault(o.type, []).append(i)

    # 打つ順番（PTCG のセオリー）: ①掘る＆エネ加速 → ②進化 → ③どうぐ → ④エネ付与 →
    # ⑤ベンチ展開 → ⑥スタジアム → ⑦攻撃。エンジンは MAIN を毎回呼ぶので、優先順が
    # そのまま「1ターン内の手順」になる。加速/どうぐ/スタジアムを使わず腐らせる問題への対処。
    if use_trainers:
        # 0. たね確保＋展開: draw/search/加速 のトレーナー・特性（掘る＆エネを伸ばす）
        dev = [
            i
            for i in by_type.get(OptionType.PLAY, [])
            if _is_develop_play(opts[i], obs, meta)
        ]
        dev += [
            i
            for i in by_type.get(OptionType.ABILITY, [])
            if _is_develop_ability(opts[i], meta)
        ]
        if dev:
            return rng.choice(dev)

    # 1. 進化（基本的に得）
    if OptionType.EVOLVE in by_type:
        return rng.choice(by_type[OptionType.EVOLVE])

    # 2. どうぐ装着（HP/火力補助＝腐らせない。対象はエンジンが選ばせる）
    if use_trainers and OptionType.TOOL_CARD in by_type:
        return rng.choice(by_type[OptionType.TOOL_CARD])

    # 3. エネルギー付与（アクティブを優先して殴れる状態に近づける）
    if OptionType.ATTACH in by_type:
        attach = by_type[OptionType.ATTACH]
        to_active = [i for i in attach if opts[i].inPlayArea == AreaType.ACTIVE]
        return rng.choice(to_active or attach)

    # 4. 手札のたねポケモンをベンチ展開
    play_basic = [
        i
        for i in by_type.get(OptionType.PLAY, [])
        if _is_basic_pokemon_play(opts[i], obs, meta)
    ]
    if play_basic:
        return rng.choice(play_basic)

    # 5. スタジアム（**場に無い時だけ**＝自分の有益スタジアムを毎ターン上書きしない）
    if use_trainers and st is not None and not st.stadium:
        stad = [
            i
            for i in by_type.get(OptionType.PLAY, [])
            if _is_stadium_play(opts[i], obs, meta)
        ]
        if stad:
            return rng.choice(stad)

    # 6. 攻撃（最大ダメージ）
    if OptionType.ATTACK in by_type:
        return _argmax_damage_indices(by_type[OptionType.ATTACK], opts, meta)

    # 7. ターン終了
    if OptionType.END in by_type:
        return by_type[OptionType.END][0]

    # フォールバック: 先頭
    return 0


def _is_basic_pokemon_play(opt, obs: Observation, meta: CardMeta) -> bool:
    """PLAY オプションが手札のたねポケモンを場に出すものか."""
    st = obs.current
    if st is None or opt.index is None:
        return False
    me = st.players[st.yourIndex]
    if me.hand is None or opt.index >= len(me.hand):
        return False
    return meta.is_basic_pokemon(me.hand[opt.index].id)


def _hand_card_id(opt, obs: Observation) -> int | None:
    """PLAY 系オプションが手札から出すカードの id（範囲外/不明なら None）."""
    st = obs.current
    if st is None or opt.index is None:
        return None
    me = st.players[st.yourIndex]
    if me.hand is None or opt.index >= len(me.hand):
        return None
    return me.hand[opt.index].id


def _is_drawsearch_trainer_play(opt, obs: Observation, meta: CardMeta) -> bool:
    """PLAY オプションが手札の draw/search 効果トレーナー（グッズ/サポート）を使うものか."""
    cid = _hand_card_id(opt, obs)
    if cid is None or meta.card_type.get(cid) not in (
        CardType.ITEM,
        CardType.SUPPORTER,
    ):
        return False
    return bool(meta.ability_effect.get(cid, 0) & _DRAWSEARCH_MASK)


def _is_develop_play(opt, obs: Observation, meta: CardMeta) -> bool:
    """PLAY が draw/search/加速（発展手）のグッズ・サポートを使うものか."""
    cid = _hand_card_id(opt, obs)
    if cid is None or meta.card_type.get(cid) not in (
        CardType.ITEM,
        CardType.SUPPORTER,
    ):
        return False
    return bool(meta.ability_effect.get(cid, 0) & _DEVELOP_MASK)


def _is_develop_ability(opt, meta: CardMeta) -> bool:
    """ABILITY 起動が draw/search/加速 の特性か（source は opt.cardId）."""
    cid = opt.cardId
    return cid is not None and bool(meta.ability_effect.get(cid, 0) & _DEVELOP_MASK)


def _is_stadium_play(opt, obs: Observation, meta: CardMeta) -> bool:
    """PLAY が手札のスタジアムを出すものか."""
    cid = _hand_card_id(opt, obs)
    return cid is not None and meta.card_type.get(cid) == CardType.STADIUM


def _argmax_damage(opts, meta: CardMeta) -> int:
    """ATTACK オプション列の中で最大ダメージのインデックス."""
    return max(range(len(opts)), key=lambda i: meta.attack_damage(opts[i].attackId))


def _argmax_damage_indices(indices: list[int], opts, meta: CardMeta) -> int:
    """指定インデックス集合の中で最大ダメージのものを返す."""
    return max(indices, key=lambda i: meta.attack_damage(opts[i].attackId))


def _generic_select(obs: Observation, meta: CardMeta) -> list[int]:
    """MAIN/ATTACK 以外のサブ選択を無難に処理する.

    - 数値選択（ドロー枚数など）は最大化。
    - セットアップのベンチ展開は可能な限り並べる。
    - **山札からの選択（サーチ）は「たね優先→エネ優先」で maxCount まで取る**。
      旧実装は最小数（多くは 0 枚）でサーチを無駄撃ちしており、トレーナー活用の妨げだった。
      たね優先はベンチ切れ（実戦敗因の6〜7割）への直接の対策。
    - それ以外は必要最小数を先頭から選ぶ（任意選択は見送る）。
    """
    sel = obs.select
    n = len(sel.option)

    if sel.type == SelectType.COUNT:
        return [max(range(n), key=lambda i: sel.option[i].number or 0)]

    if sel.context == SelectContext.SETUP_BENCH_POKEMON:
        return list(range(min(sel.maxCount, n)))

    if sel.deck is not None and sel.maxCount > 0:
        # 山札からのサーチ: たね > エネルギー > その他 の順に価値付けして取れるだけ取る
        def rank(i: int) -> tuple[int, int]:
            cid = sel.option[i].cardId or 0
            if meta.is_basic_pokemon(cid):
                return (0, i)
            if meta.card_type.get(cid) in (
                CardType.BASIC_ENERGY,
                CardType.SPECIAL_ENERGY,
            ):
                return (1, i)
            return (2, i)

        take = max(sel.minCount, min(sel.maxCount, n))
        return sorted(sorted(range(n), key=rank)[:take])

    count = sel.minCount
    return list(range(count))
