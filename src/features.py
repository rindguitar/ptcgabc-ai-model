"""観測の特徴量化（NN 用エンコーダ・Phase 3 の土台）.

AlphaZero 型の value/policy ネットへ渡すため、観測（自分視点）を**固定長の numpy ベクトル**に、
合法手（select.option）を **(手数 × 固定長) の行列**に変換する。torch 非依存（ホストでテスト可）。

v1 はカード ID の identity ではなく**数値メタ**（HP・エネルギー種別数・状態異常等）で符号化し、
任意のデッキに汎化する固定長表現にする。カード identity を使うより情報は粗いが、まず動く土台を作る。
カード名等の Pokémon Elements は保持しない。
"""

from __future__ import annotations

import numpy as np

from cards import EFFECT_CATEGORY_COUNT, CardMeta
from cg.api import CardType, Observation, PlayerState, Pokemon

# --- 各ブロックの次元（定数として一元管理） --------------------------------
N_ENERGY_TYPES = 12  # EnergyType 0..11
N_OPTION_TYPES = 17  # OptionType 0..16
MAX_BENCH = 5

# ポケモン1体の特徴量: present, hp割合, hp絶対, エネ総数, エネ種別(12), 道具数, 進化前数
POKEMON_FEAT = 1 + 1 + 1 + 1 + N_ENERGY_TYPES + 1 + 1
# プレイヤー1人: active + 状態異常5 + ベンチ5体 + カウント5
PLAYER_FEAT = POKEMON_FEAT + 5 + MAX_BENCH * POKEMON_FEAT + 5
# グローバル: turn, is_first, 4フラグ, stadium有無
GLOBAL_FEAT = 1 + 1 + 4 + 1
# 自分の手札資源（特性/トレーナー学習用・末尾に追加）。トレーナーは挙動が異なるため種別ごとに分ける:
# ポケモン / グッズ / サポート / スタジアム / どうぐ / エネ / 特性持ちポケモン の 7 種。
HAND_FEAT = 7
# アクティブのメタ（自分/相手の2体・末尾に追加）: is_special + にげるコスト + 特性効果カテゴリ。
# 効果テキストを数値化した ability_effect ビットを net が読めるようにする（NN v2）。
ACTIVE_META_FEAT = 2 + EFFECT_CATEGORY_COUNT
# 観測ベクトル長。HAND_FEAT / ACTIVE_META は**末尾**に足す（ウォームスタートで旧重みを先頭へ流用）。
OBS_FEAT_LEN = GLOBAL_FEAT + 2 * PLAYER_FEAT + HAND_FEAT + 2 * ACTIVE_META_FEAT
# 行動の対象カードのメタ（末尾に追加）: 特性有無 / 威力効率 / HP
ACTION_CARD_FEAT = 3
# 行動の効果・盤面相互作用（末尾に追加・NN v2）: ワザ効果カテゴリ(ビット) ＋
# 相手アクティブHP / damage比 / KO可能フラグ / 弱点一致フラグ の 4。手の良し悪しを net が読める。
ACTION_EFFECT_FEAT = EFFECT_CATEGORY_COUNT + 4
# 行動特徴量長。ACTION_CARD_FEAT / ACTION_EFFECT_FEAT は**末尾**に足す（同上・ウォームスタート用）。
ACTION_FEAT_LEN = N_OPTION_TYPES + 2 + ACTION_CARD_FEAT + ACTION_EFFECT_FEAT


def _bits(mask: int, n: int) -> list[float]:
    """ビットマスクを n 個の 0/1 float に展開する（効果カテゴリの符号化）."""
    return [float((mask >> i) & 1) for i in range(n)]


def _encode_active_meta(pk: Pokemon | None, meta: CardMeta) -> list[float]:
    """アクティブ1体の効果メタ（is_special / にげるコスト / 特性効果カテゴリ）を符号化."""
    if pk is None:
        return [0.0] * ACTIVE_META_FEAT
    cid = pk.id or 0
    return [
        float(meta.is_special.get(cid, False)),
        min(meta.retreat_cost.get(cid, 0), 4) / 4,
    ] + _bits(meta.ability_effect.get(cid, 0), EFFECT_CATEGORY_COUNT)


def observation_feature_size() -> int:
    """encode_observation が返すベクトル長."""
    return OBS_FEAT_LEN


def action_feature_size() -> int:
    """encode_actions の1手あたりの特徴量長."""
    return ACTION_FEAT_LEN


def _encode_pokemon(pk: Pokemon | None) -> list[float]:
    """ポケモン1体を数値メタで符号化（不在は全 0）."""
    if pk is None:
        return [0.0] * POKEMON_FEAT
    max_hp = pk.maxHp or 1
    energy_by_type = [0] * N_ENERGY_TYPES
    for e in pk.energies:
        if 0 <= e < N_ENERGY_TYPES:
            energy_by_type[e] += 1
    feats = [
        1.0,  # present
        pk.hp / max_hp,  # hp 割合
        min(pk.hp, 400) / 400,  # hp 絶対（正規化）
        min(sum(energy_by_type), 10) / 10,  # エネ総数
    ]
    feats += [c / 6.0 for c in energy_by_type]  # エネ種別ごとの枚数
    feats.append(min(len(pk.tools), 3) / 3)
    feats.append(min(len(pk.preEvolution), 3) / 3)
    return feats


def _encode_player(ps: PlayerState) -> list[float]:
    """プレイヤー1人の盤面を符号化."""
    active = ps.active[0] if ps.active and len(ps.active) > 0 else None
    feats = _encode_pokemon(active)
    feats += [
        float(ps.poisoned),
        float(ps.burned),
        float(ps.asleep),
        float(ps.paralyzed),
        float(ps.confused),
    ]
    bench = ps.bench or []
    for i in range(MAX_BENCH):
        feats += _encode_pokemon(bench[i] if i < len(bench) else None)
    feats += [
        min(ps.handCount or 0, 20) / 20,
        min(ps.deckCount or 0, 60) / 60,
        min(len(ps.discard), 60) / 60,
        len(ps.prize) / 6.0,
        min(ps.benchMax or 0, 8) / 8,
    ]
    return feats


def _encode_hand(ps: PlayerState, meta: CardMeta) -> list[float]:
    """自分の手札の種別構成を符号化（末尾追加・特性/トレーナー判断の手掛かり）.

    手札カードは id（cardId）を持つ。トレーナーは挙動が異なるため種別ごとに分けて数える
    （サポートは1ターン1枚・グッズは無制限 等）。手札非公開（相手）や None のときは全 0。
    意味（効果文）は読まず、種別とメタ数値のみ使う。
    """
    n_poke = n_item = n_supporter = n_stadium = n_tool = n_energy = n_ability = 0
    for card in ps.hand or []:
        ct = meta.card_type.get(card.id)
        if ct == CardType.POKEMON:
            n_poke += 1
            if meta.has_ability.get(card.id):
                n_ability += 1
        elif ct == CardType.ITEM:
            n_item += 1
        elif ct == CardType.SUPPORTER:
            n_supporter += 1
        elif ct == CardType.STADIUM:
            n_stadium += 1
        elif ct == CardType.TOOL:
            n_tool += 1
        elif ct in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            n_energy += 1
    return [
        min(n_poke, 10) / 10,
        min(n_item, 6) / 6,
        min(n_supporter, 4) / 4,
        min(n_stadium, 3) / 3,
        min(n_tool, 4) / 4,
        min(n_energy, 10) / 10,
        min(n_ability, 6) / 6,
    ]


def encode_observation(obs: Observation, meta: CardMeta) -> np.ndarray:
    """観測（自分視点）を固定長 float32 ベクトルに符号化する.

    select/current が None（デッキ選択時など）は全 0 を返す。
    """
    st = obs.current
    if st is None:
        return np.zeros(OBS_FEAT_LEN, dtype=np.float32)

    yi = st.yourIndex
    is_first = 1.0 if st.firstPlayer == yi else 0.0
    glob = [
        min(st.turn, 50) / 50,
        is_first,
        float(st.supporterPlayed),
        float(st.stadiumPlayed),
        float(st.energyAttached),
        float(st.retreated),
        float(len(st.stadium) > 0),
    ]
    feats = glob + _encode_player(st.players[yi]) + _encode_player(st.players[1 - yi])
    feats += _encode_hand(st.players[yi], meta)  # 末尾追加（ウォームスタート対応）
    # アクティブの効果メタ（自分/相手）を末尾追加（NN v2・特性効果カテゴリを net に渡す）
    my_act = st.players[yi].active[0] if st.players[yi].active else None
    opp_act = st.players[1 - yi].active[0] if st.players[1 - yi].active else None
    feats += _encode_active_meta(my_act, meta) + _encode_active_meta(opp_act, meta)
    return np.asarray(feats, dtype=np.float32)


def encode_actions(obs: Observation, meta: CardMeta) -> np.ndarray:
    """合法手（select.option）を (手数 × ACTION_FEAT_LEN) の行列に符号化する.

    各手: OptionType の one-hot ＋ ワザのダメージ（正規化）＋ ターゲット有無
    ＋ 対象カードのメタ（特性有無 / 威力効率 / HP・末尾追加）。
    select が None のときは (0, ACTION_FEAT_LEN) を返す。
    """
    sel = obs.select
    if sel is None:
        return np.zeros((0, ACTION_FEAT_LEN), dtype=np.float32)

    # 盤面相互作用（KO/弱点）用に自分・相手アクティブを取得（NN v2）
    st = obs.current
    my_act = opp_act = None
    if st is not None:
        yi = st.yourIndex
        my_act = st.players[yi].active[0] if st.players[yi].active else None
        opp_act = st.players[1 - yi].active[0] if st.players[1 - yi].active else None
    my_type = meta.pokemon_type.get(my_act.id, -1) if my_act else -1
    opp_hp = opp_act.hp if opp_act else 0
    opp_weak = meta.weakness.get(opp_act.id, -1) if opp_act else -1

    rows = []
    for o in sel.option:
        row = [0.0] * N_OPTION_TYPES
        if o.type is not None and 0 <= o.type < N_OPTION_TYPES:
            row[o.type] = 1.0
        damage = meta.attack_damage(o.attackId) if o.attackId is not None else 0
        row.append(min(damage, 300) / 300)
        row.append(1.0 if (o.inPlayArea is not None or o.cardId) else 0.0)
        # 対象カードのメタ（cardId>0 のとき。特性発動/トレーナーPLAY/進化等の質を表す）
        cid = o.cardId or 0
        row.append(1.0 if meta.has_ability.get(cid) else 0.0)
        row.append(min(meta.best_efficiency.get(cid, 0.0), 60) / 60)
        row.append(min(meta.hp.get(cid, 0), 400) / 400)
        # ワザ効果カテゴリ（末尾追加・NN v2）
        amask = meta.attack_effect.get(o.attackId, 0) if o.attackId is not None else 0
        row += _bits(amask, EFFECT_CATEGORY_COUNT)
        # 盤面相互作用: 相手HP / damage比 / KO可能 / 弱点一致
        row.append(min(opp_hp, 400) / 400)
        row.append(min(damage / opp_hp, 2.0) / 2.0 if opp_hp > 0 else 0.0)
        row.append(1.0 if (opp_hp > 0 and damage >= opp_hp) else 0.0)
        row.append(1.0 if (my_type >= 0 and opp_weak == my_type) else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)
