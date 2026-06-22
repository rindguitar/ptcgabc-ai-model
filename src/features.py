"""観測の特徴量化（NN 用エンコーダ・Phase 3 の土台）.

AlphaZero 型の value/policy ネットへ渡すため、観測（自分視点）を**固定長の numpy ベクトル**に、
合法手（select.option）を **(手数 × 固定長) の行列**に変換する。torch 非依存（ホストでテスト可）。

v1 はカード ID の identity ではなく**数値メタ**（HP・エネルギー種別数・状態異常等）で符号化し、
任意のデッキに汎化する固定長表現にする。カード identity を使うより情報は粗いが、まず動く土台を作る。
カード名等の Pokémon Elements は保持しない。
"""

from __future__ import annotations

import numpy as np

from cards import CardMeta
from cg.api import Observation, PlayerState, Pokemon

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
OBS_FEAT_LEN = GLOBAL_FEAT + 2 * PLAYER_FEAT
ACTION_FEAT_LEN = N_OPTION_TYPES + 2  # type one-hot + ダメージ + ターゲット有無


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
    return np.asarray(feats, dtype=np.float32)


def encode_actions(obs: Observation, meta: CardMeta) -> np.ndarray:
    """合法手（select.option）を (手数 × ACTION_FEAT_LEN) の行列に符号化する.

    各手: OptionType の one-hot ＋ ワザのダメージ（正規化）＋ ターゲットを持つか。
    select が None のときは (0, ACTION_FEAT_LEN) を返す。
    """
    sel = obs.select
    if sel is None:
        return np.zeros((0, ACTION_FEAT_LEN), dtype=np.float32)

    rows = []
    for o in sel.option:
        row = [0.0] * N_OPTION_TYPES
        if o.type is not None and 0 <= o.type < N_OPTION_TYPES:
            row[o.type] = 1.0
        damage = meta.attack_damage(o.attackId) if o.attackId is not None else 0
        row.append(min(damage, 300) / 300)
        row.append(1.0 if (o.inPlayArea is not None or o.cardId) else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)
