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
# 山札リサイクル（§39）を発動する残デッキ閾値。1位の実測（発動中央値=残13枚）より
# 少し早め＝掘り切る前に確実に戻せる余裕を持たせる。
_RECYCLE_AT = 15
# KO 脅威推定（§48）の定数: 弱点一致の打点倍率（ゲームルール）・
# エネ1不足（次の1付与で打てる＝育成中）の割引・ベンチ攻撃役（昇格が要る）の割引
_WEAKNESS_MULT = 2
_THREAT_NEXT_TURN = 0.5
_THREAT_BENCH = 0.5


def random_agent(obs: Observation, rng: random.Random) -> list[int]:
    """合法手からランダムに選ぶ baseline."""
    sel = obs.select
    count = rng.randint(sel.minCount, sel.maxCount)
    if count == 0:
        return []
    return rng.sample(range(len(sel.option)), count)


def make_heuristic_agent(
    meta: CardMeta,
    use_trainers: bool = True,
    fetch_priors: dict[int, float] | None = None,
    attach_priors: dict[int, float] | None = None,
    bench_first: bool = False,
    evacuate_prize: float | None = None,
    fix_switch_target: bool = False,
    setup_active_rule: bool = False,
) -> Agent:
    """貪欲ヒューリスティックエージェントを生成する.

    方針（MAIN 選択）: **draw/search トレーナーを使う** → 進化 → エネ付与（アクティブ優先）
    → たねをベンチ展開 → 最大ダメージで攻撃 → ターン終了。攻撃はターンを終えるため、
    整地系を先に消化する。ドローは同ターンの選択肢を増やすので最優先。

    use_trainers=False で旧挙動（トレーナー完全不使用）に戻せる（A/B 検証用）。
    旧挙動は「グッズの価値を系統的に過小評価→league がエネ過多デッキへ収束」の根因だった。
    draw/search 以外のトレーナー（妨害・回復等）は引き続き使わない（誤爆リスク回避）。

    fetch_priors: 山札サーチの取得優先度 {cardId: 教師の取得率}（§47・デッキ別）。
    未指定なら従来のカテゴリ順のみ＝挙動不変。

    attach_priors: エネ付与先の切替確率 {アクティブの装着エネ枚数: P(ベンチ付与)}
    （mine_attach_policy.py の帯実測）。上位帯はアーキタイプ不問でベンチへ 4〜5回/戦
    エネを回す（我々1.2回）のが最普遍の操縦差で、アクティブ KO＝全エネ喪失の
    ワンサイド負け（敗時サイド0〜1）の根因だった。未指定なら従来挙動（アクティブ優先）。

    bench_first: True でエネ付与より**たねのベンチ展開を先**にする（§72）。不利な相手
    （型 F0/F4）との戦いで自ベンチ数が @10決定 1.85 vs 2.62 と序盤に育っていなかった実測
    への対処。既定 False＝挙動不変。

    evacuate_prize: 指定すると「2サイド以上取られるアクティブ」を KO 脅威（§48）が
    この閾値以上のときに退避させる（入替札→にげるの順・§80）。実測では 3サイドの
    ドローエンジンがアクティブで 0.58回/戦 KO され献上サイドの 61% を占めていた。
    未指定なら退避しない＝挙動不変。

    fix_switch_target: True で `SWITCH` の相手指定バグを修正する（§79）。旧実装は
    引きずり出し（playerIndex が相手）の候補も自分のベンチとして解決していたため、
    自分のエネ数で相手を順位付け＝実質先頭取りになっていた（0.48回/戦）。
    修正後は相手の場から**取れるサイドが最大**の個体を選ぶ（上位帯基準・§80）。
    既定 False＝挙動不変。

    setup_active_rule: True で開幕アクティブ（SETUP_ACTIVE_POKEMON）を帯基準で選ぶ（§89）。
    旧実装は先頭取り（先頭率 1.000）で、40 局面中 19 が3サイド札の開幕だった。
    既定 False＝挙動不変。
    """

    def heuristic_agent(obs: Observation, rng: random.Random) -> list[int]:
        sel = obs.select
        if sel.type == SelectType.MAIN:
            return [
                _choose_main(
                    obs,
                    meta,
                    rng,
                    use_trainers,
                    attach_priors,
                    bench_first,
                    evacuate_prize,
                )
            ]
        if sel.type == SelectType.ATTACK:
            return [_argmax_damage(sel.option, meta)]
        return _generic_select(
            obs, meta, fetch_priors, fix_switch_target, setup_active_rule
        )

    return heuristic_agent


def _choose_main(
    obs: Observation,
    meta: CardMeta,
    rng: random.Random,
    use_trainers: bool = True,
    attach_priors: dict[int, float] | None = None,
    bench_first: bool = False,
    evacuate_prize: float | None = None,
) -> int:
    """MAIN 選択での貪欲な行動選択."""
    opts = obs.select.option
    st = obs.current
    by_type: dict[int, list[int]] = {}
    for i, o in enumerate(opts):
        by_type.setdefault(o.type, []).append(i)

    # 打つ順番（PTCG のセオリー）: ⓪山札リサイクル（残デッキ僅少時）→ ①掘る＆エネ加速 →
    # ②進化 → ③どうぐ → ④エネ付与 → ⑤ベンチ展開 → ⑥スタジアム → ⑦攻撃。
    # エンジンは MAIN を毎回呼ぶので、優先順がそのまま「1ターン内の手順」になる。
    # ⓪はデッキ切れ負け対策（§39）: 1位は残~13枚でトラッシュを山へ戻して回し続ける
    # （本人のデッキ切れ負け4% vs 我々52% の差の正体）。掘る前に戻す＝この位置が必須。
    if use_trainers:
        recycle = find_forced_recycle(obs, meta)
        if recycle is not None:
            return recycle

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

        # 0.5 残りの特性もすべて起動する（§40）: 行動差分で最大のギャップ＝1位は特性
        # 8.6回/試合 vs 我々4.0。特性はエンジン回転（ドロー→展開→特性増→…）の起点で、
        # develop-mask（draw/search/accel の効果文）に載らない特性が腐っていた。
        # option に出る特性は合法なものだけ・1ターン1回制限もエンジン管理＝空振りしない。
        abilities = by_type.get(OptionType.ABILITY, [])
        if abilities:
            return rng.choice(abilities)

    # 0.6 高prize退避（§80）: サイドを2枚以上取られる札がアクティブに居て、相手の場から
    #     次に落とされうるなら下がる。実測（226戦）では 3サイドのドローエンジンが
    #     アクティブで 0.58回/戦 KO され、献上サイドの 61% を占めていた。
    #     手段は ①自分のアクティブを下げる入替札（エネ不要）→ ②にげる の順。
    #     ただし今このターンに相手アクティブを倒せるなら攻撃を優先する（リーサル）。
    if evacuate_prize is not None and st is not None:
        ev = _find_evacuation(obs, meta, by_type, opts, evacuate_prize)
        if ev is not None:
            return ev

    # 1. 進化（基本的に得）
    if OptionType.EVOLVE in by_type:
        return rng.choice(by_type[OptionType.EVOLVE])

    # 2. どうぐ装着（HP/火力補助＝腐らせない。対象はエンジンが選ばせる）
    if use_trainers and OptionType.TOOL_CARD in by_type:
        return rng.choice(by_type[OptionType.TOOL_CARD])

    # 3./4. エネ付与とベンチ展開。bench_first=True なら順序を入れ替える（§72）:
    #    盤面が薄いまま殴り合いに入ると後続が立たない——不利な型との戦いで自ベンチ数が
    #    序盤（@10決定）だけ 1.85 vs 2.62 と負けていた実測への対処。既定は従来順。
    play_basic = [
        i
        for i in by_type.get(OptionType.PLAY, [])
        if _is_basic_pokemon_play(opts[i], obs, meta)
    ]
    if bench_first and play_basic:
        return rng.choice(play_basic)

    # 3. エネルギー付与: 既定はアクティブ優先（殴れる状態に近づける）。attach_priors が
    #    あれば上位帯の実測 P(ベンチ付与|アクティブの装着エネ枚数) に従って確率的にベンチの
    #    次アタッカーへ回す（帯共通の型: 空でも49%・2枚以上なら~80%がベンチ行き。
    #    アクティブ KO＝全エネ喪失のワンサイド負けを防ぐ）。
    if OptionType.ATTACH in by_type:
        attach = by_type[OptionType.ATTACH]
        to_active = [i for i in attach if opts[i].inPlayArea == AreaType.ACTIVE]
        to_bench = [i for i in attach if opts[i].inPlayArea == AreaType.BENCH]
        if attach_priors and to_active and to_bench and st is not None:
            active = (st.players[st.yourIndex].active or [None])[0]
            act_e = len(active.energyCards or []) if active is not None else 0
            p_bench = attach_priors.get(min(act_e, max(attach_priors)), 0.0)
            if rng.random() < p_bench:
                return _pick_bench_attach(to_bench, opts, st, meta)
        return rng.choice(to_active or attach)

    # 4. 手札のたねポケモンをベンチ展開（bench_first なら上で消化済み）
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

    # 6. にげる（攻撃できない active の救済・§31）: 現 active では攻撃不能で、ベンチに
    #    より多くエネが乗った子がいる時だけ交代する（にげるコストでエネを浪費しない保守条件。
    #    従来は RETREAT を一切使わず「詰み active」のまま殴られ続けていた）。
    if (
        OptionType.RETREAT in by_type
        and OptionType.ATTACK not in by_type
        and st is not None
    ):
        me = st.players[st.yourIndex]
        active = me.active[0] if me.active else None
        act_e = len(active.energyCards or []) if active else 0
        bench_e = max(
            (len(p.energyCards or []) for p in (me.bench or []) if p), default=0
        )
        if bench_e > act_e:
            return by_type[OptionType.RETREAT][0]

    # 7. 攻撃（最大ダメージ）
    if OptionType.ATTACK in by_type:
        return _argmax_damage_indices(by_type[OptionType.ATTACK], opts, meta)

    # 8. ターン終了
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


def _is_recycle_play(opt, obs: Observation, meta: CardMeta) -> bool:
    """PLAY が手札の山札リサイクル札（トラッシュ→山・§39）を使うものか."""
    cid = _hand_card_id(opt, obs)
    return cid is not None and meta.is_deck_recycle.get(cid, False)


def find_forced_recycle(
    obs: Observation, meta: CardMeta, recycle_at: int | None = None
) -> int | None:
    """残デッキ僅少時に打つべき山札リサイクル PLAY の option index を返す（無ければ None）.

    §39 の⓪段（デッキ切れ対策）の判定を関数として公開し、heuristic の MAIN 優先順と
    nn_mcts の探索前プレチェック（§43）で共用する。判定: 1. MAIN 選択であること →
    2. 手札にリサイクル札の PLAY がある → 3. 残デッキ ≤ 閾値なら先頭の index。
    recycle_at で発動閾値を上書きできる（未指定は _RECYCLE_AT・v3.5 系の閾値 A/B 用）。
    """
    sel = obs.select
    st = obs.current
    if sel is None or st is None or sel.type != SelectType.MAIN:
        return None
    recycle = [
        i
        for i, o in enumerate(sel.option)
        if o.type == OptionType.PLAY and _is_recycle_play(o, obs, meta)
    ]
    if not recycle:
        return None
    me = st.players[st.yourIndex]
    limit = _RECYCLE_AT if recycle_at is None else recycle_at
    if (me.deckCount or 0) > limit:
        return None
    return recycle[0]


def ko_threat(attacker, defender, meta: CardMeta) -> float:
    """攻め側の場から受け側アクティブへの KO 脅威度を 0〜1 で返す（§48）.

    流れ: 1. 受け側アクティブの残 HP を取る → 2. 攻め側の場の各ポケモン×各ワザで
    有効打点（弱点一致で×2）が残 HP に届くものを脅威候補に → 3. エネ充足度で重み付け
    （不足0=今打てる 1.0 / 不足1=次の1付与で打てる「育成中」0.5 / 不足2以上=0）、
    ベンチの攻撃役は昇格が要るのでさらに×0.5 → 4. 全候補の最大値を返す。
    近似: 色拘束・ワザ効果・どうぐ補正は見ない（枚数のみ）。正確な合法性・効果は
    エンジンのシミュレーションが担保する＝これは value/heuristic への事前信号。

    attacker/defender は PlayerState（テストではフェイク可）。
    """
    tgt = (defender.active or [None])[0] if defender is not None else None
    if attacker is None or tgt is None:
        return 0.0
    tgt_hp = tgt.hp or 0
    if tgt_hp <= 0:
        return 0.0
    weak = meta.weakness.get(tgt.id, -1)

    threat = 0.0
    candidates = [((attacker.active or [None])[0], 1.0)] + [
        (p, _THREAT_BENCH) for p in (attacker.bench or [])
    ]
    for pk, base_w in candidates:
        if pk is None or base_w <= threat:  # これ以上更新できない候補は飛ばす
            continue
        n_energy = len(pk.energies or [])
        mult = (
            _WEAKNESS_MULT
            if weak != -1 and weak == meta.pokemon_type.get(pk.id, -1)
            else 1
        )
        for aid in meta.card_attacks.get(pk.id, []):
            if meta.attack_damage(aid) * mult < tgt_hp:
                continue  # 届かないワザは脅威でない
            shortfall = meta.attack_cost.get(aid, 1) - n_energy
            w = (
                1.0
                if shortfall <= 0
                else (_THREAT_NEXT_TURN if shortfall == 1 else 0.0)
            )
            threat = max(threat, base_w * w)
    return threat


def _pick_bench_attach(indices: list[int], opts, st, meta: CardMeta) -> int:
    """ベンチ付与の対象を「攻撃準備に最も近い子」にする.

    選び方: 1. 最小の不足エネ（attack_cost − 装着数・撃てるワザが無い子は最後）
    → 2. 装着数が多い → 3. HP が高い。エネを散らして誰も完成しない事態を避け、
    次のアタッカーを一点集中で立てる（上位帯の「後続充電」の意図を写す）。
    """
    bench = st.players[st.yourIndex].bench or []

    def readiness(i: int):
        b = opts[i].inPlayIndex
        pk = bench[b] if b is not None and 0 <= b < len(bench) else None
        if pk is None:
            return (99, 0, 0)
        n_e = len(pk.energyCards or [])
        shortfalls = [
            max(meta.attack_cost.get(aid, 1) - n_e, 0)
            for aid in meta.card_attacks.get(pk.id, [])
        ]
        sf = min(shortfalls) if shortfalls else 99
        return (sf, -n_e, -(pk.hp or 0))

    return min(indices, key=readiness)


def _argmax_damage(opts, meta: CardMeta) -> int:
    """ATTACK オプション列の中で最大ダメージのインデックス."""
    return max(range(len(opts)), key=lambda i: meta.attack_damage(opts[i].attackId))


def _find_evacuation(
    obs: Observation,
    meta: CardMeta,
    by_type: dict[int, list[int]],
    opts,
    threshold: float,
) -> int | None:
    """高prize のアクティブを退避させる option index を返す（不要なら None）.

    判定の流れ: 1. アクティブが 2サイド以上の札か → 2. ベンチに交代先が居るか →
    3. 相手の場からの KO 脅威（§48 ko_threat）が閾値以上か → 4. こちらが相手アクティブを
    今倒せるなら攻撃を優先（退避しない）→ 5. 入替札（エネ不要）→ にげる の順で手段を返す。
    """
    st = obs.current
    if st is None:
        return None
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    active = (me.active or [None])[0]
    if active is None:
        return None
    # 1. 守る価値（取られるサイド枚数）が無いなら何もしない
    if meta.prize_value.get(active.id, 1) < 2:
        return None
    # 2. 交代先が居なければ退避できない
    if not [p for p in (me.bench or []) if p is not None]:
        return None
    # 3. 相手の場からの KO 脅威（0〜1）が閾値未満なら急がない
    if ko_threat(opp, me, meta) < threshold:
        return None
    # 4. リーサル優先: 今の攻撃で相手アクティブを落とせるなら退避しない
    opp_active = (opp.active or [None])[0]
    if opp_active is not None and OptionType.ATTACK in by_type:
        best = max(
            (meta.attack_damage(opts[i].attackId) for i in by_type[OptionType.ATTACK]),
            default=0,
        )
        if best >= (opp_active.hp or 0):
            return None
    # 5. 手段: 入替札（エネを失わない）→ にげる（コスト分のエネを失う）
    switches = [
        i
        for i in by_type.get(OptionType.PLAY, [])
        if _is_self_switch_play(opts[i], obs, meta)
    ]
    if switches:
        return switches[0]
    if OptionType.RETREAT in by_type:
        return by_type[OptionType.RETREAT][0]
    return None


def _is_self_switch_play(opt, obs: Observation, meta: CardMeta) -> bool:
    """PLAY が「自分のアクティブをベンチへ下げる」入替札か（§80）."""
    cid = _hand_card_id(opt, obs)
    return cid is not None and meta.is_self_switch.get(cid, False)


def _argmax_damage_indices(indices: list[int], opts, meta: CardMeta) -> int:
    """指定インデックス集合の中で最大ダメージのものを返す."""
    return max(indices, key=lambda i: meta.attack_damage(opts[i].attackId))


def _generic_select(
    obs: Observation,
    meta: CardMeta,
    fetch_priors: dict[int, float] | None = None,
    fix_switch_target: bool = False,
    setup_active_rule: bool = False,
) -> list[int]:
    """MAIN/ATTACK 以外のサブ選択を無難に処理する.

    - 数値選択（ドロー枚数など）は最大化。
    - セットアップのベンチ展開は可能な限り並べる。
    - **山札からの選択（サーチ）は maxCount まで取る**。旧実装は最小数（多くは 0 枚）で
      サーチを無駄撃ちしており、トレーナー活用の妨げだった。
    - **カテゴリ順は TO_HAND のみ `supporterPlayed` で分岐する**（§69）。上位帯実測の
      選好指数 lift が同じ「たねポケモン」で正反対だったため:
      サポート未使用（before）は **lift 1.93**（＝これから掘れるので先に場を作る札を取る）、
      使用後（after）は **lift 0.46**（＝掘り手段が尽きた後はたねを避ける）。
      → 未使用なら「たね > エネ > その他」・使用後は「エネ > その他 > たね」。
      §52 は after バケットの知見をサーチ全体へ一般化していた（採掘元の条件へ限定し直す）。
      TO_HAND 以外（TO_BENCH/ATTACH_TO 等）は構造的に無風なので after 側の順を使う。
    - **fetch_priors（§47）があるサーチでは、教師が実際に取っている札（取得率 > 0）を
      その率の降順でカテゴリ順より前に置く**（リサイクル札等のキーカードをサーチで
      埋没させない・§43 の発火機会＝札の可用性レバー）。未指定なら従来挙動。
      ⚠️ **取得率 0（提示されたのに教師が一度も取らなかった札）は前に出さない**。
      「priors に載っているか」だけで判定していた旧実装は、教師が避けた札をエネより
      上位に置いていた（実戦で 0.2〜0.4 回/戦発火・§68/§69）。
    - **昇格/交代先（TO_ACTIVE/SWITCH）はエネが乗り HP の残る子を優先**（§31。旧実装は
      先頭固定＝KO 後に空のたねを前に出してサイドレースを落としていた）。
    - `fix_switch_target=True` で **相手を引きずり出す候補（playerIndex が相手）を相手の
      ベンチとして解決**し、取れるサイドが最大の個体を選ぶ（§79 のバグ修正・既定 off）。
    - `setup_active_rule=True` で **開幕アクティブに特性持ち/重い札を置かない**（§89・既定 off）。
    - それ以外は必要最小数を先頭から選ぶ（任意選択は見送る）。
    """
    sel = obs.select
    n = len(sel.option)

    # 山へ戻す選択（リサイクル札の後続・§39）: 可能な最大数を戻す。既定の「最小数」だと
    # リサイクルを打っても 0 枚戻し＝空振りになる（E2E プローブで実測・8回全て空振りだった）。
    if sel.context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
        take = min(len(sel.option), max(sel.minCount, sel.maxCount))
        return list(range(take))

    if sel.context in (SelectContext.TO_ACTIVE, SelectContext.SWITCH):
        st = obs.current
        me = st.players[st.yourIndex] if st is not None else None
        opp = st.players[1 - st.yourIndex] if st is not None else None

        def promo_score(i: int) -> tuple[int, int, int]:
            # 流れ: 1. 候補が自分/相手どちらの場を指すか決める → 2. その場のベンチから
            # 個体を解決する → 3. 自分なら「育っている子」・相手なら「重い子」で順位付け。
            # エンジン実測: TO_ACTIVE の option は inPlayArea/inPlayIndex が None で、
            # `index` がベンチ位置を指す。inPlayIndex 形式にも防御的に対応する。
            o = sel.option[i]
            pk = None
            # 1. 所有者の解決（§79）: SWITCH には相手のベンチを指す候補（引きずり出し）が
            #    混ざる（実測 0.48回/戦）。旧実装は playerIndex を見ず全て自分の場として
            #    解決し、自分のベンチのエネ数で相手を順位付けていた（＝実質先頭取り）。
            is_opp = (
                fix_switch_target
                and st is not None
                and o.playerIndex is not None
                and o.playerIndex != st.yourIndex
            )
            owner = opp if is_opp else me
            bench = (owner.bench or []) if owner is not None else []
            if (
                o.inPlayArea == AreaType.BENCH
                and o.inPlayIndex is not None
                and o.inPlayIndex < len(bench)
            ):
                pk = bench[o.inPlayIndex]
            elif o.index is not None and o.index < len(bench):
                pk = bench[o.index]
            if pk is None:  # 解決できない選択肢は従来順（末尾寄せ）
                return (-1, -1, -i)
            # 3. 相手を引きずり出す時は基準が反転する: 上位帯は**取れるサイドが最大**の個体を
            #    引きずり出す（ランダム比 +0.239・§79/§80）。同値なら残 HP の少ない方＝
            #    倒し切りやすい方を選ぶ。自分の場は従来どおり「エネが乗り HP の残る子」。
            if is_opp:
                return (meta.prize_value.get(pk.id, 1), -(pk.hp or 0), -i)
            return (len(pk.energyCards or []), pk.hp or 0, -i)

        take = min(n, max(sel.minCount, 1))
        ranked = sorted(range(n), key=promo_score, reverse=True)
        return sorted(ranked[:take])

    if sel.type == SelectType.COUNT:
        return [max(range(n), key=lambda i: sel.option[i].number or 0)]

    if sel.context == SelectContext.SETUP_BENCH_POKEMON:
        return list(range(min(sel.maxCount, n)))

    if setup_active_rule and sel.context == SelectContext.SETUP_ACTIVE_POKEMON:
        # 開幕アクティブ（§89）: 上位帯は**特性持ちを前に置かない**（差が付く局面で
        # 特性持ちを選ぶ率 0.260 vs ランダム基準 0.475・n=192）。開幕札のサイド枚数も
        # 帯は {1:320, 2:41, 3:6} と 1 に偏る（我々は 40 局面中 19 が3サイド札だった）。
        # 順位: 1. 特性なし → 2. 取られるサイドが少ない → 3. HP が高い → 4. 元の順。
        st = obs.current
        hand = (st.players[st.yourIndex].hand or []) if st is not None else []

        def setup_score(i: int) -> tuple[int, int, int, int]:
            idx = sel.option[i].index
            if idx is None or idx >= len(hand) or hand[idx] is None:
                return (1, 9, 0, i)  # 解決できない候補は後回し
            cid = hand[idx].id
            return (
                int(meta.has_ability.get(cid, False)),
                meta.prize_value.get(cid, 1),
                -(meta.hp.get(cid, 0)),
                i,
            )

        return [min(range(n), key=setup_score)]

    if sel.deck is not None and sel.maxCount > 0:
        # 山札からのサーチ: 1. 教師が実際に取っている札（取得率 > 0）を率の降順 →
        # 2. 残りはカテゴリ順（TO_HAND は supporterPlayed で分岐・§69）、で取れるだけ取る
        # ⚠️ 山札検索の option は cardId を持たない（area/index/playerIndex/type のみ・
        # 実測確認済み）。実カードは sel.deck[option.index].id で解決する（旧実装は
        # cardId 直読みで常に 0 ＝カテゴリ分岐が機能していなかったバグ）。
        def _search_card_id(i: int) -> int:
            idx = sel.option[i].index
            if idx is None or sel.deck is None or not (0 <= idx < len(sel.deck)):
                return 0
            return sel.deck[idx].id or 0

        # そのターン既にサポートを使ったか（エンジン公式フラグ）。取れないときは未使用扱い
        # ＝採掘側 mine_search_role_priors.tohand_bucket と同じ判定にする。
        supporter_played = bool(obs.current is not None and obs.current.supporterPlayed)
        basics_first = sel.context == SelectContext.TO_HAND and not supporter_played

        def category_tier(cid: int) -> int:
            is_energy = meta.card_type.get(cid) in (
                CardType.BASIC_ENERGY,
                CardType.SPECIAL_ENERGY,
            )
            if basics_first:  # サポート未使用: たね > エネ > その他
                if meta.is_basic_pokemon(cid):
                    return 1
                return 2 if is_energy else 3
            # サポート使用後（および TO_HAND 以外）: エネ > その他 > たね
            if is_energy:
                return 1
            return 3 if meta.is_basic_pokemon(cid) else 2

        def rank(i: int) -> tuple[int, float, int]:
            cid = _search_card_id(i)
            prior = fetch_priors.get(cid) if fetch_priors else None
            if prior is not None and prior > 0.0:
                return (0, -prior, i)
            return (category_tier(cid), 0.0, i)

        take = max(sel.minCount, min(sel.maxCount, n))
        return sorted(sorted(range(n), key=rank)[:take])

    count = sel.minCount
    return list(range(count))
