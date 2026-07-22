"""上位帯（意図的に DL したリーダーボード相手・data/replays/others/）の
**山札サーチ/捨て札コストの「役割」別優先度**を集計する（§50・§51・§47/§49 の続き）.

§47 の cardId 単位 fetch_priors は「教師のデッキ構成と取得率が対応する」ため、
提出デッキと**完全一致**しないと信頼できない（デッキ完全一致は非現実的・ユーザー指摘）。
また §49 の調査で、cardId 完全一致に釣られて**自チームの対戦ログ（マッチメイキング相手＝
実力の裏付けなし）を「教師」として使ってしまう誤り**が判明した。

本スクリプトは cardId でなく**役割カテゴリ**（たね/進化ポケモン・基本/特殊エネ・
トレーナー(draw/search)・トレーナー(energy_accel)・山札リサイクル札・その他トレーナー・
どうぐ・スタジアム）に変換して集計する。デッキが違っても比較できるため、**教師プールを
「意図的に DL した上位帯（others/）」全員でプールし、母数を稼げる**（デッキタイプは
関係ない可能性がある＝ユーザー指摘）。教師プールは**既定で others/ のみ**（自チームの
対戦ログは実力の裏付けがないため除外・§49 の教訓）。

**バケット分割（§50）**: 全体（overall）の役割別取得率は「サーチ効果の種類」と
「サポート使用有無」の2要因が混ざった数字。切り分けるため、最頻の TO_HAND（手札に加える
＝一般的な「サーチ」・全体の75%実測）コンテキストに限定し、①1局面内で提示された
card_type が単一か複数混在か（実測: option==deck による「制限なし」判定は0件だった＝
TO_HAND 内でも85%が単一型に絞られている。複数型混在＝「トレーナー全般」等のより広い
サーチの代理指標） ②State.supporterPlayed（エンジン公式フラグ）でそのターンサポート
使用済みか、の2×2＝4バケットを追加で出す。

**候補数の正規化＝選好指数 lift（§51）**: 型混在局面は1局面あたりの提示数が役割ごとに
大きく偏るため、素の取得率は「候補が多いと薄まる」効果と混ざる。局面ごとに一様選択の
帰無仮説での期待取得数 `k×(offered_r/N)`（k=取得数, N=提示数）を計算し、実際の取得数との
比 Σ実際/Σ期待 を役割別に報告する（1.0=候補数なり＝優先度なし、>1=優先、<1=忌避）。
単一型局面では自明に1.0になるため mixed バケットのみ計算する。

**捨て札コストの優先度（§51・ユーザー提案）**: サーチ効果を使う際にコストとして手札を
捨てる場合、何を優先的に捨てているか。DISCARD（context=8）選択のうち、
`sel.effect.id`（発動中カードの ID・実測確認済み）の効果カテゴリに search ビットが
立つものだけを「サーチ効果のコスト」として抽出する（discard→search が同一 effect.id で
連続することを実測確認済み）。カードは `sel.deck` でなく手札（`AreaType.HAND`・実測
確認済み）由来のため `me.hand[option.index].id` で解決する。廃棄率は取得率と**解釈が逆**
（高い＝軽視され捨てられやすい、低い＝温存される）。実測 N は極小（DISCARD 全体で59局面・
サーチコストに絞ると10局面）のため参考程度。

**冪等・永続**（他マイニングと同じ運用）: 処理済み episode+席とバケット×役割別カウントを
--state に蓄積する。ペアリングは steps[i+1] 正実装（§45）。

出力はすべて数値・カテゴリ名まで（Pokémon Elements 非表示・規約遵守）。

    python scripts/mine_search_role_priors.py [--dir data/replays/others]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from cards import EFFECT_CATEGORIES, CardMeta, load_card_meta  # noqa: E402
from cg.api import CardType, Observation, SelectContext, to_observation_class  # noqa: E402
from mine_fetch_priorities import _search_card_id  # noqa: E402

_DRAWSEARCH_MASK = (1 << EFFECT_CATEGORIES.index("draw")) | (
    1 << EFFECT_CATEGORIES.index("search")
)
_SEARCH_BIT = 1 << EFFECT_CATEGORIES.index("search")
_ACCEL_MASK = 1 << EFFECT_CATEGORIES.index("energy_accel")
_TRAINER_TYPES = (CardType.ITEM, CardType.SUPPORTER)
_MIXED_BUCKETS = ("tohand_mixed_before", "tohand_mixed_after")


def card_role(cid: int, meta: CardMeta) -> str:
    """cardId をデッキ非依存の役割カテゴリへ変換する.

    判定順: 1. 山札リサイクル札（§39・独立フラグ）→ 2. トレーナー
    （draw/search → energy_accel → その他）→ 3. スタジアム/どうぐ →
    4. ポケモン（たね/進化）→ 5. エネルギー（基本/特殊）→ 6. その他。
    """
    if meta.is_deck_recycle.get(cid, False):
        return "recycle"
    ctype = meta.card_type.get(cid)
    if ctype in _TRAINER_TYPES:
        eff = meta.ability_effect.get(cid, 0)
        if eff & _DRAWSEARCH_MASK:
            return "trainer_drawsearch"
        if eff & _ACCEL_MASK:
            return "trainer_accel"
        return "trainer_other"
    if ctype == CardType.STADIUM:
        return "stadium"
    if ctype == CardType.TOOL:
        return "tool"
    if ctype == CardType.POKEMON:
        return "basic_pokemon" if meta.is_basic.get(cid, False) else "evolution_pokemon"
    if ctype == CardType.BASIC_ENERGY:
        return "basic_energy"
    if ctype == CardType.SPECIAL_ENERGY:
        return "special_energy"
    return "other"


def _record_search(sel, act: list[int], meta: CardMeta, role_counts: dict[str, list[int]]) -> None:
    """1つのサーチ局面を role_counts（role -> [取得, 提示]）へ加算する（bucket 非依存の中身）."""
    for i in range(len(sel.option)):
        role_counts[card_role(_search_card_id(sel, i), meta)][1] += 1
    for idx in act:
        if 0 <= idx < len(sel.option):
            role_counts[card_role(_search_card_id(sel, idx), meta)][0] += 1


def _record_lift(sel, act: list[int], meta: CardMeta, lift_counts: dict[str, list[float]]) -> None:
    """候補数で正規化した選好指数 lift 用のカウントを加算する（§51）.

    局面内の提示数N・取得数kから、役割rの一様選択時の期待取得数
    expected_r = k×(offered_r/N) を計算し、実際の取得数と合わせて積算する
    （最終的に Σ実際/Σ期待 が lift）。
    """
    n_options = len(sel.option)
    if n_options == 0:
        return
    role_of = [card_role(_search_card_id(sel, i), meta) for i in range(n_options)]
    offered_by_role: dict[str, int] = defaultdict(int)
    for r in role_of:
        offered_by_role[r] += 1
    taken_by_role: dict[str, int] = defaultdict(int)
    for idx in act:
        if 0 <= idx < n_options:
            taken_by_role[role_of[idx]] += 1
    k = sum(taken_by_role.values())
    for role, offered_r in offered_by_role.items():
        lift_counts[role][0] += taken_by_role.get(role, 0)
        lift_counts[role][1] += k * (offered_r / n_options)


def tohand_bucket(obs: Observation, sel, meta: CardMeta) -> str | None:
    """TO_HAND サーチを「型混在×サポート使用」で分類する（§50・実測に基づく2軸）.

    型混在: 1局面で提示された option の実カード card_type が複数種か（単一に絞られた
    「ポケモン限定」等のサーチと、より広い「トレーナー全般」等のサーチを区別する代理指標）。
    サポート使用: State.supporterPlayed（エンジン公式フラグ・自前のターン追跡は不要）。
    TO_HAND 以外の context は対象外（None）。
    """
    if sel.context != SelectContext.TO_HAND:
        return None
    types = {meta.card_type.get(_search_card_id(sel, i)) for i in range(len(sel.option))}
    mixed = "mixed" if len(types) > 1 else "single"
    sp = "after" if (obs.current is not None and obs.current.supporterPlayed) else "before"
    return f"tohand_{mixed}_{sp}"


def is_search_cost_discard(sel, meta: CardMeta) -> bool:
    """DISCARD 選択が『サーチ効果のコスト』として発生したものか（§51）.

    sel.effect（発動中カードの Card・実測確認済み）の効果カテゴリに search ビットが
    立つカードかで判定する（discard→search が同一 effect.id で連続することを実測確認済み）。
    """
    eff = sel.effect
    if eff is None:
        return False
    return bool(meta.ability_effect.get(eff.id, 0) & _SEARCH_BIT)


def _discard_card_id(obs: Observation, sel, option_i: int) -> int:
    """DISCARD 選択の実カード ID を解決する（§51・sel.deck は None＝手札由来。
    option.area=HAND・実測確認済み。sel.deck 由来の search と違い me.hand[index].id で解決）."""
    idx = sel.option[option_i].index
    st = obs.current
    if idx is None or st is None:
        return 0
    me = st.players[st.yourIndex]
    hand = me.hand or []
    if not (0 <= idx < len(hand)):
        return 0
    return hand[idx].id or 0


def _record_discard(
    obs: Observation, sel, act: list[int], meta: CardMeta, role_counts: dict[str, list[int]]
) -> None:
    """1つの DISCARD 選択を role_counts へ加算する（『廃棄率』＝取得率と解釈が逆・§51）."""
    for i in range(len(sel.option)):
        role_counts[card_role(_discard_card_id(obs, sel, i), meta)][1] += 1
    for idx in act:
        if 0 <= idx < len(sel.option):
            role_counts[card_role(_discard_card_id(obs, sel, idx), meta)][0] += 1


def mine_episode(
    ep: dict,
    seat: int,
    meta: CardMeta,
    buckets: dict[str, dict[str, list[int]]],
    bucket_n: dict[str, int],
    lift: dict[str, list[float]],
) -> int:
    """1 episode の指定席から全サーチ/DISCARD局面を集計し、局面数を返す.

    流れ: 1. steps[i+1] 正実装ペアリング（§45）で obs/act を対応付け →
    2. サーチ局面（sel.deck あり）なら buckets["overall"] へ常に加算 → TO_HAND なら
    型混在×サポート使用のサブバケット（§50）＋ mixed バケットは lift も加算（§51） →
    3. DISCARD 局面（context=8）ならサーチコスト判定の上 discard_all / discard_search_cost
    バケットへ加算（§51）。bucket_n は**局面数**（提示延べ枚数と違い1局面=1）。
    """
    steps = ep.get("steps") or []
    n = 0
    for i, step in enumerate(steps):
        rec = step[seat]
        if rec.get("status") != "ACTIVE" or not rec.get("observation"):
            continue
        if i + 1 >= len(steps):
            continue
        act = steps[i + 1][seat].get("action")
        if not isinstance(act, list):
            continue
        obs = to_observation_class(rec["observation"])
        sel = obs.select
        if sel is None or sel.maxCount < 1 or not sel.option:
            continue
        if sel.deck is not None:
            _record_search(sel, act, meta, buckets["overall"])
            bucket_n["overall"] += 1
            key = tohand_bucket(obs, sel, meta)
            if key is not None:
                _record_search(sel, act, meta, buckets[key])
                bucket_n[key] += 1
                if key in _MIXED_BUCKETS:
                    _record_lift(sel, act, meta, lift[key])
            n += 1
        elif sel.context == SelectContext.DISCARD:
            _record_discard(obs, sel, act, meta, buckets["discard_all"])
            bucket_n["discard_all"] += 1
            if is_search_cost_discard(sel, meta):
                _record_discard(obs, sel, act, meta, buckets["discard_search_cost"])
                bucket_n["discard_search_cost"] += 1
            n += 1
    return n


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"episodes": [], "buckets": {}, "bucket_n": {}, "lift": {}}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="山札サーチ/捨て札コストの役割別優先度のマイニング（§50/§51）")
    p.add_argument(
        "--dir",
        default="data/replays/others",
        help="episode JSON のルート（既定=others/＝意図的に DL した上位帯のみ。"
        "自チームの対戦ログ(マッチメイキング相手)は実力の裏付けがないため既定除外・§49）",
    )
    p.add_argument(
        "--state",
        default="data/fetch_priors/role_state.json",
        help="累積カウントの永続先（冪等・絞ったら生 JSON は削除してよい）",
    )
    p.add_argument("--out", default="data/fetch_priors/role_priors.json", help="出力 JSON")
    args = p.parse_args()

    meta = load_card_meta()
    state = _load_state(args.state)
    done = set(state["episodes"])
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for bucket, roles in state["buckets"].items():
        for role, to in roles.items():
            buckets[bucket][role] = list(to)
    bucket_n: dict[str, int] = defaultdict(int, state.get("bucket_n", {}))
    lift: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for bucket, roles in state.get("lift", {}).items():
        for role, to in roles.items():
            lift[bucket][role] = list(to)

    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    if not paths:
        raise SystemExit(f"JSON がありません: {args.dir}")
    n_new_ep = n_new_sel = 0
    teams_seen = set()
    for path in paths:
        with open(path) as f:
            ep = json.load(f)
        eid = str(
            ep.get("info", {}).get("EpisodeId")
            or os.path.splitext(os.path.basename(path))[0]
        )
        names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
        for seat in (0, 1):
            tag = f"{eid}#{seat}"
            if tag in done:
                continue
            done.add(tag)
            n_new_sel += mine_episode(ep, seat, meta, buckets, bucket_n, lift)
            n_new_ep += 1
            teams_seen.add(names[seat])

    os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
    out_state = {
        "episodes": sorted(done),
        "buckets": {b: dict(roles) for b, roles in buckets.items()},
        "bucket_n": dict(bucket_n),
        "lift": {b: dict(roles) for b, roles in lift.items()},
    }
    with open(args.state, "w") as f:
        json.dump(out_state, f)

    priors = {
        b: {r: (t / o if o > 0 else 0.0) for r, (t, o) in roles.items()}
        for b, roles in buckets.items()
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "source_dir": args.dir,
                "episodes": len(done),
                "priors": priors,
                "counts": out_state["buckets"],
                "lift": out_state["lift"],
            },
            f,
            indent=1,
        )

    print(
        f"新規 {n_new_ep} 席（{len(teams_seen)}チーム）を集計（累計 {len(done)} 席）→ {args.state}"
    )
    bucket_order = [
        "overall", "tohand_single_before", "tohand_single_after",
        "tohand_mixed_before", "tohand_mixed_after",
        "discard_all", "discard_search_cost",
    ]
    for bucket in bucket_order:
        roles = buckets.get(bucket)
        if not roles:
            continue
        n_offered = sum(o for _, o in roles.values())  # 提示延べ枚数（水増しされうる）
        n = bucket_n.get(bucket, 0)  # 局面数（独立サンプルの実数・統計的に信頼すべき単位）
        label = "廃棄率" if bucket.startswith("discard") else "取得率"
        print(f"\n=== {bucket}（局面数 {n}・提示延べ {n_offered}枚・{label}）===")
        if bucket != "overall" and n < 30:
            print(f"  ⚠️ 局面数 {n} と薄く、役割別の比較は参考程度に留める（§50/§51）")
        ordered = sorted(roles.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else -1))
        for role, (t, o) in ordered:
            if o == 0:
                continue
            print(f"  {role:20s}: {label} {t / o:.2f}（{t}/{o}）")
        if bucket in lift and bucket in _MIXED_BUCKETS:
            print("  --- 選好指数 lift（候補数正規化・1.0=候補数なり §51）---")
            ordered_lift = sorted(
                lift[bucket].items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else -1)
            )
            for role, (t, e) in ordered_lift:
                if e <= 0:
                    continue
                print(f"    {role:20s}: lift {t / e:.2f}（実{t:.1f}/期待{e:.1f}）")
    print(f"\n→ {args.out}")
    print(
        "\n※ 現行 _generic_select の固定順は「たね(basic_pokemon) > エネ > その他」。"
        "tohand_* バケットの順位とズレていれば固定順の精緻化を検討する。"
    )


if __name__ == "__main__":
    main()
