"""上位帯（意図的に DL したリーダーボード相手・data/replays/others/）の
**山札サーチ「役割」別優先度**を集計する（§50・§47/§49 の続き）.

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

**バケット分割（§50 ユーザー指摘の2軸で分離）**: 全体（overall）の役割別取得率は
「サーチ効果の種類」と「サポート使用有無」の2要因が混ざった数字。切り分けるため、
最頻の TO_HAND（手札に加える＝一般的な「サーチ」・全体の75%実測）コンテキストに限定し、
①1局面内で提示された card_type が単一か複数混在か（実測: option==deck による「制限なし」
判定は0件だった＝TO_HAND 内でも85%が単一型に絞られている。複数型混在＝「トレーナー全般」
等のより広いサーチの代理指標） ②State.supporterPlayed（エンジン公式フラグ）でそのターン
サポート使用済みか、の2×2＝4バケットを追加で出す。**tohand_mixed_before は実測 N=11 と
極小**（役割8〜9種に対し1局面/種未満）で、統計的に意味を持たないため件数を明示し、
解釈に注意を促す。

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
_ACCEL_MASK = 1 << EFFECT_CATEGORIES.index("energy_accel")
_TRAINER_TYPES = (CardType.ITEM, CardType.SUPPORTER)


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


def mine_episode_roles(
    ep: dict, seat: int, meta: CardMeta, buckets: dict[str, dict[str, list[int]]]
) -> int:
    """1 episode の指定席から全サーチ局面を集計し、局面数を返す.

    流れ: 1. steps[i+1] 正実装ペアリング（§45）で obs/act を対応付け →
    2. サーチ局面なら buckets["overall"] へ常に加算 → 3. TO_HAND なら型混在×サポート使用の
    サブバケットにも加算（§50）。
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
        if sel is None or sel.deck is None or sel.maxCount < 1 or not sel.option:
            continue
        _record_search(sel, act, meta, buckets["overall"])
        key = tohand_bucket(obs, sel, meta)
        if key is not None:
            _record_search(sel, act, meta, buckets[key])
        n += 1
    return n


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"episodes": [], "buckets": {}}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="山札サーチ役割別優先度のマイニング（§50）")
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
            n_new_sel += mine_episode_roles(ep, seat, meta, buckets)
            n_new_ep += 1
            teams_seen.add(names[seat])

    os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
    out_state = {
        "episodes": sorted(done),
        "buckets": {b: dict(roles) for b, roles in buckets.items()},
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
            {"source_dir": args.dir, "episodes": len(done), "priors": priors, "counts": out_state["buckets"]},
            f,
            indent=1,
        )

    print(
        f"新規 {n_new_ep} 席（{len(teams_seen)}チーム）を集計（累計 {len(done)} 席）→ {args.state}"
    )
    bucket_order = ["overall", "tohand_single_before", "tohand_single_after",
                     "tohand_mixed_before", "tohand_mixed_after"]
    for bucket in bucket_order:
        roles = buckets.get(bucket)
        if not roles:
            continue
        n_decisions = sum(o for _, o in roles.values())  # 提示延べ数（局面数の目安）
        print(f"\n=== {bucket}（提示延べ {n_decisions}）===")
        if bucket != "overall" and n_decisions < 100:
            print("  ⚠️ サンプルが薄く、役割別の比較は参考程度に留める（§50）")
        ordered = sorted(roles.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else -1))
        for role, (t, o) in ordered:
            if o == 0:
                continue
            print(f"  {role:20s}: 取得率 {t / o:.2f}（{t}/{o}）")
    print(f"\n→ {args.out}")
    print(
        "\n※ 現行 _generic_select の固定順は「たね(basic_pokemon) > エネ > その他」。"
        "tohand_* バケットの順位とズレていれば固定順の精緻化を検討する。"
    )


if __name__ == "__main__":
    main()
