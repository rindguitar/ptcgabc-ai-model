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

出力は現行 `_generic_select` の固定順（たね>エネ>その他）が上位帯の実際の基準と
合っているかの検証材料になる。ズレていれば、この役割別優先度で固定順を精緻化できる。

**冪等・永続**（他マイニングと同じ運用）: 処理済み episode+席と役割別カウントを
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
from cg.api import CardType, to_observation_class  # noqa: E402
from mine_fetch_priorities import _search_card_id  # noqa: E402

_DRAWSEARCH_MASK = (1 << EFFECT_CATEGORIES.index("draw")) | (
    1 << EFFECT_CATEGORIES.index("search")
)
_ACCEL_MASK = 1 << EFFECT_CATEGORIES.index("energy_accel")
_TRAINER_TYPES = (CardType.ITEM, CardType.SUPPORTER)

# 出力する役割の既定順（decisions.md §50 の報告で「現行固定順との比較」に使う）
ROLE_ORDER = (
    "basic_pokemon",
    "evolution_pokemon",
    "basic_energy",
    "special_energy",
    "trainer_drawsearch",
    "trainer_accel",
    "recycle",
    "tool",
    "stadium",
    "trainer_other",
    "other",
)


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


def _count_search_roles(
    obs, act: list[int], meta: CardMeta, counts: dict[str, list[int]]
) -> bool:
    """1つの山札サーチ選択を counts（role -> [取得, 提示]）へ加算する."""
    sel = obs.select
    if sel is None or sel.deck is None or sel.maxCount < 1 or not sel.option:
        return False
    for i in range(len(sel.option)):
        counts[card_role(_search_card_id(sel, i), meta)][1] += 1
    for idx in act:
        if 0 <= idx < len(sel.option):
            counts[card_role(_search_card_id(sel, idx), meta)][0] += 1
    return True


def mine_episode_roles(
    ep: dict, seat: int, meta: CardMeta, counts: dict[str, list[int]]
) -> int:
    """1 episode の指定席から全サーチ局面を役割別に集計し、局面数を返す.

    ペアリングは §45 の正実装: obs[i] への応答は steps[i+1][seat].action。
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
        if _count_search_roles(obs, act, meta, counts):
            n += 1
    return n


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"episodes": [], "counts": {}}
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
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for role, to in state["counts"].items():
        counts[role] = list(to)

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
            n_new_sel += mine_episode_roles(ep, seat, meta, counts)
            n_new_ep += 1
            teams_seen.add(names[seat])

    os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
    state = {"episodes": sorted(done), "counts": {r: c for r, c in counts.items()}}
    with open(args.state, "w") as f:
        json.dump(state, f)

    priors = {r: (t / o if o > 0 else 0.0) for r, (t, o) in counts.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"source_dir": args.dir, "episodes": len(done), "priors": priors, "counts": dict(counts)},
            f,
            indent=1,
        )

    print(
        f"新規 {n_new_ep} 席（{len(teams_seen)}チーム）・サーチ局面 {n_new_sel} を集計"
        f"（累計 {len(done)} 席）→ {args.state}"
    )
    print(f"=== 役割別サーチ取得率（{args.dir}・全チームプール）→ {args.out} ===")
    ordered = sorted(
        counts.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else -1)
    )
    for role, (t, o) in ordered:
        if o == 0:
            continue
        print(f"  {role:20s}: 取得率 {t / o:.2f}（{t}/{o}）")
    print(
        "\n※ 現行 _generic_select の固定順は「たね(basic_pokemon) > エネ > その他」。"
        "上表の順位とズレていれば固定順の精緻化を検討する。"
    )


if __name__ == "__main__":
    main()
