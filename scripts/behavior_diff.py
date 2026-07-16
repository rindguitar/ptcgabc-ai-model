"""2チームの replay から操縦の**行動差分レポート**を出す（§40・汎用）.

「違いを一個ずつ潰す」のでなく**測れる次元を全部並べて一度に見る**ための道具。
同一デッキ同士で比較すれば共適応の交絡を最小化できる（例: TeamA vs 我々の a4066acd）。

測る次元（すべて ID・数値・カテゴリまで＝Pokémon Elements 非表示）:
  A. MAIN 行動の内訳/試合（ATTACK/ABILITY/PLAY item/sup/stad/ATTACH active|bench/EVOLVE/RETREAT/TOOL/END）
  B. ペース: 総ターン・初攻撃ターン・サイド取得ペース・deckCount 軌跡（T5/T10/T15）・リサイクル
  C. 盤面運用: ベンチ数（T5/T10）・にげる回数
  D. サーチ取得の中身: 山札サーチで取った cardId 上位（フェッチ優先順位の差）
obs/action の対応は §33 検証済みの「ACTIVE の select ← 次 step の action」。

    python scripts/behavior_diff.py \
        --team-a "TeamA" --dir-a data/replays/others \
        --team-b "R.I" --dir-b data/replays/alphago
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from cg.api import AreaType, OptionType, SelectType  # noqa: E402
from deck import card_category  # noqa: E402


def _med(v: list) -> float:
    if not v:
        return float("nan")
    v = sorted(v)
    return v[len(v) // 2]


def _classify_main(opt, obs_d: dict, meta) -> str:
    """MAIN で選んだ option を人が読める行動ラベルへ（生 dict ベース・軽量）."""
    t = opt.get("type")
    if t == OptionType.ATTACK:
        return "攻撃"
    if t == OptionType.ABILITY:
        return "特性"
    if t == OptionType.EVOLVE:
        return "進化"
    if t == OptionType.RETREAT:
        return "にげる"
    if t == OptionType.TOOL_CARD:
        return "どうぐ"
    if t == OptionType.END:
        return "ターン終了"
    if t == OptionType.ATTACH:
        area = opt.get("inPlayArea")
        return "エネ付与(active)" if area == AreaType.ACTIVE else "エネ付与(bench)"
    if t == OptionType.PLAY:
        cur = obs_d.get("current") or {}
        me = (cur.get("players") or [{}, {}])[cur.get("yourIndex", 0)]
        hand = me.get("hand") or []
        idx = opt.get("index")
        if idx is not None and idx < len(hand):
            cid = hand[idx].get("id")
            cat = card_category(meta, cid)
            if cat == "poke":
                return "たね展開"
            return f"PLAY:{cat}"
        return "PLAY:?"
    return f"other:{t}"


def collect(paths: list[str], team: str, meta) -> dict:
    """1チーム分の行動統計を集める（流れ: 対局走査 → ACTIVE 決定と次stepの action を対応
    → A:MAIN 内訳 B:ペース C:盤面 D:サーチ取得 を試合単位で集計）."""
    games = 0
    main_per_game = Counter()  # 行動ラベル -> 総数（後で /games）
    turns = []
    first_attack_turn = []
    deck_at = defaultdict(list)  # turn -> deckCount
    bench_at = defaultdict(list)
    prize_taken_at10 = []
    recycle_games = 0
    fetch_ids = Counter()  # 山札サーチで取った cardId
    for path in paths:
        with open(path) as f:
            ep = json.load(f)
        names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
        steps = ep.get("steps") or []
        for seat in (0, 1):
            if names[seat] != team:
                continue
            games += 1
            fa = None
            prev_dc = None
            jumped = False
            seen_turn = set()
            for i, step in enumerate(steps):
                rec = step[seat]
                obs_d = rec.get("observation") or {}
                cur = obs_d.get("current") or {}
                players = cur.get("players")
                if players:
                    me = players[cur.get("yourIndex", seat)]
                    dc = me.get("deckCount")
                    turn = cur.get("turn") or 0
                    if dc is not None:
                        if prev_dc is not None and dc - prev_dc >= 5:
                            jumped = True
                        prev_dc = dc
                        if turn not in seen_turn:
                            seen_turn.add(turn)
                            deck_at[turn].append(dc)
                            bench_at[turn].append(len(me.get("bench") or []))
                            if turn == 10:
                                prize_taken_at10.append(6 - len(me.get("prize") or []))
                if rec.get("status") != "ACTIVE" or i + 1 >= len(steps):
                    continue
                act = steps[i + 1][seat].get("action")
                sel = obs_d.get("select") or {}
                opts = sel.get("option") or []
                if not isinstance(act, list) or not act:
                    continue
                # A) MAIN pick-one の行動分類
                if (
                    sel.get("type") == SelectType.MAIN
                    and len(opts) > 1
                    and len(act) == 1
                    and 0 <= act[0] < len(opts)
                ):
                    label = _classify_main(opts[act[0]], obs_d, meta)
                    main_per_game[label] += 1
                    if label == "攻撃" and fa is None:
                        fa = (cur.get("turn") or 0)
                # D) 山札サーチの取得中身。option に cardId は無く、option["index"] が
                #    select["deck"] 配列（{id,serial,...} の列）を指す（生 dict の実測）。
                deck_arr = sel.get("deck")
                if deck_arr and (sel.get("maxCount") or 0) >= 1:
                    for a in act:
                        if 0 <= a < len(opts):
                            di = opts[a].get("index")
                            if di is not None and 0 <= di < len(deck_arr):
                                cid = (deck_arr[di] or {}).get("id")
                                if cid:
                                    fetch_ids[cid] += 1
            if fa:
                first_attack_turn.append(fa)
            turns.append(max(seen_turn) if seen_turn else 0)
            if jumped:
                recycle_games += 1
    return {
        "games": games,
        "main": main_per_game,
        "turns": turns,
        "first_attack": first_attack_turn,
        "deck_at": deck_at,
        "bench_at": bench_at,
        "prize10": prize_taken_at10,
        "recycle": recycle_games,
        "fetch": fetch_ids,
    }


def report(a: dict, b: dict, name_a: str, name_b: str, meta) -> None:
    ga, gb = max(1, a["games"]), max(1, b["games"])
    print(f"=== 行動差分: {name_a}（{a['games']}試合） vs {name_b}（{b['games']}試合）===\n")
    print("--- A) MAIN 行動 回数/試合（差の大きい順） ---")
    labels = set(a["main"]) | set(b["main"])
    rows = []
    for lb in labels:
        ra, rb = a["main"][lb] / ga, b["main"][lb] / gb
        rows.append((abs(ra - rb), lb, ra, rb))
    for _, lb, ra, rb in sorted(rows, reverse=True):
        print(f"  {lb:16s}: {name_a} {ra:5.1f} / {name_b} {rb:5.1f}  (差 {ra-rb:+.1f})")
    print("\n--- B) ペース ---")
    print(f"  総ターン中央値      : {_med(a['turns']):.0f} / {_med(b['turns']):.0f}")
    print(f"  初攻撃ターン中央値  : {_med(a['first_attack']):.0f} / {_med(b['first_attack']):.0f}")
    print(f"  T10までの獲得サイド : {_med(a['prize10']):.0f} / {_med(b['prize10']):.0f}")
    for t in (5, 10, 15):
        print(f"  T{t} 残デッキ中央値  : {_med(a['deck_at'].get(t, [])):.0f} / {_med(b['deck_at'].get(t, [])):.0f}")
    print(f"  リサイクル発動率    : {a['recycle']/ga:.0%} / {b['recycle']/gb:.0%}")
    print("\n--- C) 盤面 ---")
    for t in (5, 10):
        print(f"  T{t} ベンチ数中央値  : {_med(a['bench_at'].get(t, [])):.0f} / {_med(b['bench_at'].get(t, [])):.0f}")
    print("\n--- D) 山札サーチで取る札 上位（cardId: 回/試合） ---")
    fa = {c: n / ga for c, n in a["fetch"].items()}
    fb = {c: n / gb for c, n in b["fetch"].items()}
    tops = sorted(set(list(fa) + list(fb)), key=lambda c: -(max(fa.get(c, 0), fb.get(c, 0))))[:10]
    for c in tops:
        cat = card_category(meta, c)
        print(f"  id{c}({cat:4s}): {name_a} {fa.get(c,0):4.1f} / {name_b} {fb.get(c,0):4.1f}  (差 {fa.get(c,0)-fb.get(c,0):+.1f})")


def main() -> None:
    p = argparse.ArgumentParser(description="2チームの操縦行動差分レポート")
    p.add_argument("--team-a", required=True)
    p.add_argument("--dir-a", required=True)
    p.add_argument("--team-b", required=True)
    p.add_argument("--dir-b", required=True)
    args = p.parse_args()
    meta = load_card_meta()
    pa = sorted(glob.glob(os.path.join(args.dir_a, "**", "*.json"), recursive=True))
    pb = sorted(glob.glob(os.path.join(args.dir_b, "**", "*.json"), recursive=True))
    if not pa or not pb:
        raise SystemExit("JSON が見つかりません（--dir-a/--dir-b を確認）")
    a = collect(pa, args.team_a, meta)
    b = collect(pb, args.team_b, meta)
    report(a, b, args.team_a, args.team_b, meta)


if __name__ == "__main__":
    main()
