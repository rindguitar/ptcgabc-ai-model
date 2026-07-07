"""NN policy の診断（Phase 3・torch・Docker）.

「policy が手を順位付けできているか／文脈フリーのショートカットに陥っていないか」を測る。
heuristic で対戦を進めながら各 MAIN/ATTACK 決定を解析する（net 自身では進めない＝中立な状態分布）。

測る指標（cardId/効果文は出さず数値のみ）:
  - shortcut度  : net top手 == 最高ダメージ手 の割合（高いほど「ダメージだけ見る」文脈無視の疑い）
  - 教師再現度  : net top手 == ISMCTS の手 の割合（蒸留が効いているか）
  - 教師shortcut: ISMCTS の手 == 最高ダメージ手 の割合（教師自身が文脈フリーでないか）
  - 自己一致率  : 同一局面で ISMCTS を2回引いた argmax の一致率（教師の確率性＝one-hot
                  ラベル雑音の下限。低い→soft-π 向き / 高いのに再現度低い→net の容量不足）
  - 文脈感度    : KO 可能手がある局面で net / 教師 が KO 手を選ぶ割合
  - 集中度      : priors の平均最大値・平均エントロピー（policy が自信を持てているか）
さらに教師強度: ISMCTS(TB) vs heuristic の勝率（教師が弱いと的も悪い）。

実行（Docker）:
    make exec CMD="python scripts/diagnose_policy.py --net models/pvnet_distill_best.pt"
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import torch  # noqa: E402

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from deckopt import _load_pool, default_opponent_paths  # noqa: E402
from harness import evaluate  # noqa: E402
from ismcts import make_ismcts_agent  # noqa: E402
from nn_eval import make_net_evaluator  # noqa: E402
from nn_mcts import _MCTS_SELECT_TYPES  # noqa: E402
from train import load_net  # noqa: E402


def _argmax(xs) -> int:
    return int(max(range(len(xs)), key=lambda i: xs[i]))


def analyze(net, meta, decks, rng, n_decisions, teacher_tb, device) -> dict:
    """heuristic で進めつつ各 MAIN/ATTACK 決定で net/教師/最高ダメージ手を比較する."""
    evaluator = make_net_evaluator(net, meta, device)
    heuristic = make_heuristic_agent(meta)
    s = {
        "n": 0,
        "net_eq_dmg": 0,
        "n_teacher": 0,
        "net_eq_teacher": 0,
        "teacher_eq_dmg": 0,
        "n_ko": 0,
        "net_ko": 0,
        "teacher_ko": 0,
        "sum_max": 0.0,
        "sum_ent": 0.0,
        "n_selfagree": 0,
        "teacher_selfagree": 0,
    }
    deck_i = 0
    while s["n"] < n_decisions:
        deck = decks[deck_i % len(decks)]
        deck_i += 1
        teacher = (
            make_ismcts_agent(meta, deck, deck, time_budget=teacher_tb)
            if teacher_tb > 0
            else None
        )
        obs_dict, _ = battle_start(deck, deck)
        try:
            for _ in range(100_000):
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    break
                sel = obs.select
                if sel is None:
                    break
                if sel.type in _MCTS_SELECT_TYPES and len(sel.option) > 1:
                    _, priors = evaluator(obs)
                    net_top = _argmax(priors)
                    dmgs = [
                        meta.attack_damage(o.attackId) if o.attackId is not None else 0
                        for o in sel.option
                    ]
                    dmg_top = _argmax(dmgs) if any(d > 0 for d in dmgs) else None
                    yi = obs.current.yourIndex
                    opp = obs.current.players[1 - yi]
                    opp_hp = opp.active[0].hp if opp.active else 0
                    ko = [i for i, d in enumerate(dmgs) if opp_hp > 0 and d >= opp_hp]

                    s["n"] += 1
                    if dmg_top is not None and net_top == dmg_top:
                        s["net_eq_dmg"] += 1
                    mx = max(priors)
                    s["sum_max"] += mx
                    s["sum_ent"] += -sum(p * math.log(p + 1e-9) for p in priors)
                    if ko:
                        s["n_ko"] += 1
                        if net_top in ko:
                            s["net_ko"] += 1
                    if teacher is not None:
                        t_act = teacher(obs, rng)
                        t_top = t_act[0] if len(t_act) == 1 else None
                        if t_top is not None:
                            s["n_teacher"] += 1
                            if net_top == t_top:
                                s["net_eq_teacher"] += 1
                            if dmg_top is not None and t_top == dmg_top:
                                s["teacher_eq_dmg"] += 1
                            if ko and t_top in ko:
                                s["teacher_ko"] += 1
                            # 教師の自己一致率: 同一局面で ISMCTS をもう一度引き argmax 一致を測る。
                            # 低い=教師が確率的→one-hot ラベルが局面ごとに矛盾→policy_loss の
                            # 下限が高い（soft-π 向き）。高いのに教師再現度が低い=net の
                            # 容量/特徴/学習量不足（soft-π では直らない）。
                            t_act2 = teacher(obs, rng)
                            t_top2 = t_act2[0] if len(t_act2) == 1 else None
                            if t_top2 is not None:
                                s["n_selfagree"] += 1
                                if t_top2 == t_top:
                                    s["teacher_selfagree"] += 1
                    if s["n"] >= n_decisions:
                        break
                obs_dict = battle_select(heuristic(obs, rng))
        finally:
            battle_finish()
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="NN policy の診断")
    p.add_argument("--net", default="models/pvnet_distill_best.pt")
    p.add_argument(
        "--deck",
        nargs="+",
        default=None,
        help="解析デッキ（未指定なら実メタ優先＝gauntlet/>replays/opp_decks/>data/*.csv の先頭3つ）",
    )
    p.add_argument("--decisions", type=int, default=200, help="解析する決定数")
    p.add_argument(
        "--teacher-tb",
        type=float,
        default=0.25,
        help="教師 ISMCTS の1手秒（0 で教師比較を省略）",
    )
    p.add_argument(
        "--strength-games",
        type=int,
        default=40,
        help="教師強度（ISMCTS vs heuristic）の試合数（0 で省略）",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_card_meta()
    paths = args.deck or default_opponent_paths()  # 未指定は実メタ優先
    decks = _load_pool(paths)[:3]  # 非デッキ CSV（カードデータ等）は除外される
    if not decks:
        raise SystemExit("解析できる60枚デッキが見つかりません（--deck で指定）")
    rng = random.Random(args.seed)
    net = load_net(args.net, device)

    print(f"=== policy 診断: {args.net}（{len(decks)}デッキ・{args.decisions}決定）===")
    s = analyze(net, meta, decks, rng, args.decisions, args.teacher_tb, device)
    n = max(1, s["n"])
    nt = max(1, s["n_teacher"])
    nk = max(1, s["n_ko"])
    nsa = max(1, s["n_selfagree"])
    print(f"解析決定数 = {s['n']}（うち KO可能局面 {s['n_ko']}）")
    print(
        f"shortcut度 : net top == 最高ダメージ手 = {s['net_eq_dmg'] / n:.2f}"
        "（高いほど文脈無視の疑い）"
    )
    if args.teacher_tb > 0:
        print(
            f"教師再現度 : net top == ISMCTS手     = {s['net_eq_teacher'] / nt:.2f}"
            f"（{s['n_teacher']}局面）"
        )
        print(
            f"教師shortcut: ISMCTS手 == 最高ダメージ = {s['teacher_eq_dmg'] / nt:.2f}"
        )
        print(
            f"教師自己一致率: 同一局面で ISMCTS 2回一致 = {s['teacher_selfagree'] / nsa:.2f}"
            f"（{s['n_selfagree']}局面）"
        )
        print(f"文脈感度(教師): KO局面でKOを選ぶ        = {s['teacher_ko'] / nk:.2f}")
    print(f"文脈感度(net) : KO局面でKOを選ぶ        = {s['net_ko'] / nk:.2f}")
    print(
        f"集中度: 平均最大prior = {s['sum_max'] / n:.2f}  平均エントロピー = {s['sum_ent'] / n:.2f}"
    )

    if args.strength_games > 0:
        print(f"\n=== 教師強度: ISMCTS(TB={args.teacher_tb}) vs heuristic ===")
        t = make_ismcts_agent(meta, decks[0], decks[0], time_budget=args.teacher_tb)
        res = evaluate(
            t, make_heuristic_agent(meta), decks[0], rng, args.strength_games
        )
        print(
            f"勝率 = {res['win_rate_a']:.3f}（{args.strength_games}試合・>0.6 なら教師は有用）"
        )

    print("\n読み方:")
    print("  shortcut度が高く文脈感度(net)が低い → 文脈無視のショートカット学習")
    print("  教師再現度が低い → そもそも蒸留が効いていない（容量/特徴/学習不足）")
    print("  教師強度が低い(<0.6) → 教師が弱い＝的が悪い（TBを上げる）")
    print("  平均最大priorが低い(≈1/手数) → policy が自信を持てず平坦")
    print("  【soft-π か 容量 かの切り分け】")
    print(
        "    自己一致率が低い(≲0.6) → 教師が確率的でone-hotラベルが矛盾 → soft-π が正解"
    )
    print(
        "    自己一致率が高い(≳0.8)のに教師再現度が低い → net が fit 不能 → 容量/特徴/学習量"
    )


if __name__ == "__main__":
    main()
