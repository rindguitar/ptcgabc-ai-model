"""進化計算によるデッキ最適化（固定プールへの最悪ケース勝率最大化）.

候補デッキを集団で保持し、各世代で「相手プール各デッキへの勝率の**最小値（最悪ケース）**」
を適応度に上位を残し、合法性保存の変異で子を作る。操縦は共通の高速方策（既定: ヒューリスティック）。

これは iterated best response / double oracle（リーグ）の**心臓部**。loop で包むと、
出力デッキをプールに追加→再最適化…で頑健なデッキへ近づく。

注意: ヒューリスティック操縦はコンボを回せず単純な攻撃デッキを過大評価する（操縦バイアス）。
最終候補は ISMCTS 操縦で再評価する前提。デッキ＝カード ID 60要素リスト（名称は扱わない）。
"""

from __future__ import annotations

import argparse
import glob
import random
import sys

from agents import Agent, make_heuristic_agent
from cards import CardMeta, load_card_meta
from deck import DECK_SIZE, is_legal, load_deck, mutate, random_legal_deck, save_deck
from harness import evaluate_decks


def fitness(
    deck: list[int],
    pool: list[list[int]],
    agent: Agent,
    rng: random.Random,
    games_per_opp: int,
) -> dict:
    """候補 deck の適応度。プール各デッキへの勝率と、その最小値/平均を返す."""
    win_rates = [
        evaluate_decks(deck, opp, agent, rng, games_per_opp)["win_rate_a"]
        for opp in pool
    ]
    return {
        "min": min(win_rates),
        "mean": sum(win_rates) / len(win_rates),
        "per_opp": win_rates,
    }


def _mutate_n(
    deck: list[int], ref_deck: list[int], rng: random.Random, n: int
) -> list[int]:
    """合法性保存の変異を n 回重ねる."""
    d = deck
    for _ in range(n):
        d = mutate(d, ref_deck, rng)
    return d


def _make_child(
    parents: list[list[int]],
    base: list[list[int]],
    ref_deck: list[int],
    rng: random.Random,
    mutations_per_child: int,
    max_swaps: int,
    explore_frac: float,
) -> list[int]:
    """子デッキを生成する.

    確率 explore_frac で**多様性注入**（多数変異で別アーキタイプ寄りの合法デッキ＝局所最適を脱出）。
    それ以外は親を変異。max_swaps>1 のとき変異枚数を 1..max_swaps で可変にし、近所も遠方も探る。
    """
    if explore_frac > 0 and rng.random() < explore_frac:
        return random_legal_deck(rng.choice(base), rng, swaps=max(20, max_swaps * 5))
    n = rng.randint(1, max_swaps) if max_swaps > 1 else mutations_per_child
    return _mutate_n(rng.choice(parents), ref_deck, rng, n)


def evolve(
    pool: list[list[int]],
    meta: CardMeta | None = None,
    *,
    seeds: list[list[int]] | None = None,
    rng: random.Random | None = None,
    pop_size: int = 16,
    generations: int = 8,
    games_per_opp: int = 12,
    elite: int = 4,
    mutations_per_child: int = 2,
    max_swaps: int = 1,
    explore_frac: float = 0.0,
    objective: str = "min",
    agent: Agent | None = None,
    ref_deck: list[int] | None = None,
    verbose: bool = False,
) -> dict:
    """進化計算でプールへの最悪ケース（既定）勝率を最大化するデッキを探す.

    max_swaps>1 で変異枚数を可変化（近所も遠方も探る）、explore_frac>0 で毎世代ランダムな
    合法デッキを混ぜて多様な軸・カードを探索する（局所最適の脱出）。

    Returns:
        dict: best デッキと適応度、世代ごとの履歴。
    """
    rng = rng or random.Random(0)
    meta = meta or load_card_meta()
    agent = agent or make_heuristic_agent(meta)
    ref_deck = ref_deck or pool[0]
    base = seeds or list(pool)

    # 初期集団: シードから変異＋多様性注入で散らす
    population = [
        _make_child(
            base, base, ref_deck, rng, mutations_per_child, max_swaps, explore_frac
        )
        for _ in range(pop_size)
    ]

    history = []
    best_key, best_fit, best_deck = (-1.0, -1.0), None, None
    for gen in range(generations):
        scored = []
        for d in population:
            f = fitness(d, pool, agent, rng, games_per_opp)
            scored.append((f[objective], f, d))
        # 主目的（最悪ケース等）→ 同点は平均勝率で tie-break
        scored.sort(key=lambda x: (x[0], x[1]["mean"]), reverse=True)

        top_key = (scored[0][0], scored[0][1]["mean"])
        if top_key > best_key:
            best_key, best_fit, best_deck = top_key, scored[0][1], scored[0][2]

        history.append(
            {"gen": gen, objective: scored[0][0], "per_opp": scored[0][1]["per_opp"]}
        )
        if verbose:
            per = [round(x, 2) for x in scored[0][1]["per_opp"]]
            print(
                f"gen {gen}: best {objective}={scored[0][0]:.3f} mean={scored[0][1]['mean']:.3f} per_opp={per}"
            )

        # 次世代: エリート保存 + 変異/多様性注入
        elites = [d for _, _, d in scored[:elite]]
        population = list(elites)
        while len(population) < pop_size:
            population.append(
                _make_child(
                    elites,
                    base,
                    ref_deck,
                    rng,
                    mutations_per_child,
                    max_swaps,
                    explore_frac,
                )
            )

    return {"deck": best_deck, "fitness": best_fit, "history": history}


def _load_pool(paths: list[str]) -> list[list[int]]:
    """60枚デッキの CSV 群をプールとして読み込む.

    カードデータ等の非デッキ CSV（パース不能・60枚でない）は黙って除外する。
    """
    pool = []
    for p in paths:
        try:
            d = load_deck(p)
        except ValueError:
            continue  # int 列でない（カードデータ CSV 等）
        if len(d) == DECK_SIZE:
            pool.append(d)
    if not pool:
        raise ValueError("有効な 60 枚デッキがプールにありません")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="進化計算デッキ最適化（最悪ケース勝率最大化）"
    )
    parser.add_argument(
        "--pool",
        nargs="+",
        default=None,
        help="相手プールの deck CSV 群（未指定なら data/*.csv の 60枚デッキ）",
    )
    parser.add_argument("--pop", type=int, default=16, help="集団サイズ")
    parser.add_argument("--gens", type=int, default=8, help="世代数")
    parser.add_argument("--games", type=int, default=12, help="相手1体あたりの対戦数")
    parser.add_argument(
        "--objective", choices=["min", "mean"], default="min", help="適応度"
    )
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument(
        "--out", default="models/optimized_deck.csv", help="出力デッキ CSV"
    )
    args = parser.parse_args()

    pool_paths = args.pool or sorted(glob.glob("data/*.csv"))
    pool = _load_pool(pool_paths)
    print(
        f"プール {len(pool)} デッキで最適化（objective={args.objective}, pop={args.pop}, gens={args.gens}）"
    )

    rng = random.Random(args.seed)
    meta = load_card_meta()
    result = evolve(
        pool,
        meta,
        rng=rng,
        pop_size=args.pop,
        generations=args.gens,
        games_per_opp=args.games,
        objective=args.objective,
        verbose=True,
    )

    fit = result["fitness"]
    print(f"\n最良: {args.objective}={fit[args.objective]:.3f} mean={fit['mean']:.3f}")
    if not is_legal(result["deck"], pool[0]):
        print("警告: 出力デッキが非合法（保存しません）")
        sys.exit(1)
    save_deck(result["deck"], args.out)
    print(f"出力デッキを {args.out} に保存（追跡外）")


if __name__ == "__main__":
    main()
