"""bounded double-oracle リーグ（頑健なデッキの反復探索）.

各反復で「現在のプールへの最悪ケース勝率を最大化する対抗デッキ」を進化計算で作り、
プールに追加する。プールは上限 K で打ち切り、超過時は**最も冗長な進化枠デッキ**を追い出す。
実メタの seed デッキは固定（pin）して試験官として残す。各反復の候補は別枠の
「チャンピオン保管庫」に記録し、最終的に**蓄積した全デッキへの最悪ケースが最大の1デッキ**を選ぶ。

これにより、際限ないプール肥大を避けつつ（コスト一定）、循環・多様性崩壊も防ぐ。
操縦は高速なヒューリスティック（最終候補は別途 ISMCTS で再評価する前提）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

from agents import Agent, make_heuristic_agent
from cards import CardMeta, load_card_meta
from deck import DECK_SIZE, is_legal, save_deck
from deckopt import _load_pool, evolve
from harness import evaluate_decks


def deck_similarity(a: list[int], b: list[int]) -> float:
    """2デッキの重なり（multiset 共通枚数 / 60）。1.0 で完全一致."""
    common = sum((Counter(a) & Counter(b)).values())
    return common / DECK_SIZE


def _most_redundant_index(decks: list[list[int]]) -> int:
    """最も冗長なデッキ（他デッキとの最大類似度が最も高いもの）の添字."""
    best_i, best_sim = 0, -1.0
    for i, d in enumerate(decks):
        sim = max(
            (deck_similarity(d, e) for j, e in enumerate(decks) if j != i),
            default=0.0,
        )
        if sim > best_sim:
            best_i, best_sim = i, sim
    return best_i


def worst_case(
    deck: list[int],
    pool: list[list[int]],
    agent: Agent,
    rng: random.Random,
    games_per_opp: int,
) -> float:
    """deck の pool への最悪ケース勝率（自分自身は除外）."""
    opps = [p for p in pool if p != deck]
    if not opps:
        return 1.0
    return min(
        evaluate_decks(deck, opp, agent, rng, games_per_opp)["win_rate_a"]
        for opp in opps
    )


def _rng_state_to_json(rng: random.Random) -> list:
    """random.Random の内部状態を JSON 化可能な形にする."""
    version, keys, gauss = rng.getstate()
    return [version, list(keys), gauss]


def _rng_state_from_json(state: list):
    """JSON から random.Random の内部状態（タプル）を復元する."""
    version, keys, gauss = state
    return (version, tuple(keys), gauss)


def save_state(path: str, state: dict) -> None:
    """リーグ状態を JSON で原子的に保存（tmp→replace。models/ 配下・追跡外）."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)  # 書き込み中クラッシュでも本体は壊れない


def load_state(path: str) -> dict:
    """保存済みリーグ状態を読み込む."""
    with open(path) as f:
        return json.load(f)


def run_league(
    seed_decks: list[list[int]],
    meta: CardMeta | None = None,
    *,
    rng: random.Random | None = None,
    cap: int = 10,
    iterations: int = 8,
    games_per_opp: int = 10,
    eval_games: int | None = None,
    plateau: int = 3,
    agent: Agent | None = None,
    evolve_kwargs: dict | None = None,
    verbose: bool = False,
    checkpoint_path: str | None = None,
    resume: bool = False,
    incumbent: list[int] | None = None,
) -> dict:
    """bounded double-oracle リーグを回し、最も頑健なチャンピオンデッキを返す.

    incumbent（前チャンピオン）を渡すと最終選抜の候補に含めるため、出力チャンピオンは
    前チャンピオン以上の評価が保証される（測定ノイズの範囲で**非弱化の世代更新**）。

    Returns:
        dict: champion デッキ、全プール最悪ケース勝率、保管庫サイズ、履歴、retained。
    """
    rng = rng or random.Random(0)
    meta = meta or load_card_meta()
    agent = agent or make_heuristic_agent(meta)
    eval_games = eval_games or games_per_opp
    evolve_kwargs = evolve_kwargs or {}

    pinned = [list(d) for d in seed_decks]  # 実メタ: 固定の試験官
    max_evolved = max(1, cap - len(pinned))

    # 状態の初期化 or チェックポイントからの復元
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        st = load_state(checkpoint_path)
        evolved = st["evolved"]
        archive = st["archive"]
        champions = st["champions"]
        history = st["history"]
        best_score = st["best_score"]
        no_improve = st["no_improve"]
        done_iters = st["done_iters"]
        if st.get("rng_state"):
            rng.setstate(_rng_state_from_json(st["rng_state"]))
        if verbose:
            print(
                f"レジューム: 反復 {done_iters} から再開 "
                f"(evolved={len(evolved)}, archive={len(archive)})"
            )
    else:
        evolved = []
        archive = [list(d) for d in seed_decks]  # これまで見た全デッキ
        champions = []  # 各反復の候補（最終選抜用）
        history = []
        best_score, no_improve, done_iters = -1.0, 0, 0

    for _ in range(iterations):
        pool = pinned + evolved
        res = evolve(
            pool,
            meta,
            rng=rng,
            games_per_opp=games_per_opp,
            agent=agent,
            **evolve_kwargs,
        )
        cand = res["deck"]
        score = res["fitness"]["min"]  # 作成時点プールへの最悪ケース
        archive.append(cand)
        champions.append(cand)
        evolved.append(cand)

        # 上限超: 最も冗長な進化枠デッキを追い出す（多様性を保つ）
        if len(evolved) > max_evolved:
            evolved.pop(_most_redundant_index(evolved))

        done_iters += 1
        if score > best_score + 1e-9:
            best_score, no_improve = score, 0
        else:
            no_improve += 1
        history.append(
            {
                "iter": done_iters - 1,
                "cand_worstcase_vs_pool": score,
                "pool_size": len(pool),
            }
        )
        if verbose:
            print(
                f"iter {done_iters - 1}: cand worst-case vs pool={score:.3f} "
                f"pool={len(pool)} evolved={len(evolved)} archive={len(archive)}"
            )

        # 各反復後にチェックポイント保存（クラッシュ耐性・再開可能に）
        if checkpoint_path:
            save_state(
                checkpoint_path,
                {
                    "evolved": evolved,
                    "archive": archive,
                    "champions": champions,
                    "history": history,
                    "best_score": best_score,
                    "no_improve": no_improve,
                    "done_iters": done_iters,
                    "seeds_count": len(pinned),
                    "rng_state": _rng_state_to_json(rng),
                },
            )

        # プラトー検出
        if no_improve >= plateau:
            if verbose:
                print(f"プラトー {plateau} 反復 → 停止")
            break

    # 最終選抜: 進化候補 ＋ 前チャンピオン(incumbent) を蓄積プール全体への
    # 最悪ケースで再評価し最良を選ぶ。incumbent を候補に含めることで非弱化を保証する。
    candidates = list(champions)
    if incumbent is not None:
        candidates.append(list(incumbent))
    if not candidates:
        raise ValueError(
            "最終選抜の候補がありません（iterations>=1 か incumbent を指定）"
        )
    scored = [(worst_case(d, archive, agent, rng, eval_games), d) for d in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_wc, best_deck = scored[0]
    retained = incumbent is not None and best_deck == list(incumbent)
    return {
        "champion": best_deck,
        "champion_worstcase_vs_archive": best_wc,
        "archive_size": len(archive),
        "history": history,
        "retained": retained,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="bounded double-oracle デッキリーグ")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="seed（固定）デッキ CSV 群（未指定なら data/*.csv の60枚デッキ）",
    )
    parser.add_argument(
        "--extra-seeds",
        nargs="+",
        default=None,
        help="seed に追加で固定する CSV（例: 既存チャンピオン models/champion_deck.csv）",
    )
    parser.add_argument("--cap", type=int, default=10, help="プール上限 K")
    parser.add_argument("--iters", type=int, default=8, help="リーグ反復数")
    parser.add_argument("--games", type=int, default=10, help="相手1体あたりの対戦数")
    parser.add_argument("--pop", type=int, default=12, help="evolve の集団サイズ")
    parser.add_argument("--gens", type=int, default=6, help="evolve の世代数")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument(
        "--plateau",
        type=int,
        default=4,
        help="最悪ケースが改善しない反復がこの数に達したら早期停止（大きくすると止めない）",
    )
    parser.add_argument(
        "--out", default="models/champion_deck.csv", help="出力チャンピオン CSV"
    )
    parser.add_argument(
        "--checkpoint",
        default="models/league/state.json",
        help="チェックポイントの保存先（各反復後に保存・追跡外）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="チェックポイントから続きを再開する（--iters は今回の追加反復数）",
    )
    args = parser.parse_args()

    seed_paths = args.seeds or sorted(glob.glob("data/*.csv"))
    seeds = _load_pool(seed_paths)
    if args.extra_seeds:
        # 既存チャンピオン等を固定の試験官として追加（重複は除外）
        for extra in _load_pool(args.extra_seeds):
            if extra not in seeds:
                seeds.append(extra)
    mode = "再開" if args.resume else "新規"
    print(
        f"seed {len(seeds)} デッキでリーグ{mode}"
        f"（cap={args.cap}, 追加iters={args.iters}, extra={len(args.extra_seeds or [])}）"
    )

    # 既存チャンピオン（出力先）があれば最終選抜の候補に入れ、非弱化の世代更新にする
    incumbent = None
    if os.path.exists(args.out):
        try:
            incumbent = _load_pool([args.out])[0]
        except (ValueError, IndexError):
            incumbent = None
    if incumbent is not None:
        print("前チャンピオンを最終選抜候補に含めます（非弱化保証）")

    rng = random.Random(args.seed)
    meta = load_card_meta()
    result = run_league(
        seeds,
        meta,
        rng=rng,
        cap=args.cap,
        iterations=args.iters,
        games_per_opp=args.games,
        plateau=args.plateau,
        evolve_kwargs={"pop_size": args.pop, "generations": args.gens},
        verbose=True,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        incumbent=incumbent,
    )

    status = (
        "据え置き（前チャンピオンが最良）"
        if result.get("retained")
        else "更新（新チャンピオン）"
    )
    print(
        f"\nチャンピオン[{status}]: 全プール最悪ケース勝率="
        f"{result['champion_worstcase_vs_archive']:.3f} (archive {result['archive_size']} デッキ)"
    )
    if not is_legal(result["champion"], seeds[0]):
        print("警告: チャンピオンが非合法（保存しません）")
        return
    save_deck(result["champion"], args.out)
    print(f"チャンピオンを {args.out} に保存（追跡外）")


if __name__ == "__main__":
    main()
