"""AlphaZero 反復学習ドライバ（Phase 3・torch・Docker）.

{現ネットで self-play → 学習 → 保存} を反復し、PVNet（policy/value）を強くする。
torch が要るので Docker で実行する:
    make train
    make exec CMD="python scripts/train_alphazero.py --iterations 20 --games 16"

出力ネット models/pvnet.pt は追跡外。--resume で続きから。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch  # noqa: E402

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from harness import evaluate  # noqa: E402
from net import PVNet  # noqa: E402
from nn_eval import make_net_evaluator  # noqa: E402
from nn_mcts import make_nn_mcts_agent  # noqa: E402
from distill import generate_ismcts_samples  # noqa: E402
from selfplay import generate_samples  # noqa: E402
from train import load_net, load_net_warmstart, save_net, train  # noqa: E402


def _eval_vs_heuristic(net, meta, decks, device, games_total, sims, dets, rng) -> float:
    """現ネット操縦(NN-MCTS) vs heuristic を**複数デッキ平均**で評価する（汎化を測る）.

    best 選抜が1デッキに偏ると、そのデッキだけ強い net を選び他デッキで弱くなる（実測の症状）。
    games_total をデッキ数で配分して総試合数を概ね一定に保ち、デッキ横断の平均勝率を返す。
    """
    evaluator = make_net_evaluator(net, meta, device)
    heuristic = make_heuristic_agent(meta)
    per = max(4, games_total // len(decks))
    rates = []
    for deck in decks:
        nn_agent = make_nn_mcts_agent(
            meta,
            deck,
            deck,
            evaluator=evaluator,
            n_simulations=sims,
            n_determinizations=dets,
        )
        res = evaluate(nn_agent, heuristic, deck, rng, per)
        rates.append(res["win_rate_a"])
    return sum(rates) / len(rates)


def main() -> None:
    p = argparse.ArgumentParser(description="AlphaZero 反復学習")
    p.add_argument(
        "--deck",
        nargs="+",
        default=["data/deck.csv"],
        help="学習に使うデッキ（複数可。毎試合ランダム選択＝汎用 pilot 化）。"
        "例: --deck models/champion_deck.csv data/sample_deck.csv",
    )
    p.add_argument(
        "--out", default="models/pvnet.pt", help="学習ネットの保存先（追跡外）"
    )
    p.add_argument("--iterations", type=int, default=10, help="反復数")
    p.add_argument("--games", type=int, default=8, help="反復ごとの self-play 試合数")
    p.add_argument("--sims", type=int, default=64, help="1手あたりの MCTS 反復")
    p.add_argument("--dets", type=int, default=2, help="determinization 数")
    p.add_argument("--epochs", type=int, default=2, help="反復ごとの学習 epoch")
    p.add_argument("--batch", type=int, default=32, help="バッチサイズ（勾配累積）")
    p.add_argument(
        "--lr", type=float, default=5e-4, help="学習率（継続学習の安定重視でやや低め）"
    )
    p.add_argument(
        "--best-out",
        default="models/pvnet_best.pt",
        help="eval 最良時のネット保存先（学習が不安定でも最良を失わない・追跡外）",
    )
    p.add_argument("--seed", type=int, default=0, help="乱数シード")
    p.add_argument("--resume", action="store_true", help="既存ネットから続行")
    p.add_argument(
        "--init-from",
        default=None,
        help="初回の初期重み（--out が無い時のみ使う）。例: 蒸留ネットを種に selfplay で"
        "ISMCTS 超えを狙う improve 用 models/pvnet_distill_best.pt",
    )
    p.add_argument(
        "--teacher",
        choices=["selfplay", "ismcts"],
        default="selfplay",
        help="データ収集元。ismcts は ISMCTS 教師を真似る蒸留（崩壊回避・強い土台）",
    )
    p.add_argument(
        "--teacher-time-budget",
        type=float,
        default=0.1,
        help="ismcts 教師の1手あたり秒（蒸留時のみ・大きいほど強い教師だが遅い）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="蒸留サンプル収集の並列プロセス数（CPU コア数推奨・1回の試行数を増やす）",
    )
    p.add_argument(
        "--buffer", type=int, default=20, help="リプレイバッファに保持する直近反復数"
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="N反復ごとに vs heuristic 評価（0=無効）",
    )
    p.add_argument("--eval-games", type=int, default=12, help="評価の試合数")
    p.add_argument(
        "--log", default="models/train_log.csv", help="進捗ログ CSV（追記・追跡外）"
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_card_meta()
    decks = [
        load_deck(p) for p in args.deck
    ]  # 学習用デッキ群（複数なら毎試合ランダム選択）
    rng = random.Random(args.seed)
    if len(decks) > 1:
        print(
            f"学習デッキ {len(decks)} 種（多様化で汎用 pilot 化）。"
            f"eval/best 選抜は全{len(decks)}デッキ平均（汎化を測る）"
        )

    def _load_with_warmstart(path: str, label: str):
        """厳密読み込み→失敗時はウォームスタート（特徴量拡張対応）でネットを得る."""
        try:
            n = load_net(path, device)
            print(f"{label}: {path} を読み込み（device={device}）")
        except RuntimeError:
            n = load_net_warmstart(path, device)
            print(
                f"{label}(ウォームスタート): {path} の旧重みを引き継ぎ（device={device}）"
            )
        return n

    if args.resume and os.path.exists(args.out):
        net = _load_with_warmstart(args.out, "レジューム")
    elif args.init_from and os.path.exists(args.init_from):
        # 初回のみ種ネットから開始（例: 蒸留ネットを種に selfplay で ISMCTS 超えを狙う）
        net = _load_with_warmstart(args.init_from, "初期重み（種）")
    else:
        net = PVNet()
        print(f"新規ネットで開始（device={device}）")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 進捗ログ（追記）。寝ている間の経過を朝に確認できるよう、各反復をディスクに残す。
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    new_log = not os.path.exists(args.log)
    log_file = open(args.log, "a", newline="")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(
            [
                "time",
                "iter",
                "new_samples",
                "buffer",
                "value_loss",
                "policy_loss",
                "eval_winrate",
            ]
        )
        log_file.flush()

    # リプレイバッファ: 直近 args.buffer 反復分の自己対戦サンプルを保持し、まとめて学習する
    buffer: deque[list] = deque(maxlen=args.buffer)
    # eval 最良勝率。学習が不安定でも最良モデルを別ファイルに残す。
    # 1h を細かく回しても積みあがるよう、best の基準値を sidecar に永続化し resume で引き継ぐ
    # （引き継がないと再開のたびに best_wr=-1 にリセットされ、劣化モデルで best を上書きしてしまう）。
    best_meta_path = args.best_out + ".meta.json"
    best_wr = -1.0
    if args.resume and os.path.exists(best_meta_path):
        try:
            with open(best_meta_path) as f:
                best_wr = float(json.load(f).get("best_wr", -1.0))
            print(f"best 基準を引き継ぎ: {best_wr:.3f}（{best_meta_path}）")
        except (ValueError, OSError):
            best_wr = -1.0
    for it in range(args.iterations):
        if args.teacher == "ismcts":
            # 蒸留: ISMCTS 教師の選択を真似る（弱い net からの自己対戦崩壊を回避し強い土台を作る）
            new_samples = generate_ismcts_samples(
                meta,
                decks,
                args.games,
                rng,
                time_budget=args.teacher_time_budget,
                n_workers=args.workers,
            )
        elif args.workers and args.workers > 1:
            # 並列 self-play: 現ネットを一時保存し、各 worker が CPU で読み込んで収集する
            # （NN-MCTS は batch=1 推論＋cgエンジン＝CPU寄り。CPU 並列で試行数をコア数倍に）。
            snap = args.out + ".snap"
            save_net(net, snap)
            from nn_collect import generate_samples_parallel

            new_samples = generate_samples_parallel(
                snap,
                decks,
                args.games,
                rng,
                args.workers,
                n_simulations=args.sims,
                n_determinizations=args.dets,
            )
        else:
            evaluator = make_net_evaluator(net, meta, device)
            new_samples = generate_samples(
                meta,
                decks,
                evaluator,
                args.games,
                rng,
                n_simulations=args.sims,
                n_determinizations=args.dets,
            )
        buffer.append(new_samples)
        train_data = [s for lst in buffer for s in lst]  # バッファ全体で学習
        history = train(
            net,
            train_data,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=device,
        )
        save_net(net, args.out)
        last = (
            history[-1]
            if history
            else {"value_loss": float("nan"), "policy_loss": float("nan")}
        )
        print(
            f"iter {it}: new={len(new_samples)} buffer={len(train_data)} "
            f"value_loss={last['value_loss']:.4f} policy_loss={last['policy_loss']:.4f} "
            f"-> saved {args.out}"
        )
        eval_wr = ""
        if args.eval_every and (it + 1) % args.eval_every == 0:
            wr = _eval_vs_heuristic(
                net, meta, decks, device, args.eval_games, args.sims, args.dets, rng
            )
            eval_wr = f"{wr:.3f}"
            print(
                f"  [eval] iter {it}: NN-MCTS vs heuristic 勝率(全{len(decks)}デッキ平均) = {wr:.3f}"
            )
            # 最良を別ファイルに退避（崩壊しても最良を失わない）。eval はノイズありなので
            # 確定判断は別途 make eval-net（40+試合）で行う前提。
            if wr > best_wr:
                best_wr = wr
                save_net(net, args.best_out)
                # 基準値を永続化（resume で引き継ぎ＝run を跨いで best が単調に上がる）
                with open(best_meta_path, "w") as f:
                    json.dump({"best_wr": best_wr}, f)
                print(f"    -> best 更新（{wr:.3f}）→ {args.best_out} に保存")
        # 進捗を CSV に追記（各反復ごとに flush＝途中で落ちても残る）
        log_writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                it,
                len(new_samples),
                len(train_data),
                f"{last['value_loss']:.4f}",
                f"{last['policy_loss']:.4f}",
                eval_wr,
            ]
        )
        log_file.flush()
    log_file.close()
    print(f"完了（進捗ログ: {args.log}）")


if __name__ == "__main__":
    main()
