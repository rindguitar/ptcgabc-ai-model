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
from selfplay import generate_samples  # noqa: E402
from train import load_net, load_net_warmstart, save_net, train  # noqa: E402


def _eval_vs_heuristic(net, meta, deck, device, games, sims, dets, rng) -> float:
    """現ネット操縦(NN-MCTS) vs heuristic の勝率（進捗確認用）."""
    evaluator = make_net_evaluator(net, meta, device)
    nn_agent = make_nn_mcts_agent(
        meta,
        deck,
        deck,
        evaluator=evaluator,
        n_simulations=sims,
        n_determinizations=dets,
    )
    res = evaluate(nn_agent, make_heuristic_agent(meta), deck, rng, games)
    return res["win_rate_a"]


def main() -> None:
    p = argparse.ArgumentParser(description="AlphaZero 反復学習")
    p.add_argument("--deck", default="data/deck.csv", help="self-play に使うデッキ")
    p.add_argument(
        "--out", default="models/pvnet.pt", help="学習ネットの保存先（追跡外）"
    )
    p.add_argument("--iterations", type=int, default=10, help="反復数")
    p.add_argument("--games", type=int, default=8, help="反復ごとの self-play 試合数")
    p.add_argument("--sims", type=int, default=64, help="1手あたりの MCTS 反復")
    p.add_argument("--dets", type=int, default=2, help="determinization 数")
    p.add_argument("--epochs", type=int, default=2, help="反復ごとの学習 epoch")
    p.add_argument("--batch", type=int, default=32, help="バッチサイズ（勾配累積）")
    p.add_argument("--lr", type=float, default=1e-3, help="学習率")
    p.add_argument("--seed", type=int, default=0, help="乱数シード")
    p.add_argument("--resume", action="store_true", help="既存ネットから続行")
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
    deck = load_deck(args.deck)
    rng = random.Random(args.seed)

    if args.resume and os.path.exists(args.out):
        try:
            net = load_net(args.out, device)
            print(f"レジューム: {args.out} を読み込み（device={device}）")
        except RuntimeError:
            # 特徴量を拡張した等で形が変わった場合は旧重みを引き継ぐ（末尾追加前提）
            net = load_net_warmstart(args.out, device)
            print(
                f"レジューム(ウォームスタート): {args.out} の旧重みを引き継ぎ（device={device}）"
            )
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
    for it in range(args.iterations):
        evaluator = make_net_evaluator(net, meta, device)
        new_samples = generate_samples(
            meta,
            deck,
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
                net, meta, deck, device, args.eval_games, args.sims, args.dets, rng
            )
            eval_wr = f"{wr:.3f}"
            print(f"  [eval] iter {it}: NN-MCTS vs heuristic 勝率 = {wr:.3f}")
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
