"""AlphaZero 反復学習ドライバ（Phase 3・torch・Docker）.

{現ネットで self-play → 学習 → 保存} を反復し、PVNet（policy/value）を強くする。
torch が要るので Docker で実行する:
    make train
    make exec CMD="python scripts/train_alphazero.py --iterations 20 --games 16"

出力ネット models/pvnet.pt は追跡外。--resume で続きから。
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch  # noqa: E402

from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from net import PVNet  # noqa: E402
from nn_eval import make_net_evaluator  # noqa: E402
from selfplay import generate_samples  # noqa: E402
from train import load_net, save_net, train  # noqa: E402


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
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_card_meta()
    deck = load_deck(args.deck)
    rng = random.Random(args.seed)

    if args.resume and os.path.exists(args.out):
        net = load_net(args.out, device)
        print(f"レジューム: {args.out} を読み込み（device={device}）")
    else:
        net = PVNet()
        print(f"新規ネットで開始（device={device}）")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for it in range(args.iterations):
        evaluator = make_net_evaluator(net, meta, device)
        samples = generate_samples(
            meta,
            deck,
            evaluator,
            args.games,
            rng,
            n_simulations=args.sims,
            n_determinizations=args.dets,
        )
        history = train(
            net,
            samples,
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
            f"iter {it}: samples={len(samples)} "
            f"value_loss={last['value_loss']:.4f} policy_loss={last['policy_loss']:.4f} "
            f"-> saved {args.out}"
        )
    print("完了")


if __name__ == "__main__":
    main()
