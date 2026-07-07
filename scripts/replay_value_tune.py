"""実戦 replay の (state, z) で **value 頭だけ**を fine-tune する（torch・Docker）.

design-decisions §24: 盤面（ベンチ切れ）信号は実戦の相手分布にしか無く、教師対局からは学べない。
本スクリプトは実戦 z（＝薄い盤面で実際に負けた結果）を value に回帰する＝学習で盤面感度を
獲得する唯一の経路。policy_weight=0 で policy/特徴は触らず value 頭のみ動かす（我々の手の
クローンや相手の手の雑音で policy を汚さない）。過学習を避けるため holdout で早期に見切る。

判定は scripts/diagnose_value_board.py（--net-a 元 net / --net-b 出力）で盤面感度 B−A を測る。

    make exec CMD="python scripts/replay_value_tune.py --init models/pvnet_operative.pt \\
        --out models/pvnet_replay.pt --epochs 4"
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import torch  # noqa: E402

from selfplay import Sample  # noqa: E402
from train import load_net, load_net_warmstart, save_net, train  # noqa: E402
from features import ACTION_FEAT_LEN  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="実戦 z で value 頭を fine-tune")
    p.add_argument("--init", default="models/pvnet_operative.pt", help="起点 net")
    p.add_argument("--samples", default="data/replays/value_samples.npz")
    p.add_argument("--out", default="models/pvnet_replay.pt")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--holdout", type=float, default=0.15, help="検証に回す割合")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not os.path.exists(args.samples):
        raise SystemExit(f"サンプルが無い: {args.samples}（先に make replay-extract）")
    d = np.load(args.samples, allow_pickle=True)
    states, zs = d["states"].astype(np.float32), d["z"].astype(np.float32)
    # value のみ学習: ダミー行動(1手)・π=[1]（policy_weight=0 なので勾配に寄与しない）
    dummy_a = np.zeros((1, ACTION_FEAT_LEN), dtype=np.float32)
    dummy_pi = np.ones((1,), dtype=np.float32)
    samples = [Sample(s, dummy_a, dummy_pi, float(z)) for s, z in zip(states, zs)]

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * args.holdout)
    val, tr = samples[:n_val], samples[n_val:]
    print(f"サンプル {len(samples)}（学習 {len(tr)} / 検証 {len(val)}）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        net = load_net(args.init, device)
    except RuntimeError:
        net = load_net_warmstart(args.init, device)
    net.to(device)  # load_net は device へ移さない（開始時 val_loss の cuda/cpu 衝突を防ぐ）

    def val_loss() -> float:
        net.eval()
        with torch.no_grad():
            tot = 0.0
            for s in val:
                v, _ = net(
                    torch.from_numpy(s.state).to(device),
                    torch.from_numpy(s.actions).to(device),
                )
                tot += abs(float(v) - s.z)  # 平均絶対誤差（解釈しやすい）
        return tot / max(1, len(val))

    print(f"開始 検証MAE = {val_loss():.4f}")
    best_mae, best_state = float("inf"), None
    for ep in range(args.epochs):
        hist = train(
            net,
            tr,
            epochs=1,
            batch_size=args.batch,
            lr=args.lr,
            policy_weight=0.0,  # value のみ
            device=device,
        )
        mae = val_loss()
        print(
            f"epoch {ep}: 学習value_loss={hist[-1]['value_loss']:.4f} 検証MAE={mae:.4f}"
        )
        if mae < best_mae:  # 早期見切り（過学習前の最良を保存）
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    save_net(net, args.out)
    print(f"最良（検証MAE {best_mae:.4f}）を保存 → {args.out}")
    print(
        "判定: make exec CMD=\"python scripts/diagnose_value_board.py "
        f"--net-a {args.init} --net-b {args.out}\"  で盤面感度 B−A を確認"
    )


if __name__ == "__main__":
    main()
