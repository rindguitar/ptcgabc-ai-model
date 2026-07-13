"""教師チームの (局面→選択) で policy を fine-tune する（行動クローン・torch・Docker）.

§33: 上位チーム（例: 現1位）の実戦決定を教師にした行動クローン。教師は §25 の
ISMCTS 蒸留より強い可能性がある（1位は軽量方策で勝率 0.7 超＝方策の質の実証）。
既定は **policy のみ**（value_weight 0）: operative の value は我々の資産で、
§25 で value fine-tune は強さを崩した実測があるため触らない。

判定は外部 A/B（§30 の作法）:
    make eval-net EVAL_NET=models/pvnet_teacher.pt   # vs 実メタ（教師デッキで）
    make eval-net EVAL_NET=models/pvnet_operative.pt # 同条件比較

    make teacher-tune TEACHER_SAMPLES=data/replays/teacher_<slug>.npz
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


def main() -> None:
    p = argparse.ArgumentParser(description="教師の実戦決定で policy を行動クローン")
    p.add_argument("--samples", required=True, help="extract_teacher_samples の npz")
    p.add_argument("--init", default="models/pvnet_operative.pt", help="起点 net")
    p.add_argument("--out", default="models/pvnet_teacher.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="低 LR（小データの過学習対策）")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument(
        "--value-weight",
        type=float,
        default=0.0,
        help="z への value 損失の重み（既定 0＝policy のみ。§25 の教訓）",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = np.load(args.samples, allow_pickle=True)
    samples = [
        Sample(s, a, pi, float(z))
        for s, a, pi, z in zip(d["states"], d["actions"], d["pis"], d["z"])
    ]
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * args.holdout))
    val, tr = samples[:n_val], samples[n_val:]
    print(f"サンプル {len(samples)}（学習 {len(tr)} / 検証 {len(val)}）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        net = load_net(args.init, device)
    except RuntimeError:
        net = load_net_warmstart(args.init, device)
    net.to(device)

    def top1_acc() -> float:
        """検証集合で「教師と同じ手を最尤にできた率」＝クローンの直接指標."""
        net.eval()
        hit = 0
        with torch.no_grad():
            for s in val:
                _, logits = net(
                    torch.from_numpy(np.asarray(s.state)).to(device),
                    torch.from_numpy(np.asarray(s.actions)).to(device),
                )
                if int(logits.argmax()) == int(np.asarray(s.pi).argmax()):
                    hit += 1
        return hit / len(val)

    print(f"開始 検証 top-1 一致率 = {top1_acc():.3f}")
    best_acc, best_state = -1.0, None
    for ep in range(args.epochs):
        hist = train(
            net,
            tr,
            epochs=1,
            batch_size=args.batch,
            lr=args.lr,
            value_weight=args.value_weight,
            policy_weight=1.0,
            device=device,
        )
        acc = top1_acc()
        print(f"epoch {ep}: policy_loss={hist[-1]['policy_loss']:.4f} 検証一致率={acc:.3f}")
        if acc > best_acc:  # 過学習前の最良を保存
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    save_net(net, args.out)
    print(f"最良（検証一致率 {best_acc:.3f}）を保存 → {args.out}")
    print(
        "判定（外部 A/B・教師デッキで）: make eval-net EVAL_NET=" + args.out + " と "
        "EVAL_NET=models/pvnet_operative.pt を同条件比較（§30 の作法）"
    )


if __name__ == "__main__":
    main()
