"""PVNet の学習（Phase 3・torch・Docker）.

self-play サンプル (state, actions, π, z) から value(BCE) + policy(cross-entropy) を最小化する。
合法手数 n が可変なのでサンプル単位で損失を計算し、batch ごとに勾配ステップ（勾配累積）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from net import PVNet


def save_net(net: PVNet, path: str) -> None:
    """ネットの重みを保存（追跡外の models/ 配下想定）."""
    torch.save(net.state_dict(), path)


def load_net(path: str, device: str = "cpu") -> PVNet:
    """保存済みネットを読み込む."""
    net = PVNet()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net


def train(
    net: PVNet,
    samples: list,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    value_weight: float = 1.0,
    device: str = "cpu",
) -> list[dict]:
    """samples で PVNet を学習し、epoch ごとの損失履歴を返す."""
    if not samples:
        return []
    dev = torch.device(device)
    net.to(dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        order = torch.randperm(len(samples)).tolist()
        sum_v, sum_p, count = 0.0, 0.0, 0
        opt.zero_grad()
        for bi, si in enumerate(order):
            s = samples[si]
            state = torch.from_numpy(s.state).to(dev)
            actions = torch.from_numpy(s.actions).to(dev)
            pi = torch.from_numpy(s.pi).to(dev)

            value, logits = net(state, actions)
            v_loss = F.binary_cross_entropy(value, torch.tensor(float(s.z), device=dev))
            p_loss = -(pi * F.log_softmax(logits, dim=-1)).sum()
            loss = value_weight * v_loss + p_loss
            (loss / batch_size).backward()

            sum_v += float(v_loss)
            sum_p += float(p_loss)
            count += 1
            if (bi + 1) % batch_size == 0:
                opt.step()
                opt.zero_grad()
        opt.step()  # 端数バッチ
        opt.zero_grad()
        history.append(
            {"epoch": epoch, "value_loss": sum_v / count, "policy_loss": sum_p / count}
        )
    return history
