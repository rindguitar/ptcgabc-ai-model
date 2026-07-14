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
    """保存済みネットを読み込む（厳密一致）."""
    net = PVNet()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return net


def load_net_warmstart(path: str, device: str = "cpu") -> PVNet:
    """特徴量を末尾追加で拡張した新アーキへ、旧重みを引き継いで読み込む（ウォームスタート）.

    入力に接する層（trunk[0] / policy_head[0]）だけ形が変わる。旧重みを**新しい大きい重みの
    先頭スロットへコピー**し、**増えた入力列は 0** にする。こうすると新特徴は初期状態で出力に
    影響せず、学習開始時の挙動は旧モデルと一致する（iter120 等の進捗を破棄しない）。

    前提: 新特徴は obs/action ベクトルの**末尾**に追加されていること（先頭の並びが不変）。
    形が完全一致する層はそのままコピー。新アーキで縮んだ次元には未対応（拡張専用）。
    """
    net = PVNet()
    old = torch.load(path, map_location=device, weights_only=True)
    new_sd = net.state_dict()
    merged = {}
    for k, new_w in new_sd.items():
        ow = old.get(k)
        if ow is None or ow.shape == new_w.shape:
            merged[k] = ow if ow is not None else new_w
            continue
        # 入力次元が拡大した層: 0 埋めの新テンソルに旧重みを先頭スロットへ複写
        w = torch.zeros_like(new_w)
        slices = tuple(slice(0, min(o, n)) for o, n in zip(ow.shape, new_w.shape))
        w[slices] = ow[slices]
        merged[k] = w
    net.load_state_dict(merged)
    return net


def train(
    net: PVNet,
    samples: list,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    device: str = "cpu",
) -> list[dict]:
    """samples で PVNet を学習し、epoch ごとの損失履歴を返す.

    policy_weight=0 で **value のみ学習**（実戦 replay の z を value 頭に回帰する用途。
    replay のダミー行動で policy を汚さない）。
    """
    # 流れ: 1. epoch ごとにサンプルをシャッフル → 2. 1件ずつ forward し
    # value(BCE)+policy(交差エントロピー) の加重和を勾配蓄積 → 3. batch_size 件ごとに step
    # （行動数が可変で通常のバッチ化ができないため、勾配蓄積で擬似バッチにする）。
    if not samples:
        return []
    dev = torch.device(device)
    net.to(dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        # 1. シャッフル（試合内の相関を崩す）
        order = torch.randperm(len(samples)).tolist()
        sum_v, sum_p, count = 0.0, 0.0, 0
        opt.zero_grad()
        for bi, si in enumerate(order):
            s = samples[si]
            state = torch.from_numpy(s.state).to(dev)
            actions = torch.from_numpy(s.actions).to(dev)
            pi = torch.from_numpy(s.pi).to(dev)

            # 2. 損失 = value_weight×BCE(value, z) + policy_weight×CE(π, logits) を勾配蓄積
            value, logits = net(state, actions)
            v_loss = F.binary_cross_entropy(value, torch.tensor(float(s.z), device=dev))
            p_loss = -(pi * F.log_softmax(logits, dim=-1)).sum()
            loss = value_weight * v_loss + policy_weight * p_loss
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
