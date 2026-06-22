"""学習ループのテスト（torch 依存・Docker で実行）.

Docker で実行: make exec CMD="python -m pytest tests/test_train.py"
"""

import math
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DECK = os.path.join(ROOT, "data", "deck.csv")
sys.path.insert(0, SRC)

torch = pytest.importorskip("torch", reason="torch は Docker のみ")
pytest.importorskip("cg.sim", reason="cabt Engine (cg) が見つからない")
if not os.path.exists(DECK):
    pytest.skip("data/deck.csv が無い", allow_module_level=True)

from cards import load_card_meta  # noqa: E402
from deck import load_deck  # noqa: E402
from net import PVNet  # noqa: E402
from selfplay import generate_samples  # noqa: E402
from train import load_net, save_net, train  # noqa: E402


def test_train_runs_and_updates(tmp_path):
    """self-play サンプルで学習が回り、損失が有限・重みが更新される."""
    meta = load_card_meta()
    deck = load_deck(DECK)
    samples = generate_samples(
        meta,
        deck,
        None,
        n_games=1,
        rng=random.Random(0),
        n_simulations=8,
        n_determinizations=1,
    )
    assert len(samples) > 0

    net = PVNet()
    before = [p.detach().clone() for p in net.parameters()]
    history = train(net, samples, epochs=2, batch_size=8, lr=1e-2)

    assert len(history) == 2
    for h in history:
        assert math.isfinite(h["value_loss"])
        assert math.isfinite(h["policy_loss"])
    changed = any(not torch.equal(b, a) for b, a in zip(before, net.parameters()))
    assert changed

    # 保存・読み込みの往復
    path = str(tmp_path / "net.pt")
    save_net(net, path)
    loaded = load_net(path)
    for a, b in zip(net.parameters(), loaded.parameters()):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
