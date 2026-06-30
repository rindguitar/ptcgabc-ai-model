"""value/policy ネット＋NN評価器のテスト（torch 依存・Docker で実行）.

torch はホスト venv に無いため、ホストの `make test` ではスキップされる。
Docker で実行: make exec CMD="python -m pytest tests/test_net.py"
"""

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

from agents import make_heuristic_agent  # noqa: E402
from cards import load_card_meta  # noqa: E402
from cg.api import SelectType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from deck import load_deck  # noqa: E402
from features import ACTION_FEAT_LEN, OBS_FEAT_LEN  # noqa: E402
from net import PVNet  # noqa: E402
from nn_eval import make_net_evaluator  # noqa: E402
from nn_mcts import make_nn_mcts_agent  # noqa: E402
from train import load_net_warmstart  # noqa: E402


@pytest.fixture(scope="module")
def meta():
    return load_card_meta()


def _advance_to_main(deck, rng, meta):
    heuristic = make_heuristic_agent(meta)
    obs_dict, _ = battle_start(deck, deck)
    for _ in range(2000):
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            return None
        if obs.select is None:
            return None
        if (
            obs.select.type == SelectType.MAIN
            and obs.current.turn >= 3
            and len(obs.select.option) > 1
        ):
            return obs
        obs_dict = battle_select(heuristic(obs, rng))
    return None


def test_net_forward_shapes():
    """PVNet が value[0,1] と logits(n,) を返す."""
    net = PVNet()
    state = torch.randn(OBS_FEAT_LEN)
    n = 7
    actions = torch.randn(n, ACTION_FEAT_LEN)
    value, logits = net(state, actions)
    assert value.ndim == 0
    assert 0.0 <= float(value) <= 1.0
    assert logits.shape == (n,)


def test_net_evaluator(meta):
    """評価器が value[0,1] と option 整列の priors を返す."""
    deck = load_deck(DECK)
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        evaluator = make_net_evaluator(PVNet(), meta)
        value, priors = evaluator(obs)
        assert 0.0 <= value <= 1.0
        assert len(priors) == len(obs.select.option)
        assert abs(sum(priors) - 1.0) < 1e-5
    finally:
        battle_finish()


def test_warmstart_roundtrips_same_arch(tmp_path):
    """同一アーキの保存→load_net_warmstart で全重みが一致し forward が通る.

    NN v2 で cardId Embedding を入れアーキを刷新したため、旧「末尾列を 0 埋め」型の
    warm-start 契約は廃止（再学習前提）。ここでは同一アーキの round-trip 健全性を確認する。
    """
    net = PVNet()
    path = tmp_path / "n.pt"
    torch.save(net.state_dict(), path)

    loaded = load_net_warmstart(str(path))
    for k, v in net.state_dict().items():
        assert torch.allclose(loaded.state_dict()[k], v), k
    value, logits = loaded(torch.randn(OBS_FEAT_LEN), torch.randn(5, ACTION_FEAT_LEN))
    assert 0.0 <= float(value) <= 1.0 and logits.shape == (5,)


def test_nn_mcts_with_net(meta):
    """NN 評価器を載せた PUCT が決定点で合法手を返す."""
    deck = load_deck(DECK)
    evaluator = make_net_evaluator(PVNet(), meta)
    agent = make_nn_mcts_agent(
        meta, deck, deck, evaluator=evaluator, n_simulations=16, n_determinizations=2
    )
    obs = _advance_to_main(deck, random.Random(0), meta)
    try:
        assert obs is not None
        action = agent(obs, random.Random(0))
        sel = obs.select
        assert sel.minCount <= len(action) <= sel.maxCount
        assert all(0 <= i < len(sel.option) for i in action)
    finally:
        battle_finish()
