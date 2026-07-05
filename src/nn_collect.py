"""NN 評価器を使う self-play サンプルの並列収集（torch・CPU 推論をプロセス分散）.

NN-MCTS self-play の主コストは (a) cg エンジンの前進シミュレーション (b) batch=1 の小さな
ネット推論。どちらも CPU 寄りで（GPU の batch=1 推論はむしろ転送オーバヘッドで遅い）、各試合は
独立なので ProcessPoolExecutor で CPU コアに分散すると「1回あたりの試行数」がほぼコア数倍に増える。

各 worker は **CPU 上にネットを読み込み**評価器を作る（1 枚の GPU を奪い合わない）。親プロセスが
CUDA を初期化済みでも安全なよう、**spawn** コンテキストで起動する（fork+CUDA のハングを回避）。
torch 依存のため、torch 非依存の selfplay.py 本体とは分離している。
"""

from __future__ import annotations

import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor

from selfplay import Sample, as_deck_list, play_selfplay_game

# worker プロセスごとの状態（spawn 後に _init で構築）
_W: dict = {}


def _init(
    net_path: str, pool: list[list[int]], game_kwargs: dict, board_bonus: float = 0.0
) -> None:
    """worker 初期化: meta とネット(CPU)・評価器・デッキプールを構築する.

    board_bonus>0 なら評価器に盤面補正を注入＝**盤面を読む探索**の訪問分布を収集し、
    net に蒸留する（AlphaZero の償却）。value は明示特徴（v2.3）＋勝敗ラベルから回帰する。
    """
    from cards import load_card_meta
    from nn_eval import make_net_evaluator, wrap_board_bonus
    from train import load_net

    meta = load_card_meta()
    net = load_net(net_path, "cpu")
    net.eval()
    evaluator = make_net_evaluator(net, meta, "cpu")
    if board_bonus:
        evaluator = wrap_board_bonus(evaluator, board_bonus)
    _W["meta"] = meta
    _W["evaluator"] = evaluator
    _W["pool"] = pool
    _W["kwargs"] = game_kwargs


def _game(seed: int) -> list[Sample]:
    """worker: 与えられた seed でデッキを選び1試合の self-play サンプルを返す."""
    rng = random.Random(seed)
    deck = rng.choice(_W["pool"])
    return play_selfplay_game(_W["meta"], deck, _W["evaluator"], rng, **_W["kwargs"])


def generate_samples_parallel(
    net_path: str,
    decks,
    n_games: int,
    rng: random.Random,
    n_workers: int,
    board_bonus: float = 0.0,
    **game_kwargs,
) -> list[Sample]:
    """net_path のネットを CPU で読み込み、n_games の self-play を並列収集する.

    net_path: 現ネットを保存したファイル（各 worker が CPU で読み込む）。
    board_bonus: 収集評価器への盤面補正 α（0 で無効・_init 参照）。
    game_kwargs: play_selfplay_game へ渡す（n_simulations / n_determinizations 等）。
    """
    pool = as_deck_list(decks)
    seeds = [rng.randrange(2**31) for _ in range(n_games)]
    samples: list[Sample] = []
    ctx = mp.get_context("spawn")  # fork+CUDA のハング回避
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init,
        initargs=(net_path, pool, game_kwargs, board_bonus),
    ) as ex:
        for game_samples in ex.map(_game, seeds):
            samples.extend(game_samples)
    return samples
