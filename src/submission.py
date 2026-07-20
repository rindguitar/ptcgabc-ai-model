"""Kaggle 提出形式へのアダプタ.

我々の方策 `(obs, rng) -> list[int]` を、公式の `agent(obs_dict) -> list[int]` に包む。
初手（obs.select is None）はデッキ60枚（カードID）を返し、以降は方策で選ぶ。

提出物としても、公式ビュアー（kaggle_environments の env.render(html)）でチャンピオンの
動きを観るためにも、この形式が必要。本大会の1試合600秒クロックに対応するため、ISMCTS は
game_budget（既定 540 = 安全マージン）で動かす。

相手デッキ opp_deck は実戦では未知のため、既定では自分のデッキを仮置きする
（determinize の相手隠れ札サンプリング用の事前分布。観測からの推定は今後の改良余地）。
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable

from cards import load_card_meta
from cg.api import to_observation_class

DECK_SIZE = 60


def _resolve_path(path: str) -> str:
    """ローカルに無ければ Kaggle 提出パス(/kaggle_simulations/agent/)にフォールバック."""
    if not os.path.exists(path):
        alt = "/kaggle_simulations/agent/" + os.path.basename(path)
        if os.path.exists(alt):
            return alt
    return path


def _read_deck_csv(path: str) -> list[int]:
    """deck CSV を読む。ローカルに無ければ Kaggle 提出パスにフォールバック."""
    path = _resolve_path(path)
    with open(path) as f:
        ids = [int(line) for line in f.read().splitlines() if line.strip()]
    if len(ids) != DECK_SIZE:
        raise ValueError(f"デッキは {DECK_SIZE} 枚必要（実際 {len(ids)} 枚）")
    return ids


def _load_fetch_priors(path: str) -> dict[int, float] | None:
    """同梱の fetch_priors.json（§47・山札サーチの取得優先度）を読む。無ければ None＝従来挙動.

    形式: {"priors": {cardId: 取得率}}（mine_fetch_priorities.py の出力）または
    {cardId: 取得率} の素の辞書。キーは int に正規化して返す。
    """
    import json

    path = _resolve_path(path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    priors = d.get("priors", d) if isinstance(d, dict) else {}
    return {int(k): float(v) for k, v in priors.items()} or None


def _load_opp_pool(dir_path: str) -> list[list[int]]:
    """相手候補デッキ群（60枚 CSV）を dir から読む。無ければ /kaggle 提出パスにフォールバック."""
    import glob

    if not os.path.isdir(dir_path):
        alt = "/kaggle_simulations/agent/" + os.path.basename(dir_path)
        if os.path.isdir(alt):
            dir_path = alt
    pool = []
    for p in sorted(glob.glob(os.path.join(dir_path, "*.csv"))):
        try:
            d = [int(x) for x in open(p).read().splitlines() if x.strip()]
        except ValueError:
            continue
        if len(d) == DECK_SIZE:
            pool.append(d)
    return pool


def make_kaggle_agent(
    policy: str = "ismcts",
    deck: list[int] | None = None,
    deck_path: str = "deck.csv",
    opp_deck: list[int] | None = None,
    opp_pool_dir: str | None = None,
    meta=None,
    seed: int = 0,
    game_budget: float | None = 540.0,
    time_budget: float = 0.5,
    fetch_priors_path: str = "fetch_priors.json",
    **policy_kwargs,
) -> Callable[[dict], list[int]]:
    """公式形式 `agent(obs_dict) -> list[int]` を返す.

    Args:
        policy: "ismcts" / "nn"（floored NN-MCTS・要 torch＋net_path）/ "heuristic" / "random"。
        deck: 自分のデッキ（None なら deck_path から読む）。
        opp_deck: 相手デッキの事前分布（None なら deck を流用）。
        opp_pool_dir: 相手候補デッキ群のディレクトリ（指定すると観測整合で相手デッキを推定）。
        game_budget/time_budget: ISMCTS の時間管理（game_budget 優先）。
        fetch_priors_path: 山札サーチ優先度 JSON（§47・同梱時のみ有効・無ければ従来挙動）。
    """
    meta = meta or load_card_meta()
    deck = list(deck) if deck is not None else _read_deck_csv(deck_path)
    opp_deck = list(opp_deck) if opp_deck is not None else list(deck)
    opp_pool = _load_opp_pool(opp_pool_dir) if opp_pool_dir else None
    fetch_priors = _load_fetch_priors(fetch_priors_path)
    rng = random.Random(seed)

    if policy == "heuristic":
        from agents import make_heuristic_agent

        inner = make_heuristic_agent(meta, fetch_priors=fetch_priors)
    elif policy == "random":
        from agents import random_agent

        inner = random_agent
    elif policy == "ismcts":
        from ismcts import make_ismcts_agent

        inner = make_ismcts_agent(
            meta,
            deck,
            opp_deck,
            time_budget=time_budget,
            game_budget=game_budget,
            opp_pool=opp_pool,
            fetch_priors=fetch_priors,
            **policy_kwargs,
        )
    elif policy == "nn":
        # floored NN-MCTS（PUCT＋接地 floor＝pilot≥heuristic 保証）。torch は nn 選択時のみ
        # import（ismcts/heuristic 提出では torch 不要のまま）。batch=1 推論は CPU が速く
        # 提出環境の GPU 有無にも依存しないため、常に CPU で読む。
        from agents import make_heuristic_agent
        from nn_eval import make_net_evaluator
        from nn_mcts import make_nn_mcts_agent
        from train import load_net

        net_path = _resolve_path(policy_kwargs.pop("net_path", "pvnet.pt"))
        net = load_net(net_path, "cpu")
        net.eval()
        evaluator = make_net_evaluator(net, meta, "cpu")
        # 盤面補正（board-blind の即効処置・注入テストで α=0.2 が最良 +0.075）
        board_bonus = policy_kwargs.pop("board_bonus", 0.0)
        if board_bonus:
            from nn_eval import wrap_board_bonus

            evaluator = wrap_board_bonus(evaluator, board_bonus)
        # KO 脅威の対称差注入（§48・エネ充足度で重み付けした受け/詰めの事前信号）
        threat_bonus = policy_kwargs.pop("threat_bonus", 0.0)
        if threat_bonus:
            from nn_eval import wrap_threat_bonus

            evaluator = wrap_threat_bonus(evaluator, meta, threat_bonus)
        inner_nn = make_nn_mcts_agent(
            meta,
            deck,
            opp_deck,
            evaluator=evaluator,
            n_simulations=policy_kwargs.pop("n_simulations", 64),
            n_determinizations=policy_kwargs.pop("n_determinizations", 2),
            floor_rollouts=policy_kwargs.pop("floor_rollouts", 8),
            # 適応 sims（§31）: 残り予算から今手の sims を配分（床=64・天井=8倍）。
            # 実測で NN 提出は 600 秒中 ~550 秒を残していた＝未使用の推論計算資源を使う。
            game_budget=game_budget,
            opp_pool=opp_pool,
            fetch_priors=fetch_priors,
            **policy_kwargs,
        )
        # 時間ガード: NN-MCTS は sims 固定で ISMCTS のような自前の時計を持たない。
        # 累積消費が game_budget を超えたら heuristic（瞬時）へ退避し、600秒クロック
        # 超過（時間切れ負け）を防ぐ。通常は 1手 ~0.1秒 × 数百手 << 540秒で発動しない保険。
        heuristic_fb = make_heuristic_agent(meta, fetch_priors=fetch_priors)
        spent = [0.0]

        def inner(obs, inner_rng):
            if game_budget is not None and spent[0] >= game_budget:
                return heuristic_fb(obs, inner_rng)
            t0 = time.perf_counter()
            act = inner_nn(obs, inner_rng)
            spent[0] += time.perf_counter() - t0
            return act
    else:
        raise ValueError(f"未知の policy: {policy}")

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)  # 初手: デッキ60枚を返す
        return inner(obs, rng)

    return agent
