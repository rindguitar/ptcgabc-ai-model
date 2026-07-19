"""Kaggle replay（episode JSON）から value 学習サンプル (state, z) を抽出・永続化する.

self-play/distill/特徴/教師あり の4経路すべてで value の盤面感度が動かなかった真因は
「ベンチ切れ＝実戦の相手分布の現象で、教師対局データに信号が無い」こと（decisions.md §24）。
**実戦 replay の z（＝薄い盤面で実際に負けた結果）には信号が入っている**ので、これを value 頭に
回帰すれば、学習で初めて盤面感度を獲得できる可能性がある（唯一データに信号が入る経路）。

抽出: 各 episode の**両席**の各決定局面で、その席視点の観測を encode し、その席の最終結果 z を付す。
（相手席も入れる＝上手い相手が薄い盤面を罰したデータを含める＝信号の本体）。policy は学ばない
（我々の手のクローンは自己蒸留・相手の手は雑音）ため state と z のみ保存する。

**冪等・永続**: data/replays/value_samples.npz に追記し、処理済み episode_id を記録。集計後の JSON は
削除してよい（サンプルが残る）。torch 非依存（features は numpy・cg.api の dataclass 化のみ使う）。

    python scripts/extract_replay_samples.py            # make replay-extract
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from cg.api import SelectType, to_observation_class  # noqa: E402
from features import OBS_FEAT_LEN, encode_observation  # noqa: E402

_DECISION_TYPES = (SelectType.MAIN, SelectType.ATTACK)


def _team_names(ep: dict) -> list[str]:
    return ep.get("info", {}).get("TeamNames") or ["?", "?"]


def _z_for_seat(ep: dict, seat: int) -> float | None:
    """その席の勝敗 z。rewards が欠損（エラー/タイムアウトで結果なし）なら None."""
    r = ep.get("rewards") or [None, None]
    mine, opp = r[seat], r[1 - seat]
    if mine is None or opp is None:
        return None
    if mine == opp:
        return 0.5
    return 1.0 if mine > opp else 0.0


def _episode_id(ep: dict, path: str) -> str:
    return str(
        ep.get("info", {}).get("EpisodeId")
        or os.path.splitext(os.path.basename(path))[0]
    )


def extract_episode(ep: dict, meta) -> list[tuple[np.ndarray, float]]:
    """両席・全決定局面から (state, z) を集める（決定局面のみ＝情報のある盤面）."""
    out: list[tuple[np.ndarray, float]] = []
    steps = ep.get("steps") or []
    for seat in (0, 1):
        z = _z_for_seat(ep, seat)
        if z is None:  # 結果欠損（エラー/タイムアウト）＝value ラベル無し→この席はスキップ
            continue
        for step in steps:
            obsd = step[seat].get("observation")
            if not obsd:
                continue
            obs = to_observation_class(obsd)
            st = obs.current
            if st is None or st.result != -1 or obs.select is None:
                continue
            if obs.select.type not in _DECISION_TYPES or len(obs.select.option) <= 1:
                continue
            out.append((encode_observation(obs, meta), z))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="replay から value 学習サンプルを抽出")
    p.add_argument("--dir", default="data/replays", help="replay JSON のルート")
    p.add_argument("--out", default="data/replays/value_samples.npz")
    args = p.parse_args()

    # 既存サンプル＋処理済み episode を読み込む（冪等）
    states, zs, done = [], [], set()
    if os.path.exists(args.out):
        d = np.load(args.out, allow_pickle=True)
        states, zs = list(d["states"]), list(d["z"])
        done = set(str(x) for x in d["episodes"])

    meta = load_card_meta()
    paths = sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True))
    n_new_ep = n_new_s = 0
    for path in paths:
        with open(path) as f:
            ep = json.load(f)
        eid = _episode_id(ep, path)
        if eid in done:
            continue
        # セルフ検証マッチ（相手＝自分）は情報ゼロなので除外
        names = _team_names(ep)
        if names[0] == names[1]:
            done.add(eid)
            continue
        pairs = extract_episode(ep, meta)
        for s, z in pairs:
            states.append(s.astype(np.float32))
            zs.append(np.float32(z))
        done.add(eid)
        n_new_ep += 1
        n_new_s += len(pairs)

    if states:
        np.savez_compressed(
            args.out,
            states=np.asarray(states, dtype=np.float32),
            z=np.asarray(zs, dtype=np.float32),
            episodes=np.asarray(sorted(done)),
        )
    z_arr = np.asarray(zs, dtype=np.float32)
    print(
        f"新規 {n_new_ep} episode / {n_new_s} サンプルを追加 → {args.out}\n"
        f"累計: {len(states)} サンプル（次元 {OBS_FEAT_LEN}）・勝ちラベル比率 "
        f"{float(z_arr.mean()):.3f}・処理済み episode {len(done)}"
    )
    print(
        "※ JSON の削除は teacher-extract も済ませてから（行動クローンは生 JSON が必要）"
    )


if __name__ == "__main__":
    main()
