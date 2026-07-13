"""指定チームの replay から方策クローン用サンプル (state, actions, π, z) を抽出する（汎用）.

**任意のチームを教師にできる**（§33: 現1位・自分・将来の1位——`--team` を変えるだけ）。
episode JSON には両席の観測と行動が入っているので、教師席の「局面 → 選んだ手」を
one-hot π として収集する＝行動クローンの教師データ。value 用の z（教師席の勝敗）も付す。

- 対象は net が扱う pick-one 決定（MAIN/ATTACK・選択肢2以上）のみ。
- 冪等・追記（処理済み episode を npz に記録・JSON は抽出後に削除可）。
- torch 非依存（ホストで動く）。学習は scripts/teacher_tune.py（Docker）。

    python scripts/extract_teacher_samples.py --team <TeamName>   # make teacher-extract
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from cg.api import SelectType, to_observation_class  # noqa: E402
from features import encode_actions, encode_observation  # noqa: E402

_DECISION_TYPES = (SelectType.MAIN, SelectType.ATTACK)


def team_slug(team: str) -> str:
    """チーム名をファイル名に使える形へ（英数以外は _）."""
    return re.sub(r"[^A-Za-z0-9]+", "_", team).strip("_").lower() or "team"


def _z_for_seat(ep: dict, seat: int) -> float | None:
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


def extract_episode(ep: dict, team: str, meta) -> list[tuple]:
    """教師席の pick-one 決定から (state, actions, one-hot π, z) を集める.

    episode のプロトコル（実データ 225 ファイル×14,602 決定で 100% 整合を確認済み）:
    - 決定 = `status == "ACTIVE"` の step にある select（INACTIVE にも観測が乗るが手番でない）
    - その決定への応答 = **次 step** の同席 action（1ステップ遅れ）
    セルフマッチ（両席が教師）は両席とも収集する（どちらも教師の方策）。
    """
    names = ep.get("info", {}).get("TeamNames") or ["?", "?"]
    steps = ep.get("steps") or []
    out: list[tuple] = []
    for seat in (0, 1):
        if names[seat] != team:
            continue
        z = _z_for_seat(ep, seat)
        if z is None:  # 結果欠損（エラー/タイムアウト）＝ラベル無し
            continue
        for i, step in enumerate(steps):
            rec = step[seat]
            if rec.get("status") != "ACTIVE" or not rec.get("observation"):
                continue
            if i + 1 >= len(steps):
                continue
            act = steps[i + 1][seat].get("action")
            if not isinstance(act, list) or len(act) != 1:
                continue  # pick-one の決定のみ（デッキ提出・複数選択は対象外）
            obs = to_observation_class(rec["observation"])
            st, sel = obs.current, obs.select
            if st is None or st.result != -1 or sel is None:
                continue
            if sel.type not in _DECISION_TYPES or len(sel.option) <= 1:
                continue
            idx = int(act[0])
            if not (0 <= idx < len(sel.option)):
                continue  # 観測と行動の不整合は捨てる（防御）
            pi = np.zeros(len(sel.option), dtype=np.float32)
            pi[idx] = 1.0
            out.append(
                (encode_observation(obs, meta), encode_actions(obs, meta), pi, z)
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="教師チームの方策クローン用サンプル抽出")
    p.add_argument("--team", required=True, help="教師チーム名（TeamNames と一致）")
    p.add_argument("--dir", default="data/replays", help="replay JSON のルート（再帰）")
    p.add_argument("--out", default=None, help="出力 npz（既定 data/replays/teacher_<slug>.npz）")
    args = p.parse_args()
    out = args.out or f"data/replays/teacher_{team_slug(args.team)}.npz"

    states, actions, pis, zs, done = [], [], [], [], set()
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True)
        states, actions, pis, zs = (
            list(d["states"]),
            list(d["actions"]),
            list(d["pis"]),
            list(d["z"]),
        )
        done = set(str(x) for x in d["episodes"])

    meta = load_card_meta()
    n_ep = n_s = 0
    for path in sorted(glob.glob(os.path.join(args.dir, "**", "*.json"), recursive=True)):
        with open(path) as f:
            ep = json.load(f)
        eid = _episode_id(ep, path)
        if eid in done:
            continue
        done.add(eid)
        rows = extract_episode(ep, args.team, meta)
        if not rows:
            continue
        for s, a, pi, z in rows:
            states.append(s.astype(np.float32))
            actions.append(a.astype(np.float32))
            pis.append(pi)
            zs.append(np.float32(z))
        n_ep += 1
        n_s += len(rows)

    if states:
        np.savez_compressed(
            out,
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=object),
            pis=np.asarray(pis, dtype=object),
            z=np.asarray(zs, dtype=np.float32),
            episodes=np.asarray(sorted(done)),
        )
    print(
        f"教師 {args.team}: 新規 {n_ep} episode / {n_s} サンプル → {out}\n"
        f"累計 {len(states)} サンプル（勝ちラベル比率 "
        f"{float(np.mean(zs)) if zs else float('nan'):.3f}）"
    )
    print("学習: make teacher-tune TEACHER_SAMPLES=" + out)


if __name__ == "__main__":
    main()
