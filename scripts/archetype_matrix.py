"""型（アーキタイプ）マッチアップ行列 — 上位帯の第三者対戦から型相性を実測する（Layer 1）.

背景（§61/§62）: ローカル eval-deck は共適応バイアスで型選定に使えず、提出枠 A/B は高価。
一方 others/ の replay は**他人同士の試合**＝我々の操縦バイアスが構造的にゼロの実測データ。
これを型ファミリー単位に集約し、「今の帯で構造的に勝っている型」を操縦交絡なしで測る。

流れ: 1. opp_decks/*.csv を構成特徴（ポケ/エネ/たね/ユニーク枚数）ベクトルへ変換
     → 2. k-means（seed 固定・標準化済み特徴）で型ファミリーへクラスタリング
     → 3. others/ の各 episode から両席のデッキ hash と勝者を抽出（_deck_of 再利用・§45）
     → 4. ファミリー×ファミリーの勝率行列と総合強さ（ミラー除外）を集計
     → 5. レポート表示＋ data/archetype_matrix.json へ永続化（追跡外）

    python scripts/archetype_matrix.py
    python scripts/archetype_matrix.py --k 8 --min-episode 87000000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from analyze_replays import _deck_of  # noqa: E402  正実装の再利用（§45）
from cards import load_card_meta  # noqa: E402
from cg.api import CardType  # noqa: E402

DECK_SIZE = 60
# 表示用の既知 hash 注記（数値・カテゴリのみ＝Pokémon Elements なし）
KNOWN = {
    "65c6b47e": "自デッキ（遅滞）",
    "9e3ece3f": "天敵（消耗型）",
    "04621784": "持続カウンター",
    "a4066acd": "旧王者",
    "4041254f": "現王者",
    "965525fe": "王者キラー（§62棄却）",
    "a8c57d4b": "旧自デッキ（攻撃型）",
}


def deck_hash(ids: list[int]) -> str:
    """analyze_replays._save_deck と同一のデッキ内容ハッシュ."""
    return hashlib.md5(",".join(map(str, sorted(ids))).encode()).hexdigest()[:8]


def deck_features(ids: list[int], meta) -> list[float]:
    """デッキ 60 枚 → 構成特徴 [ポケモン, エネ, たね, ユニーク枚数].

    トレーナー枚数は 60-ポケ-エネ で線形従属のため特徴に入れない。
    """
    poke = sum(1 for i in ids if meta.card_type.get(i) == CardType.POKEMON)
    ene = sum(1 for i in ids if meta.card_type.get(i) == CardType.BASIC_ENERGY)
    basic = sum(1 for i in ids if meta.is_basic_pokemon(i))
    return [float(poke), float(ene), float(basic), float(len(set(ids)))]


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """最小実装の k-means（seed 固定＝決定的）。ラベル配列を返す.

    1. 標準化済み入力を仮定 → 2. ランダムな k 点で初期化 → 3. 割当と重心更新を反復
    → 4. 空クラスタは最遠点で再初期化（クラスタ数を必ず k に保つ）。
    """
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if (new_labels == labels).all() and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            member = x[labels == c]
            if len(member):
                centers[c] = member.mean(axis=0)
            else:  # 空クラスタ → 現重心から最遠の点で再初期化
                centers[c] = x[dist.min(axis=1).argmax()]
    return labels


def cluster_decks(
    decks: dict[str, list[int]], meta, k: int, seed: int = 0
) -> tuple[dict[str, int], np.ndarray]:
    """hash→クラスタ id の割当と、表示用の生スケール重心を返す."""
    hashes = sorted(decks)
    feats = np.array([deck_features(decks[h], meta) for h in hashes])
    # 標準化（列ごとに平均0/分散1）してからクラスタリング
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    labels = kmeans((feats - feats.mean(axis=0)) / std, k=k, seed=seed)
    assign = {h: int(c) for h, c in zip(hashes, labels)}
    centroids = np.array(
        [feats[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(4)
         for c in range(k)]
    )
    return assign, centroids


def episode_matchup(ep: dict) -> tuple[str, str, int] | None:
    """episode → (席0のデッキhash, 席1のデッキhash, 勝者席). 抽出不能・引分は None."""
    d0, d1 = _deck_of(ep, 0), _deck_of(ep, 1)
    rewards = ep.get("rewards") or [None, None]
    if d0 is None or d1 is None or rewards[0] is None or rewards[1] is None:
        return None
    if rewards[0] == rewards[1]:  # 引分（レアケース）は勝敗集計から除外
        return None
    return deck_hash(d0), deck_hash(d1), 0 if rewards[0] > rewards[1] else 1


def build_matrix(
    matchups: list[tuple[str, str, int]], assign: dict[str, int], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """ファミリー×ファミリーの (勝ち数, 試合数) 行列を作る（未知 hash の試合は捨てる）."""
    wins = np.zeros((k, k))
    games = np.zeros((k, k))
    for h0, h1, winner in matchups:
        if h0 not in assign or h1 not in assign:
            continue
        c0, c1 = assign[h0], assign[h1]
        games[c0, c1] += 1
        games[c1, c0] += 1
        w = c0 if winner == 0 else c1
        lose = c1 if winner == 0 else c0
        wins[w, lose] += 1
    return wins, games


def main() -> None:
    p = argparse.ArgumentParser(description="型マッチアップ行列（Layer 1・操縦交絡なし）")
    p.add_argument("--deck-dir", default="data/replays/opp_decks", help="デッキ CSV 群")
    p.add_argument(
        "--replay-dirs", nargs="+", default=["data/replays/others"],
        help="第三者対戦 replay のディレクトリ群",
    )
    p.add_argument("--k", type=int, default=6, help="型ファミリー数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min-episode", type=int, default=0, help="この episode id 未満は除外（鮮度管理）"
    )
    p.add_argument("--out", default="data/archetype_matrix.json")
    args = p.parse_args()

    meta = load_card_meta()

    # 1. デッキ読込 → クラスタリング
    decks: dict[str, list[int]] = {}
    for path in sorted(glob.glob(os.path.join(args.deck_dir, "opp_*.csv"))):
        ids = [int(x) for x in open(path).read().split()]
        if len(ids) == DECK_SIZE:
            decks[deck_hash(ids)] = ids
    assign, centroids = cluster_decks(decks, meta, k=args.k, seed=args.seed)
    print(f"デッキ {len(decks)} 件 → {args.k} ファミリー")

    # 2. 第三者対戦からマッチアップ抽出
    matchups: list[tuple[str, str, int]] = []
    freq: Counter[str] = Counter()  # 帯での出現頻度（hash 単位）
    for d in args.replay_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(path)[:-5]
            if base.isdigit() and int(base) < args.min_episode:
                continue
            try:
                ep = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue
            m = episode_matchup(ep)
            if m:
                matchups.append(m)
                freq[m[0]] += 1
                freq[m[1]] += 1
    wins, games = build_matrix(matchups, assign, args.k)
    print(f"第三者対戦 {len(matchups)} 試合（両席デッキ抽出＋勝敗あり）\n")

    # 3. レポート: ファミリー概況（重心・所属数・帯の代表 hash）
    members: dict[int, list[str]] = defaultdict(list)
    for h, c in assign.items():
        members[c].append(h)
    print(f"{'fam':<4}{'所属':>5}{'ポケ':>6}{'エネ':>5}{'たね':>5}{'ユニーク':>7}  帯で頻出の hash（注記）")
    for c in range(args.k):
        top = sorted(members[c], key=lambda h: -freq.get(h, 0))[:3]
        note = "  ".join(
            f"{h}({freq.get(h, 0)}戦{'・' + KNOWN[h] if h in KNOWN else ''})" for h in top
        )
        po, en, ba, un = centroids[c]
        print(f"F{c:<3}{len(members[c]):>5}{po:>6.1f}{en:>5.1f}{ba:>5.1f}{un:>7.1f}  {note}")

    # 4. レポート: 行列（行ファミリーから見た勝率(試合数)）と総合強さ
    print("\n行列（行から見た勝率・括弧は試合数・ミラー対角含む）:")
    header = "     " + "".join(f"{'F' + str(c):>10}" for c in range(args.k))
    print(header)
    for r in range(args.k):
        cells = []
        for c in range(args.k):
            n = games[r, c]
            cells.append(f"{wins[r, c] / n:>5.2f}({n:>2.0f})" if n else f"{'-':>9}")
        print(f"F{r:<4}" + "".join(f"{s:>10}" for s in cells))

    print("\n総合強さ（ミラー除外・対他ファミリー全試合の勝率）:")
    strength = {}
    for c in range(args.k):
        w = wins[c].sum() - wins[c, c]
        n = games[c].sum() - games[c, c]
        strength[c] = (w / n if n else 0.5, int(n))
    my_fam = assign.get("65c6b47e")
    for c, (s, n) in sorted(strength.items(), key=lambda kv: -kv[1][0]):
        mark = " ← 自デッキの型" if c == my_fam else ""
        print(f"  F{c}: {s:.3f} (n={n}){mark}")

    # 5. JSON 永続化（data/ = 追跡外）
    out = {
        "k": args.k,
        "seed": args.seed,
        "assign": assign,
        "centroids": centroids.tolist(),
        "wins": wins.tolist(),
        "games": games.tolist(),
        "strength": {str(c): [s, n] for c, (s, n) in strength.items()},
        "n_matchups": len(matchups),
        "replay_dirs": args.replay_dirs,
        "min_episode": args.min_episode,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n→ {args.out} に保存")


if __name__ == "__main__":
    main()
