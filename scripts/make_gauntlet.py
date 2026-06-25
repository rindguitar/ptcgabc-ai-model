"""多様な相手デッキ「ガントレット」を生成する（過学習対策）.

league/gate/eval を 5枚のサンプルメタだけに対して回すと、その5枚に過学習して実戦（多様な
未知デッキ）で弱くなる（実測: リーダーボードで悪化）。対策として相手プールを多様化する:

  - サンプルメタ（強い・実在の軸）
  - これまでのチャンピオン系バックアップ（別最適化由来の強デッキ）
  - 全エネルギー色の mono-type 構造化デッキ（アーキタイプの網羅）

これに対し robust なデッキを選べば「どんな相手でも」に近づく（完璧な proxy ではないが
5メタより遥かにマシ）。出力は models/gauntlet/gauntlet_NN.csv（中立名・Pokémon Element 非露出）。
"""

from __future__ import annotations

import glob
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cards import load_card_meta  # noqa: E402
from deck import load_deck, save_deck, structured_deck  # noqa: E402
from deckopt import _load_pool  # noqa: E402


def main() -> None:
    out_dir = "models/gauntlet"
    os.makedirs(out_dir, exist_ok=True)
    meta = load_card_meta()
    rng = random.Random(0)

    decks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(d: list[int]) -> None:
        if len(d) != 60:
            return
        key = tuple(sorted(d))
        if key not in seen:
            seen.add(key)
            decks.append(d)

    # 1) サンプルメタ
    metas = _load_pool(sorted(glob.glob("data/*.csv")))
    for d in metas:
        add(d)
    template = decks[0]

    # 2) チャンピオン系バックアップ（別最適化由来の強デッキ・重複は自動除外）
    champ_paths = sorted(glob.glob("models/champions/*.csv")) + [
        "models/champion_best.csv",
        "models/champion_deck.csv",
    ]
    for p in champ_paths:
        if os.path.exists(p):
            try:
                add(load_deck(p))
            except ValueError:
                pass

    # 3) 全色 mono-type 構造化デッキ（アーキタイプの網羅）
    for color in sorted(meta.basic_energy_id):
        add(structured_deck(template, meta, rng, energy_type=color))

    # 旧ガントレットを消してから中立名で書き出す
    for old in glob.glob(os.path.join(out_dir, "*.csv")):
        os.remove(old)
    for i, d in enumerate(decks):
        save_deck(d, os.path.join(out_dir, f"gauntlet_{i:02d}.csv"))

    print(f"ガントレット {len(decks)} デッキを {out_dir}/ に生成")
    print(
        f"  内訳目安: メタ{len(metas)} ＋ チャンピオン系 ＋ mono-type{len(meta.basic_energy_id)}色"
    )


if __name__ == "__main__":
    main()
