# STATUS

最終更新: 2026-07-12（セッション終了時に必ず更新）

## 現在のフェーズ
デッキ軸=**加速入り新 best が昇格**（初の通し成功）。NN 軸=凍結解除ラウンド1は不採用だが仮説生存
（§30）＝distill 継ぎ足し続行。提出は再ビルド待ち。

## 前回やったこと
- **ratchet 昇格成功**: 加速入り探索デッキが確認評価込みで旧 best 超え（new 0.188/0.479 > best 0.125/0.464）。
- **凍結解除ラウンド1（§30）**: ゼロから distill-1h → 外部 A/B（同条件・実メタ16）で
  operative 0.594/0.300 > distill_best 0.537/0.100 → 不採用・`pvnet_distill_v24_candidate.pt` に退避。
  ただし1hでノイズ圏まで迫った＝**伸び代は iter 数**。`pvnet_distill.pt` から継ぎ足し可能。
- 学習デッキへの構成射影を train_alphazero に組込（849eeff・TRAIN_DECKS 12種すべて加速≥2）。

## 次の一手（優先順）
1. **ネットデッキ検証（§29b）**: 我々に勝率0.7超の頻出実メタデッキを自デッキ候補として測る。
   `make eval-deck EVAL_DECK=data/replays/opp_decks/opp_04621784.csv`（a8c57d4b も）。
   best 超えなら `cp → models/champion_deck.csv` → `make champion-gate` で昇格。
2. `make gauntlet-real` で判定プールを更新（実メタ 132→191 に増加済み）。
3. **distill 継ぎ足し**: `make distill-1h`（resume・rm 不要）。新 distill_best は提出前に必ず外部 A/B
   （存在＝自動採用の罠・§30）。採用基準: mean ≥ operative かつ worst が劣らない。
4. 並行: `ratchet-nn`（無印1iter≈35分）→ `make champion-gate` 随時。

## 未解決・保留中の問題
- **加速注入は実戦で効果なし（§29b・1日目）**: 0枚負け率59%/58%で不変・条件付き勝率も無効果。
  デッキ側仮説は棄却方向＝「自作進化デッキの天井」を疑いネットデッキ検証へ。増量（MIN_ACCEL 2→4）は
  条件付き勝率が負なので保留。
- distill 新 net の「素の加速プレイ率」診断が未整備（diagnose_policy に加速率を足すと判定が締まる）。
- distill_best の worst 崩壊（0.100×2マッチ）の原因未調査（iter 増で消えるか要観測）。

## 直近の決定事項
- 2026-07-10: ratchet/gate 分離・`GATE_FLOOR` 0→8 復元（floor は「改良ヒューリスティックに加速を打たせる経路」でもある）→ 詳細 docs/learning/design-decisions.md §29。
- NN 凍結解除の根拠: §24「データに無い信号は学べない」の前提が変化（教師 ISMCTS が改良ヒューリスティック経由で加速を実演・訓練対局に加速札が入る）。improve でなく **distill 継ぎ足し**から。
