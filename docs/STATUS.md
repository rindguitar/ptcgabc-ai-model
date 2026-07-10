# STATUS

最終更新: 2026-07-11（セッション終了時に必ず更新）

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
1. **`make submission-nn` で再提出**（operative＋昇格した加速入り best デッキ）→ Kaggle へ。
2. 日次 `make replays` で**0枚負けが減るか**追跡（§29 の実戦検証・本丸）。
3. **distill 継ぎ足し**: `make distill-1h`（resume・rm 不要）を日中に数回 or overnight。
   新しい distill_best が生まれたら **提出前に必ず外部 A/B**
   （`make eval-net EVAL_NET=models/pvnet_distill_best.pt` vs `EVAL_NET=models/pvnet_operative.pt`・
   distill_best は存在＝自動採用なので負けたら退避）。採用基準: mean ≥ operative かつ worst が劣らない。
4. 並行: `ratchet-nn`（無印1iter≈35分）でデッキ軸の上積み → `make champion-gate` 随時。

## 未解決・保留中の問題
- 新デッキ＋加速で 0枚負け（サイドレース競り負け）が実戦で減るか未検証（提出後の replay 待ち）。
- distill 新 net の「素の加速プレイ率」診断が未整備（diagnose_policy に加速率を足すと判定が締まる）。
- distill_best の worst 崩壊（0.100×2マッチ）の原因未調査（iter 増で消えるか要観測）。

## 直近の決定事項
- 2026-07-10: ratchet/gate 分離・`GATE_FLOOR` 0→8 復元（floor は「改良ヒューリスティックに加速を打たせる経路」でもある）→ 詳細 docs/learning/design-decisions.md §29。
- NN 凍結解除の根拠: §24「データに無い信号は学べない」の前提が変化（教師 ISMCTS が改良ヒューリスティック経由で加速を実演・訓練対局に加速札が入る）。improve でなく **distill 継ぎ足し**から。
