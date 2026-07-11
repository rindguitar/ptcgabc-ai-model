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
1. **champion-gate（実行中）の結果確認**: ネットデッキ a8c57d4b（eval-deck mean **0.777**/worst 0.350・
   歴代最高）が通れば昇格 → `make submission-nn`＋`make submission` で**両操縦・新デッキ再提出**。
   ※ 再提出には §31 の適応 sims＋操縦改善も自動で乗る（提出経路のみ有効）。
2. 提出前に `make smoke-submission` で 600 秒クロックを実測（適応 sims の初回検証・重要）。
3. 日次 `make replays`: 0枚負け率と勝率の推移（ネットデッキ＋§31 の実戦効果）。
4. **distill 継ぎ足し**: `make distill-1h`（resume・rm 不要）。昇格後は TRAIN_DECKS の先頭が
   ネットデッキ＝ミラーを極める訓練になる。新 distill_best は提出前に必ず外部 A/B（§30 の罠）。
5. 並行: `ratchet-nn`（ネットデッキを種に周辺変異を探索）→ `make champion-gate` 随時。

## 未解決・保留中の問題
- **加速注入は実戦で効果なし（§29b・1日目）**: 0枚負け率59%/58%で不変・条件付き勝率も無効果。
  デッキ側仮説は棄却方向＝「自作進化デッキの天井」を疑いネットデッキ検証へ。増量（MIN_ACCEL 2→4）は
  条件付き勝率が負なので保留。
- distill 新 net の「素の加速プレイ率」診断が未整備（diagnose_policy に加速率を足すと判定が締まる）。
- distill_best の worst 崩壊（0.100×2マッチ）の原因未調査（iter 増で消えるか要観測）。

## 直近の決定事項
- 2026-07-10: ratchet/gate 分離・`GATE_FLOOR` 0→8 復元（floor は「改良ヒューリスティックに加速を打たせる経路」でもある）→ 詳細 docs/learning/design-decisions.md §29。
- NN 凍結解除の根拠: §24「データに無い信号は学べない」の前提が変化（教師 ISMCTS が改良ヒューリスティック経由で加速を実演・訓練対局に加速札が入る）。improve でなく **distill 継ぎ足し**から。
