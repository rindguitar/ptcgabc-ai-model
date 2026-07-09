# STATUS

最終更新: 2026-07-10（セッション終了時に必ず更新）

## 現在のフェーズ
デッキ軸の探索を回しつつ、**NN 凍結解除（distill 継ぎ足し）**を検討中。
提出は ISMCTS / floored NN の A/B を Kaggle で並走・日次 replay で監視。

## 前回やったこと
- ratchet から gate を分離（`ratchet-nn`=探索のみ・純探索 ≈35分/iter）。昇格は `make champion-gate`（floor8）を随時（§29・commit ab71d1f）。
- 「0枚負け」の真因を replay 深掘りで特定＝**エネ加速ゼロ＋操縦がどうぐ/加速を打たない**。対策=デッキに加速ロール注入（`deck.MIN_ACCEL`）＋ヒューリスティックの打ち回し順序（§29）。
- determinize 相手デッキ推定をベイズ重み付け化（§26）・ratchet の探索相手を実メタ乱択化（§27）。

## 次の一手（優先順）
1. overnight 探索（`ratchet-nn-overnight` iters11）完了後 → `make champion-gate` で昇格判定（加速入り初の公平判定）。
2. 昇格したら `make submission-nn` で再提出 → 日次 `make replays` で**0枚負けが減るか**追跡。
3. **NN 凍結解除＝distill 継ぎ足し（前提整備済み・849eeff）**: train_alphazero が学習デッキへ構成射影を
   自動適用（TRAIN_DECKS 12種すべて加速≥2 検証済み・`--no-repair-decks` で無効化可）。実行手順:
   `rm -f models/pvnet_distill.pt`（加速ゼロ時代の蓄積を捨てる）→ `make distill-1h` →
   判定 `make eval-net EVAL_NET=models/pvnet_distill_best.pt`（vs 実メタ・非ミラー）。
   採用基準: vs 実メタ mean が operative+注入 以上（keep-best・NN_NET は distill_best 優先解決で自動切替）。

## 未解決・保留中の問題
- 新デッキ＋加速で 0枚負け（サイドレース競り負け）が実戦で減るか未検証。
- distill 新 net の「素の加速プレイ率」を測る診断が未整備（diagnose_policy に加速率を足すと判定が締まる）。

## 直近の決定事項
- 2026-07-10: ratchet/gate 分離・`GATE_FLOOR` 0→8 復元（floor は「改良ヒューリスティックに加速を打たせる経路」でもある）→ 詳細 docs/learning/design-decisions.md §29。
- NN 凍結解除の根拠: §24「データに無い信号は学べない」の前提が変化（教師 ISMCTS が改良ヒューリスティック経由で加速を実演・訓練対局に加速札が入る）。improve でなく **distill 継ぎ足し**から。
