# STATUS

最終更新: 2026-07-18（セッション終了時に必ず更新）

## 現在のフェーズ
**実戦 A/B 第2ラウンド＝操縦強化の検証**（クローン章 §37 で「デッキでなく操縦」と確定）:
- 枠1: **ismcts×a8c57d4b**（旗艦・継続中・レート維持のため再提出しない）
- 枠2: **AlphaGo型 v3×a4066acd**（`models/submission_alphago_v3.tar.gz`）
  ＝クローン事前分布＋接地ロールアウト葉（§38）＋山札リサイクル（§39）＋特性解放（§40）
  ＋相手推定 332 デッキ・floor0・盤面補正なし。**v3 の提出はユーザー側**（済みなら判定待ち）。
  ⚠️ 提出順に注意: v3 のみ提出（古い alphago 枠が落ち、ismcts 枠は残る）。

## 前回やったこと
- **§44 replay-tune の3点比較（NN 凍結論争の決着）**: eval-net に `--pilot ismcts` を配線
  （`make eval-net-ismcts`・ホスト可）し、operative／replay-tuned／ISMCTS を同一プールで比較。
  tuned 0.599 < operative 0.713（12/16 相手で劣後）→ **ロールバック・replay-tune は closed**
  （MAE 0.38→0.15 でも退行＝校正≠探索誘導）。ISMCTS(tb0.3) 0.537 < operative
  → **「凍結 NN が評価を歪める」懸念は棄却**・評価は operative 継続・凍結は「測った上での凍結」へ。
- **v3 の3点判定（scout_field で実測・§39が alphago に未伝播と判明）**: replays-daily 完走後の
  指紋比較で alphago(v3) と ismcts が別デッキ・別挙動と確定:
  alphago 勝率0.47・特性3.8/戦・**リサイクル13%**・a4066acd（poke19/ene7 エンジン型）／
  ismcts 勝率0.54・特性0.1・**リサイクル84%**・a8c57d4b（poke10/ene13）。
  **alphago の敗因1位は deck-out（敗北の43〜55%）**、ismcts は309戦で deck-out 負け3のみ。
  原因: §39 のリサイクル⓪優先は heuristic（agents.py:81-95）にあるが、alphago は MAIN を
  MCTS 経路（`_MCTS_SELECT_TYPES`）で決め **floor0 なので heuristic を参照せず** visit 最大手に委ねる
  →リサイクル札を13%しか打てず deck-out。§40 の特性も3.8/戦で 4.0 からほぼ動かず（TeamA 8.6）。
- **§42 replay の OOM 修正＋消費済み JSON の自動破棄**: `make replays` が Error 137（OOM）で落ちた。
  replay が 1116 件・4.9GB に達し `analyze_replays.py` の全件一括 load が原因。ストリーミング化で
  ピーク RSS 15GB 超→1.1GB・73秒で完走（デッキ収穫 454 件）。消費者（analyze／replay-extract）が
  処理済み episode_id を状態ファイルに記録することを利用し、`prune_replays.py`＋`make replays-prune/
  replays-daily` で「analyze＋value 両方済の JSON のみ自動破棄・自分の試合は温存」を実装。
  初回適用で others 237 件・0.96GB を解放（4.9G→4.0G）。残り others 660 件は value 抽出後に対象化。


- **§38 AlphaGo型 pilot**: クローンを操縦から降格し事前分布に。葉は接地ロールアウト
  （leaf_rollouts=1）。実戦1日目 0.441 vs ismcts 0.490（互角・続行）。
- **§39 デッキ切れの真犯人**: 残量仮説を棄却→ deckCount 軌跡の +5 ジャンプで山札リサイクル札
  （id 1129）の不使用を特定。heuristic ⓪段＋TO_DECK 最大返却で修正・E2E 確認済み。
  ※ 初版は後続 select が minCount に落ち **8回中8回0枚返却**の空振り——E2E 実測必須の教訓。
- **§40 行動差分の全量調査**（behavior_diff.py）: 11次元の差が「エンジン回転数」の1ループに集約。
  特性 8.6 vs 4.0/戦 → 開発後の特性解放を実装（採用率84%）。
- **§41 フィールドスカウト**（scout_field.py・--vs 敗因研究モード）: 上位帯はエンジン/中速加速/
  遅滞の三つ巴。1位を倒すのは同型ミラー（TeamG 51勝）・中速加速（TeamH）・遅滞（リサイクル100%・
  初攻T9）。型=デッキが決め、操縦は実行度。我々の2枠は中速加速寄り／エンジン型（v3で回転数改善）。
- 運用整備: ratchet と gate の分離（ratchet=探索のみ・gate は単独実行）・探索/判定の実メタ化・
  learning/（glossary+2語・architecture に新6スクリプト）・CLAUDE.md Git 規約に
  「マージ済みブランチは削除」を明文化し、マージ済み12ブランチをローカル・リモートとも削除。

## 次の一手（優先順）
1. **§43 リサイクル強制手【実装済み・v4/v3.5 並走 A/B へ】**（feature/alphago-recycle）:
   find_forced_recycle を agents.py に公開し nn_mcts の探索前段へ注入（E2E: 機会 4/4 発火）。
   閾値は `--recycle-at` でビルド時に指定可（未指定=既定15）。**ユーザー決定: 旗艦 ismcts を
   手放し v4（閾値15）＋v3.5（閾値25）の2枠並走**で閾値の用量反応を実戦測定。
   ビルド（ユーザー・v3 と同じ deck/net で）:
   `build_submission.py --policy nn --net <クローンnet> --floor-rollouts 0 --leaf-rollouts 1 --deck <a4066acd> --out models/submission_alphago_v4.tar.gz`
   ＋ 同コマンドに `--recycle-at 25 --out models/submission_alphago_v35.tar.gz`。
   提出→翌日 replays-daily → scout_field でリサイクル率 13%→？・deck-out 率を2枠比較。
   効いた方を残し main へマージ。
2. 次の武器候補: **フェッチ優先度の注入**（TeamA は id741 優先・我々は id305 過剰）
   → _generic_select にデッキ別事前分布。デッキ別 JSON の同梱機構の設計から。
3. （counter-meta 候補）遅滞デッキ opp_65c6b47e / opp_88cc3fe1 を eval-deck で評価
   （§39 のリサイクル実装と相性が良い）。
4. 並行（ユーザー）: make gauntlet-real → ratchet-nn-overnight → 翌朝 champion-gate。
   TeamA の定期再DL＝クローン事前分布の継ぎ足し（distill は当面スキップ）。
5. **replay の運用は今後 `make replays-daily`（analyze→replay-extract→prune --apply）に統一**。
   これで消費済み JSON は自動破棄される。残 others 660 件（~3.5GB）は次回 replays-daily で value 抽出後に消える。
   容量確認のみは `make replays-prune`（dry-run）。自分の試合を消したい時だけ `PRUNE_ARGS=--include-own`。

## 未解決・保留中の問題
- クローンの floor 抑圧疑い（floor0=0.681 > floor8=0.588）——v3 は floor0 採用済み。
- nn 学習は凍結中（§44 で実測根拠付き）。解除条件は「ISMCTS 実行を超える具体仮説」。
  value_samples は 187,236 まで蓄積済み（replay-tune での利用は §44 で closed・他用途は自由）。

## 直近の決定事項
- 2026-07-18: replay-tune はロールバック・評価は operative 継続・NN 凍結を実測で確認（§44・
  3点比較の配線 `--pilot ismcts` は eval-net に常設）。次は §43 リサイクル注入へ。
- 2026-07-18: replay の OOM 修正（ストリーミング）＋消費済み JSON の自動破棄（§42・
  破棄条件は analyze＋value 両方済・自分の試合は温存・`make replays-daily` に運用統一）。
- 2026-07-17: gate/ratchet 分離・リサイクル§39・特性解放§40・型と実行度§41・
  マージ済みブランチ削除ルール。
- 2026-07-16: クローン操縦は撤収（§37・2教師で同型失敗＝複合誤差）→ 資産は事前分布へ（§38）。
