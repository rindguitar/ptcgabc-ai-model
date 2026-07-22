# STATUS

最終更新: 2026-07-22（セッション終了時に必ず更新）

## 現在のフェーズ
**65c6b47e（遅滞デッキ）両枠運用・v2（fetch_priors×ismcts／threat_bonus×nn）1日経過で決着**（§57）:
- 枠1（ismcts）: `submission_stall_ismcts_v2.tar.gz`（fetch_priors・TeamJ分）
  実戦 0.491→**0.567**（ベンチ切れ負け14%→3.3%）＝**継続**。レート692（上限707に接近）。
- 枠2（nn）: `submission_stall_v2.tar.gz`（threat_bonus α=0.1）実戦 0.531→0.489（改善なし）。
  **threat_bonus は不採用へ**（α=0 に戻す方向・次提出で反映）。レート556（中位）と整合。
  threat_bonus は NN の value evaluator ラップ実装のため**ISMCTS には注入不可**（§48 仕様）。
- §56 で提起した「ローカル判定は実戦の改善を過小評価するか」は今回のthreat_bonusでは
  不成立（ローカルもα=0.1が最悪＝実戦と一致）。乖離の主論点は §54 の pilot 間比較のまま。

## 前回やったこと（2026-07-21・feature/fetch-priority ブランチ）
- **§49 cardId 誤読バグを修正**: 山札サーチの `option` は cardId を持たず、実カードは
  `sel.deck[option.index].id` で解決する必要があった（旧実装は常に0落ち＝「たね優先→
  エネ優先」分岐が本番で一度も機能していなかった可能性）。`_generic_select`/
  `mine_fetch_priorities` 両方を修正。§31 と同型の再発（実データ未確認のバグ）。
- **§49 教師プールの誤りを是正**: cardId 完全一致（65c6b47e）に釣られ、**自チームの
  対戦ログ（マッチメイキング相手＝実力の裏付けなし）を教師扱いしていた**とユーザー指摘で発覚
  （TeamL/TeamK は実は自分の過去対局の相手）。`mine_fetch_priorities.py` の
  既定 `--dir` を `data/replays/others`（意図的に DL した上位帯のみ）に変更。
- **§50/§51 役割別優先度マイニング新設**（`mine_search_role_priors.py`）: cardId でなく
  役割カテゴリで集計しデッキ非依存にプール（others/ 36チーム・局面数1262）。
  `sel.context`（TO_HAND/TO_BENCH/ATTACH_TO）× 型混在有無 × `supporterPlayed` の
  2軸4バケットに分離。候補数正規化（選好指数 lift）と、`sel.effect.id` で判定した
  サーチ効果コストの捨て札優先度も追加。
- **主要発見**: ① overall の「basic_energy 最優先」は TO_HAND 以外のコンテキスト混入の
  アーティファクト（TO_HAND 内では energy と pokemon はほぼ拮抗）。② lift 正規化で
  トレーナー各種の優先度差はほぼ幻（lift≈1.0）と判明、**basic_pokemon だけ明確に忌避
  （lift=0.46・局面数136で頑健）**。
- **§52 `_generic_select` を修正**: 山札サーチのカテゴリ順を「たね>エネ>その他」→
  「エネ>その他（たね含む）」に変更（上記発見に基づく）。fetch_priors（tier0）は不変。
  影響は TO_HAND の型混在局面のみ（TO_BENCH/ATTACH_TO は構造的に無風）。
- 全テスト green（88 passed・3 skipped）・ruff clean。§47/§48（fetch_priors/threat_bonus
  本体実装）は前回セッションで完了済み・データ待ちのまま。
- **§53 threat_bonus α=0 対照実測**（ユーザー実行・同ブランチ状態＝§49/§52 込み）: 最悪
  0.225/平均0.399。α=0.1（最悪0.150/平均0.386）を同条件で上回る＝α=0.1 は悪化と判明。
  ユーザー指摘により、単発レバーの α 単独スイープだけで採否を断定せず、fetch_priors
  （§47）が揃った段階で蓄積変更の合成構成も別途評価する方針に。
- **§54 現提出物（65c6b47e）1日目の replay 分析**: variant フォルダ名（ismcts/nn）は
  旧旗艦との混在に注意が必要と判明（raw JSON からデッキ hash 再計算で分離）。
  65c6b47e 限定で ismcts 28/57（0.491）・nn 26/49（0.531）＝ほぼ拮抗（n少で有意差なし）。
  デッキ切れは両枠とも低水準。天敵 9e3ece3f は nn 0/4・ismcts 遭遇なし（n=4で要継続監視）。
  `make gauntlet-real` で判定プールを実メタ544デッキから再選抜（カバー2370/3374）。
  **事故**: `make replays-daily` の prune が `others/`（§47 cardId 単位マイニングの教師
  プール・58 episode）を、cardId 単位マイニング未実行のまま削除した（詳細 decisions.md §54）。
  §47 をさらに進めるには others/ の再 DL が必要。
  **再発防止済み**: `prune_replays.py` の既定 `--keep-variants` に `others` を追加
  （自チーム試合を others/ に置くことはない前提のため独立ライフサイクルで温存・
  破棄したい場合のみ `--include-own`）。今後は replays-daily を回しても others/ は消えない。
- **§55 fetch_priors 初回マイニング**: 追加 DL 分（others/ 241 episode・新チーム）を
  `mine_fetch_priorities.py` で初回マイニング（state.json 新規作成・78 team×deck 組）。
  **提出デッキ（65c6b47e）完全一致の教師が2チーム出現**: TeamJ（3ep・
  id344=0.67/id18=0.43）・TeamK（1ep・id345=0.50）。単体ではまだ薄いが前進。
  次提出ではTeamJ側
  （`data/fetch_priors/___________65c6b47e.json`）を採用候補とする（詳細 decisions.md §55）。

## 2026-07-20
- **champion-gate（新プール＝9e3ece3f 入り）**: new 最悪0.250/平均0.458 < best 最悪0.375/平均0.641
  → 据え置き（ノイズドリフト阻止）。best の最悪 0.375 は天敵 9e3ece3f 相当＝champion の弱点が
  ローカル数値でも裏付けられた（実戦 alphago 系 0.22 と整合）。
- **確定版天敵チェック合格**: 65c6b47e×測定構成（operative＋floor8＋盤面補正0.2・Docker・
  20試合席入替）で 9e3ece3f に **0.65**。champion の 0.375 を明確に上回り、枠差し替えの
  前提クリア。ISMCTS 操縦の 0.85（保守基準線）との差は 20 試合では有意といえず、
  §44 の平均実測（operative 0.713 > ISMCTS 0.537）に従い構成は変更しない。
- **提出物ビルド済み**: `models/submission_stall_v1.tar.gz`（nn＋floor8＋board0.2＋65c6b47e・
  相手候補500デッキ同梱）。アップロードと枠選択（推奨: 弱い枠 v3.5=663）はユーザー。
- **§45 の横展開監査（ユーザー指摘・リポジトリ全域）**: off-by-one は**ゼロ**。
  episode JSON の steps を読むのは scripts 7本のみで、ペアリングする4本は全て `steps[i+1]`・
  残り3本は action 非依存 or 初手デッキ提出走査のみ。src/ は live 対局（obs に即時応答）で
  構造的に無関係、npz 消費系（teacher_tune 等）は正実装の teacher-extract 産で健全。
  ただし replay-format.md の樹形図が同一 step ペアと誤読できる記述だったため、
  t+1 規則の明記＋正実装再利用の指示を追記（再発防止）。

## 2026-07-19
- **【撤回】「誤デプロイ」は誤報＝§43 は本番で発火100%**: 正しいペアリング
  （obs[t] への応答は **steps[t+1].action**・既存スクリプトは元から正実装）で
  v4 14/14・v3.5 17/17 発火（v3 基準 8%）。誤報原因は Claude のアドホックプローブの
  off-by-one（§45・教訓: 行動ログ解析は既存実装を再利用）。
  実勢: 発火は完璧だが**機会が希少**（v4 14/64・v3.5 17/62 試合）＝律速は札の可用性
  （60枚中1枚・サイド落ち~10%・掘りで喪失）。deck-out は試合比 26%→15-16% に減少。
- **系統天敵 9e3ece3f の特定**（§46）: レートとローカル測定の乖離を追跡し、エネ厚エンジン
  （poke17/ene14・出現35件で増加中）に **alphago 系 4/18（0.22）**・旧 ismcts は 5/9 と判明。
  負け筋はデッキ切れ＝消耗戦で寝かされる。v4 の広レンジ（505-891）は概ね Elo ランダムウォーク
  ＋この天敵で、「v4 固有のメタられ」ではない（閾値差は試合の~25%にしか影響しない）。
- **判定プールの盲点を解消**: 9e3ece3f は16デッキプールに不在だった→ `make gauntlet-real` で
  再選抜し **9e3ece3f がプール入り**（カバー 2260/3152 遭遇）。ratchet は内側ループで
  gauntlet を読まない（league.py:338）ため走行中でも安全と確認の上、並行実行。
- **65c6b47e の天敵チェック（保守基準線）**: ISMCTS 操縦（--pilot ismcts・ホスト）で
  9e3ece3f に **0.85**・92985406 に **0.85**（各20試合）。遅滞はエンジンの天敵も食う。
- 運用: prune の keep-variants を**前方一致**に修正（alphago_v4 等の A/B 派生ディレクトリの
  誤削除防止・実運用で345件温存を確認）。replays-daily は ratchet と並行実行可と確認。

## 前回まで（要点のみ・詳細は decisions.md）
- **§44 replay-tune 3点比較**: tuned 0.599 < operative 0.713 → ロールバック。
  ISMCTS(tb0.3) 0.537 < operative → 評価は operative 継続・NN 凍結は実測根拠付きに。
- **§43**: リサイクル強制手を nn_mcts 探索前段へ注入・`--recycle-at` で閾値指定可。
- **§42**: replay OOM 修正（ストリーミング）＋消費済み JSON 自動破棄（replays-daily 運用）。
- **遅滞デッキ評価（計60試合/相手で再現）**: 65c6b47e 単純平均 0.728・頻度加重 0.748
  （champion best 0.631 超え）。弱点は a8c57d4b 0.15（出現0.9%）・同型 88cc 0.35（4.7%）。
  実メタ33%の a4066acd に 0.82。

## 次の一手（優先順）
1. **threat_bonus を α=0 に戻す提出（§57 の決定・未ビルド）**: nn 枠を fetch_priors 無し・
   threat_bonus 無しの構成に戻すか、この機会に nn 枠にも fetch_priors を試すか要相談。
   threat_bonus は ISMCTS に注入不可（NN value evaluator ラップのため・§48 仕様）なので
   ismcts 枠は現状（fetch_priors のみ）を維持でよい。
2. **fetch_priors（ismcts枠）は継続**: 実戦 0.491→0.567・ベンチ切れ負け14%→3.3%で効果確認
   （§57）。教師は依然TeamJ3episodeのみ＝others/ の追加 DL で厚くできれば
   より頑健になる。
3. **§56 の保留論点（eval-net 的ローカル判定の格下げ）**: 今回の threat_bonus はローカルと
   実戦が一致したため決定打にならず。乖離の主論点は §54 の pilot 間比較（ローカル
   operative≫ISMCTS だが実戦拮抗）のまま。次に判断材料が増えたタイミングで再検討。
4. **§52 の効果検証**: `_generic_select` のたね優先撤回（エネ>その他）が実戦のデッキ切れ率・
   ベンチ切れ率（§31 の当初の狙い）にどう効くか、継続観測。
   根拠が TO_HAND 型混在局面（局面数136）に偏っているため一般化は未検証。
5. gate の判定基準に頻度加重平均の併記を検討（最悪ケース基準は出現0.9%の天敵に引きずられ
   遅滞系を不当に棄却する・要議論）。
6. replay 運用は `make replays-daily` に統一（2026-07-22 実行済み・others/ は keep-variants
   化で自動破棄されなくなった＝データ量は `du -sh data/replays/` で定期確認・現状2.8GB）。
   プール刷新は `make gauntlet-real`（ratchet 並行可・gate/eval の起動前に）。

## 未解決・保留中の問題
- v4/v3.5 の閾値 A/B はレート平衡により早期収束見込み（同一提出物の継続収集はしない方針）。
  現時点の読み: 閾値差の実戦効果は小（発火機会が~25%の試合にしかない）。
- クローンの floor 抑圧疑い（floor0=0.681 > floor8=0.588）——v4/v3.5 は floor0 採用済み。
- nn 学習は凍結中（§44 で実測根拠付き）。解除条件は「ISMCTS 実行を超える具体仮説」。
  value_samples は 220,920 まで蓄積済み（replay-tune は closed・他用途は自由）。
- §47 cardId 単位フェッチ優先度の教師プール（others/）が §54 の prune で消失。再開には
  再 DL が必要（下記参照）。

## 直近の決定事項
- 2026-07-22: v2提出（fetch_priors×ismcts／threat_bonus α=0.1×nn）の1日経過結果（§57）:
  ismcts 0.491→0.567（ベンチ切れ負け14%→3.3%）で**fetch_priors継続**。nn 0.531→0.489で
  **threat_bonus不採用**（次提出で α=0 に戻す）。threat_bonus は ISMCTS 不可（実装仕様）を
  再確認。今回は実戦がローカル判定と一致し、§56 の「ローカル格下げ」論点は決定打なし。
  `make replays-daily` 実行済み（others/ は keep-variants 化で自動破棄されなくなった）。
- 2026-07-21: cardId 誤読バグ修正（§49）＋教師プールを others/ 限定に是正（自チーム対戦
  ログを教師扱いしていた誤りをユーザー指摘で発見）。役割別優先度マイニング新設（§50/§51）で
  「たね優先」が上位帯実測（lift=0.46）で否定されたため `_generic_select` を修正（§52・
  エネ>その他のみに）。効果検証は次の a/b・提出で。
- 2026-07-21（同日・続き）: threat_bonus α=0 対照実測（§53・α=0.1 は悪化と判明）。
  現提出物（65c6b47e）1日目 replay 分析（§54）: ismcts 0.491・nn 0.531 で拮抗、
  gauntlet-real で判定プール実メタ同期。**replays-daily の prune が others/（§47 教師
  プール・58 episode）を cardId 単位マイニング未実行のまま削除**（事故）＝再 DL 待ち。
  ユーザー指摘で `prune_replays.py` の既定 keep-variants に others を追加（再発防止済み）。
- 2026-07-20: 新プール gate は据え置き（new 0.458 < best 0.641）。天敵チェック合格
  （65c6×nn 構成 0.65 vs 9e3ece3f）→ submission_stall_v1 ビルド完了。
  **両枠を遅滞デッキへ差し替え決定（ユーザー）**: 枠1=ISMCTS 操縦・枠2=nn 測定構成の
  同デッキ操縦 A/B。alphago 系は退役（閾値 A/B 打ち切り・フェッチ優先度は並行実装へ）。
- 2026-07-19: 誤デプロイ誤報を撤回（§43 発火100%・原因は測定側 off-by-one §45）。
  gauntlet を実メタ同期し天敵 9e3ece3f をプール入り（§46）。prune keep-variants 前方一致化。
  平衡打開はデッキ交換（65c6×測定構成）で行く方針・枠選択はユーザー決定待ち。
- 2026-07-18: v4（閾値15）/v3.5（閾値25）並走 A/B のため旗艦 ismcts を退役（ユーザー決定）。
  replay-tune ロールバック・評価は operative 継続（§44）。replay OOM 修正＋自動破棄（§42）。
- 2026-07-17: gate/ratchet 分離・リサイクル§39・特性解放§40・型と実行度§41・
  マージ済みブランチ削除ルール。
- 2026-07-16: クローン操縦は撤収（§37）→ 資産は事前分布へ（§38）。
