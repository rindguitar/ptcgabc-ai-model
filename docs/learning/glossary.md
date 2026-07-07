# 用語集

このプロジェクトで出てくる技術用語を、初学者向けに短く説明する。詳しい使い所は
[architecture.md](architecture.md) と [design-decisions.md](design-decisions.md) を参照。

## 探索（ゲーム木 / MCTS 系）

### ゲーム木 / フォワードモデル
ある局面から「手を打つ→次の局面」を辿る木。**フォワードモデル**＝「この手を打ったら次はこうなる」を
計算できる仕組み。本プロジェクトは cabt エンジンの `search` API をフォワードモデルに使う。

### MCTS（Monte Carlo Tree Search・モンテカルロ木探索）
木を「①選択→②展開→③評価→④逆伝播」の4ステップで少しずつ育て、試行を重ねるほど良い手に
探索を集中させる手法。囲碁の AlphaGo で有名。1回の試行を **simulation（プレイアウト）** と呼ぶ。

### rollout / playout（ロールアウト）
MCTS の③評価で、葉（まだ評価していない局面）から**終局まで適当な方策で打ち切ってみて**勝敗を価値にする方法。
速いが雑（"rollout washout"＝雑なロールアウトに探索が引っ張られる問題がある）。NN を使うとロールアウト
不要で価値を直接推定できる（→ PUCT / AlphaZero）。

### UCT（Upper Confidence bounds applied to Trees）
MCTS の①選択で「Q（平均価値）＋探索ボーナス（あまり試していない手を優先）」を最大化して手を選ぶ式。
`Q + c * sqrt(ln(N)/n)`。`c` が探索の強さ。

### PUCT（Predictor + UCT）
UCT に **事前確率（priors）** を掛けた版。AlphaZero が使う。`Q + c_puct * P(a) * sqrt(N)/(1+n)`。
priors を NN（policy head）が出すので、有望な手に最初から探索を寄せられる＝少ない試行で強くなる。

### ISMCTS（Information Set MCTS）/ determinized UCT / PIMC
**不完全情報ゲーム**（相手の手札・山札が見えない）向けの MCTS。隠れ情報を1つの仮定に
**determinize（決定化＝サンプリングで仮に確定）** してから普通の木探索を回し、これを複数回違う仮定で
繰り返して集計する。PIMC（Perfect Information Monte Carlo）とも。実装は [ismcts.py](../../src/ismcts.py)。

### determinization（決定化）
「相手の山札はこれ、手札はこれ」と隠れ情報を**仮に1つ確定**させること。複数の決定化で平均すると、
特定の仮定に依存しない頑健な評価になる。[determinize.py](../../src/determinize.py)。

### time_budget / c_puct / iters_per_det / sims / dets
- `time_budget`: ISMCTS の1手あたり思考秒。大きいほど強いが遅い。
- `c_puct` / `c`: 探索ボーナスの係数（探索 vs 活用のバランス）。
- `iters_per_det`: 1つの決定化につき回す simulation 数。
- `sims`: NN-MCTS の1手あたり simulation 数。`dets`: 決定化の数。

## ニューラルネット / 強化学習

### AlphaZero
「**自己対戦**で集めたデータから policy/value ネットを学習し、そのネットで MCTS を誘導する」を
反復して強くなる枠組み。本プロジェクトの NN 操縦はこの型。

### policy head / value head / PVNet
1つのネットが2つの出力を持つ：**policy head**＝各手の良さ（確率 priors）、**value head**＝今の局面の
勝率予測 [0,1]。両方持つネットを **PVNet**（Policy-Value Net）と呼ぶ。[net.py](../../src/net.py)。

### features（特徴量） / encoding
局面や合法手を**数値ベクトルに変換**したもの。ネットの入力。HP・タイプ・KO 可能・弱点一致などの
**数値メタ＋効果カテゴリ**（下記）を使う。[features.py](../../src/features.py)。
※ 効果文そのものは**読んで数値化に使う**が、生テキストは保持・公開しない（規約: 使用は可・公開は不可）。

### 効果カテゴリ（effect categories）
カードの効果文を**汎用ゲーム機構のキーワード**で分類した数値フラグ（draw / search / heal / energy_accel /
damage_counter / status / switch / prevent など）。効果文は読めるが意味を1つずつ手で読むのは大変なので、
カテゴリのビットマスクにして net に渡す。[cards.py](../../src/cards.py) `_effect_bitmask`。

### cardId 埋め込み（embedding）
カードの ID（整数）を**学習可能なベクトル**に変換する仕組み（`nn.Embedding`）。数値メタでは表せない
「そのカード固有の強さ・癖」を学習で捉える。特徴ベクトル末尾に整数 cardId を載せ、net が分離して埋め込む。
[net.py](../../src/net.py)。

### 接地 floor（grounded floor / 安全弁）
NN-MCTS の手と heuristic の手が違うとき、両者を**実際に打って heuristic ロールアウトで比較**し、良い方を
採る仕組み（net の評価に頼らない＝接地）。**net が間違っても pilot が heuristic を下回らない**保証。
[ismcts.py](../../src/ismcts.py) `evaluate_actions_by_rollout` / [nn_mcts.py](../../src/nn_mcts.py) `floor_rollouts`。

### 蒸留（distillation）/ behavioral cloning（行動クローン）
**強い教師（ここでは ISMCTS）の選択を真似る**ようにネットを教師あり学習すること。自己対戦が弱いネットから
始めると崩壊しがちなので、まず蒸留で**安定した土台**を作る。欠点：**教師を超えられない**（天井＝教師の強さ）。

### one-hot / soft-π（ソフトターゲット）/ 温度（temperature）
方策の学習ターゲットの作り方。
- **one-hot**: 「選んだ手だけ 1、他は 0」。信号が明確だが、迷う局面の情報（僅差）を捨て**過信**しやすい。
- **soft-π**: MCTS の**訪問回数分布**をそのまま使う（例 `[0.6, 0.3, 0.1]`）。情報が多いが、探索が浅いと
  分布が**平坦**になり無情報になる。
- **温度 `τ`**: `π ∝ visits^(1/τ)` で両者を1本化。`τ→0` が one-hot、`τ=1` が生の soft。
  本プロジェクトは既定 `τ=0`（one-hot・安全）、深い教師にしたとき `τ>0` で soft を解放する。

### cross-entropy（交差エントロピー）/ BCE
- policy の損失＝**交差エントロピー** `-(π * log softmax(logits)).sum()`。soft な π もそのまま使える。
- value の損失＝**BCE（二値交差エントロピー）**（勝率予測 vs 実際の勝敗）。

### self-play（自己対戦）/ policy improvement operator（方策改善演算子）
ネット自身（を載せた MCTS）同士を戦わせてデータを作ること。**MCTS(ネット) はネット単体より強い**
（探索で1段良くなる＝方策改善演算子）。その「1段強い手」を学べばネットが上がる→また MCTS が強くなる…
の好循環で、**蒸留と違い教師（ISMCTS）を超えられる**。実装 [selfplay.py](../../src/selfplay.py)。

### replay buffer（リプレイバッファ）
直近 N 反復分の学習サンプルを溜めておき、まとめて学習する仕組み。1反復分だけだと偏る・忘れるのを防ぐ。
古いサンプルは押し出される（**サンプルは新陳代謝、重みは蓄積**）。

### warm-start（ウォームスタート）
特徴量を増やして入力次元が変わったとき、**旧モデルの重みを新しい大きい層の先頭にコピーし、増えた列は0**で
埋めて継続学習すること。過去の学習を捨てない。[train.py](../../src/train.py) `load_net_warmstart`。

### catastrophic forgetting / continual learning（継続学習）
学習を続けると古い知識を忘れる現象（catastrophic forgetting）。`--resume` で重みを継ぎ足す継続学習では、
容量飽和や分布シフトに注意（→ [design-decisions.md](design-decisions.md)「蓄積のデメリット」）。

### drift（ドリフト）/ 安全弁
自己対戦で学習が悪い方向へずれていくこと。本プロジェクトは「作業ネットの直近 eval が best を下回ったら
**best から再開**する」安全弁（`--resume-from-best`）で暴走を止める。

### 重み平均（EMA / SWA）
SGD は各 iter でパラメータが小さく揺れる。**単一 iter の net より、直近数 iter の重みを平均した net の方が
安定して強くなりやすい**という経験則を使う手法。
- **EMA（Exponential Moving Average・指数移動平均）**: `θ_ema ← d·θ_ema + (1−d)·θ`（d≈0.99）。毎 iter
  更新し、常に「最近の重みの滑らかな平均」を保つ。推論/保存には θ_ema を使う。
- **SWA（Stochastic Weight Averaging）**: 学習後半の複数チェックポイントを**単純平均**する。
- 効いる理由: 平均は損失地形の**平らで広い谷（汎化の良い解）**に寄りやすく、iter ごとの上振れ/下振れを打ち消す。
- 本プロジェクトでの用途: 「最新 iter で eval」のノイズ対策。生の最新でなく **EMA net を保存・eval** すれば、
  たまたま悪い iter を掴むリスクが減る（→ [design-decisions.md](design-decisions.md) §17）。
- **落とし穴（実測）**: decay は**更新頻度とセット**。per-step 前提の 0.999 を per-iter（1 iter 1回）更新に
  使うと 30 iter でも 97% が初期重みのまま＝**学習が実質凍結**し、eval/best がずっと種を測り続ける。
  per-iter なら 0.9 程度（半減期≈7iter）。

### Dirichlet 根ノイズ（root exploration noise）
AlphaZero 標準の探索多様化。self-play 収集で**根ノードの事前確率に乱数を混ぜる**:
`P(s,a) ← (1−ε)·p + ε·η`、`η〜Dir(α)`（本プロジェクトは ε=0.25・α=10/手数）。ネットの priors が
平坦/偏っていても探索が毎回同じ手に固まらず、**データの多様性＝改善の探索範囲**が保たれる。
**推論/評価では掛けない**（強さを測る/出すときは決定的）。実装 [nn_mcts.py](../../src/nn_mcts.py) `_apply_root_noise`。

### 収集深度と評価深度の分離（collect-sims）
self-play の学習信号は「**探索後の π − 素の policy**」の差分。よって**収集時だけ探索を深く**すれば
（`--collect-sims`）、評価/推論の速さ（`--sims`）を変えずに**教師の質＝差分**を太らせられる。ただし
「深くすれば強くなる」曲線が寝ていると（本プロジェクトは sims32≈64 で飽和）割に合わないので、
**深さの限界効用を安い eval 実験で先に測る**のが定石。

### 自己蒸留の空転 / behavioral cloning の天井
- **behavioral cloning の天井**: 教師を真似るだけ（蒸留）では**教師の強さが上限**。超えるには self-play が要る。
- **自己蒸留の空転**: self-play でも、**探索後の π が素の policy とほぼ同じ**になると「自分を自分に蒸留」する
  だけで改善が止まる（eval が横ばい）。素の policy が平坦＋探索が浅い＋相手も自分（弱い者同士）で起きる。
  対策＝探索を深く/多様に（[Dirichlet 根ノイズ](#dirichlet-根ノイズroot-exploration-noise)）・強い種から再出発・value の質を上げる。
  兆候は **eval_winrate の横ばい**（loss は下がり続けるので当てにならない・§17）。

### 注入テスト（value への手作り補正による事前検証）
特徴追加＋再訓練（GPU 日単位）に投資する**前に**、狙いの信号を評価器へ外から注入して効果を測る手法。
例: `v' = clamp01(v + α×(自駒数−敵駒数)/5)` を挟んだ floored NN を 40 試合評価。効けば「この信号を
学習させる価値がある」の実証、効かなければ安く撤退できる（design-decisions §21 の「安い代替検証」）。
実装 [nn_eval.py](../../src/nn_eval.py) `wrap_board_bonus`。学習版が完成したら注入は撤去する。

### 温度スケジュール（self-play の手選択）
self-play 中の手を「序盤 N 決定だけ π からサンプリング、以降は argmax」に切り替える運用
（AlphaZero 標準・`--opening-samples`）。サンプリングは開幕の多様性のためで、**終盤まで続けると
浅い探索の平坦な π から悪手を引き、対局の質＝z ラベルの質が壊れる**（下記「z ラベル汚染」）。
※「温度 τ」（学習ターゲットの one-hot↔soft）とは別物: あちらは**教師信号**の作り方、こちらは**打つ手**の選び方。

### z ラベル汚染 / credit assignment（貢献の帰属）
value の教師 z は「最終的に勝ったか」だけ。途中まで正しく打っていても**終盤の乱心で負けると、
その試合の全局面に z=負け が付く**（どの手が悪かったかを z は区別できない＝credit assignment 問題）。
乱心が系統的に起きる（全手サンプリング等）と、value は壊れた対局分布の相関を学ぶ。
対策＝対局の質を保つ（温度スケジュール）・独立な観測線で早期検知（カナリア）。

### 偽相関（spurious correlation）の学習
モデルが「間違い」を学ぶのではなく、**データに実在するが因果でない相関**を正しく学ぶこと。
実測例: 乱心試合が混ざった self-play では本当に「盤面が薄い側が（相手の乱心で）勝って」おり、
value は素直に「自分の場が薄い＝価値が高い」を学んだ。**符号を反転しても直らない**＝
直すべきはモデルでなくデータ生成。学習結果が常識と逆のときは、まず教材を疑う。

### カナリア評価（canary）
平滑化・選抜の**背後にある生の状態**を、安価な独立測定で常時可視化する観測線。実測例: eval/best を
EMA net で行っていたら decay 設定ミスで**学習の劣化が3 run 分隠れた**→ eval のたびに生 net を
先頭デッキ 15 試合だけ測って表示する行を追加。「便利な平滑化には必ず素通しの観測を並走させる」。

### offline 学習 / replay からの value fine-tune
環境と対戦せず、**既存の対局記録（replay）から学ぶ**こと（off-policy/offline RL の素朴版）。本プロジェクト
では実戦 replay の (盤面, 最終結果 z) で **value 頭だけ**を回帰する（`policy_weight=0`）。狙いは「学習環境
（ISMCTS 対局）に無い信号（ベンチ切れ）を、信号のある実戦データから value に入れる」こと（§24/§25）。
policy を触らないのは、我々の手のクローン＝自己蒸留・相手の手＝雑音で policy を汚さないため。

### 訓練データの信号濃度 ≠ データ源の品質
「良いデッキ／強いエージェントの対局＝良い訓練データ」とは限らない。学ばせたい現象（例: 薄い盤面で負ける）
が**濃く起きているデータ**が良い教材。実測: エネ過多の弱いデッキ replay（ベンチ切れ負け63%）は、修正済み
デッキ（同25%）より value 学習に濃い。**何を学ばせたいかで教材の良し悪しは反転する**（§25）。

## 学習ログ・指標（train_log.csv / 学習中の出力）

`models/train_log.csv` と学習中の表示に出る数値の読み方。列は
`time, iter, new_samples, buffer, value_loss, policy_loss, eval_winrate`。

### loss（損失 / ロス）
「予測と正解のズレ」を数値化したもの。**小さいほど良い**。学習は loss を勾配降下で最小化する。
value と policy で別々の loss を持つ。

### value_loss（価値損失）
value head（勝率予測）の誤差。**BCE（二値交差エントロピー）**。実際の勝敗 z にどれだけ近いか。
下がる＝局面の有利不利を当てられている。

### policy_loss（方策損失）
policy head（手の確率）の誤差。**交差エントロピー**。教師の手 π にどれだけ近いか。下がる＝良い手を
選べるようになっている。**下がらず横ばい＝手を順位付けできていない**（要改善のサイン）。
※ soft-π（分布ターゲット）だと的のエントロピー分だけ下限が上がるので、**one-hot と数値を直接比較しない**。

### iter（反復 / iteration）
{データ収集 → 学習 → 保存} の1サイクル。`--iterations` で回数を指定。

### new_samples（新規サンプル数）
その反復で新しく集めた学習サンプル（＝決定局面の数）。対戦数や試合の長さで変動する。

### buffer（リプレイバッファのサイズ）
その反復で学習に使ったサンプル総数（直近 `--buffer` 反復分を保持）。1反復分だけだと偏るので溜めて
学習する。古いサンプルは押し出される（「replay buffer」の項も参照）。

### eval_winrate（評価勝率）
学習中に NN-MCTS を heuristic 等と戦わせた勝率（傾向把握用・少試合でノイズ大）。**確定判断は
`make eval-net`（40+試合）**で行う。空欄の行は eval を回していない反復。

### epoch（エポック）
同じ学習データを何周なめるか（`--epochs`）。多いほど学習が進むが過学習にも近づく。

### batch_size / 勾配累積（gradient accumulation）
一度に勾配を更新するサンプル数（`--batch`）。本プロジェクトは合法手数 n が可変なので**サンプル毎に
loss を計算し、batch 個ためてから1回更新**する（勾配累積）。

### lr（学習率 / learning rate）
重みを1回でどれだけ動かすか（`--lr`）。大きすぎると発散・崩壊、小さすぎると遅い。継続学習では
安定重視でやや低め（既定 5e-4）。

### Adam（オプティマイザ / optimizer）
勾配から重みを更新するアルゴリズムの定番。パラメータごとに実効的な学習率を自動調整する。

### logits / softmax / log_softmax
- **logits**：softmax 前の生スコア（各手の点数）。net の policy head の出力。
- **softmax**：logits を「合計1の確率」に変換する関数。
- **log_softmax**：その log。交差エントロピーの計算で数値安定に使う。

### 価値ターゲット z / 方策ターゲット π
学習の「正解」。
- **z**：その局面の手番が最終的に勝ったか（勝1 / 分0.5 / 負0）。value head の教師。
- **π**：その局面で打つべき手の確率（教師の選択＝one-hot、または MCTS の訪問分布＝soft）。policy head の教師。

## デッキ最適化（探索 × ゲーム理論）

### double-oracle / league（リーグ）
「現在の相手プールに最も強いデッキを進化計算で作る→プールに足す」を繰り返し、頑健なデッキを探す枠組み。
プールを上限で打ち切り（bounded）、冗長なデッキを追い出して多様性を保つ。[league.py](../../src/league.py)。

### worst-case / minimax 目的
「平均勝率」ではなく「**最も相性の悪い相手への勝率（最悪ケース）を最大化**」する目的。
どんな相手にも極端に負けない頑健なデッキを狙う。

### incumbent / 非弱化（non-weakening）更新
最終選抜に**前チャンピオン（incumbent）も候補に入れる**ことで、新チャンピオンが必ず前以上になる（測定
ノイズの範囲で弱くならない）世代更新。

### keep-best ラチェット（ratchet）/ gate（ゲート）
**「確実に良くなった時だけ採用」**して後戻りしない仕組み（ラチェット＝逆回転しない歯車）。league の出力を
別の**ゲート**（多めの試合で評価）に通し、現ベストを上回った時だけ `champion_best` を更新する。
[champion_gate.py](../../scripts/champion_gate.py)。

### 共適応ロック（deck × pilot の近親交配）
デッキの良し悪しを**自分の操縦で**測ると、「操縦が使いこなせるデッキ」だけが選抜され、操縦の弱点
（例: トレーナー活用の浅さ）に合わせた歪んだデッキ（エネ過多等）へ収束する。逆に操縦はそのデッキで
訓練されるので相互に固定化する。破り方＝**外部基準**（実メタの構成分布・実戦 A/B）を持ち込む、
または**先にデッキを正して操縦をそれに合わせて訓練**する（デッキに操縦を従わせる向きで解く）。

### 構成射影（composition projection）
デッキ探索の候補を**カテゴリ枚数の許容帯**（実メタ分布から決めた範囲・COMP_BOUNDS）へ矯正する制約。
帯内は自由探索・帯外だけ「多い cardId から削減／実メタ頻度上位で補充」して射影する。探索が
評価者（操縦）の歪みに引きずられて分布外へ暴走するのを構造的に防ぐ。[deck.py](../../src/deck.py)
`repair_composition`・league の全候補誕生点（[deckopt.py](../../src/deckopt.py) `_make_child`）に適用。

## 評価・統計

### 過学習（overfitting）/ 汎化（generalization）
- **過学習**: 訓練に使った相手/デッキだけに最適化され、未知では弱くなること（症状＝train高/test低）。
- **汎化**: 未知の相手/デッキでも通用すること。本プロジェクトの NN は「学習デッキでは強いが別デッキで弱い」
  ＝汎化不足が課題（issue #2）。

### gauntlet（ガントレット）/ proxy metric（代理指標）
過学習を防ぐための**多様な相手デッキ群**。判定（gate/eval）はこれを相手に行う。ただし gauntlet は本物の
実戦（Kaggle）の**代理指標**にすぎず、gauntlet で単調改善＝Kaggle で単調改善とは限らない（issue #3）。

### 勝率ノイズ / 標準誤差(SE) / min-of-N バイアス
勝率は試行回数が少ないとブレる。標準誤差 `SE ≈ sqrt(p(1-p)/N)`（N=試合数）。20試合だと ±0.1 程度ブレる
ので、確定判断は40試合以上。「最悪ケース（min-of-N）」は**下振れバイアス**が乗る（N個の最小値は平均より
低めに出やすい）ことにも注意。逆に **max 選抜（多数の noisy な eval から最大を選ぶ）は上振れバイアス**
（選んだ値が真の実力より高く出る）＝best 選抜が上振れを掴む原因。

### 用量反応（dose-response）
介入の強さ（例: 注入 α を 0→0.05→0.1→0.2）に対して効果が**単調に増える**かを見る検証パターン。
1点の差はノイズで出るが、**複数の強さで順序が揃う**のは偶然では起きにくい（4点の厳密単調は確率 ~4%）
＝少ない試合数でも効果の実在を強く支持できる。ピーク（頭打ち）の位置は採用パラメータの根拠になる。

### Common Random Numbers（CRN・共通乱数）
2つの設定（iter や net）を比べるとき、**同じ乱数（配牌・determinization）で対戦させる**手法。差が「運」
でなく「設定の差」だけを反映するので、**比較の分散が激減**する。本プロジェクトは学習中 eval で毎回同じ
固定シードから乱数を作り直し、iter 間比較を安定化（[design-decisions.md](design-decisions.md) §17）。

### 確認評価 / ヒステリシス（best 昇格）
max 選抜の上振れ対策。候補が best を**マージン超**で上回ったときだけ、**別シードで再評価**し、`min(1回目,
2回目) > best` のときのみ昇格させる。まぐれ1回で best を上書きするのを防ぐ（1回同一 net が eval 0.556 /
確認 0.467 とブレたのを検出・保守側 0.467 を採用した実例）。

### 教師自己一致率（teacher self-agreement）
診断指標。**同一局面で確率的な教師（ISMCTS）を2回引き、argmax が一致する割合**。低い（≲0.6）＝教師が
確率的＝one-hot ラベルが局面ごとに矛盾＝policy_loss に高い下限（→ soft-π が有効）。高い（≳0.8）のに
net の教師再現度が低い＝net が fit できていない（→ 容量/特徴/学習量の問題）。soft-π か容量かを切り分ける。

## 並列・実行環境

### ProcessPoolExecutor / fork vs spawn
Python で複数プロセスに仕事を分散する仕組み。**fork**＝親をコピーして子を作る（速いが、親が CUDA を
初期化済みだと子がハングする事故がある）。**spawn**＝子を新規起動（安全だが起動が遅い）。本プロジェクトは
NN を使う並列収集で **spawn** を採用（fork+CUDA ハング回避）。[nn_collect.py](../../src/nn_collect.py)。

### GPU vs CPU / batch=1
GPU は**大きなバッチ**の行列計算が速い。MCTS は1局面ずつ（**batch=1**）小さなネットを呼ぶので、転送
オーバヘッドで GPU はむしろ遅い → **CPU 推論＋複数プロセス並列**が有効。学習（大バッチ）だけ GPU を使う。

### git worktree（作業ツリーの隔離）
同じリポジトリの**別ブランチを別ディレクトリに展開**する git 機能。長時間ジョブ（Docker が作業
ディレクトリを bind-mount して実行中）を壊さずに、互換性を破る変更（特徴次元の変更等）を並行開発
できる。ブランチを切るだけではファイル実体が共有されるため不十分、という落とし穴への対策。
追跡外の資産（エンジン・データ・モデル）は symlink で見せる。

## このプロジェクト固有

### cabt エンジン
Kaggle/Pokémon が提供する対戦シミュレータ（Competition Data・追跡外）。`battle_*`（対戦進行）と
`search_*`（フォワードモデル）の API を持つ。CPU・標準ライブラリだけで動く。

### search API / search_begin_input
エンジンの**フォワードモデル**（先読みシミュレーション）用 API。観測に入っている
`search_begin_input`（~270字のシリアライズ文字列）を `search_begin()` に渡すと、**今の局面を開始点に
した模擬対戦セッション**が開き、`search_step()` で手を進められる。ISMCTS/NN-MCTS の探索はこれで動く。
**相手の ID などではない**（探索用のチケット）。→ [replay-format.md](replay-format.md)

### episode / replay（kaggle_environments）
Kaggle 上の1試合の完全な記録（JSON）。チーム名・勝敗・全ターンの観測/行動・残り持ち時間が入って
おり、**相手の初手 action（60 カード ID）から相手デッキを復元**できる＝実メタ較正の材料。
構造と読み方は [replay-format.md](replay-format.md)、抽出は [analyze_replays.py](../../scripts/analyze_replays.py)。

### pilot（操縦）
デッキを実際に「どう戦わせるか」のエージェント。`heuristic`（速いが特性/効果を使わない）/ `ismcts`
（強いが遅い）/ `nn`（蒸留/improve した NN-MCTS）。**提出物は操縦込み**（デッキだけではない）。

### ratchet / distill / improve（運用コマンド）
- `make ratchet`: デッキ軸。best 起点で探索→ゲートで keep-best。
- `make distill`: NN の床（≈ISMCTS）を蒸留で作る（最初の一度）。
- `make improve`: 蒸留床を種に self-play で ISMCTS 超えを狙う（以後の主軸）。
- `make submission` / `submission-nn`: 提出物を作る（ISMCTS 操縦 / floored NN 操縦）。

### 提出の時間管理（game budget / 時間ガード / 600秒クロック）
本大会は**1プレイヤー 600 秒/試合の累積クロック**（超過＝時間切れ負け）。ISMCTS は 1手あたりの秒
（time_budget）と残り予算（game_budget=540 の安全マージン）で自律的に管理する。NN-MCTS は sims 固定で
自前の時計を持たないため、**時間ガード**（累積消費が game_budget を超えたら heuristic＝瞬時へ退避）を
提出アダプタに入れて時間切れ負けを防ぐ。実測は `make smoke-submission`（[smoke_submission.py](../../scripts/smoke_submission.py)）。
