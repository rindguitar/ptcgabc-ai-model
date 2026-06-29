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
局面や合法手を**数値ベクトルに変換**したもの。ネットの入力。本プロジェクトは効果文（Pokémon Element）は
読まず、HP・タイプ・特性の有無などの**数値メタ**だけを使う。[features.py](../../src/features.py)。

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
低めに出やすい）ことにも注意。

## 並列・実行環境

### ProcessPoolExecutor / fork vs spawn
Python で複数プロセスに仕事を分散する仕組み。**fork**＝親をコピーして子を作る（速いが、親が CUDA を
初期化済みだと子がハングする事故がある）。**spawn**＝子を新規起動（安全だが起動が遅い）。本プロジェクトは
NN を使う並列収集で **spawn** を採用（fork+CUDA ハング回避）。[nn_collect.py](../../src/nn_collect.py)。

### GPU vs CPU / batch=1
GPU は**大きなバッチ**の行列計算が速い。MCTS は1局面ずつ（**batch=1**）小さなネットを呼ぶので、転送
オーバヘッドで GPU はむしろ遅い → **CPU 推論＋複数プロセス並列**が有効。学習（大バッチ）だけ GPU を使う。

## このプロジェクト固有

### cabt エンジン
Kaggle/Pokémon が提供する対戦シミュレータ（Competition Data・追跡外）。`battle_*`（対戦進行）と
`search_*`（フォワードモデル）の API を持つ。CPU・標準ライブラリだけで動く。

### pilot（操縦）
デッキを実際に「どう戦わせるか」のエージェント。`heuristic`（速いが特性/効果を使わない）/ `ismcts`
（強いが遅い）/ `nn`（蒸留/improve した NN-MCTS）。**提出物は操縦込み**（デッキだけではない）。

### ratchet / distill / improve（運用コマンド）
- `make ratchet`: デッキ軸。best 起点で探索→ゲートで keep-best。
- `make distill`: NN の床（≈ISMCTS）を蒸留で作る（最初の一度）。
- `make improve`: 蒸留床を種に self-play で ISMCTS 超えを狙う（以後の主軸）。
詳細は [README ワークフロー](../../README.md) と [design-decisions.md](design-decisions.md)。
