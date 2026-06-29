# ptcgabc-ai-model

Kaggle「The Pokémon Company - PTCG AI Battle Challenge Strategy」向けの
AIモデル開発リポジトリ（**非公開**）。

## 規約上の重要な遵守事項

本リポジトリは [公式ルール](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/rules) に基づき運用する。特に:

- **Competition Data は再配布禁止**（Sec. 2.4）。参加者以外がアクセスできる場所に
  置かないこと。本リポジトリは**非公開**で運用し、`data/` 配下は `.gitignore` 済み。
- **競技終了後は Competition Data を速やかに削除**する義務がある（Sec. 2.4)。
- **Pokémon Elements**（カード名・画像・ルール・データ等）はコミット/公開しない（Sec. 2.5 / 3.18)。
- 私的なコード共有はチーム外では禁止（Sec. 3.6)。公開する場合は Kaggle フォーラムで OSI 承認ライセンス下に行う。
- 受賞時は学習/推論コードを **MIT ライセンス**で提供する義務がある（Competition-Specific: Winner License = MIT, Sec. 2.8）。

> データ・モデル等の生成物は `data/` `models/` に置く。これらは Git 追跡対象外。

## ディレクトリ構成

```
.
├── src/         # 学習・推論・環境のソース
├── configs/     # 実験設定（YAML 等）
├── notebooks/   # 試行用ノートブック
├── tests/       # テスト
├── data/        # 競技データ（Git 追跡外・終了後に削除）
├── models/      # 学習済みモデル（Git 追跡外）
└── docs/        # ルール等のドキュメント
```

## 環境構築

cabt Engine は Python + 標準ライブラリだけで動く（**GPU 不要・CPU で動作**）。
日常開発（lint / test / 自己対戦）はホストで `make` 経由で行う:

```bash
make deps     # 初回: venv(.venv) を作成し依存をインストール
make smoke    # cabt Engine の疎通確認（ランダム同士）
make bench    # baseline 評価（ヒューリスティック vs ランダム）
make check    # lint + フォーマット差分 + test
```

### GPU / Docker（Phase 3 の深層RL学習で使用）

GPU を使う深層RL学習（Phase 3・任意）でのみ Docker を使う。
前提: WSL2 + NVIDIA GPU + NVIDIA Container Toolkit。

```bash
# 1. 環境変数を用意（値はハードコードせず .env で管理）
cp .env.example .env

# 2. イメージをビルド
docker compose build

# 3. 開発シェルに入る
docker compose run --rm dev bash

# GPU が見えるか確認
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# Jupyter Lab を使う場合
docker compose up jupyter   # http://localhost:${JUPYTER_PORT}
```

設定値（イメージ名・GPU 数・ポート・パス等）はすべて `.env` で変更する。

## ワークフロー（運用）

2軸を回しながら、たまに評価する。`make help` で全コマンド一覧。

### デッキ軸（CPU・ratchet）

「best 起点で探索 → 確定試合数のゲートで“確実に良くなった時だけ”採用」を回し、
`models/champion_best.csv` を単調改善させる（ノイズで劣化しない）。**提出は best を使う**。

```bash
make ratchet                  # 1サイクル: 探索→ゲート→改善だけ採用（既定 約1.5h・CPU）
make ratchet RATCHET_ITERS=1  # 時短（約50分）
make ratchet-overnight        # 一晩版（探索多め・約6h）
make ratchet-nn               # NN操縦の高速版（探索=蒸留NN-MCTS・Docker／判定=ISMCTS）
```
- `→ 更新`＝本物の前進、`据え置き`＝ノイズ劣化を阻止（正常）。
- 探索(league)は速い5メタ・判定(gate)は多様 gauntlet で過学習を防ぐ。時間は `RATCHET_ITERS` で調整。
- **`ratchet-nn`** は探索の操縦を蒸留 NN-MCTS にする（速度ではなく**質**の改善）。現行 ratchet の ISMCTS は
  速度優先で弱い設定（tb=0.03）なので評価が雑。NN-MCTS は特性/効果を扱える強い評価者で、同等強度の
  ISMCTS より時間効率が良い（≈1/4）。NN を `improve` で強くするほど探索の質が上がるので、
  **NN が ISMCTS を安定して超えたら ratchet-nn を主軸**にする。判定(gate)は独立性のため ISMCTS のまま。

### NN軸（GPU＋CPU・distill → improve）

操縦 NN を **2段**で育てる。蒸留(`distill`)で **ISMCTS 相当の床**を作り、improve(`improve`)で
**self-play により床を超える**。蒸留だけでは教師(ISMCTS)が天井（≒五分）で止まるため、超えるには improve が要る。

> ⚠️ **distill は最初の土台作りの一度きり**（NN を壊した時の作り直し用）。improve で ISMCTS を超えた後に
> distill へ戻すと、教師=ISMCTS の天井(五分)へ NN を**引き戻して improve の成果を消す**。だから
> 改善ループは **`distill`↔`improve` ではなく `improve`↔`ratchet-nn`**（NN強化↔強いNNでデッキ探索）。

```bash
# 1段目（最初に一度）: ISMCTS を教師に蒸留＝特性/効果を扱える安定した土台（pvnet_distill_best.pt）
make distill-1h           # 約1h
make distill-overnight    # 一晩・強い教師で多め
# 2段目（以後ずっと）: 蒸留ネットを種に self-play で ISMCTS 超えを狙う（pvnet_improve_best.pt）
make improve-1h           # 約1h・前回に継ぎ足し（種は初回のみ distill_best から）
make improve              # 既定 iters40
# 作り直したい時のみ: rm models/pvnet_distill.pt（または pvnet_improve.pt）
```
- **なぜ improve で超えられるか**: MCTS(NN) は NN 単体より強い「方策改善演算子」。その**訪問回数分布
  (soft-π)** を教師に学ぶと NN が自分を上回っていく（AlphaZero の核）。以前 self-play が崩壊したのは
  **弱いネット始点**が原因で、≈ISMCTS の蒸留ネットを種にすれば回避できる（＋best保存＋低LR＋replay）。
- 収集は **CPU 並列**（`DISTILL_WORKERS`・既定 `nproc-1`）＝**1回あたりの試行数がほぼコア数倍**。
  NN-MCTS は batch=1 推論＋cgエンジンで CPU 寄りなので、GPU より CPU 並列が効く。
- `--resume` で継ぎ足し蓄積。`best` の基準勝率は `*.meta.json` に保存して run を跨いで引き継ぐので、
  **1h を細かく回しても best が単調に良くなる**（劣化モデルで best を上書きしない）。
- 超えたかの確定判断: `make eval-net EVAL_VS=ismcts EVAL_ARGS="--net models/pvnet_improve_best.pt"`。

### 評価（たまに・確定判断）

```bash
make eval-deck                                  # 現 champion vs 相手プール（ISMCTS・多めの試合）
make eval-deck EVAL_DECK=models/champions/champ_XXXX.csv  # 任意のデッキ
make eval-net EVAL_ARGS="--net models/pvnet_distill_best.pt"  # NN の強さ（vs heuristic）
```
学習中の少試合 eval は運の振れが大きい。**判断は 40 試合以上**で行う。

### 相手プール（gauntlet）

5枚のサンプルメタだけで判定すると過学習して実戦で弱くなる。多様な相手に堅くするため
ガントレットを生成しておくと、**判定側**（`eval-deck`/`champion-gate`）が自動でそれを相手に使う
（探索する `league` は速度優先で5メタのまま。過学習の歯止めは判定側で掛ける）。

```bash
make gauntlet             # models/gauntlet/ を生成（メタ＋チャンピオン系＋全色mono-type）
```
チャンピオンが何度か更新されたら作り直す。

### 提出

```bash
cp models/champion_best.csv models/champion_deck.csv
make submission           # models/submission.tar.gz を作成 → Kaggle へはユーザーが提出
```

### 注意

- `ratchet`/`distill`/`improve` はいずれも CPU を多コア使うので**同時に1つだけ**回す。
- gauntlet を入れると相手が増え評価/探索は遅くなる（その分、過学習しにくい）。
- 長時間の学習・探索はユーザーが手動実行（`CLAUDE.md` 参照）。

### 全体の回し方（まとめ）

1. **最初に一度** `make distill` で NN の床（≈ISMCTS）を作る（pvnet_distill_best.pt）。
2. **改善ループ（交互）**: `make improve`（NN 強化）↔ デッキ探索。デッキ探索の操縦は NN の強さで使い分ける:
   - **NN が ISMCTS 超え未確認のうち**は `make ratchet`（ISMCTS 探索＝無印）。
   - **`eval-net EVAL_VS=ismcts` で NN > ISMCTS を確認したら** `make ratchet-nn`（NN 探索）へ切替。
     `ratchet-nn` は強い NN（improve_best）を自動使用。
   - ※ improve 後に distill へ戻さないこと（教師=ISMCTS の天井へ引き戻すため）。
3. 判定: `make eval-net EVAL_VS=ismcts`（NN が ISMCTS 超えたか）、`make eval-deck`（デッキ）。いずれも 40 試合以上。
4. どちらの ratchet も**判定(gate)は ISMCTS**（独立判定）なので、操縦に関わらず `champion_best` は劣化しない。
5. NN が ISMCTS 超え＆heuristic 超を安定したら、提出操縦も NN に寄せる。

## 開発メモ

- 対戦は Kaggle 提供の **cabt Engine** 上で実行（提供物は Competition Data のため追跡外）。
- 手法は **ISMCTS** を中核に段階構築（詳細は `CLAUDE.md`）。
- 整形/Lint: `ruff` / テスト: `pytest`（いずれも `make` 経由・ホスト実行）。
- 深層RL学習（Phase 3・任意）でのみ PyTorch / GPU / Docker を使う。
