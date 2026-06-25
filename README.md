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
make ratchet              # 1サイクル: 探索→ゲート→改善だけ採用（CPU・gauntletありで約2h）
make ratchet-overnight    # 一晩版（じっくり探索）
```
- `→ 更新`＝本物の前進、`据え置き`＝ノイズ劣化を阻止（正常）。
- 1h しか取れない時は分割: `make league-explore-1h`（探索のみ）→ 後で `make champion-gate GATE_GAMES=20`。

### NN軸（GPU＋CPU・distill）

ISMCTS を教師に蒸留し、特性/効果を扱える操縦 NN を育てる（`--resume` で継ぎ足し蓄積）。

```bash
make distill-1h           # 約1h・前回に継ぎ足し（日中ちょくちょく）
make distill-overnight    # 一晩・強い教師で多め
# 作り直したい時: rm models/pvnet_distill.pt
```

### 評価（たまに・確定判断）

```bash
make eval-deck                                  # 現 champion vs 相手プール（ISMCTS・多めの試合）
make eval-deck EVAL_DECK=models/champions/champ_XXXX.csv  # 任意のデッキ
make eval-net EVAL_ARGS="--net models/pvnet_distill_best.pt"  # NN の強さ（vs heuristic）
```
学習中の少試合 eval は運の振れが大きい。**判断は 40 試合以上**で行う。

### 相手プール（gauntlet）

5枚のサンプルメタだけに最適化すると過学習して実戦で弱くなる。多様な相手に堅くするため
ガントレットを生成しておくと、`league`/`eval-deck`/`champion-gate` が自動でそれを相手に使う。

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

- `ratchet`（CPU）と `distill` の ISMCTS 収集（CPU）は競合するので**片方ずつ**。
- gauntlet を入れると相手が増え評価/探索は遅くなる（その分、過学習しにくい）。
- 長時間の学習・探索はユーザーが手動実行（`CLAUDE.md` 参照）。

## 開発メモ

- 対戦は Kaggle 提供の **cabt Engine** 上で実行（提供物は Competition Data のため追跡外）。
- 手法は **ISMCTS** を中核に段階構築（詳細は `CLAUDE.md`）。
- 整形/Lint: `ruff` / テスト: `pytest`（いずれも `make` 経由・ホスト実行）。
- 深層RL学習（Phase 3・任意）でのみ PyTorch / GPU / Docker を使う。
