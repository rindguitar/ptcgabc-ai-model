# CLAUDE.md

このファイルは Claude Code（および将来の自分）がこのリポジトリで作業するためのガイドです。
詳細なセットアップ手順や規約の引用は [README.md](README.md) を参照してください。

## プロジェクト概要

Kaggle「The Pokémon Company - PTCG AI Battle Challenge Strategy」（Simulation division）向けの、
ポケモンカードゲーム（PTCG）対戦 AI を開発する**非公開**リポジトリ。
最終的な提出物は対戦戦略を行うモデル／エージェントを想定（強化学習を前提に `gymnasium` を同梱）。

- 競技ページ: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy

## ⚠️ 規約上の遵守事項（作業前に必ず確認）

ルール全文は [docs/ptcgabc_rules.txt](docs/ptcgabc_rules.txt)。特に以下を**ハードに守る**こと:

- **Competition Data（`data/` 配下）は再配布禁止**（Sec. 2.4）。参加者以外がアクセスできる場所に
  置かない。リポジトリは非公開、`data/` と `models/` は `.gitignore` 済み。
- **`data/` の中身を絶対にコミット・出力・公開しない。** CSV/PDF の生データ、カード名・ワザ名・効果文・
  画像等の **Pokémon Elements**（Sec. 3.18.f）をコード、コミット、Issue、PR、ログ、ノートブック出力に
  貼り付けない。サンプルやテストのフィクスチャにも実データを混ぜない（ダミー値を使う）。
- **競技終了後は Competition Data を速やかに削除**する義務がある（Sec. 2.4）。
  → データはホスト側 `./data` にのみ保持し、削除しやすい構成を維持する。
- **私的なコード共有はチーム外で禁止**（Sec. 3.6）。公開する場合は Kaggle フォーラム上で
  OSI 承認ライセンスのもとに行う。
- 学習で使う **External Data / モデルは「全参加者が無償・容易にアクセス可能」**であること（Sec. 2.6）。
- Pokémon Elements を競技外の用途・商用・競合製品に使わない（Sec. 3.18.f）。

> Claude への指示: `data/` 配下のファイル内容を会話やコミットにそのまま展開しないこと。
> スキーマ確認のための最小限のヘッダ参照に留め、具体的なカード情報の貼り付けは避ける。

## ディレクトリ構成

```
.
├── src/         # 学習・推論・対戦環境のソース（パッケージは PYTHONPATH=/workspace で解決）
├── configs/     # 実験設定（YAML / OmegaConf 想定）
├── notebooks/   # 試行用ノートブック（出力に実データを残さない）
├── tests/       # pytest
├── data/        # 競技データ（Git 追跡外・終了後に削除）
├── models/      # 学習済みモデル（Git 追跡外）
└── docs/        # ルール等のドキュメント
```

現状 `src/` は `__init__.py` のみ。コア実装（カードデータのパーサ、対戦環境、エージェント、学習ループ）は未着手。

## データ（`data/`・追跡外）

- `EN_Card_Data.csv` / `JP_Card_Data.csv`: 各 2102 行 ＝ **ユニーク 1267 枚**（Card ID 1–1267、同一 ID で EN/JP 対応）。
  1 枚が複数ワザ／特性を持つと複数行になるため「行数 ≠ 枚数」。エンジンの `AllCard` の 1267 枚と一致＝これが大会リーガルな全カードプール（デッキ構築の探索空間）。
- `Card_ID List_EN.pdf` / `Card_ID List_JP.pdf`: カード ID 一覧（大容量）。
- CSV カラム（EN）:
  `Card ID, Card Name, Expansion, Collection No., Stage (Pokémon)/Type (Energy and Trainer),
   Rule, Category, Previous stage, HP, Type, Weakness, Resistance (Type), Retreat,
   Move Name, Cost, Damage, Effect Explanation`
- 1 枚のカードが複数ワザ／特性を持つ場合、**同一 Card ID で複数行**に分かれる（Move Name 単位）。
  パース時は Card ID でグルーピングが必要。
- エネルギー種別やコストは `{G}{R}{W}{L}...` のシンボル、コストの追加分は `●` で表記される。
- 欠損は `n/a` または空文字。両方を欠損として扱う。

## 環境・コマンド

cabt Engine は Python + 標準ライブラリだけで動く（**GPU 不要・CPU で動作**）。日常開発
（lint / test / 自己対戦）はホストで `make` 経由で行う。GPU/Docker は Phase 3 の深層RL学習でのみ使う。

```bash
make deps    # 初回: venv(.venv) を作成し依存をインストール
make smoke   # cabt Engine 疎通確認（ランダム同士・src/harness.py）
make bench   # baseline 評価（ヒューリスティック vs ランダム）
make check   # lint + フォーマット差分 + test
```

### GPU / Docker（Phase 3 用）

前提: WSL2 + NVIDIA GPU（GTX 1060 / Pascal 想定）+ NVIDIA Container Toolkit。
設定値はすべて `.env`（`cp .env.example .env`）で管理し、**ハードコードしない**。

```bash
make build                 # イメージビルド（時間がかかる→ユーザー実行）
make shell                 # 開発用コンテナで bash
make gpu-check             # コンテナから CUDA を確認
make exec CMD="..."        # 任意コマンドをコンテナ内で実行
```

コンテナ内では `WORKDIR=/workspace`、`PYTHONPATH=/workspace`（`src` を import 可能）。
`data/` `models/` はホストから bind mount される。

### 開発環境セットアップ

- **Make 優先**: Docker 操作・開発タスクは `docker compose` 直叩きでなく `make` 経由で行う（一貫性）。
- **時間のかかる操作はユーザーが手動実行**: `make build`/完全再ビルド（`make down && make build && make up`）/大量データ処理。Claude は実行せず「再ビルドが必要」と通知する。
- **Claude の役割**: コード実装・編集、`requirements.txt` 更新、短時間テスト（`make exec CMD="..."` 等）。

### Lint / Format / Test

```bash
ruff check .        # Lint
ruff format .       # フォーマット
pytest              # テスト
```

## 開発方針・規約

- フレームワーク: PyTorch（ベースイメージに同梱、`requirements.txt` には書かない）。
- 設定は `omegaconf` + `configs/*.yaml`。実験ログは `tensorboard`（`runs/` は追跡外）。
- 秘匿値・ローカル設定は `.env` のみ。コードに値を直書きしない。本物の秘密鍵が含まれていないか慎重にチェック API キー、トークン、パスワードなどの実際の値が含まれていないか 疑わしい場合はコミットせず、チームで相談。
- 新規依存は `requirements.txt` に追加（PyTorch 系は除く）。
- 受賞時は学習・推論コードを **MIT** で提供する義務があるため、OSI 非互換の依存を避ける。
- 新機能追加、既存機能のバグ修正など大きな変更を加える場合は、新ブランチを作成すること。
- 新しいブランチ作成時のルール ブランチを作成したら、作業開始前に必ずリモートにプッシュすること
- **コミット/プッシュの順序を厳守**: 実装 → テスト（成功確認）→ コミット → プッシュ（コミットとセットで必ず push）。
  - ❌ テストせずにコミット/プッシュ　❌ コミットだけして push 忘れ

### コメント・ドキュメントの日本語化
- **コメント・docstring は日本語**で記述。変数名・関数名は英語（Python 慣例）。
- 技術用語は無理に訳さず英語/カタカナ（accuracy, precision, recall, F1 score, batch, pipeline, model, dataset 等）。
- 例: `def test_perfect_predictions()` → `"""perfect predictions テスト"""`（×「完璧な予測のテスト」のようなぎこちない訳）。
