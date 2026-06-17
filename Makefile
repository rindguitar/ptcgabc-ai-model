# PTCG AI Battle Challenge - 開発用コマンドランナー
#
# 使い方: `make <ターゲット>`。引数なし（`make`）でターゲット一覧を表示する。
# ほとんどのターゲットはホスト（WSL）側で実行し、Docker コンテナを起動して処理する。
# 設定値は .env で管理され docker compose が読み込むため、ここでは再定義しない。
#
# 注意: Makefile のレシピ行は「タブ」インデント必須（スペース不可）。

# --- 設定 -------------------------------------------------------------------
# docker compose の起動コマンド。環境により `docker-compose` を使う場合は上書き可:
#   make COMPOSE=docker-compose test
COMPOSE ?= docker compose

# 使い捨てコンテナで dev サービスのコマンドを実行するための共通プレフィックス。
RUN := $(COMPOSE) run --rm dev

# .DEFAULT_GOAL を help にして、引数なし実行でヘルプを出す。
.DEFAULT_GOAL := help

# 実ファイルと衝突しないよう全ターゲットを PHONY 指定。
.PHONY: help build rebuild shell jupyter gpu-check lint format fmt-check test \
        check up down clean

# --- ヘルプ -----------------------------------------------------------------
help: ## このヘルプを表示
	@echo "PTCG AI - make ターゲット一覧:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- イメージ ---------------------------------------------------------------
build: ## Docker イメージをビルド
	$(COMPOSE) build

rebuild: ## キャッシュを使わずイメージを再ビルド
	$(COMPOSE) build --no-cache

# --- 開発シェル / ノートブック ---------------------------------------------
shell: ## 開発用コンテナで bash を起動
	$(RUN) bash

jupyter: ## Jupyter Lab を起動 (http://localhost:${JUPYTER_PORT})
	$(COMPOSE) up jupyter

gpu-check: ## コンテナから GPU（CUDA）が見えるか確認
	$(RUN) python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# --- Lint / Format / Test ---------------------------------------------------
lint: ## ruff で Lint
	$(RUN) ruff check .

format: ## ruff でフォーマット（ファイルを書き換える）
	$(RUN) ruff format .

fmt-check: ## フォーマット差分のチェックのみ（書き換えない / CI 向け）
	$(RUN) ruff format --check .

test: ## pytest を実行
	$(RUN) pytest

# まとめて品質チェック（Lint + フォーマット差分 + テスト）。
check: lint fmt-check test ## lint・fmt-check・test を順に実行

# --- 後片付け ---------------------------------------------------------------
up: ## dev サービスをバックグラウンド起動
	$(COMPOSE) up -d dev

down: ## compose で起動したコンテナを停止・削除
	$(COMPOSE) down

clean: ## __pycache__ や .pytest_cache などの生成物を削除（data/ models/ は触らない）
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} + 2>/dev/null || true
