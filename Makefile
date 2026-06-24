# PTCG AI Battle Challenge - 開発用コマンドランナー
#
# 使い方: `make <ターゲット>`。引数なし（`make`）でターゲット一覧を表示する。
#
# 住み分け:
#   - 軽い作業（lint/test/smoke）はホストの Python で直接実行（cabt Engine は
#     標準ライブラリだけで動くため Docker 不要・高速）。
#   - 重い作業（イメージビルド / GPU 学習 = Phase 3）は Docker で実行。
# 設定値は .env で管理され docker compose が読み込むため、ここでは再定義しない。
#
# 注意: Makefile のレシピ行は「タブ」インデント必須（スペース不可）。

# --- 設定 -------------------------------------------------------------------
# ホスト開発は venv(.venv) に閉じる（システム Python を汚さない / PEP 668 回避）。
# .venv があればその python を、無ければシステム python3 にフォールバックする。
VENV    ?= .venv
PY      := $(shell [ -x $(VENV)/bin/python ] && echo $(VENV)/bin/python || echo python3)
COMPOSE ?= docker compose
# 使い捨てコンテナで dev サービスのコマンドを実行するための共通プレフィックス。
RUN     := $(COMPOSE) run --rm dev

.DEFAULT_GOAL := help
.PHONY: help deps lint format fmt-check test smoke bench check \
        league league-resume league-overnight league-1h \
        league-explore league-explore-1h league-explore-overnight submission \
        train train-1h train-overnight distill eval-net \
        build rebuild shell jupyter gpu-check exec up down clean

# --- デッキリーグの既定パラメータ（make 変数で上書き可） --------------------
# 例: make league LEAGUE_GAMES=8 LEAGUE_ITERS=8
# LEAGUE_ARGS は任意の追加フラグ（例: LEAGUE_ARGS=--resume）。
# 評価操縦は ISMCTS（特性/効果/トレーナーを扱える）。heuristic より桁違いに遅いので
# games/pop/gens は小さめが既定。各反復でチェックポイント保存＝時間が来たら止めて resume 可。
LEAGUE_PILOT      ?= ismcts
LEAGUE_TIMEBUDGET ?= 0.05
_PILOT_FLAGS       = --pilot $(LEAGUE_PILOT) --time-budget $(LEAGUE_TIMEBUDGET)
LEAGUE_CAP     ?= 10
LEAGUE_ITERS   ?= 6
LEAGUE_GAMES   ?= 4
LEAGUE_POP     ?= 6
LEAGUE_GENS    ?= 3
# 既定は実行ごとにランダム（素のコマンドを繰り返すと別探索→incumbent が最良を保持し世代更新）。
# 再現したいときは make league LEAGUE_SEED=0 のように固定する。
LEAGUE_SEED    ?= $(shell python3 -c 'import random;print(random.randrange(2**31))')
LEAGUE_PLATEAU ?= 4
LEAGUE_ARGS    ?=
# 既存チャンピオンがあれば自動で固定の試験官(seed)に取り込む（無ければメタのみ）。
# メタのみで回したいときは make league LEAGUE_EXTRA= で解除。
LEAGUE_EXTRA   ?= $(wildcard models/champion_deck.csv)
_EXTRA_FLAG     = $(if $(LEAGUE_EXTRA),--extra-seeds $(LEAGUE_EXTRA),)
# 探索パラメータ（既定OFF＝従来の1軸refine）。league/league-resume で変数指定すれば探索ON。
# 例: make league LEAGUE_MAXSWAPS=12 LEAGUE_EXPLORE=0.3
LEAGUE_MAXSWAPS ?= 1
LEAGUE_EXPLORE  ?= 0
_EXPLORE_FLAGS   = --max-swaps $(LEAGUE_MAXSWAPS) --explore $(LEAGUE_EXPLORE)

# --- ヘルプ -----------------------------------------------------------------
help: ## このヘルプを表示
	@echo "PTCG AI - make ターゲット一覧:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# === ホスト実行（軽い作業・高速） ==========================================
deps: ## venv(.venv) を作成し依存をインストール（初回/更新時）
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

lint: ## ruff で Lint（ホスト）
	$(PY) -m ruff check .

format: ## ruff でフォーマット（ホスト・書き換える）
	$(PY) -m ruff format .

fmt-check: ## フォーマット差分のチェックのみ（書き換えない / CI 向け）
	$(PY) -m ruff format --check .

test: ## pytest を実行（ホスト）
	$(PY) -m pytest

smoke: ## cabt Engine 疎通確認（ランダム同士・20試合）
	$(PY) src/harness.py --a random --b random --games 20 --seed 0

bench: ## baseline 評価（ヒューリスティック vs ランダム・100試合）
	$(PY) src/harness.py --a heuristic --b random --games 100 --seed 0

# まとめて品質チェック（Lint + フォーマット差分 + テスト）。
check: lint fmt-check test ## lint・fmt-check・test を順に実行

# === デッキリーグ（bounded double-oracle） ================================
# 出力: models/champion_deck.csv / チェックポイント: models/league/state.json（追跡外）
# 操縦は ISMCTS（既定）。time は目安・ゲーム長で変動。各反復で保存＝途中で止めて league-resume 可。
league: ## ISMCTSリーグ実行（探索: LEAGUE_EXPLORE=0.3 LEAGUE_MAXSWAPS=12 / heuristicに戻す: LEAGUE_PILOT=heuristic）
	$(PY) src/league.py --cap $(LEAGUE_CAP) --iters $(LEAGUE_ITERS) \
		--games $(LEAGUE_GAMES) --pop $(LEAGUE_POP) --gens $(LEAGUE_GENS) \
		--plateau $(LEAGUE_PLATEAU) --seed $(LEAGUE_SEED) \
		$(_PILOT_FLAGS) $(_EXPLORE_FLAGS) $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-resume: ## チェックポイントから続行（make league-resume LEAGUE_ITERS=4）
	$(PY) src/league.py --cap $(LEAGUE_CAP) --iters $(LEAGUE_ITERS) \
		--games $(LEAGUE_GAMES) --pop $(LEAGUE_POP) --gens $(LEAGUE_GENS) \
		--plateau $(LEAGUE_PLATEAU) --seed $(LEAGUE_SEED) --resume \
		$(_PILOT_FLAGS) $(_EXPLORE_FLAGS) $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-overnight: ## ISMCTS一晩プリセット（約6h目安: cap12/iters10/games6/pop6/gens3・止めたら resume）
	$(PY) src/league.py --cap 12 --iters 10 --games 6 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-1h: ## ISMCTS約1時間プリセット（cap10/iters6/games4/pop6/gens3）
	$(PY) src/league.py --cap 10 --iters 6 --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-explore: ## ISMCTS多軸探索（変数で調整・大変異＋多様性注入・非弱化維持）
	$(PY) src/league.py --cap $(LEAGUE_CAP) --iters $(LEAGUE_ITERS) --games $(LEAGUE_GAMES) \
		--pop $(LEAGUE_POP) --gens $(LEAGUE_GENS) --plateau 99 --seed $(LEAGUE_SEED) \
		$(_PILOT_FLAGS) --max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-explore-1h: ## ISMCTS約1時間の探索プリセット（別アーキタイプ/カードを広く探す）
	$(PY) src/league.py --cap 10 --iters 6 --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-explore-overnight: ## ISMCTS一晩の探索プリセット（約6h目安・別アーキタイプを広く探す）
	$(PY) src/league.py --cap 12 --iters 10 --games 6 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)

# === Phase 3 学習（Docker・torch/GPU） =====================================
# 反復回数は時間の目安（実測 約30秒/反復・GPU）。ゲーム長やGPUで変動するので調整可。
TRAIN_ARGS       ?=
TRAIN_ITERS_1H   ?= 100
TRAIN_ITERS_NIGHT ?= 800
train: ## AlphaZero 反復学習（Docker）。例: make train TRAIN_ARGS="--iterations 20 --games 16"
	$(RUN) python scripts/train_alphazero.py $(TRAIN_ARGS)

train-1h: ## 約1時間の継続学習（resume＝iter300等から続行・vs heuristic評価付き）
	$(RUN) python scripts/train_alphazero.py --resume --iterations $(TRAIN_ITERS_1H) \
		--eval-every 50 --eval-games 24 $(TRAIN_ARGS)

train-overnight: ## 一晩の継続学習（約7h・resume で続行・評価付き）
	$(RUN) python scripts/train_alphazero.py --resume --iterations $(TRAIN_ITERS_NIGHT) \
		--eval-every 100 --eval-games 24 $(TRAIN_ARGS)

# 蒸留: ISMCTS 教師を真似て安定した強い土台を作る（自己対戦の崩壊を回避）。ISMCTS収集は遅い。
# 既存 pvnet.pt から続けたくない場合は DISTILL_ARGS に出力先を指定。例: DISTILL_ARGS="--out models/pvnet_distill.pt"
DISTILL_ITERS ?= 60
DISTILL_ARGS  ?=
distill: ## ISMCTS蒸留学習（Docker）。崩壊回避の強い土台作り。best も自動保存
	$(RUN) python scripts/train_alphazero.py --teacher ismcts \
		--iterations $(DISTILL_ITERS) --eval-every 20 --eval-games 24 $(DISTILL_ARGS)

# 確定判断用の offline 評価（試合数を増やして運の振れを抑える）。学習中の24は傾向把握用、
# こちらは 40+ で「NN は heuristic/ISMCTS を超えたか」を判断する。例: make eval-net EVAL_GAMES=100
EVAL_GAMES ?= 40
EVAL_VS    ?= heuristic
EVAL_ARGS  ?=
eval-net: ## 訓練済みNNの確定判断用 評価（既定: vs heuristic 40試合・Docker）
	$(RUN) python scripts/eval_net.py --vs $(EVAL_VS) --games $(EVAL_GAMES) $(EVAL_ARGS)

# === 提出 ==================================================================
submission: ## 提出パッケージ models/submission.tar.gz を作成（champion＋ISMCTS＋cg＋deck）
	$(PY) scripts/build_submission.py

# === Docker 実行（重い作業 / GPU = Phase 3） ===============================
# build は数分〜10 分以上かかるため、原則ユーザーが手動実行する（CLAUDE.md 参照）。
build: ## Docker イメージをビルド（時間がかかる）
	$(COMPOSE) build

rebuild: ## キャッシュを使わずイメージを再ビルド（時間がかかる）
	$(COMPOSE) build --no-cache

shell: ## 開発用コンテナで bash を起動
	$(RUN) bash

jupyter: ## Jupyter Lab を起動 (http://localhost:${JUPYTER_PORT})
	$(COMPOSE) up jupyter

gpu-check: ## コンテナから GPU（CUDA）が見えるか確認
	$(RUN) python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

exec: ## 任意コマンドをコンテナ内で実行。例: make exec CMD="python src/harness.py"
	$(RUN) $(CMD)

up: ## dev サービスをバックグラウンド起動
	$(COMPOSE) up -d dev

down: ## compose で起動したコンテナを停止・削除
	$(COMPOSE) down

# === 後片付け ==============================================================
clean: ## __pycache__ や .pytest_cache などの生成物を削除（data/ models/ は触らない）
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} + 2>/dev/null || true
