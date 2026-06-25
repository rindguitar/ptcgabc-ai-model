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
        league-explore-1h league-explore-overnight ratchet ratchet-overnight \
        gauntlet eval-deck champion-gate \
        train distill distill-1h distill-overnight eval-net \
        submission build rebuild shell jupyter gpu-check exec up down clean

# --- デッキ探索（ratchet が内部で使う）の既定パラメータ --------------------
# 操縦は ISMCTS（特性/効果/トレーナーを扱える）。相手は models/gauntlet/ があればそれ（多様化）。
# 各反復でチェックポイント保存＝止めても再開可。LEAGUE_SEED は実行ごとにランダム（毎回別探索）。
LEAGUE_PILOT      ?= ismcts
LEAGUE_TIMEBUDGET ?= 0.05
_PILOT_FLAGS       = --pilot $(LEAGUE_PILOT) --time-budget $(LEAGUE_TIMEBUDGET)
LEAGUE_SEED    ?= $(shell python3 -c 'import random;print(random.randrange(2**31))')
LEAGUE_ARGS    ?=
# 既存チャンピオンを固定の試験官(seed)に自動取り込み（探索の起点・無ければ相手プールのみ）。
LEAGUE_EXTRA   ?= $(wildcard models/champion_deck.csv)
_EXTRA_FLAG     = $(if $(LEAGUE_EXTRA),--extra-seeds $(LEAGUE_EXTRA),)

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

# === デッキ探索（ratchet が内部で使用・通常は ratchet 経由で呼ぶ） ==========
# 出力: models/champion_deck.csv / チェックポイント: models/league/state.json（追跡外）。
# 相手は models/gauntlet/ があればそれ（多様化）。止めても次回 league（同じ）で resume。
league-explore-1h: ## デッキ探索 約1時間（ratchet が使用・単体実行も可）
	$(PY) src/league.py --cap 10 --iters 6 --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)

league-explore-overnight: ## デッキ探索 一晩（約6h・ratchet-overnight が使用）
	$(PY) src/league.py --cap 12 --iters 10 --games 6 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)

# === Phase 3 NN（Docker・torch/GPU）=========================================
# 学習デッキ群: チャンピオン＋メタを巡回（1デッキ過学習を避け汎用 pilot 化）。先頭=評価固定。
TRAIN_DECKS ?= data/deck.csv $(wildcard models/champion_deck.csv) $(wildcard data/*_Deck.csv)
_DECKS_FLAG  = --deck $(TRAIN_DECKS)
TRAIN_ARGS  ?=
train: ## NN生 self-play（上級者向け・例: make train TRAIN_ARGS="--resume --iterations 50")
	$(RUN) python scripts/train_alphazero.py $(_DECKS_FLAG) $(TRAIN_ARGS)

# 蒸留: ISMCTS 教師を真似て安定した強い土台を作る。複数デッキ巡回＝汎用 pilot 化。
# self-play(pvnet.pt)とは別ファイルに保存し --resume で「ちょくちょく継ぎ足し」できる（日中1h×複数回で蓄積）。
# 教師は強いほど良い(DISTILL_TB↑)が収集は遅い。最初から作り直すなら rm models/pvnet_distill.pt。
# 評価は: make eval-net EVAL_ARGS="--net models/pvnet_distill_best.pt"
DISTILL_OUT   ?= models/pvnet_distill.pt
DISTILL_BEST  ?= models/pvnet_distill_best.pt
DISTILL_TB    ?= 0.25
DISTILL_ITERS ?= 60
DISTILL_ARGS  ?=
distill: ## ISMCTS蒸留（複数デッキ・強い教師・resume継ぎ足し）。長さは DISTILL_ITERS で
	$(RUN) python scripts/train_alphazero.py --teacher ismcts --resume \
		--out $(DISTILL_OUT) --best-out $(DISTILL_BEST) --teacher-time-budget $(DISTILL_TB) \
		--iterations $(DISTILL_ITERS) --eval-every 20 --eval-games 24 \
		$(_DECKS_FLAG) $(DISTILL_ARGS)

distill-1h: ## 約1時間の蒸留（前回に継ぎ足し・日中ちょくちょく用）
	$(MAKE) distill DISTILL_ITERS=50

distill-overnight: ## 一晩の蒸留（継ぎ足し・強い教師0.3で多め・約7h目安）
	$(MAKE) distill DISTILL_ITERS=350 DISTILL_TB=0.3

# 確定判断用の offline 評価（試合数を増やして運の振れを抑える）。学習中の24は傾向把握用、
# こちらは 40+ で「NN は heuristic/ISMCTS を超えたか」を判断する。例: make eval-net EVAL_GAMES=100
EVAL_GAMES ?= 40
EVAL_VS    ?= heuristic
EVAL_ARGS  ?=
eval-net: ## 訓練済みNNの確定判断用 評価（既定: vs heuristic 40試合・Docker）
	$(RUN) python scripts/eval_net.py --vs $(EVAL_VS) --games $(EVAL_GAMES) $(EVAL_ARGS)

# 多様ガントレット生成（過学習対策）。生成後は league/eval-deck/gate が自動でこれを相手に使う
# （models/gauntlet/ があれば data/*.csv より優先）。相手が多彩になる分、評価/探索は遅くなる。
gauntlet: ## 多様な相手デッキ群 models/gauntlet/ を生成（メタ＋チャンピオン系＋全色mono-type）
	$(PY) scripts/make_gauntlet.py

# デッキ強さの確定評価（vs 相手プール・ISMCTS操縦・多めの試合）。league内部の小サンプル(6試合)では
# 判定できない「本当にデッキが強くなったか」を測る。ホスト(CPU)。champions/ のバックアップと比較可。
EVAL_DECK       ?= models/champion_deck.csv
# 既定40試合: 20では運の振れで誤判断する（実測 worst 0.35→0.625 と激変）ため確定判断は40+。
EVAL_DECK_GAMES ?= 40
EVAL_DECK_ARGS  ?=
eval-deck: ## デッキ強さの確定評価（vs メタ・ISMCTS・既定 champion 20試合・ホスト）
	$(PY) scripts/eval_deck.py --deck $(EVAL_DECK) --games $(EVAL_DECK_GAMES) $(EVAL_DECK_ARGS)

# 信頼ラチェット: league 後に挟むと、新チャンピオンが best を信頼試合数で上回った時だけ昇格。
# ノイズドリフトを止め、回し続けるほど models/champion_best.csv が単調に良くなる。提出は best を使う。
# 既定40試合: 20では判断がノイズに飲まれる（worst が ±0.2 振れる）ため keep-best は40+で判定。
GATE_GAMES ?= 40
GATE_ARGS  ?=
champion-gate: ## league 後の keep-best 判定（新が best を上回った時だけ昇格・ホスト）
	$(PY) scripts/champion_gate.py --games $(GATE_GAMES) $(GATE_ARGS)

# 1サイクル自動化: best を起点に探索→確実改善だけ採用。回し続けるほど champion_best が単調改善。
# 探索の長さは変更可: make ratchet RATCHET_LEAGUE=league-explore-overnight
RATCHET_LEAGUE ?= league-explore-1h
ratchet: ## 探索→40試合ゲートを1発（best起点・確実改善だけ採用・全ホストCPU）
	@if [ -f models/champion_best.csv ]; then \
		cp models/champion_best.csv models/champion_deck.csv; \
		echo "起点を champion_best に設定（best から探索）"; \
	else echo "best 未作成: 現 champion_deck から開始（gate が初回 best を作成）"; fi
	$(MAKE) $(RATCHET_LEAGUE)
	$(MAKE) champion-gate
	@echo "ratchet 完了。最良は models/champion_best.csv（提出はこれを使う）"

ratchet-overnight: ## 一晩版 ratchet（league-explore-overnight→40試合ゲート・best起点）
	$(MAKE) ratchet RATCHET_LEAGUE=league-explore-overnight

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
