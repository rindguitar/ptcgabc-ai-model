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
        ratchet ratchet-overnight ratchet-nn ratchet-nn-overnight gauntlet-real replays replays-prune replays-daily replay-extract replay-tune eval-deck champion-gate \
        train distill distill-1h distill-overnight improve improve-1h eval-net diagnose \
        submission build rebuild shell jupyter gpu-check exec up down clean

# --- デッキ探索（ratchet が内部で使う）の既定パラメータ --------------------
# 探索操縦・相手プール・シード等。仕組みは docs/learning/architecture.md 参照。
LEAGUE_PILOT      ?= ismcts
LEAGUE_TIMEBUDGET ?= 0.03
_PILOT_FLAGS       = --pilot $(LEAGUE_PILOT) --time-budget $(LEAGUE_TIMEBUDGET)
LEAGUE_SEED    ?= $(shell python3 -c 'import random;print(random.randrange(2**31))')
LEAGUE_ARGS    ?=
LEAGUE_EXTRA   ?= $(wildcard models/champion_best.csv)  # best を探索の種に固定取込（継続性）
_EXTRA_FLAG     = $(if $(LEAGUE_EXTRA),--extra-seeds $(LEAGUE_EXTRA),)
# 探索の相手＝実メタからランダム抽出（毎実行で別の SEARCH_SAMPLE 個＝特定デッキへの過適合を回避）。
# gauntlet（実メタ・make gauntlet-real）が無ければ league 既定（公式サンプル）にフォールバック。
SEARCH_SAMPLE ?= 5
SEARCH_SEEDS  := $(shell ls models/gauntlet/*.csv 2>/dev/null | shuf -n $(SEARCH_SAMPLE))
_SEEDS_FLAG    = $(if $(strip $(SEARCH_SEEDS)),--seeds $(SEARCH_SEEDS),)
# NN 操縦の既定ネット（凍結中は operative 運用・§25）。distill 再開時は distill_best 優先。
NN_NET  ?= $(firstword $(wildcard models/pvnet_distill_best.pt) models/pvnet_operative.pt)
NN_SIMS ?= 64
SEARCH_SIMS ?= 32   # 探索の NN-MCTS sims（sims32≈64＝探索専用に半減。eval-deck 判定は 64 のまま）

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

# === Phase 3 NN（Docker・torch/GPU）=========================================
# 訓練デッキ群（混合＝汎用 pilot 化・過学習回避）。先頭=評価固定。構成の意図は design-decisions.md。
TRAIN_DECKS ?= $(wildcard models/champion_repaired.csv) \
	$(firstword $(wildcard models/champion_best.csv) data/deck.csv) \
	$(wildcard data/*_Deck.csv) $(wordlist 1,6,$(sort $(wildcard models/gauntlet/*.csv)))
_DECKS_FLAG  = --deck $(TRAIN_DECKS)
TRAIN_ARGS  ?=
train: ## NN生 self-play（上級者向け・例: make train TRAIN_ARGS="--resume --iterations 50")
	$(RUN) python scripts/train_alphazero.py $(_DECKS_FLAG) $(TRAIN_ARGS)

# 蒸留: ISMCTS 教師を真似て安定した土台を作る（--resume で継ぎ足し・並列収集）。詳細は distill.py。
DISTILL_OUT   ?= models/pvnet_distill.pt
DISTILL_BEST  ?= models/pvnet_distill_best.pt
DISTILL_TB    ?= 0.5   # 教師ISMCTSの1手秒（0.5が最適点＝弱教師回避・design-decisions）
DISTILL_ITERS ?= 60
DISTILL_TEMP  ?= 0     # 方策温度 0=one-hot（>0で soft-π＝深い教師時のみ有効）
DISTILL_WORKERS ?= $(shell n=$$(nproc 2>/dev/null || echo 2); echo $$((n>1?n-1:1)))  # 並列収集数(nproc-1)
DISTILL_ARGS  ?=
distill: ## ISMCTS蒸留（複数デッキ・強い教師・resume継ぎ足し・並列収集）。長さは DISTILL_ITERS で
	$(RUN) python scripts/train_alphazero.py --teacher ismcts --resume \
		--out $(DISTILL_OUT) --best-out $(DISTILL_BEST) --teacher-time-budget $(DISTILL_TB) \
		--distill-temp $(DISTILL_TEMP) \
		--iterations $(DISTILL_ITERS) --workers $(DISTILL_WORKERS) --eval-every 20 --eval-games 24 \
		$(_DECKS_FLAG) $(DISTILL_ARGS)

distill-1h: ## 約1時間の蒸留（前回に継ぎ足し・日中ちょくちょく用）
	$(MAKE) distill DISTILL_ITERS=50

distill-overnight: ## 強い教師で蒸留（大容量netの収束・約4-5h目安）
	$(MAKE) distill DISTILL_ITERS=120

# improve: 蒸留ネットを種に self-play で ISMCTS の天井を破る（CPU並列収集）。詳細は selfplay.py。
IMPROVE_OUT   ?= models/pvnet_improve.pt
IMPROVE_BEST  ?= models/pvnet_improve_best.pt
IMPROVE_SEED  ?= $(firstword $(wildcard models/pvnet_distill_best.pt) models/pvnet_operative.pt)
IMPROVE_ITERS ?= 40
IMPROVE_COLLECT_SIMS ?= 64   # 収集探索深度（sims32≈64で頭打ち・design-decisions）
IMPROVE_COLLECT_BONUS ?= 0   # 収集器への盤面補正（切り分け中は0）
IMPROVE_ARGS  ?=
improve: ## self-playでISMCTS超えを狙う（蒸留ネットを種に・Docker）。長さは IMPROVE_ITERS で
	$(RUN) python scripts/train_alphazero.py --teacher selfplay --resume --resume-from-best \
		--init-from $(IMPROVE_SEED) --out $(IMPROVE_OUT) --best-out $(IMPROVE_BEST) \
		--iterations $(IMPROVE_ITERS) --workers $(DISTILL_WORKERS) --ema \
		--collect-sims $(IMPROVE_COLLECT_SIMS) --collect-board-bonus $(IMPROVE_COLLECT_BONUS) \
		--eval-every 10 --eval-games 24 $(_DECKS_FLAG) $(IMPROVE_ARGS)

improve-1h: ## 約1時間の improve（前回に継ぎ足し・日中ちょくちょく用）
	$(MAKE) improve IMPROVE_ITERS=30

# NN の確定判断用 評価。既定 vs=meta＝実メタ相手プール（非ミラー・§25）。games は相手1体あたり
# ＝プール数×games（16デッキ×10＝160試合目安）。例: EVAL_NET=... / EVAL_VS=ismcts（ミラー確認）。
EVAL_GAMES ?= 10
EVAL_VS    ?= meta
EVAL_NET   ?= $(firstword $(wildcard models/pvnet_distill_best.pt) models/pvnet_operative.pt)
EVAL_ARGS  ?=
eval-net: ## 訓練済みNNの確定判断用 評価（既定: 最良net・vs 実メタ・各10試合・Docker）
	$(RUN) python scripts/eval_net.py --net $(EVAL_NET) --vs $(EVAL_VS) --games $(EVAL_GAMES) \
		--board-bonus $(JUDGE_BONUS) $(EVAL_ARGS)

# NN policy 診断（手を順位付けできるか等の切り分け）。詳細は diagnose_policy.py。
DIAGNOSE_ARGS ?=
diagnose: ## NN policy 診断（手を順位付けできるか等の切り分け・Docker）
	$(RUN) python scripts/diagnose_policy.py --net $(EVAL_NET) $(DIAGNOSE_ARGS)

# Kaggle replay 分析（日次: DL → make replays → 抽出後に JSON削除）。集計＋相手デッキ抽出。
# MY_TEAM を明示＝自分の JSON が無い時に最頻チーム（他人）を自分と誤認する事故を防ぐ。
MY_TEAM      ?= R.I
REPLAYS_ARGS ?=
replays: ## replay 分析（勝率/敗因/時間の集計＋実メタデッキ抽出・冪等・ホスト）
	$(PY) scripts/analyze_replays.py --team "$(MY_TEAM)" $(REPLAYS_ARGS)

# 消費済み JSON の破棄（analyze＋value 両方を通過した others 等・自分の試合は温存）。
PRUNE_ARGS ?=
replays-prune: ## 消費済み replay JSON を破棄（既定 dry-run・--apply で実削除・ホスト）
	$(PY) scripts/prune_replays.py $(PRUNE_ARGS)

# 日次一括: 分析（デッキ収穫）→ value サンプル抽出 → 消費済み JSON を自動破棄。
replays-daily: ## 日次: replays → replay-extract → prune --apply（消費後に自動破棄・ホスト）
	$(MAKE) replays
	$(MAKE) replay-extract
	$(PY) scripts/prune_replays.py --apply $(PRUNE_ARGS)

# リーダーボード上位の replay を data/replays/others/ に置いて実行（チーム別成績・デッキ・時間）。
top-replays: ## 上位チーム replay の分析（others/ 配下・チーム別勝率/デッキ/時間・ホスト）
	$(PY) scripts/analyze_top_replays.py $(TOP_REPLAYS_ARGS)

# 教師方策クローン（§33）: 任意チームの実戦決定を教師に policy を fine-tune。
# 例: make teacher-extract TEACHER_TEAM="<チーム名>" → make teacher-tune TEACHER_SAMPLES=<npz>
TEACHER_TEAM    ?=
TEACHER_SAMPLES ?=
teacher-extract: ## 指定チームの (局面→選択) を抽出（TEACHER_TEAM= 必須・ホスト・冪等）
	@test -n "$(TEACHER_TEAM)" || (echo "TEACHER_TEAM=<チーム名> を指定"; exit 1)
	$(PY) scripts/extract_teacher_samples.py --team "$(TEACHER_TEAM)" $(TEACHER_EXTRACT_ARGS)
teacher-tune: ## 教師の決定で policy を行動クローン（TEACHER_SAMPLES= 必須・Docker）
	@test -n "$(TEACHER_SAMPLES)" || (echo "TEACHER_SAMPLES=<npz> を指定"; exit 1)
	$(RUN) python scripts/teacher_tune.py --samples $(TEACHER_SAMPLES) $(TEACHER_TUNE_ARGS)

# 実戦 replay を value 学習に混ぜる経路（§25）。①抽出（ホスト）→ ②value頭 fine-tune（Docker）。
replay-extract: ## replay JSON から value 学習サンプル (state,z) を抽出・永続化（ホスト）
	$(PY) scripts/extract_replay_samples.py
REPLAY_TUNE_INIT ?= models/pvnet_operative.pt
replay-tune: ## 実戦 z で value 頭を fine-tune（policy 不変・Docker）→ pvnet_replay.pt
	$(RUN) python scripts/replay_value_tune.py --init $(REPLAY_TUNE_INIT) --out models/pvnet_replay.pt $(REPLAY_TUNE_ARGS)

# 実メタ較正: replay 抽出デッキで判定プールを置換。レートが上がったら取り直して再実行。
GAUNTLET_N ?= 16
gauntlet-real: ## 実メタ（replay抽出）で判定ガントレットを置換（遭遇頻度上位 GAUNTLET_N 件）
	$(PY) scripts/gauntlet_from_replays.py --n $(GAUNTLET_N)

# === 判定操縦（提出と同じ floored NN に統一）===============================
# gate/eval-deck は提出と同じ操縦で判定する（詳細は design-decisions.md）。Docker 実行。
# ISMCTS 判定に戻すには JUDGE_FLAGS="--pilot ismcts --time-budget 0.05"。
JUDGE_FLOOR ?= 8
JUDGE_BONUS ?= 0.2   # 盤面補正 α（提出と判定で同値・§24）
JUDGE_FLAGS ?= --pilot nn --net $(NN_NET) --nn-sims $(NN_SIMS) \
	--floor-rollouts $(JUDGE_FLOOR) --board-bonus $(JUDGE_BONUS)

# gate は探索ループから分離し **make champion-gate で随時**実行する（ratchet は探索のみ＝iter を稼ぐ）。
# 随時実行なので提出忠実に振れる: **floor8**（floored NN の床＝改良ヒューリスティックが加速を打つ・§29）。
# floor0 だと凍結nnが加速を打たず加速デッキの利点が見えない（提出は floor8）。sims は 32≈64。
GATE_SIMS  ?= 32
GATE_FLOOR ?= 8
GATE_JUDGE_FLAGS ?= --pilot nn --net $(NN_NET) --nn-sims $(GATE_SIMS) \
	--floor-rollouts $(GATE_FLOOR) --board-bonus $(JUDGE_BONUS)
# $(call _gate_meta,N)＝実メタ頻度上位 N を --meta に（gauntlet 無ければ空＝champion_gate 既定）。
_gate_meta = $(if $(strip $(wildcard models/gauntlet/*.csv)),--meta $(wordlist 1,$(1),$(sort $(wildcard models/gauntlet/*.csv))),)
GATE_OPPS  ?= 12
GATE_GAMES ?= 16

# デッキ強さの確定評価（vs 相手プール・floored NN 判定）。既定 champion_best（提出に使う最良）。
# 別デッキは EVAL_DECK=... 、厳密化は EVAL_DECK_GAMES=40。
EVAL_DECK       ?= $(firstword $(wildcard models/champion_best.csv) models/champion_deck.csv)
EVAL_DECK_GAMES ?= 20
EVAL_DECK_ARGS  ?=
eval-deck: ## デッキ強さの確定評価（vs 相手プール・floored NN 判定・既定 champion・Docker）
	$(RUN) python scripts/eval_deck.py --deck $(EVAL_DECK) --games $(EVAL_DECK_GAMES) \
		$(JUDGE_FLAGS) $(EVAL_DECK_ARGS)

# 信頼ラチェット gate（単独・厳格プロファイル）: 新が best を上回った時だけ昇格。約1〜1.5h。
GATE_ARGS  ?=
champion-gate: ## keep-best 判定・単独厳格（相手12・16試合・約1〜1.5h・Docker）
	$(RUN) python scripts/champion_gate.py --games $(GATE_GAMES) $(call _gate_meta,$(GATE_OPPS)) $(GATE_JUDGE_FLAGS) $(GATE_ARGS)

# デッキ探索のみ（gate は分離＝make champion-gate で随時）。best を種に探索し models/champion_deck.csv
# に出力。RATCHET_ITERS で時間調整（実測(低spec): NN純探索 ≈ 約30〜60分/iter・要再計測）。
# league は毎反復 checkpoint＝途中で止めても再開可。昇格は champion-gate が best を上回った時だけ。
RATCHET_ITERS ?= 2
ratchet: ## デッキ探索のみ（best を種に→champion_deck。昇格は make champion-gate 別途）
	$(PY) src/league.py --cap 12 --iters $(RATCHET_ITERS) --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) $(_SEEDS_FLAG) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)
	@echo "探索完了 → models/champion_deck.csv。昇格は make champion-gate（best 超えで更新）"

# ISMCTS 探索の一晩版。per-iter は未実測（NN と別・time_budget 0.03）。使うなら time で測って調整。
ratchet-overnight: ## 一晩版 ratchet（ISMCTS探索のみ・iters10・所要は未実測）
	$(MAKE) ratchet RATCHET_ITERS=10

# NN 操縦版の探索のみ（探索=NN-MCTS＝ISMCTS の ~1/4 時間）。Docker。gate は make champion-gate。
ratchet-nn: ## NN探索のみ（best を種に→champion_deck。昇格は make champion-gate 別途・Docker）
	$(RUN) python src/league.py --cap 12 --iters $(RATCHET_ITERS) --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) --pilot nn --net $(NN_NET) --nn-sims $(SEARCH_SIMS) \
		$(_SEEDS_FLAG) --max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)
	@echo "探索完了 → models/champion_deck.csv。昇格は make champion-gate（best 超えで更新）"

# 一晩探索（gate 分離で iter を増やせる。iters は純探索の実測後に調整）。翌朝 make champion-gate。
ratchet-nn-overnight: ## NN探索の一晩版（iters11・純探索・翌朝 make champion-gate で昇格判定）
	$(MAKE) ratchet-nn RATCHET_ITERS=11

# === 提出 ==================================================================
# 同梱デッキは champion_best を優先（無ければ champion_deck）。相手推定プールも同梱（§26）。
submission: ## 提出パッケージ models/submission.tar.gz を作成（champion_best＋ISMCTS＋cg＋deck）
	@deck=models/champion_best.csv; \
	if [ ! -f "$$deck" ]; then deck=models/champion_deck.csv; \
		echo "champion_best.csv 未作成→ $$deck を同梱（先に ratchet 推奨）"; \
	else echo "同梱デッキ（最良）: $$deck"; fi; \
	$(PY) scripts/build_submission.py --deck $$deck

# NN 操縦の提出。提出前に make smoke-submission で 600秒クロックを実測（issue #4）。
SUBMISSION_NET ?= $(firstword $(wildcard models/pvnet_distill_best.pt) models/pvnet_operative.pt)
submission-nn: ## NN操縦の提出パッケージ models/submission_nn.tar.gz（floored NN＋最良net）
	@deck=models/champion_best.csv; \
	if [ ! -f "$$deck" ]; then deck=models/champion_deck.csv; \
		echo "champion_best.csv 未作成→ $$deck を同梱（先に ratchet 推奨）"; \
	else echo "同梱デッキ（最良）: $$deck"; fi; \
	echo "同梱ネット: $(SUBMISSION_NET)（board-bonus $(JUDGE_BONUS)）"; \
	$(PY) scripts/build_submission.py --deck $$deck --policy nn --net $(SUBMISSION_NET) \
		--board-bonus $(JUDGE_BONUS) --out models/submission_nn.tar.gz

smoke-submission: ## 提出エージェントの煙テスト＋時間計測（600秒検証・NN は Docker）
	$(RUN) python scripts/smoke_submission.py --policy nn --net $(SUBMISSION_NET) \
		--board-bonus $(JUDGE_BONUS) $(SMOKE_ARGS)

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
