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
        ratchet ratchet-overnight ratchet-nn gauntlet eval-deck champion-gate \
        train distill distill-1h distill-overnight improve improve-1h eval-net diagnose \
        submission build rebuild shell jupyter gpu-check exec up down clean

# --- デッキ探索（ratchet が内部で使う）の既定パラメータ --------------------
# 操縦は ISMCTS（特性/効果/トレーナーを扱える）。相手は models/gauntlet/ があればそれ（多様化）。
# 各反復でチェックポイント保存＝止めても再開可。LEAGUE_SEED は実行ごとにランダム（毎回別探索）。
LEAGUE_PILOT      ?= ismcts
LEAGUE_TIMEBUDGET ?= 0.03
_PILOT_FLAGS       = --pilot $(LEAGUE_PILOT) --time-budget $(LEAGUE_TIMEBUDGET)
LEAGUE_SEED    ?= $(shell python3 -c 'import random;print(random.randrange(2**31))')
LEAGUE_ARGS    ?=
# 既存チャンピオンを固定の試験官(seed)に自動取り込み（探索の起点・無ければ相手プールのみ）。
LEAGUE_EXTRA   ?= $(wildcard models/champion_deck.csv)
_EXTRA_FLAG     = $(if $(LEAGUE_EXTRA),--extra-seeds $(LEAGUE_EXTRA),)
# NN 操縦の既定ネット。improve で ISMCTS を超えた net を優先し、無ければ蒸留(床)にフォールバック
# （ratchet-nn は「ISMCTS を超えた強い NN」で探索するのが目的なので improve_best を使う）。
NN_NET  ?= $(firstword $(wildcard models/pvnet_improve_best.pt) models/pvnet_distill_best.pt)
NN_SIMS ?= 64

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
# 教師(ISMCTS)の1手秒。診断で **TB=0.25 は heuristic 未満(0.375)＝弱教師**、TB=0.5 で 0.733、
# 1.0 は誤差内の微増で2倍遅い → **0.5 が最適点**。弱教師を蒸留しないよう既定 0.5。
DISTILL_TB    ?= 0.5
DISTILL_ITERS ?= 60
# 方策ターゲット温度。0=one-hot（既定・浅い teacher で安全）。>0 で訪問分布 soft-π を解放
# （soft は teacher を深くした時=DISTILL_TB↑ でのみ有効。例: 鋭め soft なら 0.5）。
DISTILL_TEMP  ?= 0
# 教師対戦の並列収集数。各試合は独立＝コア数まで上げると「1回あたりの試行数」がほぼ線形に増える。
# 既定は nproc-1（1コアを OS/GPU供給に空けて機械の無反応を防ぐ）。HT 環境では物理コア数推奨。
DISTILL_WORKERS ?= $(shell n=$$(nproc 2>/dev/null || echo 2); echo $$((n>1?n-1:1)))
DISTILL_ARGS  ?=
distill: ## ISMCTS蒸留（複数デッキ・強い教師・resume継ぎ足し・並列収集）。長さは DISTILL_ITERS で
	$(RUN) python scripts/train_alphazero.py --teacher ismcts --resume \
		--out $(DISTILL_OUT) --best-out $(DISTILL_BEST) --teacher-time-budget $(DISTILL_TB) \
		--distill-temp $(DISTILL_TEMP) \
		--iterations $(DISTILL_ITERS) --workers $(DISTILL_WORKERS) --eval-every 20 --eval-games 24 \
		$(_DECKS_FLAG) $(DISTILL_ARGS)

distill-1h: ## 約1時間の蒸留（前回に継ぎ足し・日中ちょくちょく用）
	$(MAKE) distill DISTILL_ITERS=50

distill-overnight: ## 強い教師(TB=0.5)で蒸留（大容量netの収束に・約4-5h目安）。長時間の伸びは improve 側で
	$(MAKE) distill DISTILL_ITERS=120

# 蒸留は教師(ISMCTS)が天井＝五分まで。improve は **蒸留ネットを種に self-play** で天井を破る。
# MCTS(NN) は NN 単体より強い方策改善演算子なので、その訪問分布(soft-π)を学べば ISMCTS を
# 超えうる（以前の崩壊は弱いネット始点が原因。≈ISMCTS の種＋best保存＋低LR＋replayで安定）。
# 収集は CPU 並列（NN-MCTS は batch=1 推論＝GPUより CPU 向き）。判定は make eval-net EVAL_VS=ismcts。
IMPROVE_OUT   ?= models/pvnet_improve.pt
IMPROVE_BEST  ?= models/pvnet_improve_best.pt
IMPROVE_SEED  ?= models/pvnet_distill_best.pt
IMPROVE_ITERS ?= 40
IMPROVE_ARGS  ?=
improve: ## self-playでISMCTS超えを狙う（蒸留ネットを種・CPU並列・best保存・resume蓄積・drift安全弁）
	$(RUN) python scripts/train_alphazero.py --teacher selfplay --resume --resume-from-best \
		--init-from $(IMPROVE_SEED) --out $(IMPROVE_OUT) --best-out $(IMPROVE_BEST) \
		--iterations $(IMPROVE_ITERS) --workers $(DISTILL_WORKERS) \
		--eval-every 10 --eval-games 24 $(_DECKS_FLAG) $(IMPROVE_ARGS)

improve-1h: ## 約1時間の improve（前回に継ぎ足し・日中ちょくちょく用）
	$(MAKE) improve IMPROVE_ITERS=30

# 確定判断用の offline 評価（試合数を増やして運の振れを抑える）。学習中の24は傾向把握用、
# こちらは 40+ で「NN は heuristic/ISMCTS を超えたか」を判断する。例: make eval-net EVAL_GAMES=100
# 既定ネットは「improve_best > distill_best > pvnet」の順で存在する最良を使う（古い pvnet.pt を
# 黙って測る事故を防ぐ）。別ネットを測るなら make eval-net EVAL_NET=models/pvnet_distill_best.pt。
EVAL_GAMES ?= 40
EVAL_VS    ?= heuristic
EVAL_NET   ?= $(firstword $(wildcard models/pvnet_improve_best.pt) $(wildcard models/pvnet_distill_best.pt) models/pvnet.pt)
EVAL_ARGS  ?=
eval-net: ## 訓練済みNNの確定判断用 評価（既定: 最良net・vs heuristic 40試合・Docker）
	$(RUN) python scripts/eval_net.py --net $(EVAL_NET) --vs $(EVAL_VS) --games $(EVAL_GAMES) $(EVAL_ARGS)

# policy 診断: net が手を順位付けできているか／文脈無視のショートカットか／教師が弱くないかを測る。
# 「value は学べるが policy は学べない」の原因切り分け用。
DIAGNOSE_ARGS ?=
diagnose: ## NN policy 診断（shortcut度/教師再現度/文脈感度/集中度/教師強度・Docker）
	$(RUN) python scripts/diagnose_policy.py --net $(EVAL_NET) $(DIAGNOSE_ARGS)

# 多様ガントレット生成（過学習対策）。生成後は league/eval-deck/gate が自動でこれを相手に使う
# （models/gauntlet/ があれば data/*.csv より優先）。相手が多彩になる分、評価/探索は遅くなる。
gauntlet: ## 多様な相手デッキ群 models/gauntlet/ を生成（メタ＋チャンピオン系＋全色mono-type）
	$(PY) scripts/make_gauntlet.py

# デッキ強さの確定評価（vs 相手プール・ISMCTS操縦・多めの試合）。league内部の小サンプル(6試合)では
# 判定できない「本当にデッキが強くなったか」を測る。ホスト(CPU)。champions/ のバックアップと比較可。
EVAL_DECK       ?= models/champion_deck.csv
# 相手が gauntlet(16デッキ)なら 20試合でも合計十分（平均は安定・最悪は相手数で網羅）。厳密化は ↑。
EVAL_DECK_GAMES ?= 20
EVAL_DECK_ARGS  ?=
eval-deck: ## デッキ強さの確定評価（vs 相手プール・ISMCTS・既定 champion・ホスト）
	$(PY) scripts/eval_deck.py --deck $(EVAL_DECK) --games $(EVAL_DECK_GAMES) $(EVAL_DECK_ARGS)

# 信頼ラチェット: league 後に挟むと、新チャンピオンが best を信頼試合数で上回った時だけ昇格。
# ノイズドリフトを止め、回し続けるほど models/champion_best.csv が単調に良くなる。提出は best を使う。
# 相手が gauntlet(16)なら 20試合で合計十分。厳密に見たいときは GATE_GAMES=40。
GATE_GAMES ?= 20
GATE_ARGS  ?=
champion-gate: ## league 後の keep-best 判定（新が best を上回った時だけ昇格・ホスト）
	$(PY) scripts/champion_gate.py --games $(GATE_GAMES) $(GATE_ARGS)

# デッキ探索→ゲートを1サイクル（best起点・確実改善だけ採用）。回すほど champion_best が単調改善。
# 探索(league)は速い5メタ・判定(gate)は多様 gauntlet。RATCHET_ITERS で時間調整
# （1≒約50分・3≒約1.5h・6≒約3h）。league はチェックポイント保存＝途中で止めても無駄にならない。
RATCHET_ITERS ?= 3
ratchet: ## デッキ探索→ゲート1サイクル（best起点／時短: RATCHET_ITERS=1, じっくり: =6）
	@if [ -f models/champion_best.csv ]; then \
		cp models/champion_best.csv models/champion_deck.csv; \
		echo "起点を champion_best に設定（best から探索）"; \
	else echo "best 未作成: 現 champion_deck から開始（gate が初回 best を作成）"; fi
	$(PY) src/league.py --cap 12 --iters $(RATCHET_ITERS) --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) $(_PILOT_FLAGS) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)
	$(PY) scripts/champion_gate.py --games 20 --time-budget 0.05 $(GATE_ARGS)
	@echo "ratchet 完了。最良は models/champion_best.csv（提出はこれを使う）"

ratchet-overnight: ## 一晩版 ratchet（探索を多め iters20・約6h・翌朝 eval-deck で確認）
	$(MAKE) ratchet RATCHET_ITERS=20

# NN 操縦版の ratchet（Docker/GPU）。探索(league)を蒸留 NN-MCTS で回す＝ISMCTS の ~1/4 時間で
# 同等強度＝同じ時間で「より多く探索」できる。判定(gate)は独立性のため ISMCTS のまま（ホスト/CPU）。
# distill で NN を強くしてから回すほど探索の質が上がる（distill↔ratchet-nn を交互に）。
ratchet-nn: ## NN操縦の高速 ratchet（探索=蒸留NN-MCTS・Docker／判定=ISMCTS・ホスト）
	@if [ -f models/champion_best.csv ]; then \
		cp models/champion_best.csv models/champion_deck.csv; \
		echo "起点を champion_best に設定（best から探索）"; \
	else echo "best 未作成: 現 champion_deck から開始（gate が初回 best を作成）"; fi
	$(RUN) python src/league.py --cap 12 --iters $(RATCHET_ITERS) --games 4 --pop 6 --gens 3 \
		--plateau 99 --seed $(LEAGUE_SEED) --pilot nn --net $(NN_NET) --nn-sims $(NN_SIMS) \
		--max-swaps 12 --explore 0.3 $(_EXTRA_FLAG) $(LEAGUE_ARGS)
	$(PY) scripts/champion_gate.py --games 20 --time-budget 0.05 $(GATE_ARGS)
	@echo "ratchet-nn 完了。最良は models/champion_best.csv（提出はこれを使う）"

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
