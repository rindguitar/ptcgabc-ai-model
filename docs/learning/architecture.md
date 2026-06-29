# コード構造（architecture）

`src/` のモジュールと `scripts/` の実行スクリプトの役割・関係をまとめる。用語は
[glossary.md](glossary.md) を参照。

## 全体像（データの流れ）

```
                    ┌─────────────── デッキ軸（CPU）───────────────┐
  data/ メタデッキ → league.py（探索）→ champion_gate.py（keep-best）→ champion_best.csv → 提出
                         │ 評価操縦に ismcts/nn を使う
                         ▼
   cards.py / deck.py / deckopt.py / features.py（デッキ・カードのモデル化）
                         ▲
                    ┌─────────────── NN軸（GPU+CPU）──────────────┐
  ISMCTS 教師 → distill.py（蒸留）→ PVNet ──┐
  自己対戦   → selfplay.py（improve）────────┤→ train.py（学習）→ pvnet_*.pt
                         │ nn_collect.py（並列収集）          │
                         ▼                                    ▼
   ismcts.py / nn_mcts.py（探索操縦）  ← nn_eval.py（NN→評価器）← net.py（PVNet）
```

対戦そのものは **cabt エンジン**（`src/cg/`・Competition Data・追跡外）が実行する。各モジュールは
その API（`battle_*` 対戦進行 / `search_*` フォワードモデル）を呼ぶ。

## `src/` モジュール

### 基盤（カード・デッキ・対戦）
| モジュール | 役割 |
|---|---|
| [cards.py](../../src/cards.py) | カードのメタ情報（HP・タイプ・特性有無・威力効率など**数値のみ**）をロード。効果文は読まない |
| [deck.py](../../src/deck.py) | デッキの表現・合法性・変異(mutate)・構成・構造化生成(structured_deck) |
| [harness.py](../../src/harness.py) | 自己対戦ハーネス（2エージェントを戦わせ勝率を返す・席入替で先手有利を打ち消す） |
| [features.py](../../src/features.py) | 局面・合法手を数値ベクトルに encode（NN 入力）。特性/手札構成も含む（末尾追加でwarm-start可） |

### 操縦（pilot）
| モジュール | 役割 |
|---|---|
| [agents.py](../../src/agents.py) | heuristic 操縦（貪欲・速いが特性/効果/トレーナーは使わない） |
| [ismcts.py](../../src/ismcts.py) | ISMCTS 操縦（determinized UCT）。`make_ismcts_agent` と、蒸留用に訪問数を返す `ismcts_aggregate` |
| [determinize.py](../../src/determinize.py) | 隠れ情報の決定化（相手山札/手札のサンプリング推定） |
| [nn_mcts.py](../../src/nn_mcts.py) | NN 誘導 MCTS（PUCT）。葉をロールアウトせず評価器で評価。`aggregate_visits` が訪問分布を返す |
| [nn_eval.py](../../src/nn_eval.py) | 学習済み PVNet を nn_mcts 用の評価器 `evaluator(obs)->(value, priors)` に変換 |

### NN（学習）
| モジュール | 役割 |
|---|---|
| [net.py](../../src/net.py) | PVNet（policy/value の2出力を持つネット）の定義 |
| [train.py](../../src/train.py) | 学習ループ（value=BCE + policy=交差エントロピー）・保存/読込・`load_net_warmstart` |
| [distill.py](../../src/distill.py) | ISMCTS 教師の蒸留データ収集。温度で one-hot↔soft 1本化・CPU 並列収集 |
| [selfplay.py](../../src/selfplay.py) | 自己対戦データ収集（improve の中核・訪問分布を soft-π に） |
| [nn_collect.py](../../src/nn_collect.py) | self-play サンプルの **CPU 並列収集**（spawn・各workerがCPUにネット読込） |

### デッキ最適化
| モジュール | 役割 |
|---|---|
| [deckopt.py](../../src/deckopt.py) | 進化計算（最悪ケース勝率の最大化）・適応度評価・相手プール解決 |
| [league.py](../../src/league.py) | bounded double-oracle リーグ。探索操縦は `--pilot {heuristic,ismcts,nn}`。チェックポイント/resume |
| [reeval.py](../../src/reeval.py) | 候補デッキを別操縦で再評価（操縦バイアス補正） |

### 提出
| モジュール | 役割 |
|---|---|
| [submission.py](../../src/submission.py) | 公式 `agent(obs_dict)` 形式へのアダプタ（`make_kaggle_agent`） |
| [main.py](../../src/main.py) | ローカル動作確認用エントリ |

## `scripts/`（実行スクリプト・`make` から呼ぶ）
| スクリプト | 対応 make | 役割 |
|---|---|---|
| [train_alphazero.py](../../scripts/train_alphazero.py) | `distill` / `improve` / `train` | 反復学習ドライバ（蒸留 or 自己対戦 → 学習 → best保存・resume蓄積） |
| [eval_net.py](../../scripts/eval_net.py) | `eval-net` | NN 操縦の強さを vs heuristic / ismcts で測る（確定判断・40+試合） |
| [eval_deck.py](../../scripts/eval_deck.py) | `eval-deck` | デッキの強さを相手プールに対し測る（pilot 選択可） |
| [champion_gate.py](../../scripts/champion_gate.py) | `champion-gate` / `ratchet` | keep-best ゲート（新が best を上回った時だけ昇格） |
| [make_gauntlet.py](../../scripts/make_gauntlet.py) | `gauntlet` | 多様な相手デッキ群を生成（過学習対策の判定用） |
| [build_submission.py](../../scripts/build_submission.py) | `submission` | 提出 tar.gz を組み立て（操縦＋デッキ＋cg を同梱） |

## 追跡外（Git 管理しない）
- `src/cg/`（cabt エンジン）・`data/`（カードデータ・メタデッキ）＝**Competition Data**。
- `models/`（学習済み `.pt`・チャンピオン CSV・gauntlet）＝生成物。
- これらは `.gitignore` 済み。詳細は [design-decisions.md](design-decisions.md)「規約と公開計画」。
