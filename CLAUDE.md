# CLAUDE.md

このファイルは Claude Code がこのリポジトリで作業する際のガイド。
**薄く保つこと。** 詳細は `docs/` に置き、必要時のみ参照する（末尾「参照ドキュメント」）。

## ⚠️ 規約（ハード遵守・Sec 2.4/2.6/3.6/3.18.f）

- `data/`（Competition Data）と `src/cg/`（エンジン）は**コミット・出力・公開禁止**
  （.gitignore 済み・競技終了後に削除。リポジトリ自体の削除義務はない）。
- **Pokémon Elements（カード名・ワザ名・効果文・デッキ内容・画像）をコード/コミット/issue/
  ログ/会話に貼らない**。参照はカード ID・数値・カテゴリまで。テストもダミー値を使う。
- 効果文等の**ローカルでの学習利用は許諾の範囲内**（禁止は公開・再配布のみ）。
- コード共有はチーム外禁止。公開は Kaggle フォーラム＋OSI ライセンス経由のみ。
  External Data/モデルは全参加者が無償アクセス可能なもの限定。受賞時は MIT 提供義務
  （OSI 非互換の依存を入れない）。

## プロジェクト概要

Kaggle「The Pokémon Company - PTCG AI Battle Challenge Strategy」（Simulation division）向け
ポケモンカード対戦 AI の**非公開**リポジトリ。ISMCTS / floored NN-MCTS（操縦）×
league/ネットデッキ（デッキ）×リーダーボード諜報（§33）の3軸で賞金圏を狙う。

### 技術スタック
- Python 3.12（ホスト venv＝cabt は標準ライブラリで動く / WSL2 Ubuntu）
- GPU 学習系は Docker（PyTorch はベースイメージ同梱＝requirements.txt に書かない）
- cabt Engine（`src/cg/`・ctypes・Competition Data）・numpy・omegaconf・pytest・ruff
- 低スペック環境＝計算量に敏感。訓練の軍拡でなく推論時・データ駆動のレバーを優先（§26/§31/§33）

## セッションプロトコル（最重要）

### 開始時
1. `docs/STATUS.md` を読む（現在地・次の一手・未解決事項）
2. 着手前に**今日のゴールを1〜3行で確認**し、ユーザーと認識を合わせる

### 終了時
1. `docs/STATUS.md` を更新（何をした / 何が残った / 次の一手 / ハマった点）
2. 設計判断があれば `docs/learning/design-decisions.md` に「## N.」で追記・新語は glossary へ
3. コミット・プッシュ漏れがないか確認する

## モデル運用ルール（コスト規律）

- 大きなファイル（replay JSON・モデル出力・ログ・CSV）を**コンテキストに丸読みしない**。
  部分読み・スクリプトでの要約・行番号指定を使う。
- 同じファイルを繰り返し読み直さない。一度読んだ内容は会話内の記憶を使う。
- 広範な探索・一括処理で親のコンテキストを汚したくない時だけサブエージェントに委譲
  （このプランでは spawn ごとにコンテキスト再構築＝原則 inline で処理）。

## 実装ワークフロー（説明可能性の担保）

**実装前に方針を先に提示し、承認を得てから書く。**
1. 方針提示: 何を・どう変えるかを箇条書き3〜5行で提示（新手法は理由・利点・欠点も）
2. 承認後に実装
3. **実装 → ruff＋テスト成功 → コミット → プッシュ**（この順序を崩さない・テストなしコミット禁止）

理由: ユーザーは「自分で説明できないコードは採用しない」方針。大きな変更は**新ブランチ**
（作成したら作業前に必ず remote へ push）。

## 開発コマンド

- 全コマンド一覧: `make help`。テスト: `make test`（=pytest）。Lint: `make lint` / `make format`
- ホスト（CPU・日常開発）: `make deps / smoke / bench / check`。replay 分析は `make replays`
- GPU/学習系は Docker: `make build / shell / exec CMD="..."`。設定は `.env` のみ（ハードコード禁止）

### ⚠️ 時間のかかる操作はユーザーが実行
以下は Claude Code で実行せず、「実行が必要」とユーザーに通知して結果を貼ってもらう:
- Docker イメージの再ビルド
- train / distill / improve / eval-net / eval-deck / league / ratchet 等の GPU・長時間コマンド
（Claude の担当はコード実装・編集・短時間テストまで）

## コーディング規約（常時適用の最小限）

- **処理の流れをコメントで説明する**: 関数・処理ブロックの冒頭や区切りに「何をどういう順で
  やるか」を日本語コメントで書く（例: `# 1. 決定局面を抽出 → 2. one-hot π を構築 → 3. npz へ永続化`）。
  ユーザーがコードを読んで流れを追えることが目的。ただし自明な1行ごとのコメント
  （`i += 1  # 加算` 等）は書かない
- コメント・docstring は日本語（技術用語は英語のまま。例: batch, pipeline）
- 変数名・関数名は英語（Python 慣例に従う）
- ハードコード禁止。設定値は引数・環境変数・`omegaconf`+`configs/*.yaml` へ
- コミットメッセージ: `feat:` / `fix:` / `docs:` / `refactor:`
- 新規依存は requirements.txt へ（PyTorch はイメージ同梱＝書かない）

## セキュリティ

- 秘匿値・ローカル設定は `.env` のみ。API キー等の実値をコミットしない（疑わしければ相談）
- コミット前に秘密鍵・トークン・Competition Data が含まれていないか確認

## 参照ドキュメント（必要時のみ読む）

| ファイル | 内容 | 読むタイミング |
|---|---|---|
| `docs/STATUS.md` | 現在地・次の一手 | **毎セッション開始時（必須）** |
| `docs/learning/design-decisions.md` | 設計判断の記録（ADR・§1〜） | 設計変更を検討するとき |
| `docs/learning/glossary.md` | 用語集 | 知らない用語が出たとき |
| `docs/learning/architecture.md` | コード構造（src/ scripts/ の役割） | どこに何があるか探すとき |
| `docs/learning/replay-format.md` | Kaggle replay JSON の構造 | replay を触るとき |
| `docs/data-notes.md` | データのスキーマ | データを触るとき |
| `README.md` | セットアップ詳細 | 環境構築時 |
