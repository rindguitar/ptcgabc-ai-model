# CLAUDE.md

Kaggle「The Pokémon Company - PTCG AI Battle Challenge Strategy」（Simulation division）向け
ポケモンカード対戦 AI の**非公開**リポジトリ。
セットアップ詳細は [README.md](README.md)、設計判断・用語・コード構造は
[docs/learning/](docs/learning/)、データのスキーマは [docs/data-notes.md](docs/data-notes.md)。

**現在地・次の一手は [docs/STATUS.md](docs/STATUS.md)（セッション開始時に読み、終了時に更新する）。**

## ⚠️ 規約（ハード遵守・Sec 2.4/2.6/3.6/3.18.f）

- `data/`（Competition Data）と `src/cg/`（エンジン）は**コミット・出力・公開禁止**
  （.gitignore 済み・競技終了後に削除。リポジトリ自体の削除義務はない）。
- **Pokémon Elements（カード名・ワザ名・効果文・デッキ内容・画像）をコード/コミット/issue/
  ログ/会話に貼らない**。参照はカード ID・数値・カテゴリまで。テストもダミー値を使う。
- 効果文等の**ローカルでの学習利用は許諾の範囲内**（禁止は公開・再配布のみ）。
- コード共有はチーム外禁止。公開は Kaggle フォーラム＋OSI ライセンス経由のみ。
  External Data/モデルは全参加者が無償アクセス可能なもの限定。受賞時は MIT 提供義務
  （OSI 非互換の依存を入れない）。

## コマンド

- ホスト（CPU・日常開発）: `make deps / smoke / bench / check`、`ruff check .`、`pytest`。
  replay 分析は `make replays`。主要ターゲットは `make help`。
- GPU/学習系は Docker: `make build / shell / exec CMD="..."`。設定は `.env` のみ
  （ハードコード禁止）。`make build` 等の重い操作はユーザーが手動実行。
- **train / distill / improve / eval-net / league 等の長時間・GPU コマンドはユーザーが実行**
  して結果を貼る。Claude の担当はコード実装・編集・短時間テストまで。

## 開発ルール

- 大きな変更は**新ブランチ**（作成したら作業前に必ず remote へ push）。
- **実装 → テスト成功 → コミット → push（セット）** の順を厳守。テストなしコミット・push 忘れ禁止。
- コメント/docstring は日本語（技術用語は英語のまま: batch, pipeline 等）。変数・関数名は英語。
- PyTorch はベースイメージ同梱＝ requirements.txt に書かない。新規依存は requirements.txt へ。
- 秘匿値・ローカル設定は `.env` のみ。API キー等の実値をコミットしない（疑わしければ相談）。
- 設定は `omegaconf` + `configs/*.yaml`、実験ログは tensorboard（`runs/` 追跡外）。
