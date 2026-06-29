# 学習ノート（docs/learning）

このプロジェクトで使っている技術・用語・設計判断・コード構造を、後から学べる形でまとめる場所。
開発しながら「知らない用語が出てきた」「なぜこの設計？」を**その都度ここに追記**していく。

## 使い方
- 分からない言葉が出たら [glossary.md](glossary.md) を引く／無ければ追記する。
- コードのどこに何があるかは [architecture.md](architecture.md)。
- 「なぜそうしたか」は [design-decisions.md](design-decisions.md)（判断の理由・利点・欠点）。
- コードへは相対リンクで直結できる（例: [ismcts.py](../../src/ismcts.py)）。

## 構成
| ファイル | 内容 |
|---|---|
| [glossary.md](glossary.md) | 用語集（MCTS / ISMCTS / AlphaZero / 蒸留 / 過学習 / 並列化 など） |
| [architecture.md](architecture.md) | コード構造（`src/` 各モジュールの役割と関係） |
| [design-decisions.md](design-decisions.md) | 設計判断の理由（なぜ ISMCTS 操縦か・distill→improve・gauntlet など） |

## ⚠️ 書いてよいこと / ダメなこと（規約）
- **書いてよい**: 一般的な技術（MCTS, NN, Python）、自分のコード構造・設計判断・数値（勝率・loss）。
- **書いてはいけない**: **Pokémon Elements**（カード名・ワザ名・効果文・デッキ構成・画像）。
  数値結果を書くときもデッキは「あるデッキ / 別デッキ」のように**伏せて**書く。
- 競技期間中はこのリポジトリを **public にしない**。公開は競技終了後に
  「Competition Data 削除 → ライセンス付与 → Kaggle フォーラム告知 → public 化」の順で行う計画
  （詳細は [design-decisions.md](design-decisions.md) の「公開計画」）。
