# Competition Data のスキーマメモ（data/・追跡外）

CLAUDE.md から移設した参照資料。**data/ の中身そのもの（カード名・効果文等の Pokémon
Elements）はどこにも貼らない**こと。ここに書くのはスキーマ・件数・記法のみ。

## カードデータ CSV
- `EN_Card_Data.csv` / `JP_Card_Data.csv`: 各 2102 行 ＝ **ユニーク 1267 枚**（Card ID 1–1267、
  同一 ID で EN/JP 対応）。エンジンの `AllCard` の 1267 枚と一致＝大会リーガルな全カードプール。
- 1 枚のカードが複数ワザ／特性を持つ場合、**同一 Card ID で複数行**（Move Name 単位）。
  「行数 ≠ 枚数」なのでパース時は Card ID でグルーピングが必要（実装済み: src/cards.py は
  エンジンの `AllCard`/`AllAttack` JSON を使い、CSV は直接使っていない）。
- CSV カラム（EN）:
  `Card ID, Card Name, Expansion, Collection No., Stage (Pokémon)/Type (Energy and Trainer),
   Rule, Category, Previous stage, HP, Type, Weakness, Resistance (Type), Retreat,
   Move Name, Cost, Damage, Effect Explanation`
- エネルギー種別やコストは `{G}{R}{W}{L}...` のシンボル、コストの追加分は `●` で表記。
- 欠損は `n/a` または空文字（両方を欠損として扱う）。
- `Card_ID List_EN.pdf` / `Card_ID List_JP.pdf`: カード ID 一覧（大容量・通常は使わない）。

## その他の data/ 配下
- `data/*_Deck.csv` 等: 公式サンプルのメタデッキ（60 行のカード ID）。
- `data/replays/`: Kaggle episode JSON（`make replays` で集計・抽出後は削除可）。
  `episodes_log.csv`（結果の永続ログ）と `opp_decks/`（実メタデッキ抽出）は残す。
