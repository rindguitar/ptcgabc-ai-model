---
paths:
  - "tests/**/*.py"
---

# テストコード規約（テストファイル編集時のみロードされる）

このプロジェクトの実テスト（`tests/test_*.py`）の作法に合わせる。既存を読んで踏襲すること。

- pytest を使用。名前は既存に合わせ `test_<機能>_<期待>` 式（例: `test_ismcts_returns_legal_action`・
  `test_repair_injects_energy_accel`）。**1テスト1アサーション強制はしない**（既存は関連 assert を複数持つ）。
- **cabt Engine 依存のテストは冒頭で `pytest.importorskip("cg.sim")`**（無い環境で自動 skip）。
  同様に `data/deck.csv` 等が無ければ `pytest.skip(..., allow_module_level=True)`。
- **⚠️ 規約: テストに実カードデータ/Pokémon Elements を書かない。カード ID は必ずダミー整数**
  （例: `[1]*60`・`list(range(1,61))`）。効果文・カード名・実デッキを埋め込まない。
- 観測のフェイクは `SimpleNamespace` で軽量に組む（`tests/test_determinize.py`・`test_agents.py` 参照）。
  ただし**本番データの形は一度 dump して確認**（§31/§33 の教訓: テストが通っても実データの形と違えば no-op）。
- 乱数は `random.Random(seed)` でシード固定＝再現性。**効果量を主張する実験は複数シードで検証**
  （1点差はノイズ・design-decisions の用量反応/CRN 参照）。強さ検証は重いので CI では回さず
  「合法な選択を返す」等の軽量確認に留める（強さは `make bench` / eval 系で別途測る）。
- 共通の重い準備（`load_card_meta` 等）は module スコープ fixture で1回だけ。

<!--
  paths にマッチするファイルを Claude が触るときだけロードされる想定（実挙動は要確認）。
  トピックが増えたら submission.md, replay.md 等をこの流儀で足す。
-->
