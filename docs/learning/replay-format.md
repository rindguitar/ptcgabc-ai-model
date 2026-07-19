# Kaggle replay（episode JSON）の構造と読み方

`data/replays/**/*.json`（Competition Data 扱い・追跡外）の中身と、
[analyze_replays.py](../../scripts/analyze_replays.py) が**何をどこから読むか**をまとめる。
形式は kaggle_environments の episode 標準形式（本大会固有ではなく Kaggle シミュレーション大会共通）。

## 全体の形

```
episode JSON
├── configuration     … 対戦設定（actTimeout / episodeSteps / runTimeout / seed）
├── info
│   ├── TeamNames[2]  … 両プレイヤーのチーム名 ← ★相手の正体はここ
│   └── EpisodeId     … エピソード（試合）の一意 ID
├── rewards[2]        … 最終報酬（勝ち 1 / 負け -1・index はプレイヤー席）
├── statuses[2]       … 終了状態（"DONE" 正常 / "ERROR" / "TIMEOUT" …）
├── specification     … スキーマ定義（観測・行動・報酬の型の説明書き）
└── steps[t][agent]   … ★本体。ターン t における各プレイヤーの記録
    ├── action        … そのステップで返した行動（int のリスト）
    ├── reward / status / info
    └── observation   … そのプレイヤーから見えた盤面
        ├── current   … 状態（players の盤面・手札枚数・サイド残数 … 自分視点）
        ├── select    … 選択肢（合法手のリスト）
        ├── logs      … 前回の選択以降に起きたイベント
        ├── remainingOverageTime … 残りの累積持ち時間（600 秒起点）
        └── search_begin_input   … エンジンの探索 API 用のシリアライズ文字列（下記）
```

## `search_begin_input` は何か（相手の ID ではない）

**エンジンのフォワードモデル（search API）を初期化するための内部状態文字列**（~270 字・各プレイヤーが
自分用を持つ）。我々の ISMCTS/NN-MCTS は「今の局面から先読みシミュレーション」を行うが、その開始点を
エンジンに伝えるのがこの文字列で、[cg/api.py](../../src/cg/api.py) の `search_begin()` にそのまま渡す。
つまり**探索用のチケット**であり、対戦相手の情報は入っていない。replay 分析では使わない。

## analyze_replays.py は何をどこから読むか

| 知りたいこと | 読む場所 | 読み方 |
|---|---|---|
| 自分はどっちの席か | `info.TeamNames` | 全エピソードに共通して出る名前＝自分（相手は入れ替わる）。`--team` で明示も可 |
| 相手は誰か | `info.TeamNames[1-自分]` | チーム名（＝Kaggle 上の相手） |
| 勝敗 | `rewards` | 自分の席の値が大きければ win（1 / -1） |
| 相手のデッキ | `steps[先頭付近][相手].action` | **初手 action が「デッキ 60 枚のカード ID リスト」**（公式形式の仕様: 最初の応答はデッキ提出）。これを CSV 保存→gauntlet の相手プールへ＝実メタ較正 |
| 時間の健全性 | `observation.remainingOverageTime` | 最後の値が残り持ち時間。600 起点でどれだけ使ったか（時間ガード検証） |
| 異常終了の有無 | `statuses` | "DONE" 以外（ERROR/TIMEOUT）が出ていないか |

## 読むときの注意

- **提出直後のエピソードはセルフ検証マッチ**（相手＝自分のチーム名）。動作確認用で、メタ情報の価値はない。
  相手デッキ抽出が意味を持つのは**他チーム戦**の replay から。
- replay には盤面・デッキ（カード ID）等の Competition Data が含まれる → **`data/` 配下（追跡外）にのみ置き、
  分析結果はチーム名・数値・ID までに留める**（カード名・効果文は出さない）。
- JSON に**保持期限の情報は無い**。Kaggle 側で古い episode が辿りにくくなる可能性に備え、**毎日ダウンロードして
  手元に蓄積**する（分析はエピソード ID で冪等）。

## 生 JSON のライフサイクル（いつ消してよいか・2026-07-15 決定）

**原則: 生 JSON は「蒸留物」を全部絞ったら削除可。永久保存は蒸留物の方**——
`episodes_log.csv`・`opp_decks/*.csv`・`value_samples.npz`・`teacher_*.npz`（いずれも冪等追記で軽い）。

削除前チェックリスト:
- **自分の対局**（ismcts/ teacher/ 等）: ① `make replays` ② `make replay-extract`
  ③ その日の深掘り分析（サイド差・使用デッキ検証等）→ 削除可。
  **例外: A/B 判定が未決着のアームは決着まで残す**（追加の深掘りは生 JSON にしか無い情報を使う）。
- **他チーム**（others/）: ① `make replays`（デッキ収穫）② `make top-replays`（成績の記録）
  ③ 教師候補は `make teacher-extract` ④ **本人の基準値算出**（手数・デッキ切れ率＝
  decisions.md §35 の教訓: クローンとの比較基準に要る）→ 削除可。
- **迷ったら削除でなくアーカイブ**: `tar czf data/replays/archive/YYYYMMDD.tar.gz <フォルダ>`
  （容量 1/5〜1/10・後から再分析可能）。アーカイブも Competition Data＝追跡外・競技終了後に削除。
