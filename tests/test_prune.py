"""prune_replays の破棄判定テスト（消費者通過・自分の試合の温存・dry-run）."""

import csv
import os
import sys

import numpy as np
import pytest

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS)

import prune_replays  # noqa: E402


def _write_json(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{}")  # 中身は問わない（prune はファイル名 stem のみ使う）


def _build(root):
    """others に A,B（両消費者済）,D（value 未消費）・ismcts に C・alphago_v4 に E（派生）を置く."""
    _write_json(os.path.join(root, "others", "A.json"))
    _write_json(os.path.join(root, "others", "B.json"))
    _write_json(os.path.join(root, "others", "D.json"))
    _write_json(os.path.join(root, "ismcts", "C.json"))
    _write_json(os.path.join(root, "alphago_v4", "E.json"))  # keep の前方一致対象
    # analyze 済み: A,B,C,D,E すべてログ済み
    with open(os.path.join(root, "episodes_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id"])
        w.writeheader()
        for eid in ("A", "B", "C", "D", "E"):
            w.writerow({"episode_id": eid})
    # value 済み: A,B,C,E（D は未消費）
    np.savez_compressed(
        os.path.join(root, "value_samples.npz"),
        episodes=np.asarray(["A", "B", "C", "E"]),
    )


def _run(root, extra):
    argv = ["prune_replays.py", "--dir", str(root)] + extra
    old = sys.argv
    sys.argv = argv
    try:
        prune_replays.main()
    finally:
        sys.argv = old


def _exists(root, *parts):
    return os.path.exists(os.path.join(root, *parts))


def test_dry_run_deletes_nothing(tmp_path):
    _build(tmp_path)
    _run(tmp_path, [])  # 既定は dry-run
    assert _exists(tmp_path, "others", "A.json")
    assert _exists(tmp_path, "others", "D.json")


def test_apply_prunes_nothing_in_keep_variants(tmp_path):
    _build(tmp_path)
    _run(tmp_path, ["--apply"])
    # others は既定 keep-variants（§47 教師プール・cardId マイニング未追跡のため独立ライフサイクル）
    # → analyze＋value 両方済でも温存
    assert _exists(tmp_path, "others", "A.json")
    assert _exists(tmp_path, "others", "B.json")
    assert _exists(tmp_path, "others", "D.json")
    # C は自分の試合（keep-variants の ismcts）→ 温存
    assert _exists(tmp_path, "ismcts", "C.json")
    # E は派生ディレクトリ（alphago_v4）＝前方一致で温存（A/B 用 replay を守る）
    assert _exists(tmp_path, "alphago_v4", "E.json")


def test_include_own_prunes_consumed_keep_variants(tmp_path):
    _build(tmp_path)
    _run(tmp_path, ["--apply", "--include-own"])
    assert not _exists(tmp_path, "ismcts", "C.json")  # 自分の試合も破棄
    assert not _exists(tmp_path, "alphago_v4", "E.json")  # 派生も含めて破棄
    # others も破棄対象に含まれる（consumed 分のみ）
    assert not _exists(tmp_path, "others", "A.json")
    assert not _exists(tmp_path, "others", "B.json")
    assert _exists(tmp_path, "others", "D.json")  # value 未消費 → 温存


def test_consumers_analyze_only_prunes_pending_value(tmp_path):
    _build(tmp_path)
    # others は既定 keep-variants なので --include-own で外して consumers 判定のみ検証
    _run(tmp_path, ["--apply", "--include-own", "--consumers", "analyze"])
    # analyze のみ必須 → D も破棄される（value 未消費でも）
    assert not _exists(tmp_path, "others", "D.json")


def test_unknown_consumer_errors(tmp_path):
    _build(tmp_path)
    with pytest.raises(SystemExit):
        _run(tmp_path, ["--consumers", "bogus"])
