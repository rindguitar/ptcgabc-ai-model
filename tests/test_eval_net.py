"""eval_net CLI の配線テスト（--pilot 基準線が torch 無しホストで起動できること）."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("cg.sim", reason="cabt Engine (cg) が見つからない")

SCRIPT = os.path.join(ROOT, "scripts", "eval_net.py")


def test_eval_net_help_runs_without_torch():
    # --pilot ismcts の基準線はホスト（torch 無し）で回す設計。torch がトップ import に
    # 戻ると venv で起動自体が失敗するため、--help の成功と --pilot の存在を配線として固定する
    res = subprocess.run(
        [sys.executable, SCRIPT, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert "--pilot" in res.stdout
    assert "ismcts" in res.stdout
