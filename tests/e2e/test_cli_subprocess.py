"""End-to-end: the installed CLI in a fresh interpreter (marked e2e; run with -m e2e)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "configs" / "scenarios" / "demo_delivery.toml"

pytestmark = pytest.mark.e2e


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "robocoord", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
        check=False,
    )


def test_module_entry_point_simulates_demo(tmp_path: Path) -> None:
    trace = tmp_path / "demo.jsonl"
    proc = run("simulate", "--scenario", str(DEMO), "--no-animate", "--trace", str(trace))
    assert proc.returncode == 0, proc.stderr
    assert "all tasks delivered" in proc.stdout
    assert trace.exists() and trace.stat().st_size > 0


def test_two_processes_produce_identical_traces(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path in (a, b):
        proc = run("simulate", "--scenario", str(DEMO), "--json", "--trace", str(path))
        assert proc.returncode == 0, proc.stderr
    assert a.read_bytes() == b.read_bytes()


def test_version_and_validation_error_exit_code() -> None:
    assert "robocoord" in run("--version").stdout
    proc = run("validate", "--scenario", "does-not-exist.toml")
    assert proc.returncode == 2 and "scenario error" in proc.stderr
