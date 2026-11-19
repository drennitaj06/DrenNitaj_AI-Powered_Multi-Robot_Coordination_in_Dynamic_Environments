"""End-to-end ``train`` and ``simulate --controller madql`` in a fresh interpreter (Issue #5)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "training" / "madql_smoke.toml"
DEMO = ROOT / "configs" / "scenarios" / "demo_delivery.toml"

pytestmark = pytest.mark.e2e


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "robocoord", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
        check=False,
    )


def test_train_then_simulate_with_the_trained_policy(tmp_path: Path) -> None:
    out = tmp_path / "run"
    proc = run("train", "--config", str(SMOKE), "--out", str(out), "--seed", "3")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["env_steps"] == 200 and summary["updates"] > 0
    for name in ("config.json", "train_log.jsonl", "validation.jsonl", "summary.json", "best.pt"):
        assert (out / name).exists(), name
    sidecar = json.loads((out / "best.json").read_text())
    assert sidecar["seeds"]["run_seed"] == 3 and sidecar["algorithm"] == "madql"
    proc = run("simulate", "--scenario", str(DEMO), "--planner", "astar", "--allocator",
               "auction", "--controller", "madql", "--checkpoint", str(out / "best.pt"),
               "--max-steps", "40", "--json")  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["physical_collisions"] == 0
    # a second run into the same directory is refused (results are never overwritten)
    again = run("train", "--config", str(SMOKE), "--out", str(out))
    assert again.returncode != 0
