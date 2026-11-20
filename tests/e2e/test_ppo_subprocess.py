"""End-to-end PPO ``train`` (and ``--resume``) and ``simulate --controller ppo`` in a fresh
interpreter (Issue #6)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "training" / "ppo_smoke.toml"
MADQL_SMOKE = ROOT / "configs" / "training" / "madql_smoke.toml"
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


def test_ppo_train_resume_and_simulate(tmp_path: Path) -> None:
    out = tmp_path / "run"
    proc = run("train", "--config", str(SMOKE), "--out", str(out), "--seed", "3")
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["algorithm"] == "ppo" and summary["env_steps"] == 200
    assert summary["updates"] > 0 and summary["robot_steps"] > 200
    for name in ("config.json", "train_log.jsonl", "validation.jsonl", "summary.json", "best.pt"):
        assert (out / name).exists(), name
    sidecar = json.loads((out / "best.json").read_text())
    assert sidecar["seeds"]["run_seed"] == 3 and sidecar["model"]["version"] == "mlp-pi-v1"
    for shield in ("off", "predictive"):
        proc = run("simulate", "--scenario", str(DEMO), "--planner", "astar", "--allocator",
                   "auction", "--controller", "ppo", "--checkpoint", str(out / "best.pt"),
                   "--shield", shield, "--max-steps", "40", "--json")  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)["physical_collisions"] == 0
    # a completed run resumes to nothing new; a fresh run into the same directory is refused
    again = run("train", "--config", str(SMOKE), "--out", str(out), "--seed", "3", "--resume")
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout)["env_steps"] == 200
    assert run("train", "--config", str(SMOKE), "--out", str(out), "--seed", "3").returncode != 0
    # --resume is a PPO feature
    madql = run("train", "--config", str(MADQL_SMOKE), "--out", str(tmp_path / "m"), "--resume")
    assert madql.returncode == 2
