"""Issue 6A Phase 4: n-step MADQL training end to end (smoke scale) and the policy-to-simulation
path of an n-step checkpoint."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from robocoord.marl.checkpoint import load_training  # noqa: E402
from robocoord.marl.evaluation import EvalTask, run_episode  # noqa: E402
from robocoord.marl.trainer import TrainingConfig, load_training_config, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "training" / "madql_smoke.toml"


def smoke(n_step: int | None, seed: int = 1, target: int | None = None) -> TrainingConfig:
    cfg = load_training_config(SMOKE, run_seed=seed)
    m = cfg.madql
    if n_step is not None:
        m = replace(m, n_step=n_step)
    if target is not None:
        m = replace(m, target_period=target)
    return replace(cfg, madql=m)


def episodes(d: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in (d / "train_log.jsonl").read_text().splitlines()]
    return [r for r in rows if r["type"] == "episode"]


def weights(d: Path) -> dict[str, Any]:
    q: dict[str, Any] = load_training(next((d / "checkpoints").glob("*.train.pt")))["q"]
    return q


def test_explicit_one_step_is_identical_to_the_default(tmp_path: Path) -> None:
    a = train(smoke(None), tmp_path / "a")
    b = train(smoke(1), tmp_path / "b")
    assert a["config_hash"] == b["config_hash"]
    assert episodes(tmp_path / "a") == episodes(tmp_path / "b")
    wa, wb = weights(tmp_path / "a"), weights(tmp_path / "b")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert (tmp_path / "a" / "best.pt").read_bytes() == (tmp_path / "b" / "best.pt").read_bytes()


def test_n_step_training_is_deterministic_and_changes_learning(tmp_path: Path) -> None:
    a = train(smoke(3), tmp_path / "a")
    b = train(smoke(3), tmp_path / "b")
    c = train(smoke(None), tmp_path / "c")
    assert episodes(tmp_path / "a") == episodes(tmp_path / "b")
    assert a["updates"] == b["updates"] and a["best_key"] == b["best_key"]
    wa, wb, wc = weights(tmp_path / "a"), weights(tmp_path / "b"), weights(tmp_path / "c")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert any(not torch.equal(wa[k], wc[k]) for k in wa)
    assert a["config_hash"] != c["config_hash"]
    cfg = json.loads((tmp_path / "a" / "config.json").read_text())
    assert cfg["config"]["madql"]["n_step"] == 3


def test_n_step_checkpoint_runs_in_the_production_evaluation(tmp_path: Path) -> None:
    train(smoke(3, target=100), tmp_path / "a")
    ckpt = str(tmp_path / "a" / "best.pt")
    for sup, dl in (("off", None), ("R+S", True)):
        task = EvalTask(
            "madql", "full", 10_000, block="A", checkpoint=ckpt, supervisor=sup, deadlock=dl
        )
        r1, r2 = run_episode(task), run_episode(task)
        timing = ("wall_s", "prepare_s", "act_s")
        assert {k: v for k, v in r1.items() if k not in timing} == {
            k: v for k, v in r2.items() if k not in timing
        }
        assert r1["physical_collisions"] == 0 and r1["conflicts"]["INV"] == 0
        assert 0 <= r1["delivered"] <= r1["released"]
