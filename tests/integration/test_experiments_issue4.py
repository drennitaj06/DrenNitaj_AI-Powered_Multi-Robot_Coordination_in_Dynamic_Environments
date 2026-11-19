"""Experiment smoke tests for Issue #4 (E3a, E3b, preliminary E5, NFR-PERF-1): the executors run,
records are written, the instances are paired and deterministic, and the summaries render."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robocoord.experiments import perf
from robocoord.experiments.allocation_runs import (
    e3a_matrix,
    e3b_scenario,
    e5_scenario,
    run_assignment,
)
from robocoord.experiments.analysis import summarize
from robocoord.experiments.runner import run_experiment
from robocoord.experiments.spec import load_spec, parse_spec

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiments"


def test_e3a_instances_are_identical_across_methods_and_guarantees_hold() -> None:
    spec = parse_spec(
        {
            "experiment": {"name": "e3a_assignment", "kind": "assignment", "seed_base": 30400},
            "factors": {"k": [7], "C": [100], "matrix": [0], "method": ["auction_scaling"]},
            "fixed": {"repeats": 1},
        }
    )
    a, b = e3a_matrix(30400, 7, 100, 0), e3a_matrix(30400, 7, 100, 0)
    assert np.array_equal(a, b) and not np.array_equal(a, e3a_matrix(30400, 7, 100, 1))
    sparse = e3a_matrix(30400, 20, 100, 0, density=0.5)
    assert 0.3 < np.count_nonzero(sparse) / sparse.size < 0.7
    results = {
        m: run_assignment(spec, {"k": 30, "C": 1000, "matrix": 2, "method": m})[1]
        for m in ("auction_scaling", "auction_fixed1", "sequential", "hungarian")
    }
    opt = results["hungarian"]["opt"]
    assert all(r["opt"] == opt for r in results.values())  # same matrix for every method
    assert results["auction_scaling"]["gap"] == 0 and results["hungarian"]["gap"] == 0
    assert 0 <= results["auction_fixed1"]["gap"] <= 30  # V ≥ OPT − k·ε, ε = 1
    assert 2 * results["sequential"]["value"] >= opt


def test_scenario_generators_pair_their_factors() -> None:
    a = e3b_scenario(
        20100, size=24, density=0.15, robots=4, rate=0.16, arrivals_until=200, max_steps=400
    )
    b = e3b_scenario(
        20100, size=24, density=0.15, robots=4, rate=0.16, arrivals_until=200, max_steps=400
    )
    assert a.grid.to_rows() == b.grid.to_rows() and a.robots == b.robots
    kw = dict(size=32, density=0.15, robots=8, tasks_per_robot=2, obstacles=8, max_steps=400,
              arrival_rate=0.05, arrivals_until=200)  # fmt: skip
    static = e5_scenario(20200, "static", **kw)  # type: ignore[arg-type]
    moving = e5_scenario(20200, "obstacles", **kw)  # type: ignore[arg-type]
    full = e5_scenario(20200, "full", **kw)  # type: ignore[arg-type]
    assert static.grid.to_rows() == moving.grid.to_rows() == full.grid.to_rows()
    assert static.robots == moving.robots == full.robots and static.tasks == full.tasks
    assert (
        not static.obstacles and len(moving.obstacles) == 8 and moving.obstacles == full.obstacles
    )
    assert not moving.map_events and full.map_events and full.arrivals.kind == "bernoulli"
    with pytest.raises(ValueError):
        e5_scenario(20200, "chaos", **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["e3a_assignment", "e3b_priority", "e5_prelim"])
def test_configs_run_and_summarise(tmp_path: Path, name: str) -> None:
    spec = load_spec(CONFIGS / f"{name}.toml")
    store = run_experiment(spec, tmp_path / name, workers=1, limit=4, progress=None)
    records = store.read()
    assert len(records) == 4 and all(r["outcome"] == "done" for r in records)
    text = summarize(tmp_path / name)
    assert text.startswith(f"## {name}") and (tmp_path / name / "summary.md").exists()


def test_perf_benchmark_smoke(tmp_path: Path) -> None:
    res = perf.run(steps=15, seeds=(30500,))
    for label in ("auction", "sequential"):
        c = res["configurations"][label]
        assert c["all_steps"]["steps"] == 15
        assert set(c["slowest_1pct_share"]) == set(perf.COMPONENTS)
    assert "NFR-PERF-1" in perf.to_markdown(res)
