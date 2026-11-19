"""Evidence tooling of the endpoint-recovery / NFR-PERF-1 follow-up: the paired before/after
benchmark (workloads R and F), the recovery E5 configurations, the paired recovery comparison
and the recovery-aware undelivered-task diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.control.baseline import BaselineController
from robocoord.experiments import perf
from robocoord.experiments.allocation_analysis import compare_e5_recovery, summarize_e5_prelim
from robocoord.experiments.e5_diagnostics import classify, diagnose, to_markdown
from robocoord.experiments.runner import run_experiment
from robocoord.experiments.spec import load_spec
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.planning.endpoint import EndpointParams
from robocoord.scenario import build_scenario
from robocoord.sim.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiments"


# ---------------------------------------------------------------- benchmark


def test_workload_r_pairs_legacy_and_fast_on_identical_trajectories() -> None:
    res = perf.run_workloads(("R",), steps=40, seeds=(30500,), impls=("legacy", "fast"))
    assert res["impls"] == ["legacy", "fast"] and "ac_line" in res["power_start"]
    assert set(res["trajectories_identical"]) == {f"R {c.label}" for c in perf.WORKLOADS["R"]}
    assert all(res["trajectories_identical"].values())
    for cfg in perf.WORKLOADS["R"]:
        legacy = res["runs"]["legacy"]["R"][cfg.label]
        fast = res["runs"]["fast"]["R"][cfg.label]
        assert legacy["outcome"] == fast["outcome"]  # identical behaviour
        s = fast["all_steps"]
        assert {"median_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "exceedances"} <= set(s)
        assert s["exceedances"] == round(s["exceedance_pct"] * s["steps"] / 100)
        assert fast["first_steps"]["steps"] == 1 and fast["config"]["endpoint"]
        # O1/O2 + symmetric reuse compute fewer fields on the cold first step
        assert fast["bfs_fields_mean"]["first_steps"] < legacy["bfs_fields_mean"]["first_steps"]
    n = sum(c["all_steps"]["steps"] for i in res["runs"] for c in res["runs"][i]["R"].values())
    assert len(res["samples"]) == n and {s["impl"] for s in res["samples"]} == {"legacy", "fast"}
    assert "Identical state digests at every step" in perf.workloads_markdown(res)


@pytest.mark.slow
def test_workload_f_runs_every_configuration() -> None:
    res = perf.run_workloads(("F",), steps=30, seeds=(30500,), impls=("legacy", "fast"))
    assert len(res["runs"]["fast"]["F"]) == 8 and all(res["trajectories_identical"].values())
    for c in res["runs"]["fast"]["F"].values():
        assert c["outcome"]["physical_collisions"] == 0 and c["config"]["shield"]


def test_benchmark_cli_validation_and_samples(tmp_path: Path) -> None:
    assert perf.main(["--out", str(tmp_path / "a"), "--workload", "X"]) == 2
    assert perf.main(["--out", str(tmp_path / "b"), "--impl", "turbo"]) == 2
    res = perf.run_workloads(("R",), steps=15, seeds=(30501,), impls=("fast",))
    perf.write_workload_samples(res, tmp_path / "s.csv")
    lines = (tmp_path / "s.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("impl,workload,config,seed,t,total_ms") and lines[0].endswith(
        "state"
    )
    assert len(lines) == 1 + len(res["samples"]) and res["trajectories_identical"] == {}


# ---------------------------------------------------------------- E5 recovery configurations


@pytest.mark.parametrize(
    ("recovery", "pair"),
    [("e5_recovery", "e5_prelim"), ("e5_recovery_reference", "e5_prelim_reference")],
)
def test_recovery_configs_pair_with_the_stored_halves(recovery: str, pair: str) -> None:
    on, off = load_spec(CONFIGS / f"{recovery}.toml"), load_spec(CONFIGS / f"{pair}.toml")
    assert on.factors == off.factors and on.seed_base == off.seed_base
    assert on.planners == off.planners and on.name == recovery
    assert dict(on.fixed) == {**dict(off.fixed), "endpoint": "on"} and "endpoint" not in off.fixed


def test_recovery_comparison_and_summary(tmp_path: Path) -> None:
    stored: dict[str, list[dict[str, Any]]] = {}
    for name in ("e5_prelim", "e5_recovery"):
        spec = load_spec(CONFIGS / f"{name}.toml")
        stored[name] = run_experiment(
            spec, tmp_path / name, workers=1, limit=4, progress=None
        ).read()
    assert all(r["metrics"]["endpoint"] is None for r in stored["e5_prelim"])
    assert all(r["metrics"]["endpoint"] is not None for r in stored["e5_recovery"])
    text = compare_e5_recovery(stored["e5_prelim"], stored["e5_recovery"])
    assert "Paired instances: 4" in text and "retreats_started" in text
    assert "Endpoint recovery: on." in summarize_e5_prelim(stored["e5_recovery"])
    assert "Endpoint recovery: off." in summarize_e5_prelim(stored["e5_prelim"])
    spec = load_spec(CONFIGS / "e5_recovery.toml")
    records = [
        dict(r, metrics=dict(r["metrics"], tasks_released=r["metrics"]["tasks_released"] + 1))
        for r in stored["e5_recovery"]
    ]  # every record replayed (in memory only)
    result = diagnose(spec, records)
    assert result["replay_mismatches"] == 0 and len(result["episodes"]) == 4


# ---------------------------------------------------------------- recovery-aware diagnostics


def episode(data: dict[str, Any]) -> tuple[Engine, AllocationCoordinator, dict[int, int]]:
    coord = AllocationCoordinator(
        make_planner("astar"),
        AuctionAllocator(),
        deadlock=DetourParams(),
        endpoint=EndpointParams(),
    )
    eng = Engine(build_scenario(data), coord)
    ctrl = BaselineController()
    ctrl.reset(len(eng.robots))
    last: dict[int, int] = {}
    while not eng.done:
        before = [r.position for r in eng.robots]
        eng.step(ctrl.act)
        for r, b in zip(eng.robots, before, strict=True):
            if r.position != b:
                last[r.id] = eng.t
    return eng, coord, last


def test_diagnostics_record_the_recovery_outcome() -> None:
    eng, coord, last = episode(
        {
            "scenario": {"name": "parked", "seed": 1, "max_steps": 120},
            "map": {"rows": ["." * 7] * 5},
            "robots": [{"start": [0, 2]}],
            "obstacles": [{"start": [3, 2], "model": "patrol", "waypoints": [[3, 2]]}],
            "tasks": [{"pickup": [1, 2], "dropoff": [3, 2], "priority": 1, "release": 0}],
        }
    )
    (item,) = classify(eng, coord, last)
    assert item["cause"] == "goal_obstacle"
    assert item["recovery"]["attempts_exhausted"] and item["recovery"]["retreat_attempts"] == 3
    eng, coord, last = episode(
        {
            "scenario": {"name": "trapped", "seed": 1, "max_steps": 40},
            "map": {"rows": ["#####", "#...#", "#####"]},
            "robots": [{"start": [1, 1]}, {"start": [3, 1]}],
            "tasks": [{"pickup": [1, 1], "dropoff": [3, 1], "priority": 1, "release": 0}],
        }
    )
    (item,) = classify(eng, coord, last)
    assert item["cause"] == "goal_idle_robot" and item["recovery"]["no_retreat_cell"]
    assert item["occupant_parking"]["parking_blocked_dynamic"]
    md = to_markdown(
        {
            "experiment": "x",
            "replay_mismatches": 0,
            "episodes": [
                {
                    "params": {"planner": "astar", "allocator": "auction", "dynamics": "static"},
                    "undelivered": [item],
                }
            ],
        }
    )
    assert "Endpoint recovery state of the 1 goal-occupancy items" in md
    assert "no_retreat_cell 1" in md and "occupant_parking_blocked_dynamic 1" in md
