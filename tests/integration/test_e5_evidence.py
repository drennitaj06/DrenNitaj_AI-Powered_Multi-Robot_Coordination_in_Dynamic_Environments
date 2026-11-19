"""Evidence tooling added after the Project Issue #4 audit: the reference-controller E5 executor,
the undelivered-task diagnostics (each cause on a constructed episode, derived by hand), replay
fidelity, the paired controller comparison, and the raw-sample performance output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.control.base import Controller
from robocoord.control.baseline import BaselineController
from robocoord.control.reference import ReferenceFollower
from robocoord.experiments import perf
from robocoord.experiments.allocation_analysis import compare_e5_controllers, summarize_e5_prelim
from robocoord.experiments.allocation_runs import run_system_episode, system_episode_setup
from robocoord.experiments.e5_diagnostics import CAUSES, classify, diagnose, to_markdown
from robocoord.experiments.runner import run_experiment
from robocoord.experiments.spec import load_spec
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.scenario import build_scenario
from robocoord.sim.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "experiments"


def episode(
    data: dict[str, Any], controller: Controller | None = None
) -> tuple[Engine, Any, dict[int, int]]:
    coord = AllocationCoordinator(make_planner("astar"), AuctionAllocator())
    eng = Engine(build_scenario(data), coord)
    ctrl = controller or BaselineController()
    ctrl.reset(len(eng.robots))
    last: dict[int, int] = {}
    while not eng.done:
        before = [r.position for r in eng.robots]
        eng.step(ctrl.act)
        for r, b in zip(eng.robots, before, strict=True):
            if r.position != b:
                last[r.id] = eng.t
    return eng, coord, last


def corridor(n: int, rows: int = 1) -> list[str]:
    return ["#" * (n + 2)] + ["#" + "." * n + "#"] * rows + ["#" * (n + 2)]


def causes(items: list[dict[str, Any]]) -> list[str]:
    return sorted(i["cause"] for i in items)


# ---------------------------------------------------------------- executor


def test_reference_controller_executor_is_paired_with_baseline() -> None:
    spec = load_spec(CONFIGS / "e5_prelim_reference.toml")
    base = load_spec(CONFIGS / "e5_prelim.toml")
    assert spec.factors == base.factors and spec.seed_base == base.seed_base
    assert {k: v for k, v in spec.fixed.items() if k != "controller"} == dict(base.fixed)
    params = {"planner": "astar", "allocator": "auction", "dynamics": "static", "seed": 20200}
    _, coord, ctrl = system_episode_setup(spec, params)
    assert isinstance(ctrl, ReferenceFollower) and coord.resolver is None
    _, _, bctrl = system_episode_setup(base, params)
    assert isinstance(bctrl, BaselineController)
    outcome, m = run_system_episode(spec, params)
    assert outcome == "done" and m["controller"] == "reference" and m["deadlock"] is None


# ---------------------------------------------------------------- diagnostics: one cause each


def test_goal_occupied_by_idle_robot() -> None:
    """Robot 0 (x = 1) takes the task (pickup 2, drop-off 5); robot 1 stands idle on x = 5 and
    never moves (no parking): the drop-off stays occupied."""
    eng, coord, last = episode(
        {
            "scenario": {"name": "park", "seed": 1, "max_steps": 30},
            "map": {"rows": corridor(8)},
            "robots": [{"start": [1, 1]}, {"start": [5, 1]}],
            "tasks": [{"pickup": [2, 1], "dropoff": [5, 1], "priority": 1, "release": 0}],
        }
    )
    assert causes(classify(eng, coord, last)) == ["goal_idle_robot"]


def test_route_blocked_by_busy_robot_and_horizon() -> None:
    """Single-lane head-on (stage 5A case): both carriers are blocked by the other, a busy
    robot. With a 3-step cap the same robots are still moving: censored by the horizon."""
    data = {
        "scenario": {"name": "head-on", "seed": 1, "max_steps": 30},
        "map": {"rows": corridor(10)},
        "robots": [{"start": [1, 1]}, {"start": [10, 1]}],
        "tasks": [
            {"pickup": [2, 1], "dropoff": [10, 1], "priority": 1, "release": 0},
            {"pickup": [9, 1], "dropoff": [1, 1], "priority": 2, "release": 0},
        ],
    }
    eng, coord, last = episode(data)
    assert causes(classify(eng, coord, last)) == ["route_busy_robot", "route_busy_robot"]
    eng, coord, last = episode(data, ReferenceFollower())
    assert causes(classify(eng, coord, last)) == ["route_busy_robot", "route_busy_robot"]
    data["scenario"]["max_steps"] = 3  # type: ignore[index]
    eng, coord, last = episode(data)
    assert causes(classify(eng, coord, last)) == ["horizon", "horizon"]


def test_route_contested_free_cell_for_the_plain_follower() -> None:
    """Junction (cross map): both robots carry their task and need the free centre (4, 4) as
    next cell. The reference follower proposes it every step, the arbiter freezes both: a
    vertex contest on a FREE cell (no T4). The baseline lets one pass and both deliver."""
    cross = (
        ["#########"]
        + ["#" + ("." * 7 if y == 4 else "###.###") + "#" for y in range(7, 0, -1)]
        + ["#########"]
    )
    data = {
        "scenario": {"name": "junction", "seed": 1, "max_steps": 30},
        "map": {"rows": cross},
        "robots": [{"start": [3, 4]}, {"start": [4, 3]}],
        "tasks": [
            {"pickup": [3, 4], "dropoff": [4, 7], "priority": 1, "release": 0},
            {"pickup": [4, 3], "dropoff": [4, 6], "priority": 1, "release": 0},
        ],
    }
    eng, coord, last = episode(data, ReferenceFollower())
    assert causes(classify(eng, coord, last)) == ["route_contested_cell", "route_contested_cell"]
    eng, _, _ = episode(data)
    assert eng.summary().tasks_delivered == 2


def test_goal_occupied_by_obstacle() -> None:
    """A patrol obstacle with the drop-off as its only waypoint parks there for good; the
    carrier waits in front of it (robot–obstacle goal deadlock)."""
    eng, coord, last = episode(
        {
            "scenario": {"name": "obst", "seed": 1, "max_steps": 30},
            "map": {"rows": corridor(10)},
            "robots": [{"start": [1, 1]}],
            "obstacles": [{"start": [9, 1], "model": "patrol", "waypoints": [[7, 1]]}],
            "tasks": [{"pickup": [2, 1], "dropoff": [7, 1], "priority": 1, "release": 0}],
        }
    )
    items = classify(eng, coord, last)
    assert causes(items) == ["goal_obstacle"] and items[0]["obstacle_model"] == "patrol"


def test_unassigned_unavailable_after_map_event() -> None:
    eng, coord, last = episode(
        {
            "scenario": {"name": "unavail", "seed": 1, "max_steps": 20},
            "map": {"rows": corridor(10)},
            "robots": [{"start": [1, 1]}],
            "tasks": [{"pickup": [9, 1], "dropoff": [10, 1], "priority": 1, "release": 5}],
            "map_events": [{"time": 3, "cell": [10, 1], "action": "block"}],
        }
    )
    assert causes(classify(eng, coord, last)) == ["unassigned_unavailable"]


# ---------------------------------------------------------------- replay fidelity, comparison


def test_diagnose_replays_reproduce_the_records(tmp_path: Path) -> None:
    stored: dict[str, list[dict[str, Any]]] = {}
    for name in ("e5_prelim", "e5_prelim_reference"):
        spec = load_spec(CONFIGS / f"{name}.toml")
        store = run_experiment(spec, tmp_path / name, workers=1, limit=6, progress=None)
        stored[name] = store.read()
        # mark every record "incomplete" (in memory only) so that each one is replayed and checked
        records = [
            dict(r, metrics=dict(r["metrics"], tasks_released=r["metrics"]["tasks_released"] + 1))
            for r in stored[name]
        ]
        result = diagnose(spec, records)
        assert result["replay_mismatches"] == 0 and len(result["episodes"]) == 6
        assert all(ep["replay_delivered"] == ep["delivered"] for ep in result["episodes"])
        assert "Undelivered tasks" in to_markdown(result)
        assert all(i["cause"] in CAUSES for ep in result["episodes"] for i in ep["undelivered"])
    base, ref = stored["e5_prelim"], stored["e5_prelim_reference"]
    text = compare_e5_controllers(base, ref)
    assert "Paired instances: 6" in text and "| baseline |" in text and "| reference |" in text
    assert "controller: reference" in summarize_e5_prelim(ref)


# ---------------------------------------------------------------- performance raw samples


def test_perf_raw_samples_and_overwrite_guard(tmp_path: Path) -> None:
    res = perf.run(steps=12, seeds=(30500,))
    for label in ("auction", "sequential"):
        c, samples = res["configurations"][label], res["samples"][label]
        assert len(samples) == c["all_steps"]["steps"] == 12
        assert samples[0]["t"] == 0 and samples[0]["bfs_fields"] > 0  # cold cache at t = 0
        e = c["exceedances"]
        assert e["total"] == e["first_step"] + e["map_event_step"] + e["other"]
        assert c["first_steps"]["steps"] == 1
    perf.write_samples(res, tmp_path / "s.csv")
    lines = (tmp_path / "s.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("allocator,seed,t,total_ms") and len(lines) == 1 + 24
    (tmp_path / "nfr_perf1.json").write_text("{}", encoding="utf-8")
    assert perf.main(["--out", str(tmp_path), "--steps", "5"]) == 2  # refuses to overwrite
    assert (tmp_path / "nfr_perf1.json").read_text(encoding="utf-8") == "{}"


def test_diagnostics_cli_and_empty_report(tmp_path: Path) -> None:
    from robocoord.experiments import e5_diagnostics

    assert e5_diagnostics.main([]) == 2  # usage error
    empty = {"experiment": "x", "replay_mismatches": 0, "episodes": []}
    text = to_markdown(empty)
    assert "Incomplete episodes replayed: 0" in text and "| **total** | **0** |" in text
    spec = load_spec(CONFIGS / "e5_prelim.toml")
    run_experiment(spec, tmp_path / "d", workers=1, limit=1, progress=None)
    assert e5_diagnostics.main([str(tmp_path / "d"), str(CONFIGS / "e5_prelim.toml")]) == 0
    assert (tmp_path / "d" / "undelivered.json").exists()
    assert (tmp_path / "d" / "undelivered.md").exists()


@pytest.mark.parametrize("cause", CAUSES)
def test_cause_names_are_documented(cause: str) -> None:
    from robocoord.experiments import e5_diagnostics

    assert f"``{cause}``" in (e5_diagnostics.__doc__ or "") or cause in (
        "route_busy_robot",
        "route_idle_robot",
    )
