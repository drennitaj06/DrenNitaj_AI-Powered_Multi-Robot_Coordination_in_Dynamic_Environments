"""Endpoint recovery in full episodes (engine + allocation + planner + monitor + controller +
shield + arbiter; algorithms §5.6). Each scenario is small enough to be followed by hand; the
off/on pairs show what the recovery changes and that ``endpoint=None`` keeps the earlier
behaviour."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.control.base import Controller
from robocoord.control.baseline import BaselineController
from robocoord.control.reference import ReferenceFollower
from robocoord.control.shield import Shield
from robocoord.experiments.allocation_runs import e5_scenario, run_system_episode
from robocoord.experiments.e5_diagnostics import classify
from robocoord.experiments.spec import load_spec
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.planning.endpoint import EndpointParams
from robocoord.scenario import build_scenario
from robocoord.sim.engine import Engine

ROOT = Path(__file__).resolve().parents[2]


def run(
    data: dict[str, Any],
    *,
    endpoint: bool = True,
    controller: str = "baseline",
    planner: str = "astar",
    allocator: str = "auction",
    shield: str = "off",
) -> tuple[Engine, AllocationCoordinator, dict[str, Any]]:
    alloc = AuctionAllocator() if allocator == "auction" else SequentialAllocator()
    coord = AllocationCoordinator(
        make_planner(planner),
        alloc,
        deadlock=DetourParams() if controller == "baseline" else None,
        endpoint=EndpointParams() if endpoint else None,
    )
    eng = Engine(
        build_scenario(data), coord, shield=Shield("predictive") if shield != "off" else None
    )
    ctrl: Controller = BaselineController() if controller == "baseline" else ReferenceFollower()
    ctrl.reset(len(eng.robots))
    last: dict[int, int] = {}
    while not eng.done:
        before = [r.position for r in eng.robots]
        eng.step(ctrl.act)
        for r, b in zip(eng.robots, before, strict=True):
            if r.position != b:
                last[r.id] = eng.t
    m = eng.summary()
    assert m.physical_collisions == 0
    stats = m.extra["planning"].get("endpoint") or {}
    return eng, coord, {"m": m, "ep": stats, "last": last}


def open_rows(w: int, h: int) -> list[str]:
    return ["." * w] * h


def corridor(n: int) -> list[str]:
    return ["#" * (n + 2), "#" + "." * n + "#", "#" * (n + 2)]


def delivered(res: dict[str, Any]) -> int:
    return int(res["m"].tasks_delivered)


# ---------------------------------------------------------------- idle robot on an endpoint

REUSE = {
    # robot 0 delivers task 0 at (3, 2) and becomes idle there; task 1 (released at t = 6) needs
    # the same cell as its drop-off and goes to robot 1 (its pickup is next to robot 1)
    "scenario": {"name": "reuse", "seed": 1, "max_steps": 60},
    "map": {"rows": open_rows(7, 5)},
    "robots": [{"start": [1, 2]}, {"start": [6, 0]}],
    "tasks": [
        {"pickup": [2, 2], "dropoff": [3, 2], "priority": 1, "release": 0},
        {"pickup": [6, 1], "dropoff": [3, 2], "priority": 1, "release": 6},
    ],
}


@pytest.mark.parametrize("controller", ["baseline", "reference"])
def test_endpoint_reuse_after_delivery(controller: str) -> None:
    eng, coord, off = run(REUSE, endpoint=False, controller=controller)
    assert delivered(off) == 1
    assert [i["cause"] for i in classify(eng, coord, off["last"])] == ["goal_idle_robot"]
    eng, _, on = run(REUSE, controller=controller)
    assert delivered(on) == 2 and eng.done and eng.terminated
    ep = on["ep"]
    assert ep["parking_needed"] == ep["parking_started"] == ep["parking_completed"] == 1
    assert ep["idle_on_endpoint_at_end"] == 0 and ep["unresolved_idle_robot_at_end"] == 0
    calls = on["m"].extra["planning"]["calls_by_trigger"]
    assert calls["T8"] >= 1  # the parking goal is a T8 goal change


def test_shared_endpoints_between_tasks() -> None:
    """Task 1's pickup is task 0's drop-off and task 2 reuses the same drop-off: every task is
    delivered with recovery on."""
    data = {
        "scenario": {"name": "shared", "seed": 2, "max_steps": 80},
        "map": {"rows": open_rows(7, 5)},
        "robots": [{"start": [0, 2]}, {"start": [6, 4]}, {"start": [6, 0]}],
        "tasks": [
            {"pickup": [1, 2], "dropoff": [3, 2], "priority": 1, "release": 0},
            {"pickup": [3, 2], "dropoff": [5, 4], "priority": 1, "release": 4},
            {"pickup": [6, 1], "dropoff": [3, 2], "priority": 2, "release": 8},
        ],
    }
    _, _, on = run(data)
    assert delivered(on) == 3


def test_identical_pickup_and_dropoff_cells_are_rejected_by_the_model() -> None:
    from robocoord.scenario import ScenarioError

    with pytest.raises(ScenarioError):
        build_scenario(
            {
                "scenario": {"name": "same", "seed": 1, "max_steps": 5},
                "map": {"rows": open_rows(3, 1)},
                "robots": [{"start": [0, 0]}],
                "tasks": [{"pickup": [1, 0], "dropoff": [1, 0], "priority": 1, "release": 0}],
            }
        )


NO_PARKING = {
    # corridor x = 1 … 3: robot 1 idle on (3, 1) (a leaf); robot 0 carries a task to (3, 1).
    # Robot 1 can only leave through robot 0's cell and robot 0 has no retreat cell.
    "scenario": {"name": "trapped", "seed": 1, "max_steps": 40},
    "map": {"rows": corridor(3)},
    "robots": [{"start": [1, 1]}, {"start": [3, 1]}],
    "tasks": [{"pickup": [1, 1], "dropoff": [3, 1], "priority": 1, "release": 0}],
}


def test_no_safe_parking_or_retreat_cell_is_reported() -> None:
    _, coord, on = run(NO_PARKING)
    ep = on["ep"]
    assert delivered(on) == 0
    assert ep["parking_blocked_dynamic"] == 1 and ep["no_retreat_cell"] == 1
    assert ep["blockages_persistent_robot"] == 1
    assert ep["unresolved_idle_robot_at_end"] == 1 and ep["idle_on_endpoint_at_end"] == 1
    assert coord.endpoint is not None and coord.endpoint.robot_report(0)["no_retreat_cell"]


# ---------------------------------------------------------------- obstacles on an endpoint


def patrol_on_goal(role: str) -> dict[str, Any]:
    """Open 7 × 5 map. The obstacle walks west along y = 2 towards its single waypoint (0, 2).
    Robot 0 stands on (2, 2) from t = 0; the task (released at t = 2) has (3, 2) as ``role``
    (as drop-off, the pickup is the robot's own cell, so it carries the task from t = 2).
    The obstacle reaches (3, 2) at t = 2, wants (2, 2) (the robot's cell) and stays: obstacles
    never enter robot cells, and the robot waits for its goal (the E5 mutual blockage)."""
    pickup, dropoff = ([3, 2], [5, 4]) if role == "pickup" else ([2, 2], [3, 2])
    data: dict[str, Any] = {
        "scenario": {"name": f"patrol-{role}", "seed": 1, "max_steps": 80},
        "map": {"rows": open_rows(7, 5)},
        "robots": [{"start": [2, 2]}],
        "obstacles": [{"start": [5, 2], "model": "patrol", "waypoints": [[0, 2]]}],
        "tasks": [{"pickup": pickup, "dropoff": dropoff, "priority": 1, "release": 2}],
    }
    return data


@pytest.mark.parametrize("role", ["pickup", "dropoff"])
def test_persistent_patrol_on_an_endpoint_is_recovered(role: str) -> None:
    eng, coord, off = run(patrol_on_goal(role), endpoint=False)
    assert delivered(off) == 0
    assert [i["cause"] for i in classify(eng, coord, off["last"])] == ["goal_obstacle"]
    _, _, on = run(patrol_on_goal(role))
    ep = on["ep"]
    assert delivered(on) == 1
    assert ep["blockages_persistent_obstacle"] >= 1 and ep["retreats_started"] >= 1
    assert ep["retreats_succeeded"] >= 1 and ep["goal_reached_after_recovery"] >= 1
    assert ep["unresolved_obstacle_at_end"] == 0


def test_temporary_obstacle_occupancy_clears_without_recovery() -> None:
    """A bouncing obstacle crosses the drop-off (3, 2) vertically; the carrier waits less than
    W steps and delivers without any retreat."""
    data = {
        "scenario": {"name": "crossing", "seed": 1, "max_steps": 60},
        "map": {"rows": open_rows(7, 5)},
        "robots": [{"start": [0, 2]}],
        "obstacles": [{"start": [3, 4], "model": "bounce", "direction": "down"}],
        "tasks": [{"pickup": [1, 2], "dropoff": [3, 2], "priority": 1, "release": 0}],
    }
    _, _, on = run(data)
    ep = on["ep"]
    assert delivered(on) == 1 and ep["retreats_started"] == 0
    assert ep["blockages_persistent"] == 0


def test_never_leaving_obstacle_exhausts_the_attempts_and_is_reported() -> None:
    """A one-waypoint patrol parked on the drop-off never leaves (a documented unresolvable
    case): M = 3 retreats time out, the episode is exhausted and reported at the end."""
    data = {
        "scenario": {"name": "parked", "seed": 1, "max_steps": 120},
        "map": {"rows": open_rows(7, 5)},
        "robots": [{"start": [0, 2]}],
        "obstacles": [{"start": [3, 2], "model": "patrol", "waypoints": [[3, 2]]}],
        "tasks": [{"pickup": [1, 2], "dropoff": [3, 2], "priority": 1, "release": 0}],
    }
    _, coord, on = run(data)
    ep = on["ep"]
    assert delivered(on) == 0
    assert ep["retreats_started"] == ep["retreats_timed_out"] == 3
    assert ep["attempts_exhausted"] == 1 and ep["unresolved_obstacle_at_end"] == 1
    assert ep["retreats_succeeded"] == 0
    assert coord.endpoint is not None and coord.endpoint.robot_report(0)["attempts_exhausted"]


# ---------------------------------------------------------------- compatibility and safety

COMBOS = list(
    itertools.product(("astar", "rrtstar"), ("auction", "sequential"), ("baseline", "reference"))
)


@pytest.mark.parametrize(("planner", "allocator", "controller"), COMBOS)
@pytest.mark.parametrize("shield", ["off", "predictive"])
def test_every_combination_recovers(
    planner: str, allocator: str, controller: str, shield: str
) -> None:
    _, _, reuse = run(
        REUSE, planner=planner, allocator=allocator, controller=controller, shield=shield
    )
    assert delivered(reuse) == 2 and reuse["ep"]["parking_completed"] == 1
    _, _, patrol = run(
        patrol_on_goal("dropoff"),
        planner=planner,
        allocator=allocator,
        controller=controller,
        shield=shield,
    )
    assert delivered(patrol) == 1 and patrol["ep"]["retreats_succeeded"] >= 1
    if controller == "baseline":
        c = patrol["m"].conflicts
        assert c["RRv"] == c["RRs"] == c["ROs"] == c["INV"] == 0  # G19 for any suggestions


@pytest.mark.parametrize("controller", ["baseline", "reference"])
@pytest.mark.parametrize("shield", ["off", "predictive"])
def test_e5_episodes_with_recovery_are_safe(controller: str, shield: str) -> None:
    """Development seeds (not E5 test seeds), full dynamics: no physical collision, the baseline
    keeps 0 robot–robot conflicts, and the counter identities of `04` §5.6 hold."""
    spec = load_spec(ROOT / "configs" / "experiments" / "e5_prelim.toml")
    spec = replace(spec, fixed={**spec.fixed, "controller": controller, "endpoint": "on"})
    for seed in (10000, 10003):
        params = {"planner": "astar", "allocator": "auction", "dynamics": "full", "seed": seed}
        if shield == "predictive":
            scenario = e5_scenario(
                seed,
                "full",
                size=32,
                density=0.15,
                robots=8,
                tasks_per_robot=2,
                obstacles=8,
                max_steps=400,
                arrival_rate=0.05,
                arrivals_until=200,
            )
            coord = AllocationCoordinator(
                make_planner("astar"),
                AuctionAllocator(),
                deadlock=DetourParams() if controller == "baseline" else None,
                endpoint=EndpointParams(),
            )
            eng = Engine(scenario, coord, shield=Shield("predictive"))
            ctrl: Controller = (
                BaselineController() if controller == "baseline" else ReferenceFollower()
            )
            ctrl.reset(len(eng.robots))
            while not eng.done:
                eng.step(ctrl.act)
            m = eng.summary()
            assert "shield" in m.extra
            ep = m.extra["planning"]["endpoint"]
            physical, conflicts = m.physical_collisions, m.conflicts
        else:
            outcome, rec = run_system_episode(spec, params)
            assert outcome == "done"
            ep, physical, conflicts = rec["endpoint"], rec["physical_collisions"], rec["conflicts"]
        assert physical == 0
        if controller == "baseline":
            assert conflicts["RRv"] == conflicts["RRs"] == 0
        assert ep["retreats_started"] == (
            ep["retreats_succeeded"]
            + ep["retreats_timed_out"]
            + ep["retreats_aborted"]
            + ep["retreats_cancelled"]
            + ep["retreats_active_at_end"]
        )
        assert ep["parking_started"] == (
            ep["parking_completed"]
            + ep["parking_cancelled"]
            + ep["parking_aborted"]
            + ep["parkings_active_at_end"]
        )


def test_endpoint_off_equals_absent_key_and_adds_no_recovery() -> None:
    """Within the current code: an explicit ``endpoint = "off"`` gives exactly the metrics of the
    absent key (both mean no recovery), the coordinator has no endpoint statistics and no T8
    trigger, and an invalid value is rejected. This does not compare against the code before
    endpoint recovery existed; that comparison is the offline replay of the stored records,
    documented in ``docs/verification-log.md`` V-11 (not a repository test)."""
    spec = load_spec(ROOT / "configs" / "experiments" / "e5_prelim.toml")
    params = {"planner": "astar", "allocator": "auction", "dynamics": "full", "seed": 10001}
    _, a = run_system_episode(spec, params)
    off = replace(spec, fixed={**spec.fixed, "endpoint": "off"})
    _, b = run_system_episode(off, params)
    a.pop("wall_s_per_step")
    b.pop("wall_s_per_step")
    assert a == b and a["endpoint"] is None and "T8" not in a["calls_by_trigger"]
    with pytest.raises(ValueError):
        run_system_episode(replace(spec, fixed={**spec.fixed, "endpoint": "maybe"}), params)


def test_deterministic_with_recovery() -> None:
    _, ca, a = run(patrol_on_goal("pickup"), planner="rrtstar")
    _, cb, b = run(patrol_on_goal("pickup"), planner="rrtstar")
    assert a["m"].extra["planning"] == b["m"].extra["planning"] and a["last"] == b["last"]
    assert ca.endpoint is not None and cb.endpoint is not None
    assert [e.to_json() for e in ca.endpoint.events] == [e.to_json() for e in cb.endpoint.events]
