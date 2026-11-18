"""Planner + path monitor inside the engine: X-5 (planning part), X-12 attribution, X-1 for both
planners, plan-adherence metric, CLI selection (Issue #3, acceptance criterion 4)."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from robocoord.cli import main
from robocoord.control.base import SuggestionFollower
from robocoord.core.types import Action
from robocoord.planning import make_planner
from robocoord.planning.coordinator import PlanningCoordinator
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.engine import Engine
from robocoord.sim.trace import TraceWriter
from robocoord.sim.view import WorldView
from robocoord.world.entities import TaskStatus

ROOT = Path(__file__).resolve().parents[2]
PLANNERS = [("astar", {}), ("rrtstar", {"max_iterations": 400})]

CORRIDORS = [
    "#########",
    "#.......#",
    "#.#####.#",
    "#.......#",
    "#########",
]


def _scenario(**over: Any) -> Any:
    data: dict[str, Any] = {
        "scenario": {"name": "t", "seed": 1, "max_steps": 60},
        "map": {"rows": CORRIDORS},
        "robots": [{"start": [1, 1]}],
        "tasks": [{"pickup": [7, 1], "dropoff": [1, 3], "priority": 1, "release": 0}],
        "script": {"actions": [""], "assignments": [[0]]},
    }
    data.update(over)
    return build_scenario(data)


def _run(spec: Any, planner: str, opts: Mapping[str, object], controller: Any = None) -> Engine:
    coord = PlanningCoordinator(make_planner(planner, **opts), spec.script.assignments)
    eng = Engine(spec, coord)
    ctrl = controller or SuggestionFollower()
    ctrl.reset(len(eng.robots))
    eng.run(ctrl.act)
    return eng


def _calls(eng: Engine) -> list[tuple[int, int, str, str]]:
    coord = eng.coordinator
    assert isinstance(coord, PlanningCoordinator) and coord.monitor is not None
    return [(c.t, c.robot, c.trigger, c.status) for c in coord.monitor.calls]


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_follow_plan_without_events_only_t1(planner: str, opts: Mapping[str, object]) -> None:
    eng = _run(_scenario(), planner, opts)
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 1
    assert [c[2] for c in _calls(eng)] == ["T1", "T1"]  # one call per leg, no replanning
    assert m.plan_adherence == 1.0 and m.conflicts_total == 0
    planning = m.extra["planning"]
    assert planning["calls_by_attribution"] == {"allocation": 2}
    assert planning["legs_completed"] == 2 and planning["deviation_rate_per_100"] == 0.0


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_x5_map_event_on_path_triggers_t2_and_detour(
    planner: str, opts: Mapping[str, object]
) -> None:
    spec = _scenario(map_events=[{"time": 2, "cell": [5, 1], "action": "block"}])
    eng = _run(spec, planner, opts)
    calls = _calls(eng)
    assert (2, 0, "T2", "ok") in calls  # detected at the step of the event
    assert eng.terminated and eng.summary().tasks_delivered == 1  # via the upper corridor


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_x5_map_event_on_goal_hold_after_pickup(planner: str, opts: Mapping[str, object]) -> None:
    spec = _scenario(
        map_events=[
            {"time": 9, "cell": [1, 3], "action": "block"},
            {"time": 20, "cell": [1, 3], "action": "unblock"},
        ]
    )
    eng = _run(spec, planner, opts)
    calls = _calls(eng)
    # picked at t = 6 → drop-off leg; the goal is blocked at t = 9: unreachable, hold, retry (T5)
    assert (9, 0, "T2", "unreachable") in calls
    retries = [c for c in calls if c[2] == "T5"]
    assert [c[0] for c in retries][:2] == [12, 15]  # every T_cd = 3 steps
    m = eng.summary()
    assert m.tasks_delivered == 1 and m.blocked_goal_robot_steps == 11  # t = 9 … 19
    assert m.extra["planning"]["releases"] == 0


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_x5_map_event_on_pickup_releases_task(planner: str, opts: Mapping[str, object]) -> None:
    spec = _scenario(
        map_events=[
            {"time": 2, "cell": [7, 1], "action": "block"},
            {"time": 12, "cell": [7, 1], "action": "unblock"},
        ]
    )
    released: list[tuple[int, TaskStatus]] = []
    coord = PlanningCoordinator(make_planner(planner, **opts), [[0]])
    eng = Engine(spec, coord)
    follower = SuggestionFollower()
    while not eng.done:
        out = eng.step(follower.act)
        released.append((out.t, eng.book.tasks[0].status))
    assert (2, 0, "T2", "unreachable") in _calls(eng)
    assert coord.releases == 1
    # the released task is pending (and unavailable) while its pickup is blocked
    assert (3, TaskStatus.PENDING) in released
    assert eng.terminated and eng.summary().tasks_delivered == 1  # re-taken after the unblock


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_x5_obstacle_crossing_path_triggers_t2(planner: str, opts: Mapping[str, object]) -> None:
    # an obstacle patrols the end of the upper corridor, crossing the robot's path
    spec = _scenario(
        tasks=[{"pickup": [7, 3], "dropoff": [1, 1], "priority": 1, "release": 0}],
        robots=[{"start": [1, 3]}],
        obstacles=[{"start": [6, 3], "model": "patrol", "waypoints": [[6, 3], [7, 3]]}],
    )
    eng = _run(spec, planner, opts)
    triggers = {c[2] for c in _calls(eng)}
    assert "T2" in triggers
    assert eng.summary().physical_collisions == 0


class _Deviator(SuggestionFollower):
    """Follows the plan, but once (at t = 2) moves up instead (if allowed)."""

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        out = super().act(view, observations)
        if view.t == 2:
            out[0] = Action.UP if view.masks[0][Action.UP] else Action.LEFT
        return out


def test_x12_deviation_triggers_t3_and_adherence_is_hand_computed() -> None:
    spec = _scenario(
        map={"rows": ["." * 6] * 3},
        robots=[{"start": [0, 0]}],
        tasks=[{"pickup": [5, 0], "dropoff": [0, 2], "priority": 1, "release": 0}],
    )
    eng = _run(spec, "astar", {}, _Deviator())
    calls = _calls(eng)
    assert (3, 0, "T3", "ok") in calls
    m = eng.summary()
    # every non-idle robot-step has a valid plan; exactly one of them deviated
    nonidle = round(m.utilisation * m.steps)
    assert m.plan_adherence == pytest.approx((nonidle - 1) / nonidle)
    assert m.extra["planning"]["deviation_rate_per_100"] == pytest.approx(100.0 / nonidle)


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_x1_traces_identical_for_both_planners(planner: str, opts: Mapping[str, object]) -> None:
    spec = load_scenario(ROOT / "configs/scenarios/planning_detour.toml")

    def trace() -> str:
        buf = io.StringIO()
        coord = PlanningCoordinator(make_planner(planner, **opts), spec.script.assignments)  # type: ignore[union-attr]
        eng = Engine(spec, coord, trace=TraceWriter(buf))
        eng.run(SuggestionFollower().act)
        return buf.getvalue()

    a, b = trace(), trace()
    assert a == b and '"plans"' in a
    assert '"time_s"' not in a  # no timings in traces


def test_planners_selectable_from_cli(capsys: pytest.CaptureFixture[str]) -> None:
    scenario = str(ROOT / "configs/scenarios/planning_detour.toml")
    for planner in ("astar", "rrtstar"):
        assert (
            main(
                [
                    "simulate",
                    "--scenario",
                    scenario,
                    "--planner",
                    planner,
                    "--no-animate",
                    "--rrt-iterations",
                    "300",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert f"planning ({planner})" in out and "delivered 2/2" in out
    assert main(["simulate", "--scenario", scenario, "--controller", "follow", "--no-animate"]) == 2
    rw = str(ROOT / "configs/scenarios/random_warehouse.toml")
    assert main(["simulate", "--scenario", rw, "--planner", "astar", "--no-animate"]) == 2


def test_release_is_not_reoffered_until_something_changes() -> None:
    """Review finding (V-6): a robot cut off from an *available* pickup releases the task once and
    is not re-offered it every T_cd steps; after a map event opens the wall it is re-offered and
    the task is delivered."""
    spec = build_scenario(
        {
            "scenario": {"name": "cutoff", "seed": 1, "max_steps": 60},
            "map": {"rows": [".@....", ".@....", ".@...."]},
            "robots": [{"start": [0, 0]}],
            "tasks": [{"pickup": [3, 0], "dropoff": [5, 2], "priority": 1, "release": 0}],
            "map_events": [{"time": 12, "cell": [1, 1], "action": "unblock"}],
            "script": {"actions": [""], "assignments": [[0]]},
        }
    )
    eng = _run(spec, "astar", {})
    coord = eng.coordinator
    assert isinstance(coord, PlanningCoordinator)
    assert coord.releases == 1
    calls = _calls(eng)
    assert calls[0] == (0, 0, "T1", "unreachable")
    assert calls[1][0] == 12 and calls[1][2:] == ("T1", "ok")  # re-offered after the map event
    assert eng.terminated and eng.summary().tasks_delivered == 1
