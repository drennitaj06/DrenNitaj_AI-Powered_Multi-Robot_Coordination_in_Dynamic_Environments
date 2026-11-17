"""Engine integration tests: pipeline, invariants (X-2), determinism (X-1), metrics (X-6)."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.control.base import RandomController, ScriptedController, StayController
from robocoord.core.rng import make_streams
from robocoord.core.types import Action
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.conflicts import physical_violations
from robocoord.sim.coordinator import AssignmentAPI, ScriptedCoordinator
from robocoord.sim.engine import Engine, EngineStateError
from robocoord.sim.trace import TraceWriter, read_trace
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode, TaskStatus

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "configs" / "scenarios" / "demo_delivery.toml"
RANDOM = ROOT / "configs" / "scenarios" / "random_warehouse.toml"
S, U, D, L, R = Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT


def scenario(**over: Any) -> Any:
    data: dict[str, Any] = {
        "scenario": {"name": "x", "seed": 0, "max_steps": 20},
        "map": {"rows": ["......"]},
        "robots": [{"start": [0, 0]}],
        "tasks": [{"pickup": [2, 0], "dropoff": [5, 0], "priority": 2, "release": 1}],
        "script": {"actions": ["RRRRR"], "assignments": [[0]]},
    }
    data.update(over)
    return build_scenario(data)


def run_scripted(
    spec: Any, trace: TraceWriter | None = None, max_steps: int | None = None
) -> Engine:
    assert spec.script is not None
    eng = Engine(
        spec, ScriptedCoordinator(spec.script.assignments), trace=trace, max_steps=max_steps
    )
    ctrl = ScriptedController(spec.script.actions)
    ctrl.reset(len(eng.robots))
    eng.run(ctrl.act)
    return eng


# ------------------------------------------------------------------ hand-computed episode (X-6)


def test_single_robot_episode_hand_computed() -> None:
    eng = run_scripted(scenario())
    m = eng.summary()
    # release at t=1 → assigned at step 1 (robot at (1,0)); at (2,0) at t=2 → picked at 2;
    # reaches (5,0) at t=5 → delivered at 5; flow time = 5 - 1 = 4
    task = eng.book.tasks[0]
    assert (task.pickup_time, task.completion_time) == (2, 5)
    assert eng.terminated and not eng.truncated and m.steps == 5
    assert m.tasks_delivered == m.tasks_released == 1 and m.makespan == 5
    assert m.mean_flow_time == 4.0 and m.weighted_flow_time == 4.0
    assert m.flow_time_by_priority == {"1": None, "2": 4.0, "3": None}
    assert m.waiting_max_by_priority["2"] == 1 and m.waiting_censored == 0
    # legs: (1,0)->(2,0) optimal 1, moves 1; (2,0)->(5,0) optimal 3, moves 3
    assert m.legs_completed == 2 and m.path_efficiency == 1.0
    # non-idle robot-steps: steps 1..4 (4 of 5), none of them 'stay'
    assert m.utilisation == pytest.approx(4 / 5) and m.wait_ratio == 0.0
    assert m.conflicts_total == 0 and m.physical_collisions == 0
    assert m.throughput_per_100_steps == pytest.approx(20.0)
    assert m.conflicts_avoided is None  # no planner suggestions in Issue #2


def test_waiting_and_detour_metrics() -> None:
    # the robot waits two steps and makes a detour: path efficiency < 1, wait ratio > 0
    spec = build_scenario(
        {
            "scenario": {"name": "w", "max_steps": 30},
            "map": {"rows": ["...", "..."]},
            "robots": [{"start": [0, 0]}],
            "tasks": [{"pickup": [2, 0], "dropoff": [0, 0]}],
            "script": {"actions": ["SS U RR D LL"], "assignments": [[0]]},
        }
    )
    m = run_scripted(spec).summary()
    assert m.tasks_delivered == 1
    # leg 1: optimal 2, moves 4 (U R R D); leg 2: optimal 2, moves 2 → 4/6
    assert m.path_efficiency == pytest.approx(4 / 6)
    assert m.wait_ratio == pytest.approx(2 / 8)


def test_censored_waiting_and_truncation() -> None:
    spec = scenario(script={"actions": ["S"], "assignments": [[]]})
    eng = run_scripted(spec, max_steps=7)
    m = eng.summary()
    assert eng.truncated and not eng.terminated and m.steps == 7
    assert m.tasks_released == 1 and m.tasks_delivered == 0 and m.completion_rate == 0.0
    assert m.waiting_censored == 1 and m.waiting_max_by_priority["2"] == 6  # age at t=7
    assert m.makespan is None and m.mean_flow_time is None


# ------------------------------------------------------------------ conflicts in the engine


def test_conflicts_recorded_and_prevented_in_engine() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "c", "max_steps": 4},
            "map": {"rows": ["....."]},
            "robots": [{"start": [0, 0]}, {"start": [2, 0]}, {"start": [4, 0]}],
            "script": {"actions": ["R", "SL", "L"], "assignments": [[], [], []]},
        }
    )
    eng = Engine(spec, ScriptedCoordinator(spec.script.assignments))  # type: ignore[union-attr]
    ctrl = ScriptedController(spec.script.actions)  # type: ignore[union-attr]
    ctrl.reset(3)
    out = eng.step(ctrl.act)
    # step 0: r0 R → (1,0) ok; r1 S; r2 L → (3,0) ok: no conflict
    assert out.events == frozenset() and eng.robots[2].position == (3, 0)
    out = eng.step(ctrl.act)
    # step 1: r1 L → (1,0) occupied by r0 (stays) → RRv, r1 frozen; r2 retries? no, r2 script ended
    assert {(e.kind_code, e.entities) for e in out.events} == {("RRv", ("r0", "r1"))}
    assert eng.robots[1].position == (2, 0)
    assert eng.metrics.conflict_counts["RRv"] == 1


def test_obstacle_never_enters_robot_and_map_event_deferral() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "o", "max_steps": 6},
            "map": {"rows": ["...."]},
            "robots": [{"start": [1, 0]}],
            "obstacles": [{"start": [3, 0], "model": "bounce", "direction": "left"}],
            "map_events": [{"time": 0, "cell": [1, 0], "action": "block"}],
        }
    )
    eng = Engine(spec)
    stay = StayController()
    for _ in range(4):
        eng.step(stay.act)
        assert eng.obstacles[0].position != eng.robots[0].position
        assert eng.grid.is_free((1, 0))  # block deferred while the robot is there
    assert eng.obstacles[0].position == (2, 0)  # waiting next to the robot


# ------------------------------------------------------------------ determinism (X-1)


def trace_bytes(path: Path, seed: int | None = None, controller: str = "scripted") -> str:
    spec = load_scenario(path, seed=seed)
    buf = io.StringIO()
    if controller == "scripted":
        assert spec.script is not None
        eng = Engine(spec, ScriptedCoordinator(spec.script.assignments), trace=TraceWriter(buf))
        ctrl: Any = ScriptedController(spec.script.actions)
    else:
        eng = Engine(spec, trace=TraceWriter(buf))
        ctrl = RandomController(make_streams(spec.seed).generator("policy"))
    ctrl.reset(len(eng.robots))
    eng.run(ctrl.act)
    return buf.getvalue()


def test_traces_are_byte_identical_for_same_seed() -> None:
    assert trace_bytes(DEMO) == trace_bytes(DEMO)
    a = trace_bytes(RANDOM, controller="random")
    assert a == trace_bytes(RANDOM, controller="random")
    assert a != trace_bytes(RANDOM, seed=2, controller="random")


def test_trace_structure(tmp_path: Path) -> None:
    spec = load_scenario(DEMO)
    with TraceWriter(tmp_path / "t.jsonl") as tw:
        eng = run_scripted(spec, trace=tw)
    recs = read_trace(tmp_path / "t.jsonl")
    assert recs[0]["type"] == "header" and recs[-1]["type"] == "summary"
    steps = [r for r in recs if r["type"] == "step"]
    assert len(steps) == eng.t and steps[-1]["t"] == eng.t
    assert recs[-1]["metrics"]["tasks_delivered"] == 4


def test_demo_scenario_delivers_everything_safely() -> None:
    eng = run_scripted(load_scenario(DEMO))
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 4 and m.physical_collisions == 0
    assert all(t.status is TaskStatus.DELIVERED for t in eng.book.tasks.values())
    assert not eng.grid.is_blocked((6, 3))  # blocked at t=15, unblocked at t=24


# ---------------------------------------------------------- invariants under random play (X-2)


@settings(max_examples=40)
@given(st.integers(0, 10_000), st.sampled_from(["random", "warehouse", "rooms"]))
def test_invariants_hold_under_random_controller(seed: int, kind: str) -> None:
    gen: dict[str, Any] = {"kind": kind, "width": 10, "height": 8}
    if kind == "random":
        gen["density"] = 0.2
    spec = build_scenario(
        {
            "scenario": {"name": "p", "seed": seed, "max_steps": 40},
            "map": {"generator": gen},
            "placement": {"robots": 6, "obstacles": 3, "tasks": 2},
            "arrivals": {
                "kind": "poisson",
                "rate": 0.2,
                "priorities": [0.5, 0.3, 0.2],
                "until": 30,
            },
            "map_events": [{"time": 5, "cell": [0, 0], "action": "block"}],
        }
    )
    eng = Engine(spec)
    ctrl = RandomController(make_streams(seed).generator("policy"))
    while not eng.done:
        before = [r.position for r in eng.robots]
        out = eng.step(ctrl.act)
        after = [r.position for r in eng.robots]
        obs = [o.position for o in eng.obstacles]
        assert physical_violations(before, after, obs, eng.grid) == []
        assert not any(e.kind_code == "INV" for e in out.events)  # masked controller (G2)
        assert len(set(obs)) == len(obs)
    assert eng.summary().physical_collisions == 0


# ---------------------------------------------------------- coordinator API and counterfactual


class SuggestingCoordinator:
    """Test stub: assigns task 0 to robot 0 and suggests fixed actions (as a planner would)."""

    name = "stub"

    def __init__(self, suggestions: Sequence[Action]) -> None:
        self.s = list(suggestions)

    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        pass

    def update(self, view: WorldView, api: AssignmentAPI) -> None:
        pass

    def suggested_actions(self, view: WorldView) -> Sequence[Action] | None:
        return self.s


def test_counterfactual_avoided_and_introduced_sets() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "cf", "max_steps": 1},
            "map": {"rows": ["....."]},
            "robots": [{"start": [0, 0]}, {"start": [2, 0]}, {"start": [4, 0]}],
        }
    )
    # reference: r0 R and r1 L both into (1,0) → one RRv conflict
    eng = Engine(spec, SuggestingCoordinator([R, L, S]))
    out = eng.step(lambda v: {0: S, 1: S, 2: L})  # controller: only r2 moves, into free (3,0)
    assert out.reference_events is not None and len(out.reference_events) == 1
    m = eng.summary()
    assert (
        m.reference_conflicts,
        m.conflicts_avoided,
        m.conflicts_introduced,
        m.conflicts_persisted,
    ) == (1, 1, 0, 0)
    assert m.avoidance_rate == 1.0
    # controller that follows the reference exactly: avoided = introduced = 0 (X-10 identity)
    eng2 = Engine(spec, SuggestingCoordinator([R, L, S]))
    eng2.step(lambda v: {0: R, 1: L, 2: S})
    m2 = eng2.summary()
    assert (m2.conflicts_avoided, m2.conflicts_introduced, m2.conflicts_persisted) == (0, 0, 1)
    # controller introducing a conflict the reference did not have
    eng3 = Engine(spec, SuggestingCoordinator([S, S, S]))
    eng3.step(lambda v: {0: S, 1: R, 2: L})
    m3 = eng3.summary()
    assert (m3.reference_conflicts, m3.conflicts_introduced) == (0, 1)
    assert m3.avoidance_rate is None


def test_assignment_api_enforces_state_machine() -> None:
    eng = Engine(scenario())
    with pytest.raises(EngineStateError):
        eng.assign_task(0, 0)  # task not released yet (release=1)
    eng.step(lambda v: {0: S})
    eng.prepare_step()
    eng.assign_task(0, 0)
    assert eng.robots[0].mode is RobotMode.TO_PICKUP
    with pytest.raises(EngineStateError):
        eng.assign_task(0, 0)  # robot no longer idle
    eng.unassign_task(0)
    assert eng.book.tasks[0].status is TaskStatus.PENDING and eng.robots[0].mode is RobotMode.IDLE
    with pytest.raises(EngineStateError):
        eng.unassign_task(0)
    with pytest.raises(ValueError):
        eng.apply_actions({})  # missing actions
    eng.apply_actions({0: S})
    with pytest.raises(EngineStateError):
        eng.apply_actions({0: S})  # must prepare first


def test_unavailable_task_after_map_event() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "u", "max_steps": 5},
            "map": {"rows": ["....."]},
            "robots": [{"start": [0, 0]}],
            "tasks": [{"pickup": [1, 0], "dropoff": [4, 0]}],
            "map_events": [{"time": 0, "cell": [3, 0], "action": "block"}],
        }
    )
    eng = Engine(spec)
    view = eng.prepare_step()
    assert view.tasks[0].status is TaskStatus.PENDING and not view.tasks[0].available
    with pytest.raises(EngineStateError):
        eng.assign_task(0, 0)


def test_stepping_finished_episode_raises() -> None:
    eng = run_scripted(scenario())
    with pytest.raises(EngineStateError):
        eng.prepare_step()
    eng.reset()
    assert eng.t == 0 and not eng.done
