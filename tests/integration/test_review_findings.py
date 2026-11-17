"""Tests added by the Issue #2 final verification: each one pins a finding of that review.

* corridor demo with fully hand-computed metrics (conflicts, cascade, deferred map event);
* scripted coordinator must respect task availability (it used to crash);
* blocked-goal metric (model §9.1), per-100-robot-step rates;
* G1 holds for controllers that ignore the action mask (full engine, not just the arbiter);
* coordinator interface exposes the oracle and random streams (architecture §5.4–5.5);
* every shipped example scenario loads, validates and runs;
* scenarios without tasks are pure-motion sandboxes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.control.base import RandomController, ScriptedController, StayController
from robocoord.core.rng import make_streams
from robocoord.core.types import Action
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.conflicts import physical_violations
from robocoord.sim.coordinator import AssignmentAPI, ScriptedCoordinator
from robocoord.sim.engine import Engine
from robocoord.sim.view import WorldView
from robocoord.world.entities import TaskStatus

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "configs" / "scenarios"


def scripted_engine(path: Path) -> tuple[Engine, ScriptedController]:
    spec = load_scenario(path)
    assert spec.script is not None
    eng = Engine(spec, ScriptedCoordinator(spec.script.assignments))
    ctrl = ScriptedController(spec.script.actions)
    ctrl.reset(len(eng.robots))
    return eng, ctrl


# ------------------------------------------------------------------ corridor demo, by hand


def test_corridor_queue_hand_computed() -> None:
    """Derivation: r0 waits 4 steps (t=0..3). In each of them r1 moves into r0's cell (vertex
    conflict, r1 at fault, both frozen) and r2 wants r1's cell (cascade → blocked move).
    At t=4 all three advance in lockstep (following moves are legal) and pick up at t=5.
    Each then walks 5 more cells; all deliver at t=10."""
    eng, ctrl = scripted_engine(SCENARIO_DIR / "corridor_queue.toml")
    wall = (4, 1)
    applied_at: list[int] = []
    while not eng.done:
        out = eng.step(ctrl.act)
        if out.t <= 4:  # steps 0..3
            assert [(e.kind_code, e.entities, e.at_fault) for e in out.events] == [
                ("RRv", ("r0", "r1"), (1,))
            ]
            assert out.blocked == frozenset({2})
        else:
            assert out.events == frozenset() and out.blocked == frozenset()
        if out.map_events:
            applied_at.append(out.t)
        # a wall must never appear on top of a robot
        assert not (eng.grid.is_blocked(wall) and wall in {r.position for r in eng.robots})
    m = eng.summary()
    assert eng.terminated and m.steps == 10 and m.makespan == 10
    assert [eng.book.tasks[j].pickup_time for j in range(3)] == [5, 5, 5]
    assert [eng.book.tasks[j].completion_time for j in range(3)] == [10, 10, 10]
    assert [r.position for r in eng.robots] == [(10, 1), (9, 1), (8, 1)]
    assert m.conflicts == {"RRv": 4, "RRs": 0, "ROv": 0, "ROs": 0, "INV": 0}
    assert m.blocked_moves == 4 and m.physical_collisions == 0
    assert m.conflicts_per_100_robot_steps == pytest.approx(100 * 4 / 30)
    assert m.blocked_moves_per_100_robot_steps == pytest.approx(100 * 4 / 30)
    assert m.conflict_rates_per_100_robot_steps["RRv"] == pytest.approx(100 * 4 / 30)
    assert m.conflict_rates_per_100_robot_steps["RRs"] == 0.0
    assert m.wait_ratio == pytest.approx(4 / 30)  # only r0's four scripted STAYs
    assert m.utilisation == 1.0 and m.path_efficiency == 1.0 and m.legs_completed == 6
    assert m.mean_flow_time == 10.0 and m.weighted_flow_time == 10.0
    # map event scheduled at t=1 is deferred until the convoy has left cell (4,1) at the start of
    # step 7 (r0 at (7,1), r1 at (6,1), r2 at (5,1)) → recorded in the outcome with t = 8
    assert applied_at == [8]
    assert eng.grid.is_blocked(wall)


# ------------------------------------------------------------------ coordinator vs availability


def test_scripted_coordinator_waits_for_unavailable_task() -> None:
    """The task becomes unreachable at t=0 (wall at (3,0)) and available again at t=4. The old
    coordinator tried to assign it immediately and crashed with EngineStateError."""
    spec = build_scenario(
        {
            "scenario": {"name": "avail", "max_steps": 10},
            "map": {"rows": ["......."]},
            "robots": [{"start": [0, 0]}],
            "tasks": [{"pickup": [1, 0], "dropoff": [5, 0]}],
            "map_events": [
                {"time": 0, "cell": [3, 0], "action": "block"},
                {"time": 4, "cell": [3, 0], "action": "unblock"},
            ],
            "script": {"actions": [""], "assignments": [[0]]},
        }
    )
    eng = Engine(spec, ScriptedCoordinator([[0]]))
    stay = StayController()
    statuses = []
    for _ in range(6):
        eng.step(stay.act)
        statuses.append(eng.book.tasks[0].status)
    # steps 0..3: unavailable → stays PENDING; assigned during step 4
    assert statuses[:4] == [TaskStatus.PENDING] * 4
    assert statuses[4] is TaskStatus.ASSIGNED and statuses[5] is TaskStatus.ASSIGNED


# ------------------------------------------------------------------ blocked-goal metric


def test_blocked_goal_robot_steps_hand_computed() -> None:
    """Robot picks up at t=1 (goal (5,0)). A wall appears at (3,0) at the start of step 2, cutting
    the goal off from the robot at (2,0). Steps 2..5 each hold a task whose goal is unreachable."""
    spec = build_scenario(
        {
            "scenario": {"name": "bg", "max_steps": 6},
            "map": {"rows": ["......"]},
            "robots": [{"start": [0, 0]}],
            "tasks": [{"pickup": [1, 0], "dropoff": [5, 0]}],
            "map_events": [{"time": 2, "cell": [3, 0], "action": "block"}],
            "script": {"actions": ["RRSSSS"], "assignments": [[0]]},
        }
    )
    eng = Engine(spec, ScriptedCoordinator([[0]]))
    ctrl = ScriptedController(["RRSSSS"])
    ctrl.reset(1)
    m = eng.run(ctrl.act)
    assert eng.truncated and m.steps == 6
    assert m.blocked_goal_robot_steps == 4
    assert m.tasks_delivered == 0 and m.waiting_censored == 0  # the task was picked up at t=1


def test_blocked_goal_is_zero_without_map_events() -> None:
    eng, ctrl = scripted_engine(SCENARIO_DIR / "demo_delivery.toml")
    assert eng.run(ctrl.act).blocked_goal_robot_steps == 0


# ------------------------------------------------------------------ G1 for unmasked controllers


@settings(max_examples=25)
@given(st.integers(0, 10_000))
def test_engine_invariants_hold_for_controllers_that_ignore_the_mask(seed: int) -> None:
    """Uniformly random actions *including moves into walls*: every wall move must surface as an
    INV event, be cancelled, and never produce a physical collision."""
    spec = build_scenario(
        {
            "scenario": {"name": "unmasked", "seed": seed, "max_steps": 40},
            "map": {"generator": {"kind": "random", "width": 8, "height": 6, "density": 0.25}},
            "placement": {"robots": 6, "obstacles": 3, "tasks": 2},
            "arrivals": {
                "kind": "poisson",
                "rate": 0.15,
                "priorities": [0.5, 0.3, 0.2],
                "until": 30,
            },
            "map_events": [{"time": 3, "cell": [0, 0], "action": "block"}],
        }
    )
    eng = Engine(spec)
    rng = np.random.default_rng(seed)
    while not eng.done:
        before = [r.position for r in eng.robots]
        out = eng.step(lambda v: {i: Action(int(rng.integers(0, 5))) for i in range(len(v.robots))})
        after = [r.position for r in eng.robots]
        # the grid cannot change between detection and here (map events precede detection)
        wall_moves = sum(
            not eng.grid.is_free((before[i][0] + a.delta[0], before[i][1] + a.delta[1]))
            for i, a in enumerate(out.actions)
        )
        assert sum(e.kind_code == "INV" for e in out.events) == wall_moves
        assert (
            physical_violations(before, after, [o.position for o in eng.obstacles], eng.grid) == []
        )
    assert eng.summary().physical_collisions == 0


# ------------------------------------------------------------------ interface readiness


class _ProbeCoordinator:
    """Records what the engine hands a coordinator; stands in for the future auction/planners."""

    name = "probe"

    def __init__(self) -> None:
        self.api: AssignmentAPI | None = None
        self.draws: list[float] = []

    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        self.api = api
        self.draws.append(float(api.streams.generator("planner").random()))

    def update(self, view: WorldView, api: AssignmentAPI) -> None:
        assert api is self.api

    def suggested_actions(self, view: WorldView) -> None:
        return None


def test_coordinators_receive_oracle_and_named_streams() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "probe", "seed": 5, "max_steps": 3},
            "map": {"rows": ["...."]},
            "robots": [{"start": [0, 0]}],
        }
    )
    probe = _ProbeCoordinator()
    eng = Engine(spec, probe)
    assert probe.api is eng and probe.api.oracle is eng.oracle
    assert probe.api.oracle.distance((0, 0), (3, 0)) == 3
    # the planner stream is exactly the documented one and restarts identically on reset()
    expected = float(make_streams(5).generator("planner").random())
    assert probe.draws == [expected]
    eng.reset()
    assert probe.draws == [expected, expected]
    # a different run seed gives a different planner stream
    eng2 = Engine(spec, _ProbeCoordinator(), seed=6)
    assert eng2.streams.generator("planner").random() != expected


# ------------------------------------------------------------------ shipped example scenarios


EXAMPLES = sorted(SCENARIO_DIR.glob("*.toml"))


def test_at_least_three_example_scenarios_ship() -> None:
    assert len(EXAMPLES) >= 3, [p.name for p in EXAMPLES]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_every_example_scenario_loads_and_runs(path: Path) -> None:
    spec = load_scenario(path)
    spec.validate()
    if spec.script is not None:
        eng, ctrl = scripted_engine(path)
        m = eng.run(ctrl.act)
        # scripted demos are designed to finish: every task delivered, nothing unsafe
        assert eng.terminated and m.tasks_delivered == m.tasks_released > 0
    else:
        eng = Engine(spec)
        ctrl2 = RandomController(make_streams(spec.seed).generator("policy"))
        ctrl2.reset(len(eng.robots))
        m = eng.run(ctrl2.act)
        assert m.steps == spec.max_steps  # no allocator yet: runs to the step limit
        assert m.conflicts["INV"] == 0  # masked controller
    assert m.physical_collisions == 0


# ------------------------------------------------------------------ sandbox scenarios


def test_scenario_without_tasks_runs_to_max_steps() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "sandbox", "max_steps": 7},
            "map": {"rows": ["...."]},
            "robots": [{"start": [0, 0]}],
            "obstacles": [{"start": [3, 0], "model": "bounce", "direction": "left"}],
        }
    )
    eng = Engine(spec)
    m = eng.run(StayController().act)
    assert eng.truncated and not eng.terminated and m.steps == 7
    assert m.tasks_released == 0 and m.completion_rate is None
