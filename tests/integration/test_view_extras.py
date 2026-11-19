"""Opt-in ``WorldView`` extras for learned controllers (Issue #5, Step 0).

The extras (reference paths, capped goal distances) are read-only: requesting them must not
change the simulation. Traces and metrics with and without them are byte-identical.
"""

from __future__ import annotations

import io
import itertools

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.control.baseline import BaselineController
from robocoord.control.reference import ReferenceFollower
from robocoord.core.types import action_between, manhattan
from robocoord.experiments.allocation_runs import e5_scenario
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.planning.endpoint import EndpointParams
from robocoord.scenario.spec import ScenarioSpec
from robocoord.sim.engine import VIEW_EXTRAS, Engine
from robocoord.sim.trace import TraceWriter

EXTRAS = frozenset({"paths", "goal_distance"})


def small(seed: int = 10000, dynamics: str = "full") -> ScenarioSpec:
    return e5_scenario(
        seed,
        dynamics,
        size=16,
        density=0.15,
        robots=4,
        tasks_per_robot=2,
        obstacles=4,
        max_steps=80,
        arrival_rate=0.05,
        arrivals_until=40,
    )


def planner(name: str) -> object:
    return make_planner(name, max_iterations=300) if name == "rrtstar" else make_planner(name)


def run(spec: ScenarioSpec, plan: str, ctrl_name: str, extras: frozenset[str]) -> tuple[str, dict]:
    coord = AllocationCoordinator(
        planner(plan),  # type: ignore[arg-type]
        AuctionAllocator(),
        deadlock=DetourParams() if ctrl_name == "baseline" else None,
        endpoint=EndpointParams(),
    )
    buf = io.StringIO()
    engine = Engine(spec, coord, trace=TraceWriter(buf), view_extras=extras, path_limit=6)
    ctrl = BaselineController() if ctrl_name == "baseline" else ReferenceFollower()
    ctrl.reset(len(engine.robots))
    while not engine.done:
        view = engine.prepare_step()
        if extras:
            check_view(engine, view)
        engine.apply_actions(ctrl.act(view))
    return buf.getvalue(), engine.summary().to_dict()


def check_view(engine: Engine, view: object) -> None:
    from robocoord.sim.view import WorldView

    assert isinstance(view, WorldView)
    assert view.reference_paths is not None and view.goal_distance is not None
    assert view.suggested is not None
    for r in view.robots:
        path = view.reference_paths[r.id]
        if path is not None:
            assert path[0] == r.position, "a reported path always starts at the robot"
            assert len(path) <= engine.path_limit + 1
            for a, b in itertools.pairwise(path):
                assert manhattan(a, b) == 1
            if len(path) >= 2:
                assert view.suggested[r.id] == action_between(path[0], path[1])
        d = view.goal_distance[r.id]
        if r.goal is None:
            assert d is None
        else:
            assert d == engine.oracle.capped_distance(r.position, r.goal)


def test_extras_absent_by_default() -> None:
    coord = AllocationCoordinator(make_planner("astar"), AuctionAllocator())
    engine = Engine(small(), coord)
    view = engine.prepare_step()
    assert view.reference_paths is None and view.goal_distance is None


def test_unknown_extra_and_bad_limit_rejected() -> None:
    with pytest.raises(ValueError, match="unknown view extras"):
        Engine(small(), None, view_extras=frozenset({"bogus"}))
    with pytest.raises(ValueError, match="path_limit"):
        Engine(small(), None, path_limit=0)
    assert frozenset({"paths", "goal_distance"}) == VIEW_EXTRAS


def test_paths_none_without_a_path_reporting_coordinator() -> None:
    engine = Engine(small(), None, view_extras=EXTRAS)
    view = engine.prepare_step()
    assert view.reference_paths is None  # NullCoordinator plans nothing
    assert view.goal_distance == tuple(None for _ in view.robots)


@pytest.mark.parametrize("ctrl_name", ["reference", "baseline"])
def test_extras_do_not_change_traces_astar(ctrl_name: str) -> None:
    spec = small()
    off = run(spec, "astar", ctrl_name, frozenset())
    on = run(spec, "astar", ctrl_name, EXTRAS)
    assert on[0] == off[0]
    assert on[1] == off[1]


@pytest.mark.slow
def test_extras_do_not_change_traces_rrtstar() -> None:
    spec = small(10001)
    off = run(spec, "rrtstar", "reference", frozenset())
    on = run(spec, "rrtstar", "reference", EXTRAS)
    assert on[0] == off[0]
    assert on[1] == off[1]
