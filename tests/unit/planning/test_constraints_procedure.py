"""Shared constraint builder, two-stage procedure and planner registry (model §5.2–5.3)."""

from __future__ import annotations

import numpy as np
import pytest

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.planning import (
    PLANNERS,
    ConstraintParams,
    PlanQuery,
    build_constraints,
    make_planner,
    plan_with_procedure,
)
from robocoord.planning.base import PlanConstraints, PlanStats, lattice_cost, path_is_valid
from robocoord.planning.constraints import proximity_set, sense
from robocoord.planning.procedure import reachable

# corridor map: the only connection between left and right halves is the cell (2, 1)
CHOKE = OccupancyGrid.from_rows(
    [
        "..@..",
        ".....",
        "..@..",
    ]
)

PLANNER_CASES = [("astar", {}), ("rrtstar", {"max_iterations": 3000, "eta": 1.0})]


def _plan(name: str, opts: dict[str, object], start, goal, constraints, seed: int = 0):  # type: ignore[no-untyped-def]
    return plan_with_procedure(
        make_planner(name, **opts), PlanQuery(start, goal, constraints, np.random.default_rng(seed))
    )


def test_empty_sensed_set_gives_unit_density_and_no_blocks() -> None:
    c = build_constraints(CHOKE, (0, 0), (4, 2), obstacles={}, params=ConstraintParams())
    assert c.dynamic_blocked == frozenset() and c.proximity == frozenset()
    assert all(c.density((x, y)) == 1.0 for x in range(5) for y in range(3))
    # obstacles exist but none within the sensing radius → also empty
    far = build_constraints(
        OccupancyGrid.empty(20, 1), (0, 0), (19, 0), {(10, 0): (1, 0)}, ConstraintParams(3, 1, 2.0)
    )
    assert far.dynamic_blocked == frozenset() and far.proximity == frozenset()


def test_sensing_prediction_and_proximity() -> None:
    grid = OccupancyGrid.empty(9, 9)
    obstacles = {(5, 4): (1, 0), (4, 7): (0, 1), (8, 8): (0, 0)}  # last one is out of range
    sensed = sense(grid, (4, 4), obstacles, sensing_radius=3)
    assert sensed.current == frozenset({(5, 4), (4, 7)})
    assert sensed.predicted == frozenset({(6, 4), (4, 8)})
    # prediction into a wall or out of bounds is dropped
    walled = OccupancyGrid.from_rows(["...", "..@", "..."])
    assert sense(walled, (0, 1), {(1, 1): (1, 0)}, 3).predicted == frozenset()
    assert sense(walled, (0, 1), {(2, 2): (1, 0)}, 3).predicted == frozenset()
    q = proximity_set(grid, {(0, 0)}, 1)
    assert q == {(0, 0), (1, 0), (0, 1)}
    # the window scan and the list scan agree (many obstacles → window scan)
    many = {(x, y): (0, 0) for x in range(9) for y in range(9) if (x + y) % 2 == 0}
    s1 = sense(grid, (4, 4), many, 1)
    assert s1.current == frozenset(c for c in many if max(abs(c[0] - 4), abs(c[1] - 4)) <= 1)


def test_start_and_goal_are_never_dynamically_blocked() -> None:
    grid = OccupancyGrid.empty(6, 1)
    obstacles = {(1, 0): (1, 0), (5, 0): (0, 0), (4, 0): (-1, 0)}  # predicted (2,0); on goal (5,0)
    c = build_constraints(grid, (2, 0), (5, 0), obstacles, ConstraintParams(5, 1, 2.0))
    assert (2, 0) not in c.dynamic_blocked and (5, 0) not in c.dynamic_blocked
    assert (1, 0) in c.dynamic_blocked and (4, 0) in c.dynamic_blocked
    assert (5, 0) in c.proximity  # the goal may be penalised, never blocked
    extra = build_constraints(grid, (2, 0), (5, 0), {}, extra_blocked=[(3, 0), (5, 0), (2, 0)])
    assert extra.dynamic_blocked == frozenset({(3, 0)})


@pytest.mark.parametrize(("name", "opts"), PLANNER_CASES)
def test_obstacle_on_goal_leaves_goal_plannable(name: str, opts: dict[str, object]) -> None:
    grid = OccupancyGrid.empty(6, 3)
    c = build_constraints(grid, (0, 1), (5, 1), {(5, 1): (0, 0)}, ConstraintParams(10, 1, 2.0))
    r = _plan(name, opts, (0, 1), (5, 1), c)
    assert r.status == "ok" and not r.degraded
    assert r.path is not None and path_is_valid(r.path, (0, 1), (5, 1), c)


@pytest.mark.parametrize(("name", "opts"), PLANNER_CASES)
def test_dynamic_chokepoint_triggers_degraded_fallback(name: str, opts: dict[str, object]) -> None:
    c = build_constraints(CHOKE, (0, 1), (4, 1), {(2, 1): (0, 0)}, ConstraintParams(3, 1, 2.0))
    assert (2, 1) in c.dynamic_blocked
    assert not reachable(c, (0, 1), (4, 1))
    r = _plan(name, opts, (0, 1), (4, 1), c)
    assert r.status == "ok" and r.degraded
    assert r.path is not None and (2, 1) in r.path
    assert path_is_valid(r.path, (0, 1), (4, 1), c.without_dynamic())
    if name == "astar":  # A* proves stage 1 infeasible itself, then runs stage 2
        assert r.stats.stages == 2
    else:  # RRT*: the BFS guard skips stage 1 without spending iterations
        assert r.stats.stages == 1 and r.stats.guard_bfs == 2


@pytest.mark.parametrize(("name", "opts"), PLANNER_CASES)
def test_static_unreachability_and_invalid_queries(name: str, opts: dict[str, object]) -> None:
    walled = OccupancyGrid.from_rows(["..@..", "..@..", "..@.."])
    c = build_constraints(walled, (0, 0), (4, 0), {(3, 1): (0, 0)})
    r = _plan(name, opts, (0, 0), (4, 0), c)
    assert r.status == "unreachable" and not r.degraded and r.path is None
    assert r.stats.iterations == 0  # RRT* never spends budget on an unreachable goal
    # goal inside S(t) (after a map event) → unreachable
    assert _plan(name, opts, (0, 0), (2, 1), c).status == "unreachable"
    # start or goal out of bounds, start in S(t) → invalid_query
    assert _plan(name, opts, (0, 0), (9, 9), c).status == "invalid_query"
    assert _plan(name, opts, (-1, 0), (1, 1), c).status == "invalid_query"
    assert _plan(name, opts, (2, 0), (1, 1), c).status == "invalid_query"


@pytest.mark.parametrize(("name", "opts"), PLANNER_CASES)
def test_start_equals_goal(name: str, opts: dict[str, object]) -> None:
    c = build_constraints(CHOKE, (1, 1), (1, 1))
    r = _plan(name, opts, (1, 1), (1, 1), c)
    assert r.status == "ok" and r.path == ((1, 1),) and r.cost == 0.0


def test_rrtstar_budget_exhausted_only_when_reachable() -> None:
    c = build_constraints(OccupancyGrid.empty(8, 8), (0, 0), (7, 7))
    r = _plan("rrtstar", {"max_iterations": 0}, (0, 0), (7, 7), c)
    assert r.status == "budget_exhausted" and r.stats.guard_bfs == 1


def test_registry_and_config_errors() -> None:
    assert set(PLANNERS) == {"astar", "rrtstar"}
    assert make_planner("astar").name == "astar"
    rrt = make_planner("rrtstar", max_iterations=10, neighbor_index="brute")
    assert rrt.name == "rrtstar"
    with pytest.raises(ConfigError, match="astar") as info:
        make_planner("dijkstra")
    assert info.value.field == "planner"
    with pytest.raises(ConfigError):
        make_planner("astar", foo=1)
    with pytest.raises(ConfigError):
        make_planner("rrtstar", bogus=1)
    with pytest.raises(ConfigError):
        make_planner("rrtstar", neighbor_index="octree")
    with pytest.raises(ConfigError):
        make_planner("rrtstar", gamma_factor=0.5)
    with pytest.raises(ConfigError):
        ConstraintParams(sensing_radius=-1)


def test_helpers() -> None:
    c = PlanConstraints(OccupancyGrid.empty(3, 1), frozenset({(1, 0)}), frozenset({(2, 0)}), 2.0)
    assert lattice_cost([(0, 0), (1, 0), (2, 0)], c) == 1.0 + 3.0
    assert not path_is_valid([(0, 0), (1, 0), (2, 0)], (0, 0), (2, 0), c)
    assert not path_is_valid([(0, 0), (2, 0)], (0, 0), (2, 0), c.without_dynamic())
    assert c.free_area() == 2
    merged = PlanStats(expansions=2, peak_open=5, timed_out=True).merged(
        PlanStats(expansions=3, peak_open=4)
    )
    assert (merged.expansions, merged.peak_open, merged.timed_out, merged.stages) == (5, 5, True, 2)
    assert "expansions" in merged.counters()
