"""A* against independent oracles (``04`` §2.3 tests 1–5; acceptance criterion 1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robocoord.core.grid import OccupancyGrid
from robocoord.planning import AStarPlanner, ConstraintParams, PlanQuery, build_constraints
from robocoord.planning.base import lattice_cost, path_is_valid
from robocoord.planning.procedure import reachable
from tests.unit.planning.cases import bfs_steps, dijkstra_oracle, random_constraints, random_grid

N_UNIT = 600  # unit costs: BFS oracle
N_WEIGHTED = 600  # proximity costs: O(V²) Dijkstra oracle


def _instance(seed: int, weighted: bool):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    grid = random_grid(rng)
    while grid.num_free() == 0:
        grid = random_grid(rng)
    free = grid.free_cells()
    start = free[int(rng.integers(len(free)))]
    goal = free[int(rng.integers(len(free)))]
    if weighted:
        c = random_constraints(rng, grid, start, goal)
    else:
        c = build_constraints(grid, start, goal)
    return start, goal, c


def _bfs_args(seed: int):  # type: ignore[no-untyped-def]
    start, goal, c = _instance(seed, weighted=False)
    return c, start, goal


def _check(start, goal, c, expected: float) -> bool:  # type: ignore[no-untyped-def]
    r = AStarPlanner().plan(PlanQuery(start, goal, c))
    if math.isinf(expected):
        assert r.status == "unreachable" and r.path is None
        assert not reachable(c, start, goal)
        return False
    assert r.status == "ok" and r.path is not None
    assert r.cost == pytest.approx(expected, abs=1e-9)
    assert path_is_valid(r.path, start, goal, c)
    assert lattice_cost(r.path, c) == pytest.approx(r.cost, abs=1e-9)
    # expanded nodes ≤ reachable cells
    assert r.stats.expansions <= sum(c.free_flat)
    return True


def test_astar_equals_bfs_on_unit_cost_instances() -> None:
    solved = sum(
        _check(*_instance(s, weighted=False), bfs_steps(*_bfs_args(s))) for s in range(N_UNIT)
    )
    assert N_UNIT // 4 < solved < N_UNIT  # both outcomes are exercised


def test_astar_equals_dijkstra_on_weighted_instances() -> None:
    n_penalised = solved = 0
    for seed in range(10_000, 10_000 + N_WEIGHTED):
        start, goal, c = _instance(seed, weighted=True)
        n_penalised += bool(c.penalised_flat)
        solved += _check(start, goal, c, dijkstra_oracle(c, start, goal))
    assert n_penalised > N_WEIGHTED // 4  # the weighted branch is genuinely exercised
    assert N_WEIGHTED // 4 < solved < N_WEIGHTED


def test_astar_prefers_cheaper_detour_over_penalised_cells() -> None:
    grid = OccupancyGrid.empty(5, 3)
    # obstacle at (2, 2) sensed; Q covers (2, 1). With λ = 3 the straight path costs
    # 1 + 4 + 1 + 1 = 7, the 6-step detour through y = 0 costs 6.
    c = build_constraints(grid, (0, 1), (4, 1), {(2, 2): (0, 0)}, ConstraintParams(3, 1, 3.0))
    r = AStarPlanner().plan(PlanQuery((0, 1), (4, 1), c))
    assert r.path is not None and (2, 1) not in r.path
    assert r.cost == 6.0 and len(r.path) - 1 == 6
    # with λ = 0 the straight 4-step path is optimal
    c0 = build_constraints(grid, (0, 1), (4, 1), {(2, 2): (0, 0)}, ConstraintParams(3, 1, 0.0))
    assert AStarPlanner().plan(PlanQuery((0, 1), (4, 1), c0)).cost == 4.0


def test_astar_deterministic_and_edge_grids() -> None:
    c = build_constraints(OccupancyGrid.empty(7, 7), (0, 0), (6, 6))
    a = AStarPlanner().plan(PlanQuery((0, 0), (6, 6), c))
    b = AStarPlanner().plan(PlanQuery((0, 0), (6, 6), c))
    assert a.path == b.path and a.stats.expansions == b.stats.expansions
    # tie-breaking towards larger g expands only nodes on one optimal path in open space
    assert a.stats.expansions == 12
    one = build_constraints(OccupancyGrid.empty(1, 1), (0, 0), (0, 0))
    assert AStarPlanner().plan(PlanQuery((0, 0), (0, 0), one)).path == ((0, 0),)
    row = build_constraints(OccupancyGrid.empty(9, 1), (0, 0), (8, 0))
    assert AStarPlanner().plan(PlanQuery((0, 0), (8, 0), row)).cost == 8.0
    enclosed = OccupancyGrid.from_rows(["...", ".@.", "@.@"])
    ce = build_constraints(enclosed, (0, 2), (1, 0))
    assert AStarPlanner().plan(PlanQuery((0, 2), (1, 0), ce)).status == "unreachable"
    assert AStarPlanner().plan(PlanQuery((0, 2), (5, 5), ce)).status == "invalid_query"
