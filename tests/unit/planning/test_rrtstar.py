"""RRT*: validity, monotone best cost, tree-cost invariant, determinism, counters, convergence
(``04`` §3.9 tests 5–8; acceptance criterion 2)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.grid import OccupancyGrid
from robocoord.planning import PlanQuery, RRTStarConfig, RRTStarPlanner, build_constraints
from robocoord.planning.base import lattice_cost, path_is_valid
from robocoord.planning.geometry import discretise, polyline_length, remove_loops
from robocoord.planning.procedure import plan_with_procedure, reachable
from robocoord.planning.rrtstar import TRACE_FIELDS, gamma_star
from tests.unit.planning.cases import random_constraints


def _solvable_query(seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    while True:
        w, h = int(rng.integers(3, 11)), int(rng.integers(3, 11))
        grid = OccupancyGrid(rng.random((h, w)) < rng.uniform(0, 0.35))
        free = grid.free_cells()
        if len(free) < 2:
            continue
        s = free[int(rng.integers(len(free)))]
        g = free[int(rng.integers(len(free)))]
        c = random_constraints(rng, grid, s, g)
        if s != g and reachable(c, s, g):
            return s, g, c, rng


@settings(max_examples=60)
@given(st.integers(0, 100_000), st.sampled_from(["kdtree", "brute"]))
def test_paths_valid_monotone_and_invariant(seed: int, index: str) -> None:
    s, g, c, rng = _solvable_query(seed)
    cfg = RRTStarConfig(
        max_iterations=150,
        eta=float(rng.choice([0.7, 1.5, 3.0])),
        neighbor_index=index,
        record_trace=True,
        check_invariants=True,  # raises TreeInvariantError after any bad iteration
    )
    r = RRTStarPlanner(cfg).plan(PlanQuery(s, g, c, rng))
    assert r.status in ("ok", "budget_exhausted")
    trace = r.stats.best_cost_trace
    assert trace is not None and len(trace) == r.stats.iterations == 150
    assert all(b <= a for a, b in itertools.pairwise(trace))  # monotone
    if r.ok:
        assert r.path is not None and path_is_valid(r.path, s, g, c)
        assert r.waypoints is not None and r.cost == trace[-1]
        raw = discretise(r.waypoints)
        assert tuple(remove_loops(raw)) == r.path
        assert len(set(r.path)) == len(r.path)  # no cell visited twice
        assert lattice_cost(r.path, c) <= lattice_cost(raw, c)
        if c.lambda_prox == 0 or not c.proximity:
            assert r.cost == pytest.approx(polyline_length(r.waypoints), rel=1e-9)
        assert 0 < r.stats.first_solution_iteration <= 150
        assert math.isinf(trace[r.stats.first_solution_iteration - 2]) or (
            r.stats.first_solution_iteration == 1
        )
    else:
        assert all(math.isinf(b) for b in trace)


def test_determinism_and_index_independence() -> None:
    s, g, c, _ = _solvable_query(7)

    def run(seed: int, index: str):  # type: ignore[no-untyped-def]
        cfg = RRTStarConfig(max_iterations=400, neighbor_index=index)
        return RRTStarPlanner(cfg).plan(PlanQuery(s, g, c, np.random.default_rng(seed)))

    a, b = run(1, "kdtree"), run(1, "kdtree")
    assert (a.path, a.cost, a.waypoints) == (b.path, b.cost, b.waypoints)
    # the two indexes return identical query results, so the whole run is identical
    br = run(1, "brute")
    assert (br.path, br.cost, br.waypoints, br.stats.near_k, br.stats.near_kprime) == (
        a.path,
        a.cost,
        a.waypoints,
        a.stats.near_k,
        a.stats.near_kprime,
    )
    assert a.stats.nn_visited < br.stats.nn_visited  # pruning works
    assert run(2, "kdtree").waypoints != a.waypoints  # a different seed explores differently


def test_work_counters_are_consistent() -> None:
    c = build_constraints(OccupancyGrid.empty(12, 12), (0, 0), (11, 11))
    cfg = RRTStarConfig(max_iterations=600, record_trace=True, neighbor_index="kdtree")
    r = RRTStarPlanner(cfg).plan(PlanQuery((0, 0), (11, 11), c, np.random.default_rng(3)))
    tr = r.stats.iteration_trace
    assert tr is not None and set(tr) == set(TRACE_FIELDS)
    assert np.all(tr["k"] <= tr["n"])  # k ≤ n
    assert np.all(tr["k"] <= tr["kprime"])  # disk ⊆ bounding square
    assert int(tr["k"].sum()) == r.stats.near_k
    assert int(tr["kprime"].sum()) == r.stats.near_kprime
    assert int(tr["s"].sum()) == r.stats.propagation >= r.stats.rewires  # s ≥ rewires
    assert int(tr["added"].sum()) == r.stats.tree_size - 1
    assert int(tr["nn_visited"].sum()) == r.stats.nn_visited
    assert int(tr["nn_nodes"].sum()) == r.stats.nn_nodes >= r.stats.nn_queries
    assert int(tr["range_nodes"].sum()) == r.stats.range_nodes
    assert int(tr["range_tested"].sum()) == r.stats.range_nodes_tested >= r.stats.range_nodes
    assert int(tr["rebuild"].sum()) == r.stats.rebuild_points - 1  # root inserted before loop
    assert int(tr["rebuild_work"].sum()) == r.stats.rebuild_work - 1
    assert r.stats.rebuild_work >= r.stats.rebuild_points
    assert r.stats.rewires > 0 and r.stats.radius_queries == r.stats.tree_size - 1


def test_convergence_sanity_on_empty_map() -> None:
    """Euclidean cost approaches the straight-line optimum as n grows. This is a finite-budget
    statistical sanity check over 6 seeds (tolerance: mean ratio ≤ 1.03), not a test of asymptotic
    optimality, which is not claimed for the implemented radius (G7)."""
    c = build_constraints(OccupancyGrid.empty(16, 16), (1, 1), (14, 14))
    optimum = math.hypot(12.5, 12.5)  # centre of (1,1) to the nearest point of square(14,14)

    def mean_ratio(n: int) -> float:
        cfg = RRTStarConfig(max_iterations=n)
        costs = [
            RRTStarPlanner(cfg).plan(PlanQuery((1, 1), (14, 14), c, np.random.default_rng(s))).cost
            for s in range(6)
        ]
        assert all(x >= optimum - 1e-9 for x in costs)  # never below the true optimum
        return float(np.mean(costs)) / optimum

    small, large = mean_ratio(150), mean_ratio(900)
    assert large < small and large <= 1.03


def test_edge_cases() -> None:
    corridor = OccupancyGrid.empty(12, 1)
    c = build_constraints(corridor, (0, 0), (11, 0))
    r = plan_with_procedure(
        RRTStarPlanner(RRTStarConfig(max_iterations=400, eta=1.0)),
        PlanQuery((0, 0), (11, 0), c, np.random.default_rng(0)),
    )
    assert r.ok and r.path == tuple((x, 0) for x in range(12))
    zero = RRTStarPlanner(RRTStarConfig(max_iterations=0))
    assert zero.plan(PlanQuery((0, 0), (11, 0), c, np.random.default_rng(0))).status == (
        "budget_exhausted"
    )
    timed = RRTStarPlanner(RRTStarConfig(max_iterations=10**7, time_limit_s=0.0))
    res = timed.plan(PlanQuery((0, 0), (11, 0), c, np.random.default_rng(0)))
    assert res.stats.timed_out and res.stats.iterations == 0
    first = RRTStarPlanner(RRTStarConfig(max_iterations=10**6, stop_on_first_solution=True))
    rf = first.plan(PlanQuery((0, 0), (11, 0), c, np.random.default_rng(0)))
    assert rf.ok and rf.stats.iterations == rf.stats.first_solution_iteration
    # no rng supplied: a fixed default generator keeps the planner deterministic
    a = RRTStarPlanner(RRTStarConfig(max_iterations=50)).plan(PlanQuery((0, 0), (11, 0), c))
    b = RRTStarPlanner(RRTStarConfig(max_iterations=50)).plan(PlanQuery((0, 0), (11, 0), c))
    assert a.waypoints == b.waypoints
    assert RRTStarPlanner().plan(PlanQuery((0, 0), (99, 0), c)).status == "invalid_query"
    assert gamma_star(math.pi) == pytest.approx(2 * math.sqrt(1.5))


def test_regression_loop_cut_from_discretised_path() -> None:
    """Review finding (V-6): the discretised polyline of this query revisits a cell (an
    A→B→A detour). The executable path must have the loop cut, stay valid and get shorter."""
    from robocoord.experiments.planning_runs import make_map, make_pair

    grid = make_map("d20", 16, 30000, 0)
    s, g = make_pair(grid, (30000, 0, 16, 0, 6))
    c = build_constraints(grid, s, g)
    r = plan_with_procedure(
        RRTStarPlanner(RRTStarConfig(max_iterations=600)),
        PlanQuery(s, g, c, np.random.default_rng(6)),
    )
    assert r.ok and r.path is not None and r.waypoints is not None
    raw = discretise(r.waypoints)
    assert len(set(raw)) < len(raw)  # the raw discretisation has a loop
    assert (len(raw) - 1, len(r.path) - 1) == (18, 16)
    assert len(set(r.path)) == len(r.path)
    assert path_is_valid(r.path, s, g, c)
