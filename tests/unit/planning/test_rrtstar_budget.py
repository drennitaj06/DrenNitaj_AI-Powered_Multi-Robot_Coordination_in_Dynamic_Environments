"""RRT* budget control (Issue 6A P2): ``refine_iterations`` (first solution + k) and ``patience``
(no improvement for m iterations).

The central property: a stopped run is an exact prefix of the full-budget run with the same random
stream, so it returns the best path of that prefix (valid), its cost equals the full run's best cost
at the stop iteration (never lower than the full run's final cost), and a query without a solution
runs the whole budget and fails exactly as before."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.planning import (
    PlanQuery,
    RRTStarConfig,
    RRTStarPlanner,
    build_constraints,
    make_planner,
)
from robocoord.planning.base import PlanResult, path_is_valid
from tests.unit.planning.test_rrtstar import _solvable_query


def _plan(cfg: RRTStarConfig, s, g, c, seed: int) -> PlanResult:  # type: ignore[no-untyped-def]
    return RRTStarPlanner(cfg).plan(PlanQuery(s, g, c, np.random.default_rng(seed)))


def expected_stop(trace: tuple[float, ...], k: int | None, m: int | None) -> int:
    """Stop iteration implied by the full run's best-cost trace (independent re-statement)."""
    n = len(trace)
    first = next((i + 1 for i, b in enumerate(trace) if not math.isinf(b)), None)
    if first is None:
        return n
    last_best, last_imp = math.inf, first
    for it in range(first, n + 1):
        b = trace[it - 1]
        if b < last_best:
            last_best, last_imp = b, it
        if k is not None and it - first >= k:
            return it
        if m is not None and it - last_imp >= m:
            return it
    return n


def test_defaults_are_off_and_validated() -> None:
    cfg = RRTStarConfig()
    assert cfg.refine_iterations is None and cfg.patience is None
    for bad in (-1, 1.5, True):
        with pytest.raises(ConfigError):
            RRTStarConfig(refine_iterations=bad)  # type: ignore[arg-type]
    for bad in (0, -3, 2.0, False):
        with pytest.raises(ConfigError):
            RRTStarConfig(patience=bad)  # type: ignore[arg-type]
    RRTStarConfig(refine_iterations=0, patience=1)  # valid edge values
    p = make_planner("rrtstar", refine_iterations=5, patience=7)
    assert isinstance(p, RRTStarPlanner)
    assert (p.config.refine_iterations, p.config.patience) == (5, 7)


def test_disabled_rules_reproduce_the_full_run_exactly() -> None:
    for seed in range(12):
        s, g, c, _ = _solvable_query(seed)
        base = _plan(RRTStarConfig(max_iterations=300, record_trace=True), s, g, c, seed)
        off = _plan(
            RRTStarConfig(
                max_iterations=300, record_trace=True, refine_iterations=None, patience=None
            ),
            s,
            g,
            c,
            seed,
        )
        assert (off.path, off.cost, off.status, off.waypoints) == (
            base.path,
            base.cost,
            base.status,
            base.waypoints,
        )
        ca, cb = off.stats.counters(), base.stats.counters()
        ca.pop("time_s")
        cb.pop("time_s")
        assert ca == cb
        assert off.stats.best_cost_trace == base.stats.best_cost_trace


@settings(max_examples=80, deadline=None)
@given(
    st.integers(0, 100_000),
    st.sampled_from([None, 0, 1, 5, 40]),
    st.sampled_from([None, 1, 3, 20, 60]),
    st.sampled_from(["brute", "kdtree"]),
)
def test_stopped_run_is_a_prefix_of_the_full_run(
    seed: int, k: int | None, m: int | None, index: str
) -> None:
    s, g, c, rng = _solvable_query(seed)
    eta = float(rng.choice([0.7, 1.5, 3.0]))
    n = 200
    full = _plan(
        RRTStarConfig(max_iterations=n, eta=eta, neighbor_index=index, record_trace=True),
        s,
        g,
        c,
        seed,
    )
    cfg = RRTStarConfig(
        max_iterations=n,
        eta=eta,
        neighbor_index=index,
        record_trace=True,
        check_invariants=True,
        refine_iterations=k,
        patience=m,
    )
    a = _plan(cfg, s, g, c, seed)
    b = _plan(cfg, s, g, c, seed)
    ft = full.stats.best_cost_trace
    at = a.stats.best_cost_trace
    assert ft is not None and at is not None
    stop = expected_stop(ft, k, m)
    # determinism
    assert (a.path, a.cost, a.status, a.stats.iterations) == (
        b.path,
        b.cost,
        b.status,
        b.stats.iterations,
    )
    # exact prefix: same iterations, same best-cost history, same first solution
    assert a.stats.iterations == stop == len(at)
    assert at == ft[:stop]
    assert a.stats.first_solution_iteration == full.stats.first_solution_iteration
    assert all(y <= x for x, y in itertools.pairwise(at))  # best cost non-increasing
    # success is unchanged; a failure runs the whole budget
    assert a.status == full.status
    if not full.ok:
        assert a.stats.iterations == n and a.path is None
        return
    assert a.path is not None and path_is_valid(a.path, s, g, c)
    assert a.cost == at[-1] >= full.cost  # never better than the full run, never invalid
    if stop == n:
        assert (a.path, a.cost, a.waypoints) == (full.path, full.cost, full.waypoints)
    if k == 0:
        first = _plan(
            RRTStarConfig(
                max_iterations=n, eta=eta, neighbor_index=index, stop_on_first_solution=True
            ),
            s,
            g,
            c,
            seed,
        )
        assert (first.path, first.cost) == (a.path, a.cost)


def _walled() -> tuple[OccupancyGrid, tuple[int, int], tuple[int, int]]:
    occ = np.zeros((7, 7), dtype=bool)
    occ[:, 3] = True  # wall: the goal side is unreachable
    return OccupancyGrid(occ), (0, 3), (6, 3)


def test_no_solution_runs_the_whole_budget_and_fails_as_before() -> None:
    grid, s, g = _walled()
    c = build_constraints(grid, s, g)
    full = _plan(RRTStarConfig(max_iterations=150), s, g, c, 3)
    for k, m in ((0, None), (10, None), (None, 1), (5, 5)):
        r = _plan(RRTStarConfig(max_iterations=150, refine_iterations=k, patience=m), s, g, c, 3)
        assert r.status == full.status == "budget_exhausted"
        assert r.path is None and r.stats.iterations == full.stats.iterations == 150
        assert r.stats.first_solution_iteration == -1


def test_late_first_solution_is_capped_by_the_budget() -> None:
    # a long corridor with no goal bias: the first solution comes late
    occ = np.ones((3, 30), dtype=bool)
    occ[1, :] = False
    grid = OccupancyGrid(occ)
    s, g = (0, 1), (29, 1)
    c = build_constraints(grid, s, g)
    base = RRTStarConfig(max_iterations=3000, goal_bias=0.0, record_trace=True)
    full = _plan(base, s, g, c, 11)
    assert full.ok
    i1 = full.stats.first_solution_iteration
    assert i1 > 50  # late
    near_end = 3000 - i1 + 10  # k larger than the remaining budget
    r = _plan(
        RRTStarConfig(
            max_iterations=3000, goal_bias=0.0, record_trace=True, refine_iterations=near_end
        ),
        s,
        g,
        c,
        11,
    )
    assert r.stats.iterations == 3000 and (r.path, r.cost) == (full.path, full.cost)
    r5 = _plan(RRTStarConfig(max_iterations=3000, goal_bias=0.0, refine_iterations=5), s, g, c, 11)
    assert r5.stats.iterations == i1 + 5 and r5.ok


def test_patience_restarts_after_each_improvement() -> None:
    # an open map improves repeatedly after the first solution; patience must count from the
    # latest strict improvement, not from the first solution
    grid = OccupancyGrid(np.zeros((12, 12), dtype=bool))
    s, g = (0, 0), (11, 11)
    c = build_constraints(grid, s, g)
    full = _plan(RRTStarConfig(max_iterations=1500, record_trace=True), s, g, c, 5)
    trace = full.stats.best_cost_trace
    assert trace is not None
    finite = [b for b in trace if not math.isinf(b)]
    assert len(set(finite)) >= 3  # repeated improvement after the first solution
    i1 = full.stats.first_solution_iteration
    m = 30
    r = _plan(RRTStarConfig(max_iterations=1500, record_trace=True, patience=m), s, g, c, 5)
    stop = r.stats.iterations
    assert stop == expected_stop(trace, None, m)
    rt = r.stats.best_cost_trace
    assert rt is not None
    # the stop iteration is m after the last improvement of the prefix
    last_imp = max(i + 1 for i in range(i1 - 1, stop) if i + 1 == i1 or rt[i] < rt[i - 1])
    assert stop - last_imp == m
    # and every earlier gap between improvements was shorter than m
    imps = [i + 1 for i in range(i1 - 1, stop) if i + 1 == i1 or rt[i] < rt[i - 1]]
    assert all(b - a < m for a, b in itertools.pairwise(imps))


def test_no_improvement_stops_exactly_after_the_window() -> None:
    # a straight line in a 1-wide corridor: the first path is already optimal
    occ = np.ones((3, 8), dtype=bool)
    occ[1, :] = False
    grid = OccupancyGrid(occ)
    s, g = (0, 1), (7, 1)
    c = build_constraints(grid, s, g)
    full = _plan(RRTStarConfig(max_iterations=500, record_trace=True), s, g, c, 2)
    assert full.ok
    i1 = full.stats.first_solution_iteration
    trace = full.stats.best_cost_trace
    assert trace is not None
    r = _plan(RRTStarConfig(max_iterations=500, patience=25, record_trace=True), s, g, c, 2)
    assert r.stats.iterations == expected_stop(trace, None, 25)
    assert r.stats.iterations >= i1 + 25
    assert r.ok and r.cost >= full.cost


def test_both_rules_stop_at_whichever_triggers_first() -> None:
    for seed in range(10):
        s, g, c, _ = _solvable_query(seed)
        full = _plan(RRTStarConfig(max_iterations=250, record_trace=True), s, g, c, seed)
        trace = full.stats.best_cost_trace
        assert trace is not None
        both = _plan(
            RRTStarConfig(max_iterations=250, refine_iterations=30, patience=10), s, g, c, seed
        )
        assert both.stats.iterations == min(
            expected_stop(trace, 30, None), expected_stop(trace, None, 10)
        )
