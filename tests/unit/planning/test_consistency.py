"""X-11 planner policy consistency (acceptance criterion 3): on identical queries A* and RRT*
(generous budget) return the same ``status`` and the same ``degraded`` flag."""

from __future__ import annotations

from collections import Counter

import numpy as np

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.planning import (
    AStarPlanner,
    ConstraintParams,
    PlanQuery,
    RRTStarConfig,
    RRTStarPlanner,
    build_constraints,
    plan_with_procedure,
)
from robocoord.planning.procedure import reachable
from robocoord.scenario.generators import rooms
from tests.unit.planning.cases import random_constraints, random_grid

GENEROUS = RRTStarConfig(max_iterations=50_000, eta=1.5, stop_on_first_solution=True)


def _chokepoints(grid: OccupancyGrid, s: Cell, g: Cell) -> list[Cell]:
    """Free cells whose removal disconnects s from g (computed by brute force)."""
    out = []
    for c in grid.free_cells():
        if c in (s, g):
            continue
        cons = build_constraints(grid, s, g, extra_blocked=[c])
        if not reachable(cons, s, g):
            out.append(c)
    return out


def _queries() -> list[tuple[Cell, Cell, object]]:
    queries: list[tuple[Cell, Cell, object]] = []
    for seed in range(120):  # random maps, obstacles on and near goals
        rng = np.random.default_rng(50_000 + seed)
        grid = random_grid(rng, max_size=10, max_density=0.35)
        free = grid.free_cells()
        if len(free) < 2:
            continue
        s = free[int(rng.integers(len(free)))]
        g = free[int(rng.integers(len(free)))]
        if seed % 10 == 0:  # goal inside S(t), as after a map event
            blocked = [(x, y) for y in range(grid.height) for x in range(grid.width)]
            blocked = [c for c in blocked if grid.is_blocked(c)]
            if blocked:
                g = blocked[0]
        queries.append((s, g, random_constraints(rng, grid, s, g)))
    for seed in range(120):  # rooms maps: an obstacle sits on a door (chokepoint) near the start
        rng = np.random.default_rng(60_000 + seed)
        grid = rooms(int(rng.integers(5, 11)), int(rng.integers(5, 11)), 2, rng)
        free = grid.free_cells()
        s = free[int(rng.integers(len(free)))]
        g = free[int(rng.integers(len(free)))]
        chokes = _chokepoints(grid, s, g)
        obstacles: dict[Cell, Cell] = {}
        if chokes:
            obstacles[chokes[int(rng.integers(len(chokes)))]] = (0, 0)
        if rng.random() < 0.3:
            obstacles[g] = (0, 0)
        params = ConstraintParams(sensing_radius=int(rng.integers(2, 12)))
        queries.append((s, g, build_constraints(grid, s, g, obstacles, params)))
    for seed in range(80):  # empty sensed sets
        rng = np.random.default_rng(70_000 + seed)
        grid = random_grid(rng, max_size=10, max_density=0.3)
        free = grid.free_cells()
        if len(free) < 2:
            continue
        s = free[int(rng.integers(len(free)))]
        g = free[int(rng.integers(len(free)))]
        queries.append((s, g, build_constraints(grid, s, g)))
    return queries


def test_x11_astar_and_rrtstar_agree_on_status_and_degraded() -> None:
    queries = _queries()
    assert len(queries) >= 300
    outcomes: Counter[tuple[str, bool]] = Counter()
    for i, (s, g, c) in enumerate(queries):
        a = plan_with_procedure(AStarPlanner(), PlanQuery(s, g, c))  # type: ignore[arg-type]
        r = plan_with_procedure(
            RRTStarPlanner(GENEROUS),
            PlanQuery(s, g, c, np.random.default_rng(i)),  # type: ignore[arg-type]
        )
        assert (a.status, a.degraded) == (r.status, r.degraded), (i, s, g, a.status, r.status)
        outcomes[(a.status, a.degraded)] += 1
    # every policy branch is exercised
    assert outcomes[("ok", False)] > 0
    assert outcomes[("ok", True)] > 0
    assert outcomes[("unreachable", False)] > 0
