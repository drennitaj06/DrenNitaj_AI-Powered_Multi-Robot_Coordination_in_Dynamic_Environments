"""Random planning instances and independent oracles shared by the planning tests."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.planning.base import PlanConstraints
from robocoord.planning.constraints import ConstraintParams, build_constraints


def random_grid(
    rng: np.random.Generator, max_size: int = 12, max_density: float = 0.45
) -> OccupancyGrid:
    w = int(rng.integers(1, max_size + 1))
    h = int(rng.integers(1, max_size + 1))
    density = float(rng.uniform(0.0, max_density))
    return OccupancyGrid(rng.random((h, w)) < density)


def random_constraints(
    rng: np.random.Generator, grid: OccupancyGrid, start: Cell, goal: Cell
) -> PlanConstraints:
    """Random sensed obstacles (some on/near the goal), built by the shared builder."""
    free = grid.free_cells()
    k = int(rng.integers(0, min(len(free), 6) + 1))
    obstacles: dict[Cell, Cell] = {}
    picks = [free[int(i)] for i in rng.choice(len(free), size=k, replace=False)] if k else []
    if rng.random() < 0.3:
        picks.append(goal)  # an obstacle on the goal cell
    for c in picks:
        v = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(0, 5))]
        obstacles[c] = v
    params = ConstraintParams(
        sensing_radius=int(rng.integers(0, 6)),
        inflation_radius=int(rng.integers(0, 3)),
        lambda_prox=float(rng.choice([0.0, 0.5, 2.0])),
    )
    return build_constraints(grid, start, goal, obstacles, params)


def dijkstra_oracle(c: PlanConstraints, start: Cell, goal: Cell) -> float:
    """Independent O(V²) Dijkstra (array scan, no heap) on the constrained graph."""
    cells = [(x, y) for y in range(c.height) for x in range(c.width) if c.is_free((x, y))]
    if start not in cells or goal not in cells:
        return math.inf
    dist = {v: math.inf for v in cells}
    dist[start] = 0.0
    done: set[Cell] = set()
    while True:
        u = None
        best = math.inf
        for v in cells:
            if v not in done and dist[v] < best:
                u, best = v, dist[v]
        if u is None:
            return math.inf
        if u == goal:
            return best
        done.add(u)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dx, u[1] + dy)
            if v in dist and v not in done:
                nd = best + c.density(v)
                if nd < dist[v]:
                    dist[v] = nd


def bfs_steps(c: PlanConstraints, start: Cell, goal: Cell) -> float:
    """Independent BFS step distance on the constrained graph (inf if unreachable)."""
    if not (c.is_free(start) and c.is_free(goal)):
        return math.inf
    seen = {start: 0}
    queue: deque[Cell] = deque([start])
    while queue:
        u = queue.popleft()
        if u == goal:
            return float(seen[u])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dx, u[1] + dy)
            if v not in seen and c.is_free(v):
                seen[v] = seen[u] + 1
                queue.append(v)
    return math.inf
