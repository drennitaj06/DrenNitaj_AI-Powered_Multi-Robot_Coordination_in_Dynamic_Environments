"""BFS distance fields and a version-aware distance cache (model §1; algorithms §2.1)."""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np
import numpy.typing as npt

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import MOVES, Cell

UNREACHABLE = -1


def bfs_distance_field(grid: OccupancyGrid, target: Cell) -> npt.NDArray[np.int32]:
    """Shortest 4-connected step distance from every cell to ``target``.

    Returns an ``int32`` array indexed ``[y, x]``; :data:`UNREACHABLE` (−1) marks cells that
    cannot reach ``target`` (including blocked cells). If ``target`` itself is blocked, every
    entry is −1.

    Time: O(V) worst case. Space: O(V).
    """
    w, h = grid.width, grid.height
    dist = np.full((h, w), UNREACHABLE, dtype=np.int32)
    if not grid.is_free(target):
        return dist
    blocked = grid.blocked
    dist[target[1], target[0]] = 0
    queue: deque[Cell] = deque([target])
    deltas = [a.delta for a in MOVES]
    while queue:
        x, y = queue.popleft()
        d = int(dist[y, x]) + 1
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx] and dist[ny, nx] == UNREACHABLE:
                dist[ny, nx] = d
                queue.append((nx, ny))
    return dist


class DistanceOracle:
    """LRU cache of BFS distance fields keyed by ``(target, grid.version)``.

    The oracle holds a reference to a (mutable) grid; any map event bumps the grid version, so
    stale fields are never returned.
    """

    def __init__(self, grid: OccupancyGrid, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._grid = grid
        self._capacity = capacity
        self._cache: OrderedDict[tuple[Cell, int], npt.NDArray[np.int32]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def grid(self) -> OccupancyGrid:
        return self._grid

    def field(self, target: Cell) -> npt.NDArray[np.int32]:
        """Distance field to ``target`` for the current grid version (cached)."""
        key = (target, self._grid.version)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        fld = bfs_distance_field(self._grid, target)
        fld.flags.writeable = False
        self._cache[key] = fld
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return fld

    def distance(self, a: Cell, b: Cell) -> int | None:
        """``D_t(a, b)`` or ``None`` if unreachable (or either cell is blocked/out of bounds)."""
        if not (self._grid.in_bounds(a) and self._grid.in_bounds(b)):
            return None
        d = int(self.field(b)[a[1], a[0]])
        return None if d == UNREACHABLE else d

    def capped_distance(self, a: Cell, b: Cell) -> int:
        """``D̄(a, b) = min(D(a, b), W·H)`` (model §1)."""
        d = self.distance(a, b)
        cap = self._grid.width * self._grid.height
        return cap if d is None else min(d, cap)

    def reachable(self, a: Cell, b: Cell) -> bool:
        return self.distance(a, b) is not None


def connected_components(grid: OccupancyGrid) -> npt.NDArray[np.int32]:
    """Label free cells by 4-connected component (0, 1, …); blocked cells are −1.

    Time: O(V).
    """
    w, h = grid.width, grid.height
    labels = np.full((h, w), -1, dtype=np.int32)
    blocked = grid.blocked
    deltas = [a.delta for a in MOVES]
    current = 0
    for y in range(h):
        for x in range(w):
            if blocked[y, x] or labels[y, x] != -1:
                continue
            labels[y, x] = current
            queue: deque[Cell] = deque([(x, y)])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in deltas:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx] and labels[ny, nx] == -1:
                        labels[ny, nx] = current
                        queue.append((nx, ny))
            current += 1
    return labels
