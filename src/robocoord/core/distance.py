"""BFS distance fields, connected components and a version-aware distance cache (model §1;
algorithms §2.1).

Implementation (NFR-PERF-1 optimisation O1/O2, Project Issue #4; `04` §2.1):

* **O1 — flat BFS.** Cells are indexed ``i = y·W + x``. The traversal runs on Python lists (no
  per-cell NumPy scalar indexing); the :class:`DistanceOracle` caches, per grid version, the list
  of free 4-neighbours of every free cell, so each field is one pass over that list. The result
  is converted to the same ``int32`` ``[y, x]`` array as before. BFS distances do not depend on
  the order in which neighbours are expanded, so the fields are identical to the earlier
  implementation (kept as :func:`bfs_distance_field_legacy`, the reference of the equivalence
  tests and of the before/after benchmark).
* **O2 — component reachability.** On the undirected 4-connected grid, ``D_t(a, b) < ∞`` iff
  ``a`` and ``b`` are free and in the same connected component. :meth:`DistanceOracle.reachable`
  answers from component labels computed once per grid version (one O(V) pass) instead of a
  distance field per target. Every map event changes the grid version, so the labels (and the
  neighbour lists and fields) of an older version are never used.
* **Symmetric reuse.** ``D_t(a, b) = D_t(b, a)`` on the undirected grid, so
  :meth:`DistanceOracle.distance` answers from a cached field to ``a`` when none to ``b`` is
  cached (and the auction benefits read the pickup→drop-off leg from the pickup's field).

``DistanceOracle(grid, legacy=True)`` (or :func:`legacy_distance` for every oracle created inside
the block) reproduces the earlier behaviour exactly: legacy BFS fields and ``reachable`` through a
distance field. It exists only for the paired before/after benchmark and the equivalence tests.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import MOVES, Cell

UNREACHABLE = -1


def _free_flat(grid: OccupancyGrid) -> list[bool]:
    """Free flags in index order ``i = y·W + x``."""
    free: list[bool] = (~grid.blocked).ravel().tolist()
    return free


def _bfs_index(free: list[bool], w: int, h: int, start: int) -> list[int]:
    """Flat BFS by index arithmetic (no neighbour lists); ``start`` must be free."""
    n = w * h
    dist = [UNREACHABLE] * n
    dist[start] = 0
    queue = [start]
    head = 0
    last_row = n - w
    while head < len(queue):
        i = queue[head]
        head += 1
        d = dist[i] + 1
        x = i % w
        if x > 0 and free[i - 1] and dist[i - 1] < 0:
            dist[i - 1] = d
            queue.append(i - 1)
        if x < w - 1 and free[i + 1] and dist[i + 1] < 0:
            dist[i + 1] = d
            queue.append(i + 1)
        if i >= w and free[i - w] and dist[i - w] < 0:
            dist[i - w] = d
            queue.append(i - w)
        if i < last_row and free[i + w] and dist[i + w] < 0:
            dist[i + w] = d
            queue.append(i + w)
    return dist


def _neighbour_lists(free: list[bool], w: int, h: int) -> list[list[int]]:
    """Free 4-neighbours of every free cell (empty list for blocked cells). Time O(V)."""
    n = w * h
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        if not free[i]:
            continue
        x = i % w
        lst = adj[i]
        if x > 0 and free[i - 1]:
            lst.append(i - 1)
        if x < w - 1 and free[i + 1]:
            lst.append(i + 1)
        if i >= w and free[i - w]:
            lst.append(i - w)
        if i + w < n and free[i + w]:
            lst.append(i + w)
    return adj


def _bfs_lists(adj: list[list[int]], n: int, start: int) -> list[int]:
    """Flat BFS over neighbour lists; ``start`` must be free."""
    dist = [UNREACHABLE] * n
    dist[start] = 0
    queue = [start]
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        d = dist[i] + 1
        for j in adj[i]:
            if dist[j] < 0:
                dist[j] = d
                queue.append(j)
    return dist


def _as_field(dist: list[int], w: int, h: int) -> npt.NDArray[np.int32]:
    return np.array(dist, dtype=np.int32).reshape(h, w)


def bfs_distance_field(grid: OccupancyGrid, target: Cell) -> npt.NDArray[np.int32]:
    """Shortest 4-connected step distance from every cell to ``target``.

    Returns an ``int32`` array indexed ``[y, x]``; :data:`UNREACHABLE` (−1) marks cells that
    cannot reach ``target`` (including blocked cells). If ``target`` itself is blocked, every
    entry is −1.

    Time: O(V) worst case. Space: O(V).
    """
    w, h = grid.width, grid.height
    if not grid.is_free(target):
        return np.full((h, w), UNREACHABLE, dtype=np.int32)
    return _as_field(_bfs_index(_free_flat(grid), w, h, target[1] * w + target[0]), w, h)


def bfs_distance_field_legacy(grid: OccupancyGrid, target: Cell) -> npt.NDArray[np.int32]:
    """The implementation before optimisation O1 (per-cell NumPy indexing); same result as
    :func:`bfs_distance_field`. Kept as the reference of the equivalence tests and the
    before/after benchmark."""
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


def _component_labels(free: list[bool], w: int, h: int) -> list[int]:
    """Flat component labels 0, 1, … in order of each component's first cell in row-major
    (``y`` then ``x``) scan order; −1 for blocked cells. Time O(V)."""
    n = w * h
    labels = [-1] * n
    current = 0
    last_row = n - w
    for s in range(n):
        if not free[s] or labels[s] >= 0:
            continue
        labels[s] = current
        queue = [s]
        head = 0
        while head < len(queue):
            i = queue[head]
            head += 1
            x = i % w
            if x > 0 and free[i - 1] and labels[i - 1] < 0:
                labels[i - 1] = current
                queue.append(i - 1)
            if x < w - 1 and free[i + 1] and labels[i + 1] < 0:
                labels[i + 1] = current
                queue.append(i + 1)
            if i >= w and free[i - w] and labels[i - w] < 0:
                labels[i - w] = current
                queue.append(i - w)
            if i < last_row and free[i + w] and labels[i + w] < 0:
                labels[i + w] = current
                queue.append(i + w)
        current += 1
    return labels


def connected_components(grid: OccupancyGrid) -> npt.NDArray[np.int32]:
    """Label free cells by 4-connected component (0, 1, … in row-major order of each
    component's first cell); blocked cells are −1.

    Time: O(V).
    """
    w, h = grid.width, grid.height
    return np.array(_component_labels(_free_flat(grid), w, h), dtype=np.int32).reshape(h, w)


class DistanceOracle:
    """LRU cache of BFS distance fields keyed by ``(target, grid.version)``, plus per-version
    neighbour lists and component labels.

    The oracle holds a reference to a (mutable) grid; any map event bumps the grid version, so
    stale fields, neighbour lists and labels are never used. ``misses`` counts distance fields
    computed; ``label_builds`` counts component labellings.
    """

    #: Default of ``legacy`` for oracles created without it (see :func:`legacy_distance`).
    default_legacy = False

    def __init__(
        self, grid: OccupancyGrid, capacity: int = 256, *, legacy: bool | None = None
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._grid = grid
        self._capacity = capacity
        self._legacy = DistanceOracle.default_legacy if legacy is None else legacy
        self._cache: OrderedDict[tuple[Cell, int], npt.NDArray[np.int32]] = OrderedDict()
        self._adj: tuple[int, list[list[int]]] | None = None
        self._labels: tuple[int, list[int]] | None = None
        self.hits = 0
        self.misses = 0
        self.label_builds = 0

    @property
    def grid(self) -> OccupancyGrid:
        return self._grid

    @property
    def legacy(self) -> bool:
        return self._legacy

    def _neighbours(self) -> list[list[int]]:
        v = self._grid.version
        if self._adj is None or self._adj[0] != v:
            g = self._grid
            self._adj = (v, _neighbour_lists(_free_flat(g), g.width, g.height))
        return self._adj[1]

    def component_labels(self) -> list[int]:
        """Flat component labels (index ``y·W + x``) of the current grid version (cached)."""
        v = self._grid.version
        if self._labels is None or self._labels[0] != v:
            g = self._grid
            self._labels = (v, _component_labels(_free_flat(g), g.width, g.height))
            self.label_builds += 1
        return self._labels[1]

    def field(self, target: Cell) -> npt.NDArray[np.int32]:
        """Distance field to ``target`` for the current grid version (cached)."""
        key = (target, self._grid.version)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        g = self._grid
        if self._legacy:
            fld = bfs_distance_field_legacy(g, target)
        elif not g.is_free(target):
            fld = np.full((g.height, g.width), UNREACHABLE, dtype=np.int32)
        else:
            w, h = g.width, g.height
            fld = _as_field(_bfs_lists(self._neighbours(), w * h, target[1] * w + target[0]), w, h)
        fld.flags.writeable = False
        self._cache[key] = fld
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return fld

    def distance(self, a: Cell, b: Cell) -> int | None:
        """``D_t(a, b)`` or ``None`` if unreachable (or either cell is blocked/out of bounds)."""
        if not (self._grid.in_bounds(a) and self._grid.in_bounds(b)):
            return None
        if not self._legacy:
            v = self._grid.version
            if (b, v) not in self._cache and (a, v) in self._cache:
                # D is symmetric on the undirected grid: reuse the cached field to ``a``
                fa = self._cache[(a, v)]
                self._cache.move_to_end((a, v))
                self.hits += 1
                d = int(fa[b[1], b[0]])
                return None if d == UNREACHABLE else d
        d = int(self.field(b)[a[1], a[0]])
        return None if d == UNREACHABLE else d

    def capped_distance(self, a: Cell, b: Cell) -> int:
        """``D̄(a, b) = min(D(a, b), W·H)`` (model §1)."""
        d = self.distance(a, b)
        cap = self._grid.width * self._grid.height
        return cap if d is None else min(d, cap)

    def reachable(self, a: Cell, b: Cell) -> bool:
        """``D_t(a, b) < ∞``: both cells free and in one 4-connected component (O2; exact)."""
        if self._legacy:
            return self.distance(a, b) is not None
        g = self._grid
        if not (g.in_bounds(a) and g.in_bounds(b)):
            return False
        labels = self.component_labels()
        la = labels[a[1] * g.width + a[0]]
        return la >= 0 and la == labels[b[1] * g.width + b[0]]


@contextmanager
def legacy_distance(enabled: bool = True) -> Iterator[None]:
    """Oracles created inside the block use the pre-optimisation implementation (benchmark
    before/after pairing and equivalence tests only)."""
    previous = DistanceOracle.default_legacy
    DistanceOracle.default_legacy = enabled
    try:
        yield
    finally:
        DistanceOracle.default_legacy = previous
