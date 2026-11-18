"""Neighbour indexes for RRT* (algorithms §3.2) and the brute-force reference index.

Both indexes answer exactly the same queries with identical results:

* ``nearest(q)`` — the id of the point with the smallest squared distance
  ``dx*dx + dy*dy``; ties go to the **lowest id**.
* ``radius(q, r)`` — ascending ids of the points with ``|dx| ≤ r``, ``|dy| ≤ r`` and
  ``dx*dx + dy*dy ≤ r*r`` (closed disk).

Squared distances are computed with the same IEEE operations in both indexes, so ties and disk
borders are decided identically (the brute-force index is the test oracle of the k-d tree).

Work counters (cumulative, for ``04`` §3.6 / experiment E1; exact definitions in
``planning/kdtree.py``): ``nn_visited`` (points whose distance a nearest query evaluated),
``nn_nodes`` (tree nodes examined by nearest queries, including pruned and internal nodes),
``range_points`` (k′: points inside the query's bounding square), ``range_nodes`` (tree nodes a
radius query expanded), ``range_nodes_tested`` (tree nodes examined, including pruned ones),
``rebuild_points`` (points placed into newly built static trees) and ``rebuild_work`` (points
processed by those builds). The tree counters are 0 for the brute-force index, which evaluates
all ``n`` points per nearest query.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from robocoord.core.errors import ConfigError

IdArray = npt.NDArray[np.int64]


class NeighborIndex(Protocol):
    name: str
    nn_visited: int
    nn_nodes: int
    range_points: int
    range_nodes: int
    range_nodes_tested: int
    rebuild_points: int
    rebuild_work: int

    def __len__(self) -> int: ...

    def insert(self, x: float, y: float) -> int:
        """Add a point; returns its id (``0, 1, 2, …`` in insertion order)."""
        ...

    def nearest(self, x: float, y: float) -> int: ...

    def radius(self, x: float, y: float, r: float) -> IdArray: ...


class BruteForceIndex:
    """Points in a growing NumPy array; every query is one vectorised scan.

    Insert: O(1) amortised (array doubling). Nearest and radius: Θ(n) worst and expected.
    """

    name = "brute"

    def __init__(self, capacity: int = 256) -> None:
        self._xy = np.empty((max(capacity, 1), 2), dtype=np.float64)
        self._n = 0
        self.nn_visited = 0
        self.nn_nodes = 0  # no tree: always 0
        self.range_points = 0
        self.range_nodes = 0
        self.range_nodes_tested = 0
        self.rebuild_points = 0
        self.rebuild_work = 0

    def __len__(self) -> int:
        return self._n

    def insert(self, x: float, y: float) -> int:
        if self._n == len(self._xy):
            grown = np.empty((2 * len(self._xy), 2), dtype=np.float64)
            grown[: self._n] = self._xy[: self._n]
            self._xy = grown
        self._xy[self._n] = (x, y)
        self._n += 1
        return self._n - 1

    def nearest(self, x: float, y: float) -> int:
        if self._n == 0:
            raise ValueError("nearest() on an empty index")
        dx = self._xy[: self._n, 0] - x
        dy = self._xy[: self._n, 1] - y
        d2 = dx * dx + dy * dy
        self.nn_visited += self._n
        return int(np.argmin(d2))  # first occurrence = lowest id on ties

    def radius(self, x: float, y: float, r: float) -> IdArray:
        dx = self._xy[: self._n, 0] - x
        dy = self._xy[: self._n, 1] - y
        in_square = (np.abs(dx) <= r) & (np.abs(dy) <= r)
        self.range_points += int(np.count_nonzero(in_square))
        within = in_square & (dx * dx + dy * dy <= r * r)
        return np.flatnonzero(within).astype(np.int64)


def make_index(name: str, leaf_size: int = 8) -> NeighborIndex:
    """Index factory by name: ``"kdtree"`` (Bentley–Saxe logarithmic method) or ``"brute"``."""
    if name == "brute":
        return BruteForceIndex()
    if name == "kdtree":
        from robocoord.planning.kdtree import LogKDTreeIndex

        return LogKDTreeIndex(leaf_size=leaf_size)
    raise ConfigError("neighbor_index", f"unknown index {name!r}; expected 'kdtree' or 'brute'")
