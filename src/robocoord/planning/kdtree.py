"""Static alternating-median 2-d trees and the Bentley–Saxe logarithmic method (algorithms §3.2).

* :class:`StaticKDTree` — built once from a point set by recursive median splits that alternate
  between the x and y axes (median via ``numpy.argpartition``, O(m) per level). Leaves hold at most
  ``leaf_size`` points. Depth ≤ ⌈log₂(m / b)⌉, build O(m log m), nodes in flat lists.
* :class:`LogKDTreeIndex` — the dynamic structure: slot ``i`` is empty or holds a static tree of
  exactly ``2^i`` points (the binary representation of ``n``). Insert rebuilds the lowest empty
  slot from the new point and all smaller slots. Queries visit every non-empty slot.

Bounds (``04`` §3.6, named): insert O(log² n) amortised, O(n log n) worst case for one insert;
nearest O(n) worst case, expected O(log² n) only under the random-point model (assumption R);
radius (bounding-square traversal + disk filter) O(√n + k′) worst case.

Median ties: equal coordinates may fall on both sides of a split; the median property holds with
``≤``/``≥`` and introselect is deterministic, so trees are reproducible.

Work counters (all cumulative on :class:`LogKDTreeIndex`):

* ``nn_visited`` — points whose distance a nearest query evaluated (leaf points only);
* ``nn_nodes`` — tree nodes a nearest query examined (popped and box-tested), **including** nodes
  pruned by the bounding-box test and internal nodes;
* ``range_points`` — k′, points inside the query's bounding square;
* ``range_nodes`` — tree nodes a radius query *expanded* (box intersects the square);
* ``range_nodes_tested`` — tree nodes a radius query examined, including pruned ones;
* ``rebuild_points`` — points placed into newly built static trees: per insertion, the size
  ``2^i`` of the new slot tree (the inserted point plus the points moved from smaller slots);
* ``rebuild_work`` — points processed by those builds, Σ over every built node of its subtree size
  (the min/max and ``argpartition`` work of each level). This is the quantity the O(m log m) build
  bound and the amortised O(log² n) insertion bound are about.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from robocoord.planning.neighbors import IdArray


class StaticKDTree:
    """A balanced 2-d tree over a fixed set of point ids (coordinates read from ``xs``, ``ys``)."""

    __slots__ = (
        "axis",
        "build_work",
        "depth",
        "hi",
        "leaf_size",
        "left",
        "lo",
        "maxx",
        "maxy",
        "minx",
        "miny",
        "pid",
        "px",
        "py",
        "right",
        "size",
        "split",
    )

    def __init__(
        self,
        ids: npt.NDArray[np.int64],
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        leaf_size: int = 8,
    ) -> None:
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        self.leaf_size = leaf_size
        self.size = len(ids)
        perm = np.array(ids, dtype=np.int64)
        self.axis: list[int] = []
        self.split: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.lo: list[int] = []
        self.hi: list[int] = []
        self.minx: list[float] = []
        self.maxx: list[float] = []
        self.miny: list[float] = []
        self.maxy: list[float] = []
        self.depth = 0
        self.build_work = 0  # Σ subtree sizes over all built nodes
        if self.size:
            self._build(perm, xs, ys, 0, self.size, 0)
        self.pid: list[int] = perm.tolist()
        self.px: list[float] = xs[perm].tolist()
        self.py: list[float] = ys[perm].tolist()

    def _build(
        self,
        perm: npt.NDArray[np.int64],
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        lo: int,
        hi: int,
        depth: int,
    ) -> int:
        node = len(self.lo)
        self.depth = max(self.depth, depth)
        self.build_work += hi - lo
        seg = perm[lo:hi]
        sx, sy = xs[seg], ys[seg]
        self.lo.append(lo)
        self.hi.append(hi)
        self.minx.append(float(sx.min()))
        self.maxx.append(float(sx.max()))
        self.miny.append(float(sy.min()))
        self.maxy.append(float(sy.max()))
        self.left.append(-1)
        self.right.append(-1)
        axis = depth % 2
        self.axis.append(axis)
        if hi - lo <= self.leaf_size:
            self.split.append(0.0)
            return node
        key = sx if axis == 0 else sy
        m = (hi - lo) // 2
        order = np.argpartition(key, m)
        perm[lo:hi] = seg[order]
        self.split.append(float(key[order[m]]))
        left = self._build(perm, xs, ys, lo, lo + m, depth + 1)
        right = self._build(perm, xs, ys, lo + m, hi, depth + 1)
        self.left[node] = left
        self.right[node] = right
        return node

    def is_leaf(self, node: int) -> bool:
        return self.left[node] < 0

    def nearest(
        self, qx: float, qy: float, best_d2: float, best_id: int
    ) -> tuple[float, int, int, int]:
        """Improve ``(best_d2, best_id)`` with this tree's points.

        Returns ``(d2, id, points evaluated, nodes examined)``; nodes examined counts every node
        popped from the stack, including internal nodes and nodes then pruned.

        Subtrees are pruned only when their bounding box is strictly farther than the best
        distance, so equal-distance points with lower ids are never missed.
        """
        if not self.size:
            return best_d2, best_id, 0, 0
        px, py, pid = self.px, self.py, self.pid
        left, right, axis, split = self.left, self.right, self.axis, self.split
        minx, maxx, miny, maxy = self.minx, self.maxx, self.miny, self.maxy
        visited = 0
        examined = 0
        stack = [0]
        while stack:
            node = stack.pop()
            examined += 1
            bx = (
                minx[node] - qx
                if qx < minx[node]
                else (qx - maxx[node] if qx > maxx[node] else 0.0)
            )
            by = (
                miny[node] - qy
                if qy < miny[node]
                else (qy - maxy[node] if qy > maxy[node] else 0.0)
            )
            if bx * bx + by * by > best_d2:
                continue
            lc = left[node]
            if lc < 0:
                for i in range(self.lo[node], self.hi[node]):
                    dx = px[i] - qx
                    dy = py[i] - qy
                    d2 = dx * dx + dy * dy
                    visited += 1
                    if d2 < best_d2 or (d2 == best_d2 and pid[i] < best_id):
                        best_d2, best_id = d2, pid[i]
                continue
            rc = right[node]
            q = qx if axis[node] == 0 else qy
            if q < split[node]:  # visit the nearer child first (pushed last)
                stack.append(rc)
                stack.append(lc)
            else:
                stack.append(lc)
                stack.append(rc)
        return best_d2, best_id, visited, examined

    def radius(
        self, qx: float, qy: float, r: float, r2: float, out: list[int]
    ) -> tuple[int, int, int]:
        """Append ids within the closed disk to ``out``.

        Returns ``(k′ points in the bounding square, nodes expanded, nodes examined)``. Nodes are
        pruned against the bounding square of the disk; *examined* includes pruned nodes,
        *expanded* only nodes whose box intersects the square."""
        if not self.size:
            return 0, 0, 0
        px, py, pid = self.px, self.py, self.pid
        left, right = self.left, self.right
        minx, maxx, miny, maxy = self.minx, self.maxx, self.miny, self.maxy
        kprime = 0
        nodes = 0
        examined = 0
        stack = [0]
        while stack:
            node = stack.pop()
            examined += 1
            if (
                qx - maxx[node] > r
                or minx[node] - qx > r
                or qy - maxy[node] > r
                or miny[node] - qy > r
            ):
                continue
            nodes += 1
            lc = left[node]
            if lc < 0:
                for i in range(self.lo[node], self.hi[node]):
                    dx = px[i] - qx
                    dy = py[i] - qy
                    if abs(dx) <= r and abs(dy) <= r:
                        kprime += 1
                        if dx * dx + dy * dy <= r2:
                            out.append(pid[i])
                continue
            stack.append(right[node])
            stack.append(lc)
        return kprime, nodes, examined


class LogKDTreeIndex:
    """Bentley–Saxe logarithmic method over :class:`StaticKDTree` slots."""

    name = "kdtree"

    def __init__(self, leaf_size: int = 8, capacity: int = 256) -> None:
        self.leaf_size = leaf_size
        self._xs = np.empty(max(capacity, 1), dtype=np.float64)
        self._ys = np.empty(max(capacity, 1), dtype=np.float64)
        self._n = 0
        self.slots: list[StaticKDTree | None] = []
        self.nn_visited = 0
        self.nn_nodes = 0
        self.range_points = 0
        self.range_nodes = 0
        self.range_nodes_tested = 0
        self.rebuild_points = 0
        self.rebuild_work = 0

    def __len__(self) -> int:
        return self._n

    def insert(self, x: float, y: float) -> int:
        if self._n == len(self._xs):
            self._xs = np.concatenate([self._xs, np.empty(len(self._xs), dtype=np.float64)])
            self._ys = np.concatenate([self._ys, np.empty(len(self._ys), dtype=np.float64)])
        pid = self._n
        self._xs[pid] = x
        self._ys[pid] = y
        self._n += 1
        i = 0
        while i < len(self.slots) and self.slots[i] is not None:
            i += 1
        if i == len(self.slots):
            self.slots.append(None)
        parts = [np.array([pid], dtype=np.int64)]
        for j in range(i):
            tree = self.slots[j]
            assert tree is not None
            parts.append(np.array(tree.pid, dtype=np.int64))
            self.slots[j] = None
        ids = np.concatenate(parts)
        tree = StaticKDTree(ids, self._xs, self._ys, self.leaf_size)
        self.slots[i] = tree
        self.rebuild_points += len(ids)
        self.rebuild_work += tree.build_work
        return pid

    def nearest(self, x: float, y: float) -> int:
        if self._n == 0:
            raise ValueError("nearest() on an empty index")
        best_d2, best_id = float("inf"), -1
        for tree in reversed(self.slots):  # largest slot first: a good bound early
            if tree is not None:
                best_d2, best_id, visited, examined = tree.nearest(x, y, best_d2, best_id)
                self.nn_visited += visited
                self.nn_nodes += examined
        return best_id

    def radius(self, x: float, y: float, r: float) -> IdArray:
        out: list[int] = []
        r2 = r * r
        for tree in self.slots:
            if tree is not None:
                kprime, nodes, examined = tree.radius(x, y, r, r2, out)
                self.range_points += kprime
                self.range_nodes += nodes
                self.range_nodes_tested += examined
        out.sort()
        return np.array(out, dtype=np.int64)

    def slot_sizes(self) -> list[int]:
        """Size of each slot (0 for empty): must equal the binary representation of ``n``."""
        return [0 if t is None else t.size for t in self.slots]
