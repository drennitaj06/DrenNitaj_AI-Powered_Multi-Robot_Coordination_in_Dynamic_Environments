"""Segment geometry for RRT*: supercover traversal, collision test, weighted cost and
discretisation of polylines into executable cell paths (model §5.2; algorithms §3.3).

Conventions. Cell ``(x, y)`` is the square ``[x, x+1) × [y, y+1)``; ``cell(p) = (⌊p_x⌋, ⌊p_y⌋)``.

* The **supercover** of a segment is the set of cells whose *closed* square intersects it. A
  segment is collision-free iff every supercover cell is free; in particular, at an exact
  lattice-corner crossing all four touching cells must be free (Lemma 5.1).
* The traversal (Amanatides & Woo, 1987) walks the cells in order, one axis step at a time, with
  the parametric interval ``[t_in, t_out]`` spent in each. At an exact corner the x-step side
  cell is put on the path (with a zero-length interval) and the y-step side cell is checked only.
* The **weighted cost** is ``‖q − p‖₂ · Σ (t_out − t_in) ω(cell)``: the exact integral of the
  piecewise-constant density along the segment (cells by the ``⌊·⌋`` convention).

Every segment is evaluated in a **canonical direction** (lexicographically smaller endpoint
first), so collision results and costs are bitwise symmetric: ``c(p, q) == c(q, p)``.

Time: O(ℓ + 1) per segment spanning ℓ cells (Guaranteed).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from robocoord.core.types import Cell

#: Two crossings closer than this (in the segment parameter) count as one exact corner crossing.
#: This only ever adds side cells to the check, so it is conservative.
CORNER_EPS = 1e-9

Point = tuple[float, float]


def _cells_at_point(x: float, y: float) -> list[Cell]:
    """All cells whose closed square contains the point (1, 2 or 4 cells)."""
    fx, fy = math.floor(x), math.floor(y)
    xs = [fx, fx - 1] if x == fx else [fx]
    ys = [fy, fy - 1] if y == fy else [fy]
    return [(a, b) for a in xs for b in ys]


def _walk_canonical(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[list[Cell], list[tuple[float, float]], list[Cell]]:
    """Walk from ``(x0, y0)`` to ``(x1, y1)`` (canonical order assumed).

    Returns ``(path, intervals, side)``: the 4-connected ordered path cells, their parameter
    intervals, and the check-only cells (corner y-sides and closed-square boundary contacts).
    """
    dx, dy = x1 - x0, y1 - y0
    cx, cy = math.floor(x0), math.floor(y0)
    ex, ey = math.floor(x1), math.floor(y1)
    inf = math.inf
    if dx > 0:
        sx, tmx, tdx = 1, (cx + 1 - x0) / dx, 1.0 / dx
    elif dx < 0:
        sx, tmx, tdx = -1, (cx - x0) / dx, -1.0 / dx
    else:
        sx, tmx, tdx = 0, inf, inf
    if dy > 0:
        sy, tmy, tdy = 1, (cy + 1 - y0) / dy, 1.0 / dy
    elif dy < 0:
        sy, tmy, tdy = -1, (cy - y0) / dy, -1.0 / dy
    else:
        sy, tmy, tdy = 0, inf, inf
    path: list[Cell] = []
    intervals: list[tuple[float, float]] = []
    side: list[Cell] = []
    t = 0.0
    while cx != ex or cy != ey:
        can_x, can_y = cx != ex, cy != ey
        if can_x and can_y and abs(tmx - tmy) <= CORNER_EPS:
            tn = min(max(tmx, t), 1.0)
            path.append((cx, cy))
            intervals.append((t, tn))
            path.append((cx + sx, cy))  # x-step side cell: on the path, zero length
            intervals.append((tn, tn))
            side.append((cx, cy + sy))  # y-step side cell: checked only
            cx += sx
            cy += sy
            tmx += tdx
            tmy += tdy
        elif can_x and (not can_y or tmx < tmy):
            tn = min(max(tmx, t), 1.0)
            path.append((cx, cy))
            intervals.append((t, tn))
            cx += sx
            tmx += tdx
        else:
            tn = min(max(tmy, t), 1.0)
            path.append((cx, cy))
            intervals.append((t, tn))
            cy += sy
            tmy += tdy
        t = tn
    path.append((cx, cy))
    intervals.append((t, 1.0))
    # closed-square contacts: endpoints on cell boundaries, segments lying on a grid line
    if x0 == math.floor(x0) or y0 == math.floor(y0):
        side.extend(_cells_at_point(x0, y0))
    if x1 == math.floor(x1) or y1 == math.floor(y1):
        side.extend(_cells_at_point(x1, y1))
    if dy == 0 and y0 == math.floor(y0):
        side.extend((a, b - 1) for a, b in path)
    if dx == 0 and x0 == math.floor(x0):
        side.extend((a - 1, b) for a, b in path)
    return path, intervals, side


def _canonical(p: Point, q: Point) -> tuple[Point, Point, bool]:
    return (q, p, True) if (q[0], q[1]) < (p[0], p[1]) else (p, q, False)


def supercover(p: Point, q: Point) -> set[Cell]:
    """The cells whose closed square intersects segment ``pq`` (the collision-check set)."""
    a, b, _ = _canonical(p, q)
    path, _, side = _walk_canonical(a[0], a[1], b[0], b[1])
    return set(path) | set(side)


def traverse(p: Point, q: Point) -> tuple[list[Cell], list[tuple[float, float]]]:
    """Ordered 4-connected path cells from ``cell(p)`` to ``cell(q)`` with parameter intervals
    (``t = 0`` at ``p``). Zero-length entries are corner side cells."""
    a, b, swapped = _canonical(p, q)
    path, intervals, _ = _walk_canonical(a[0], a[1], b[0], b[1])
    if swapped:
        path.reverse()
        intervals = [(1.0 - t1, 1.0 - t0) for t0, t1 in reversed(intervals)]
    return path, intervals


def segment_cost(
    p: Point,
    q: Point,
    free_flat: Sequence[bool],
    width: int,
    height: int,
    penalised: frozenset[int] | set[int],
    heavy: float,
) -> tuple[float, int]:
    """Collision test and weighted cost of segment ``pq`` in one traversal.

    Returns ``(cost, cells_examined)``; ``cost`` is ``inf`` if any supercover cell is out of bounds
    or not free. ``penalised`` holds flat indices of ``Q_i``; ``heavy = 1 + λ_prox``.
    Time: O(ℓ + 1).
    """
    a, b, _ = _canonical(p, q)
    path, intervals, side = _walk_canonical(a[0], a[1], b[0], b[1])
    n_cells = 0
    for x, y in side:
        n_cells += 1
        if not (0 <= x < width and 0 <= y < height) or not free_flat[y * width + x]:
            return math.inf, n_cells
    acc = 0.0
    for (x, y), (t0, t1) in zip(path, intervals, strict=True):
        n_cells += 1
        if not (0 <= x < width and 0 <= y < height):
            return math.inf, n_cells
        idx = y * width + x
        if not free_flat[idx]:
            return math.inf, n_cells
        acc += (t1 - t0) * (heavy if idx in penalised else 1.0)
    return math.hypot(b[0] - a[0], b[1] - a[1]) * acc, n_cells


def discretise(points: Sequence[Point]) -> list[Cell]:
    """Polyline → executable 4-connected cell path (``04`` §3.3).

    Walks each segment's traversal in order and removes consecutive duplicates. Every produced
    cell belongs to the supercover of its segment, so a polyline whose segments passed the
    collision test yields a cell path that is free on the same snapshot (Guaranteed).
    """
    if not points:
        return []
    out: list[Cell] = [(math.floor(points[0][0]), math.floor(points[0][1]))]
    for p, q in itertools.pairwise(points):
        cells, _ = traverse(p, q)
        for c in cells:
            if c != out[-1]:
                out.append(c)
    return out


def remove_loops(cells: Sequence[Cell]) -> list[Cell]:
    """Cut every loop from a 4-connected cell path: when a cell is visited again, the cells
    between its two visits are dropped.

    The result is a sub-sequence of the input that keeps the first and last cell and visits every
    cell at most once. It stays valid on the same snapshot: every kept cell was already
    collision-checked, and the join point is the same cell, so consecutive cells remain 4-adjacent.
    Its lattice cost ``Σ κ`` cannot increase, because only cells (each with ``κ ≥ 1``) are removed.
    Time: O(L) amortised.
    """
    out: list[Cell] = []
    where: dict[Cell, int] = {}
    for c in cells:
        j = where.get(c)
        if j is None:
            where[c] = len(out)
            out.append(c)
            continue
        for removed in out[j + 1 :]:  # drop the loop c → … → c
            del where[removed]
        del out[j + 1 :]
    return out


def polyline_length(points: Sequence[Point]) -> float:
    """Euclidean length of a polyline."""
    return float(sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in itertools.pairwise(points)))
