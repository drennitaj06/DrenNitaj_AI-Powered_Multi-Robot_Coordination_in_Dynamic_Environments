"""Supercover traversal, weighted segment cost and discretisation (``04`` §3.3, §3.9 test 1)."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from robocoord.core.grid import OccupancyGrid
from robocoord.planning.base import PlanConstraints
from robocoord.planning.geometry import (
    discretise,
    polyline_length,
    remove_loops,
    segment_cost,
    supercover,
    traverse,
)

Point = tuple[float, float]


def touches_closed_square(p: Point, q: Point, cx: int, cy: int, margin: float = 0.0) -> bool:
    """Exact oracle: does the closed segment meet the closed square of cell (cx, cy), optionally
    inflated by ``margin`` on every side?

    Liang–Barsky clipping in rational arithmetic (floats convert exactly to Fractions).
    """
    x0, y0, x1, y1 = (Fraction(v) for v in (*p, *q))
    m = Fraction(margin)
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = Fraction(0), Fraction(1)
    for pp, qq in (
        (-dx, x0 - cx + m),
        (dx, cx + 1 - x0 + m),
        (-dy, y0 - cy + m),
        (dy, cy + 1 - y0 + m),
    ):
        if pp == 0:
            if qq < 0:
                return False
        else:
            r = qq / pp
            if pp < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
    return t0 <= t1


def oracle_cells(p: Point, q: Point) -> set[tuple[int, int]]:
    xs = range(math.floor(min(p[0], q[0])) - 1, math.floor(max(p[0], q[0])) + 2)
    ys = range(math.floor(min(p[1], q[1])) - 1, math.floor(max(p[1], q[1])) + 2)
    return {(x, y) for x in xs for y in ys if touches_closed_square(p, q, x, y)}


coord = st.floats(0.0, 9.999, allow_nan=False, allow_infinity=False)
lattice = st.integers(0, 18).map(lambda v: v / 2)  # integers and half-integers: degenerate cases


@given(coord, coord, coord, coord)
def test_supercover_equals_exact_oracle_random(x0: float, y0: float, x1: float, y1: float) -> None:
    """Exact equality except within the documented corner tolerance: the implementation may add
    a side cell when the segment passes within ~1e-9 of a lattice corner (conservative)."""
    got, exact = supercover((x0, y0), (x1, y1)), oracle_cells((x0, y0), (x1, y1))
    assert exact <= got
    for cx, cy in got - exact:
        assert touches_closed_square((x0, y0), (x1, y1), cx, cy, margin=1e-7)


@given(lattice, lattice, lattice, lattice)
def test_supercover_equals_exact_oracle_on_lattice_points(
    x0: float, y0: float, x1: float, y1: float
) -> None:
    """Corners, grid-line segments and boundary endpoints (the degenerate cases)."""
    assert supercover((x0, y0), (x1, y1)) == oracle_cells((x0, y0), (x1, y1))


def test_explicit_corner_cases() -> None:
    # diagonal through the exact corner (1, 1): all four touching cells
    assert supercover((0.5, 0.5), (1.5, 1.5)) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    # anti-diagonal through (1, 1)
    assert supercover((0.5, 1.5), (1.5, 0.5)) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    # segment lying on the grid line y = 1 touches both rows
    assert supercover((0.5, 1.0), (2.5, 1.0)) == {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}
    # interior crossing (no corner): exactly the two cells
    assert supercover((0.5, 0.5), (1.5, 0.7)) == {(0, 0), (1, 0)}
    # path order: x-step side cell at the corner, zero-length interval
    path, intervals = traverse((0.5, 0.5), (1.5, 1.5))
    assert path == [(0, 0), (1, 0), (1, 1)]
    assert intervals[1][0] == intervals[1][1] == 0.5
    rev, _ = traverse((1.5, 1.5), (0.5, 0.5))
    assert rev == [(1, 1), (1, 0), (0, 0)]


def _constraints(rows: list[str], prox: set[tuple[int, int]] | None = None, lam: float = 0.0):  # type: ignore[no-untyped-def]
    return PlanConstraints(OccupancyGrid.from_rows(rows), frozenset(), frozenset(prox or ()), lam)


def _cost(c: PlanConstraints, p: Point, q: Point) -> float:
    return segment_cost(p, q, c.free_flat, c.width, c.height, c.penalised_flat, 1 + c.lambda_prox)[
        0
    ]


def test_lemma_5_1_corner_requires_both_side_cells_free() -> None:
    p, q = (0.5, 0.5), (1.5, 1.5)
    assert _cost(_constraints(["..", ".."]), p, q) == pytest.approx(math.sqrt(2), rel=1e-12)
    assert _cost(_constraints(["..", "@."]), p, q) == math.inf  # side cell (0, 1) blocked
    assert _cost(_constraints(["..", ".@"]), p, q) == math.inf  # (1, 0) blocked
    # out of bounds counts as blocked
    assert _cost(_constraints([".."]), (0.5, 0.5), (2.5, 0.5)) == math.inf


@given(st.integers(0, 10_000))
def test_accepted_segments_give_free_4_connected_chains(seed: int) -> None:
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid(rng.random((6, 6)) < 0.3)
    c = PlanConstraints(grid)
    p = (float(rng.uniform(0, 6)), float(rng.uniform(0, 6)))
    q = (float(rng.uniform(0, 6)), float(rng.uniform(0, 6)))
    cost = _cost(c, p, q)
    cells = supercover(p, q)
    assert (cost != math.inf) == all(c.is_free(x) for x in cells)
    path, intervals = traverse(p, q)
    assert set(path) <= cells
    assert path[0] == (math.floor(p[0]), math.floor(p[1]))
    assert path[-1] == (math.floor(q[0]), math.floor(q[1]))
    for a, b in itertools.pairwise(path):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    assert intervals[0][0] == 0.0 and intervals[-1][1] == 1.0
    assert all(t0 <= t1 for t0, t1 in intervals)


@given(st.integers(0, 10_000))
def test_weighted_cost_matches_numerical_integration(seed: int) -> None:
    rng = np.random.default_rng(seed)
    prox = {(int(x), int(y)) for x, y in rng.integers(0, 6, size=(8, 2))}
    lam = float(rng.uniform(0.5, 3.0))
    c = PlanConstraints(OccupancyGrid.empty(6, 6), frozenset(), frozenset(prox), lam)
    p = (float(rng.uniform(0, 6)), float(rng.uniform(0, 6)))
    q = (float(rng.uniform(0, 6)), float(rng.uniform(0, 6)))
    n = 20_000
    t = (np.arange(n) + 0.5) / n
    xs = np.floor(p[0] + t * (q[0] - p[0])).astype(int)
    ys = np.floor(p[1] + t * (q[1] - p[1])).astype(int)
    dens = np.array([c.density((int(a), int(b))) for a, b in zip(xs, ys, strict=True)])
    length = math.hypot(q[0] - p[0], q[1] - p[1])
    numeric = length * float(dens.mean())
    # midpoint rule: error ≤ length · λ · (number of crossings) / n
    assert abs(_cost(c, p, q) - numeric) <= length * lam * 14 / n + 1e-12
    # symmetric bitwise, and equal to Euclidean length without penalties
    assert _cost(c, p, q) == _cost(c, q, p)
    assert _cost(PlanConstraints(c.static), p, q) == pytest.approx(length, rel=1e-12)


def test_discretise_polyline() -> None:
    pts = [(0.5, 0.5), (2.7, 0.6), (2.6, 2.4), (0.5, 2.5)]
    cells = discretise(pts)
    assert cells[0] == (0, 0) and cells[-1] == (0, 2)
    for a, b in itertools.pairwise(cells):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    assert discretise([]) == []
    assert discretise([(1.2, 1.2)]) == [(1, 1)]
    assert polyline_length([(0.0, 0.0), (3.0, 4.0)]) == 5.0


def test_remove_loops_examples() -> None:
    assert remove_loops([]) == []
    assert remove_loops([(0, 0)]) == [(0, 0)]
    # A → B → A → C: the detour to B is cut
    assert remove_loops([(0, 0), (1, 0), (0, 0), (0, 1)]) == [(0, 0), (0, 1)]
    # nested and consecutive loops
    path = [(0, 0), (1, 0), (2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (3, 0)]
    assert remove_loops(path) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    # the start is revisited: everything before the last visit is cut
    assert remove_loops([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, -1)]) == [(0, 0), (0, -1)]


@given(st.integers(0, 100_000))
def test_remove_loops_keeps_a_valid_simple_subpath(seed: int) -> None:
    rng = np.random.default_rng(seed)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    walk = [(0, 0)]
    for _ in range(int(rng.integers(0, 60))):
        dx, dy = moves[int(rng.integers(4))]
        walk.append((walk[-1][0] + dx, walk[-1][1] + dy))
    out = remove_loops(walk)
    assert out[0] == walk[0] and out[-1] == walk[-1]
    assert len(set(out)) == len(out)  # simple
    assert set(out) <= set(walk)  # only original (already checked) cells
    for a, b in itertools.pairwise(out):  # still 4-connected
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    it = iter(walk)  # a sub-sequence of the input
    assert all(c in it for c in out)
