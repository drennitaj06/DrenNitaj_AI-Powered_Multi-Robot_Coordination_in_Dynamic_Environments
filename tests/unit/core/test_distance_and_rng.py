from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from robocoord.core.distance import (
    UNREACHABLE,
    DistanceOracle,
    bfs_distance_field,
    connected_components,
)
from robocoord.core.grid import OccupancyGrid
from robocoord.core.rng import STREAM_NAMES, make_streams
from tests.strategies import grids


def floyd_warshall(grid: OccupancyGrid) -> dict[tuple[tuple[int, int], tuple[int, int]], float]:
    """Independent all-pairs oracle (O(V^3)), written without reusing library code."""
    cells = grid.free_cells()
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    d = np.full((n, n), np.inf)
    for i in range(n):
        d[i, i] = 0
    for c in cells:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (c[0] + dx, c[1] + dy)
            if nb in idx:
                d[idx[c], idx[nb]] = 1
    for k in range(n):
        d = np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    return {(a, b): float(d[idx[a], idx[b]]) for a in cells for b in cells}


@given(grids(max_size=6, max_density=0.5))
def test_bfs_matches_floyd_warshall(grid: OccupancyGrid) -> None:
    ref = floyd_warshall(grid)
    for target in grid.free_cells():
        fld = bfs_distance_field(grid, target)
        for src in grid.free_cells():
            expected = ref[(src, target)]
            got = int(fld[src[1], src[0]])
            if np.isinf(expected):
                assert got == UNREACHABLE
            else:
                assert got == expected
        # blocked cells are always unreachable
        ys, xs = np.nonzero(grid.blocked)
        assert all(fld[y, x] == UNREACHABLE for x, y in zip(xs, ys, strict=True))


def test_bfs_blocked_target_gives_all_unreachable() -> None:
    g = OccupancyGrid.from_rows([".@."])
    assert (bfs_distance_field(g, (1, 0)) == UNREACHABLE).all()


def test_oracle_cache_and_invalidation() -> None:
    g = OccupancyGrid.from_rows([".....", "....."])
    oracle = DistanceOracle(g, capacity=2)
    assert oracle.distance((0, 0), (4, 0)) == 4
    assert oracle.distance((0, 0), (4, 0)) == 4
    assert (oracle.hits, oracle.misses) == (1, 1)
    # wall with one gap at the top row: path must detour
    g.set_blocked((2, 0), True)
    assert oracle.distance((0, 0), (4, 0)) == 6
    g.set_blocked((2, 1), True)
    assert oracle.distance((0, 0), (4, 0)) is None
    assert not oracle.reachable((0, 0), (4, 0))
    assert oracle.capped_distance((0, 0), (4, 0)) == 10  # W*H cap
    assert oracle.distance((0, 0), (9, 9)) is None  # out of bounds
    # capacity respected
    oracle.field((0, 1))
    oracle.field((1, 1))
    oracle.field((3, 1))
    assert len(oracle._cache) == 2


def test_connected_components() -> None:
    g = OccupancyGrid.from_rows(["..@..", "..@..", "@@@.."])
    labels = connected_components(g)
    assert labels[2, 0] == labels[1, 1]  # top-left block
    assert labels[2, 0] != labels[2, 4]
    assert labels[0, 0] == -1


def test_streams_are_deterministic_and_independent() -> None:
    a = make_streams(42)
    b = make_streams(42)
    assert tuple(a.sequences) == STREAM_NAMES
    for name in STREAM_NAMES:
        assert a.generator(name).integers(0, 10**9) == b.generator(name).integers(0, 10**9)
    # a stream restarts from the same state every time it is requested
    assert a.generator("tasks").random() == a.generator("tasks").random()
    # different names / seeds give different values
    assert a.generator("tasks").random() != a.generator("obstacles").random()
    assert a.generator("tasks").random() != make_streams(43).generator("tasks").random()
    # per-entity generators are deterministic and distinct
    g1 = [g.random() for g in a.generators("obstacles", 3)]
    g2 = [g.random() for g in b.generators("obstacles", 3)]
    assert g1 == g2 and len(set(g1)) == 3
    assert a.int_seed("torch") == b.int_seed("torch") >= 0
    with pytest.raises(KeyError):
        a.generator("nope")
    with pytest.raises(ValueError):
        make_streams(-1)


@given(st.integers(0, 2**32))
def test_int_seed_fits_63_bits(seed: int) -> None:
    assert 0 <= make_streams(seed).int_seed("torch") < 2**63
