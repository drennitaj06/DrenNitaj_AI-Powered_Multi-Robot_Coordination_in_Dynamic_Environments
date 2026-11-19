"""NFR-PERF-1 optimisations O1 (flat BFS) and O2 (component reachability) are exact: they agree
with the pre-optimisation implementation on random grids, before and after map events, and on
the E5 maps. The legacy labelling below is the earlier ``connected_components`` verbatim."""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.distance import (
    UNREACHABLE,
    DistanceOracle,
    bfs_distance_field,
    bfs_distance_field_legacy,
    connected_components,
    legacy_distance,
)
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import MOVES
from robocoord.experiments.allocation_runs import e5_scenario
from tests.strategies import grids


def legacy_components(grid: OccupancyGrid) -> npt.NDArray[np.int32]:
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
            queue = deque([(x, y)])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in deltas:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx] and labels[ny, nx] == -1:
                        labels[ny, nx] = current
                        queue.append((nx, ny))
            current += 1
    return labels


@settings(max_examples=80, deadline=None)
@given(grids(max_size=9, max_density=0.5))
def test_flat_bfs_and_labels_equal_the_legacy_implementation(grid: OccupancyGrid) -> None:
    for y in range(grid.height):
        for x in range(grid.width):  # every target, blocked ones included
            legacy = bfs_distance_field_legacy(grid, (x, y))
            assert np.array_equal(bfs_distance_field(grid, (x, y)), legacy)
            assert bfs_distance_field(grid, (x, y)).dtype == np.int32
    assert np.array_equal(connected_components(grid), legacy_components(grid))


@settings(max_examples=50, deadline=None)
@given(grids(min_size=2, max_size=8, max_density=0.4), st.data())
def test_oracle_matches_legacy_oracle_across_map_events(
    grid: OccupancyGrid, data: st.DataObject
) -> None:
    """Fast and legacy oracles on the same mutable grid: identical fields, distances and
    reachability before and after every block/unblock event (cached state is never stale)."""
    fast, slow = DistanceOracle(grid), DistanceOracle(grid, legacy=True)
    assert not fast.legacy and slow.legacy
    cells = [(x, y) for y in range(grid.height) for x in range(grid.width)]
    events = data.draw(st.lists(st.sampled_from(cells), max_size=4))
    for k in range(len(events) + 1):
        for a in cells:
            for b in cells[:: max(1, len(cells) // 6)]:
                assert (
                    fast.reachable(a, b)
                    == slow.reachable(a, b)
                    == (slow.distance(a, b) is not None)
                )
                assert fast.distance(a, b) == slow.distance(a, b)
        assert fast.reachable((0, 0), (-1, 0)) is False and fast.reachable((99, 0), (0, 0)) is False
        if k < len(events):
            c = events[k]
            grid.set_blocked(c, not grid.is_blocked(c))  # a map event: new grid version


def test_fast_oracle_labels_once_per_version_and_fields_are_read_only() -> None:
    g = OccupancyGrid.from_rows([".....", ".....", "....."])
    o = DistanceOracle(g)
    assert o.reachable((0, 0), (4, 2)) and o.label_builds == 1 and o.misses == 0
    assert o.reachable((1, 1), (3, 0)) and o.label_builds == 1  # cached for the version
    g.set_blocked((2, 0), True)
    g.set_blocked((2, 1), True)
    g.set_blocked((2, 2), True)
    assert not o.reachable((0, 0), (4, 2)) and o.label_builds == 2
    assert not o.reachable((2, 1), (2, 1))  # a blocked cell reaches nothing
    fld = o.field((4, 2))
    assert not fld.flags.writeable and fld[0, 0] == UNREACHABLE
    assert (o.field((2, 0)) == UNREACHABLE).all()  # blocked target


def test_legacy_switch_restores_the_default() -> None:
    g = OccupancyGrid.from_rows(["..."])
    with legacy_distance():
        assert DistanceOracle(g).legacy
        with legacy_distance(False):
            assert not DistanceOracle(g).legacy
        assert DistanceOracle(g).legacy
    assert not DistanceOracle(g).legacy


def test_e5_maps_identical_under_both_implementations() -> None:
    """The E5 / benchmark generator (which labels components) yields the same maps, and the
    oracles agree on every field the episode could ask for (sampled targets)."""
    for seed in (20200, 30500):
        a = e5_scenario(
            seed,
            "full",
            size=32,
            density=0.15,
            robots=8,
            tasks_per_robot=2,
            obstacles=8,
            max_steps=400,
            arrival_rate=0.05,
            arrivals_until=200,
        )
        with legacy_distance():
            b = e5_scenario(
                seed,
                "full",
                size=32,
                density=0.15,
                robots=8,
                tasks_per_robot=2,
                obstacles=8,
                max_steps=400,
                arrival_rate=0.05,
                arrivals_until=200,
            )
        assert np.array_equal(a.grid.blocked, b.grid.blocked) and a.tasks == b.tasks
        fast, slow = DistanceOracle(a.grid), DistanceOracle(a.grid, legacy=True)
        for task in a.tasks:
            for target in (task.pickup, task.dropoff):
                assert np.array_equal(fast.field(target), slow.field(target))
