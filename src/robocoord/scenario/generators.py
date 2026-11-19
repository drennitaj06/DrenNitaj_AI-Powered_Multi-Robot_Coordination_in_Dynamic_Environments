"""Synthetic map generators. Every generator returns a grid whose free space is 4-connected."""

from __future__ import annotations

import numpy as np

from robocoord.core.distance import connected_components
from robocoord.core.grid import OccupancyGrid


def keep_largest_component(grid: OccupancyGrid) -> OccupancyGrid:
    """Block every free cell outside the largest 4-connected free component (ties: lowest label)."""
    labels = connected_components(grid)
    if (labels < 0).all():
        return grid.copy()
    counts = np.bincount(labels[labels >= 0])
    keep = int(np.argmax(counts))
    return OccupancyGrid(labels != keep)


def random_density(
    width: int, height: int, density: float, rng: np.random.Generator
) -> OccupancyGrid:
    """I.i.d. obstacles with probability ``density``, restricted to the largest free component."""
    if not 0.0 <= density < 1.0:
        raise ValueError("density must be in [0, 1)")
    grid = OccupancyGrid(rng.random((height, width)) < density)
    return keep_largest_component(grid)


def rooms(width: int, height: int, room_size: int, rng: np.random.Generator) -> OccupancyGrid:
    """A lattice of ``room_size × room_size`` rooms separated by 1-cell walls.

    Every wall between two horizontally or vertically adjacent rooms gets one door at a random
    position, so the room graph is a full grid and the free space is connected.
    """
    if room_size < 1:
        raise ValueError("room_size must be >= 1")
    period = room_size + 1
    blocked = np.zeros((height, width), dtype=bool)
    xs = np.arange(width)
    ys = np.arange(height)
    blocked[:, xs % period == room_size] = True
    blocked[ys % period == room_size, :] = True
    n_rx = (width + period - 1) // period
    n_ry = (height + period - 1) // period
    for ry in range(n_ry):
        for rx in range(n_rx):
            x0, y0 = rx * period, ry * period
            x1 = min(x0 + room_size, width)  # exclusive
            y1 = min(y0 + room_size, height)
            if x1 <= x0 or y1 <= y0:
                continue
            wall_x = x0 + room_size
            if wall_x < width - 1:  # door to the right neighbour
                blocked[int(rng.integers(y0, y1)), wall_x] = False
            wall_y = y0 + room_size
            if wall_y < height - 1:  # door to the upper neighbour
                blocked[wall_y, int(rng.integers(x0, x1))] = False
    return keep_largest_component(OccupancyGrid(blocked))


def warehouse(width: int, height: int, shelf_length: int = 4) -> OccupancyGrid:
    """Rows of shelf blocks with aisles; a free 1-cell border and cross aisles keep it connected.

    Deterministic (no randomness): shelf rows at ``y = 2, 5, 8, …`` (two aisle rows in between),
    shelf segments of ``shelf_length`` separated by one-cell cross aisles.
    """
    if shelf_length < 1:
        raise ValueError("shelf_length must be >= 1")
    blocked = np.zeros((height, width), dtype=bool)
    for y in range(2, height - 2, 3):
        x = 2
        while x < width - 2:
            end = min(x + shelf_length, width - 2)
            blocked[y, x:end] = True
            x = end + 1
    return keep_largest_component(OccupancyGrid(blocked))
