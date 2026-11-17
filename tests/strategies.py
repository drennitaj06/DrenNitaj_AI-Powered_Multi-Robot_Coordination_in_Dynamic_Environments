"""Reusable Hypothesis strategies for grids and agent placements."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from robocoord.core.grid import OccupancyGrid


@st.composite
def grids(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 8,
    max_density: float = 0.4,
) -> OccupancyGrid:
    """Random small occupancy grids."""
    w = draw(st.integers(min_size, max_size))
    h = draw(st.integers(min_size, max_size))
    density = draw(st.floats(0.0, max_density))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    return OccupancyGrid(rng.random((h, w)) < density)
