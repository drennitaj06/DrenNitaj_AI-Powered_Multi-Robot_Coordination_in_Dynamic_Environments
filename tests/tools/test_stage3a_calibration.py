"""Tests of the Stage 3a calibration tool's pure helpers (no replay, no stored data needed)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import stage3a_calibration as cal  # noqa: E402

from robocoord.control.supervisor import is_oscillating  # noqa: E402


def test_first_at_least() -> None:
    assert cal.first_at_least([0, 1, 2, 0, 1, 2, 3], 3) == 6
    assert cal.first_at_least([0, 1, 2], 3) is None


def test_first_osc_matches_the_supervisor_rule() -> None:
    """A robot that walks 3 cells, then shuttles between two cells: the O trigger (K = 4, 2 cells,
    3 moves) fires at the first offset with >= 4 steps without progress whose last 5 positions
    cover <= 2 cells with >= 3 moves — the supervisor's own rule."""
    hist = [(0, 0), (1, 0), (2, 0), (3, 0), (2, 0), (3, 0), (2, 0), (3, 0), (2, 0)]
    since = [0, 0, 0, 0, 1, 2, 3, 4, 5]  # progress up to offset 3
    j = cal.first_osc(hist, since, 4, 2, 3)
    assert j == 7
    assert is_oscillating(hist[j - 4 : j + 1], 4, 2, 3)
    assert is_oscillating(hist[j - 5 : j], 4, 2, 3) and since[j - 1] < 4  # earlier: progress
    assert cal.first_osc(hist, [0] * len(hist), 4, 2, 3) is None  # progressing: never


def test_grids_are_valid_supervisor_params() -> None:
    from robocoord.control.supervisor import SupervisorParams

    for k, c, m in cal.O_GRID:
        SupervisorParams(osc_window=k, osc_cells=c, osc_moves=m)
    for w in cal.W_GRID:
        SupervisorParams(stall_window=w)


def test_population_is_block_b_only() -> None:
    import pytest

    if not (ROOT / cal.A_STAR).exists():
        pytest.skip("stored results not available")
    import os

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        pop = cal.population()
    finally:
        os.chdir(cwd)
    assert len(pop) == 2700
    assert all(r["seed"] in cal.BLOCK_B for r in pop)
    assert sum(r["planner"] == "astar" for r in pop) == 2160
