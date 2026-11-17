from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

from robocoord.core.grid import OccupancyGrid
from robocoord.core.mapio import MapFormatError, format_movingai, parse_movingai
from robocoord.core.types import Action, action_between, add, chebyshev, manhattan
from tests.strategies import grids


def test_actions_and_deltas() -> None:
    assert Action.UP.delta == (0, 1)
    assert Action.LEFT.delta == (-1, 0)
    assert Action.from_symbol("r") is Action.RIGHT
    assert [a.symbol for a in Action] == ["S", "U", "D", "L", "R"]
    with pytest.raises(ValueError):
        Action.from_symbol("X")
    assert action_between((2, 2), (2, 3)) is Action.UP
    assert action_between((2, 2), (2, 2)) is Action.STAY
    with pytest.raises(ValueError):
        action_between((0, 0), (1, 1))
    assert add((1, 2), (3, -4)) == (4, -2)
    assert manhattan((0, 0), (3, -4)) == 7
    assert chebyshev((0, 0), (3, -4)) == 4


def test_from_rows_orientation_top_row_is_highest_y() -> None:
    g = OccupancyGrid.from_rows(["@..", "...", "..#"])
    assert (g.width, g.height) == (3, 3)
    assert g.is_blocked((0, 2))  # top-left in text
    assert g.is_blocked((2, 0))  # bottom-right in text
    assert g.is_free((1, 1))
    assert g.to_rows() == ["@..", "...", "..@"]


def test_bounds_and_neighbors_order() -> None:
    g = OccupancyGrid.from_rows(["...", ".@.", "..."])
    assert not g.in_bounds((-1, 0))
    assert g.is_blocked((3, 0))  # out of bounds counts as blocked
    assert list(g.neighbors((1, 0))) == [(0, 0), (2, 0)]  # UP is blocked (1,1)
    assert list(g.neighbors((0, 1))) == [(0, 2), (0, 0)]  # UP, DOWN; RIGHT blocked
    assert g.num_free() == 8
    assert (1, 1) not in g.free_cells()


def test_set_blocked_bumps_version_only_on_change() -> None:
    g = OccupancyGrid.empty(3, 2)
    assert g.version == 0
    assert g.set_blocked((1, 1), True)
    assert g.version == 1
    assert not g.set_blocked((1, 1), True)
    assert g.version == 1
    with pytest.raises(ValueError):
        g.set_blocked((5, 5), True)


def test_blocked_view_is_read_only() -> None:
    g = OccupancyGrid.empty(2, 2)
    with pytest.raises(ValueError):
        g.blocked[0, 0] = True


def test_invalid_rows() -> None:
    with pytest.raises(ValueError):
        OccupancyGrid.from_rows([])
    with pytest.raises(ValueError):
        OccupancyGrid.from_rows(["..", "."])
    with pytest.raises(ValueError):
        OccupancyGrid.from_rows([".x"])


@given(grids())
def test_movingai_round_trip(grid: OccupancyGrid) -> None:
    assert parse_movingai(format_movingai(grid)) == grid


def test_movingai_parse_example() -> None:
    text = "type octile\nheight 2\nwidth 3\nmap\n.T.\n@..\n"
    g = parse_movingai(text)
    assert g.is_blocked((1, 1)) and g.is_blocked((0, 0)) and g.is_free((2, 0))


@pytest.mark.parametrize(
    "text",
    [
        "type octile\nheight 2\nwidth 2\n..\n..\n",  # missing 'map'
        "type octile\nwidth 2\nmap\n..\n",  # missing height
        "type octile\nheight 2\nwidth 2\nmap\n..\n",  # too few rows
        "type octile\nheight 1\nwidth 3\nmap\n..\n",  # wrong width
    ],
)
def test_movingai_errors(text: str) -> None:
    with pytest.raises(MapFormatError):
        parse_movingai(text)


def test_grid_equality() -> None:
    a = OccupancyGrid(np.array([[0, 1]], dtype=bool))
    assert a == a.copy()
    assert a != OccupancyGrid.empty(2, 1)
