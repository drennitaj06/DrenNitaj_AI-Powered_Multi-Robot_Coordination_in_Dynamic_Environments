"""Issue 6A Phase 1 (P1): repair before replanning on T3 deviations.

The repair rejoins the remaining reference path through a connector of at most ``repair_radius``
moves over free cells outside ``B_i`` and ``Q_i``; it never replaces a degraded plan, and a
repaired path must pass the T2 check. With ``repair_radius = 0`` the monitor is unchanged.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell, add
from robocoord.planning import AStarPlanner, ConstraintParams
from robocoord.planning.monitor import MonitorParams, PathMonitor, RobotInput

OPEN = ["." * 9] * 5


def _monitor(rows: list[str] = OPEN, radius: int = 3) -> PathMonitor:
    return PathMonitor(
        AStarPlanner(),
        OccupancyGrid.from_rows(rows),
        1,
        params=MonitorParams(repair_radius=radius),
        constraint_params=ConstraintParams(3, 1, 0.0),
    )


def _one(
    m: PathMonitor, t: int, pos: Cell, goal: Cell, obs: dict[Cell, Cell] | None = None
) -> list[str]:
    return [c.trigger for c in m.update(t, [RobotInput(0, pos, goal)], obs or {})]


def _connected(path: tuple[Cell, ...]) -> bool:
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in itertools.pairwise(path))


def test_params_validation_and_default_off() -> None:
    assert MonitorParams().repair_radius == 0
    with pytest.raises(ConfigError):
        MonitorParams(repair_radius=-1)


def test_one_step_off_path_is_repaired_without_a_planner_call() -> None:
    m = _monitor()
    assert _one(m, 0, (0, 2), (8, 2)) == ["T1"]
    assert _one(m, 1, (0, 3), (8, 2)) == []  # off the path by one cell: repaired
    st_ = m.states[0]
    assert st_.path is not None and st_.path[0] == (0, 3) and st_.path[-1] == (8, 2)
    assert _connected(st_.path) and len(st_.path) - 1 == 9  # one step back + 8 along the path
    assert m.repairs == 1 and m.repair_failures == 0 and m.deviations == 1
    assert [c.trigger for c in m.calls] == ["T1"]
    assert _one(m, 2, st_.path[1], (8, 2)) == []  # following the repaired path


def test_radius_zero_keeps_the_original_t3_call() -> None:
    m = _monitor(radius=0)
    _one(m, 0, (0, 2), (8, 2))
    assert _one(m, 1, (0, 3), (8, 2)) == ["T3"]
    assert m.repairs == 0 and m.repair_failures == 0


def test_too_far_from_the_path_falls_back_to_replanning() -> None:
    m = _monitor(radius=1)
    _one(m, 0, (0, 0), (8, 0))
    assert _one(m, 1, (0, 2), (8, 0)) == ["T3"]  # two cells away: repair fails, replanned
    assert m.repairs == 0 and m.repair_failures == 1
    assert m.states[0].path is not None and m.states[0].path[0] == (0, 2)


def test_connector_avoids_sensed_obstacles_and_t2_blocks_repair() -> None:
    m = _monitor()
    _one(m, 0, (0, 2), (8, 2))
    # the robot steps to (1, 3); an obstacle sits on the path at (2, 2) ahead: the repaired path
    # would cross it, so the T2 check rejects the repair and the planner is called
    assert _one(m, 1, (1, 3), (8, 2), {(2, 2): (0, 0)}) == ["T3"]
    assert m.repair_failures == 1
    path = m.states[0].path
    assert path is not None and (2, 2) not in path


def test_skipping_ahead_beyond_the_window_is_a_zero_length_repair() -> None:
    m = _monitor()
    _one(m, 0, (0, 2), (8, 2))
    path = m.states[0].path
    assert path is not None
    assert _one(m, 1, path[6], (8, 2)) == []  # beyond H_r = 3: T3, but already on the path
    assert m.repairs == 1 and m.states[0].path == path[6:]


def test_degraded_plans_are_never_repaired() -> None:
    m = _monitor()
    _one(m, 0, (0, 2), (8, 2))
    st_ = m.states[0]
    st_.degraded = True
    assert _one(m, 1, (0, 3), (8, 2)) == ["T3"]
    assert m.repairs == 0


@settings(max_examples=150, deadline=None)
@given(
    walls=st.sets(st.tuples(st.integers(0, 8), st.integers(0, 4)), max_size=10),
    moves=st.lists(st.integers(0, 4), min_size=1, max_size=25),
    radius=st.integers(1, 4),
)
def test_property_repaired_paths_are_valid(walls: set[Cell], moves: list[int], radius: int) -> None:
    """Whatever the robot does, the reference path stays 4-connected over free cells, starts at
    the robot and ends at the goal; every T3 is either repaired or replanned; deterministic."""
    from robocoord.core.types import Action

    start, goal = (0, 2), (8, 2)
    rows = [
        "".join("#" if (x, y) in walls - {start, goal} else "." for x in range(9)) for y in range(5)
    ]
    runs = []
    for _ in range(2):
        m = _monitor(rows, radius)
        pos, out = start, []
        for t, a in enumerate(moves):
            out.append(tuple(_one(m, t, pos, goal)))
            s = m.states[0]
            if s.path is not None:
                assert s.path[s.index] == pos and s.path[-1] == goal
                assert _connected(s.path) and all(m.grid.is_free(c) for c in s.path)
            nxt = add(pos, Action(a).delta)
            if m.grid.is_free(nxt):
                pos = nxt
        t3 = m.deviations
        assert m.repairs + m.repair_failures == t3
        runs.append((out, [c.trigger for c in m.calls], m.repairs))
    assert runs[0] == runs[1]
