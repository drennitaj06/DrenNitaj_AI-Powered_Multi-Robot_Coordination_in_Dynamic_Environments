"""Deadlock resolver units (algorithms §5.5; Issue #4 stage 5B): cycle finding, passing cells,
detection thresholds, yielder selection, state machine, attempt bound, map-change recovery.

The resolver is driven with a stub monitor (reference paths given explicitly) and hand-chosen
positions, so every expected outcome follows from the definitions in the module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.planning import make_planner
from robocoord.planning.base import PlanStats
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import (
    DeadlockResolver,
    DetourParams,
    DetourState,
    find_cycles,
    passing_cell,
)
from robocoord.planning.monitor import MonitorParams, PlanCall

# ---------------------------------------------------------------- pure helpers


def test_find_cycles() -> None:
    assert find_cycles({}) == []
    assert find_cycles({0: 1, 1: 0}) == [(0, 1)]
    assert find_cycles({3: 2, 2: 3, 0: 1}) == [(2, 3)]  # chain 0 → 1 is not a cycle
    assert find_cycles({0: 1, 1: 2, 2: 3, 3: 0, 5: 4, 4: 5, 6: 0}) == [(0, 1, 2, 3), (4, 5)]
    assert find_cycles({2: 0, 0: 1, 1: 2}) == [(0, 1, 2)]  # rotated to the smallest member


def test_passing_cell_nearest_tie_forbidden_blocked_radius() -> None:
    grid = OccupancyGrid.empty(5, 5)
    # all four neighbours at distance 1: smallest (x, y) wins
    assert passing_cell(grid, (2, 2), set(), set()) == (1, 2)
    assert passing_cell(grid, (2, 2), set(), {(1, 2)}) == (2, 1)
    # blocked cells are not entered; forbidden cells may be crossed but not chosen
    line = OccupancyGrid.empty(6, 1)
    assert passing_cell(line, (0, 0), {(1, 0)}, set()) is None
    assert passing_cell(line, (0, 0), set(), {(1, 0), (2, 0)}) == (3, 0)
    assert passing_cell(line, (0, 0), set(), {(1, 0), (2, 0)}, radius=2) is None
    assert passing_cell(line, (0, 0), set(), {(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)}) is None


def test_params_validation() -> None:
    with pytest.raises(ConfigError):
        DetourParams(detect_steps=0)
    with pytest.raises(ConfigError):
        DetourParams(hold_extra=-1)
    with pytest.raises(ConfigError):
        DetourParams(search_radius=0)
    with pytest.raises(ConfigError):
        DetourParams(alpha=-0.1)
    with pytest.raises(ConfigError):  # detection must precede T4
        AllocationCoordinator(
            make_planner("astar"),
            AuctionAllocator(),
            monitor_params=MonitorParams(stall_threshold=2),
            deadlock=DetourParams(detect_steps=2),
        )


# ---------------------------------------------------------------- resolver with a stub monitor


@dataclass
class Stub:
    """Reference paths per robot; ``next_cell`` is the cell after the robot's position."""

    paths: dict[int, tuple[Cell, ...]] = field(default_factory=dict)

    def reference_path(self, robot: int) -> tuple[Cell, ...] | None:
        return self.paths.get(robot)

    def next_cell(self, robot: int, position: Cell) -> Cell | None:
        p = self.paths.get(robot)
        if not p or p[0] != position or len(p) < 2:
            return None
        return p[1]


class Map:
    def __init__(self, grid: OccupancyGrid) -> None:
        self.grid = grid

    def is_free(self, c: Cell) -> bool:
        return self.grid.is_free(c)

    @property
    def version(self) -> int:
        return self.grid.version


def corridor(lanes: int) -> Map:
    return Map(OccupancyGrid.empty(10, lanes))


def head_on(pos: list[Cell]) -> Stub:
    """Robot 0 heads right to x = 9, robot 1 heads left to x = 0, on lane y = 0."""
    (x0, _), (x1, _) = pos
    return Stub(
        {0: tuple((x, 0) for x in range(x0, 10)), 1: tuple((x, 0) for x in range(x1, -1, -1))}
    )


def observe(
    res: DeadlockResolver,
    t: int,
    m: Map,
    pos: list[Cell],
    goals: dict[int, Cell | None],
    prio: dict[int, float],
    mon: Stub,
    obstacles: dict[Cell, Cell] | None = None,
    calls: list[PlanCall] | None = None,
) -> list[str]:
    res.overrides(t, goals)
    res.observe(t, m, pos, goals, prio, obstacles or {}, mon, calls or [])  # type: ignore[arg-type]
    return [e.kind for e in res.step_events]


GOALS: dict[int, Cell | None] = {0: (9, 0), 1: (0, 0)}


def test_persistent_detection_threshold_and_yielder_by_priority() -> None:
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver(DetourParams(detect_steps=3))
    mon = head_on(pos)
    prio = {0: 1.0, 1: 2.0}
    assert observe(res, 0, m, pos, GOALS, prio, mon) == []  # count 1: temporary
    assert observe(res, 1, m, pos, GOALS, prio, mon) == []  # count 2: temporary
    assert observe(res, 2, m, pos, GOALS, prio, mon) == ["detected_persistent", "detours_started"]
    d = res.detours[0]  # lower priority yields
    assert d.waypoint == (4, 1) and d.state is DetourState.PENDING and d.others == (1,)
    assert d.avoid == frozenset({(5, 0)})


def test_equal_priority_higher_id_yields() -> None:
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    mon = head_on(pos)
    for t in range(2):
        observe(res, t, m, pos, GOALS, {0: 1.0, 1: 1.0}, mon)
    assert list(res.detours) == [1] and res.detours[1].waypoint == (5, 1)


def test_no_passing_place_is_reported_once_and_cleared_by_map_change() -> None:
    m, pos = corridor(1), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    mon = head_on(pos)
    prio = {0: 1.0, 1: 2.0}
    observe(res, 0, m, pos, GOALS, prio, mon)
    assert observe(res, 1, m, pos, GOALS, prio, mon) == ["detected_persistent", "no_passing_place"]
    assert observe(res, 2, m, pos, GOALS, prio, mon) == []  # not re-reported
    assert res.summary()["unresolved_at_end"] == 1 and not res.detours
    m.grid.set_blocked((9, 0), True)  # any map change: re-evaluate
    assert observe(res, 3, m, pos, GOALS, prio, mon) == [
        "cleared_on_map_change",
        "detected_persistent",
        "no_passing_place",
    ]


def test_fallback_to_next_candidate() -> None:
    """Robot 0 (lowest priority) is boxed in; robot 1 has a side cell and yields instead."""
    grid = OccupancyGrid.empty(10, 2)
    for x in range(0, 5):
        grid.set_blocked((x, 1), True)  # no second lane on robot 0's side
    m, pos = Map(grid), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    mon = head_on(pos)
    for t in range(2):
        observe(res, t, m, pos, GOALS, {0: 1.0, 1: 2.0}, mon)
    assert list(res.detours) == [1] and res.detours[1].waypoint == (5, 1)


def test_passing_cell_excludes_predicted_obstacle_and_reserved_cells() -> None:
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    mon = head_on(pos)
    obstacles = {(3, 1): (1, 0)}  # predicted at (4, 1), the natural passing cell
    for t in range(2):
        observe(res, t, m, pos, GOALS, {0: 1.0, 1: 2.0}, mon, obstacles)
    assert res.detours[0].waypoint not in {(4, 1), (3, 1)}


def test_recurrent_detection() -> None:
    """The same pair meets, separates and meets again: on the 3rd occurrence within the window
    it is a deadlock at once (repeated mutual blocking), without waiting detect_steps."""
    m = corridor(2)
    res = DeadlockResolver(DetourParams(detect_steps=3, recur_count=3, recur_window=20))
    prio = {0: 1.0, 1: 2.0}
    apart = [(1, 0), (8, 0)]
    meet = [(4, 0), (5, 0)]
    events = []
    for t, pos in enumerate([meet, apart, meet, apart, meet]):
        events.append(observe(res, t, m, pos, GOALS, prio, head_on(pos)))
    assert events[:4] == [[], [], [], []]
    assert events[4] == ["detected_recurrent", "detours_started"]


def advance(res: DeadlockResolver, t: int, goals: dict[int, Cell | None]) -> None:
    res.overrides(t, goals)


def test_release_horizon_and_rejoin() -> None:
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    prio = {0: 1.0, 1: 2.0}
    for t in range(2):
        observe(res, t, m, pos, GOALS, prio, head_on(pos))
    over = res.overrides(2, GOALS)  # PENDING → TO_WAYPOINT
    assert over[0].goal == (4, 1) and over[0].detour and over[0].avoid == {(5, 0)}
    assert over[1].hold_stall and not over[1].detour
    # robot 0 stepped aside; robot 1 still on robot 0's route: not released
    pos = [(4, 1), (5, 0)]
    mon = Stub({0: ((4, 1),), 1: tuple((x, 0) for x in range(5, -1, -1))})
    res.observe(2, m, pos, GOALS, prio, {}, mon, [])  # type: ignore[arg-type]
    assert res.detours[0].state is DetourState.AT_WAYPOINT
    # robot 1 passed x = 4: off the route (x ≥ 4) and heading left: released
    pos = [(4, 1), (3, 0)]
    mon = Stub({0: ((4, 1),), 1: tuple((x, 0) for x in range(3, -1, -1))})
    res.overrides(3, GOALS)
    res.observe(3, m, pos, GOALS, prio, {}, mon, [])  # type: ignore[arg-type]
    assert [e.kind for e in res.step_events] == ["released"] and not res.detours


@pytest.mark.parametrize("reason", ["blocked", "winner_path", "unreachable", "no_progress"])
def test_abort_reasons_then_redetection(reason: str) -> None:
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    prio = {0: 1.0, 1: 2.0}
    mon = head_on(pos)
    for t in range(2):
        observe(res, t, m, pos, GOALS, prio, mon)
    calls: list[PlanCall] = []
    if reason == "blocked":
        m.grid.set_blocked((4, 1), True)
    elif reason == "winner_path":
        mon = Stub({**mon.paths, 1: ((5, 0), (4, 1), (4, 0), (3, 0))})
    elif reason == "unreachable":
        calls = [PlanCall(2, 0, "T7", "unreachable", False, -1, PlanStats())]
    steps = 5 if reason == "no_progress" else 1  # abort at t − last_progress ≥ 4, i.e. t = 6
    kinds: list[str] = []
    for t in range(2, 2 + steps):
        kinds += observe(res, t, m, pos, GOALS, prio, mon, calls=calls)
    assert "aborted" in kinds


def test_attempt_bound_then_unresolved() -> None:
    """Detours that never resolve (the yielder never moves): at most 1 + max_retries starts per
    (member set, goals), then ``unresolved_retry_limit``."""
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver(DetourParams(max_retries=1, abort_steps=1))
    mon = head_on(pos)
    kinds: list[str] = []
    for t in range(12):
        kinds += observe(res, t, m, pos, GOALS, {0: 1.0, 1: 2.0}, mon)
    assert kinds.count("detours_started") == 2
    assert kinds.count("unresolved_retry_limit") == 1
    new_goals: dict[int, Cell | None] = {0: (8, 0), 1: (0, 0)}  # new task: fresh budget
    more = observe(res, 12, m, pos, new_goals, {0: 1.0, 1: 2.0}, mon)
    assert "detours_started" in more


def test_cancel_on_task_goal_change_and_timeout() -> None:
    m, pos = corridor(2), [(4, 0), (5, 0)]
    res = DeadlockResolver()
    prio = {0: 1.0, 1: 2.0}
    for t in range(2):
        observe(res, t, m, pos, GOALS, prio, head_on(pos))
    res.overrides(2, {0: (7, 0), 1: (0, 0)})  # pre-emption gave robot 0 another task goal
    assert [e.kind for e in res.step_events] == ["cancelled"] and not res.detours
    # timeout: the winner never leaves the route
    res = DeadlockResolver(DetourParams(hold_extra=0))
    for t in range(2):
        observe(res, t, m, pos, GOALS, prio, head_on(pos))
    # hold = 2·|P_w| + hold_extra with |P_w| = 5 moves at detection: arrival t = 2, timeout t = 12
    stuck = Stub({0: ((4, 1),), 1: ((5, 0), (4, 0), (3, 0))})
    kinds: list[str] = []
    for t in range(2, 14):
        res.overrides(t, GOALS)
        res.observe(t, m, [(4, 1), (5, 0)], GOALS, prio, {}, stuck, [])  # type: ignore[arg-type]
        kinds += [e.kind for e in res.step_events]
    assert kinds == ["released_timeout"]


def test_two_cycles_reserve_distinct_cells() -> None:
    m = Map(OccupancyGrid.empty(10, 3))
    pos = [(4, 0), (5, 0), (4, 2), (5, 2)]
    mon = Stub(
        {
            0: tuple((x, 0) for x in range(4, 10)),
            1: tuple((x, 0) for x in range(5, -1, -1)),
            2: tuple((x, 2) for x in range(4, 10)),
            3: tuple((x, 2) for x in range(5, -1, -1)),
        }
    )
    goals: dict[int, Cell | None] = {0: (9, 0), 1: (0, 0), 2: (9, 2), 3: (0, 2)}
    res = DeadlockResolver()
    for t in range(2):
        observe(res, t, m, pos, goals, {0: 1.0, 1: 2.0, 2: 1.0, 3: 2.0}, mon)
    cells = {d.waypoint for d in res.detours.values()}
    assert len(res.detours) == 2 and len(cells) == 2  # (4, 1) used once
