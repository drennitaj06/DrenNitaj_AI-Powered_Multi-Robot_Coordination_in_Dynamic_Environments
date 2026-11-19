"""Path-monitor support for the deadlock detour (Issue #4 stage 5B): T7 attribution, legs not
split by a detour, cancellation to a new goal (T1), stall suspension (T4 held), extra avoided
cells, and the ``next_cell`` query. The defaults of ``RobotInput`` give the Issue #3 behaviour,
which the existing ``test_monitor.py`` keeps checking."""

from __future__ import annotations

from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.planning import make_planner
from robocoord.planning.monitor import PathMonitor, RobotInput


def monitor(grid: OccupancyGrid, n: int = 1) -> PathMonitor:
    oracle = DistanceOracle(grid)
    return PathMonitor(
        make_planner("astar"),
        grid,
        n,
        static_reachable=oracle.reachable,
        static_distance=oracle.distance,
    )


def step(mon: PathMonitor, t: int, pos: Cell, goal: Cell | None, **kw: object) -> list[str]:
    return [c.trigger for c in mon.update(t, [RobotInput(0, pos, goal, **kw)], {})]  # type: ignore[arg-type]


def test_t7_into_and_out_of_a_detour_keeps_one_leg() -> None:
    grid = OccupancyGrid.empty(8, 2)
    mon = monitor(grid)
    goal, s = (7, 0), (2, 1)
    assert step(mon, 0, (0, 0), goal) == ["T1"]
    assert step(mon, 1, (1, 0), goal) == []
    assert step(mon, 2, (2, 0), s, detour=True) == ["T7"]  # into the passing cell
    assert mon.states[0].detour and mon.states[0].task_goal == goal
    assert mon.next_cell(0, (2, 0)) == (2, 1)
    assert step(mon, 3, (2, 1), s, detour=True) == []  # at the passing cell: no leg closed
    assert mon.legs == [] and mon.next_cell(0, (2, 1)) is None
    assert step(mon, 4, (2, 1), goal) == ["T7"]  # back to the task goal
    assert not mon.states[0].detour
    pos = (2, 1)
    for t in range(5, 15):
        path = mon.reference_path(0)
        assert path is not None
        if pos == goal:
            break
        pos = path[1]
        step(mon, t, pos, goal)
    mon.finish(15, [goal])
    assert len(mon.legs) == 1 and mon.legs[0].start_t == 0  # one leg, from the original T1


def test_cancelled_detour_to_a_new_goal_is_t1_and_a_new_leg() -> None:
    grid = OccupancyGrid.empty(8, 2)
    mon = monitor(grid)
    step(mon, 0, (0, 0), (7, 0))
    step(mon, 1, (1, 0), (1, 1), detour=True)
    assert step(mon, 2, (1, 1), (0, 1)) == ["T1"]  # a different goal: not a return
    assert not mon.states[0].detour and mon.states[0].leg_start == 2
    mon.finish(3, [(1, 1)])
    assert mon.legs == []  # standing on the old passing cell does not close a leg


def test_hold_stall_suspends_t4() -> None:
    grid = OccupancyGrid.empty(6, 1)
    mon = monitor(grid, 2)
    goal = (5, 0)
    blocker = RobotInput(1, (2, 0), None)

    def go(t: int, hold: bool) -> list[str]:
        calls = mon.update(t, [RobotInput(0, (1, 0), goal, hold_stall=hold), blocker], {})
        return [c.trigger for c in calls if c.robot == 0]

    assert go(0, False) == ["T1"]
    assert [go(t, True) for t in range(1, 9)] == [[]] * 8  # stalled 8 steps, held: no T4
    assert [go(t, False) for t in range(9, 13)] == [[], [], [], ["T4"]]  # counting resumes


def test_avoid_cells_shape_the_plan() -> None:
    grid = OccupancyGrid.empty(5, 3)
    mon = monitor(grid)
    calls = mon.update(0, [RobotInput(0, (0, 1), (4, 1), avoid=frozenset({(2, 1)}))], {})
    assert calls[0].status == "ok" and not calls[0].degraded
    path = mon.reference_path(0)
    assert path is not None and (2, 1) not in path
