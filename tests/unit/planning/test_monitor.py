"""X-12 path monitor: triggers T1–T6 fire exactly when specified, rejoin advances without
replanning, the goal is never stall-blocked, and every call is attributed (model §5.4)."""

from __future__ import annotations

import pytest

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell
from robocoord.planning import AStarPlanner, ConstraintParams, RRTStarConfig, RRTStarPlanner
from robocoord.planning.monitor import TRIGGERS, MonitorParams, PathMonitor, RobotInput

OPEN = ["." * 9] * 5  # 9 × 5 open map


def _monitor(rows: list[str] = OPEN, n: int = 1, **kw: object) -> PathMonitor:
    grid = OccupancyGrid.from_rows(rows)
    return PathMonitor(AStarPlanner(), grid, n, constraint_params=ConstraintParams(3, 1, 0.0), **kw)  # type: ignore[arg-type]


def _one(
    m: PathMonitor, t: int, pos: Cell, goal: Cell | None, obs: dict[Cell, Cell] | None = None
) -> list[str]:
    return [c.trigger for c in m.update(t, [RobotInput(0, pos, goal)], obs or {})]


def test_t1_then_following_and_rejoin_need_no_replanning() -> None:
    m = _monitor()
    assert _one(m, 0, (0, 2), (8, 2)) == ["T1"]
    path = m.states[0].path
    assert path is not None and len(path) == 9
    t = 1
    for c in path[1:4]:  # following the path
        assert _one(m, t, c, (8, 2)) == []
        t += 1
    # skipping ahead by 3 cells (= H_r) is a rejoin: index advances, no replanning
    assert _one(m, t, path[6], (8, 2)) == [] and m.states[0].index == 6
    assert m.suggested(0, path[6]) is Action.RIGHT
    assert _one(m, t + 1, path[8], (8, 2)) == []  # at the goal
    assert m.suggested(0, path[8]) is Action.STAY
    assert [c.trigger for c in m.calls] == ["T1"]


def test_t3_off_path_and_beyond_rejoin_window() -> None:
    m = _monitor()
    _one(m, 0, (0, 2), (8, 2))
    assert _one(m, 1, (0, 3), (8, 2)) == ["T3"]  # moved off the path
    path = m.states[0].path
    assert path is not None and path[0] == (0, 3)
    assert _one(m, 2, path[1], (8, 2)) == []
    # a jump of H_r + 1 cells is outside the rejoin window → off-path
    assert _one(m, 3, path[1 + 4], (8, 2)) == ["T3"]
    assert m.calls[-1].attribution == "controller"


def test_t2_obstacle_or_map_event_ahead_within_horizon() -> None:
    m = _monitor()
    _one(m, 0, (0, 2), (8, 2))
    # an obstacle on the path 3 cells ahead (within H_v = 5 and r_s = 3) → T2, detour planned
    assert _one(m, 1, (1, 2), (8, 2), {(4, 2): (0, 0)}) == ["T2"]
    path = m.states[0].path
    assert path is not None and (4, 2) not in path and m.calls[-1].attribution == "environment"
    # a static block (map event) 2 cells ahead → T2
    m2 = _monitor()
    _one(m2, 0, (0, 2), (8, 2))
    m2.grid.set_blocked((3, 2), True)
    assert _one(m2, 1, (1, 2), (8, 2)) == ["T2"]
    # beyond the validity horizon H_v = 5: no trigger
    m3 = _monitor()
    _one(m3, 0, (0, 2), (8, 2))
    m3.grid.set_blocked((7, 2), True)
    assert _one(m3, 1, (1, 2), (8, 2)) == []
    # an obstacle on the goal cell never triggers T2
    m4 = _monitor()
    _one(m4, 0, (0, 2), (3, 2))
    assert _one(m4, 1, (1, 2), (3, 2), {(3, 2): (0, 0)}) == []


def test_t4_stall_blocks_next_cell_but_never_the_goal() -> None:
    # robot 1 (idle, no goal) is parked on robot 0's next path cell: occupied, not an obstacle
    m = _monitor(n=2)
    robots = lambda: [RobotInput(0, (0, 2), (8, 2)), RobotInput(1, (1, 2), None)]  # noqa: E731
    assert [c.trigger for c in m.update(0, robots(), {})] == ["T1"]
    assert m.states[0].path is not None and m.states[0].path[1] == (1, 2)
    fired = [[c.trigger for c in m.update(t, robots(), {})] for t in range(1, 5)]
    assert fired == [[], [], [], ["T4"]]  # exactly after T_stall = 4 steps
    assert m.calls[-1].attribution == "interaction"
    assert m.states[0].blocks == {(1, 2): 4 + 4}
    path = m.states[0].path
    assert path is not None and (1, 2) not in path  # detour
    # the block expires after T_stall steps
    m.update(8, robots(), {})
    assert m.states[0].blocks == {}
    # the goal is never stall-blocked: a robot parked on the goal makes the robot wait
    g = _monitor(n=2)
    stay = lambda: [RobotInput(0, (0, 2), (1, 2)), RobotInput(1, (1, 2), None)]  # noqa: E731
    g.update(0, stay(), {})
    assert all(not g.update(t, stay(), {}) for t in range(1, 12))
    assert g.states[0].blocks == {}


def test_t5_retry_after_failure_and_cooldown() -> None:
    m = _monitor(["....@...."] * 5)
    assert _one(m, 0, (0, 2), (8, 2)) == ["T1"]
    assert m.calls[-1].status == "unreachable" and m.states[0].path is None
    assert m.suggested(0, (0, 2)) is Action.STAY
    assert [_one(m, t, (0, 2), (8, 2)) for t in (1, 2)] == [[], []]
    assert _one(m, 3, (0, 2), (8, 2)) == ["T5"]  # T_cd = 3 after the failed call
    assert m.calls[-1].attribution == "retry"
    m.grid.set_blocked((4, 0), False)  # a map event opens the wall
    assert _one(m, 6, (0, 2), (8, 2)) == ["T5"] and m.states[0].path is not None


def test_t6_degraded_plan_retried_after_cooldown() -> None:
    m = _monitor(["....@....", ".........", "....@...."])
    obstacle = {(4, 1): (0, 0)}  # sits on the only gap (2 cells ahead after one step)
    assert _one(m, 0, (2, 1), (8, 1), obstacle) == ["T1"]
    st = m.states[0]
    assert st.degraded and st.path is not None and (4, 1) in st.path
    assert _one(m, 1, (2, 1), (8, 1), obstacle) == []  # degraded: no T2 re-fire
    assert _one(m, 2, (2, 1), (8, 1), obstacle) == []
    assert _one(m, 3, (2, 1), (8, 1), obstacle) == ["T6"]
    assert _one(m, 4, (2, 1), (8, 1), {}) == []  # cooldown restarts at the T6 call
    assert _one(m, 6, (2, 1), (8, 1), {}) == ["T6"] and not m.states[0].degraded


def test_one_call_per_step_attribution_and_reset_on_idle() -> None:
    m = _monitor()
    _one(m, 0, (0, 2), (8, 2))
    # goal change and deviation at the same step → one call, attributed to T1
    assert _one(m, 1, (0, 3), (8, 0)) == ["T1"]
    assert _one(m, 2, (0, 3), None) == [] and m.states[0].path is None
    assert set(TRIGGERS) == {"T1", "T2", "T3", "T4", "T5", "T6"}
    with pytest.raises(ConfigError):
        MonitorParams(cooldown=0)


def test_monitor_with_rrtstar_and_legs() -> None:
    grid = OccupancyGrid.from_rows(OPEN)
    m = PathMonitor(
        RRTStarPlanner(RRTStarConfig(max_iterations=300)),
        grid,
        1,
        seed=5,
        static_distance=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
    )
    _one(m, 0, (0, 0), (5, 0))
    path = m.states[0].path
    assert path is not None
    for t, c in enumerate(path[1:], start=1):
        assert _one(m, t, c, (5, 0)) == []
    _one(m, len(path), path[-1], None)  # delivered: leg closed
    leg = m.legs[0]
    assert (leg.moves, leg.reference_length, leg.static_distance) == (len(path) - 1,) * 2 + (5,)


def test_t5_retries_after_budget_exhausted() -> None:
    """RRT* with a zero budget: the goal is reachable (BFS guard) but no path is found. The robot
    has no valid plan, stays, and retries with T5 every T_cd steps (model §5.3 step 5)."""
    grid = OccupancyGrid.from_rows(OPEN)
    m = PathMonitor(RRTStarPlanner(RRTStarConfig(max_iterations=0)), grid, 1)
    assert _one(m, 0, (0, 2), (8, 2)) == ["T1"]
    assert m.calls[-1].status == "budget_exhausted" and m.states[0].path is None
    assert m.suggested(0, (0, 2)) is Action.STAY
    fired = [(t, _one(m, t, (0, 2), (8, 2))) for t in range(1, 10)]
    assert [t for t, f in fired if f] == [3, 6, 9]
    assert all(f in ([], ["T5"]) for _, f in fired)
    assert {c.status for c in m.calls} == {"budget_exhausted"}
    assert all(c.attribution == "retry" for c in m.calls[1:])


DEGRADED_MAP = ["....@....", ".........", "....@...."]


def _stalled_on_degraded(cooldown: int) -> list[tuple[int, list[str]]]:
    """Robot follows a degraded plan to (3, 1); a stationary obstacle keeps the next path cell
    (4, 1), the only gap, occupied."""
    m = _monitor(DEGRADED_MAP, params=MonitorParams(cooldown=cooldown))
    obstacle = {(4, 1): (0, 0)}
    out = [(0, _one(m, 0, (2, 1), (8, 1), obstacle))]
    assert m.states[0].degraded and m.states[0].path is not None
    out += [(t, _one(m, t, (3, 1), (8, 1), obstacle)) for t in range(1, 10)]
    return out


def test_degraded_plan_and_stall_interaction() -> None:
    """With the defaults (T_cd = 3 < T_stall = 4) the T6 retry of a degraded plan comes first and
    each replan resets the stall counter, so T4 never fires while the plan stays degraded."""
    fired = [(t, f) for t, f in _stalled_on_degraded(cooldown=3) if f]
    assert fired == [(0, ["T1"]), (3, ["T6"]), (6, ["T6"]), (9, ["T6"])]
    # with a cooldown longer than the stall threshold, T4 fires first (after 4 stalled steps)
    fired = [(t, f) for t, f in _stalled_on_degraded(cooldown=6) if f]
    assert fired[:2] == [(0, ["T1"]), (4, ["T4"])]
