"""Endpoint recovery (algorithms §5.6): the cell searches, the cut-vertex rule and both state
machines, driven directly with hand-built states (no engine). Expected cells are derived by
hand from the rules in the module docstring."""

from __future__ import annotations

from collections import deque

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.planning.endpoint import (
    END_KEYS,
    STAT_KEYS,
    EndpointParams,
    EndpointRecovery,
    RobotState,
    active_endpoints,
    articulation_points,
    nearest_cell,
)
from robocoord.sim.view import TaskView
from robocoord.world.entities import RobotMode, TaskStatus
from tests.strategies import grids

IDLE, PICK, DROP = RobotMode.IDLE, RobotMode.TO_PICKUP, RobotMode.TO_DROPOFF


def corridor(n: int) -> OccupancyGrid:
    """Cells (1, 1) … (n, 1), walls around."""
    return OccupancyGrid.from_rows(["#" * (n + 2), "#" + "." * n + "#", "#" * (n + 2)])


def open_grid(w: int, h: int) -> OccupancyGrid:
    return OccupancyGrid.from_rows(["." * w] * h)


def task(i: int, p: Cell, d: Cell, status: TaskStatus) -> TaskView:
    return TaskView(i, p, d, 1, 0, status, None, True)


def robot(pos: Cell, mode: RobotMode = IDLE, task: int | None = None, goal: Cell | None = None):
    return RobotState(pos, mode, task, goal)


# ---------------------------------------------------------------- parameters and helpers


def test_params_defaults_and_validation() -> None:
    p = EndpointParams()
    assert (p.wait_steps, p.hold_steps, p.max_attempts, p.retreat_radius, p.abort_steps) == (
        4,
        8,
        3,
        8,
        4,
    )
    for name in ("wait_steps", "hold_steps", "max_attempts", "retreat_radius", "abort_steps"):
        with pytest.raises(ConfigError):
            EndpointParams(**{name: 0})


def test_active_endpoints_follow_the_task_status() -> None:
    tasks = [
        task(0, (1, 1), (2, 2), TaskStatus.PENDING),
        task(1, (3, 3), (4, 4), TaskStatus.ASSIGNED),
        task(2, (5, 5), (6, 6), TaskStatus.PICKED),  # pickup no longer needed
        task(3, (7, 7), (8, 8), TaskStatus.DELIVERED),
        task(4, (9, 9), (0, 0), TaskStatus.UNRELEASED),
    ]
    assert active_endpoints(tasks) == {(1, 1), (2, 2), (3, 3), (4, 4), (6, 6)}


def test_nearest_cell_ties_radius_and_blocking() -> None:
    g = open_grid(5, 5)
    # distance-1 cells of (2, 2) are (1,2), (3,2), (2,1), (2,3); the (x, y) minimum is (1, 2)
    assert nearest_cell(g, (2, 2), set(), lambda c: True) == (1, 2)
    assert nearest_cell(g, (2, 2), set(), lambda c: c == (4, 4), radius=3) is None
    assert nearest_cell(g, (2, 2), set(), lambda c: c == (4, 4), radius=4) == (4, 4)
    # a wall of blocked cells around the start: nothing reachable
    wall = {(1, 2), (3, 2), (2, 1), (2, 3)}
    assert nearest_cell(g, (2, 2), wall, lambda c: True) is None
    assert nearest_cell(corridor(3), (1, 1), set(), lambda c: c[0] == 3) == (3, 1)


def _components(cells: set[Cell]) -> int:
    seen: set[Cell] = set()
    count = 0
    for c in cells:
        if c in seen:
            continue
        count += 1
        queue = deque([c])
        seen.add(c)
        while queue:
            x, y = queue.popleft()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in cells and n not in seen:
                    seen.add(n)
                    queue.append(n)
    return count


@settings(max_examples=60, deadline=None)
@given(grids(max_size=7, max_density=0.45), st.data())
def test_articulation_points_match_brute_force(grid: OccupancyGrid, data: st.DataObject) -> None:
    """``c`` is a cut vertex iff removing it increases the number of components (independent
    brute force, recomputing components for every cell)."""
    free = set(grid.free_cells())
    removed = set(data.draw(st.sets(st.sampled_from(sorted(free)), max_size=3))) if free else set()
    rest = free - removed
    base = _components(rest)
    expected = {c for c in rest if _components(rest - {c}) > base}
    assert articulation_points(grid, removed) == expected


def test_articulation_points_of_a_corridor_and_a_ring() -> None:
    assert articulation_points(corridor(5), set()) == {(2, 1), (3, 1), (4, 1)}
    ring = OccupancyGrid.from_rows(["...", ".#.", "..."])
    assert articulation_points(ring, set()) == set()
    # without (0, 1) the ring is the path (0,0)–(1,0)–(2,0)–(2,1)–(2,2)–(1,2)–(0,2): its two
    # ends are leaves, every interior cell is a cut vertex
    assert articulation_points(ring, {(0, 1)}) == {(1, 0), (2, 0), (2, 1), (2, 2), (1, 2)}


# ---------------------------------------------------------------- parking


def test_parking_picks_the_nearest_safe_cell_and_completes() -> None:
    """Corridor x = 1 … 5. Robot 0 is idle on (3, 1), the drop-off of the task robot 1 (at x = 1)
    carries. Interior cells 2–4 are cut vertices, (1, 1) is occupied: the only safe cell is the
    leaf (5, 1)."""
    g = corridor(5)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    rec = EndpointRecovery()
    robots = [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1))]
    rec.observe(0, g, robots, tasks, [], [])
    assert rec.parkings[0].cell == (5, 1)
    assert rec.stats["parking_needed"] == rec.stats["parking_started"] == 1
    over = rec.overrides(1, robots, set())
    o = over[0]
    assert (o.goal, o.detour, o.hold_stall, o.trigger) == ((5, 1), True, False, "T8")
    assert o.avoid == {(1, 1)}  # robot 1 is at distance 2 (waiting for the endpoint)
    assert 1 not in over
    rec.observe(1, g, [robot((4, 1)), robots[1]], tasks, [], [])
    rec.overrides(2, robots, set())
    rec.observe(2, g, [robot((5, 1)), robots[1]], tasks, [], [])
    assert rec.stats["parking_completed"] == 1 and not rec.parkings
    assert rec.overrides(3, robots, set()) == {}


def test_parking_is_cancelled_when_the_robot_gets_a_task() -> None:
    g = corridor(5)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    rec = EndpointRecovery()
    rec.observe(0, g, [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1))], tasks, [], [])
    busy = [robot((3, 1), PICK, 1, (4, 1)), robot((1, 1), DROP, 0, (3, 1))]
    assert rec.overrides(1, busy, set()) == {}
    assert rec.stats["parking_cancelled"] == 1 and not rec.parkings
    s = rec.summary()
    assert s["parking_started"] == s["parking_completed"] + s["parking_cancelled"] + s[
        "parking_aborted"
    ] + len(rec.parkings)


def test_parking_aborts_without_progress_and_retries_later() -> None:
    g = corridor(5)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    rec = EndpointRecovery()
    robots = [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1))]
    rec.observe(0, g, robots, tasks, [], [])
    for t in range(1, 5):  # the robot never moves
        rec.overrides(t, robots, set())
        rec.observe(t, g, robots, tasks, [], [])
    assert rec.stats["parking_aborted"] == 1 and not rec.parkings
    for t in range(5, 9):  # retry after abort_steps = 4
        rec.overrides(t, robots, set())
        rec.observe(t, g, robots, tasks, [], [])
    assert rec.stats["parking_started"] == 2 and rec.stats["parking_needed"] == 1


def test_resolver_held_robot_is_not_restarted_until_released() -> None:
    """While the deadlock resolver overrides an idle robot, its parking is cancelled once and
    not restarted (no repeated start/cancel counting); after the release it parks again within
    the same need episode."""
    g = corridor(5)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    rec = EndpointRecovery()
    robots = [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1))]
    rec.observe(0, g, robots, tasks, [], [])
    for t in range(1, 6):  # held by the resolver for 5 steps
        assert 0 not in rec.overrides(t, robots, {0})
        rec.observe(t, g, robots, tasks, [], [])
    s = rec.stats
    assert (s["parking_started"], s["parking_cancelled"], s["parking_needed"]) == (1, 1, 1)
    rec.overrides(6, robots, set())  # released
    rec.observe(6, g, robots, tasks, [], [])
    assert s["parking_started"] == 2 and s["parking_needed"] == 1 and 0 in rec.parkings


def test_no_safe_parking_cell_is_reported_once() -> None:
    """Corridor x = 1 … 3; a pending task needs (1, 1) and (3, 1). Robot 0 idles on (3, 1):
    (2, 1) is a cut vertex and (1, 1) an active endpoint, so no safe cell exists at all."""
    g = corridor(3)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PENDING)]
    rec = EndpointRecovery()
    robots = [robot((3, 1))]
    for t in range(6):
        rec.overrides(t, robots, set())
        rec.observe(t, g, robots, tasks, [], [])
    assert rec.stats["no_parking_cell"] == 1 and rec.stats["parking_started"] == 0
    assert rec.robot_report(0)["no_parking_cell"]
    rec.finish(robots, tasks, [])
    assert rec.summary()["idle_on_endpoint_at_end"] == 1


def test_parking_blocked_by_a_robot_is_dynamic() -> None:
    """Corridor x = 1 … 3. Robot 1 carries the task (picked, so (1, 1) is no endpoint any more)
    and stands on (1, 1), the only safe cell: a cell exists but cannot be reached now."""
    g = corridor(3)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    rec = EndpointRecovery()
    robots = [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1))]
    for t in range(3):
        rec.observe(t, g, robots, tasks, [], [])
    assert rec.stats["parking_blocked_dynamic"] == 1 and rec.stats["no_parking_cell"] == 0
    assert rec.robot_report(0)["parking_blocked_dynamic"]


def test_parking_respects_stationary_idle_robots_and_reserved_cells() -> None:
    """Corridor x = 1 … 5 with robot 2 idle (stationary) on the leaf (5, 1): removing (5, 1)
    makes (4, 1) a leaf, but (4, 1) is then next to the stationary robot and still allowed; it is
    the nearest safe cell. With (4, 1) reserved by the resolver nothing is left."""
    g = corridor(5)
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    robots = [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1)), robot((5, 1))]
    rec = EndpointRecovery()
    rec.observe(0, g, robots, tasks, [], [])
    assert rec.parkings[0].cell == (4, 1)
    rec2 = EndpointRecovery()
    rec2.observe(0, g, robots, tasks, [], [], reserved_external={(4, 1)})
    assert not rec2.parkings and rec2.stats["parking_blocked_dynamic"] == 1


# ---------------------------------------------------------------- retreat

GOAL = (3, 2)
WAIT = (2, 2)
TASKS = [task(0, (6, 4), GOAL, TaskStatus.PICKED)]


def carrier(pos: Cell = WAIT) -> RobotState:
    return robot(pos, DROP, 0, GOAL)


def obstacle(at: Cell = GOAL) -> list[tuple[int, Cell, Cell]]:
    return [(0, at, (0, 0))]


def test_temporary_occupancy_clears_without_recovery() -> None:
    g, rec = open_grid(7, 5), EndpointRecovery()
    for t in range(3):  # 3 < W = 4 steps adjacent
        rec.observe(t, g, [carrier()], TASKS, obstacle(), [])
    rec.observe(3, g, [carrier()], TASKS, obstacle((4, 2)), [])  # the obstacle moved on
    assert rec.stats["blockages_temporary"] == 1
    assert rec.stats["blockages_persistent"] == rec.stats["retreats_started"] == 0


def test_persistent_obstacle_starts_a_retreat_that_succeeds() -> None:
    """After W = 4 steps the robot retreats. BFS level 2 from (2, 2) (the obstacle cell blocks
    the search) holds (0,2), (1,1), (1,3), (2,0), (2,4), (3,1), (3,3); (3, 1) and (3, 3) are next
    to the obstacle, so the (x, y) minimum of the rest is (0, 2)."""
    g, rec = open_grid(7, 5), EndpointRecovery()
    for t in range(4):
        rec.overrides(t, [carrier()], set())
        rec.observe(t, g, [carrier()], TASKS, obstacle(), [])
    assert rec.stats["blockages_persistent"] == rec.stats["blockages_persistent_obstacle"] == 1
    assert rec.retreats[0].cell == (0, 2) and rec.stats["retreats_started"] == 1
    o = rec.overrides(4, [carrier()], set())[0]
    assert (o.goal, o.detour, o.hold_stall, o.trigger) == ((0, 2), True, True, "T8")
    rec.observe(4, g, [carrier((1, 2))], TASKS, obstacle(), [])
    rec.overrides(5, [carrier((1, 2))], set())
    rec.observe(5, g, [carrier((0, 2))], TASKS, obstacle((2, 2)), [])  # the goal is free
    assert rec.stats["retreats_succeeded"] == 1 and not rec.retreats
    restore = rec.overrides(6, [carrier((0, 2))], set())[0]
    assert (restore.goal, restore.detour, restore.trigger) == (GOAL, False, "T8")
    rec.observe(6, g, [carrier(GOAL)], TASKS, obstacle((1, 2)), [])
    assert rec.stats["goal_reached_after_recovery"] == 1


def test_retreat_timeouts_then_exhaustion_and_unresolved_at_end() -> None:
    """The obstacle never leaves: M = 3 retreats time out after H = 8 steps each; after W more
    steps waiting the episode is exhausted (counted once) and no fourth retreat starts."""
    g, rec = open_grid(7, 5), EndpointRecovery()
    for t in range(120):
        pos = rec.retreats[0].cell if 0 in rec.retreats else WAIT
        rec.overrides(t, [carrier(pos)], set())
        rec.observe(t, g, [carrier(pos)], TASKS, obstacle(), [])
    s = rec.stats
    assert s["retreats_started"] == s["retreats_timed_out"] == 3
    assert s["attempts_exhausted"] == 1 and s["blockages_persistent"] == 1
    assert s["blockages_temporary"] == 0 and s["retreats_succeeded"] == 0
    rep = rec.robot_report(0)
    assert rep["attempts_exhausted"] and rep["retreat_attempts"] == 3
    rec.finish([carrier()], TASKS, obstacle())
    assert rec.summary()["unresolved_obstacle_at_end"] == 1


def test_no_retreat_cell_in_a_corridor_is_reported_once() -> None:
    """Corridor x = 1 … 4: robot at (2, 1), goal (3, 1) held. Cells at distance ≥ 2 from both
    would be x ≤ 0 or beyond the obstacle (unreachable): none."""
    g, rec = corridor(4), EndpointRecovery()
    tasks = [task(0, (4, 1), (3, 1), TaskStatus.PICKED)]
    me = robot((2, 1), DROP, 0, (3, 1))
    for t in range(12):
        rec.observe(t, g, [me], tasks, obstacle((3, 1)), [])
    assert rec.stats["no_retreat_cell"] == 1 and rec.stats["retreats_started"] == 0
    assert rec.robot_report(0)["no_retreat_cell"]


def test_retreat_is_cancelled_by_a_goal_change_or_the_resolver() -> None:
    g = open_grid(7, 5)
    for reason in ("goal", "resolver"):
        rec = EndpointRecovery()
        for t in range(4):
            rec.observe(t, g, [carrier()], TASKS, obstacle(), [])
        assert 0 in rec.retreats
        if reason == "goal":
            assert rec.overrides(4, [robot(WAIT, DROP, 1, (5, 4))], set()) == {}
        else:
            assert rec.overrides(4, [carrier()], {0}) == {}
        assert rec.stats["retreats_cancelled"] == 1 and not rec.retreats


def test_retreat_aborts_without_progress() -> None:
    g, rec = open_grid(7, 5), EndpointRecovery()
    for t in range(8):  # retreat starts at t = 3, no move for abort_steps = 4
        rec.overrides(t, [carrier()], set())
        rec.observe(t, g, [carrier()], TASKS, obstacle(), [])
    assert rec.stats["retreats_aborted"] == 1


def test_idle_robot_occupant_counts_as_a_robot_blockage() -> None:
    """The goal is held by an idle robot that cannot park (its only exit is the carrier)."""
    g, rec = corridor(3), EndpointRecovery()
    tasks = [task(0, (1, 1), (3, 1), TaskStatus.PICKED)]
    robots = [robot((2, 1), DROP, 0, (3, 1)), robot((3, 1))]
    for t in range(5):
        rec.observe(t, g, robots, tasks, [], [])
    assert rec.stats["blockages_persistent_robot"] == 1
    assert rec.stats["no_retreat_cell"] == 1 and rec.stats["parking_blocked_dynamic"] == 1
    rec.finish(robots, tasks, [])
    assert rec.summary()["unresolved_idle_robot_at_end"] == 1


def test_parking_steps_do_not_count_as_active_for_the_deviation_rate() -> None:
    """An idle robot driving to a parking cell (``count_active=False``) adds neither active
    robot-steps nor deviations; the same goal with ``count_active=True`` does (model §9.3)."""
    from robocoord.planning import make_planner
    from robocoord.planning.monitor import PathMonitor, RobotInput

    g = open_grid(7, 5)
    for counted in (False, True):
        mon = PathMonitor(make_planner("astar"), g, 1)
        kw = {"detour": True, "trigger": "T8", "count_active": counted}
        assert [c.trigger for c in mon.update(0, [RobotInput(0, (0, 0), (4, 0), **kw)], {})] == [
            "T8"
        ]
        mon.update(1, [RobotInput(0, (0, 1), (4, 0), **kw)], {})  # off the path: T3
        assert [c.trigger for c in mon.calls] == ["T8", "T3"]
        assert (mon.active_robot_steps, mon.deviations) == ((2, 1) if counted else (0, 0))


def test_summary_has_every_documented_key_and_events_serialise() -> None:
    rec = EndpointRecovery()
    rec.observe(
        0,
        corridor(5),
        [robot((3, 1)), robot((1, 1), DROP, 0, (3, 1))],
        [task(0, (1, 1), (3, 1), TaskStatus.PICKED)],
        [],
        [],
    )
    assert set(rec.summary()) == set(STAT_KEYS) | set(END_KEYS)
    assert [e.to_json() for e in rec.events] == [
        ["parking_needed", 0, [3, 1]],
        ["parking_started", 0, [5, 1]],
    ]
