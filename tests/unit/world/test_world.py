from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell
from robocoord.world.arrivals import BernoulliArrivals, NoArrivals, PoissonArrivals
from robocoord.world.entities import (
    InvalidTransitionError,
    Robot,
    RobotMode,
    Task,
    TaskBook,
    TaskRequest,
    TaskStatus,
)
from robocoord.world.events import MapEvent, MapEventSchedule
from robocoord.world.motion import Bounce, Patrol, RandomWalk, resolve_obstacle_moves

# ---------------------------------------------------------------- task state machine

LEGAL = {
    (TaskStatus.UNRELEASED, TaskStatus.PENDING),
    (TaskStatus.PENDING, TaskStatus.ASSIGNED),
    (TaskStatus.ASSIGNED, TaskStatus.PENDING),
    (TaskStatus.ASSIGNED, TaskStatus.PICKED),
    (TaskStatus.PICKED, TaskStatus.DELIVERED),
}


def _make(status: TaskStatus) -> Task:
    t = Task(0, (0, 0), (1, 0), 1, 0)
    t.status = status
    return t


def _apply(task: Task, target: TaskStatus) -> None:
    {
        TaskStatus.PENDING: lambda: (
            task.do_release() if task.status is TaskStatus.UNRELEASED else task.do_unassign()
        ),
        TaskStatus.ASSIGNED: lambda: task.do_assign(3),
        TaskStatus.PICKED: lambda: task.do_pick(5),
        TaskStatus.DELIVERED: lambda: task.do_deliver(9),
        TaskStatus.UNRELEASED: lambda: task._move(TaskStatus.UNRELEASED),
    }[target]()


@pytest.mark.parametrize("src", list(TaskStatus))
@pytest.mark.parametrize("dst", list(TaskStatus))
def test_task_transitions_exhaustive(src: TaskStatus, dst: TaskStatus) -> None:
    task = _make(src)
    if (src, dst) in LEGAL:
        _apply(task, dst)
        assert task.status is dst
    else:
        with pytest.raises(InvalidTransitionError):
            _apply(task, dst)
        assert task.status is src


def test_task_lifecycle_records_times_and_robot() -> None:
    task = Task(7, (0, 0), (2, 0), 2, release=4)
    assert task.age(2) == 0 and task.age(10) == 6
    task.do_release()
    assert task.is_open
    task.do_assign(1)
    assert task.robot == 1
    task.do_unassign()
    assert task.robot is None
    task.do_assign(2)
    task.do_pick(6)
    task.do_deliver(9)
    assert (task.pickup_time, task.completion_time, task.is_open) == (6, 9, False)


def test_robot_goal_and_taskbook() -> None:
    book = TaskBook()
    t0 = book.create(TaskRequest((1, 1), (3, 3), 1, 0))
    t1 = book.create(TaskRequest((2, 2), (4, 4), 2, 5))
    assert (t0.id, t1.id) == (0, 1)
    r = Robot(0, (0, 0))
    assert r.goal(book.tasks) is None
    r.task, r.mode = 0, RobotMode.TO_PICKUP
    assert r.goal(book.tasks) == (1, 1)
    r.mode = RobotMode.TO_DROPOFF
    assert r.goal(book.tasks) == (3, 3)
    assert book.with_status(TaskStatus.UNRELEASED) == [t0, t1]


# ---------------------------------------------------------------- motion models

GRID = OccupancyGrid.from_rows(
    [
        ".....",
        ".@@@.",
        ".....",
    ]
)


def test_patrol_follows_shortest_path_and_cycles() -> None:
    oracle = DistanceOracle(GRID)
    rng = np.random.default_rng(0)
    model = Patrol([(0, 0), (4, 2)])
    pos: Cell = (0, 0)
    visited = [pos]
    for _ in range(12):
        pos = model.propose(pos, GRID, oracle, rng)
        visited.append(pos)
    assert (4, 2) in visited
    assert visited[6] == (4, 2)  # shortest path length 6
    assert visited[12] == (0, 0)  # and back
    for a, b in itertools.pairwise(visited):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
        assert GRID.is_free(b)


def test_bounce_reverses_at_walls() -> None:
    oracle = DistanceOracle(GRID)
    rng = np.random.default_rng(0)
    model = Bounce(Action.RIGHT)
    pos: Cell = (3, 0)
    seq = []
    for _ in range(4):
        pos = model.propose(pos, GRID, oracle, rng)
        seq.append(pos)
    assert seq == [(4, 0), (3, 0), (2, 0), (1, 0)]
    trapped = OccupancyGrid.from_rows(["@.@"])
    assert Bounce(Action.LEFT).propose((1, 0), trapped, DistanceOracle(trapped), rng) == (1, 0)
    with pytest.raises(ValueError):
        Bounce(Action.STAY)


def test_random_walk_is_seeded() -> None:
    oracle = DistanceOracle(GRID)
    a = [RandomWalk().propose((2, 2), GRID, oracle, np.random.default_rng(5)) for _ in range(3)]
    b = [RandomWalk().propose((2, 2), GRID, oracle, np.random.default_rng(5)) for _ in range(3)]
    assert a == b


def test_obstacle_move_rule() -> None:
    grid = OccupancyGrid.empty(5, 1)
    # obstacle 0 wants the robot cell, obstacle 1 wants a cell already claimed by obstacle 2? order!
    cur = [(0, 0), (3, 0), (4, 0)]
    prop = [(1, 0), (2, 0), (3, 0)]
    out = resolve_obstacle_moves(cur, prop, grid, robot_cells={(1, 0)})
    # 0 blocked by robot; 1 moves to (2,0); 2 cannot enter (3,0) because it was occupied at t
    assert out == [(0, 0), (2, 0), (4, 0)]
    # two obstacles proposing the same free cell: the first (by index) wins
    out = resolve_obstacle_moves([(0, 0), (2, 0)], [(1, 0), (1, 0)], grid, set())
    assert out == [(1, 0), (2, 0)]
    # proposals into walls or out of bounds are rejected
    out = resolve_obstacle_moves([(0, 0)], [(-1, 0)], grid, set())
    assert out == [(0, 0)]


@given(st.integers(0, 10_000))
def test_obstacle_moves_never_overlap(seed: int) -> None:
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid(rng.random((5, 5)) < 0.2)
    free = grid.free_cells()
    k = min(len(free), 6)
    idx = rng.choice(len(free), size=k, replace=False)
    cur = [free[int(i)] for i in idx]
    robots = set(free[int(i)] for i in rng.choice(len(free), size=min(3, len(free)), replace=False))
    robots -= set(cur)
    oracle = DistanceOracle(grid)
    props = [RandomWalk().propose(c, grid, oracle, rng) for c in cur]
    out = resolve_obstacle_moves(cur, props, grid, robots)
    assert len(set(out)) == len(out)
    assert not (set(out) & robots)
    assert all(grid.is_free(c) for c in out)
    for a, b in zip(cur, out, strict=True):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1


# ---------------------------------------------------------------- arrivals


def test_arrival_processes_are_seeded_and_valid() -> None:
    oracle = DistanceOracle(GRID)
    proc = BernoulliArrivals(rate=0.5, priorities=[0.2, 0.3, 0.5], until=20)
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    a = [r for t in range(30) for r in proc.releases(t, GRID, oracle, rng_a)]
    b = [r for t in range(30) for r in proc.releases(t, GRID, oracle, rng_b)]
    assert a == b and len(a) > 0
    assert all(r.release <= 20 for r in a)
    for r in a:
        assert r.pickup != r.dropoff and GRID.is_free(r.pickup) and GRID.is_free(r.dropoff)
        assert 1 <= r.priority <= 3
    assert proc.has_future_releases(20) and not proc.has_future_releases(21)
    pois = PoissonArrivals(rate=2.0, priorities=[1.0], until=5)
    counts = [len(pois.releases(t, GRID, oracle, np.random.default_rng(t))) for t in range(200)]
    assert sum(counts[:6]) > 0 and sum(counts[6:]) == 0
    assert NoArrivals().releases(0, GRID, oracle, rng_a) == []
    assert not NoArrivals().has_future_releases(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": -0.1, "priorities": [1.0], "until": 5},
        {"rate": 0.5, "priorities": [0.5, 0.4], "until": 5},
        {"rate": 0.5, "priorities": [], "until": 5},
        {"rate": 0.5, "priorities": [1.0], "until": -1},
        {"rate": 1.5, "priorities": [1.0], "until": 5},
    ],
)
def test_arrival_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BernoulliArrivals(**kwargs)  # type: ignore[arg-type]


def test_poisson_mean_rate_statistically() -> None:
    oracle = DistanceOracle(OccupancyGrid.empty(6, 6))
    grid = OccupancyGrid.empty(6, 6)
    proc = PoissonArrivals(rate=0.7, priorities=[1.0], until=10_000)
    rng = np.random.default_rng(123)
    n = sum(len(proc.releases(t, grid, oracle, rng)) for t in range(3000))
    # mean 2100, sd ~46: a 5-sigma band
    assert abs(n - 2100) < 5 * np.sqrt(2100)


# ---------------------------------------------------------------- map events


def test_map_events_with_deferral() -> None:
    grid = OccupancyGrid.empty(3, 1)
    sched = MapEventSchedule(
        [MapEvent(2, (1, 0), True), MapEvent(0, (2, 0), True), MapEvent(5, (2, 0), False)]
    )
    assert sched.apply_due(0, grid, occupied=set()) == [MapEvent(0, (2, 0), True)]
    assert grid.is_blocked((2, 0))
    # at t=2 the cell (1,0) is occupied: block is deferred
    assert sched.apply_due(2, grid, occupied={(1, 0)}) == []
    assert sched.deferred == (MapEvent(2, (1, 0), True),)
    assert sched.apply_due(3, grid, occupied={(1, 0)}) == []
    applied = sched.apply_due(4, grid, occupied=set())
    assert applied == [MapEvent(2, (1, 0), True)] and grid.is_blocked((1, 0))
    assert sched.has_future_events()
    assert sched.apply_due(5, grid, occupied=set()) == [MapEvent(5, (2, 0), False)]
    assert grid.is_free((2, 0)) and not sched.has_future_events()
    assert MapEvent(1, (0, 0), True).describe() == "block (0, 0)"
