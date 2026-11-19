"""Allocation policy (algorithms §4.3; model §8; FR-TASK-5): triggers, at most one auction per
step, eligibility, application order, pre-emption, retry. Uses hand-built views and a fake
``AssignmentAPI`` that enforces the engine's state-machine rules (assign only idle robots and
unheld tasks; unassign only to_pickup robots)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from robocoord.allocation import AllocationProblem
from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.base import AllocationResult, SolveStats
from robocoord.allocation.policy import TRIGGER_KINDS, AllocationPolicy
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.sim.view import MapView, RobotView, TaskView, WorldView
from robocoord.world.entities import RobotMode, TaskStatus

IDLE, PICK, DROP = RobotMode.IDLE, RobotMode.TO_PICKUP, RobotMode.TO_DROPOFF


@dataclass
class World:
    """Minimal mutable world: robots (position, mode, task) and tasks, on a corridor grid."""

    grid: OccupancyGrid = field(default_factory=lambda: OccupancyGrid.empty(40, 1))
    robots: dict[int, list] = field(default_factory=dict)  # type: ignore[type-arg]
    tasks: dict[int, list] = field(default_factory=dict)  # type: ignore[type-arg]
    t: int = 0

    def __post_init__(self) -> None:
        self.oracle = DistanceOracle(self.grid)
        self.streams = None
        self.log: list[tuple[str, int, int | None]] = []

    def add_robot(self, rid: int, x: int, mode: RobotMode = IDLE, task: int | None = None) -> None:
        self.robots[rid] = [(x, 0), mode, task]
        if task is not None:
            self.tasks[task][4] = TaskStatus.ASSIGNED if mode is PICK else TaskStatus.PICKED
            self.tasks[task][5] = rid

    def add_task(self, tid: int, px: int, dx: int, level: int = 1, release: int = 0) -> None:
        status = TaskStatus.PENDING if release <= self.t else TaskStatus.UNRELEASED
        self.tasks[tid] = [(px, 0), (dx, 0), level, release, status, None]

    def available(self, tid: int) -> bool:
        p, d, _, _, status, _ = self.tasks[tid]
        return (
            status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)
            and self.grid.is_free(p)
            and self.grid.is_free(d)
            and self.oracle.reachable(p, d)
        )

    def view(self) -> WorldView:
        robots = []
        for rid, (pos, mode, task) in sorted(self.robots.items()):
            goal: Cell | None = None
            if task is not None:
                goal = self.tasks[task][0] if mode is PICK else self.tasks[task][1]
            robots.append(RobotView(rid, pos, mode, task, goal))
        tasks = tuple(
            TaskView(tid, p, d, lvl, rel, st, r, self.available(tid))
            for tid, (p, d, lvl, rel, st, r) in sorted(self.tasks.items())
        )
        return WorldView(self.t, 1000, MapView(self.grid), tuple(robots), (), tasks, (), None)

    # ---- AssignmentAPI (engine rules)
    def assign_task(self, robot: int, task: int) -> None:
        assert self.robots[robot][1] is IDLE, f"robot {robot} not idle"
        assert self.available(task) and self.tasks[task][4] is TaskStatus.PENDING, task
        self.robots[robot][1:] = [PICK, task]
        self.tasks[task][4:] = [TaskStatus.ASSIGNED, robot]
        self.log.append(("assign", robot, task))

    def unassign_task(self, robot: int) -> None:
        assert self.robots[robot][1] is PICK, f"robot {robot} has no pre-pickup task"
        task = self.robots[robot][2]
        self.robots[robot][1:] = [IDLE, None]
        self.tasks[task][4:] = [TaskStatus.PENDING, None]
        self.log.append(("unassign", robot, task))


def step(policy: AllocationPolicy, w: World):  # type: ignore[no-untyped-def]
    return policy.step(w.view(), w, w.oracle.distance)  # type: ignore[arg-type]


@pytest.fixture(params=["auction", "sequential"])
def policy(request: pytest.FixtureRequest) -> AllocationPolicy:
    alloc = AuctionAllocator() if request.param == "auction" else SequentialAllocator()
    return AllocationPolicy(alloc)


# ---------------------------------------------------------------- triggers


def test_triggers_and_at_most_one_auction_per_step(policy: AllocationPolicy) -> None:
    w = World()
    w.add_task(10, 5, 6)
    w.add_task(11, 30, 31, release=3)
    w.add_robot(0, 0)
    w.add_robot(1, 20)
    policy.reset(w.view())
    state, ev = step(policy, w)  # t = 0: task 10 released
    assert ev is not None and ev.triggers == ("released",) and ev.assigned == ((0, 10),)
    assert state[0].goal == (5, 0) and state[0].mode is PICK and state[1].goal is None
    for t in (1, 2):  # nothing happened: no auction
        w.t = t
        assert step(policy, w)[1] is None
    # t = 3: a release, a delivery (robot 0 idle again) and a map event in the same step
    w.t = 3
    w.tasks[11][4] = TaskStatus.PENDING
    w.tasks[10][4:] = [TaskStatus.DELIVERED, None]
    w.robots[0][1:] = [IDLE, None]
    w.grid.set_blocked((39, 0), True)
    _, ev = step(policy, w)
    assert ev is not None and ev.triggers == ("released", "delivered", "idle", "map_event")
    assert ev.assigned == ((1, 11),)  # robot 1 is closer (|20 − 30| vs |0 − 30|)
    assert [e.t for e in policy.events] == [0, 3]  # one event per triggering step
    summary = policy.summary()
    assert summary["auctions"] == 2 and summary["assignments"] == 2
    assert set(summary["triggers"]) == set(TRIGGER_KINDS)  # type: ignore[arg-type]


def test_release_on_unreachable_triggers_next_step(policy: AllocationPolicy) -> None:
    w = World()
    w.add_task(10, 5, 6)
    w.add_robot(0, 0)
    policy.reset(w.view())
    step(policy, w)
    w.t = 1
    w.unassign_task(0)  # what the coordinator does after an ``unreachable`` plan
    policy.notify_release(0)
    _, ev = step(policy, w)
    assert ev is not None and ev.triggers == ("unreachable",)  # not also ``idle``


def test_retry_after_timeout() -> None:
    class Stub:
        name = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def allocate(self, problem: AllocationProblem) -> AllocationResult:
            self.calls += 1
            if self.calls == 1:
                return AllocationResult((), 0.0, SolveStats("stub", problem.size, timed_out=True))
            return AuctionAllocator().allocate(problem)

    stub = Stub()
    policy = AllocationPolicy(stub)
    w = World()
    w.add_task(10, 5, 6)
    w.add_robot(0, 0)
    policy.reset(w.view())
    _, ev = step(policy, w)
    assert ev is not None and ev.stats.timed_out and ev.assigned == () and w.log == []
    w.t = 1
    _, ev = step(policy, w)
    assert ev is not None and ev.triggers == ("retry",) and ev.assigned == ((0, 10),)
    assert policy.summary()["timeouts"] == 1


def test_skipped_without_eligible_robots(policy: AllocationPolicy) -> None:
    w = World()
    w.add_task(10, 5, 6)
    w.add_task(11, 8, 9, release=1)
    w.add_robot(0, 0, DROP, 10)
    policy.reset(w.view())
    w.t = 1
    w.tasks[11][4] = TaskStatus.PENDING
    _, ev = step(policy, w)
    assert ev is None and policy.skipped == 1 and policy.summary()["auctions"] == 0


# ---------------------------------------------------------------- eligibility and application


def test_eligibility_carried_tasks_never_preempted(policy: AllocationPolicy) -> None:
    """A to_dropoff robot is not in R_f; an urgent task goes to an eligible robot or waits."""
    w = World()
    w.add_task(10, 5, 6)
    w.add_task(11, 4, 3, level=3)
    w.add_robot(0, 4, DROP, 10)  # carrying task 10, standing on the urgent pickup
    w.add_robot(1, 30)
    policy.reset(w.view())
    _, ev = step(policy, w)
    assert ev is not None and ev.eligible_robots == 1 and ev.available_tasks == 1
    assert ev.assigned == ((1, 11),) and w.robots[0][1:] == [DROP, 10]


def test_unavailable_task_is_excluded_and_its_holder_released(policy: AllocationPolicy) -> None:
    w = World()
    w.add_task(10, 5, 6)
    w.add_task(11, 15, 16)
    w.add_robot(0, 0)
    policy.reset(w.view())
    step(policy, w)
    assert w.robots[0][1:] == [PICK, 10]
    w.t = 1
    # drop-off of task 10 blocked: 10 unavailable. Task 11 stays available (15 → 16) but the
    # corridor is cut, so the pair (0, 11) is infeasible.
    w.grid.set_blocked((6, 0), True)
    _, ev = step(policy, w)
    assert ev is not None and ev.triggers == ("map_event",) and ev.available_tasks == 1
    assert ev.unassigned == ((0, 10),) and ev.lost == ((0, 10),)
    assert w.tasks[10][4] is TaskStatus.PENDING and w.robots[0][1:] == [IDLE, None]


def test_task_moves_between_robots_in_one_event(policy: AllocationPolicy) -> None:
    """Robot 0 holds task 10; robot 1 becomes idle much closer to it. One event unassigns robot 0
    and assigns robot 1 (the fake API asserts the order), and robot 0 is left without a task
    because max cardinality is 1 here."""
    w = World()
    w.add_task(10, 25, 26)
    w.add_task(20, 2, 3)
    w.add_robot(0, 0)
    w.add_robot(1, 24, DROP, 20)
    w.tasks[20][4:] = [TaskStatus.PICKED, 1]
    policy.reset(w.view())
    step(policy, w)
    assert w.robots[0][1:] == [PICK, 10]
    w.t = 1
    w.robots[1] = [(24, 0), IDLE, None]  # robot 1 delivered task 20
    w.tasks[20][4:] = [TaskStatus.DELIVERED, None]
    _, ev = step(policy, w)
    assert ev is not None and ev.unassigned == ((0, 10),) and ev.assigned == ((1, 10),)
    assert ev.lost == ((0, 10),) and ev.switched == ()
    assert [x[0] for x in w.log[-2:]] == ["unassign", "assign"]
    assert policy.summary()["tasks_reassigned"] == 1


@pytest.mark.parametrize("level", [1, 3])
def test_urgent_task_preempts_only_beyond_hysteresis(policy: AllocationPolicy, level: int) -> None:
    """Robot at x = 10 heading to task 10 (x = 30). A new task released at x = 8:
    level 3 → gain = κ(4 − 1) + (21 − 3) > h → switch; level 1 → gain = 18 > h → switch too
    (it is much closer). Then a same-level task 2 cells closer than the current one (gain = 2 = h)
    does not cause a switch."""
    w = World()
    w.add_task(10, 30, 31)
    w.add_robot(0, 10)
    policy.reset(w.view())
    step(policy, w)
    w.t = 1
    w.add_task(11, 8, 7, level=level)
    _, ev = step(policy, w)
    assert ev is not None and ev.switched == ((0, 10, 11),)
    assert w.tasks[10][4] is TaskStatus.PENDING and w.robots[0][1:] == [PICK, 11]
    # robot now holds 11 (c = 2 + 1 = 3); a same-level task with c = 1 gives gain 2 = h
    w.t = 2
    w.tasks[10][4] = TaskStatus.DELIVERED  # remove it from the picture
    w.add_task(12, 10, 11, level=level)  # c = 0 + 1 = 1
    _, ev = step(policy, w)
    assert ev is not None and ev.switched == () and w.robots[0][1:] == [PICK, 11]


def test_deterministic(policy: AllocationPolicy) -> None:
    def run() -> list[tuple[str, int, int | None]]:
        rng = np.random.default_rng(7)
        w = World()
        for tid in range(8):
            w.add_task(
                tid,
                int(rng.integers(0, 39)),
                int(rng.integers(0, 39)),
                int(rng.integers(1, 4)),
                tid,
            )
        for rid in range(3):
            w.add_robot(rid, int(rng.integers(0, 39)))
        policy.reset(w.view())
        for t in range(8):
            w.t = t
            for tid in w.tasks:
                if w.tasks[tid][3] == t and w.tasks[tid][4] is TaskStatus.UNRELEASED:
                    w.tasks[tid][4] = TaskStatus.PENDING
            step(policy, w)
        return w.log

    assert run() == run()
