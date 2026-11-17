"""Robots, tasks and dynamic obstacles; the task state machine (model §3–4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from robocoord.core.types import Cell

if TYPE_CHECKING:
    import numpy as np

    from robocoord.world.motion import MotionModel


class InvalidTransitionError(RuntimeError):
    """Raised when a task or robot is asked to perform an illegal state transition."""


class RobotMode(Enum):
    IDLE = "idle"
    TO_PICKUP = "to_pickup"
    TO_DROPOFF = "to_dropoff"


class TaskStatus(Enum):
    UNRELEASED = "unreleased"
    PENDING = "pending"
    ASSIGNED = "assigned"
    PICKED = "picked"
    DELIVERED = "delivered"


_ALLOWED: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.UNRELEASED: frozenset({TaskStatus.PENDING}),
    TaskStatus.PENDING: frozenset({TaskStatus.ASSIGNED}),
    TaskStatus.ASSIGNED: frozenset({TaskStatus.PENDING, TaskStatus.PICKED}),
    TaskStatus.PICKED: frozenset({TaskStatus.DELIVERED}),
    TaskStatus.DELIVERED: frozenset(),
}


@dataclass(slots=True)
class Task:
    """Pickup-and-delivery task ``j = (p_j, d_j, π_j, ρ_j)`` with its lifecycle state."""

    id: int
    pickup: Cell
    dropoff: Cell
    priority: int
    release: int
    status: TaskStatus = TaskStatus.UNRELEASED
    robot: int | None = None
    pickup_time: int | None = None
    completion_time: int | None = None

    def _move(self, new: TaskStatus) -> None:
        if new not in _ALLOWED[self.status]:
            raise InvalidTransitionError(
                f"task {self.id}: illegal transition {self.status.value} -> {new.value}"
            )
        self.status = new

    def do_release(self) -> None:
        self._move(TaskStatus.PENDING)

    def do_assign(self, robot: int) -> None:
        self._move(TaskStatus.ASSIGNED)
        self.robot = robot

    def do_unassign(self) -> None:
        self._move(TaskStatus.PENDING)
        self.robot = None

    def do_pick(self, t: int) -> None:
        self._move(TaskStatus.PICKED)
        self.pickup_time = t

    def do_deliver(self, t: int) -> None:
        self._move(TaskStatus.DELIVERED)
        self.completion_time = t

    @property
    def is_open(self) -> bool:
        """Released and not yet picked (pending or assigned)."""
        return self.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)

    def age(self, t: int) -> int:
        """Waiting age ``a_j(t) = t − ρ_j`` (0 before release)."""
        return max(0, t - self.release)


@dataclass(slots=True)
class Robot:
    """A robot: position, mode and assigned task (model §3)."""

    id: int
    position: Cell
    mode: RobotMode = RobotMode.IDLE
    task: int | None = None

    def goal(self, tasks: dict[int, Task]) -> Cell | None:
        """``g_i``: pickup cell, drop-off cell, or ``None`` when idle."""
        if self.task is None or self.mode is RobotMode.IDLE:
            return None
        task = tasks[self.task]
        return task.pickup if self.mode is RobotMode.TO_PICKUP else task.dropoff


@dataclass(slots=True)
class DynamicObstacle:
    """A moving obstacle with its own motion model and random generator (model §2)."""

    id: int
    position: Cell
    model: MotionModel
    rng: np.random.Generator
    previous: Cell | None = None

    @property
    def velocity(self) -> Cell:
        """``v_k(t) = z_k(t) − z_k(t−1)``; zero before the first move."""
        if self.previous is None:
            return (0, 0)
        return (self.position[0] - self.previous[0], self.position[1] - self.previous[1])


@dataclass(frozen=True, slots=True)
class TaskRequest:
    """A task to be created (from a schedule or an arrival process)."""

    pickup: Cell
    dropoff: Cell
    priority: int
    release: int


@dataclass(slots=True)
class TaskBook:
    """All tasks of an episode, with deterministic id allocation."""

    tasks: dict[int, Task] = field(default_factory=dict)

    def create(self, req: TaskRequest) -> Task:
        task = Task(len(self.tasks), req.pickup, req.dropoff, req.priority, req.release)
        self.tasks[task.id] = task
        return task

    def with_status(self, *statuses: TaskStatus) -> list[Task]:
        return [t for t in self.tasks.values() if t.status in statuses]
