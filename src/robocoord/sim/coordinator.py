"""Coordinator interface: the strategic/tactical layer plugged into the engine (architecture §2).

In Issue #2 only trivial coordinators exist. The auction allocator (Issue #4) and the planners with
their path monitor (Issue #3) will be composed behind this same interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from robocoord.core.distance import DistanceOracle
from robocoord.core.rng import SeedStreams
from robocoord.core.types import Action, Cell
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode, TaskStatus


class AssignmentAPI(Protocol):
    """What the engine offers a coordinator: state-machine-checked task operations and the shared
    services the approved design needs (architecture §5.4–5.5).

    * ``oracle`` — the episode's cached BFS distances (auction benefits, planner reachability).
    * ``streams`` — the episode's named random streams (planners: ``streams.generator("planner")``).
    """

    oracle: DistanceOracle
    streams: SeedStreams

    def assign_task(self, robot: int, task: int) -> None: ...

    def unassign_task(self, robot: int) -> None: ...


class Coordinator(Protocol):
    name: str

    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        """Called by ``Engine.reset()``; coordinators fetch their random generators here."""
        ...

    def update(self, view: WorldView, api: AssignmentAPI) -> None:
        """Pipeline step 3: (re)allocate tasks and (re)plan. Called once per step."""
        ...

    def suggested_actions(self, view: WorldView) -> Sequence[Action] | None:
        """The planner's suggested action per robot (``None`` = no planner attached)."""
        ...


@runtime_checkable
class PlanReporting(Protocol):
    """Optional extension of a coordinator that plans (Issue #3): the engine records plan
    adherence (model §9.3), adds deterministic planning fields to trace step records and the
    planning statistics to the episode summary."""

    def plan_valid(self) -> tuple[bool, ...]: ...

    def trace_fields(self) -> dict[str, Any]: ...

    def episode_stats(self) -> dict[str, Any]: ...

    def finish(self, view: WorldView) -> None:
        """Called once when the episode ends, with the final state."""
        ...


@runtime_checkable
class PathReporting(Protocol):
    """Optional extension of a planning coordinator (Issue #5): the current reference path of
    every robot, read by the engine only when ``view_extras`` asks for paths."""

    def reference_paths(self, limit: int) -> tuple[tuple[Cell, ...] | None, ...]:
        """Per robot: the remaining path ``c_ℓ, …, c_{ℓ+limit}`` or ``None`` without a plan."""
        ...


class NullCoordinator:
    """Assigns nothing and suggests nothing: tasks are never picked up (no allocator yet)."""

    name = "none"

    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        return None

    def update(self, view: WorldView, api: AssignmentAPI) -> None:
        return None

    def suggested_actions(self, view: WorldView) -> Sequence[Action] | None:
        return None


class ScriptedCoordinator:
    """Demo coordinator: each robot takes the tasks listed for it, in order (no optimisation).

    A robot that is idle takes its next listed task as soon as that task is released, still
    unassigned and *available* (reachable on the current map, model §4). If the next task is not
    yet available the robot waits for it; it never skips ahead in its list.
    """

    name = "scripted"

    def __init__(self, assignments: Sequence[Sequence[int]]) -> None:
        self._plan = [list(lst) for lst in assignments]
        self._next = [0] * len(self._plan)

    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        if len(self._plan) != len(view.robots):
            raise ValueError("one assignment list per robot is required")
        self._next = [0] * len(self._plan)

    def update(self, view: WorldView, api: AssignmentAPI) -> None:
        takeable = {t.id for t in view.tasks if t.status is TaskStatus.PENDING and t.available}
        for robot in view.robots:
            if robot.mode is not RobotMode.IDLE:
                continue
            queue = self._plan[robot.id]
            k = self._next[robot.id]
            if k < len(queue) and queue[k] in takeable:
                api.assign_task(robot.id, queue[k])
                takeable.discard(queue[k])
                self._next[robot.id] = k + 1

    def suggested_actions(self, view: WorldView) -> Sequence[Action] | None:
        return None
