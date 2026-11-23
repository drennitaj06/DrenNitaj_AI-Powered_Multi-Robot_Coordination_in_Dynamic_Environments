"""Read-only snapshots handed to coordinators, controllers, renderers and (later) RL wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell, add
from robocoord.world.entities import RobotMode, TaskStatus


class MapView:
    """Read-only facade over the (mutable) occupancy grid."""

    __slots__ = ("_grid",)

    def __init__(self, grid: OccupancyGrid) -> None:
        self._grid = grid

    @property
    def width(self) -> int:
        return self._grid.width

    @property
    def height(self) -> int:
        return self._grid.height

    @property
    def version(self) -> int:
        return self._grid.version

    @property
    def blocked(self) -> npt.NDArray[np.bool_]:
        return self._grid.blocked

    def in_bounds(self, c: Cell) -> bool:
        return self._grid.in_bounds(c)

    def is_free(self, c: Cell) -> bool:
        return self._grid.is_free(c)

    def rows(self) -> list[str]:
        return self._grid.to_rows()


@dataclass(frozen=True, slots=True)
class RobotView:
    id: int
    position: Cell
    mode: RobotMode
    task: int | None
    goal: Cell | None


@dataclass(frozen=True, slots=True)
class ObstacleView:
    id: int
    position: Cell
    velocity: Cell


@dataclass(frozen=True, slots=True)
class TaskView:
    id: int
    pickup: Cell
    dropoff: Cell
    priority: int
    release: int
    status: TaskStatus
    robot: int | None
    available: bool


@dataclass(frozen=True, slots=True)
class WorldView:
    """Snapshot of the state at time ``t`` (before the robots act).

    ``masks[i][a]`` is True if action ``a`` is allowed for robot ``i`` (static feasibility, ADR-10).
    ``suggested`` holds the planner's suggested actions, or ``None`` when no planner is attached.
    Obstacle positions at ``t + 1`` are deliberately *not* included (model §2).

    Optional extras (Issue #5, ``Engine(view_extras=...)``; ``None`` unless requested, and only on
    the view a controller acts on):

    * ``reference_paths[i]`` — robot ``i``'s remaining reference path ``c_ℓ, …`` (``c_ℓ`` is its
      own cell), truncated to the engine's ``path_limit`` moves, or ``None`` without a valid plan;
    * ``goal_distance[i]`` — the capped static distance ``D̄(x_i, g_i)`` (model §1), or ``None``
      when the robot is idle.
    """

    t: int
    max_steps: int
    map: MapView
    robots: tuple[RobotView, ...]
    obstacles: tuple[ObstacleView, ...]
    tasks: tuple[TaskView, ...]
    masks: tuple[tuple[bool, ...], ...]
    suggested: tuple[Action, ...] | None
    reference_paths: tuple[tuple[Cell, ...] | None, ...] | None = None
    goal_distance: tuple[int | None, ...] | None = None

    def robot_cells(self) -> dict[Cell, int]:
        return {r.position: r.id for r in self.robots}

    def obstacle_cells(self) -> dict[Cell, int]:
        return {o.position: o.id for o in self.obstacles}

    def allowed(self, robot: int) -> list[Action]:
        return [a for a in Action if self.masks[robot][a]]


def action_mask(grid: OccupancyGrid, position: Cell) -> tuple[bool, ...]:
    """Static feasibility mask (always on): exclude out-of-bounds and static-obstacle targets."""
    return tuple(grid.is_free(add(position, a.delta)) for a in Action)
