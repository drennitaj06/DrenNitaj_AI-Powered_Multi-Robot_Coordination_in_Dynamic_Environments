"""Dynamic-obstacle motion models and the obstacle move rule (model §2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from robocoord.core.distance import UNREACHABLE, DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell, add


class MotionModel(Protocol):
    """Proposes an obstacle's desired next cell ``z̃_k(t+1)``.

    Models may keep internal state (patrol index, bounce direction); they must be deterministic
    given their generator.
    """

    name: str

    def propose(
        self, position: Cell, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> Cell: ...


class RandomWalk:
    """Uniform choice among stay and the four moves (proposal may be infeasible)."""

    name = "random_walk"

    def propose(
        self, position: Cell, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> Cell:
        action = Action(int(rng.integers(0, len(Action))))
        return add(position, action.delta)


class Patrol:
    """Cycle through waypoints, moving one step along a shortest path to the current one."""

    name = "patrol"

    def __init__(self, waypoints: Sequence[Cell]) -> None:
        if not waypoints:
            raise ValueError("patrol needs at least one waypoint")
        self.waypoints: tuple[Cell, ...] = tuple(waypoints)
        self.index = 0

    def propose(
        self, position: Cell, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> Cell:
        if position == self.waypoints[self.index]:
            self.index = (self.index + 1) % len(self.waypoints)
        target = self.waypoints[self.index]
        if position == target:
            return position
        field = oracle.field(target)
        here = int(field[position[1], position[0]]) if grid.in_bounds(position) else UNREACHABLE
        best: Cell = position
        best_d = here if here != UNREACHABLE else np.iinfo(np.int32).max
        for nb in grid.neighbors(position):  # canonical order → deterministic tie-break
            d = int(field[nb[1], nb[0]])
            if d != UNREACHABLE and d < best_d:
                best, best_d = nb, d
        return best


class Bounce:
    """Move straight in a fixed direction; reverse when the next cell is statically blocked."""

    name = "bounce"

    def __init__(self, direction: Action) -> None:
        if direction is Action.STAY:
            raise ValueError("bounce direction must be a movement action")
        self.direction: Action = direction

    def propose(
        self, position: Cell, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> Cell:
        nxt = add(position, self.direction.delta)
        if grid.is_free(nxt):
            return nxt
        self.direction = _REVERSE[self.direction]
        nxt = add(position, self.direction.delta)
        return nxt if grid.is_free(nxt) else position


_REVERSE = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}


def resolve_obstacle_moves(
    current: Sequence[Cell],
    proposals: Sequence[Cell],
    grid: OccupancyGrid,
    robot_cells: set[Cell] | frozenset[Cell],
) -> list[Cell]:
    """Apply the obstacle move rule in fixed index order (model §2).

    A proposal is accepted only if it is in bounds and free, not occupied at time ``t`` by a
    robot or any dynamic obstacle, and not already claimed by an earlier obstacle this step.
    Otherwise the obstacle stays. Consequently obstacles never overlap, never swap and never
    enter a robot-occupied cell.

    Time: O(K) expected.
    """
    if len(current) != len(proposals):
        raise ValueError("current and proposals must have equal length")
    occupied_now = set(current) | set(robot_cells)
    claimed: set[Cell] = set()
    result: list[Cell] = []
    for here, want in zip(current, proposals, strict=True):
        nxt = want
        if nxt != here and (not grid.is_free(nxt) or nxt in occupied_now or nxt in claimed):
            nxt = here
        if nxt != here and abs(nxt[0] - here[0]) + abs(nxt[1] - here[1]) != 1:
            nxt = here  # defensive: models may only propose unit moves
        claimed.add(nxt)
        result.append(nxt)
    return result


__all__ = ["Bounce", "MotionModel", "Patrol", "RandomWalk", "resolve_obstacle_moves"]
