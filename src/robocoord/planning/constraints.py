"""The shared dynamic constraint builder (model §5.2; algorithms §2.2).

``build_constraints`` is the **only** place where ``Z_i``, ``Ẑ_i``, ``B_i``, ``Q_i`` and ``ω_i``
are computed; A* and RRT* receive the same :class:`PlanConstraints` object.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.planning.base import PlanConstraints


class FreeCellQuery(Protocol):
    """What :func:`sense` needs from a map: ``OccupancyGrid`` or the read-only ``MapView``."""

    def is_free(self, c: Cell) -> bool: ...


@dataclass(frozen=True, slots=True)
class ConstraintParams:
    """Sensing radius ``r_s`` (Chebyshev), inflation radius ``r_infl`` (Manhattan) and proximity
    penalty ``λ_prox``. Defaults from model §11."""

    sensing_radius: int = 3
    inflation_radius: int = 1
    lambda_prox: float = 2.0

    def __post_init__(self) -> None:
        if self.sensing_radius < 0:
            raise ConfigError("constraints.sensing_radius", "must be >= 0")
        if self.inflation_radius < 0:
            raise ConfigError("constraints.inflation_radius", "must be >= 0")
        if self.lambda_prox < 0:
            raise ConfigError("constraints.lambda_prox", "must be >= 0")


@dataclass(frozen=True, slots=True)
class SensedObstacles:
    """Result of sensing: ``Z_i`` and the one-step predictions ``Ẑ_i`` (model §5.2)."""

    current: frozenset[Cell]
    predicted: frozenset[Cell]


def sense(
    grid: FreeCellQuery,
    position: Cell,
    obstacles: Mapping[Cell, Cell],
    sensing_radius: int,
) -> SensedObstacles:
    """``Z_i`` = obstacles within Chebyshev distance ``r_s`` of ``position``; ``Ẑ_i`` = their
    constant-velocity predictions ``z + v`` that are free in-bounds cells.

    ``obstacles`` maps each obstacle's current cell to its velocity. The scan visits the
    ``(2 r_s + 1)²`` window or the obstacle list, whichever is smaller. Time: O(min(r_s², K)).
    """
    x0, y0 = position
    r = sensing_radius
    current: list[Cell] = []
    if (2 * r + 1) ** 2 <= len(obstacles):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                c = (x0 + dx, y0 + dy)
                if c in obstacles:
                    current.append(c)
    else:
        current = [z for z in obstacles if max(abs(z[0] - x0), abs(z[1] - y0)) <= r]
    predicted: set[Cell] = set()
    for z in current:
        v = obstacles[z]
        p = (z[0] + v[0], z[1] + v[1])
        if grid.is_free(p):
            predicted.add(p)
    return SensedObstacles(frozenset(current), frozenset(predicted))


def proximity_set(grid: OccupancyGrid, sensed: Iterable[Cell], inflation_radius: int) -> set[Cell]:
    """``Q_i``: in-bounds cells within Manhattan distance ``r_infl`` of a sensed obstacle.

    Empty when nothing is sensed (``min ∅ = +∞``). Time: O(|Z_i| r_infl²).
    """
    r = inflation_radius
    out: set[Cell] = set()
    for zx, zy in sensed:
        for dx in range(-r, r + 1):
            rem = r - abs(dx)
            for dy in range(-rem, rem + 1):
                c = (zx + dx, zy + dy)
                if grid.in_bounds(c):
                    out.add(c)
    return out


def build_constraints(
    grid: OccupancyGrid,
    start: Cell,
    goal: Cell,
    obstacles: Mapping[Cell, Cell] | None = None,
    params: ConstraintParams | None = None,
    extra_blocked: Iterable[Cell] = (),
) -> PlanConstraints:
    """Build the constraint snapshot of robot ``i`` for one query (model §5.2).

    * ``B_i = (Z_i ∪ Ẑ_i ∪ extra) ∖ {start, goal}``: the robot's own cell and its goal cell are
      never dynamically blocked. ``extra_blocked`` is the path monitor's temporary stall block
      list (T4), which is dynamic in nature and therefore dropped by the stage-2 fallback too.
    * ``Q_i`` from the sensed (current) obstacle cells; it may contain the goal.
    * ``λ_prox`` from ``params``.
    """
    params = params or ConstraintParams()
    sensed = sense(grid, start, obstacles or {}, params.sensing_radius)
    blocked = set(sensed.current) | set(sensed.predicted) | set(extra_blocked)
    blocked.discard(start)
    blocked.discard(goal)
    prox = proximity_set(grid, sensed.current, params.inflation_radius)
    return PlanConstraints(
        static=grid,
        dynamic_blocked=frozenset(blocked),
        proximity=frozenset(prox),
        lambda_prox=params.lambda_prox,
    )
