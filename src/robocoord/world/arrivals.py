"""Task arrival processes (model §4; FR-ENV-4).

Scheduled tasks are created from the scenario directly; the processes here generate *additional*
tasks during the episode. Each draws only from the ``tasks`` random stream it is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.world.entities import TaskRequest

#: Maximum attempts to draw a valid (distinct, reachable) pickup/drop-off pair.
MAX_DRAW_ATTEMPTS = 100


class ArrivalProcess(Protocol):
    name: str

    def releases(
        self, t: int, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> list[TaskRequest]: ...

    def has_future_releases(self, t: int) -> bool:
        """True if tasks may still be generated at some step ``>= t``."""
        ...


class NoArrivals:
    name = "none"

    def releases(
        self, t: int, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> list[TaskRequest]:
        return []

    def has_future_releases(self, t: int) -> bool:
        return False


class _RandomArrivals:
    """Shared logic: count of arrivals per step × random valid task."""

    name = "random"

    def __init__(self, rate: float, priorities: Sequence[float], until: int) -> None:
        if rate < 0:
            raise ValueError("rate must be >= 0")
        probs = np.asarray(priorities, dtype=float)
        if (
            probs.ndim != 1
            or probs.size == 0
            or (probs < 0).any()
            or not np.isclose(probs.sum(), 1)
        ):
            raise ValueError("priorities must be a non-empty probability vector summing to 1")
        if until < 0:
            raise ValueError("until must be >= 0")
        self.rate = rate
        self.priorities = probs
        self.until = until
        self.failed_draws = 0

    def _count(self, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def releases(
        self, t: int, grid: OccupancyGrid, oracle: DistanceOracle, rng: np.random.Generator
    ) -> list[TaskRequest]:
        if t > self.until:
            return []
        out: list[TaskRequest] = []
        free = grid.free_cells()
        for _ in range(self._count(rng)):
            req = draw_task(t, free, oracle, self.priorities, rng)
            if req is None:
                self.failed_draws += 1
            else:
                out.append(req)
        return out

    def has_future_releases(self, t: int) -> bool:
        return t <= self.until and self.rate > 0


class BernoulliArrivals(_RandomArrivals):
    """At most one arrival per step, with probability ``rate`` (geometric inter-arrival times)."""

    name = "bernoulli"

    def __init__(self, rate: float, priorities: Sequence[float], until: int) -> None:
        if rate > 1:
            raise ValueError("Bernoulli rate must be <= 1")
        super().__init__(rate, priorities, until)

    def _count(self, rng: np.random.Generator) -> int:
        return int(rng.random() < self.rate)


class PoissonArrivals(_RandomArrivals):
    """``Poisson(rate)`` arrivals per step (homogeneous)."""

    name = "poisson"

    def _count(self, rng: np.random.Generator) -> int:
        return int(rng.poisson(self.rate))


def draw_task(
    t: int,
    free: Sequence[tuple[int, int]],
    oracle: DistanceOracle,
    priorities: np.ndarray,
    rng: np.random.Generator,
) -> TaskRequest | None:
    """Draw a task with distinct, mutually reachable pickup and drop-off cells."""
    if len(free) < 2:
        return None
    for _ in range(MAX_DRAW_ATTEMPTS):
        i, j = rng.choice(len(free), size=2, replace=False)
        p, d = free[int(i)], free[int(j)]
        if oracle.reachable(p, d):
            level = int(rng.choice(len(priorities), p=priorities)) + 1
            return TaskRequest(pickup=p, dropoff=d, priority=level, release=t)
    return None
