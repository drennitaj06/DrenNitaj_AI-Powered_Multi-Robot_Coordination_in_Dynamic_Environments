"""Planner interface: constraints, queries, results, statistics (model §5; architecture §5.4).

Both required planners (A* and RRT*) implement :class:`Planner` and are always called through the
shared two-stage procedure :func:`robocoord.planning.procedure.plan_with_procedure`, so they cannot
diverge in their goal or dynamic-obstacle policy.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import cached_property
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell

PlanStatus = Literal["ok", "unreachable", "budget_exhausted", "invalid_query"]
STATUSES: tuple[PlanStatus, ...] = ("ok", "unreachable", "budget_exhausted", "invalid_query")


@dataclass(frozen=True)
class PlanConstraints:
    """The constraint snapshot of one query (model §5.2), built by ``build_constraints``.

    * ``static`` — the occupancy grid ``S(t)``. It must not be mutated while a query runs (the
      engine plans between map events, so this holds).
    * ``dynamic_blocked`` — ``B_i``: sensed and predicted obstacle cells (plus the monitor's
      temporary stall block list), never containing the query's start or goal.
    * ``proximity`` — ``Q_i``: cells within ``r_infl`` (Manhattan) of a sensed obstacle.
    * ``lambda_prox`` — proximity penalty; density ``ω = 1 + λ·1[cell ∈ Q_i]``.
    """

    static: OccupancyGrid
    dynamic_blocked: frozenset[Cell] = frozenset()
    proximity: frozenset[Cell] = frozenset()
    lambda_prox: float = 0.0

    @property
    def width(self) -> int:
        return self.static.width

    @property
    def height(self) -> int:
        return self.static.height

    def density(self, c: Cell) -> float:
        """``ω_i`` of cell ``c`` (also the A* cost of entering ``c``)."""
        return 1.0 + self.lambda_prox if c in self.proximity else 1.0

    def is_free(self, c: Cell) -> bool:
        """Free for this query: in bounds, not in ``S(t)`` and not in ``B_i``."""
        return self.static.is_free(c) and c not in self.dynamic_blocked

    def without_dynamic(self) -> PlanConstraints:
        """Stage-2 constraints (model §5.3): ``S(t)`` only, keeping the density ``ω_i``."""
        return replace(self, dynamic_blocked=frozenset())

    # ---------------------------------------------------------------- flat views (cached)
    @cached_property
    def free_flat(self) -> list[bool]:
        """Row-major (``y*W + x``) freeness w.r.t. ``S(t) ∪ B_i``. Built once per query, O(V)."""
        free: list[bool] = (~self.static.blocked).ravel().tolist()
        w = self.width
        for x, y in self.dynamic_blocked:
            if 0 <= x < w and 0 <= y < self.height:
                free[y * w + x] = False
        return free

    @cached_property
    def free_array(self) -> npt.NDArray[np.bool_]:
        """``[y, x]`` freeness w.r.t. ``S(t) ∪ B_i`` (for BFS guards)."""
        arr = np.array(self.free_flat, dtype=bool).reshape(self.height, self.width)
        arr.flags.writeable = False
        return arr

    @cached_property
    def penalised_flat(self) -> frozenset[int]:
        """Flat indices of ``Q_i`` cells that are in bounds (empty if ``λ = 0``)."""
        if self.lambda_prox == 0.0:
            return frozenset()
        w, h = self.width, self.height
        return frozenset(y * w + x for x, y in self.proximity if 0 <= x < w and 0 <= y < h)

    def free_area(self) -> int:
        """``μ(X_free)`` in cells: the number of free cells w.r.t. ``S(t) ∪ B_i``."""
        return int(sum(self.free_flat))


@dataclass(frozen=True)
class PlanQuery:
    """One planning query: start and goal cells on one constraint snapshot."""

    start: Cell
    goal: Cell
    constraints: PlanConstraints
    rng: np.random.Generator | None = None


@dataclass(frozen=True)
class PlanStats:
    """Work counters of one query (hardware independent) plus wall time.

    A* fills ``expansions``, ``pushes`` and ``peak_open``. RRT* fills the iteration counters of
    ``04`` §3.6 (definitions in ``planning/kdtree.py``):

    * ``nn_visited`` — points whose distance nearest-neighbour queries evaluated;
    * ``nn_nodes`` — k-d tree nodes nearest queries examined, including pruned and internal nodes
      (0 for the brute-force index);
    * ``near_k`` (Σ k), ``near_kprime`` (Σ k′);
    * ``range_nodes`` — k-d tree nodes radius queries expanded (box intersects the query square);
      ``range_nodes_tested`` — nodes they examined, including pruned ones;
    * ``propagation`` (Σ s);
    * ``rebuild_points`` — points placed into newly built static trees (per insertion the new slot
      size, inserted point included); ``rebuild_work`` — points processed by those builds
      (Σ subtree sizes of all built nodes);
    * ``segment_checks`` and ``segment_cells``.

    The index counters include the insertion of the root, which happens before the first
    iteration. ``guard_bfs`` counts reachability BFS runs of the procedure, ``stages`` the number
    of planner stages run.
    """

    planner: str = ""
    expansions: int = 0
    pushes: int = 0
    peak_open: int = 0
    iterations: int = 0
    tree_size: int = 0
    nn_queries: int = 0
    nn_visited: int = 0
    nn_nodes: int = 0
    radius_queries: int = 0
    near_k: int = 0
    near_kprime: int = 0
    range_nodes: int = 0
    range_nodes_tested: int = 0
    rewires: int = 0
    propagation: int = 0
    rebuild_points: int = 0
    rebuild_work: int = 0
    segment_checks: int = 0
    segment_cells: int = 0
    guard_bfs: int = 0
    stages: int = 1
    time_s: float = 0.0
    timed_out: bool = False
    first_solution_iteration: int = -1
    best_cost_trace: tuple[float, ...] | None = None
    iteration_trace: Mapping[str, Any] | None = None

    def merged(self, later: PlanStats) -> PlanStats:
        """Counters summed over two stages; traces and flags taken from ``later``."""
        out: dict[str, Any] = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(later, f.name)
            if f.name == "peak_open":
                out[f.name] = max(a, b)
            elif isinstance(a, bool) or f.name in (
                "planner",
                "best_cost_trace",
                "iteration_trace",
                "first_solution_iteration",
                "tree_size",
            ):
                out[f.name] = b if f.name != "timed_out" else (a or b)
            elif isinstance(a, int | float):
                out[f.name] = a + b
            else:  # pragma: no cover - defensive
                out[f.name] = b
        return PlanStats(**out)

    def counters(self) -> dict[str, float]:
        """Scalar counters as a flat dict (traces omitted), e.g. for run records."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("planner", "best_cost_trace", "iteration_trace")
        }


@dataclass(frozen=True)
class PlanResult:
    """Outcome of one query.

    ``path`` is a 4-connected cell sequence from start to goal (``None`` unless ``status == "ok"``).
    ``cost`` is the planner's own objective: the lattice cost ``Σ κ_i`` for A*, the weighted
    polyline length ``∫ ω_i ds`` for RRT* (``inf`` on failure). ``waypoints`` is the RRT* polyline
    (cell coordinates), ``None`` for A*. ``degraded`` marks a stage-2 (static fallback) result.
    """

    path: tuple[Cell, ...] | None
    cost: float
    status: PlanStatus
    degraded: bool = False
    stats: PlanStats = PlanStats()
    waypoints: tuple[tuple[float, float], ...] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def failure(status: PlanStatus, stats: PlanStats) -> PlanResult:
    return PlanResult(None, float("inf"), status, False, stats)


class Planner(Protocol):
    """A single-stage planner. It plans on exactly the constraints it is given and returns
    ``ok``, ``unreachable`` (it proved no path exists) or ``budget_exhausted`` (it gave up).

    ``decides_reachability`` is True for complete planners (A*). For the others the procedure runs
    a BFS guard first (Lemma 5.1), so their budget is only spent on solvable queries.
    """

    name: str
    decides_reachability: bool

    def plan(self, query: PlanQuery) -> PlanResult: ...


def lattice_cost(path: tuple[Cell, ...] | list[Cell], constraints: PlanConstraints) -> float:
    """``Σ_{ℓ≥1} κ_i(c_ℓ)``: the A* objective evaluated on any cell path."""
    return float(sum(constraints.density(c) for c in path[1:]))


def path_is_valid(
    path: tuple[Cell, ...] | list[Cell], start: Cell, goal: Cell, constraints: PlanConstraints
) -> bool:
    """Endpoints correct, consecutive cells 4-adjacent, every cell free w.r.t. ``S(t) ∪ B_i``."""
    if not path or path[0] != start or path[-1] != goal:
        return False
    for a, b in itertools.pairwise(path):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return all(constraints.is_free(c) for c in path)
