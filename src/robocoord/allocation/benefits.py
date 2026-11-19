"""Benefit construction for the task-allocation assignment problem (model §4, §8; algorithms §4.1).

For eligible robots ``i`` (idle or to_pickup) and available tasks ``j`` (pending, or assigned but
not picked) at time ``t`` the model's benefit is

    a_ij = M + κ·w̃_j(t) − c_ij + h·1[τ_i = j],     c_ij = D(x_i, p_j) + D(p_j, d_j),

with the aged weight ``w̃_j(t) = w(π_j)·(1 + α·(t − ρ_j))``, ``w(π) = 2^(π−1)``, and pairs with
``c_ij = ∞`` infeasible. Allocators work on an **integer, square** version of it:

* **Scaling (s).** Distances and ``h`` are integers, so ``s·c_ij`` and ``s·h`` are exact. The only
  rounded quantity is the priority value ``P_j = round(s·κ·w̃_j)``, so every benefit differs from
  ``s·a_ij`` by at most ``1/2``, i.e. by at most ``1/(2s)`` in original units, and a matching of
  ``q`` pairs by at most ``q/(2s)`` (model §8).
* **Offset (M).** The model's offset ``M = 1 + k(B + c_max)`` with ``B = max_j κw̃_j + h`` is
  applied to the **scaled integer components**: ``M_int = 1 + k·(B_int + C_max)`` with
  ``B_int = max_j P_j + s·h`` and ``C_max = s·c_max``. The model's argument then holds exactly in
  integers: every feasible benefit is ≥ 1, and for ``q ≤ k − 1`` any matching of ``q + 1``
  feasible pairs is worth strictly more than any matching of ``q`` pairs, for every ``κ`` and
  ``k``. Scaling the real-valued ``M`` and rounding afterwards is not guaranteed to keep that
  strict inequality in general: the accumulated rounding error can reach about ``k`` integer
  units, while the margin is ``s(1 + B)``. With the model defaults (``B ≥ κ = 10``) this would only
  fail for ``k`` above about 2 000, and with ``κ = 0`` nothing is rounded at all; the integer form
  removes the condition entirely. See `04` §4.1 implementation notes.
* **Padding.** The matrix is padded to ``k × k`` with ``k = max(|R_f|, |J_p|)``; dummy pairs and
  infeasible pairs get benefit ``0`` (algorithms §4.1 proves the optima correspond).

Guaranteed (G10, together with an exact assignment solver): the optimal padded assignment, with
zero-benefit pairs dropped, is a **maximum-cardinality** feasible matching that, among those,
maximises ``Σ (P_j − s·c_ij + s·h·1[τ_i = j])``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from robocoord.core.errors import ConfigError
from robocoord.core.types import Cell

IntMatrix = npt.NDArray[np.int64]
FloatMatrix = npt.NDArray[np.float64]

#: Largest absolute value allowed for any total of ``k`` integer benefits (int64 safety margin).
_INT_LIMIT = 2**62


@dataclass(frozen=True, slots=True)
class BenefitConfig:
    """Priority scale ``κ``, hysteresis ``h``, aging rate ``α``, integer scale ``s`` (model §11:
    10, 2, 0.01, 100). ``h`` must be an integer so that ``s·h`` is exact."""

    kappa: float = 10.0
    hysteresis: int = 2
    alpha: float = 0.01
    scale: int = 100

    def __post_init__(self) -> None:
        if not (self.kappa >= 0 and math.isfinite(self.kappa)):
            raise ConfigError("allocation.kappa", "must be a finite number >= 0")
        if not isinstance(self.hysteresis, int) or self.hysteresis < 0:
            raise ConfigError("allocation.hysteresis", "must be an integer >= 0")
        if not (self.alpha >= 0 and math.isfinite(self.alpha)):
            raise ConfigError("allocation.alpha", "must be a finite number >= 0")
        if not isinstance(self.scale, int) or self.scale < 1:
            raise ConfigError("allocation.scale", "must be an integer >= 1")


@dataclass(frozen=True, slots=True)
class RobotCandidate:
    """An eligible robot: its cell and its current pre-pickup task (``None`` if idle)."""

    id: int
    position: Cell
    task: int | None = None


@dataclass(frozen=True, slots=True)
class TaskCandidate:
    """An available task: pickup, drop-off, priority level ``π ≥ 1`` and release time ``ρ``."""

    id: int
    pickup: Cell
    dropoff: Cell
    priority: int
    release: int


def priority_weight(level: int) -> float:
    """``w(π) = 2^(π−1)`` (model §4)."""
    if level < 1:
        raise ValueError("priority levels start at 1")
    return float(2 ** (level - 1))


def effective_weight(level: int, release: int, t: int, alpha: float) -> float:
    """Aged weight ``w̃_j(t) = w(π_j)(1 + α·a_j(t))``, waiting age ``a_j(t) = max(t − ρ_j, 0)``."""
    return priority_weight(level) * (1.0 + alpha * max(t - release, 0))


class _TaskLike(Protocol):
    @property
    def id(self) -> int: ...
    @property
    def priority(self) -> int: ...
    @property
    def release(self) -> int: ...


class _RobotLike(Protocol):
    @property
    def id(self) -> int: ...
    @property
    def task(self) -> int | None: ...


def robot_priorities(
    robots: Sequence[_RobotLike], tasks: Sequence[_TaskLike], t: int, alpha: float
) -> dict[int, float]:
    """Effective priority ``π̃_i`` of each robot (algorithms §5.3): the aged weight of its
    assigned or carried task, ``0`` when idle (below every task, weights are ≥ 1). Shared by the
    baseline controller and the deadlock resolver. Time O(N + M)."""
    by_id = {tk.id: tk for tk in tasks}
    out: dict[int, float] = {}
    for r in robots:
        tk = by_id.get(r.task) if r.task is not None else None
        out[r.id] = 0.0 if tk is None else effective_weight(tk.priority, tk.release, t, alpha)
    return out


@dataclass(frozen=True)
class AllocationProblem:
    """One allocation instance, in original units and as the integer square matrix allocators use.

    * ``cost[i, j]`` — ``c_ij`` (``inf`` if infeasible); ``priority_value[j]`` — ``κ·w̃_j(t)``.
    * ``value[i, j]`` — ``κ·w̃_j − c_ij + h·1[τ_i = j]`` (``-inf`` if infeasible): the benefit
      without the offset, the quantity optimised among maximum-cardinality matchings.
    * ``matrix`` — padded ``k × k`` int64 benefits (``0`` = dummy or infeasible); rows ``<
      len(robots)`` and columns ``< len(tasks)`` are real, the rest are padding.
    * ``offset`` — ``M_int``; ``core[i, j] = matrix[i, j] − offset`` for feasible real pairs.
    """

    robots: tuple[int, ...]
    tasks: tuple[int, ...]
    cost: FloatMatrix
    priority_value: FloatMatrix
    value: FloatMatrix
    feasible: npt.NDArray[np.bool_]
    matrix: IntMatrix
    offset: int
    config: BenefitConfig
    t: int
    current: Mapping[int, int] = field(default_factory=dict)  # robot → current pre-pickup task

    @property
    def size(self) -> int:
        """Square size ``k = max(|R_f|, |J_p|)``."""
        return int(self.matrix.shape[0])

    def pairs(self, assignment: Sequence[int]) -> list[tuple[int, int]]:
        """Convert a square assignment (``assignment[row] = column``) into ``(robot, task)`` pairs,
        dropping dummy rows/columns and infeasible (zero-benefit) pairs."""
        out = []
        for row, col in enumerate(assignment):
            if row < len(self.robots) and 0 <= col < len(self.tasks) and self.feasible[row, col]:
                out.append((self.robots[row], self.tasks[col]))
        return out

    def matching_value(self, pairs: Sequence[tuple[int, int]]) -> float:
        """``Σ value[i, j]`` over the pairs, in original units."""
        r = {rid: i for i, rid in enumerate(self.robots)}
        c = {tid: j for j, tid in enumerate(self.tasks)}
        return float(sum(self.value[r[i], c[j]] for i, j in pairs))


DistanceFn = Callable[[Cell, Cell], int | None]


def build_problem(
    robots: Sequence[RobotCandidate],
    tasks: Sequence[TaskCandidate],
    t: int,
    distance: DistanceFn,
    config: BenefitConfig | None = None,
) -> AllocationProblem:
    """Build the benefit matrices for one allocation event (model §8).

    ``distance(a, b)`` returns the static shortest-path distance ``D_t(a, b)`` or ``None`` if
    ``b`` is unreachable from ``a`` (e.g. :meth:`DistanceOracle.distance`).

    Time: O(n·m) distance look-ups plus the BFS fields they need (cached by the oracle:
    one field per distinct pickup cell, O(V) each on a cache miss); O(k²) memory. ``distance``
    must be symmetric (a static shortest-path distance on the undirected grid is).
    """
    cfg = config or BenefitConfig()
    n, m = len(robots), len(tasks)
    if len({r.id for r in robots}) != n or len({j.id for j in tasks}) != m:
        raise ValueError("robot and task ids must be unique")
    for task in tasks:
        if task.priority < 1:
            raise ValueError(f"task {task.id}: priority must be >= 1")
    k = max(n, m)
    s = cfg.scale

    # travel cost c_ij = D(x_i, p_j) + D(p_j, d_j); infinite if either leg is unreachable.
    # D is symmetric on the undirected 4-connected grid (model §1), so D(p_j, d_j) is read as
    # D(d_j, p_j): it uses the field to p_j that the D(x_i, p_j) terms need anyway — one BFS
    # field per task instead of two on a cold cache (NFR-PERF-1; identical values). A legacy
    # oracle (the before/after benchmark's "before") keeps the original argument order.
    legacy = bool(getattr(getattr(distance, "__self__", None), "legacy", False))
    leg2 = np.array(
        [
            _dist_or_inf(distance, j.pickup, j.dropoff)
            if legacy
            else _dist_or_inf(distance, j.dropoff, j.pickup)
            for j in tasks
        ],
        dtype=np.float64,
    )
    cost = np.full((n, m), math.inf)
    for i, r in enumerate(robots):
        for j, task in enumerate(tasks):
            if math.isfinite(leg2[j]):
                cost[i, j] = _dist_or_inf(distance, r.position, task.pickup) + leg2[j]
    feasible = np.isfinite(cost)

    prio = np.array(
        [cfg.kappa * effective_weight(j.priority, j.release, t, cfg.alpha) for j in tasks],
        dtype=np.float64,
    )
    task_col = {task.id: j for j, task in enumerate(tasks)}
    current = {r.id: r.task for r in robots if r.task is not None and r.task in task_col}
    hyst = np.zeros((n, m), dtype=np.float64)
    for i, r in enumerate(robots):
        if r.id in current:
            hyst[i, task_col[current[r.id]]] = cfg.hysteresis
    value = np.where(feasible, prio[None, :] - np.where(feasible, cost, 0.0) + hyst, -math.inf)

    # integer components: P_j rounded; s·c_ij and s·h exact
    p_int = np.array([round(s * float(p)) for p in prio], dtype=np.int64)
    matrix = np.zeros((k, k), dtype=np.int64)
    offset = 0
    if feasible.any():
        cost_int = np.where(feasible, cost, 0.0).astype(np.int64) * s
        hyst_int = hyst.astype(np.int64) * s
        b_int = int(p_int.max()) + s * cfg.hysteresis
        c_max = int(cost_int[feasible].max())
        offset = 1 + k * (b_int + c_max)
        core = p_int[None, :] - cost_int + hyst_int
        bound = k * (offset + int(np.abs(core).max()))
        if bound >= _INT_LIMIT:
            raise OverflowError(
                f"integer benefits too large for int64 (k={k}, offset={offset}); "
                "reduce the integer scale s"
            )
        matrix[:n, :m] = np.where(feasible, offset + core, 0)
    return AllocationProblem(
        robots=tuple(r.id for r in robots),
        tasks=tuple(task.id for task in tasks),
        cost=cost,
        priority_value=prio,
        value=value,
        feasible=feasible,
        matrix=matrix,
        offset=offset,
        config=cfg,
        t=t,
        current=current,
    )


def _dist_or_inf(distance: DistanceFn, a: Cell, b: Cell) -> float:
    d = distance(a, b)
    return math.inf if d is None else float(d)
