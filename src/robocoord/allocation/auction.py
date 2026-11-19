"""Bertsekas' forward auction for the assignment problem, with ε-scaling (algorithms §4.1; ADR-5).

The production allocator. SciPy's Hungarian solver is only the test and experiment oracle.

**Exact integer arithmetic.** All benefits, prices and ε are int64:

* *ε-scaling mode* multiplies the benefits by ``k + 1``. An integer ε of 1 then corresponds to
  ``ε = 1/(k+1) < 1/k`` in original units, which is exactly the documented convergence condition:
  with integer benefits, a final ε-CS assignment is **optimal** (``k·ε < 1``; Guaranteed). The
  schedule follows the pseudocode: ``ε₀ = max(C'/θ, 1)``, then ``ε ← max(⌊ε/θ⌋, 1)`` with prices
  kept between phases, stopping after the phase with ``ε = 1`` (``C'`` = largest scaled benefit).
* *fixed-ε mode* runs one phase from zero prices with a given integer ε in original units. The
  result satisfies ``V ≥ OPT − k·ε`` (Guaranteed, algorithms §4.1). For a non-negative matrix
  with largest entry ``C`` the phase makes at most ``k·(⌊C/ε⌋ + 1) + 1`` bids (Guaranteed, see
  `04` §4.1 notes for the derivation).

Bids are Gauss–Seidel: one unassigned person bids at a time (FIFO queue); the bid goes to the
lowest-index best object; the price rises by ``v* − w* + ε`` where ``w*`` is the second-best value
(``w* = v*`` when ``k = 1``); the displaced owner rejoins the queue.

Time per bid: O(k) (vectorised). Without scaling O(k·C/ε) bids, i.e. O(k²·C/ε) worst case
(pseudo-polynomial); with ε-scaling O(k·A·log(kC)) (Bertsekas & Eckstein 1988, cited, not
re-derived here). Space O(k²).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robocoord.allocation.base import (
    AllocationResult,
    SolveStats,
    SquareSolution,
    result_from_solution,
)
from robocoord.allocation.benefits import AllocationProblem
from robocoord.core.errors import ConfigError

IntMatrix = npt.NDArray[np.int64]
IntVector = npt.NDArray[np.int64]
_INT_LIMIT = 2**62
_CHECK_EVERY = 256  # bids between wall-clock checks


class _TimeoutError(Exception):
    pass


@dataclass
class _Phase:
    owner: list[int]  # object → person (−1 = unowned)
    assigned: list[int]  # person → object (−1 = unassigned)
    bids: int


def _run_phase(
    benefits: IntMatrix,
    prices: IntVector,
    eps: int,
    deadline: float | None,
    max_bids: int | None,
    bids_before: int,
) -> _Phase:
    """One forward-auction phase from an empty assignment (prices are updated in place)."""
    k = benefits.shape[0]
    owner = [-1] * k
    assigned = [-1] * k
    queue = deque(range(k))
    bids = 0
    floor = np.iinfo(np.int64).min
    while queue:
        i = queue.popleft()
        values = benefits[i] - prices
        j = int(np.argmax(values))  # first maximum: lowest index on ties
        best = int(values[j])
        if k == 1:
            second = best
        else:
            values[j] = floor
            second = int(values.max())
        prices[j] += best - second + eps
        prev = owner[j]
        if prev >= 0:
            assigned[prev] = -1
            queue.append(prev)
        owner[j] = i
        assigned[i] = j
        bids += 1
        if (bids_before + bids) % _CHECK_EVERY == 0:  # cumulative, so short phases still check
            _check_caps(deadline, max_bids, bids_before + bids)
    return _Phase(owner, assigned, bids)


def _check_caps(deadline: float | None, max_bids: int | None, bids: int) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise _TimeoutError
    if max_bids is not None and bids > max_bids:
        raise _TimeoutError


def epsilon_cs_violation(
    benefits: IntMatrix, assignment: tuple[int, ...] | list[int], prices: IntVector, eps: int
) -> int:
    """Largest violation of ε-complementary slackness; ``≤ 0`` means every person holds an
    object within ε of its best net value:
    ``max_i [max_j (a_ij − q_j) − ε − (a_{i,j_i} − q_{j_i})]``.
    """
    k = benefits.shape[0]
    if k == 0:
        return 0
    net = benefits - prices[None, :]
    best = net.max(axis=1)
    held = net[np.arange(k), np.asarray(assignment)]
    return int((best - eps - held).max())


@dataclass(frozen=True)
class AuctionTrace:
    """Final prices and ε (in the units the auction worked in), for ε-CS checks."""

    prices: IntVector
    epsilon: int
    unit_scale: int  # benefits were multiplied by this (k + 1 with ε-scaling, 1 otherwise)


def solve(
    matrix: IntMatrix,
    *,
    scaling: bool = True,
    epsilon: int = 1,
    theta: int = 5,
    time_limit_s: float | None = None,
    max_bids: int | None = None,
) -> tuple[SquareSolution, AuctionTrace]:
    """Solve the square integer assignment problem ``max Σ A[i, σ(i)]`` by auction.

    ``scaling=True``: ε-scaling to ``ε = 1/(k+1)`` (exact optimum). ``scaling=False``: one phase
    with the integer ``epsilon`` (``V ≥ OPT − k·epsilon``). Negative entries are allowed.
    """
    a = np.asarray(matrix, dtype=np.int64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got shape {a.shape}")
    if theta < 2:
        raise ConfigError("auction.theta", "must be >= 2")
    if epsilon < 1:
        raise ConfigError("auction.epsilon", "must be an integer >= 1")
    k = int(a.shape[0])
    method = "auction-scaling" if scaling else f"auction-fixed-eps{epsilon}"
    t0 = time.perf_counter()
    if k == 0:
        stats = SolveStats(method, 0, final_epsilon=0.0)
        return SquareSolution((), 0, stats), AuctionTrace(np.zeros(0, np.int64), 0, 1)
    scale = k + 1 if scaling else 1
    span = int(np.abs(a).max()) if a.size else 0
    if (span + 1) * scale * (k + 2) * 4 >= _INT_LIMIT:
        raise OverflowError("benefits too large for exact int64 auction arithmetic")
    b = a * scale
    prices = np.zeros(k, dtype=np.int64)
    deadline = None if time_limit_s is None else t0 + time_limit_s
    eps = max(int(b.max()) // theta, 1) if scaling else epsilon
    bids = phases = 0
    try:
        while True:
            phase = _run_phase(b, prices, eps, deadline, max_bids, bids)
            bids += phase.bids
            phases += 1
            if not scaling or eps == 1:
                break
            _check_caps(deadline, max_bids, bids)  # also at every phase boundary
            eps = max(eps // theta, 1)
    except _TimeoutError:
        stats = SolveStats(
            method, k, bids, phases, eps / scale, time.perf_counter() - t0, timed_out=True
        )
        return SquareSolution(None, 0, stats), AuctionTrace(prices, eps, scale)
    assignment = tuple(phase.assigned)
    value = int(a[np.arange(k), np.asarray(assignment)].sum())
    stats = SolveStats(method, k, bids, phases, eps / scale, time.perf_counter() - t0)
    return SquareSolution(assignment, value, stats), AuctionTrace(prices, eps, scale)


class AuctionAllocator:
    """The production allocator: ε-scaling auction (exact optimum on the integer benefits).

    ``mode="fixed"`` (with ``epsilon``) exists for experiment E3a only.
    """

    name = "auction"

    def __init__(
        self,
        mode: str = "scaling",
        epsilon: int = 1,
        theta: int = 5,
        time_limit_s: float | None = None,
    ) -> None:
        if mode not in ("scaling", "fixed"):
            raise ConfigError("allocator.auction.mode", "expected 'scaling' or 'fixed'")
        self.mode = mode
        self.epsilon = epsilon
        self.theta = theta
        self.time_limit_s = time_limit_s

    def allocate(self, problem: AllocationProblem) -> AllocationResult:
        sol, _ = solve(
            problem.matrix,
            scaling=self.mode == "scaling",
            epsilon=self.epsilon,
            theta=self.theta,
            time_limit_s=self.time_limit_s,
        )
        return result_from_solution(problem, sol)
