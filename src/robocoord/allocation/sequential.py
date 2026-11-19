"""Sequential single-item auction: the greedy baseline allocator (algorithms §4.2).

**Rule (as specified).** While a free robot and a feasible task remain, every free robot bids its
benefit for its best remaining feasible task, the globally highest bid wins, and the pair is
removed. Ties: a robot bids for its lowest-index best task; among equal bids the lowest robot index
wins. "Feasible" means a positive entry of the padded integer matrix (``0`` = dummy or
infeasible, model §8).

**Implementation.** The round winner is always the remaining feasible pair that is smallest in
the order ``(−A_ij, i, j)``: robot ``i``'s bid is its smallest ``(−A_ij, j)``, and the winner is
the smallest ``(−A_ij, i)`` among the bids. So one sort of the feasible pairs by that key and one
scan that keeps every pair whose robot and task are still free give exactly the round-by-round
outcome. Time O(F log F) for ``F ≤ n·m`` feasible pairs (the documented ``O(nm log m)`` with
pre-sorted bid lists, up to the log factor), memory O(F). The tests compare this against a
literal round-by-round simulation written from the rule above.

**Guarantees (on the integer matrix it is given, which includes the offset M).**

* Greedy maximum-weight matching with non-negative weights is a **½-approximation**:
  ``2·V ≥ OPT`` (Guaranteed; every optimal pair loses to a greedy pair sharing an endpoint, and
  each greedy pair blocks at most two optimal pairs).
* The result is a **maximal** matching, so its cardinality is at least half the maximum
  (Guaranteed). It is not maximum in general.
* Neither statement transfers to original units (``value`` = ``κw̃ − c + h``, which can be
  negative). See `04` §4.2 implementation notes.
"""

from __future__ import annotations

import time

import numpy as np

from robocoord.allocation.auction import IntMatrix
from robocoord.allocation.base import (
    AllocationResult,
    SolveStats,
    SquareSolution,
    result_from_solution,
)
from robocoord.allocation.benefits import AllocationProblem


def solve(matrix: IntMatrix) -> SquareSolution:
    """Greedy matching of a square non-negative integer matrix (positive entries = feasible).

    Returns a full permutation: rows and columns the auction left unmatched are paired in index
    order, and every such pair has benefit 0 (otherwise the matching would not be maximal), so the
    value is unchanged. ``stats.bids`` is the number of rounds, i.e. winning bids.
    """
    a = np.asarray(matrix, dtype=np.int64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got shape {a.shape}")
    if a.size and int(a.min()) < 0:
        raise ValueError("the sequential auction needs non-negative benefits (0 = infeasible)")
    t0 = time.perf_counter()
    k = int(a.shape[0])
    rows, cols = np.nonzero(a > 0)
    order = np.lexsort((cols, rows, -a[rows, cols]))  # last key is primary: (−A, i, j)
    assigned = [-1] * k
    col_taken = [False] * k
    rounds = 0
    for idx in order.tolist():
        i, j = int(rows[idx]), int(cols[idx])
        if assigned[i] < 0 and not col_taken[j]:
            assigned[i] = j
            col_taken[j] = True
            rounds += 1
    value = int(sum(int(a[i, j]) for i, j in enumerate(assigned) if j >= 0))
    free_cols = iter(j for j in range(k) if not col_taken[j])
    assignment = tuple(j if j >= 0 else next(free_cols) for j in assigned)
    stats = SolveStats("sequential", k, bids=rounds, time_s=time.perf_counter() - t0)
    return SquareSolution(assignment, value, stats)


class SequentialAllocator:
    """Baseline allocator for E3 and the end-to-end matrix (not optimal; ½-approximation)."""

    name = "sequential"

    def allocate(self, problem: AllocationProblem) -> AllocationResult:
        return result_from_solution(problem, solve(problem.matrix))
