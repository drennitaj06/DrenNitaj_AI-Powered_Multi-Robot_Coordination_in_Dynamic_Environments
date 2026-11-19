"""Allocator interface (architecture §5.5): allocators solve the integer padded assignment
problem built by :func:`robocoord.allocation.benefits.build_problem`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from robocoord.allocation.benefits import AllocationProblem


@dataclass(frozen=True)
class SolveStats:
    """Work counters of one assignment solve (hardware independent) plus wall time.

    ``bids`` — auction bids (or sequential-auction rounds); ``phases`` — ε-scaling phases;
    ``final_epsilon`` — the last ε in original benefit units; ``timed_out`` — the solve hit its
    wall-clock or bid cap and returned no assignment.
    """

    method: str
    k: int
    bids: int = 0
    phases: int = 0
    final_epsilon: float = 0.0
    time_s: float = 0.0
    timed_out: bool = False


@dataclass(frozen=True)
class SquareSolution:
    """Solution of a square integer assignment problem: ``assignment[row] = column``.

    ``assignment`` is ``None`` only if the solve timed out. ``value`` is ``Σ A[i, assignment[i]]``
    in the matrix's own units.
    """

    assignment: tuple[int, ...] | None
    value: int
    stats: SolveStats


@dataclass(frozen=True)
class AllocationResult:
    """Robot → task pairs chosen for one allocation event (dummy and infeasible pairs dropped)."""

    pairs: tuple[tuple[int, int], ...]
    value: float  # Σ value[i, j] of the pairs, original units (offset excluded)
    stats: SolveStats


class Allocator(Protocol):
    name: str

    def allocate(self, problem: AllocationProblem) -> AllocationResult: ...


def result_from_solution(problem: AllocationProblem, sol: SquareSolution) -> AllocationResult:
    if sol.assignment is None:
        return AllocationResult((), 0.0, sol.stats)
    pairs = tuple(problem.pairs(list(sol.assignment)))
    return AllocationResult(pairs, problem.matching_value(pairs), sol.stats)
