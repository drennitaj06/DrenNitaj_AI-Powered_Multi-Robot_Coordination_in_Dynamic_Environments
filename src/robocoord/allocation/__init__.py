"""Market-based task allocation (Issue #4; model §8; algorithms §4; ADR-5)."""

from __future__ import annotations

from robocoord.allocation.benefits import (
    AllocationProblem,
    BenefitConfig,
    RobotCandidate,
    TaskCandidate,
    build_problem,
    effective_weight,
    priority_weight,
)

__all__ = [
    "AllocationProblem",
    "BenefitConfig",
    "RobotCandidate",
    "TaskCandidate",
    "build_problem",
    "effective_weight",
    "priority_weight",
]
