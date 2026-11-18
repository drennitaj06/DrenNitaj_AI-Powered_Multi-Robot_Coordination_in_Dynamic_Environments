"""Online path planning: A* and RRT* behind one interface (Issue #3; architecture §5.4, §6).

Both planners are required, implemented independently and selectable by name::

    planner = make_planner("astar")
    planner = make_planner("rrtstar", max_iterations=2000, neighbor_index="kdtree")
    result = plan_with_procedure(planner, PlanQuery(start, goal, constraints, rng))
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from robocoord.core.errors import ConfigError
from robocoord.planning.astar import AStarPlanner
from robocoord.planning.base import (
    PlanConstraints,
    Planner,
    PlanQuery,
    PlanResult,
    PlanStats,
    PlanStatus,
)
from robocoord.planning.constraints import ConstraintParams, build_constraints
from robocoord.planning.procedure import plan_with_procedure
from robocoord.planning.rrtstar import RRTStarConfig, RRTStarPlanner


def _make_astar(**options: Any) -> Planner:
    if options:
        raise ConfigError("planner.astar", f"A* takes no options, got {sorted(options)}")
    return AStarPlanner()


def _make_rrtstar(**options: Any) -> Planner:
    known = set(RRTStarConfig.__dataclass_fields__)
    unknown = set(options) - known
    if unknown:
        raise ConfigError(
            "planner.rrtstar", f"unknown option(s) {sorted(unknown)}; expected {sorted(known)}"
        )
    return RRTStarPlanner(RRTStarConfig(**options))


#: Name → factory. The same names are used by the CLI and experiment specs (architecture §6).
PLANNERS: Mapping[str, Callable[..., Planner]] = {"astar": _make_astar, "rrtstar": _make_rrtstar}


def make_planner(name: str, **options: Any) -> Planner:
    """Create a planner by name; unknown names fail with the valid choices listed."""
    try:
        factory = PLANNERS[name]
    except KeyError:
        raise ConfigError(
            "planner", f"unknown planner {name!r}; expected one of {sorted(PLANNERS)}"
        ) from None
    return factory(**options)


__all__ = [
    "PLANNERS",
    "AStarPlanner",
    "ConstraintParams",
    "PlanConstraints",
    "PlanQuery",
    "PlanResult",
    "PlanStats",
    "PlanStatus",
    "Planner",
    "RRTStarConfig",
    "RRTStarPlanner",
    "build_constraints",
    "make_planner",
    "plan_with_procedure",
]
