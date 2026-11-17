"""Scenario specification (immutable) and its validation (FR-ENV-8; test X-3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell

MOTION_MODELS = ("random_walk", "patrol", "bounce")
ARRIVAL_KINDS = ("none", "bernoulli", "poisson")


class ScenarioError(ValueError):
    """Invalid scenario. ``field`` names the offending entry, e.g. ``tasks[2].pickup``."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


@dataclass(frozen=True, slots=True)
class ObstacleSpec:
    start: Cell
    model: str = "random_walk"
    waypoints: tuple[Cell, ...] = ()
    direction: Action | None = None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    pickup: Cell
    dropoff: Cell
    priority: int = 1
    release: int = 0


@dataclass(frozen=True, slots=True)
class ArrivalSpec:
    kind: str = "none"
    rate: float = 0.0
    priorities: tuple[float, ...] = (1.0,)
    until: int = 0


@dataclass(frozen=True, slots=True)
class MapEventSpec:
    time: int
    cell: Cell
    block: bool


@dataclass(frozen=True, slots=True)
class ScriptSpec:
    """Hand-written behaviour for the scripted demo controller/coordinator.

    ``actions[i]`` is robot ``i``'s action string (S/U/D/L/R; spaces ignored); after it ends the
    robot stays. ``assignments[i]`` lists the task indices robot ``i`` takes, in order.
    """

    actions: tuple[str, ...]
    assignments: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    grid: OccupancyGrid
    robots: tuple[Cell, ...]
    seed: int = 0
    max_steps: int = 100
    description: str = ""
    num_priorities: int = 3
    obstacles: tuple[ObstacleSpec, ...] = ()
    tasks: tuple[TaskSpec, ...] = ()
    arrivals: ArrivalSpec = field(default_factory=ArrivalSpec)
    map_events: tuple[MapEventSpec, ...] = ()
    script: ScriptSpec | None = None

    def validate(self) -> None:
        """Raise :class:`ScenarioError` naming the first invalid field."""
        validate_scenario(self)


def _check_cell(grid: OccupancyGrid, cell: Cell, where: str, *, must_be_free: bool = True) -> None:
    if not grid.in_bounds(cell):
        raise ScenarioError(where, f"cell {cell} is outside the {grid.width}x{grid.height} map")
    if must_be_free and grid.is_blocked(cell):
        raise ScenarioError(where, f"cell {cell} is a static obstacle")


def validate_scenario(spec: ScenarioSpec) -> None:
    grid = spec.grid
    if not spec.name:
        raise ScenarioError("scenario.name", "must be non-empty")
    if spec.max_steps < 1:
        raise ScenarioError("scenario.max_steps", "must be >= 1")
    if spec.seed < 0:
        raise ScenarioError("scenario.seed", "must be >= 0")
    if spec.num_priorities < 1:
        raise ScenarioError("scenario.num_priorities", "must be >= 1")
    if not spec.robots:
        raise ScenarioError("robots", "at least one robot is required")

    occupied: dict[Cell, str] = {}
    for i, start in enumerate(spec.robots):
        where = f"robots[{i}].start"
        _check_cell(grid, start, where)
        if start in occupied:
            raise ScenarioError(where, f"cell {start} is already used by {occupied[start]}")
        occupied[start] = f"robots[{i}]"

    for k, obs in enumerate(spec.obstacles):
        where = f"obstacles[{k}]"
        _check_cell(grid, obs.start, f"{where}.start")
        if obs.start in occupied:
            raise ScenarioError(
                f"{where}.start", f"cell {obs.start} is already used by {occupied[obs.start]}"
            )
        occupied[obs.start] = where
        if obs.model not in MOTION_MODELS:
            raise ScenarioError(
                f"{where}.model", f"unknown model {obs.model!r}; choose from {MOTION_MODELS}"
            )
        if obs.model == "patrol":
            if not obs.waypoints:
                raise ScenarioError(f"{where}.waypoints", "patrol needs at least one waypoint")
            for w, wp in enumerate(obs.waypoints):
                _check_cell(grid, wp, f"{where}.waypoints[{w}]")
        if obs.model == "bounce" and (obs.direction is None or obs.direction is Action.STAY):
            raise ScenarioError(
                f"{where}.direction", "bounce needs a direction (up/down/left/right)"
            )

    oracle = DistanceOracle(grid)
    for j, task in enumerate(spec.tasks):
        where = f"tasks[{j}]"
        _check_cell(grid, task.pickup, f"{where}.pickup")
        _check_cell(grid, task.dropoff, f"{where}.dropoff")
        if task.pickup == task.dropoff:
            raise ScenarioError(where, "pickup and dropoff must differ")
        if not 1 <= task.priority <= spec.num_priorities:
            raise ScenarioError(f"{where}.priority", f"must be in 1..{spec.num_priorities}")
        if task.release < 0:
            raise ScenarioError(f"{where}.release", "must be >= 0")
        if not oracle.reachable(task.pickup, task.dropoff):
            raise ScenarioError(
                where, "dropoff is not reachable from pickup on the initial map (task unavailable)"
            )

    arr = spec.arrivals
    if arr.kind not in ARRIVAL_KINDS:
        raise ScenarioError(
            "arrivals.kind", f"unknown kind {arr.kind!r}; choose from {ARRIVAL_KINDS}"
        )
    if arr.kind != "none":
        if arr.rate < 0 or (arr.kind == "bernoulli" and arr.rate > 1):
            raise ScenarioError(
                "arrivals.rate", "must be in [0, 1] for bernoulli, >= 0 for poisson"
            )
        if len(arr.priorities) != spec.num_priorities:
            raise ScenarioError("arrivals.priorities", f"needs {spec.num_priorities} probabilities")
        if any(p < 0 for p in arr.priorities) or abs(sum(arr.priorities) - 1.0) > 1e-9:
            raise ScenarioError("arrivals.priorities", "must be non-negative and sum to 1")
        if arr.until < 0:
            raise ScenarioError("arrivals.until", "must be >= 0")

    for e, ev in enumerate(spec.map_events):
        _check_cell(grid, ev.cell, f"map_events[{e}].cell", must_be_free=False)
        if ev.time < 0:
            raise ScenarioError(f"map_events[{e}].time", "must be >= 0")

    if spec.script is not None:
        n = len(spec.robots)
        if len(spec.script.actions) != n:
            raise ScenarioError("script.actions", f"needs one action string per robot ({n})")
        if len(spec.script.assignments) != n:
            raise ScenarioError("script.assignments", f"needs one task list per robot ({n})")
        for i, actions in enumerate(spec.script.actions):
            for ch in actions:
                if ch != " " and ch.upper() not in "SUDLR":
                    raise ScenarioError(f"script.actions[{i}]", f"invalid action symbol {ch!r}")
        seen: set[int] = set()
        for i, lst in enumerate(spec.script.assignments):
            for j in lst:
                if not 0 <= j < len(spec.tasks):
                    raise ScenarioError(
                        f"script.assignments[{i}]", f"task index {j} does not exist"
                    )
                if j in seen:
                    raise ScenarioError(f"script.assignments[{i}]", f"task {j} is assigned twice")
                seen.add(j)
