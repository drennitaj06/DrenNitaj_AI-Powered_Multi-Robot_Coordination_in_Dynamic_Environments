"""TOML scenario loader. File format: ``docs/03-architecture.md``, Appendix A."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.core.grid import OccupancyGrid
from robocoord.core.mapio import MapFormatError, load_movingai_map
from robocoord.core.rng import make_streams
from robocoord.core.types import Action, Cell
from robocoord.scenario import generators
from robocoord.scenario.spec import (
    ArrivalSpec,
    MapEventSpec,
    ObstacleSpec,
    ScenarioError,
    ScenarioSpec,
    ScriptSpec,
    TaskSpec,
)
from robocoord.world.arrivals import draw_task

_DIRECTIONS = {"up": Action.UP, "down": Action.DOWN, "left": Action.LEFT, "right": Action.RIGHT}


def load_scenario(path: str | Path, *, seed: int | None = None) -> ScenarioSpec:
    """Load and validate a scenario file. ``seed`` overrides ``scenario.seed``."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ScenarioError("file", f"scenario file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ScenarioError("file", f"invalid TOML: {exc}") from None
    return build_scenario(data, base_dir=path.parent, seed=seed)


def build_scenario(
    data: Mapping[str, Any], *, base_dir: Path | None = None, seed: int | None = None
) -> ScenarioSpec:
    """Build and validate a :class:`ScenarioSpec` from parsed TOML data."""
    known = {
        "scenario",
        "map",
        "robots",
        "obstacles",
        "tasks",
        "arrivals",
        "map_events",
        "script",
        "placement",
    }
    for key in data:
        if key not in known:
            raise ScenarioError(key, f"unknown section; expected one of {sorted(known)}")
    meta = _table(data, "scenario")
    name = _get(meta, "name", str, "scenario.name")
    base_seed = _get(meta, "seed", int, "scenario.seed", default=0)
    run_seed = base_seed if seed is None else seed
    max_steps = _get(meta, "max_steps", int, "scenario.max_steps", default=100)
    description = _get(meta, "description", str, "scenario.description", default="")
    num_priorities = _get(meta, "num_priorities", int, "scenario.num_priorities", default=3)

    map_rng = make_streams(run_seed).generator("map")
    grid = _build_map(_table(data, "map"), base_dir, map_rng)

    robots = tuple(
        _cell(r.get("start"), f"robots[{i}].start") for i, r in enumerate(_array(data, "robots"))
    )
    obstacles = tuple(
        _obstacle(o, f"obstacles[{k}]") for k, o in enumerate(_array(data, "obstacles"))
    )
    tasks = tuple(_task(t, f"tasks[{j}]") for j, t in enumerate(_array(data, "tasks")))

    if "placement" in data:
        robots, obstacles, tasks = _random_placement(
            _table(data, "placement"), grid, robots, obstacles, tasks, map_rng
        )

    arrivals = _arrivals(data.get("arrivals"))
    events = tuple(
        _map_event(e, f"map_events[{i}]") for i, e in enumerate(_array(data, "map_events"))
    )
    script = _script(data.get("script"))

    spec = ScenarioSpec(
        name=name,
        grid=grid,
        robots=robots,
        seed=run_seed,
        max_steps=max_steps,
        description=description,
        num_priorities=num_priorities,
        obstacles=obstacles,
        tasks=tasks,
        arrivals=arrivals,
        map_events=events,
        script=script,
    )
    spec.validate()
    return spec


# ---------------------------------------------------------------------------- helpers
def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ScenarioError(key, "missing or not a table")
    return value


def _array(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, Mapping) for v in value):
        raise ScenarioError(key, "must be an array of tables ([[...]])")
    return value


def _get(table: Mapping[str, Any], key: str, typ: type, where: str, default: Any = ...) -> Any:
    if key not in table:
        if default is ...:
            raise ScenarioError(where, "is required")
        return default
    value = table[key]
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
        raise ScenarioError(where, f"expected {typ.__name__}, got {type(value).__name__}")
    return value


def _cell(value: Any, where: str) -> Cell:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        raise ScenarioError(where, f"expected [x, y] integer pair, got {value!r}")
    return (value[0], value[1])


def _build_map(
    table: Mapping[str, Any], base_dir: Path | None, rng: np.random.Generator
) -> OccupancyGrid:
    sources = [k for k in ("rows", "file", "generator") if k in table]
    if len(sources) != 1:
        raise ScenarioError("map", "specify exactly one of 'rows', 'file' or 'generator'")
    if "rows" in table:
        rows = table["rows"]
        if not isinstance(rows, list) or not all(isinstance(r, str) for r in rows):
            raise ScenarioError("map.rows", "must be a list of strings")
        try:
            return OccupancyGrid.from_rows(rows)
        except ValueError as exc:
            raise ScenarioError("map.rows", str(exc)) from None
    if "file" in table:
        file = Path(_get(table, "file", str, "map.file"))
        if base_dir is not None and not file.is_absolute():
            file = base_dir / file
        try:
            return load_movingai_map(file)
        except FileNotFoundError:
            raise ScenarioError("map.file", f"file not found: {file}") from None
        except MapFormatError as exc:
            raise ScenarioError("map.file", str(exc)) from None
    gen = table["generator"]
    if not isinstance(gen, Mapping):
        raise ScenarioError("map.generator", "must be a table")
    kind = _get(gen, "kind", str, "map.generator.kind")
    width = _get(gen, "width", int, "map.generator.width")
    height = _get(gen, "height", int, "map.generator.height")
    if width < 1 or height < 1:
        raise ScenarioError("map.generator", "width and height must be >= 1")
    try:
        if kind == "random":
            density = _get(gen, "density", float, "map.generator.density", default=0.2)
            return generators.random_density(width, height, density, rng)
        if kind == "rooms":
            room = _get(gen, "room_size", int, "map.generator.room_size", default=4)
            return generators.rooms(width, height, room, rng)
        if kind == "warehouse":
            shelf = _get(gen, "shelf_length", int, "map.generator.shelf_length", default=4)
            return generators.warehouse(width, height, shelf)
        if kind == "empty":
            return OccupancyGrid.empty(width, height)
    except ValueError as exc:
        raise ScenarioError("map.generator", str(exc)) from None
    raise ScenarioError(
        "map.generator.kind", f"unknown generator {kind!r}; choose random, rooms, warehouse, empty"
    )


def _obstacle(o: Mapping[str, Any], where: str) -> ObstacleSpec:
    start = _cell(o.get("start"), f"{where}.start")
    model = _get(o, "model", str, f"{where}.model", default="random_walk")
    waypoints = tuple(
        _cell(w, f"{where}.waypoints[{i}]") for i, w in enumerate(o.get("waypoints", []))
    )
    direction = None
    if "direction" in o:
        name = _get(o, "direction", str, f"{where}.direction").lower()
        if name not in _DIRECTIONS:
            raise ScenarioError(f"{where}.direction", f"unknown direction {name!r}")
        direction = _DIRECTIONS[name]
    return ObstacleSpec(start, model, waypoints, direction)


def _task(t: Mapping[str, Any], where: str) -> TaskSpec:
    return TaskSpec(
        pickup=_cell(t.get("pickup"), f"{where}.pickup"),
        dropoff=_cell(t.get("dropoff"), f"{where}.dropoff"),
        priority=_get(t, "priority", int, f"{where}.priority", default=1),
        release=_get(t, "release", int, f"{where}.release", default=0),
    )


def _arrivals(a: Any) -> ArrivalSpec:
    if a is None:
        return ArrivalSpec()
    if not isinstance(a, Mapping):
        raise ScenarioError("arrivals", "must be a table")
    pr = a.get("priorities", [1.0])
    if not isinstance(pr, list) or not all(isinstance(p, int | float) for p in pr):
        raise ScenarioError("arrivals.priorities", "must be a list of numbers")
    return ArrivalSpec(
        kind=_get(a, "kind", str, "arrivals.kind", default="none"),
        rate=_get(a, "rate", float, "arrivals.rate", default=0.0),
        priorities=tuple(float(p) for p in pr),
        until=_get(a, "until", int, "arrivals.until", default=0),
    )


def _map_event(e: Mapping[str, Any], where: str) -> MapEventSpec:
    action = _get(e, "action", str, f"{where}.action")
    if action not in ("block", "unblock"):
        raise ScenarioError(f"{where}.action", "must be 'block' or 'unblock'")
    return MapEventSpec(
        time=_get(e, "time", int, f"{where}.time"),
        cell=_cell(e.get("cell"), f"{where}.cell"),
        block=action == "block",
    )


def _script(s: Any) -> ScriptSpec | None:
    if s is None:
        return None
    if not isinstance(s, Mapping):
        raise ScenarioError("script", "must be a table")
    actions = s.get("actions", [])
    assignments = s.get("assignments", [])
    if not isinstance(actions, list) or not all(isinstance(a, str) for a in actions):
        raise ScenarioError("script.actions", "must be a list of strings")
    if not isinstance(assignments, list) or not all(
        isinstance(lst, list) and all(isinstance(j, int) for j in lst) for lst in assignments
    ):
        raise ScenarioError("script.assignments", "must be a list of integer lists")
    return ScriptSpec(tuple(actions), tuple(tuple(lst) for lst in assignments))


def _random_placement(
    table: Mapping[str, Any],
    grid: OccupancyGrid,
    robots: tuple[Cell, ...],
    obstacles: tuple[ObstacleSpec, ...],
    tasks: tuple[TaskSpec, ...],
    rng: np.random.Generator,
) -> tuple[tuple[Cell, ...], tuple[ObstacleSpec, ...], tuple[TaskSpec, ...]]:
    """Add randomly placed robots, obstacles and tasks (drawn from the ``map`` stream)."""
    from robocoord.core.distance import DistanceOracle

    n_robots = _get(table, "robots", int, "placement.robots", default=0)
    n_obs = _get(table, "obstacles", int, "placement.obstacles", default=0)
    n_tasks = _get(table, "tasks", int, "placement.tasks", default=0)
    model = _get(table, "obstacle_model", str, "placement.obstacle_model", default="random_walk")
    max_release = _get(table, "max_release", int, "placement.max_release", default=0)
    pr = table.get("priorities", [1.0])
    if model not in ("random_walk",):
        raise ScenarioError(
            "placement.obstacle_model", "only 'random_walk' is supported for random placement"
        )
    used = set(robots) | {o.start for o in obstacles}
    free = [c for c in grid.free_cells() if c not in used]
    need = n_robots + n_obs
    if need > len(free):
        raise ScenarioError(
            "placement", f"needs {need} free cells, map has only {len(free)} available"
        )
    picks = rng.choice(len(free), size=need, replace=False)
    cells = [free[int(i)] for i in picks]
    new_robots = tuple(cells[:n_robots])
    new_obs = tuple(ObstacleSpec(c, "random_walk") for c in cells[n_robots:])
    oracle = DistanceOracle(grid)
    probs = np.asarray(pr, dtype=float)
    if probs.ndim != 1 or probs.size == 0 or not np.isclose(probs.sum(), 1.0):
        raise ScenarioError("placement.priorities", "must be a probability vector")
    all_free = grid.free_cells()
    new_tasks: list[TaskSpec] = []
    for _ in range(n_tasks):
        release = int(rng.integers(0, max_release + 1))
        req = draw_task(release, all_free, oracle, probs, rng)
        if req is None:
            raise ScenarioError("placement.tasks", "could not draw a reachable task")
        new_tasks.append(TaskSpec(req.pickup, req.dropoff, req.priority, req.release))
    return robots + new_robots, obstacles + new_obs, tasks + tuple(new_tasks)
