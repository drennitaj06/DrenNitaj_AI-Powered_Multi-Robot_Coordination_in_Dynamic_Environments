"""Executors of the planner experiments (05 §B.4): E1 static queries, E2 replanning episodes.

Instances are generated deterministically from the experiment's ``seed_base`` and the run's
factor values only, never from the planner, so A* and RRT* see identical maps and queries
(paired design, 05 §B.1).
"""

from __future__ import annotations

import math
import time
import tracemalloc
from collections.abc import Mapping
from typing import Any

import numpy as np

from robocoord.control.base import SuggestionFollower
from robocoord.core.distance import bfs_distance_field
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell
from robocoord.experiments.spec import ExperimentSpec, PlannerSetting
from robocoord.planning import PlanQuery, build_constraints, make_planner, plan_with_procedure
from robocoord.planning.base import PlanResult, lattice_cost
from robocoord.planning.coordinator import PlanningCoordinator
from robocoord.planning.geometry import polyline_length
from robocoord.planning.rrtstar import TRACE_FIELDS, gamma_star
from robocoord.scenario.generators import random_density, rooms, warehouse
from robocoord.scenario.spec import MapEventSpec, ObstacleSpec, ScenarioSpec, ScriptSpec, TaskSpec
from robocoord.sim.engine import Engine

#: Synthetic map families of E1a (05 §B.4).
FAMILIES: dict[str, tuple[str, float]] = {
    "open": ("random", 0.0),
    "d10": ("random", 0.10),
    "d20": ("random", 0.20),
    "d30": ("random", 0.30),
    "d35": ("random", 0.35),
    "rooms": ("rooms", 5.0),
    "warehouse": ("warehouse", 4.0),
}
_FAMILY_INDEX = {name: i for i, name in enumerate(FAMILIES)}


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


def make_map(family: str, size: int, seed_base: int, map_index: int) -> OccupancyGrid:
    kind, value = FAMILIES[family]
    rng = _rng(seed_base, _FAMILY_INDEX[family], size, map_index)
    if kind == "random":
        return random_density(size, size, value, rng)
    if kind == "rooms":
        return rooms(size, size, int(value), rng)
    return warehouse(size, size, int(value))


def make_pair(grid: OccupancyGrid, key: tuple[int, ...]) -> tuple[Cell, Cell]:
    """Two distinct free cells (the generators keep one connected component, so the pair is
    reachable)."""
    free = grid.free_cells()
    rng = _rng(*key)
    i, j = rng.choice(len(free), size=2, replace=False)
    return free[int(i)], free[int(j)]


def euclid_lower_bound(start: Cell, goal: Cell) -> float:
    """Distance from the centre of ``start`` to the nearest point of ``square(goal)``: the exact
    Euclidean optimum on an obstacle-free map, a lower bound otherwise."""
    cx, cy = start[0] + 0.5, start[1] + 0.5
    dx = max(goal[0] - cx, 0.0, cx - (goal[0] + 1))
    dy = max(goal[1] - cy, 0.0, cy - (goal[1] + 1))
    return math.hypot(dx, dy)


def _planner_options(
    setting: PlannerSetting | None, params: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    name = setting.name if setting else str(params.get("planner_name", "rrtstar"))
    options = dict(setting.options) if setting else {}
    if "n_max" in params:
        options["max_iterations"] = int(params["n_max"])
    if "index" in params:
        options["neighbor_index"] = str(params["index"])
    return name, options


def _bin_trace(trace: Mapping[str, Any]) -> dict[str, list[float]]:
    """Per-iteration counters averaged in log₂-spaced bins of the tree size ``n``."""
    n = np.asarray(trace["n"], dtype=np.float64)
    if n.size == 0:
        return {}
    edges = 2.0 ** np.arange(0, math.ceil(math.log2(max(n.max(), 1))) + 2)
    which = np.digitize(n, edges)
    out: dict[str, list[float]] = {"n_mean": [], "iterations": []}
    for f in TRACE_FIELDS:
        if f != "n":
            out[f] = []
    for b in np.unique(which):
        sel = which == b
        out["n_mean"].append(float(n[sel].mean()))
        out["iterations"].append(int(sel.sum()))
        for f in TRACE_FIELDS:
            if f != "n":
                out[f].append(float(np.asarray(trace[f])[sel].mean()))
    return out


def run_query(spec: ExperimentSpec, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """E1: one static planning query. Returns ``(outcome, metrics)``."""
    family, size = str(params["family"]), int(params["size"])
    map_index, pair_index = int(params["map"]), int(params["pair"])
    grid = make_map(family, size, spec.seed_base, map_index)
    key = (spec.seed_base, _FAMILY_INDEX[family], size, map_index, pair_index)
    start, goal = make_pair(grid, key)
    constraints = build_constraints(grid, start, goal)  # static query: no sensed obstacles
    setting = spec.planners.get(str(params.get("planner", "")))
    name, options = _planner_options(setting, params)
    fixed = spec.fixed
    record_trace = bool(fixed.get("record_trace", False))
    checkpoints = [int(c) for c in fixed.get("checkpoints", [])]
    if name == "rrtstar":
        options.setdefault("time_limit_s", spec.timeout_s)
        if record_trace or checkpoints:
            options["record_trace"] = True
    planner = make_planner(name, **options)

    def solve() -> PlanResult:
        return plan_with_procedure(planner, PlanQuery(start, goal, constraints, _rng(*key, 7)))

    res = solve()
    bfs = int(bfs_distance_field(grid, goal)[start[1], start[0]])
    st = res.stats
    m: dict[str, Any] = {
        "planner": name,
        "options": {k: v for k, v in options.items() if k != "time_limit_s"},
        "start": list(start),
        "goal": list(goal),
        "free_cells": grid.num_free(),
        "bfs_steps": bfs,
        "euclid_lower_bound": euclid_lower_bound(start, goal),
        "status": res.status,
        "time_s": st.time_s,
        "counters": st.counters(),
    }
    if res.path is not None:
        steps = len(res.path) - 1
        m["path_steps"] = steps
        m["lattice_cost"] = lattice_cost(res.path, constraints)
        m["step_gap"] = steps / bfs if bfs > 0 else 1.0
        m["length"] = polyline_length(res.waypoints) if res.waypoints else float(steps)
        m["planner_cost"] = res.cost
    if st.iteration_trace is not None and record_trace:
        m["trace_bins"] = _bin_trace(st.iteration_trace)
    if st.best_cost_trace is not None and checkpoints:
        bct = st.best_cost_trace
        gamma = options.get("gamma_factor", 1.5) * gamma_star(constraints.free_area())
        eta = float(options.get("eta", 2.0))
        m["checkpoints"] = {str(c): bct[c - 1] if c <= len(bct) else None for c in checkpoints}
        m["radius_at"] = {
            str(c): min(gamma * math.sqrt(math.log(c) / c), eta) if c >= 2 else eta
            for c in checkpoints
        }
    if fixed.get("measure_memory", False):
        tracemalloc.start()
        solve()
        m["peak_memory_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    outcome = "timeout" if st.timed_out else res.status
    return outcome, m


# ====================================================================== E2: replanning episodes


def e2_scenario(
    seed: int,
    n_obstacles: int,
    events: str,
    team: int,
    *,
    size: int = 24,
    density: float = 0.15,
    tasks_per_robot: int = 3,
    max_steps: int = 300,
    n_events: int = 6,
    event_duration: int = 20,
) -> ScenarioSpec:
    """A paired E2 instance: map, robots and tasks depend only on ``(seed, team)``; obstacles and
    map events come from separate streams, so they do not change the map or the tasks."""
    grid = random_density(size, size, density, _rng(seed, 1))
    free = grid.free_cells()
    place = _rng(seed, 2, team)
    robots = [free[int(i)] for i in place.choice(len(free), size=team, replace=False)]
    tasks: list[TaskSpec] = []
    for _ in range(tasks_per_robot * team):
        a, b = place.choice(len(free), size=2, replace=False)
        tasks.append(TaskSpec(free[int(a)], free[int(b)], int(place.integers(1, 4)), 0))
    obs_rng = _rng(seed, 3)
    candidates = [c for c in free if c not in set(robots)]
    picks: list[int] = (
        obs_rng.choice(len(candidates), size=n_obstacles, replace=False).tolist()
        if n_obstacles
        else []
    )
    obstacles: list[ObstacleSpec] = []
    for j, i in enumerate(picks):
        cell = candidates[int(i)]
        if j % 2 == 0:
            obstacles.append(ObstacleSpec(cell, "random_walk"))
        else:
            far = free[int(obs_rng.integers(len(free)))]
            obstacles.append(ObstacleSpec(cell, "patrol", (cell, far)))
    map_events: list[MapEventSpec] = []
    if events == "low":
        ev_rng = _rng(seed, 4)
        for _ in range(n_events):
            t0 = int(ev_rng.integers(5, max_steps // 2))
            cell = free[int(ev_rng.integers(len(free)))]
            map_events.append(MapEventSpec(t0, cell, True))
            map_events.append(MapEventSpec(t0 + event_duration, cell, False))
        map_events.sort(key=lambda e: (e.time, e.cell, not e.block))
    elif events != "none":
        raise ValueError(f"unknown events level {events!r}")
    assignments = tuple(tuple(range(i, tasks_per_robot * team, team)) for i in range(team))
    spec = ScenarioSpec(
        name=f"e2-s{seed}-k{n_obstacles}-{events}-n{team}",
        grid=grid,
        robots=tuple(robots),
        seed=seed,
        max_steps=max_steps,
        obstacles=tuple(obstacles),
        tasks=tuple(tasks),
        map_events=tuple(map_events),
        script=ScriptSpec(tuple("" for _ in range(team)), assignments),
    )
    spec.validate()
    return spec


def run_episode(spec: ExperimentSpec, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """E2: one episode with the reference plan follower."""
    fixed = spec.fixed
    scenario = e2_scenario(
        int(params["seed"]),
        int(params["obstacles"]),
        str(params["events"]),
        int(params["team"]),
        size=int(fixed.get("size", 24)),
        density=float(fixed.get("density", 0.15)),
        tasks_per_robot=int(fixed.get("tasks_per_robot", 3)),
        max_steps=int(fixed.get("max_steps", 300)),
    )
    setting = spec.planners[str(params["planner"])]
    planner = make_planner(setting.name, **dict(setting.options))
    assert scenario.script is not None
    coord = PlanningCoordinator(planner, scenario.script.assignments)
    engine = Engine(scenario, coord)
    follower = SuggestionFollower()
    t0 = time.perf_counter()
    timed_out = False
    while not engine.done:
        engine.step(follower.act)
        if time.perf_counter() - t0 > spec.timeout_s:
            timed_out = True
            break
    wall = time.perf_counter() - t0
    summary = engine.summary()
    times = np.asarray(coord.call_times(), dtype=np.float64)
    m: dict[str, Any] = {
        "planner": setting.name,
        "steps": summary.steps,
        "terminated": engine.terminated,
        "tasks_released": summary.tasks_released,
        "tasks_delivered": summary.tasks_delivered,
        "completion_rate": summary.completion_rate,
        "makespan": summary.makespan,
        "mean_flow_time": summary.mean_flow_time,
        "conflicts": summary.conflicts,
        "conflicts_per_100": summary.conflicts_per_100_robot_steps,
        "blocked_moves": summary.blocked_moves,
        "blocked_goal_robot_steps": summary.blocked_goal_robot_steps,
        "wait_ratio": summary.wait_ratio,
        "plan_adherence": summary.plan_adherence,
        "path_efficiency": summary.path_efficiency,
        "planning": summary.extra.get("planning"),
        "latency_s": {
            "calls": int(times.size),
            "median": float(np.median(times)) if times.size else None,
            "p95": float(np.quantile(times, 0.95)) if times.size else None,
            "max": float(times.max()) if times.size else None,
            "total": float(times.sum()),
        },
        "wall_s_per_step": wall / max(summary.steps, 1),
    }
    return ("timeout" if timed_out else "done"), m
