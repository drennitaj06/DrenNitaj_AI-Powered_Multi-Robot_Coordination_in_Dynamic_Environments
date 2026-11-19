"""Executors of the Issue #4 experiments (05 §B.4): E3a assignment solves, E3b priority-sensitivity
episodes, and the preliminary non-learned E5 slice.

Instances depend only on the experiment's ``seed_base`` / the run's seed and factor values that
define the instance, never on the method or variant under comparison, so every method sees the
identical matrix and every variant the identical map, robots and task stream (paired design,
05 §B.1). SciPy (the Hungarian oracle) is a dev/test dependency: E3a needs ``pip install -e
".[dev]"``; it is imported lazily.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from robocoord.allocation.auction import solve as auction_solve
from robocoord.allocation.base import SquareSolution
from robocoord.allocation.benefits import BenefitConfig
from robocoord.allocation.sequential import solve as sequential_solve
from robocoord.control.baseline import BaselineController
from robocoord.control.reference import ReferenceFollower
from robocoord.core.types import Cell
from robocoord.experiments.spec import ExperimentSpec
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.planning.endpoint import EndpointParams
from robocoord.scenario.generators import random_density
from robocoord.scenario.spec import (
    ArrivalSpec,
    MapEventSpec,
    ObstacleSpec,
    ScenarioSpec,
    TaskSpec,
)
from robocoord.sim.engine import Engine

E3A_METHODS = ("auction_scaling", "auction_fixed1", "sequential", "hungarian")


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


# ================================================================== E3a: assignment solves


def e3a_matrix(seed_base: int, k: int, c: int, index: int, density: float = 1.0) -> np.ndarray:
    """A k × k integer benefit matrix with entries uniform in {1, …, C}; a fraction ``1 −
    density`` of the entries is set to 0 (infeasible), as in padded benefit problems. Depends
    only on ``(seed_base, k, C, index, density)``."""
    rng = _rng(seed_base, k, c, index, round(density * 1000))
    a = rng.integers(1, c + 1, size=(k, k), dtype=np.int64)
    if density < 1.0:
        a[rng.random((k, k)) >= density] = 0
    return a


def _hungarian(a: np.ndarray) -> tuple[int, int]:
    # SciPy 1.18.1 ships no type stubs (no scipy-stubs installed by policy): oracle only
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

    r, c = linear_sum_assignment(a, maximize=True)
    return int(a[r, c].sum()), int(np.count_nonzero(a[r, c]))


def run_assignment(spec: ExperimentSpec, params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """E3a: solve one matrix with one method; the optimum comes from the Hungarian oracle."""
    k, c, index = int(params["k"]), int(params["C"]), int(params["matrix"])
    method = str(params["method"])
    density = float(spec.fixed.get("density", 1.0))
    repeats = int(spec.fixed.get("repeats", 3))
    cap = float(spec.fixed.get("solve_cap_s", 60.0))
    a = e3a_matrix(spec.seed_base, k, c, index, density)
    opt, opt_card = _hungarian(a)
    times: list[float] = []
    value, card, bids, phases, timed_out = 0, 0, 0, 0, False
    for _ in range(repeats):
        t0 = time.perf_counter()
        if method == "hungarian":
            value, card = _hungarian(a)
        else:
            sol: SquareSolution
            if method == "auction_scaling":
                sol, _ = auction_solve(a, scaling=True, time_limit_s=cap)
            elif method == "auction_fixed1":
                sol, _ = auction_solve(a, scaling=False, epsilon=1, time_limit_s=cap)
            elif method == "sequential":
                sol = sequential_solve(a)
            else:
                raise ValueError(f"unknown E3a method {method!r}")
            timed_out = sol.stats.timed_out
            bids, phases = sol.stats.bids, sol.stats.phases
            if sol.assignment is not None:
                value = sol.value
                card = int(np.count_nonzero(a[np.arange(k), np.asarray(sol.assignment)]))
        times.append(time.perf_counter() - t0)
        if timed_out:
            break
    m = {
        "method": method,
        "k": k,
        "C": c,
        "density": float(np.count_nonzero(a)) / a.size,
        "opt": opt,
        "value": value,
        "gap": opt - value,
        "ratio": value / opt if opt else None,
        "cardinality": card,
        "max_cardinality": opt_card,
        "bids": bids,
        "phases": phases,
        "time_s": float(np.median(times)),
        "time_all_s": times,
    }
    return ("timeout" if timed_out else "done"), m


# ================================================================== scenarios (E3b, E5)


def _robots_and_free(grid: Any, seed: int, n: int) -> tuple[list[Cell], list[Cell]]:
    free = grid.free_cells()
    place = _rng(seed, 2, n)
    robots = [free[int(i)] for i in place.choice(len(free), size=n, replace=False)]
    return robots, free


def e3b_scenario(
    seed: int,
    *,
    size: int,
    density: float,
    robots: int,
    rate: float,
    arrivals_until: int,
    max_steps: int,
    priorities: tuple[float, ...] = (0.5, 0.3, 0.2),
) -> ScenarioSpec:
    """Scarce-robot suite: N robots on a random map, tasks arriving by a Bernoulli process with
    three priority levels. The map, robot starts and the arrival stream depend only on
    ``seed`` (the arrival stream is drawn by the engine from the scenario seed)."""
    grid = random_density(size, size, density, _rng(seed, 1))
    starts, _ = _robots_and_free(grid, seed, robots)
    spec = ScenarioSpec(
        name=f"e3b-s{seed}",
        grid=grid,
        robots=tuple(starts),
        seed=seed,
        max_steps=max_steps,
        num_priorities=len(priorities),
        arrivals=ArrivalSpec("bernoulli", rate, priorities, arrivals_until),
    )
    spec.validate()
    return spec


def e5_scenario(
    seed: int,
    dynamics: str,
    *,
    size: int,
    density: float,
    robots: int,
    tasks_per_robot: int,
    obstacles: int,
    max_steps: int,
    arrival_rate: float,
    arrivals_until: int,
    n_events: int = 6,
    event_duration: int = 20,
) -> ScenarioSpec:
    """E5 instance for ``dynamics`` ∈ {static, obstacles, full}: the same map, robots and
    scheduled tasks for every level; ``obstacles`` adds moving obstacles; ``full`` adds map
    events and Bernoulli arrivals (separate random streams, so lower levels are unchanged)."""
    if dynamics not in ("static", "obstacles", "full"):
        raise ValueError(f"unknown dynamics {dynamics!r}")
    grid = random_density(size, size, density, _rng(seed, 1))
    starts, free = _robots_and_free(grid, seed, robots)
    place = _rng(seed, 5)
    tasks: list[TaskSpec] = []
    for _ in range(tasks_per_robot * robots):
        a, b = place.choice(len(free), size=2, replace=False)
        tasks.append(TaskSpec(free[int(a)], free[int(b)], int(place.integers(1, 4)), 0))
    obs: list[ObstacleSpec] = []
    if dynamics in ("obstacles", "full"):
        obs_rng = _rng(seed, 3)
        candidates = [c for c in free if c not in set(starts)]
        for j, i in enumerate(obs_rng.choice(len(candidates), size=obstacles, replace=False)):
            cell = candidates[int(i)]
            if j % 2 == 0:
                obs.append(ObstacleSpec(cell, "random_walk"))
            else:
                far = free[int(obs_rng.integers(len(free)))]
                obs.append(ObstacleSpec(cell, "patrol", (cell, far)))
    events: list[MapEventSpec] = []
    arrivals = ArrivalSpec()
    if dynamics == "full":
        ev_rng = _rng(seed, 4)
        for _ in range(n_events):
            t0 = int(ev_rng.integers(5, max_steps // 2))
            cell = free[int(ev_rng.integers(len(free)))]
            events.append(MapEventSpec(t0, cell, True))
            events.append(MapEventSpec(t0 + event_duration, cell, False))
        events.sort(key=lambda e: (e.time, e.cell, not e.block))
        arrivals = ArrivalSpec("bernoulli", arrival_rate, (0.5, 0.3, 0.2), arrivals_until)
    spec = ScenarioSpec(
        name=f"e5-s{seed}-{dynamics}",
        grid=grid,
        robots=tuple(starts),
        seed=seed,
        max_steps=max_steps,
        obstacles=tuple(obs),
        tasks=tuple(tasks),
        map_events=tuple(events),
        arrivals=arrivals,
    )
    spec.validate()
    return spec


# ================================================================== episodes


def _run(engine: Engine, ctrl: Any, cap_s: float) -> tuple[bool, int, float]:
    """Step to the end; returns (timed out, robot moves, wall seconds)."""
    t0 = time.perf_counter()
    moves = 0
    ctrl.reset(len(engine.robots))
    while not engine.done:
        before = [r.position for r in engine.robots]
        engine.step(ctrl.act)
        moves += sum(r.position != p for r, p in zip(engine.robots, before, strict=True))
        if time.perf_counter() - t0 > cap_s:
            return True, moves, time.perf_counter() - t0
    return False, moves, time.perf_counter() - t0


def _episode_metrics(engine: Engine, moves: int, wall: float) -> dict[str, Any]:
    s = engine.summary()
    planning = s.extra.get("planning", {})
    delivered = s.tasks_delivered
    return {
        "steps": s.steps,
        "terminated": engine.terminated,
        "tasks_released": s.tasks_released,
        "tasks_delivered": delivered,
        "completion_rate": s.completion_rate,
        "makespan": s.makespan,
        "mean_flow_time": s.mean_flow_time,
        "weighted_flow_time": s.weighted_flow_time,
        "flow_time_by_priority": s.flow_time_by_priority,
        "waiting_max_by_priority": s.waiting_max_by_priority,
        "waiting_p95_by_priority": s.waiting_p95_by_priority,
        "waiting_censored": s.waiting_censored,
        "robot_moves": moves,
        "travel_per_delivered_task": moves / delivered if delivered else None,
        "utilisation": s.utilisation,
        "wait_ratio": s.wait_ratio,
        "conflicts": s.conflicts,
        "conflicts_per_100": s.conflicts_per_100_robot_steps,
        "blocked_moves": s.blocked_moves,
        "reference_conflicts": s.reference_conflicts,
        "conflicts_avoided": s.conflicts_avoided,
        "conflicts_introduced": s.conflicts_introduced,
        "conflicts_persisted": s.conflicts_persisted,
        "avoidance_rate": s.avoidance_rate,
        "plan_adherence": s.plan_adherence,
        "path_efficiency": s.path_efficiency,
        "physical_collisions": s.physical_collisions,
        "calls_by_trigger": planning.get("calls_by_trigger"),
        "deviation_rate_per_100": planning.get("deviation_rate_per_100"),
        "allocation": planning.get("allocation"),
        "deadlock": planning.get("deadlock"),
        "endpoint": planning.get("endpoint"),
        "controller": s.extra.get("controller"),
        "wall_s_per_step": wall / max(s.steps, 1),
    }


#: E3b variants: (κ, α, h) — default model §11 values and the three single-factor ablations.
E3B_VARIANTS: dict[str, tuple[float, float, int]] = {
    "default": (10.0, 0.01, 2),
    "kappa0": (0.0, 0.01, 2),
    "alpha0": (10.0, 0.0, 2),
    "h0": (10.0, 0.01, 0),
}


def run_allocation_episode(
    spec: ExperimentSpec, params: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """E3b: one scarce-robot episode, A* + auction + baseline (resolver on, its default)."""
    fixed = spec.fixed
    seed, variant = int(params["seed"]), str(params["variant"])
    kappa, alpha, h = E3B_VARIANTS[variant]
    scenario = e3b_scenario(
        seed,
        size=int(fixed.get("size", 24)),
        density=float(fixed.get("density", 0.15)),
        robots=int(fixed.get("robots", 4)),
        rate=float(fixed["arrival_rate"]),
        arrivals_until=int(fixed.get("arrivals_until", 200)),
        max_steps=int(fixed.get("max_steps", 400)),
    )
    from robocoord.allocation.auction import AuctionAllocator

    coord = AllocationCoordinator(
        make_planner("astar"),
        AuctionAllocator(),
        benefit_config=BenefitConfig(kappa=kappa, hysteresis=h, alpha=alpha),
        deadlock=DetourParams(alpha=alpha),
    )
    engine = Engine(scenario, coord)
    ctrl = BaselineController(alpha=alpha)
    engine.controller_stats = ctrl.episode_stats
    timed_out, moves, wall = _run(engine, ctrl, spec.timeout_s)
    m = _episode_metrics(engine, moves, wall)
    m.update({"variant": variant, "kappa": kappa, "alpha": alpha, "hysteresis": h})
    return ("timeout" if timed_out else "done"), m


def system_episode_setup(
    spec: ExperimentSpec, params: Mapping[str, Any]
) -> tuple[Engine, AllocationCoordinator, BaselineController | ReferenceFollower]:
    """Build the engine, coordinator and controller of one preliminary-E5 run (shared by the
    executor and the deterministic replays of :mod:`robocoord.experiments.e5_diagnostics`).

    ``fixed.controller`` (default ``"baseline"``) selects the controller with its documented CLI
    defaults: ``baseline`` = right-of-way baseline with the deadlock resolver on;
    ``reference`` = reference follower, resolver off. The shield is off for both. Everything
    else (generator, seeds, metrics, caps) is identical, so runs of the two controllers on the
    same seed are paired.

    ``fixed.endpoint`` (``"on"`` / ``"off"``) enables endpoint recovery (algorithms §5.6) for
    either controller. An experiment file **without** the key means ``"off"``: the E5 record
    stores produced before endpoint recovery existed (``e5_prelim``, ``e5_prelim_reference``)
    stay reproducible by replay. New configurations state the key explicitly.
    """
    controller = str(spec.fixed.get("controller", "baseline"))
    if controller not in ("baseline", "reference"):
        raise ValueError(f"unknown E5 controller {controller!r}")
    endpoint = str(spec.fixed.get("endpoint", "off"))
    if endpoint not in ("on", "off"):
        raise ValueError(f"fixed.endpoint must be 'on' or 'off', got {endpoint!r}")
    fixed = spec.fixed
    scenario = e5_scenario(
        int(params["seed"]),
        str(params["dynamics"]),
        size=int(fixed.get("size", 32)),
        density=float(fixed.get("density", 0.15)),
        robots=int(fixed.get("robots", 8)),
        tasks_per_robot=int(fixed.get("tasks_per_robot", 2)),
        obstacles=int(fixed.get("obstacles", 8)),
        max_steps=int(fixed.get("max_steps", 400)),
        arrival_rate=float(fixed.get("arrival_rate", 0.05)),
        arrivals_until=int(fixed.get("arrivals_until", 200)),
    )
    setting = spec.planners[str(params["planner"])]
    from robocoord.allocation.auction import AuctionAllocator
    from robocoord.allocation.sequential import SequentialAllocator

    alloc = AuctionAllocator() if params["allocator"] == "auction" else SequentialAllocator()
    coord = AllocationCoordinator(
        make_planner(setting.name, **dict(setting.options)),
        alloc,
        deadlock=DetourParams() if controller == "baseline" else None,
        endpoint=EndpointParams() if endpoint == "on" else None,
    )
    engine = Engine(scenario, coord)
    ctrl: BaselineController | ReferenceFollower = (
        BaselineController() if controller == "baseline" else ReferenceFollower()
    )
    engine.controller_stats = ctrl.episode_stats
    return engine, coord, ctrl


def run_system_episode(
    spec: ExperimentSpec, params: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Preliminary E5: planner × allocator × dynamics for one non-learned controller (see
    :func:`system_episode_setup`)."""
    engine, _, ctrl = system_episode_setup(spec, params)
    timed_out, moves, wall = _run(engine, ctrl, spec.timeout_s)
    m = _episode_metrics(engine, moves, wall)
    m.update(
        {
            "planner": spec.planners[str(params["planner"])].name,
            "allocator": str(params["allocator"]),
            "controller": str(spec.fixed.get("controller", "baseline")),
        }
    )
    return ("timeout" if timed_out else "done"), m
