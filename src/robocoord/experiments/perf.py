"""NFR-PERF-1 benchmark (01 requirements; 05 §B.2): wall time of one headless simulation step,
8 robots, 32 × 32 grid, A*; target < 20 ms on a typical laptop CPU (no RL inference exists).

Methodology (reproducible, documented in `05` §B.6):

* **Workload.** The E5 "full" generator (the most dynamic non-learned setting): 32 × 32 random
  map with 15 % obstacles, 8 robots, 16 scheduled tasks, 8 moving obstacles, map events and
  Bernoulli arrivals. Seeds from the micro-benchmark range 30 500 – 30 509 (not test seeds).
* **Pipeline measured in full:** A* + auction allocator (and, separately, sequential) +
  right-of-way baseline + deadlock resolver + predictive shield + conflict detection,
  arbitration, metrics. Traces are off (a headless step); no component is skipped.
* **Procedure.** One unmeasured warm-up episode, then one natural episode per seed (it ends
  when every task is delivered and no arrivals remain, or at ``steps``). ``time.perf_counter``
  around each ``Engine.step``; component times by wrapping the allocation policy, the path
  monitor, the deadlock resolver, the controller and the shield (the rest = engine: views,
  conflict detection, arbitration, metrics). Single process, no other load started by the
  benchmark.
* **Reported.** Median, p95, p99 and maximum step time over all steps and, separately, over
  *busy* steps (at least one robot holds a task), so idle steps cannot flatter the result; the
  fraction of steps under 20 ms; the per-component median / p95 / mean; and the mean component
  share of the slowest 1 % of steps (where spikes come from). Hardware and software from
  ``environment_info()``.
* **Raw evidence (added after the Issue #4 audit; workload and timing unchanged).** Every
  measured step is written to ``nfr_perf1_samples.csv`` (seed, t, total and component ms, map
  event, BFS fields computed). The aggregates add first-step, map-event-step and steady-state
  statistics and the exceedance count by kind. The step index, map-event flag and BFS counter
  are read outside the timed region. An existing result is never replaced unless
  ``--overwrite`` is given.

* **Workloads R and F (added with endpoint recovery and the distance-field optimisation).**
  :func:`run_workloads` measures, with the same procedure and seeds:

  - **R** (requirement-focused): 8 robots, 32 × 32, A*, headless, the system defaults with an
    allocator (baseline controller, deadlock resolver on, endpoint recovery on, shield off),
    each allocator, ``static`` (no moving obstacles) and ``obstacles`` (8 moving obstacles);
  - **F** (full system): ``full`` dynamics (moving obstacles, map events, arrivals), predictive
    shield, endpoint recovery on, resolver as the CLI default (on for the baseline),
    planner {A*, RRT* with the E5 budget n_max = 1 000} × allocator × controller {baseline,
    reference}.

  Every step is timed and kept; the statistics add p90 and the number and percentage of steps
  ≥ 20 ms. The distance implementation before (``legacy``) and after (``fast``) optimisation
  O1/O2 are measured interleaved, seed by seed, in one process, and the AC/battery state is
  recorded at the start and end (battery power slows every step several-fold). Each sample
  carries a digest of the state after the step (robot and obstacle positions, modes, task
  statuses), read outside the timed region, so the two implementations can be checked to follow
  identical trajectories step by step.

Run: ``python -m robocoord.experiments.perf --out results/perf`` (workloads R and F;
``--legacy`` for the original Issue #4 workload).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.base import Allocator
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.control.baseline import BaselineController
from robocoord.control.reference import ReferenceFollower
from robocoord.control.shield import Shield
from robocoord.core.distance import legacy_distance
from robocoord.experiments.allocation_runs import e5_scenario
from robocoord.experiments.records import environment_info
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.planning.endpoint import EndpointParams
from robocoord.sim.engine import Engine

TARGET_MS = 20.0
SEEDS = tuple(range(30500, 30510))


def _timed(store: dict[str, float], name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            store[name] += time.perf_counter() - t0

    return wrapper


def _episode(seed: int, allocator: Allocator, steps: int) -> dict[str, list[float]]:
    scenario = e5_scenario(
        seed,
        "full",
        size=32,
        density=0.15,
        robots=8,
        tasks_per_robot=2,
        obstacles=8,
        max_steps=steps,
        arrival_rate=0.05,
        arrivals_until=200,
    )
    coord = AllocationCoordinator(make_planner("astar"), allocator, deadlock=DetourParams())
    ctrl = BaselineController()
    shield = Shield("predictive")
    engine = Engine(scenario, coord, shield=shield)
    ctrl.reset(len(engine.robots))
    acc: dict[str, float] = defaultdict(float)
    assert coord.monitor is not None and coord.resolver is not None
    coord.policy.step = _timed(acc, "allocation", coord.policy.step)  # type: ignore[method-assign]
    coord.monitor.update = _timed(acc, "planning", coord.monitor.update)  # type: ignore[method-assign]
    coord.resolver.overrides = _timed(acc, "deadlock", coord.resolver.overrides)  # type: ignore[method-assign]
    coord.resolver.observe = _timed(acc, "deadlock", coord.resolver.observe)  # type: ignore[method-assign]
    shield.apply = _timed(acc, "shield", shield.apply)  # type: ignore[method-assign]
    act = _timed(acc, "controller", ctrl.act)
    out: dict[str, list[float]] = defaultdict(list)
    while not engine.done:
        acc.clear()
        busy = any(r.task is not None for r in engine.robots)
        t_step, fields_before = engine.t, engine.oracle.misses  # read outside the timed region
        t0 = time.perf_counter()
        outcome = engine.step(act)
        total = time.perf_counter() - t0
        out["total"].append(total)
        out["busy"].append(1.0 if busy else 0.0)
        out["t"].append(float(t_step))
        out["map_event"].append(1.0 if outcome.map_events else 0.0)
        out["bfs_fields"].append(float(engine.oracle.misses - fields_before))
        parts = 0.0
        for name in ("allocation", "planning", "deadlock", "controller", "shield"):
            out[name].append(acc[name])
            parts += acc[name]
        out["engine_rest"].append(total - parts)
    return out


COMPONENTS = ("allocation", "planning", "deadlock", "controller", "shield", "engine_rest")


def _stats(a: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {"steps": 0}
    return {
        "steps": int(a.size),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.quantile(a, 0.90)),
        "p95_ms": float(np.quantile(a, 0.95)),
        "p99_ms": float(np.quantile(a, 0.99)),
        "max_ms": float(a.max()),
        "fraction_under_target": float(np.mean(a < TARGET_MS)),
        "exceedances": int(np.sum(a >= TARGET_MS)),
        "exceedance_pct": float(100.0 * np.mean(a >= TARGET_MS)),
    }


# ================================================================== workloads R and F


@dataclass(frozen=True, slots=True)
class PerfConfig:
    """One benchmark configuration (see the module docstring)."""

    workload: str
    label: str
    dynamics: str
    planner: str = "astar"
    allocator: str = "auction"
    controller: str = "baseline"
    shield: bool = False
    endpoint: bool = True


WORKLOADS: dict[str, tuple[PerfConfig, ...]] = {
    "R": tuple(
        PerfConfig("R", f"{dyn}/{alloc}", dyn, allocator=alloc)
        for dyn in ("static", "obstacles")
        for alloc in ("auction", "sequential")
    ),
    "F": tuple(
        PerfConfig("F", f"{pl}/{alloc}/{ctrl}", "full", pl, alloc, ctrl, shield=True)
        for pl in ("astar", "rrtstar")
        for alloc in ("auction", "sequential")
        for ctrl in ("baseline", "reference")
    ),
}


def _digest(engine: Engine) -> int:
    """CRC-32 of the state after a step (positions, modes, task statuses); deterministic."""
    state = (
        tuple((r.position, r.mode.value, r.task) for r in engine.robots),
        tuple(o.position for o in engine.obstacles),
        tuple((tk.id, tk.status.value) for tk in engine.book.tasks.values()),
        engine.grid.version,
    )
    return zlib.crc32(repr(state).encode("utf-8"))


def _config_episode(cfg: PerfConfig, seed: int, steps: int) -> dict[str, list[float]]:
    scenario = e5_scenario(
        seed,
        cfg.dynamics,
        size=32,
        density=0.15,
        robots=8,
        tasks_per_robot=2,
        obstacles=8,
        max_steps=steps,
        arrival_rate=0.05,
        arrivals_until=200,
    )
    alloc: Allocator = AuctionAllocator() if cfg.allocator == "auction" else SequentialAllocator()
    options = {"max_iterations": 1000} if cfg.planner == "rrtstar" else {}  # as in E5 (n_max)
    coord = AllocationCoordinator(
        make_planner(cfg.planner, **options),
        alloc,
        deadlock=DetourParams() if cfg.controller == "baseline" else None,  # CLI --deadlock auto
        endpoint=EndpointParams() if cfg.endpoint else None,
    )
    ctrl: BaselineController | ReferenceFollower = (
        BaselineController() if cfg.controller == "baseline" else ReferenceFollower()
    )
    shield = Shield("predictive") if cfg.shield else None
    engine = Engine(scenario, coord, shield=shield)
    ctrl.reset(len(engine.robots))
    acc: dict[str, float] = defaultdict(float)
    assert coord.monitor is not None
    coord.policy.step = _timed(acc, "allocation", coord.policy.step)  # type: ignore[method-assign]
    coord.monitor.update = _timed(acc, "planning", coord.monitor.update)  # type: ignore[method-assign]
    if coord.resolver is not None:
        coord.resolver.overrides = _timed(acc, "deadlock", coord.resolver.overrides)  # type: ignore[method-assign]
        coord.resolver.observe = _timed(acc, "deadlock", coord.resolver.observe)  # type: ignore[method-assign]
    if coord.endpoint is not None:
        coord.endpoint.overrides = _timed(acc, "endpoint", coord.endpoint.overrides)  # type: ignore[method-assign]
        coord.endpoint.observe = _timed(acc, "endpoint", coord.endpoint.observe)  # type: ignore[method-assign]
    if shield is not None:
        shield.apply = _timed(acc, "shield", shield.apply)  # type: ignore[method-assign]
    act = _timed(acc, "controller", ctrl.act)
    out: dict[str, list[float]] = defaultdict(list)
    while not engine.done:
        acc.clear()
        busy = any(r.task is not None for r in engine.robots)
        t_step, fields_before = engine.t, engine.oracle.misses  # read outside the timed region
        t0 = time.perf_counter()
        outcome = engine.step(act)
        total = time.perf_counter() - t0
        out["total"].append(total)
        out["busy"].append(1.0 if busy else 0.0)
        out["t"].append(float(t_step))
        out["map_event"].append(1.0 if outcome.map_events else 0.0)
        out["bfs_fields"].append(float(engine.oracle.misses - fields_before))
        out["state"].append(float(_digest(engine)))
        parts = 0.0
        for name in CONFIG_COMPONENTS[:-1]:
            out[name].append(acc[name])
            parts += acc[name]
        out["engine_rest"].append(total - parts)
    s = engine.summary()
    out["delivered"] = [float(s.tasks_delivered)]
    out["released"] = [float(s.tasks_released)]
    out["physical"] = [float(s.physical_collisions)]
    return out


CONFIG_COMPONENTS = (
    "allocation",
    "planning",
    "deadlock",
    "endpoint",
    "controller",
    "shield",
    "engine_rest",
)


#: Distance implementations a workload run can measure: ``legacy`` = before optimisation O1/O2
#: (:func:`~robocoord.core.distance.legacy_distance`), ``fast`` = after.
IMPLS: tuple[str, ...] = ("legacy", "fast")


def power_state() -> dict[str, Any]:
    """AC line and battery state (Windows ``GetSystemPowerStatus``; ``unknown`` elsewhere).
    Recorded with every workload run: step times on battery power are several times slower."""
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        status = _Status()
        kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 (absent off Windows)
        if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return {"ac_line": "unknown"}
        ac = {0: "offline (battery)", 1: "online"}.get(status.ACLineStatus, "unknown")
        battery = status.BatteryLifePercent if status.BatteryLifePercent <= 100 else None
        return {"ac_line": ac, "battery_percent": battery}
    except (AttributeError, OSError):  # not Windows
        return {"ac_line": "unknown"}


def _aggregate(
    cfg: PerfConfig, series: dict[str, list[float]], outcome: dict[str, int]
) -> dict[str, Any]:
    total = np.asarray(series["total"]) * 1e3
    first = np.asarray(series["t"]) == 0
    event = (np.asarray(series["map_event"]) > 0) & ~first
    steady = ~first & ~event
    comp = {}
    for c in CONFIG_COMPONENTS:
        a = np.asarray(series[c]) * 1e3
        comp[c] = {
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.quantile(a, 0.95)),
            "max_ms": float(a.max()),
        }
    bfs = np.asarray(series["bfs_fields"])
    return {
        "config": {
            "dynamics": cfg.dynamics,
            "planner": cfg.planner,
            "allocator": cfg.allocator,
            "controller": cfg.controller,
            "shield": cfg.shield,
            "endpoint": cfg.endpoint,
        },
        "all_steps": _stats(total),
        "first_steps": _stats(total[first]),
        "map_event_steps": _stats(total[event]),
        "steady_steps": _stats(total[steady]),
        "bfs_fields_mean": {
            "first_steps": float(bfs[first].mean()) if first.any() else 0.0,
            "map_event_steps": float(bfs[event].mean()) if event.any() else 0.0,
            "steady_steps": float(bfs[steady].mean()) if steady.any() else 0.0,
        },
        "components": comp,
        "outcome": outcome,
    }


def run_workloads(
    names: tuple[str, ...] = ("R", "F"),
    steps: int = 400,
    seeds: tuple[int, ...] = SEEDS,
    impls: tuple[str, ...] = ("fast",),
) -> dict[str, Any]:
    """Workloads R and F (module docstring); every step is kept in ``samples``.

    With several ``impls`` (``("legacy", "fast")``) every configuration and seed is run once per
    implementation, **interleaved** in the same process (the order alternates from seed to
    seed), so that before and after are measured under the same machine conditions; the state
    digests then show whether both followed identical trajectories.
    """
    for impl in impls:
        if impl not in IMPLS:
            raise ValueError(f"unknown distance implementation {impl!r}")
    results: dict[str, Any] = {
        "target_ms": TARGET_MS,
        "steps_per_episode": steps,
        "seeds": list(seeds),
        "impls": list(impls),
        "environment": environment_info(),
        "power_start": power_state(),
        "runs": {impl: {name: {} for name in names} for impl in impls},
        "trajectories_identical": {},
        "samples": [],
    }
    for name in names:
        for cfg in WORKLOADS[name]:
            for impl in impls:  # warm-up, not measured
                with legacy_distance(impl == "legacy"):
                    _config_episode(cfg, seeds[0], min(steps, 60))
            series: dict[str, dict[str, list[float]]] = {i: defaultdict(list) for i in impls}
            outcome = {i: {"delivered": 0, "released": 0, "physical_collisions": 0} for i in impls}
            states: dict[str, dict[int, list[float]]] = {i: {} for i in impls}
            for k, seed in enumerate(seeds):
                for impl in impls if k % 2 == 0 else tuple(reversed(impls)):
                    with legacy_distance(impl == "legacy"):
                        ep = _config_episode(cfg, seed, steps)
                    o = outcome[impl]
                    o["delivered"] += int(ep.pop("delivered")[0])
                    o["released"] += int(ep.pop("released")[0])
                    o["physical_collisions"] += int(ep.pop("physical")[0])
                    states[impl][seed] = list(ep["state"])
                    ep["seed"] = [float(seed)] * len(ep["total"])
                    for key, vals in ep.items():
                        series[impl][key] += vals
            for impl in impls:
                s = series[impl]
                results["runs"][impl][name][cfg.label] = _aggregate(cfg, s, outcome[impl])
                for i in range(len(s["total"])):
                    results["samples"].append(
                        {
                            "impl": impl,
                            "workload": name,
                            "config": cfg.label,
                            "seed": int(s["seed"][i]),
                            "t": int(s["t"][i]),
                            "total_ms": float(s["total"][i] * 1e3),
                            **{f"{c}_ms": float(s[c][i] * 1e3) for c in CONFIG_COMPONENTS},
                            "map_event": bool(s["map_event"][i] > 0),
                            "bfs_fields": int(s["bfs_fields"][i]),
                            "busy": bool(s["busy"][i] > 0),
                            "state": int(s["state"][i]),
                        }
                    )
            if len(impls) > 1:
                ref = states[impls[0]]
                results["trajectories_identical"][f"{name} {cfg.label}"] = all(
                    states[i] == ref for i in impls[1:]
                )
    results["power_end"] = power_state()
    return results


def workloads_markdown(res: dict[str, Any]) -> str:
    env = res["environment"]
    lines = [
        f"NFR-PERF-1 workloads (target < {res['target_ms']:.0f} ms per headless step, every "
        f"step measured). {len(res['seeds'])} natural episodes (cap {res['steps_per_episode']} "
        "steps) per configuration and distance implementation after one unmeasured warm-up "
        f"episode; implementations {res['impls']} interleaved seed by seed.",
        "",
        f"Environment: {json.dumps(env, sort_keys=True)}",
        f"Power at start: {res['power_start']}; at end: {res['power_end']}",
        "",
        "| impl | workload | configuration | steps | n | median | p90 | p95 | p99 | max "
        "| ≥ 20 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for impl, workloads in res["runs"].items():
        for name, configs in workloads.items():
            for label, c in configs.items():
                for which in ("all_steps", "first_steps", "map_event_steps", "steady_steps"):
                    s = c[which]
                    if not s.get("steps"):
                        continue
                    lines.append(
                        f"| {impl} | {name} | {label} | {which.replace('_steps', '')} | "
                        f"{s['steps']} | {s['median_ms']:.2f} | {s['p90_ms']:.2f} | "
                        f"{s['p95_ms']:.2f} | {s['p99_ms']:.2f} | {s['max_ms']:.2f} | "
                        f"{s['exceedances']} ({s['exceedance_pct']:.2f} %) |"
                    )
    lines += ["", "Outcome per configuration (delivered / released, physical collisions):", ""]
    for impl, workloads in res["runs"].items():
        for name, configs in workloads.items():
            for label, c in configs.items():
                o = c["outcome"]
                lines.append(
                    f"- {impl} {name} {label}: {o['delivered']}/{o['released']}, physical "
                    f"{o['physical_collisions']}"
                )
    if res["trajectories_identical"]:
        same = res["trajectories_identical"]
        lines += [
            "",
            f"Identical state digests at every step across implementations: "
            f"{sum(same.values())} of {len(same)} configurations"
            + ("" if all(same.values()) else f" (differ: {[k for k, v in same.items() if not v]})"),
        ]
    return "\n".join(lines)


def write_workload_samples(res: dict[str, Any], path: Path) -> None:
    fields = [
        "impl",
        "workload",
        "config",
        "seed",
        "t",
        "total_ms",
        *(f"{c}_ms" for c in CONFIG_COMPONENTS),
        "map_event",
        "bfs_fields",
        "busy",
        "state",
    ]
    rows = [",".join(fields)]
    rows += [",".join(str(s[f]) for f in fields) for s in res["samples"]]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(steps: int = 400, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    results: dict[str, Any] = {
        "target_ms": TARGET_MS,
        "steps_per_episode": steps,
        "seeds": list(seeds),
        "environment": environment_info(),
        "configurations": {},
    }
    results["samples"] = {}
    for label, make in (("auction", AuctionAllocator), ("sequential", SequentialAllocator)):
        _episode(seeds[0], make(), min(steps, 60))  # warm-up, not measured
        series: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            ep = _episode(seed, make(), steps)
            ep["seed"] = [float(seed)] * len(ep["total"])
            for name, vals in ep.items():
                series[name] += vals
        total = np.asarray(series["total"]) * 1e3
        busy = np.asarray(series["busy"]) > 0
        first = np.asarray(series["t"]) == 0
        event = np.asarray(series["map_event"]) > 0
        steady = ~first & ~event
        over = total >= TARGET_MS
        results["samples"][label] = [
            {
                "seed": int(series["seed"][i]),
                "t": int(series["t"][i]),
                "total_ms": float(total[i]),
                **{f"{c}_ms": float(series[c][i] * 1e3) for c in COMPONENTS},
                "map_event": bool(event[i]),
                "bfs_fields": int(series["bfs_fields"][i]),
                "busy": bool(busy[i]),
            }
            for i in range(total.size)
        ]
        comp = {}
        for name in ("total", *COMPONENTS):
            a = np.asarray(series[name]) * 1e3
            comp[name] = {
                "median_ms": float(np.median(a)),
                "p95_ms": float(np.quantile(a, 0.95)),
                "mean_ms": float(a.mean()),
            }
        slow = total >= np.quantile(total, 0.99)
        spike_share = {
            name: float((np.asarray(series[name]) * 1e3)[slow].sum() / total[slow].sum())
            for name in COMPONENTS
        }
        all_steps = _stats(total)
        bfs = np.asarray(series["bfs_fields"])
        results["configurations"][label] = {
            "all_steps": all_steps,
            "busy_steps": _stats(total[busy]),
            "first_steps": _stats(total[first]),
            "map_event_steps": _stats(total[event & ~first]),
            "steady_steps": _stats(total[steady]),
            "exceedances": {
                "total": int(over.sum()),
                "first_step": int((over & first).sum()),
                "map_event_step": int((over & event & ~first).sum()),
                "other": int((over & steady).sum()),
            },
            "bfs_fields_mean": {
                "first_steps": float(bfs[first].mean()) if first.any() else 0.0,
                "map_event_steps": float(bfs[event & ~first].mean())
                if (event & ~first).any()
                else 0.0,
                "steady_steps": float(bfs[steady].mean()) if steady.any() else 0.0,
            },
            "bfs_time_correlation": float(np.corrcoef(bfs, total)[0, 1]) if total.size > 1 else 0.0,
            "meets_target_median": all_steps["median_ms"] < TARGET_MS,
            "meets_target_p99": all_steps["p99_ms"] < TARGET_MS,
            "meets_target_max": all_steps["max_ms"] < TARGET_MS,
            "components": comp,
            "slowest_1pct_share": spike_share,
        }
    return results


def to_markdown(res: dict[str, Any]) -> str:
    env = res["environment"]
    lines = [
        f"NFR-PERF-1 (target < {res['target_ms']:.0f} ms per headless step; 8 robots, 32 × 32, "
        f"A*). {len(res['seeds'])} natural episodes (cap {res['steps_per_episode']} steps) per "
        "configuration, after one unmeasured warm-up episode.",
        "",
        f"Environment: {json.dumps(env, sort_keys=True)}",
        "",
        "| allocator | steps | n | median ms | p95 ms | p99 ms | max ms | < 20 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, c in res["configurations"].items():
        for which in ("all_steps", "busy_steps", "first_steps", "map_event_steps", "steady_steps"):
            s = c[which]
            lines.append(
                f"| {label} | {which.replace('_', ' ')} | {s['steps']} | {s['median_ms']:.2f} | "
                f"{s['p95_ms']:.2f} | {s['p99_ms']:.2f} | {s['max_ms']:.2f} | "
                f"{100 * s['fraction_under_target']:.1f} % |"
            )
    lines += ["", "Components (median / p95 / mean ms per step):", ""]
    lines += ["| allocator | component | median | p95 | mean |", "|---|---|---|---|---|"]
    for label, c in res["configurations"].items():
        for name, s in c["components"].items():
            lines.append(
                f"| {label} | {name} | {s['median_ms']:.3f} | {s['p95_ms']:.3f} | "
                f"{s['mean_ms']:.3f} |"
            )
    lines += ["", "Share of the slowest 1 % of steps by component:", ""]
    for label, c in res["configurations"].items():
        shares = ", ".join(f"{k} {100 * v:.0f} %" for k, v in c["slowest_1pct_share"].items())
        lines.append(f"- {label}: {shares}")
    lines += [
        "",
        "Steps ≥ 20 ms by kind (first step = cold distance-field cache; map-event step = every "
        "cached field invalidated); mean BFS fields computed per step:",
        "",
    ]
    for label, c in res["configurations"].items():
        e, b = c["exceedances"], c["bfs_fields_mean"]
        lines.append(
            f"- {label}: {e['total']} steps ≥ 20 ms = {e['first_step']} first steps + "
            f"{e['map_event_step']} map-event steps + {e['other']} other; BFS fields per step: "
            f"first {b['first_steps']:.1f}, map-event {b['map_event_steps']:.1f}, steady "
            f"{b['steady_steps']:.2f}; correlation(BFS fields, step time) "
            f"{c['bfs_time_correlation']:.3f}"
        )
    return "\n".join(lines)


def write_samples(res: dict[str, Any], path: Path) -> None:
    """Every measured step as one CSV row (raw evidence for the aggregates)."""
    fields = [
        "allocator",
        "seed",
        "t",
        "total_ms",
        *(f"{c}_ms" for c in COMPONENTS),
        "map_event",
        "bfs_fields",
        "busy",
    ]
    rows = [",".join(fields)]
    for label, samples in res["samples"].items():
        for s in samples:
            rows.append(",".join([label, *(str(s[f]) for f in fields[1:])]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NFR-PERF-1 step-time benchmark")
    parser.add_argument("--out", type=Path, default=Path("results") / "perf")
    parser.add_argument("--steps", type=int, default=400, help="episode cap (steps)")
    parser.add_argument(
        "--workload",
        default="R,F",
        help="comma-separated workloads to run (R, F; default R,F)",
    )
    parser.add_argument(
        "--impl",
        default="legacy,fast",
        help="distance implementations, interleaved per seed: legacy (before O1/O2), fast "
        "(after); default legacy,fast (paired before/after)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="run the original Issue #4 workload (results/perf, results/perf_r2) instead",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing result in --out"
    )
    args = parser.parse_args(argv)
    if (args.out / "nfr_perf1.json").exists() and not args.overwrite:
        print(f"{args.out / 'nfr_perf1.json'} exists; choose another --out or pass --overwrite")
        return 2
    if args.legacy:
        res = run(args.steps)
        args.out.mkdir(parents=True, exist_ok=True)
        write_samples(res, args.out / "nfr_perf1_samples.csv")
        md = to_markdown(res)
    else:
        names = tuple(n.strip() for n in args.workload.split(",") if n.strip())
        unknown = [n for n in names if n not in WORKLOADS]
        if unknown or not names:
            print(f"unknown workload(s) {unknown}; choose from {sorted(WORKLOADS)}")
            return 2
        impls = tuple(i.strip() for i in args.impl.split(",") if i.strip())
        if not impls or any(i not in IMPLS for i in impls):
            print(f"unknown implementation(s) in {impls}; choose from {IMPLS}")
            return 2
        res = run_workloads(names, args.steps, impls=impls)
        args.out.mkdir(parents=True, exist_ok=True)
        write_workload_samples(res, args.out / "nfr_perf1_samples.csv")
        md = workloads_markdown(res)
    aggregates = {k: v for k, v in res.items() if k != "samples"}
    (args.out / "nfr_perf1.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    (args.out / "nfr_perf1.md").write_text(md + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
