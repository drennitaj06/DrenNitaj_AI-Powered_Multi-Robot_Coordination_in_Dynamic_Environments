"""RRT*-in-training feasibility pilot (Issue #6, approved D7 protocol). Requires PyTorch.

Measures the real PPO collection-and-update loop (``marl.rollout`` + ``marl.ppo``; E = 8,
T_r = 125, updates on, one torch thread) per curriculum stage, with RRT* (``n_max`` = 1 000) and
with A* on the same scenario seeds, for two action sources:

* ``ppo`` — an untrained, seeded MAPPO actor (the start of training: high deviation from the
  planner's suggestions, so many replanning triggers);
* ``reference`` — the planner's suggested actions (a low-deviation bound); PPO updates still run
  on these samples so the per-step cost includes them.

Per cell (planner × source × stage): ``steps`` environment steps (default 2 000) from the stage's
distribution, the scenario seeds taken from the fixed training order (identical in every cell),
capped at ``cap`` seconds (default 3 600; a capped cell is recorded as a timeout and its
projection is then a lower bound). Measured: environment steps per second, mean / median / p95
time of one environment step (``step_arrays`` including the next preparation, plus resets),
planner calls per environment step and the planner's share of the wall time (all ``plan`` calls
timed), PPO update time, and the peak process memory.

Projection per planner and source: ``T_proj = Σ_stage share · 1 000 000 / throughput + T_val``,
with ``T_val`` the measured validation time of a 1 M-step run (A* block A, unchanged by the
training planner). Decision (D7): the untrained-actor projection ≤ 24 h → feasible; only the
reference projection ≤ 24 h → borderline (reported, decided by the student); both > 24 h →
A*-only training with cross-planner evaluation. The pilot never decides to train.

Usage::

    python -m robocoord.marl.pilot --out results/pilot_rrt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.marl.config import EnvConfig, ObsParams
from robocoord.marl.env import F32, MultiRobotEnv
from robocoord.marl.global_state import global_state_size
from robocoord.marl.observation import S_SUGGEST
from robocoord.marl.rollout import RolloutBuffer, collect_rollout, last_values
from robocoord.marl.scenarios import CURRICULUM, training_seed_order
from robocoord.marl.vector_env import SyncVectorEnv
from robocoord.scenario.spec import ScenarioSpec

PLANNERS = ("astar", "rrtstar")
SOURCES = ("ppo", "reference")
THRESHOLD_H = 24.0
RRT_ITERATIONS = 1000


class Clock:
    """Accumulates planner time and calls (process-wide, single-threaded)."""

    def __init__(self) -> None:
        self.plan_s = 0.0
        self.plan_calls = 0


@contextlib.contextmanager
def timed_planners(clock: Clock) -> Iterator[None]:
    """Wrap the planner factory so every ``plan`` call of every new planner is timed."""
    import robocoord.planning as planning

    original = planning.make_planner

    def factory(name: str, **options: Any) -> Any:
        planner = original(name, **options)
        inner = planner.plan

        def plan(query: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return inner(query)
            finally:
                clock.plan_s += time.perf_counter() - t0
                clock.plan_calls += 1

        setattr(planner, "plan", plan)  # noqa: B010 - instance-level timing wrapper
        return planner

    planning.make_planner = factory
    try:
        yield
    finally:
        planning.make_planner = original


class TimedEnv(MultiRobotEnv):
    """Records the duration of every ``step_arrays`` (one environment step, including the next
    preparation) and every ``reset_spec``."""

    def __init__(self, config: EnvConfig, times: list[float]) -> None:
        super().__init__(config=config)
        self.times = times

    def step_arrays(self, actions: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return super().step_arrays(actions)
        finally:
            self.times.append(time.perf_counter() - t0)

    def reset_spec(self, spec: ScenarioSpec) -> Any:
        t0 = time.perf_counter()
        try:
            return super().reset_spec(spec)
        finally:
            if self.times:  # a reset during the run belongs to the step that triggered it
                self.times[-1] += time.perf_counter() - t0


def peak_rss() -> float | None:
    from robocoord.marl.ppo_trainer import peak_rss_mb

    return peak_rss_mb()


def run_cell(
    planner: str, source: str, stage: int, steps: int, cap: float, run_seed: int = 1
) -> dict[str, Any]:
    """One pilot cell; deterministic scenario seeds, fresh untrained actor."""
    import torch

    from robocoord.marl.networks import configure_torch
    from robocoord.marl.ppo import PPO, PPOParams, masked_log_probs

    params = PPOParams(variant="mappo")
    root = np.random.SeedSequence(run_seed)
    sample, shuffle, torch_seq = root.spawn(3)
    sample_rng, shuffle_rng = np.random.default_rng(sample), np.random.default_rng(shuffle)
    configure_torch(int(torch_seq.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1)), True, 1)
    agent = PPO(ObsParams().dim, params)
    rrt = RRT_ITERATIONS if planner == "rrtstar" else None
    cfg = EnvConfig(planner=planner, rrt_iterations=rrt)
    order = training_seed_order(0)
    dist = CURRICULUM[stage].dist
    position = {"i": 0}

    def next_scenario(_k: int) -> ScenarioSpec:
        seed = order[position["i"]]
        position["i"] += 1
        return dist.sample(seed)

    suggest = ObsParams().grid_size + S_SUGGEST

    def act(o: F32, m: np.ndarray) -> tuple[np.ndarray, F32]:
        if source == "ppo":
            return agent.act(o, m, sample_rng)
        a = o[:, suggest : suggest + 5].argmax(axis=1)
        a = np.where(m[np.arange(len(a)), a], a, 0).astype(np.int64)
        with torch.no_grad():
            lp = masked_log_probs(agent.actor(torch.from_numpy(o)), torch.from_numpy(m))
        return a, lp.gather(1, torch.from_numpy(a)[:, None])[:, 0].numpy().astype(np.float32)

    times: list[float] = []
    clock = Clock()
    record: dict[str, Any] = {"planner": planner, "source": source, "stage": stage + 1,
                              "stage_share": CURRICULUM[stage].fraction, "target_steps": steps,
                              "cap_s": cap}  # fmt: skip
    t_update = 0.0
    env_steps = robot_steps = episodes = 0
    t_start = time.perf_counter()
    timeout = False
    with timed_planners(clock):
        envs = [TimedEnv(cfg, times) for _ in range(params.n_envs)]
        venv = SyncVectorEnv(envs, next_scenario, with_state=True)
        t_reset = time.perf_counter()
        venv.reset()
        reset_s = time.perf_counter() - t_reset
        while env_steps < steps:
            buf = RolloutBuffer(params.rollout, params.n_envs, ObsParams().dim,
                                state_dim=global_state_size(8))  # fmt: skip
            stats = collect_rollout(venv, buf, act, agent.value, centralised=True)
            lv, valid = last_values(venv, buf, agent.value, centralised=True)
            buf.compute_gae(lv, valid, params.gamma, params.lam)
            tu = time.perf_counter()
            agent.update(buf.flatten(), shuffle_rng)
            t_update += time.perf_counter() - tu
            env_steps += stats.env_steps
            robot_steps += stats.robot_steps
            episodes += len(stats.episodes)
            if time.perf_counter() - t_start > cap and env_steps < steps:
                timeout = True
                break
    wall = time.perf_counter() - t_start
    arr = np.asarray(times)
    record.update(
        env_steps=env_steps,
        robot_steps=robot_steps,
        episodes_finished=episodes,
        timeout=timeout,
        wall_s=wall,
        initial_reset_s=reset_s,
        env_steps_per_s=env_steps / wall,
        step_mean_ms=1000 * float(arr.mean()),
        step_median_ms=1000 * float(np.median(arr)),
        step_p95_ms=1000 * float(np.quantile(arr, 0.95)),
        step_max_ms=1000 * float(arr.max()),
        planner_calls=clock.plan_calls,
        planner_calls_per_env_step=clock.plan_calls / env_steps,
        planner_s=clock.plan_s,
        planner_share=clock.plan_s / wall,
        update_s=t_update,
        update_share=t_update / wall,
        scenario_seeds_used=position["i"],
        peak_rss_mb=peak_rss(),
    )
    record["step_times_ms"] = [round(1000 * x, 4) for x in times]
    return record


def project(cells: Sequence[dict[str, Any]], validation_s: float) -> dict[str, Any]:
    """``T_proj`` per planner and source (hours) and the D7 decision."""
    out: dict[str, Any] = {"validation_s_per_run": validation_s, "threshold_h": THRESHOLD_H}
    for planner in PLANNERS:
        for source in SOURCES:
            cs = [c for c in cells if c["planner"] == planner and c["source"] == source]
            if len(cs) != len(CURRICULUM):
                continue
            train_s = sum(c["stage_share"] * 1_000_000 / c["env_steps_per_s"] for c in cs)
            out[f"{planner}/{source}"] = {
                "training_h": train_s / 3600,
                "total_h": (train_s + validation_s) / 3600,
                "lower_bound": any(c["timeout"] for c in cs),
            }
    rrt_ppo = out.get("rrtstar/ppo", {}).get("total_h")
    rrt_ref = out.get("rrtstar/reference", {}).get("total_h")
    if rrt_ppo is not None and rrt_ppo <= THRESHOLD_H:
        decision = "feasible"
    elif rrt_ref is not None and rrt_ref <= THRESHOLD_H:
        decision = "borderline"
    elif rrt_ppo is not None and rrt_ref is not None:
        decision = "infeasible"
    else:
        decision = "incomplete"
    out["decision"] = decision
    for source in SOURCES:
        a, r = out.get(f"astar/{source}"), out.get(f"rrtstar/{source}")
        if a and r:
            out[f"rrt_over_astar_training_time/{source}"] = r["training_h"] / a["training_h"]
    return out


def main(argv: Sequence[str] | None = None, run: Callable[..., dict[str, Any]] = run_cell) -> int:
    ap = argparse.ArgumentParser(prog="python -m robocoord.marl.pilot")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--cap", type=float, default=3600.0)
    ap.add_argument("--planners", default=",".join(PLANNERS))
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--validation-s", type=float, required=True,
                    help="measured validation time of one 1 M-step run (seconds)")  # fmt: skip
    args = ap.parse_args(argv)
    if (args.out / "cells.jsonl").exists():
        raise FileExistsError(f"{args.out} already holds a pilot")
    args.out.mkdir(parents=True, exist_ok=True)
    from robocoord.marl.trainer import _environment

    (args.out / "environment.json").write_text(
        json.dumps({"environment": _environment(), "argv": list(argv or sys.argv[1:])}, indent=2),
        "utf-8",
    )
    cells = []
    with (args.out / "cells.jsonl").open("w", encoding="utf-8") as fh:
        for planner in args.planners.split(","):
            for source in args.sources.split(","):
                for stage in range(len(CURRICULUM)):
                    cell = run(planner, source, stage, args.steps, args.cap)
                    fh.write(json.dumps(cell) + "\n")
                    fh.flush()
                    cells.append(cell)
                    brief = {k: v for k, v in cell.items() if k != "step_times_ms"}
                    print(json.dumps(brief), flush=True)
    summary = project(cells, args.validation_s)
    (args.out / "projection.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
