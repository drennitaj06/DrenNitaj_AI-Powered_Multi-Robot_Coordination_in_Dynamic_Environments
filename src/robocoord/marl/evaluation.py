"""Episode evaluation for validation (block A) and development comparisons (block B).

Every controller runs on the plain engine (no look-ahead; horizon ``T`` = ``max_steps``), with the
approved compositions (D2), shield off unless requested:

* ``baseline`` — right-of-way baseline, deadlock resolver on, endpoint recovery on;
* ``reference`` — reference follower, resolver off, endpoint recovery on;
* ``madql`` — the learned controller (greedy), resolver off, endpoint recovery on.

Per episode it records the task completion, the mean over robots of the discounted *base* return
(model §7 without potential-based shaping; the mean is the same for every ``β`` because team
mixing preserves the sum), conflicts and the engine's episode metrics. NumPy only, except that
``madql`` loads its checkpoint with PyTorch inside the worker.

Pooled task completion is ``Σ delivered / Σ released`` (as in the Issue #4 reporting).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robocoord.core.types import Action
from robocoord.marl.config import EnvConfig, RewardParams
from robocoord.marl.reward import step_rewards
from robocoord.marl.scenarios import LEVELS, check_seeds
from robocoord.sim.engine import Engine
from robocoord.world.entities import RobotMode

CONTROLLERS = ("baseline", "reference", "madql")
#: RRT* iteration budget of the Issue #4 system comparison (``configs/experiments/e5_*.toml``).
RRT_ITERATIONS = 1000


@dataclass(frozen=True, slots=True)
class EvalTask:
    controller: str
    level: str
    seed: int
    planner: str = "astar"
    block: str = "A"
    checkpoint: str | None = None
    shield: str = "off"
    rrt_iterations: int = RRT_ITERATIONS
    label: str = ""


def _system(task: EvalTask) -> tuple[Any, Any, frozenset[str]]:
    if task.controller not in CONTROLLERS:
        raise ValueError(f"unknown controller {task.controller!r}; expected {CONTROLLERS}")
    from robocoord.marl.env import make_coordinator

    cfg = EnvConfig(
        planner=task.planner,
        deadlock=task.controller == "baseline",
        endpoint=True,
        rrt_iterations=task.rrt_iterations if task.planner == "rrtstar" else None,
    )
    coord = make_coordinator(cfg)
    if task.controller == "baseline":
        from robocoord.control.baseline import BaselineController

        return coord, BaselineController(), frozenset()
    if task.controller == "reference":
        from robocoord.control.reference import ReferenceFollower

        return coord, ReferenceFollower(), frozenset()
    if task.checkpoint is None:
        raise ValueError("the madql controller needs a checkpoint")
    import torch

    from robocoord.marl.env import VIEW_EXTRAS
    from robocoord.marl.policy import LearnedController

    torch.set_num_threads(1)  # one intra-op thread per evaluation process (ADR-8)
    return coord, LearnedController.from_checkpoint(task.checkpoint), VIEW_EXTRAS


def run_episode(task: EvalTask, reward: RewardParams | None = None) -> dict[str, Any]:
    """Run one episode and return its record (deterministic for a fixed task)."""
    check_seeds([task.seed], task.block)
    reward = reward or RewardParams()
    spec = LEVELS[task.level].sample(task.seed)
    coord, ctrl, extras = _system(task)
    shield = None
    if task.shield != "off":
        from robocoord.control.shield import Shield

        shield = Shield(task.shield)
    engine = Engine(spec, coord, shield=shield, view_extras=extras, path_limit=6)
    ctrl.reset(len(engine.robots))
    n = len(engine.robots)
    ret = np.zeros(n)
    disc = 1.0
    zeros = np.zeros(n)
    t0 = time.perf_counter()
    while not engine.done:
        view = engine.prepare_step()
        nonidle = [r.mode is not RobotMode.IDLE for r in engine.robots]
        actions: dict[int, Action] = ctrl.act(view)
        outcome = engine.apply_actions(actions)
        prios = {tid: engine.book.tasks[tid].priority for _, tid in outcome.delivered}
        ret += disc * step_rewards(reward, nonidle, outcome, prios, zeros, zeros).base
        disc *= reward.gamma
    wall = time.perf_counter() - t0
    m = engine.summary().to_dict()
    planning = m["extra"].get("planning", {})
    return {
        **{k: v for k, v in asdict(task).items() if k != "checkpoint"},
        "checkpoint": os.path.basename(task.checkpoint) if task.checkpoint else None,
        "steps": m["steps"],
        "terminated": engine.terminated,
        "released": m["tasks_released"],
        "delivered": m["tasks_delivered"],
        "completion": m["completion_rate"],
        "base_return": float(ret.mean()),
        "conflicts": m["conflicts"],
        "conflicts_total": m["conflicts_total"],
        "conflicts_per_100": m["conflicts_per_100_robot_steps"],
        "physical_collisions": m["physical_collisions"],
        "mean_flow_time": m["mean_flow_time"],
        "weighted_flow_time": m["weighted_flow_time"],
        "wait_ratio": m["wait_ratio"],
        "deviation_rate_per_100": planning.get("deviation_rate_per_100"),
        "planner_calls": planning.get("calls"),
        "wall_s": wall,
    }


def run_tasks(tasks: Sequence[EvalTask], workers: int = 1) -> list[dict[str, Any]]:
    """Run tasks, optionally in worker processes; results keep the task order."""
    if workers <= 1 or len(tasks) <= 1:
        return [run_episode(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_episode, tasks, chunksize=1))


def summarise(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pooled completion and mean base return, overall and per scenario level."""
    recs = list(records)

    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        released = sum(r["released"] for r in rs)
        delivered = sum(r["delivered"] for r in rs)
        return {
            "episodes": len(rs),
            "completion": delivered / released if released else None,
            "base_return": float(np.mean([r["base_return"] for r in rs])) if rs else None,
            "conflicts_per_100": float(np.mean([r["conflicts_per_100"] for r in rs]))
            if rs
            else None,
            "terminated": sum(bool(r["terminated"]) for r in rs),
            "physical_collisions": sum(r["physical_collisions"] for r in rs),
        }

    out: dict[str, Any] = {"overall": block(recs)}
    for lvl in LEVELS:
        rs = [r for r in recs if r["level"] == lvl]
        if rs:
            out[lvl] = block(rs)
    return out


def selection_key(summary: dict[str, Any]) -> tuple[float, float]:
    """Approved checkpoint-selection rule: maximise overall completion, then mean base return."""
    o = summary["overall"]
    return (float(o["completion"] or 0.0), float(o["base_return"] or -np.inf))


def block_a_tasks(
    checkpoint: str | None, controller: str = "madql", seeds: int = 20
) -> list[EvalTask]:
    """The block-A A* validation episodes: the first ``seeds`` seeds × 3 levels (60 when all 20
    are used, the approved setting; fewer only for smoke tests)."""
    from robocoord.marl.scenarios import BLOCK_A

    return [
        EvalTask(controller, lvl, s, "astar", "A", checkpoint)
        for lvl in LEVELS
        for s in BLOCK_A[:seeds]
    ]


def block_a_rrt_tasks(checkpoint: str | None, controller: str = "madql") -> list[EvalTask]:
    """RRT* validation: the first 10 block-A seeds of the full level (reported, never selects)."""
    from robocoord.marl.scenarios import BLOCK_A_RRT

    return [EvalTask(controller, "full", s, "rrtstar", "A", checkpoint) for s in BLOCK_A_RRT]
