"""Episode evaluation for validation (block A) and development comparisons (block B).

Every controller runs on the plain engine (no look-ahead; horizon ``T`` = ``max_steps``), with the
approved compositions (D2), shield off unless requested:

* ``baseline`` — right-of-way baseline, deadlock resolver on, endpoint recovery on (the
  *system-level* benchmark of Issue #5);
* ``baseline-noresolver`` — the same controller with the resolver **off** (Issue #6, D11): the
  *controller-level* comparison, matched to the learned controllers' composition;
* ``reference`` — reference follower, resolver off, endpoint recovery on;
* ``madql`` / ``ppo`` — the learned controllers (greedy by default; the explicit ``stochastic``
  mode is the Issue #6 diagnostic), resolver off, endpoint recovery on.

The allocator is the auction unless a task names the sequential auction (shield ablation, Issue
#6). Stochastic episodes draw from ``SeedSequence([tag, scenario seed, level index, replicate])``,
which does not depend on the policy, so every policy faces the same stream per instance.

Per episode it records the task completion, the mean over robots of the discounted *base* return
(model §7 without potential-based shaping; the mean is the same for every ``β`` because team
mixing preserves the sum), conflicts and the engine's episode metrics. NumPy only, except that
``madql`` loads its checkpoint with PyTorch inside the worker.

Pooled task completion is ``Σ delivered / Σ released`` (as in the Issue #4 reporting).

Conflict metrics (D10; the denominators differ and are named apart):

* ``conflicts_per_100`` — proposed conflicts per 100 robot-steps over **all** robot-steps of the
  episode (``sim.metrics``; idle robots included). Training logs (``marl.env``) use a different
  denominator, non-idle robot-steps; the two are never mixed;
* ``conflicts_total`` — proposed conflicts per episode (absolute);
* ``conflicts_per_delivered`` — ``conflicts_total / delivered``, undefined (``None``) when nothing
  was delivered.

``shaped_return`` adds the potential-based shaping to the base return through the telescoping
identity ``Σ_t γ^t F_t = γ^T Φ(s_T) − Φ(s_0)`` (mean over robots, ζ of the reward parameters),
with ``Φ(s_T) = 0`` at termination and, at the horizon, ``Φ`` of the final (unprepared) state.
``prepare_s`` is the time spent in ``prepare_step`` (events, obstacles, allocation and planning)
and ``act_s`` the controller's time.
"""

from __future__ import annotations

import os
import time
import zlib
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robocoord.core.types import Action
from robocoord.marl.config import EnvConfig, RewardParams
from robocoord.marl.reward import potentials, step_rewards
from robocoord.marl.scenarios import LEVELS, check_seeds
from robocoord.sim.engine import Engine
from robocoord.world.entities import RobotMode

CONTROLLERS = ("baseline", "baseline-noresolver", "reference", "madql", "ppo")
LEARNED = ("madql", "ppo")
MODES = ("greedy", "stochastic")
#: Tag of the stochastic-diagnostic random streams (Issue #6, D6).
STOCHASTIC_TAG = zlib.crc32(b"issue6-stochastic-diagnostic")
#: ε of the MADQL stochastic mode: its training behaviour policy after the decay.
MADQL_EPSILON = 0.05
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
    allocator: str = "auction"
    mode: str = "greedy"
    replicate: int = 0


def stochastic_rng(task: EvalTask) -> np.random.Generator:
    """The diagnostic's stream for one instance and replicate (independent of the policy)."""
    level = list(LEVELS).index(task.level)
    return np.random.default_rng(
        np.random.SeedSequence([STOCHASTIC_TAG, task.seed, level, task.replicate])
    )


def _system(task: EvalTask) -> tuple[Any, Any, frozenset[str]]:
    if task.controller not in CONTROLLERS:
        raise ValueError(f"unknown controller {task.controller!r}; expected {CONTROLLERS}")
    if task.mode not in MODES:
        raise ValueError(f"unknown mode {task.mode!r}; expected {MODES}")
    if task.mode != "greedy" and task.controller not in LEARNED:
        raise ValueError("only learned controllers have a stochastic mode")
    from robocoord.marl.env import make_coordinator

    cfg = EnvConfig(
        planner=task.planner,
        allocator=task.allocator,
        deadlock=task.controller == "baseline",
        endpoint=True,
        rrt_iterations=task.rrt_iterations if task.planner == "rrtstar" else None,
    )
    coord = make_coordinator(cfg)
    if task.controller in ("baseline", "baseline-noresolver"):
        from robocoord.control.baseline import BaselineController

        return coord, BaselineController(), frozenset()
    if task.controller == "reference":
        from robocoord.control.reference import ReferenceFollower

        return coord, ReferenceFollower(), frozenset()
    if task.checkpoint is None:
        raise ValueError(f"the {task.controller} controller needs a checkpoint")
    import torch

    from robocoord.marl.env import VIEW_EXTRAS
    from robocoord.marl.policy import LearnedController

    torch.set_num_threads(1)  # one intra-op thread per evaluation process (ADR-8)
    rng = stochastic_rng(task) if task.mode == "stochastic" else None
    ctrl = LearnedController.from_checkpoint(
        task.checkpoint, mode=task.mode, epsilon=MADQL_EPSILON, rng=rng
    )
    if (ctrl.output == "logits") != (task.controller == "ppo"):
        raise ValueError(f"{task.checkpoint} is not a {task.controller} checkpoint")
    return coord, ctrl, VIEW_EXTRAS


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
    phi0: Any = None
    t_prepare = t_act = 0.0
    t0 = time.perf_counter()
    while not engine.done:
        tp = time.perf_counter()
        view = engine.prepare_step()
        ta = time.perf_counter()
        t_prepare += ta - tp
        if phi0 is None:
            phi0 = _phi(engine)
        nonidle = [r.mode is not RobotMode.IDLE for r in engine.robots]
        actions: dict[int, Action] = ctrl.act(view)
        t_act += time.perf_counter() - ta
        outcome = engine.apply_actions(actions)
        prios = {tid: engine.book.tasks[tid].priority for _, tid in outcome.delivered}
        ret += disc * step_rewards(reward, nonidle, outcome, prios, zeros, zeros).base
        disc *= reward.gamma
    wall = time.perf_counter() - t0
    phi_end = np.zeros(n) if engine.terminated else _phi(engine)
    shaping = reward.zeta * (disc * phi_end - (phi0 if phi0 is not None else np.zeros(n)))
    m = engine.summary().to_dict()
    planning = m["extra"].get("planning", {})
    shield_stats = m["extra"].get("shield") or {}
    delivered = m["tasks_delivered"]
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
        # Issue #6 additions (absent from Issue #5 records)
        "robots": n,
        "undelivered": m["tasks_released"] - delivered,
        "conflicts_per_delivered": m["conflicts_total"] / delivered if delivered else None,
        "shaped_return": float((ret + shaping).mean()),
        "shield_interventions": shield_stats.get("interventions", 0),
        "deadlock": task.controller == "baseline",
        "makespan": m["makespan"],
        "prepare_s": t_prepare,
        "act_s": t_act,
    }


def _phi(engine: Engine) -> Any:
    g = engine.grid
    return potentials(engine.robots, engine.book.tasks, engine.oracle, g.width, g.height)


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
