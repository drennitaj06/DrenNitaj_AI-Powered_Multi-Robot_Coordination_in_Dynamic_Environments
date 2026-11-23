"""Base reward, phase-aware potential shaping and team mixing (model §7; algorithms §8). NumPy only.

For robot ``i`` at step ``t`` (``j = τ_i``):

* base: ``r_i = −c_step·1[μ_i ≠ idle] + R_pick·1^pick_i + R_del·w(π_j)·1^del_i
  − c_rr·n^RR_i − c_ro·n^RO_i − c_sh·n^SH_i`` with ``w(π) = 2^(π−1)``; the conflict counts are
  the events of the proposal reaching the arbiter in which ``i`` is at fault
  (``ConflictEvent.at_fault``), the shield count its interventions on ``i``;
* potential: ``Φ_i(s) = −Λ_i(s) / (W + H)`` with ``Λ_i = D̄(x_i, p_j) + D̄(p_j, d_j)`` before
  pickup, ``D̄(x_i, d_j)`` after it, 0 when idle (``D̄`` on the static map of ``s``);
* shaping: ``F_i = γ Φ_i(s') − Φ_i(s)``; individual reward ``r_i + ζ F_i``;
* team mixing: ``(1 − β)·r_i^ind + (β / N) Σ_k r_k^ind``.

``s`` is the state the robots observed (after the step's allocation and replanning) and ``s'``
the next observed state; at termination every robot is idle, so ``Φ(s') = 0``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robocoord.allocation.benefits import priority_weight
from robocoord.core.distance import DistanceOracle
from robocoord.marl.config import RewardParams
from robocoord.sim.engine import StepOutcome
from robocoord.world.entities import Robot, RobotMode, Task

F64 = npt.NDArray[np.float64]


def remaining_route(robot: Robot, tasks: Mapping[int, Task], oracle: DistanceOracle) -> int:
    """``Λ_i``: the capped remaining route length of the robot's task (0 when idle)."""
    if robot.mode is RobotMode.IDLE or robot.task is None:
        return 0
    task = tasks[robot.task]
    if robot.mode is RobotMode.TO_PICKUP:
        return oracle.capped_distance(robot.position, task.pickup) + oracle.capped_distance(
            task.pickup, task.dropoff
        )
    return oracle.capped_distance(robot.position, task.dropoff)


def potentials(
    robots: Sequence[Robot],
    tasks: Mapping[int, Task],
    oracle: DistanceOracle,
    width: int,
    height: int,
) -> F64:
    """``Φ_i(s) = −Λ_i(s) / (W + H)`` for every robot."""
    norm = float(width + height)
    return np.array([-remaining_route(r, tasks, oracle) / norm for r in robots], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class StepRewards:
    """Per-robot reward components of one transition."""

    total: F64  # team-mixed reward used for learning
    base: F64  # r_i^base
    shaping: F64  # ζ F_i
    individual: F64  # r_i^base + ζ F_i
    rr: npt.NDArray[np.int64]
    ro: npt.NDArray[np.int64]
    sh: npt.NDArray[np.int64]


def conflict_counts(
    outcome: StepOutcome, n: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """At-fault robot–robot and robot–obstacle conflicts, and shield interventions, per robot."""
    rr = np.zeros(n, dtype=np.int64)
    ro = np.zeros(n, dtype=np.int64)
    sh = np.zeros(n, dtype=np.int64)
    for e in outcome.events:
        kind = e.kind
        if kind.is_robot_robot:
            for i in e.at_fault:
                rr[i] += 1
        elif kind.is_robot_obstacle:
            for i in e.at_fault:
                ro[i] += 1
    for ev in outcome.shield_events:
        sh[int(ev.robot)] += 1
    return rr, ro, sh


def step_rewards(
    params: RewardParams,
    nonidle: Sequence[bool],
    outcome: StepOutcome,
    task_priority: Mapping[int, int],
    phi_before: F64,
    phi_after: F64,
) -> StepRewards:
    """The model §7 reward of one transition ``s → s'``."""
    n = len(nonidle)
    rr, ro, sh = conflict_counts(outcome, n)
    base = np.where(np.asarray(nonidle, dtype=bool), -params.c_step, 0.0).astype(np.float64)
    for robot, _task in outcome.picked:
        base[robot] += params.r_pick
    for robot, task in outcome.delivered:
        base[robot] += params.r_del * priority_weight(task_priority[task])
    base -= params.c_rr * rr + params.c_ro * ro + params.c_sh * sh
    shaping = params.zeta * (params.gamma * phi_after - phi_before)
    individual = base + shaping
    total = (1.0 - params.beta) * individual + params.beta * float(individual.mean())
    return StepRewards(total, base, shaping, individual, rr, ro, sh)


def team_potential(phi: F64, beta: float) -> F64:
    """``Ψ_i = (1 − β) Φ_i + (β / N) Σ_k Φ_k`` (model §7.1)."""
    return (1.0 - beta) * phi + beta * float(phi.mean())
