"""Episode metrics (model §9). Everything is computed from exact per-step records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from robocoord.core.types import Action
from robocoord.sim.conflicts import ConflictEvent, ConflictKind
from robocoord.world.entities import TaskBook, TaskStatus


def priority_weight(level: int) -> float:
    """``w(π) = 2^(π−1)`` (model §4)."""
    return float(2 ** (level - 1))


@dataclass(slots=True)
class _Leg:
    optimal: int | None  # static shortest distance at leg start (None if unreachable)
    moves: int = 0


@dataclass
class EpisodeMetrics:
    steps: int
    robots: int
    # task completion
    tasks_released: int
    tasks_delivered: int
    completion_rate: float | None
    makespan: int | None
    mean_flow_time: float | None
    weighted_flow_time: float | None
    flow_time_by_priority: dict[str, float | None]
    waiting_max_by_priority: dict[str, int | None]
    waiting_p95_by_priority: dict[str, float | None]
    waiting_censored: int
    throughput_per_100_steps: float
    # efficiency
    path_efficiency: float | None
    legs_completed: int
    utilisation: float
    wait_ratio: float | None
    blocked_moves: int
    blocked_moves_per_100_robot_steps: float
    blocked_goal_robot_steps: int
    # safety
    conflicts: dict[str, int]
    conflicts_total: int
    conflicts_per_100_robot_steps: float
    conflict_rates_per_100_robot_steps: dict[str, float]
    conflicts_prevented: int
    physical_collisions: int
    # counterfactual (None when no planner suggestions were available)
    reference_conflicts: int | None = None
    conflicts_avoided: int | None = None
    conflicts_introduced: int | None = None
    conflicts_persisted: int | None = None
    avoidance_rate: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = _round(asdict(self))
        return result


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


class MetricsRecorder:
    """Accumulates per-step quantities; :meth:`summary` produces :class:`EpisodeMetrics`."""

    def __init__(self, num_robots: int, num_priorities: int) -> None:
        self.num_robots = num_robots
        self.num_priorities = num_priorities
        self.steps = 0
        self.robot_steps = 0
        self.nonidle_steps = 0
        self.nonidle_stays = 0
        self.blocked_moves = 0
        self.blocked_goal_robot_steps = 0
        self.conflict_counts = {k.value: 0 for k in ConflictKind}
        self.physical_collisions = 0
        self._legs: dict[int, _Leg] = {}
        self.leg_optimal_sum = 0
        self.leg_moves_sum = 0
        self.legs_completed = 0
        self.legs_unmeasurable = 0
        self._cf_available = False
        self.reference_conflicts = 0
        self.avoided = 0
        self.introduced = 0
        self.persisted = 0

    # ------------------------------------------------------------------ legs (path efficiency)
    def leg_start(self, robot: int, optimal: int | None) -> None:
        self._legs[robot] = _Leg(optimal)

    def leg_end(self, robot: int) -> None:
        leg = self._legs.pop(robot, None)
        if leg is None:
            return
        if leg.optimal is None:
            self.legs_unmeasurable += 1
            return
        self.legs_completed += 1
        self.leg_optimal_sum += leg.optimal
        self.leg_moves_sum += leg.moves

    def leg_abort(self, robot: int) -> None:
        self._legs.pop(robot, None)

    # ------------------------------------------------------------------ per step
    def record_step(
        self,
        nonidle: Sequence[bool],
        proposal: Sequence[Action],
        moved: Sequence[bool],
        events: frozenset[ConflictEvent],
        blocked: frozenset[int],
        reference_events: frozenset[ConflictEvent] | None,
        controller_events: frozenset[ConflictEvent] | None,
        physical_violations: int = 0,
        blocked_goal: int = 0,
    ) -> None:
        self.steps += 1
        self.robot_steps += len(proposal)
        self.blocked_goal_robot_steps += blocked_goal
        for i, busy in enumerate(nonidle):
            if busy:
                self.nonidle_steps += 1
                if proposal[i] is Action.STAY:
                    self.nonidle_stays += 1
            if moved[i] and i in self._legs:
                self._legs[i].moves += 1
        for e in events:
            self.conflict_counts[e.kind_code] += 1
        self.blocked_moves += len(blocked)
        self.physical_collisions += physical_violations
        if reference_events is not None and controller_events is not None:
            self._cf_available = True
            self.reference_conflicts += len(reference_events)
            self.avoided += len(reference_events - controller_events)
            self.introduced += len(controller_events - reference_events)
            self.persisted += len(reference_events & controller_events)

    # ------------------------------------------------------------------ summary
    def summary(self, tasks: TaskBook, t_end: int) -> EpisodeMetrics:
        released = [t for t in tasks.tasks.values() if t.status is not TaskStatus.UNRELEASED]
        delivered = [t for t in released if t.status is TaskStatus.DELIVERED]
        levels = range(1, self.num_priorities + 1)

        def flow(t: Any) -> int:
            return int(t.completion_time - t.release)

        mean_flow = float(np.mean([flow(t) for t in delivered])) if delivered else None
        wsum = sum(priority_weight(t.priority) for t in delivered)
        weighted = (
            sum(priority_weight(t.priority) * flow(t) for t in delivered) / wsum
            if delivered
            else None
        )
        flow_by: dict[str, float | None] = {}
        wait_max: dict[str, int | None] = {}
        wait_p95: dict[str, float | None] = {}
        censored = 0
        for lvl in levels:
            d = [flow(t) for t in delivered if t.priority == lvl]
            flow_by[str(lvl)] = float(np.mean(d)) if d else None
            waits: list[int] = []
            for t in released:
                if t.priority != lvl:
                    continue
                if t.pickup_time is not None:
                    waits.append(t.pickup_time - t.release)
                else:
                    waits.append(t_end - t.release)
                    censored += 1
            wait_max[str(lvl)] = max(waits) if waits else None
            wait_p95[str(lvl)] = float(np.percentile(waits, 95)) if waits else None
        all_done = bool(released) and len(delivered) == len(released)
        completions = [t.completion_time for t in delivered if t.completion_time is not None]
        makespan = max(completions) if all_done else None
        total = sum(self.conflict_counts.values())

        def per_100(count: int) -> float:
            return 100.0 * count / self.robot_steps if self.robot_steps else 0.0

        rate = None
        if self._cf_available and self.reference_conflicts:
            rate = self.avoided / self.reference_conflicts
        return EpisodeMetrics(
            steps=self.steps,
            robots=self.num_robots,
            tasks_released=len(released),
            tasks_delivered=len(delivered),
            completion_rate=len(delivered) / len(released) if released else None,
            makespan=makespan,
            mean_flow_time=mean_flow,
            weighted_flow_time=weighted,
            flow_time_by_priority=flow_by,
            waiting_max_by_priority=wait_max,
            waiting_p95_by_priority=wait_p95,
            waiting_censored=censored,
            throughput_per_100_steps=100.0 * len(delivered) / self.steps if self.steps else 0.0,
            path_efficiency=(self.leg_optimal_sum / self.leg_moves_sum)
            if self.leg_moves_sum
            else None,
            legs_completed=self.legs_completed,
            utilisation=self.nonidle_steps / self.robot_steps if self.robot_steps else 0.0,
            wait_ratio=self.nonidle_stays / self.nonidle_steps if self.nonidle_steps else None,
            blocked_moves=self.blocked_moves,
            blocked_moves_per_100_robot_steps=per_100(self.blocked_moves),
            blocked_goal_robot_steps=self.blocked_goal_robot_steps,
            conflicts=dict(self.conflict_counts),
            conflicts_total=total,
            conflicts_per_100_robot_steps=per_100(total),
            conflict_rates_per_100_robot_steps={
                kind: per_100(n) for kind, n in self.conflict_counts.items()
            },
            conflicts_prevented=total,
            physical_collisions=self.physical_collisions,
            reference_conflicts=self.reference_conflicts if self._cf_available else None,
            conflicts_avoided=self.avoided if self._cf_available else None,
            conflicts_introduced=self.introduced if self._cf_available else None,
            conflicts_persisted=self.persisted if self._cf_available else None,
            avoidance_rate=rate,
            extra={"legs_unmeasurable": self.legs_unmeasurable},
        )
