"""Planning coordinator: fixture task assignment + path monitor + planner (Issue #3).

Until the auction allocator exists (Issue #4), goals come from **fixture assignments** (each robot
takes the tasks listed for it, in order, as with the scripted demo coordinator). The planner and
path monitor are the real Issue #3 components; the coordinator exposes the suggested action
``â^plan`` per robot, which a plan-following controller plays.

Goal-availability policy (model §5.3 step 4): a robot whose goal is statically ``unreachable``
releases its task if it has not picked it up yet, and holds it, staying in place and retrying
after ``T_cd``, if the task is already picked.

Re-offer rule after a release (**Issue #3 fixture logic only**). The fixture has no other robot to
give a released task to, so it offers it again to the same robot, but only once the robot's static
reachability *can* have changed: after a map event (the grid version changed) or after the robot
has moved. Re-offering every ``T_cd`` steps in an unchanged situation would only repeat the same
``unreachable`` call and release (the churn found in the Issue #3 review). The task itself stays
pending; as in model §4 it is only taken while it is *available*. In Issue #4 the auction
allocator replaces this: a released task is re-auctioned and can go to another robot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from robocoord.core.types import Action, Cell
from robocoord.planning.base import STATUSES, Planner
from robocoord.planning.constraints import ConstraintParams
from robocoord.planning.monitor import TRIGGERS, MonitorParams, PathMonitor, PlanCall, RobotInput
from robocoord.sim.coordinator import AssignmentAPI
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode, TaskStatus


class PlanningCoordinator:
    """Fixture assignments + reference planning with event-triggered replanning."""

    def __init__(
        self,
        planner: Planner,
        assignments: Sequence[Sequence[int]],
        *,
        monitor_params: MonitorParams | None = None,
        constraint_params: ConstraintParams | None = None,
    ) -> None:
        self.planner = planner
        self.name = f"fixture+{planner.name}"
        self._plan = [list(a) for a in assignments]
        self._next = [0] * len(self._plan)
        # robot → (released task, grid version, robot cell) at the time of the release
        self._released: dict[int, tuple[int, int, Cell]] = {}
        self.monitor_params = monitor_params or MonitorParams()
        self.constraint_params = constraint_params or ConstraintParams()
        self.monitor: PathMonitor | None = None
        self._step_calls: list[PlanCall] = []
        self.releases = 0

    # ------------------------------------------------------------------ Coordinator protocol
    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        if len(self._plan) != len(view.robots):
            raise ValueError("one assignment list per robot is required")
        self._next = [0] * len(self._plan)
        self._released = {}
        self.releases = 0
        self._step_calls = []
        oracle = api.oracle
        self.monitor = PathMonitor(
            self.planner,
            oracle.grid,
            len(view.robots),
            params=self.monitor_params,
            constraint_params=self.constraint_params,
            seed=api.streams.int_seed("planner"),
            static_reachable=oracle.reachable,
            static_distance=oracle.distance,
        )

    def update(self, view: WorldView, api: AssignmentAPI) -> None:
        assert self.monitor is not None, "reset() must be called first"
        t = view.t
        goals: dict[int, Cell | None] = {r.id: r.goal for r in view.robots}
        modes = {r.id: r.mode for r in view.robots}
        tasks = {tk.id: tk for tk in view.tasks}
        takeable = {tk.id for tk in view.tasks if tk.status is TaskStatus.PENDING and tk.available}
        version = view.map.version
        for r in view.robots:  # fixture allocation
            if r.mode is not RobotMode.IDLE:
                continue
            queue, k = self._plan[r.id], self._next[r.id]
            released = self._released.get(r.id)
            if released is not None and k < len(queue) and queue[k] == released[0]:
                if (version, r.position) == released[1:]:
                    continue  # nothing changed since the release: it would fail again
                del self._released[r.id]
            if k < len(queue) and queue[k] in takeable:
                api.assign_task(r.id, queue[k])
                takeable.discard(queue[k])
                self._next[r.id] = k + 1
                goals[r.id] = tasks[queue[k]].pickup
                modes[r.id] = RobotMode.TO_PICKUP
        obstacles = {o.position: o.velocity for o in view.obstacles}
        robots = [RobotInput(r.id, r.position, goals[r.id]) for r in view.robots]
        calls = self.monitor.update(t, robots, obstacles)
        for call in calls:  # release policy for statically unreachable pickups
            if call.status == "unreachable" and modes[call.robot] is RobotMode.TO_PICKUP:
                api.unassign_task(call.robot)
                self._next[call.robot] -= 1
                task = self._plan[call.robot][self._next[call.robot]]
                self._released[call.robot] = (task, version, robots[call.robot].position)
                self.releases += 1
                self.monitor.release(call.robot)
        self._step_calls = calls

    def suggested_actions(self, view: WorldView) -> Sequence[Action] | None:
        assert self.monitor is not None
        return [self.monitor.suggested(r.id, r.position) for r in view.robots]

    # ------------------------------------------------------------------ reporting (engine hooks)
    def plan_valid(self) -> tuple[bool, ...]:
        assert self.monitor is not None
        return tuple(st.valid for st in self.monitor.states)

    def finish(self, view: WorldView) -> None:
        """Called once by the engine when the episode ends (closes the last legs)."""
        assert self.monitor is not None
        self.monitor.finish(view.t, [r.position for r in view.robots])

    def trace_fields(self) -> dict[str, Any]:
        """Deterministic per-step trace additions (no timings)."""
        if not self._step_calls:
            return {}
        return {
            "plans": [
                [c.robot, c.trigger, c.status, int(c.degraded), c.path_length]
                for c in self._step_calls
            ]
        }

    def episode_stats(self) -> dict[str, Any]:
        """Deterministic planning statistics for the episode summary (model §9.3, §9.4)."""
        assert self.monitor is not None
        mon = self.monitor
        calls = mon.calls
        by_trigger = Counter(c.trigger for c in calls)
        status_mix = Counter(
            "ok_degraded" if c.status == "ok" and c.degraded else c.status for c in calls
        )
        exec_ratios = [
            (leg.end_t - leg.start_t) / leg.reference_length
            for leg in mon.legs
            if leg.reference_length is not None and leg.reference_length > 0
        ]
        move_ratios = [
            leg.moves / leg.static_distance
            for leg in mon.legs
            if leg.static_distance is not None and leg.static_distance > 0
        ]
        exec_ratio = float(np.mean(exec_ratios)) if exec_ratios else None
        move_ratio = float(np.mean(move_ratios)) if move_ratios else None
        work = Counter[str]()
        for c in calls:
            for key in ("expansions", "iterations", "near_k", "nn_visited", "propagation"):
                work[key] += int(getattr(c.stats, key))
        return {
            "planner": self.planner.name,
            "calls": len(calls),
            "calls_by_trigger": {k: by_trigger.get(k, 0) for k in TRIGGERS},
            "calls_by_attribution": dict(
                sorted(Counter(TRIGGERS[c.trigger] for c in calls).items())
            ),
            "status_mix": {
                k: status_mix.get(k, 0) for k in (*STATUSES[:1], "ok_degraded", *STATUSES[1:])
            },
            "deviation_rate_per_100": (
                100.0 * by_trigger.get("T3", 0) / mon.active_robot_steps
                if mon.active_robot_steps
                else None
            ),
            "legs_completed": len(mon.legs),
            "leg_steps_over_reference": exec_ratio,
            "leg_moves_over_static": move_ratio,
            "releases": self.releases,
            "work": dict(work),
        }

    def call_times(self) -> list[float]:
        """Wall-clock seconds per planner call (not deterministic; never written to traces)."""
        assert self.monitor is not None
        return [c.stats.time_s for c in self.monitor.calls]
