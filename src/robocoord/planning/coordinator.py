"""Planning coordinators: task assignment + path monitor + planner (Issues #3 and #4).

Two coordinators share the path monitor, the release/hold policy and the reporting hooks:

* :class:`AllocationCoordinator` (Issue #4): goals come from the event-triggered
  :class:`~robocoord.allocation.policy.AllocationPolicy` with a pluggable allocator (auction or
  sequential).
* :class:`PlanningCoordinator` (Issue #3): goals come from **fixture assignments** (each robot
  takes the tasks listed for it, in order, as with the scripted demo coordinator). Kept unchanged
  for the Issue #3 tests, experiments E1/E2 and the ``[script]`` demos.

The planner and path monitor expose the suggested action ``â^plan`` per robot, which a
plan-following controller plays.

Goal-availability policy (model §5.3 step 4): a robot whose goal is statically ``unreachable``
releases its task if it has not picked it up yet, and holds it, staying in place and retrying
after ``T_cd``, if the task is already picked.

What happens to a released task differs:

* *Allocation:* the task returns to pending and the next step's auction re-allocates it (at most
  one auction per step, and this step's auction has already run). The releasing robot cannot get
  it back while its pickup stays unreachable, because the pair is then infeasible (model §8).
* *Fixture (Issue #3 logic only):* the fixture has no other robot to give a released task to, so
  it offers it again to the same robot, but only once the robot's static reachability *can* have
  changed: after a map event (the grid version changed) or after the robot has moved.
  Re-offering every ``T_cd`` steps in an unchanged situation would only repeat the same
  ``unreachable`` call and release (the churn found in the Issue #3 review).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robocoord.allocation.base import Allocator
from robocoord.allocation.benefits import BenefitConfig, robot_priorities
from robocoord.allocation.policy import AllocationEvent, AllocationPolicy
from robocoord.core.errors import ConfigError
from robocoord.core.types import Action, Cell
from robocoord.planning.base import STATUSES, Planner
from robocoord.planning.constraints import ConstraintParams
from robocoord.planning.deadlock import DeadlockResolver, DetourParams
from robocoord.planning.endpoint import EndpointParams, EndpointRecovery, RobotState
from robocoord.planning.monitor import (
    BASE_TRIGGERS,
    TRIGGERS,
    MonitorParams,
    PathMonitor,
    PlanCall,
    RobotInput,
)
from robocoord.sim.coordinator import AssignmentAPI
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode, TaskStatus


class _MonitoredCoordinator:
    """Shared part: path monitor, release/hold policy, suggestions and reporting hooks.

    Subclasses implement :meth:`_assign` (goals after this step's assignment) and may react to
    releases in :meth:`_on_release`.
    """

    name: str

    def __init__(
        self,
        planner: Planner,
        *,
        monitor_params: MonitorParams | None = None,
        constraint_params: ConstraintParams | None = None,
    ) -> None:
        self.planner = planner
        self.monitor_params = monitor_params or MonitorParams()
        self.constraint_params = constraint_params or ConstraintParams()
        self.monitor: PathMonitor | None = None
        self._step_calls: list[PlanCall] = []
        self.releases = 0
        self._trigger_keys: tuple[str, ...] = BASE_TRIGGERS  # T7 only with the resolver

    # ------------------------------------------------------------------ hooks
    def _robot_inputs(self, view: WorldView, goals: dict[int, Cell | None]) -> list[RobotInput]:
        return [RobotInput(r.id, r.position, goals[r.id]) for r in view.robots]

    def _skip_release(self, robot: int) -> bool:
        """True if this step's ``unreachable`` call was not for the robot's task goal."""
        return False

    def _after_monitor(
        self,
        view: WorldView,
        goals: dict[int, Cell | None],
        obstacles: dict[Cell, Cell],
        calls: list[PlanCall],
    ) -> None:
        return None

    def _reset_assignment(self, view: WorldView, api: AssignmentAPI) -> None:
        raise NotImplementedError

    def _assign(
        self, view: WorldView, api: AssignmentAPI
    ) -> tuple[dict[int, Cell | None], dict[int, RobotMode]]:
        """Make this step's assignment changes; return every robot's goal and mode after them."""
        raise NotImplementedError

    def _on_release(self, robot: int, version: int, position: Cell) -> None:
        return None

    # ------------------------------------------------------------------ Coordinator protocol
    def reset(self, view: WorldView, api: AssignmentAPI) -> None:
        self._reset_assignment(view, api)
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
        goals, modes = self._assign(view, api)
        version = view.map.version
        obstacles = {o.position: o.velocity for o in view.obstacles}
        robots = self._robot_inputs(view, goals)
        calls = self.monitor.update(view.t, robots, obstacles)
        for call in calls:  # release policy for statically unreachable pickups
            if (
                call.status == "unreachable"
                and modes[call.robot] is RobotMode.TO_PICKUP
                and not self._skip_release(call.robot)
            ):
                api.unassign_task(call.robot)
                self._on_release(call.robot, version, robots[call.robot].position)
                self.releases += 1
                self.monitor.release(call.robot)
        self._after_monitor(view, goals, obstacles, calls)
        self._step_calls = calls

    def suggested_actions(self, view: WorldView) -> Sequence[Action] | None:
        assert self.monitor is not None
        return [self.monitor.suggested(r.id, r.position) for r in view.robots]

    def reference_paths(self, limit: int) -> tuple[tuple[Cell, ...] | None, ...]:
        """Remaining reference path per robot, at most ``limit`` moves (``PathReporting``)."""
        assert self.monitor is not None
        out: list[tuple[Cell, ...] | None] = []
        for i in range(len(self.monitor.states)):
            path = self.monitor.reference_path(i)
            out.append(None if path is None else path[: limit + 1])
        return tuple(out)

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
            "calls_by_trigger": {k: by_trigger.get(k, 0) for k in self._trigger_keys},
            "calls_by_attribution": dict(
                sorted(Counter(TRIGGERS[c.trigger] for c in calls).items())
            ),
            "status_mix": {
                k: status_mix.get(k, 0) for k in (*STATUSES[:1], "ok_degraded", *STATUSES[1:])
            },
            "deviation_rate_per_100": (
                100.0 * mon.deviations / mon.active_robot_steps if mon.active_robot_steps else None
            ),
            "legs_completed": len(mon.legs),
            "leg_steps_over_reference": exec_ratio,
            "leg_moves_over_static": move_ratio,
            "releases": self.releases,
            "work": dict(work),
            # additive, only with repair on (Issue 6A Phase 1)
            **(
                {"repairs": mon.repairs, "repair_failures": mon.repair_failures}
                if self.monitor_params.repair_radius > 0
                else {}
            ),
        }

    def call_times(self) -> list[float]:
        """Wall-clock seconds per planner call (not deterministic; never written to traces)."""
        assert self.monitor is not None
        return [c.stats.time_s for c in self.monitor.calls]


@dataclass(frozen=True, slots=True)
class _RobotTask:
    id: int
    task: int | None


class AllocationCoordinator(_MonitoredCoordinator):
    """Event-triggered allocation (auction or sequential) + reference planning (Issue #4).

    ``deadlock``: optional :class:`~robocoord.planning.deadlock.DetourParams` enabling the
    deadlock-detour resolver (stage 5B). Off by default: behaviour and traces are then exactly
    those of stages 4–5A. Intended for the right-of-way baseline controller, whose collision
    property does not depend on the suggested actions (algorithms §5.5).

    ``endpoint``: optional :class:`~robocoord.planning.endpoint.EndpointParams` enabling
    endpoint recovery (idle-robot parking and retreat from a persistently held task goal;
    algorithms §5.6). The *system* defaults (CLI, experiment executor, benchmark) turn it on for
    every controller; this constructor keeps it opt-in like ``deadlock`` so that the stage-level
    unit tests keep pinning the exact behaviour of each stage. When both are on, the resolver
    has precedence: a robot it overrides is never in an endpoint recovery.
    """

    def __init__(
        self,
        planner: Planner,
        allocator: Allocator,
        *,
        benefit_config: BenefitConfig | None = None,
        monitor_params: MonitorParams | None = None,
        constraint_params: ConstraintParams | None = None,
        deadlock: DetourParams | None = None,
        endpoint: EndpointParams | None = None,
    ) -> None:
        super().__init__(
            planner, monitor_params=monitor_params, constraint_params=constraint_params
        )
        self.allocator = allocator
        self.policy = AllocationPolicy(allocator, benefit_config)
        self._distance: Any = None
        self._step_event: AllocationEvent | None = None
        self._tasks: dict[int, int | None] = {}
        self.resolver: DeadlockResolver | None = None
        self.name = f"{allocator.name}+{planner.name}"
        if deadlock is not None:
            if deadlock.detect_steps >= self.monitor_params.stall_threshold:
                raise ConfigError(
                    "deadlock.detect_steps",
                    "must be < monitor.stall_threshold so that detection precedes T4",
                )
            self.resolver = DeadlockResolver(deadlock)
            self._trigger_keys = (*self._trigger_keys, "T7")
            self.name += "+detour"
        self.endpoint: EndpointRecovery | None = None
        self._modes: dict[int, RobotMode] = {}
        self._goals: dict[int, Cell | None] = {}
        if endpoint is not None:
            self.endpoint = EndpointRecovery(endpoint)
            self._trigger_keys = (*self._trigger_keys, "T8")
            self.name += "+endpoint"

    def _reset_assignment(self, view: WorldView, api: AssignmentAPI) -> None:
        self.policy.reset(view)
        self._distance = api.oracle.distance
        self._step_event = None
        self._tasks = {r.id: r.task for r in view.robots}
        self._modes = {r.id: r.mode for r in view.robots}
        self._goals = {r.id: r.goal for r in view.robots}
        if self.resolver is not None:
            self.resolver.reset(view.map.version)
        if self.endpoint is not None:
            self.endpoint.reset(view.map.version)

    def _assign(
        self, view: WorldView, api: AssignmentAPI
    ) -> tuple[dict[int, Cell | None], dict[int, RobotMode]]:
        state, event = self.policy.step(view, api, self._distance)
        self._step_event = event
        self._tasks = {i: s.task for i, s in state.items()}
        self._modes = {i: s.mode for i, s in state.items()}
        self._goals = {i: s.goal for i, s in state.items()}
        return {i: s.goal for i, s in state.items()}, {i: s.mode for i, s in state.items()}

    def _on_release(self, robot: int, version: int, position: Cell) -> None:
        self.policy.notify_release(robot)

    # ------------------------------------------------------------------ resolver / endpoint hooks
    def _endpoint_robots(self, view: WorldView) -> list[RobotState]:
        return [
            RobotState(r.position, self._modes[r.id], self._tasks.get(r.id), self._goals[r.id])
            for r in view.robots
        ]

    def _robot_inputs(self, view: WorldView, goals: dict[int, Cell | None]) -> list[RobotInput]:
        if self.resolver is None and self.endpoint is None:
            return super()._robot_inputs(view, goals)
        over = self.resolver.overrides(view.t, goals) if self.resolver is not None else {}
        if self.endpoint is not None:
            extra = self.endpoint.overrides(view.t, self._endpoint_robots(view), set(over))
            over = {**over, **extra}
        out = []
        for r in view.robots:
            o = over.get(r.id)
            if o is None:
                out.append(RobotInput(r.id, r.position, goals[r.id]))
            else:
                out.append(
                    RobotInput(
                        r.id,
                        r.position,
                        o.goal,
                        o.detour,
                        o.hold_stall,
                        o.avoid,
                        o.trigger,
                        o.count_active,
                    )
                )
        return out

    def detour_members(self) -> frozenset[int]:
        """Robots that are members of an active deadlock detour (read-only; Issue 6A Phase 3).
        Empty without the resolver."""
        if self.resolver is None:
            return frozenset()
        out: set[int] = set()
        for d in self.resolver.detours.values():
            out.update(d.members)
        return frozenset(out)

    def _skip_release(self, robot: int) -> bool:
        return (self.resolver is not None and self.resolver.in_detour(robot)) or (
            self.endpoint is not None and self.endpoint.in_recovery(robot)
        )

    def _after_monitor(
        self,
        view: WorldView,
        goals: dict[int, Cell | None],
        obstacles: dict[Cell, Cell],
        calls: list[PlanCall],
    ) -> None:
        assert self.monitor is not None
        if self.resolver is not None:
            robots = [_RobotTask(i, self._tasks.get(i)) for i in range(len(view.robots))]
            prio = robot_priorities(robots, view.tasks, view.t, self.resolver.params.alpha)
            self.resolver.observe(
                view.t,
                view.map,
                [r.position for r in view.robots],
                goals,
                prio,
                obstacles,
                self.monitor,
                calls,
            )
        if self.endpoint is not None:
            detours = self.resolver.detours if self.resolver is not None else {}
            self.endpoint.observe(
                view.t,
                view.map,
                self._endpoint_robots(view),
                view.tasks,
                [(o.id, o.position, o.velocity) for o in view.obstacles],
                calls,
                busy_external=frozenset(detours),
                reserved_external=frozenset(d.waypoint for d in detours.values()),
            )

    def finish(self, view: WorldView) -> None:
        super().finish(view)
        if self.endpoint is not None:
            robots = [RobotState(r.position, r.mode, r.task, r.goal) for r in view.robots]
            self.endpoint.finish(
                robots, view.tasks, [(o.id, o.position, o.velocity) for o in view.obstacles]
            )

    def trace_fields(self) -> dict[str, Any]:
        out = super().trace_fields()
        e = self._step_event
        if e is not None:
            out["alloc"] = {
                "triggers": list(e.triggers),
                "assigned": [list(p) for p in e.assigned],
                "unassigned": [list(p) for p in e.unassigned],
            }
        if self.resolver is not None and self.resolver.step_events:
            out["deadlock"] = [ev.to_json() for ev in self.resolver.step_events]
        if self.endpoint is not None and self.endpoint.step_events:
            out["endpoint"] = [ev.to_json() for ev in self.endpoint.step_events]
        return out

    def episode_stats(self) -> dict[str, Any]:
        stats = super().episode_stats()
        stats["allocation"] = self.policy.summary()
        if self.resolver is not None:
            stats["deadlock"] = self.resolver.summary()
        if self.endpoint is not None:
            stats["endpoint"] = self.endpoint.summary()
        return stats


class PlanningCoordinator(_MonitoredCoordinator):
    """Fixture assignments + reference planning with event-triggered replanning (Issue #3)."""

    def __init__(
        self,
        planner: Planner,
        assignments: Sequence[Sequence[int]],
        *,
        monitor_params: MonitorParams | None = None,
        constraint_params: ConstraintParams | None = None,
    ) -> None:
        super().__init__(
            planner, monitor_params=monitor_params, constraint_params=constraint_params
        )
        self.name = f"fixture+{planner.name}"
        self._plan = [list(a) for a in assignments]
        self._next = [0] * len(self._plan)
        # robot → (released task, grid version, robot cell) at the time of the release
        self._released: dict[int, tuple[int, int, Cell]] = {}

    def _reset_assignment(self, view: WorldView, api: AssignmentAPI) -> None:
        if len(self._plan) != len(view.robots):
            raise ValueError("one assignment list per robot is required")
        self._next = [0] * len(self._plan)
        self._released = {}

    def _assign(
        self, view: WorldView, api: AssignmentAPI
    ) -> tuple[dict[int, Cell | None], dict[int, RobotMode]]:
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
        return goals, modes

    def _on_release(self, robot: int, version: int, position: Cell) -> None:
        self._next[robot] -= 1
        task = self._plan[robot][self._next[robot]]
        self._released[robot] = (task, version, position)
