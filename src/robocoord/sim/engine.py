"""The simulation engine: one deterministic step pipeline (architecture §2).

Step ``t`` → ``t + 1``:

1. apply events due at ``t`` (task releases, arrivals, map events with deferral);
2. dynamic obstacles propose and resolve their moves (``z(t+1)``; hidden from controllers);
3. the coordinator (re)assigns tasks — and, from Issue #3, (re)plans;
4. observations: a :class:`~robocoord.sim.view.WorldView` with action masks and suggestions;
   the controller returns a joint proposal;
5. the optional safety shield (Issue #4, ``control.shield``): ``a^ctrl → a^arb``; identity if
   none is configured;
6. conflict detection + arbitration on ``a^arb``; physical-collision invariant check (G1);
7. counterfactual conflict sets ``K^ref`` (the suggested actions, masked → stay) and
   ``K^ctrl`` (the controller's raw proposal) when suggestions exist (model §9.2, X-10), and the
   shield's prevented / introduced sets ``K^ctrl ∖ K^arb`` / ``K^arb ∖ K^ctrl``;
8. pickups and deliveries at ``t + 1``;
9. path monitor bookkeeping: done by a planning coordinator at the start of the next step's
   pipeline step 3 (it sees the positions at ``t + 1``, which is equivalent);
10. metrics and trace.

The RL wrapper (Issue #5) calls :meth:`Engine.prepare_step` and :meth:`Engine.apply_actions`
separately; :meth:`Engine.step` composes them for interactive and experiment loops.

Termination (model §6.3): every task of the episode is delivered and no further arrivals can
occur. A scenario that never contains a task (no scheduled tasks, no arrivals) is a pure-motion
sandbox and runs to ``max_steps`` instead of terminating vacuously at ``t = 1``. The post-step
physical-collision check raises :class:`PhysicalCollisionError`, so the recorded
``physical_collisions`` count is 0 for every episode that completes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from robocoord.core.distance import DistanceOracle
from robocoord.core.rng import make_streams
from robocoord.core.types import Action, Cell
from robocoord.scenario.spec import ScenarioSpec
from robocoord.sim.conflicts import (
    ConflictEvent,
    arbitrate,
    detect_conflicts,
    physical_violations,
)
from robocoord.sim.coordinator import Coordinator, NullCoordinator, PathReporting, PlanReporting
from robocoord.sim.metrics import EpisodeMetrics, MetricsRecorder
from robocoord.sim.trace import TraceWriter
from robocoord.sim.view import (
    MapView,
    ObstacleView,
    RobotView,
    TaskView,
    WorldView,
    action_mask,
)
from robocoord.world.arrivals import (
    ArrivalProcess,
    BernoulliArrivals,
    NoArrivals,
    PoissonArrivals,
)
from robocoord.world.entities import (
    DynamicObstacle,
    Robot,
    RobotMode,
    Task,
    TaskBook,
    TaskRequest,
    TaskStatus,
)
from robocoord.world.events import MapEvent, MapEventSchedule
from robocoord.world.motion import Bounce, MotionModel, Patrol, RandomWalk, resolve_obstacle_moves


class PhysicalCollisionError(RuntimeError):
    """The post-arbitration invariant was violated: a simulator defect (G1)."""


class EngineStateError(RuntimeError):
    """Engine API used in the wrong order (e.g. stepping a finished episode)."""


_MODE_CODE = {RobotMode.IDLE: "I", RobotMode.TO_PICKUP: "P", RobotMode.TO_DROPOFF: "D"}

#: Optional view data for learned controllers (Issue #5); see ``WorldView``.
VIEW_EXTRAS: frozenset[str] = frozenset({"paths", "goal_distance"})


class SafetyLayer(Protocol):
    """What the engine needs from a shield (implemented by ``control.shield.Shield``)."""

    mode: str

    @property
    def enabled(self) -> bool: ...

    def reset(self) -> None: ...

    def apply(self, view: WorldView, proposal: Sequence[Action]) -> tuple[list[Action], list[Any]]:
        """The proposal reaching the arbiter and the interventions (each has ``to_json``)."""
        ...

    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class StepOutcome:
    """One step's result. ``actions`` is the controller's raw proposal ``a^ctrl``; ``events`` the
    conflicts of the proposal reaching the arbiter (``a^arb``, after the shield);
    ``reference_events`` / ``controller_events`` the counterfactual sets ``K^ref`` / ``K^ctrl``
    (``None`` without suggestions); ``shield_events`` the shield's interventions."""

    t: int  # time after the step
    actions: tuple[Action, ...]
    events: frozenset[ConflictEvent]
    blocked: frozenset[int]
    released: tuple[int, ...]
    assigned: tuple[tuple[int, int], ...]
    picked: tuple[tuple[int, int], ...]
    delivered: tuple[tuple[int, int], ...]
    map_events: tuple[MapEvent, ...]
    reference_events: frozenset[ConflictEvent] | None
    terminated: bool
    truncated: bool
    controller_events: frozenset[ConflictEvent] | None = None
    arbitrated_actions: tuple[Action, ...] | None = None
    shield_events: tuple[Any, ...] = ()

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class _Pending:
    """State produced by :meth:`Engine.prepare_step`, consumed by :meth:`Engine.apply_actions`."""

    view: WorldView
    obstacles_now: list[Cell]
    obstacles_next: list[Cell]
    released: list[int] = field(default_factory=list)
    map_events: list[MapEvent] = field(default_factory=list)
    assigned: list[tuple[int, int]] = field(default_factory=list)
    unassigned: list[int] = field(default_factory=list)
    new_tasks: list[int] = field(default_factory=list)


def _make_model(
    spec_model: str, waypoints: tuple[Cell, ...], direction: Action | None
) -> MotionModel:
    if spec_model == "patrol":
        return Patrol(waypoints)
    if spec_model == "bounce":
        assert direction is not None
        return Bounce(direction)
    return RandomWalk()


class Engine:
    """Deterministic headless simulator for one scenario instance."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        coordinator: Coordinator | None = None,
        *,
        seed: int | None = None,
        trace: TraceWriter | None = None,
        stop_when_done: bool = True,
        max_steps: int | None = None,
        shield: SafetyLayer | None = None,
        controller_stats: Callable[[], dict[str, Any]] | None = None,
        view_extras: frozenset[str] = frozenset(),
        path_limit: int = 16,
    ) -> None:
        scenario.validate()
        unknown = set(view_extras) - VIEW_EXTRAS
        if unknown:
            raise ValueError(f"unknown view extras {sorted(unknown)}; known: {sorted(VIEW_EXTRAS)}")
        if path_limit < 1:
            raise ValueError("path_limit must be >= 1")
        # Opt-in data for learned controllers (Issue #5). Added only to the view a controller acts
        # on; reading it has no side effect on the simulation, its traces or its metrics.
        self.view_extras = frozenset(view_extras)
        self.path_limit = path_limit
        self.scenario = scenario
        self.shield = shield if shield is not None and shield.enabled else None
        self.controller_stats = controller_stats  # adds extra["controller"] to the summary
        self.coordinator: Coordinator = coordinator or NullCoordinator()
        self._reporting = self.coordinator if isinstance(self.coordinator, PlanReporting) else None
        self.seed = scenario.seed if seed is None else seed
        self.trace = trace
        self.stop_when_done = stop_when_done
        self.max_steps = scenario.max_steps if max_steps is None else max_steps
        self.reset()

    # ================================================================== setup
    def reset(self) -> WorldView:
        spec = self.scenario
        streams = make_streams(self.seed)
        self.streams = streams  # exposed to coordinators through the AssignmentAPI
        self.grid = spec.grid.copy()
        self.oracle = DistanceOracle(self.grid)
        self.robots = [Robot(i, c) for i, c in enumerate(spec.robots)]
        obs_rngs = streams.generators("obstacles", len(spec.obstacles))
        self.obstacles = [
            DynamicObstacle(k, o.start, _make_model(o.model, o.waypoints, o.direction), obs_rngs[k])
            for k, o in enumerate(spec.obstacles)
        ]
        self.book = TaskBook()
        for ts in spec.tasks:  # scheduled tasks keep their scenario index as id
            self.book.create(TaskRequest(ts.pickup, ts.dropoff, ts.priority, ts.release))
        self._task_rng = streams.generator("tasks")
        self.arrivals = self._make_arrivals()
        self.events = MapEventSchedule(MapEvent(e.time, e.cell, e.block) for e in spec.map_events)
        self.metrics = MetricsRecorder(len(self.robots), spec.num_priorities)
        if self.shield is not None:
            self.shield.reset()
            self.metrics.enable_shield()
        self.t = 0
        self.terminated = False
        self.truncated = False
        self._pending: _Pending | None = None
        view = self._build_view(None)
        self.coordinator.reset(view, self)
        if self.trace is not None:
            self.trace.write(self._header())
        return view

    def _make_arrivals(self) -> ArrivalProcess:
        a = self.scenario.arrivals
        if a.kind == "bernoulli":
            return BernoulliArrivals(a.rate, a.priorities, a.until)
        if a.kind == "poisson":
            return PoissonArrivals(a.rate, a.priorities, a.until)
        return NoArrivals()

    # ================================================================== AssignmentAPI
    def assign_task(self, robot: int, task: int) -> None:
        r = self.robots[robot]
        tk = self.book.tasks[task]
        if r.mode is not RobotMode.IDLE:
            raise EngineStateError(f"robot {robot} is not idle (mode {r.mode.value})")
        if not self.task_available(tk):
            raise EngineStateError(f"task {task} is not available")
        tk.do_assign(robot)
        r.mode, r.task = RobotMode.TO_PICKUP, task
        self.metrics.leg_start(robot, self.oracle.distance(r.position, tk.pickup))
        if self._pending is not None:
            self._pending.assigned.append((robot, task))

    def unassign_task(self, robot: int) -> None:
        r = self.robots[robot]
        if r.mode is not RobotMode.TO_PICKUP or r.task is None:
            raise EngineStateError(f"robot {robot} has no pre-pickup task to release")
        self.book.tasks[r.task].do_unassign()
        r.mode, r.task = RobotMode.IDLE, None
        self.metrics.leg_abort(robot)
        if self._pending is not None:
            self._pending.unassigned.append(robot)

    def task_available(self, task: Task) -> bool:
        """Released, unpicked, and pickup→dropoff reachable on the current map (model §4)."""
        return (
            task.is_open
            and self.grid.is_free(task.pickup)
            and self.grid.is_free(task.dropoff)
            and self.oracle.reachable(task.pickup, task.dropoff)
        )

    # ================================================================== stepping
    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def prepare_step(self) -> WorldView:
        """Pipeline steps 1–4 (up to the observation). Returns the view the controller acts on."""
        if self.done:
            raise EngineStateError("episode is finished; call reset()")
        if self._pending is not None:
            return self._pending.view
        t = self.t
        robot_cells = {r.position for r in self.robots}
        obstacle_cells = [o.position for o in self.obstacles]
        pending = _Pending(
            view=self._build_view(None), obstacles_now=obstacle_cells, obstacles_next=obstacle_cells
        )
        self._pending = pending

        # 1. events: scheduled releases, arrivals, map events
        for task in self.book.tasks.values():
            if task.status is TaskStatus.UNRELEASED and task.release <= t:
                task.do_release()
                pending.released.append(task.id)
        for req in self.arrivals.releases(t, self.grid, self.oracle, self._task_rng):
            task = self.book.create(req)
            task.do_release()
            pending.released.append(task.id)
            pending.new_tasks.append(task.id)
        pending.map_events = self.events.apply_due(t, self.grid, robot_cells | set(obstacle_cells))

        # 2. obstacle moves (resolved now, revealed after the robots act)
        proposals = [
            o.model.propose(o.position, self.grid, self.oracle, o.rng) for o in self.obstacles
        ]
        pending.obstacles_next = resolve_obstacle_moves(
            obstacle_cells, proposals, self.grid, robot_cells
        )

        # 3. coordinator
        self.coordinator.update(self._build_view(None), self)

        # 4. view with masks and suggestions
        suggestions = self.coordinator.suggested_actions(self._build_view(None))
        if suggestions is not None and len(suggestions) != len(self.robots):
            raise EngineStateError("coordinator must suggest one action per robot")
        pending.view = self._build_view(
            tuple(suggestions) if suggestions is not None else None, extras=True
        )
        return pending.view

    def apply_actions(self, actions: Mapping[int, Action]) -> StepOutcome:
        """Pipeline steps 5–10 for the controller's joint proposal."""
        pending = self._pending
        if pending is None:
            raise EngineStateError("call prepare_step() before apply_actions()")
        n = len(self.robots)
        missing = [i for i in range(n) if i not in actions]
        if missing:
            raise ValueError(f"no action for robots {missing}")
        proposal = [Action(actions[i]) for i in range(n)]
        t = self.t
        positions = [r.position for r in self.robots]
        nonidle = [r.mode is not RobotMode.IDLE for r in self.robots]
        blocked_goal = 0  # robots holding a task whose goal is statically unreachable (model §9.1)
        for r in self.robots:
            goal = r.goal(self.book.tasks)
            if goal is not None and not self.oracle.reachable(r.position, goal):
                blocked_goal += 1

        # 5. shield (a^ctrl → a^arb); identity without one
        shield_events: list[Any] = []
        arb_proposal = proposal
        if self.shield is not None:
            arb_proposal, shield_events = self.shield.apply(pending.view, proposal)
            if not shield_events:
                arb_proposal = proposal  # nothing replaced: same object, no second detection

        # 6. detection + arbitration + invariant
        events = detect_conflicts(
            t, positions, arb_proposal, pending.obstacles_now, pending.obstacles_next, self.grid
        )
        resolution = arbitrate(positions, arb_proposal, events)
        violations = physical_violations(
            positions, resolution.positions, pending.obstacles_next, self.grid
        )
        if violations:
            raise PhysicalCollisionError(f"t={t}: " + "; ".join(violations))

        # 7. counterfactual reference conflicts
        reference_events: frozenset[ConflictEvent] | None = None
        controller_events: frozenset[ConflictEvent] | None = None
        if self.shield is not None:
            raw = (
                events
                if arb_proposal is proposal
                else detect_conflicts(
                    t, positions, proposal, pending.obstacles_now, pending.obstacles_next, self.grid
                )
            )
            self.metrics.record_shield(len(shield_events), raw, events)
        if pending.view.suggested is not None:
            ref = [
                s if pending.view.masks[i][s] else Action.STAY
                for i, s in enumerate(pending.view.suggested)
            ]
            reference_events = detect_conflicts(
                t, positions, ref, pending.obstacles_now, pending.obstacles_next, self.grid
            )
            controller_events = (
                events
                if arb_proposal is proposal
                else detect_conflicts(
                    t, positions, proposal, pending.obstacles_now, pending.obstacles_next, self.grid
                )
            )

        if self._reporting is not None and pending.view.suggested is not None:
            self.metrics.record_adherence(
                nonidle, self._reporting.plan_valid(), proposal, pending.view.suggested
            )

        # commit motion
        moved = [resolution.positions[i] != positions[i] for i in range(n)]
        for r, pos in zip(self.robots, resolution.positions, strict=True):
            r.position = pos
        for o, nxt in zip(self.obstacles, pending.obstacles_next, strict=True):
            o.previous, o.position = o.position, nxt
        self.metrics.record_step(
            nonidle,
            proposal,
            moved,
            events,
            resolution.blocked,
            reference_events,
            controller_events,
            blocked_goal=blocked_goal,
        )

        # 8. pickups and deliveries at t + 1
        picked: list[tuple[int, int]] = []
        delivered: list[tuple[int, int]] = []
        for r in self.robots:
            if r.task is None:
                continue
            task = self.book.tasks[r.task]
            if r.mode is RobotMode.TO_PICKUP and r.position == task.pickup:
                task.do_pick(t + 1)
                self.metrics.leg_end(r.id)
                r.mode = RobotMode.TO_DROPOFF
                self.metrics.leg_start(r.id, self.oracle.distance(task.pickup, task.dropoff))
                picked.append((r.id, task.id))
            elif r.mode is RobotMode.TO_DROPOFF and r.position == task.dropoff:
                task.do_deliver(t + 1)
                self.metrics.leg_end(r.id)
                r.mode, r.task = RobotMode.IDLE, None
                delivered.append((r.id, task.id))

        # termination / truncation
        self.t = t + 1
        self._pending = None
        all_delivered = bool(self.book.tasks) and all(
            tk.status is TaskStatus.DELIVERED for tk in self.book.tasks.values()
        )
        self.terminated = (
            self.stop_when_done and all_delivered and not self.arrivals.has_future_releases(self.t)
        )
        self.truncated = not self.terminated and self.t >= self.max_steps

        outcome = StepOutcome(
            t=self.t,
            actions=tuple(proposal),
            events=events,
            blocked=resolution.blocked,
            released=tuple(pending.released),
            assigned=tuple(pending.assigned),
            picked=tuple(picked),
            delivered=tuple(delivered),
            map_events=tuple(pending.map_events),
            reference_events=reference_events,
            terminated=self.terminated,
            truncated=self.truncated,
            controller_events=controller_events,
            arbitrated_actions=tuple(arb_proposal) if arb_proposal is not proposal else None,
            shield_events=tuple(shield_events),
        )
        if self.done and self._reporting is not None:
            self._reporting.finish(self._build_view(None))
        if self.trace is not None:
            self.trace.write(self._step_record(outcome, pending))
            if self.done:
                self.trace.write({"type": "summary", "metrics": self.summary().to_dict()})
        return outcome

    def step(self, controller: Callable[[WorldView], Mapping[int, Action]]) -> StepOutcome:
        """Convenience: ``prepare_step`` → ``controller(view)`` → ``apply_actions``."""
        view = self.prepare_step()
        return self.apply_actions(controller(view))

    def summary(self) -> EpisodeMetrics:
        metrics = self.metrics.summary(self.book, self.t)
        if self._reporting is not None:
            metrics.extra["planning"] = self._reporting.episode_stats()
        if self.shield is not None:
            metrics.extra["shield"] = {**self.shield.stats(), **self.metrics.shield_summary()}
        if self.controller_stats is not None:
            metrics.extra["controller"] = self.controller_stats()
        return metrics

    # ================================================================== views and traces
    def _build_view(
        self, suggested: tuple[Action, ...] | None, *, extras: bool = False
    ) -> WorldView:
        tasks = self.book.tasks
        paths: tuple[tuple[Cell, ...] | None, ...] | None = None
        goal_distance: tuple[int | None, ...] | None = None
        if extras and "paths" in self.view_extras and isinstance(self.coordinator, PathReporting):
            paths = self.coordinator.reference_paths(self.path_limit)
        if extras and "goal_distance" in self.view_extras:
            goals = [r.goal(tasks) for r in self.robots]
            goal_distance = tuple(
                None if g is None else self.oracle.capped_distance(r.position, g)
                for r, g in zip(self.robots, goals, strict=True)
            )
        return WorldView(
            t=self.t,
            max_steps=self.max_steps,
            map=MapView(self.grid),
            robots=tuple(
                RobotView(r.id, r.position, r.mode, r.task, r.goal(tasks)) for r in self.robots
            ),
            obstacles=tuple(ObstacleView(o.id, o.position, o.velocity) for o in self.obstacles),
            tasks=tuple(
                TaskView(
                    tk.id,
                    tk.pickup,
                    tk.dropoff,
                    tk.priority,
                    tk.release,
                    tk.status,
                    tk.robot,
                    self.task_available(tk),
                )
                for tk in tasks.values()
            ),
            masks=tuple(action_mask(self.grid, r.position) for r in self.robots),
            suggested=suggested,
            reference_paths=paths,
            goal_distance=goal_distance,
        )

    def _header(self) -> dict[str, object]:
        spec = self.scenario
        return {
            "type": "header",
            "format": 1,
            "scenario": spec.name,
            "seed": self.seed,
            "width": self.grid.width,
            "height": self.grid.height,
            "map": self.grid.to_rows(),
            "max_steps": self.max_steps,
            "num_priorities": spec.num_priorities,
            "robots": [list(r.position) for r in self.robots],
            "obstacles": [[*o.position, o.model.name] for o in self.obstacles],
            "tasks": [
                [tk.id, *tk.pickup, *tk.dropoff, tk.priority, tk.release]
                for tk in self.book.tasks.values()
            ],
            "coordinator": self.coordinator.name,
            **({"shield": self.shield.mode} if self.shield is not None else {}),
        }

    def _step_record(self, out: StepOutcome, pending: _Pending) -> dict[str, object]:
        rec: dict[str, object] = {
            "type": "step",
            "t": out.t,
            "actions": "".join(a.symbol for a in out.actions),
            "robots": [
                [*r.position, _MODE_CODE[r.mode], -1 if r.task is None else r.task]
                for r in self.robots
            ],
            "obstacles": [list(o.position) for o in self.obstacles],
            "events": [e.to_json() for e in sorted(out.events)],
            "blocked": sorted(out.blocked),
            "released": list(out.released),
            "assigned": [list(p) for p in out.assigned],
            "picked": [list(p) for p in out.picked],
            "delivered": [list(p) for p in out.delivered],
        }
        if pending.new_tasks:
            rec["new_tasks"] = [
                [
                    j,
                    *self.book.tasks[j].pickup,
                    *self.book.tasks[j].dropoff,
                    self.book.tasks[j].priority,
                    self.book.tasks[j].release,
                ]
                for j in pending.new_tasks
            ]
        if pending.unassigned:
            rec["unassigned"] = list(pending.unassigned)
        if out.map_events:
            rec["map_events"] = [
                [*e.cell, "block" if e.block else "unblock"] for e in out.map_events
            ]
        if pending.view.suggested is not None:  # X-10: the real suggested actions (Issue #4)
            rec["suggested"] = "".join(a.symbol for a in pending.view.suggested)
        if out.reference_events is not None:
            rec["reference_events"] = [e.to_json() for e in sorted(out.reference_events)]
        if out.shield_events:
            rec["shield"] = [e.to_json() for e in out.shield_events]
            assert out.arbitrated_actions is not None
            rec["arb_actions"] = "".join(a.symbol for a in out.arbitrated_actions)
        if self._reporting is not None:
            rec.update(self._reporting.trace_fields())
        return rec

    # ================================================================== convenience
    def run(
        self,
        controller: Callable[[WorldView], Mapping[int, Action]],
        on_step: Callable[[Engine, StepOutcome], None] | None = None,
    ) -> EpisodeMetrics:
        """Run until termination or truncation; returns the episode metrics."""
        while not self.done:
            outcome = self.step(controller)
            if on_step is not None:
                on_step(self, outcome)
        return self.summary()
