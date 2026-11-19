"""Allocation policy: triggers, eligibility, pre-emption and application (algorithms §4.3;
model §8; FR-TASK-5).

**Triggers** (algorithms §4.3): a task was released; a task was delivered; a robot became idle;
a robot released its task because its pickup was statically ``unreachable``; a map event changed
the grid. Each step, every trigger seen since the last step is collected and **at most one
auction** is run.

* A release on ``unreachable`` is detected by the path monitor *after* this step's auction, so it
  is carried over and triggers the auction of the next step (``unreachable``).
* A solve that hits its time or bid cap changes nothing and is retried at the next step
  (``retry``). The production auction has no cap by default.
* Aging changes benefits continuously, but is not a trigger: as specified, it acts at the next
  allocation event.

**Eligibility** (model §8). Robots in mode idle or to_pickup (``R_f``); carried tasks are never
pre-empted. Tasks that are *available* (model §4) with status pending or assigned (``J_p``); a
to_pickup robot's current task ``τ_i`` gets the hysteresis bonus ``h``.

**Application.** For every eligible robot, in id order, compare the new task ``σ_i`` with
``τ_i``. First every robot whose task changes or is lost is unassigned (its task returns to
pending), then every robot with a new task is assigned, so a task can move from one robot to
another within one event. A robot whose current task became unavailable and gets no new task is
unassigned (the task stays pending and excluded while unavailable, model §5.3 step 4). The new
goal ``g_i`` (pickup cell) is what makes the path monitor replan with trigger T1.

**Pre-emption** happens only through this auction and only beyond the hysteresis margin, because
the hysteresis term is part of the benefits (model §8, FR-TASK-5). Picked tasks are never in
``J_p``.

**Ties keep the current assignment** (:func:`prefer_current`). Model §8: a robot keeps its task
"unless another assignment is better by more than ``h``", so at a gain of exactly ``h`` it must
keep it. With integer distances such exact ties occur, and either optimal matching could be
returned. The allocator therefore gets the matrix ``(k+1)·A + 1[(i, j) current]``, an exact
lexicographic refinement: every optimum of it is an optimum of ``A`` that keeps as many current
assignments as possible (proof in the function).

Determinism: robots and tasks are processed in id order and both allocators are deterministic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from robocoord.allocation.base import Allocator, SolveStats
from robocoord.allocation.benefits import (
    AllocationProblem,
    BenefitConfig,
    DistanceFn,
    RobotCandidate,
    TaskCandidate,
    build_problem,
)
from robocoord.core.types import Cell
from robocoord.sim.coordinator import AssignmentAPI
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode, TaskStatus

TRIGGER_KINDS: tuple[str, ...] = (
    "released",  # a task was released (scheduled or arrival)
    "delivered",  # a task was delivered
    "idle",  # a robot became idle (other than through this policy)
    "unreachable",  # a robot released its task: pickup statically unreachable (previous step)
    "map_event",  # the grid changed
    "retry",  # the previous solve hit its time/bid cap
)


def prefer_current(problem: AllocationProblem) -> AllocationProblem:
    """Break ties between optimal matchings in favour of keeping current assignments.

    Returns the problem with matrix ``A' = (k+1)·A + E``, where ``E_ij = 1`` for each feasible
    current pair ``(i, τ_i)`` and 0 otherwise (dummy and infeasible entries stay 0).

    *Proof (exact).* For a matching ``X`` let ``V(X) = Σ A`` and ``e(X)`` = number of current pairs
    in ``X``, ``0 ≤ e(X) ≤ k``. Then ``V'(X) = (k+1)·V(X) + e(X)``. If ``V(X) > V(Y)`` then, the
    entries being integers, ``V(X) ≥ V(Y) + 1`` and ``V'(X) ≥ (k+1)V(Y) + k + 1 > V'(Y)``. So every
    maximiser of ``V'`` maximises ``V`` (so the auction stays exact and of maximum cardinality),
    and among them maximises ``e``. For the sequential auction, ``A_ij > A_uv`` implies
    ``A'_ij > A'_uv``: sorting by ``A'`` is sorting by ``A`` with ties broken towards current
    pairs, i.e. greedy on ``A`` with another tie rule. The ½ bound (its proof holds for any tie
    rule) and maximality therefore still hold for ``A``. Real-valued values (``problem.value``)
    are unchanged.
    """
    k = problem.size
    if k == 0 or not problem.current:
        return problem
    rows = {rid: i for i, rid in enumerate(problem.robots)}
    cols = {tid: j for j, tid in enumerate(problem.tasks)}
    matrix = problem.matrix * (k + 1)
    if int(problem.matrix.max()) >= 2**62 // (k + 2):
        raise OverflowError("tie-break scaling would overflow int64; reduce the integer scale s")
    for rid, tid in problem.current.items():
        i, j = rows[rid], cols[tid]
        if problem.feasible[i, j]:
            matrix[i, j] += 1
    return replace(problem, matrix=matrix)


@dataclass(frozen=True)
class AllocationEvent:
    """One allocation event: why it ran, what it saw and what it changed."""

    t: int
    triggers: tuple[str, ...]
    eligible_robots: int  # |R_f|
    available_tasks: int  # |J_p|
    pairs: tuple[tuple[int, int], ...]  # the allocator's robot → task matching
    assigned: tuple[tuple[int, int], ...]  # (robot, task) assignments applied this event
    unassigned: tuple[tuple[int, int], ...]  # (robot, previous task) released this event
    switched: tuple[tuple[int, int, int], ...]  # (robot, old task, new task): pre-emptions
    value: float  # Σ value of the pairs, original units (offset excluded)
    stats: SolveStats

    @property
    def lost(self) -> tuple[tuple[int, int], ...]:
        """Robots that lost their task and got none (pre-empted or task unavailable)."""
        new = {r for r, _ in self.assigned}
        return tuple((r, j) for r, j in self.unassigned if r not in new)


@dataclass(frozen=True)
class RobotState:
    """A robot's mode, task and goal after an allocation event."""

    mode: RobotMode
    task: int | None
    goal: Cell | None


class AllocationPolicy:
    """Event-triggered allocation with a pluggable :class:`Allocator` (auction or sequential)."""

    def __init__(self, allocator: Allocator, config: BenefitConfig | None = None) -> None:
        self.allocator = allocator
        self.config = config or BenefitConfig()
        self.events: list[AllocationEvent] = []
        self.trigger_counts: Counter[str] = Counter()
        self.skipped = 0  # triggered, but no eligible robot or no available task
        self._released: set[int] = set()
        self._delivered: set[int] = set()
        self._modes: dict[int, RobotMode] = {}
        self._version = 0
        self._carry: set[str] = set()

    def reset(self, view: WorldView) -> None:
        """Start of an episode. Nothing is marked as seen, so tasks released at ``t = 0``
        trigger the first auction."""
        self.events = []
        self.trigger_counts = Counter()
        self.skipped = 0
        self._released = set()
        self._delivered = set()
        self._modes = {r.id: r.mode for r in view.robots}
        self._version = view.map.version
        self._carry = set()

    # ------------------------------------------------------------------ triggers
    def detect(self, view: WorldView) -> tuple[str, ...]:
        """Triggers since the previous call, in :data:`TRIGGER_KINDS` order; updates the
        snapshots."""
        fired = set(self._carry)
        self._carry = set()
        released = {tk.id for tk in view.tasks if tk.status is not TaskStatus.UNRELEASED}
        if released - self._released:
            fired.add("released")
        self._released = released
        delivered = {tk.id for tk in view.tasks if tk.status is TaskStatus.DELIVERED}
        if delivered - self._delivered:
            fired.add("delivered")
        self._delivered = delivered
        if any(
            r.mode is RobotMode.IDLE and self._modes.get(r.id) is not RobotMode.IDLE
            for r in view.robots
        ):
            fired.add("idle")
        self._modes = {r.id: r.mode for r in view.robots}
        if view.map.version != self._version:
            fired.add("map_event")
            self._version = view.map.version
        return tuple(k for k in TRIGGER_KINDS if k in fired)

    def notify_release(self, robot: int) -> None:
        """A robot released its task after an ``unreachable`` plan (model §5.3 step 4): trigger a
        re-auction at the next step (at most one auction per step)."""
        self._carry.add("unreachable")
        self._modes[robot] = RobotMode.IDLE  # not a separate ``idle`` trigger

    # ------------------------------------------------------------------ one step
    def step(
        self, view: WorldView, api: AssignmentAPI, distance: DistanceFn
    ) -> tuple[dict[int, RobotState], AllocationEvent | None]:
        """Detect triggers and, if any fired, run one allocation and apply it.

        Returns every robot's state after the step and the event (``None`` if no auction ran).
        """
        state = {r.id: RobotState(r.mode, r.task, r.goal) for r in view.robots}
        triggers = self.detect(view)
        if not triggers:
            return state, None
        self.trigger_counts.update(triggers)
        eligible = [r for r in view.robots if r.mode in (RobotMode.IDLE, RobotMode.TO_PICKUP)]
        available = [tk for tk in view.tasks if tk.available]  # implies pending or assigned
        holding = {r.id: r.task for r in eligible if r.task is not None}
        if not eligible or (not available and not holding):
            self.skipped += 1
            return state, None
        robots = [
            RobotCandidate(r.id, r.position, r.task if r.mode is RobotMode.TO_PICKUP else None)
            for r in eligible
        ]
        tasks = [
            TaskCandidate(tk.id, tk.pickup, tk.dropoff, tk.priority, tk.release) for tk in available
        ]
        problem = build_problem(robots, tasks, view.t, distance, self.config)
        result = self.allocator.allocate(prefer_current(problem))
        if result.stats.timed_out:
            self._carry.add("retry")
            event = AllocationEvent(
                view.t, triggers, len(robots), len(tasks), (), (), (), (), 0.0, result.stats
            )
            self.events.append(event)
            return state, event

        new = dict(result.pairs)
        pickups = {tk.id: tk.pickup for tk in available}
        unassigned = [(rid, old) for rid, old in sorted(holding.items()) if new.get(rid) != old]
        assigned = [(rid, j) for rid, j in sorted(new.items()) if holding.get(rid) != j]
        for rid, _ in unassigned:  # first free every task that changes hands ...
            api.unassign_task(rid)
            state[rid] = RobotState(RobotMode.IDLE, None, None)
        for rid, j in assigned:  # ... then assign
            api.assign_task(rid, j)
            state[rid] = RobotState(RobotMode.TO_PICKUP, j, pickups[j])
        for rid, st in state.items():
            self._modes[rid] = st.mode  # changes made here are not ``idle`` triggers
        switched = tuple(
            (rid, old, new[rid]) for rid, old in unassigned if new.get(rid) is not None
        )
        event = AllocationEvent(
            t=view.t,
            triggers=triggers,
            eligible_robots=len(robots),
            available_tasks=len(tasks),
            pairs=tuple(sorted(result.pairs)),
            assigned=tuple(assigned),
            unassigned=tuple(unassigned),
            switched=switched,
            value=result.value,
            stats=result.stats,
        )
        self.events.append(event)
        return state, event

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict[str, object]:
        """Deterministic allocation statistics for the episode summary (no timings)."""
        ev = self.events
        reassigned = Counter[int]()
        for e in ev:
            for _, old in e.unassigned:
                reassigned[old] += 1
        return {
            "allocator": self.allocator.name,
            "auctions": len(ev),
            "skipped": self.skipped,
            "triggers": {k: self.trigger_counts.get(k, 0) for k in TRIGGER_KINDS},
            "assignments": sum(len(e.assigned) for e in ev),
            "preemptions": sum(len(e.switched) for e in ev),  # robot switched task
            "lost": sum(len(e.lost) for e in ev),  # robot lost its task, got none
            "tasks_reassigned": len(reassigned),  # tasks taken back at least once
            "timeouts": sum(1 for e in ev if e.stats.timed_out),
            "bids": sum(e.stats.bids for e in ev),
        }
