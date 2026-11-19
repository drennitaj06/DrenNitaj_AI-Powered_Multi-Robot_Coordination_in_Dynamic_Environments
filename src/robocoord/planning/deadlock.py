"""Deadlock detection and detour resolution (algorithms §5.5; Issue #4 stage 5B).

A coordinator-level, **opt-in** mechanism. It never chooses robot actions: it only replaces the
goal the path monitor plans to (a *passing cell*) for one robot of a deadlocked cycle, so the
controller (normally the right-of-way baseline) keeps all of its rules. T4 remains the planning
monitor's stall trigger; the detour is attributed to the separate trigger T7.

**Wait-for graph.** Robot ``i`` is *active* if it is on a valid plan and not at its goal; its next
planned cell is ``n_i``. There is an edge ``i → j`` iff robot ``i`` is active and ``n_i`` is
robot ``j``'s cell. Every robot has at most one outgoing edge, so cycles are vertex-disjoint and
found in O(N); on the 4-connected grid (bipartite by the parity of ``x + y``) every cycle has
even length: a head-on pair or a rotation of ≥ 4 robots.

**Detection.** A cycle with the same members on the same cells for ``detect_steps`` consecutive
steps (*persistent*), or the ``recur_count``-th new occurrence of the same member set within
``recur_window`` steps (*recurrent*: repeated mutual blocking dissolved by replans). Waiting that
is not part of a detected deadlock is *temporary waiting*. A robot blocked by an idle robot or a
robot at its goal is not in a cycle and is not handled by this resolver. Idle robots standing on
an active task endpoint are parked by endpoint recovery (``planning/endpoint.py``, trigger T8),
not here.

**Resolution.** Candidates in ascending ``(π̃_i, −i)`` order (lowest effective priority first; on
equal priority the higher id yields); the first candidate with a *passing cell* yields. The
passing cell ``s`` is the nearest cell, by BFS from the candidate over static free cells with
every robot and obstacle cell blocked, that is (P1) on no other member's remaining reference path,
(P2) not an obstacle's constant-velocity prediction, (P3) not reserved by another detour, and not
the candidate's own task goal; ties by ``(x, y)``. No passing cell for any candidate → the
deadlock is reported **unresolved** (``no_passing_place``); nothing is forced.

**Detour state machine** (per yielder): PENDING → TO_WAYPOINT (goal = ``s``, T7) → AT_WAYPOINT →
released (goal = task goal, T7) when every other member is at its goal, or is off the yielder's
route ``P_y⁰`` at detection and its next ``release_horizon`` planned cells avoid ``P_y⁰``; or
released by timeout after ``hold`` steps at ``s``. Aborted when ``s`` becomes blocked, another
member's plan now runs through ``s``, the detour plan is ``unreachable``, or the yielder makes no
progress for ``abort_steps``; a still-persisting cycle is then detected again. Cancelled when the
yielder's task goal changes. At most ``1 + max_retries`` detours are started per (member set,
member task goals); beyond that the deadlock is reported unresolved (``unresolved_retry_limit``).
A map change clears unresolved marks and attempt counts. While a detour is active, T4 is
suspended for its members.

*Changes from the reviewed design (documented in `04` §5.5):* the release condition looks only
``release_horizon`` cells ahead (the whole-path test kept yielders waiting long after the conflict
had dissolved), and the retry bound counts detour *starts*, not only aborts, so the earlier release
cannot create an unbounded detour loop.

Determinism: no randomness; robots, cycles and cells are processed in fixed orders.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from robocoord.core.errors import ConfigError
from robocoord.core.types import Cell
from robocoord.planning.monitor import PathMonitor, PlanCall

STAT_KEYS: tuple[str, ...] = (
    "detected_persistent",
    "detected_recurrent",
    "detours_started",
    "released",
    "released_timeout",
    "aborted",
    "cancelled",
    "no_passing_place",
    "no_passing_place_dynamic",
    "unresolved_retry_limit",
    "cleared_on_map_change",
    "detour_robot_steps",
)


@dataclass(frozen=True, slots=True)
class DetourParams:
    """Resolver parameters (proposed defaults of the stage 5B design; algorithms §5.5).

    ``detect_steps`` (T_dl) must be smaller than the monitor's ``stall_threshold`` (checked by the
    coordinator) so that detection precedes T4. ``search_radius`` ``None`` = unbounded BFS.
    ``alpha`` must equal the allocation's aging rate (effective priority ``π̃``).
    """

    detect_steps: int = 2
    recur_count: int = 3
    recur_window: int = 20
    abort_steps: int = 4
    hold_extra: int = 4
    max_retries: int = 2
    release_horizon: int = 5
    search_radius: int | None = None
    alpha: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "detect_steps",
            "recur_count",
            "recur_window",
            "abort_steps",
            "max_retries",
            "release_horizon",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"deadlock.{name}", "must be >= 1")
        if self.hold_extra < 0:
            raise ConfigError("deadlock.hold_extra", "must be >= 0")
        if self.search_radius is not None and self.search_radius < 1:
            raise ConfigError("deadlock.search_radius", "must be >= 1 or None")
        if not self.alpha >= 0:
            raise ConfigError("deadlock.alpha", "must be >= 0")


class DetourState(Enum):
    PENDING = "pending"
    TO_WAYPOINT = "to_waypoint"
    AT_WAYPOINT = "at_waypoint"


@dataclass
class Detour:
    yielder: int
    members: frozenset[int]
    others: tuple[int, ...]
    waypoint: Cell
    avoid: frozenset[Cell]  # the other members' cells at detection: not on the detour route
    route0: frozenset[Cell]  # the yielder's remaining task path at detection (P_y⁰)
    task_goal: Cell
    hold: int  # max steps at the passing cell before a timeout release
    state: DetourState
    started: int
    last_pos: Cell
    last_progress: int
    arrived: int | None = None


@dataclass(frozen=True, slots=True)
class DeadlockEvent:
    t: int
    kind: str
    members: tuple[int, ...]
    robot: int | None = None
    cell: Cell | None = None

    def to_json(self) -> list[object]:
        return [self.kind, list(self.members), self.robot, list(self.cell) if self.cell else None]


@dataclass(frozen=True, slots=True)
class Override:
    """What the coordinator passes to the monitor for one robot this step (``trigger``: the
    attribution of a temporary-goal change, T7 here; endpoint recovery uses T8;
    ``count_active``: see :class:`~robocoord.planning.monitor.RobotInput`)."""

    goal: Cell | None
    detour: bool = False
    hold_stall: bool = False
    avoid: frozenset[Cell] = frozenset()
    trigger: str = "T7"
    count_active: bool = True


class _Free(Protocol):
    def is_free(self, c: Cell) -> bool: ...

    @property
    def version(self) -> int: ...


def find_cycles(succ: Mapping[int, int]) -> list[tuple[int, ...]]:
    """Cycles of a functional graph (out-degree ≤ 1), each starting at its smallest member, sorted
    by that member. Time O(N)."""
    color: dict[int, int] = {}  # 1 = on the current walk, 2 = done
    cycles: list[tuple[int, ...]] = []
    for start in sorted(succ):
        if start in color:
            continue
        walk: list[int] = []
        v: int | None = start
        while v is not None and v not in color:
            color[v] = 1
            walk.append(v)
            v = succ.get(v)
        if v is not None and color.get(v) == 1:
            cyc = walk[walk.index(v) :]
            k = cyc.index(min(cyc))
            cycles.append(tuple(cyc[k:] + cyc[:k]))
        for u in walk:
            color[u] = 2
    return sorted(cycles, key=lambda c: c[0])


def passing_cell(
    grid: _Free,
    start: Cell,
    blocked: set[Cell],
    forbidden: set[Cell],
    radius: int | None = None,
) -> Cell | None:
    """Nearest cell ``s ≠ start`` reachable from ``start`` through free, unblocked cells with
    ``s ∉ forbidden``; ties by ``(x, y)``. ``None`` if there is none. Time O(V)."""
    seen = {start}
    frontier = [start]
    dist = 0
    while frontier:
        dist += 1
        if radius is not None and dist > radius:
            return None
        nxt: list[Cell] = []
        for x, y in frontier:
            for c in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):
                if c in seen or c in blocked or not grid.is_free(c):
                    continue
                seen.add(c)
                nxt.append(c)
        hits = [c for c in nxt if c not in forbidden]
        if hits:
            return min(hits)
        frontier = nxt
    return None


class DeadlockResolver:
    """Opt-in deadlock detection and detour resolution for :class:`AllocationCoordinator`."""

    def __init__(self, params: DetourParams | None = None) -> None:
        self.params = params or DetourParams()
        self.reset(0)

    def reset(self, map_version: int) -> None:
        self.detours: dict[int, Detour] = {}
        self.stats: Counter[str] = Counter()
        self.events: list[DeadlockEvent] = []  # all events of the episode
        self.step_events: list[DeadlockEvent] = []
        self._persist: dict[tuple[tuple[int, ...], tuple[Cell, ...]], int] = {}
        self._occurrences: dict[frozenset[int], deque[int]] = {}
        # detour attempts per (member set, member task goals): at most 1 + max_retries
        self._attempts: Counter[tuple[frozenset[int], tuple[Cell | None, ...]]] = Counter()
        # member set → earliest step to re-check a temporarily blocked passing place
        self._recheck: dict[frozenset[int], int] = {}
        self._unresolved: dict[frozenset[int], tuple[int, tuple[Cell | None, ...]]] = {}
        self._version = map_version

    # ------------------------------------------------------------------ before the monitor
    def overrides(self, t: int, task_goals: Mapping[int, Cell | None]) -> dict[int, Override]:
        """Goal overrides for this step; cancels detours whose task goal changed."""
        self.step_events = []
        out: dict[int, Override] = {}
        for y in sorted(self.detours):
            d = self.detours[y]
            if task_goals.get(y) != d.task_goal:
                self._emit(t, "cancelled", d.members, y, d.waypoint)
                del self.detours[y]
                continue
            if d.state is DetourState.PENDING:
                d.state, d.last_progress = DetourState.TO_WAYPOINT, t
            avoid = d.avoid if d.state is DetourState.TO_WAYPOINT else frozenset()
            out[y] = Override(d.waypoint, detour=True, hold_stall=True, avoid=avoid)
            for j in d.others:
                if j not in self.detours:
                    out[j] = Override(task_goals.get(j), hold_stall=True)
            self.stats["detour_robot_steps"] += 1
        return out

    def in_detour(self, robot: int) -> bool:
        d = self.detours.get(robot)
        return d is not None and d.state is not DetourState.PENDING

    # ------------------------------------------------------------------ after the monitor
    def observe(
        self,
        t: int,
        grid: _Free,
        positions: Sequence[Cell],
        task_goals: Mapping[int, Cell | None],
        priorities: Mapping[int, float],
        obstacles: Mapping[Cell, Cell],
        monitor: PathMonitor,
        calls: Sequence[PlanCall],
    ) -> None:
        """Update active detours, then detect deadlocks on the state at ``t``."""
        if grid.version != self._version:
            self._version = grid.version
            for members in sorted(self._unresolved, key=sorted):
                self._emit(t, "cleared_on_map_change", members)
            self._unresolved.clear()
            self._attempts.clear()
            self._recheck.clear()

        def remaining(j: int) -> tuple[Cell, ...]:
            return monitor.reference_path(j) or ()

        failed = {c.robot for c in calls if c.status == "unreachable"}
        for y in sorted(self.detours):
            self._update_detour(t, grid, self.detours[y], positions, task_goals, remaining, failed)

        n = len(positions)
        at = {positions[i]: i for i in range(n)}
        succ: dict[int, int] = {}
        for i in range(n):
            nxt = monitor.next_cell(i, positions[i])
            if nxt is not None and nxt in at:
                succ[i] = at[nxt]
        busy = set(self.detours)
        for d in self.detours.values():
            busy.update(d.members)
        seen: dict[tuple[tuple[int, ...], tuple[Cell, ...]], int] = {}
        for cyc in find_cycles(succ):
            members = frozenset(cyc)
            key = (tuple(sorted(cyc)), tuple(positions[i] for i in sorted(cyc)))
            count = self._persist.get(key, 0) + 1
            seen[key] = count
            if count == 1:
                occ = self._occurrences.setdefault(members, deque())
                occ.append(t)
                while occ and occ[0] <= t - self.params.recur_window:
                    occ.popleft()
            if members & busy:
                continue  # managed by an active detour (abort / release rules)
            if count >= self.params.detect_steps:
                kind = "detected_persistent"
            elif count == 1 and len(self._occurrences[members]) >= self.params.recur_count:
                kind = "detected_recurrent"
            else:
                continue  # temporary (so far)
            stamp = (self._version, tuple(task_goals.get(i) for i in sorted(members)))
            if members in self._unresolved:
                if self._unresolved[members] == stamp:
                    continue  # already reported; waits for a map or goal change
                del self._unresolved[members]  # a member's goal changed: evaluate again
            if t < self._recheck.get(members, t):
                continue  # temporarily blocked passing places: re-check later
            self._emit(t, kind, members)
            attempt_key = (members, stamp[1])
            if self._attempts[attempt_key] > self.params.max_retries:
                self._unresolved[members] = stamp
                self._emit(t, "unresolved_retry_limit", members)
                continue
            status = self._start(
                t, grid, cyc, positions, task_goals, priorities, obstacles, remaining
            )
            if status == "started":
                self._attempts[attempt_key] += 1
                self._recheck.pop(members, None)
            elif status == "dynamic":
                self._recheck[members] = t + self.params.abort_steps
                self._emit(t, "no_passing_place_dynamic", members)
            else:
                self._unresolved[members] = stamp
                self._emit(t, "no_passing_place", members)
        self._persist = seen

    # ------------------------------------------------------------------ internals
    def _start(
        self,
        t: int,
        grid: _Free,
        cyc: tuple[int, ...],
        positions: Sequence[Cell],
        task_goals: Mapping[int, Cell | None],
        priorities: Mapping[int, float],
        obstacles: Mapping[Cell, Cell],
        remaining: Any,
    ) -> str:
        """Start a detour: ``"started"``; else ``"dynamic"`` if a passing cell exists once
        obstacles and robots outside the cycle are ignored (temporary blockage), or ``"static"``
        if none exists even then (re-evaluated only after a map or goal change)."""
        members = frozenset(cyc)
        robots_at = set(positions)
        blocked = robots_at | set(obstacles)
        predicted = {
            (z[0] + v[0], z[1] + v[1])
            for z, v in obstacles.items()
            if grid.is_free((z[0] + v[0], z[1] + v[1]))
        }
        reserved = {d.waypoint for d in self.detours.values()}
        for y in sorted(cyc, key=lambda i: (priorities.get(i, 0.0), -i)):
            goal = task_goals.get(y)
            if goal is None:
                continue
            others = tuple(j for j in cyc if j != y)
            forbidden = set(predicted) | reserved | {goal}
            for j in others:
                forbidden.update(remaining(j))
            s = passing_cell(
                grid, positions[y], blocked - {positions[y]}, forbidden, self.params.search_radius
            )
            if s is None:
                continue
            longest = max((len(remaining(j)) - 1 for j in others), default=0)
            self.detours[y] = Detour(
                yielder=y,
                members=members,
                others=others,
                waypoint=s,
                avoid=frozenset(positions[j] for j in others),
                route0=frozenset(remaining(y) or (positions[y],)),
                task_goal=goal,
                hold=2 * max(longest, 0) + self.params.hold_extra,
                state=DetourState.PENDING,
                started=t,
                last_pos=positions[y],
                last_progress=t,
            )
            self._emit(t, "detours_started", members, y, s)
            return "started"
        member_cells = {positions[i] for i in cyc}
        for y in cyc:
            goal = task_goals.get(y)
            if goal is None:
                continue
            forbidden = reserved | {goal}
            for j in cyc:
                if j != y:
                    forbidden.update(remaining(j))
            relaxed = passing_cell(
                grid,
                positions[y],
                member_cells - {positions[y]},
                forbidden,
                self.params.search_radius,
            )
            if relaxed is not None:
                return "dynamic"
        return "static"

    def _update_detour(
        self,
        t: int,
        grid: _Free,
        d: Detour,
        positions: Sequence[Cell],
        task_goals: Mapping[int, Cell | None],
        remaining: Any,
        failed: set[int],
    ) -> None:
        if d.state is DetourState.PENDING:
            return
        y, pos = d.yielder, positions[d.yielder]
        if pos != d.last_pos:
            d.last_pos, d.last_progress = pos, t
        if d.state is DetourState.TO_WAYPOINT and pos == d.waypoint:
            d.state, d.arrived = DetourState.AT_WAYPOINT, t
        horizon = self.params.release_horizon
        clear = all(
            positions[j] == task_goals.get(j)
            or (positions[j] not in d.route0 and not (set(remaining(j)[: horizon + 1]) & d.route0))
            for j in d.others
        )
        if clear:
            self._finish(t, d, "released")
            return
        invalid = (
            not grid.is_free(d.waypoint)
            or y in failed
            or any(d.waypoint in remaining(j) for j in d.others)
        )
        stalled = (
            d.state is DetourState.TO_WAYPOINT and t - d.last_progress >= self.params.abort_steps
        )
        if invalid or stalled:
            self._finish(t, d, "aborted")
        elif d.arrived is not None and t - d.arrived >= d.hold:
            self._finish(t, d, "released_timeout")

    def _finish(self, t: int, d: Detour, kind: str) -> None:
        self._emit(t, kind, d.members, d.yielder, d.waypoint)
        del self.detours[d.yielder]

    def _emit(
        self,
        t: int,
        kind: str,
        members: frozenset[int] | tuple[int, ...],
        robot: int | None = None,
        cell: Cell | None = None,
    ) -> None:
        ev = DeadlockEvent(t, kind, tuple(sorted(members)), robot, cell)
        self.events.append(ev)
        self.step_events.append(ev)
        self.stats[kind] += 1

    def summary(self) -> dict[str, Any]:
        """Deterministic counters for the episode summary."""
        out: dict[str, Any] = {k: self.stats.get(k, 0) for k in STAT_KEYS}
        out["unresolved_at_end"] = len(self._unresolved)
        out["active_at_end"] = len(self.detours)
        return out
