"""Endpoint recovery: idle-robot parking and retreat from persistently held task goals
(algorithms §5.6; model §5.3; Project Issue #4 endpoint follow-up).

A coordinator-level mechanism of the same kind as the deadlock resolver (`04` §5.5): it never
chooses actions. It only replaces the goal the path monitor plans to, for one robot at a time,
with a *temporary* cell (monitor trigger **T8**). The planner, monitor, controller, shield and
arbiter run unchanged, so every safety property of those layers holds for the resulting
proposals (the arbiter's Lemma 3.1 for any proposal; the baseline's collision property G19 for any
suggested actions). Obstacle motion (assumption A-5) is not changed.

**Active endpoints** ``E(t)``: the pickup and drop-off cells of released tasks that are pending
or assigned, and the drop-off cells of picked tasks.

**Parking (an idle robot on an active endpoint).** An idle robot (after this step's allocation)
standing on a cell of ``E(t)`` gets a parking goal: the nearest cell, by BFS from the robot over
static free cells not occupied by another robot or an obstacle, that is (K1) not in ``E(t)``,
(K2) not reserved (another robot's parking or retreat cell, or a deadlock passing cell),
(K3) not a cut vertex of the static free graph without the cells of the other stationary idle
robots and the reserved cells (parking there would disconnect the map), and (K4) not occupied;
ties by ``(x, y)``. The cells of robots within distance 2 at the start are avoided by its plans
(the robots waiting for the endpoint). The robot keeps priority 0 (it is idle), so every
task-holding robot has right of way over it. Parking ends when the robot reaches the cell
(*completed*), is assigned a task or taken over by the deadlock resolver (*cancelled*), or makes
no progress for
``abort_steps``, finds its cell statically blocked or turned into an active endpoint, or gets an
``unreachable`` plan (*aborted*; a new cell is chosen ``abort_steps`` later). If no cell exists
the robot stays and this is reported: ``no_parking_cell`` when no cell satisfies K1–K3 in the
robot's static component, ``parking_blocked_dynamic`` when one exists but none can be reached
now past robots and obstacles (re-checked every step).

**Retreat (a persistently held goal).** A robot carrying or assigned a task, adjacent to its
task goal ``g`` while ``g`` is occupied by the same *occupant* (a dynamic obstacle, or an idle
robot) waits. A *blockage episode* starts with the first such step and ends when the occupant
leaves ``g`` or the robot's task goal changes. It is *temporary* if it ends before the robot
has waited ``wait_steps`` (W) consecutive steps adjacent to ``g``, else *persistent*. On
reaching W the robot retreats to the nearest cell (BFS from the robot over static free cells
not occupied by robots or obstacles, radius ``retreat_radius``) at Manhattan distance ≥ 2 from
the occupant **and** from the robot's current cell (the cells the occupant could enter next),
not in ``E(t)``, not reserved and not an obstacle's constant-velocity prediction; ties by
``(x, y)``. It plans there (T8) with T4 suspended, holds at most ``hold_steps`` (H) steps, and
then restores its task goal (T8). An attempt *succeeds* when the occupant leaves ``g`` (the goal
is re-assessed every step), *times out* after H steps at the retreat cell, is *aborted* after
``abort_steps`` steps without progress, a statically blocked retreat cell or an ``unreachable``
plan, and is *cancelled* when the task goal changes or the resolver takes the robot over. At most
``max_attempts`` (M) attempts per blockage episode; when the robot has waited W steps again after
the last one the episode is *exhausted*: the robot keeps waiting and the blockage is reported
(``attempts_exhausted``; ``unresolved_*_at_end`` if it lasts to the end of the episode).

**Not claimed:** recovery does not guarantee delivery. Known unresolvable cases (`04` §5.6): an
occupant that never leaves (a one-waypoint patrol parked on the endpoint); an endpoint whose only
access is a corridor that an obstacle re-enters after the retreat (head-on with an obstacle that
waits for the robot); no retreat cell within the radius; no safe parking cell.

Determinism: no randomness; robots and cells are processed in fixed orders.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from robocoord.core.errors import ConfigError
from robocoord.core.types import Cell
from robocoord.planning.deadlock import Override
from robocoord.planning.monitor import PlanCall
from robocoord.world.entities import RobotMode, TaskStatus

#: Episode counters (``04`` §5.6 defines each one; every event is counted once).
STAT_KEYS: tuple[str, ...] = (
    "blockages_temporary",
    "blockages_persistent",
    "blockages_persistent_obstacle",
    "blockages_persistent_robot",
    "retreats_started",
    "retreats_succeeded",
    "retreats_timed_out",
    "retreats_aborted",
    "retreats_cancelled",
    "attempts_exhausted",
    "no_retreat_cell",
    "goal_reached_after_recovery",
    "parking_needed",
    "parking_started",
    "parking_completed",
    "parking_cancelled",
    "parking_aborted",
    "no_parking_cell",
    "parking_blocked_dynamic",
    "retreat_robot_steps",
    "parking_robot_steps",
)
END_KEYS: tuple[str, ...] = (
    "unresolved_obstacle_at_end",
    "unresolved_idle_robot_at_end",
    "idle_on_endpoint_at_end",
    "retreats_active_at_end",
    "parkings_active_at_end",
)


@dataclass(frozen=True, slots=True)
class EndpointParams:
    """Fixed before any validation run (documented in `04` §5.6; not tuned):

    * ``wait_steps`` W = 4 = the monitor's ``T_stall``, the model's existing bound for "a stall
      is not momentary" (model §11);
    * ``hold_steps`` H = 8 = 2·T_stall: time for the occupant to leave once the robot has
      cleared its exit;
    * ``max_attempts`` M = 3 retreats per blockage episode;
    * ``retreat_radius`` = 8 cells (BFS radius of the retreat search);
    * ``abort_steps`` = 4, as the deadlock resolver's ``abort_steps``.
    """

    wait_steps: int = 4
    hold_steps: int = 8
    max_attempts: int = 3
    retreat_radius: int = 8
    abort_steps: int = 4

    def __post_init__(self) -> None:
        for name in ("wait_steps", "hold_steps", "max_attempts", "retreat_radius", "abort_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"endpoint.{name}", "must be >= 1")


class _Map(Protocol):
    def is_free(self, c: Cell) -> bool: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def version(self) -> int: ...


class _TaskLike(Protocol):
    @property
    def pickup(self) -> Cell: ...

    @property
    def dropoff(self) -> Cell: ...

    @property
    def status(self) -> TaskStatus: ...


def active_endpoints(tasks: Iterable[_TaskLike]) -> frozenset[Cell]:
    """``E(t)``: endpoints still needed by released, undelivered tasks."""
    out: set[Cell] = set()
    for tk in tasks:
        if tk.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
            out.add(tk.pickup)
            out.add(tk.dropoff)
        elif tk.status is TaskStatus.PICKED:
            out.add(tk.dropoff)
    return frozenset(out)


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nbrs(c: Cell) -> tuple[Cell, Cell, Cell, Cell]:
    x, y = c
    return ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y))


def nearest_cell(
    grid: _Map,
    start: Cell,
    blocked: set[Cell] | frozenset[Cell],
    accept: Callable[[Cell], bool],
    radius: int | None = None,
) -> Cell | None:
    """Nearest cell ``c ≠ start`` reachable from ``start`` through static free cells not in
    ``blocked`` with ``accept(c)``; ties by ``(x, y)``; ``None`` if there is none within
    ``radius`` (``None`` = unbounded). Time O(V)."""
    seen = {start}
    frontier = [start]
    dist = 0
    while frontier:
        dist += 1
        if radius is not None and dist > radius:
            return None
        nxt: list[Cell] = []
        for c0 in frontier:
            for c in _nbrs(c0):
                if c in seen or c in blocked or not grid.is_free(c):
                    continue
                seen.add(c)
                nxt.append(c)
        hits = [c for c in nxt if accept(c)]
        if hits:
            return min(hits)
        frontier = nxt
    return None


def articulation_points(grid: _Map, removed: set[Cell] | frozenset[Cell]) -> frozenset[Cell]:
    """Cut vertices of the 4-connected graph of static free cells not in ``removed``
    (iterative Hopcroft–Tarjan; each component handled separately). Time O(V)."""
    w, h = grid.width, grid.height

    def ok(c: Cell) -> bool:
        return c not in removed and grid.is_free(c)

    disc: dict[Cell, int] = {}
    low: dict[Cell, int] = {}
    cut: set[Cell] = set()
    timer = 0
    for y in range(h):
        for x in range(w):
            root = (x, y)
            if root in disc or not ok(root):
                continue
            disc[root] = low[root] = timer
            timer += 1
            children = 0
            stack: list[tuple[Cell, Cell | None, int]] = [(root, None, 0)]
            while stack:
                v, parent, k = stack[-1]
                nb = _nbrs(v)
                advanced = False
                while k < 4:
                    u = nb[k]
                    k += 1
                    if not ok(u):
                        continue
                    if u not in disc:
                        stack[-1] = (v, parent, k)
                        disc[u] = low[u] = timer
                        timer += 1
                        if v == root:
                            children += 1
                        stack.append((u, v, 0))
                        advanced = True
                        break
                    if u != parent:
                        low[v] = min(low[v], disc[u])
                if advanced:
                    continue
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[v])
                    if p != root and low[v] >= disc[p]:
                        cut.add(p)
            if children > 1:
                cut.add(root)
    return frozenset(cut)


@dataclass
class Blockage:
    """One blockage episode of a robot's task goal (see the module docstring)."""

    task: int
    goal: Cell
    occupant: tuple[str, int]  # ("obstacle", id) or ("robot", id)
    started: int
    wait: int = 0  # consecutive steps waited adjacent to the goal
    persistent: bool = False
    attempts: int = 0
    exhausted: bool = False
    no_cell: bool = False


@dataclass
class Retreat:
    robot: int
    task: int
    task_goal: Cell
    cell: Cell
    started: int
    last_pos: Cell
    last_progress: int
    arrived: int | None = None


@dataclass
class Parking:
    robot: int
    cell: Cell
    avoid: frozenset[Cell]
    started: int
    last_pos: Cell
    last_progress: int


@dataclass
class _Need:
    """An idle robot standing on an active endpoint (one *need episode* per stay)."""

    cell: Cell
    retry_at: int = 0
    static_stamp: tuple[int, frozenset[Cell]] | None = None
    counted_static: bool = False
    counted_dynamic: bool = False


@dataclass(frozen=True, slots=True)
class EndpointEvent:
    t: int
    kind: str
    robot: int
    cell: Cell | None = None

    def to_json(self) -> list[object]:
        return [self.kind, self.robot, list(self.cell) if self.cell else None]


@dataclass(frozen=True, slots=True)
class RobotState:
    """What the recovery needs to know about one robot at time ``t`` (after allocation)."""

    position: Cell
    mode: RobotMode
    task: int | None
    goal: Cell | None  # the task goal (pickup or drop-off), None when idle


class EndpointRecovery:
    """Parking and retreat goals for :class:`~robocoord.planning.coordinator.AllocationCoordinator`
    (see the module docstring)."""

    def __init__(self, params: EndpointParams | None = None) -> None:
        self.params = params or EndpointParams()
        self.reset(0)

    def reset(self, map_version: int) -> None:
        self.stats: Counter[str] = Counter()
        self.events: list[EndpointEvent] = []
        self.step_events: list[EndpointEvent] = []
        self.retreats: dict[int, Retreat] = {}
        self.parkings: dict[int, Parking] = {}
        self.blockages: dict[int, Blockage] = {}
        self._needs: dict[int, _Need] = {}
        self._restoring: set[int] = set()
        self._recovered: dict[int, tuple[int, Cell]] = {}
        self._excluded: frozenset[int] = frozenset()
        self._cut_cache: tuple[tuple[int, frozenset[Cell]], frozenset[Cell]] | None = None
        self._version = map_version
        self.end: dict[str, int] = dict.fromkeys(END_KEYS, 0)

    # ------------------------------------------------------------------ before the monitor
    def overrides(
        self, t: int, robots: Sequence[RobotState], exclude: set[int] | frozenset[int]
    ) -> dict[int, Override]:
        """Goal overrides for this step. ``exclude``: robots the deadlock resolver overrides
        (the resolver has precedence); their retreat or parking is cancelled."""
        self.step_events = []
        # robots the resolver holds this step: no recovery is started or tracked for them until
        # it lets them go (otherwise a parking would be restarted and cancelled every step)
        self._excluded = frozenset(exclude)
        out: dict[int, Override] = {}
        for i in sorted(self._restoring):
            if i not in exclude and robots[i].goal is not None:
                out[i] = Override(robots[i].goal, trigger="T8")
        self._restoring.clear()
        for i in sorted(self.retreats):
            rt = self.retreats[i]
            if i in exclude or robots[i].goal != rt.task_goal or robots[i].task != rt.task:
                self._end_retreat(t, rt, "retreats_cancelled", restore=False)
                continue
            out[i] = Override(rt.cell, detour=True, hold_stall=True, trigger="T8")
            self.stats["retreat_robot_steps"] += 1
        for i in sorted(self.parkings):
            pk = self.parkings[i]
            if i in exclude or robots[i].mode is not RobotMode.IDLE:
                self._end_parking(t, pk, "parking_cancelled")
                continue
            out[i] = Override(
                pk.cell, detour=True, avoid=pk.avoid, trigger="T8", count_active=False
            )
            self.stats["parking_robot_steps"] += 1
        return out

    def in_recovery(self, robot: int) -> bool:
        return robot in self.retreats or robot in self.parkings

    def reserved(self) -> frozenset[Cell]:
        return frozenset(
            [rt.cell for rt in self.retreats.values()] + [p.cell for p in self.parkings.values()]
        )

    # ------------------------------------------------------------------ after the monitor
    def observe(
        self,
        t: int,
        grid: _Map,
        robots: Sequence[RobotState],
        tasks: Iterable[_TaskLike],
        obstacles: Sequence[tuple[int, Cell, Cell]],
        calls: Sequence[PlanCall],
        busy_external: set[int] | frozenset[int] = frozenset(),
        reserved_external: set[Cell] | frozenset[Cell] = frozenset(),
    ) -> None:
        """Update retreats and parkings, track blockage episodes and start new recoveries on the
        state at ``t``. ``obstacles``: (id, position, velocity); ``busy_external``: robots in a
        deadlock detour; ``reserved_external``: deadlock passing cells."""
        if grid.version != self._version:
            self._version = grid.version
            for need in self._needs.values():
                need.static_stamp = None  # a map change can create a parking cell
        endpoints = active_endpoints(tasks)
        positions = [r.position for r in robots]
        obstacle_at = {pos: (oid, vel) for oid, pos, vel in obstacles}
        failed = {c.robot for c in calls if c.status == "unreachable"}
        busy_external = frozenset(busy_external) | self._excluded
        self._update_retreats(t, grid, robots, obstacle_at, failed)
        self._update_parkings(t, grid, robots, endpoints, failed)
        self._track_blockages(
            t, grid, robots, endpoints, obstacle_at, busy_external, reserved_external
        )
        self._start_parkings(
            t, grid, robots, endpoints, obstacle_at, busy_external, reserved_external
        )
        for i in sorted(self._recovered):
            task, goal = self._recovered[i]
            if positions[i] == goal:
                self.stats["goal_reached_after_recovery"] += 1
                self._emit(t, "goal_reached_after_recovery", i, goal)
                del self._recovered[i]
            elif robots[i].task != task:
                del self._recovered[i]

    # ------------------------------------------------------------------ retreats
    def _occupant(
        self,
        goal: Cell,
        me: int,
        robots: Sequence[RobotState],
        obstacle_at: Mapping[Cell, tuple[int, Cell]],
    ) -> tuple[str, int] | None:
        if goal in obstacle_at:
            return ("obstacle", obstacle_at[goal][0])
        for j, r in enumerate(robots):
            if j != me and r.position == goal and r.mode is RobotMode.IDLE:
                return ("robot", j)
        return None

    def _update_retreats(
        self,
        t: int,
        grid: _Map,
        robots: Sequence[RobotState],
        obstacle_at: Mapping[Cell, tuple[int, Cell]],
        failed: set[int],
    ) -> None:
        for i in sorted(self.retreats):
            rt = self.retreats[i]
            pos = robots[i].position
            blk = self.blockages.get(i)
            occupant = self._occupant(rt.task_goal, i, robots, obstacle_at)
            if blk is None or occupant != blk.occupant:
                self._end_retreat(t, rt, "retreats_succeeded")
                self.blockages.pop(i, None)  # the episode ends: the occupant left the goal
                self._recovered[i] = (rt.task, rt.task_goal)
                continue
            if pos != rt.last_pos:
                rt.last_pos, rt.last_progress = pos, t
            if rt.arrived is None and pos == rt.cell:
                rt.arrived = t
            if rt.arrived is not None and t - rt.arrived >= self.params.hold_steps:
                self._end_retreat(t, rt, "retreats_timed_out")
                blk.wait = 0
            elif (
                not grid.is_free(rt.cell)
                or i in failed
                or (rt.arrived is None and t - rt.last_progress >= self.params.abort_steps)
            ):
                self._end_retreat(t, rt, "retreats_aborted")
                blk.wait = 0

    def _end_retreat(self, t: int, rt: Retreat, kind: str, *, restore: bool = True) -> None:
        self.stats[kind] += 1
        self._emit(t, kind, rt.robot, rt.cell)
        del self.retreats[rt.robot]
        if restore:
            self._restoring.add(rt.robot)

    def _track_blockages(
        self,
        t: int,
        grid: _Map,
        robots: Sequence[RobotState],
        endpoints: frozenset[Cell],
        obstacle_at: Mapping[Cell, tuple[int, Cell]],
        busy_external: set[int] | frozenset[int],
        reserved_external: set[Cell] | frozenset[Cell],
    ) -> None:
        p = self.params
        for i, r in enumerate(robots):
            if i in self.retreats:
                continue  # managed by its retreat (success ends the episode)
            blk = self.blockages.get(i)
            occupant = (
                self._occupant(r.goal, i, robots, obstacle_at)
                if r.goal is not None and r.task is not None
                else None
            )
            if blk is not None and (
                occupant != blk.occupant or r.task != blk.task or r.goal != blk.goal
            ):
                if not blk.persistent:
                    self.stats["blockages_temporary"] += 1
                del self.blockages[i]
                blk = None
            if occupant is None or r.goal is None or r.task is None:
                continue
            if _manhattan(r.position, r.goal) != 1 or i in busy_external:
                if blk is not None:
                    blk.wait = 0
                continue
            if blk is None:
                blk = self.blockages[i] = Blockage(r.task, r.goal, occupant, t)
            blk.wait += 1
            if blk.wait < p.wait_steps:
                continue
            if not blk.persistent:
                blk.persistent = True
                self.stats["blockages_persistent"] += 1
                self.stats[f"blockages_persistent_{occupant[0]}"] += 1
                self._emit(t, "blockages_persistent", i, r.goal)
            if blk.attempts >= p.max_attempts:
                if not blk.exhausted:
                    blk.exhausted = True
                    self.stats["attempts_exhausted"] += 1
                    self._emit(t, "attempts_exhausted", i, r.goal)
                continue
            cell = self._retreat_cell(grid, i, robots, endpoints, obstacle_at, reserved_external)
            if cell is None:
                if not blk.no_cell:
                    blk.no_cell = True
                    self.stats["no_retreat_cell"] += 1
                    self._emit(t, "no_retreat_cell", i, r.goal)
                blk.wait = 0  # search again after another W steps
                continue
            blk.attempts += 1
            blk.wait = 0
            self.retreats[i] = Retreat(i, r.task, r.goal, cell, t, r.position, t)
            self.stats["retreats_started"] += 1
            self._emit(t, "retreats_started", i, cell)

    def _retreat_cell(
        self,
        grid: _Map,
        i: int,
        robots: Sequence[RobotState],
        endpoints: frozenset[Cell],
        obstacle_at: Mapping[Cell, tuple[int, Cell]],
        reserved_external: set[Cell] | frozenset[Cell],
    ) -> Cell | None:
        r = robots[i]
        assert r.goal is not None
        occupant_cell = r.goal
        here = r.position
        blocked = {q.position for j, q in enumerate(robots) if j != i} | set(obstacle_at)
        predicted = {(z[0] + v[0], z[1] + v[1]) for z, (_, v) in obstacle_at.items() if v != (0, 0)}
        reserved = self.reserved() | reserved_external

        def accept(c: Cell) -> bool:
            return (
                _manhattan(c, occupant_cell) >= 2
                and _manhattan(c, here) >= 2
                and c not in endpoints
                and c not in reserved
                and c not in predicted
            )

        return nearest_cell(grid, here, blocked, accept, self.params.retreat_radius)

    # ------------------------------------------------------------------ parking
    def _update_parkings(
        self,
        t: int,
        grid: _Map,
        robots: Sequence[RobotState],
        endpoints: frozenset[Cell],
        failed: set[int],
    ) -> None:
        for i in sorted(self.parkings):
            pk = self.parkings[i]
            pos = robots[i].position
            if pos == pk.cell:
                self._end_parking(t, pk, "parking_completed")
                self._needs.pop(i, None)
                continue
            if pos != pk.last_pos:
                pk.last_pos, pk.last_progress = pos, t
            if (
                not grid.is_free(pk.cell)
                or pk.cell in endpoints
                or i in failed
                or t - pk.last_progress >= self.params.abort_steps
            ):
                self._end_parking(t, pk, "parking_aborted")
                need = self._needs.get(i)
                if need is not None:
                    need.retry_at = t + self.params.abort_steps

    def _end_parking(self, t: int, pk: Parking, kind: str) -> None:
        self.stats[kind] += 1
        self._emit(t, kind, pk.robot, pk.cell)
        del self.parkings[pk.robot]

    def _start_parkings(
        self,
        t: int,
        grid: _Map,
        robots: Sequence[RobotState],
        endpoints: frozenset[Cell],
        obstacle_at: Mapping[Cell, tuple[int, Cell]],
        busy_external: set[int] | frozenset[int],
        reserved_external: set[Cell] | frozenset[Cell],
    ) -> None:
        for i, r in enumerate(robots):
            if r.mode is not RobotMode.IDLE or i in self.parkings or i in busy_external:
                if r.mode is not RobotMode.IDLE:
                    self._needs.pop(i, None)
                continue
            need = self._needs.get(i)
            if r.position not in endpoints:
                self._needs.pop(i, None)
                continue
            if need is None or need.cell != r.position:
                need = self._needs[i] = _Need(r.position)
                self.stats["parking_needed"] += 1
                self._emit(t, "parking_needed", i, r.position)
            if t < need.retry_at:
                continue
            stamp = (grid.version, endpoints)
            if need.static_stamp == stamp:
                continue  # no cell exists and nothing static has changed
            kind, cell = self._parking_cell(
                grid, i, robots, endpoints, obstacle_at, reserved_external
            )
            if cell is not None:
                # robots next to the endpoint (typically the ones waiting for it): not on the
                # way out, so that leaving does not run head-on into them
                near = frozenset(
                    q.position
                    for j, q in enumerate(robots)
                    if j != i and _manhattan(q.position, r.position) <= 2
                )
                self.parkings[i] = Parking(i, cell, near, t, r.position, t)
                self.stats["parking_started"] += 1
                self._emit(t, "parking_started", i, cell)
            elif kind == "static":
                need.static_stamp = stamp
                if not need.counted_static:
                    need.counted_static = True
                    self.stats["no_parking_cell"] += 1
                    self._emit(t, "no_parking_cell", i, r.position)
            elif not need.counted_dynamic:
                need.counted_dynamic = True
                self.stats["parking_blocked_dynamic"] += 1
                self._emit(t, "parking_blocked_dynamic", i, r.position)

    def _cut_vertices(self, grid: _Map, removed: frozenset[Cell]) -> frozenset[Cell]:
        key = (grid.version, removed)
        if self._cut_cache is not None and self._cut_cache[0] == key:
            return self._cut_cache[1]
        cut = articulation_points(grid, removed)
        self._cut_cache = (key, cut)
        return cut

    def _parking_cell(
        self,
        grid: _Map,
        i: int,
        robots: Sequence[RobotState],
        endpoints: frozenset[Cell],
        obstacle_at: Mapping[Cell, tuple[int, Cell]],
        reserved_external: set[Cell] | frozenset[Cell],
    ) -> tuple[str, Cell | None]:
        """``("ok", cell)``, ``("dynamic", None)`` or ``("static", None)`` (module docstring)."""
        here = robots[i].position
        stationary = frozenset(
            q.position
            for j, q in enumerate(robots)
            if j != i and q.mode is RobotMode.IDLE and j not in self.parkings
        )
        reserved = self.reserved() | reserved_external
        removed = stationary | reserved
        cut = self._cut_vertices(grid, removed)

        def safe(c: Cell) -> bool:
            return c not in endpoints and c not in removed and c not in cut

        occupied = {q.position for j, q in enumerate(robots) if j != i} | set(obstacle_at)
        cell = nearest_cell(grid, here, occupied, lambda c: safe(c) and c not in occupied)
        if cell is not None:
            return "ok", cell
        if nearest_cell(grid, here, frozenset(), safe) is not None:
            return "dynamic", None
        return "static", None

    # ------------------------------------------------------------------ end of episode
    def finish(
        self,
        robots: Sequence[RobotState],
        tasks: Iterable[_TaskLike],
        obstacles: Sequence[tuple[int, Cell, Cell]],
    ) -> None:
        """End-of-episode state: unresolved blockages and recoveries still running."""
        endpoints = active_endpoints(tasks)
        obstacle_at = {pos: (oid, vel) for oid, pos, vel in obstacles}
        # the episode can end on the very step a recovered robot reaches its goal (termination
        # when everything is delivered), before any further observe()
        for i in sorted(self._recovered):
            if robots[i].position == self._recovered[i][1]:
                self.stats["goal_reached_after_recovery"] += 1
        self._recovered.clear()
        end = dict.fromkeys(END_KEYS, 0)
        for i, blk in self.blockages.items():
            r = robots[i]
            if not blk.persistent or r.goal != blk.goal:
                continue
            if self._occupant(blk.goal, i, robots, obstacle_at) == blk.occupant:
                key = "obstacle" if blk.occupant[0] == "obstacle" else "idle_robot"
                end[f"unresolved_{key}_at_end"] += 1
        end["idle_on_endpoint_at_end"] = sum(
            1 for r in robots if r.mode is RobotMode.IDLE and r.position in endpoints
        )
        end["retreats_active_at_end"] = len(self.retreats)
        end["parkings_active_at_end"] = len(self.parkings)
        self.end = end

    # ------------------------------------------------------------------ reporting
    def robot_report(self, robot: int) -> dict[str, Any]:
        """Recovery state of one robot (for the E5 diagnostics)."""
        blk = self.blockages.get(robot)
        need = self._needs.get(robot)
        return {
            "blockage_persistent": bool(blk and blk.persistent),
            "retreat_attempts": blk.attempts if blk else 0,
            "attempts_exhausted": bool(blk and blk.exhausted),
            "no_retreat_cell": bool(blk and blk.no_cell),
            "retreating": robot in self.retreats,
            "parking": robot in self.parkings,
            "no_parking_cell": bool(need and need.counted_static and need.static_stamp),
            "parking_blocked_dynamic": bool(need and need.counted_dynamic),
        }

    def _emit(self, t: int, kind: str, robot: int, cell: Cell | None = None) -> None:
        ev = EndpointEvent(t, kind, robot, cell)
        self.events.append(ev)
        self.step_events.append(ev)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: self.stats.get(k, 0) for k in STAT_KEYS}
        out.update(self.end)
        return out


__all__ = [
    "END_KEYS",
    "STAT_KEYS",
    "Blockage",
    "EndpointEvent",
    "EndpointParams",
    "EndpointRecovery",
    "Parking",
    "Retreat",
    "RobotState",
    "active_endpoints",
    "articulation_points",
    "nearest_cell",
]
