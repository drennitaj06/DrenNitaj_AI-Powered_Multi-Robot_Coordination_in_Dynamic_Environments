"""Path monitor: rejoin, replanning triggers T1–T8, stall block list, attribution (model §5.4;
algorithms §5.2).

The monitor is controller-agnostic: it only sees positions, goals and obstacles at the start of
each step and decides whether a robot needs a new reference path. At most one planner call per
robot per step, attributed to the first trigger found. The checks run in this order: goal changes
(T1; T7 and T8 when the coordinator supplies a temporary goal), then T5 (no valid plan), T3
(off-path), T2 (environment), T4 (stall) and T6 (degraded plan). T7 and T8 are produced only with
the deadlock resolver and endpoint recovery, respectively. Every call goes through the shared
procedure and is logged.

Implementation choices (documented in ``04`` §5.2 notes):

* **T2 and the goal cell**: "excluding the goal" applies to the dynamic set (the goal is never in
  ``B_i``). The goal is still checked against ``S(t)`` at every step (O(1)), otherwise a map
  event on the goal would never be noticed and the ``unreachable`` policy of model §5.3 step 4
  could not apply.
* **T2 on degraded plans** checks only ``S(t)``: a degraded path crosses dynamically blocked
  cells by construction, so a dynamic check would re-fire every step. Dynamic re-checks of
  degraded plans are the job of T6 (cooldown ``T_cd``).
* **T4 never targets the goal**: a stall in front of an occupied goal cell is waiting, which is
  the occupied-goal policy of model §5.3; the counter is reset and no replan is made.
* **T5** fires for a robot without a valid plan once ``T_cd`` steps have passed since the failed
  call.
* **Repair before replanning** (Issue 6A Phase 1, proposal P1; off by default,
  ``repair_radius = 0``): on a T3 deviation the monitor first searches, breadth-first over
  statically free cells outside ``B_i(t)`` and the proximity set ``Q_i(t)``, a connector of at
  most ``repair_radius`` moves from the robot to a cell of its *remaining* reference path (the
  shortest connector; ties → the cell furthest along). The repaired path is the connector plus
  the rest of the old path; it is kept only if it passes the T2 check (the next ``H_v`` cells
  free of ``S(t) ∪ B_i(t)``). Otherwise, or on a degraded plan, the T3 planner call is made as
  before. A repaired deviation is still a deviation (``deviations``), but not a planner call;
  repairs are counted separately. With ``repair_radius = 0`` the monitor is unchanged.

Time: O(H_r + H_v) per robot per step, plus planner calls.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell, action_between
from robocoord.planning.base import Planner, PlanQuery, PlanResult, PlanStats
from robocoord.planning.constraints import ConstraintParams, build_constraints
from robocoord.planning.procedure import plan_with_procedure

TRIGGERS: dict[str, str] = {
    "T1": "allocation",  # goal changed
    "T2": "environment",  # a path cell ahead entered S(t) ∪ B_i(t)
    "T3": "controller",  # off-path after the robot's own action
    "T4": "interaction",  # stall in front of an occupied cell
    "T5": "retry",  # previous call failed, cooldown elapsed
    "T6": "retry",  # degraded plan, cooldown elapsed
    "T7": "deadlock",  # deadlock detour: goal set to / restored from a passing cell (Issue #4)
    "T8": "endpoint",  # endpoint recovery: retreat / parking goal set or task goal restored
}
BASE_TRIGGERS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T6")


@dataclass(frozen=True, slots=True)
class MonitorParams:
    """``H_v``, ``H_r``, ``T_cd``, ``T_stall`` (model §11 defaults 5, 3, 3, 4)."""

    validity_horizon: int = 5
    rejoin_window: int = 3
    cooldown: int = 3
    stall_threshold: int = 4
    repair_radius: int = 0  # Issue 6A Phase 1 (P1): 0 = off (the original monitor)

    def __post_init__(self) -> None:
        for name in ("validity_horizon", "rejoin_window", "cooldown", "stall_threshold"):
            if getattr(self, name) < 1:
                raise ConfigError(f"monitor.{name}", "must be >= 1")
        if self.repair_radius < 0:
            raise ConfigError("monitor.repair_radius", "must be >= 0")


@dataclass(frozen=True, slots=True)
class PlanCall:
    """One logged planner call."""

    t: int
    robot: int
    trigger: str
    status: str
    degraded: bool
    path_length: int  # moves on the returned path (−1 on failure)
    stats: PlanStats

    @property
    def attribution(self) -> str:
        return TRIGGERS[self.trigger]


@dataclass(slots=True)
class LegRecord:
    """A completed leg (goal reached): executed steps/moves vs the first reference path."""

    robot: int
    start_t: int
    end_t: int
    moves: int
    reference_length: int | None
    static_distance: int | None


@dataclass
class RobotPlanState:
    goal: Cell | None = None
    path: tuple[Cell, ...] | None = None
    index: int = 0
    degraded: bool = False
    planned_at: int = -1
    retry_at: int = 0
    stall: int = 0
    blocks: dict[Cell, int] = field(default_factory=dict)  # stall block list: cell → expiry
    last_pos: Cell | None = None
    leg_start: int = 0
    leg_moves: int = 0
    leg_ref: int | None = None
    leg_static: int | None = None
    detour: bool = False  # the current goal is a deadlock-detour passing cell
    task_goal: Cell | None = None  # the goal to restore after the detour

    @property
    def valid(self) -> bool:
        return self.path is not None


@dataclass(frozen=True, slots=True)
class RobotInput:
    """Per-step monitor input. The last four fields are used only by the deadlock resolver
    (Issue #4 stage 5B) and endpoint recovery; their defaults give exactly the Issue #3
    behaviour.

    * ``detour`` — ``goal`` is a temporary cell (a passing, retreat or parking cell): the goal
      change is attributed to ``trigger`` and the current leg is neither closed nor restarted;
      restoring the saved task goal is attributed to ``trigger`` too.
    * ``hold_stall`` — suspend the T4 stall counter (a detour is resolving this robot's stall).
    * ``avoid`` — extra dynamically blocked cells for this step's checks and plans.
    * ``trigger`` — the attribution of such a goal change: ``T7`` (deadlock resolver) or ``T8``
      (endpoint recovery).
    * ``count_active`` — False for an *idle* robot driving to a parking cell: its steps are not
      counted in ``active_robot_steps`` and its T3 calls not in ``deviations``, so the deviation
      rate keeps its definition (T3 per 100 non-idle robot-steps, model §9.3).
    """

    id: int
    position: Cell
    goal: Cell | None
    detour: bool = False
    hold_stall: bool = False
    avoid: frozenset[Cell] = frozenset()
    trigger: str = "T7"
    count_active: bool = True


def _extra(st: RobotPlanState, r: RobotInput) -> Iterable[Cell]:
    """Stall block list plus the resolver's per-step ``avoid`` cells (unchanged if empty)."""
    return st.blocks if not r.avoid else set(st.blocks) | r.avoid


class PathMonitor:
    """Per-robot reference paths kept valid by event-triggered replanning."""

    def __init__(
        self,
        planner: Planner,
        grid: OccupancyGrid,
        n_robots: int,
        *,
        params: MonitorParams | None = None,
        constraint_params: ConstraintParams | None = None,
        seed: int = 0,
        static_reachable: Callable[[Cell, Cell], bool] | None = None,
        static_distance: Callable[[Cell, Cell], int | None] | None = None,
    ) -> None:
        self.planner = planner
        self.grid = grid
        self.params = params or MonitorParams()
        self.constraint_params = constraint_params or ConstraintParams()
        self.seed = seed
        self._static_reachable = static_reachable
        self._static_distance = static_distance
        self.states = [RobotPlanState() for _ in range(n_robots)]
        self.calls: list[PlanCall] = []
        self.legs: list[LegRecord] = []
        self.active_robot_steps = 0
        self.deviations = 0  # T3 calls of counted (non-idle) robots
        self.repairs = 0  # T3 deviations repaired without a planner call (P1)
        self.repair_failures = 0  # T3 deviations whose repair failed (a planner call followed)
        self.last_results: dict[int, PlanResult] = {}

    # ------------------------------------------------------------------ per step
    def update(
        self, t: int, robots: Sequence[RobotInput], obstacles: Mapping[Cell, Cell]
    ) -> list[PlanCall]:
        """Bookkeeping for the state at time ``t`` and the triggered planner calls."""
        occupied = {r.position for r in robots} | set(obstacles)
        made: list[PlanCall] = []
        for r in robots:
            before = self.active_robot_steps
            trigger = self._bookkeep(t, r, obstacles, occupied)
            if not r.count_active:
                self.active_robot_steps = before
            if trigger is not None:
                if trigger == "T3" and r.count_active:
                    self.deviations += 1
                if trigger == "T3" and self.params.repair_radius > 0:
                    if self._repair(r, obstacles):
                        self.repairs += 1
                        continue
                    self.repair_failures += 1
                made.append(self._plan(t, r, trigger, obstacles))
        self.calls.extend(made)
        return made

    def _bookkeep(
        self, t: int, r: RobotInput, obstacles: Mapping[Cell, Cell], occupied: set[Cell]
    ) -> str | None:
        st = self.states[r.id]
        pos, goal = r.position, r.goal
        if st.last_pos is not None and pos != st.last_pos:
            st.leg_moves += 1
        st.last_pos = pos
        st.blocks = {c: e for c, e in st.blocks.items() if e > t}
        if (
            goal != st.goal
            and goal is not None
            and (r.detour or (st.detour and goal == st.task_goal))
        ):
            # T7 / T8: into a temporary cell, between temporary cells, or back to the saved task
            # goal. The leg (trip to the task goal) continues: no close, no restart.
            if not st.detour:
                st.task_goal = st.goal
            st.goal, st.path, st.index, st.degraded, st.stall = goal, None, 0, False, 0
            st.blocks.clear()
            st.detour = r.detour
            if not r.detour:
                st.task_goal = None
            self.active_robot_steps += 1
            return r.trigger
        if goal != st.goal:
            if st.goal is not None and pos == st.goal and not st.detour:
                self._close_leg(r.id, t)
            st.detour, st.task_goal = False, None
            st.goal, st.path, st.index, st.degraded, st.stall = goal, None, 0, False, 0
            st.blocks.clear()
            if goal is None:
                return None
            st.leg_start, st.leg_moves, st.leg_ref = t, 0, None
            st.leg_static = self._static_distance(pos, goal) if self._static_distance else None
            self.active_robot_steps += 1
            return "T1"
        if goal is None:
            return None
        self.active_robot_steps += 1
        p = self.params
        if st.path is None:
            return "T5" if t >= st.retry_at else None
        path = st.path
        # ---- advance / rejoin within {ℓ, …, ℓ + H_r}
        window_end = min(st.index + p.rejoin_window, len(path) - 1)
        found = -1
        for j in range(window_end, st.index - 1, -1):
            if path[j] == pos:
                found = j
                break
        if found < 0:
            return "T3"
        if found > st.index:
            st.index = found
            st.stall = 0
        if st.index == len(path) - 1:
            return None  # at the goal
        # ---- T2: next H_v cells (excluding the goal) entered S(t) ∪ B_i(t)
        ahead = path[st.index + 1 : st.index + 1 + p.validity_horizon]
        dynamic: frozenset[Cell] = frozenset()
        if not st.degraded:
            dynamic = build_constraints(
                self.grid, pos, goal, obstacles, self.constraint_params, _extra(st, r)
            ).dynamic_blocked
        if self.grid.is_blocked(goal):  # a map event closed the goal cell (model §5.3 step 4)
            return "T2"
        for c in ahead:
            if self.grid.is_blocked(c) or (c != goal and c in dynamic):
                return "T2"
        # ---- T4: stall in front of an occupied cell
        nxt = path[st.index + 1]
        if r.hold_stall:
            st.stall = 0  # a deadlock detour is resolving this stall (T7), not T4
        elif nxt in occupied and nxt != goal:
            st.stall += 1
            if st.stall >= p.stall_threshold:
                st.stall = 0
                st.blocks[nxt] = t + p.stall_threshold
                return "T4"
        else:
            st.stall = 0
        # ---- T6: degraded plan, retry after the cooldown
        if st.degraded and t >= st.planned_at + p.cooldown:
            return "T6"
        return None

    def _plan(
        self, t: int, r: RobotInput, trigger: str, obstacles: Mapping[Cell, Cell]
    ) -> PlanCall:
        st = self.states[r.id]
        assert st.goal is not None
        constraints = build_constraints(
            self.grid, r.position, st.goal, obstacles, self.constraint_params, _extra(st, r)
        )
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(t, r.id)))
        res = plan_with_procedure(
            self.planner,
            PlanQuery(r.position, st.goal, constraints, rng),
            static_reachable=self._static_reachable,
        )
        self.last_results[r.id] = res
        st.planned_at = t
        st.stall = 0
        if res.ok:
            assert res.path is not None
            st.path, st.index, st.degraded = res.path, 0, res.degraded
            if st.leg_ref is None:
                st.leg_ref = len(res.path) - 1
        else:
            st.path, st.index, st.degraded = None, 0, False
            st.retry_at = t + self.params.cooldown
        stats = res.stats
        if stats.best_cost_trace is not None or stats.iteration_trace is not None:
            stats = replace(stats, best_cost_trace=None, iteration_trace=None)
        return PlanCall(
            t,
            r.id,
            trigger,
            res.status,
            res.degraded,
            len(res.path) - 1 if res.path is not None else -1,
            stats,
        )

    def _repair(self, r: RobotInput, obstacles: Mapping[Cell, Cell]) -> bool:
        """P1: rejoin the remaining reference path by a connector of at most ``repair_radius``
        moves; ``True`` (and the path replaced) on success. Deterministic (canonical neighbour
        order; the shortest connector, then the furthest path cell)."""
        st = self.states[r.id]
        if st.path is None or st.degraded or st.goal is None:
            return False
        pos, goal, path = r.position, st.goal, st.path
        cons = build_constraints(
            self.grid, pos, goal, obstacles, self.constraint_params, _extra(st, r)
        )
        dynamic = cons.dynamic_blocked
        avoid = dynamic | cons.proximity
        targets: dict[Cell, int] = {}
        for j in range(st.index, len(path)):
            targets[path[j]] = j  # the last occurrence: furthest along
        prev: dict[Cell, Cell | None] = {pos: None}
        frontier = [pos]
        depth = 0
        while True:
            hits = [c for c in frontier if c in targets]
            if hits:
                break
            if depth >= self.params.repair_radius:
                return False
            nxt: list[Cell] = []
            for c in frontier:
                for n in self.grid.neighbors(c):
                    if n in prev or (n in avoid and n != goal):
                        continue
                    prev[n] = c
                    nxt.append(n)
            if not nxt:
                return False
            frontier = nxt
            depth += 1
        best = max(hits, key=lambda c: targets[c])
        connector: list[Cell] = []
        node: Cell | None = best
        while node is not None:
            connector.append(node)
            node = prev[node]
        connector.reverse()
        repaired = (*connector, *path[targets[best] + 1 :])
        if self.grid.is_blocked(goal):
            return False
        for c in repaired[1 : 1 + self.params.validity_horizon]:
            if self.grid.is_blocked(c) or (c != goal and c in dynamic):
                return False
        st.path, st.index, st.stall = repaired, 0, 0
        return True

    def _close_leg(self, robot: int, t: int) -> None:
        st = self.states[robot]
        self.legs.append(LegRecord(robot, st.leg_start, t, st.leg_moves, st.leg_ref, st.leg_static))

    def finish(self, t: int, positions: Sequence[Cell]) -> None:
        """End of episode: count the last moves and close legs whose goal was reached."""
        for i, pos in enumerate(positions):
            st = self.states[i]
            if st.last_pos is not None and pos != st.last_pos:
                st.leg_moves += 1
            st.last_pos = pos
            if st.goal is not None and pos == st.goal and not st.detour:
                self._close_leg(i, t)
                st.goal, st.path = None, None

    def release(self, robot: int) -> None:
        """Forget the robot's goal and plan (its task was released); no leg is recorded."""
        st = self.states[robot]
        st.goal, st.path, st.index, st.degraded, st.stall = None, None, 0, False, 0
        st.blocks.clear()
        st.detour, st.task_goal = False, None

    def next_cell(self, robot: int, position: Cell) -> Cell | None:
        """``c_{ℓ+1}`` if the robot is on a valid plan and not at its goal, else ``None``."""
        st = self.states[robot]
        if st.path is None or st.index >= len(st.path) - 1 or st.path[st.index] != position:
            return None
        return st.path[st.index + 1]

    # ------------------------------------------------------------------ queries
    def suggested(self, robot: int, position: Cell) -> Action:
        """``â^plan``: towards ``c_{ℓ+1}``; ``stay`` without a valid plan or at the goal."""
        st = self.states[robot]
        if st.path is None or st.index >= len(st.path) - 1 or st.path[st.index] != position:
            return Action.STAY
        return action_between(position, st.path[st.index + 1])

    def has_valid_plan(self, robot: int) -> bool:
        return self.states[robot].valid

    def reference_path(self, robot: int) -> tuple[Cell, ...] | None:
        """The remaining reference path ``c_ℓ, …, c_L`` (``None`` without a valid plan)."""
        st = self.states[robot]
        return None if st.path is None else st.path[st.index :]
