"""Path monitor: rejoin, replanning triggers T1–T6, stall block list, attribution (model §5.4;
algorithms §5.2).

The monitor is controller-agnostic: it only sees positions, goals and obstacles at the start of
each step and decides whether a robot needs a new reference path. At most one planner call per
robot per step; when several triggers fire, the call is attributed to the first in the order
T1, T2, T3, T4, T5, T6. Every call goes through the shared procedure and is logged.

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

Time: O(H_r + H_v) per robot per step, plus planner calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
}


@dataclass(frozen=True, slots=True)
class MonitorParams:
    """``H_v``, ``H_r``, ``T_cd``, ``T_stall`` (model §11 defaults 5, 3, 3, 4)."""

    validity_horizon: int = 5
    rejoin_window: int = 3
    cooldown: int = 3
    stall_threshold: int = 4

    def __post_init__(self) -> None:
        for name in ("validity_horizon", "rejoin_window", "cooldown", "stall_threshold"):
            if getattr(self, name) < 1:
                raise ConfigError(f"monitor.{name}", "must be >= 1")


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

    @property
    def valid(self) -> bool:
        return self.path is not None


@dataclass(frozen=True, slots=True)
class RobotInput:
    id: int
    position: Cell
    goal: Cell | None


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
        self.last_results: dict[int, PlanResult] = {}

    # ------------------------------------------------------------------ per step
    def update(
        self, t: int, robots: Sequence[RobotInput], obstacles: Mapping[Cell, Cell]
    ) -> list[PlanCall]:
        """Bookkeeping for the state at time ``t`` and the triggered planner calls."""
        occupied = {r.position for r in robots} | set(obstacles)
        made: list[PlanCall] = []
        for r in robots:
            trigger = self._bookkeep(t, r, obstacles, occupied)
            if trigger is not None:
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
        if goal != st.goal:
            if st.goal is not None and pos == st.goal:
                self._close_leg(r.id, t)
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
                self.grid, pos, goal, obstacles, self.constraint_params, st.blocks
            ).dynamic_blocked
        if self.grid.is_blocked(goal):  # a map event closed the goal cell (model §5.3 step 4)
            return "T2"
        for c in ahead:
            if self.grid.is_blocked(c) or (c != goal and c in dynamic):
                return "T2"
        # ---- T4: stall in front of an occupied cell
        nxt = path[st.index + 1]
        if nxt in occupied and nxt != goal:
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
            self.grid, r.position, st.goal, obstacles, self.constraint_params, st.blocks
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
            if st.goal is not None and pos == st.goal:
                self._close_leg(i, t)
                st.goal, st.path = None, None

    def release(self, robot: int) -> None:
        """Forget the robot's goal and plan (its task was released); no leg is recorded."""
        st = self.states[robot]
        st.goal, st.path, st.index, st.degraded, st.stall = None, None, 0, False, 0
        st.blocks.clear()

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
