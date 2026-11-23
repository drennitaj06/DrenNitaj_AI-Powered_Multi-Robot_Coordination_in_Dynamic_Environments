"""Execution Supervisor (Issue 6A Phase 3; design ``docs/research/2026-11-21-issue-6a-phase3-
execution-supervisor-design.md``).

An optional engine stage between the controller's proposal and the predictive shield::

    policy proposal a_pol → SUPERVISOR → shield (optional) → arbiter → motion

It is **off by default** (the engine is then byte-identical to the code without it), it never
participates in training, and it never touches the policy, its network outputs, the reward, task
states or goals. For every robot it returns exactly one of the policy's action, ``STAY`` or the
planner's suggested action (masked-valid), so the arbiter — unchanged — remains the final
authority and Lemma 3.1 (no physical collision) still holds.

Mechanisms (selected by ``mode``: ``"off"``, ``"full"`` = ``"R+S+O"``, or any ``+``-joined subset):

* **S — stall breaker.** A robot with a task whose best static goal distance ``D̄`` (view extra
  ``goal_distance``) has not improved for ``stall_window`` *counted* steps starts a recovery: for
  at most ``hold`` steps it executes the **filtered suggestion** (the planner's suggested action,
  or ``STAY`` if it is ``STAY``, masked, into a robot / obstacle cell, or into a predicted obstacle
  cell — the baseline controller's rules 1–4). Excused (neither counted nor resetting): the
  suggestion is ``STAY`` or filtered to ``STAY``, the robot lost an R contest in the previous step,
  it is a member of an active resolver detour, or it is in recovery. A recovery ends on
  ``progress_cells`` cells of progress since its start, after ``hold`` steps, after ``stay_limit``
  consecutive ``STAY`` actions, on a leg change (task / mode / goal) or when the robot joins a
  detour.
* **O — oscillation detector.** Over the last ``osc_window`` steps of the leg: at most
  ``osc_cells`` distinct cells, at least ``osc_moves`` moves and no progress → an S recovery starts
  at once. *Recovery-cycle guard* (replaces the design's anti-backtrack rule, which in testing
  blocked legitimate path moves): a recovery of any kind ends (``cycle``) as soon as the robot
  moves onto a cell it already occupied during that recovery, so a recovery can never itself
  oscillate (e.g. under alternating RRT* replans).
* **R — right-of-way** on the joint proposal (after S/O): a vertex contest for a **free** cell
  keeps the mover with the highest key ``(losses ≥ L_max, π̃, losses, −id)`` and turns the others
  into ``STAY``. Moves into a cell whose occupant stays and head-on **swaps** are left to the
  arbiter, which cancels them (same motion; their conflicts stay visible in its record); they are
  only counted. R cannot resolve swaps — a persisting swap is a two-robot wait-for cycle, the
  existing resolver's case. ``π̃`` is the baseline's effective
  priority (:func:`robocoord.allocation.benefits.robot_priorities`); ``losses`` counts consecutive
  R contest losses (the fairness override bounds them).
* **D — deadlock recovery** is the existing resolver (``planning/deadlock.py``) enabled by
  configuration; the supervisor only reads its detour membership (passed in by the engine).
  Detour members are excused from S and O (their task-goal distance rises during a detour). With
  S on, a detour member whose policy declines a movable filtered suggestion for ``stall_window``
  consecutive steps gets a recovery of kind ``D`` (design decision D2b): the filtered suggestion
  (which now leads to the passing cell) for at most ``hold`` steps, ended when the detour ends.

Deterministic: no randomness; robots and cells are processed in fixed orders. Time per step
O(N · (osc_window + min((2 r_s + 1)², K))) plus O(N²) in the worst case of the R fixed point
(each iteration turns at least one move into ``STAY``).
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from robocoord.allocation.benefits import robot_priorities
from robocoord.core.errors import ConfigError
from robocoord.core.types import Action, Cell, add
from robocoord.planning.constraints import sense
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode

MECHANISMS: tuple[str, ...] = ("R", "S", "O")
MODES_HELP = "'off', 'full' or a '+'-joined subset of R, S, O (e.g. 'R+S')"
#: Reasons of a changed action (events) and of recovery starts / ends (statistics).
CHANGE_REASONS: tuple[str, ...] = ("recovery", "R_loss")
END_REASONS: tuple[str, ...] = (
    "progress",
    "hold",
    "stay_limit",
    "cycle",
    "leg",
    "detour",
    "detour_end",
)
#: Recovery kinds: S (stall), O (oscillation), D (a detour member ignoring its detour, D2b).
RECOVERY_KINDS: tuple[str, ...] = ("S", "O", "D")


def parse_mode(mode: str) -> frozenset[str]:
    """The mechanisms of a supervisor mode string."""
    if mode == "off":
        return frozenset()
    if mode == "full":
        return frozenset(MECHANISMS)
    parts = mode.split("+")
    if not parts or any(p not in MECHANISMS for p in parts) or len(set(parts)) != len(parts):
        raise ConfigError("supervisor.mode", f"unknown mode {mode!r}; expected {MODES_HELP}")
    return frozenset(parts)


@dataclass(frozen=True, slots=True)
class SupervisorParams:
    """Parameters. The defaults are the design values (§5); evaluation uses the Stage 3a
    calibrated :data:`FROZEN_PARAMS`."""

    stall_window: int = 12  # W
    hold: int = 8  # H
    progress_cells: int = 2  # Δ_prog
    stay_limit: int = 3  # H_stay
    osc_window: int = 8  # K_osc
    osc_moves: int = 4  # m_osc
    osc_cells: int = 2
    fairness_losses: int = 3  # L_max
    alpha: float = 0.01  # aging rate of π̃ (must equal the allocation's)
    sensing_radius: int = 3  # r_s of the predicted-obstacle filter (as the baseline)
    #: Optional escalation (off by default; needs the student's approval before use): a robot
    #: that stalls again in a leg in which it already had a recovery re-triggers after
    #: ``repeat_window`` counted steps, and that recovery is not ended by progress (only by
    #: hold, cycle, stay limit, leg or detour). Worst-case override share per leg becomes
    #: hold / (hold + repeat_window) instead of hold / (hold + stall_window).
    escalate: bool = False
    repeat_window: int = 4

    def __post_init__(self) -> None:
        for name in (
            "stall_window",
            "hold",
            "progress_cells",
            "stay_limit",
            "osc_window",
            "osc_moves",
            "osc_cells",
            "fairness_losses",
            "repeat_window",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"supervisor.{name}", "must be >= 1")
        if self.osc_moves > self.osc_window:
            raise ConfigError("supervisor.osc_moves", "must be <= osc_window")
        if self.alpha < 0 or self.sensing_radius < 0:
            raise ConfigError("supervisor.alpha/sensing_radius", "must be >= 0")


#: Stage 3a frozen configuration (2026-11-21; protocol
#: ``docs/research/2026-11-21-issue-6a-stage3a-calibration-freeze.md``), calibrated on the stored
#: block-B greedy episodes only. The class defaults above stay the design values (they drive the
#: scripted software tests); every evaluation run uses these, selected by planner. Changing them
#: after the freeze invalidates the Phase 3 pre-registration.
FROZEN_PARAMS: Mapping[str, SupervisorParams] = MappingProxyType(
    {
        "astar": SupervisorParams(stall_window=24, osc_window=14, osc_cells=3, osc_moves=7),
        "rrtstar": SupervisorParams(stall_window=32, osc_window=8, osc_cells=3, osc_moves=4),
    }
)


def frozen_params(planner: str) -> SupervisorParams:
    """The Stage 3a frozen parameters for ``planner`` (``astar`` or ``rrtstar``)."""
    try:
        return FROZEN_PARAMS[planner]
    except KeyError:
        raise ConfigError("supervisor.planner", f"no frozen parameters for {planner!r}") from None


@dataclass(frozen=True, slots=True)
class SupervisorEvent:
    """Robot ``robot``'s proposed ``policy`` action was replaced by ``executed`` for ``reason``."""

    t: int
    robot: int
    reason: str
    policy: Action
    executed: Action

    def to_json(self) -> list[object]:
        return [self.robot, self.reason, self.policy.symbol, self.executed.symbol]


# --------------------------------------------------------------------------- pure helpers


def filtered_suggestion(
    view: WorldView,
    robot: int,
    robot_cells: Iterable[Cell],
    obstacles: Mapping[Cell, Cell],
    sensing_radius: int,
) -> tuple[Action, str]:
    """The baseline controller's rules 1–4 applied to the planner's suggestion: the action and
    the reason (``move``, ``no_plan``, ``masked``, ``robot_ahead``, ``obstacle_ahead``,
    ``predicted_obstacle``)."""
    if view.suggested is None:
        return Action.STAY, "no_plan"
    a = view.suggested[robot]
    if a is Action.STAY:
        return Action.STAY, "no_plan"
    if not view.masks[robot][a]:
        return Action.STAY, "masked"
    pos = view.robots[robot].position
    c = add(pos, a.delta)
    if c in set(robot_cells):
        return Action.STAY, "robot_ahead"
    if c in obstacles:
        return Action.STAY, "obstacle_ahead"
    if c in sense(view.map, pos, obstacles, sensing_radius).predicted:
        return Action.STAY, "predicted_obstacle"
    return a, "move"


def is_oscillating(history: Sequence[Cell], window: int, max_cells: int, min_moves: int) -> bool:
    """``True`` iff the last ``window`` steps (``window + 1`` positions) visit at most
    ``max_cells`` distinct cells with at least ``min_moves`` moves."""
    if len(history) < window + 1:
        return False
    h = list(history)[-(window + 1) :]
    moves = sum(1 for a, b in pairwise(h) if a != b)
    return len(set(h)) <= max_cells and moves >= min_moves


def right_of_way(
    positions: Sequence[Cell], actions: Sequence[Action], key: Mapping[int, tuple[Any, ...]]
) -> tuple[list[Action], dict[int, str], dict[str, int]]:
    """R: every vertex contest for a **free** cell (at least two movers target it and no robot
    stays on it) keeps the mover with the highest key; the other movers become ``STAY``
    (``R_loss``). Returns the new actions, the changed robots' reasons and the counts of the
    motion-neutral conflicts R deliberately leaves to the arbiter (``occupied``: a move into a
    cell whose occupant stays; ``swap``: a head-on exchange). Changing those would not change
    the motion — the arbiter cancels them anyway — but would hide the policy's conflicts from the
    arbiter's record. One pass suffices: turning a mover into ``STAY`` can never create a new
    free-cell contest (the set of movers only shrinks, and a cell with a staying occupant is
    never free)."""
    out = list(actions)
    n = len(out)
    targets = [add(positions[i], out[i].delta) for i in range(n)]
    occ = {p: i for i, p in enumerate(positions)}
    by_target: dict[Cell, list[int]] = defaultdict(list)
    for i in range(n):
        by_target[targets[i]].append(i)
    reason: dict[int, str] = {}
    left = {"occupied": 0, "swap": 0}
    for c in sorted(by_target):
        ids = by_target[c]
        movers = [i for i in ids if out[i] is not Action.STAY]
        if len(movers) < len(ids):  # the occupant stays: left to the arbiter
            left["occupied"] += len(movers)
        elif len(movers) >= 2:
            w = max(movers, key=lambda i: key[i])
            for i in movers:
                if i != w:
                    out[i] = Action.STAY
                    reason[i] = "R_loss"
    for i in range(n):
        j = occ.get(targets[i])
        if (
            j is not None
            and j > i
            and actions[i] is not Action.STAY
            and actions[j] is not Action.STAY
            and targets[j] == positions[i]
        ):
            left["swap"] += 1
    return out, reason, left


# --------------------------------------------------------------------------- the supervisor


@dataclass
class _Recovery:
    reason: str  # "S", "O" or "D"
    start_best: int
    remaining: int
    stays: int = 0
    visited: set[Cell] = field(default_factory=set)


@dataclass
class _RobotState:
    leg: tuple[Any, ...] | None = None
    best: int | None = None
    counter: int = 0
    last_progress_t: int = 0
    history: deque[Cell] = field(default_factory=deque)
    recovery: _Recovery | None = None
    losses: int = 0
    lost_last: bool = False
    prev_pos: Cell | None = None
    detour_ignored: int = 0
    leg_recoveries: int = 0  # recoveries started in the current leg


class ExecutionSupervisor:
    """The supervisor stage (see the module docstring). ``ExecutionSupervisor("off")`` is the
    identity and reports ``enabled = False`` (the engine then does not install it)."""

    def __init__(self, mode: str = "full", params: SupervisorParams | None = None) -> None:
        self.mechanisms = parse_mode(mode)
        self.mode = mode
        self.params = params or SupervisorParams()
        self.reset()

    @property
    def enabled(self) -> bool:
        return bool(self.mechanisms)

    def reset(self) -> None:
        self._robots: dict[int, _RobotState] = {}
        self.changes: Counter[str] = Counter()
        self.recoveries_started: Counter[str] = Counter()
        self.recoveries_ended: Counter[str] = Counter()
        self.recovery_steps = 0
        self.modified_robot_steps = 0
        self.eligible_robot_steps = 0
        self.robot_steps = 0
        self.policy_stays_nonidle = 0
        self.executed_stays_nonidle = 0
        self.max_consecutive_losses = 0
        self.r_left_to_arbiter: Counter[str] = Counter()
        self.steps = 0
        # Stage 3a monitoring (per robot; reporting only, no effect on the actions)
        self.modified_by_robot: Counter[int] = Counter()
        self.eligible_by_robot: Counter[int] = Counter()
        self.recoveries_by_robot: Counter[int] = Counter()
        self.r_losses_by_robot: Counter[int] = Counter()
        self.max_leg_recoveries = 0
        self.max_steps_without_progress = 0

    # ------------------------------------------------------------------ helpers
    def _state(self, robot: int) -> _RobotState:
        if robot not in self._robots:
            self._robots[robot] = _RobotState(history=deque(maxlen=self.params.osc_window + 1))
        return self._robots[robot]

    def _end(self, st: _RobotState, why: str) -> None:
        st.recovery = None
        st.counter = 0
        st.history.clear()
        self.recoveries_ended[why] += 1

    def _start(self, st: _RobotState, why: str, robot: int) -> None:
        assert st.best is not None
        st.recovery = _Recovery(why, st.best, self.params.hold)
        st.detour_ignored = 0
        st.leg_recoveries += 1
        st.history.clear()
        self.recoveries_started[why] += 1
        self.recoveries_by_robot[robot] += 1
        self.max_leg_recoveries = max(self.max_leg_recoveries, st.leg_recoveries)

    # ------------------------------------------------------------------ the stage
    def apply(
        self, view: WorldView, proposal: Sequence[Action], detour: frozenset[int] = frozenset()
    ) -> tuple[list[Action], list[SupervisorEvent]]:
        """The supervised proposal and the changed actions (events). ``detour``: robots that are
        members of an active resolver detour (from the coordinator)."""
        p = self.params
        use_s, use_o, use_r = (
            "S" in self.mechanisms,
            "O" in self.mechanisms,
            "R" in self.mechanisms,
        )
        if (use_s or use_o) and view.goal_distance is None:
            raise ConfigError("supervisor", "S and O need the view extra 'goal_distance'")
        t = view.t
        self.steps += 1
        out = list(proposal)
        why: dict[int, str] = {}
        robot_cells = [r.position for r in view.robots]
        obstacles: dict[Cell, Cell] = {o.position: o.velocity for o in view.obstacles}
        in_recovery: list[int] = []
        for r in view.robots:
            i = r.id
            st = self._state(i)
            pos = r.position
            self.robot_steps += 1
            nonidle = r.mode is not RobotMode.IDLE
            if nonidle and proposal[i] is Action.STAY:
                self.policy_stays_nonidle += 1
            if not (use_s or use_o):
                st.prev_pos = pos
                continue
            d = view.goal_distance[i] if view.goal_distance is not None else None
            leg = (r.task, r.mode, r.goal) if nonidle and r.goal is not None else None
            eligible = leg is not None and d is not None
            progressed = False
            if leg != st.leg:
                if st.recovery is not None:
                    self._end(st, "leg")
                st.leg, st.best, st.counter, st.last_progress_t = leg, d, 0, t
                st.leg_recoveries = 0
                st.history.clear()
            elif eligible and st.best is not None and d is not None and d < st.best:
                st.best, st.counter, st.last_progress_t, progressed = d, 0, t, True
            st.history.append(pos)
            if not eligible:
                st.prev_pos = pos
                continue
            self.eligible_robot_steps += 1
            self.eligible_by_robot[i] += 1
            self.max_steps_without_progress = max(
                self.max_steps_without_progress, t - st.last_progress_t
            )
            assert st.best is not None
            filtered, _ = filtered_suggestion(view, i, robot_cells, obstacles, p.sensing_radius)
            member = i in detour
            # ---- recovery termination
            rec = st.recovery
            if rec is not None:
                if rec.reason == "D" and not member:
                    self._end(st, "detour_end")
                elif rec.reason != "D" and member:
                    self._end(st, "detour")
                elif (
                    rec.reason != "D"
                    and not (p.escalate and st.leg_recoveries > 1)
                    and rec.start_best - st.best >= p.progress_cells
                ):
                    self._end(st, "progress")
                elif pos in rec.visited and pos != st.prev_pos:
                    self._end(st, "cycle")
                elif rec.remaining <= 0:
                    self._end(st, "hold")
                elif rec.stays >= p.stay_limit:
                    self._end(st, "stay_limit")
            # ---- stall counting (excused steps neither count nor reset)
            excused = filtered is Action.STAY or st.lost_last or member
            if st.recovery is None and not progressed and not excused:
                st.counter += 1
            if not member:
                st.detour_ignored = 0
            elif st.recovery is None:
                declined = filtered is not Action.STAY and proposal[i] != filtered
                st.detour_ignored = st.detour_ignored + 1 if declined else 0
            # ---- triggers
            if st.recovery is None:
                if member:
                    if use_s and st.detour_ignored >= p.stall_window:
                        self._start(st, "D", i)
                elif (
                    use_o
                    and t - st.last_progress_t >= p.osc_window
                    and is_oscillating(st.history, p.osc_window, p.osc_cells, p.osc_moves)
                ):
                    self._start(st, "O", i)
                elif use_s and st.counter >= (
                    p.repeat_window if p.escalate and st.leg_recoveries else p.stall_window
                ):
                    self._start(st, "S", i)
            # ---- recovery action
            if st.recovery is not None:
                st.recovery.visited.add(pos)
                if filtered != out[i]:
                    out[i] = filtered
                    why[i] = "recovery"
                in_recovery.append(i)
            st.prev_pos = pos
        if use_r:
            prio = robot_priorities(view.robots, view.tasks, t, p.alpha)
            key = {}
            for r in view.robots:
                losses = self._state(r.id).losses
                key[r.id] = (losses >= p.fairness_losses, prio[r.id], losses, -r.id)
            new, r_reason, left = right_of_way(robot_cells, out, key)
            self.r_left_to_arbiter.update(left)
            for i, reason in r_reason.items():
                if new[i] != out[i]:
                    out[i] = new[i]
                    why.setdefault(i, reason)
            losers = {i for i, reason in r_reason.items() if reason == "R_loss"}
            for r in view.robots:
                st = self._state(r.id)
                if r.id in losers:
                    self.r_losses_by_robot[r.id] += 1
                    st.losses += 1
                    st.lost_last = True
                    self.max_consecutive_losses = max(self.max_consecutive_losses, st.losses)
                else:
                    st.losses = 0
                    st.lost_last = False
        for i in in_recovery:
            rec = self._state(i).recovery
            assert rec is not None
            rec.remaining -= 1
            rec.stays = rec.stays + 1 if out[i] is Action.STAY else 0
            self.recovery_steps += 1
        events = []
        for i in sorted(why):
            if out[i] != proposal[i]:
                events.append(SupervisorEvent(t, i, why[i], proposal[i], out[i]))
                self.changes[why[i]] += 1
                self.modified_by_robot[i] += 1
        self.modified_robot_steps += len(events)
        for r in view.robots:
            if r.mode is not RobotMode.IDLE and out[r.id] is Action.STAY:
                self.executed_stays_nonidle += 1
        return out, events

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "params": {k: getattr(self.params, k) for k in self.params.__slots__},
            "steps": self.steps,
            "robot_steps": self.robot_steps,
            "eligible_robot_steps": self.eligible_robot_steps,
            "modified_robot_steps": self.modified_robot_steps,
            "modified_share": self.modified_robot_steps / self.robot_steps
            if self.robot_steps
            else None,
            "changes": {k: self.changes.get(k, 0) for k in CHANGE_REASONS},
            "recoveries_started": {k: self.recoveries_started.get(k, 0) for k in RECOVERY_KINDS},
            "recoveries_ended": {k: self.recoveries_ended.get(k, 0) for k in END_REASONS},
            "recovery_steps": self.recovery_steps,
            "policy_stays_nonidle": self.policy_stays_nonidle,
            "executed_stays_nonidle": self.executed_stays_nonidle,
            "max_consecutive_r_losses": self.max_consecutive_losses,
            "r_left_to_arbiter": {
                k: self.r_left_to_arbiter.get(k, 0) for k in ("occupied", "swap")
            },
            "monitor": self._monitor(),
        }

    def _monitor(self) -> dict[str, Any]:
        """Stage 3a monitoring of cumulative interventions, fairness and starvation (per robot
        lists indexed by robot id). ``max_robot_modified_share``: the largest share of a robot's
        eligible steps whose action was changed; ``max_steps_without_progress``: the longest time
        any eligible robot spent in a leg without improving ``D̄`` (a starvation indicator)."""
        ids = range(max(self._robots, default=-1) + 1)
        shares = [
            self.modified_by_robot[i] / self.eligible_by_robot[i]
            for i in ids
            if self.eligible_by_robot[i]
        ]
        return {
            "modified_by_robot": [self.modified_by_robot[i] for i in ids],
            "eligible_by_robot": [self.eligible_by_robot[i] for i in ids],
            "recoveries_by_robot": [self.recoveries_by_robot[i] for i in ids],
            "r_losses_by_robot": [self.r_losses_by_robot[i] for i in ids],
            "max_robot_modified_share": max(shares) if shares else None,
            "max_recoveries_in_a_leg": self.max_leg_recoveries,
            "max_steps_without_progress": self.max_steps_without_progress,
        }
