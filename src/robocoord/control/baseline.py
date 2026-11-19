"""Rule-based baseline controller with deterministic right-of-way (algorithms §5.3; model §5.5).

For robot ``i`` with suggested action ``â_i`` (the path monitor's ``â^plan``) and target
``c = x_i + Δ(â_i)``, in this order:

1. ``â_i = stay`` (no valid plan, or at the goal) → **stay** (``no_plan``).
2. ``â_i`` is masked (static obstacle or out of bounds) → **stay** (``masked``). Cannot happen
   for a monitor suggestion; kept so the action mask is always respected.
3. ``c`` is occupied at ``t`` by a robot → **stay** (``robot_ahead``); by a dynamic obstacle
   → **stay** (``obstacle_ahead``).
4. ``c ∈ Ẑ_i(t)``, the constant-velocity predictions of the obstacles robot ``i`` senses
   (``planning.constraints.sense``, the single definition of ``Ẑ_i``) → **stay**
   (``predicted_obstacle``).
5. Another robot ``j`` (necessarily adjacent to ``c``) also has suggested target ``c`` and
   ``(π̃_j, −j) > (π̃_i, −i)`` → **stay**, yielding right-of-way (``yield_priority`` if
   ``π̃_j > π̃_i``, ``yield_tie`` if the priorities are equal and ``j < i``).
6. Otherwise → **move** to ``c``.

``π̃_i`` is the effective priority of the robot's carried or assigned task, the aged weight
``w(π_j)(1 + α·max(t − ρ_j, 0))`` of model §4 with the allocation's ``α``; an idle robot has
``π̃ = 0``, lower than any task (weights are ≥ 1). The age keeps growing after pickup (the
engine's ``Task.age``), an interpretation documented in `04` §5.3 notes.

**Collision property (Guaranteed, proof in `04` §5.3 notes).** If every robot plays this rule on
the same view, the joint proposal contains no robot–robot conflict (vertex or swap), no
robot–obstacle swap and no invalid move. A robot–obstacle *vertex* conflict needs an obstacle
whose move to ``c`` was not predicted (it turned, or was outside the sensing radius).

**Not a deadlock rule.** Two robots facing each other both stay (step 3), as do robots in a
cycle; the controller itself never detours. Stalls are left to the controller-agnostic path
monitor (T4). Deadlock detours (the opt-in resolver, ``planning/deadlock.py``, trigger T7) and
endpoint recovery (``planning/endpoint.py``, trigger T8) change only the goals the monitor plans
to; this controller's rules are unchanged by either.

Deterministic: no randomness; decisions depend only on the view. Time per step
O(M + N·min((2r_s+1)², K)): the M task records are indexed once, cells and claims are hashed
(O(N)), and ``Ẑ_i`` is sensed only for robots that reach step 4 (K obstacles). Memory O(N + M).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from robocoord.allocation.benefits import robot_priorities
from robocoord.core.errors import ConfigError
from robocoord.core.types import Action, Cell, add
from robocoord.planning.constraints import sense
from robocoord.sim.view import WorldView

DECISIONS: tuple[str, ...] = (
    "move",
    "no_plan",
    "masked",
    "robot_ahead",
    "obstacle_ahead",
    "predicted_obstacle",
    "yield_priority",
    "yield_tie",
)


class BaselineController:
    """Deterministic right-of-way baseline (``name = "baseline"``).

    ``alpha`` must equal the allocation's aging rate (default 0.01, model §11);
    ``sensing_radius`` is ``r_s`` (default 3, as the planners' ``ConstraintParams``).
    """

    name = "baseline"

    def __init__(self, *, alpha: float = 0.01, sensing_radius: int = 3) -> None:
        if not alpha >= 0:
            raise ConfigError("baseline.alpha", "must be >= 0")
        if sensing_radius < 0:
            raise ConfigError("baseline.sensing_radius", "must be >= 0")
        self.alpha = alpha
        self.sensing_radius = sensing_radius
        self.decisions: Counter[str] = Counter()
        self.robot_steps = 0
        self.last: dict[int, str] = {}  # robot → decision of the latest step

    def reset(self, n_robots: int) -> None:
        self.decisions = Counter()
        self.robot_steps = 0
        self.last = {}

    # ------------------------------------------------------------------ priorities
    def effective_priority(self, view: WorldView) -> dict[int, float]:
        """``π̃_i`` per robot: aged weight of its task, 0 when idle."""
        return robot_priorities(view.robots, view.tasks, view.t, self.alpha)

    # ------------------------------------------------------------------ the rule
    def decide(self, view: WorldView) -> dict[int, tuple[Action, str]]:
        """Action and decision reason per robot (see the module docstring)."""
        suggested = view.suggested
        robots = view.robots
        if suggested is None:
            return {r.id: (Action.STAY, "no_plan") for r in robots}
        prio = self.effective_priority(view)
        robot_cells = {r.position for r in robots}
        obstacles: dict[Cell, Cell] = {o.position: o.velocity for o in view.obstacles}
        target: dict[int, Cell] = {}
        claims: dict[Cell, list[int]] = {}
        for r in robots:
            a = suggested[r.id]
            if a is not Action.STAY and view.masks[r.id][a]:
                c = add(r.position, a.delta)
                target[r.id] = c
                claims.setdefault(c, []).append(r.id)
        # the right-of-way winner of each claimed cell: max (π̃, −id)
        winner = {c: max(ids, key=lambda j: (prio[j], -j)) for c, ids in claims.items()}
        out: dict[int, tuple[Action, str]] = {}
        for r in robots:
            a = suggested[r.id]
            if a is Action.STAY:
                out[r.id] = (Action.STAY, "no_plan")
                continue
            if not view.masks[r.id][a]:
                out[r.id] = (Action.STAY, "masked")
                continue
            c = target[r.id]
            if c in robot_cells:
                out[r.id] = (Action.STAY, "robot_ahead")
            elif c in obstacles:
                out[r.id] = (Action.STAY, "obstacle_ahead")
            elif c in sense(view.map, r.position, obstacles, self.sensing_radius).predicted:
                out[r.id] = (Action.STAY, "predicted_obstacle")
            elif winner[c] != r.id:
                w = winner[c]
                reason = "yield_priority" if prio[w] > prio[r.id] else "yield_tie"
                out[r.id] = (Action.STAY, reason)
            else:
                out[r.id] = (a, "move")
        return out

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        decided = self.decide(view)
        self.last = {i: d[1] for i, d in decided.items()}
        self.decisions.update(self.last.values())
        self.robot_steps += len(decided)
        return {i: d[0] for i, d in decided.items()}

    # ------------------------------------------------------------------ reporting
    def episode_stats(self) -> dict[str, Any]:
        """Decision counts per reason and the share of active robot-steps spent waiting.

        *Active* = the robot had a suggestion to follow (all decisions except ``no_plan``).
        ``waits_per_active`` splits waiting into its causes; ``yields`` counts right-of-way
        decisions (priority vs id tie-break). Deterministic.
        """
        d = self.decisions
        active = self.robot_steps - d.get("no_plan", 0)
        waits = active - d.get("move", 0)
        return {
            "controller": self.name,
            "robot_steps": self.robot_steps,
            "active_robot_steps": active,
            "decisions": {k: d.get(k, 0) for k in DECISIONS},
            "yields": d.get("yield_priority", 0) + d.get("yield_tie", 0),
            "wait_fraction": (waits / active) if active else None,
        }
