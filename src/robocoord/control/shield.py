"""Predictive safety shield (algorithms §5.4; model §5.5 execution layers; ADR-10).

Applied by the engine between the controller and the arbiter (pipeline step 5)::

    SHIELD(a_i):  c ← x_i + Δ(a_i)
                  if a_i ≠ stay and c ∈ Z_i(t) ∪ Ẑ_i(t): record (t, SH, i); return stay
                  return a_i

``Z_i`` / ``Ẑ_i``: the obstacles robot ``i`` senses (Chebyshev radius ``r_s``) and their
constant-velocity predictions, from :func:`robocoord.planning.constraints.sense`, the single
definition shared with the planners and the baseline controller.

**Properties.**

* *Guaranteed:* the shield only replaces a move by ``stay``; it never creates a move. A conflict
  in which the **shielded** robot moves (is at fault) therefore cannot be created by it.
* *Not guaranteed:* a robot that stays because of the shield can become the *target* of another
  robot's move (a vertex conflict at fault of that other robot) — measured as
  ``shield["introduced"]``. Robot–obstacle conflicts from unpredicted obstacle turns remain.
  Robot–robot conflicts are outside its scope. It does not mask invalid moves (an invalid
  proposal still reaches the arbiter as ``INV``, so masking defects stay visible).
* With the right-of-way baseline and the same ``r_s`` it never intervenes: the baseline already
  stays in front of current and predicted obstacle cells.

``mode``: ``"off"`` (identity; the default in training and the main experiments, model §5.5) or
``"predictive"``. Deterministic; O(min((2r_s+1)², K)) per moving robot.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from robocoord.core.errors import ConfigError
from robocoord.core.types import Action, Cell, add
from robocoord.planning.constraints import sense
from robocoord.sim.view import WorldView

SHIELD_MODES: tuple[str, ...] = ("off", "predictive")


@dataclass(frozen=True, slots=True)
class ShieldEvent:
    """``(t, SH, i)``: robot ``i``'s move ``action`` into ``cell`` was replaced by ``stay``."""

    t: int
    robot: int
    action: Action
    cell: Cell
    reason: str  # "current" (c ∈ Z_i) or "predicted" (c ∈ Ẑ_i ∖ Z_i)

    def to_json(self) -> list[object]:
        return [self.robot, self.action.symbol, list(self.cell), self.reason]


class Shield:
    """The predictive shield; ``Shield("off")`` is the identity."""

    def __init__(self, mode: str = "predictive", *, sensing_radius: int = 3) -> None:
        if mode not in SHIELD_MODES:
            raise ConfigError("shield", f"expected one of {SHIELD_MODES}, got {mode!r}")
        if sensing_radius < 0:
            raise ConfigError("shield.sensing_radius", "must be >= 0")
        self.mode = mode
        self.sensing_radius = sensing_radius
        self.interventions: Counter[str] = Counter()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def reset(self) -> None:
        self.interventions = Counter()

    def apply(
        self, view: WorldView, proposal: Sequence[Action]
    ) -> tuple[list[Action], list[ShieldEvent]]:
        """The proposal reaching the arbiter, and the interventions of this step."""
        out = list(proposal)
        if not self.enabled:
            return out, []
        obstacles = {o.position: o.velocity for o in view.obstacles}
        events: list[ShieldEvent] = []
        if not obstacles:
            return out, events
        for r in view.robots:
            a = out[r.id]
            if a is Action.STAY:
                continue
            c = add(r.position, a.delta)
            sensed = sense(view.map, r.position, obstacles, self.sensing_radius)
            if c in sensed.current:
                reason = "current"
            elif c in sensed.predicted:
                reason = "predicted"
            else:
                continue
            out[r.id] = Action.STAY
            events.append(ShieldEvent(view.t, r.id, a, c, reason))
            self.interventions[reason] += 1
        return out, events

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "interventions": sum(self.interventions.values()),
            "interventions_current": self.interventions.get("current", 0),
            "interventions_predicted": self.interventions.get("predicted", 0),
        }
