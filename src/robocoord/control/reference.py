"""Reference follower: the diagnostic plan-following controller (algorithms §5.3; model §9.2).

``a_i = â_i^plan`` — the coordinator's suggested action — or ``stay`` if there is no suggestion
or the suggestion is masked (static obstacle / out of bounds; a monitor suggestion never is).
It defines the counterfactual of model §9.2: for this controller the controller proposal *is*
the reference proposal ``a^ref``, so ``N^avoided = N^introduced = 0`` and its conflict set equals
``K^ref`` at every step (test X-10).

It reproduces the Issue #3 plain plan follower (``control.base.SuggestionFollower``, kept as the
``follow`` alias) action for action; it adds decision counts. Deterministic, O(N) per step.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from robocoord.core.types import Action
from robocoord.sim.view import WorldView

REFERENCE_DECISIONS: tuple[str, ...] = ("follow", "no_plan", "masked")


class ReferenceFollower:
    name = "reference"

    def __init__(self) -> None:
        self.decisions: Counter[str] = Counter()
        self.robot_steps = 0

    def reset(self, n_robots: int) -> None:
        self.decisions = Counter()
        self.robot_steps = 0

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        out: dict[int, Action] = {}
        for r in view.robots:
            a = Action.STAY if view.suggested is None else view.suggested[r.id]
            if a is Action.STAY:
                self.decisions["no_plan"] += 1  # no plan, at the goal, or a suggested wait
                out[r.id] = Action.STAY
            elif not view.masks[r.id][a]:
                self.decisions["masked"] += 1
                out[r.id] = Action.STAY
            else:
                self.decisions["follow"] += 1
                out[r.id] = a
        self.robot_steps += len(out)
        return out

    def episode_stats(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "robot_steps": self.robot_steps,
            "decisions": {k: self.decisions.get(k, 0) for k in REFERENCE_DECISIONS},
        }
