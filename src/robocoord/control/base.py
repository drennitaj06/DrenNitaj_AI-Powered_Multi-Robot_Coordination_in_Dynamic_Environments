"""Controller interface (operational level) and the simple controllers of Issue #2.

Learned controllers (MADQL, PPO; Issues #5–#6), the rule-based baseline, the reference follower
and the safety shield (Issue #4) implement the same :class:`Controller` protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from robocoord.core.types import Action
from robocoord.sim.view import WorldView


class Controller(Protocol):
    name: str

    def reset(self, n_robots: int) -> None: ...

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        """Return one action per robot id. Observations are used by learned controllers."""
        ...


class StayController:
    name = "stay"

    def reset(self, n_robots: int) -> None:
        return None

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        return {r.id: Action.STAY for r in view.robots}


class SuggestionFollower:
    """Plays the coordinator's suggested action ``â^plan`` (masked; ``stay`` otherwise).

    The minimal plan follower used by the Issue #3 tests and experiment E2. The reference-follower
    controller of Issue #4 (``control.reference``) supersedes it in the controller registry.
    """

    name = "follow"

    def reset(self, n_robots: int) -> None:
        return None

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        if view.suggested is None:
            return {r.id: Action.STAY for r in view.robots}
        return {
            r.id: s if view.masks[r.id][s] else Action.STAY
            for r, s in zip(view.robots, view.suggested, strict=True)
        }


class ScriptedController:
    """Plays a fixed action string per robot (S/U/D/L/R, spaces ignored); then stays.

    Each robot has a script pointer. A scripted *move* that the arbiter cancelled (the robot did
    not move although the move was statically feasible) is **retried** on the next step, so a
    stopped robot waits instead of skipping part of its route. A move into a wall is proposed
    once (recorded as an ``INV`` event, cancelled) and then skipped. ``S`` always advances.
    """

    name = "scripted"

    def __init__(self, scripts: Sequence[str]) -> None:
        self._scripts = [[Action.from_symbol(c) for c in s if c != " "] for s in scripts]
        self._ptr = [0] * len(self._scripts)
        self._last: list[tuple[tuple[int, int], Action] | None] = [None] * len(self._scripts)

    def reset(self, n_robots: int) -> None:
        if n_robots != len(self._scripts):
            raise ValueError(f"script has {len(self._scripts)} robots, scenario has {n_robots}")
        self._ptr = [0] * n_robots
        self._last = [None] * n_robots

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        out: dict[int, Action] = {}
        for r in view.robots:
            last = self._last[r.id]
            if last is not None:
                prev_pos, prev_action = last
                cancelled = prev_action is not Action.STAY and r.position == prev_pos
                if not (cancelled and view.masks[r.id][prev_action]):
                    self._ptr[r.id] += 1
            script = self._scripts[r.id]
            k = self._ptr[r.id]
            action = script[k] if k < len(script) else Action.STAY
            self._last[r.id] = (r.position, action) if k < len(script) else None
            out[r.id] = action
        return out

    def finished(self, robot: int) -> bool:
        return self._ptr[robot] >= len(self._scripts[robot])


class RandomController:
    """Uniformly random *masked* actions from a seeded generator (stress tests, demos)."""

    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def reset(self, n_robots: int) -> None:
        return None

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        out: dict[int, Action] = {}
        for r in view.robots:
            allowed = view.allowed(r.id)
            out[r.id] = allowed[int(self._rng.integers(0, len(allowed)))]
        return out
