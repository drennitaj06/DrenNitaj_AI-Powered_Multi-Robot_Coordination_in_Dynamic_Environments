"""Shared primitive types: cells, actions and displacements (model §1, §3)."""

from __future__ import annotations

from enum import IntEnum

type Cell = tuple[int, int]
"""Grid cell ``(x, y)``; ``x`` grows to the right, ``y`` grows upwards (``UP`` = +y)."""


class Action(IntEnum):
    """The five primitive robot actions (model §3)."""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    @property
    def delta(self) -> Cell:
        """Displacement ``Δ(a)`` of this action."""
        return DELTAS[self]

    @property
    def symbol(self) -> str:
        """One-letter script symbol (S, U, D, L, R)."""
        return _SYMBOLS[self]

    @classmethod
    def from_symbol(cls, symbol: str) -> Action:
        """Parse a one-letter script symbol (case-insensitive)."""
        try:
            return _FROM_SYMBOL[symbol.upper()]
        except KeyError:
            raise ValueError(
                f"unknown action symbol {symbol!r}; expected one of S, U, D, L, R"
            ) from None


DELTAS: dict[Action, Cell] = {
    Action.STAY: (0, 0),
    Action.UP: (0, 1),
    Action.DOWN: (0, -1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

_SYMBOLS: dict[Action, str] = {
    Action.STAY: "S",
    Action.UP: "U",
    Action.DOWN: "D",
    Action.LEFT: "L",
    Action.RIGHT: "R",
}
_FROM_SYMBOL: dict[str, Action] = {v: k for k, v in _SYMBOLS.items()}

#: The four movement actions in canonical order (used for deterministic tie-breaking).
MOVES: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


def add(cell: Cell, delta: Cell) -> Cell:
    """Return ``cell + delta``."""
    return (cell[0] + delta[0], cell[1] + delta[1])


def manhattan(a: Cell, b: Cell) -> int:
    """Manhattan distance ``‖a − b‖₁``."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: Cell, b: Cell) -> int:
    """Chebyshev distance ``‖a − b‖∞``."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def action_between(a: Cell, b: Cell) -> Action:
    """The action moving from ``a`` to the 4-neighbour (or identical) cell ``b``."""
    d = (b[0] - a[0], b[1] - a[1])
    for action, delta in DELTAS.items():
        if delta == d:
            return action
    raise ValueError(f"cells {a} and {b} are not 4-adjacent")
