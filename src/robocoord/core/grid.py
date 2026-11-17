"""Static occupancy grid ``S(t)`` and the 4-connected free-space graph (model §1)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from robocoord.core.types import MOVES, Cell, add

FREE_CHAR = "."
BLOCKED_CHAR = "@"
_BLOCKED_INPUT = frozenset(
    "@#TOW"
)  # MovingAI: @ O T W are impassable; '#' accepted for convenience
_FREE_INPUT = frozenset(".GS")


class OccupancyGrid:
    """A ``W × H`` grid of free/blocked cells.

    Storage is a boolean NumPy array indexed ``[y, x]``. Text rows are written top-down,
    i.e. the first text row is ``y = H - 1`` (so ``UP`` points to the top of the screen).
    Every mutation increments :attr:`version`, which invalidates cached distance fields.
    """

    __slots__ = ("_blocked", "_version")

    def __init__(self, blocked: npt.ArrayLike) -> None:
        arr = np.array(blocked, dtype=bool)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError(f"grid must be a non-empty 2-D array, got shape {arr.shape}")
        self._blocked: npt.NDArray[np.bool_] = arr
        self._version = 0

    # ------------------------------------------------------------------ construction
    @classmethod
    def empty(cls, width: int, height: int) -> OccupancyGrid:
        """An obstacle-free grid."""
        if width < 1 or height < 1:
            raise ValueError("width and height must be >= 1")
        return cls(np.zeros((height, width), dtype=bool))

    @classmethod
    def from_rows(cls, rows: Sequence[str]) -> OccupancyGrid:
        """Parse text rows (top row first). ``.`` = free, ``@``/``#`` = blocked."""
        if not rows:
            raise ValueError("map must have at least one row")
        width = len(rows[0])
        blocked = np.zeros((len(rows), width), dtype=bool)
        height = len(rows)
        for r, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"row {r} has length {len(row)}, expected {width}")
            for x, ch in enumerate(row):
                if ch in _BLOCKED_INPUT:
                    blocked[height - 1 - r, x] = True
                elif ch not in _FREE_INPUT:
                    raise ValueError(f"row {r}, column {x}: unknown map character {ch!r}")
        return cls(blocked)

    def to_rows(self) -> list[str]:
        """Inverse of :meth:`from_rows` (uses ``.`` and ``@``)."""
        return [
            "".join(BLOCKED_CHAR if b else FREE_CHAR for b in self._blocked[y])
            for y in range(self.height - 1, -1, -1)
        ]

    def copy(self) -> OccupancyGrid:
        """Independent copy (version reset to 0)."""
        return OccupancyGrid(self._blocked.copy())

    # ------------------------------------------------------------------ queries
    @property
    def width(self) -> int:
        return int(self._blocked.shape[1])

    @property
    def height(self) -> int:
        return int(self._blocked.shape[0])

    @property
    def version(self) -> int:
        """Incremented on every change; used as a cache key."""
        return self._version

    @property
    def blocked(self) -> npt.NDArray[np.bool_]:
        """Read-only view of the ``[y, x]`` blocked array."""
        view = self._blocked.view()
        view.flags.writeable = False
        return view

    def in_bounds(self, c: Cell) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def is_blocked(self, c: Cell) -> bool:
        """True if ``c`` is out of bounds or a static obstacle."""
        return not self.in_bounds(c) or bool(self._blocked[c[1], c[0]])

    def is_free(self, c: Cell) -> bool:
        """True if ``c`` is in bounds and not a static obstacle."""
        return self.in_bounds(c) and not bool(self._blocked[c[1], c[0]])

    def neighbors(self, c: Cell) -> Iterator[Cell]:
        """Free 4-neighbours of ``c`` in canonical order (UP, DOWN, LEFT, RIGHT)."""
        for action in MOVES:
            n = add(c, action.delta)
            if self.is_free(n):
                yield n

    def free_cells(self) -> list[Cell]:
        """All free cells, sorted by ``(x, y)``."""
        ys, xs = np.nonzero(~self._blocked)
        return sorted(zip(xs.tolist(), ys.tolist(), strict=True))

    def num_free(self) -> int:
        return int((~self._blocked).sum())

    # ------------------------------------------------------------------ mutation
    def set_blocked(self, c: Cell, blocked: bool) -> bool:
        """Set the state of ``c``. Returns True if the state changed (and bumps the version)."""
        if not self.in_bounds(c):
            raise ValueError(f"cell {c} is out of bounds")
        if bool(self._blocked[c[1], c[0]]) == blocked:
            return False
        self._blocked[c[1], c[0]] = blocked
        self._version += 1
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OccupancyGrid) and bool(
            np.array_equal(self._blocked, other._blocked)
        )

    def __hash__(self) -> int:  # pragma: no cover - grids are mutable; hashing is not supported
        raise TypeError("OccupancyGrid is mutable and unhashable")

    def __repr__(self) -> str:
        return f"OccupancyGrid({self.width}x{self.height}, free={self.num_free()})"
