"""Scheduled map events with the deferral rule (model §1; FR-ENV-3)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Cell


@dataclass(frozen=True, slots=True)
class MapEvent:
    time: int
    cell: Cell
    block: bool  # True = block, False = unblock

    def describe(self) -> str:
        return f"{'block' if self.block else 'unblock'} {self.cell}"


class MapEventSchedule:
    """Applies due map events; a *block* on an occupied cell waits until the cell is vacated."""

    def __init__(self, events: Iterable[MapEvent]) -> None:
        # stable sort keeps the scenario's order for events at the same time
        self._pending: list[MapEvent] = sorted(events, key=lambda e: e.time)
        self._deferred: list[MapEvent] = []

    def apply_due(self, t: int, grid: OccupancyGrid, occupied: set[Cell]) -> list[MapEvent]:
        """Apply events due at ``t`` plus previously deferred ones; return the applied events."""
        due = self._deferred + [e for e in self._pending if e.time <= t]
        self._pending = [e for e in self._pending if e.time > t]
        self._deferred = []
        applied: list[MapEvent] = []
        for event in due:
            if event.block and event.cell in occupied:
                self._deferred.append(event)
                continue
            if grid.set_blocked(event.cell, event.block):
                applied.append(event)
        return applied

    @property
    def deferred(self) -> tuple[MapEvent, ...]:
        return tuple(self._deferred)

    def has_future_events(self) -> bool:
        return bool(self._pending or self._deferred)
