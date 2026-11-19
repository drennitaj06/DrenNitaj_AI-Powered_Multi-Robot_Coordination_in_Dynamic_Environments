"""Conflict detection and arbitration (model §3; algorithms §1).

Two pure functions:

* :func:`detect_conflicts` — the set of *proposed* conflict events of a joint proposal. It is
  also applied to the counterfactual reference proposal, so controller and reference conflicts
  are computed by identical code.
* :func:`arbitrate` — the emergency-stop layer: cancels every move involved in an event and,
  by a worklist cascade, every move into a cell whose occupant was frozen. Guarantees that no
  physical collision occurs (Lemma 3.1).

Both run in O(N + K) expected time.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell, add


class ConflictKind(Enum):
    RR_VERTEX = "RRv"
    RR_SWAP = "RRs"
    RO_VERTEX = "ROv"
    RO_SWAP = "ROs"
    INVALID = "INV"

    @property
    def is_robot_robot(self) -> bool:
        return self in (ConflictKind.RR_VERTEX, ConflictKind.RR_SWAP)

    @property
    def is_robot_obstacle(self) -> bool:
        return self in (ConflictKind.RO_VERTEX, ConflictKind.RO_SWAP)


def robot_tag(i: int) -> str:
    return f"r{i}"


def obstacle_tag(k: int) -> str:
    return f"o{k}"


@dataclass(frozen=True, order=True, slots=True)
class ConflictEvent:
    """``(t, type, entity pair)``; identity ignores the derived ``at_fault`` field."""

    t: int
    kind_code: str
    entities: tuple[str, ...]
    at_fault: tuple[int, ...] = field(default=(), compare=False)

    @property
    def kind(self) -> ConflictKind:
        return ConflictKind(self.kind_code)

    def to_json(self) -> list[object]:
        return [self.kind_code, list(self.entities), list(self.at_fault)]


def _event(
    t: int, kind: ConflictKind, entities: Sequence[str], at_fault: Sequence[int]
) -> ConflictEvent:
    return ConflictEvent(t, kind.value, tuple(sorted(entities)), tuple(sorted(at_fault)))


def detect_conflicts(
    t: int,
    positions: Sequence[Cell],
    proposal: Sequence[Action],
    obstacles_now: Sequence[Cell],
    obstacles_next: Sequence[Cell],
    grid: OccupancyGrid,
) -> frozenset[ConflictEvent]:
    """All proposed conflict events of ``proposal`` at state ``(positions, obstacles)``.

    ``obstacles_next`` are the already-resolved obstacle positions at ``t + 1`` (they do not depend
    on robot actions, model §2).
    """
    n = len(positions)
    if len(proposal) != n:
        raise ValueError("proposal must contain one action per robot")
    targets = [add(positions[i], proposal[i].delta) for i in range(n)]
    occ = {c: i for i, c in enumerate(positions)}
    obs_now = {c: k for k, c in enumerate(obstacles_now)}
    obs_next = {c: k for k, c in enumerate(obstacles_next)}
    events: set[ConflictEvent] = set()
    by_target: dict[Cell, list[int]] = defaultdict(list)

    for i in range(n):
        moving = proposal[i] is not Action.STAY
        tgt = targets[i]
        by_target[tgt].append(i)
        if not grid.is_free(tgt):
            events.add(_event(t, ConflictKind.INVALID, [robot_tag(i)], [i]))
        if moving and tgt in obs_next:
            k = obs_next[tgt]
            events.add(_event(t, ConflictKind.RO_VERTEX, [robot_tag(i), obstacle_tag(k)], [i]))
        if tgt in obs_now:
            k = obs_now[tgt]
            if obstacles_next[k] == positions[i]:
                events.add(_event(t, ConflictKind.RO_SWAP, [robot_tag(i), obstacle_tag(k)], [i]))
        if moving and tgt in occ:
            j = occ[tgt]
            if j > i and targets[j] == positions[i]:
                events.add(_event(t, ConflictKind.RR_SWAP, [robot_tag(i), robot_tag(j)], [i, j]))

    for robots in by_target.values():
        if len(robots) < 2:
            continue
        for a_idx in range(len(robots)):
            for b_idx in range(a_idx + 1, len(robots)):
                i, j = robots[a_idx], robots[b_idx]
                fault = [r for r in (i, j) if proposal[r] is not Action.STAY]
                events.add(_event(t, ConflictKind.RR_VERTEX, [robot_tag(i), robot_tag(j)], fault))
    return frozenset(events)


@dataclass(frozen=True, slots=True)
class Resolution:
    positions: tuple[Cell, ...]
    frozen: frozenset[int]
    blocked: frozenset[int]  # frozen only by the cascade (no event of their own)


def robots_in_events(events: frozenset[ConflictEvent]) -> set[int]:
    out: set[int] = set()
    for e in events:
        for tag in e.entities:
            if tag.startswith("r"):
                out.add(int(tag[1:]))
    return out


def arbitrate(
    positions: Sequence[Cell], proposal: Sequence[Action], events: frozenset[ConflictEvent]
) -> Resolution:
    """Cancel every conflicting move and cascade (worklist; each robot frozen at most once)."""
    n = len(positions)
    targets = [add(positions[i], proposal[i].delta) for i in range(n)]
    by_target: dict[Cell, list[int]] = defaultdict(list)
    for i, tgt in enumerate(targets):
        by_target[tgt].append(i)
    frozen = robots_in_events(events)
    direct = set(frozen)
    work = deque(sorted(frozen))
    while work:
        j = work.popleft()
        for i in by_target.get(positions[j], ()):
            if i not in frozen:
                frozen.add(i)
                work.append(i)
    new_positions = tuple(positions[i] if i in frozen else targets[i] for i in range(n))
    return Resolution(new_positions, frozenset(frozen), frozenset(frozen - direct))


def physical_violations(
    before: Sequence[Cell],
    after: Sequence[Cell],
    obstacles_next: Sequence[Cell],
    grid: OccupancyGrid,
) -> list[str]:
    """Post-resolution invariant check (G1). Returns human-readable violations (empty = OK)."""
    problems: list[str] = []
    seen: dict[Cell, int] = {}
    for i, c in enumerate(after):
        if c in seen:
            problems.append(f"robots {seen[c]} and {i} share cell {c}")
        seen[c] = i
        if not grid.is_free(c):
            problems.append(f"robot {i} is in a static obstacle or out of bounds at {c}")
        if abs(c[0] - before[i][0]) + abs(c[1] - before[i][1]) > 1:
            problems.append(f"robot {i} jumped from {before[i]} to {c}")
    obs = set(obstacles_next)
    for i, c in enumerate(after):
        if c in obs:
            problems.append(f"robot {i} overlaps a dynamic obstacle at {c}")
    prev = {c: i for i, c in enumerate(before)}
    for i, c in enumerate(after):
        j = prev.get(c)
        if j is not None and i < j and after[j] == before[i]:
            problems.append(f"robots {i} and {j} swapped through each other")
    return problems
