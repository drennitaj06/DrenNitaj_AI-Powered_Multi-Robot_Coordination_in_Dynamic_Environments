"""Conflict detection and arbitration vs an independent naive oracle (algorithms §1, X-10 basis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell
from robocoord.sim.conflicts import (
    ConflictKind,
    arbitrate,
    detect_conflicts,
    physical_violations,
)
from robocoord.world.motion import resolve_obstacle_moves

S, U, D, L, R = Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
LINE = OccupancyGrid.empty(6, 1)
OPEN = OccupancyGrid.empty(5, 5)


def kinds(events: frozenset) -> list[tuple[str, tuple[str, ...]]]:  # type: ignore[type-arg]
    return sorted((e.kind_code, e.entities) for e in events)


def run(
    pos: list[Cell],
    acts: list[Action],
    obs: list[Cell] | None = None,
    obs_next: list[Cell] | None = None,
    grid: OccupancyGrid = LINE,
):  # type: ignore[no-untyped-def]
    obs = obs or []
    obs_next = obs_next if obs_next is not None else obs
    ev = detect_conflicts(0, pos, acts, obs, obs_next, grid)
    res = arbitrate(pos, acts, ev)
    assert physical_violations(pos, list(res.positions), obs_next, grid) == []
    return ev, res


def test_no_conflict_and_following_moves() -> None:
    ev, res = run([(0, 0), (1, 0)], [R, R])  # 0 follows 1
    assert ev == frozenset() and res.positions == ((1, 0), (2, 0))


def test_vertex_conflict_freezes_both() -> None:
    ev, res = run([(0, 0), (2, 0)], [R, L])
    assert kinds(ev) == [("RRv", ("r0", "r1"))]
    assert next(iter(ev)).at_fault == (0, 1)
    assert res.positions == ((0, 0), (2, 0)) and res.blocked == frozenset()


def test_moving_into_staying_robot_only_mover_at_fault() -> None:
    ev, res = run([(0, 0), (1, 0)], [R, S])
    (e,) = ev
    assert e.kind is ConflictKind.RR_VERTEX and e.at_fault == (0,)
    assert res.positions == ((0, 0), (1, 0))


def test_swap_conflict() -> None:
    ev, res = run([(0, 0), (1, 0)], [R, L])
    assert kinds(ev) == [("RRs", ("r0", "r1"))]
    assert res.positions == ((0, 0), (1, 0))


def test_cascade_blocked_moves() -> None:
    # 2 follows 1 follows 0; 0 runs into a staying robot 3 → 0 collides, 1 and 2 blocked
    ev, res = run([(1, 0), (2, 0), (3, 0), (0, 0)], [L, L, L, S])
    assert kinds(ev) == [("RRv", ("r0", "r3"))]
    assert res.frozen == frozenset({0, 1, 2, 3}) and res.blocked == frozenset({1, 2})


def test_rotation_cycle_is_allowed() -> None:
    pos = [(0, 0), (1, 0), (1, 1), (0, 1)]
    ev, res = run(pos, [R, U, L, D], grid=OPEN)
    assert ev == frozenset()
    assert res.positions == ((1, 0), (1, 1), (0, 1), (0, 0))


def test_three_robots_one_cell() -> None:
    ev, _ = run([(1, 0), (0, 1), (2, 1)], [U, R, L], grid=OPEN)
    assert len(ev) == 3 and {e.kind for e in ev} == {ConflictKind.RR_VERTEX}


def test_invalid_moves() -> None:
    grid = OccupancyGrid.from_rows([".@."])
    ev, res = run([(0, 0)], [L], grid=grid)
    assert kinds(ev) == [("INV", ("r0",))]
    ev, res = run([(0, 0)], [R], grid=grid)
    assert kinds(ev) == [("INV", ("r0",))] and res.positions == ((0, 0),)


def test_robot_obstacle_vertex_and_swap() -> None:
    ev, res = run([(0, 0)], [R], obs=[(2, 0)], obs_next=[(1, 0)])
    assert kinds(ev) == [("ROv", ("o0", "r0"))]
    # An obstacle entering a robot-occupied cell is impossible under the obstacle move rule
    # (model §2), so this state never reaches arbitration; detection alone must still flag the swap.
    ev = detect_conflicts(0, [(0, 0)], [R], [(1, 0)], [(0, 0)], LINE)
    assert ("ROs", ("o0", "r0")) in kinds(ev)
    # moving into the cell an obstacle is leaving is fine
    ev, res = run([(0, 0)], [R], obs=[(1, 0)], obs_next=[(2, 0)])
    assert ev == frozenset() and res.positions == ((1, 0),)


# --------------------------------------------------------------------------- naive oracle


def naive_events(pos, acts, obs, obs_next, grid):  # type: ignore[no-untyped-def]
    """Direct transcription of the model §3 definitions with O(N^2) pair loops."""
    out = set()
    tgt = [(p[0] + a.delta[0], p[1] + a.delta[1]) for p, a in zip(pos, acts, strict=True)]
    n = len(pos)
    for i in range(n):
        if not grid.is_free(tgt[i]):
            out.add(("INV", (f"r{i}",)))
        for k in range(len(obs)):
            if acts[i] is not S and tgt[i] == obs_next[k]:
                out.add(("ROv", tuple(sorted((f"r{i}", f"o{k}")))))
            if tgt[i] == obs[k] and obs_next[k] == pos[i]:
                out.add(("ROs", tuple(sorted((f"r{i}", f"o{k}")))))
        for j in range(i + 1, n):
            if tgt[i] == tgt[j]:
                out.add(("RRv", (f"r{i}", f"r{j}")))
            if tgt[i] == pos[j] and tgt[j] == pos[i] and tgt[i] != pos[i]:
                out.add(("RRs", (f"r{i}", f"r{j}")))
    return out, tgt


def naive_arbitrate(pos, tgt, event_robots):  # type: ignore[no-untyped-def]
    frozen = set(event_robots)
    while True:
        cur = {j: (pos[j] if j in frozen else tgt[j]) for j in range(len(pos))}
        new = {
            i
            for i in range(len(pos))
            if i not in frozen and any(j != i and tgt[i] == cur[j] for j in range(len(pos)))
        }
        if not new:
            return frozen, tuple(cur[i] for i in range(len(pos)))
        frozen |= new


@st.composite
def joint_situations(draw):  # type: ignore[no-untyped-def]
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    w, h = int(rng.integers(2, 7)), int(rng.integers(1, 7))
    grid = OccupancyGrid(rng.random((h, w)) < rng.uniform(0, 0.35))
    free = grid.free_cells()
    if len(free) < 2:
        grid = OccupancyGrid.empty(w, max(h, 2))
        free = grid.free_cells()
    n_rob = int(rng.integers(1, min(len(free), 8) + 1))
    n_obs = int(rng.integers(0, min(len(free) - n_rob, 4) + 1))
    cells = [free[int(i)] for i in rng.choice(len(free), size=n_rob + n_obs, replace=False)]
    pos, obs = cells[:n_rob], cells[n_rob:]
    props = [
        (
            c[0] + Action(int(rng.integers(0, 5))).delta[0],
            c[1] + Action(int(rng.integers(0, 5))).delta[1],
        )
        for c in obs
    ]
    props = [
        p if abs(p[0] - c[0]) + abs(p[1] - c[1]) <= 1 else c
        for p, c in zip(props, obs, strict=True)
    ]
    obs_next = resolve_obstacle_moves(obs, props, grid, set(pos))
    acts = [Action(int(a)) for a in rng.integers(0, 5, size=n_rob)]
    return grid, pos, acts, obs, obs_next


@settings(max_examples=600)
@given(joint_situations())
def test_detection_and_arbitration_match_naive_oracle(case) -> None:  # type: ignore[no-untyped-def]
    grid, pos, acts, obs, obs_next = case
    ev = detect_conflicts(0, pos, acts, obs, obs_next, grid)
    expected, tgt = naive_events(pos, acts, obs, obs_next, grid)
    assert {(e.kind_code, e.entities) for e in ev} == expected
    res = arbitrate(pos, acts, ev)
    event_robots = {int(x[1:]) for _, ents in expected for x in ents if x.startswith("r")}
    frozen, positions = naive_arbitrate(pos, tgt, event_robots)
    assert res.frozen == frozen
    assert res.positions == positions
    assert res.blocked == frozen - event_robots
    # (a) no physical collision, (b) non-frozen robots executed their move exactly
    assert physical_violations(pos, list(res.positions), obs_next, grid) == []
    for i in range(len(pos)):
        if i not in res.frozen:
            assert res.positions[i] == tgt[i]
    # at-fault robots always moved
    for e in ev:
        assert all(acts[i] is not S for i in e.at_fault)


def test_physical_violation_detector_flags_problems() -> None:
    grid = OccupancyGrid.from_rows(["..@"])
    probs = physical_violations([(0, 0), (1, 0)], [(1, 0), (0, 0)], [], grid)
    assert any("swapped" in p for p in probs)
    probs = physical_violations([(0, 0), (1, 0)], [(1, 0), (1, 0)], [], grid)
    assert any("share" in p for p in probs)
    probs = physical_violations([(1, 0)], [(2, 0)], [], grid)
    assert any("static" in p for p in probs)
    probs = physical_violations([(0, 0)], [(1, 0)], [(1, 0)], grid)
    assert any("dynamic" in p for p in probs)
