"""Reference follower and safety shield (algorithms §5.3–5.4; Issue #4 stage 6).

Oracles: the Issue #3 ``SuggestionFollower`` (the reference follower must reproduce it), and the
simulator's own ``detect_conflicts`` / ``resolve_obstacle_moves`` for the shield properties.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.control.base import SuggestionFollower
from robocoord.control.reference import ReferenceFollower
from robocoord.control.shield import Shield
from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action
from robocoord.sim.conflicts import ConflictKind, detect_conflicts
from tests.unit.control.test_baseline import ONE, _random_world, make_view

S, U, D, L, R = Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
LINE = OccupancyGrid.empty(8, 1)

# ---------------------------------------------------------------- reference follower


@settings(max_examples=300)
@given(st.integers(0, 10**7))
def test_reference_reproduces_the_plain_plan_follower(seed: int) -> None:
    world = _random_world(seed, constant_velocity=True)
    if world is None:
        return
    view = world[1]
    assert ReferenceFollower().act(view) == SuggestionFollower().act(view)


def test_reference_decisions() -> None:
    ctrl = ReferenceFollower()
    ctrl.reset(3)
    out = ctrl.act(make_view(LINE, [((1, 0), ONE), ((3, 0), ONE), ((5, 0), ONE)], [R, S, U]))
    assert out == {0: R, 1: S, 2: S}  # UP leaves the 8 × 1 grid: masked
    assert ctrl.act(make_view(LINE, [((1, 0), ONE)], None)) == {0: S}
    stats = ctrl.episode_stats()
    assert stats["decisions"] == {"follow": 1, "no_plan": 2, "masked": 1}
    assert stats["robot_steps"] == 4 and stats["controller"] == "reference"


# ---------------------------------------------------------------- shield: cases


def test_shield_off_is_identity_and_validation() -> None:
    view = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((3, 0), (0, 0))])
    assert Shield("off").apply(view, [R]) == ([R], [])
    assert not Shield("off").enabled and Shield().enabled
    with pytest.raises(ConfigError):
        Shield("aggressive")
    with pytest.raises(ConfigError):
        Shield(sensing_radius=-1)


def test_shield_replacements_and_logging() -> None:
    shield = Shield()
    # move into a current obstacle cell → stay, reason "current"
    view = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((3, 0), (1, 0))], t=7)
    out, ev = shield.apply(view, [R])
    assert out == [S] and len(ev) == 1
    assert (ev[0].t, ev[0].robot, ev[0].action, ev[0].cell, ev[0].reason) == (
        7,
        0,
        R,
        (3, 0),
        "current",
    )
    # move into a predicted obstacle cell → stay, reason "predicted"
    view = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((4, 0), (-1, 0))])
    out, ev = shield.apply(view, [R])
    assert out == [S] and ev[0].reason == "predicted"
    # safe move, stay, and an unsensed obstacle (radius) are untouched
    view = make_view(LINE, [((2, 0), ONE)], [L], obstacles=[((4, 0), (-1, 0))])
    assert shield.apply(view, [L]) == ([L], [])
    assert shield.apply(view, [S]) == ([S], [])
    far = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((4, 0), (-1, 0))])
    assert Shield(sensing_radius=1).apply(far, [R]) == ([R], [])
    assert shield.stats() == {
        "mode": "predictive",
        "interventions": 2,
        "interventions_current": 1,
        "interventions_predicted": 1,
    }
    shield.reset()
    assert shield.stats()["interventions"] == 0


def test_shield_leaves_robot_robot_conflicts_and_invalid_moves_alone() -> None:
    """Out of scope: RR vertex and swap proposals and invalid (masked) moves pass unchanged, so
    the arbiter still sees and counts them."""
    shield = Shield()
    vertex = make_view(LINE, [((1, 0), ONE), ((3, 0), ONE)], [R, L], obstacles=[((7, 0), (0, 0))])
    assert shield.apply(vertex, [R, L]) == ([R, L], [])
    swap = make_view(LINE, [((2, 0), ONE), ((3, 0), ONE)], [R, L], obstacles=[((7, 0), (0, 0))])
    assert shield.apply(swap, [R, L]) == ([R, L], [])
    invalid = make_view(LINE, [((2, 0), ONE)], [U], obstacles=[((7, 0), (0, 0))])
    out, _ = shield.apply(invalid, [U])
    assert out == [U]
    events = detect_conflicts(0, [(2, 0)], out, [(7, 0)], [(7, 0)], LINE)
    assert {e.kind for e in events} == {ConflictKind.INVALID}


# ---------------------------------------------------------------- shield: properties


@settings(max_examples=400)
@given(st.integers(0, 10**7), st.sampled_from([True, False]))
def test_shield_only_replaces_moves_by_stay_and_is_never_at_fault(seed: int, cv: bool) -> None:
    """For arbitrary simultaneous proposals of all robots: every output action is the input or
    ``stay``; no conflict of the shielded proposal has a shielded robot at fault; the shield is
    deterministic."""
    world = _random_world(seed, constant_velocity=cv)
    if world is None:
        return
    grid, view, now, nxt = world
    assert view.suggested is not None
    proposal = list(view.suggested)  # arbitrary actions, masked ones included
    out, ev = Shield().apply(view, proposal)
    assert all(b == a or b is S for a, b in zip(proposal, out, strict=True))
    shielded = {e.robot for e in ev}
    assert shielded == {i for i, (a, b) in enumerate(zip(proposal, out, strict=True)) if a != b}
    positions = [r.position for r in view.robots]
    for e in detect_conflicts(view.t, positions, out, now, nxt, grid):
        assert not (set(e.at_fault) & shielded)
    assert Shield().apply(view, proposal) == (out, ev)


@settings(max_examples=400)
@given(st.integers(0, 10**7))
def test_shield_removes_robot_obstacle_conflicts_under_constant_velocity(seed: int) -> None:
    """With obstacles that stay or keep their velocity and r_s ≥ 2, the shielded proposal of
    the plain plan follower has no robot–obstacle conflict (the shield's purpose); with
    arbitrary obstacle turns this is not guaranteed (see the next test)."""
    world = _random_world(seed, constant_velocity=True)
    if world is None:
        return
    grid, view, now, nxt = world
    ctrl = ReferenceFollower().act(view)
    raw = [ctrl[r.id] for r in view.robots]
    out, _ = Shield().apply(view, raw)
    positions = [r.position for r in view.robots]
    events = detect_conflicts(view.t, positions, out, now, nxt, grid)
    assert not any(e.kind.is_robot_obstacle for e in events)


def test_shield_cannot_prevent_unpredicted_obstacle_turns() -> None:
    """Not claimed: an obstacle that turns into the robot's target is not predicted."""
    found = False
    for seed in range(400):
        world = _random_world(seed, constant_velocity=False)
        if world is None:
            continue
        grid, view, now, nxt = world
        ctrl = ReferenceFollower().act(view)
        out, _ = Shield().apply(view, [ctrl[r.id] for r in view.robots])
        positions = [r.position for r in view.robots]
        events = detect_conflicts(view.t, positions, out, now, nxt, grid)
        found = found or any(e.kind is ConflictKind.RO_VERTEX for e in events)
    assert found
