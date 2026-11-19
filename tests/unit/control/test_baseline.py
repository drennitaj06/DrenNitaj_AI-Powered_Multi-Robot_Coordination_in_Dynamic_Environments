"""Baseline right-of-way controller (algorithms §5.3; Issue #4 stage 5A).

Hand-built views check each rule; property tests check the collision property against the
simulator's own conflict detector (``sim.conflicts.detect_conflicts``) and obstacle move rule
(``world.motion.resolve_obstacle_moves``), which are independent of the controller.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.control.base import SuggestionFollower
from robocoord.control.baseline import DECISIONS, BaselineController
from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell, add
from robocoord.sim.conflicts import ConflictKind, detect_conflicts
from robocoord.sim.view import MapView, ObstacleView, RobotView, TaskView, WorldView, action_mask
from robocoord.world.entities import RobotMode, TaskStatus
from robocoord.world.motion import resolve_obstacle_moves

S, U, D, L, R = Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT


def make_view(
    grid: OccupancyGrid,
    robots: list[tuple[Cell, tuple[int, int] | None]],  # (position, (level, release) or None)
    suggested: list[Action] | None,
    obstacles: list[tuple[Cell, Cell]] = (),  # type: ignore[assignment]
    t: int = 0,
) -> WorldView:
    """Robot i carries task i (if given) in mode to_pickup; idle otherwise."""
    rv, tv = [], []
    for i, (pos, task) in enumerate(robots):
        if task is None:
            rv.append(RobotView(i, pos, RobotMode.IDLE, None, None))
        else:
            level, release = task
            rv.append(RobotView(i, pos, RobotMode.TO_PICKUP, i, pos))
            tv.append(TaskView(i, pos, pos, level, release, TaskStatus.ASSIGNED, i, True))
    ov = tuple(ObstacleView(k, z, v) for k, (z, v) in enumerate(obstacles))
    masks = tuple(action_mask(grid, pos) for pos, _ in robots)
    sug = tuple(suggested) if suggested is not None else None
    return WorldView(t, 100, MapView(grid), tuple(rv), ov, tuple(tv), masks, sug)


def decide(view: WorldView, **kw: object) -> dict[int, tuple[Action, str]]:
    return BaselineController(**kw).decide(view)  # type: ignore[arg-type]


LINE = OccupancyGrid.empty(8, 1)
ONE = (1, 0)  # level 1 task released at t = 0


# ---------------------------------------------------------------- single-robot rules


def test_move_stay_and_mask() -> None:
    assert decide(make_view(LINE, [((2, 0), ONE)], [R])) == {0: (R, "move")}
    assert decide(make_view(LINE, [((2, 0), ONE)], [S])) == {0: (S, "no_plan")}
    assert decide(make_view(LINE, [((2, 0), ONE)], None)) == {0: (S, "no_plan")}
    # UP leaves the 8 × 1 grid: masked, never proposed
    assert decide(make_view(LINE, [((2, 0), ONE)], [U])) == {0: (S, "masked")}


def test_occupied_and_predicted_cells() -> None:
    robots = [((2, 0), ONE), ((3, 0), ONE)]
    assert decide(make_view(LINE, robots, [R, S]))[0] == (S, "robot_ahead")
    view = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((3, 0), (0, 0))])
    assert decide(view)[0] == (S, "obstacle_ahead")
    # obstacle at x = 4 moving left: predicted at x = 3, the robot's target
    view = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((4, 0), (-1, 0))])
    assert decide(view)[0] == (S, "predicted_obstacle")
    assert decide(view, sensing_radius=1)[0] == (R, "move")  # not sensed: documented limit
    # moving away: not predicted at the target
    view = make_view(LINE, [((2, 0), ONE)], [R], obstacles=[((4, 0), (1, 0))])
    assert decide(view)[0] == (R, "move")


# ---------------------------------------------------------------- right-of-way


def test_vertex_conflict_priority_then_id() -> None:
    """Robots at x = 1 and x = 3 both want x = 2."""
    hi_right = make_view(LINE, [((1, 0), (1, 0)), ((3, 0), (3, 0))], [R, L])
    assert decide(hi_right) == {0: (S, "yield_priority"), 1: (L, "move")}
    hi_left = make_view(LINE, [((1, 0), (3, 0)), ((3, 0), (1, 0))], [R, L])
    assert decide(hi_left) == {0: (R, "move"), 1: (S, "yield_priority")}
    tie = make_view(LINE, [((1, 0), ONE), ((3, 0), ONE)], [R, L])
    assert decide(tie) == {0: (R, "move"), 1: (S, "yield_tie")}  # lower id wins


def test_idle_robot_is_lowest() -> None:
    view = make_view(LINE, [((1, 0), None), ((3, 0), ONE)], [R, L])
    assert decide(view) == {0: (S, "yield_priority"), 1: (L, "move")}


def test_aging_breaks_equal_levels() -> None:
    """Same level; robot 1's task is older, so its effective weight is higher at t = 50."""
    view = make_view(LINE, [((1, 0), (2, 40)), ((3, 0), (2, 0))], [R, L], t=50)
    assert decide(view) == {0: (S, "yield_priority"), 1: (L, "move")}
    assert decide(view, alpha=0.0) == {0: (R, "move"), 1: (S, "yield_tie")}


@pytest.mark.parametrize(
    ("levels", "winner"),
    [((1, 1, 1, 1), 0), ((1, 2, 1, 1), 1), ((1, 1, 3, 3), 2), ((3, 2, 1, 3), 0), ((1, 1, 1, 2), 3)],
)
def test_four_robots_compete_for_one_cell(levels: tuple[int, ...], winner: int) -> None:
    grid = OccupancyGrid.empty(5, 5)
    pos = [(1, 2), (3, 2), (2, 1), (2, 3)]  # all adjacent to the centre (2, 2)
    view = make_view(grid, [(p, (lv, 0)) for p, lv in zip(pos, levels, strict=True)], [R, L, U, D])
    out = decide(view)
    movers = [i for i, (a, _) in out.items() if a is not S]
    assert movers == [winner]
    for i, (_, reason) in out.items():
        if i != winner:
            expect = "yield_priority" if levels[winner] > levels[i] else "yield_tie"
            assert reason == expect


def test_three_robots_compete() -> None:
    grid = OccupancyGrid.empty(5, 5)
    view = make_view(grid, [((1, 2), ONE), ((3, 2), ONE), ((2, 1), ONE)], [R, L, U])
    out = decide(view)
    assert [i for i, (a, _) in out.items() if a is not S] == [0]


def test_edge_swap_and_cycle_both_wait() -> None:
    swap = make_view(LINE, [((2, 0), (3, 0)), ((3, 0), ONE)], [R, L])
    assert decide(swap) == {0: (S, "robot_ahead"), 1: (S, "robot_ahead")}
    grid = OccupancyGrid.empty(2, 2)  # a 4-cycle of robots rotating
    cyc = make_view(
        grid, [((0, 0), ONE), ((1, 0), ONE), ((1, 1), ONE), ((0, 1), ONE)], [R, U, L, D]
    )
    assert all(d == (S, "robot_ahead") for d in decide(cyc).values())


def test_following_is_conservative() -> None:
    """A robot behind a robot that moves away waits one step (cell occupied at t)."""
    view = make_view(LINE, [((1, 0), ONE), ((2, 0), ONE)], [R, R])
    assert decide(view) == {0: (S, "robot_ahead"), 1: (R, "move")}


# ---------------------------------------------------------------- statistics and validation


def test_episode_stats_and_reset() -> None:
    ctrl = BaselineController()
    ctrl.reset(2)
    ctrl.act(make_view(LINE, [((1, 0), ONE), ((3, 0), ONE)], [R, L]))
    ctrl.act(make_view(LINE, [((1, 0), ONE), ((4, 0), None)], [R, S]))
    s = ctrl.episode_stats()
    assert s["robot_steps"] == 4 and s["active_robot_steps"] == 3
    assert s["decisions"]["move"] == 2 and s["decisions"]["yield_tie"] == 1
    assert s["yields"] == 1 and s["wait_fraction"] == pytest.approx(1 / 3)
    assert set(s["decisions"]) == set(DECISIONS) and ctrl.last == {0: "move", 1: "no_plan"}
    ctrl.reset(2)
    assert (
        ctrl.episode_stats()["robot_steps"] == 0 and ctrl.episode_stats()["wait_fraction"] is None
    )


def test_validation() -> None:
    with pytest.raises(ConfigError):
        BaselineController(alpha=-1)
    with pytest.raises(ConfigError):
        BaselineController(sensing_radius=-1)


# ---------------------------------------------------------------- collision property


def _random_world(seed: int, constant_velocity: bool):  # type: ignore[no-untyped-def]
    """Random grid, robots with random tasks and *arbitrary* suggested actions (not just plan
    moves), obstacles with velocities, and the obstacles' next cells by the simulator's rule:
    constant velocity, or an arbitrary unit move."""
    rng = np.random.default_rng(seed)
    w, h = int(rng.integers(2, 8)), int(rng.integers(2, 8))
    grid = OccupancyGrid(rng.random((h, w)) < rng.uniform(0.0, 0.35))
    free = grid.free_cells()
    if len(free) < 2:
        return None
    rng.shuffle(free)  # type: ignore[arg-type]
    n = int(rng.integers(1, min(9, len(free)) + 1))
    k = int(rng.integers(0, min(5, len(free) - n) + 1))
    robots = []
    for c in free[:n]:
        task = None if rng.random() < 0.2 else (int(rng.integers(1, 4)), int(rng.integers(0, 30)))
        robots.append((c, task))
    moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    obstacles = [(c, moves[int(rng.integers(5))]) for c in free[n : n + k]]
    suggested = [Action(int(rng.integers(5))) for _ in robots]
    view = make_view(grid, robots, suggested, obstacles, t=30)
    now = [z for z, _ in obstacles]
    if constant_velocity:
        proposals = [add(z, v) for z, v in obstacles]
    else:
        proposals = [add(z, moves[int(rng.integers(5))]) for z, _ in obstacles]
    nxt = resolve_obstacle_moves(now, proposals, grid, {c for c, _ in robots})
    return grid, view, now, nxt


@settings(max_examples=400)
@given(st.integers(0, 10**7), st.sampled_from([2, 3, 5]))
def test_no_conflicts_when_obstacles_move_as_predicted(seed: int, r_s: int) -> None:
    """Guaranteed (`04` §5.3 notes): all robots on the baseline, sensing radius ≥ 2, every
    obstacle either stays or moves by its velocity → no conflict of any type."""
    world = _random_world(seed, constant_velocity=True)
    if world is None:
        return
    grid, view, now, nxt = world
    acts = BaselineController(sensing_radius=r_s).act(view)
    proposal = [acts[r.id] for r in view.robots]
    positions = [r.position for r in view.robots]
    assert detect_conflicts(view.t, positions, proposal, now, nxt, grid) == frozenset()


@settings(max_examples=400)
@given(st.integers(0, 10**7))
def test_only_unpredicted_obstacle_vertex_conflicts_otherwise(seed: int) -> None:
    """With arbitrary obstacle moves, the only possible conflicts are robot–obstacle vertex
    conflicts, and each involves an obstacle that did not move as predicted."""
    world = _random_world(seed, constant_velocity=False)
    if world is None:
        return
    grid, view, now, nxt = world
    acts = BaselineController().act(view)
    proposal = [acts[r.id] for r in view.robots]
    positions = [r.position for r in view.robots]
    events = detect_conflicts(view.t, positions, proposal, now, nxt, grid)
    vel = {o.id: o.velocity for o in view.obstacles}
    for e in events:
        assert e.kind is ConflictKind.RO_VERTEX
        k = next(int(x[1:]) for x in e.entities if x.startswith("o"))
        assert nxt[k] != add(now[k], vel[k])  # the obstacle turned


def test_property_is_not_vacuous() -> None:
    """On the same random worlds a plain plan follower does produce robot–robot conflicts."""
    rr = 0
    for seed in range(300):
        world = _random_world(seed, constant_velocity=True)
        if world is None:
            continue
        grid, view, now, nxt = world
        acts = SuggestionFollower().act(view)
        proposal = [acts[r.id] for r in view.robots]
        positions = [r.position for r in view.robots]
        events = detect_conflicts(view.t, positions, proposal, now, nxt, grid)
        rr += sum(1 for e in events if e.kind.is_robot_robot)
    assert rr > 0


def test_deterministic() -> None:
    world = _random_world(11, constant_velocity=True)
    assert world is not None
    view = world[1]
    assert BaselineController().decide(view) == BaselineController().decide(view)
