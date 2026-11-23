"""Unit and property tests of the Execution Supervisor (Issue 6A Phase 3): S, O, R, D, safety,
determinism and fairness on synthetic views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.control.supervisor import (
    ExecutionSupervisor,
    SupervisorParams,
    filtered_suggestion,
    is_oscillating,
    parse_mode,
    right_of_way,
)
from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell, add
from robocoord.sim.conflicts import arbitrate, detect_conflicts, physical_violations
from robocoord.sim.view import MapView, ObstacleView, RobotView, TaskView, WorldView, action_mask
from robocoord.world.entities import RobotMode, TaskStatus

S, U, D, L, R = Action.STAY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT
SMALL = SupervisorParams(
    stall_window=3,
    hold=4,
    progress_cells=2,
    stay_limit=2,
    osc_window=4,
    osc_moves=3,
    osc_cells=2,
    fairness_losses=2,
)


def view(
    t: int,
    robots: Sequence[tuple[Cell, Cell | None]],
    suggested: Sequence[Action] | None,
    gd: Sequence[int | None] | None,
    *,
    grid: OccupancyGrid | None = None,
    priorities: Sequence[int] | None = None,
    obstacles: Sequence[tuple[Cell, Cell]] = (),
) -> WorldView:
    """Robot ``i`` at ``robots[i][0]``, goal ``robots[i][1]`` (``None`` = idle), task ``i``."""
    g = grid or OccupancyGrid(np.zeros((8, 8), dtype=bool))
    rv, tv = [], []
    for i, (pos, goal) in enumerate(robots):
        busy = goal is not None
        rv.append(
            RobotView(
                i, pos, RobotMode.TO_PICKUP if busy else RobotMode.IDLE, i if busy else None, goal
            )
        )
        if goal is not None:
            pr = priorities[i] if priorities else 1
            tv.append(TaskView(i, goal, goal, pr, 0, TaskStatus.ASSIGNED, i, False))
    return WorldView(
        t=t,
        max_steps=400,
        map=MapView(g),
        robots=tuple(rv),
        obstacles=tuple(ObstacleView(k, p, v) for k, (p, v) in enumerate(obstacles)),
        tasks=tuple(tv),
        masks=tuple(action_mask(g, r.position) for r in rv),
        suggested=None if suggested is None else tuple(suggested),
        reference_paths=None,
        goal_distance=None if gd is None else tuple(gd),
    )


def acts(
    sup: ExecutionSupervisor,
    v: WorldView,
    proposal: Sequence[Action],
    detour: frozenset[int] = frozenset(),
) -> list[Action]:
    out, _ = sup.apply(v, list(proposal), detour)
    return out


# --------------------------------------------------------------------------- configuration


def test_parse_mode_and_enabled() -> None:
    assert parse_mode("off") == frozenset() and not ExecutionSupervisor("off").enabled
    assert parse_mode("full") == {"R", "S", "O"}
    assert parse_mode("R+S") == {"R", "S"} and parse_mode("O") == {"O"}
    for bad in ("", "X", "R+R", "full+R", "r"):
        with pytest.raises(ConfigError):
            parse_mode(bad)


@pytest.mark.parametrize(
    "kw",
    [{"stall_window": 0}, {"hold": 0}, {"osc_moves": 9}, {"alpha": -1.0}, {"fairness_losses": 0}],
)
def test_params_validation(kw: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        SupervisorParams(**kw)


def test_s_and_o_need_goal_distance_but_r_does_not() -> None:
    v = view(0, [((1, 1), (5, 1))], [R], None)
    with pytest.raises(ConfigError):
        ExecutionSupervisor("S").apply(v, [S])
    assert acts(ExecutionSupervisor("R"), v, [S]) == [S]


# --------------------------------------------------------------------------- pure helpers


def test_filtered_suggestion_reasons() -> None:
    g = OccupancyGrid.from_rows(["......", "......", "......", "..#..."])  # wall at (2, 0)
    base = [((1, 1), (5, 1)), ((3, 3), (5, 3))]
    v = view(0, base, [R, S], [4, 2], grid=g)
    cells = [r.position for r in v.robots]
    assert filtered_suggestion(v, 0, cells, {}, 3) == (R, "move")
    assert filtered_suggestion(v, 1, cells, {}, 3) == (S, "no_plan")
    v2 = view(0, [((1, 1), (5, 1)), ((2, 1), (5, 3))], [R, S], [4, 2], grid=g)
    assert filtered_suggestion(v2, 0, [r.position for r in v2.robots], {}, 3)[1] == "robot_ahead"
    vm = view(0, [((1, 0), (5, 1))], [R], [4], grid=g)  # (2, 0) is the wall '#'
    assert filtered_suggestion(vm, 0, [(1, 0)], {}, 3)[1] == "masked"
    assert filtered_suggestion(v, 0, cells, {(2, 1): (0, 0)}, 3)[1] == "obstacle_ahead"
    assert filtered_suggestion(v, 0, cells, {(2, 2): (0, -1)}, 3)[1] == "predicted_obstacle"
    assert filtered_suggestion(view(0, base, None, [4, 2]), 0, cells, {}, 3)[1] == "no_plan"


def test_is_oscillating() -> None:
    a, b, c = (1, 1), (1, 2), (2, 2)
    assert is_oscillating([a, b, a, b, a], 4, 2, 3)
    assert not is_oscillating([a, b, a, b], 4, 2, 3)  # too short
    assert not is_oscillating([a, b, c, b, a], 4, 2, 3)  # three cells
    assert not is_oscillating([a, a, a, b, b], 4, 2, 3)  # one move
    assert is_oscillating([c, c, a, b, a, b, a], 4, 2, 3)  # only the last window counts


def test_right_of_way_vertex_contest_keeps_the_highest_key() -> None:
    pos = [(1, 2), (3, 2)]
    out, why, left = right_of_way(pos, [R, L], {0: (False, 1.0, 0, 0), 1: (False, 2.0, 0, -1)})
    assert out == [S, L] and why == {0: "R_loss"} and left == {"occupied": 0, "swap": 0}
    three = right_of_way([(1, 2), (3, 2), (2, 3)], [R, L, D], {0: (0,), 1: (2,), 2: (1,)})
    assert three[0] == [S, L, S] and three[1] == {0: "R_loss", 2: "R_loss"}


def test_motion_neutral_conflicts_are_left_to_the_arbiter() -> None:
    # a move into a staying robot's cell, and a head-on swap: unchanged, only counted
    out, why, left = right_of_way([(1, 2), (2, 2)], [R, S], {0: (0,), 1: (1,)})
    assert out == [R, S] and why == {} and left == {"occupied": 1, "swap": 0}
    out, why, left = right_of_way([(1, 2), (2, 2)], [R, L], {0: (0,), 1: (1,)})
    assert out == [R, L] and why == {} and left == {"occupied": 0, "swap": 1}
    # a follower into a contest loser's cell is not changed by R; the arbiter cascades it
    out, why, _ = right_of_way([(0, 2), (1, 2), (3, 2)], [R, R, L], {0: (0,), 1: (0,), 2: (1,)})
    assert out == [R, S, L] and why == {1: "R_loss"}
    pos = [(0, 2), (1, 2), (3, 2)]
    res = arbitrate(
        pos, out, detect_conflicts(0, pos, out, [], [], OccupancyGrid(np.zeros((6, 6), dtype=bool)))
    )
    assert list(res.positions) == [(0, 2), (1, 2), (2, 2)]


# --------------------------------------------------------------------------- S


def test_stall_breaker_triggers_after_w_counted_steps_and_ends_on_progress() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    outs = [acts(sup, view(t, [((1, 1), (6, 1))], [R], [5]), [S]) for t in range(3)]
    assert outs == [[S], [S], [R]]  # counter 1, 2, 3 = W -> recovery
    assert sup.recoveries_started["S"] == 1
    assert acts(sup, view(3, [((2, 1), (6, 1))], [R], [4]), [S]) == [R]
    assert acts(sup, view(4, [((3, 1), (6, 1))], [R], [3]), [S]) == [S]  # 2 cells: ended
    assert sup.recoveries_ended["progress"] == 1


def test_policy_that_moves_is_never_overridden_by_s() -> None:
    sup = ExecutionSupervisor("S+O", SMALL)
    for t in range(8):
        pos = (1 + t, 1)
        assert acts(sup, view(t, [(pos, (7, 1))], [R], [6 - t]), [R]) == [R]
    assert sup.recoveries_started == {} or sum(sup.recoveries_started.values()) == 0


@pytest.mark.parametrize("case", ["robot_ahead", "no_plan", "predicted"])
def test_legitimate_waiting_is_excused(case: str) -> None:
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(10):
        if case == "robot_ahead":
            v = view(t, [((1, 1), (6, 1)), ((2, 1), None)], [R, S], [5, None])
        elif case == "no_plan":
            v = view(t, [((1, 1), (6, 1))], [S], [5])
        else:
            v = view(t, [((1, 1), (6, 1))], [R], [5], obstacles=[((2, 2), (0, -1))])
        out = acts(sup, v, [S] * len(v.robots))
        assert out[0] is S
    assert sum(sup.recoveries_started.values()) == 0


def test_recovery_ends_by_hold_and_by_stay_limit() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(3):
        acts(sup, view(t, [((1, 1), (6, 1))], [R], [5]), [S])
    # moving without progress (D̄ unchanged) for the whole hold of 4 steps
    for t, x in zip(range(3, 7), (2, 3, 4, 5), strict=True):
        acts(sup, view(t, [((x, 1), (6, 1))], [R], [5]), [S])
    acts(sup, view(7, [((6, 2), (6, 1))], [R], [5]), [S])
    assert sup.recoveries_ended["hold"] == 1
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(3):
        acts(sup, view(t, [((1, 1), (6, 1))], [R], [5]), [S])
    # the suggested cell becomes occupied: the recovery stays, then ends after stay_limit = 2
    for t in range(3, 6):
        acts(sup, view(t, [((1, 1), (6, 1)), ((2, 1), None)], [R, S], [5, None]), [S, S])
    assert sup.recoveries_ended["stay_limit"] == 1


def test_leg_change_resets_and_ends_recovery() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(3):
        acts(sup, view(t, [((1, 1), (6, 1))], [R], [5]), [S])
    assert sup._robots[0].recovery is not None
    acts(sup, view(3, [((1, 1), (1, 6))], [U], [5]), [S])  # new goal
    assert sup.recoveries_ended["leg"] == 1 and sup._robots[0].counter == 1


# --------------------------------------------------------------------------- O and the cycle guard


def test_oscillation_detected_and_recovered() -> None:
    sup = ExecutionSupervisor("O", SMALL)
    a, b = (1, 1), (1, 2)
    outs = []
    for t, p in enumerate([a, b, a, b, a]):
        outs.append(acts(sup, view(t, [(p, (6, 1))], [R], [5]), [U if p == a else D]))
    assert outs[-1] == [R] and sup.recoveries_started["O"] == 1
    assert outs[:-1] == [[U], [D], [U], [D]]


def test_recovery_cycle_guard_ends_a_recovery_that_revisits_a_cell() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(3):
        acts(sup, view(t, [((2, 1), (6, 1))], [R], [5]), [S])
    acts(sup, view(3, [((3, 1), (6, 1))], [L], [5]), [S])  # suggestion flips (e.g. a replan)
    acts(sup, view(4, [((2, 1), (6, 1))], [R], [5]), [S])  # back on a visited cell
    assert sup.recoveries_ended["cycle"] == 1


# --------------------------------------------------------------------------- D (detour membership)


def test_detour_members_are_excused_and_d_recovery_d2b() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    member = frozenset({0})
    outs = [acts(sup, view(t, [((1, 1), (6, 1))], [U], [5]), [S], member) for t in range(3)]
    # never an S recovery for a member; after W = 3 declined suggestions a D recovery starts
    assert outs == [[S], [S], [U]] and sup.recoveries_started == {"D": 1} | {}
    assert sup._robots[0].counter == 0
    acts(sup, view(3, [((1, 2), (6, 1))], [R], [6]), [S], frozenset())  # detour over
    assert sup.recoveries_ended["detour_end"] == 1


def test_member_that_follows_its_detour_is_left_alone() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(6):
        assert acts(sup, view(t, [((1, 1 + t), (6, 1))], [U], [5 + t]), [U], frozenset({0})) == [U]
    assert sum(sup.recoveries_started.values()) == 0


def test_s_recovery_ends_when_the_robot_joins_a_detour() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    for t in range(3):
        acts(sup, view(t, [((1, 1), (6, 1))], [R], [5]), [S])
    acts(sup, view(3, [((1, 1), (6, 1))], [R], [5]), [S], frozenset({0}))
    assert sup.recoveries_ended["detour"] == 1


# --------------------------------------------------------------------------- R fairness


def test_r_fairness_bounds_consecutive_losses() -> None:
    sup = ExecutionSupervisor("R", SMALL)  # L_max = 2
    winners = []
    for t in range(9):
        v = view(t, [((1, 2), (6, 2)), ((3, 2), (0, 2))], [R, L], [5, 3], priorities=[1, 3])
        out = acts(sup, v, [R, L])
        winners.append(0 if out == [R, S] else 1)
    assert winners == [1, 1, 0, 1, 1, 0, 1, 1, 0]  # the low-priority robot wins every 3rd step
    assert sup.max_consecutive_losses == 2


def test_r_loss_excuses_the_next_stall_step() -> None:
    """Robot 0 loses most contests (lower priority): the step after each loss is excused, so its
    stall counter only advances after its wins. (Robot 1, which wins but never progresses in this
    static view, is a genuine stall and may trigger S; it is not what this test checks.)"""
    sup = ExecutionSupervisor("R+S", SMALL)
    for t in range(6):
        v = view(t, [((1, 2), (6, 2)), ((3, 2), (0, 2))], [R, L], [5, 3], priorities=[1, 3])
        acts(sup, v, [R, L])
        assert sup._robots[0].recovery is None
    assert sup._robots[0].counter <= 2


# --------------------------------------------------------------------------- events and stats


def test_events_stats_and_reset() -> None:
    sup = ExecutionSupervisor("full", SMALL)
    v = view(0, [((1, 2), (6, 2)), ((3, 2), (0, 2))], [R, L], [5, 3], priorities=[1, 3])
    out, ev = sup.apply(v, [R, L])
    assert out == [S, L] and len(ev) == 1 and ev[0].to_json() == [0, "R_loss", "R", "S"]
    s = sup.stats()
    assert s["modified_robot_steps"] == 1 and s["changes"]["R_loss"] == 1
    assert s["policy_stays_nonidle"] == 0 and s["executed_stays_nonidle"] == 1
    sup.reset()
    assert sup.stats()["modified_robot_steps"] == 0 and sup._robots == {}


# --------------------------------------------------------------------------- properties

cells = st.tuples(st.integers(0, 5), st.integers(0, 5))
action = st.sampled_from(list(Action))


@settings(max_examples=300, deadline=None)
@given(st.lists(cells, min_size=1, max_size=8, unique=True), st.data())
def test_property_right_of_way_is_conflict_free_idempotent_and_only_stops(
    pos: list[Cell], data: Any
) -> None:
    acts_in = data.draw(st.lists(action, min_size=len(pos), max_size=len(pos)))
    keys = {
        i: (data.draw(st.booleans()), data.draw(st.integers(0, 3)), 0, -i) for i in range(len(pos))
    }
    out, why, _ = right_of_way(pos, acts_in, keys)
    for i, (a, b) in enumerate(zip(acts_in, out, strict=True)):
        assert b == a or (b is S and why.get(i) == "R_loss")
    targets = [add(p, a.delta) for p, a in zip(pos, out, strict=True)]
    staying = {pos[i] for i, a in enumerate(out) if a is S}
    movers = [i for i, a in enumerate(out) if a is not S]
    free_targets = [targets[i] for i in movers if targets[i] not in staying]
    assert len(free_targets) == len(set(free_targets))  # no shared free target remains
    assert right_of_way(pos, out, keys)[0] == out  # a fixed point
    for i, j in ((i, j) for i in range(len(pos)) for j in range(len(pos)) if i != j):
        # every loser lost against a higher key contesting the same cell
        if (
            why.get(i) == "R_loss"
            and targets[j] == add(pos[i], acts_in[i].delta)
            and out[j] is not S
        ):
            assert keys[j] > keys[i]


@settings(max_examples=200, deadline=None)
@given(st.lists(cells, min_size=1, max_size=6, unique=True), st.data())
def test_property_supervised_output_alphabet_masks_and_arbiter_safety(
    pos: list[Cell], data: Any
) -> None:
    g = OccupancyGrid(np.zeros((6, 6), dtype=bool))
    n = len(pos)
    goals = [data.draw(st.one_of(st.none(), cells)) for _ in range(n)]
    masks = [action_mask(g, p) for p in pos]
    sug = [
        data.draw(st.sampled_from([a for a in Action if masks[i][a]])) for i, _ in enumerate(pos)
    ]
    prop = [
        data.draw(st.sampled_from([a for a in Action if masks[i][a]])) for i, _ in enumerate(pos)
    ]
    sup = ExecutionSupervisor(
        "full", SupervisorParams(stall_window=1, hold=2, osc_window=2, osc_moves=1)
    )
    for t in range(3):  # a few steps, so recoveries start
        gd = [None if goals[i] is None else data.draw(st.integers(0, 9)) for i in range(n)]
        v = view(t, list(zip(pos, goals, strict=True)), sug, gd, grid=g)
        out, _ = sup.apply(v, prop)
        for i in range(n):
            assert out[i] in (prop[i], S, sug[i])
            assert masks[i][out[i]]
        ev = detect_conflicts(t, pos, out, [], [], g)
        res = arbitrate(pos, out, ev)
        assert physical_violations(pos, res.positions, [], g) == []


@settings(max_examples=100, deadline=None)
@given(st.lists(cells, min_size=2, max_size=5, unique=True), st.data())
def test_property_determinism(pos: list[Cell], data: Any) -> None:
    n = len(pos)
    steps = []
    for t in range(5):
        sug = data.draw(st.lists(action, min_size=n, max_size=n))
        prop = data.draw(st.lists(action, min_size=n, max_size=n))
        gd = data.draw(st.lists(st.integers(0, 9), min_size=n, max_size=n))
        steps.append((view(t, [(p, (5, 5)) for p in pos], sug, gd), prop))
    a, b = ExecutionSupervisor("full", SMALL), ExecutionSupervisor("full", SMALL)
    assert [a.apply(v, p)[0] for v, p in steps] == [b.apply(v, p)[0] for v, p in steps]
    assert a.stats() == b.stats()


# --------------------------------------------------------------------------- optional escalation


def _stall_then_progress(sup: ExecutionSupervisor) -> list[list[Action]]:
    """A frozen policy: stall until a recovery, progress 2 cells under it, then freeze again."""
    out, x, d = [], 1, 6
    for t in range(40):
        a = acts(sup, view(t, [((x, 1), (7, 1))], [R], [d]), [S])
        out.append(a)
        if a == [R]:
            x, d = x + 1, d - 1
        if d == 0:
            break
    return out


def test_default_has_no_escalation() -> None:
    sup = ExecutionSupervisor("S", SMALL)
    _stall_then_progress(sup)
    assert SupervisorParams().escalate is False
    assert sup.recoveries_ended["progress"] >= 2  # every recovery ends after 2 cells


def test_escalation_retriggers_sooner_and_holds() -> None:
    plain = ExecutionSupervisor("S", SMALL)
    esc = ExecutionSupervisor(
        "S",
        SupervisorParams(
            **{
                **{k: getattr(SMALL, k) for k in SMALL.__slots__},
                "escalate": True,
                "repeat_window": 1,
            }
        ),
    )
    n_plain = len(_stall_then_progress(plain))
    n_esc = len(_stall_then_progress(esc))
    assert n_esc < n_plain
    assert esc.recoveries_ended["progress"] == 1  # only the first recovery of the leg


def test_monitor_statistics_per_robot() -> None:
    """Stage 3a monitoring: per-robot modifications, eligible steps, recoveries and R losses, the
    largest per-robot override share, recoveries per leg and the longest no-progress time."""
    sup = ExecutionSupervisor("full", SMALL)
    _stall_then_progress(sup)
    mon = sup.stats()["monitor"]
    assert mon["eligible_by_robot"] == [sup.eligible_robot_steps]
    assert mon["modified_by_robot"] == [sup.modified_robot_steps] and sup.modified_robot_steps
    assert mon["recoveries_by_robot"] == [sum(sup.recoveries_started.values())]
    assert mon["max_recoveries_in_a_leg"] == sum(sup.recoveries_started.values()) >= 2
    assert mon["max_robot_modified_share"] == sup.modified_robot_steps / sup.eligible_robot_steps
    assert mon["max_steps_without_progress"] >= SMALL.stall_window
    assert mon["r_losses_by_robot"] == [0]
    r = ExecutionSupervisor("R", SMALL)
    v = view(0, [((1, 2), (6, 2)), ((3, 2), (0, 2))], [R, L], [5, 3], priorities=[1, 3])
    acts(r, v, [R, L])
    assert r.stats()["monitor"]["r_losses_by_robot"] == [1, 0]
    assert ExecutionSupervisor("full").stats()["monitor"]["max_robot_modified_share"] is None


def test_stage3a_frozen_parameters_are_pinned() -> None:
    """The Stage 3a freeze (protocol §4): any change here invalidates the pre-registration."""
    from robocoord.control.supervisor import FROZEN_PARAMS, frozen_params

    design = SupervisorParams()
    common = {
        "hold": 8,
        "progress_cells": 2,
        "stay_limit": 3,
        "fairness_losses": 3,
        "alpha": 0.01,
        "sensing_radius": 3,
        "escalate": False,
        "repeat_window": 4,
    }
    expected = {
        "astar": {**common, "stall_window": 24, "osc_window": 14, "osc_cells": 3, "osc_moves": 7},
        "rrtstar": {**common, "stall_window": 32, "osc_window": 8, "osc_cells": 3, "osc_moves": 4},
    }
    assert set(FROZEN_PARAMS) == set(expected)
    for planner, values in expected.items():
        p = frozen_params(planner)
        assert {k: getattr(p, k) for k in p.__slots__} == values
        assert p.escalate is False  # escalation stays disabled
    assert design.stall_window == 12 and design.osc_window == 8  # design values unchanged
    with pytest.raises(ConfigError):
        frozen_params("dijkstra")
