"""Weighted priority, hysteresis and aging (algorithms §4.4; model §4, §8.1; Issue #4 plan step 4).

What is tested is the *weighted* rule. No test asserts strict precedence or starvation freedom
(model §12 G11, G13; acceptance criterion 2).

Exactness: with α = 1/100 and s = 100 the priority term is ``P_j = round(s·κ·w̃_j) =
κ·w(π_j)·(100 + a_j)``, an integer for integer κ, and every distance is an integer. The expected
outcomes below are computed from the model's formulas with integers or ``Fraction`` only,
independently of the allocator.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.allocation import (
    AllocationProblem,
    BenefitConfig,
    RobotCandidate,
    TaskCandidate,
    build_problem,
)
from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.policy import prefer_current
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid

ALLOCATORS = [AuctionAllocator(), SequentialAllocator()]
ORACLE = DistanceOracle(OccupancyGrid.empty(60, 1))  # 60 × 1 corridor: D = |Δx|


def task(tid: int, pickup_x: int, level: int, release: int = 0, leg: int = 1) -> TaskCandidate:
    return TaskCandidate(tid, (pickup_x, 0), (pickup_x + leg, 0), level, release)


def problem(robots: list[RobotCandidate], tasks: list[TaskCandidate], t: int = 0, **cfg: object):  # type: ignore[no-untyped-def]
    return build_problem(robots, tasks, t, ORACLE.distance, BenefitConfig(**cfg))  # type: ignore[arg-type]


def priority_ints(p: AllocationProblem) -> dict[int, int]:
    """``P_j`` recovered from the integer matrix of a one-robot problem without a current task:
    ``A_0j = M + P_j − s·c_0j``."""
    s = p.config.scale
    return {
        tid: int(p.matrix[0, j]) - p.offset + s * int(p.cost[0, j]) for j, tid in enumerate(p.tasks)
    }


def exact_p(kappa: int, level: int, age: int) -> int:
    return kappa * 2 ** (level - 1) * (100 + age)  # s·κ·w·(1 + a/100) with s = 100


# ---------------------------------------------------------------- the weighted threshold


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=lambda a: a.name)
@settings(max_examples=150)
@given(
    kappa=st.integers(1, 20),
    lo=st.integers(1, 2),
    gap=st.integers(1, 2),
    age_hi=st.integers(0, 60),
    age_lo=st.integers(0, 60),
    x_lo=st.integers(1, 55),
    x_hi=st.integers(1, 55),
)
def test_threshold_both_sides(
    alloc: object, kappa: int, lo: int, gap: int, age_hi: int, age_lo: int, x_lo: int, x_hi: int
) -> None:
    """One robot, levels hi > lo: the high task is chosen iff κ(w̃_hi − w̃_lo) > c_hi − c_lo
    (model §8.1). At exact equality both are optimal; the result is deterministic."""
    hi = lo + gap
    t = 60
    robot = [RobotCandidate(0, (0, 0))]
    tasks = [task(1, x_lo, lo, t - age_lo), task(2, x_hi, hi, t - age_hi)]
    p = problem(robot, tasks, t, kappa=float(kappa))
    # exact, in units of 1/s: s·κ·(w̃_hi − w̃_lo) vs s·(c_hi − c_lo)
    lhs = exact_p(kappa, hi, age_hi) - exact_p(kappa, lo, age_lo)
    rhs = 100 * ((x_hi + 1) - (x_lo + 1))
    res = alloc.allocate(p)  # type: ignore[attr-defined]
    assert len(res.pairs) == 1
    chosen = res.pairs[0][1]
    if lhs > rhs:
        assert chosen == 2
    elif lhs < rhs:
        assert chosen == 1
    else:  # exact tie: both optimal; deterministic
        assert chosen in (1, 2) and alloc.allocate(p).pairs == res.pairs  # type: ignore[attr-defined]


def test_threshold_explicit_instance() -> None:
    """κ = 10, levels 1 vs 2, age 0: the level-2 task buys a detour of < 10 cells.
    Low task at x = 1 (c = 2); high task at x = d (c = d + 1); high iff 10 > d − 1."""
    robot = [RobotCandidate(0, (0, 0))]
    for d in (2, 5, 10, 12, 20):
        p = problem(robot, [task(1, 1, 1), task(2, d, 2)])
        expect = 2 if d - 1 < 10 else 1
        for alloc in ALLOCATORS:
            assert alloc.allocate(p).pairs == ((0, expect),)
    tie = problem(robot, [task(1, 1, 1), task(2, 11, 2)])  # 10 == 11 − 1
    assert len({int(tie.matrix[0, 0]), int(tie.matrix[0, 1])}) == 1  # an exact tie


def test_not_strict_priority() -> None:
    """A much closer low-priority task is preferred (the intended trade-off, model §8.1)."""
    p = problem([RobotCandidate(0, (0, 0))], [task(1, 1, 1), task(2, 40, 3)])
    for alloc in ALLOCATORS:
        assert alloc.allocate(p).pairs == ((0, 1),)


# ---------------------------------------------------------------- hysteresis


@pytest.mark.parametrize("alloc", ALLOCATORS, ids=lambda a: a.name)
@pytest.mark.parametrize("h", [0, 2, 5])
def test_hysteresis_switch_iff_gain_exceeds_h(alloc: object, h: int) -> None:
    """Robot at x = 20 holds task A (pickup x = 20 + dA). Task B (same level and age) at
    x = 20 − dB. Gain of switching = c_A − c_B = dA − dB. No switch iff gain ≤ h (model §8:
    "unless another assignment is better by more than h"); the equality case relies on
    :func:`prefer_current`."""
    for gain in range(-2, h + 4):
        d_a, d_b = 10, 10 - gain
        robot = [RobotCandidate(0, (20, 0), task=1)]
        both = [task(1, 20 + d_a, 1), task(2, 20 - d_b, 1, leg=-1)]
        for tasks in (both, both[::-1]):  # the result must not depend on the task order
            p = problem(robot, tasks, hysteresis=h)
            res = alloc.allocate(prefer_current(p))  # type: ignore[attr-defined]
            assert res.pairs == ((0, 2 if gain > h else 1),), (h, gain)
            if gain == h:  # without the tie-break both matchings are optimal
                assert int(p.matrix[0, 0]) == int(p.matrix[0, 1])


def test_prefer_current_is_exact_refinement() -> None:
    """On random benefit problems with current tasks: the refined matrix's optimum is optimal
    for the original integer matrix (Hungarian oracle) and keeps at least as many current pairs
    as any other optimum (here: at least the Hungarian solution's)."""
    from scipy.optimize import linear_sum_assignment

    from tests.unit.allocation.test_benefits import random_instance

    checked = 0
    for seed in range(300):
        base = random_instance(seed, max_n=6, max_m=6)
        if base.size == 0 or not base.tasks or not base.robots:
            continue
        # give every second robot a feasible current task (distinct tasks)
        cur: dict[int, int] = {}
        for i, rid in enumerate(base.robots):
            js = [j for j in range(len(base.tasks)) if base.feasible[i, j]]
            if js and i % 2 == 0 and base.tasks[js[0]] not in cur.values():
                cur[rid] = base.tasks[js[0]]
        p = prefer_current(_with_current(base, cur))
        r, c = linear_sum_assignment(base.matrix, maximize=True)
        opt = int(base.matrix[r, c].sum())
        res = AuctionAllocator().allocate(p)
        rows = {rid: i for i, rid in enumerate(base.robots)}
        cols = {tid: j for j, tid in enumerate(base.tasks)}
        value = sum(int(base.matrix[rows[i], cols[j]]) for i, j in res.pairs)
        assert value == opt
        kept = sum(1 for i, j in res.pairs if cur.get(i) == j)
        hung_pairs = base.pairs(list(c[r.argsort()]))
        assert kept >= sum(1 for i, j in hung_pairs if cur.get(i) == j)
        checked += 1
    assert checked > 100


def test_prefer_current_overflow_guard_and_no_op() -> None:
    import numpy as np

    p = problem([RobotCandidate(0, (0, 0), task=1)], [task(1, 5, 1), task(2, 9, 1)])
    assert prefer_current(_with_current(p, {})).matrix is p.matrix  # nothing to prefer
    huge = _with_current(p, {0: 1})
    from dataclasses import replace

    huge = replace(huge, matrix=np.full((2, 2), 2**61, dtype=np.int64))
    with pytest.raises(OverflowError):
        prefer_current(huge)


def _with_current(p: AllocationProblem, current: dict[int, int]) -> AllocationProblem:
    from dataclasses import replace

    return replace(p, current=current)


# ---------------------------------------------------------------- aging P1–P3 (priority term)


@settings(max_examples=200)
@given(
    level=st.integers(1, 3),
    r1=st.integers(0, 200),
    dr=st.integers(0, 200),
    extra=st.integers(0, 400),
    kappa=st.integers(1, 20),
)
def test_p1_fifo_within_level(level: int, r1: int, dr: int, extra: int, kappa: int) -> None:
    r2 = r1 + dr
    t = r2 + extra
    p = problem(
        [RobotCandidate(0, (0, 0))], [task(1, 5, level, r1), task(2, 5, level, r2)], t, kappa=kappa
    )
    pj = priority_ints(p)
    assert pj[1] == exact_p(kappa, level, t - r1) and pj[2] == exact_p(kappa, level, t - r2)
    assert pj[1] >= pj[2]


@pytest.mark.parametrize("new_level", [1, 2, 3])
def test_p2_outranks_new_tasks_after_a_star(new_level: int) -> None:
    """A* = (w_max/w_min − 1)/α = (4 − 1)·100 = 300 for P = 3, α = 1/100. A level-1 task of age
    ≥ 300 has priority term ≥ that of any task released now; at age 299 a new level-3 task
    still outranks it (the threshold is tight)."""
    a_star = int((Fraction(4, 1) - 1) / Fraction(1, 100))
    assert a_star == 300
    for age in (a_star, a_star + 1, a_star + 57):
        t = age
        p = problem([RobotCandidate(0, (0, 0))], [task(1, 5, 1, 0), task(2, 5, new_level, t)], t)
        pj = priority_ints(p)
        assert pj[1] >= pj[2]
    t = a_star - 1
    p = problem([RobotCandidate(0, (0, 0))], [task(1, 5, 1, 0), task(2, 5, 3, t)], t)
    assert priority_ints(p)[1] < priority_ints(p)[2]


@settings(max_examples=200)
@given(
    lo=st.integers(1, 2),
    gap=st.integers(1, 2),
    rho=st.integers(0, 50),
    delta=st.integers(0, 400),
    extra=st.integers(0, 600),
)
def test_p3_exact_condition_and_temporary_advantage(
    lo: int, gap: int, rho: int, delta: int, extra: int
) -> None:
    """π_j < π_l, ρ_l = ρ_j + Δ: j outranks l iff α·w_l·Δ ≥ (w_l − w_j)(1 + α·a_j(t))
    (model §4, P3), and any advantage of j is temporary."""
    hi = lo + gap
    w_j, w_l = 2 ** (lo - 1), 2 ** (hi - 1)
    alpha = Fraction(1, 100)
    t = rho + delta + extra
    p = problem([RobotCandidate(0, (0, 0))], [task(1, 5, lo, rho), task(2, 5, hi, rho + delta)], t)
    pj = priority_ints(p)
    a_j = t - rho
    condition = alpha * w_l * delta >= (w_l - w_j) * (1 + alpha * a_j)
    assert (pj[1] >= pj[2]) == condition
    # temporary: beyond t* the higher level wins again (RHS grows with t)
    t_late = t + 100 * delta + 1
    late = problem(
        [RobotCandidate(0, (0, 0))], [task(1, 5, lo, rho), task(2, 5, hi, rho + delta)], t_late
    )
    lp = priority_ints(late)
    assert lp[1] < lp[2]
