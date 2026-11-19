"""Auction (algorithms §4.1 tests 1–5; Issue #4 plan step 2; acceptance criterion 1).

Oracle: ``scipy.optimize.linear_sum_assignment`` (Hungarian, ADR-5), independent of the auction.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import linear_sum_assignment

from robocoord.allocation import BenefitConfig
from robocoord.allocation.auction import AuctionAllocator, epsilon_cs_violation, solve
from robocoord.core.errors import ConfigError
from tests.unit.allocation.test_benefits import random_instance


def hungarian_value(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    r, c = linear_sum_assignment(a, maximize=True)
    return int(a[r, c].sum())


def random_matrix(seed: int, max_k: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, max_k + 1))
    c = int(rng.choice([1, 3, 10, 1000, 10**6]))  # small C → many ties
    a = rng.integers(0, c + 1, size=(k, k))
    if rng.random() < 0.3:  # zero blocks, like padding and infeasible pairs
        a[rng.random((k, k)) < 0.4] = 0
    return a.astype(np.int64)


def is_permutation(assignment: tuple[int, ...], k: int) -> bool:
    return sorted(assignment) == list(range(k))


# ---------------------------------------------------------------- ε-scaling: exact optimum


@settings(max_examples=200)
@given(st.integers(0, 10**7))
def test_scaling_auction_equals_hungarian(seed: int) -> None:
    a = random_matrix(seed)
    sol, trace = solve(a, scaling=True)
    k = a.shape[0]
    assert sol.assignment is not None and is_permutation(sol.assignment, k)
    assert sol.value == hungarian_value(a)
    assert sol.stats.final_epsilon < 1 / k  # documented convergence condition (ε = 1/(k+1))
    # ε-CS holds at termination, in the auction's (k+1)-scaled integer units
    assert (
        epsilon_cs_violation(a * trace.unit_scale, sol.assignment, trace.prices, trace.epsilon) <= 0
    )


@settings(max_examples=200)
@given(st.integers(0, 10**7))
def test_auction_on_benefit_problems_has_max_cardinality(seed: int) -> None:
    """Rectangular problems with infeasible pairs, through the real benefit construction: the
    auction reaches the Hungarian optimum of the *integer* problem exactly, with the same
    (maximum) cardinality and no dummy or infeasible pair (acceptance criterion 1).

    In original units two integer-optimal matchings can differ, because only the priority term
    is rounded: by at most q/s for q pairs (each pair's error is ≤ 1/(2s), model §8). This
    happened in development (92.0 vs 92.1 with s = 1), so the test checks exactly that bound."""
    p = random_instance(seed, max_n=12, max_m=12)
    if p.size == 0:
        return
    res = AuctionAllocator().allocate(p)
    r, c = linear_sum_assignment(p.matrix, maximize=True)
    oracle_pairs = p.pairs(list(c[np.argsort(r)]))
    assert len(res.pairs) == len(oracle_pairs)  # maximum cardinality (G10)

    def int_total(pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> int:
        rr = {rid: i for i, rid in enumerate(p.robots)}
        cc = {tid: j for j, tid in enumerate(p.tasks)}
        return int(sum(p.matrix[rr[i], cc[j]] for i, j in pairs))

    assert int_total(res.pairs) == int_total(oracle_pairs) == int(p.matrix[r, c].sum())
    q, s = len(res.pairs), p.config.scale
    assert abs(res.value - p.matching_value(oracle_pairs)) <= q / s + 1e-9
    rows = {rid: i for i, rid in enumerate(p.robots)}
    cols = {tid: j for j, tid in enumerate(p.tasks)}
    assert all(p.feasible[rows[i], cols[j]] for i, j in res.pairs)
    assert len({i for i, _ in res.pairs}) == len({j for _, j in res.pairs}) == len(res.pairs)


# ---------------------------------------------------------------- fixed ε: additive and bid bounds


@settings(max_examples=200)
@given(st.integers(0, 10**7), st.sampled_from([1, 2, 7, 50, 1000]))
def test_fixed_epsilon_within_k_eps_and_bid_bound(seed: int, eps: int) -> None:
    """V ≥ OPT − k·ε (Guaranteed) and, for a non-negative matrix with largest entry C, at most
    k·(⌊C/ε⌋ + 1) + 1 bids from zero prices (Guaranteed; derivation in `04` §4.1 notes)."""
    a = random_matrix(seed)
    k = a.shape[0]
    sol, trace = solve(a, scaling=False, epsilon=eps)
    assert sol.assignment is not None and is_permutation(sol.assignment, k)
    opt = hungarian_value(a)
    assert opt - k * eps <= sol.value <= opt
    assert epsilon_cs_violation(a, sol.assignment, trace.prices, eps) <= 0
    c = int(a.max())
    assert sol.stats.bids <= k * (c // eps + 1) + 1
    assert sol.stats.phases == 1


def test_fixed_epsilon_gap_can_be_positive() -> None:
    """The kε bound is not vacuous: a fixed ε can return a strictly suboptimal assignment."""
    found = False
    for seed in range(2000):
        a = np.random.default_rng(seed).integers(0, 6, size=(4, 4))
        sol, _ = solve(a, scaling=False, epsilon=3)
        gap = hungarian_value(a) - sol.value
        assert 0 <= gap <= 4 * 3
        found = found or gap > 0
    assert found


# ---------------------------------------------------------------- edge cases, determinism, guards


def test_edge_cases_and_determinism() -> None:
    empty, _ = solve(np.zeros((0, 0), dtype=np.int64))
    assert empty.assignment == () and empty.value == 0
    one, _ = solve(np.array([[7]]))
    assert one.assignment == (0,) and one.value == 7
    ties, _ = solve(np.full((5, 5), 3))
    assert ties.assignment is not None and is_permutation(ties.assignment, 5)
    neg, _ = solve(np.array([[-5, -1], [-2, -9]]))
    assert neg.value == hungarian_value(np.array([[-5, -1], [-2, -9]])) == -3
    a = random_matrix(4242, max_k=30)
    s1, _ = solve(a)
    s2, _ = solve(a)
    assert s1.assignment == s2.assignment and s1.stats.bids == s2.stats.bids
    with pytest.raises(ValueError):
        solve(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(ConfigError):
        solve(a, theta=1)
    with pytest.raises(ConfigError):
        solve(a, scaling=False, epsilon=0)
    with pytest.raises(OverflowError):
        solve(np.full((3, 3), 2**60, dtype=np.int64))
    with pytest.raises(ConfigError):
        AuctionAllocator(mode="magic")


def test_timeout_returns_no_assignment() -> None:
    # the caps are checked every 256 bids; this instance needs more than 256 bids
    a = np.random.default_rng(1).integers(0, 10**6, size=(60, 60)).astype(np.int64)
    full, _ = solve(a, scaling=False, epsilon=1)
    assert full.stats.bids > 256
    sol, _ = solve(a, scaling=False, epsilon=1, max_bids=10)
    assert sol.assignment is None and sol.stats.timed_out
    late, _ = solve(a, scaling=True, time_limit_s=0.0)
    assert late.assignment is None and late.stats.timed_out
    res = AuctionAllocator(time_limit_s=0.0).allocate(random_instance(3, 6, 6, BenefitConfig()))
    assert res.pairs == () or not res.stats.timed_out  # tiny problems may finish before a check


def test_allocator_on_hand_built_problem() -> None:
    from robocoord.allocation import RobotCandidate, TaskCandidate, build_problem
    from robocoord.core.distance import DistanceOracle
    from robocoord.core.grid import OccupancyGrid

    oracle = DistanceOracle(OccupancyGrid.empty(10, 1))
    robots = [RobotCandidate(0, (0, 0)), RobotCandidate(1, (9, 0))]
    tasks = [
        TaskCandidate(10, (1, 0), (2, 0), 1, 0),  # near robot 0
        TaskCandidate(11, (8, 0), (7, 0), 1, 0),  # near robot 1
        TaskCandidate(12, (5, 0), (4, 0), 1, 0),  # in the middle: left unassigned
    ]
    res = AuctionAllocator().allocate(build_problem(robots, tasks, 0, oracle.distance))
    assert sorted(res.pairs) == [(0, 10), (1, 11)]
    assert res.stats.method == "auction-scaling" and res.stats.bids > 0
