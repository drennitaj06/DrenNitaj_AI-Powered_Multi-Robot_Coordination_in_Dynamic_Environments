"""Sequential single-item auction (algorithms §4.2; Issue #4 plan step 3).

Oracles, all independent of the module under test:
* a literal round-by-round simulation of the rule in `04` §4.2 (the module uses one sort instead);
* ``scipy.optimize.linear_sum_assignment`` (Hungarian) for the optimum and maximum cardinality;
* full enumeration (``best_perfect_assignment``) for small instances.

The ½ guarantee is on the integer objective the allocator is given (offset included); values in
original units carry no such guarantee, and a test shows it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.allocation import BenefitConfig, RobotCandidate, TaskCandidate, build_problem
from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.sequential import SequentialAllocator, solve
from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from tests.unit.allocation.test_auction import hungarian_value, random_matrix
from tests.unit.allocation.test_benefits import best_perfect_assignment, random_instance

# ---------------------------------------------------------------- independent oracles


def literal_rounds(a: np.ndarray) -> list[tuple[int, int]]:
    """`04` §4.2 as written: each round, every free robot bids its benefit for its best remaining
    feasible task (lowest task index on ties); the highest bid wins (lowest robot index on ties)."""
    k = a.shape[0]
    free_robots, free_tasks = set(range(k)), set(range(k))
    won: list[tuple[int, int]] = []
    while True:
        bids = []
        for i in sorted(free_robots):
            options = [(int(a[i, j]), j) for j in sorted(free_tasks) if a[i, j] > 0]
            if options:
                best = max(v for v, _ in options)
                bids.append((best, i, min(j for v, j in options if v == best)))
        if not bids:
            return won
        top = max(b for b, _, _ in bids)
        _, i, j = min((i, i, j) for b, i, j in bids if b == top)
        won.append((i, j))
        free_robots.discard(i)
        free_tasks.discard(j)


def max_cardinality(a: np.ndarray) -> int:
    return hungarian_value((a > 0).astype(np.int64))


def matched(assignment: tuple[int, ...], a: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i, j in enumerate(assignment) if a[i, j] > 0]


# ---------------------------------------------------------------- the rule and the ½ guarantee


@settings(max_examples=200)
@given(st.integers(0, 10**7))
def test_matches_literal_rounds_and_half_approximation(seed: int) -> None:
    a = random_matrix(seed, max_k=40)  # many ties for small C; zero blocks = infeasible pairs
    sol = solve(a)
    k = a.shape[0]
    assert sol.assignment is not None and sorted(sol.assignment) == list(range(k))
    pairs = matched(sol.assignment, a)
    assert sorted(pairs) == sorted(literal_rounds(a))  # same pairs as the specified rule
    assert sol.value == sum(int(a[i, j]) for i, j in pairs)
    assert sol.stats.bids == len(pairs)
    opt = hungarian_value(a)
    assert 2 * sol.value >= opt  # ½-approximation (Guaranteed, non-negative weights)
    assert sol.value <= opt
    # maximal: no feasible pair with both ends unmatched; hence cardinality ≥ ½ maximum
    rows, cols = {i for i, _ in pairs}, {j for _, j in pairs}
    free_rows = [i for i in range(k) if i not in rows]
    free_cols = [j for j in range(k) if j not in cols]
    assert not any(a[i, j] > 0 for i in free_rows for j in free_cols)
    assert 2 * len(pairs) >= max_cardinality(a)


@settings(max_examples=200)
@given(st.integers(0, 10**7))
def test_on_benefit_problems(seed: int) -> None:
    """Through the real benefit construction (rectangular, infeasible pairs, priorities, aging).

    * ½-approximation on the integer objective (offset included), against Hungarian.
    * Maximum cardinality. This holds for every problem `build_problem` makes with a grid
      distance: reachability is connectivity, so the feasible pairs form a complete bipartite
      block per connected component, and a maximal matching of such blocks is maximum.
    """
    p = random_instance(seed, max_n=12, max_m=12)
    if p.size == 0:
        return
    sol = solve(p.matrix)
    opt = hungarian_value(p.matrix)
    assert 2 * sol.value >= opt
    k = p.size  # the offset gives the much stronger (k−1)/(k+1) ratio (`04` §4.2 notes)
    assert (k + 1) * sol.value >= (k - 1) * opt
    res = SequentialAllocator().allocate(p)
    assert len(res.pairs) == max_cardinality(p.matrix)
    rows = {rid: i for i, rid in enumerate(p.robots)}
    cols = {tid: j for j, tid in enumerate(p.tasks)}
    assert all(p.feasible[rows[i], cols[j]] for i, j in res.pairs)
    assert res.value == pytest.approx(p.matching_value(res.pairs))
    assert res.stats.method == "sequential"


# ---------------------------------------------------------------- where greedy is optimal


@settings(max_examples=100)
@given(st.integers(0, 10**7))
def test_optimal_on_additive_and_single_robot_instances(seed: int) -> None:
    """Two classes where greedy is provably optimal, checked by full enumeration / Hungarian.

    * ``A_ij = u_i + v_j > 0`` on a complete square: every perfect matching has the same value,
      and a maximal matching of a complete square is perfect.
    * One robot: greedy takes the row maximum, which is the optimum."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 7))
    a = rng.integers(1, 50, size=k)[:, None] + rng.integers(0, 50, size=k)[None, :]
    assert solve(a).value == best_perfect_assignment(a)[0]

    p = random_instance(seed, max_n=1, max_m=8)
    if p.size:
        assert solve(p.matrix).value == hungarian_value(p.matrix)


def test_optimal_on_separated_corridor_instance() -> None:
    oracle = DistanceOracle(OccupancyGrid.empty(10, 1))
    robots = [RobotCandidate(0, (0, 0)), RobotCandidate(1, (9, 0))]
    tasks = [
        TaskCandidate(10, (1, 0), (2, 0), 1, 0),
        TaskCandidate(11, (8, 0), (7, 0), 1, 0),
        TaskCandidate(12, (5, 0), (4, 0), 1, 0),
    ]
    p = build_problem(robots, tasks, 0, oracle.distance)
    seq, auc = SequentialAllocator().allocate(p), AuctionAllocator().allocate(p)
    assert sorted(seq.pairs) == sorted(auc.pairs) == [(0, 10), (1, 11)]


# ---------------------------------------------------------------- constructed counter-examples


def test_counterexample_bound_is_tight_and_cardinality_not_maximum() -> None:
    """General matrices: the ½ bound is attained (with the documented tie rule), and greedy can
    miss maximum cardinality. 0 = infeasible."""
    tight = np.array([[2, 2], [2, 0]])
    sol = solve(tight)
    assert matched(sol.assignment, tight) == [(0, 0)]  # type: ignore[arg-type]
    assert sol.value == 2 and hungarian_value(tight) == 4  # ratio exactly ½
    strict = np.array([[3, 2], [2, 0]])  # no ties: greedy 3, optimum 2 + 2 = 4
    sol = solve(strict)
    assert sol.value == 3 and hungarian_value(strict) == 4
    assert len(matched(sol.assignment, strict)) == 1 and max_cardinality(strict) == 2  # type: ignore[arg-type]


def corridor_problem(kappa: float):  # type: ignore[no-untyped-def]
    """8 × 1 corridor. Robot 0 at x=2 is nearest to both pickups; robot 1 at x=6.

    costs c_ij = D(x_i, p_j) + D(p_j, d_j):   task 20 (p=3, d=4)   task 21 (p=0, d=1)
                                    robot 0          1 + 1 = 2            2 + 1 = 3
                                    robot 1          3 + 1 = 4            6 + 1 = 7
    Greedy takes the cheapest pair (0, 20), leaving (1, 21): cost 9. The optimum is
    (0, 21) + (1, 20): cost 7."""
    oracle = DistanceOracle(OccupancyGrid.empty(8, 1))
    robots = [RobotCandidate(0, (2, 0)), RobotCandidate(1, (6, 0))]
    tasks = [TaskCandidate(20, (3, 0), (4, 0), 1, 0), TaskCandidate(21, (0, 0), (1, 0), 1, 0)]
    return build_problem(robots, tasks, 0, oracle.distance, BenefitConfig(kappa=kappa))


def test_counterexample_on_benefit_model() -> None:
    p = corridor_problem(10.0)
    seq, auc = SequentialAllocator().allocate(p), AuctionAllocator().allocate(p)
    assert sorted(seq.pairs) == [(0, 20), (1, 21)]
    assert sorted(auc.pairs) == [(0, 21), (1, 20)]
    assert seq.value == pytest.approx(2 * 10 - 9) and auc.value == pytest.approx(2 * 10 - 7)
    seq_int, opt_int = solve(p.matrix).value, hungarian_value(p.matrix)
    assert seq_int < opt_int <= 2 * seq_int  # suboptimal, within the integer ½ bound


def test_half_bound_does_not_transfer_to_original_units() -> None:
    """Same instance with κ = 4: in original units greedy is negative (8 − 9 = −1) while the
    optimum is positive (8 − 7 = 1), so no ratio bound holds there; the integer bound still does."""
    p = corridor_problem(4.0)
    seq, auc = SequentialAllocator().allocate(p), AuctionAllocator().allocate(p)
    assert seq.value == pytest.approx(-1.0) and auc.value == pytest.approx(1.0)
    assert 2 * seq.value < auc.value
    assert 2 * solve(p.matrix).value >= hungarian_value(p.matrix)


# ---------------------------------------------------------------- edge cases


def test_edge_cases_and_validation() -> None:
    empty = solve(np.zeros((0, 0), dtype=np.int64))
    assert empty.assignment == () and empty.value == 0
    assert solve(np.array([[7]])).assignment == (0,)
    none = solve(np.zeros((3, 3), dtype=np.int64))  # all infeasible: full permutation, value 0
    assert none.assignment == (0, 1, 2) and none.value == 0 and none.stats.bids == 0
    a = random_matrix(99, max_k=20)
    assert solve(a).assignment == solve(a).assignment  # deterministic
    with pytest.raises(ValueError):
        solve(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(ValueError):
        solve(np.array([[1, -1], [0, 2]]))
    p = random_instance(5, 0, 4)  # no robots
    assert SequentialAllocator().allocate(p).pairs == ()
