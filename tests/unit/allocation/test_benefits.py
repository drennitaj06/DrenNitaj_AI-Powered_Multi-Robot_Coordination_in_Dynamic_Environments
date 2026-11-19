"""Benefit construction (model §8; algorithms §4.1; Issue #4 plan step 1): offset and maximum
cardinality, padding equivalence, infeasible pairs, rounding bound. Oracles are brute-force
enumerations written independently of the module under test."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.allocation import (
    BenefitConfig,
    RobotCandidate,
    TaskCandidate,
    build_problem,
    effective_weight,
    priority_weight,
)
from robocoord.core.distance import DistanceOracle
from robocoord.core.errors import ConfigError
from robocoord.core.grid import OccupancyGrid

# ---------------------------------------------------------------- independent brute-force oracles


def partial_matchings(feasible: np.ndarray) -> Iterator[list[tuple[int, int]]]:
    """Every matching (set of disjoint feasible (row, col) pairs), including the empty one."""
    n, m = feasible.shape

    def rec(
        i: int, used: frozenset[int], acc: list[tuple[int, int]]
    ) -> Iterator[list[tuple[int, int]]]:
        if i == n:
            yield list(acc)
            return
        yield from rec(i + 1, used, acc)  # robot i unmatched
        for j in range(m):
            if feasible[i, j] and j not in used:
                acc.append((i, j))
                yield from rec(i + 1, used | {j}, acc)
                acc.pop()

    yield from rec(0, frozenset(), [])


def best_perfect_assignment(matrix: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Maximum-value perfect assignment of a square matrix by full enumeration (k ≤ 6)."""
    k = matrix.shape[0]
    best, arg = -1, ()
    for perm in itertools.permutations(range(k)):
        v = int(sum(matrix[i, perm[i]] for i in range(k)))
        if v > best:
            best, arg = v, perm
    return best, arg


# ---------------------------------------------------------------- random instances on real grids


def random_instance(seed: int, max_n: int = 5, max_m: int = 5, cfg: BenefitConfig | None = None):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    w, h = int(rng.integers(2, 8)), int(rng.integers(2, 8))
    grid = OccupancyGrid(rng.random((h, w)) < rng.uniform(0.0, 0.4))  # may be disconnected
    free = grid.free_cells()
    oracle = DistanceOracle(grid)
    n, m = int(rng.integers(0, max_n + 1)), int(rng.integers(0, max_m + 1))
    pick = lambda: free[int(rng.integers(len(free)))]  # noqa: E731
    t = int(rng.integers(0, 50))
    tasks = []
    for j in range(m if len(free) >= 2 else 0):
        p, d = pick(), pick()
        tasks.append(
            TaskCandidate(100 + j, p, d, int(rng.integers(1, 4)), int(rng.integers(0, t + 1)))
        )
    robots = []
    for i in range(n):
        current = tasks[int(rng.integers(len(tasks)))].id if tasks and rng.random() < 0.3 else None
        robots.append(RobotCandidate(i, pick(), current))
    if cfg is None:
        cfg = BenefitConfig(
            kappa=float(rng.choice([0.0, 10.0, 3.7])),
            hysteresis=int(rng.choice([0, 2])),
            alpha=float(rng.choice([0.0, 0.01, 0.05])),
            scale=int(rng.choice([1, 100])),
        )
    return build_problem(robots, tasks, t, oracle.distance, cfg)


# ---------------------------------------------------------------- definitions and hand computation


def test_weights_aging_and_config() -> None:
    assert [priority_weight(p) for p in (1, 2, 3)] == [1.0, 2.0, 4.0]
    assert effective_weight(3, release=10, t=30, alpha=0.01) == pytest.approx(4.0 * 1.2)
    assert effective_weight(2, release=10, t=5, alpha=0.5) == 2.0  # not yet released: age 0
    with pytest.raises(ValueError):
        priority_weight(0)
    for bad in (
        {"kappa": -1.0},
        {"kappa": math.inf},
        {"hysteresis": -1},
        {"hysteresis": 1.5},
        {"alpha": -0.1},
        {"scale": 0},
    ):
        with pytest.raises(ConfigError):
            BenefitConfig(**bad)  # type: ignore[arg-type]
    assert BenefitConfig() == BenefitConfig(kappa=10.0, hysteresis=2, alpha=0.01, scale=100)


def test_hand_computed_instance() -> None:
    """3×1 open corridor, model defaults. Every entry checked by hand."""
    grid = OccupancyGrid.empty(6, 1)
    oracle = DistanceOracle(grid)
    robots = [RobotCandidate(0, (0, 0)), RobotCandidate(1, (5, 0), task=7)]
    tasks = [
        TaskCandidate(7, pickup=(2, 0), dropoff=(4, 0), priority=2, release=0),  # age 3
        TaskCandidate(8, pickup=(5, 0), dropoff=(0, 0), priority=1, release=3),  # age 0
    ]
    p = build_problem(robots, tasks, t=3, distance=oracle.distance)
    # c_ij = D(x_i, p_j) + D(p_j, d_j)
    assert p.cost.tolist() == [[2 + 2, 5 + 5], [3 + 2, 0 + 5]]
    # κ w̃: task 7: 10·2·(1 + 0.01·3) = 20.6; task 8: 10·1·1 = 10
    assert p.priority_value.tolist() == pytest.approx([20.6, 10.0])
    # value = κw̃ − c + h·1[current]; robot 1 currently holds task 7 (h = 2)
    assert np.allclose(p.value, [[16.6, 0.0], [17.6, 5.0]])
    # integer form (s = 100): P = [2060, 1000]; B_int = 2060 + 200; C_max = 1000; k = 2
    assert p.offset == 1 + 2 * (2260 + 1000)
    core = np.array([[2060 - 400, 1000 - 1000], [2060 - 500 + 200, 1000 - 500]])
    assert p.matrix.tolist() == (p.offset + core).tolist()
    assert p.current == {1: 7}
    assert p.pairs([1, 0]) == [(0, 8), (1, 7)]
    assert p.matching_value([(0, 8), (1, 7)]) == pytest.approx(0.0 + 17.6)


def test_infeasible_pairs_and_padding() -> None:
    #   map (top row first): a wall splits columns 0–1 from 3–4
    grid = OccupancyGrid.from_rows(["..@..", "..@..", "..@.."])
    oracle = DistanceOracle(grid)
    robots = [RobotCandidate(0, (0, 0)), RobotCandidate(1, (4, 0)), RobotCandidate(2, (1, 2))]
    tasks = [
        TaskCandidate(0, (1, 0), (0, 2), 1, 0),  # left side only
        TaskCandidate(1, (3, 1), (0, 1), 3, 0),  # pickup → drop-off unreachable: never feasible
    ]
    p = build_problem(robots, tasks, t=0, distance=oracle.distance)
    assert p.feasible.tolist() == [[True, False], [False, False], [True, False]]
    assert p.size == 3 and p.matrix.shape == (3, 3)
    assert (p.matrix[p.feasible.nonzero()] >= 1).all()
    assert (p.matrix[:3, :2][~p.feasible] == 0).all() and (p.matrix[:, 2] == 0).all()
    # nothing feasible at all: an all-zero matrix and an empty mapping
    empty = build_problem([RobotCandidate(0, (4, 0))], tasks[:1], 0, oracle.distance)
    assert empty.offset == 0 and empty.matrix.tolist() == [[0]] and empty.pairs([0]) == []
    none = build_problem([], [], 0, oracle.distance)
    assert none.size == 0 and none.pairs([]) == []


def test_input_validation_and_overflow_guard() -> None:
    oracle = DistanceOracle(OccupancyGrid.empty(3, 1))
    t0 = TaskCandidate(0, (0, 0), (2, 0), 1, 0)
    with pytest.raises(ValueError):
        build_problem(
            [RobotCandidate(0, (0, 0)), RobotCandidate(0, (1, 0))], [t0], 0, oracle.distance
        )
    with pytest.raises(ValueError):
        build_problem(
            [RobotCandidate(0, (0, 0))],
            [TaskCandidate(0, (0, 0), (2, 0), 0, 0)],
            0,
            oracle.distance,
        )
    huge = BenefitConfig(kappa=1e15, scale=10**6)
    with pytest.raises(OverflowError):
        build_problem([RobotCandidate(0, (0, 0))], [t0], 0, oracle.distance, huge)


# ---------------------------------------------------------------- properties against brute force


@settings(max_examples=300)
@given(st.integers(0, 10**6))
def test_offset_gives_max_cardinality_and_padding_equivalence(seed: int) -> None:
    """The optimal padded assignment (enumerated) has maximum feasible cardinality, maximises the
    integer core among maximum-cardinality matchings, and its value equals the best *partial*
    feasible matching's value (padding equivalence, `04` §4.1)."""
    p = random_instance(seed)
    n, m = len(p.robots), len(p.tasks)
    if p.size == 0:
        return
    feas = p.feasible
    # 1. structure: k×k, feasible entries ≥ 1, dummy / infeasible entries 0
    assert p.matrix.shape == (max(n, m),) * 2
    assert (p.matrix[:n, :m][feas] >= 1).all()
    assert (p.matrix[:n, :m][~feas] == 0).all()
    assert (p.matrix[n:, :] == 0).all() and (p.matrix[:, m:] == 0).all()
    # 2. brute force over all partial feasible matchings
    matchings = list(partial_matchings(feas))
    max_card = max(len(mm) for mm in matchings)
    core = p.matrix[:n, :m].astype(np.int64) - p.offset
    best_core = max(sum(int(core[i, j]) for i, j in mm) for mm in matchings if len(mm) == max_card)
    best_partial = max(sum(int(p.matrix[i, j]) for i, j in mm) for mm in matchings)
    # 3. optimal padded perfect assignment by enumeration
    value, perm = best_perfect_assignment(p.matrix)
    chosen = [(i, perm[i]) for i in range(n) if perm[i] < m and feas[i, perm[i]]]
    assert value == best_partial  # padding equivalence
    assert len(chosen) == max_card  # maximum cardinality (G10)
    assert sum(int(core[i, j]) for i, j in chosen) == best_core  # optimal among max-cardinality
    assert len(p.pairs(list(perm))) == max_card


@settings(max_examples=300)
@given(st.integers(0, 10**6))
def test_rounding_bound(seed: int) -> None:
    """Each integer benefit differs from s·a_ij by at most 1/2 (≤ 1/(2s) in original units), so a
    matching of q pairs differs by at most q/(2s); costs and hysteresis are exact."""
    p = random_instance(seed)
    n, m = len(p.robots), len(p.tasks)
    if not p.feasible.any():
        return
    s = p.config.scale
    core = (p.matrix[:n, :m] - p.offset).astype(np.float64)
    err = np.abs(core - s * np.where(p.feasible, p.value, 0.0))[p.feasible]
    assert (err <= 0.5 + 1e-9).all()
    # only the priority term is rounded: the signed error is identical down each task column
    for j in range(m):
        rows = p.feasible[:, j]
        if rows.sum() > 1:
            e = core[rows, j] - s * p.value[rows, j]
            assert np.allclose(e, e[0], atol=1e-9)


@settings(max_examples=60)
@given(st.integers(0, 10**6), st.sampled_from([(10.0, 2), (0.0, 2), (10.0, 0), (0.0, 0)]))
def test_offset_margin_holds_for_large_instances(seed: int, kappa_h: tuple[float, int]) -> None:
    """For any q ≤ k − 1, every (q+1)-matching beats every q-matching iff the smallest feasible
    benefit exceeds q·(largest − smallest); checked at the hardest q = k − 1 on instances far too
    large for enumeration (k up to 120), including the κ = 0 and h = 0 ablations of E3b."""
    kappa, h = kappa_h
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid.empty(30, 30)
    oracle = DistanceOracle(grid)
    cells = grid.free_cells()
    pick = lambda: cells[int(rng.integers(len(cells)))]  # noqa: E731
    n, m = int(rng.integers(1, 40)), int(rng.integers(1, 121))
    t = 500
    tasks = [
        TaskCandidate(j, pick(), pick(), int(rng.integers(1, 4)), int(rng.integers(0, t)))
        for j in range(m)
    ]
    robots = [
        RobotCandidate(i, pick(), tasks[i % m].id if rng.random() < 0.5 else None) for i in range(n)
    ]
    p = build_problem(robots, tasks, t, oracle.distance, BenefitConfig(kappa=kappa, hysteresis=h))
    vals = p.matrix[:n, :m][p.feasible]
    lo, hi, k = int(vals.min()), int(vals.max()), p.size
    assert lo >= 1
    assert k * lo > (k - 1) * hi  # ⇒ (q+1)·lo > q·hi for every q ≤ k − 1
