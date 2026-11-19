"""Neighbour indexes: k-d tree invariants, Bentley–Saxe slots, equivalence with brute force
(``04`` §3.9 tests 2–4; acceptance criterion 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robocoord.core.errors import ConfigError
from robocoord.planning.kdtree import LogKDTreeIndex, StaticKDTree
from robocoord.planning.neighbors import BruteForceIndex, make_index


def _points(rng: np.random.Generator, n: int, snap: bool) -> np.ndarray:
    pts = rng.uniform(0, 20, size=(n, 2))
    if snap:  # many duplicates and exact ties
        pts = np.round(pts * 2) / 2
    return pts


@given(st.integers(0, 100_000), st.integers(1, 300), st.booleans(), st.integers(1, 9))
def test_kdtree_equals_brute_force(seed: int, n: int, snap: bool, leaf: int) -> None:
    rng = np.random.default_rng(seed)
    pts = _points(rng, n, snap)
    kd, bf = LogKDTreeIndex(leaf_size=leaf, capacity=1), BruteForceIndex(capacity=1)
    for i, (x, y) in enumerate(pts):
        assert kd.insert(float(x), float(y)) == bf.insert(float(x), float(y)) == i
        if i % 7 == 0 or i == n - 1:  # interleave queries with inserts
            for _ in range(3):
                qx, qy = (float(v) for v in _points(rng, 1, snap)[0])
                r = float(rng.choice([0.0, 0.5, 1.0, 2.5, 30.0]))
                assert kd.nearest(qx, qy) == bf.nearest(qx, qy)
                assert kd.radius(qx, qy, r).tolist() == bf.radius(qx, qy, r).tolist()
    # identical k′ counts (points inside the bounding squares of all radius queries)
    assert kd.range_points == bf.range_points
    # queries exactly at stored points
    for x, y in pts[:5]:
        assert kd.nearest(float(x), float(y)) == bf.nearest(float(x), float(y))


def test_nearest_ties_go_to_lowest_id() -> None:
    for index in (LogKDTreeIndex(leaf_size=1), BruteForceIndex()):
        for x, y in [(2.0, 0.0), (0.0, 0.0), (-2.0, 0.0), (0.0, 0.0)]:
            index.insert(x, y)
        assert index.nearest(0.0, 0.0) == 1
        assert index.nearest(1.0, 0.0) == 0  # (2,0) and (0,0) equidistant → id 0
        assert index.radius(0.0, 0.0, 2.0).tolist() == [0, 1, 2, 3]
        assert index.radius(0.0, 0.0, 1.999).tolist() == [1, 3]


@settings(max_examples=60)
@given(st.integers(0, 1100))
def test_logarithmic_method_slot_invariants(n: int) -> None:
    rng = np.random.default_rng(n)
    index = LogKDTreeIndex(leaf_size=4)
    for x, y in rng.uniform(0, 10, size=(n, 2)):
        index.insert(float(x), float(y))
    sizes = index.slot_sizes()
    assert sizes == [(1 << i) if (n >> i) & 1 else 0 for i in range(len(sizes))]
    ids = sorted(i for t in index.slots if t is not None for i in t.pid)
    assert ids == list(range(n))  # every point exactly once
    # each point was rebuilt into larger slots at most ⌊log2 n⌋ + 1 times
    if n:
        assert index.rebuild_points <= n * (math.floor(math.log2(n)) + 1)


@given(st.integers(0, 100_000), st.integers(1, 500), st.integers(1, 10), st.booleans())
def test_static_tree_invariants(seed: int, m: int, leaf: int, snap: bool) -> None:
    rng = np.random.default_rng(seed)
    pts = _points(rng, m, snap)
    xs, ys = pts[:, 0].copy(), pts[:, 1].copy()
    tree = StaticKDTree(np.arange(m, dtype=np.int64), xs, ys, leaf)
    assert sorted(tree.pid) == list(range(m))
    assert tree.depth <= max(math.ceil(math.log2(m / leaf)), 0) + 1

    def check(node: int, depth: int) -> None:
        lo, hi = tree.lo[node], tree.hi[node]
        ids = tree.pid[lo:hi]
        # bounding box is tight and contains the subtree
        assert tree.minx[node] == xs[ids].min() and tree.maxx[node] == xs[ids].max()
        assert tree.miny[node] == ys[ids].min() and tree.maxy[node] == ys[ids].max()
        assert tree.axis[node] == depth % 2  # alternating split axes
        if tree.is_leaf(node):
            assert hi - lo <= leaf
            return
        assert hi - lo > leaf
        key = xs if tree.axis[node] == 0 else ys
        left, right = tree.left[node], tree.right[node]
        lids = tree.pid[tree.lo[left] : tree.hi[left]]
        rids = tree.pid[tree.lo[right] : tree.hi[right]]
        assert len(lids) == (hi - lo) // 2  # median split
        assert key[lids].max() <= tree.split[node] <= key[rids].min()  # median property
        check(left, depth + 1)
        check(right, depth + 1)

    check(0, 0)


def test_factory_and_empty_index() -> None:
    assert make_index("brute").name == "brute"
    assert make_index("kdtree", 4).name == "kdtree"
    with pytest.raises(ConfigError):
        make_index("rtree")
    for index in (LogKDTreeIndex(), BruteForceIndex()):
        with pytest.raises(ValueError):
            index.nearest(0.0, 0.0)
        assert index.radius(0.0, 0.0, 1.0).tolist() == []
        assert len(index) == 0
    with pytest.raises(ValueError):
        StaticKDTree(np.arange(1, dtype=np.int64), np.zeros(1), np.zeros(1), 0)
    empty = StaticKDTree(np.arange(0, dtype=np.int64), np.zeros(0), np.zeros(0))
    assert empty.nearest(0.0, 0.0, math.inf, -1) == (math.inf, -1, 0, 0)
    assert empty.radius(0.0, 0.0, 1.0, 1.0, []) == (0, 0, 0)


# ---------------------------------------------------------------- work-counter definitions
# Regression tests for the Issue #3 review: the node counters must include internal and pruned
# nodes, and the rebuild counters must match their documented definitions exactly.


def _ctz(t: int) -> int:
    return (t & -t).bit_length() - 1


def _total_nodes(index: LogKDTreeIndex) -> int:
    return sum(len(t.lo) for t in index.slots if t is not None)


def _nonempty(index: LogKDTreeIndex) -> int:
    return sum(t is not None for t in index.slots)


@given(st.integers(0, 100_000), st.integers(1, 400), st.integers(1, 8))
def test_rebuild_counters_match_their_definitions(seed: int, n: int, leaf: int) -> None:
    rng = np.random.default_rng(seed)
    index = LogKDTreeIndex(leaf_size=leaf, capacity=1)
    expected_work = 0
    for t, (x, y) in enumerate(rng.uniform(0, 10, size=(n, 2)), start=1):
        index.insert(float(x), float(y))
        tree = index.slots[_ctz(t)]  # the t-th insertion builds slot ctz(t) of size 2^ctz(t)
        assert tree is not None and tree.size == 1 << _ctz(t)
        # build work = Σ subtree sizes over all built nodes; between m and m·(depth + 1)
        assert tree.build_work == sum(h - lo for lo, h in zip(tree.lo, tree.hi, strict=True))
        assert tree.size <= tree.build_work <= tree.size * (tree.depth + 1)
        expected_work += tree.build_work
    # rebuild_points: per insertion the size of the newly built tree, inserted point included
    assert index.rebuild_points == sum(1 << _ctz(t) for t in range(1, n + 1))
    assert index.rebuild_work == expected_work


@given(st.integers(0, 100_000), st.integers(1, 300), st.integers(1, 8))
def test_node_counters_include_internal_and_pruned_nodes(seed: int, n: int, leaf: int) -> None:
    rng = np.random.default_rng(seed)
    index = LogKDTreeIndex(leaf_size=leaf)
    for x, y in rng.uniform(0, 10, size=(n, 2)):
        index.insert(float(x), float(y))
    total, slots = _total_nodes(index), _nonempty(index)
    # radius covering everything: every node is examined and expanded
    index.radius(5.0, 5.0, 100.0)
    assert index.range_nodes == index.range_nodes_tested == total
    # radius far away: only the slot roots are examined, none is expanded
    index.range_nodes = index.range_nodes_tested = 0
    index.radius(1000.0, 1000.0, 1.0)
    assert (index.range_nodes, index.range_nodes_tested) == (0, slots)
    # each examined node is a root or a child of an expanded internal node
    for _ in range(5):
        index.range_nodes = index.range_nodes_tested = 0
        qx, qy, r = (float(v) for v in rng.uniform(0, 10, size=3))
        index.radius(qx, qy, r)
        assert index.range_nodes <= index.range_nodes_tested <= slots + 2 * index.range_nodes
    # nearest: at least every slot root; points are only evaluated in examined leaves
    for _ in range(5):
        index.nn_nodes = index.nn_visited = 0
        index.nearest(*(float(v) for v in rng.uniform(0, 10, size=2)))
        assert index.nn_nodes >= slots
        assert index.nn_visited <= index.nn_nodes * leaf
    # a tree with more than one leaf: the internal root is examined as well as a leaf
    big = StaticKDTree(np.arange(4 * leaf, dtype=np.int64), *rng.uniform(0, 1, (2, 4 * leaf)), leaf)
    _, _, points, nodes = big.nearest(0.5, 0.5, math.inf, -1)
    assert nodes >= 2 and points >= 1
    # a single-leaf tree: exactly one node, all its points evaluated
    small = StaticKDTree(np.arange(leaf, dtype=np.int64), *rng.uniform(0, 1, (2, leaf)), leaf)
    assert small.nearest(0.5, 0.5, math.inf, -1)[2:] == (leaf, 1)


def test_brute_force_counters() -> None:
    index = BruteForceIndex()
    for i in range(10):
        index.insert(float(i), 0.0)
    index.nearest(3.2, 0.0)
    index.radius(3.0, 0.0, 1.5)
    assert (index.nn_visited, index.nn_nodes) == (10, 0)
    assert (index.range_points, index.range_nodes, index.range_nodes_tested) == (3, 0, 0)
    assert (index.rebuild_points, index.rebuild_work) == (0, 0)
