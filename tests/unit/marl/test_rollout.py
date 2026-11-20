"""Rollout storage and GAE against hand computations (algorithms §7 test 1; Issue #6)."""

from __future__ import annotations

import numpy as np
import pytest

from robocoord.marl.rollout import RolloutBuffer

G, L = 0.99, 0.95


def fill(buf: RolloutBuffer, k: int, t: int, n: int, r: float, v: float, **kw: object) -> None:
    buf.add(
        t,
        k,
        obs=np.full((n, buf.obs_dim), t, dtype=np.float32),
        masks=np.ones((n, 5), dtype=bool),
        actions=np.zeros(n, dtype=np.int64),
        logp=np.zeros(n, dtype=np.float32),
        values=np.full(n, v, dtype=np.float32),
        rewards=np.full(n, r),
        terminated=bool(kw.get("terminated", False)),
        truncated=bool(kw.get("truncated", False)),
        boot_values=kw.get("boot"),  # type: ignore[arg-type]
    )


def one_stream(**ends: object) -> RolloutBuffer:
    buf = RolloutBuffer(3, 1, 4, max_robots=1)
    for t in range(3):
        kw = {}
        if ends.get("term") == t:
            kw["terminated"] = True
        if ends.get("trunc") == t:
            kw["truncated"] = True
            kw["boot"] = np.array([2.0], dtype=np.float32)
        fill(buf, 0, t, 1, r=1.0, v=0.5, **kw)
    return buf


def test_gae_without_episode_end() -> None:
    buf = one_stream()
    buf.compute_gae(np.array([[0.5]], np.float32), np.array([[True]]), G, L)
    d = 1 + G * 0.5 - 0.5
    a2 = d
    a1 = d + G * L * a2
    a0 = d + G * L * a1
    assert buf.advantages[:, 0, 0] == pytest.approx([a0, a1, a2], rel=1e-6)
    assert buf.returns[:, 0, 0] == pytest.approx([a0 + 0.5, a1 + 0.5, a2 + 0.5], rel=1e-6)


def test_gae_termination_cuts_and_bootstraps_zero() -> None:
    buf = one_stream(term=1)
    buf.compute_gae(np.array([[0.5]], np.float32), np.array([[True]]), G, L)
    d_next = 1 + G * 0.5 - 0.5
    a2 = d_next  # a new episode starts at t = 2
    a1 = 1 + G * 0.0 - 0.5  # terminated: V_next = 0, no carry from t = 2
    a0 = (1 + G * 0.5 - 0.5) + G * L * a1
    assert buf.advantages[:, 0, 0] == pytest.approx([a0, a1, a2], rel=1e-6)


def test_gae_truncation_bootstraps_the_final_value_and_cuts() -> None:
    buf = one_stream(trunc=1)
    buf.compute_gae(np.array([[0.5]], np.float32), np.array([[True]]), G, L)
    a2 = 1 + G * 0.5 - 0.5
    a1 = 1 + G * 2.0 - 0.5  # V(final state) = 2.0 from boot_values, not V_{t+1} = 0.5
    a0 = (1 + G * 0.5 - 0.5) + G * L * a1
    assert buf.advantages[:, 0, 0] == pytest.approx([a0, a1, a2], rel=1e-6)


def test_last_step_bootstraps_with_the_last_values() -> None:
    buf = one_stream()
    buf.compute_gae(np.array([[3.0]], np.float32), np.array([[True]]), G, L)
    assert buf.advantages[2, 0, 0] == pytest.approx(1 + G * 3.0 - 0.5, rel=1e-6)


def test_streams_are_independent_and_padding_is_excluded() -> None:
    buf = RolloutBuffer(2, 2, 3, max_robots=3)
    # copy 0: two robots, different rewards; copy 1: one robot, terminates at t = 0, then a
    # three-robot episode starts in the same copy
    for t in range(2):
        buf.add(t, 0, obs=np.zeros((2, 3), np.float32), masks=np.ones((2, 5), bool),
                actions=np.zeros(2, np.int64), logp=np.zeros(2, np.float32),
                values=np.array([0.0, 1.0], np.float32), rewards=[1.0, 0.0],
                terminated=False, truncated=False)  # fmt: skip
    buf.add(0, 1, obs=np.zeros((1, 3), np.float32), masks=np.ones((1, 5), bool),
            actions=np.zeros(1, np.int64), logp=np.zeros(1, np.float32),
            values=np.array([0.2], np.float32), rewards=[5.0], terminated=True,
            truncated=False)  # fmt: skip
    buf.add(1, 1, obs=np.zeros((3, 3), np.float32), masks=np.ones((3, 5), bool),
            actions=np.zeros(3, np.int64), logp=np.zeros(3, np.float32),
            values=np.zeros(3, np.float32), rewards=[1.0, 1.0, 1.0], terminated=False,
            truncated=False)  # fmt: skip
    last = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], np.float32)
    valid = np.array([[True, True, False], [True, True, True]])
    buf.compute_gae(last, valid, G, L)
    # copy 0 robot 0: r = 1, V = 0 at both steps
    a01 = 1.0
    a00 = 1.0 + G * L * a01
    assert buf.advantages[:, 0, 0] == pytest.approx([a00, a01])
    # copy 0 robot 1: r = 0, V = 1 → δ = γ − 1
    d = G * 1.0 - 1.0
    assert buf.advantages[:, 0, 1] == pytest.approx([d + G * L * d, d])
    # copy 1: the terminated robot does not see the next episode
    assert buf.advantages[0, 1, 0] == pytest.approx(5.0 - 0.2)
    assert buf.advantages[0, 1, 1:].tolist() == [0.0, 0.0]  # padding
    s = buf.flatten()
    assert len(s) == 2 + 2 + 1 + 3 and s.states is None
    assert np.all(buf.returns[~buf.valid] == 0)


def test_a_running_stream_may_not_vanish() -> None:
    buf = RolloutBuffer(1, 1, 2, max_robots=2)
    fill(buf, 0, 0, 2, r=0.0, v=0.0)
    with pytest.raises(ValueError, match="discontinuity"):
        buf.compute_gae(np.zeros((1, 2), np.float32), np.array([[True, False]]), G, L)


def test_input_validation() -> None:
    buf = RolloutBuffer(1, 1, 2, max_robots=1, state_dim=3)
    with pytest.raises(ValueError, match="global state"):
        fill(buf, 0, 0, 1, r=0, v=0)
    with pytest.raises(ValueError, match="bootstrap"):
        buf.add(0, 0, obs=np.zeros((1, 2), np.float32), masks=np.ones((1, 5), bool),
                actions=np.zeros(1, np.int64), logp=np.zeros(1, np.float32),
                values=np.zeros(1, np.float32), rewards=[0.0], terminated=False, truncated=True,
                state=np.zeros(3, np.float32))  # fmt: skip
    with pytest.raises(ValueError, match="exceed"):
        RolloutBuffer(1, 1, 2, max_robots=1).add(
            0, 0, obs=np.zeros((2, 2), np.float32), masks=np.ones((2, 5), bool),
            actions=np.zeros(2, np.int64), logp=np.zeros(2, np.float32),
            values=np.zeros(2, np.float32), rewards=[0.0, 0.0], terminated=False,
            truncated=False)  # fmt: skip
    with pytest.raises(ValueError, match="not complete"):
        RolloutBuffer(2, 1, 2).compute_gae(np.zeros((1, 8), np.float32), np.ones((1, 8), bool),
                                           G, L)  # fmt: skip


def test_flatten_repeats_the_copy_state_for_each_robot() -> None:
    buf = RolloutBuffer(1, 2, 2, max_robots=3, state_dim=2)
    for k, n in ((0, 2), (1, 3)):
        buf.add(0, k, obs=np.zeros((n, 2), np.float32), masks=np.ones((n, 5), bool),
                actions=np.zeros(n, np.int64), logp=np.zeros(n, np.float32),
                values=np.zeros(n, np.float32), rewards=np.zeros(n), terminated=False,
                truncated=False, state=np.array([k, 10 + k], np.float32))  # fmt: skip
    buf.compute_gae(np.zeros((2, 3), np.float32), np.ones((2, 3), bool), G, L)
    s = buf.flatten()
    assert s.states is not None
    assert s.states.tolist() == [[0, 10], [0, 10], [1, 11], [1, 11], [1, 11]]
    assert buf.nbytes > 0
