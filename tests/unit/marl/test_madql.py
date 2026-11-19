"""MADQL components (algorithms §6 tests 1–5)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from robocoord.marl.config import ObsParams  # noqa: E402
from robocoord.marl.madql import (  # noqa: E402
    MADQL,
    MADQLParams,
    double_dqn_targets,
    masked_epsilon_greedy,
)
from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.replay import Batch, ObsCodec, ReplayBuffer  # noqa: E402

D = ObsParams().dim


def rand_obs(rng: np.random.Generator, n: int) -> np.ndarray:
    """Observation rows with the real value sets: 0/1 channels, priority levels π/P, scalars."""
    p = ObsParams()
    k2 = p.window**2
    grid = rng.integers(0, 2, size=(n, p.grid_size)).astype(np.float32)
    grid[:, 4 * k2 : 5 * k2] = rng.integers(0, 4, size=(n, k2)).astype(np.float32) / np.float32(3)
    scal = rng.normal(size=(n, 13)).astype(np.float32)
    return np.concatenate([grid, scal], axis=1)


# ---------------------------------------------------------------------------------- replay
def test_codec_round_trip_is_exact() -> None:
    rng = np.random.default_rng(0)
    obs = rand_obs(rng, 50)
    codec = ObsCodec()
    g, s = codec.encode(obs)
    assert g.dtype == np.uint8 and s.dtype == np.float32
    assert np.array_equal(codec.decode(g, s), obs)


def test_replay_fifo_overwrite_and_dtypes() -> None:
    buf = ReplayBuffer(5)
    rng = np.random.default_rng(1)
    for k in range(3):
        obs = rand_obs(rng, 3)
        buf.add(obs, [k] * 3, [float(k)] * 3, obs, np.ones((3, 5), bool), [0, 0, 1])
    assert len(buf) == 5 and buf.added == 9 and buf.pos == 4
    # writes: k=0 → slots 0,1,2; k=1 → 3,4,0; k=2 → 1,2,3 (oldest entries overwritten first)
    assert buf.actions.tolist() == [1, 2, 2, 2, 1]
    assert buf.dones.tolist() == [1, 0, 0, 1, 0]
    b = buf.sample(7, np.random.default_rng(2))
    assert b.obs.dtype == np.float32 and b.actions.dtype == np.int64
    assert b.dones.dtype == np.float32 and b.next_masks.dtype == bool


def test_replay_sampling_is_uniform_and_seeded() -> None:
    buf = ReplayBuffer(10)
    rng = np.random.default_rng(3)
    obs = rand_obs(rng, 10)
    buf.add(obs, np.arange(10) % 5, np.arange(10), obs, np.ones((10, 5), bool), np.zeros(10))
    a = buf.sample(20_000, np.random.default_rng(4))
    b = buf.sample(20_000, np.random.default_rng(4))
    assert np.array_equal(a.rewards, b.rewards) and np.array_equal(a.obs, b.obs)
    counts = np.bincount(a.rewards.astype(int), minlength=10)
    chi2 = ((counts - 2000) ** 2 / 2000).sum()
    assert chi2 < 27.88  # χ²(9) 99.9 % quantile


def test_replay_memory_under_300_mb_for_200k() -> None:
    small = ReplayBuffer(1000)
    per = small.nbytes / 1000
    assert per * 200_000 < 300e6
    assert per < 820  # ≈ 0.8 kB per transition (design D)


# ---------------------------------------------------------------------------------- targets
def test_double_dqn_targets_vs_numpy() -> None:
    rng = np.random.default_rng(5)
    qo = rng.normal(size=(6, 5)).astype(np.float32)
    qt = rng.normal(size=(6, 5)).astype(np.float32)
    masks = rng.random((6, 5)) < 0.6
    masks[:, 0] = True
    r = rng.normal(size=6).astype(np.float32)
    d = np.array([0, 1, 0, 0, 1, 0], np.float32)  # 1 = termination; truncation stays 0
    y = double_dqn_targets(
        torch.from_numpy(qo),
        torch.from_numpy(qt),
        torch.from_numpy(masks),
        torch.from_numpy(r),
        torch.from_numpy(d),
        0.9,
    ).numpy()
    for i in range(6):
        a = int(np.argmax(np.where(masks[i], qo[i], -np.inf)))  # selection by the online net
        expect = r[i] + 0.9 * (1 - d[i]) * qt[i, a]  # evaluation by the target net
        assert y[i] == pytest.approx(expect, rel=1e-6)
    assert y[1] == pytest.approx(r[1]) and y[4] == pytest.approx(r[4])


def test_double_dqn_selects_with_online_not_target() -> None:
    qo = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    qt = torch.tensor([[5.0, 2.0, 9.0, 0.0, 0.0]])
    m = torch.ones(1, 5, dtype=torch.bool)
    y = double_dqn_targets(qo, qt, m, torch.zeros(1), torch.zeros(1), 1.0)
    assert float(y[0]) == 2.0  # online picks action 1; target's own max (9) is not used
    m2 = m.clone()
    m2[0, 1] = False  # a masked action is never selected in the target
    y2 = double_dqn_targets(qo, qt, m2, torch.zeros(1), torch.zeros(1), 1.0)
    assert float(y2[0]) == 5.0  # remaining ties at 0 → lowest index 0


# ---------------------------------------------------------------------------------- acting
@settings(max_examples=200, deadline=None)
@given(
    st.integers(0, 2**32 - 1),
    st.floats(0.0, 1.0),
    st.lists(st.lists(st.booleans(), min_size=5, max_size=5), min_size=1, max_size=12),
)
def test_masked_epsilon_greedy_never_selects_masked(seed: int, eps: float, rows: list) -> None:
    masks = np.array(rows, dtype=bool)
    masks[:, 0] = True  # stay is never masked
    rng = np.random.default_rng(seed)
    q = rng.normal(size=masks.shape).astype(np.float32) * 10
    a = masked_epsilon_greedy(q, masks, eps, rng)
    assert all(masks[i, a[i]] for i in range(len(a)))
    if eps == 0.0:
        assert np.array_equal(a, np.where(masks, q, -np.inf).argmax(axis=1))


def test_epsilon_schedule() -> None:
    p = MADQLParams()
    assert p.epsilon(0, 1000) == 1.0
    assert p.epsilon(150, 1000) == pytest.approx(0.525)
    assert p.epsilon(300, 1000) == pytest.approx(0.05) == p.epsilon(999, 1000)


# ---------------------------------------------------------------------------------- learning
def test_overfits_a_fixed_batch() -> None:
    configure_torch(0)
    rng = np.random.default_rng(6)
    agent = MADQL(D, MADQLParams(lr=1e-3, hidden=(64, 64)))
    batch = Batch(
        obs=rand_obs(rng, 32),
        actions=rng.integers(0, 5, 32),
        rewards=rng.normal(size=32).astype(np.float32),
        next_obs=rand_obs(rng, 32),
        next_masks=np.ones((32, 5), bool),
        dones=np.ones(32, np.float32),  # targets are the rewards
    )
    first = agent.update(batch)
    for _ in range(400):
        last = agent.update(batch)
    assert last < 1e-3 < first


def chain_q_star(n: int, gamma: float) -> np.ndarray:
    """Chain 0 … n−1; action 4 (right) moves right, 3 (left) moves left (wall at 0); reaching
    state n−1 gives reward 1 and terminates. Q*(s, right) = γ^(n−2−s), Q*(s, left) = γ Q*(s−1∨0)."""
    v = np.array([gamma ** (n - 2 - s) for s in range(n - 1)])
    q = np.zeros((n - 1, 2))
    q[:, 1] = v
    q[:, 0] = gamma * v[np.maximum(np.arange(n - 1) - 1, 0)]
    return q


def test_chain_mdp_learning_sanity() -> None:
    """Learned Q approaches the known Q* of a small chain (algorithms §6 test 3)."""
    configure_torch(1)
    n, gamma = 6, 0.9
    dim = 8
    agent = MADQL(dim, MADQLParams(lr=3e-3, gamma=gamma, hidden=(32, 32), batch_size=64))

    def onehot(s: np.ndarray) -> np.ndarray:
        out = np.zeros((len(s), dim), np.float32)
        out[np.arange(len(s)), s] = 1.0
        return out

    mask = np.zeros((1, 5), bool)
    mask[0, 3] = mask[0, 4] = True
    states = np.repeat(np.arange(n - 1), 2)
    acts = np.tile([3, 4], n - 1)
    nxt = np.where(acts == 4, states + 1, np.maximum(states - 1, 0))
    rew = (nxt == n - 1).astype(np.float32)
    done = (nxt == n - 1).astype(np.float32)
    batch = Batch(
        onehot(states),
        acts.astype(np.int64),
        rew,
        onehot(nxt),
        np.repeat(mask, len(states), axis=0),
        done,
    )
    for it in range(3000):
        agent.update(batch)
        if it % 50 == 0:
            agent.sync_target()
    q = agent.q_values(onehot(np.arange(n - 1)))[:, [3, 4]]
    assert np.abs(q - chain_q_star(n, gamma)).max() < 0.05
    assert (q[:, 1] > q[:, 0]).all()  # the greedy policy is optimal


def test_updates_are_deterministic() -> None:
    def run() -> list[np.ndarray]:
        configure_torch(7)
        rng = np.random.default_rng(8)
        agent = MADQL(D, MADQLParams(hidden=(32,)))
        buf = ReplayBuffer(100)
        obs = rand_obs(rng, 50)
        buf.add(
            obs,
            rng.integers(0, 5, 50),
            rng.normal(size=50),
            obs,
            np.ones((50, 5), bool),
            np.zeros(50),
        )
        for _ in range(10):
            agent.update(buf.sample(16, rng))
        return [p.detach().numpy().copy() for p in agent.q.parameters()]

    a, b = run(), run()
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
