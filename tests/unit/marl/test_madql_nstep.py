"""Issue 6A Phase 4: optional n-step MADQL targets (off by default)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from robocoord.core.errors import ConfigError  # noqa: E402
from robocoord.marl.madql import (  # noqa: E402
    MADQL,
    MADQLParams,
    double_dqn_targets,
    masked_epsilon_greedy,
)
from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.replay import Batch, NStepAccumulator, ReplayBuffer  # noqa: E402
from robocoord.marl.trainer import load_training_config  # noqa: E402
from tests.unit.marl.test_madql import D, rand_obs  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GAMMA = 0.99


# ---------------------------------------------------------------------------------- params
def test_default_is_one_step_and_the_issue5_dictionary() -> None:
    p = MADQLParams()
    assert p.n_step == 1
    assert "n_step" not in p.to_dict()
    cfg = load_training_config(ROOT / "configs" / "training" / "madql.toml")
    assert "n_step" not in cfg.to_dict()["madql"]
    assert MADQLParams(n_step=3).to_dict()["n_step"] == 3


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, None])
def test_invalid_n_step_is_rejected(bad: object) -> None:
    with pytest.raises(ConfigError, match="n_step"):
        MADQLParams(n_step=bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------- accumulator
def _drive(
    n: int, rewards: np.ndarray, terminated: bool, copies: int = 1, copy: int = 0
) -> tuple[list[Any], np.ndarray, np.ndarray]:
    """Push one episode of ``len(rewards)`` steps (rows: robots); return the emitted batches,
    the observations of steps 0…L (o_L = final next observation) and the actions."""
    steps, m = rewards.shape
    acc = NStepAccumulator(n, GAMMA, copies)
    obs = np.arange((steps + 1) * m * 2, dtype=np.float32).reshape(steps + 1, m, 2)
    acts = (np.arange(steps * m) % 5).reshape(steps, m)
    masks = np.ones((m, 5), bool)
    out = []
    for t in range(steps):
        last = t == steps - 1
        b = acc.push(
            copy, obs[t], acts[t], rewards[t], obs[t + 1], masks,
            terminated and last, (not terminated) and last,
        )  # fmt: skip
        out.append(b)
    assert acc.pending(copy) == 0
    return out, obs, acts


@settings(max_examples=150, deadline=None)
@given(
    n=st.integers(1, 5),
    steps=st.integers(1, 15),
    m=st.integers(1, 3),
    terminated=st.booleans(),
    seed=st.integers(0, 10_000),
)
def test_every_step_is_emitted_once_with_the_n_step_return(
    n: int, steps: int, m: int, terminated: bool, seed: int
) -> None:
    rewards = np.random.default_rng(seed).normal(size=(steps, m))
    out, obs, acts = _drive(n, rewards, terminated)
    rows = [b for b in out if b is not None]
    o = np.concatenate([b.obs for b in rows])
    a = np.concatenate([b.actions for b in rows])
    r = np.concatenate([b.rewards for b in rows])
    nx = np.concatenate([b.next_obs for b in rows])
    d = np.concatenate([b.dones for b in rows])
    h = np.concatenate([b.horizons for b in rows])
    assert len(o) == steps * m  # exactly once
    assert np.array_equal(o, obs[:steps].reshape(-1, 2))  # in time order
    assert np.array_equal(a, acts.reshape(-1))
    for t in range(steps):
        hh = min(n, steps - t)
        sl = slice(t * m, (t + 1) * m)
        assert (h[sl] == hh).all()
        ret = sum(GAMMA**k * rewards[t + k] for k in range(hh))
        assert np.allclose(r[sl], ret.astype(np.float32), rtol=0, atol=1e-6)
        assert np.array_equal(nx[sl], obs[t + hh])
        assert (d[sl] == float(terminated and t + hh == steps)).all()
    # nothing is emitted before n steps have been seen (unless the episode ended)
    for t, b in enumerate(out):
        if t < min(n, steps) - 1:
            assert b is None


def test_one_step_accumulator_equals_the_raw_transitions() -> None:
    rewards = np.random.default_rng(0).normal(size=(6, 2))
    out, obs, _ = _drive(1, rewards, terminated=True)
    for t, b in enumerate(out):
        assert b is not None and (b.horizons == 1).all()
        assert np.array_equal(b.obs, obs[t]) and np.array_equal(b.next_obs, obs[t + 1])
        assert np.allclose(b.rewards, rewards[t].astype(np.float32))
        assert (b.dones == float(t == 5)).all()


def test_copies_are_independent() -> None:
    acc = NStepAccumulator(3, GAMMA, 2)
    m = np.ones((1, 5), bool)
    o = np.zeros((1, 2), np.float32)
    assert acc.push(0, o, [0], [1.0], o, m, False, False) is None
    assert acc.push(1, o, [0], [1.0], o, m, False, False) is None
    assert acc.pending(0) == 1 and acc.pending(1) == 1
    b = acc.push(1, o, [0], [1.0], o, m, False, True)  # copy 1 ends: flushes only copy 1
    assert b is not None and len(b.rewards) == 2 and acc.pending(0) == 1


def test_accumulator_validation() -> None:
    with pytest.raises(ValueError):
        NStepAccumulator(0, GAMMA, 1)
    with pytest.raises(ValueError):
        NStepAccumulator(3, 1.0, 1)


# ---------------------------------------------------------------------------------- targets
def test_tensor_gamma_target_equals_the_scalar_one_and_the_manual_one() -> None:
    rng = np.random.default_rng(3)
    qo = torch.from_numpy(rng.normal(size=(8, 5)).astype(np.float32))
    qt = torch.from_numpy(rng.normal(size=(8, 5)).astype(np.float32))
    masks = torch.from_numpy(rng.random((8, 5)) < 0.6)
    masks[:, 0] = True
    r = torch.from_numpy(rng.normal(size=8).astype(np.float32))
    d = torch.from_numpy((rng.random(8) < 0.3).astype(np.float32))
    one = double_dqn_targets(qo, qt, masks, r, d, GAMMA)
    vec = double_dqn_targets(qo, qt, masks, r, d, torch.full((8,), GAMMA, dtype=torch.float32))
    assert torch.allclose(one, vec)
    h = np.array([1, 2, 3, 3, 1, 2, 3, 1], dtype=np.float32)
    g = torch.from_numpy(np.power(np.float32(GAMMA), h).astype(np.float32))
    y = double_dqn_targets(qo, qt, masks, r, d, g).numpy()
    best = np.where(masks.numpy(), qo.numpy(), -np.inf).argmax(axis=1)
    manual = r.numpy() + GAMMA**h * (1 - d.numpy()) * qt.numpy()[np.arange(8), best]
    assert np.allclose(y, manual, atol=1e-6)


def _batch(rng: np.random.Generator, horizons: np.ndarray | None) -> Batch:
    return Batch(
        obs=rand_obs(rng, 16),
        actions=rng.integers(0, 5, 16),
        rewards=rng.normal(size=16).astype(np.float32),
        next_obs=rand_obs(rng, 16),
        next_masks=np.ones((16, 5), bool),
        dones=np.zeros(16, np.float32),
        horizons=horizons,
    )


def test_n_step_loss_with_unit_horizons_equals_the_one_step_loss() -> None:
    configure_torch(5)
    a = MADQL(D, MADQLParams(hidden=(16,)))
    configure_torch(5)
    b = MADQL(D, MADQLParams(hidden=(16,), n_step=3))
    batch = _batch(np.random.default_rng(1), np.ones(16, np.float32))
    assert torch.allclose(a.loss(batch), b.loss(batch))
    h3 = _batch(np.random.default_rng(1), np.full(16, 3, np.float32))
    assert not torch.allclose(a.loss(h3), b.loss(h3))  # γ^3 is used when h = 3


def test_n_step_learner_needs_horizons() -> None:
    agent = MADQL(D, MADQLParams(hidden=(8,), n_step=2))
    with pytest.raises(ValueError, match="horizons"):
        agent.loss(_batch(np.random.default_rng(0), None))


def test_buffer_stores_horizons_and_defaults_to_one() -> None:
    rng = np.random.default_rng(2)
    buf = ReplayBuffer(50)
    obs = rand_obs(rng, 10)
    m = np.ones((10, 5), bool)
    buf.add(obs, np.zeros(10), np.zeros(10), obs, m, np.zeros(10))
    buf.add(obs, np.zeros(10), np.zeros(10), obs, m, np.zeros(10), np.full(10, 3))
    s = buf.sample(200, np.random.default_rng(0))
    assert s.horizons is not None and set(np.unique(s.horizons)) <= {1.0, 3.0}
    assert (buf.horizons[:10] == 1).all() and (buf.horizons[10:20] == 3).all()


# ---------------------------------------------------------------------------------- actions
@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 10_000), eps=st.sampled_from([0.0, 0.05, 0.5, 1.0]))
def test_epsilon_greedy_never_returns_a_masked_action(seed: int, eps: float) -> None:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(32, 5)).astype(np.float32)
    masks = rng.random((32, 5)) < 0.5
    masks[:, 0] = True  # STAY is never masked (env contract)
    a = masked_epsilon_greedy(q, masks, eps, rng)
    assert masks[np.arange(32), a].all()
    if eps == 0.0:
        assert np.array_equal(a, np.where(masks, q, -np.inf).argmax(axis=1))


def test_gamma_power_is_exact_for_small_horizons() -> None:
    for h in (1, 2, 3, 5):
        g = float(np.power(np.float32(GAMMA), np.float32(h)))
        assert math.isclose(g, GAMMA**h, rel_tol=1e-6)
