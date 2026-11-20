"""PPO rollouts against the real environment (Issue #6, Revision 2 regression tests R1–R6).

The centralised critic must bootstrap a truncated episode with the **final** observation and the
**final** global state, captured before the vector environment's auto-reset — never with the reset
episode's state — and a terminated episode with 0. The Issue #5 path (``with_state=False``) is
unchanged.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest

from robocoord.marl.config import ObsParams
from robocoord.marl.env import MultiRobotEnv
from robocoord.marl.global_state import global_state_size
from robocoord.marl.observation import S_SUGGEST
from robocoord.marl.rollout import RolloutBuffer, collect_rollout, last_values
from robocoord.marl.vector_env import SyncVectorEnv
from robocoord.scenario.spec import ScenarioSpec
from tests.integration.test_marl_env import hand_spec, small

T_INDEX = global_state_size(8) - 2  # the t / T scalar of φ(s)
SUGGEST = ObsParams().grid_size + S_SUGGEST
G, L = 0.99, 0.95


def stay(obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(len(obs), dtype=np.int64), np.zeros(len(obs), dtype=np.float32)


def follow(obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The planner's suggestion read from the observation (stay when masked)."""
    a = obs[:, SUGGEST : SUGGEST + 5].argmax(axis=1)
    a = np.where(masks[np.arange(len(a)), a], a, 0)
    return a.astype(np.int64), np.zeros(len(obs), dtype=np.float32)


def queue(*specs: ScenarioSpec) -> Callable[[int], ScenarioSpec]:
    """Per-copy scenario queues; a copy's spec repeats once its queue is exhausted."""
    pending = [list(s) if isinstance(s, tuple) else [s] for s in specs]

    def nxt(k: int) -> ScenarioSpec:
        return pending[k].pop(0) if len(pending[k]) > 1 else pending[k][0]

    return nxt


def run_reference(spec: ScenarioSpec, steps: int) -> MultiRobotEnv:
    env = MultiRobotEnv()
    env.reset_spec(spec)
    for _ in range(steps):
        env.step_arrays(np.zeros(len(spec.robots), dtype=np.int64))
    return env


# ------------------------------------------------------------------------------- R1, R2, R6
def test_r1_truncated_copy_exposes_the_final_and_the_reset_state() -> None:
    first, second = small(10006, "obstacles", robots=2, steps=3), small(10007, "obstacles", 3, 9)
    venv = SyncVectorEnv([MultiRobotEnv()], queue((first, second)), with_state=True)
    venv.reset()
    for _ in range(3):
        res = venv.step([np.zeros(len(o), np.int64) for o in venv.obs])
    assert res.transitions[0].truncated and res.final_states is not None and res.states is not None
    final, after = res.final_states[0], res.states[0]
    assert final is not None and final[T_INDEX] == pytest.approx(1.0)
    assert after[T_INDEX] == pytest.approx(0.0)
    np.testing.assert_array_equal(final, run_reference(first, 3).global_state())
    fresh = MultiRobotEnv()
    fresh.reset_spec(second)
    np.testing.assert_array_equal(after, fresh.global_state())
    assert not np.array_equal(final, after)


def test_r2_terminated_copy_has_no_final_state() -> None:
    venv = SyncVectorEnv([MultiRobotEnv()], queue(hand_spec(3)), with_state=True)
    venv.reset()
    for _ in range(3):
        res = venv.step([follow(o, m)[0] for o, m in zip(venv.obs, venv.masks, strict=True)])
    assert res.transitions[0].terminated
    assert res.final_states == [None]


def test_r6_default_vector_env_is_the_issue5_path() -> None:
    specs = (small(10006, "obstacles", robots=2, steps=3), small(10007, "obstacles", 3, 9))
    plain = SyncVectorEnv([MultiRobotEnv()], queue(specs))
    stated = SyncVectorEnv([MultiRobotEnv()], queue(specs), with_state=True)
    for v in (plain, stated):
        v.reset()
    for _ in range(5):
        acts = [np.zeros(len(o), np.int64) for o in plain.obs]
        a, b = plain.step(acts), stated.step(acts)
        assert a.states is None and a.final_states is None and plain.states == []
        assert all(np.array_equal(x, y) for x, y in zip(a.obs, b.obs, strict=True))
        for ta, tb in zip(a.transitions, b.transitions, strict=True):
            assert np.array_equal(ta.next_obs, tb.next_obs)
            assert np.array_equal(ta.rewards, tb.rewards)
            assert (ta.terminated, ta.truncated) == (tb.terminated, tb.truncated)


# ------------------------------------------------------------------------------- R3, R4
def test_r3_mappo_bootstrap_uses_the_final_global_state() -> None:
    """Stub critic V = t/T of φ(s): the truncation bootstrap must be exactly 1.0 (a leaked reset
    state would give 0.0)."""
    first, second = small(10006, "obstacles", robots=2, steps=3), small(10008, "obstacles", 2, 9)
    venv = SyncVectorEnv([MultiRobotEnv()], queue((first, second)), with_state=True)
    venv.reset()

    def value(obs: np.ndarray, states: np.ndarray | None) -> np.ndarray:
        assert states is not None and len(states) == len(obs)
        return states[:, T_INDEX].astype(np.float32)

    buf = RolloutBuffer(5, 1, 356, state_dim=global_state_size(8))
    collect_rollout(venv, buf, stay, value, centralised=True)
    assert buf.truncated[2, 0] and not buf.truncated[:2, 0].any()
    assert buf.boot_values[2, 0, :2].tolist() == [1.0, 1.0]
    lv, valid = last_values(venv, buf, value, centralised=True)
    buf.compute_gae(lv, valid, G, L)
    r = buf.rewards[2, 0, :2]
    expected = r + G * 1.0 - np.float32(2 / 3)
    np.testing.assert_allclose(buf.advantages[2, 0, :2], expected, rtol=1e-6)
    assert buf.states is not None and buf.states[3, 0, T_INDEX] == 0.0  # the new episode


def test_r4_ippo_bootstrap_uses_the_true_final_observation() -> None:
    first, second = small(10006, "obstacles", robots=2, steps=3), small(10008, "obstacles", 2, 9)
    venv = SyncVectorEnv([MultiRobotEnv()], queue((first, second)))
    venv.reset()

    def value(obs: np.ndarray, states: np.ndarray | None) -> np.ndarray:
        assert states is None
        return obs.astype(np.float64).sum(axis=1).astype(np.float32)

    buf = RolloutBuffer(5, 1, 356)
    collect_rollout(venv, buf, stay, value, centralised=False)
    ref = run_reference(first, 3)
    final_obs = ref._obs
    assert final_obs is not None
    np.testing.assert_allclose(buf.boot_values[2, 0, :2], value(final_obs, None), rtol=1e-6)
    reset_obs = MultiRobotEnv().reset_spec(second)[0]
    assert not np.allclose(buf.boot_values[2, 0, :2], value(reset_obs, None))


def test_centralised_collection_requires_global_states() -> None:
    venv = SyncVectorEnv([MultiRobotEnv()], queue(small(10006, "obstacles", 2, 3)))
    venv.reset()
    with pytest.raises(ValueError, match="with_state"):
        collect_rollout(venv, RolloutBuffer(1, 1, 356, state_dim=259), stay,
                        lambda o, s: np.zeros(len(o), np.float32), centralised=True)  # fmt: skip


# ------------------------------------------------------------------------------- R5
def reference_gae(buf: RolloutBuffer, lv: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """An independent, segment-based GAE: split each (copy, slot) series into episodes."""
    out = np.zeros_like(buf.advantages, dtype=np.float64)
    for k in range(buf.envs):
        ends = [t for t in range(buf.steps) if buf.terminated[t, k] or buf.truncated[t, k]]
        bounds = [-1, *ends] + ([] if ends and ends[-1] == buf.steps - 1 else [buf.steps - 1])
        for a, b in itertools.pairwise(bounds):
            seg = range(a + 1, b + 1)
            n = int(buf.valid[a + 1, k].sum())
            for i in range(n):
                acc = 0.0
                for t in reversed(seg):
                    if t == b and buf.terminated[t, k]:
                        nv = 0.0
                    elif t == b and buf.truncated[t, k]:
                        nv = float(buf.boot_values[t, k, i])
                    elif t == b:
                        nv = float(lv[k, i])
                    else:
                        nv = float(buf.values[t + 1, k, i])
                    delta = float(buf.rewards[t, k, i]) + gamma * nv - float(buf.values[t, k, i])
                    acc = delta + (gamma * lam * acc if t != b else 0.0)
                    out[t, k, i] = acc
    return out


def test_r5_mixed_ends_team_sizes_and_resets() -> None:
    specs = [
        (hand_spec(3), hand_spec(3)),  # 1 robot, terminates at t = 3, twice
        (small(10006, "obstacles", robots=2, steps=4), small(10007, "obstacles", 5, 9)),
        small(10008, "obstacles", robots=4, steps=2),  # truncates every 2 steps
    ]
    venv = SyncVectorEnv([MultiRobotEnv() for _ in specs], queue(*specs), with_state=True)
    venv.reset()

    def value(obs: np.ndarray, states: np.ndarray | None) -> np.ndarray:
        assert states is not None
        return (0.1 * obs[:, :20].sum(axis=1) + states[:, T_INDEX]).astype(np.float32)

    buf = RolloutBuffer(7, 3, 356, state_dim=global_state_size(8))
    stats = collect_rollout(venv, buf, follow, value, centralised=True)
    assert buf.terminated[:, 0].tolist() == [False, False, True, False, False, True, False]
    assert buf.truncated[:, 1].tolist() == [False, False, False, True, False, False, False]
    assert buf.truncated[:, 2].tolist() == [False, True, False, True, False, True, False]
    # validity: one robot in copy 0; 2 then 5 robots in copy 1; always 4 in copy 2
    assert buf.valid[:, 0].sum(axis=1).tolist() == [1] * 7
    assert buf.valid[:, 1].sum(axis=1).tolist() == [2, 2, 2, 2, 5, 5, 5]
    assert buf.valid[:, 2].sum(axis=1).tolist() == [4] * 7
    assert stats.env_steps == 21 and stats.robot_steps == int(buf.valid.sum())
    assert len(stats.episodes) == 2 + 1 + 3
    lv, valid = last_values(venv, buf, value, centralised=True)
    buf.compute_gae(lv, valid, G, L)
    np.testing.assert_allclose(buf.advantages, reference_gae(buf, lv, G, L), rtol=1e-5, atol=1e-6)
    assert np.all(buf.advantages[~buf.valid] == 0)
    # every truncation bootstrap saw a horizon state (t/T = 1)
    for t, k in zip(*np.nonzero(buf.truncated), strict=True):
        n = int(buf.valid[t, k].sum())
        assert np.all(buf.boot_values[t, k, :n] >= 1.0)
