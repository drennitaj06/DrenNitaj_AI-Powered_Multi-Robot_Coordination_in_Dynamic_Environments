"""On-policy rollout storage and generalised advantage estimation for PPO (algorithms §7; Issue #6).
NumPy only.

Layout: padded arrays ``[T_r, E, N_max]`` (time, environment copy, robot slot). Team sizes differ
between copies and change at auto-resets; ``valid[t, k, i]`` marks the slots that hold a robot of
the episode running in copy ``k`` at step ``t``. Per (copy, slot) the samples form a *stream*.

GAE per stream, computed backwards in ``O(T_r)``:

* ``δ_t = r_t + γ V_next(t) − V_t`` with ``V_next(t)`` equal to
  - ``0`` if the episode **terminated** at ``t`` (every robot is idle, ``Φ = 0``);
  - ``V(final observation, final global state)`` if it was **truncated** at ``t`` — the value of
    the look-ahead state at the horizon, stored as ``boot_values[t]`` before the auto-reset;
  - ``V_{t+1}`` of the same stream otherwise (``last_values`` at the end of the rollout);
* ``Â_t = δ_t + γλ (1 − end_t) Â_{t+1}`` with ``end_t`` = terminated **or** truncated at ``t``, so
  a stream never crosses an auto-reset (the slot may hold a different robot afterwards);
* ``R̂_t = Â_t + V_t``.

Invalid slots get ``Â = R̂ = 0`` and are excluded from every loss by :meth:`flatten`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from robocoord.marl.config import N_ACTIONS
from robocoord.marl.vector_env import SyncVectorEnv

F32 = npt.NDArray[np.float32]
#: ``act(obs, masks) -> (actions, log-probabilities)`` for a batch of robots.
ActFn = Callable[[F32, npt.NDArray[np.bool_]], tuple[npt.NDArray[np.int64], F32]]
#: ``value(obs, states) -> values``; ``states`` is ``None`` for a local (IPPO) critic and holds
#: one global-state row per robot (its copy's ``φ(s)``) for a centralised (MAPPO) critic.
ValueFn = Callable[[F32, F32 | None], F32]


@dataclass(frozen=True, slots=True)
class Samples:
    """The valid samples of one rollout, flattened (row order: time, copy, slot)."""

    obs: F32  # (n, d_o)
    masks: npt.NDArray[np.bool_]  # (n, 5)
    actions: npt.NDArray[np.int64]  # (n,)
    logp: F32  # (n,) log-probabilities under the collecting policy
    values: F32  # (n,) critic values at collection
    advantages: F32  # (n,)
    returns: F32  # (n,)
    states: F32 | None  # (n, d_s) global state of each sample's step (MAPPO), else None

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class RolloutBuffer:
    """Padded ``[T_r, E, N_max]`` storage of one PPO iteration."""

    def __init__(
        self,
        steps: int,
        envs: int,
        obs_dim: int,
        *,
        max_robots: int = 8,
        state_dim: int | None = None,
    ) -> None:
        if steps < 1 or envs < 1 or max_robots < 1:
            raise ValueError("steps, envs and max_robots must be >= 1")
        self.steps, self.envs, self.max_robots = steps, envs, max_robots
        self.obs_dim, self.state_dim = obs_dim, state_dim
        shape = (steps, envs, max_robots)
        self.obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.masks = np.zeros((*shape, N_ACTIONS), dtype=bool)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.logp = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.boot_values = np.zeros(shape, dtype=np.float32)
        self.valid = np.zeros(shape, dtype=bool)
        self.terminated = np.zeros((steps, envs), dtype=bool)
        self.truncated = np.zeros((steps, envs), dtype=bool)
        self.states = (
            None if state_dim is None else np.zeros((steps, envs, state_dim), dtype=np.float32)
        )
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self._filled = np.zeros((steps, envs), dtype=bool)

    @property
    def nbytes(self) -> int:
        arrays = [self.obs, self.masks, self.actions, self.logp, self.values, self.rewards,
                  self.boot_values, self.valid, self.terminated, self.truncated, self.advantages,
                  self.returns]  # fmt: skip
        return sum(a.nbytes for a in arrays) + (0 if self.states is None else self.states.nbytes)

    def add(
        self,
        t: int,
        k: int,
        *,
        obs: F32,
        masks: npt.NDArray[np.bool_],
        actions: npt.NDArray[np.int64],
        logp: F32,
        values: F32,
        rewards: npt.ArrayLike,
        terminated: bool,
        truncated: bool,
        boot_values: F32 | None = None,
        state: F32 | None = None,
    ) -> None:
        """Store copy ``k``'s step ``t``: the observation the actions were chosen from, the
        actions, their log-probabilities and values, and the transition's rewards and end flags.
        ``boot_values`` (the critic's values of the final state) is required at truncation."""
        n = obs.shape[0]
        if n > self.max_robots:
            raise ValueError(f"{n} robots exceed max_robots = {self.max_robots}")
        if terminated and truncated:
            raise ValueError("a transition is either terminated or truncated, not both")
        if truncated and boot_values is None:
            raise ValueError("a truncated transition needs the bootstrap values of its final state")
        if (self.states is None) != (state is None):
            raise ValueError("pass a global state exactly when the buffer stores them (MAPPO)")
        self.obs[t, k, :n] = obs
        self.obs[t, k, n:] = 0.0
        self.masks[t, k, :n] = masks
        self.masks[t, k, n:] = False
        self.actions[t, k, :n] = actions
        self.logp[t, k, :n] = logp
        self.values[t, k, :n] = values
        self.rewards[t, k, :n] = np.asarray(rewards, dtype=np.float32)
        self.valid[t, k] = False
        self.valid[t, k, :n] = True
        self.terminated[t, k] = terminated
        self.truncated[t, k] = truncated
        self.boot_values[t, k] = 0.0
        if truncated:
            assert boot_values is not None
            self.boot_values[t, k, :n] = boot_values
        if self.states is not None:
            self.states[t, k] = state
        self._filled[t, k] = True

    def compute_gae(
        self, last_values: F32, last_valid: npt.NDArray[np.bool_], gamma: float, lam: float
    ) -> None:
        """Fill ``advantages`` and ``returns``. ``last_values[k, i]`` is the critic's value of the
        observation the next action of copy ``k`` would be chosen from (after the rollout)."""
        if not self._filled.all():
            raise ValueError("the rollout is not complete")
        adv_next = np.zeros((self.envs, self.max_robots), dtype=np.float64)
        v_next = np.asarray(last_values, dtype=np.float64)
        valid_next = np.asarray(last_valid, dtype=bool)
        for t in range(self.steps - 1, -1, -1):
            term = self.terminated[t][:, None]
            trunc = self.truncated[t][:, None]
            end = term | trunc
            valid = self.valid[t]
            # a running stream continues in the same slot of the next step
            if np.any(valid & ~end & ~valid_next):
                raise ValueError(f"stream discontinuity at step {t}: a running robot vanished")
            bootstrap = np.where(term, 0.0, np.where(trunc, self.boot_values[t], v_next))
            delta = self.rewards[t] + gamma * bootstrap - self.values[t]
            adv = delta + gamma * lam * np.where(end, 0.0, adv_next)
            adv = np.where(valid, adv, 0.0)
            self.advantages[t] = adv
            self.returns[t] = np.where(valid, adv + self.values[t], 0.0)
            adv_next = adv
            v_next = self.values[t].astype(np.float64)
            valid_next = valid

    def flatten(self) -> Samples:
        """The valid samples, in (time, copy, slot) order."""
        v = self.valid
        states = None
        if self.states is not None:
            states = np.broadcast_to(self.states[:, :, None, :], (*v.shape, self.states.shape[-1]))[
                v
            ]
        return Samples(
            obs=self.obs[v],
            masks=self.masks[v],
            actions=self.actions[v],
            logp=self.logp[v],
            values=self.values[v],
            advantages=self.advantages[v],
            returns=self.returns[v],
            states=None if states is None else np.ascontiguousarray(states),
        )


@dataclass(slots=True)
class CollectStats:
    """Bookkeeping of one collection: finished episodes and step counts."""

    episodes: list[dict[str, Any]] = field(default_factory=list)
    env_steps: int = 0  # joint steps summed over the copies
    robot_steps: int = 0  # robot actions taken (valid samples)


def _repeat_states(states: list[F32] | None, sizes: list[int]) -> F32 | None:
    if states is None:
        return None
    return np.concatenate(
        [np.repeat(s[None, :], n, axis=0) for s, n in zip(states, sizes, strict=True)]
    )


def collect_rollout(
    venv: SyncVectorEnv,
    buffer: RolloutBuffer,
    act: ActFn,
    value: ValueFn,
    *,
    centralised: bool,
) -> CollectStats:
    """Fill ``buffer`` with ``buffer.steps`` joint steps of every copy of ``venv`` and compute
    nothing else (call :meth:`RolloutBuffer.compute_gae` with :func:`last_values` afterwards).

    With ``centralised=True`` (MAPPO) the vector environment must have ``with_state=True``: the
    critic sees each robot's observation with its copy's ``φ(s)``, and a truncated copy is
    bootstrapped with its **final** observation and **final** global state, captured before the
    auto-reset; the post-reset state is never used for it. A terminated copy bootstraps with 0.
    """
    if centralised and not venv.with_state:
        raise ValueError("a centralised critic needs SyncVectorEnv(with_state=True)")
    if buffer.envs != venv.num_envs:
        raise ValueError("the buffer and the vector environment disagree on the number of copies")
    stats = CollectStats()
    for t in range(buffer.steps):
        obs, masks = list(venv.obs), list(venv.masks)
        states = list(venv.states) if centralised else None
        sizes = [len(o) for o in obs]
        cat_obs = np.concatenate(obs)
        actions, logp = act(cat_obs, np.concatenate(masks))
        values = value(cat_obs, _repeat_states(states, sizes))
        cuts = np.cumsum(sizes)[:-1]
        a_split = np.split(actions, cuts)
        l_split, v_split = np.split(logp, cuts), np.split(values, cuts)
        res = venv.step(a_split)
        for k, tr in enumerate(res.transitions):
            boot = None
            if tr.truncated:
                final_state = None
                if centralised:
                    assert res.final_states is not None
                    final_state = res.final_states[k]
                    assert final_state is not None
                    final_state = np.repeat(final_state[None, :], sizes[k], axis=0)
                boot = value(tr.next_obs, final_state)
            buffer.add(
                t,
                k,
                obs=obs[k],
                masks=masks[k],
                actions=a_split[k],
                logp=l_split[k],
                values=v_split[k],
                rewards=tr.rewards,
                terminated=tr.terminated,
                truncated=tr.truncated,
                boot_values=boot,
                state=None if states is None else states[k],
            )
        stats.episodes.extend(res.episodes)
        stats.env_steps += venv.num_envs
        stats.robot_steps += int(sum(sizes))
    return stats


def last_values(
    venv: SyncVectorEnv, buffer: RolloutBuffer, value: ValueFn, *, centralised: bool
) -> tuple[F32, npt.NDArray[np.bool_]]:
    """Critic values of the observations the next actions would be chosen from, padded
    ``[E, N_max]``, and their validity."""
    sizes = [len(o) for o in venv.obs]
    states = list(venv.states) if centralised else None
    vals = value(np.concatenate(venv.obs), _repeat_states(states, sizes))
    out = np.zeros((venv.num_envs, buffer.max_robots), dtype=np.float32)
    valid = np.zeros((venv.num_envs, buffer.max_robots), dtype=bool)
    for k, part in enumerate(np.split(vals, np.cumsum(sizes)[:-1])):
        out[k, : sizes[k]] = part
        valid[k, : sizes[k]] = True
    return out, valid
