"""Replay memory of per-robot transitions (algorithms §6). NumPy only.

Layout (design D, target < 300 MB for 200 k transitions): window channels are stored as
``uint8`` — the 0/1 channels as they are, the priority channel as the integer level ``π`` — and
the 13 scalars as ``float32``. :class:`ObsCodec` restores the float observation **exactly**
(the builder computes ``π / P`` in float32 and the codec repeats that computation). About
2 × (343 + 52) + 5 + 10 bytes ≈ 0.8 kB per transition, ≈ 160 MB for 200 k.

FIFO overwrite once full; uniform sampling from a seeded generator.

Issue 6A Phase 4: :class:`NStepAccumulator` (optional; used only when ``MADQLParams.n_step`` > 1)
turns the one-step transitions of each vectorised copy into n-step transitions
``(o_t, a_t, Σ_{k<h} γ^k r_{t+k}, o_{t+h}, m_{t+h}, d, h)``; the buffer then also stores the
horizon ``h`` (1 for every one-step transition, so the one-step path is unchanged).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robocoord.marl.config import CH_ROBOT_PRIO, N_ACTIONS, N_CHANNELS, N_SCALARS, ObsParams

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]


class ObsCodec:
    """Lossless compression of observation rows (see the module docstring)."""

    def __init__(self, params: ObsParams | None = None) -> None:
        self.params = params or ObsParams()
        k2 = self.params.window**2
        self.grid_size = self.params.grid_size
        self._prio = slice(CH_ROBOT_PRIO * k2, (CH_ROBOT_PRIO + 1) * k2)
        self._scale = np.float32(self.params.max_priority)
        assert self.grid_size == N_CHANNELS * k2

    def encode(self, obs: F32) -> tuple[U8, F32]:
        grid = obs[:, : self.grid_size].copy()
        grid[:, self._prio] = np.rint(grid[:, self._prio] * self._scale)
        return grid.astype(np.uint8), obs[:, self.grid_size :].astype(np.float32)

    def decode(self, grid: U8, scal: F32) -> F32:
        out = np.empty((grid.shape[0], self.grid_size + N_SCALARS), dtype=np.float32)
        g = grid.astype(np.float32)
        g[:, self._prio] = g[:, self._prio] / self._scale
        out[:, : self.grid_size] = g
        out[:, self.grid_size :] = scal
        return out


@dataclass(frozen=True, slots=True)
class Batch:
    obs: F32
    actions: npt.NDArray[np.int64]
    rewards: npt.NDArray[np.float32]
    next_obs: F32
    next_masks: npt.NDArray[np.bool_]
    dones: npt.NDArray[np.float32]  # 1 only for termination, never for truncation
    horizons: npt.NDArray[np.float32] | None = None  # n-step horizon h of each row (Phase 4)


class ReplayBuffer:
    """Circular buffer of ``(o, a, r, o', m', d)`` with ``d = 1`` only for termination."""

    def __init__(self, capacity: int, params: ObsParams | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.codec = ObsCodec(params)
        g = self.codec.grid_size
        self.capacity = capacity
        self.grid = np.zeros((capacity, g), dtype=np.uint8)
        self.scal = np.zeros((capacity, N_SCALARS), dtype=np.float32)
        self.next_grid = np.zeros((capacity, g), dtype=np.uint8)
        self.next_scal = np.zeros((capacity, N_SCALARS), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, N_ACTIONS), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.uint8)
        self.horizons = np.ones(capacity, dtype=np.uint8)
        self.pos = 0
        self.size = 0
        self.added = 0  # total transitions ever added

    def __len__(self) -> int:
        return self.size

    @property
    def nbytes(self) -> int:
        arrays = (self.grid, self.scal, self.next_grid, self.next_scal, self.actions,
                  self.rewards, self.next_masks, self.dones, self.horizons)  # fmt: skip
        return sum(a.nbytes for a in arrays)

    def add(
        self,
        obs: F32,
        actions: npt.ArrayLike,
        rewards: npt.ArrayLike,
        next_obs: F32,
        next_masks: npt.NDArray[np.bool_],
        dones: npt.ArrayLike,
        horizons: npt.ArrayLike | None = None,
    ) -> None:
        """Append a batch of transitions (oldest entries are overwritten once full).
        ``horizons`` (n-step transitions only) defaults to 1."""
        b = obs.shape[0]
        g, s = self.codec.encode(obs)
        ng, ns = self.codec.encode(next_obs)
        idx = (self.pos + np.arange(b)) % self.capacity
        self.grid[idx], self.scal[idx] = g, s
        self.next_grid[idx], self.next_scal[idx] = ng, ns
        self.actions[idx] = np.asarray(actions, dtype=np.uint8)
        self.rewards[idx] = np.asarray(rewards, dtype=np.float32)
        self.next_masks[idx] = next_masks
        self.dones[idx] = np.asarray(dones, dtype=np.uint8)
        self.horizons[idx] = 1 if horizons is None else np.asarray(horizons, dtype=np.uint8)
        self.pos = int((self.pos + b) % self.capacity)
        self.size = min(self.capacity, self.size + b)
        self.added += b

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Uniform sample with replacement."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        return Batch(
            obs=self.codec.decode(self.grid[idx], self.scal[idx]),
            actions=self.actions[idx].astype(np.int64),
            rewards=self.rewards[idx].copy(),
            next_obs=self.codec.decode(self.next_grid[idx], self.next_scal[idx]),
            next_masks=self.next_masks[idx].copy(),
            dones=self.dones[idx].astype(np.float32),
            horizons=self.horizons[idx].astype(np.float32),
        )


@dataclass(frozen=True, slots=True)
class NStepBatch:
    """n-step transitions emitted by :class:`NStepAccumulator` for one copy."""

    obs: F32
    actions: npt.NDArray[np.int64]
    rewards: npt.NDArray[np.float32]
    next_obs: F32
    next_masks: npt.NDArray[np.bool_]
    dones: npt.NDArray[np.float32]
    horizons: npt.NDArray[np.int64]


class NStepAccumulator:
    """Per vectorised copy, a queue of the last ``n`` one-step transitions of the running episode
    (all robots of the copy together: the team is fixed within an episode).

    * while the episode runs, each step emits the oldest queued step with ``h = n``, the n-step
      reward sum and the newest next observation (``d = 0``);
    * when the episode ends, every queued step is emitted (oldest first) with ``h`` = its number
      of remaining steps, the final next observation and ``d = 1`` only for termination, so a
      truncated tail bootstraps from the look-ahead observation exactly as the one-step target.

    Every one-step transition is emitted exactly once, in time order per copy. Rewards are summed
    in float64 and stored as float32. NumPy only.
    """

    def __init__(self, n: int, gamma: float, copies: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        self.n = n
        self.gamma = gamma
        self.queues: list[deque[tuple[F32, npt.NDArray[np.int64], npt.NDArray[np.float64]]]] = [
            deque() for _ in range(copies)
        ]

    def pending(self, copy: int) -> int:
        return len(self.queues[copy])

    def push(
        self,
        copy: int,
        obs: F32,
        actions: npt.ArrayLike,
        rewards: npt.ArrayLike,
        next_obs: F32,
        next_masks: npt.NDArray[np.bool_],
        terminated: bool,
        truncated: bool,
    ) -> NStepBatch | None:
        """Add one step of copy ``copy``; returns the transitions that are complete now."""
        q = self.queues[copy]
        q.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                np.asarray(actions, dtype=np.int64).copy(),
                np.asarray(rewards, dtype=np.float64).copy(),
            )
        )
        if terminated or truncated:
            items = list(q)
            q.clear()
            return self._emit(items, next_obs, next_masks, float(terminated), len(items))
        if len(q) == self.n:
            items = list(q)
            q.popleft()
            return self._emit(items, next_obs, next_masks, 0.0, 1)
        return None

    def _emit(
        self,
        items: list[tuple[F32, npt.NDArray[np.int64], npt.NDArray[np.float64]]],
        next_obs: F32,
        next_masks: npt.NDArray[np.bool_],
        done: float,
        count: int,
    ) -> NStepBatch:
        """The first ``count`` items as transitions ending at ``next_obs``."""
        length = len(items)
        obs, acts, rews, hs = [], [], [], []
        for j in range(count):
            r = np.zeros_like(items[j][2])
            for k in range(length - 1, j - 1, -1):
                r = items[k][2] + self.gamma * r
            obs.append(items[j][0])
            acts.append(items[j][1])
            rews.append(r)
            hs.append(length - j)
        m = items[0][0].shape[0]
        return NStepBatch(
            obs=np.concatenate(obs),
            actions=np.concatenate(acts),
            rewards=np.concatenate(rews).astype(np.float32),
            next_obs=np.concatenate([next_obs] * count),
            next_masks=np.concatenate([next_masks] * count),
            dones=np.full(m * count, done, dtype=np.float32),
            horizons=np.repeat(np.asarray(hs, dtype=np.int64), m),
        )
