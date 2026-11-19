"""Replay memory of per-robot transitions (algorithms §6). NumPy only.

Layout (design D, target < 300 MB for 200 k transitions): window channels are stored as
``uint8`` — the 0/1 channels as they are, the priority channel as the integer level ``π`` — and
the 13 scalars as ``float32``. :class:`ObsCodec` restores the float observation **exactly**
(the builder computes ``π / P`` in float32 and the codec repeats that computation). About
2 × (343 + 52) + 5 + 10 bytes ≈ 0.8 kB per transition, ≈ 160 MB for 200 k.

FIFO overwrite once full; uniform sampling from a seeded generator.
"""

from __future__ import annotations

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
        self.pos = 0
        self.size = 0
        self.added = 0  # total transitions ever added

    def __len__(self) -> int:
        return self.size

    @property
    def nbytes(self) -> int:
        arrays = (self.grid, self.scal, self.next_grid, self.next_scal, self.actions,
                  self.rewards, self.next_masks, self.dones)  # fmt: skip
        return sum(a.nbytes for a in arrays)

    def add(
        self,
        obs: F32,
        actions: npt.ArrayLike,
        rewards: npt.ArrayLike,
        next_obs: F32,
        next_masks: npt.NDArray[np.bool_],
        dones: npt.ArrayLike,
    ) -> None:
        """Append a batch of transitions (oldest entries are overwritten once full)."""
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
        )
