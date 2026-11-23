"""MADQL: parameter-shared independent Double DQN with team-mixed rewards and action masking
(ADR-3; algorithms §6). Requires PyTorch.

* one Q-network shared by all robots, a target network, a common replay of per-robot transitions;
* masked ε-greedy action selection (uniform over allowed actions with probability ε, D5);
* Double-DQN target ``y = r + γ (1 − d) Q_θ⁻(o', argmax_{a' ∈ m'} Q_θ(o', a'))`` with ``d = 1``
  only for termination (truncation bootstraps), Huber loss, Adam, gradient-norm clip.

Issue 6A Phase 4 (optional, off by default): ``n_step`` > 1 replaces the one-step target by the
n-step Double-DQN target ``y = Σ_{k<h} γ^k r_{t+k} + γ^h (1 − d) Q_θ⁻(o_{t+h}, argmax …)`` with
``h = n`` or, when the episode ends earlier, the number of remaining steps
(``replay.NStepAccumulator``).
With ``n_step = 1`` (the default) every computation, the configuration dictionary and its hash
are exactly those of Issue #5.

No convergence guarantee is claimed (G16): the other robots' policies change while learning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from robocoord.core.errors import ConfigError
from robocoord.marl.config import N_ACTIONS
from robocoord.marl.networks import build_mlp, masked_argmax
from robocoord.marl.replay import Batch


@dataclass(frozen=True, slots=True)
class MADQLParams:
    """Hyper-parameters (initial values of algorithms §6). Step counts are environment steps
    summed over all vectorised copies (D3); ``warmup`` counts stored transitions."""

    lr: float = 5e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup: int = 10_000
    train_freq: int = 4
    target_period: int = 2_000
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    grad_clip: float = 10.0
    hidden: tuple[int, ...] = field(default=(256, 256))
    n_step: int = 1  # Issue 6A Phase 4: n-step return length; 1 = the Issue #5 one-step target

    def __post_init__(self) -> None:
        positive = ("lr", "batch_size", "buffer_size", "train_freq", "target_period", "grad_clip")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ConfigError(f"madql.{name}", "must be > 0")
        if self.warmup < 0:
            raise ConfigError("madql.warmup", "must be >= 0")
        if not 0.0 < self.gamma < 1.0:
            raise ConfigError("madql.gamma", "must be in (0, 1)")
        if not 0.0 <= self.eps_end <= self.eps_start <= 1.0:
            raise ConfigError("madql.eps", "need 0 <= eps_end <= eps_start <= 1")
        if not 0.0 < self.eps_fraction <= 1.0:
            raise ConfigError("madql.eps_fraction", "must be in (0, 1]")
        if not self.hidden or any(h < 1 for h in self.hidden):
            raise ConfigError("madql.hidden", "need at least one positive layer width")
        if isinstance(self.n_step, bool) or not isinstance(self.n_step, int) or self.n_step < 1:
            raise ConfigError("madql.n_step", "must be an integer >= 1")

    def epsilon(self, step: int, total: int) -> float:
        """Linear decay from ``eps_start`` to ``eps_end`` over ``eps_fraction`` of ``total``."""
        horizon = max(1.0, self.eps_fraction * total)
        frac = min(1.0, step / horizon)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        if self.n_step == 1:  # default: the Issue #5 dictionary (and config hash) unchanged
            del d["n_step"]
        return d


def masked_epsilon_greedy(
    q: npt.NDArray[np.float32],
    masks: npt.NDArray[np.bool_],
    epsilon: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Per row: a uniformly random *allowed* action with probability ``ε``, else the masked
    arg-max (lowest index on ties). Masked actions are never returned."""
    n = q.shape[0]
    if not masks.any(axis=1).all():
        raise ValueError("every row needs at least one allowed action")
    greedy = np.where(masks, q, -np.inf).argmax(axis=1)
    explore = rng.random(n) < epsilon
    out: npt.NDArray[np.int64] = np.asarray(greedy, dtype=np.int64)
    for i in np.flatnonzero(explore):
        allowed = np.flatnonzero(masks[i])
        out[i] = int(allowed[rng.integers(len(allowed))])
    return out


def double_dqn_targets(
    q_next_online: torch.Tensor,
    q_next_target: torch.Tensor,
    next_masks: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float | torch.Tensor,
) -> torch.Tensor:
    """``r + γ (1 − d) Q_target(o', argmax_{a' ∈ m'} Q_online(o', a'))``. ``gamma`` may be a
    per-row tensor (``γ^h`` of the n-step target); ``rewards`` are then the n-step sums."""
    best = masked_argmax(q_next_online, next_masks)
    q_sel = q_next_target.gather(1, best.unsqueeze(1)).squeeze(1)
    return rewards + gamma * (1.0 - dones) * q_sel


class MADQL:
    """Learner state: online and target networks, optimiser, update counter."""

    algorithm = "madql"
    variant = "parameter-shared independent Double DQN, team-mixed rewards, action masking"

    def __init__(self, obs_dim: int, params: MADQLParams | None = None) -> None:
        self.params = params or MADQLParams()
        self.obs_dim = obs_dim
        self.q: nn.Sequential = build_mlp(obs_dim, self.params.hidden, N_ACTIONS)
        self.target: nn.Sequential = build_mlp(obs_dim, self.params.hidden, N_ACTIONS)
        self.target.load_state_dict(self.q.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.params.lr)
        self.updates = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def q_values(self, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        out: npt.NDArray[np.float32] = self.q(torch.from_numpy(np.ascontiguousarray(obs))).numpy()
        return out

    def act(
        self,
        obs: npt.NDArray[np.float32],
        masks: npt.NDArray[np.bool_],
        epsilon: float,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        return masked_epsilon_greedy(self.q_values(obs), masks, epsilon, rng)

    # ------------------------------------------------------------------ learning
    def loss(self, batch: Batch) -> torch.Tensor:
        obs = torch.from_numpy(batch.obs)
        nxt = torch.from_numpy(batch.next_obs)
        gamma: float | torch.Tensor = self.params.gamma
        if self.params.n_step > 1:
            if batch.horizons is None:
                raise ValueError("an n-step learner needs batches with horizons")
            gamma = torch.from_numpy(
                np.power(np.float32(self.params.gamma), batch.horizons).astype(np.float32)
            )
        with torch.no_grad():
            y = double_dqn_targets(
                self.q(nxt),
                self.target(nxt),
                torch.from_numpy(batch.next_masks),
                torch.from_numpy(batch.rewards),
                torch.from_numpy(batch.dones),
                gamma,
            )
        q = self.q(obs).gather(1, torch.from_numpy(batch.actions).unsqueeze(1)).squeeze(1)
        return nn.functional.smooth_l1_loss(q, y)

    def update(self, batch: Batch) -> float:
        loss = self.loss(batch)
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(self.q.parameters(), self.params.grad_clip)
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.q.state_dict())

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Any]:
        return {
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "params": self.params.to_dict(),
            "obs_dim": self.obs_dim,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q.load_state_dict(state["q"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
