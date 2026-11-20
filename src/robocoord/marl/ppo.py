"""PPO for multiple robots: MAPPO (centralised critic) and IPPO (local critic) in one core
(ADR-4; algorithms §7; Issue #6). Requires PyTorch.

* **Actor** ``π_φ(a | o_i)``: one MLP ``356 → 256 → 256 → 5`` shared by every robot (no robot
  identity, so team sizes may vary), orthogonal initialisation (hidden gain √2, output gain 0.01).
  Masked actions get the logit ``finfo.min``; the policy is the categorical distribution over the
  allowed actions, and its entropy is taken over the allowed actions only.
* **Critic** ``V_ψ``: one MLP shared by every robot (orthogonal, hidden gain √2, output gain 1).
  MAPPO feeds ``[o_i, φ(s)]`` (615 inputs: the observation and the global state of model §6.2,
  which contains ``t / T``); IPPO feeds ``o_i`` (356). The switch changes only the critic input
  and its input width; the actor is identical. Execution needs the actor only (CTDE).
* **Loss** per minibatch (advantages normalised over the minibatch's valid samples):
  ``−E[min(ρ Â, clip(ρ, 1 ± ε_c) Â)] + c_v E[(V − R̂)²] − c_e E[H]``.
* **Update**: ``K`` epochs over shuffled minibatches, Adam (one optimiser, actor and critic
  parameter groups), the gradient norms of actor and critic clipped **separately** (D5), and an
  early stop of the iteration once a minibatch's approximate KL exceeds ``target_kl``.

The approximate KL is the non-negative estimator ``E[(ρ − 1) − log ρ]``. No monotonic-improvement
guarantee is claimed (G16).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from robocoord.core.errors import ConfigError
from robocoord.marl.config import N_ACTIONS
from robocoord.marl.global_state import global_state_size
from robocoord.marl.networks import build_mlp
from robocoord.marl.policy import sample_masked
from robocoord.marl.rollout import Samples

VARIANTS = ("mappo", "ippo")


@dataclass(frozen=True, slots=True)
class PPOParams:
    """Hyper-parameters (algorithms §7 defaults; ``rollout`` = 125 and separate clipping are the
    approved Issue #6 decisions D4 and D5)."""

    variant: str = "mappo"
    n_envs: int = 8
    rollout: int = 125
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    target_kl: float | None = 0.02
    actor_grad_clip: float = 0.5
    critic_grad_clip: float = 0.5
    hidden: tuple[int, ...] = field(default=(256, 256))
    critic_hidden: tuple[int, ...] = field(default=(256, 256))
    global_blocks: int = 8
    max_robots: int = 8

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ConfigError("ppo.variant", f"expected one of {VARIANTS}, got {self.variant!r}")
        ints = ("n_envs", "rollout", "epochs", "minibatches", "global_blocks", "max_robots")
        for name in ints:
            if getattr(self, name) < 1:
                raise ConfigError(f"ppo.{name}", "must be >= 1")
        for name in ("lr", "adam_eps", "clip", "actor_grad_clip", "critic_grad_clip"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"ppo.{name}", "must be > 0")
        for name in ("vf_coef", "ent_coef"):
            if getattr(self, name) < 0:
                raise ConfigError(f"ppo.{name}", "must be >= 0")
        if not 0.0 < self.gamma < 1.0 or not 0.0 <= self.lam <= 1.0:
            raise ConfigError("ppo.gamma", "need 0 < gamma < 1 and 0 <= lam <= 1")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ConfigError("ppo.target_kl", "must be > 0 or omitted")
        for name in ("hidden", "critic_hidden"):
            if not getattr(self, name) or any(h < 1 for h in getattr(self, name)):
                raise ConfigError(f"ppo.{name}", "need at least one positive layer width")

    @property
    def centralised(self) -> bool:
        return self.variant == "mappo"

    @property
    def iteration_steps(self) -> int:
        """Environment steps per iteration, summed over the copies (D3)."""
        return self.n_envs * self.rollout

    def critic_input(self, obs_dim: int) -> int:
        return obs_dim + (global_state_size(self.global_blocks) if self.centralised else 0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["critic_hidden"] = list(self.critic_hidden)
        return d


def orthogonal_init(net: nn.Sequential, output_gain: float) -> nn.Sequential:
    """Orthogonal weights (gain √2 for hidden layers, ``output_gain`` for the last), zero biases."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears):
        gain = output_gain if i == len(linears) - 1 else math.sqrt(2.0)
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    return net


def build_actor(obs_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    return orthogonal_init(build_mlp(obs_dim, hidden, N_ACTIONS), 0.01)


def build_critic(in_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    return orthogonal_init(build_mlp(in_dim, hidden, 1), 1.0)


# ------------------------------------------------------------------ masked categorical
def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return torch.where(masks, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))


def masked_log_probs(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of every action; masked actions get ``-inf``-like values."""
    return torch.log_softmax(masked_logits(logits, masks), dim=-1)


def masked_entropy(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """``−Σ_{a allowed} π(a) log π(a)`` per row (masked actions contribute exactly 0)."""
    logp = masked_log_probs(logits, masks)
    p = logp.exp()
    return -torch.where(masks, p * logp, torch.zeros_like(logp)).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class LossTerms:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def normalise(adv: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit standard deviation over the (valid) samples given."""
    if adv.numel() < 2:
        return adv - adv.mean()
    return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def ppo_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    params: PPOParams,
) -> LossTerms:
    """The clipped surrogate, value and entropy terms of one minibatch of valid samples."""
    logp_all = masked_log_probs(logits, masks)
    logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    adv = normalise(advantages)
    surrogate = torch.min(ratio * adv, ratio.clamp(1.0 - params.clip, 1.0 + params.clip) * adv)
    policy = -surrogate.mean()
    value = ((values - returns) ** 2).mean()
    entropy = masked_entropy(logits, masks).mean()
    total = policy + params.vf_coef * value - params.ent_coef * entropy
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > params.clip).float().mean()
    return LossTerms(total, policy, value, entropy, approx_kl, clip_fraction)


def explained_variance(values: npt.ArrayLike, returns: npt.ArrayLike) -> float | None:
    """``1 − Var(R̂ − V) / Var(R̂)``; undefined (``None``) when the returns are constant."""
    v = np.asarray(values, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    var = float(r.var())
    return None if var == 0.0 else 1.0 - float((r - v).var()) / var


class PPO:
    """Learner state: shared actor, shared critic (MAPPO or IPPO input), one Adam optimiser."""

    algorithm = "ppo"

    def __init__(self, obs_dim: int, params: PPOParams | None = None) -> None:
        self.params = params or PPOParams()
        self.obs_dim = obs_dim
        self.critic_dim = self.params.critic_input(obs_dim)
        self.actor = build_actor(obs_dim, self.params.hidden)
        self.critic = build_critic(self.critic_dim, self.params.critic_hidden)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=self.params.lr,
            eps=self.params.adam_eps,
        )
        self.updates = 0  # minibatch gradient steps

    @property
    def variant(self) -> str:
        return self.params.variant

    @property
    def variant_description(self) -> str:
        critic = "centralised critic [o_i, φ(s)]" if self.params.centralised else "local critic o_i"
        return f"{self.variant.upper()}: shared masked-categorical actor, shared {critic}"

    # ------------------------------------------------------------------ acting
    def critic_inputs(self, obs: torch.Tensor, states: torch.Tensor | None) -> torch.Tensor:
        if self.params.centralised:
            if states is None:
                raise ValueError("the MAPPO critic needs the global state of every sample")
            return torch.cat([obs, states], dim=1)
        return obs

    @torch.no_grad()
    def act(
        self,
        obs: npt.NDArray[np.float32],
        masks: npt.NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Sample one action per robot from the masked policy; returns actions and their
        log-probabilities (computed exactly as in the update)."""
        o = torch.from_numpy(np.ascontiguousarray(obs))
        m = torch.from_numpy(np.ascontiguousarray(masks))
        logits = self.actor(o)
        actions = sample_masked(logits.numpy().astype(np.float64), masks, rng)
        logp = masked_log_probs(logits, m).gather(1, torch.from_numpy(actions).unsqueeze(1))
        return actions, logp.squeeze(1).numpy().astype(np.float32)

    @torch.no_grad()
    def value(
        self, obs: npt.NDArray[np.float32], states: npt.NDArray[np.float32] | None
    ) -> npt.NDArray[np.float32]:
        o = torch.from_numpy(np.ascontiguousarray(obs))
        s = None if states is None else torch.from_numpy(np.ascontiguousarray(states))
        out: npt.NDArray[np.float32] = self.critic(self.critic_inputs(o, s)).squeeze(1).numpy()
        return out

    # ------------------------------------------------------------------ learning
    def update(self, samples: Samples, rng: np.random.Generator) -> dict[str, Any]:
        """``K`` epochs of shuffled minibatch updates on one rollout's valid samples."""
        p = self.params
        n = len(samples)
        if n == 0:
            raise ValueError("no valid samples to learn from")
        obs = torch.from_numpy(samples.obs)
        masks = torch.from_numpy(samples.masks)
        actions = torch.from_numpy(samples.actions)
        old_logp = torch.from_numpy(samples.logp)
        adv = torch.from_numpy(samples.advantages)
        ret = torch.from_numpy(samples.returns)
        states = None if samples.states is None else torch.from_numpy(samples.states)
        crit_in = self.critic_inputs(obs, states)
        logs: dict[str, list[float]] = {k: [] for k in (
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
            "actor_grad_norm", "critic_grad_norm")}  # fmt: skip
        epochs_run, stopped = 0, False
        for _ in range(p.epochs):
            perm = rng.permutation(n)
            for idx in np.array_split(perm, p.minibatches):
                if idx.size == 0:
                    continue
                i = torch.from_numpy(idx)
                terms = ppo_loss(
                    self.actor(obs[i]),
                    masks[i],
                    actions[i],
                    old_logp[i],
                    adv[i],
                    self.critic(crit_in[i]).squeeze(1),
                    ret[i],
                    p,
                )
                self.optimizer.zero_grad()
                terms.total.backward()  # type: ignore[no-untyped-call]
                g_actor = nn.utils.clip_grad_norm_(self.actor.parameters(), p.actor_grad_clip)
                g_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), p.critic_grad_clip)
                self.optimizer.step()
                self.updates += 1
                for key, val in (
                    ("policy_loss", terms.policy), ("value_loss", terms.value),
                    ("entropy", terms.entropy), ("approx_kl", terms.approx_kl),
                    ("clip_fraction", terms.clip_fraction), ("actor_grad_norm", g_actor),
                    ("critic_grad_norm", g_critic),
                ):  # fmt: skip
                    logs[key].append(float(val.detach()))
                if p.target_kl is not None and float(terms.approx_kl) > p.target_kl:
                    stopped = True
                    break
            epochs_run += 1
            if stopped:
                break
        out: dict[str, Any] = {k: float(np.mean(v)) for k, v in logs.items() if v}
        out.update(
            epochs_run=epochs_run,
            kl_early_stop=stopped,
            minibatch_updates=len(logs["policy_loss"]),
            samples=n,
            explained_variance=explained_variance(samples.values, samples.returns),
            lr=p.lr,
        )
        return out

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "params": self.params.to_dict(),
            "obs_dim": self.obs_dim,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["params"] != self.params.to_dict() or state["obs_dim"] != self.obs_dim:
            raise ValueError("the saved PPO state belongs to a different configuration")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
