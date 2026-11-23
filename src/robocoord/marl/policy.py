"""The learned controller (architecture §5.6): a trained policy network acting on the observation
built from the view. Requires PyTorch.

It implements the ordinary :class:`~robocoord.control.base.Controller` protocol and is selectable
as ``madql`` (Issue #5, a Q-network, ``mlp-q-v1``) or ``ppo`` (Issue #6, an actor, ``mlp-pi-v1``)
with either planner and with the shield on or off. Its output is the *requested* action per robot:
the mask is respected, and the optional shield and the always-on arbiter of the engine may still
replace or cancel a request. It contains no rule-based fallback, so it stays distinguishable from
the baseline. The engine must provide the view extras ``{"paths", "goal_distance"}``.

Execution modes (Issue #6, D6):

* ``greedy`` (default, unchanged from Issue #5): masked arg-max of the network outputs, ties to the
  lowest action index. For a PPO actor this is the mode of the masked categorical policy.
* ``stochastic`` (an explicit evaluation mode for the stochastic diagnostic, never the default):
  for a Q-network, masked ε-greedy (uniform over allowed actions with probability ε, the MADQL
  training behaviour); for an actor, a sample from the masked categorical policy. The random
  stream is a NumPy generator supplied by the caller (seeded per episode by the evaluation).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from robocoord.core.types import Action
from robocoord.marl.config import ObsParams
from robocoord.marl.networks import masked_argmax
from robocoord.marl.observation import ObservationBuilder, action_mask_array
from robocoord.sim.view import WorldView

MODES = ("greedy", "stochastic")


def masked_probabilities(
    logits: npt.NDArray[np.float64], masks: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Softmax over the allowed actions of each row (masked actions get probability 0)."""
    z = np.where(masks, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.where(masks, np.exp(z), 0.0)
    out: npt.NDArray[np.float64] = e / e.sum(axis=1, keepdims=True)
    return out


def sample_masked(
    logits: npt.NDArray[np.float64], masks: npt.NDArray[np.bool_], rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """One action per row from the masked categorical distribution (inverse CDF, float64), so
    the draw depends only on the logits and the NumPy stream. Masked actions are never drawn."""
    p = masked_probabilities(logits, masks)
    cdf = np.cumsum(p, axis=1)
    u = np.asarray(rng.random(size=(p.shape[0], 1))) * cdf[:, -1:]
    a = (cdf <= u).sum(axis=1)
    a = np.minimum(a, p.shape[1] - 1)
    # guard against round-off landing on a masked action: take the last allowed one below it
    for i in np.flatnonzero(~masks[np.arange(len(a)), a]):
        allowed = np.flatnonzero(masks[i])
        a[i] = int(allowed[np.searchsorted(allowed, a[i], side="right") - 1])
    out: npt.NDArray[np.int64] = a.astype(np.int64)
    return out


class LearnedController:
    def __init__(
        self,
        net: nn.Module,
        obs: ObsParams | None = None,
        name: str = "madql",
        *,
        output: str = "q-values",
        mode: str = "greedy",
        epsilon: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if output not in ("q-values", "logits"):
            raise ValueError(f"unknown network output {output!r}")
        if mode == "stochastic" and rng is None:
            raise ValueError("the stochastic mode needs a seeded NumPy generator")
        self.net = net.eval()
        self.builder = ObservationBuilder(obs)
        self.name = name
        self.output = output
        self.mode = mode
        self.epsilon = epsilon
        self.rng = rng
        self.decisions: Counter[str] = Counter()
        self.meta: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        mode: str = "greedy",
        epsilon: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> LearnedController:
        from robocoord.marl.checkpoint import MODEL_SPECS, load_inference

        net, meta, obs = load_inference(path)
        ctrl = cls(
            net,
            obs,
            name=str(meta.get("algorithm", "madql")),
            output=MODEL_SPECS[meta["model"]["version"]],
            mode=mode,
            epsilon=epsilon,
            rng=rng,
        )
        ctrl.meta = meta
        return ctrl

    def reset(self, n_robots: int) -> None:
        self.decisions = Counter()

    def choose(
        self, obs: npt.NDArray[np.float32], masks: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.int64]:
        """Actions for a batch of observations in the controller's mode."""
        with torch.no_grad():
            out = self.net(torch.from_numpy(np.ascontiguousarray(obs)))
            if self.mode == "greedy":
                return np.asarray(masked_argmax(out, torch.from_numpy(masks)).numpy(), np.int64)
            values = out.numpy().astype(np.float64)
        assert self.rng is not None
        if self.output == "logits":
            return sample_masked(values, masks, self.rng)
        from robocoord.marl.madql import masked_epsilon_greedy

        return masked_epsilon_greedy(values.astype(np.float32), masks, self.epsilon, self.rng)

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        obs = self.builder.build(view)
        masks = action_mask_array(view)
        acts = self.choose(obs, masks)
        if view.suggested is not None:
            for i, a in enumerate(acts):
                self.decisions["follow" if int(a) == int(view.suggested[i]) else "deviate"] += 1
        return {r.id: Action(int(acts[r.id])) for r in view.robots}

    def episode_stats(self) -> dict[str, Any]:
        total = sum(self.decisions.values())
        stats: dict[str, Any] = {
            "decisions": dict(self.decisions),
            "share_following_suggestion": self.decisions["follow"] / total if total else None,
        }
        if self.mode != "greedy":  # the greedy output is exactly the Issue #5 one
            stats["mode"] = self.mode
        return stats


def greedy_actions(net: nn.Module, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Masked greedy actions for a batch of observations (used by tests and the trainer)."""
    with torch.no_grad():
        return masked_argmax(net(torch.from_numpy(obs)), torch.from_numpy(masks)).numpy()
