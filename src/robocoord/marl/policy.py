"""The learned controller (architecture §5.6): a trained Q-network acting greedily on the
observation built from the view. Requires PyTorch.

It implements the ordinary :class:`~robocoord.control.base.Controller` protocol and is selectable
as ``madql`` with either planner and with the shield on or off. Its output is the *requested*
action per robot: the mask is respected (masked arg-max), and the optional shield and the
always-on arbiter of the engine may still replace or cancel a request. It contains no rule-based
fallback, so it stays distinguishable from the baseline. The engine must provide the view extras
``{"paths", "goal_distance"}``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from robocoord.core.types import Action
from robocoord.marl.config import ObsParams
from robocoord.marl.networks import masked_argmax
from robocoord.marl.observation import ObservationBuilder, action_mask_array
from robocoord.sim.view import WorldView


class LearnedController:
    def __init__(self, net: nn.Module, obs: ObsParams | None = None, name: str = "madql") -> None:
        self.net = net.eval()
        self.builder = ObservationBuilder(obs)
        self.name = name
        self.decisions: Counter[str] = Counter()
        self.meta: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> LearnedController:
        from robocoord.marl.checkpoint import load_inference

        net, meta, obs = load_inference(path)
        ctrl = cls(net, obs, name=str(meta.get("algorithm", "madql")))
        ctrl.meta = meta
        return ctrl

    def reset(self, n_robots: int) -> None:
        self.decisions = Counter()

    @torch.no_grad()
    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        obs = self.builder.build(view)
        masks = action_mask_array(view)
        q = self.net(torch.from_numpy(obs))
        acts = masked_argmax(q, torch.from_numpy(masks)).numpy()
        if view.suggested is not None:
            for i, a in enumerate(acts):
                self.decisions["follow" if int(a) == int(view.suggested[i]) else "deviate"] += 1
        return {r.id: Action(int(acts[r.id])) for r in view.robots}

    def episode_stats(self) -> dict[str, Any]:
        total = sum(self.decisions.values())
        return {
            "decisions": dict(self.decisions),
            "share_following_suggestion": self.decisions["follow"] / total if total else None,
        }


def greedy_actions(net: nn.Module, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Masked greedy actions for a batch of observations (used by tests and the trainer)."""
    with torch.no_grad():
        return masked_argmax(net(torch.from_numpy(obs)), torch.from_numpy(masks)).numpy()
