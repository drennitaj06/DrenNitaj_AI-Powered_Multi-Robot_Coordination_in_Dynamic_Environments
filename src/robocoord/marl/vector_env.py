"""Synchronous batch of environments with auto-reset (ADR-6). NumPy only.

Each copy may have a different team size. ``step`` returns, per copy, the transition of the step
(with the true final observation when the episode ended, needed for the truncation bootstrap and
for GAE in Issue #6) and the observation the next action is chosen from (the first observation of
the new episode after an auto-reset).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from robocoord.marl.env import F32, MultiRobotEnv, Transition
from robocoord.scenario.spec import ScenarioSpec


@dataclass(frozen=True, slots=True)
class VecStep:
    transitions: list[Transition]
    obs: list[F32]  # observation for the next action (post auto-reset)
    masks: list[npt.NDArray[np.bool_]]
    episodes: list[dict[str, Any]]  # summaries of the episodes that ended in this step


class SyncVectorEnv:
    """``E`` environments stepped in lock-step; ``next_scenario()`` supplies every new episode."""

    def __init__(
        self, envs: Sequence[MultiRobotEnv], next_scenario: Callable[[int], ScenarioSpec]
    ) -> None:
        if not envs:
            raise ValueError("at least one environment is required")
        self.envs = list(envs)
        self.next_scenario = next_scenario  # called with the copy index
        self.obs: list[F32] = []
        self.masks: list[npt.NDArray[np.bool_]] = []

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> tuple[list[F32], list[npt.NDArray[np.bool_]]]:
        self.obs, self.masks = [], []
        for k, env in enumerate(self.envs):
            o, m, _ = env.reset_spec(self.next_scenario(k))
            self.obs.append(o)
            self.masks.append(m)
        return self.obs, self.masks

    def step(self, actions: Sequence[npt.NDArray[np.int64]]) -> VecStep:
        if len(actions) != len(self.envs):
            raise ValueError("one action array per environment is required")
        transitions: list[Transition] = []
        episodes: list[dict[str, Any]] = []
        for k, (env, act) in enumerate(zip(self.envs, actions, strict=True)):
            tr = env.step_arrays(np.asarray(act, dtype=np.int64))
            transitions.append(tr)
            if tr.done:
                episodes.append(tr.info["episode"])
                o, m, _ = env.reset_spec(self.next_scenario(k))
            else:
                o, m = tr.next_obs, tr.next_masks
            self.obs[k], self.masks[k] = o, m
        return VecStep(transitions, list(self.obs), list(self.masks), episodes)
