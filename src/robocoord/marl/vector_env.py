"""Synchronous batch of environments with auto-reset (ADR-6). NumPy only.

Each copy may have a different team size. ``step`` returns, per copy, the transition of the step
(with the true final observation when the episode ended, needed for the truncation bootstrap and
for GAE in Issue #6) and the observation the next action is chosen from (the first observation of
the new episode after an auto-reset).

Centralised state (Issue #6, opt-in with ``with_state=True``; off by default so the Issue #5
MADQL path is unchanged):

* ``states[k]`` — the global state ``φ(s)`` of the observation the next action of copy ``k`` is
  chosen from (after an auto-reset: the new episode's first state);
* ``final_states[k]`` — for a copy whose episode was **truncated** in this step, ``φ(s_T)`` of the
  look-ahead state, captured after ``step_arrays()`` and *before* ``reset_spec()``, so a critic
  bootstrap can never see the reset episode's state. ``None`` for terminated copies (their value
  is 0 and the environment does not refresh the view at termination) and for running copies.
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
    states: list[F32] | None = None  # with_state only: φ(s) for the next action (post reset)
    final_states: list[F32 | None] | None = None  # with_state only: φ(s_T) of truncated copies


class SyncVectorEnv:
    """``E`` environments stepped in lock-step; ``next_scenario()`` supplies every new episode."""

    def __init__(
        self,
        envs: Sequence[MultiRobotEnv],
        next_scenario: Callable[[int], ScenarioSpec],
        *,
        with_state: bool = False,
    ) -> None:
        if not envs:
            raise ValueError("at least one environment is required")
        self.envs = list(envs)
        self.next_scenario = next_scenario  # called with the copy index
        self.with_state = with_state
        self.obs: list[F32] = []
        self.masks: list[npt.NDArray[np.bool_]] = []
        self.states: list[F32] = []

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> tuple[list[F32], list[npt.NDArray[np.bool_]]]:
        self.obs, self.masks, self.states = [], [], []
        for k, env in enumerate(self.envs):
            o, m, _ = env.reset_spec(self.next_scenario(k))
            self.obs.append(o)
            self.masks.append(m)
            if self.with_state:
                self.states.append(env.global_state())
        return self.obs, self.masks

    def step(self, actions: Sequence[npt.NDArray[np.int64]]) -> VecStep:
        if len(actions) != len(self.envs):
            raise ValueError("one action array per environment is required")
        transitions: list[Transition] = []
        episodes: list[dict[str, Any]] = []
        finals: list[F32 | None] = []
        for k, (env, act) in enumerate(zip(self.envs, actions, strict=True)):
            tr = env.step_arrays(np.asarray(act, dtype=np.int64))
            transitions.append(tr)
            final: F32 | None = None
            if self.with_state and tr.truncated:
                final = env.global_state()  # the look-ahead state at T, before the reset
            finals.append(final)
            if tr.done:
                episodes.append(tr.info["episode"])
                o, m, _ = env.reset_spec(self.next_scenario(k))
            else:
                o, m = tr.next_obs, tr.next_masks
            self.obs[k], self.masks[k] = o, m
            if self.with_state:
                self.states[k] = env.global_state()
        if not self.with_state:
            return VecStep(transitions, list(self.obs), list(self.masks), episodes)
        return VecStep(
            transitions, list(self.obs), list(self.masks), episodes, list(self.states), finals
        )
