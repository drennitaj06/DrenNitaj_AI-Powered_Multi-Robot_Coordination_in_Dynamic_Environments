"""The multi-agent environment around the existing simulator (ADR-6; model §6.3). NumPy only.

``MultiRobotEnv`` drives :class:`~robocoord.sim.engine.Engine` through ``prepare_step`` /
``apply_actions``; it never simulates anything itself. Allocation (auction) and planning (A\\* or
RRT\\*, path monitor) are part of the environment; the policy chooses one *requested* primitive
action per robot. The always-on action mask excludes only statically infeasible moves (static
obstacle or out of bounds; ``stay`` is never masked); the optional shield and the always-on
arbiter belong to the transition and may cancel a request. Rewards attribute conflicts to the
request (``ConflictEvent.at_fault``).

Episode end (approved design R2):

* **termination** (every task delivered and no future release) takes precedence; the episode ends
  with ``terminated = True`` and ``Φ(s') = 0`` (every robot is idle);
* **truncation** at the horizon ``T`` (the scenario's ``max_steps``): the engine runs with
  ``max_steps = T + 1`` and, after the ``T``-th transition, the environment performs the normal
  ``prepare_step()`` for time ``T`` — events, arrivals, allocation and replanning as at every
  other step — without acting. The returned final observation, the potential used for the last
  reward and the episode metrics are all taken from that prepared state, so the bootstrap target
  sees a regular observation, never a stale one.

Training-episode metrics therefore describe the prepared state at ``T``. For truncated episodes they
can differ from an ``Engine(max_steps=T)`` run only in the counts the look-ahead adds (tasks
released at ``T``, the planning counters of the extra update). Validation and development
comparisons use the plain engine (``marl.evaluation``) for every controller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from robocoord.core.types import Action
from robocoord.marl.config import N_ACTIONS, EnvConfig
from robocoord.marl.global_state import global_state
from robocoord.marl.observation import ObservationBuilder, action_mask_array
from robocoord.marl.reward import StepRewards, potentials, step_rewards
from robocoord.scenario.spec import ScenarioSpec
from robocoord.sim.engine import Engine, EngineStateError, SafetyLayer
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode, TaskStatus

F32 = npt.NDArray[np.float32]
VIEW_EXTRAS = frozenset({"paths", "goal_distance"})


def make_coordinator(config: EnvConfig) -> Any:
    """The approved system composition for learned controllers (D2)."""
    from robocoord.allocation.auction import AuctionAllocator
    from robocoord.allocation.sequential import SequentialAllocator
    from robocoord.planning import make_planner
    from robocoord.planning.coordinator import AllocationCoordinator
    from robocoord.planning.deadlock import DetourParams
    from robocoord.planning.endpoint import EndpointParams

    options: dict[str, Any] = {}
    if config.planner == "rrtstar" and config.rrt_iterations is not None:
        options["max_iterations"] = config.rrt_iterations
    allocator = AuctionAllocator() if config.allocator == "auction" else SequentialAllocator()
    return AllocationCoordinator(
        make_planner(config.planner, **options),
        allocator,
        deadlock=DetourParams() if config.deadlock else None,
        endpoint=EndpointParams() if config.endpoint else None,
    )


def make_shield(config: EnvConfig) -> SafetyLayer | None:
    if config.shield == "off":
        return None
    from robocoord.control.shield import Shield

    return Shield(config.shield)


@dataclass(frozen=True, slots=True)
class Transition:
    """One environment step for all robots (arrays in robot-id order)."""

    next_obs: F32  # the observation of s' (zeros when terminated: never bootstrapped)
    next_masks: npt.NDArray[np.bool_]
    rewards: npt.NDArray[np.float64]
    terminated: bool
    truncated: bool
    components: StepRewards
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class MultiRobotEnv:
    """PettingZoo-parallel-style dictionaries with Gymnasium termination/truncation flags."""

    def __init__(
        self,
        scenario_fn: Callable[[int], ScenarioSpec] | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        self.scenario_fn = scenario_fn
        self.config = config or EnvConfig()
        self.builder = ObservationBuilder(self.config.obs)
        self.engine: Engine | None = None
        self.horizon = 0
        self._view: WorldView | None = None
        self._obs: F32 | None = None
        self._masks: npt.NDArray[np.bool_] | None = None
        self._phi = np.zeros(0)
        self._done = True
        self._ep: dict[str, Any] = {}

    # ------------------------------------------------------------------ properties
    @property
    def obs_dim(self) -> int:
        return self.config.obs.dim

    @property
    def n_robots(self) -> int:
        assert self.engine is not None
        return len(self.engine.robots)

    @property
    def view(self) -> WorldView:
        assert self._view is not None
        return self._view

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int) -> tuple[dict[int, F32], dict[str, Any]]:
        if self.scenario_fn is None:
            raise ValueError("reset(seed) needs a scenario_fn; use reset_spec(spec)")
        return self._as_dict(*self.reset_spec(self.scenario_fn(seed)))

    def reset_spec(self, spec: ScenarioSpec) -> tuple[F32, npt.NDArray[np.bool_], dict[str, Any]]:
        """Start an episode on ``spec``; returns (observations, masks, info) as arrays."""
        cfg = self.config
        self.horizon = spec.max_steps
        self.engine = Engine(
            spec,
            make_coordinator(cfg),
            shield=make_shield(cfg),
            max_steps=spec.max_steps + 1,  # the environment truncates at T itself (look-ahead)
            view_extras=VIEW_EXTRAS,
            path_limit=cfg.obs.path_cells,
        )
        self._done = False
        self._ep = {
            "seed": spec.seed,
            "scenario": spec.name,
            "robots": len(spec.robots),
            "base_return": np.zeros(len(spec.robots)),
            "discount": 1.0,
            "rr": 0,
            "ro": 0,
            "sh": 0,
            "max_abs_shaping": 0.0,
            "shaping_gt1": 0,
            "robot_steps": 0,
        }
        self._observe(self.engine.prepare_step())
        assert self._obs is not None and self._masks is not None
        return self._obs, self._masks, {"t": 0}

    def _observe(self, view: WorldView) -> None:
        assert self.engine is not None
        e = self.engine
        self._view = view
        self._obs = self.builder.build(view)
        self._masks = action_mask_array(view)
        self._phi = potentials(e.robots, e.book.tasks, e.oracle, e.grid.width, e.grid.height)

    # ------------------------------------------------------------------ step
    def action_masks(self) -> dict[int, npt.NDArray[np.bool_]]:
        assert self._masks is not None
        return {i: self._masks[i] for i in range(len(self._masks))}

    def global_state(self) -> F32:
        return global_state(self.view, self.horizon, self.config.global_blocks)

    def step(
        self, actions: Mapping[int, int]
    ) -> tuple[dict[int, F32], dict[int, float], bool, bool, dict[str, Any]]:
        n = self.n_robots
        missing = [i for i in range(n) if i not in actions]
        if missing:
            raise ValueError(f"no action for robots {missing}")
        tr = self.step_arrays(np.array([int(actions[i]) for i in range(n)], dtype=np.int64))
        obs = {i: tr.next_obs[i] for i in range(n)}
        rewards = {i: float(tr.rewards[i]) for i in range(n)}
        return obs, rewards, tr.terminated, tr.truncated, tr.info

    def step_arrays(self, actions: npt.NDArray[np.int64]) -> Transition:
        """Apply the requested actions of all robots; the arrays form of :meth:`step`."""
        if self._done or self.engine is None:
            raise EngineStateError("episode is finished; call reset_spec() / reset()")
        e = self.engine
        n = len(e.robots)
        if actions.shape != (n,) or actions.min() < 0 or actions.max() >= N_ACTIONS:
            raise ValueError(f"expected {n} actions in 0..{N_ACTIONS - 1}, got {actions!r}")
        nonidle = [r.mode is not RobotMode.IDLE for r in e.robots]
        phi_before = self._phi
        outcome = e.apply_actions({i: Action(int(a)) for i, a in enumerate(actions)})
        prios = {tid: e.book.tasks[tid].priority for _, tid in outcome.delivered}
        terminated = e.terminated
        truncated = not terminated and e.t >= self.horizon
        if terminated:
            phi_after = potentials(e.robots, e.book.tasks, e.oracle, e.grid.width, e.grid.height)
            next_obs = np.zeros((n, self.obs_dim), dtype=np.float32)
            next_masks = np.zeros((n, N_ACTIONS), dtype=bool)
            next_masks[:, Action.STAY] = True
        else:
            self._observe(e.prepare_step())  # normal preparation; at t = T it is the look-ahead
            assert self._obs is not None and self._masks is not None
            next_obs, next_masks, phi_after = self._obs, self._masks, self._phi
        comp = step_rewards(self.config.reward, nonidle, outcome, prios, phi_before, phi_after)
        self._phi = phi_after
        info = self._account(comp, outcome, sum(nonidle))
        info["t"] = e.t
        if terminated or truncated:
            self._done = True
            info["episode"] = self._episode_summary(terminated, truncated)
        return Transition(next_obs, next_masks, comp.total, terminated, truncated, comp, info)

    # ------------------------------------------------------------------ bookkeeping
    def _account(self, comp: StepRewards, outcome: Any, active: int) -> dict[str, Any]:
        ep = self._ep
        ep["base_return"] += ep["discount"] * comp.base
        ep["discount"] *= self.config.reward.gamma
        rr, ro, sh = int(comp.rr.sum()), int(comp.ro.sum()), int(comp.sh.sum())
        ep["rr"] += rr
        ep["ro"] += ro
        ep["sh"] += sh
        ep["robot_steps"] += active
        big = np.abs(comp.shaping)
        ep["max_abs_shaping"] = max(ep["max_abs_shaping"], float(big.max(initial=0.0)))
        ep["shaping_gt1"] += int((big > 1.0).sum())
        return {
            "conflicts_rr": rr,
            "conflicts_ro": ro,
            "shield": sh,
            "picked": len(outcome.picked),
            "delivered": len(outcome.delivered),
        }

    def _episode_summary(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        e = self.engine
        assert e is not None
        tasks = e.book.tasks.values()
        released = sum(tk.status is not TaskStatus.UNRELEASED for tk in tasks)
        delivered = sum(tk.status is TaskStatus.DELIVERED for tk in tasks)
        ep = self._ep
        return {
            "seed": ep["seed"],
            "scenario": ep["scenario"],
            "robots": ep["robots"],
            "steps": e.t,
            "terminated": terminated,
            "truncated": truncated,
            "released": released,
            "delivered": delivered,
            "completion": delivered / released if released else None,
            "base_return": float(np.mean(ep["base_return"])),
            "conflicts_rr": ep["rr"],
            "conflicts_ro": ep["ro"],
            "shield": ep["sh"],
            "conflicts_per_100": 100.0 * (ep["rr"] + ep["ro"]) / max(1, ep["robot_steps"]),
            "max_abs_shaping": ep["max_abs_shaping"],
            "shaping_gt1": ep["shaping_gt1"],
            "physical_collisions": e.summary().physical_collisions,
        }

    @staticmethod
    def _as_dict(
        obs: F32, masks: npt.NDArray[np.bool_], info: dict[str, Any]
    ) -> tuple[dict[int, F32], dict[str, Any]]:
        info = {**info, "action_masks": {i: masks[i] for i in range(len(masks))}}
        return {i: obs[i] for i in range(len(obs))}, info
