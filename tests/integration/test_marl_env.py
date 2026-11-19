"""The MARL environment around the real engine (Issue #5, Steps 1–2).

Truncation look-ahead (T-a … T-d), termination precedence, X-13 on recorded episodes,
determinism, size independence, the vector environment and the seed blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robocoord.core.types import Action
from robocoord.experiments.allocation_runs import e5_scenario
from robocoord.marl.config import EnvConfig, RewardParams
from robocoord.marl.env import VIEW_EXTRAS, MultiRobotEnv, make_coordinator
from robocoord.marl.global_state import global_state_size
from robocoord.marl.observation import ObservationBuilder
from robocoord.marl.reward import potentials, step_rewards, team_potential
from robocoord.marl.scenarios import (
    BLOCK_A,
    BLOCK_B,
    CURRICULUM,
    LEVELS,
    RESERVED_TEST,
    SeedBlockError,
    check_seeds,
    stage_at,
    training_seed_order,
)
from robocoord.marl.vector_env import SyncVectorEnv
from robocoord.scenario import build_scenario
from robocoord.scenario.spec import ScenarioSpec
from robocoord.sim.engine import Engine
from robocoord.world.entities import RobotMode


def small(
    seed: int = 10000, dynamics: str = "full", robots: int = 4, steps: int = 60
) -> ScenarioSpec:
    return e5_scenario(
        seed,
        dynamics,
        size=16,
        density=0.15,
        robots=robots,
        tasks_per_robot=2,
        obstacles=4,
        max_steps=steps,
        arrival_rate=0.05,
        arrivals_until=steps // 2,
    )


def follow(env: MultiRobotEnv) -> np.ndarray:
    """The reference action: the suggestion if allowed, else stay."""
    v = env.view
    assert v.suggested is not None
    return np.array([int(s) if v.masks[i][s] else 0 for i, s in enumerate(v.suggested)])


def rrt_config() -> EnvConfig:
    return EnvConfig(planner="rrtstar", rrt_iterations=300)


# ---------------------------------------------------------------------------------- truncation
def replay_reference(spec: ScenarioSpec, cfg: EnvConfig, actions: list[np.ndarray]) -> Any:
    """A plain engine performing the same transitions, then a normal prepare at time T."""
    engine = Engine(
        spec,
        make_coordinator(cfg),
        max_steps=spec.max_steps + 5,
        view_extras=VIEW_EXTRAS,
        path_limit=cfg.obs.path_cells,
    )
    outcome = None
    for a in actions:
        engine.prepare_step()
        nonidle = [r.mode is not RobotMode.IDLE for r in engine.robots]
        phi0 = potentials(engine.robots, engine.book.tasks, engine.oracle, 16, 16)
        outcome = engine.apply_actions({i: Action(int(x)) for i, x in enumerate(a)})
    view = engine.prepare_step()  # the same logical time T, a regular preparation
    phi1 = potentials(engine.robots, engine.book.tasks, engine.oracle, 16, 16)
    prios = {tid: engine.book.tasks[tid].priority for _, tid in outcome.delivered}
    rew = step_rewards(cfg.reward, nonidle, outcome, prios, phi0, phi1)
    return engine, view, phi1, rew


@pytest.mark.parametrize(
    "cfg", [EnvConfig(), pytest.param(rrt_config(), marks=pytest.mark.slow)], ids=["astar", "rrt"]
)
def test_truncation_lookahead_equals_normal_transition(cfg: EnvConfig) -> None:
    """T-a and T-b: final observation, reward, Φ and metrics at the same logical time."""
    spec = small(steps=25)
    env = MultiRobotEnv(config=cfg)
    env.reset_spec(spec)
    actions: list[np.ndarray] = []
    while True:
        a = follow(env)
        actions.append(a)
        tr = env.step_arrays(a)
        if tr.done:
            break
    assert tr.truncated and not tr.terminated and len(actions) == 25
    engine, view, phi1, rew = replay_reference(spec, cfg, actions)
    ref_obs = ObservationBuilder(cfg.obs).build(view)
    assert np.array_equal(tr.next_obs, ref_obs)
    assert np.array_equal(tr.next_masks, np.asarray(view.masks))
    assert np.array_equal(tr.rewards, rew.total)
    assert np.array_equal(env._phi, phi1)
    ep = tr.info["episode"]
    tasks = engine.book.tasks.values()
    assert ep["released"] == sum(t.status.value != "unreleased" for t in tasks)
    assert ep["delivered"] == sum(t.status.value == "delivered" for t in tasks)
    assert ep["steps"] == engine.t == 25


def test_lookahead_metrics_differ_from_plain_engine_only_in_released_counts() -> None:
    """T-b: compared with ``Engine(max_steps=T)`` only tasks released at ``T`` can differ."""
    spec = small(steps=30)
    env = MultiRobotEnv()
    env.reset_spec(spec)
    actions = []
    while True:
        a = follow(env)
        actions.append(a)
        tr = env.step_arrays(a)
        if tr.done:
            break
    plain = Engine(spec, make_coordinator(EnvConfig()))
    for a in actions:
        plain.prepare_step()
        plain.apply_actions({i: Action(int(x)) for i, x in enumerate(a)})
    assert plain.truncated
    s = plain.summary()
    ep = tr.info["episode"]
    released_at_t = sum(t.release == 30 for t in env.engine.book.tasks.values())  # type: ignore[union-attr]
    assert ep["delivered"] == s.tasks_delivered
    assert ep["released"] - s.tasks_released == released_at_t


def hand_spec(max_steps: int) -> ScenarioSpec:
    return build_scenario(
        {
            "scenario": {"name": "line", "seed": 0, "max_steps": max_steps},
            "map": {"rows": ["......"]},
            "robots": [{"start": [0, 0]}],
            "tasks": [{"pickup": [1, 0], "dropoff": [3, 0], "priority": 2, "release": 0}],
        }
    )


def run_follow(env: MultiRobotEnv, spec: ScenarioSpec) -> list[Any]:
    env.reset_spec(spec)
    out = []
    while True:
        tr = env.step_arrays(follow(env))
        out.append(tr)
        if tr.done:
            return out


def test_termination_takes_precedence_at_the_horizon() -> None:
    """T-c: the last task is delivered exactly at T = 3 → terminated, no look-ahead, Φ = 0."""
    env = MultiRobotEnv()
    trs = run_follow(env, hand_spec(3))
    last = trs[-1]
    assert len(trs) == 3 and last.terminated and not last.truncated
    assert env.engine is not None and env.engine._pending is None  # no prepare_step after the end
    assert np.all(last.next_obs == 0) and env._phi.tolist() == [0.0]
    assert last.components.base[0] == pytest.approx(-0.01 + 1.0 * 2)  # step cost + R_del·w(2)
    # the same scenario cut one step earlier is truncated with a look-ahead preparation
    trs2 = run_follow(env, hand_spec(2))
    assert trs2[-1].truncated and not trs2[-1].terminated
    assert env.engine is not None and env.engine._pending is not None


def test_reset_after_episode_equals_fresh_environment() -> None:
    """T-d."""
    spec = small()
    used = MultiRobotEnv()
    run_follow(used, small(10001, steps=20))
    o1, m1, _ = used.reset_spec(spec)
    o2, m2, _ = MultiRobotEnv().reset_spec(spec)
    assert np.array_equal(o1, o2) and np.array_equal(m1, m2)


# ---------------------------------------------------------------------------------- X-13
def open2_spec() -> ScenarioSpec:
    """Two robots, three tasks on an open 6×6 map: the reference policy delivers everything."""
    return build_scenario(
        {
            "scenario": {"name": "open2", "seed": 0, "max_steps": 100},
            "map": {"rows": ["......", "..#...", "......", "...#..", "......", "......"]},
            "robots": [{"start": [0, 0]}, {"start": [5, 5]}],
            "tasks": [
                {"pickup": [1, 4], "dropoff": [4, 1], "priority": 1, "release": 0},
                {"pickup": [5, 0], "dropoff": [0, 5], "priority": 3, "release": 2},
                {"pickup": [2, 2], "dropoff": [5, 3], "priority": 2, "release": 4},
            ],
        }
    )


@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("which", ["terminated", "truncated"])
def test_shaping_telescopes_over_recorded_episodes(beta: float, which: str) -> None:
    cfg = EnvConfig(reward=RewardParams(beta=beta))
    env = MultiRobotEnv(config=cfg)
    env.reset_spec(open2_spec() if which == "terminated" else small(10002, steps=40))
    g = cfg.reward.gamma
    psi0 = team_potential(env._phi, beta)
    total = np.zeros_like(psi0)
    t = 0
    while True:
        tr = env.step_arrays(follow(env))
        mixed = (1 - beta) * tr.components.shaping + beta * tr.components.shaping.mean()
        total += g**t * mixed
        t += 1
        if tr.done:
            break
    assert (tr.terminated, tr.truncated) == (which == "terminated", which == "truncated")
    psi_t = team_potential(env._phi, beta)
    assert np.allclose(total, cfg.reward.zeta * (g**t * psi_t - psi0), atol=1e-12)
    if tr.terminated:  # zero potential at termination
        assert np.all(psi_t == 0.0)
        assert np.allclose(total, -cfg.reward.zeta * psi0, atol=1e-12)


# ---------------------------------------------------------------------------------- API
def test_dict_api_and_masks() -> None:
    env = MultiRobotEnv(lambda s: small(s))
    obs, _info = env.reset(10003)
    assert set(obs) == {0, 1, 2, 3} and all(o.shape == (356,) for o in obs.values())
    masks = env.action_masks()
    assert all(m[Action.STAY] for m in masks.values())  # stay is never masked
    for i, m in masks.items():
        assert m.tolist() == list(env.view.masks[i])
    _obs2, rew, term, trunc, _info2 = env.step({i: 0 for i in obs})
    assert set(rew) == set(obs) and isinstance(term, bool) and isinstance(trunc, bool)
    assert env.global_state().shape == (global_state_size(8),)
    with pytest.raises(ValueError, match="no action"):
        env.step({0: 0})


def test_determinism_same_spec_same_actions() -> None:
    def roll() -> list[np.ndarray]:
        env = MultiRobotEnv()
        env.reset_spec(small(10004))
        rng = np.random.default_rng(3)
        out = []
        while True:
            a = np.array([rng.choice(np.flatnonzero(m)) for m in env._masks])  # type: ignore[union-attr]
            tr = env.step_arrays(a)
            out.extend([tr.next_obs, tr.rewards])
            if tr.done:
                return out

    a, b = roll(), roll()
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize(("robots", "size"), [(2, 16), (8, 32)])
def test_observation_size_independent_of_team_and_map(robots: int, size: int) -> None:
    spec = e5_scenario(
        10005,
        "full",
        size=size,
        density=0.15,
        robots=robots,
        tasks_per_robot=2,
        obstacles=4,
        max_steps=50,
        arrival_rate=0.05,
        arrivals_until=20,
    )
    env = MultiRobotEnv()
    obs, masks, _ = env.reset_spec(spec)
    assert obs.shape == (robots, 356) and masks.shape == (robots, 5)
    assert env.global_state().shape == (259,)


def test_vector_env_auto_reset_with_different_team_sizes() -> None:
    specs = [
        small(10006, "obstacles", robots=2, steps=5),
        small(10007, "obstacles", robots=4, steps=8),
    ]
    counter = {"k": 0}

    def next_scenario(k: int) -> ScenarioSpec:
        counter["k"] += 1
        return specs[k]

    venv = SyncVectorEnv([MultiRobotEnv(), MultiRobotEnv()], next_scenario)
    obs, _masks = venv.reset()
    assert [o.shape[0] for o in obs] == [2, 4]
    ended = []
    for _ in range(8):
        res = venv.step([np.zeros(len(o), dtype=np.int64) for o in venv.obs])
        ended.extend(res.episodes)
        for k, tr in enumerate(res.transitions):
            if tr.done:  # the transition keeps the true final observation …
                assert tr.next_obs.shape == (specs[k].robots.__len__(), 356)
                # … while the next action is chosen from the new episode's first observation
                assert res.obs[k].shape[0] == len(specs[k].robots)
    assert [e["steps"] for e in ended] == [5, 8]
    assert counter["k"] == 2 + 2


# ---------------------------------------------------------------------------------- seeds
def test_seed_blocks_are_disjoint_and_enforced() -> None:
    assert set(BLOCK_A).isdisjoint(BLOCK_B)
    assert set(BLOCK_A) | set(BLOCK_B) == set(range(10_000, 10_100))
    assert check_seeds([10_000, 10_019], "A") == [10_000, 10_019]
    with pytest.raises(SeedBlockError):
        check_seeds([10_020], "A")
    with pytest.raises(SeedBlockError):
        check_seeds([10_019], "B")
    with pytest.raises(SeedBlockError):
        check_seeds([20_200], "B")
    order = training_seed_order()
    assert sorted(order) == list(range(10_000)) and order == training_seed_order()
    assert not set(order) & set(RESERVED_TEST)


def test_curriculum_and_levels_match_the_approved_design() -> None:
    assert sum(s.fraction for s in CURRICULUM) == pytest.approx(1.0)
    assert [stage_at(x, 100) for x in (0, 19, 20, 49, 50, 99)] == [0, 0, 1, 1, 2, 2]
    for lvl, dist in LEVELS.items():
        spec = dist.sample(10_000)
        assert spec.grid.width == spec.grid.height == 32 and len(spec.robots) == 8
        assert spec.max_steps == 400 and len(spec.tasks) == 16
        assert len(spec.obstacles) == (0 if lvl == "static" else 8)
        assert (len(spec.map_events) > 0) == (lvl == "full")
        assert (spec.arrivals.kind == "bernoulli") == (lvl == "full")
    # the validation instances are exactly the Issue #4 generator settings
    ref = e5_scenario(
        10_000,
        "full",
        size=32,
        density=0.15,
        robots=8,
        tasks_per_robot=2,
        obstacles=8,
        max_steps=400,
        arrival_rate=0.05,
        arrivals_until=200,
    )
    assert LEVELS["full"].sample(10_000) == ref
    for st in CURRICULUM:
        p = st.dist.parameters(7)
        assert p == st.dist.parameters(7)  # a function of (stage, seed) only
        spec = st.dist.sample(7)
        assert len(spec.robots) in st.dist.robots and spec.grid.width in st.dist.sizes
