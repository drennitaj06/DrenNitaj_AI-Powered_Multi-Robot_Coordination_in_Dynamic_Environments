"""PPO learning sanity (algorithms §7 test 4; Issue #6): a masked bandit and a chain MDP.

These show only that the implementation learns simple known behaviours with the real rollout
buffer, GAE (termination and truncation) and update; they say nothing about convergence on the
robot-coordination task.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.ppo import PPO, PPOParams, masked_log_probs  # noqa: E402
from robocoord.marl.rollout import RolloutBuffer  # noqa: E402

STATE_DIM = 259


def test_masked_bandit_learns_the_best_allowed_arm() -> None:
    """Arm 4 pays most but is masked; arm 2 is the best allowed arm."""
    configure_torch(0)
    params = PPOParams(variant="ippo", hidden=(16, 16), critic_hidden=(16, 16), lr=1e-2,
                       ent_coef=0.0, target_kl=None)  # fmt: skip
    agent = PPO(3, params)
    rng = np.random.default_rng(0)
    obs = np.tile(np.array([1.0, 0.0, 0.5], np.float32), (8, 1))
    masks = np.ones((8, 5), dtype=bool)
    masks[:, 4] = False
    pay = np.array([0.0, 0.2, 1.0, 0.1, 5.0])
    for _ in range(40):
        buf = RolloutBuffer(16, 1, 3, max_robots=8)
        for t in range(16):
            a, logp = agent.act(obs, masks, rng)
            assert not np.any(a == 4)
            buf.add(t, 0, obs=obs, masks=masks, actions=a, logp=logp,
                    values=agent.value(obs, None), rewards=pay[a], terminated=True,
                    truncated=False)  # fmt: skip
        buf.compute_gae(np.zeros((1, 8), np.float32), np.ones((1, 8), bool), 0.99, 0.95)
        agent.update(buf.flatten(), rng)
    p = masked_log_probs(agent.actor(torch.from_numpy(obs[:1])), torch.from_numpy(masks[:1]))
    probs = p.exp()[0].detach().numpy()
    assert probs.argmax() == 2 and probs[2] > 0.9 and probs[4] == 0.0


@pytest.mark.parametrize("variant", ["ippo", "mappo"])
def test_chain_mdp_learns_to_walk_right(variant: str) -> None:
    """States 0…4, start 0; action 1 moves right, 2 left (masked at 0), others stay. Reaching 4
    pays +1 and terminates; each step costs 0.01; the horizon of 12 steps truncates."""
    configure_torch(1)
    params = PPOParams(variant=variant, hidden=(32, 32), critic_hidden=(32, 32), lr=3e-3,
                       target_kl=None)  # fmt: skip
    agent = PPO(5, params)
    rng = np.random.default_rng(1)
    envs, horizon, steps = 4, 12, 32

    def ob(s: int) -> np.ndarray:
        o = np.zeros((1, 5), np.float32)
        o[0, s] = 1.0
        return o

    def state(t: int) -> np.ndarray:
        g = np.zeros(STATE_DIM, np.float32)
        g[STATE_DIM - 2] = t / horizon
        return g

    def mask(s: int) -> np.ndarray:
        m = np.ones((1, 5), dtype=bool)
        m[0, 2] = s > 0
        return m

    def crit(t: int) -> np.ndarray | None:
        return state(t)[None, :] if variant == "mappo" else None

    pos, time_ = [0] * envs, [0] * envs
    for _ in range(60):
        buf = RolloutBuffer(steps, envs, 5, max_robots=1,
                            state_dim=STATE_DIM if variant == "mappo" else None)  # fmt: skip
        for t in range(steps):
            for k in range(envs):
                s, tt = pos[k], time_[k]
                a, logp = agent.act(ob(s), mask(s), rng)
                v = agent.value(ob(s), crit(tt))
                nxt = min(4, s + 1) if a[0] == 1 else (max(0, s - 1) if a[0] == 2 else s)
                tt += 1
                term = nxt == 4
                trunc = not term and tt >= horizon
                r = -0.01 + (1.0 if term else 0.0)
                boot = agent.value(ob(nxt), crit(tt)) if trunc else None
                buf.add(t, k, obs=ob(s), masks=mask(s), actions=a, logp=logp, values=v,
                        rewards=[r], terminated=term, truncated=trunc, boot_values=boot,
                        state=state(time_[k]) if variant == "mappo" else None)  # fmt: skip
                pos[k], time_[k] = (0, 0) if term or trunc else (nxt, tt)
        lv = np.array([agent.value(ob(pos[k]), crit(time_[k]))[0] for k in range(envs)])
        buf.compute_gae(lv.reshape(envs, 1).astype(np.float32), np.ones((envs, 1), bool),
                        0.99, 0.95)  # fmt: skip
        agent.update(buf.flatten(), rng)
    with torch.no_grad():
        logits = agent.actor(torch.eye(5)[:4])
    greedy = logits.argmax(dim=1).tolist()
    assert greedy == [1, 1, 1, 1]
