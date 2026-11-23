"""PPO core (algorithms §7 tests 2, 3, 5; Issue #6): masked categorical, clipped loss, value loss,
advantage normalisation, approximate KL, separate gradient clipping, deterministic sampling and the
MAPPO/IPPO critic switch."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from robocoord.core.errors import ConfigError  # noqa: E402
from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.policy import masked_probabilities, sample_masked  # noqa: E402
from robocoord.marl.ppo import (  # noqa: E402
    PPO,
    PPOParams,
    explained_variance,
    masked_entropy,
    masked_log_probs,
    normalise,
    ppo_loss,
)
from robocoord.marl.rollout import Samples  # noqa: E402

MASKS = torch.tensor([[True, True, False, True, False], [False, False, True, False, False]])


def test_masked_categorical_probabilities_and_entropy() -> None:
    logits = torch.tensor([[1.0, 2.0, 50.0, 0.5, -3.0], [0.1, 9.0, 0.2, 3.0, 4.0]])
    p = masked_log_probs(logits, MASKS).exp()
    assert torch.all(p[~MASKS] == 0.0)
    assert torch.allclose(p.sum(dim=1), torch.ones(2))
    allowed = torch.softmax(torch.tensor([1.0, 2.0, 0.5]), dim=0)
    assert torch.allclose(p[0, [0, 1, 3]], allowed)
    h = masked_entropy(logits, MASKS)
    assert h[0] == pytest.approx(float(-(allowed * allowed.log()).sum()), rel=1e-5)
    assert h[1] == 0.0  # a single allowed action: no entropy, and no NaN from masked terms
    # the numpy version used for sampling agrees with torch
    np.testing.assert_allclose(
        masked_probabilities(logits.numpy().astype(np.float64), MASKS.numpy()), p.numpy(),
        rtol=1e-5, atol=1e-7,
    )  # fmt: skip


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    rows=st.lists(st.lists(st.booleans(), min_size=5, max_size=5), min_size=1, max_size=8),
)
def test_sampling_never_selects_a_masked_action(seed: int, rows: list[list[bool]]) -> None:
    masks = np.array(rows, dtype=bool)
    masks[:, 0] = True  # stay is never masked in the environment
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 5, masks.shape)
    a = sample_masked(logits, masks, rng)
    assert masks[np.arange(len(a)), a].all()


def test_sampling_is_deterministic_and_matches_the_distribution() -> None:
    logits = np.array([[0.0, 1.0, 2.0, -1.0, 0.5]] * 20_000)
    masks = np.ones_like(logits, dtype=bool)
    masks[:, 3] = False
    a1 = sample_masked(logits, masks, np.random.default_rng(5))
    a2 = sample_masked(logits, masks, np.random.default_rng(5))
    assert np.array_equal(a1, a2)
    freq = np.bincount(a1, minlength=5) / len(a1)
    np.testing.assert_allclose(freq, masked_probabilities(logits[:1], masks[:1])[0], atol=0.012)
    assert freq[3] == 0.0


def loss_inputs(ratio: float, adv: list[float]) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(len(adv), 5, requires_grad=True)
    masks = torch.ones(len(adv), 5, dtype=torch.bool)
    actions = torch.zeros(len(adv), dtype=torch.long)
    cur = masked_log_probs(logits, masks)[:, 0].detach()
    return {"logits": logits, "masks": masks, "actions": actions,
            "old_logp": cur - float(np.log(ratio)), "advantages": torch.tensor(adv),
            "values": torch.zeros(len(adv)), "returns": torch.zeros(len(adv))}  # fmt: skip


def test_clipped_objective_at_ratio_one_and_outside_the_clip_range() -> None:
    p = PPOParams()
    inp = loss_inputs(1.0, [1.0, -1.0, 2.0, -2.0])
    t = ppo_loss(**inp, params=p)
    adv = normalise(inp["advantages"])
    assert t.policy.item() == pytest.approx(-adv.mean().item(), abs=1e-6)
    assert t.approx_kl.item() == pytest.approx(0.0, abs=1e-7) and t.clip_fraction.item() == 0.0
    # ρ = 1.5 > 1 + ε. Advantages are normalised inside the loss: [1, 2] → [−1, +1]. For the
    # positive one the clipped term is the minimum (no gradient); for the negative one the
    # unclipped term is, so its gradient flows.
    inp = loss_inputs(1.5, [1.0, 2.0])
    t = ppo_loss(**inp, params=p)
    t.policy.backward()
    grad = inp["logits"].grad
    assert torch.all(grad[1] == 0) and torch.any(grad[0] != 0)
    assert t.clip_fraction.item() == 1.0
    assert t.approx_kl.item() == pytest.approx(0.5 - np.log(1.5), rel=1e-5)


def test_value_loss_and_total() -> None:
    p = PPOParams(vf_coef=0.5, ent_coef=0.01)
    inp = loss_inputs(1.0, [0.0, 0.0])
    inp["values"] = torch.tensor([1.0, 3.0])
    inp["returns"] = torch.tensor([0.0, 1.0])
    t = ppo_loss(**inp, params=p)
    assert t.value.item() == pytest.approx((1 + 4) / 2)
    assert t.total.item() == pytest.approx(
        t.policy.item() + 0.5 * t.value.item() - 0.01 * t.entropy.item(), rel=1e-6
    )


def test_advantage_normalisation() -> None:
    a = normalise(torch.tensor([1.0, 2.0, 3.0, 10.0]))
    assert float(a.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(a.std(unbiased=False)) == pytest.approx(1.0, rel=1e-5)
    assert float(normalise(torch.tensor([4.0]))[0]) == 0.0


def samples(n: int = 64, seed: int = 0, ret_scale: float = 1.0, states: bool = True) -> Samples:
    r = np.random.default_rng(seed)
    return Samples(
        obs=r.normal(size=(n, 356)).astype(np.float32),
        masks=np.ones((n, 5), dtype=bool),
        actions=r.integers(0, 5, n),
        logp=np.full(n, np.log(0.2), dtype=np.float32),
        values=np.zeros(n, dtype=np.float32),
        advantages=r.normal(size=n).astype(np.float32),
        returns=(ret_scale * r.normal(size=n)).astype(np.float32),
        states=r.normal(size=(n, 259)).astype(np.float32) if states else None,
    )


def test_actor_and_critic_gradients_are_clipped_separately() -> None:
    """A critic loss a million times larger must not change the actor's update at all."""
    out = []
    for scale in (1.0, 1e6):
        configure_torch(3)
        agent = PPO(356, PPOParams(epochs=1, minibatches=1, target_kl=None))
        logs = agent.update(samples(ret_scale=scale), np.random.default_rng(0))
        out.append(([p.detach().clone() for p in agent.actor.parameters()], logs))
    (a1, l1), (a2, l2) = out
    assert all(torch.equal(x, y) for x, y in zip(a1, a2, strict=True))
    assert l2["critic_grad_norm"] > 1e3 * l1["critic_grad_norm"]


def test_mappo_and_ippo_differ_only_in_the_critic_input() -> None:
    configure_torch(7)
    mappo = PPO(356, PPOParams(variant="mappo"))
    configure_torch(7)
    ippo = PPO(356, PPOParams(variant="ippo"))
    assert mappo.critic_dim == 615 and ippo.critic_dim == 356
    pairs = zip(mappo.actor.state_dict().values(), ippo.actor.state_dict().values(), strict=True)
    assert all(torch.equal(a, b) for a, b in pairs)
    s = samples(8)
    assert s.states is not None
    other = s.states + 1.0
    assert not np.allclose(mappo.value(s.obs, s.states), mappo.value(s.obs, other))
    np.testing.assert_array_equal(ippo.value(s.obs, None), ippo.value(s.obs, None))
    with pytest.raises(ValueError, match="global state"):
        mappo.value(s.obs, None)


def test_update_logs_and_kl_early_stop() -> None:
    configure_torch(1)
    agent = PPO(356, PPOParams(epochs=4, minibatches=4))
    logs = agent.update(samples(128), np.random.default_rng(1))
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "explained_variance",
        "lr",
        "samples",
    ):
        assert key in logs, key  # fmt: skip
    assert logs["samples"] == 128 and agent.updates == logs["minibatch_updates"]
    configure_torch(1)
    tight = PPO(356, PPOParams(epochs=4, minibatches=4, target_kl=1e-12, lr=1e-2))
    logs = tight.update(samples(128), np.random.default_rng(1))
    assert logs["kl_early_stop"] and logs["epochs_run"] < 4
    assert logs["minibatch_updates"] < 16


def test_updates_are_deterministic_and_state_round_trips() -> None:
    def run() -> PPO:
        configure_torch(9)
        agent = PPO(356, PPOParams(target_kl=None))
        agent.update(samples(96), np.random.default_rng(4))
        return agent

    a, b = run(), run()
    pairs = zip(a.actor.state_dict().values(), b.actor.state_dict().values(), strict=True)
    assert all(torch.equal(x, y) for x, y in pairs)
    c = PPO(356, PPOParams(target_kl=None))
    c.load_state_dict(a.state_dict())
    assert c.updates == a.updates
    s = samples(4)
    np.testing.assert_array_equal(c.value(s.obs, s.states), a.value(s.obs, s.states))
    with pytest.raises(ValueError, match="different configuration"):
        PPO(356, PPOParams(variant="ippo")).load_state_dict(a.state_dict())


def test_act_returns_allowed_actions_and_their_log_probabilities() -> None:
    configure_torch(2)
    agent = PPO(356, PPOParams())
    s = samples(32)
    masks = np.ones((32, 5), dtype=bool)
    masks[:, 2] = False
    acts, logp = agent.act(s.obs, masks, np.random.default_rng(0))
    assert not np.any(acts == 2)
    ref = masked_log_probs(agent.actor(torch.from_numpy(s.obs)), torch.from_numpy(masks))
    np.testing.assert_allclose(logp, ref.gather(1, torch.from_numpy(acts)[:, None])[:, 0].detach(),
                               rtol=1e-6)  # fmt: skip


def test_explained_variance_and_parameter_validation() -> None:
    assert explained_variance([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert explained_variance([0, 0], [1, 1]) is None
    assert PPOParams().iteration_steps == 1_000
    with pytest.raises(ConfigError):
        PPOParams(variant="vdn")
    with pytest.raises(ConfigError):
        PPOParams(rollout=0)
