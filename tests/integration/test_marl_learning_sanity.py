"""Learning sanity (algorithms §6 test 6; marked ``training``, run with ``-m training``).

A short end-to-end MADQL training on a 2-robot 6×6 static family must improve the greedy policy's
return over a uniformly random masked policy, checked statistically across scenario seeds for
two training seeds. Only this *sanity* claim is tested; performance claims come from experiments.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robocoord.experiments.stats import wilcoxon_p  # noqa: E402
from robocoord.marl.checkpoint import load_inference  # noqa: E402
from robocoord.marl.env import MultiRobotEnv  # noqa: E402
from robocoord.marl.policy import greedy_actions  # noqa: E402
from robocoord.marl.scenarios import ScenarioDist, Stage  # noqa: E402
from robocoord.marl.trainer import load_training_config, train  # noqa: E402

pytestmark = pytest.mark.training

ROOT = Path(__file__).resolve().parents[2]
TINY = ScenarioDist("sanity-6x6", "static", (2,), (6,), (0.0,), (0,), 60)
EVAL_SEEDS = range(10_000, 10_020)


def episode_return(
    env: MultiRobotEnv, seed: int, policy: object, rng: np.random.Generator
) -> float:
    obs, masks, _ = env.reset_spec(TINY.sample(seed))
    total, disc = np.zeros(len(obs)), 1.0
    while True:
        if policy is None:
            a = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        else:
            a = greedy_actions(policy, obs, masks)  # type: ignore[arg-type]
        tr = env.step_arrays(a)
        total += disc * tr.components.base
        disc *= env.config.reward.gamma
        if tr.done:
            return float(total.mean())
        obs, masks = tr.next_obs, tr.next_masks


@pytest.mark.parametrize("run_seed", [1, 2])
def test_short_training_beats_random_policy(run_seed: int, tmp_path: Path) -> None:
    cfg = load_training_config(
        ROOT / "configs" / "training" / "madql_smoke.toml", run_seed=run_seed
    )
    cfg = replace(
        cfg,
        total_steps=30_000,
        madql=replace(cfg.madql, warmup=1_000, buffer_size=50_000, hidden=(64, 64), batch_size=64),
    )
    train(cfg, tmp_path, curriculum=(Stage(1.0, TINY),))
    net, _, _ = load_inference(next((tmp_path / "checkpoints").glob("*.pt")))
    env = MultiRobotEnv()
    rng = np.random.default_rng(0)
    learned = [episode_return(env, s, net, rng) for s in EVAL_SEEDS]
    random = [episode_return(env, s, None, rng) for s in EVAL_SEEDS]
    assert np.mean(learned) > np.mean(random)
    assert wilcoxon_p(learned, random) < 0.05
