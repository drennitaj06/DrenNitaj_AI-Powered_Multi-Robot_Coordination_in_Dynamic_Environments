"""PPO learning sanity (algorithms §7 test 6; marked ``training``, run with ``-m training``).

A short end-to-end MAPPO/IPPO training on the 2-robot 6×6 static family of the MADQL sanity
test must improve the greedy policy's return over a uniformly random masked policy, checked
statistically across scenario seeds. Only this *sanity* claim is tested; performance claims come
from the experiments.
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
from robocoord.marl.ppo_trainer import load_ppo_config, train_ppo  # noqa: E402
from robocoord.marl.scenarios import Stage  # noqa: E402
from tests.integration.test_marl_learning_sanity import (  # noqa: E402
    EVAL_SEEDS,
    TINY,
    episode_return,
)

pytestmark = pytest.mark.training
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("variant", ["mappo", "ippo"])
def test_short_ppo_training_beats_random_policy(variant: str, tmp_path: Path) -> None:
    cfg = load_ppo_config(ROOT / "configs" / "training" / "ppo_smoke.toml", run_seed=1)
    cfg = replace(
        cfg,
        total_steps=30_000,
        log_interval=10_000,
        ppo=replace(cfg.ppo, variant=variant, n_envs=8, rollout=125, minibatches=4,
                    hidden=(64, 64), critic_hidden=(64, 64)),
    )  # fmt: skip
    train_ppo(cfg, tmp_path, curriculum=(Stage(1.0, TINY),))
    net, _, _ = load_inference(next((tmp_path / "checkpoints").glob("*.pt")))
    env = MultiRobotEnv()
    rng = np.random.default_rng(0)
    learned = [episode_return(env, s, net, rng) for s in EVAL_SEEDS]
    random = [episode_return(env, s, None, rng) for s in EVAL_SEEDS]
    assert np.mean(learned) > np.mean(random)
    assert wilcoxon_p(learned, random) < 0.05
