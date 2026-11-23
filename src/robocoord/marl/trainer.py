"""MADQL training pipeline (architecture §7; algorithms §6). Requires PyTorch.

Reproducible from ``(config, run seed)``:

* scenario seeds come from 0–9 999 in one fixed order that is the same for every run
  (``scenarios.training_seed_order(sampler_seed)``); the ``k``-th episode started uses the
  ``k``-th seed, and the curriculum stage active at that environment step chooses the
  distribution (stage parameters are a function of the stage and the seed);
* the algorithm's randomness comes from streams derived from the run seed only: ``explore``
  (ε-greedy), ``replay`` (minibatch sampling) and ``torch`` (network initialisation);
* deterministic algorithms and one intra-op thread are enabled (ADR-8).

Step counts are environment steps summed over the ``E`` vectorised copies (D3). Validation runs
the 60 block-A A* episodes with the greedy policy; the checkpoint selection is lexicographic —
overall completion, then mean discounted base return (approved). RRT* validation (10 block-A
seeds, full dynamics) is reported but never selects. Training uses A* only (approved).

Issue 6A Phase 4: with ``madql.n_step`` > 1 the transitions pass through a
:class:`~robocoord.marl.replay.NStepAccumulator` (one queue per copy) before they are stored;
with the default ``n_step = 1`` the loop is exactly the Issue #5 one.
"""

from __future__ import annotations

import json
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.core.errors import ConfigError
from robocoord.marl.checkpoint import (
    REWARD_SPEC_VERSION,
    config_hash,
    save_inference,
    save_training,
)
from robocoord.marl.config import EnvConfig, ObsParams, RewardParams
from robocoord.marl.env import MultiRobotEnv
from robocoord.marl.evaluation import (
    block_a_rrt_tasks,
    block_a_tasks,
    run_tasks,
    selection_key,
    summarise,
)
from robocoord.marl.madql import MADQL, MADQLParams
from robocoord.marl.networks import configure_torch
from robocoord.marl.replay import NStepAccumulator, ReplayBuffer
from robocoord.marl.scenarios import CURRICULUM, Stage, stage_at, training_seed_order
from robocoord.marl.vector_env import SyncVectorEnv
from robocoord.scenario.spec import ScenarioSpec


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    interval: int = 50_000  # environment steps between block-A validations (0 = only at the end)
    rrt_interval: int = 250_000  # environment steps between RRT* validations (0 = only at the end)
    rrt: bool = True
    workers: int = 3
    seeds_per_level: int = 20  # block-A seeds per level (20 = the approved 60 A* episodes)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    name: str = "madql"
    run_seed: int = 1
    total_steps: int = 500_000
    n_envs: int = 4
    sampler_seed: int = 0
    log_interval: int = 10_000
    deterministic: bool = True
    threads: int = 1
    madql: MADQLParams = field(default_factory=MADQLParams)
    env: EnvConfig = field(default_factory=EnvConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.n_envs < 1:
            raise ConfigError("run", "total_steps and n_envs must be >= 1")
        if self.env.planner != "astar":
            raise ConfigError("env.planner", "initial MADQL training uses A* only (approved)")
        if self.env.shield != "off":
            raise ConfigError("env.shield", "the shield is off during training (ADR-10)")
        if abs(self.madql.gamma - self.env.reward.gamma) > 1e-12:
            raise ConfigError("madql.gamma", "must equal reward.gamma (the shaping discount)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["madql"] = self.madql.to_dict()
        return d


def load_training_config(path: str | Path, **overrides: Any) -> TrainingConfig:
    """Parse a TOML training configuration (``configs/training/madql*.toml``)."""
    data = tomllib.loads(Path(path).read_text("utf-8"))
    run = dict(data.get("run", {}))
    madql = dict(data.get("madql", {}))
    if "hidden" in madql:
        madql["hidden"] = tuple(madql["hidden"])
    env = dict(data.get("env", {}))
    obs = ObsParams(**data.get("obs", {}))
    reward = RewardParams(**data.get("reward", {}))
    cfg = TrainingConfig(
        **run,
        madql=MADQLParams(**madql),
        env=EnvConfig(**env, obs=obs, reward=reward),
        validation=ValidationConfig(**data.get("validation", {})),
    )
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


class Trainer:
    def __init__(
        self,
        cfg: TrainingConfig,
        out_dir: str | Path,
        curriculum: Sequence[Stage] = CURRICULUM,
    ) -> None:
        self.cfg = cfg
        self.curriculum = tuple(curriculum)  # default: the approved curriculum (design R7)
        self.out = Path(out_dir)
        if (self.out / "train_log.jsonl").exists():
            raise FileExistsError(f"{self.out} already holds a training run; use a new directory")
        self.out.mkdir(parents=True, exist_ok=True)
        root = np.random.SeedSequence(cfg.run_seed)
        explore, replay, torch_seq = root.spawn(3)
        self.explore_rng = np.random.default_rng(explore)
        self.replay_rng = np.random.default_rng(replay)
        torch_seed = int(torch_seq.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
        configure_torch(torch_seed, cfg.deterministic, cfg.threads)
        self.agent = MADQL(cfg.env.obs.dim, cfg.madql)
        self.buffer = ReplayBuffer(cfg.madql.buffer_size, cfg.env.obs)
        self.order = training_seed_order(cfg.sampler_seed)
        self.episode_index = 0
        self.env_steps = 0
        self.best: tuple[float, float] | None = None
        self.best_step = -1
        self.cfg_dict = cfg.to_dict()
        self.cfg_hash = config_hash(self.cfg_dict)
        self._log = (self.out / "train_log.jsonl").open("a", encoding="utf-8")
        self._val = (self.out / "validation.jsonl").open("a", encoding="utf-8")
        (self.out / "config.json").write_text(
            json.dumps(
                {"config": self.cfg_dict, "hash": self.cfg_hash, "environment": _environment()},
                indent=2,
            ),
            "utf-8",
        )

    # ------------------------------------------------------------------ scenarios
    def next_scenario(self, _copy: int) -> ScenarioSpec:
        if self.episode_index >= len(self.order):
            raise RuntimeError("the 10 000 training scenario seeds are exhausted")
        seed = self.order[self.episode_index]
        self.episode_index += 1
        stage = stage_at(self.env_steps, self.cfg.total_steps, self.curriculum)
        return self.curriculum[stage].dist.sample(seed)

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict[str, Any]:
        cfg, p = self.cfg, self.cfg.madql
        envs = [MultiRobotEnv(config=cfg.env) for _ in range(cfg.n_envs)]
        venv = SyncVectorEnv(envs, self.next_scenario)
        obs, masks = venv.reset()
        nstep = NStepAccumulator(p.n_step, p.gamma, cfg.n_envs) if p.n_step > 1 else None
        next_update, next_target = p.train_freq, p.target_period
        next_log = cfg.log_interval
        v = cfg.validation
        next_val = v.interval if v.interval > 0 else cfg.total_steps
        next_rrt = v.rrt_interval if v.rrt_interval > 0 else cfg.total_steps
        losses: list[float] = []
        episodes: list[dict[str, Any]] = []
        t_start = time.perf_counter()
        t_val = 0.0
        while self.env_steps < cfg.total_steps:
            eps = p.epsilon(self.env_steps, cfg.total_steps)
            sizes = [len(o) for o in obs]
            acts = self.agent.act(np.concatenate(obs), np.concatenate(masks), eps, self.explore_rng)
            split = np.split(acts, np.cumsum(sizes)[:-1])
            res = venv.step(split)
            for k, tr in enumerate(res.transitions):
                n = sizes[k]
                if nstep is None:
                    self.buffer.add(obs[k], split[k], tr.rewards, tr.next_obs, tr.next_masks,
                                    np.full(n, float(tr.terminated)))  # fmt: skip
                    continue
                out = nstep.push(k, obs[k], split[k], tr.rewards, tr.next_obs, tr.next_masks,
                                 tr.terminated, tr.truncated)  # fmt: skip
                if out is not None:
                    self.buffer.add(out.obs, out.actions, out.rewards, out.next_obs,
                                    out.next_masks, out.dones, out.horizons)  # fmt: skip
            self.env_steps += cfg.n_envs
            while self.env_steps >= next_update:
                if len(self.buffer) >= p.warmup:
                    losses.append(
                        self.agent.update(self.buffer.sample(p.batch_size, self.replay_rng))
                    )
                next_update += p.train_freq
            while self.env_steps >= next_target:
                self.agent.sync_target()
                next_target += p.target_period
            for ep in res.episodes:
                stage = stage_at(self.env_steps, cfg.total_steps, self.curriculum)
                ep = {**ep, "env_steps": self.env_steps, "epsilon": eps, "stage": stage}
                episodes.append(ep)
                self._write(self._log, {"type": "episode", **ep})
            obs, masks = res.obs, res.masks
            if self.env_steps >= next_log:
                elapsed = time.perf_counter() - t_start
                self._write(self._log, {
                    "type": "progress", "env_steps": self.env_steps, "epsilon": eps,
                    "loss_mean": float(np.mean(losses)) if losses else None,
                    "updates": self.agent.updates, "buffer": len(self.buffer),
                    "episodes": self.episode_index, "wall_s": elapsed, "validation_s": t_val,
                    "train_steps_per_s": self.env_steps / max(1e-9, elapsed - t_val),
                })  # fmt: skip
                losses = []
                next_log += cfg.log_interval
            final = self.env_steps >= cfg.total_steps
            if self.env_steps >= next_val or final:
                tv = time.perf_counter()
                run_rrt = v.rrt and (self.env_steps >= next_rrt or final)
                self.validate(run_rrt)
                t_val += time.perf_counter() - tv
                while next_val <= self.env_steps:
                    next_val += v.interval if v.interval > 0 else cfg.total_steps
                if run_rrt:
                    while next_rrt <= self.env_steps:
                        next_rrt += v.rrt_interval if v.rrt_interval > 0 else cfg.total_steps
        wall = time.perf_counter() - t_start
        summary = {
            "env_steps": self.env_steps,
            "episodes": self.episode_index,
            "updates": self.agent.updates,
            "wall_s": wall,
            "validation_s": t_val,
            "best_step": self.best_step,
            "best_key": self.best,
            "config_hash": self.cfg_hash,
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
        self._log.close()
        self._val.close()
        return summary

    # ------------------------------------------------------------------ validation
    def _meta(self, validation: dict[str, Any]) -> dict[str, Any]:
        return {
            "algorithm": MADQL.algorithm,
            "variant": MADQL.variant,
            "reward": {"version": REWARD_SPEC_VERSION, **asdict(self.cfg.env.reward)},
            "config_hash": self.cfg_hash,
            "config": self.cfg_dict,
            "seeds": {"run_seed": self.cfg.run_seed, "sampler_seed": self.cfg.sampler_seed},
            "env_steps": self.env_steps,
            "validation": validation,
            "experiment_id": f"{self.cfg.name}-seed{self.cfg.run_seed}",
        }

    def validate(self, rrt: bool) -> dict[str, Any]:
        step = self.env_steps
        ckpt = self.out / "checkpoints" / f"step_{step:08d}.pt"
        save_inference(ckpt, self.agent.q, hidden=self.cfg.madql.hidden, obs=self.cfg.env.obs,
                       meta=self._meta({"pending": True}))  # fmt: skip
        save_training(self.out / "checkpoints" / f"step_{step:08d}.train.pt",
                      {**self.agent.state_dict(), "env_steps": step})  # fmt: skip
        tasks = block_a_tasks(str(ckpt), seeds=self.cfg.validation.seeds_per_level)
        recs = run_tasks(tasks, self.cfg.validation.workers)
        summ = summarise(recs)
        key = selection_key(summ)
        entry: dict[str, Any] = {"type": "validation", "env_steps": step, "planner": "astar",
                                 "summary": summ, "key": list(key), "records": recs}  # fmt: skip
        if rrt:
            rrecs = run_tasks(block_a_rrt_tasks(str(ckpt)), self.cfg.validation.workers)
            entry["rrtstar"] = {"summary": summarise(rrecs), "records": rrecs}
        improved = self.best is None or key > self.best
        if improved:
            self.best, self.best_step = key, step
        entry["best_so_far"] = {"env_steps": self.best_step, "key": list(self.best or key)}
        self._write(self._val, entry)
        val = {"block_a_astar": summ, "key": list(key), "selection": "completion, then base return"}
        if rrt:
            val["block_a_rrtstar_full"] = entry["rrtstar"]["summary"]
        save_inference(ckpt, self.agent.q, hidden=self.cfg.madql.hidden, obs=self.cfg.env.obs,
                       meta=self._meta(val))  # fmt: skip
        if improved:
            save_inference(self.out / "best.pt", self.agent.q, hidden=self.cfg.madql.hidden,
                           obs=self.cfg.env.obs, meta=self._meta(val))  # fmt: skip
        return entry

    @staticmethod
    def _write(fh: Any, rec: dict[str, Any]) -> None:
        fh.write(json.dumps(rec, default=_json_default) + "\n")
        fh.flush()


def _environment() -> dict[str, Any]:
    """Software and hardware of the run (``05`` B.2)."""
    import os
    import platform

    import torch

    import robocoord

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "robocoord": robocoord.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def train(
    cfg: TrainingConfig, out_dir: str | Path, curriculum: Sequence[Stage] = CURRICULUM
) -> dict[str, Any]:
    return Trainer(cfg, out_dir, curriculum).train()
