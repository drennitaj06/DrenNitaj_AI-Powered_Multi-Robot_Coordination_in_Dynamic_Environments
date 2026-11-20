"""PPO training pipeline (MAPPO / IPPO; architecture §7; algorithms §7; Issue #6). Requires PyTorch.

A separate trainer (D9): the MADQL trainer of Issue #5 is unchanged. Everything that defines the
comparison is shared with it — the environment composition, the curriculum and the scenario order,
block-A validation and the checkpoint-selection rule — through ``marl.scenarios``,
``marl.evaluation`` and ``marl.checkpoint``.

Reproducible from ``(config, run seed)``:

* scenario seeds come from 0–9 999 in the fixed order of ``scenarios.training_seed_order`` (the
  same for every run and every algorithm); the stage active at the episode's start chooses the
  distribution;
* the algorithm's randomness comes from streams derived from the run seed only: ``sample``
  (action sampling during collection), ``shuffle`` (minibatches) and ``torch`` (initialisation);
* deterministic algorithms and one intra-op thread (ADR-8).

Step counts are environment steps summed over the ``E`` copies (one environment step is one joint
step of one copy, D3); robot-steps (actions taken) are reported separately. One iteration collects
``E · T_r`` = 1 000 environment steps (D4), so validations every 50 000 steps fall on iteration
boundaries.

**Resume (D8).** Training state is saved at every validation point
(``checkpoints/step_XXXXXXXX.train.pt``: actor, critic, optimiser, iteration, environment and
robot steps, the scenario position, the NumPy and torch random states, the best checkpoint so far
and the configuration hash). ``resume`` continues from the latest such point. **It is not bitwise
identical to an uninterrupted run:** the episodes in progress at the save point are not restored —
every copy restarts with the next scenario seeds of the fixed order, and its unfinished episode is
discarded. Log records written after the resume point by the interrupted attempt are moved to
``*.pre_resume_<n>.jsonl`` (never deleted), and the resume is recorded in the log and the summary.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robocoord.core.errors import ConfigError
from robocoord.marl.checkpoint import (
    ACTOR_SPEC_VERSION,
    OBS_SPEC_VERSION,
    REWARD_SPEC_VERSION,
    config_hash,
    load_training,
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
from robocoord.marl.global_state import global_state_size
from robocoord.marl.networks import configure_torch
from robocoord.marl.ppo import PPO, PPOParams
from robocoord.marl.rollout import RolloutBuffer, collect_rollout, last_values
from robocoord.marl.scenarios import CURRICULUM, Stage, stage_at, training_seed_order
from robocoord.marl.trainer import ValidationConfig, _environment, _json_default
from robocoord.marl.vector_env import SyncVectorEnv
from robocoord.scenario.spec import ScenarioSpec

CRITIC_SPEC_VERSION = "mlp-v-v1"


@dataclass(frozen=True, slots=True)
class PPOTrainingConfig:
    name: str = "mappo"
    run_seed: int = 1
    total_steps: int = 1_000_000
    sampler_seed: int = 0
    log_interval: int = 10_000
    deterministic: bool = True
    threads: int = 1
    allow_rrtstar: bool = False  # RRT* training needs explicit approval (D7); pilot only
    ppo: PPOParams = field(default_factory=PPOParams)
    env: EnvConfig = field(default_factory=EnvConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        step = self.ppo.iteration_steps
        if self.total_steps < step or self.total_steps % step:
            raise ConfigError("run.total_steps", f"must be a positive multiple of E·T_r = {step}")
        if self.validation.interval % step:
            raise ConfigError("validation.interval", f"must be a multiple of E·T_r = {step}")
        if self.env.planner != "astar" and not self.allow_rrtstar:
            raise ConfigError("env.planner", "PPO training uses A* unless RRT* is approved (D7)")
        if self.env.shield != "off":
            raise ConfigError("env.shield", "the shield is off during training (ADR-10)")
        if abs(self.ppo.gamma - self.env.reward.gamma) > 1e-12:
            raise ConfigError("ppo.gamma", "must equal reward.gamma (the shaping discount)")
        if self.ppo.global_blocks != self.env.global_blocks:
            raise ConfigError("ppo.global_blocks", "must equal env.global_blocks")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ppo"] = self.ppo.to_dict()
        d["algorithm"] = "ppo"
        return d


def is_ppo_config(path: str | Path) -> bool:
    return "ppo" in tomllib.loads(Path(path).read_text("utf-8"))


def load_ppo_config(path: str | Path, **overrides: Any) -> PPOTrainingConfig:
    """Parse a TOML PPO training configuration (``configs/training/{mappo,ippo,…}.toml``)."""
    data = tomllib.loads(Path(path).read_text("utf-8"))
    ppo = dict(data.get("ppo", {}))
    for key in ("hidden", "critic_hidden"):
        if key in ppo:
            ppo[key] = tuple(ppo[key])
    cfg = PPOTrainingConfig(
        **data.get("run", {}),
        ppo=PPOParams(**ppo),
        env=EnvConfig(
            **data.get("env", {}),
            obs=ObsParams(**data.get("obs", {})),
            reward=RewardParams(**data.get("reward", {})),
        ),
        validation=ValidationConfig(**data.get("validation", {})),
    )
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB (validation workers are separate processes).

    Windows: ``GetProcessMemoryInfo`` (peak working set); elsewhere ``getrusage``. The branch is
    selected with ``sys.platform`` (not ``os.name``) because type checkers understand that check:
    each branch is checked only against its own platform's stubs (``ctypes.WinDLL`` exists only
    in the Windows stubs, ``resource`` only in the POSIX ones).
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]  # fmt: skip

            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit pseudo-handle
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
            ]  # fmt: skip
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            c = Counters()
            c.cb = ctypes.sizeof(c)
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None
            return float(c.PeakWorkingSetSize) / 2**20
        else:
            import resource

            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB; bytes on macOS
            return float(kb) / (2**20 if sys.platform == "darwin" else 2**10)
    except Exception:  # pragma: no cover - platform specific
        return None


class PPOTrainer:
    def __init__(
        self,
        cfg: PPOTrainingConfig,
        out_dir: str | Path,
        curriculum: Sequence[Stage] = CURRICULUM,
        *,
        resume: bool = False,
    ) -> None:
        self.cfg = cfg
        self.curriculum = tuple(curriculum)
        self.out = Path(out_dir)
        if not resume and (self.out / "train_log.jsonl").exists():
            raise FileExistsError(f"{self.out} already holds a training run; use a new directory")
        self.out.mkdir(parents=True, exist_ok=True)
        root = np.random.SeedSequence(cfg.run_seed)
        sample, shuffle, torch_seq = root.spawn(3)
        self.sample_rng = np.random.default_rng(sample)
        self.shuffle_rng = np.random.default_rng(shuffle)
        torch_seed = int(torch_seq.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
        configure_torch(torch_seed, cfg.deterministic, cfg.threads)
        self.obs_dim = cfg.env.obs.dim
        self.agent = PPO(self.obs_dim, cfg.ppo)
        self.order = training_seed_order(cfg.sampler_seed)
        self.episode_index = 0
        self.env_steps = 0
        self.robot_steps = 0
        self.iteration = 0
        self.best: tuple[float, float] | None = None
        self.best_step = -1
        self.t_val = 0.0
        self.wall_before = 0.0  # wall time of earlier attempts (resume)
        self.resumes: list[dict[str, Any]] = []
        self.wraps: list[dict[str, Any]] = []  # wrap-arounds of the training seed order
        self.cfg_dict = cfg.to_dict()
        self.cfg_hash = config_hash(self.cfg_dict)
        if resume:
            self._restore()
        else:
            (self.out / "config.json").write_text(
                json.dumps(
                    {"config": self.cfg_dict, "hash": self.cfg_hash,
                     "environment": _environment()}, indent=2,
                ),
                "utf-8",
            )  # fmt: skip
        self._log = (self.out / "train_log.jsonl").open("a", encoding="utf-8")
        self._val = (self.out / "validation.jsonl").open("a", encoding="utf-8")
        if self.resumes:
            self._write(self._log, {"type": "resume", **self.resumes[-1]})

    # ------------------------------------------------------------------ resume
    def _restore(self) -> None:
        points = sorted((self.out / "checkpoints").glob("step_*.train.pt"))
        if not points:
            raise FileNotFoundError(f"{self.out} has no training checkpoint to resume from")
        state = load_training(points[-1])
        if state["config_hash"] != self.cfg_hash:
            raise ValueError("the configuration differs from the run being resumed")
        self.agent.load_state_dict(state["agent"])
        self.iteration = int(state["iteration"])
        self.env_steps = int(state["env_steps"])
        self.robot_steps = int(state["robot_steps"])
        self.episode_index = int(state["episode_index"])
        self.sample_rng.bit_generator.state = state["sample_rng"]
        self.shuffle_rng.bit_generator.state = state["shuffle_rng"]
        torch.set_rng_state(state["torch_rng"])
        best = state["best"]
        self.best = None if best is None else (float(best[0]), float(best[1]))
        self.best_step = int(state["best_step"])
        self.t_val = float(state["validation_s"])
        self.wall_before = float(state["wall_s"])
        self.resumes = [*state.get("resumes", [])]
        moved = self._quarantine_after(self.env_steps, len(self.resumes) + 1)
        self.resumes.append(
            {
                "resumed_at_env_steps": self.env_steps,
                "from": points[-1].name,
                "unfinished_episodes_discarded": self.cfg.ppo.n_envs,
                "records_moved": moved,
                "bitwise_identical": False,
            }
        )

    def _quarantine_after(self, step: int, n: int) -> dict[str, int]:
        """Move records the interrupted attempt wrote after ``step`` aside (never delete them)."""
        moved: dict[str, int] = {}
        for name in ("train_log.jsonl", "validation.jsonl"):
            path = self.out / name
            if not path.exists():
                continue
            keep: list[str] = []
            later: list[str] = []
            for line in path.read_text("utf-8").splitlines():
                rec = json.loads(line)
                (later if int(rec.get("env_steps", 0)) > step else keep).append(line)
            if later:
                shutil.copyfile(path, path.with_suffix(f".pre_resume_{n}.jsonl"))
                path.write_text("".join(x + "\n" for x in keep), "utf-8")
            moved[name] = len(later)
        return moved

    # ------------------------------------------------------------------ scenarios
    def next_scenario(self, _copy: int) -> ScenarioSpec:
        """The next seed of the fixed training order. PPO finishes episodes much faster than
        MADQL did, so the 10 000 seeds can run out within the budget; the order then starts again
        from its beginning (same seeds, same order, training block 0–9 999 only). Every wrap is
        logged; the stage active at that time chooses the distribution, as always."""
        n = len(self.order)
        if self.episode_index and self.episode_index % n == 0:
            self.wraps.append(
                {
                    "type": "seed_order_wrap",
                    "episode_index": self.episode_index,
                    "env_steps": self.env_steps,
                    "pass": self.episode_index // n + 1,
                }
            )
        seed = self.order[self.episode_index % n]
        self.episode_index += 1
        stage = stage_at(self.env_steps, self.cfg.total_steps, self.curriculum)
        return self.curriculum[stage].dist.sample(seed)

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict[str, Any]:
        cfg, p = self.cfg, self.cfg.ppo
        envs = [MultiRobotEnv(config=cfg.env) for _ in range(p.n_envs)]
        venv = SyncVectorEnv(envs, self.next_scenario, with_state=p.centralised)
        venv.reset()
        state_dim = global_state_size(p.global_blocks) if p.centralised else None

        def act(o: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return self.agent.act(o, m, self.sample_rng)

        v = cfg.validation
        next_val = self._next_multiple(v.interval)
        next_rrt = self._next_multiple(v.rrt_interval)
        next_log = self._next_multiple(cfg.log_interval)
        window: list[dict[str, Any]] = []
        t_start = time.perf_counter()
        rollout_bytes = 0
        while self.env_steps < cfg.total_steps:
            buf = RolloutBuffer(p.rollout, p.n_envs, self.obs_dim, max_robots=p.max_robots,
                                state_dim=state_dim)  # fmt: skip
            rollout_bytes = buf.nbytes
            stats = collect_rollout(venv, buf, act, self.agent.value, centralised=p.centralised)
            lv, valid = last_values(venv, buf, self.agent.value, centralised=p.centralised)
            buf.compute_gae(lv, valid, p.gamma, p.lam)
            logs = self.agent.update(buf.flatten(), self.shuffle_rng)
            self.iteration += 1
            self.env_steps += stats.env_steps
            self.robot_steps += stats.robot_steps
            window.append(logs)
            for wrap in self.wraps:
                self._write(self._log, wrap)
            self.wraps.clear()
            for ep in stats.episodes:
                stage = stage_at(self.env_steps, cfg.total_steps, self.curriculum)
                self._write(self._log, {"type": "episode", **ep, "env_steps": self.env_steps,
                                        "iteration": self.iteration, "stage": stage})  # fmt: skip
            if self.env_steps >= next_log:
                self._progress(window, t_start, rollout_bytes)
                window = []
                next_log += cfg.log_interval
            final = self.env_steps >= cfg.total_steps
            if self.env_steps >= next_val or final:
                tv = time.perf_counter()
                run_rrt = v.rrt and (self.env_steps >= next_rrt or final)
                self.validate(run_rrt)
                self.t_val += time.perf_counter() - tv
                while next_val <= self.env_steps:
                    next_val += v.interval if v.interval > 0 else cfg.total_steps
                if run_rrt:
                    while next_rrt <= self.env_steps:
                        next_rrt += v.rrt_interval if v.rrt_interval > 0 else cfg.total_steps
                self._save_state(time.perf_counter() - t_start)
        wall = self.wall_before + time.perf_counter() - t_start
        summary = {
            "algorithm": "ppo",
            "variant": p.variant,
            "env_steps": self.env_steps,
            "robot_steps": self.robot_steps,
            "iterations": self.iteration,
            "episodes": self.episode_index,
            "updates": self.agent.updates,
            "wall_s": wall,
            "validation_s": self.t_val,
            "peak_rss_mb": peak_rss_mb(),
            "rollout_bytes": rollout_bytes,
            "best_step": self.best_step,
            "best_key": self.best,
            "config_hash": self.cfg_hash,
            "resumes": self.resumes,
            "seed_order_passes": self.episode_index / len(self.order),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
        self._log.close()
        self._val.close()
        return summary

    def _next_multiple(self, interval: int) -> int:
        if interval <= 0:
            return self.cfg.total_steps
        return (self.env_steps // interval + 1) * interval

    def _progress(self, window: list[dict[str, Any]], t_start: float, rollout_bytes: int) -> None:
        elapsed = self.wall_before + time.perf_counter() - t_start
        rec: dict[str, Any] = {
            "type": "progress",
            "env_steps": self.env_steps,
            "robot_steps": self.robot_steps,
            "iteration": self.iteration,
            "updates": self.agent.updates,
            "episodes": self.episode_index,
            "wall_s": elapsed,
            "validation_s": self.t_val,
            "train_steps_per_s": self.env_steps / max(1e-9, elapsed - self.t_val),
            "peak_rss_mb": peak_rss_mb(),
            "rollout_bytes": rollout_bytes,
        }
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "actor_grad_norm",
            "critic_grad_norm",
            "epochs_run",
            "samples",
            "lr",
        ):
            vals = [w[key] for w in window if w.get(key) is not None]
            rec[key] = float(np.mean(vals)) if vals else None  # fmt: skip
        ev = [w["explained_variance"] for w in window if w.get("explained_variance") is not None]
        rec["explained_variance"] = float(np.mean(ev)) if ev else None
        rec["kl_early_stops"] = sum(bool(w.get("kl_early_stop")) for w in window)
        self._write(self._log, rec)

    # ------------------------------------------------------------------ validation and state
    def _meta(self, validation: dict[str, Any]) -> dict[str, Any]:
        p = self.cfg.ppo
        critic = self.agent.critic
        return {
            "algorithm": "ppo",
            "variant": p.variant,
            "variant_description": self.agent.variant_description,
            "evaluation_mode": "greedy",
            "critic": {
                "version": CRITIC_SPEC_VERSION,
                "input": self.agent.critic_dim,
                "inputs": "[o_i, phi(s)]" if p.centralised else "o_i",
                "hidden": list(p.critic_hidden),
                "parameters": sum(t.numel() for t in critic.state_dict().values()),
                "included": False,  # inference checkpoints hold the actor only (ADR-9)
            },
            "reward": {"version": REWARD_SPEC_VERSION, **asdict(self.cfg.env.reward)},
            "config_hash": self.cfg_hash,
            "config": self.cfg_dict,
            "seeds": {"run_seed": self.cfg.run_seed, "sampler_seed": self.cfg.sampler_seed},
            "env_steps": self.env_steps,
            "robot_steps": self.robot_steps,
            "iteration": self.iteration,
            "validation": validation,
            "experiment_id": f"{self.cfg.name}-seed{self.cfg.run_seed}",
            "checkpoint_kind": "inference",
        }

    def _export(self, path: Path, validation: dict[str, Any]) -> Path:
        return save_inference(path, self.agent.actor, hidden=self.cfg.ppo.hidden,
                              obs=self.cfg.env.obs, meta=self._meta(validation),
                              model_version=ACTOR_SPEC_VERSION)  # fmt: skip

    def validate(self, rrt: bool) -> dict[str, Any]:
        step = self.env_steps
        ckpt = self.out / "checkpoints" / f"step_{step:08d}.pt"
        self._export(ckpt, {"pending": True})
        tasks = block_a_tasks(str(ckpt), controller="ppo",
                              seeds=self.cfg.validation.seeds_per_level)  # fmt: skip
        recs = run_tasks(tasks, self.cfg.validation.workers)
        summ = summarise(recs)
        key = selection_key(summ)
        entry: dict[str, Any] = {"type": "validation", "env_steps": step, "planner": "astar",
                                 "summary": summ, "key": list(key), "records": recs}  # fmt: skip
        if rrt:
            rrecs = run_tasks(block_a_rrt_tasks(str(ckpt), controller="ppo"),
                              self.cfg.validation.workers)  # fmt: skip
            entry["rrtstar"] = {"summary": summarise(rrecs), "records": rrecs}
        improved = self.best is None or key > self.best
        if improved:
            self.best, self.best_step = key, step
        entry["best_so_far"] = {"env_steps": self.best_step, "key": list(self.best or key)}
        self._write(self._val, entry)
        val = {"block_a_astar": summ, "key": list(key), "selection": "completion, then base return"}
        if rrt:
            val["block_a_rrtstar_full"] = entry["rrtstar"]["summary"]
        self._export(ckpt, val)
        if improved:
            self._export(self.out / "best.pt", val)
        return entry

    def _save_state(self, wall_now: float) -> None:
        save_training(
            self.out / "checkpoints" / f"step_{self.env_steps:08d}.train.pt",
            {
                "kind": "training",
                "agent": self.agent.state_dict(),
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "robot_steps": self.robot_steps,
                "episode_index": self.episode_index,
                "sample_rng": self.sample_rng.bit_generator.state,
                "shuffle_rng": self.shuffle_rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "best": None if self.best is None else list(self.best),
                "best_step": self.best_step,
                "validation_s": self.t_val,
                "wall_s": self.wall_before + wall_now,
                "config_hash": self.cfg_hash,
                "specs": {"actor": ACTOR_SPEC_VERSION, "critic": CRITIC_SPEC_VERSION,
                          "observation": OBS_SPEC_VERSION, "reward": REWARD_SPEC_VERSION},
                "resumes": self.resumes,
            },
        )  # fmt: skip

    @staticmethod
    def _write(fh: Any, rec: dict[str, Any]) -> None:
        fh.write(json.dumps(rec, default=_json_default) + "\n")
        fh.flush()


def train_ppo(
    cfg: PPOTrainingConfig,
    out_dir: str | Path,
    curriculum: Sequence[Stage] = CURRICULUM,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    return PPOTrainer(cfg, out_dir, curriculum, resume=resume).train()
