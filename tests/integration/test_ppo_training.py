"""PPO training pipeline, checkpoints, resume, the learned controller and evaluation (Issue #6).

X-7 for the PPO actor (inference-only export, sidecar, tamper check), reproducibility from config
+ seed, iteration-boundary resume (documented as not bitwise identical), the ``ppo`` controller in
the CLI with both shields, the stochastic evaluation mode and the matched baseline.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robocoord.cli import main  # noqa: E402
from robocoord.marl.checkpoint import (  # noqa: E402
    ACTOR_SPEC_VERSION,
    SIDECAR_FIELDS,
    load_inference,
    load_training,
    save_inference,
)
from robocoord.marl.config import ObsParams  # noqa: E402
from robocoord.marl.env import MultiRobotEnv  # noqa: E402
from robocoord.marl.evaluation import (  # noqa: E402
    EvalTask,
    run_episode,
    stochastic_rng,
)
from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.policy import LearnedController  # noqa: E402
from robocoord.marl.ppo import PPO, PPOParams  # noqa: E402
from robocoord.marl.ppo_trainer import PPOTrainer, load_ppo_config, train_ppo  # noqa: E402
from tests.integration.test_marl_env import small  # noqa: E402
from tests.integration.test_marl_training import checkpoint as madql_checkpoint  # noqa: E402, F401

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "training" / "ppo_smoke.toml"
SCEN = ROOT / "configs" / "scenarios"


def export(agent: PPO, path: Path, hidden: tuple[int, ...]) -> Path:
    return save_inference(path, agent.actor, hidden=hidden, obs=ObsParams(), meta={
        "algorithm": "ppo", "variant": agent.variant, "config_hash": "x", "config": {},
        "seeds": {"run_seed": 0}, "env_steps": 0, "validation": {}, "experiment_id": "test",
        "evaluation_mode": "greedy",
    }, model_version=ACTOR_SPEC_VERSION)  # fmt: skip


@pytest.fixture(scope="module")
def ppo_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configure_torch(13)
    agent = PPO(ObsParams().dim, PPOParams(hidden=(32, 32), critic_hidden=(32, 32)))
    return export(agent, tmp_path_factory.mktemp("ppo") / "actor.pt", (32, 32))


# ---------------------------------------------------------------------------------- X-7
def test_actor_export_round_trip_is_inference_only(tmp_path: Path) -> None:
    configure_torch(4)
    agent = PPO(ObsParams().dim, PPOParams())  # the full 356-256-256-5 actor
    path = export(agent, tmp_path / "a.pt", (256, 256))
    net, meta, _ = load_inference(path)
    assert set(SIDECAR_FIELDS) <= set(meta) and meta["model"]["version"] == ACTOR_SPEC_VERSION
    assert set(net.state_dict()) == set(agent.actor.state_dict())  # no critic, no optimiser
    assert path.stat().st_size < 5 * 2**20
    obs = np.random.default_rng(0).normal(size=(16, 356)).astype(np.float32)
    with torch.no_grad():
        assert torch.equal(net(torch.from_numpy(obs)), agent.actor(torch.from_numpy(obs)))
    ctrl = LearnedController.from_checkpoint(path)
    assert ctrl.output == "logits" and ctrl.name == "ppo"


def test_tampered_actor_checkpoint_is_rejected(ppo_checkpoint: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(ppo_checkpoint.read_bytes() + b"x")
    bad.with_suffix(".json").write_text(ppo_checkpoint.with_suffix(".json").read_text())
    with pytest.raises(ValueError, match="SHA-256"):
        load_inference(bad)
    meta = json.loads(ppo_checkpoint.with_suffix(".json").read_text())
    meta["model"]["version"] = "mlp-pi-v9"
    other = tmp_path / "other.pt"
    other.write_bytes(ppo_checkpoint.read_bytes())
    other.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unsupported"):
        load_inference(other)


# ---------------------------------------------------------------------------------- training
def smoke(**overrides: object) -> object:
    return load_ppo_config(SMOKE, **overrides)


def run_dir(tmp: Path, cfg: object, name: str) -> tuple[dict, Path]:
    out = tmp / name
    return train_ppo(cfg, out), out  # type: ignore[arg-type]


def log_without_timing(path: Path) -> list[dict]:
    skip = {"wall_s", "validation_s", "train_steps_per_s", "peak_rss_mb"}
    return [{k: v for k, v in json.loads(x).items() if k not in skip}
            for x in path.read_text().splitlines()]  # fmt: skip


@pytest.mark.parametrize("variant", ["mappo", "ippo"])
def test_training_is_reproducible_from_config_and_seed(tmp_path: Path, variant: str) -> None:
    cfg = smoke()
    cfg = replace(cfg, ppo=replace(cfg.ppo, variant=variant))  # type: ignore[attr-defined]
    s1, a = run_dir(tmp_path, cfg, "a")
    s2, b = run_dir(tmp_path, cfg, "b")
    _, c = run_dir(tmp_path, replace(cfg, run_seed=2), "c")
    assert s1["env_steps"] == s2["env_steps"] == 200 and s1["robot_steps"] > 200
    assert (a / "best.pt").read_bytes() == (b / "best.pt").read_bytes()
    assert log_without_timing(a / "train_log.jsonl") == log_without_timing(b / "train_log.jsonl")
    assert (a / "best.pt").read_bytes() != (c / "best.pt").read_bytes()
    side = json.loads((a / "best.json").read_text())
    assert side["variant"] == variant and side["critic"]["included"] is False
    assert side["critic"]["input"] == (615 if variant == "mappo" else 356)
    state = load_training(a / "checkpoints" / "step_00000200.train.pt")
    assert state["kind"] == "training" and "critic" in state["agent"]
    assert state["specs"]["actor"] == ACTOR_SPEC_VERSION


def test_training_refuses_to_overwrite_and_checks_its_configuration(tmp_path: Path) -> None:
    run_dir(tmp_path, smoke(), "a")
    with pytest.raises(FileExistsError):
        train_ppo(smoke(), tmp_path / "a")  # type: ignore[arg-type]
    cfg = smoke()
    with pytest.raises(Exception, match="multiple"):
        replace(cfg, total_steps=150)  # type: ignore[type-var]
    with pytest.raises(Exception, match="RRT"):
        replace(cfg, env=replace(cfg.env, planner="rrtstar"))  # type: ignore[attr-defined]
    with pytest.raises(Exception, match="shield"):
        replace(cfg, env=replace(cfg.env, shield="predictive"))  # type: ignore[attr-defined]


class InterruptedRunError(Exception):
    pass


def test_resume_at_an_iteration_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = smoke(total_steps=400)
    cfg = replace(
        cfg,
        log_interval=100,  # type: ignore[type-var]
        validation=replace(cfg.validation, interval=200),
    )  # type: ignore[attr-defined]
    out = tmp_path / "run"
    original = PPOTrainer._progress

    def interrupt_at_300(self: PPOTrainer, *args: object) -> None:
        original(self, *args)  # type: ignore[arg-type]
        if self.env_steps == 300:
            raise InterruptedRunError

    monkeypatch.setattr(PPOTrainer, "_progress", interrupt_at_300)
    with pytest.raises(InterruptedRunError):
        train_ppo(cfg, out)  # type: ignore[arg-type]
    monkeypatch.setattr(PPOTrainer, "_progress", original)
    summary = train_ppo(cfg, out, resume=True)  # type: ignore[arg-type]
    assert summary["env_steps"] == 400
    (res,) = summary["resumes"]
    assert res["resumed_at_env_steps"] == 200 and res["bitwise_identical"] is False
    assert res["unfinished_episodes_discarded"] == cfg.ppo.n_envs  # type: ignore[attr-defined]
    assert res["records_moved"]["train_log.jsonl"] >= 1
    log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["env_steps"] for r in log if r["type"] == "progress"] == [100, 200, 300, 400]
    assert any(r["type"] == "resume" for r in log)
    assert (out / "train_log.pre_resume_1.jsonl").exists()  # the interrupted attempt is kept
    with pytest.raises(ValueError, match="configuration differs"):
        PPOTrainer(replace(cfg, run_seed=9), out, resume=True)  # type: ignore[type-var]
    with pytest.raises(FileNotFoundError):
        PPOTrainer(cfg, tmp_path / "empty", resume=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------- controller
def cli(*args: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = main(list(args))
    return code, buf.getvalue()


@pytest.mark.parametrize("planner", ["astar", pytest.param("rrtstar", marks=pytest.mark.slow)])
@pytest.mark.parametrize("shield", ["off", "predictive"])
def test_cli_selects_ppo(ppo_checkpoint: Path, planner: str, shield: str) -> None:
    code, out = cli("simulate", "--scenario", str(SCEN / "demo_delivery.toml"), "--planner",
                    planner, "--allocator", "auction", "--controller", "ppo", "--checkpoint",
                    str(ppo_checkpoint), "--shield", shield, "--rrt-iterations", "300",
                    "--max-steps", "30", "--json")  # fmt: skip
    assert code == 0
    m = json.loads(out)
    assert m["physical_collisions"] == 0 and "controller" in m["extra"]


def test_cli_rejects_a_checkpoint_of_the_other_algorithm(
    ppo_checkpoint: Path,
    madql_checkpoint: Path,  # noqa: F811
) -> None:
    base = ("simulate", "--scenario", str(SCEN / "demo_delivery.toml"), "--planner", "astar",
            "--allocator", "auction", "--max-steps", "5", "--json")  # fmt: skip
    assert cli(*base, "--controller", "madql", "--checkpoint", str(ppo_checkpoint))[0] == 2
    assert cli(*base, "--controller", "ppo", "--checkpoint", str(madql_checkpoint))[0] == 2
    assert cli(*base, "--controller", "ppo")[0] == 2


def test_env_and_ppo_controller_choose_identical_greedy_actions(ppo_checkpoint: Path) -> None:
    env = MultiRobotEnv()
    obs, masks, _ = env.reset_spec(small(10009))
    ctrl = LearnedController.from_checkpoint(ppo_checkpoint)
    acts = ctrl.act(env.view)
    net, _, _ = load_inference(ppo_checkpoint)
    with torch.no_grad():
        logits = net(torch.from_numpy(obs))
    expected = torch.where(torch.from_numpy(masks), logits, torch.finfo(logits.dtype).min)
    assert [int(acts[i]) for i in range(len(acts))] == expected.argmax(1).tolist()


# ---------------------------------------------------------------------------------- evaluation
def test_stochastic_mode_is_seeded_per_instance_not_per_policy(
    ppo_checkpoint: Path,
    madql_checkpoint: Path,  # noqa: F811
) -> None:
    t1 = EvalTask("ppo", "static", 10_000, checkpoint=str(ppo_checkpoint), mode="stochastic",
                  replicate=1, label="mappo-seed1")  # fmt: skip
    t2 = replace(t1, controller="madql", checkpoint=str(madql_checkpoint), label="madql-seed2")
    assert stochastic_rng(t1).random(5).tolist() == stochastic_rng(t2).random(5).tolist()
    assert stochastic_rng(t1).random() != stochastic_rng(replace(t1, replicate=2)).random()
    r1, r2 = run_episode(t1), run_episode(t1)
    drop = {"wall_s", "prepare_s", "act_s"}
    assert {k: v for k, v in r1.items() if k not in drop} == {
        k: v for k, v in r2.items() if k not in drop
    }
    assert r1["mode"] == "stochastic" and r1["replicate"] == 1 and r1["physical_collisions"] == 0
    m = run_episode(replace(t2, level="static"))
    assert m["mode"] == "stochastic" and m["physical_collisions"] == 0
    with pytest.raises(ValueError, match="not a ppo"):
        run_episode(replace(t1, checkpoint=str(madql_checkpoint)))
    with pytest.raises(ValueError, match="stochastic"):
        run_episode(EvalTask("baseline", "static", 10_000, mode="stochastic"))


def test_matched_baseline_runs_without_the_resolver() -> None:
    orig = run_episode(EvalTask("baseline", "static", 10_000))
    matched = run_episode(EvalTask("baseline-noresolver", "static", 10_000))
    assert orig["deadlock"] is True and matched["deadlock"] is False
    for r in (orig, matched):
        assert r["physical_collisions"] == 0
        for key in (
            "robots",
            "undelivered",
            "conflicts_per_delivered",
            "shaped_return",
            "shield_interventions",
            "prepare_s",
            "act_s",
            "allocator",
        ):
            assert key in r, key  # fmt: skip
        assert r["undelivered"] == r["released"] - r["delivered"]
    seq = run_episode(EvalTask("baseline-noresolver", "static", 10_000, allocator="sequential"))
    assert seq["allocator"] == "sequential"


def test_training_seed_order_wraps_around_instead_of_failing(tmp_path: Path) -> None:
    """PPO can finish more than 10 000 episodes within the budget: the fixed order restarts
    (training seeds only) and every wrap is logged."""
    cfg = smoke(total_steps=400)
    trainer = PPOTrainer(cfg, tmp_path / "wrap")  # type: ignore[arg-type]
    trainer.order = trainer.order[:3]
    seeds = [trainer.next_scenario(0).seed for _ in range(7)]
    assert seeds == [*trainer.order, *trainer.order, trainer.order[0]]
    assert [w["pass"] for w in trainer.wraps] == [2, 3]
    assert all(0 <= s <= 9_999 for s in seeds)
    summary = trainer.train()
    log = [json.loads(x) for x in (tmp_path / "wrap" / "train_log.jsonl").read_text().splitlines()]
    assert sum(r["type"] == "seed_order_wrap" for r in log) >= 2
    assert summary["seed_order_passes"] > 2
