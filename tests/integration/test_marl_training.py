"""Training pipeline, checkpoints and the learned controller (Issue #5, Step 4).

X-7 checkpoint round-trip, training reproducibility from config + seed, X-1 with a learned
controller, the controller with both planners and the shield on or off, CLI selection, and the
evaluation records used for validation (block A) and development comparisons (block B).
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
    SIDECAR_FIELDS,
    load_inference,
    load_training,
    save_inference,
)
from robocoord.marl.config import EnvConfig, ObsParams  # noqa: E402
from robocoord.marl.env import VIEW_EXTRAS, MultiRobotEnv, make_coordinator  # noqa: E402
from robocoord.marl.evaluation import (  # noqa: E402
    EvalTask,
    block_a_tasks,
    run_episode,
    selection_key,
    summarise,
)
from robocoord.marl.madql import MADQL, MADQLParams  # noqa: E402
from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.policy import LearnedController, greedy_actions  # noqa: E402
from robocoord.marl.scenarios import LEVELS, SeedBlockError  # noqa: E402
from robocoord.marl.trainer import load_training_config, train  # noqa: E402
from robocoord.sim.engine import Engine  # noqa: E402
from robocoord.sim.trace import TraceWriter  # noqa: E402
from tests.integration.test_marl_env import small  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "configs" / "training" / "madql_smoke.toml"
FULL = ROOT / "configs" / "training" / "madql.toml"
SCEN = ROOT / "configs" / "scenarios"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small random-initialised policy saved as an inference checkpoint."""
    configure_torch(11)
    agent = MADQL(ObsParams().dim, MADQLParams(hidden=(32, 32)))
    path = tmp_path_factory.mktemp("ckpt") / "policy.pt"
    save_inference(path, agent.q, hidden=(32, 32), obs=ObsParams(), meta={
        "algorithm": "madql", "variant": MADQL.variant, "config_hash": "x", "config": {},
        "seeds": {"run_seed": 0}, "env_steps": 0, "validation": {}, "experiment_id": "test",
    })  # fmt: skip
    return path


# ---------------------------------------------------------------------------------- X-7
def test_checkpoint_round_trip_identical_actions(checkpoint: Path) -> None:
    configure_torch(11)
    agent = MADQL(ObsParams().dim, MADQLParams(hidden=(32, 32)))  # same seed → same weights
    net, meta, obs_params = load_inference(checkpoint)
    rng = np.random.default_rng(0)
    obs = rng.random((64, ObsParams().dim)).astype(np.float32)
    masks = rng.random((64, 5)) < 0.7
    masks[:, 0] = True
    assert np.array_equal(greedy_actions(agent.q, obs, masks), greedy_actions(net, obs, masks))
    assert obs_params == ObsParams()
    for key in SIDECAR_FIELDS:
        assert key in meta, key
    assert checkpoint.stat().st_size < 5 * 1024 * 1024  # ADR-9 limit for committed weights
    state = torch.load(checkpoint, weights_only=True)
    assert set(state) == {k for k, _ in net.state_dict().items()}  # inference-only weights


def test_tampered_checkpoint_is_rejected(checkpoint: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(checkpoint.read_bytes() + b"x")
    bad.with_suffix(".json").write_text(checkpoint.with_suffix(".json").read_text())
    with pytest.raises(ValueError, match="SHA-256"):
        load_inference(bad)


# ---------------------------------------------------------------------------------- training
def run_smoke(out: Path, seed: int = 1) -> dict:
    return train(load_training_config(SMOKE, run_seed=seed), out)


def test_training_is_reproducible_from_config_and_seed(tmp_path: Path) -> None:
    a = run_smoke(tmp_path / "a")
    b = run_smoke(tmp_path / "b")
    c = run_smoke(tmp_path / "c", seed=2)

    def episodes(d: Path) -> list[dict]:
        rows = [json.loads(x) for x in (d / "train_log.jsonl").read_text().splitlines()]
        return [r for r in rows if r["type"] == "episode"]

    assert episodes(tmp_path / "a") == episodes(tmp_path / "b")
    assert a["best_key"] == b["best_key"] and a["updates"] == b["updates"]
    wa = load_training(next((tmp_path / "a" / "checkpoints").glob("*.train.pt")))
    wb = load_training(next((tmp_path / "b" / "checkpoints").glob("*.train.pt")))
    wc = load_training(next((tmp_path / "c" / "checkpoints").glob("*.train.pt")))
    for k in wa["q"]:
        assert torch.equal(wa["q"][k], wb["q"][k])
    assert any(not torch.equal(wa["q"][k], wc["q"][k]) for k in wa["q"])
    # the same scenario sequence for every run seed; only the algorithm's streams differ
    assert [e["seed"] for e in episodes(tmp_path / "a")] == [
        e["seed"] for e in episodes(tmp_path / "c")
    ][: len(episodes(tmp_path / "a"))]
    assert c["config_hash"] != a["config_hash"]  # the run seed is part of the config


def test_training_refuses_to_overwrite(tmp_path: Path) -> None:
    run_smoke(tmp_path / "a")
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path / "a")


def test_training_config_guards() -> None:
    cfg = load_training_config(FULL)
    assert cfg.env.planner == "astar" and cfg.env.shield == "off" and cfg.env.deadlock is False
    assert cfg.env.endpoint is True and cfg.validation.seeds_per_level == 20
    assert cfg.madql == MADQLParams()  # the documented initial hyper-parameters
    from robocoord.core.errors import ConfigError

    with pytest.raises(ConfigError, match="A\\* only"):
        replace(cfg, env=replace(cfg.env, planner="rrtstar"))


# ------------------------------------------------------------------------- X-1 / controller
def trace_of(ckpt: Path, planner: str, shield: str) -> str:
    cfg = EnvConfig(planner=planner, rrt_iterations=300 if planner == "rrtstar" else None)
    shield_obj = None
    if shield != "off":
        from robocoord.control.shield import Shield

        shield_obj = Shield(shield)
    buf = io.StringIO()
    engine = Engine(small(10008, steps=40), make_coordinator(cfg), shield=shield_obj,
                    trace=TraceWriter(buf), view_extras=VIEW_EXTRAS, path_limit=6)  # fmt: skip
    ctrl = LearnedController.from_checkpoint(ckpt)
    ctrl.reset(len(engine.robots))
    m = engine.run(ctrl.act)
    assert m.physical_collisions == 0
    return buf.getvalue()


@pytest.mark.parametrize("planner", ["astar", pytest.param("rrtstar", marks=pytest.mark.slow)])
@pytest.mark.parametrize("shield", ["off", "predictive"])
def test_learned_controller_deterministic_traces(
    checkpoint: Path, planner: str, shield: str
) -> None:
    a = trace_of(checkpoint, planner, shield)
    b = trace_of(checkpoint, planner, shield)
    assert a == b and a.count('"type":"step"') == 40


def test_learned_controller_needs_view_extras(checkpoint: Path) -> None:
    engine = Engine(small(), make_coordinator(EnvConfig()))
    ctrl = LearnedController.from_checkpoint(checkpoint)
    from robocoord.marl.observation import ObservationError

    with pytest.raises(ObservationError):
        ctrl.act(engine.prepare_step())


def test_env_and_controller_choose_identical_greedy_actions(checkpoint: Path) -> None:
    """Training (env) and execution (controller) build the same observation from the view."""
    env = MultiRobotEnv()
    obs, masks, _ = env.reset_spec(small(10009))
    ctrl = LearnedController.from_checkpoint(checkpoint)
    acts = ctrl.act(env.view)
    assert [int(acts[i]) for i in range(len(acts))] == greedy_actions(ctrl.net, obs, masks).tolist()


def cli(*args: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = main(list(args))
    return code, buf.getvalue()


@pytest.mark.parametrize("planner", ["astar", pytest.param("rrtstar", marks=pytest.mark.slow)])
@pytest.mark.parametrize("shield", ["off", "predictive"])
def test_cli_selects_madql(checkpoint: Path, planner: str, shield: str) -> None:
    code, out = cli("simulate", "--scenario", str(SCEN / "demo_delivery.toml"), "--planner",
                    planner, "--allocator", "auction", "--controller", "madql", "--checkpoint",
                    str(checkpoint), "--shield", shield, "--rrt-iterations", "300",
                    "--max-steps", "30", "--json")  # fmt: skip
    assert code == 0
    m = json.loads(out)
    assert m["physical_collisions"] == 0 and m["steps"] <= 30
    assert "controller" in m["extra"]


def test_cli_madql_without_checkpoint_fails_cleanly() -> None:
    code, _ = cli("simulate", "--scenario", str(SCEN / "demo_delivery.toml"), "--planner", "astar",
                  "--allocator", "auction", "--controller", "madql", "--no-animate")  # fmt: skip
    assert code == 2


# ---------------------------------------------------------------------------------- evaluation
def test_evaluation_records_and_selection_rule(checkpoint: Path) -> None:
    recs = [run_episode(t) for t in block_a_tasks(str(checkpoint), seeds=1)]
    assert [r["level"] for r in recs] == list(LEVELS)
    base = run_episode(EvalTask("baseline", "static", 10_000))
    assert base["physical_collisions"] == 0 and 0.0 <= base["completion"] <= 1.0
    s = summarise([*recs, base])
    assert set(s) == {"overall", "static", "obstacles", "full"}
    hi = {"overall": {"completion": 0.9, "base_return": -5.0}}
    lo = {"overall": {"completion": 0.8, "base_return": 9.0}}
    tie = {"overall": {"completion": 0.9, "base_return": -4.0}}
    assert selection_key(hi) > selection_key(lo) and selection_key(tie) > selection_key(hi)


def test_evaluation_refuses_seeds_outside_the_block() -> None:
    with pytest.raises(SeedBlockError):
        run_episode(EvalTask("baseline", "static", 10_020, block="A"))
    with pytest.raises(SeedBlockError):
        run_episode(EvalTask("baseline", "static", 20_200, block="B"))
