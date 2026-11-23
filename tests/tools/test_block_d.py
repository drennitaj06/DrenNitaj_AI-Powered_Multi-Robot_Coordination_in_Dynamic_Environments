"""Tests of the Block D runner and analysis. **No test generates a Block D scenario**: an autouse
sentinel replaces the scenario sampler for Block D seeds with one that raises, and every check
that would hash Block D instances runs on synthetic fingerprints."""

from __future__ import annotations

import copy
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "selection"))
import block_c_analysis as bca
import block_d
import block_d_analysis as bda

SEEDS = {"astar": tuple(range(12_000, 12_012)), "rrtstar": tuple(range(12_000, 12_006))}
DESIGN = bda.Design(
    pairs={r: {"old": f"step_{r}_old.pt", "new": f"step_{r}_new.pt"} for r in bda.PAIR_RUNS},
    unchanged={"ippo_seed2": "step_u2.pt", "ippo_seed3": "step_u3.pt"},
)
FPS = {lvl: {s: f"{lvl}-{s}" for s in SEEDS["astar"]} for lvl in bda.LEVELS}


class BlockDGenerationError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_block_d_generation(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Any call of the scenario sampler with a Block D seed raises (and is recorded)."""
    calls: list[int] = []
    try:
        from robocoord.marl.scenarios import ScenarioDist
    except ImportError:  # pragma: no cover
        yield calls
        return
    original = ScenarioDist.sample

    def sentinel(self: Any, seed: int) -> Any:
        if int(seed) in range(12_000, 12_200):
            calls.append(int(seed))
            raise BlockDGenerationError(f"Block D seed {seed} must not be generated in tests")
        return original(self, seed)

    monkeypatch.setattr(ScenarioDist, "sample", sentinel)
    yield calls


def rec(planner: str, unit: str, arm: str, lvl: str, s: int, **kw: Any) -> dict[str, Any]:
    ck = (DESIGN.pairs[unit][arm] if arm in bda.PAIR_ARMS
          else DESIGN.unchanged.get(unit) if arm == bda.V5_ARM else None)  # fmt: skip
    r: dict[str, Any] = {
        **bda.COMMON, "controller": bda.CONTROLLER[arm], "planner": planner, "label": unit,
        "checkpoint": ck, "arm": arm, "level": lvl, "seed": s, "robots": 4, "max_steps": 100,
        "instance_sha256": f"{lvl}-{s}", "status": "ok", "steps": 100, "released": 10,
        "delivered": 9, "conflicts_total": 3, "conflicts": {"RRv": 3, "INV": 0},
        "physical_collisions": 0, "terminated": False, "base_return": 1.0,
        "mean_flow_time": 20.0, "planner_calls": 12, "wall_s": 0.5, "act_s": 0.1,
        "prepare_s": 0.2,
    }  # fmt: skip
    r.update(kw)
    return r


def records(seed: int = 0, effect: float = 0.0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for planner in bda.PLANNERS:
        for unit, arm in DESIGN.units(planner):
            for lvl in bda.LEVELS:
                for s in SEEDS[planner]:
                    mu = 6.0 * (1 - effect) if (unit, arm) == ("mappo_seed1", "new") else 6.0
                    k = int(rng.negative_binomial(2.0, 2.0 / (2.0 + mu)))
                    rel = int(rng.integers(8, 14))
                    d = rel - int(rng.integers(0, 2))
                    out.append(rec(planner, unit, arm, lvl, s, released=rel, delivered=d,
                                   conflicts_total=k, conflicts={"RRv": k, "INV": 0},
                                   base_return=float(rng.normal()), terminated=d == rel,
                                   steps=100 if d < rel else 80))  # fmt: skip
    return out


def build(recs: list[dict[str, Any]]) -> bda.Data:
    return bda.build(recs, DESIGN, FPS, seeds=SEEDS)


# --------------------------------------------------------------------------- frozen constants


def test_design_constants() -> None:
    assert range(12_000, 12_200) == bda.BLOCK_D
    assert range(12_000, 12_060) == bda.BLOCK_D_RRT
    assert bda.PAIR_RUNS == ("mappo_seed1", "mappo_seed2", "mappo_seed3", "ippo_seed1")
    assert bda.UNCHANGED_RUNS == ("ippo_seed2", "ippo_seed3")
    assert bda.FAMILY_ALPHA == {"F1": 0.03, "F2": 0.01, "F3": 0.01}
    assert sum(bda.FAMILY_ALPHA.values()) == pytest.approx(0.05)
    assert (bda.CI_ALPHA, bda.N_BOOT, bda.BOOT_SEED, bda.MC_K) == (0.05, 10_000, 0, 2.0)
    sizes = {f: len(bda.PAIR_RUNS) * len(outs) for f, (_, outs) in bda.FAMILIES.items()}
    assert sizes == {"F1": 8, "F2": 12, "F3": 8}
    assert bda.FAMILIES["F3"][0] == "rrtstar" and bda.FAMILIES["F1"][0] == "astar"
    assert bda.COMMON["mode"] == "greedy" and bda.COMMON["shield"] == "off"
    for r in (*block_d.COMPARE_WITH.values(), *block_d.FORBIDDEN.values()):
        assert not set(r) & set(bda.BLOCK_D)


def test_reused_block_c_analysis_is_the_frozen_file() -> None:
    bda.verify_frozen_dependency()
    assert bda.N_BOOT == bca.N_BOOT and bda.BOOT_SEED == bca.BOOT_SEED


def test_frozen_dependency_mismatch_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "fake.py"
    p.write_text("x = 1\n")
    with pytest.raises(bda.ConfigError):
        bda.verify_frozen_dependency(p)


# --------------------------------------------------------------------------- seed guards


@pytest.mark.parametrize(("planner", "s"), [("astar", 12_000), ("astar", 12_199),
                                            ("rrtstar", 12_000), ("rrtstar", 12_059)])  # fmt: skip
def test_guard_accepts_the_design(planner: str, s: int) -> None:
    block_d.guard_task(planner, s, "D")


@pytest.mark.parametrize(
    ("planner", "s", "block"),
    [("rrtstar", 12_060, "D"), ("rrtstar", 12_199, "D"), ("astar", 12_200, "D"),
     ("astar", 11_999, "D"), ("astar", 11_000, "D"), ("astar", 13_000, "D"),
     ("astar", 10_000, "D"), ("astar", 20_000, "D"), ("astar", 0, "D"),
     ("astar", 12_000, "C"), ("astar", 12_000, "A"), ("dijkstra", 12_000, "D")],
)  # fmt: skip
def test_guard_rejects_everything_else(planner: str, s: int, block: str) -> None:
    with pytest.raises(ValueError):
        block_d.guard_task(planner, s, block)


def test_worker_seed_check() -> None:
    assert block_d.worker_check_seeds([12_000, 12_199], "D") == [12_000, 12_199]
    for bad, block in ((11_000, "D"), (20_000, "D"), (12_000, "C"), (12_200, "D")):
        with pytest.raises(ValueError):
            block_d.worker_check_seeds([bad], block)


def test_sentinel_blocks_block_d_generation(no_block_d_generation: list[int]) -> None:
    pytest.importorskip("robocoord")
    from robocoord.marl.scenarios import LEVELS

    with pytest.raises(BlockDGenerationError):
        LEVELS["static"].sample(12_000)
    assert no_block_d_generation == [12_000]
    no_block_d_generation.clear()


def test_worker_guard_runs_before_the_scenario_generator(no_block_d_generation: list[int]) -> None:
    pytest.importorskip("robocoord")
    from robocoord.marl.evaluation import EvalTask

    bad = [("rrtstar", 12_100, "D"), ("astar", 11_000, "D"), ("astar", 12_000, "C")]
    for planner, s, block in bad:
        t = EvalTask("ppo", "full", s, planner, block, "x.pt", label="mappo_seed1")
        with pytest.raises(ValueError):
            block_d._run(("mappo_seed1", "old", t))
    assert no_block_d_generation == []


def test_fingerprint_helper_refuses_training_and_test_seeds() -> None:
    for s in (0, 9_999, 20_000):
        with pytest.raises(ValueError):
            block_d.fingerprints([s])


# --------------------------------------------------------------------------- task matrix


def test_task_matrix_counts_pairing_and_order(no_block_d_generation: list[int]) -> None:
    pytest.importorskip("robocoord")
    jobs = block_d.tasks(DESIGN)
    assert len(jobs) == 8_220
    by_planner = {p: [j for j in jobs if j[2].planner == p] for p in bda.PLANNERS}
    assert len(by_planner["astar"]) == 6_600 and len(by_planner["rrtstar"]) == 1_620
    count: dict[tuple[str, str], int] = {}
    for unit, arm, t in jobs:
        kind = (t.planner, "pair" if arm in bda.PAIR_ARMS else arm)
        count[kind] = count.get(kind, 0) + 1
        assert t.block == "D" and t.mode == "greedy" and t.shield == "off"
        assert t.allocator == "auction" and t.rrt_iterations == 1000 and t.replicate == 0
        assert t.seed in bda.SEEDS[t.planner]
        if unit == bda.BASELINE:
            assert t.controller == "baseline-noresolver" and t.checkpoint is None
        else:
            assert t.controller == "ppo" and t.checkpoint.endswith(".pt")
    assert count == {("astar", "pair"): 4_800, ("astar", "v5"): 1_200,
                     ("astar", "baseline"): 600, ("rrtstar", "pair"): 1_440,
                     ("rrtstar", "baseline"): 180}  # fmt: skip
    assert not any(t.planner == "rrtstar" and u in bda.UNCHANGED_RUNS for u, _, t in jobs)
    # interleaved: the first 11 jobs are every A* arm of seed 12000 / static
    first = jobs[:11]
    assert {(t.seed, t.level, t.planner) for _, _, t in first} == {(12_000, "static", "astar")}
    assert len({(u, a) for u, a, _ in first}) == 11
    pairs: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for unit, arm, t in jobs:
        if arm in bda.PAIR_ARMS:
            pairs.setdefault((t.planner, t.level, t.seed, unit), {})[arm] = t
    assert len(pairs) == 2_400 + 720 == 3_120
    seed_units = {(p, s, u) for p, _, s, u in pairs}
    assert len(seed_units) == 4 * 200 + 4 * 60 == 1_040
    for pair in pairs.values():
        a, b = pair["old"], pair["new"]
        for f in ("controller", "level", "seed", "planner", "block", "shield", "rrt_iterations",
                  "label", "allocator", "mode", "replicate"):  # fmt: skip
            assert getattr(a, f) == getattr(b, f)
        assert a.checkpoint != b.checkpoint
    assert no_block_d_generation == []


# --------------------------------------------------------------------------- freeze / check


def _fake_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    files = [tmp_path / f"f{i}.txt" for i in range(3)]
    for f in files:
        f.write_text(f.name)
    monkeypatch.setattr(block_d, "FROZEN", tuple(files))
    monkeypatch.setattr(block_d, "freeze_manifest", lambda: {
        "hashes": block_d.frozen_hashes(), "scbc_code_matches_selections": True,
        "selections_matches_recorded": True})  # fmt: skip
    return files


def test_freeze_list_covers_scbc_and_the_test_suite() -> None:
    names = {str(p).replace("\\", "/") for p in block_d.FROZEN}
    assert {
        "docs/research/2026-11-21-block-d-protocol.md",
        "tools/selection/block_d.py",
        "tools/selection/block_d_analysis.py",
        "tools/selection/block_c_analysis.py",
        "tools/selection/block_c.py",
        "tools/selection/selection_block.py",
        "tools/selection/scbc_frozen.py",
        "tools/selection/scbc.py",
        "tests/tools/test_block_d.py",
    } == names


@pytest.mark.parametrize("bad", ["scbc_code_matches_selections", "selections_matches_recorded"])
def test_freeze_refuses_changed_scbc_or_selections(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_frozen(tmp_path, monkeypatch)
    man = {"hashes": {}, "scbc_code_matches_selections": True, "selections_matches_recorded": True}
    monkeypatch.setattr(block_d, "freeze_manifest", lambda: {**man, bad: False})
    with pytest.raises(SystemExit):
        block_d.freeze(tmp_path / "out")
    assert not (tmp_path / "out" / "plan_freeze.json").exists()


def test_real_freeze_manifest_is_read_only_and_consistent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    if not (root / "results" / "training").exists():
        pytest.skip("results/ not present")
    pytest.importorskip("torch")
    import os

    cwd = os.getcwd()
    try:
        os.chdir(root)
        before = sorted(p.name for p in (root / "results" / "dev6a").iterdir())
        man = block_d.freeze_manifest()
        assert sorted(p.name for p in (root / "results" / "dev6a").iterdir()) == before
    finally:
        os.chdir(cwd)
    assert man["scbc_code_matches_selections"] and man["selections_matches_recorded"]
    assert set(man["hashes"]) == {str(p).replace("\\", "/") for p in block_d.FROZEN}
    assert len(man["checkpoints"]) == 4 * 4 + 2 * 3
    assert set(man["environment"]) == {"python", "numpy", "torch", "platform"}


# --------------------------------------------------------------------------- real worker processes


class SamplerCalledError(RuntimeError):
    """Raised by the probe sampler installed inside a worker process."""


def _probe_init() -> None:
    """Worker initializer: the runner's own initializer, then a sampler that never generates."""
    block_d._init_worker()
    from robocoord.marl.scenarios import ScenarioDist

    def never(self: Any, seed: int) -> Any:
        raise SamplerCalledError(f"sampler called with seed {seed}")

    ScenarioDist.sample = never  # type: ignore[method-assign]


def _worker_state() -> tuple[int, bool, bool]:
    from robocoord.marl import evaluation

    return (os.getpid(), vars(evaluation)["check_seeds"] is block_d.worker_check_seeds,
            bool(getattr(evaluation.run_episode, "block_d_guard", False)))  # fmt: skip


def _worker_run_episode(task: Any) -> None:
    from robocoord.marl.evaluation import run_episode

    run_episode(task)


def _start_methods() -> list[str]:
    import multiprocessing as mp

    return [m for m in ("spawn", "fork") if m in mp.get_all_start_methods()]


@pytest.mark.parametrize("method", _start_methods())
def test_seed_guard_is_enforced_inside_real_worker_processes(method: str) -> None:
    """A real process pool with the runner's initializer: out-of-range tasks are rejected inside
    the worker before the sampler is called, both through the runner's task function and through
    the production ``run_episode`` called directly; an in-range task reaches the (probe) sampler
    only after the guards. The probe sampler raises and never generates a scenario."""
    pytest.importorskip("robocoord")
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from robocoord.marl.evaluation import EvalTask

    bad = [("rrtstar", 12_060, "D"), ("rrtstar", 12_199, "D"), ("astar", 11_000, "D"),
           ("astar", 13_000, "D"), ("astar", 20_000, "D"), ("astar", 12_000, "C"),
           ("astar", 12_200, "D")]  # fmt: skip
    with ProcessPoolExecutor(1, mp_context=mp.get_context(method), initializer=_probe_init) as pool:
        pid, replaced, wrapped = pool.submit(_worker_state).result()
        assert pid != os.getpid() and replaced and wrapped
        for planner, s, block in bad:
            t = EvalTask("ppo", "full", s, planner, block, "x.pt", label="mappo_seed1")
            # layer 1: the runner's task function
            with pytest.raises(ValueError, match=r"not allowed"):
                pool.submit(block_d._run, ("mappo_seed1", "old", t)).result()
            # layers 2-3: the production episode runner itself, wrapped in the worker
            with pytest.raises(ValueError, match=r"not allowed"):
                pool.submit(_worker_run_episode, t).result()
        ok = EvalTask("ppo", "full", 12_000, "astar", "D", "x.pt", label="mappo_seed1")
        with pytest.raises(SamplerCalledError):
            pool.submit(block_d._run, ("mappo_seed1", "old", ok)).result()


def test_freeze_precedes_check_and_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _fake_frozen(tmp_path, monkeypatch)
    out = tmp_path / "out"
    rep = block_d.freeze(out)
    assert set(rep["hashes"]) == {str(f).replace("\\", "/") for f in files}
    assert block_d._require_freeze(out)["hashes"] == rep["hashes"]
    files[0].write_text("changed")
    with pytest.raises(SystemExit):
        block_d._require_freeze(out)
    for name in ("check.json", "records.jsonl"):
        out2 = tmp_path / f"o_{name}"
        out2.mkdir()
        (out2 / name).write_text("{}")
        with pytest.raises(SystemExit):
            block_d.freeze(out2)


def test_check_and_run_refuse_without_freeze(tmp_path: Path,
                                             no_block_d_generation: list[int]) -> None:  # fmt: skip
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit, match="not frozen"):
        block_d.check(out)
    with pytest.raises(SystemExit, match="not frozen"):
        block_d.run(out, 1)
    assert no_block_d_generation == []


@pytest.mark.parametrize("cmd", ["check", "run", "replay"])
def test_cli_needs_block_d_authorization(cmd: str, tmp_path: Path,
                                         no_block_d_generation: list[int]) -> None:  # fmt: skip
    with pytest.raises(SystemExit, match="authorized-block-d"):
        block_d.main([cmd, "--out", str(tmp_path)])
    assert no_block_d_generation == []


def test_cli_freeze_needs_authorization(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="authorized-freeze"):
        block_d.main(["freeze", "--out", str(tmp_path)])
    assert not (tmp_path / "plan_freeze.json").exists()


def _synthetic_check_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                         fd: dict[str, list[str]], train: str = "") -> Path:  # fmt: skip
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "training" / "r").mkdir(parents=True)
    (tmp_path / "results" / "x.jsonl").write_text('{"seed": 11000}\n')
    if train:
        (tmp_path / "results" / "training" / "r" / "log.jsonl").write_text(
            json.dumps({"instance_sha256": train}) + "\n")  # fmt: skip
    files = _fake_frozen(tmp_path, monkeypatch)
    assert files
    stored_c = {lvl: [f"C-{lvl}-{s}" for s in block_d.COMPARE_WITH["C"]] for lvl in bda.LEVELS}
    (tmp_path / "c.json").write_text(json.dumps({"fingerprints": stored_c}))
    monkeypatch.setattr(block_d, "BLOCK_C_CHECK", tmp_path / "c.json")
    monkeypatch.setattr(block_d, "picks", lambda: {"ok": True, "runs": {}})
    monkeypatch.setattr(bda, "verify_frozen_dependency", lambda *a: None)

    def fake_fp(block: Any) -> dict[str, list[str]]:
        seeds = list(block)
        if seeds == list(bda.BLOCK_D):
            return fd
        name = next(n for n, r in block_d.COMPARE_WITH.items() if list(r) == seeds)
        return {lvl: [f"{name}-{lvl}-{s}" for s in seeds] for lvl in bda.LEVELS}

    monkeypatch.setattr(block_d, "fingerprints", fake_fp)
    out = tmp_path / "out"
    block_d.freeze(out)
    return out


def _fd(dup: bool = False, clash: str = "") -> dict[str, list[str]]:
    fd = {lvl: [f"D-{lvl}-{s}" for s in bda.BLOCK_D] for lvl in bda.LEVELS}
    if dup:
        fd["full"][1] = fd["full"][0]
    if clash:
        fd["static"][5] = clash
    return fd


def test_check_logic_on_synthetic_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_block_d_generation: list[int]
) -> None:
    out = _synthetic_check_env(tmp_path, monkeypatch, _fd())
    rep = block_d.check(out)
    assert rep["ok"] and rep["files_with_block_d_seeds"] == {}
    assert rep["training_disjointness"]["verified"] is False
    assert rep["unique_instances_in_D"] == {lvl: 200 for lvl in bda.LEVELS}
    assert "previous_in_memory_generations" in rep
    assert no_block_d_generation == []


TRAIN_HASH = "a" * 64


@pytest.mark.parametrize("bad", ["dup", "clash_S", "clash_train", "seed_in_results"])
def test_check_fails_on_any_violation(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clash = {"clash_S": "S-static-13000", "clash_train": TRAIN_HASH}.get(bad, "")
    fd = _fd(dup=bad == "dup", clash=clash)
    out = _synthetic_check_env(tmp_path, monkeypatch, fd,
                               train=TRAIN_HASH if bad == "clash_train" else "")  # fmt: skip
    if bad == "seed_in_results":
        (tmp_path / "results" / "y.jsonl").write_text('{"seed": 12007}')
    rep = block_d.check(out)
    assert not rep["ok"]
    if bad == "clash_train":
        assert rep["training_disjointness"] == {"verified": True, "overlap": 1}


def test_scan_ignores_the_output_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "dev6a" / "block_d").mkdir(parents=True)
    (tmp_path / "results" / "dev6a" / "block_d" / "records.jsonl").write_text('{"seed": 12000}')
    (tmp_path / "results" / "a.json").write_text('{"seed": 12001, "x": {"seed": 1}}')
    n, hits = block_d.scan_results(Path("results/dev6a/block_d"))
    assert n == 1 and hits == {str(Path("results/a.json")): 1}


def test_real_training_logs_store_no_instance_hashes() -> None:
    root = Path(__file__).resolve().parents[2]
    if not (root / "results" / "training").exists():
        pytest.skip("results/ not present")
    import os

    cwd = os.getcwd()
    try:
        os.chdir(root)
        assert block_d.stored_training_instance_hashes() == set()
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------- gate


def test_complete_records_are_usable() -> None:
    d = build(records())
    assert all(d.usable.values()) and not any(d.problems.values())
    a = d.arrays["astar", "ippo_seed2", "v5"]
    assert a["episodes"].sum() == 3 * len(SEEDS["astar"])


@pytest.mark.parametrize(
    ("mutate", "word"),
    [
        (lambda r: r.update(status="error", error_type="PhysicalCollisionError"), "failed"),
        (lambda r: r.update(status="error", error_type="InvalidTransitionError"), "failed"),
        (lambda r: r.update(delivered=99), "invalid"),
        (lambda r: r.update(terminated=False, steps=50), "invalid"),
        (lambda r: r.update(base_return=None), "invalid"),
        (lambda r: r.update(conflicts={"RRv": 0}), "invalid"),
    ],
)
def test_unusable_episode_makes_run_planner_not_evaluable(mutate: Any, word: str) -> None:
    recs = records()
    i = next(i for i, r in enumerate(recs) if r["planner"] == "rrtstar"
             and r["label"] == "mappo_seed1" and r["arm"] == "new")  # fmt: skip
    mutate(recs[i])
    d = build(recs)
    assert not d.usable["mappo_seed1", "rrtstar"]
    assert d.usable["mappo_seed1", "astar"]
    assert any(p.startswith(word) for p in d.problems["mappo_seed1", "rrtstar"])
    assert sum(not v for v in d.usable.values()) == 1


def test_missing_and_duplicate_episodes() -> None:
    recs = records()
    gone = recs.pop(3)
    d = build(recs)
    assert not d.usable[gone["label"], gone["planner"]]
    recs = records()
    recs.append(copy.deepcopy(recs[3]))
    d = build(recs)
    assert any(p.startswith("duplicate") for p in d.problems[recs[3]["label"], recs[3]["planner"]])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(mode="stochastic"),
        lambda r: r.update(block="C"),
        lambda r: r.update(shield="predictive"),
        lambda r: r.update(allocator="sequential"),
        lambda r: r.update(arm="new" if r["arm"] == "old" else "old"),
        lambda r: r.update(checkpoint="step_other.pt"),
        lambda r: r.update(label="madql_seed1"),
        lambda r: r.update(seed=12_500),
        lambda r: r.update(instance_sha256="other"),
        lambda r: r.update(robots=5),
        lambda r: r.update(controller="baseline"),
    ],
)
def test_configuration_mismatch_aborts(mutate: Any) -> None:
    recs = records()
    mutate(recs[0])
    with pytest.raises(bda.ConfigError):
        build(recs)


def test_rrtstar_seed_outside_subset_and_unchanged_run_on_rrtstar_abort() -> None:
    for extra in (rec("rrtstar", "mappo_seed1", "old", "static", 12_011),
                  rec("rrtstar", "ippo_seed2", "v5", "static", 12_000)):  # fmt: skip
        with pytest.raises(bda.ConfigError):
            build([*records(), extra])


def test_baseline_with_a_checkpoint_aborts() -> None:
    recs = records()
    i = next(i for i, r in enumerate(recs) if r["label"] == bda.BASELINE)
    recs[i]["checkpoint"] = "x.pt"
    with pytest.raises(bda.ConfigError):
        build(recs)


# --------------------------------------------------------------------------- analysis


def test_analysis_structure_alpha_and_labels() -> None:
    res = bda.analyse(build(records()), n_boot=499)
    fams = res["families"]
    assert {f: len(v) for f, v in fams.items()} == {"F1": 8, "F2": 12, "F3": 8}
    for f, rows in fams.items():
        alpha = bda.FAMILY_ALPHA[f]
        p = [r["p_for_holm"] for r in rows]
        assert [r["p_holm"] for r in rows] == pytest.approx(bca.holm(p))
        assert [r["holm_threshold"] for r in rows] == pytest.approx(bda.holm_thresholds(p, alpha))
        for r in rows:
            assert r["alpha_family"] == alpha
            assert r["reject"] == (r["p_holm"] <= alpha)
            assert r["ci95_nominal"][0] <= r["estimate"] <= r["ci95_nominal"][1]
            assert isinstance(r["mc_sensitive"], bool)
    assert "unadjusted" in res["labels"]["intervals"]
    assert all("p" not in r for r in res["per_level"])
    assert all("p" not in r for r in res["vs_matched_baseline"])
    assert len(res["rrtstar_supportive_outcomes"]) == 4 * 3


def test_holm_thresholds_known_values() -> None:
    thr = bda.holm_thresholds([0.004, 0.02, 0.001, 0.5], 0.03)
    assert thr == pytest.approx([0.03 / 3, 0.03 / 2, 0.03 / 4, 0.03 / 1])


def test_same_p_can_be_rejected_in_f1_but_not_f3() -> None:
    p = 0.02
    assert bca.holm([p])[0] <= bda.FAMILY_ALPHA["F1"]
    assert not bca.holm([p])[0] <= bda.FAMILY_ALPHA["F3"]


def test_estimates_equal_the_frozen_block_c_primitives() -> None:
    data = build(records(seed=3))
    res = bda.analyse(data, n_boot=999)
    row = next(r for r in res["families"]["F1"]
               if r["run"] == "mappo_seed2" and r["outcome"] == "completion")  # fmt: skip
    args = bda._args(data, "astar", ("mappo_seed2", "old"), ("mappo_seed2", "new"),
                     "completion", [0, 1, 2])  # fmt: skip
    inf = bca.bootstrap_t(bca.paired_ratio, args, bca.boot_index(12, 999, 0), 0.05)
    assert row["estimate"] == inf.estimate and row["p"] == inf.p and row["se"] == inf.se
    assert row["ci95_nominal"] == list(inf.ci)


def test_planted_effect_is_detected_in_f1() -> None:
    res = bda.analyse(build(records(seed=7, effect=0.85)), n_boot=999)
    conf = {r["run"]: r for r in res["families"]["F1"] if r["outcome"] == "conflicts_per_episode"}
    assert conf["mappo_seed1"]["estimate"] < 0 and conf["mappo_seed1"]["reject"]


def test_not_evaluable_pair_has_no_estimates_anywhere() -> None:
    recs = records(seed=7, effect=0.85)
    i = next(i for i, r in enumerate(recs) if r["planner"] == "rrtstar"
             and r["label"] == "mappo_seed1")  # fmt: skip
    recs.pop(i)
    res = bda.analyse(build(recs), n_boot=499)
    f3 = [r for r in res["families"]["F3"] if r["run"] == "mappo_seed1"]
    assert len(res["families"]["F3"]) == 8  # m unchanged
    for r in f3:
        assert r["evaluable"] is False and r["p_for_holm"] == 1.0 and r["reject"] is False
        assert not {"estimate", "ci95_nominal", "p", "se", "old", "new"} & set(r)
        assert r["mc_sensitive"] is False
    assert all(r["evaluable"] for r in res["families"]["F1"])
    assert not [r for r in res["per_level"] if r["run"] == "mappo_seed1"
                and r["planner"] == "rrtstar"]  # fmt: skip
    assert not [r for r in res["rrtstar_supportive_outcomes"] if r["run"] == "mappo_seed1"]
    vb = [r for r in res["vs_matched_baseline"] if r["run"] == "mappo_seed1"
          and r["planner"] == "rrtstar"]  # fmt: skip
    assert vb and all(not r["data_complete"] and "estimate" not in r for r in vb)


def test_unusable_descriptive_unit_reports_counts_only() -> None:
    recs = records()
    i = next(i for i, r in enumerate(recs) if r["label"] == "ippo_seed2")
    recs[i].update(status="error", error_type="RuntimeError")
    res = bda.analyse(build(recs), n_boot=199)
    rows = [r for r in res["summaries"] if r["unit"] == "ippo_seed2"]
    assert rows and all(not r["data_complete"] and "completion" not in r for r in rows)
    desc = bda.descriptives(recs, DESIGN)
    allrow = next(r for r in desc if r["unit"] == "ippo_seed2" and r["level"] == "all")
    assert allrow["error_types"] == {"RuntimeError": 1} and allrow["classes"]["failed"] == 1


def test_mc_sensitivity_flag() -> None:
    data = build(records(seed=11, effect=0.35))
    res = bda.analyse(data, n_boot=999)
    for rows in res["families"].values():
        for r in rows:
            if r["evaluable"]:
                expect = abs(r["p"] - r["holm_threshold"]) <= 2 * r["p_mc_se"]
                assert r["mc_sensitive"] == expect


def test_analysis_is_deterministic_and_order_invariant() -> None:
    recs = records(seed=5)
    a = bda.analyse(build(recs), n_boot=199)
    b = bda.analyse(build(list(reversed(recs))), n_boot=199)
    assert a == b


def test_descriptives_classes_and_safety() -> None:
    recs = records()
    recs[0].update(conflicts={"RRv": recs[0]["conflicts_total"] - 1, "INV": 1})
    desc = bda.descriptives(recs, DESIGN)
    tot = sum(r["episodes"] for r in desc if r["level"] == "all")
    assert tot == len(recs) == 3 * (11 * 12 + 9 * 6)
    for r in desc:
        assert sum(r["classes"].values()) == r["episodes"]
        assert set(r["classes"]) <= {"successful", "incomplete", "horizon_all_delivered"}
    assert sum(r["invalid_moves_INV"] for r in desc if r["level"] == "all") == 1


def test_pairing_counts_on_synthetic_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(block_d, "SEEDS", SEEDS)
    p = block_d.pairing(records(), DESIGN)
    assert p["astar"] == {"episode_pairs": 4 * 3 * 12, "expected": 4 * 3 * 12,
                          "identical_paired_fields": 4 * 3 * 12, "seed_units": 4 * 12}  # fmt: skip
    assert p["rrtstar"]["episode_pairs"] == 4 * 3 * 6
