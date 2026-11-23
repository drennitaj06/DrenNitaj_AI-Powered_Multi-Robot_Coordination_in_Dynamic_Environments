"""Tests of the Issue 6A Phase 6 preparation (runner, guards, freeze gates, observer, analysis).

No test generates a held-out (Block V) scenario: an autouse sentinel makes the scenario sampler
raise on any Block V seed in this process, and the worker-process test uses a probe that never
runs an allowed task. The observer test uses Block C development data only.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("phase6", "phase4", "phase2", "phase1", "phase3", "selection"):
    sys.path.insert(0, str(ROOT / "tools" / sub))

import phase6 as p6  # noqa: E402
import phase6_analysis as p6a  # noqa: E402

HIST = ROOT / "results" / "training" / "madql_seed1" / "best.pt"
PHASE1_1A = ROOT / "results" / "dev6a" / "phase1" / "eval" / "records_1a.jsonl"


class BlockVGenerationError(AssertionError):
    pass


@pytest.fixture(autouse=True)
def _no_block_v_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from robocoord.marl.scenarios import ScenarioDist

    original = ScenarioDist.sample

    def sentinel(self: Any, seed: int) -> Any:
        if int(seed) in p6.BLOCK_V:
            raise BlockVGenerationError(f"Block V seed {seed} must not be generated in tests")
        return original(self, seed)

    monkeypatch.setattr(ScenarioDist, "sample", sentinel)


def test_sentinel_blocks_block_v_generation() -> None:
    from robocoord.marl.scenarios import LEVELS

    with pytest.raises(BlockVGenerationError):
        LEVELS["static"].sample(14_000)


def test_design_counts_and_matrix() -> None:
    assert p6.episode_counts() == {"astar": 17_400, "rrtstar": 4_680, "total": 22_080}
    assert (p6.BLOCK_V.start, p6.BLOCK_V.stop, p6.BLOCK_V_RRT.stop) == (14_000, 14_200, 14_060)
    assert ("v2", "OFF") in p6.MATRIX["astar"] and ("v2", "OFF") not in p6.MATRIX["rrtstar"]
    assert p6.POLICY_RUNS["i6"] == ("mappo_seed1", "mappo_seed2", "mappo_seed3", "ippo_seed1")
    js = p6.jobs({"policies": {}})
    assert len(js) == 22_080 and len({p6.key(j) for j in js}) == 22_080
    assert {j["seed"] for j in js if j["planner"] == "rrtstar"} == set(p6.BLOCK_V_RRT)
    assert {j["seed"] for j in js} == set(p6.BLOCK_V)


def test_block_v_is_disjoint_from_every_other_block() -> None:
    for name, r in p6.FORBIDDEN.items():
        assert not set(r) & set(p6.BLOCK_V), name


@pytest.mark.parametrize(
    ("planner", "seed", "block"),
    [("astar", 11_000, "C"), ("astar", 12_000, "V"), ("astar", 13_000, "V"), ("astar", 20_000, "V"),
     ("astar", 5, "V"), ("astar", 10_050, "V"), ("rrtstar", 14_060, "V"), ("astar", 14_200, "V"),
     ("astar", 14_000, "C"), ("astar", 11_119, "V")],
)  # fmt: skip
def test_guard_rejects_outside_the_design(planner: str, seed: int, block: str) -> None:
    with pytest.raises(ValueError):
        p6.guard(planner, seed, block)


def test_guard_accepts_the_design() -> None:
    p6.guard("astar", 14_000)
    p6.guard("astar", 14_199)
    p6.guard("rrtstar", 14_059)
    with pytest.raises(ValueError):
        p6.anchor_guard("astar", 14_000, "V")
    p6.anchor_guard("rrtstar", 11_039, "C")


def test_tasks_encode_the_forward_configuration() -> None:
    base = {"stage": "6", "planner": "rrtstar", "level": "full", "seed": 14_000, "checkpoint": "x"}
    off = p6.make_task({**base, "policy": "hist", "cond": "OFF", "run": "madql_seed1"})
    rsd = p6.make_task({**base, "policy": "v2", "cond": "RSD", "run": "madql_seed2"})
    orig = p6.make_task({**base, "policy": "base", "cond": "ORIG", "run": "baseline"})
    assert (off.supervisor, off.deadlock, rsd.supervisor, rsd.deadlock) == (
        "off",
        False,
        "R+S",
        True,
    )
    assert rsd.label == "madql-v2-n3-seed2" and rsd.block == "V"
    assert (orig.controller, orig.supervisor, orig.deadlock) == ("baseline", "off", None)
    for t in (off, rsd, orig):
        assert (t.repair, t.rrt_refine, t.rrt_patience, t.rrt_iterations) == (0, None, None, 1000)
        assert (t.mode, t.allocator, t.shield) == ("greedy", "auction", "off")
    with pytest.raises(ValueError):
        p6.make_task({**base, "policy": "base", "cond": "RSD", "run": "baseline"})
    with pytest.raises(ValueError):
        p6.make_task({**base, "policy": "hist", "cond": "ORIG", "run": "madql_seed1"})


def test_commands_refuse_without_authorization(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="authorized-freeze"):
        p6.freeze(tmp_path, authorized=False)
    with pytest.raises(SystemExit, match="authorized-precheck"):
        p6.check(tmp_path, False)
    for fn in (lambda: p6.run(tmp_path, 1, False), lambda: p6.replay(tmp_path, 1, False),
               lambda: p6.instance_check(), lambda: p6.block_v_gate(tmp_path, False)):  # fmt: skip
        with pytest.raises(SystemExit, match="authorized-execution"):
            fn()
    with pytest.raises(SystemExit, match="no freeze record"):
        p6.check(tmp_path, True)


def test_freeze_refuses_after_check_and_detects_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proto = tmp_path / "protocol.md"
    proto.write_text("v1", "utf-8")
    monkeypatch.setattr(p6, "PROTOCOL", proto)
    monkeypatch.setattr(p6, "FREEZE_FILES", (str(proto),))
    out = tmp_path / "eval"
    monkeypatch.setattr(p6, "backup_status", lambda: {"ok": False, "reason": "test"})
    with pytest.raises(SystemExit, match="no verified backups and no owner-approved waiver"):
        p6.freeze(out, authorized=True)
    monkeypatch.setattr(p6, "backup_status", lambda: {"ok": True, "manifest_sha256": "x"})
    p6.freeze(out, authorized=True)
    assert p6.verify_freeze(out)["backups"]["ok"]
    assert p6.verify_freeze(out)["design"]["episodes"]["total"] == 22_080
    proto.write_text("v2", "utf-8")
    with pytest.raises(SystemExit, match="changed since the freeze"):
        p6.verify_freeze(out)
    (out / "check.json").write_text("{}", "utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        p6.freeze(out, authorized=True)


def test_guards_inside_real_worker_processes() -> None:
    methods = [m for m in ("spawn", "fork") if m in mp.get_all_start_methods()]
    for method in methods:
        ctx = mp.get_context(method)
        with ProcessPoolExecutor(1, mp_context=ctx, initializer=p6._init_worker,
                                 initargs=("v", False)) as ex:  # fmt: skip
            bad = ex.submit(p6._worker_probe, "astar", 11_000, "C").result()
            assert bad["guards_installed"] == "True"
            assert all(bad[k].startswith("rejected") for k in ("check_seeds", "guard_task",
                                                               "run_episode"))  # fmt: skip
            for seed, planner in ((12_000, "astar"), (14_100, "rrtstar"), (20_000, "astar")):
                r = ex.submit(p6._worker_probe, planner, seed, "V").result()
                assert r["guard_task"].startswith("rejected")
                assert r["run_episode"].startswith("rejected")
            ok = ex.submit(p6._worker_probe, "rrtstar", 14_059, "V").result()
            assert ok["check_seeds"] == ok["guard_task"] == "passed" and "run_episode" not in ok


def test_scan_finds_block_v_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "x").mkdir(parents=True)
    (tmp_path / "results" / "x" / "a.jsonl").write_text('{"seed": 11000}\n', "utf-8")
    assert p6.scan_results(tmp_path / "results" / "out")["clean"]
    (tmp_path / "results" / "x" / "b.jsonl").write_text('{"seed": 14005}\n', "utf-8")
    s = p6.scan_results(tmp_path / "results" / "out")
    assert not s["clean"] and len(s["files_with_block_v_seeds"]) == 1


@pytest.mark.skipif(not HIST.exists() or not PHASE1_1A.exists(), reason="stored data absent")
def test_observer_on_block_c_reproduces_phase1_and_counts_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robocoord.control import supervisor as sup
    from robocoord.marl import evaluation
    from robocoord.marl.policy import LearnedController

    monkeypatch.setattr(sup.ExecutionSupervisor, "apply", sup.ExecutionSupervisor.apply)
    monkeypatch.setattr(evaluation, "check_seeds", getattr(evaluation, "check_seeds"))  # noqa: B009
    monkeypatch.setattr(evaluation, "run_episode", evaluation.run_episode)
    monkeypatch.setattr(LearnedController, "choose", LearnedController.choose)
    p6._init_worker("c", True)
    stored = next(
        json.loads(x) for x in PHASE1_1A.read_text("utf-8").splitlines()
        if '"rsd_r0"' in x and '"madql_seed1"' in x and '"seed": 11001' in x and '"static"' in x
    )  # fmt: skip
    j = {"stage": "test", "arm": "hist:RSD", "policy": "hist", "cond": "RSD", "run": "madql_seed1",
         "planner": "astar", "level": "static", "seed": 11_001, "checkpoint": str(HIST),
         "checkpoint_sha256": stored["checkpoint_sha256"]}  # fmt: skip
    rec = p6._run(j, observe=True, block="C")
    shared = {k for k in stored if k in rec} - p6.ANCHOR_SKIP
    assert shared and all(stored[k] == rec[k] for k in shared)
    p = rec["p6"]
    assert p["k_arb"] == {k: rec["conflicts"][k] for k in p["k_arb"]}
    assert 0 < p["moves_arb_nonidle"] <= p["moves_arb"] <= p["robot_steps"]
    assert p["robot_steps"] == rec["robots"] * rec["steps"]
    assert p["k_ctrl"] is not None and p["k_ctrl_available"]
    # no supervisor change → the arbiter's input equals the controller's moves up to S/R changes
    assert abs(p["moves_arb"] - p["moves_ctrl"]) <= rec["sup_changed"]


def test_analysis_maps_and_outcomes() -> None:
    assert p6a.historical_system("madql_seed2") == "hist:OFF"
    assert p6a.historical_system("mappo_seed1") == "i6:OFF"
    assert p6a.historical_system("ippo_seed3") == "v5:OFF"
    assert p6a.forward_system("madql_seed1") == "v2:RSD"
    assert p6a.forward_system("ippo_seed2") == "v5:RSD"
    r = {"p6": {"k_arb": {"RRv": 2, "RRs": 1, "ROv": 3, "ROs": 1, "INV": 0},
                "k_ctrl": {"RRv": 4, "RRs": 0, "ROv": 3, "ROs": 0, "INV": 0}, "moves_arb": 400},
         "robots": 8, "steps": 100}  # fmt: skip
    assert p6a.ro_per_1000_moves(r) == (4000.0, 400.0)
    assert p6a.rr_ctrl_per_100(r) == (400.0, 800.0)
    assert p6a.rr_arb_per_100(r) == (300.0, 800.0)
    assert p6a.ALPHA_L == {"P1": 0.03, "P2": 0.005, "P3": 0.01, "P4": 0.005}
    assert p6a.ALPHA_V2 == {"P1": 0.03, "P2": 0.005, "E": 0.01, "P4": 0.005}
    assert sum(p6a.ALPHA_L.values()) <= 0.05 + 1e-12 and sum(p6a.ALPHA_V2.values()) <= 0.05 + 1e-12


def test_ceiling_rule_is_used_for_the_no_harm_bound() -> None:
    old_d, old_r = np.array([16.0, 16.0]), np.array([16.0, 16.0])
    new_d = np.array([15.0, 16.0])
    row = {"inf": {"ci95": [-np.inf, np.inf], "ci95_percentile": [-0.01, 0.0]}}
    lb = p6a.lower_bound(row, [old_d, old_r, new_d, old_r])
    assert lb["value"] == min(-0.01, -1 / 32)
    assert p6a.lower_bound({"inf": None}, [])["value"] == -np.inf


def _fam_row(unit: str, hyp: str, est: float, rejected: bool) -> dict[str, Any]:
    return {"unit": unit, "hypothesis": hyp, "inf": {"estimate": est}, "rejected": rejected}


def test_system_decision_rules() -> None:
    fam = {
        "P1": [
            _fam_row("madql", "completion", 0.3, True),
            _fam_row("ippo", "completion", 0.01, True),
            _fam_row("mappo", "completion", 0.0, False),
        ],
        "P2": [],
        "P3": [],
        "P4": [_fam_row("ippo", "ro_per_1000_moves", 0.4, False)],
    }
    res = {
        "families": fam,
        "delta1": {"madql": 0.06, "ippo": 0.06, "mappo": 0.06},
        "completion_lower_bounds": {"ippo": {"value": -0.001}, "mappo": {"value": -0.002}},
    }
    assert p6a.decide_system(res, {"pass": True}, "astar")["verdict"] == "go"
    fam["P4"][0]["rejected"] = True  # a Holm-rejected increase of RO per move is a veto
    d = p6a.decide_system(res, {"pass": True}, "astar")
    assert d["verdict"] == "inconclusive" and d["vetoes"] == ["ippo|ro_per_1000_moves"]
    assert p6a.decide_system(res, {"pass": True}, "rrtstar")["verdict"] == "no-go"
    fam["P4"][0]["rejected"] = False
    assert p6a.decide_system(res, {"pass": False}, "astar")["verdict"] == "no-go"


def test_backup_manifest_verify_detects_changes(tmp_path: Path) -> None:
    import backup_manifest as bm

    f = tmp_path / "results" / "x" / "best.pt"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"abc")
    man = {"files": {"results/x/best.pt": {"sha256": bm.sha256(f), "bytes": 3}}}
    assert bm.verify(man, tmp_path)["ok"]
    f.write_bytes(b"abd")
    assert bm.verify(man, tmp_path)["changed"] == ["results/x/best.pt"]
    f.unlink()
    assert bm.verify(man, tmp_path)["missing"] == ["results/x/best.pt"]


def test_dev_ro_move_denominator() -> None:
    import dev_ro_analysis as dra

    r = {"planning": {"active_robot_steps": 1000}, "wait_ratio": 0.25,
         "conflicts": {"ROv": 3, "ROs": 1}}  # fmt: skip
    assert dra.moves(r) == 750.0
    assert dra.per_move(r) == (4000.0, 750.0)


DEV_EXACT = ROOT / "results" / "dev6a" / "phase6" / "dev_ro_exact" / "records.jsonl"


@pytest.mark.skipif(not DEV_EXACT.exists(), reason="development records absent")
def test_analysis_and_report_end_to_end_on_relabelled_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Software test only: stored Block C development records are relabelled as a tiny Block V
    design (6 A* and 3 RRT* seeds) to exercise every analysis and report path. No scenario is
    generated, and the numbers mean nothing."""
    import copy

    import phase6_report as p6r

    monkeypatch.setattr(p6, "BLOCK_V", range(14_000, 14_006))
    monkeypatch.setattr(p6, "BLOCK_V_RRT", range(14_000, 14_003))
    dev: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    with DEV_EXACT.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if int(r["seed"]) < 11_006:
                dev[(r["arm"], r["run"], r["planner"], r["level"], int(r["seed"]))] = r
    fallback = {"rrtstar": {"hist:OFF": "hist:RSD", "v5:OFF": "hist:RSD", "v5:RSD": "hist:RSD",
                            "i6:OFF": "hist:RSD", "base:ORIG": "hist:RSD"},
                "astar": {"i6:OFF": "v5:OFF", "base:ORIG": "v5:OFF"}}  # fmt: skip
    out = tmp_path / "eval"
    out.mkdir()
    n = 0
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for j in p6.jobs({"policies": {}}):
            src_arm = fallback[j["planner"]].get(j["arm"], j["arm"])
            valid = p6.MADQL_RUNS if src_arm.startswith(("hist", "v2")) else p6.PPO_RUNS
            src_run = j["run"] if j["run"] in valid else valid[0]
            c_seed = 11_000 + (j["seed"] - 14_000)
            r = copy.deepcopy(dev[(src_arm, src_run, j["planner"], j["level"], c_seed)])
            r.update({k: j[k] for k in ("arm", "policy", "cond", "run", "planner", "level")})
            r.update(seed=j["seed"], block="V", instance_sha256=f"{j['level']}-{j['seed']}")
            if j["cond"] == "OFF":
                r.update(deadlock=False)
                r.pop("supervisor", None)
            if j["cond"] == "ORIG":
                r.update(deadlock=True)
                r.pop("supervisor", None)
            if j["policy"] == "v2":
                r["label"] = "madql-v2-n3-" + j["run"][-5:]
            fh.write(json.dumps(r) + "\n")
            n += 1
    (out / "replay.json").write_text(json.dumps({"identical": True, "replayed": 1}), "utf-8")
    fps = {lv: {"fingerprints": {str(s): f"{lv}-{s}" for s in p6.BLOCK_V}} for lv in p6.LEVELS}
    (out / "instance_check.json").write_text(json.dumps({"ok": True, "levels": fps}), "utf-8")
    assert n == p6.episode_counts()["total"]
    res = p6a.analyse(out)
    assert res["completeness"]["complete"]
    assert "P3" in res["H-L2"]["astar"]["families"]
    assert "P3" not in res["H-L2"]["rrtstar"]["families"]
    assert all("P3" not in res["H-L12"][p]["families"] for p in p6.PLANNERS)
    assert len(res["H-L12"]["astar"]["families"]["P4"]) == 9
    assert [r["hypothesis"] for r in res["H-v2"]["FA"]["families"]["P4"]][-1] == "ro_per_1000_moves"
    assert set(res["context_v2_vs_forward_ppo"]["astar"]) == set(p6a.RO_OUTCOMES)
    assert set(res["summary"]) >= {"supervisor_confirmed", "madql_v2_confirmed",
                                   "integrated_system_confirmed", "error_control"}  # fmt: skip
    assert res["summary"]["error_control"]["global_bound_any_false_rejection"] == pytest.approx(
        1 - 0.95**7
    )
    for planner in p6.PLANNERS:
        assert set(res["arms"][planner]) == {f"{p}:{c}" for p, c in p6.MATRIX[planner]}
    text = p6r.render(json.loads(json.dumps(res, default=float)))
    assert "H-L12" in text and "ro_per_1000_moves" in text and "Context" in text


def test_backup_record_requires_a_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import backup_manifest as bm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bm, "PROTECTED", ("results/x",))
    src = tmp_path / "results" / "x"
    src.mkdir(parents=True)
    (src / "best.pt").write_bytes(b"abc")
    man = bm.manifest()
    man["madql_v2_best"] = {}
    out = Path("results/dev6a/phase6/backup")
    out.mkdir(parents=True)
    (out / "manifest_sha256.json").write_text(json.dumps(man), "utf-8")
    dest = tmp_path / "copy"
    (dest / "results" / "x").mkdir(parents=True)
    (dest / "results" / "x" / "best.pt").write_bytes(b"abc")
    with pytest.raises(SystemExit, match="manifest itself"):
        bm.main(["record", "--root", str(dest), "--label", "a"])
    (dest / out).mkdir(parents=True)
    (dest / out / "manifest_sha256.json").write_bytes((out / "manifest_sha256.json").read_bytes())
    assert bm.main(["record", "--root", str(dest), "--label", "a"]) == 0
    rec = json.loads((out / "backup_record.json").read_text("utf-8"))
    assert rec["destinations"][0]["verified"]["ok"] and rec["destinations"][0]["label"] == "a"
    with pytest.raises(SystemExit, match="already recorded"):
        bm.main(["record", "--root", str(dest), "--label", "a"])
    (dest / "results" / "x" / "best.pt").write_bytes(b"abd")
    assert bm.main(["verify", "--root", str(dest)]) == 1


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any = None) -> Path:
    """Relabelled development records as a tiny Block V design (software fixture only)."""
    import copy

    monkeypatch.setattr(p6, "BLOCK_V", range(14_000, 14_006))
    monkeypatch.setattr(p6, "BLOCK_V_RRT", range(14_000, 14_003))
    dev: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    with DEV_EXACT.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if int(r["seed"]) < 11_006:
                dev[(r["arm"], r["run"], r["planner"], r["level"], int(r["seed"]))] = r
    fallback = {"rrtstar": {"hist:OFF": "hist:RSD", "v5:OFF": "hist:RSD", "v5:RSD": "hist:RSD",
                            "i6:OFF": "hist:RSD", "base:ORIG": "hist:RSD"},
                "astar": {"i6:OFF": "v5:OFF", "base:ORIG": "v5:OFF"}}  # fmt: skip
    out = tmp_path / "eval"
    out.mkdir()
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for j in p6.jobs({"policies": {}}):
            src_arm = fallback[j["planner"]].get(j["arm"], j["arm"])
            valid = p6.MADQL_RUNS if src_arm.startswith(("hist", "v2")) else p6.PPO_RUNS
            src_run = j["run"] if j["run"] in valid else valid[0]
            r = copy.deepcopy(dev[(src_arm, src_run, j["planner"], j["level"],
                                   11_000 + (j["seed"] - 14_000))])  # fmt: skip
            r.update({k: j[k] for k in ("arm", "policy", "cond", "run", "planner", "level")})
            r.update(seed=j["seed"], block="V", instance_sha256=f"{j['level']}-{j['seed']}")
            if j["cond"] != "RSD":
                r.update(deadlock=j["cond"] == "ORIG")
                r.pop("supervisor", None)
            if j["policy"] == "v2":
                r["label"] = "madql-v2-n3-" + j["run"][-5:]
            if mutate is not None:
                r = mutate(j, r)
            fh.write(json.dumps(r) + "\n")
    (out / "replay.json").write_text(json.dumps({"identical": True, "replayed": 1}), "utf-8")
    fps = {lv: {"fingerprints": {str(s): f"{lv}-{s}" for s in p6.BLOCK_V}} for lv in p6.LEVELS}
    (out / "instance_check.json").write_text(json.dumps({"ok": True, "levels": fps}), "utf-8")
    return out


@pytest.mark.skipif(not DEV_EXACT.exists(), reason="development records absent")
def test_failed_and_invalid_episodes_are_not_evaluable_and_block_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protocol §5 / Block D §7: an exception or an invalid record is never dropped or imputed.
    G0 fails, every row that involves it is 'not evaluable', and nothing can be confirmed."""
    err_key = ("v2:RSD", "madql_seed1", "astar", "full", 14_001)
    bad_key = ("v5:OFF", "ippo_seed2", "rrtstar", "static", 14_002)

    def mutate(j: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
        if p6.key(j) == err_key:
            return {**{k: j[k] for k in ("stage", "arm", "policy", "cond", "run", "planner",
                                         "level", "seed")},
                    "status": "error", "error_type": "InvalidTransitionError"}  # fmt: skip
        if p6.key(j) == bad_key:
            r["delivered"] = r["released"] + 1  # invalid: delivered > released
        return r

    res = p6a.analyse(_fixture(tmp_path, monkeypatch, mutate))
    assert res["completeness"]["errors"] == 1 and res["completeness"]["invalid"] == 1
    assert res["g0"]["astar"]["errors"] == [list(err_key)] and not res["g0"]["astar"]["pass"]
    assert res["g0"]["rrtstar"]["invalid_records"][0]["key"] == list(bad_key)
    assert not res["g0"]["rrtstar"]["pass"]
    fa = res["H-v2"]["FA"]
    assert fa["families"]["P1"][0]["inf"] is None and not fa["families"]["P1"][0]["rejected"]
    assert fa["decision"]["verdict"] == "reject"  # G0 failure
    l2r = {r["unit"]: r for r in res["H-L2"]["rrtstar"]["families"]["P2"]}
    assert l2r["ippo_seed2"]["inf"] is None and l2r["ippo_seed1"]["inf"] is not None
    assert res["H-L2"]["astar"]["decision"]["verdict"] == "no-go"
    s = res["summary"]
    assert not (s["supervisor_confirmed"] or s["madql_v2_confirmed"]
                or s["integrated_system_confirmed"])  # fmt: skip
    assert res["arms"]["astar"]["v2:RSD"]["unusable_episodes"] == 1


@pytest.mark.skipif(not DEV_EXACT.exists(), reason="development records absent")
def test_missing_replay_blocks_every_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _fixture(tmp_path, monkeypatch)
    (out / "replay.json").unlink()
    s = p6a.analyse(out)["summary"]
    assert s["replay_identical"] is False
    assert not (s["supervisor_confirmed"] or s["madql_v2_confirmed"]
                or s["integrated_system_confirmed"])  # fmt: skip


def test_analysis_refuses_without_fixed_hash_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(SystemExit, match="PYTHONHASHSEED=0"):
        p6a.main(["--out", "nonexistent"])


@pytest.mark.skipif(not DEV_EXACT.exists(), reason="development records absent")
def test_an_arm_with_no_usable_episode_is_reported_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phase6_report as p6r

    def mutate(j: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
        if j["arm"] == "base:ORIG" and j["planner"] == "rrtstar":
            return {**{k: j[k] for k in ("stage", "arm", "policy", "cond", "run", "planner",
                                         "level", "seed")}, "status": "error"}  # fmt: skip
        return r

    res = p6a.analyse(_fixture(tmp_path, monkeypatch, mutate))
    cell = res["arms"]["rrtstar"]["base:ORIG"]
    assert cell["data_incomplete"] and cell["unusable_episodes"] == cell["episodes"] == 9
    assert not res["g0"]["rrtstar"]["pass"] and not res["summary"]["supervisor_confirmed"]
    assert "data incomplete" in p6r.render(json.loads(json.dumps(res, default=float)))


@pytest.mark.skipif(not DEV_EXACT.exists(), reason="development records absent")
def test_not_evaluable_rows_never_carry_a_p_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protocol §5 / Block D §7: p = 1 is only the Holm input of a not-evaluable row; no
    estimate, interval or (adjusted) p-value is stored or printed for it."""
    import phase6_report as p6r

    err = ("hist:RSD", "madql_seed2", "astar", "obstacles", 14_003)

    def mutate(j: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
        if p6.key(j) == err:
            return {**{k: j[k] for k in ("stage", "arm", "policy", "cond", "run", "planner",
                                         "level", "seed")}, "status": "error"}  # fmt: skip
        return r

    res = p6a.analyse(_fixture(tmp_path, monkeypatch, mutate))
    rows = [r for fam in res["H-L2"]["astar"]["families"].values() for r in fam]
    ne = [r for r in rows if not r["evaluable"]]
    ok = [r for r in rows if r["evaluable"]]
    assert ne and ok
    for r in ne:
        assert r["inf"] is None and r["p_holm"] is None and r["holm_input_p"] == 1.0
        assert not r["rejected"] and not r["mc_sensitive"] and "not evaluable" in r["note"]
    assert all(isinstance(r["p_holm"], float) and "holm_input_p" not in r for r in ok)
    text = p6r.render(json.loads(json.dumps(res, default=float)))
    line = next(x for x in text.splitlines() if "madql_seed2 | completion | **not evaluable**" in x)
    assert "Holm input p = 1 by rule" in line and "| 1 |" not in line and "| 1.0" not in line


# ------------------------------------------------------------ amendment A1 (Block V timing)


@pytest.fixture
def sample_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Records every scenario generation (any seed) in this process."""
    from robocoord.marl.scenarios import ScenarioDist

    calls: list[int] = []
    original = ScenarioDist.sample

    def counting(self: Any, seed: int) -> Any:
        calls.append(int(seed))
        if int(seed) in p6.BLOCK_V:
            raise BlockVGenerationError(f"Block V seed {seed} generated")
        return original(self, seed)

    monkeypatch.setattr(ScenarioDist, "sample", counting)
    return calls


def test_precheck_never_generates_or_fingerprints_block_v(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_calls: list[int]
) -> None:
    """A1: the pre-run check validates every non-Block-V requirement and generates no scenario."""
    monkeypatch.setattr(
        p6, "verify_freeze", lambda out: {"design": {"episodes": {"total": 22_080}}}
    )
    monkeypatch.setattr(p6, "scan_results", lambda out: {"clean": True, "files_scanned": 0})
    monkeypatch.setattr(p6, "checkpoints", lambda: {"ok": True, "policies": {}})
    monkeypatch.setattr(p6, "backup_status", lambda: {"ok": True})
    (tmp_path / "plan_freeze.json").write_text("{}", "utf-8")
    res = p6.check(tmp_path, authorized=True)
    assert res["ok"] and res["episodes"] == 22_080
    assert sample_calls == []  # no scenario of any block was generated
    text = (tmp_path / "check.json").read_text("utf-8")
    assert "fingerprints" not in text and "instances" not in json.loads(text)
    assert "deferred to run" in json.loads(text)["block_v_instance_check"]
    assert not (tmp_path / "instance_check.json").exists()


def test_block_v_generation_is_impossible_before_execution_authorization(
    tmp_path: Path, sample_calls: list[int]
) -> None:
    with pytest.raises(SystemExit, match="authorized-execution"):
        p6.instance_check()
    with pytest.raises(SystemExit, match="authorized-execution"):
        p6.block_v_gate(tmp_path, authorized=False)
    with pytest.raises(SystemExit, match="authorized-execution"):
        p6.run(tmp_path, 1, authorized=False)
    with pytest.raises(SystemExit, match="no freeze record"):  # authorized, but nothing frozen
        p6.run(tmp_path, 1, authorized=True)
    assert sample_calls == [] and not (tmp_path / "instance_check.json").exists()
    with pytest.raises(SystemExit, match="authorized-execution"):
        p6.main(["run", "--out", str(tmp_path), "--authorized-precheck"])  # wrong flag


def _gate_stub(out: Path) -> dict[str, Any]:
    return {"ok": True, "checkpoints": {"ok": True, "policies": {}}}


def test_run_aborts_before_any_episode_when_the_block_v_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(p6, "_gate", _gate_stub)
    monkeypatch.setattr(p6, "instance_check", lambda execution_authorized=False: {"ok": False})

    class NoPool:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise AssertionError("an episode pool was started")

    monkeypatch.setattr(p6, "ProcessPoolExecutor", NoPool)
    with pytest.raises(SystemExit, match="aborted before any episode"):
        p6.run(tmp_path, 1, authorized=True)
    assert not (tmp_path / "records.jsonl").exists()
    assert json.loads((tmp_path / "instance_check.json").read_text("utf-8"))["ok"] is False
    assert "ABORT" in (tmp_path / "run.log").read_text("utf-8")
    with pytest.raises(SystemExit, match="aborted before any episode"):  # also on a re-run
        p6.run(tmp_path, 1, authorized=True)


def test_run_performs_the_block_v_check_once_and_before_any_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []

    def ic(execution_authorized: bool = False) -> dict[str, Any]:
        assert execution_authorized
        order.append("instance_check")
        return {"ok": True, "levels": {}}

    class Pool:
        def __init__(self, *a: Any, **k: Any) -> None:
            order.append("episodes")
            raise RuntimeError("stop before running episodes (test)")

    monkeypatch.setattr(p6, "_gate", _gate_stub)
    monkeypatch.setattr(p6, "instance_check", ic)
    monkeypatch.setattr(p6, "ProcessPoolExecutor", Pool)
    for _ in range(2):  # a resumed run re-uses the recorded check and never regenerates
        with pytest.raises(RuntimeError):
            p6.run(tmp_path, 1, authorized=True)
    assert order == ["instance_check", "episodes", "episodes"]


@pytest.mark.skipif(not DEV_EXACT.exists(), reason="development records absent")
def test_records_must_match_the_pre_episode_block_v_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(j: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
        if p6.key(j) == ("v5:RSD", "mappo_seed3", "astar", "static", 14_004):
            r["instance_sha256"] = "different"
        return r

    out = _fixture(tmp_path, monkeypatch, mutate)
    res = p6a.analyse(out)
    assert not res["g0"]["astar"]["instances_match_pre_episode_check"]
    assert not res["g0"]["astar"]["pass"] and not res["summary"]["integrated_system_confirmed"]
    assert res["g0"]["rrtstar"]["instances_match_pre_episode_check"]
    (out / "instance_check.json").unlink()
    assert not p6a.analyse(out)["g0"]["rrtstar"]["pass"]  # no pre-episode check: G0 fails


def test_d12_waiver_lets_the_gate_pass_without_claiming_d12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner-approved waiver (2026-11-23) replaces the two copies for this execution only:
    D12 is reported as *not satisfied*, no backup record is created, and the originals must
    still verify against the manifest."""
    import backup_manifest as bm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bm, "PROTECTED", ("results/x",))
    (tmp_path / "results" / "x").mkdir(parents=True)
    (tmp_path / "results" / "x" / "best.pt").write_bytes(b"abc")
    bdir = Path("results/dev6a/phase6/backup")
    bdir.mkdir(parents=True)
    man = bm.manifest()
    man["madql_v2_best"] = {}
    (bdir / "manifest_sha256.json").write_text(json.dumps(man), "utf-8")
    monkeypatch.setattr(p6, "BACKUP_MANIFEST", bdir / "manifest_sha256.json")
    monkeypatch.setattr(p6, "BACKUP_RECORD", bdir / "backup_record.json")
    monkeypatch.setattr(p6, "BACKUP_WAIVER", bdir / "d12_waiver.json")
    assert not p6.backup_status()["ok"]  # neither copies nor a waiver
    waiver = {"owner_approved": True, "d12_satisfied": False,
              "scope": "Issue 6A Phase 6 execution (this protocol version)"}  # fmt: skip
    (bdir / "d12_waiver.json").write_text(json.dumps({**waiver, "owner_approved": "yes"}), "utf-8")
    assert not p6.backup_status()["ok"]  # malformed waiver
    (bdir / "d12_waiver.json").write_text(json.dumps(waiver), "utf-8")
    st = p6.backup_status()
    assert st["ok"] and st["d12_waived"] and not st["d12_satisfied"] and st["originals_ok"]
    assert st["waiver_sha256"] and not (bdir / "backup_record.json").exists()
    (tmp_path / "results" / "x" / "best.pt").write_bytes(b"abd")  # a checkpoint changed
    st = p6.backup_status()
    assert not st["ok"] and not st["originals_ok"]
