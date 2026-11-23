"""Tests of the Issue 6A Phase 4 (MADQL-v2) tools: design, guards, the observer, the selection
rule, the ceiling rule and the diagnostics' spell logic."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("phase4", "phase2", "phase1", "phase3", "selection"):
    sys.path.insert(0, str(ROOT / "tools" / sub))

import diagnose  # noqa: E402
import phase4 as p4  # noqa: E402
import phase4_analysis as p4a  # noqa: E402
import select_candidate as sc  # noqa: E402
import train as p4t  # noqa: E402

HIST = ROOT / "results" / "training" / "madql_seed1" / "best.pt"


def test_training_design_is_the_pre_declared_factorial() -> None:
    assert p4t.CANDIDATES == {
        "n3": {"n_step": 3, "target_period": 2000},
        "t8": {"n_step": 1, "target_period": 8000},
        "n3t8": {"n_step": 3, "target_period": 8000},
    }
    assert p4t.SEEDS == (1, 2, 3)
    cfg = p4t.config("n3t8", 2)
    base = p4t.config("t8", 2)
    assert (cfg.run_seed, cfg.total_steps, cfg.n_envs, cfg.sampler_seed) == (2, 1_000_000, 4, 0)
    assert (cfg.madql.n_step, cfg.madql.target_period) == (3, 8000)
    # everything else equals the historical configuration
    from dataclasses import replace

    from robocoord.marl.trainer import load_training_config

    hist = load_training_config(ROOT / "configs" / "training" / "madql.toml", run_seed=2)
    assert replace(cfg.madql, n_step=1, target_period=2000) == hist.madql
    assert cfg.env == hist.env and base.env == hist.env
    assert replace(cfg.validation, workers=3) == hist.validation


def test_evaluation_design_and_job_counts() -> None:
    assert p4.STAGES["4s"]["arms"] == ("hist", "n3", "t8", "n3t8")
    assert p4.STAGES["4c"]["conds"] == ("FA", "NA", "FR")
    assert p4.CONDS == {"FA": ("astar", "R+S", True), "NA": ("astar", "off", False),
                        "FR": ("rrtstar", "R+S", True)}  # fmt: skip
    runs = {r: {"path": "x", "sha256": "y"} for r in p4.RUNS}
    ck = {
        "arms": {
            a: {"candidate": None if a == "hist" else a, "runs": runs}
            for a in ("hist", "n3", "t8", "n3t8", "v2")
        }
    }  # fmt: skip
    assert len(p4.jobs("4s", ck)) == 4 * 2 * 3 * 3 * 80
    js = p4.jobs("4c", ck)
    assert len(js) == 2 * 3 * 3 * (120 + 120 + 40)
    assert {j["seed"] for j in js if j["cond"] == "FR"} == set(range(11_000, 11_040))
    assert {j["seed"] for j in p4.jobs("4s", ck)} == set(range(13_000, 13_080))
    t = p4.make_task({**js[0], "cond": "NA", "planner": "astar", "policy_arm": "hist"})
    assert (t.supervisor, t.deadlock, t.repair, t.rrt_patience, t.rrt_iterations) == (
        "off",
        False,
        0,
        None,
        1000,
    )


@pytest.mark.parametrize(
    ("stage", "cond", "seed"),
    [("4s", "FA", 11_000), ("4c", "FA", 13_000), ("4c", "FR", 11_050), ("4s", "NA", 12_000),
     ("4c", "FA", 20_000), ("4s", "FA", 10_000), ("4c", "NA", 5)],
)  # fmt: skip
def test_guard_rejects_seeds_outside_the_design(stage: str, cond: str, seed: int) -> None:
    with pytest.raises(ValueError):
        p4.guard(stage, cond, seed)


def test_guard_accepts_the_design() -> None:
    p4.guard("4s", "NA", 13_079)
    p4.guard("4c", "FR", 11_039)
    p4.guard("4c", "FA", 11_119)


@pytest.mark.skipif(not HIST.exists(), reason="historical checkpoint not present")
def test_observer_counts_are_consistent_and_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    from robocoord.marl import evaluation
    from robocoord.marl.policy import LearnedController

    # register the originals so that the worker hooks are undone after the test
    monkeypatch.setattr(evaluation, "check_seeds", getattr(evaluation, "check_seeds"))  # noqa: B009
    monkeypatch.setattr(LearnedController, "choose", LearnedController.choose)
    p4._init_worker("4s", True)
    j = {
        "stage": "4s", "arm": "hist:FA", "policy_arm": "hist", "candidate": None, "cond": "FA",
        "run": "madql_seed1", "planner": "astar", "level": "static", "seed": 13_001,
        "checkpoint": str(HIST), "checkpoint_sha256": "x",
    }  # fmt: skip
    a = p4._run(j)
    b = p4._run(j, observe=False)
    assert a["status"] == b["status"] == "ok"
    skip = set(p4.TIMING) | set(p4.OBS_FIELDS)
    assert {k: v for k, v in a.items() if k not in skip} == {
        k: v for k, v in b.items() if k not in skip
    }
    m = a["madql"]
    assert sum(m["actions"]) == m["nonidle_steps"] == m["gap_n"]
    assert m["actions"][0] == m["stay"] >= m["stay_free"] + m["stay_occupied"]
    assert sum(m["gap_hist_all"]) == m["gap_n"]
    assert sum(m["gap_hist_stay"]) == m["stay"]
    assert sum(m["gap_hist_stay"][: p4.GAP_EDGES.index(0.0) + 1]) == 0  # STAY ⇔ gap ≥ 0
    assert m["self_loop_stay"] <= m["stay"] and m["spell_ge20_steps"] <= m["stay"]
    assert m["inference_calls"] == a["steps"] and m["inference_s"] > 0


def test_spells_and_resolution() -> None:
    #        robot 0: S S S M S S (t 0..5); robot 1: S M with a gap at t=1 (idle)
    ep = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    rob = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    t = np.array([0, 1, 2, 3, 4, 5, 0, 2])
    stay = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    length, resolved = diagnose.stay_spells(ep, rob, t, stay)
    assert length.tolist() == [3, 3, 3, 0, 2, 2, 1, 0]
    assert resolved.tolist() == [True, True, True, False, False, False, False, False]


def _tab(f: dict[str, float], n: dict[str, float], flow: dict[str, float]) -> dict[str, Any]:
    return {
        a: {"FA": {"completion": f[a], "flow_time_delivered": flow[a]}, "NA": {"completion": n[a]}}
        for a in f
    }


def test_selection_rule() -> None:
    f = {"hist": 0.997, "n3": 0.996, "t8": 0.990, "n3t8": 0.998}
    n = {"hist": 0.62, "n3": 0.70, "t8": 0.80, "n3t8": 0.698}
    flow = {"hist": 50.0, "n3": 48.0, "t8": 40.0, "n3t8": 45.0}
    s = sc.select(_tab(f, n, flow))
    assert s["eligible"] == ["n3", "n3t8"]  # t8 loses > 0.005 on F
    assert s["selected"] == "n3t8" and s["selected_eligible"]  # tie within 0.005 → lower flow
    n2 = {"hist": 0.62, "n3": 0.60, "t8": 0.55, "n3t8": 0.61}
    s2 = sc.select(_tab(f, n2, flow))
    assert s2["eligible"] == [] and s2["selected"] == "n3t8" and not s2["selected_eligible"]


def test_ceiling_rule_ignores_gains() -> None:
    old_d = np.array([20.0, 20.0, 19.0])
    old_r = np.array([20.0, 20.0, 20.0])
    new_d = np.array([19.0, 20.0, 20.0])  # one loss, one gain
    b = p4a.ceiling_bound([old_d, old_r, new_d, old_r])
    assert math.isclose(b, -1 / 60)
    row = {"inf": {"ci95": [-math.inf, math.inf], "ci95_percentile": [-0.001, 0.0]}}
    lb = p4a.lower_bound(row, [old_d, old_r, new_d, old_r])
    assert math.isclose(lb["value"], -1 / 60) and lb["source"].startswith("ceiling")
    finite = {"inf": {"ci95": [-0.01, 0.02]}}
    assert p4a.lower_bound(finite, [old_d, old_r, new_d, old_r]) == {
        "value": -0.01,
        "source": "bootstrap-t",
    }


def test_analysis_families_and_alpha() -> None:
    assert p4a.ALPHA == {"P1": 0.03, "P2": 0.005, "E": 0.01, "P4": 0.005}
    assert [n for n, _ in p4a.e_family("FA")] == [
        "planner_calls_per_100_robot_steps",
        "common_flow_time",
        "supervisor_changes_per_100_robot_steps",
    ]
    assert p4a.e_family("NA")[2][0] == "stuck_robots_per_episode"
    a = {
        "tasks": [
            [1, 1, 0, 5, 10, "delivered"],
            [2, 3, 0, 6, 20, "delivered"],
            [3, 1, 0, None, None, "assigned"],
        ]
    }
    b = {
        "tasks": [
            [1, 1, 0, 4, 8, "delivered"],
            [2, 3, 0, None, None, "assigned"],
            [3, 1, 0, 9, 30, "delivered"],
        ]
    }
    assert p4a.common_flow(a, b) == (10.0, 1.0, 8.0, 1.0)  # only task 1 is common


def test_posthoc_conflict_outcomes() -> None:
    import posthoc_conflicts as ph

    r = {"conflicts": {"ROv": 3, "RRv": 1}, "madql": {"nonidle_steps": 50, "stay": 10}}
    assert ph.OUTCOMES["ROv_per_episode"](r) == (3.0, 1.0)
    assert ph.OUTCOMES["ROv_per_1000_moves"](r) == (3000.0, 40.0)
    assert ph.OUTCOMES["RRv_per_1000_moves"](r) == (1000.0, 40.0)
    assert set(ph.OUTCOMES) == {
        "ROv_per_episode", "ROv_per_1000_moves", "RRv_per_episode", "RRv_per_1000_moves"
    }  # fmt: skip
