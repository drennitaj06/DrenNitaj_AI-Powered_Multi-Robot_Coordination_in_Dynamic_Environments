"""Tests of the Phase 2 (P2, RRT* budget control) microbenchmark and Block C tools."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for sub in ("phase2", "phase1", "phase3", "selection"):
    sys.path.insert(0, str(ROOT / "tools" / sub))

import micro  # noqa: E402
import phase2 as p2  # noqa: E402
import phase2_analysis as p2a  # noqa: E402
import phase3 as p3  # noqa: E402


def test_microbenchmark_design() -> None:
    from robocoord.marl.evaluation import RRT_ITERATIONS

    assert micro.BUDGET == RRT_ITERATIONS == p2.BUDGET == 1000
    assert micro.CONFIGS["full"] == (None, None)
    assert [k for k in micro.CONFIGS if k.startswith("k")] == [f"k{k}" for k in micro.REFINE]
    assert [k for k in micro.CONFIGS if k.startswith("m")] == [f"m{m}" for m in micro.PATIENCE]
    assert len(micro.CONFIGS) == 12 and micro.TOLERANCE == 1.01
    assert micro.STRATUM_SIZES == (16, 32)
    jobs = micro._static_jobs(micro.E1A_SPEC)
    assert len(jobs) == 2800 and len({micro._key(j) for j in jobs}) == 2800
    assert len(micro._static_jobs(micro.E1C_SPEC)) == 20


def test_static_job_checks_the_prefix_property() -> None:
    j = {"seed_base": 30000, "family": "d20", "size": 16, "map": 0, "pair": 3}
    r = micro.static_job(j)
    assert set(r["configs"]) == set(micro.CONFIGS)
    full = r["configs"]["full"]
    assert full["iterations"] == 1000
    for lab, row in r["configs"].items():
        if lab == "full":
            continue
        assert row["prefix_ok"] is True
        assert row["status"] == full["status"]
        assert row["iterations"] <= 1000
        if full["path_steps"] is not None:
            assert row["planner_cost"] >= full["planner_cost"]


def _rec(fam: str, size: int, steps: dict[str, int | None], its: dict[str, int]) -> dict[str, Any]:
    return {
        "family": fam,
        "size": size,
        "bfs_steps": 5,
        "configs": {
            lab: {
                "status": "ok" if steps.get(lab, steps["full"]) is not None else "budget_exhausted",
                "path_steps": steps.get(lab, steps["full"]),
                "planner_cost": float(steps.get(lab, steps["full"]) or 0),
                "iterations": its.get(lab, 1000),
                "segment_checks": 1,
                "first_solution_iteration": 10,
                "time_s": 0.01,
                "prefix_ok": True,
            }
            for lab in micro.CONFIGS
        },
    }


def test_summarise_ratio_of_sums_and_sizes() -> None:
    recs = [
        _rec("open", 16, {"full": 10, "k0": 12}, {"k0": 100}),
        _rec("open", 16, {"full": 10, "k0": 10}, {"k0": 50}),
        _rec("open", 64, {"full": 10, "k0": 30}, {"k0": 1}),  # outside the stratum
    ]
    s = micro.summarise(recs, (16, 32))
    assert math.isclose(s["k0"]["open"]["q_steps"], 22 / 20)
    assert s["k0"]["all"]["mean_iterations"] == 75
    assert s["k0"]["open"]["share_steps_identical"] == 0.5
    assert s["full"]["open"]["q_steps"] == 1.0


def test_block_c_design_and_task_mapping() -> None:
    assert p2.ARMS == ("full", "p2")
    assert p2.STAGES["2b"] == ("baseline",) and len(p2.STAGES["2l"]) == 9
    sel = {"label": "m400", "refine": None, "patience": 400}
    ck = {"runs": {r: {"path": f"{r}.pt", "sha256": "x"} for r in p3.RUNS}}
    js = p2.jobs("2l", ck) + p2.jobs("2b", ck)
    assert len(js) == 2 * 9 * 120 + 2 * 120
    assert {j["seed"] for j in js} == set(range(11_000, 11_040))
    learned = next(j for j in js if j["run"] == "mappo_seed2" and j["arm"] == "p2")
    t = p2.make_task(learned, sel)
    assert (t.controller, t.supervisor, t.deadlock, t.repair, t.planner, t.block) == (
        "ppo",
        "R+S",
        True,
        0,
        "rrtstar",
        "C",
    )
    assert (t.rrt_iterations, t.rrt_refine, t.rrt_patience) == (1000, None, 400)
    base = next(j for j in js if j["run"] == "baseline" and j["arm"] == "full")
    tb = p2.make_task(base, sel)
    assert (tb.controller, tb.supervisor, tb.deadlock, tb.rrt_refine, tb.rrt_patience) == (
        "baseline",
        "off",
        None,
        None,
        None,
    )
    assert set(p2a.CONDITIONS) == {"L", "B"}
    assert p2a.ALPHA == {"P1": 0.03, "P2": 0.005, "E": 0.01, "P4": 0.005}


def test_iterations_outcome() -> None:
    r = {"rrt": {"iterations": 5000}, "robots": 8, "steps": 100}
    assert p2a.iterations_per_robot_step(r) == (5000.0, 800.0)
