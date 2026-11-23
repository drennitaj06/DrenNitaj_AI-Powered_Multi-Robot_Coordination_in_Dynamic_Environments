"""Tests of the Phase 0 stored-record diagnostics (synthetic records only; nothing simulated)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "diagnostics"))
import phase0_stored as p0


def sysrec(
    calls: int, und: int, *, robots: int | None = 8, dev: float = 2.0, **kw: Any
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "controller": "ppo",
        "label": "mappo_seed1",
        "planner": "astar",
        "mode": "greedy",
        "shield": "off",
        "allocator": "auction",
        "steps": 400,
        "released": 16,
        "delivered": 16 - und,
        "planner_calls": calls,
        "deviation_rate_per_100": dev,
        "wall_s": 2.0,
    }
    if robots is not None:
        r["robots"] = robots
    r.update(kw)
    return r


def test_dist_and_spearman_edge_cases() -> None:
    assert p0.dist([]) == {"n": 0}
    d = p0.dist([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["q00"] == 1.0 and d["q100"] == 4.0
    assert d["q50"] == 2.5
    assert p0.spearman([1, 2], [1, 2]) is None
    assert p0.spearman([1, 1, 1], [1, 2, 3]) is None
    assert p0.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_system_replanning_rates_bounds_and_default_robots() -> None:
    recs = [sysrec(320, 0), sysrec(640, 2), sysrec(960, 4, robots=None)]
    out = p0.system_replanning({"src": recs})
    (key,) = out
    g = out[key]
    assert key == "src|ppo|mappo_seed1||astar|greedy|off|auction"
    assert g["episodes"] == 3 and g["incomplete"] == 2
    assert g["calls_per_100_robot_steps_pooled"] == pytest.approx(100 * 1920 / (3 * 8 * 400))
    assert g["calls_per_100_robot_steps_successful"]["n"] == 1
    assert g["calls_per_100_robot_steps_incomplete"]["mean"] == pytest.approx(
        (100 * 640 / 3200 + 100 * 960 / 3200) / 2
    )
    assert g["t3_share_upper_bound"] == pytest.approx(
        min(1.0, 2.0 / g["calls_per_100_robot_steps_pooled"])
    )
    assert g["spearman_call_rate_vs_undelivered"] == pytest.approx(1.0)
    assert g["robots_field_present"] is False


def test_system_replanning_skips_failed_episodes_and_groups_arms() -> None:
    recs = [sysrec(100, 0, arm="old"), sysrec(100, 0, arm="new"), sysrec(1, 0, status="error")]
    out = p0.system_replanning({"s": recs})
    assert len(out) == 2 and all(v["episodes"] == 1 for v in out.values())


def test_trigger_breakdown_both_record_layouts() -> None:
    flat = {
        "params": {"planner": "astar", "dynamics": "static", "allocator": "auction"},
        "metrics": {
            "controller": {"controller": "baseline"},
            "calls_by_trigger": {"T1": 6, "T3": 2, "T7": 2},
        },
    }
    nested = {
        "params": {"planner": "rrtstar", "obstacles": 4, "events": "none", "team": 4},
        "metrics": {"planning": {"calls_by_trigger": {"T1": 3, "T3": 1}}},
    }
    out = p0.trigger_breakdown({"a": [flat, flat], "b": [nested], "c": [{"metrics": {}}]})
    ka = "a|baseline|astar|static||auction"
    assert out[ka]["calls"] == 20 and out[ka]["calls_by_trigger"]["T3"] == 4
    assert out[ka]["share_by_trigger"]["T1"] == pytest.approx(0.6)
    kb = "b|plan_follower|rrtstar|obstacles=4,events=none|4|"
    assert out[kb]["share_by_trigger"]["T3"] == pytest.approx(0.25)
    assert len(out) == 2


def test_failure_mode_association() -> None:
    def ep(calls: int, classes: list[str]) -> dict[str, Any]:
        return {
            "algorithm": "madql",
            "replayed": {"planner_calls": calls, "steps": 400},
            "robots": [{"class": c} for c in classes],
        }

    reps = [
        ep(100, ["frozen"]),
        ep(200, ["oscillating"]),
        ep(300, ["oscillating"] * 2),
        ep(400, ["oscillating"] * 3),
    ]
    out = p0.failure_mode_association(reps)["madql"]
    assert out["failed_episodes"] == 4
    assert out["spearman_rate_vs_oscillating_robots"] == pytest.approx(1.0)
    assert out["rate_without_oscillating"]["n"] == 1


def _q(
    planner: str, status: str, fsi: int, its: int, n_max: int, fam: str = "d20"
) -> dict[str, Any]:
    return {
        "params": {"planner": planner, "family": fam, "size": 32},
        "metrics": {
            "status": status,
            "options": {"max_iterations": n_max},
            "counters": {"first_solution_iteration": fsi, "iterations": its},
        },
    }


def test_static_queries_first_solution_and_budget_after() -> None:
    e1a = [
        _q("rrtstar-500", "ok", 50, 500, 500),
        _q("rrtstar-500", "ok", 150, 500, 500),
        _q("rrtstar-500", "budget_exhausted", -1, 500, 500),
        _q("astar", "ok", -1, 0, 0),
    ]
    e1b = [
        {
            "params": {"family": "open", "size": 64, "n_max": 1000, "index": "brute"},
            "metrics": {
                "status": "ok",
                "counters": {"first_solution_iteration": 40, "iterations": 1000},
            },
        }
    ]
    out = p0.static_queries(e1a, e1b)
    g = out["e1a|d20|32|500"]
    assert (g["queries"], g["ok"], g["budget_exhausted"], g["all_ran_full_budget"]) == (
        3,
        2,
        1,
        True,
    )
    assert g["first_solution_iteration"]["mean"] == 100
    assert g["iterations_after_first_solution"]["mean"] == pytest.approx((450 + 350) / 2)
    assert g["share_of_budget_after_first_solution"]["mean"] == pytest.approx(0.8)
    assert g["share_of_all_iterations_spent_without_any_solution"] == pytest.approx(1 / 3)
    assert out["e1b|open|64|1000|brute"]["iterations_after_first_solution"]["mean"] == 960


def test_convergence_ratios_skip_pre_solution_checkpoints() -> None:
    r = {
        "params": {"family": "open", "size": 32},
        "metrics": {
            "counters": {"first_solution_iteration": 30},
            "checkpoints": {"10": "inf", "50": 8.0, "100": 6.0, "1000": 4.0},
        },
    }
    out = p0.convergence([r])["open|32"]
    ratios = out["cost_over_first_found_cost"]
    assert "10" not in ratios
    assert ratios["50"]["mean"] == 1.0 and ratios["1000"]["mean"] == pytest.approx(0.5)
    assert list(ratios) == ["50", "100", "1000"]


def test_replanning_rrt_status_mix() -> None:
    def ep(ok: int, ex: int) -> dict[str, Any]:
        return {
            "params": {"planner": "rrtstar", "team": 4, "obstacles": 8, "events": "low"},
            "metrics": {
                "latency_s": {"median": 0.14},
                "planning": {
                    "calls": ok + ex,
                    "work": {"iterations": 1000 * (ok + ex)},
                    "status_mix": {"ok": ok, "budget_exhausted": ex},
                },
            },
        }

    out = p0.replanning_rrt([ep(9, 1), ep(10, 0), {"params": {"planner": "astar"}}])
    g = out["team=4|obstacles=8|events=low"]
    assert g["calls"] == 20 and g["budget_exhausted_share"] == pytest.approx(0.05)
    assert g["iterations_per_call"] == 1000


def test_learned_rrt_only_rrtstar_ok_records() -> None:
    recs = [
        sysrec(50, 0, planner="rrtstar"),
        sysrec(50, 0),
        sysrec(1, 0, planner="rrtstar", status="error"),
    ]
    out = p0.learned_rrt({"s": recs})
    (g,) = out.values()
    assert g["episodes"] == 1 and g["wall_s_per_call_upper_bound"]["mean"] == pytest.approx(0.04)


def test_missing_inventory_is_explicit() -> None:
    assert {
        "madql_q_values",
        "calls_by_trigger_learned",
        "rrt_per_call_learned",
        "unnecessary_replanning",
    } <= set(p0.MISSING)
    assert all(isinstance(v, str) and len(v) > 40 for v in p0.MISSING.values())
    assert not math.isnan(p0.EVAL_ROBOTS)
