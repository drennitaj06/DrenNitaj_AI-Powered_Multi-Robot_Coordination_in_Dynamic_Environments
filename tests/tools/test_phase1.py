"""Tests of the Phase 1 evaluation tools and of the repair switch in ``run_episode``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("phase1", "phase3", "selection"):
    sys.path.insert(0, str(ROOT / "tools" / sub))

import phase1 as p1  # noqa: E402
import phase1_analysis as p1a  # noqa: E402


def test_design() -> None:
    assert p1.REPAIR_RADIUS == 3
    from robocoord.planning.monitor import MonitorParams

    assert MonitorParams().rejoin_window == p1.REPAIR_RADIUS  # = H_r, not tuned
    assert set(p1.ARMS) == {"rsd_r0", "rsd_r3", "off_r0", "off_r3"}
    assert p1.SUPERVISION["rsd"] == ("R+S", True)  # the approved R + S + D (Phase 3 §11)
    sizes = {s: len(p1.ARMS) * len(r) * 27 for s, (_, r) in p1.STAGES.items()}
    assert sizes == {"1a": 12_960, "1b": 4_320}
    assert set(p1a.CONDITIONS.values()) == {
        (p, s) for p in ("astar", "rrtstar") for s in ("rsd", "off")
    }


def test_make_task_maps_arms() -> None:
    j = {
        "arm": "rsd_r3",
        "run": "mappo_seed1",
        "planner": "astar",
        "level": "full",
        "seed": 11_000,
        "checkpoint": "x.pt",
    }
    t = p1.make_task(j)
    assert (t.supervisor, t.deadlock, t.repair, t.block, t.controller) == (
        "R+S",
        True,
        3,
        "C",
        "ppo",
    )
    t0 = p1.make_task({**j, "arm": "off_r0", "run": "madql_seed2"})
    assert (t0.supervisor, t0.deadlock, t0.repair, t0.controller) == ("off", False, 0, "madql")


def test_outcome_helpers() -> None:
    r = {
        "planner_calls": 30,
        "robots": 8,
        "steps": 100,
        "conflicts_total": 4,
        "planning": {"expansions": 800, "iterations": 5000},
    }
    assert p1a.calls_per_100(r) == (3000.0, 800.0)
    assert p1a.work("astar")(r) == (800.0, 800.0)
    assert p1a.work("rrtstar")(r) == (5000.0, 800.0)
    assert p1a.conflicts(r) == (4.0, 1.0)


def test_repair_switch_is_additive_and_off_by_default() -> None:
    ckpt = ROOT / "results" / "training" / "madql_seed1" / "best.pt"
    if not ckpt.exists():
        pytest.skip("checkpoints not available")
    from dataclasses import replace

    from robocoord.marl.evaluation import EvalTask, run_episode

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        base = EvalTask("madql", "obstacles", 10_000, block="A", checkpoint=str(ckpt), label="m")
        off = run_episode(base)
        on = run_episode(replace(base, repair=3))
    finally:
        os.chdir(cwd)
    assert "repair_radius" not in off and "repairs" not in off
    assert list(on)[: len(off)] == list(off)  # historical fields first, then the additions
    assert list(on)[len(off) :] == [
        "repair_radius",
        "repairs",
        "repair_failures",
        "calls_by_trigger",
    ]
    assert on["repair_radius"] == 3 and on["repairs"] >= 0 and on["physical_collisions"] == 0
