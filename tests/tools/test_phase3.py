"""Tests of the Phase 3 evaluation tools (design, guards, observer neutrality, statistics)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "phase3"))
sys.path.insert(0, str(ROOT / "tools" / "selection"))

import phase3 as p3  # noqa: E402
import phase3_analysis as pa  # noqa: E402


def test_factorial_design_is_complete_and_unique() -> None:
    cells = [c for _, _, c in p3.ARMS.values() if c is not None]
    assert sorted(cells) == sorted((r, s, d) for r in (0, 1) for s in (0, 1) for d in (0, 1))
    from robocoord.control.supervisor import parse_mode

    for mode, deadlock, cell in p3.ARMS.values():
        mech = parse_mode(mode)
        if cell is None:  # "S without O": full minus O, resolver on
            assert (mech, deadlock) == (frozenset({"R", "S"}), True)
            continue
        assert ("R" in mech) == bool(cell[0])
        assert ({"S", "O"} <= mech) == bool(cell[1]) and (bool(mech & {"S", "O"}) == bool(cell[1]))
        assert deadlock == bool(cell[2])
    assert p3.ARMS["off"][:2] == ("off", False) and p3.ARMS["full"][:2] == ("full", True)


def test_stage_sizes_match_the_design() -> None:
    sizes = {s: len(a) * len(r) * 9 * 3 for s, (_, a, r) in p3.STAGES.items()}
    assert sizes == {"3b": 6480, "3c": 22680, "3e": 2160}  # design §10
    assert p3.STAGES["3e"][0] == "rrtstar" and p3.STAGES["3e"][2] == range(11_000, 11_040)


def test_main_effect_pairs_differ_in_one_factor_only() -> None:
    for i in range(3):
        pairs = [(c, tuple(1 if k == i else c[k] for k in range(3))) for c in pa.CELL if c[i] == 0]
        assert len(pairs) == 4
        for lo, hi in pairs:
            assert sum(a != b for a, b in zip(lo, hi, strict=True)) == 1


@pytest.mark.parametrize(
    ("planner", "seed", "ok"),
    [
        ("astar", 11_000, True),
        ("astar", 11_119, True),
        ("rrtstar", 11_039, True),
        ("rrtstar", 11_040, False),
        ("astar", 12_000, False),
        ("astar", 10_050, False),
        ("astar", 13_000, False),
        ("astar", 20_000, False),
    ],
)
def test_guard(planner: str, seed: int, ok: bool) -> None:
    if ok:
        p3.guard(planner, seed, "C")
    else:
        with pytest.raises(ValueError):
            p3.guard(planner, seed, "C")
    with pytest.raises(ValueError):
        p3.guard("astar", 11_000, "B")


def test_common_top_flow_uses_tasks_delivered_in_both_arms() -> None:
    a = {
        "tasks": [
            [0, 3, 0, 5, 10, "delivered"],
            [1, 3, 2, 6, None, "picked"],
            [2, 1, 0, 3, 4, "delivered"],
        ]
    }
    b = {
        "tasks": [
            [0, 3, 0, 4, 8, "delivered"],
            [1, 3, 2, 6, 20, "delivered"],
            [2, 1, 0, 3, 9, "delivered"],
        ]
    }
    assert pa.common_top_flow(a, b) == (10.0, 1.0, 8.0, 1.0)
    assert pa.top_undelivered(a) == (1.0, 1.0) and pa.top_undelivered(b) == (0.0, 1.0)


def test_infer_identical_arms_and_not_evaluable() -> None:
    idx = np.random.default_rng(0).integers(0, 20, size=(200, 20))
    y = np.arange(20, dtype=float)
    x = np.full(20, 30.0)
    same = pa.infer([y, x, y, x], idx)
    assert same is not None and same["estimate"] == 0.0 and same["p"] == 1.0
    assert pa.infer([y, np.zeros(20), y, x], idx) is None
    better = pa.infer([y, x, y + 3, x], idx)
    assert better is not None and better["estimate"] == pytest.approx(0.1) and better["p"] < 0.05
    main = pa.infer([y, x, y + 3, x] * 4, idx, groups=4)
    assert main is not None and main["estimate"] == pytest.approx(0.1)


def test_holm_family_keeps_not_evaluable_rows() -> None:
    rows: list[dict[str, Any]] = [{"inf": {"p": 0.001}}, {"inf": None}, {"inf": {"p": 0.02}}]
    pa.holm_family(rows, 0.03)
    assert rows[0]["rejected"] and not rows[1]["rejected"] and rows[1]["p_holm"] == 1.0
    assert rows[2]["p_holm"] == pytest.approx(0.04) and not rows[2]["rejected"]
    assert math.isclose(rows[0]["own_threshold"], 0.01)


def test_observer_is_neutral_and_classifies() -> None:
    """``run_episode`` with the observer gives the same record as without it (block A)."""
    ckpt = ROOT / "results" / "training" / "madql_seed1" / "best.pt"
    if not ckpt.exists():
        pytest.skip("checkpoints not available")
    import os

    from observer import Phase3Observer

    from robocoord.marl.evaluation import EvalTask, run_episode

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        task = EvalTask(
            "madql",
            "full",
            10_000,
            block="A",
            checkpoint=str(ckpt),
            label="madql-seed1",
            supervisor="full",
            deadlock=True,
        )
        obs = Phase3Observer()
        a = run_episode(task, observer=obs)
        b = run_episode(task)
    finally:
        os.chdir(cwd)
    timing = {"wall_s", "prepare_s", "act_s"}
    assert {k: v for k, v in a.items() if k not in timing} == {
        k: v for k, v in b.items() if k not in timing
    }
    res = obs.result()
    assert res["sup_invalid"] == 0 and len(res["tasks"]) >= a["released"]
    assert all(s["class"] in pa.ALL_CLASSES for s in res["stuck"])
    flows = [t[4] - t[2] for t in res["tasks"] if t[4] is not None]
    assert sum(flows) / len(flows) == pytest.approx(a["mean_flow_time"], abs=1e-5)
