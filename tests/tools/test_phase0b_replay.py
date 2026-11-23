"""Tests of the Phase 0b replay helpers (synthetic inputs; no episode is simulated here)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "diagnostics"))
import phase0b_replay as p0b


def q(*v: float) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def m(*v: bool) -> np.ndarray:
    return np.asarray(v, dtype=bool)


def test_q_gap_sign_and_best_move() -> None:
    gap, best_q, best = p0b.q_gap(q(1.0, 0.5, 0.9, 0.2, 0.9), m(True, True, True, True, True))
    assert gap == pytest.approx(0.1) and best == 2 and best_q == pytest.approx(0.9)
    gap, _, best = p0b.q_gap(q(0.0, 0.5, 2.0, 0.2, 0.9), m(True, True, False, True, True))
    assert best == 4 and gap == pytest.approx(-0.9)


def test_q_gap_undefined_without_allowed_move() -> None:
    gap, best_q, best = p0b.q_gap(q(1.0, 2.0, 2.0, 2.0, 2.0), m(True, False, False, False, False))
    assert math.isnan(gap) and math.isnan(best_q) and best == -1


def test_greedy_choice_matches_production_tie_rule() -> None:
    torch = pytest.importorskip("torch")
    from robocoord.marl.networks import masked_argmax

    rng = np.random.default_rng(0)
    for _ in range(500):
        vals = rng.integers(-2, 3, size=5).astype(np.float32)  # many exact ties
        mask = rng.random(5) < 0.7
        mask[0] = True
        prod = int(masked_argmax(torch.from_numpy(vals[None]), torch.from_numpy(mask[None]))[0])
        assert p0b.greedy_choice(vals, mask) == prod
        gap, _, _ = p0b.q_gap(vals, mask)
        if not math.isnan(gap):
            assert (prod == p0b.STAY) == (gap >= 0)


def test_category_rules() -> None:
    assert p0b.category("frozen", True, False, False) == "frozen"
    assert p0b.category("oscillating", True, True, True) == "oscillating"
    assert p0b.category("conflict_loop", True, True, False) == "other_stuck"
    assert p0b.category("frozen", False, True, True) == "moving"
    assert p0b.category(None, True, True, False) == "waiting"
    assert p0b.category(None, False, False, False) == "waiting"


def test_gap_summary_tie_break_and_shares() -> None:
    gap = np.array([0.0, 0.0, 0.5, -0.2, 1e-5, math.nan])
    chosen = np.array([0, 0, 0, 3, 0, 0])
    s = p0b.gap_summary(gap, chosen)
    assert s["robot_steps"] == 6 and s["no_allowed_move"] == 1
    assert s["share_gap_eq0"] == pytest.approx(2 / 5)
    assert s["share_gap_gt0"] == pytest.approx(2 / 5)
    assert s["stay_chosen"] == 4 and s["stay_chosen_by_tie_break"] == 2
    assert s["share_abs_gap_le"]["0.0001"] == pytest.approx(3 / 5)
    assert s["stay_chosen_with_gap_gt"]["0.01"] == 1


def _ep(label: str, level: str, cbt: dict[str, int], **kw: Any) -> dict[str, Any]:
    e: dict[str, Any] = {
        "label": label,
        "level": level,
        "calls_by_trigger": cbt,
        "robot_steps": 3200,
        "active_robot_steps": 1600,
        "replayed": {"released": 16, "delivered": 15},
        "stuck_classes": {"2": "oscillating"},
        "calls_by_robot_last_window": {"2": {"T3": 40, "T1": 2}},
    }
    e.update(kw)
    return e


def test_trigger_tables_exact_shares_and_denominators() -> None:
    eps = [
        _ep("madql-seed1", "static", {"T1": 10, "T3": 30, "T2": 0}),
        _ep("madql-seed1", "full", {"T1": 20, "T3": 40, "T2": 10}),
    ]
    t = p0b.trigger_tables(eps)
    g = t["groups"]["madql|all|all"]
    assert g["calls"] == 110 and g["counts"]["T3"] == 70
    assert g["shares"]["T3"] == pytest.approx(70 / 110)
    assert g["t3_per_100_nonidle_robot_steps"] == pytest.approx(100 * 70 / 3200)
    assert g["nonidle_share_of_robot_steps"] == pytest.approx(0.5)
    assert t["groups"]["madql|madql-seed1|static"]["calls"] == 40
    cls = t["stuck_robots_last_window"]["madql"]["oscillating"]
    assert cls["stuck_robots"] == 2 and cls["t3_share"] == pytest.approx(80 / 84)


def test_qgap_tables_split_by_category_and_pool() -> None:
    st = {
        "gap": np.array([0.0, 0.3, -0.5]),
        "gap_sug": np.array([0.0, math.nan, -0.5]),
        "chosen": np.array([0, 0, 1]),
        "category": np.array([0, 0, 3], dtype=np.int8),
    }
    eps = [{"label": "madql-seed1", "level": "static", "states": st}]
    out = p0b.qgap_tables(eps)
    fr = out["madql-seed1|static|frozen"]
    assert fr["robot_steps"] == 2 and fr["stay_chosen_by_tie_break"] == 1
    assert out["all|all|moving"]["share_gap_lt0"] == 1.0
    assert out["all|all|frozen"]["share_suggested_move_available"] == 0.5
