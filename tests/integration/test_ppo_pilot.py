"""The RRT*-in-training feasibility pilot: one real cell and the D7 projection rule (Issue #6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from robocoord.marl.pilot import THRESHOLD_H, main, project, run_cell


def test_a_pilot_cell_measures_the_real_loop() -> None:
    cell = run_cell("astar", "reference", 0, steps=1000, cap=600)
    assert cell["env_steps"] == 1000 and not cell["timeout"]
    assert len(cell["step_times_ms"]) == 1000 and cell["env_steps_per_s"] > 0
    assert 0 < cell["planner_share"] < 1 and cell["planner_calls"] > 0
    assert cell["stage"] == 1 and cell["stage_share"] == pytest.approx(0.2)
    assert cell["update_s"] > 0 and cell["robot_steps"] >= 2000


def fake(planner: str, source: str, rate: float, timeout: bool = False) -> list[dict[str, Any]]:
    return [{"planner": planner, "source": source, "stage": s + 1, "stage_share": f,
             "env_steps_per_s": rate, "timeout": timeout}
            for s, f in enumerate((0.2, 0.3, 0.5))]  # fmt: skip


@pytest.mark.parametrize(
    ("ppo_rate", "ref_rate", "decision"),
    [(100.0, 100.0, "feasible"), (5.0, 100.0, "borderline"), (5.0, 5.0, "infeasible")],
)
def test_projection_and_decision_rule(ppo_rate: float, ref_rate: float, decision: str) -> None:
    cells = [
        *fake("astar", "ppo", 200.0),
        *fake("astar", "reference", 200.0),
        *fake("rrtstar", "ppo", ppo_rate),
        *fake("rrtstar", "reference", ref_rate),
    ]
    out = project(cells, validation_s=3600.0)
    assert out["decision"] == decision
    assert out["rrtstar/ppo"]["total_h"] == pytest.approx(1e6 / ppo_rate / 3600 + 1.0)
    assert out["threshold_h"] == THRESHOLD_H
    assert out["rrt_over_astar_training_time/ppo"] == pytest.approx(200.0 / ppo_rate)


def test_timeouts_mark_a_lower_bound_and_missing_cells_are_incomplete() -> None:
    out = project([*fake("rrtstar", "ppo", 5.0, timeout=True)], validation_s=0.0)
    assert out["rrtstar/ppo"]["lower_bound"] is True and out["decision"] == "incomplete"


def test_pilot_cli_writes_cells_and_projection(tmp_path: Path) -> None:
    calls = []

    def stub(planner: str, source: str, stage: int, steps: int, cap: float) -> dict[str, Any]:
        calls.append((planner, source, stage))
        return {**fake(planner, source, 50.0)[stage], "step_times_ms": [1.0]}

    assert main(["--out", str(tmp_path), "--validation-s", "0"], run=stub) == 0
    assert len(calls) == 2 * 2 * 3
    assert len((tmp_path / "cells.jsonl").read_text().splitlines()) == 12
    assert json.loads((tmp_path / "projection.json").read_text())["decision"] == "feasible"
    with pytest.raises(FileExistsError):
        main(["--out", str(tmp_path), "--validation-s", "0"], run=stub)
