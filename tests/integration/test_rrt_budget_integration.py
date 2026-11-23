"""Issue 6A P2 in the production episode runner: the RRT* budget-control switch is additive
(off by default, records unchanged), reaches the planner the path monitor uses, and keeps the
episode safe."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from robocoord.marl.evaluation import EvalTask, run_episode


class _Calls:
    """Read-only observer: keeps the path monitor's planner-call log at the end."""

    def __init__(self) -> None:
        self.engine: Any = None

    def before(self, engine: Any, view: Any, actions: Any) -> None:
        self.engine = engine

    def after(self, engine: Any, outcome: Any) -> None:
        self.engine = engine

    def calls(self) -> list[Any]:
        return list(self.engine.coordinator.monitor.calls)


BASE = EvalTask("baseline", "obstacles", 10_000, planner="rrtstar", block="A", rrt_iterations=200)


def test_switch_is_additive_and_off_by_default() -> None:
    off = run_episode(BASE)
    explicit_off = run_episode(replace(BASE, rrt_refine=None, rrt_patience=None))
    assert "rrt_refine" not in off and "rrt_patience" not in off
    timing = {"wall_s", "prepare_s", "act_s"}
    assert {k: v for k, v in off.items() if k not in timing} == {
        k: v for k, v in explicit_off.items() if k not in timing
    }
    on = run_episode(replace(BASE, rrt_patience=20))
    assert list(on)[: len(off)] == list(off)  # historical fields first, then the additions
    assert list(on)[len(off) :] == ["rrt_refine", "rrt_patience"]
    assert (on["rrt_refine"], on["rrt_patience"]) == (None, 20)
    assert on["physical_collisions"] == 0 and on["conflicts"]["INV"] == 0


def test_rule_reaches_the_monitor_planner_and_stops_calls_early() -> None:
    full, stopped = _Calls(), _Calls()
    run_episode(BASE, observer=full)
    run_episode(replace(BASE, rrt_refine=10), observer=stopped)
    planner = stopped.engine.coordinator.monitor.planner
    assert planner.config.refine_iterations == 10 and planner.config.patience is None
    assert full.engine.coordinator.monitor.planner.config.refine_iterations is None
    full_calls = [c for c in full.calls() if c.stats.iterations > 0]
    stop_calls = [c for c in stopped.calls() if c.stats.iterations > 0]
    assert full_calls and stop_calls
    assert all(c.stats.iterations == 200 for c in full_calls)  # anytime: the whole budget
    for c in stop_calls:
        i1 = c.stats.first_solution_iteration
        if c.status == "ok" and i1 > 0:
            assert c.stats.iterations == min(200, i1 + 10)
        else:
            assert c.stats.iterations == 200  # no solution: the whole budget, as before
    assert any(c.stats.iterations < 200 for c in stop_calls)


def test_switch_requires_rrtstar() -> None:
    with pytest.raises(ValueError, match="rrtstar"):
        run_episode(replace(BASE, planner="astar", rrt_refine=5))
