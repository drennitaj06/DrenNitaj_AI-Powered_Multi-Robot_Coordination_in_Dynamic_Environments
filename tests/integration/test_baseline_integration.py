"""Baseline controller in the engine (Issue #4 stage 5A): planner × allocator compatibility,
right-of-way at a junction with 2 and 3 competing robots, waiting without deviation, head-on
encounters, baseline metrics and determinism.

These tests run with the deadlock resolver **disabled** and with endpoint recovery **disabled**
(both are opt-in in the bare ``AllocationCoordinator`` constructor, and these tests pass neither
argument). They therefore record the baseline's own behaviour: the head-on tests show that the
controller-agnostic T4 trigger does **not** resolve them. The resolver's behaviour is tested in
``test_deadlock_integration.py``, and endpoint recovery in ``test_endpoint_integration.py``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.control.base import SuggestionFollower
from robocoord.control.baseline import BaselineController
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.engine import Engine
from robocoord.sim.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = [
    "corridor_queue",
    "demo_delivery",
    "planning_detour",
    "rooms_dynamic",
    "random_warehouse",
]
PLANNERS = [("astar", {}), ("rrtstar", {"max_iterations": 400})]
ALLOCATORS = {"auction": AuctionAllocator, "sequential": SequentialAllocator}

# 9 × 9 cross: a horizontal corridor y = 4 (x = 1..7) and a vertical one x = 4 (y = 1..7)
CROSS = (
    ["#########"]
    + ["#" + ("." * 7 if y == 4 else "###.###") + "#" for y in range(7, 0, -1)]
    + ["#########"]
)


def engine(
    spec: Any,
    controller: Any,
    planner: str = "astar",
    allocator: str = "auction",
    trace: Any = None,
) -> Engine:
    opts = {"max_iterations": 400} if planner == "rrtstar" else {}
    coord = AllocationCoordinator(make_planner(planner, **opts), ALLOCATORS[allocator]())
    eng = Engine(spec, coord, trace=trace)
    controller.reset(len(eng.robots))
    return eng


def run(spec: Any, controller: Any, **kw: Any) -> Engine:
    eng = engine(spec, controller, **kw)
    eng.run(controller.act)
    return eng


# ---------------------------------------------------------------- compatibility and the property


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
@pytest.mark.parametrize("name", SCENARIOS)
def test_no_robot_robot_conflicts_with_any_planner_and_allocator(
    name: str, planner: str, opts: dict[str, Any], allocator: str
) -> None:
    """Guaranteed (`04` §5.3 notes): with every robot on the baseline, no RR (vertex or swap),
    no RO swap and no invalid move is ever proposed; only unpredicted obstacle moves can still
    give RO vertex conflicts. Checked on every demo scenario with both planners and allocators."""
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    ctrl = BaselineController()
    eng = run(spec, ctrl, planner=planner, allocator=allocator)
    c = eng.summary().conflicts
    assert c["RRv"] == c["RRs"] == c["ROs"] == c["INV"] == 0
    stats = ctrl.episode_stats()
    assert stats["robot_steps"] == eng.t * len(eng.robots)


def test_follower_comparison_is_not_vacuous() -> None:
    """On the same scenarios the plain plan follower proposes many RR conflicts."""
    total = 0
    for name in ("demo_delivery", "random_warehouse"):
        spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
        c = run(spec, SuggestionFollower()).summary().conflicts
        total += c["RRv"] + c["RRs"]
    assert total > 0


# ---------------------------------------------------------------- right-of-way at a junction


def junction(levels: list[int], starts: list[tuple[int, int]], drops: list[tuple[int, int]]) -> Any:
    """Robots start on their own pickup next to the centre (4, 4); once they carry their task
    (t = 1), all their paths lead through the centre to the upper arm."""
    return build_scenario(
        {
            "scenario": {"name": "junction", "seed": 1, "max_steps": 40},
            "map": {"rows": CROSS},
            "robots": [{"start": list(s)} for s in starts],
            "tasks": [
                {"pickup": list(s), "dropoff": list(d), "priority": lv, "release": 0}
                for s, d, lv in zip(starts, drops, levels, strict=True)
            ],
        }
    )


@pytest.mark.parametrize(
    ("levels", "order"),
    [([1, 3], [1, 0]), ([3, 1], [0, 1]), ([2, 2], [0, 1])],  # priority, then the lower id
)
def test_two_robots_vertex_conflict_at_junction(levels: list[int], order: list[int]) -> None:
    starts = [(3, 4), (4, 3)]  # left of and below the centre
    drops = {order[0]: (4, 7), order[1]: (4, 6)}  # the first through goes farthest
    spec = junction(levels, starts, [drops[0], drops[1]])
    ctrl = BaselineController()
    eng = engine(spec, ctrl)
    passed: list[int] = []
    decisions: list[dict[int, str]] = []
    while not eng.done:
        view = eng.prepare_step()
        acts = ctrl.act(view)
        decisions.append(dict(ctrl.last))
        eng.apply_actions(acts)
        passed += [r.id for r in eng.robots if r.position == (4, 4) and r.id not in passed]
    m = eng.summary()
    assert passed == order and m.tasks_delivered == 2 and m.conflicts_total == 0
    loser = order[1]
    expect = "yield_priority" if levels[0] != levels[1] else "yield_tie"
    assert decisions[1][loser] == expect  # t = 1: both carry their task and claim the centre
    assert decisions[1][order[0]] == "move"


@pytest.mark.parametrize(
    ("levels", "order"),
    [([1, 2, 3], [2, 1, 0]), ([3, 1, 2], [0, 2, 1]), ([1, 1, 1], [0, 1, 2])],
)
def test_three_robots_compete_and_pass_in_right_of_way_order(
    levels: list[int], order: list[int]
) -> None:
    """Three robots claim the centre at t = 1. They pass one at a time in (π̃, −id) order; the
    others wait (yield, then robot ahead) without leaving their path (no T3) and without any
    conflict. All three deliver."""
    starts = [(3, 4), (5, 4), (4, 3)]
    far = [(4, 7), (4, 6), (4, 5)]
    drops = {robot: far[k] for k, robot in enumerate(order)}
    spec = junction(levels, starts, [drops[0], drops[1], drops[2]])
    ctrl = BaselineController()
    eng = engine(spec, ctrl)
    passed: list[int] = []
    while not eng.done:
        eng.apply_actions(ctrl.act(eng.prepare_step()))
        passed += [r.id for r in eng.robots if r.position == (4, 4) and r.id not in passed]
    m = eng.summary()
    assert passed == order
    assert eng.terminated and m.tasks_delivered == 3 and m.conflicts_total == 0
    stats = ctrl.episode_stats()
    assert stats["yields"] >= 3  # two at t = 1, one more when the centre frees again
    assert 0 < stats["wait_fraction"] < 1
    assert m.plan_adherence is not None and m.plan_adherence < 1.0  # waiting ≠ following
    coord = eng.coordinator
    assert isinstance(coord, AllocationCoordinator) and coord.monitor is not None
    assert all(c.trigger != "T3" for c in coord.monitor.calls)  # waiting is not a deviation


# ---------------------------------------------------------------- head-on (no detour rule yet)


@pytest.mark.parametrize("lanes", [1, 2])
def test_head_on_is_conflict_free_but_not_resolved(lanes: int) -> None:
    """Two robots carry tasks to opposite ends of a corridor (1 or 2 lanes wide).

    Observed and recorded, not desired: the baseline proposes no conflict (the follower proposes
    a swap every step), but the robots never pass. T4 fires for **both** robots at the same step
    and both replan symmetrically, so they meet again, even when a second lane exists. A
    baseline-specific deadlock rule is the subject of stage 5B; T4 is not that rule."""
    rows = ["#" * 12] + ["#" + "." * 10 + "#"] * lanes + ["#" * 12]
    spec = build_scenario(
        {
            "scenario": {"name": "head-on", "seed": 1, "max_steps": 60},
            "map": {"rows": rows},
            "robots": [{"start": [1, 1]}, {"start": [10, 1]}],
            "tasks": [
                {"pickup": [2, 1], "dropoff": [10, 1], "priority": 1, "release": 0},
                {"pickup": [9, 1], "dropoff": [1, 1], "priority": 2, "release": 0},
            ],
        }
    )
    ctrl = BaselineController()
    eng = run(spec, ctrl)
    m = eng.summary()
    assert m.conflicts_total == 0 and m.tasks_delivered == 0 and eng.truncated
    assert ctrl.episode_stats()["decisions"]["robot_ahead"] > 0
    coord = eng.coordinator
    assert isinstance(coord, AllocationCoordinator) and coord.monitor is not None
    t4 = [(c.t, c.robot) for c in coord.monitor.calls if c.trigger == "T4"]
    first = min(t for t, _ in t4)
    assert {r for t, r in t4 if t == first} == {0, 1}  # symmetric T4
    follower = run(spec, SuggestionFollower()).summary()
    assert follower.conflicts["RRs"] > 0 and follower.tasks_delivered == 0


# ---------------------------------------------------------------- determinism


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize("name", ["rooms_dynamic", "random_warehouse"])
def test_deterministic_traces(name: str, allocator: str) -> None:
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    outs, stats = [], []
    for _ in range(2):
        buf = io.StringIO()
        ctrl = BaselineController()
        run(spec, ctrl, allocator=allocator, trace=TraceWriter(buf))
        outs.append(buf.getvalue())
        stats.append(ctrl.episode_stats())
    assert outs[0] == outs[1] and stats[0] == stats[1]
