"""Allocation policy inside the engine (Issue #4 plan step 5; X-5 allocation part; FR-TASK-4/5).

Planner × allocator, assignment changes → T1 replanning, urgent-task pre-emption, hysteresis,
priority ordering, map-event re-allocation, a constructed aging scenario and determinism.

The controller is the existing plan follower (``SuggestionFollower``, Issue #3); the baseline and
the reference follower are not part of this stage. Expected outcomes are derived by hand from the
benefit formula (model §8) in each docstring.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation import BenefitConfig
from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.control.base import SuggestionFollower
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.engine import Engine
from robocoord.sim.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[2]
ALLOCATORS = {"auction": AuctionAllocator, "sequential": SequentialAllocator}
PLANNERS = [("astar", {}), ("rrtstar", {"max_iterations": 400})]
CORRIDOR = ["#" * 32, "#" + "." * 30 + "#", "#" * 32]  # free cells x = 1..30, y = 1


def scenario(
    robots: list[int], tasks: list[dict[str, Any]], max_steps: int = 200, **over: Any
) -> Any:
    data: dict[str, Any] = {
        "scenario": {"name": "alloc", "seed": 3, "max_steps": max_steps},
        "map": {"rows": CORRIDOR},
        "robots": [{"start": [x, 1]} for x in robots],
        "tasks": [
            {
                "pickup": [t["p"], 1],
                "dropoff": [t["d"], 1],
                "priority": t.get("level", 1),
                "release": t.get("release", 0),
            }
            for t in tasks
        ],
    }
    data.update(over)
    return build_scenario(data)


def run(
    spec: Any,
    allocator: str = "auction",
    planner: str = "astar",
    opts: dict[str, Any] | None = None,
    config: BenefitConfig | None = None,
    trace: Any = None,
) -> Engine:
    coord = AllocationCoordinator(
        make_planner(planner, **(opts or {})), ALLOCATORS[allocator](), benefit_config=config
    )
    eng = Engine(spec, coord, trace=trace)
    ctrl = SuggestionFollower()
    ctrl.reset(len(eng.robots))
    eng.run(ctrl.act)
    return eng


def coord_of(eng: Engine) -> AllocationCoordinator:
    assert isinstance(eng.coordinator, AllocationCoordinator)
    return eng.coordinator


def t1_calls(eng: Engine) -> list[tuple[int, int]]:
    mon = coord_of(eng).monitor
    assert mon is not None
    return [(c.t, c.robot) for c in mon.calls if c.trigger == "T1"]


# ---------------------------------------------------------------- planner × allocator


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_planner_x_allocator_delivers(allocator: str, planner: str, opts: dict[str, Any]) -> None:
    """Two robots at the ends, two tasks near them: one auction at t = 0 assigns each robot its
    near task (both allocators find it), both robots plan with T1 at t = 0, all delivered."""
    spec = scenario([2, 28], [{"p": 5, "d": 3}, {"p": 25, "d": 27}])
    eng = run(spec, allocator, planner, opts)
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 2 and m.conflicts_total == 0
    ev = coord_of(eng).policy.events
    assert ev[0].t == 0 and ev[0].triggers == ("released",)
    assert ev[0].assigned == ((0, 0), (1, 1))
    assert (0, 0) in t1_calls(eng) and (0, 1) in t1_calls(eng)
    alloc = m.extra["planning"]["allocation"]
    assert alloc["allocator"] == allocator and alloc["assignments"] == 2
    assert alloc["preemptions"] == 0 and alloc["timeouts"] == 0
    assert len({e.t for e in ev}) == len(ev)  # at most one auction per step


# ---------------------------------------------------------------- pre-emption and replanning


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_urgent_task_preempts_and_triggers_t1(allocator: str) -> None:
    """Robot at x = 10 is assigned L (level 1, pickup 28) at t = 0 and walks right. At t = 3 it is
    at x = 13 and U (level 3, pickup 6, drop-off 4) is released.
    Values: L: 10·1.03 − (15 + 1) + h = −3.7; U: 10·4 − (7 + 2) = 31 → switch (gain ≫ h).
    The assignment change must make the monitor replan robot 0 with T1 at t = 3, towards U."""
    spec = scenario([10], [{"p": 28, "d": 29}, {"p": 6, "d": 4, "level": 3, "release": 3}])
    buf = io.StringIO()
    eng = run(spec, allocator, trace=TraceWriter(buf))
    coord = coord_of(eng)
    ev = {e.t: e for e in coord.policy.events}
    assert ev[3].triggers == ("released",) and ev[3].switched == ((0, 0, 1),)
    assert (3, 0) in t1_calls(eng)
    mon = coord.monitor
    assert mon is not None
    first_t1_at_3 = next(c for c in mon.calls if c.t == 3 and c.robot == 0)
    assert first_t1_at_3.trigger == "T1" and first_t1_at_3.status == "ok"
    tasks = eng.book.tasks
    assert tasks[1].pickup_time is not None and tasks[0].pickup_time is not None
    assert tasks[1].pickup_time < tasks[0].pickup_time  # urgent first, L re-auctioned later
    assert eng.summary().tasks_delivered == 2
    stats = eng.summary().extra["planning"]["allocation"]
    assert stats["preemptions"] == 1 and stats["tasks_reassigned"] == 1
    trace_alloc = [r for r in map(_json, buf.getvalue().splitlines()) if "alloc" in r]
    assert any(r["alloc"]["unassigned"] == [[0, 0]] for r in trace_alloc)


def _json(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)  # type: ignore[no-any-return]


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize(("pickup", "switch"), [(5, False), (7, True)])
def test_hysteresis_in_simulation(allocator: str, pickup: int, switch: bool) -> None:
    """Robot at x = 10 takes A (pickup 20, released 0); at t = 2 it is at x = 12 and B (same
    level, released 2, drop-off one cell left of its pickup) appears.
    c_A = 8 + 1 = 9, w̃_A = 1.02; B at 5: c_B = 7 + 1 = 8, gain = 1 − 0.2 = 0.8 ≤ h → keep;
    B at 7: c_B = 5 + 1 = 6, gain = 3 − 0.2 = 2.8 > h → switch."""
    spec = scenario([10], [{"p": 20, "d": 21}, {"p": pickup, "d": pickup - 1, "release": 2}])
    eng = run(spec, allocator)
    ev = {e.t: e for e in coord_of(eng).policy.events}
    assert ev[2].triggers == ("released",)
    assert ev[2].switched == (((0, 0, 1),) if switch else ())
    assert ((2, 0) in t1_calls(eng)) == switch  # replanning only if the goal changed
    assert eng.summary().tasks_delivered == 2


def test_carried_task_never_preempted() -> None:
    """Robot 0 picks its task at x = 3 at t = 1, then carries it. An urgent task released at
    t = 2 right next to it goes to robot 1 (the only eligible robot), never to robot 0."""
    spec = scenario([2, 29], [{"p": 3, "d": 12}, {"p": 4, "d": 5, "level": 3, "release": 2}])
    eng = run(spec)
    ev = {e.t: e for e in coord_of(eng).policy.events}
    assert ev[2].eligible_robots == 1 and ev[2].assigned == ((1, 1),)
    carried = eng.book.tasks[0]
    assert carried.robot == 0 and carried.completion_time is not None


# ---------------------------------------------------------------- priority ordering


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_priority_ordering_weighted(allocator: str) -> None:
    """One robot at x = 15, two tasks released together.
    (a) Equal travel (c = 6): level 3 at x = 20 before level 1 at x = 10.
    (b) Weighted, not strict: level 2 at x = 29 (c = 15) vs level 1 at x = 14 (c = 2):
        κΔw = 10 < Δc = 13, so the low task goes first."""
    eng = run(scenario([15], [{"p": 10, "d": 9}, {"p": 20, "d": 21, "level": 3}]), allocator)
    t = eng.book.tasks
    assert t[1].pickup_time is not None and t[0].pickup_time is not None
    assert t[1].pickup_time < t[0].pickup_time
    eng = run(scenario([15], [{"p": 14, "d": 13}, {"p": 29, "d": 30, "level": 2}]), allocator)
    t = eng.book.tasks
    assert t[0].pickup_time is not None and t[1].pickup_time is not None
    assert t[0].pickup_time < t[1].pickup_time


def test_x6_task_metrics_by_hand_with_priority_levels() -> None:
    """X-6 (task part, Issue #4): one robot at x = 15, a level-3 task 20 → 21 and a level-1 task
    10 → 9, both released at 0. By hand: the level-3 task is picked at t = 5 and delivered at
    t = 6; the robot then needs 11 moves from x = 21 to x = 10, so the level-1 task is picked at
    t = 17 and delivered at t = 18. Mean flow (6 + 18)/2 = 12; F_w = (4·6 + 1·18)/(4 + 1) = 8.4;
    waits 5 (level 3) and 17 (level 1); nothing censored."""
    eng = run(scenario([15], [{"p": 10, "d": 9}, {"p": 20, "d": 21, "level": 3}]))
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 2 and m.makespan == 18
    assert m.mean_flow_time == 12.0 and m.weighted_flow_time == pytest.approx(8.4)
    assert m.flow_time_by_priority == {"1": 18.0, "2": None, "3": 6.0}
    assert m.waiting_max_by_priority == {"1": 17, "2": None, "3": 5}
    assert m.waiting_p95_by_priority["1"] == 17.0 and m.waiting_censored == 0


# ---------------------------------------------------------------- map events


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_map_event_reallocates_without_unreachable_release(allocator: str) -> None:
    """Robot 0 (x = 5) takes T (pickup 20); robot 1 (x = 28) carries a task 29 → 26 until it
    delivers at t = 4. At t = 3 a map event closes x = 10: (0, T) becomes infeasible, so the
    map-event auction in that step takes T from robot 0 (robot 1 is not eligible yet), and the
    delivery at t = 4 triggers the auction that gives T to robot 1. The path monitor never sees
    an ``unreachable`` pickup (no release)."""
    spec = scenario(
        [5, 28],
        [{"p": 20, "d": 21}, {"p": 29, "d": 26}],
        map_events=[{"time": 3, "cell": [10, 1], "action": "block"}],
    )
    eng = run(spec, allocator)
    ev = {e.t: e for e in coord_of(eng).policy.events}
    assert ev[0].assigned == ((0, 0), (1, 1))
    assert "map_event" in ev[3].triggers and ev[3].unassigned == ((0, 0),)
    assert ev[4].assigned == ((1, 0),)
    assert eng.book.tasks[0].completion_time is not None and eng.book.tasks[0].robot == 1
    assert coord_of(eng).releases == 0
    assert eng.summary().extra["planning"]["allocation"]["lost"] >= 1


def test_unreachable_release_when_the_map_event_auction_is_capped() -> None:
    """The release path of model §5.3 step 4 inside the allocation coordinator. The allocator's
    solve at the map-event step (t = 3) hits its cap, so robot 0 keeps a pickup that the map
    event has cut off; the monitor's T2 call returns ``unreachable``, the robot releases the
    task, and the next step's auction is triggered by ``unreachable`` and ``retry``."""
    from robocoord.allocation import AllocationProblem
    from robocoord.allocation.base import AllocationResult, SolveStats

    class CappedAtT3:
        name = "capped"

        def __init__(self) -> None:
            self.calls = 0

        def allocate(self, problem: AllocationProblem) -> AllocationResult:
            self.calls += 1
            if problem.t == 3:
                return AllocationResult((), 0.0, SolveStats("capped", problem.size, timed_out=True))
            return AuctionAllocator().allocate(problem)

    spec = scenario(
        [5],
        [{"p": 20, "d": 21}],
        max_steps=12,
        map_events=[{"time": 3, "cell": [10, 1], "action": "block"}],
    )
    coord = AllocationCoordinator(make_planner("astar"), CappedAtT3())
    eng = Engine(spec, coord)
    ctrl = SuggestionFollower()
    ctrl.reset(1)
    eng.run(ctrl.act)
    assert coord.releases == 1
    mon = coord.monitor
    assert mon is not None
    assert any(c.t == 3 and c.status == "unreachable" for c in mon.calls)
    ev = {e.t: e for e in coord.policy.events}
    assert ev[3].stats.timed_out and "map_event" in ev[3].triggers
    assert ev[4].triggers == ("unreachable", "retry") and ev[4].assigned == ()
    assert eng.book.tasks[0].robot is None  # pending, infeasible for the only robot


# ---------------------------------------------------------------- aging (constructed scenario)


def _stream(l_pickup: int, h_level: int) -> Any:
    """One robot at x = 10; a level-1 task L released at t = 0; tasks H (11 → 12) released every
    2 steps until t = 120. Serving one H takes exactly 2 steps (12 → 11 → 12), so the robot
    becomes free every 2 steps with exactly one fresh H pending."""
    tasks = [{"p": l_pickup, "d": l_pickup + 1}] + [
        {"p": 11, "d": 12, "level": h_level, "release": r} for r in range(0, 121, 2)
    ]
    return scenario([10], tasks, max_steps=400)


def test_aging_mitigates_waiting_in_constructed_scenario() -> None:
    """A constructed demonstration of starvation *mitigation*, not a guarantee (model §4, G13).

    L at x = 16 (c = 5 from x = 12), H of level 2 (fresh value 20 − 2 = 18).
    * α = 0: L's value 10 − 5 = 5 never beats 18, so L waits until the H stream ends.
    * α = 0.1: L wins once 10(1 + 0.1a) − 5 > 18, i.e. after about 14 steps of waiting; on its
      4-step trip the H released meanwhile (age ≤ 2: value ≤ 20·1.2 − 4 = 20) do not reach L's
      value plus h, so no pre-emption.
    Only the comparison on this instance is asserted."""
    picked = {}
    for alpha in (0.0, 0.1):
        eng = run(_stream(16, 2), config=BenefitConfig(alpha=alpha))
        picked[alpha] = eng.book.tasks[0].pickup_time
        assert eng.summary().tasks_delivered == 62
    assert picked[0.0] is not None and picked[0.1] is not None
    assert picked[0.0] > 120  # without aging: only after the stream of other tasks ends
    assert picked[0.1] < 40  # with aging: early on this instance


def test_aging_gives_no_service_guarantee_under_a_backlog() -> None:
    """Counter-example kept on purpose (model §4 'Not claimed', reason (i) and P3).

    Same stream with L far away (x = 25, c = 14) and H of level 3. With α = 0.1 L does win an
    auction (around t = 42), but while the robot walks to it, H tasks accumulate and *age*
    (weight 4·(1 + 0.1a) grows four times faster than L's), so an aged H pre-empts L. L is then
    served only when the stream ends, as without aging."""
    eng = run(_stream(25, 3), config=BenefitConfig(alpha=0.1))
    events = coord_of(eng).policy.events
    won = [e.t for e in events if (0, 0) in e.assigned]
    lost = [e.t for e in events if any(old == 0 for _, old in e.unassigned)]
    assert won and lost and lost[0] > won[0]  # L was assigned, then pre-empted
    picked = eng.book.tasks[0].pickup_time
    assert picked is not None and picked > 120


# ---------------------------------------------------------------- determinism and invariants


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize("name", ["random_warehouse", "rooms_dynamic"])
def test_deterministic_traces_and_one_auction_per_step(allocator: str, name: str) -> None:
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    outs = []
    for _ in range(2):
        buf = io.StringIO()
        eng = run(spec, allocator, trace=TraceWriter(buf))
        outs.append(buf.getvalue())
        events = coord_of(eng).policy.events
        assert len({e.t for e in events}) == len(events)
        assert all(len(e.pairs) <= min(e.eligible_robots, e.available_tasks) for e in events)
    assert outs[0] == outs[1]
