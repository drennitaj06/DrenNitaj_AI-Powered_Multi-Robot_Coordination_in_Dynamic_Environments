"""Deadlock-detour resolver in the engine (Issue #4 stage 5B; algorithms §5.5).

Every expected outcome below was first observed in a scratch run and is asserted as recorded:
two-lane head-on resolved (both planners × both allocators, unequal and equal priority),
single-lane head-on reported unresolved (no passing place), a 4-robot rotation resolved, a map
event invalidating the passing cell (abort, re-detection, resolution), an obstacle delaying a
detour, the collision property with the resolver on, determinism, and off-by-default behaviour.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.control.baseline import BaselineController
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.engine import Engine
from robocoord.sim.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[2]
ALLOCATORS = {"auction": AuctionAllocator, "sequential": SequentialAllocator}
PLANNERS = [("astar", {}), ("rrtstar", {"max_iterations": 400})]


def run(
    spec: Any,
    *,
    planner: str = "astar",
    allocator: str = "auction",
    detour: bool = True,
    trace: Any = None,
    ctrl: BaselineController | None = None,
) -> tuple[Engine, AllocationCoordinator, BaselineController]:
    opts = {"max_iterations": 400} if planner == "rrtstar" else {}
    coord = AllocationCoordinator(
        make_planner(planner, **opts),
        ALLOCATORS[allocator](),
        deadlock=DetourParams() if detour else None,
    )
    eng = Engine(spec, coord, trace=trace)
    ctrl = ctrl or BaselineController()
    ctrl.reset(len(eng.robots))
    eng.run(ctrl.act)
    return eng, coord, ctrl


def head_on(lanes: int, levels: tuple[int, int] = (1, 2), **extra: Any) -> Any:
    """Robots carry tasks to opposite ends of a 10-cell corridor (1 or 2 lanes); they meet at
    x = 5 / 6 at t = 4 (robot 0 carries level ``levels[0]``, robot 1 ``levels[1]``)."""
    rows = ["#" * 12] + ["#" + "." * 10 + "#"] * lanes + ["#" * 12]
    data = {
        "scenario": {"name": "head-on", "seed": 1, "max_steps": 60},
        "map": {"rows": rows},
        "robots": [{"start": [1, 1]}, {"start": [10, 1]}],
        "tasks": [
            {"pickup": [2, 1], "dropoff": [10, 1], "priority": levels[0], "release": 0},
            {"pickup": [9, 1], "dropoff": [1, 1], "priority": levels[1], "release": 0},
        ],
    }
    data.update(extra)
    return build_scenario(data)


def kinds(coord: AllocationCoordinator) -> list[tuple[int, str, int | None]]:
    assert coord.resolver is not None
    return [(e.t, e.kind, e.robot) for e in coord.resolver.events]


# ---------------------------------------------------------------- the two head-on cases


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
@pytest.mark.parametrize(("levels", "yielder", "cell"), [((1, 2), 0, (5, 2)), ((2, 2), 1, (6, 2))])
def test_two_lane_head_on_is_resolved(
    planner: str,
    opts: dict[str, Any],
    allocator: str,
    levels: tuple[int, int],
    yielder: int,
    cell: tuple[int, int],
) -> None:
    """Detected at t = 5 (2 steps), the lower-priority robot (on a tie: the higher id) steps
    into the other lane (T7 at t = 6), is released at t = 9 and rejoins (T7 at t = 10); both
    tasks are delivered and no conflict is proposed."""
    eng, coord, _ = run(head_on(2, levels), planner=planner, allocator=allocator)
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 2 and m.conflicts_total == 0
    assert kinds(coord) == [
        (5, "detected_persistent", None),
        (5, "detours_started", yielder),
        (9, "released", yielder),
    ]
    assert coord.resolver is not None and coord.resolver.events[1].cell == cell
    mon = coord.monitor
    assert mon is not None
    assert [(c.t, c.robot) for c in mon.calls if c.trigger == "T7"] == [(6, yielder), (10, yielder)]
    assert all(c.trigger != "T4" for c in mon.calls)  # detection precedes T4; T4 held meanwhile
    stats = m.extra["planning"]
    assert stats["calls_by_trigger"]["T7"] == 2 and stats["calls_by_attribution"]["deadlock"] == 2
    assert stats["deadlock"]["released"] == 1 and stats["deadlock"]["unresolved_at_end"] == 0
    assert stats["legs_completed"] == 4  # the detour does not split the drop-off leg


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
def test_single_lane_head_on_is_reported_unresolved(planner: str, opts: dict[str, Any]) -> None:
    """No cell off the other robot's path exists for either robot: reported once as
    ``no_passing_place`` (static), nothing is forced, no conflict, no T7."""
    eng, coord, ctrl = run(head_on(1), planner=planner)
    m = eng.summary()
    assert eng.truncated and m.tasks_delivered == 0 and m.conflicts_total == 0
    assert kinds(coord) == [(5, "detected_persistent", None), (5, "no_passing_place", None)]
    assert m.extra["planning"]["deadlock"]["unresolved_at_end"] == 1
    assert m.extra["planning"]["calls_by_trigger"]["T7"] == 0
    assert ctrl.episode_stats()["decisions"]["robot_ahead"] > 0


# ---------------------------------------------------------------- larger cycle


OPEN = ["######", "#....#", "#....#", "#....#", "#....#", "######"]
ROTATION = [((2, 2), (2, 4)), ((2, 3), (4, 3)), ((3, 3), (3, 1)), ((3, 2), (1, 2))]


@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
@pytest.mark.parametrize(
    ("levels", "yielder", "cell"),
    [((1, 1, 1, 1), 3, (4, 2)), ((3, 3, 3, 1), 3, (4, 2)), ((1, 3, 3, 3), 0, (2, 1))],
)
def test_four_robot_rotation_is_resolved(
    planner: str, opts: dict[str, Any], levels: tuple[int, ...], yielder: int, cell: Any
) -> None:
    """Four robots in a 2 × 2 block, each needing the next one's cell (a 4-cycle; odd cycles
    cannot occur on the grid). The lowest (π̃, −id) robot yields; the rotation drains and all
    four deliver. One constructed instance: no general k-cycle progress claim."""
    spec = build_scenario(
        {
            "scenario": {"name": "rotation", "seed": 1, "max_steps": 40},
            "map": {"rows": OPEN},
            "robots": [{"start": list(s)} for s, _ in ROTATION],
            "tasks": [
                {"pickup": list(s), "dropoff": list(d), "priority": lv, "release": 0}
                for (s, d), lv in zip(ROTATION, levels, strict=True)
            ],
        }
    )
    eng, coord, _ = run(spec, planner=planner)
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 4 and m.conflicts_total == 0
    assert kinds(coord)[:2] == [(2, "detected_persistent", None), (2, "detours_started", yielder)]
    assert coord.resolver is not None and coord.resolver.events[1].cell == cell
    assert coord.resolver.events[1].members == (0, 1, 2, 3)


# ---------------------------------------------------------------- disruptions


def test_map_event_on_passing_cell_aborts_then_resolves() -> None:
    """The passing cell (5, 2) is walled at t = 6: the detour is aborted, the persisting cycle
    is detected again and a new passing cell (4, 2) resolves it."""
    spec = head_on(2, map_events=[{"time": 6, "cell": [5, 2], "action": "block"}])
    eng, coord, _ = run(spec)
    m = eng.summary()
    assert eng.terminated and m.tasks_delivered == 2 and m.conflicts_total == 0
    assert kinds(coord) == [
        (5, "detected_persistent", None),
        (5, "detours_started", 0),
        (6, "aborted", 0),
        (8, "detected_persistent", None),
        (8, "detours_started", 0),
        (12, "released", 0),
    ]
    assert coord.resolver is not None and coord.resolver.events[4].cell == (4, 2)


def test_obstacle_delays_the_detour_without_conflict() -> None:
    """A bouncing obstacle in the passing lane makes the yielder wait twice on its detour
    (predicted / occupied cell); the detour still completes and both tasks are delivered."""
    rows = ["#" * 16, "#" + "." * 14 + "#", "#" + "." * 14 + "#", "#" * 16]
    spec = build_scenario(
        {
            "scenario": {"name": "obs", "seed": 1, "max_steps": 80},
            "map": {"rows": rows},
            "robots": [{"start": [1, 1]}, {"start": [14, 1]}],
            "tasks": [
                {"pickup": [2, 1], "dropoff": [14, 1], "priority": 1, "release": 0},
                {"pickup": [13, 1], "dropoff": [1, 1], "priority": 2, "release": 0},
            ],
            "obstacles": [{"start": [4, 2], "model": "bounce", "direction": "left"}],
        }
    )
    coord = AllocationCoordinator(
        make_planner("astar"), AuctionAllocator(), deadlock=DetourParams()
    )
    eng = Engine(spec, coord)
    ctrl = BaselineController()
    ctrl.reset(2)
    waits = 0
    while not eng.done:
        view = eng.prepare_step()
        acts = ctrl.act(view)
        assert coord.resolver is not None
        waits += sum(
            ctrl.last[y] in ("obstacle_ahead", "predicted_obstacle") for y in coord.resolver.detours
        )
        eng.apply_actions(acts)
    m = eng.summary()
    assert waits == 2 and m.tasks_delivered == 2 and m.conflicts_total == 0
    assert [k for _, k, _ in kinds(coord)] == ["detected_persistent", "detours_started", "released"]


# ---------------------------------------------------------------- safety, determinism, defaults


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize(("planner", "opts"), PLANNERS)
@pytest.mark.parametrize(
    "name",
    ["corridor_queue", "demo_delivery", "planning_detour", "rooms_dynamic", "random_warehouse"],
)
def test_collision_property_with_resolver(
    name: str, planner: str, opts: dict[str, Any], allocator: str
) -> None:
    """The stage 5A property is unchanged with the resolver on (it only changes goals)."""
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    eng, _, _ = run(spec, planner=planner, allocator=allocator)
    c = eng.summary().conflicts
    assert c["RRv"] == c["RRs"] == c["ROs"] == c["INV"] == 0


@pytest.mark.parametrize("name", ["rooms_dynamic", "random_warehouse"])
def test_deterministic_with_resolver(name: str) -> None:
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    outs = []
    for _ in range(2):
        buf = io.StringIO()
        run(spec, trace=TraceWriter(buf))
        outs.append(buf.getvalue())
    assert outs[0] == outs[1]


def test_trace_records_deadlock_events() -> None:
    buf = io.StringIO()
    run(head_on(2), trace=TraceWriter(buf))
    recs = [json.loads(line) for line in buf.getvalue().splitlines()]
    dl = [(r["t"], r["deadlock"]) for r in recs if "deadlock" in r]
    assert dl[0][1][0][0] == "detected_persistent" and dl[0][1][1][:3] == [
        "detours_started",
        [0, 1],
        0,
    ]
    assert recs[0]["coordinator"] == "auction+astar+detour"


def test_off_by_default() -> None:
    """Without ``deadlock=`` the coordinator, its name, its statistics and the trigger table are
    those of stages 4–5A (byte-identical traces were also checked against a pre-5B snapshot of
    46 configurations; see the verification log)."""
    coord = AllocationCoordinator(make_planner("astar"), AuctionAllocator())
    assert coord.resolver is None and coord.name == "auction+astar"
    eng = Engine(head_on(1), coord)
    ctrl = BaselineController()
    ctrl.reset(2)
    m = eng.run(ctrl.act)
    planning = m.extra["planning"]
    assert "deadlock" not in planning and "T7" not in planning["calls_by_trigger"]
    assert planning["calls_by_trigger"]["T4"] > 0  # stage 5A behaviour: symmetric T4 replans
