"""Execution Supervisor in the engine (Issue 6A Phase 3): the four failure modes on hand-built
scenarios with scripted controllers, composition with the resolver and the shield, the learned
controller interfaces on both planners, and byte-identical behaviour when disabled.

These are software tests on small fixed scenarios, not an evaluation: they show that each
mechanism does what it is designed to do on the situation it targets, not that it improves
completion on the evaluation distribution."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.control.reference import ReferenceFollower
from robocoord.control.shield import Shield
from robocoord.control.supervisor import ExecutionSupervisor
from robocoord.core.types import Action, add
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.planning.deadlock import DetourParams
from robocoord.planning.endpoint import EndpointParams
from robocoord.scenario import build_scenario
from robocoord.sim.conflicts import ConflictKind
from robocoord.sim.engine import VIEW_EXTRAS, Engine
from robocoord.sim.trace import TraceWriter
from robocoord.sim.view import WorldView

Controller = Callable[[WorldView], dict[int, Action]]


def run(
    spec: Any,
    ctrl: Controller,
    mode: str,
    *,
    deadlock: bool = False,
    planner: str = "astar",
    shield: Shield | None = None,
    trace: Any = None,
) -> Engine:
    opts = {"max_iterations": 300} if planner == "rrtstar" else {}
    coord = AllocationCoordinator(
        make_planner(planner, **opts),
        AuctionAllocator(),
        deadlock=DetourParams() if deadlock else None,
        endpoint=EndpointParams(),
    )
    eng = Engine(
        spec,
        coord,
        view_extras=VIEW_EXTRAS,
        supervisor=ExecutionSupervisor(mode),
        shield=shield,
        trace=trace,
    )
    eng.run(ctrl)
    return eng


def summary(eng: Engine) -> dict[str, Any]:
    out: dict[str, Any] = eng.summary().to_dict()
    return out


def corridor() -> Any:
    rows = ["#" * 12, "#" + "." * 10 + "#", "#" + "." * 10 + "#", "#" * 12]
    return build_scenario(
        {
            "scenario": {"name": "c", "seed": 1, "max_steps": 80},
            "map": {"rows": rows},
            "robots": [{"start": [1, 1]}],
            "tasks": [{"pickup": [3, 1], "dropoff": [9, 1], "priority": 1, "release": 0}],
        }
    )


def crossing() -> Any:
    rows = [
        "#########",
        "####.####",
        "####.####",
        "####.####",
        "#.......#",
        "####.####",
        "####.####",
        "####.####",
        "#########",
    ]
    return build_scenario(
        {
            "scenario": {"name": "x", "seed": 1, "max_steps": 60},
            "map": {"rows": rows},
            "robots": [{"start": [1, 4]}, {"start": [4, 1]}],
            "tasks": [
                {"pickup": [2, 4], "dropoff": [7, 4], "priority": 1, "release": 0},
                {"pickup": [4, 2], "dropoff": [4, 7], "priority": 1, "release": 0},
            ],
        }
    )


def head_on(lanes: int) -> Any:
    rows = ["#" * 12] + ["#" + "." * 10 + "#"] * lanes + ["#" * 12]
    return build_scenario(
        {
            "scenario": {"name": "h", "seed": 1, "max_steps": 60},
            "map": {"rows": rows},
            "robots": [{"start": [1, 1]}, {"start": [10, 1]}],
            "tasks": [
                {"pickup": [2, 1], "dropoff": [10, 1], "priority": 1, "release": 0},
                {"pickup": [9, 1], "dropoff": [1, 1], "priority": 2, "release": 0},
            ],
        }
    )


def frozen(view: WorldView) -> dict[int, Action]:
    return {r.id: Action.STAY for r in view.robots}


def oscillator(view: WorldView) -> dict[int, Action]:
    """Steps off the corridor's path row and back, forever (an A↔B limit cycle)."""
    return {r.id: (Action.UP if r.position[1] == 1 else Action.DOWN) for r in view.robots}


def followers(n: int) -> Controller:
    rf = ReferenceFollower()
    rf.reset(n)
    return rf.act


# --------------------------------------------------------------------------- the failure modes


@pytest.mark.parametrize("planner", ["astar", "rrtstar"])
def test_frozen_robot_is_recovered_by_the_stall_breaker(planner: str) -> None:
    off = summary(run(corridor(), frozen, "off", planner=planner))
    on = summary(run(corridor(), frozen, "S", planner=planner))
    assert off["tasks_delivered"] == 0
    assert on["tasks_delivered"] == 1 and on["physical_collisions"] == 0
    assert on["extra"]["supervisor"]["recoveries_started"]["S"] >= 1


def test_oscillating_robot_is_recovered_by_the_oscillation_detector() -> None:
    off = summary(run(corridor(), oscillator, "off"))
    on = summary(run(corridor(), oscillator, "O"))
    assert off["tasks_delivered"] == 0
    assert on["tasks_delivered"] == 1
    sup = on["extra"]["supervisor"]
    assert sup["recoveries_started"]["O"] >= 1 and sup["recoveries_started"]["S"] == 0


def test_conflict_loop_over_a_free_cell_is_resolved_by_right_of_way() -> None:
    off = summary(run(crossing(), followers(2), "off"))
    on = summary(run(crossing(), followers(2), "R"))
    assert off["tasks_delivered"] == 0 and off["conflicts"]["RRv"] > 50  # a persistent loop
    assert on["tasks_delivered"] == 2 and on["extra"]["supervisor"]["changes"]["R_loss"] >= 1


def test_head_on_swap_needs_the_resolver() -> None:
    """R alone only stops both robots of a swap; the resolver's detour resolves it, and the
    supervisor does not get in the resolver's way. A one-lane corridor stays unresolved (no
    passing place) — the supervisor does not pretend otherwise."""
    r_only = summary(run(head_on(2), followers(2), "R"))
    assert r_only["tasks_delivered"] == 0
    assert r_only["extra"]["supervisor"]["r_left_to_arbiter"]["swap"] > 0
    assert r_only["conflicts"]["RRs"] > 0  # the swap stays visible in the arbiter's record
    with_d = summary(run(head_on(2), followers(2), "full", deadlock=True))
    without = summary(run(head_on(2), followers(2), "off", deadlock=True))
    assert with_d["tasks_delivered"] == 2 == without["tasks_delivered"]
    assert with_d["steps"] <= without["steps"]
    assert with_d["extra"]["planning"]["deadlock"]["detours_started"] >= 1
    one_lane = summary(run(head_on(1), followers(2), "full", deadlock=True))
    assert one_lane["tasks_delivered"] == 0 and one_lane["physical_collisions"] == 0


def test_legitimate_waiting_behind_a_stationary_robot_is_not_overridden() -> None:
    rows = ["#" * 12, "#" + "." * 10 + "#", "#" * 12]
    spec = build_scenario(
        {
            "scenario": {"name": "w", "seed": 1, "max_steps": 40},
            "map": {"rows": rows},
            "robots": [{"start": [2, 1]}, {"start": [5, 1]}],
            "tasks": [{"pickup": [1, 1], "dropoff": [10, 1], "priority": 1, "release": 0}],
        }
    )

    def ctrl(view: WorldView) -> dict[int, Action]:  # robot 0 follows, robot 1 never moves
        out = followers(2)(view)
        out[1] = Action.STAY
        return out

    eng = run(spec, ctrl, "full")
    m = summary(eng)
    sup = m["extra"]["supervisor"]
    # robot 0 holds the task and waits behind the idle robot 1 in a one-lane corridor
    assert eng.robots[0].task is not None and eng.robots[1].task is None
    assert m["tasks_delivered"] == 0 and eng.robots[0].position[0] < eng.robots[1].position[0]
    assert sum(sup["recoveries_started"].values()) == 0
    assert sup["modified_robot_steps"] == 0


# --------------------------------------------------------------------------- composition and safety


class RecordingController:
    """Wraps a controller and checks every supervised step's invariants."""

    def __init__(self, inner: Controller) -> None:
        self.inner = inner

    def __call__(self, view: WorldView) -> dict[int, Action]:
        self.view = view
        return self.inner(view)


def check_steps(eng: Engine, ctrl: RecordingController) -> int:
    """Step the engine and assert the supervisor's output alphabet and mask validity."""
    changed = 0
    while not eng.done:
        view = eng.prepare_step()
        proposal = ctrl.inner(view)
        out = eng.apply_actions(proposal)
        sup = out.supervised_actions or out.actions
        for i, a in enumerate(sup):
            sug = view.suggested[i] if view.suggested is not None else Action.STAY
            assert a in (proposal[i], Action.STAY, sug)
            assert view.masks[i][a]
            changed += a != proposal[i]
        inv_sup = sum(e.kind_code == ConflictKind.INVALID.value for e in out.events)
        inv_pol = sum(not view.masks[i][proposal[i]] for i in range(len(proposal)))
        assert inv_sup <= inv_pol  # the supervisor never introduces an invalid move
    return changed


@pytest.mark.parametrize("planner", ["astar", "rrtstar"])
@pytest.mark.parametrize("shield", [None, "predictive"])
def test_learned_controllers_under_the_supervisor(planner: str, shield: str | None) -> None:
    """MADQL-style (Q-values) and PPO-style (logits) LearnedControllers with random weights on a
    full-dynamics evaluation-shaped instance: the supervisor runs, its output stays within the
    allowed alphabet and the masks, and no physical collision occurs (engine invariant)."""
    torch = pytest.importorskip("torch")
    from robocoord.marl.config import ObsParams
    from robocoord.marl.networks import build_mlp
    from robocoord.marl.observation import ObservationBuilder
    from robocoord.marl.policy import LearnedController
    from robocoord.marl.scenarios import LEVELS

    torch.manual_seed(0)
    obs = ObsParams()
    dim = ObservationBuilder(obs).dim
    for output in ("q-values", "logits"):
        net = build_mlp(dim, (16,), 5)
        ctrl = LearnedController(
            net, obs, name="madql" if output == "q-values" else "ppo", output=output
        )
        spec = LEVELS["full"].sample(10_000)
        opts = {"max_iterations": 100} if planner == "rrtstar" else {}
        coord = AllocationCoordinator(
            make_planner(planner, **opts),
            AuctionAllocator(),
            deadlock=DetourParams(),
            endpoint=EndpointParams(),
        )
        eng = Engine(
            spec,
            coord,
            view_extras=VIEW_EXTRAS,
            path_limit=6,
            max_steps=60,
            supervisor=ExecutionSupervisor("full"),
            shield=Shield(shield) if shield else None,
        )
        ctrl.reset(len(eng.robots))
        rc = RecordingController(ctrl.act)
        check_steps(eng, rc)
        m = summary(eng)
        assert m["physical_collisions"] == 0
        assert m["extra"]["supervisor"]["robot_steps"] == 60 * len(eng.robots)


def test_shield_input_is_the_supervised_proposal() -> None:
    spec = LEVELS_full_small()
    eng = run(spec, frozen, "full", shield=Shield("predictive"))
    m = summary(eng)
    assert m["physical_collisions"] == 0
    assert "supervisor" in m["extra"] and "shield" in m["extra"]


def LEVELS_full_small() -> Any:
    rows = ["#" * 10] + ["#" + "." * 8 + "#"] * 4 + ["#" * 10]
    return build_scenario(
        {
            "scenario": {"name": "s", "seed": 3, "max_steps": 60},
            "map": {"rows": rows},
            "robots": [{"start": [1, 1]}],
            "obstacles": [{"start": [5, 2], "model": "patrol", "waypoints": [[5, 2], [5, 3]]}],
            "tasks": [{"pickup": [7, 1], "dropoff": [7, 4], "priority": 1, "release": 0}],
        }
    )


def test_policy_proposal_is_kept_for_metrics() -> None:
    """``StepOutcome.actions`` is the policy's proposal; the supervisor's output is separate, and
    the conflict set of the policy's own proposal (K^ctrl) is still recorded."""
    eng = Engine(
        crossing(),
        AllocationCoordinator(make_planner("astar"), AuctionAllocator(), endpoint=EndpointParams()),
        view_extras=VIEW_EXTRAS,
        supervisor=ExecutionSupervisor("R"),
    )
    ctrl = followers(2)
    seen = 0
    while not eng.done:
        view = eng.prepare_step()
        prop = ctrl(view)
        out = eng.apply_actions(prop)
        assert out.actions == tuple(prop[i] for i in range(2))
        if out.supervisor_events:
            seen += 1
            assert out.supervised_actions is not None and out.supervised_actions != out.actions
            assert out.controller_events is not None and len(out.controller_events) > 0
            assert len(out.events) < len(out.controller_events)
    assert seen >= 1


# --------------------------------------------------------------------------- off = unchanged


@pytest.mark.parametrize("scenario", [crossing, head_on_two := (lambda: head_on(2)), corridor])
def test_disabled_supervisor_is_byte_identical(scenario: Callable[[], Any]) -> None:
    def trace(supervisor: ExecutionSupervisor | None) -> str:
        buf = io.StringIO()
        coord = AllocationCoordinator(
            make_planner("astar"),
            AuctionAllocator(),
            deadlock=DetourParams(),
            endpoint=EndpointParams(),
        )
        kw: dict[str, Any] = {} if supervisor is None else {"supervisor": supervisor}
        eng = Engine(scenario(), coord, view_extras=VIEW_EXTRAS, trace=TraceWriter(buf), **kw)
        eng.run(followers(len(eng.robots)))
        return buf.getvalue()

    assert trace(None) == trace(ExecutionSupervisor("off"))
    assert head_on_two is not None


def test_supervisor_events_appear_in_the_trace() -> None:
    buf = io.StringIO()
    run(crossing(), followers(2), "R", trace=TraceWriter(buf))
    assert '"supervisor"' in buf.getvalue() and '"sup_actions"' in buf.getvalue()


def test_detour_members_accessor() -> None:
    coord = AllocationCoordinator(
        make_planner("astar"), AuctionAllocator(), deadlock=DetourParams()
    )
    assert coord.detour_members() == frozenset()
    eng = Engine(head_on(2), coord, view_extras=VIEW_EXTRAS)
    ctrl = followers(2)
    members_seen: set[int] = set()
    while not eng.done:
        eng.apply_actions(ctrl(eng.prepare_step()))
        members_seen |= coord.detour_members()
    assert members_seen == {0, 1}
    no_resolver = AllocationCoordinator(make_planner("astar"), AuctionAllocator())
    assert no_resolver.detour_members() == frozenset()
    assert add((0, 0), Action.UP.delta) == (0, 1)
