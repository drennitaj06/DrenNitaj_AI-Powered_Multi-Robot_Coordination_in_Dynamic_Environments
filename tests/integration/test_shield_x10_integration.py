"""Shield and X-10 counterfactual metrics in the engine (Issue #4 stage 6; model §9.2).

Per step, with ``a^ref`` the suggested actions (masked → stay), ``a^ctrl`` the controller's raw
proposal and ``a^arb`` the proposal reaching the arbiter (after the shield):

* suggested conflict  — ``e ∈ K^ref``;           conflict-free suggestion — ``K^ref = ∅``;
* avoided / introduced / persisted — ``K^ref ∖ K^ctrl`` / ``K^ctrl ∖ K^ref`` / ``K^ref ∩ K^ctrl``;
* shield-prevented — ``K^ctrl ∖ K^arb``;          shield-introduced — ``K^arb ∖ K^ctrl``;
* arbiter-prevented — every ``e ∈ K^arb`` (emergency stop, Lemma 3.1);
* executed — a physical collision after resolution: 0 by the invariant (checked every step).
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from robocoord.allocation.auction import AuctionAllocator
from robocoord.allocation.sequential import SequentialAllocator
from robocoord.control.baseline import BaselineController
from robocoord.control.reference import ReferenceFollower
from robocoord.control.shield import Shield
from robocoord.core.types import Action
from robocoord.planning import make_planner
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.scenario import build_scenario, load_scenario
from robocoord.sim.engine import Engine, StepOutcome
from robocoord.sim.trace import TraceWriter
from robocoord.sim.view import WorldView

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = [
    "corridor_queue",
    "demo_delivery",
    "planning_detour",
    "rooms_dynamic",
    "random_warehouse",
]
ALLOCATORS = {"auction": AuctionAllocator, "sequential": SequentialAllocator}


def make(
    spec: Any,
    planner: str = "astar",
    allocator: str = "auction",
    shield: bool = False,
    trace: Any = None,
) -> Engine:
    opts = {"max_iterations": 400} if planner == "rrtstar" else {}
    coord = AllocationCoordinator(make_planner(planner, **opts), ALLOCATORS[allocator]())
    return Engine(spec, coord, trace=trace, shield=Shield() if shield else None)


def collect(eng: Engine, ctrl: Any) -> list[StepOutcome]:
    ctrl.reset(len(eng.robots))
    outs: list[StepOutcome] = []
    eng.run(ctrl.act, on_step=lambda _e, o: outs.append(o))
    return outs


# ---------------------------------------------------------------- X-10: reference identities


@pytest.mark.parametrize("shield", [False, True])
@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize("planner", ["astar", "rrtstar"])
@pytest.mark.parametrize("name", SCENARIOS)
def test_x10_reference_identities(name: str, planner: str, allocator: str, shield: bool) -> None:
    """X-10 (05): for the reference follower N^avoided = N^introduced = 0 and K^ctrl = K^ref at
    every step, with the shield off and on (the shield acts after a^ctrl)."""
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    eng = make(spec, planner, allocator, shield)
    outs = collect(eng, ReferenceFollower())
    assert all(o.reference_events is not None for o in outs)
    assert all(o.controller_events == o.reference_events for o in outs)
    m = eng.summary()
    assert m.conflicts_avoided == 0 and m.conflicts_introduced == 0
    assert m.conflicts_persisted == m.reference_conflicts
    if not shield:
        assert all(o.events == o.reference_events for o in outs)  # a^arb = a^ctrl = a^ref


# ---------------------------------------------------------------- the categories, constructed

CROSS = (
    ["#########"]
    + ["#" + ("." * 7 if y == 4 else "###.###") + "#" for y in range(7, 0, -1)]
    + ["#########"]
)


def junction() -> Any:
    """Two robots on their own pickups next to the centre; at t = 1 both suggestions target the
    centre (4, 4): a suggested RR vertex conflict."""
    return build_scenario(
        {
            "scenario": {"name": "junction", "seed": 1, "max_steps": 30},
            "map": {"rows": CROSS},
            "robots": [{"start": [3, 4]}, {"start": [4, 3]}],
            "tasks": [
                {"pickup": [3, 4], "dropoff": [4, 7], "priority": 2, "release": 0},
                {"pickup": [4, 3], "dropoff": [4, 6], "priority": 1, "release": 0},
            ],
        }
    )


def test_conflict_free_suggestion_and_arbiter_prevented() -> None:
    """Reference follower: t = 0 has no suggested conflict; at t = 1 the suggested vertex
    conflict is proposed, reaches the arbiter, is prevented (both robots frozen) and nothing is
    executed (no physical collision)."""
    eng = make(junction())
    outs = collect(eng, ReferenceFollower())
    assert outs[0].reference_events == frozenset()  # conflict-free suggestion
    k = outs[1]
    assert k.reference_events is not None and len(k.reference_events) == 1
    (e,) = k.reference_events
    assert e.kind_code == "RRv" and e.entities == ("r0", "r1")
    assert k.events == k.reference_events  # proposed to the arbiter ...
    m = eng.summary()
    assert m.conflicts_prevented >= 1 and m.physical_collisions == 0  # ... prevented, not executed


def test_avoided_by_the_baseline() -> None:
    """Baseline: the same suggested conflict is not proposed (robot 1 yields): avoided."""
    eng = make(junction())
    outs = collect(eng, BaselineController())
    k = outs[1]
    assert k.reference_events is not None and len(k.reference_events) == 1
    assert k.controller_events == frozenset() and k.events == frozenset()
    m = eng.summary()
    assert m.conflicts_avoided == 1 and m.conflicts_introduced == 0 and m.avoidance_rate == 1.0
    assert m.tasks_delivered == 2


class Pusher:
    """A deliberately bad controller: plays the suggestions, but robot 1 moves left at t = 3
    even if its plan says otherwise — used only to show an *introduced* conflict."""

    name = "pusher"

    def reset(self, n: int) -> None:
        return None

    def act(
        self, view: WorldView, observations: Mapping[int, Any] | None = None
    ) -> dict[int, Action]:
        out = ReferenceFollower().act(view)
        if view.t == 3:
            out[1] = Action.LEFT
        return out


def test_introduced_by_a_controller() -> None:
    spec = build_scenario(
        {
            "scenario": {"name": "push", "seed": 1, "max_steps": 6},
            "map": {"rows": ["#" * 8, "#" + "." * 6 + "#", "#" * 8]},
            "robots": [{"start": [1, 1]}, {"start": [3, 1]}],
            "tasks": [  # robot 0 delivers at (2, 1) by t = 2; a later task keeps the episode open
                {"pickup": [1, 1], "dropoff": [2, 1], "priority": 1, "release": 0},
                {"pickup": [6, 1], "dropoff": [5, 1], "priority": 1, "release": 20},
            ],
        }
    )
    eng = make(spec)
    outs = collect(eng, Pusher())
    m = eng.summary()
    intro = [o for o in outs if o.controller_events and o.reference_events is not None]
    assert intro and all(o.controller_events - o.reference_events for o in intro)  # type: ignore[operator]
    assert m.conflicts_introduced is not None and m.conflicts_introduced >= 1


def obstacle_corridor() -> Any:
    """One robot and a bouncing obstacle in a single-lane corridor: the plan becomes degraded
    (it must cross the obstacle's cells), so plain plan-following proposes RO conflicts."""
    return build_scenario(
        {
            "scenario": {"name": "obs", "seed": 1, "max_steps": 40},
            "map": {"rows": ["#" * 14, "#" + "." * 12 + "#", "#" * 14]},
            "robots": [{"start": [1, 1]}],
            "obstacles": [{"start": [8, 1], "model": "bounce", "direction": "left"}],
            "tasks": [{"pickup": [2, 1], "dropoff": [12, 1], "priority": 1, "release": 0}],
        }
    )


def test_shield_prevented_conflicts() -> None:
    off = make(obstacle_corridor())
    off_outs = collect(off, ReferenceFollower())
    on = make(obstacle_corridor(), shield=True)
    on_outs = collect(on, ReferenceFollower())
    m_off, m_on = off.summary(), on.summary()
    ro_off = m_off.conflicts["ROv"] + m_off.conflicts["ROs"]
    assert ro_off > 0  # plain plan-following proposes robot–obstacle conflicts here
    sh = m_on.extra["shield"]
    assert sh["interventions"] > 0 and sh["prevented"] > 0
    prevented = [o for o in on_outs if o.controller_events and o.controller_events - o.events]
    assert prevented and all(o.shield_events for o in prevented)
    assert all(o.arbitrated_actions is not None for o in on_outs if o.shield_events)
    assert m_on.physical_collisions == m_off.physical_collisions == 0
    assert "shield" not in m_off.extra and len(off_outs) > 0


@pytest.mark.parametrize("allocator", ALLOCATORS)
@pytest.mark.parametrize("planner", ["astar", "rrtstar"])
@pytest.mark.parametrize("name", SCENARIOS)
def test_shield_never_intervenes_for_the_baseline(name: str, planner: str, allocator: str) -> None:
    """With the same sensing radius the baseline already stays in front of current and
    predicted obstacle cells, so the shield has nothing to replace."""
    spec = load_scenario(ROOT / f"configs/scenarios/{name}.toml")
    eng = make(spec, planner, allocator, shield=True)
    outs = collect(eng, BaselineController())
    assert not any(o.shield_events for o in outs)
    assert eng.summary().extra["shield"]["interventions"] == 0


def test_shield_trace_fields_and_determinism() -> None:
    texts = []
    for _ in range(2):
        buf = io.StringIO()
        eng = make(obstacle_corridor(), shield=True, trace=TraceWriter(buf))
        collect(eng, ReferenceFollower())
        texts.append(buf.getvalue())
    assert texts[0] == texts[1]
    assert '"shield":"predictive"' in texts[0].splitlines()[0]
    assert '"arb_actions"' in texts[0] and '"suggested"' in texts[0]
