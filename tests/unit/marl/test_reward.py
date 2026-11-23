"""Reward terms and the approved closed forms (model §7; algorithms §8 test 2; design R6/B).

With ``Φ = −Λ/(W+H)``, ``F = γΦ' − Φ`` and the step cost for a non-idle robot:

* a ``k``-step cycle returning to the same state has net discounted return
  ``(1 − γ^k)(ζΛ/(W+H) − c_step/(1 − γ))`` — at the defaults ``(1 − γ^k)(ζΛ/(W+H) − 1)``,
  positive exactly when ``Λ > (W+H)/ζ``;
* a progress move beats ``stay`` by exactly ``ζγ/(W+H)``;
* finishing at ``t_f < T`` beats staying by
  ``R_pick γ^{t_p} + R_del w γ^{t_f} + (c/(1−γ))(γ^{t_f} − γ^T) + ζ|Φ_0|γ^T``;
* truncation before finishing: progress beats staying by ``ζγ^T(|Φ_0| − |Φ_T|)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from robocoord.core.distance import DistanceOracle
from robocoord.core.grid import OccupancyGrid
from robocoord.marl.config import RewardParams
from robocoord.marl.reward import potentials, remaining_route, step_rewards, team_potential
from robocoord.sim.conflicts import ConflictEvent
from robocoord.sim.engine import StepOutcome
from robocoord.world.entities import Robot, RobotMode, Task, TaskStatus

WH = 64.0  # W + H of a 32x32 map


def outcome(events=(), picked=(), delivered=(), shield=()) -> StepOutcome:  # type: ignore[no-untyped-def]
    return StepOutcome(
        t=1,
        actions=(),
        events=frozenset(events),
        blocked=frozenset(),
        released=(),
        assigned=(),
        picked=tuple(picked),
        delivered=tuple(delivered),
        map_events=(),
        reference_events=None,
        terminated=False,
        truncated=False,
        shield_events=tuple(shield),
    )


@dataclass
class FakeShieldEvent:
    robot: int


def zeros(n: int) -> np.ndarray:
    return np.zeros(n)


def test_step_cost_only_for_non_idle_robots() -> None:
    p = RewardParams(beta=0.0)
    r = step_rewards(p, [True, False], outcome(), {}, zeros(2), zeros(2))
    assert r.base.tolist() == [-0.01, 0.0]
    assert r.total.tolist() == [-0.01, 0.0]


def test_pickup_and_priority_weighted_delivery() -> None:
    p = RewardParams(beta=0.0, c_step=0.0)
    r = step_rewards(
        p,
        [True] * 3,
        outcome(picked=[(0, 5)], delivered=[(1, 7), (2, 8)]),
        {7: 1, 8: 3},
        zeros(3),
        zeros(3),
    )
    assert r.base.tolist() == [0.5, 1.0, 4.0]  # w(1) = 1, w(3) = 2^2


def test_conflicts_attributed_to_robots_at_fault() -> None:
    p = RewardParams(beta=0.0, c_step=0.0)
    events = [
        ConflictEvent(1, "RRv", ("r0", "r1"), (1,)),  # robot 0 stayed: only robot 1 at fault
        ConflictEvent(1, "RRs", ("r2", "r3"), (2, 3)),  # swap: both at fault
        ConflictEvent(1, "ROv", ("o0", "r0"), (0,)),
        ConflictEvent(1, "INV", ("r3",), (3,)),  # invalid moves are masked; not a reward term
    ]
    r = step_rewards(
        p, [True] * 4, outcome(events=events, shield=[FakeShieldEvent(2)]), {}, zeros(4), zeros(4)
    )
    assert r.rr.tolist() == [0, 1, 1, 1] and r.ro.tolist() == [1, 0, 0, 0]
    assert r.sh.tolist() == [0, 0, 1, 0]
    assert r.base.tolist() == [-1.0, -1.0, -1.5, -1.0]


def test_team_mixing_and_shaping_of_psi() -> None:
    p = RewardParams(beta=0.3)
    phi, phi2 = np.array([-0.5, -0.2, 0.0]), np.array([-0.4, -0.3, -0.1])
    r = step_rewards(p, [True, True, False], outcome(picked=[(1, 0)]), {}, phi, phi2)
    expect = 0.7 * r.individual + 0.3 * r.individual.mean()
    assert np.allclose(r.total, expect)
    # team-mixed shaping equals the shaping of Ψ (X-13)
    mixed_shaping = 0.7 * r.shaping + 0.3 * r.shaping.mean()
    psi_shaping = p.zeta * (p.gamma * team_potential(phi2, 0.3) - team_potential(phi, 0.3))
    assert np.allclose(mixed_shaping, psi_shaping)


def test_idle_robot_waiting_is_never_penalised() -> None:
    p0 = RewardParams(beta=0.0)
    for _ in range(5):
        r = step_rewards(p0, [False], outcome(), {}, zeros(1), zeros(1))
        assert r.total[0] == 0.0
    # with β > 0 an idle robot receives exactly the team share, with no own term
    p = RewardParams(beta=0.3)
    phi = np.array([-0.5, 0.0])
    r = step_rewards(p, [True, False], outcome(), {}, phi, phi)
    assert r.individual[1] == 0.0
    assert r.total[1] == pytest.approx(0.3 * r.individual.mean())


def discounted(rs: list[float], g: float) -> float:
    return sum(g**t * x for t, x in enumerate(rs))


@pytest.mark.parametrize("zeta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("k", [1, 2, 3, 8])
@pytest.mark.parametrize("lam", [10.0, 31.9, 32.1, 63.9, 64.1, 90.0, 200.0])
def test_cycle_closed_form(zeta: float, k: int, lam: float) -> None:
    p = RewardParams(beta=0.0, zeta=zeta)
    phi = np.array([-lam / WH])
    # a k-step cycle that returns to the same state: Φ is the same at both ends of every step
    rs = [step_rewards(p, [True], outcome(), {}, phi, phi).total[0] for _ in range(k)]
    expect = (1 - p.gamma**k) * (zeta * lam / WH - p.c_step / (1 - p.gamma))
    assert discounted(rs, p.gamma) == pytest.approx(expect, abs=1e-12)
    defaults = (1 - p.gamma**k) * (zeta * lam / WH - 1.0)  # c_step / (1 − γ) = 1 at the defaults
    assert expect == pytest.approx(defaults, abs=1e-12)
    assert (expect > 0) == (lam > WH / zeta)  # threshold Λ > (W + H)/ζ


@pytest.mark.parametrize("lam", [10.0, 90.0])
def test_two_cell_oscillation_closed_form(lam: float) -> None:
    p = RewardParams(beta=0.0)
    a, b = np.array([-lam / WH]), np.array([-(lam + 1) / WH])
    rs = [
        step_rewards(p, [True], outcome(), {}, a, b).total[0],
        step_rewards(p, [True], outcome(), {}, b, a).total[0],
    ]
    expect = (1 - p.gamma**2) * (lam / WH - p.c_step / (1 - p.gamma))
    assert discounted(rs, p.gamma) == pytest.approx(expect, abs=1e-12)


@pytest.mark.parametrize("lam", [5.0, 40.0, 200.0])
def test_progress_beats_stay_by_exact_gap(lam: float) -> None:
    p = RewardParams(beta=0.0)
    phi = np.array([-lam / WH])
    move = step_rewards(p, [True], outcome(), {}, phi, phi + 1 / WH).total[0]
    stay = step_rewards(p, [True], outcome(), {}, phi, phi).total[0]
    assert move - stay == pytest.approx(p.zeta * p.gamma / WH, abs=1e-12)


def test_finishing_beats_staying_closed_form() -> None:
    p = RewardParams(beta=0.0)
    g, c = p.gamma, p.c_step
    lam0, tp, tf, horizon, prio = 70.0, 20, 45, 100, 2
    # finishing: a straight route of length lam0 (pickup after tp moves, delivery after tf = lam0)
    # is not needed for the identity; any Φ path from -lam0/WH to 0 gives the same telescoped sum
    phis = np.linspace(-lam0 / WH, 0.0, tf + 1)
    fin = []
    for t in range(tf):
        picked = [(0, 0)] if t == tp - 1 else []
        delivered = [(0, 0)] if t == tf - 1 else []
        fin.append(
            step_rewards(
                p,
                [True],
                outcome(picked=picked, delivered=delivered),
                {0: prio},
                phis[t : t + 1],
                phis[t + 1 : t + 2],
            ).total[0]
        )
    stay = [
        step_rewards(p, [True], outcome(), {}, phis[:1], phis[:1]).total[0] for _ in range(horizon)
    ]
    diff = discounted(fin, g) - discounted(stay, g)
    expect = (
        p.r_pick * g ** (tp - 1)
        + p.r_del * 2 * g ** (tf - 1)
        + (c / (1 - g)) * (g**tf - g**horizon)
        + p.zeta * (lam0 / WH) * g**horizon
    )
    assert diff == pytest.approx(expect, abs=1e-10)
    assert diff > 0


def test_truncated_progress_beats_staying_closed_form() -> None:
    p = RewardParams(beta=0.0)
    g, horizon = p.gamma, 100
    phi0, phi_t = -70.0 / WH, -30.0 / WH
    phis = np.linspace(phi0, phi_t, horizon + 1)
    prog = [
        step_rewards(p, [True], outcome(), {}, phis[t : t + 1], phis[t + 1 : t + 2]).total[0]
        for t in range(horizon)
    ]
    stay = [
        step_rewards(p, [True], outcome(), {}, phis[:1], phis[:1]).total[0] for _ in range(horizon)
    ]
    diff = discounted(prog, g) - discounted(stay, g)
    assert diff == pytest.approx(p.zeta * g**horizon * (abs(phi0) - abs(phi_t)), abs=1e-12)


def task(pickup, dropoff) -> Task:  # type: ignore[no-untyped-def]
    t = Task(0, pickup, dropoff, 1, 0)
    t.status = TaskStatus.ASSIGNED
    return t


def test_potential_continuous_across_pickup() -> None:
    oracle = DistanceOracle(OccupancyGrid.empty(8, 8))
    tk = task((3, 3), (6, 1))
    before = Robot(0, (3, 3), RobotMode.TO_PICKUP, 0)
    after = Robot(0, (3, 3), RobotMode.TO_DROPOFF, 0)
    tasks = {0: tk}
    assert remaining_route(before, tasks, oracle) == remaining_route(after, tasks, oracle) == 5
    assert (
        potentials([before], tasks, oracle, 8, 8)[0] == potentials([after], tasks, oracle, 8, 8)[0]
    )


def test_capped_distance_when_goal_unreachable() -> None:
    grid = OccupancyGrid.from_rows(["..#.", "..#.", "..#."])
    oracle = DistanceOracle(grid)
    tk = task((1, 1), (3, 1))  # dropoff walled off
    r = Robot(0, (0, 1), RobotMode.TO_PICKUP, 0)
    assert remaining_route(r, {0: tk}, oracle) == 1 + 4 * 3  # D̄ = W·H when unreachable
    idle = Robot(1, (0, 0))
    assert remaining_route(idle, {0: tk}, oracle) == 0
