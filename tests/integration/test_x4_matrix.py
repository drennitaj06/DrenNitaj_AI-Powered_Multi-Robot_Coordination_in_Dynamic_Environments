"""X-4: the complete planner × allocator × controller matrix (Issue #6; ``05`` A.3).

All 16 combinations {A*, RRT*} × {auction, sequential} × {reference, baseline, MADQL, PPO} run a
50-step episode of ``demo_delivery`` (unless it terminates earlier) through the CLI's controller
factory, with seeded **untrained** learned-controller checkpoints (this tests plumbing, not
quality). Every learned controller also runs with the predictive shield on, with both planners and
both allocators (8 more cases). After every step: no physical collision, no ``INV`` event (masked
controllers), legal task-state transitions, and shield events only when the shield is on.
RRT* cases are marked ``slow``.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from robocoord.cli import make_controller, resolve_deadlock  # noqa: E402
from robocoord.control.shield import Shield  # noqa: E402
from robocoord.marl.checkpoint import ACTOR_SPEC_VERSION, save_inference  # noqa: E402
from robocoord.marl.config import ObsParams  # noqa: E402
from robocoord.marl.env import VIEW_EXTRAS  # noqa: E402
from robocoord.marl.madql import MADQL  # noqa: E402
from robocoord.marl.networks import configure_torch  # noqa: E402
from robocoord.marl.ppo import PPO  # noqa: E402
from robocoord.scenario import load_scenario  # noqa: E402
from robocoord.sim.conflicts import physical_violations  # noqa: E402
from robocoord.sim.engine import Engine  # noqa: E402
from robocoord.world.entities import TaskStatus  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "configs" / "scenarios" / "demo_delivery.toml"
STEPS = 50

# forward closure of the task state machine (several transitions can happen within one step,
# e.g. released, assigned and picked up); a task never moves backwards past these
_NEXT = {
    TaskStatus.UNRELEASED: {TaskStatus.PENDING},
    TaskStatus.PENDING: {TaskStatus.ASSIGNED},
    TaskStatus.ASSIGNED: {TaskStatus.PENDING, TaskStatus.PICKED},
    TaskStatus.PICKED: {TaskStatus.DELIVERED},
    TaskStatus.DELIVERED: set(),
}


def reachable(src: TaskStatus) -> set[TaskStatus]:
    seen, todo = {src}, [src]
    while todo:
        for nxt in _NEXT[todo.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Seeded untrained policies of the real architecture (356 → 256 → 256 → 5)."""
    out = tmp_path_factory.mktemp("x4")
    meta = {"config_hash": "x4", "config": {}, "seeds": {"run_seed": 0}, "env_steps": 0,
            "validation": {}, "experiment_id": "x4-untrained"}  # fmt: skip
    configure_torch(41)
    q = MADQL(ObsParams().dim).q
    madql_meta = {**meta, "algorithm": "madql", "variant": MADQL.variant}
    madql = save_inference(out / "madql.pt", q, hidden=(256, 256), obs=ObsParams(), meta=madql_meta)
    configure_torch(42)
    actor = PPO(ObsParams().dim).actor
    ppo = save_inference(out / "ppo.pt", actor, hidden=(256, 256), obs=ObsParams(),
                         meta={**meta, "algorithm": "ppo", "variant": "mappo"},
                         model_version=ACTOR_SPEC_VERSION)  # fmt: skip
    return {"madql": madql, "ppo": ppo}


def run_combination(
    planner: str, allocator: str, controller: str, shield: str, ckpt: Path | None
) -> Engine:
    spec = load_scenario(DEMO)
    deadlock = resolve_deadlock("auto", controller, allocator)
    rrt = 300 if planner == "rrtstar" else None
    ctrl, coord = make_controller(controller, spec, spec.seed, planner, rrt, allocator, deadlock,
                                  True, ckpt)  # fmt: skip
    learned = controller in ("madql", "ppo")
    engine = Engine(
        spec,
        coord,
        max_steps=STEPS,
        shield=Shield("predictive") if shield == "predictive" else None,
        view_extras=VIEW_EXTRAS if learned else frozenset(),
        path_limit=ObsParams().path_cells,
    )
    ctrl.reset(len(engine.robots))
    status = {t.id: t.status for t in engine.book.tasks.values()}
    shield_events = 0
    while not engine.done:
        before = [r.position for r in engine.robots]
        out = engine.step(ctrl.act)
        after = [r.position for r in engine.robots]
        obstacles = [o.position for o in engine.obstacles]
        assert physical_violations(before, after, obstacles, engine.grid) == []
        assert not any(e.kind_code == "INV" for e in out.events)
        for task in engine.book.tasks.values():
            assert task.status in reachable(status[task.id]), (task.id, status[task.id])
            status[task.id] = task.status
        if shield == "off":
            assert out.shield_events == ()
        shield_events += len(out.shield_events)
    assert engine.t == STEPS or engine.terminated
    m = engine.summary()
    assert m.physical_collisions == 0
    shield_stats = m.extra.get("shield")
    if shield == "off":
        assert shield_stats is None
    else:
        assert shield_stats is not None and shield_stats["interventions"] == shield_events
    return engine


def slow_rrt(planner: str) -> list[pytest.MarkDecorator]:
    return [pytest.mark.slow] if planner == "rrtstar" else []


MATRIX = [
    pytest.param(p, a, c, id=f"{p}-{a}-{c}", marks=slow_rrt(p))
    for p, a, c in itertools.product(
        ("astar", "rrtstar"), ("auction", "sequential"), ("reference", "baseline", "madql", "ppo")
    )
]
SHIELDED = [
    pytest.param(p, a, c, id=f"{p}-{a}-{c}-shield", marks=slow_rrt(p))
    for p, a, c in itertools.product(
        ("astar", "rrtstar"), ("auction", "sequential"), ("madql", "ppo")
    )
]


def test_matrix_has_sixteen_combinations_and_eight_shielded_variants() -> None:
    assert len(MATRIX) == 16 and len({p.id for p in MATRIX}) == 16
    assert len(SHIELDED) == 8


@pytest.mark.parametrize(("planner", "allocator", "controller"), MATRIX)
def test_x4_combination_runs_50_steps_safely(
    planner: str, allocator: str, controller: str, checkpoints: dict[str, Path]
) -> None:
    run_combination(planner, allocator, controller, "off", checkpoints.get(controller))


@pytest.mark.parametrize(("planner", "allocator", "controller"), SHIELDED)
def test_x4_learned_controllers_with_the_predictive_shield(
    planner: str, allocator: str, controller: str, checkpoints: dict[str, Path]
) -> None:
    run_combination(planner, allocator, controller, "predictive", checkpoints[controller])
