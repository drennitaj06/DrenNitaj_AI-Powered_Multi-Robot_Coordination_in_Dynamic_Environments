"""Regression: with the Execution Supervisor off (the default) and the new ``EvalTask`` fields at
their defaults, ``run_episode`` reproduces stored historical records exactly — every field except
timing, and the field order. Uses the stored block-B records and checkpoints under ``results/``
(skipped where they are absent, e.g. in CI). Also: the explicit resolver switch, the supervisor's
additive record fields, and the learned-controllers-only rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
STOCHASTIC = ROOT / "results/dev6/stochastic/records.jsonl"
MATCHED = ROOT / "results/dev6/baseline_matched/records.jsonl"
DEV5 = ROOT / "results/dev5/comparison/records.jsonl"
TIMING = {"wall_s", "prepare_s", "act_s"}

pytestmark = pytest.mark.skipif(not STOCHASTIC.exists(), reason="stored results/ not present")


def stored(path: Path, **match: Any) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if all(r.get(k) == v for k, v in match.items()):
                return dict(r)
    raise LookupError(match)


def replay(rec: dict[str, Any], monkeypatch: pytest.MonkeyPatch, **kw: Any) -> dict[str, Any]:
    pytest.importorskip("torch")
    from robocoord.marl.evaluation import EvalTask, run_episode

    monkeypatch.chdir(ROOT)
    label = rec.get("label") or rec["controller"]
    ckpt = None
    if rec["controller"] in ("madql", "ppo"):
        ckpt = f"results/training/{label.replace('-', '_')}/best.pt"
    task = EvalTask(
        rec["controller"],
        rec["level"],
        rec["seed"],
        rec["planner"],
        "B",
        ckpt,
        label=rec.get("label", ""),
        mode=rec.get("mode", "greedy"),
        replicate=rec.get("replicate", 0),
        **kw,
    )
    out: dict[str, Any] = run_episode(task)
    return out


@pytest.mark.parametrize(
    ("path", "match"),
    [
        (STOCHASTIC, {"label": "madql-seed1", "level": "full", "seed": 10020, "mode": "greedy"}),
        (STOCHASTIC, {"label": "ippo-seed1", "level": "static", "seed": 10021, "mode": "greedy"}),
        (
            STOCHASTIC,
            {
                "label": "mappo-seed2",
                "level": "obstacles",
                "seed": 10022,
                "mode": "stochastic",
                "replicate": 1,
            },
        ),
        (MATCHED, {"level": "static", "seed": 10023, "planner": "astar"}),
    ],
)
def test_default_records_are_reproduced_exactly(
    path: Path, match: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = stored(path, **match)
    new = replay(rec, monkeypatch)
    assert list(new) == list(rec)  # identical field order: nothing added at the defaults
    assert {k: v for k, v in new.items() if k not in TIMING} == {
        k: v for k, v in rec.items() if k not in TIMING
    }


def test_resolver_baseline_record_values_reproduced(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original resolver-on baseline (Issue #5 schema: compare the shared fields)."""
    rec = stored(DEV5, controller="baseline", level="static", seed=10024, planner="astar")
    new = replay(rec, monkeypatch)
    shared = (set(rec) & set(new)) - TIMING
    assert shared and {k: new[k] for k in shared} == {k: rec[k] for k in shared}
    assert new["deadlock"] is True


def test_supervisor_record_fields_are_additive(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = stored(STOCHASTIC, label="madql-seed1", level="static", seed=10020, mode="greedy")
    new = replay(rec, monkeypatch, supervisor="full")
    assert list(new)[: len(rec)] == list(rec)  # the historical fields, in order, then additions
    assert list(new)[len(rec) :] == ["supervisor", "supervisor_stats", "calls_by_trigger"]
    assert new["supervisor"] == "full" and new["physical_collisions"] == 0
    stats = new["supervisor_stats"]
    assert stats["robot_steps"] == new["robots"] * new["steps"]
    from robocoord.control.supervisor import frozen_params

    frozen = frozen_params(rec["planner"])  # evaluation runs use the Stage 3a freeze
    assert stats["params"] == {k: getattr(frozen, k) for k in frozen.__slots__}
    assert set(stats["monitor"]) >= {"max_recoveries_in_a_leg", "max_robot_modified_share"}
    assert set(new["calls_by_trigger"]) >= {"T1", "T2", "T3"}


def test_supervisor_is_for_learned_controllers_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from robocoord.marl.evaluation import EvalTask, run_episode

    monkeypatch.chdir(ROOT)
    with pytest.raises(ValueError, match="learned controllers only"):
        run_episode(EvalTask("baseline", "static", 10020, block="B", supervisor="full"))


@pytest.mark.parametrize(
    ("controller", "deadlock", "expected"),
    [
        ("baseline", None, True),
        ("baseline", False, False),
        ("ppo", None, False),
        ("ppo", True, True),
        ("madql", True, True),
        ("reference", None, False),
    ],
)
def test_explicit_resolver_switch(controller: str, deadlock: bool | None, expected: bool) -> None:
    from robocoord.marl.evaluation import EvalTask, resolver_enabled

    task = EvalTask(controller, "static", 10020, block="B", deadlock=deadlock)
    assert resolver_enabled(task) is expected
