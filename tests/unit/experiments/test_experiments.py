"""Experiment infrastructure: spec parsing, resumable runner, records, statistics, instances."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from robocoord.core.errors import ConfigError
from robocoord.experiments.planning_runs import (
    e2_scenario,
    euclid_lower_bound,
    make_map,
    make_pair,
)
from robocoord.experiments.records import RecordStore, RunRecord, _plain
from robocoord.experiments.runner import execute, run_experiment
from robocoord.experiments.spec import load_spec, parse_spec
from robocoord.experiments.stats import (
    bootstrap_ci,
    holm,
    loglog_fit,
    paired_bootstrap,
    polylog_fit,
    rank_biserial,
)

ROOT = Path(__file__).resolve().parents[3]

TINY = {
    "experiment": {"name": "tiny", "kind": "planner_queries", "seed_base": 39_500, "timeout_s": 5},
    "factors": {
        "family": ["open", "d20"],
        "size": [8],
        "map": [0],
        "pair": [0, 1],
        "planner": ["a", "r"],
    },
    "planners": [
        {"label": "a", "name": "astar"},
        {"label": "r", "name": "rrtstar", "options": {"max_iterations": 300}},
    ],
}


def test_spec_expansion_and_shipped_configs() -> None:
    spec = parse_spec(TINY)
    runs = spec.runs()
    assert len(runs) == 8 and len({r.run_id for r in runs}) == 8
    assert runs[0].run_id == "family=open|size=8|map=0|pair=0|planner=a"
    sizes = {
        "e1a_planners_static": 11_200,
        "e1b_neighbor_index": 480,
        "e1c_convergence": 20,
        "e2_replanning": 640,
    }
    for name, n in sizes.items():
        assert len(load_spec(ROOT / "configs/experiments" / f"{name}.toml").runs()) == n


@pytest.mark.parametrize(
    ("patch", "field"),
    [
        ({"experiment": {"name": "x", "kind": "magic"}}, "experiment.kind"),
        ({"experiment": {"kind": "replanning"}}, "experiment.name"),
        ({"bogus": {}}, "bogus"),
        ({"factors": {"size": []}}, "factors.size"),
        ({"factors": {"planner": ["zzz"]}}, "factors.planner"),
        ({"experiment": {"name": "x", "kind": "replanning", "workers": 0}}, "experiment.workers"),
    ],
)
def test_spec_errors_name_the_field(patch: dict[str, object], field: str) -> None:
    data = {**TINY, **patch}
    with pytest.raises(ConfigError) as info:
        parse_spec(data)
    assert info.value.field == field
    with pytest.raises(ConfigError):
        load_spec(ROOT / "missing.toml")


def test_runner_is_resumable_and_records_everything(tmp_path: Path) -> None:
    spec = parse_spec(TINY)
    store = run_experiment(spec, tmp_path, limit=3, progress=None)
    assert len(store.read()) == 3
    store = run_experiment(spec, tmp_path, progress=None)  # resumes: only the 5 missing runs
    records = store.read()
    assert len(records) == 8 and len({r["run_id"] for r in records}) == 8
    assert {r["outcome"] for r in records} <= {"ok", "budget_exhausted"}
    env = json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))
    assert "python" in env and "numpy" in env
    # a truncated last line (crash while writing) is ignored and that run is redone
    with open(store.path, "a", encoding="utf-8") as fh:
        fh.write('{"run_id": "trunc')
    assert len(RecordStore(tmp_path).read()) == 8
    a = [r for r in records if r["params"]["planner"] == "a"]
    assert all(r["metrics"]["step_gap"] == 1.0 for r in a)  # A* is step-optimal (λ = 0)


def test_execute_records_errors_and_timeouts() -> None:
    bad = parse_spec(
        {
            **TINY,
            "factors": {"family": ["nope"], "size": [8], "map": [0], "pair": [0], "planner": ["a"]},
        }
    )
    rec = execute(bad, bad.runs()[0])
    assert rec.outcome == "error" and rec.error is not None and "KeyError" in rec.error
    slow = parse_spec({**TINY, "experiment": {**TINY["experiment"], "timeout_s": 1e-9}})
    rec2 = execute(slow, next(r for r in slow.runs() if r.params["planner"] == "r"))
    assert rec2.outcome == "timeout"
    assert json.loads(RunRecord("x", "e", {}, "ok", {"v": math.inf}).to_json())["metrics"] == {
        "v": "inf"
    }
    assert _plain(np.array([1, 2])) == [1, 2] and _plain(float("nan")) is None


def test_instances_are_deterministic_and_paired() -> None:
    g1, g2 = make_map("d20", 16, 1, 0), make_map("d20", 16, 1, 0)
    assert g1 == g2 and g1 != make_map("d20", 16, 1, 1)
    assert make_pair(g1, (1, 2, 3)) == make_pair(g2, (1, 2, 3))
    s0 = e2_scenario(20_000, 0, "none", 4)
    s16 = e2_scenario(20_000, 16, "low", 4)
    assert s0.grid == s16.grid and s0.robots == s16.robots and s0.tasks == s16.tasks
    assert len(s16.obstacles) == 16 and len(s16.map_events) == 12 and not s0.map_events
    assert s16.script is not None and s16.script.assignments[1] == (1, 5, 9)
    with pytest.raises(ValueError):
        e2_scenario(1, 0, "storm", 1)
    assert euclid_lower_bound((0, 0), (0, 0)) == 0.0
    assert euclid_lower_bound((0, 0), (3, 0)) == 2.5
    assert euclid_lower_bound((0, 0), (3, 4)) == pytest.approx(math.hypot(2.5, 3.5))


def test_statistics() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 1.0, size=200)
    est = bootstrap_ci(x, n_boot=2000)
    assert est.low < 10.0 < est.high and est.n == 200
    med = bootstrap_ci(x, "median", n_boot=2000)
    assert med.low <= med.value <= med.high
    assert math.isnan(bootstrap_ci([]).value)
    d = paired_bootstrap(x + 1.0, x, n_boot=500)
    assert d.value == pytest.approx(1.0) and d.low == pytest.approx(1.0)
    assert rank_biserial([2, 3, 4], [1, 1, 1]) == 1.0
    assert rank_biserial([1, 1], [1, 1]) == 0.0
    assert rank_biserial([0, 3], [1, 1]) == pytest.approx((2 - 1) / 3)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    n = np.array([10, 20, 40, 80, 160], dtype=float)
    fit = loglog_fit(n, 3 * n**1.5, n_boot=200)
    assert fit.exponent == pytest.approx(1.5) and fit.low == pytest.approx(1.5)
    assert math.isnan(loglog_fit([1.0], [1.0]).exponent)


def test_episode_timeout_is_recorded_not_dropped() -> None:
    """An E2 episode over its wall-clock cap stops and is recorded as ``timeout`` with the metrics
    gathered so far (05 §B.1)."""
    data = {
        "experiment": {"name": "e2_tiny", "kind": "replanning", "timeout_s": 1e-9},
        "factors": {
            "planner": ["a"],
            "obstacles": [2],
            "events": ["none"],
            "team": [1],
            "seed": [20_000],
        },
        "fixed": {"size": 12, "max_steps": 200, "tasks_per_robot": 2},
        "planners": [{"label": "a", "name": "astar"}],
    }
    spec = parse_spec(data)
    rec = execute(spec, spec.runs()[0])
    assert rec.outcome == "timeout" and rec.error is None
    assert rec.metrics["steps"] == 1  # stopped after the first step
    assert rec.metrics["terminated"] is False
    # the same episode without a cap runs to completion
    ok = execute(
        parse_spec({**data, "experiment": {**data["experiment"], "timeout_s": 600}}), spec.runs()[0]
    )
    assert ok.outcome == "done" and ok.metrics["steps"] > 1


def test_polylog_fit_and_residuals() -> None:
    n = np.array([64.0, 128, 256, 512, 1024, 4096, 10_000])
    y = 3.0 * np.log(n) ** 2
    pl, pw = polylog_fit(n, y, n_boot=200), loglog_fit(n, y, n_boot=200)
    assert pl.exponent == pytest.approx(2.0) and pl.rss == pytest.approx(0.0, abs=1e-20)
    assert pw.rss > pl.rss  # a power law fits (ln n)^2 data worse over this range
    assert 0 < pw.exponent < 1
    assert math.isnan(polylog_fit([1.0, 0.5], [1.0, 2.0]).exponent)  # n ≤ 1 dropped
