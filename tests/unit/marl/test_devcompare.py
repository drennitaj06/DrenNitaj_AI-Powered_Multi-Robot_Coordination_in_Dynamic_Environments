"""Development-comparison design and statistics (block B; not E5)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

from robocoord.marl import devcompare
from robocoord.marl.devcompare import analyse, report, tasks_for, two_level_bootstrap
from robocoord.marl.scenarios import BLOCK_B, RESERVED_TEST


def test_block_b_design() -> None:
    tasks = tasks_for("baseline", "baseline")
    astar = [t for t in tasks if t.planner == "astar"]
    rrt = [t for t in tasks if t.planner == "rrtstar"]
    assert len(astar) == 240 and len(rrt) == 60
    assert all(t.block == "B" and t.seed in BLOCK_B for t in tasks)
    assert not {t.seed for t in tasks} & set(RESERVED_TEST)
    assert {t.seed for t in rrt} == set(range(10_020, 10_040))


def fake(label: str, planner: str, seed: int, comp: float) -> dict:
    return {"label": label, "planner": planner, "level": "full", "seed": seed,
            "completion": comp, "base_return": comp, "conflicts_per_100": 1 - comp,
            "released": 10, "delivered": int(comp * 10), "terminated": False,
            "physical_collisions": 0}  # fmt: skip


def test_analysis_pairs_by_instance_and_applies_holm() -> None:
    recs = []
    for s in range(10_020, 10_030):
        recs += [fake("baseline", "astar", s, 0.5), fake("reference", "astar", s, 0.4)]
        recs += [fake("madql-seed1", "astar", s, 0.6), fake("madql-seed2", "astar", s, 0.5)]
    a = analyse(recs)
    rows = [r for r in a["paired"] if r["metric"] == "completion"]
    by = {(r["policy"], r["comparator"]): r for r in rows}
    assert by[("madql-seed1", "baseline")]["mean_diff"] == pytest.approx(0.1)
    assert by[("madql-seed2", "baseline")]["mean_diff"] == pytest.approx(0.0)
    assert by[("madql-seed2", "baseline")]["wilcoxon_p"] == 1.0
    assert all(r["holm_p"] >= r["wilcoxon_p"] for r in rows)
    pooled = [
        r for r in a["pooled"] if r["metric"] == "completion" and r["comparator"] == "reference"
    ]
    assert pooled[0]["mean"] == pytest.approx(0.15)
    assert "not E5" in report(a)


def test_main_prints_the_summary_on_a_cp1252_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: the first block-B run wrote every result file, then exited 1 with
    # UnicodeEncodeError while printing "−" to a redirected Windows console.
    def fake_run(policies: object, out: Path, workers: int) -> None:
        (out / "summary.md").write_text("MADQL − comparator\n", "utf-8")

    monkeypatch.setattr(devcompare, "run", fake_run)
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert devcompare.main(["--policy", "p.pt", "--out", str(tmp_path)]) == 0
    sys.stdout.flush()
    assert raw.getvalue().decode("cp1252").startswith("MADQL ? comparator")


def test_two_level_bootstrap_brackets_the_mean() -> None:
    diffs = [np.full(20, 1.0), np.full(20, 3.0)]
    r = two_level_bootstrap(diffs, n_boot=2000)
    assert r["mean"] == pytest.approx(2.0) and r["low"] <= 2.0 <= r["high"]
    assert r["low"] >= 1.0 and r["high"] <= 3.0


# ---------------------------------------------------------------------------------- Issue #6
def test_issue6_comparators_learned_prefixes_and_derived_metrics() -> None:
    from robocoord.marl.devcompare import METRICS_ISSUE6, metric_value

    recs = []
    for s in range(10_020, 10_030):
        recs += [fake("baseline", "astar", s, 0.9), fake("baseline-noresolver", "astar", s, 0.7)]
        recs += [fake("reference", "astar", s, 0.6), fake("mappo-seed1", "astar", s, 0.8)]
        recs += [fake("ippo-seed1", "astar", s, 0.0 if s == 10_020 else 0.5)]
    for r in recs:
        r["conflicts_total"] = 3
    a = analyse(recs, metrics=METRICS_ISSUE6)
    pairs = {(r["metric"], r["policy"], r["comparator"]) for r in a["paired"]}
    assert ("completion", "mappo-seed1", "baseline-noresolver") in pairs
    assert ("conflicts_total", "ippo-seed1", "reference") in pairs
    per_del = [r for r in a["paired"] if r["metric"] == "conflicts_per_delivered"
               and r["policy"] == "ippo-seed1"]  # fmt: skip
    # the instance with nothing delivered is excluded (undefined ratio) and counted
    assert all(r["n"] == 9 and r["excluded_undefined"] == 1 for r in per_del)
    # Holm runs within each metric family, which now includes the matched baseline
    fam = [r for r in a["paired"] if r["metric"] == "completion"]
    assert len(fam) == 2 * 3 and all(r["holm_p"] >= r["wilcoxon_p"] for r in fam)
    # Issue #5 records have no conflicts_per_delivered: it is derived
    assert metric_value({"conflicts_total": 6, "delivered": 3}, "conflicts_per_delivered") == 2.0
    assert metric_value({"conflicts_total": 6, "delivered": 0}, "conflicts_per_delivered") is None


def test_algorithm_bootstrap_resamples_seeds_and_shares_instances() -> None:
    from robocoord.marl.devcompare import algorithm_bootstrap

    x = [np.arange(30, dtype=float) + 2.0, np.arange(30, dtype=float) + 4.0]
    y = [np.arange(30, dtype=float)]
    r = algorithm_bootstrap(x, y, n_boot=2000)
    assert r["mean"] == pytest.approx(3.0)
    # instance resampling is shared, so the large between-instance spread cancels: the CI only
    # reflects the between-seed spread of X (2 or 4)
    assert 2.0 - 1e-9 <= r["low"] <= r["high"] <= 4.0 + 1e-9
    assert r["high"] - r["low"] > 1.0  # the seed spread is reflected, not hidden
    with pytest.raises(ValueError):
        algorithm_bootstrap(x, [np.arange(29, dtype=float)])


def test_load_records_reads_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from robocoord.marl.devcompare import load_records

    p = tmp_path / "records.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n")
    assert load_records(p) == [{"a": 1}, {"a": 2}]
