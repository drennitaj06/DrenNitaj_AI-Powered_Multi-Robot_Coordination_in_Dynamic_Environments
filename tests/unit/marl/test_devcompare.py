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
