"""Issue #6 block-B experiment orchestration and analysis (``marl.study``): designs, the
determinism check, replicate averaging, algorithm-level and ablation statistics, run-once rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robocoord.marl import study
from robocoord.marl.evaluation import EvalTask
from robocoord.marl.scenarios import BLOCK_B, RESERVED_TEST


def rec(label: str, seed: int, comp: float, **kw: Any) -> dict[str, Any]:
    r = {"label": label, "planner": "astar", "level": "full", "seed": seed, "completion": comp,
         "base_return": comp, "conflicts_total": 2, "conflicts_per_100": 0.5, "released": 10,
         "delivered": int(comp * 10), "terminated": False, "physical_collisions": 0,
         "wall_s": 1.0}  # fmt: skip
    r.update(kw)
    return r


def write_sidecar(tmp: Path, name: str, algorithm: str, seed: int, variant: str) -> Path:
    path = tmp / f"{name}.pt"
    path.write_bytes(b"x")
    path.with_suffix(".json").write_text(json.dumps({
        "algorithm": algorithm, "variant": variant, "seeds": {"run_seed": seed},
        "config": {"name": name.split("_seed")[0]}, "sha256": "s", "config_hash": "h",
        "env_steps": 1, "validation": {},
    }))  # fmt: skip
    return path


@pytest.mark.parametrize(
    ("experiment", "per_policy"), [("ppo", 300), ("ablation", 240), ("shield", 300),
                                   ("stochastic", 960)],
)  # fmt: skip
def test_experiment_designs(
    experiment: str, per_policy: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[EvalTask] = []
    monkeypatch.setattr(study, "run_tasks", lambda tasks, workers: seen.extend(tasks) or [])
    monkeypatch.setattr(study, "HISTORICAL", tmp_path / "none.jsonl")
    pols = [write_sidecar(tmp_path, "mappo_seed1", "ppo", 1, "mappo"),
            write_sidecar(tmp_path, "madql_seed2", "madql", 2, "madql")]  # fmt: skip
    study.run_policies(tmp_path / experiment, pols, 1, experiment=experiment)
    assert len(seen) == 2 * per_policy
    assert {t.label for t in seen} == {"mappo-seed1", "madql-seed2"}
    assert all(t.block == "B" and t.seed in BLOCK_B for t in seen)
    assert not {t.seed for t in seen} & set(RESERVED_TEST)
    if experiment == "shield":
        cells = {(t.allocator, t.shield) for t in seen}
        assert cells == {("auction", "predictive"), ("sequential", "off"),
                         ("sequential", "predictive")}  # fmt: skip
        assert {t.level for t in seen} == {"full"}
    if experiment == "stochastic":
        assert {t.replicate for t in seen if t.mode == "stochastic"} == {1, 2, 3}
        assert sum(t.mode == "greedy" for t in seen) == 2 * 240
    manifest = json.loads((tmp_path / experiment / "manifest.json").read_text())
    assert set(manifest["policies"]) == {"mappo-seed1", "madql-seed2"}
    with pytest.raises(FileExistsError):
        study.run_policies(tmp_path / experiment, pols, 1, experiment=experiment)


def test_matched_baseline_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[EvalTask] = []
    monkeypatch.setattr(study, "run_tasks", lambda tasks, workers: seen.extend(tasks) or [])
    study.run_matched_baseline(tmp_path / "m", 1)
    assert len(seen) == 300 and {t.controller for t in seen} == {"baseline-noresolver"}
    assert sum(t.planner == "rrtstar" for t in seen) == 60


def test_determinism_check_reports_field_mismatches_but_ignores_timing() -> None:
    hist = [rec("madql-seed1", s, 0.5) for s in (10_020, 10_021)]
    fresh = [rec("madql-seed1", 10_020, 0.5, wall_s=9.0, mode="greedy", shaped_return=1.0),
             rec("madql-seed1", 10_021, 0.6, mode="greedy"),
             rec("madql-seed1", 10_021, 0.9, mode="stochastic")]  # fmt: skip
    out = study.determinism_check(fresh, hist)
    assert out["compared"] == 2 and out["mismatches"] == 1
    assert set(out["details"][0]["fields"]) == {"completion", "base_return", "delivered"}


def test_stochastic_replicates_are_averaged_per_instance() -> None:
    recs = [rec("ippo-seed1", 10_020, 0.4, mode="greedy"),
            rec("ippo-seed1", 10_020, 0.6, mode="stochastic", replicate=1),
            rec("ippo-seed1", 10_020, 0.8, mode="stochastic", replicate=2)]  # fmt: skip
    g = study.stochastic_groups(recs)
    key = ("astar", "full", 10_020)
    assert g["ippo-seed1/greedy"][key]["completion"] == 0.4
    assert g["ippo-seed1/stochastic"][key]["completion"] == pytest.approx(0.7)
    assert g["ippo-seed1/stochastic"][key]["replicates"] == 2


def test_algorithm_level_and_summaries() -> None:
    recs = []
    for s in range(10_020, 10_040):
        recs += [rec("mappo-seed1", s, 0.6), rec("mappo-seed2", s, 0.8), rec("ippo-seed1", s, 0.5),
                 rec("madql-seed1", s, 0.4), rec("baseline-noresolver", s, 0.9)]  # fmt: skip
    rows = study.algorithm_level(recs, metrics=("completion",))
    by = {(r["a"], r["b"]): r for r in rows}
    assert by[("mappo", "ippo")]["mean"] == pytest.approx(0.2)
    assert by[("mappo", "ippo")]["per_seed_a"] == pytest.approx([0.6, 0.8])
    assert by[("madql", "baseline-noresolver")]["mean"] == pytest.approx(-0.5)
    assert "wilcoxon_p" not in by[("mappo", "ippo")]  # no p-values from three seeds
    s = study.summary_block(recs[:5])
    assert s["shaped_return"] is None and s["conflicts_per_delivered_pooled"] == pytest.approx(
        10 / (6 + 8 + 5 + 4 + 9)
    )


def test_ablation_effects_are_balanced_on_seeds_one_and_two() -> None:
    groups: dict[str, dict] = {}
    for lab, comp in (("mappo_zeta0-seed1", 0.2), ("mappo_zeta0-seed2", 0.4),
                      ("mappo-seed1", 0.6), ("mappo-seed2", 0.6)):  # fmt: skip
        groups[lab] = {("astar", "full", s): rec(lab, s, comp) for s in range(10_020, 10_030)}
    rows = study.ablation_effects(groups, metrics=("completion",))
    (row,) = rows
    assert row["cell"] == "mappo_zeta0" and row["mean"] == pytest.approx(-0.3)
    assert row["per_seed_cell"] == pytest.approx([0.2, 0.4]) and row["instances"] == 10


def test_analyse_all_combines_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hist = tmp_path / "hist.jsonl"
    rows = [rec("madql-seed1", s, 0.4) for s in range(10_020, 10_030)]
    rows += [rec("baseline", s, 1.0) for s in range(10_020, 10_030)]
    hist.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(study, "HISTORICAL", hist)
    for part, recs in (
        ("baseline_matched", [rec("baseline-noresolver", s, 0.9) for s in range(10_020, 10_030)]),
        ("ppo", [rec("mappo-seed1", s, 0.7) for s in range(10_020, 10_030)]),
    ):
        (tmp_path / part).mkdir()
        (tmp_path / part / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = study.analyse_all(tmp_path)
    assert out["historical"]["records"] == str(hist)
    comps = {(r["policy"], r["comparator"]) for r in out["primary"]["per_policy"]["paired"]}
    assert ("mappo-seed1", "baseline-noresolver") in comps and ("madql-seed1", "baseline") in comps
    effect = [r for r in out["resolver_effect"] if r["metric"] == "completion"]
    assert effect[0]["mean_diff"] == pytest.approx(0.1)
    assert hist.read_text().count("\n") == 20  # historical records are only read
    md = study.report(out)
    assert "## Primary comparison" in md and "mappo-seed1 − baseline-noresolver" in md
    assert "## Deadlock resolver effect" in md and "## Reward ablation" not in md
    assert study.main(["analyse", "--root", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").read_text("utf-8") == md


def test_shield_analysis_labels_historical_records_as_auction_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hist = tmp_path / "hist.jsonl"
    rows = [rec("madql-seed1", s, 0.4) for s in range(10_020, 10_030)]  # no allocator / shield
    hist.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(study, "HISTORICAL", hist)
    shield = [rec("madql-seed1", s, 0.5, allocator=a, shield=sh) for s in range(10_020, 10_030)
              for a, sh in (("auction", "predictive"), ("sequential", "off"),
                            ("sequential", "predictive"))]  # fmt: skip
    (tmp_path / "shield").mkdir()
    (tmp_path / "shield" / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in shield)
    )
    out = study.analyse_all(tmp_path)["shield"]
    assert "madql-seed1/auction/off/astar" in out["summaries"]
    assert not any("//" in k for k in out["summaries"])
    auction = [r for r in out["paired"]
               if r["comparison"] == "madql-seed1 auction: predictive − off"]  # fmt: skip
    assert auction and auction[0]["mean_diff"] == pytest.approx(0.1)
    assert hist.read_text().count("\n") == 10  # historical records are only read
