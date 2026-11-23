"""Tests of the L1 follow-up tools: ``selection_block``, ``null_study``, ``rule_compare``."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "selection"))
import null_study
import rule_compare
import scbc
import selection_block


def table(k: int = 5, s: int = 30, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rel = rng.integers(40, 60, size=(1, s)).repeat(k, 0).astype(float)
    mc = np.linspace(1, 9, k)[:, None]
    und = rng.negative_binomial(0.3, 0.3 / (0.3 + np.linspace(0.2, 1.0, k)[:, None]), (k, s))
    return {
        "steps": np.arange(1, k + 1) * 50_000.0,
        "released": rel,
        "delivered": np.maximum(rel - und, 0),
        "conflicts": rng.negative_binomial(0.4, 0.4 / (0.4 + mc), (k, s)).astype(float),
        "collisions": np.zeros((k, s)),
        "ret": rng.normal(size=(k, s)),
    }  # fmt: skip


# ----------------------------------------------------------------------------- selection block
def test_selection_block_is_disjoint_from_every_other_block() -> None:
    s = set(selection_block.SELECTION_BLOCK)
    assert len(s) == 80
    for r in selection_block.OTHER_BLOCKS.values():
        assert not s & set(r)


def test_guard_accepts_only_block_s() -> None:
    assert selection_block.guard([13_000, 13_079], "S") == [13_000, 13_079]
    for bad in (11_000, 12_000, 20_000, 10_000, 13_080):
        with pytest.raises(ValueError):
            selection_block.guard([bad], "S")


def test_scenario_fingerprint_ignores_the_name_and_separates_instances() -> None:
    from dataclasses import replace

    from robocoord.marl.scenarios import LEVELS

    a = LEVELS["static"].sample(13_000)
    assert selection_block.fingerprint(a) == selection_block.fingerprint(replace(a, name="x"))
    assert selection_block.fingerprint(a) != selection_block.fingerprint(
        LEVELS["static"].sample(13_001)
    )


# ----------------------------------------------------------------------------- null study
def test_plugin_null_equalises_means_and_keeps_shapes() -> None:
    t = table()
    comp, conf = null_study.plugin_null(t)  # type: ignore[misc]
    assert np.allclose(comp.mean(1), comp.mean(1)[0]) and np.allclose(conf.mean(1), conf.mean(1)[0])
    raw = t["conflicts"]
    cv = lambda m: m.std(1) / m.mean(1)  # noqa: E731
    # multiplicative rescaling keeps the coefficient of variation
    assert np.allclose(cv(-conf), cv(raw))


def test_studentised_permutation_matches_a_loop_implementation() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 6)) * np.array([[0.5], [1.0], [3.0]])
    old = null_study.N_RESAMPLE
    null_study.N_RESAMPLE = 200
    try:
        band = null_study.bands_v3s(x, 4)[0]
    finally:
        null_study.N_RESAMPLE = old
    idx = scbc.permutation_indices(6, 3, 200, 4)
    tmax = []
    for p in range(200):
        xp = np.array([[x[idx[p, s, j], s] for s in range(6)] for j in range(3)])
        best = 0.0
        for j, k in itertools.combinations(range(3), 2):
            d = xp[j] - xp[k]
            best = max(best, abs(d.mean()) / (d.std(ddof=1) / math.sqrt(6)))
        tmax.append(best)
    assert band.q == pytest.approx(float(np.quantile(tmax, 0.95, method="higher")))
    d01 = x[0] - x[1]
    assert band.se[0, 1] == pytest.approx(d01.std(ddof=1) / math.sqrt(6))


def test_null_study_replicate_is_deterministic() -> None:
    t = table(k=4, s=40)
    w = {"wid": "t", "kind": "plugin", "reps": 1, "tables": null_study.plugin_null(t), "s": 20}
    old = null_study.N_RESAMPLE
    null_study.N_RESAMPLE = 200
    try:
        a = null_study.one(w, 3, null_study.METHODS)
        b = null_study.one(w, 3, null_study.METHODS)
    finally:
        null_study.N_RESAMPLE = old
    assert a == b
    assert all(a[f"{m}_a025_stage1"] <= a[f"{m}_a05_stage1"] for m in null_study.METHODS)


def test_wilson_interval_and_binomial_test() -> None:
    r = null_study.rate([True] * 5 + [False] * 95)
    assert r["rate"] == 0.05 and r["lo"] < 0.05 < r["hi"] and r["p_vs_005"] == pytest.approx(1.0)
    assert null_study.rate([True] * 20 + [False] * 80)["p_vs_005"] < 1e-6


# ----------------------------------------------------------------------------- rule comparison
def test_rule_compare_picks_cover_every_rule_and_stage() -> None:
    old = rule_compare.N_BOOT
    rule_compare.N_BOOT = 200
    try:
        p = rule_compare.picks(table(k=4, s=20), 0)
    finally:
        rule_compare.N_BOOT = old
    for rule in rule_compare.RULES[1:]:
        assert f"none_{rule}" in p
        for st in ("v2", "v3", "v4"):
            for tag in rule_compare.ALPHAS:
                assert 0 <= p[f"{st}_{tag}_{rule}"] < 4
    assert p["none_conflicts"] == int(np.argmin(table(k=4, s=20)["conflicts"].mean(1)))


def test_score_measures_the_pick_on_the_evaluation_table() -> None:
    t = table(k=4, s=30)
    s = rule_compare.score(t, {"x": 2})
    comp = t["delivered"].sum(1) / t["released"].sum(1)
    assert s["x"]["completion"] == pytest.approx(comp[2])
    assert s["x"]["shortfall"] == pytest.approx(comp.max() - comp[2])
    assert s["x"]["meets_099"] == (comp[2] >= 0.99)


def test_one_se_rule_picks_fewest_conflicts_within_one_paired_standard_error() -> None:
    s = 10
    rel = np.full((3, s), 50.0)
    d = rel - 1
    d[1, 0], d[1, 1] = 47, 50  # 1 task behind the best overall, noisy: inside one paired SE
    d[2, :] -= 3  # 30 tasks behind, on every seed: outside the band (SE of a constant diff = 0)
    t = {"steps": np.array([1.0, 2.0, 3.0]), "released": rel, "delivered": d,
         "conflicts": np.array([[9.0] * s, [1.0] * s, [0.0] * s]), "collisions": np.zeros((3, s)),
         "ret": np.zeros((3, s))}  # fmt: skip
    x = d / 50.0
    se = (x[0] - x[1]).std(ddof=1) / np.sqrt(s)
    gap = (d[0].sum() - d[1].sum()) / rel[0].sum()
    assert gap <= se
    assert rule_compare.one_se_pick(t, 1.0) == 1
    assert rule_compare.one_se_pick(t, 0.0) == 0
