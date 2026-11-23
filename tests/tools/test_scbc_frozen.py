"""Tests of the frozen SCB-C v5 specification: ``tools/selection/scbc_frozen.py`` (proposal §23)."""

from __future__ import annotations

import inspect
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "selection"))
import scbc
import scbc_frozen as fz

NB = 2000  # resamples in unit tests (the frozen value, 10 000, is checked separately)


def table(delivered: list[list[int]], conflicts: list[int], steps: list[int],
          released: int = 10, collisions: list[int] | None = None) -> fz.Table:  # fmt: skip
    """A table with one level per seed; ``conflicts`` are per-checkpoint totals (put on seed 0)."""
    d = np.array(delivered, dtype=float)
    k, s = d.shape
    c = np.zeros((k, s))
    c[:, 0] = conflicts
    x = np.zeros((k, s))
    if collisions:
        x[:, 0] = collisions
    names = tuple(f"step_{v:08d}.pt" for v in steps)
    return fz.Table(names, tuple(steps), d, np.full((k, s), float(released)), c, x, {}, 1)


def rec(ck: str, seed: int, level: str, **kw: Any) -> dict[str, Any]:
    r = {"checkpoint": ck, "seed": seed, "level": level, "delivered": 9, "released": 10,
         "conflicts_total": 2, "physical_collisions": 0, "controller": "ppo", "planner": "astar",
         "mode": "greedy", "shield": "off", "block": "T"}  # fmt: skip
    return r | kw


CKS = ["step_00100000.pt", "step_00050000.pt"]
SEEDS = [1, 2, 3, 4, 5]
LEVELS = ("a", "b")


def records(**override: Any) -> list[dict[str, Any]]:
    return [rec(c, s, lv, **override) for c in CKS for s in SEEDS for lv in LEVELS]


def build(recs: list[dict[str, Any]]) -> fz.Table:
    return fz.build_table(recs, CKS, seeds=SEEDS, levels=LEVELS, block_of=None)


# ----------------------------------------------------------------------------- frozen constants
def test_frozen_configuration() -> None:
    assert tuple(range(10_000, 10_020)) + tuple(range(13_000, 13_080)) == fz.SEEDS
    assert len(fz.SEEDS) == 100 and len(set(fz.SEEDS)) == 100
    assert fz.LEVELS == ("static", "obstacles", "full")
    assert (fz.ALPHA, fz.N_BOOT, fz.BOOT_SEED) == (0.05, 10_000, 0)
    assert fz.EXPECTED == {
        "controller": "ppo",
        "planner": "astar",
        "mode": "greedy",
        "shield": "off",
    }
    forbidden = set(range(11_000, 11_120)) | set(range(12_000, 12_200)) | set(range(20_000, 21_000))
    assert not forbidden & set(fz.SEEDS)  # blocks C, D and the test seeds are never selection data
    assert fz.BLOCK_OF[10_000] == "A" and fz.BLOCK_OF[13_079] == "S"


def test_no_completion_tolerance_or_floor_and_no_conflict_test() -> None:
    sig = inspect.signature(fz.select)
    assert set(sig.parameters) == {"t", "alpha", "n_boot", "seed"}
    src = inspect.getsource(fz.select)
    assert "0.005" not in src and "0.99" not in src and "floor" not in src
    assert "alpha_conflicts" not in src and "conflicts_band" not in src


# ----------------------------------------------------------------------------- loader
def test_complete_records_are_summed_over_levels_and_sorted_by_step() -> None:
    t = build(records())
    assert t.checkpoints == ("step_00050000.pt", "step_00100000.pt") and t.steps == (
        50_000,
        100_000,
    )
    assert not t.problems
    assert (t.delivered == 18).all() and (t.released == 20).all() and (t.conflicts == 4).all()


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda rs: rs.pop(3), "incomplete"),  # a missing episode
        (lambda rs: rs[3].update(delivered=None), "incomplete"),  # a null field
        (lambda rs: rs[3].pop("conflicts_total"), "incomplete"),  # a missing field
        (lambda rs: rs.append(dict(rs[3])), "invalid"),  # a duplicate episode
        (lambda rs: rs[3].update(planner="rrtstar"), "invalid"),
        (lambda rs: rs[3].update(mode="stochastic"), "invalid"),
        (lambda rs: rs[3].update(shield="on"), "invalid"),
        (lambda rs: rs[3].update(seed=99), "invalid"),  # a seed outside the selection data
        (lambda rs: rs[3].update(level="zz"), "invalid"),
        (lambda rs: rs[3].update(delivered=11), "invalid"),  # delivered > released
        (lambda rs: rs[3].update(delivered=-1), "invalid"),
        (lambda rs: rs[3].update(delivered=8.5), "invalid"),
        (lambda rs: rs[3].update(conflicts_total=float("nan")), "invalid"),
        (lambda rs: rs[3].update(delivered=True), "invalid"),
    ],
)
def test_invalid_or_missing_evaluations_exclude_the_checkpoint(mutate: Any, reason: str) -> None:
    rs = records()
    mutate(rs)  # rs[3] belongs to step_00100000 (index 1 after sorting)
    t = build(rs)
    assert t.problems == {1: reason}
    f = fz.select(t, n_boot=NB)
    assert f.excluded == {1: reason} and f.selected == 0  # never imputed, never selected


def test_an_invalid_record_takes_precedence_over_the_missing_cell_it_leaves() -> None:
    rs = records()
    rs[3]["planner"] = "rrtstar"
    rs.pop(4)
    assert build(rs).problems == {1: "invalid"}


def test_block_field_must_match_the_seed() -> None:
    rs = [rec(c, s, lv, block=fz.BLOCK_OF[s]) for c in CKS for s in fz.SEEDS for lv in fz.LEVELS]
    assert not fz.build_table(rs, CKS).problems
    rs[5]["block"] = "C"
    assert fz.build_table(rs, CKS).problems == {1: "invalid"}


def test_records_of_other_checkpoints_are_ignored_and_record_order_does_not_matter() -> None:
    rs = [*records(), rec("step_00999999.pt", 1, "a", planner="rrtstar")]
    a = fz.select(build(rs), n_boot=NB)
    random.Random(4).shuffle(rs)
    b = fz.select(build(rs), n_boot=NB)
    assert a.selected == b.selected and a.band == b.band and a.q == b.q and not a.excluded


def test_released_zero_and_collisions_exclude() -> None:
    t = table([[10] * 6, [9] * 6, [9] * 6], [1, 0, 0], [1, 2, 3], collisions=[0, 1, 0])
    t.released[2] = 0
    t.delivered[2] = 0
    f = fz.select(t, n_boot=NB)
    assert f.excluded == {1: "collision", 2: "invalid"} and f.selected == 0


def test_no_valid_checkpoint_gives_no_selection() -> None:
    t = table([[10] * 6, [9] * 6], [0, 0], [1, 2], collisions=[1, 1])
    f = fz.select(t, n_boot=NB)
    assert f.selected is None and f.selected_step is None and f.band == () and f.q is None


# ----------------------------------------------------------------------------- procedure
DESIGN = [
    [10, 10, 10, 10, 10, 6],  # A: 56 / 60
    [9, 9, 9, 10, 10, 10],  # B: 57 / 60  (highest completion: the reference j*)
    [8, 8, 8, 8, 8, 8],  # C: 48 / 60
]


def test_hand_computed_band_and_pick() -> None:
    f = fz.select(table(DESIGN, [5, 9, 0], [100, 200, 300]), n_boot=NB)
    assert f.reference == 1
    xa, xb, xc = (np.array(r) / 10 for r in DESIGN)
    se_a = np.std(xb - xa, ddof=1) / math.sqrt(6)  # 0.0792
    se_c = np.std(xb - xc, ddof=1) / math.sqrt(6)  # 0.0224
    assert f.se_to_reference[0] == pytest.approx(se_a)
    assert se_a >= 1 / 60 and se_c < 9 / 60
    assert f.band == (0, 1)  # C has the fewest conflicts but is outside the band
    assert f.selected == 0 and f.selected_step == 100  # fewest conflicts in the band
    assert f.completion[1] == pytest.approx(57 / 60) and f.conflicts_per_episode[
        0
    ] == pytest.approx(5 / 6)


def test_conflict_ties_go_to_the_latest_step_not_to_higher_completion() -> None:
    f = fz.select(table(DESIGN, [5, 5, 0], [300, 200, 100]), n_boot=NB)
    assert f.band == (0, 1) and f.selected == 0 and f.selected_step == 300  # A: lower completion


def test_reference_ties_go_to_the_latest_step() -> None:
    d = [[10, 10, 9, 10, 9, 10], [10, 10, 9, 10, 9, 10], [9, 9, 9, 9, 9, 9]]
    f = fz.select(table(d, [3, 3, 1], [100, 200, 300]), n_boot=NB)
    assert f.reference == 1  # exact tie in completion
    assert f.selected == 1  # conflict tie between the duplicates: the latest


def test_steps_must_be_unique() -> None:
    with pytest.raises(ValueError):
        fz.select(table(DESIGN, [1, 2, 3], [100, 100, 300]), n_boot=NB)


def test_completion_comparison_is_exact() -> None:
    # 1/3 vs 2/6: equal as rationals; floating sums could differ in the last bit
    t = table([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]], [4, 4], [1, 2], released=1)
    f = fz.select(t, n_boot=NB)
    assert f.reference == 1 and f.selected == 1


def test_stage1_matches_the_v2_procedure_of_the_study() -> None:
    rng = np.random.default_rng(8)
    rel = rng.integers(40, 60, size=(1, 30)).repeat(6, 0).astype(float)
    d = np.minimum(rel, rng.binomial(rel.astype(int), np.linspace(0.93, 0.99, 6)[:, None])).astype(
        float
    )
    d[0] -= 3  # one clearly worse checkpoint
    c = rng.poisson(5, (6, 30)).astype(float)
    steps = tuple(range(50_000, 350_000, 50_000))
    t = fz.Table(tuple(f"step_{s:08d}.pt" for s in steps), steps, d, rel, c, np.zeros((6, 30)))
    f = fz.select(t, n_boot=NB)
    ref = scbc.select(np.array(steps), d, rel, c, n_boot=NB, seed=0, studentize="t")
    assert f.survivors == ref.eligible_completion
    assert f.q == pytest.approx(ref.completion.q)  # type: ignore[union-attr]
    assert 0 not in f.survivors


def test_selection_is_deterministic() -> None:
    t = table(DESIGN, [5, 9, 0], [100, 200, 300])
    assert fz.select(t, n_boot=NB) == fz.select(t, n_boot=NB)


@settings(max_examples=40, deadline=None)
@given(k=st.integers(1, 6), s=st.integers(5, 12), seed=st.integers(0, 10_000))
def test_pick_is_the_fewest_conflicts_in_a_band_of_survivors(k: int, s: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rel = rng.integers(5, 15, size=(1, s)).repeat(k, 0).astype(float)
    d = np.minimum(rel, rng.binomial(rel.astype(int), rng.uniform(0.8, 1.0, (k, 1)))).astype(float)
    c = rng.integers(0, 4, (k, s)).astype(float)
    steps = tuple(int(v) for v in rng.permutation(np.arange(1, k + 1) * 1000))
    t = fz.Table(tuple(f"step_{v:08d}.pt" for v in steps), steps, d, rel, c, np.zeros((k, s)))
    f = fz.select(t, n_boot=300)
    assert f.selected is not None and f.reference is not None
    assert f.selected in f.band and set(f.band) <= set(f.survivors) and f.reference in f.band
    comp = d.sum(1) / r_sum(rel)
    assert comp[f.reference] == max(comp[i] for i in f.survivors)
    tot = c.sum(1)
    assert tot[f.selected] == min(tot[i] for i in f.band)
    tied = [i for i in f.band if tot[i] == tot[f.selected]]
    assert steps[f.selected] == max(steps[i] for i in tied)


def r_sum(rel: np.ndarray) -> np.ndarray:
    out: np.ndarray = rel.sum(1)
    return out


# ----------------------------------------------------------------------------- frozen data
S_RECORDS = Path("results/dev6a/l1_selection/selection_block/records.jsonl")


@pytest.mark.skipif(
    not S_RECORDS.exists(), reason="selection-block records not present (results/ is not tracked)"
)
def test_frozen_selections_on_blocks_a_and_s() -> None:
    expected = {"mappo_seed1": 300_000, "mappo_seed2": 500_000, "mappo_seed3": 500_000,
                "ippo_seed1": 500_000, "ippo_seed2": 250_000, "ippo_seed3": 500_000}  # fmt: skip
    for run, step in expected.items():
        t = fz.load_run(run, S_RECORDS)
        assert t.delivered.shape == (20, 100) and not t.problems
        assert fz.select(t).selected_step == step
