"""Tests of the SCB-C selection tool (Issue 6A, L1): ``tools/selection/scbc.py``."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "selection"))
import scbc


def world(k: int = 4, s: int = 20, seed: int = 1, comp: float = 0.98, conf: float = 2.0):
    rng = np.random.default_rng(seed)
    released = rng.integers(54, 67, size=(1, s)).repeat(k, 0).astype(float)
    delivered = np.minimum(released, rng.binomial(released.astype(int), comp)).astype(float)
    conflicts = rng.poisson(conf, size=(k, s)).astype(float)
    return np.arange(1, k + 1) * 50_000, delivered, released, conflicts


# ----------------------------------------------------------------------------- determinism
def test_selection_is_deterministic_for_a_fixed_seed() -> None:
    steps, d, r, c = world()
    a = scbc.select(steps, d, r, c, n_boot=2000, seed=7)
    b = scbc.select(steps, d, r, c, n_boot=2000, seed=7)
    assert a.summary() == b.summary()
    assert np.array_equal(a.completion.lower, b.completion.lower)


def test_resample_counts_rows_sum_to_the_number_of_seeds() -> None:
    w = scbc.resample_counts(20, 500, 3)
    assert w.shape == (500, 20) and (w.sum(1) == 20).all() and (w >= 0).all()
    assert np.array_equal(w, scbc.resample_counts(20, 500, 3))


# ----------------------------------------------------------------------------- pairing
def test_paired_seed_bootstrap_uses_one_resample_for_all_checkpoints() -> None:
    # strong seed heterogeneity, checkpoint 1 = checkpoint 0 minus one delivery on every seed:
    # paired, the difference is precise; resampled independently it would not be.
    rng = np.random.default_rng(0)
    released = np.full((2, 20), 60.0)
    base = rng.integers(30, 60, 20).astype(float)
    delivered = np.stack([base, base - 1])
    conflicts = np.ones((2, 20))
    sel = scbc.select([1, 2], delivered, released, conflicts, n_boot=4000, seed=0)
    se_paired = sel.completion.se[0, 1]
    w1, w2 = scbc.resample_counts(20, 4000, 1), scbc.resample_counts(20, 4000, 2)
    unpaired = (w1 @ delivered[0]) / (w1 @ released[0]) - (w2 @ delivered[1]) / (w2 @ released[1])
    assert se_paired < unpaired.std(ddof=1) / 10
    assert sel.eligible_completion == (0,)  # checkpoint 1 is shown worse
    assert sel.selected == 0


# --------------------------------------------------------------- max-T (independent checks)
def manual_q(estimate: np.ndarray, boot: np.ndarray, alpha: float) -> float:
    """Loop-based reference implementation of the studentised max statistic."""
    k = len(estimate)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    se = {}
    for i, j in pairs:
        diffs = [b[i] - b[j] for b in boot]
        m = sum(diffs) / len(diffs)
        se[(i, j)] = (sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
    tmax = []
    for b in boot:
        vals = [abs((b[i] - b[j]) - (estimate[i] - estimate[j])) / se[(i, j)]
                for i, j in pairs if se[(i, j)] > 0]  # fmt: skip
        tmax.append(max(vals))
    tmax.sort()
    n = len(tmax)
    pos = (n - 1) * (1 - alpha)
    return tmax[int(np.ceil(pos))]  # numpy "higher" quantile


def test_max_t_matches_a_hand_computed_example() -> None:
    est = np.array([0.5, 0.4, 0.45])
    boot = np.array([[0.52, 0.41, 0.44], [0.48, 0.42, 0.47], [0.55, 0.37, 0.46],
                     [0.49, 0.40, 0.43], [0.50, 0.39, 0.45]])  # fmt: skip
    band = scbc.max_t_band(est, boot, 0.2)
    assert band.q == pytest.approx(manual_q(est, boot, 0.2))
    assert np.allclose(band.lower, band.diff - band.q * band.se)


def test_two_checkpoints_reduce_to_the_bootstrap_t_interval() -> None:
    rng = np.random.default_rng(5)
    est = np.array([0.0, 0.1])
    boot = est + rng.normal(0, [0.03, 0.05], size=(5000, 2))
    band = scbc.max_t_band(est, boot, 0.05)
    se = np.std(boot[:, 0] - boot[:, 1], ddof=1)
    t = np.abs((boot[:, 0] - boot[:, 1]) - (est[0] - est[1])) / se
    assert band.q == pytest.approx(np.quantile(t, 0.95, method="higher"))


def test_exact_enumeration_of_the_seed_bootstrap_for_five_seeds() -> None:
    # every one of the 5^5 equally likely resamples, as the bootstrap distribution
    d = np.array([[10, 11, 9, 12, 10], [9, 11, 8, 12, 11], [12, 10, 10, 11, 12]], float)
    r = np.full_like(d, 12.0)
    est = d.sum(1) / r.sum(1)
    boot = []
    for idx in itertools.product(range(5), repeat=5):
        cnt = np.bincount(idx, minlength=5)
        boot.append((d @ cnt) / (r @ cnt))
    exact = scbc.max_t_band(est, np.array(boot), 0.05).q
    assert exact == pytest.approx(manual_q(est, np.array(boot), 0.05))
    w = scbc.resample_counts(5, 200_000, 11).astype(float)
    mc = scbc.max_t_band(est, (w @ d.T) / (w @ r.T), 0.05).q
    assert mc == pytest.approx(exact, rel=0.03)


def test_zero_variance_pairs_keep_a_degenerate_interval() -> None:
    est = np.array([1.0, 1.0, 0.0])
    boot = np.array([[1.0, 1.0, 0.1], [1.0, 1.0, -0.1], [1.0, 1.0, 0.0]])
    band = scbc.max_t_band(est, boot, 0.05)
    assert band.se[0, 1] == 0 and band.lower[0, 1] == 0 == band.upper[0, 1]


# ----------------------------------------------------------------------------- elimination and ties
def test_identical_checkpoints_are_all_kept_and_the_latest_step_wins() -> None:
    steps, d, r, c = world(k=4)
    d[:] = d[0]
    c[:] = c[0]
    sel = scbc.select(steps, d, r, c, n_boot=1000)
    assert sel.eligible_completion == (0, 1, 2, 3) == sel.eligible_conflicts
    assert sel.selected == 3


def test_a_clearly_worse_checkpoint_is_eliminated() -> None:
    steps, d, r, c = world(k=3, comp=0.99)
    d[1] = np.floor(r[1] * 0.6)
    sel = scbc.select(steps, d, r, c, n_boot=2000)
    assert 1 not in sel.eligible_completion and 1 not in sel.eligible_conflicts


def test_conflicts_break_completion_ties() -> None:
    steps, d, r, c = world(k=3)
    d[:] = d[0]
    c[0], c[1], c[2] = 40.0, 1.0, 50.0
    sel = scbc.select(steps, d, r, c, n_boot=2000)
    assert sel.eligible_completion == (0, 1, 2)
    assert sel.eligible_conflicts == (1,) and sel.selected == 1


def test_conflicts_never_override_a_completion_elimination() -> None:
    steps, d, r, c = world(k=2, comp=0.99)
    d[1] = np.floor(r[1] * 0.5)
    c[0], c[1] = 100.0, 0.0
    sel = scbc.select(steps, d, r, c, n_boot=2000)
    assert sel.selected == 0


# --------------------------------------------------------------- missing, invalid, small sets
def test_missing_invalid_and_colliding_checkpoints_are_excluded_with_reasons() -> None:
    steps, d, r, c = world(k=5)
    x = np.zeros_like(d)
    d[0, 3] = np.nan
    d[1, 0] = r[1, 0] + 1
    x[2, 5] = 1
    r[3] = 0
    d[3] = 0
    sel = scbc.select(steps, d, r, c, x, n_boot=500)
    assert sel.excluded == {0: "incomplete", 1: "invalid", 2: "collision", 3: "invalid"}
    assert sel.selected == 4 and sel.eligible_completion == (4,)


def test_no_valid_checkpoint_gives_no_selection() -> None:
    steps, d, r, c = world(k=2)
    d[:, 0] = np.nan
    sel = scbc.select(steps, d, r, c, n_boot=100)
    assert sel.selected is None and set(sel.excluded) == {0, 1}


@pytest.mark.parametrize("bad", ["seeds", "shape", "alpha"])
def test_bad_inputs_are_rejected(bad: str) -> None:
    steps, d, r, c = world(k=3, s=20)
    with pytest.raises(ValueError):
        if bad == "seeds":
            scbc.select(steps, d[:, :4], r[:, :4], c[:, :4])
        elif bad == "shape":
            scbc.select(steps[:2], d, r, c)
        else:
            scbc.select(steps, d, r, c, alpha_completion=0.0)


def test_old_rule_takes_the_first_maximum_of_completion_then_return() -> None:
    d = np.array([[9, 9], [10, 9], [10, 9]], float)
    r = np.full_like(d, 10.0)
    assert scbc.old_rule([1, 2, 3], d, r, [0.1, 0.5, 0.5]) == 1


# ----------------------------------------------------------------------------- properties
small = st.integers(min_value=2, max_value=6)


@settings(max_examples=40, deadline=None)
@given(k=small, s=st.integers(5, 12), seed=st.integers(0, 10_000))
def test_selection_sets_are_nested_and_permutation_invariant(k: int, s: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = rng.integers(10, 30, (k, s)).astype(float)
    d = np.floor(r * rng.uniform(0.6, 1.0, (k, s)))
    c = rng.poisson(rng.uniform(0.5, 5.0, (k, 1)), (k, s)).astype(float)
    steps = np.arange(1, k + 1)
    for mode in ("perm", "t"):
        nested = scbc.select(steps, d, r, c, n_boot=300, seed=seed, studentize=mode)
        assert nested.selected in nested.eligible_conflicts
        assert set(nested.eligible_conflicts) <= set(nested.eligible_completion)
    # exact invariance to checkpoint order holds for the seed bootstrap (the permutation method is
    # invariant only in distribution: its label draws depend on the order)
    sel = scbc.select(steps, d, r, c, n_boot=300, seed=seed, studentize="t")
    perm = rng.permutation(k)
    sel2 = scbc.select(
        steps[perm], d[perm], r[perm], c[perm], n_boot=300, seed=seed, studentize="t"
    )
    assert perm[sel2.selected] == sel.selected
    shifted = scbc.select(
        steps, d, r, c + 7.0, n_boot=300, seed=seed, studentize="t"
    )  # shift invariance
    assert shifted.selected == sel.selected


@settings(max_examples=40, deadline=None)
@given(k=small, s=st.integers(5, 12), seed=st.integers(0, 10_000))
def test_a_checkpoint_best_on_every_seed_is_never_eliminated(k: int, s: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = np.full((k, s), 20.0)
    d = np.floor(r * rng.uniform(0.5, 0.95, (k, s)))
    c = rng.uniform(1, 10, (k, s))
    best = int(rng.integers(0, k))
    d[best] = d.max(0)
    c[best] = c.min(0)
    sel = scbc.select(np.arange(k), d, r, c, n_boot=300, seed=seed)
    assert best in sel.eligible_completion and best in sel.eligible_conflicts


# ----------------------------------------------------------------------------- bootstrap-t (v2)
def test_pair_se_matches_a_loop_implementation() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=(2, 3, 6))
    w = np.array([[1, 1, 1, 1, 1, 1], [2, 0, 1, 0, 3, 0]], dtype=float)
    got = scbc.pair_se(z, w)
    for b in range(2):
        vals = [[z[b, k, s] for s in range(6) for _ in range(int(w[b, s]))] for k in range(3)]
        for j in range(3):
            for k in range(3):
                d = np.array(vals[j]) - np.array(vals[k])
                assert got[b, j, k] == pytest.approx(np.std(d, ddof=1) / np.sqrt(6), abs=1e-12)


def test_bootstrap_t_band_matches_a_manual_computation() -> None:
    est = np.array([0.5, 0.4, 0.45])
    boot = np.array([[0.52, 0.41, 0.44], [0.48, 0.42, 0.47], [0.55, 0.37, 0.46]])
    se0 = np.full((3, 3), 0.02)
    seb = np.array([np.full((3, 3), v) for v in (0.01, 0.02, 0.04)])
    band = scbc.bootstrap_t_band(est, boot, se0, seb, 0.5)
    tmax = []
    for b in range(3):
        t = [abs((boot[b, i] - boot[b, j]) - (est[i] - est[j])) / seb[b, i, j]
             for i in range(3) for j in range(i + 1, 3)]  # fmt: skip
        tmax.append(max(t))
    assert band.q == pytest.approx(np.quantile(tmax, 0.5, method="higher"))
    assert np.allclose(band.upper, band.diff + band.q * se0)


def test_delta_method_se_agrees_with_the_bootstrap_for_many_seeds() -> None:
    steps, d, r, c = world(k=3, s=400, comp=0.95)
    sel = scbc.select(steps, d, r, c, n_boot=4000, seed=1)
    w = scbc.resample_counts(400, 4000, 1).astype(float)
    b = (w @ d.T) / (w @ r.T)
    boot_se = np.std(b[:, 0] - b[:, 1], ddof=1)
    assert sel.completion.se[0, 1] == pytest.approx(boot_se, rel=0.1)


@pytest.mark.parametrize("rule, expected", [("completion", 0), ("conflicts", 2), ("latest", 2)])
def test_final_rules(rule: str, expected: int) -> None:
    members = [0, 1, 2]
    steps = np.array([1.0, 2.0, 3.0])
    comp = np.array([0.99, 0.98, 0.98])
    conf = np.array([5.0, 3.0, 1.0])
    assert scbc.choose(members, steps, comp, conf, rule) == expected


def test_v1_and_v2_are_both_selectable_and_validated() -> None:
    steps, d, r, c = world(k=4)
    v1 = scbc.select(steps, d, r, c, n_boot=500, studentize="fixed", final="latest")
    v2 = scbc.select(steps, d, r, c, n_boot=500, studentize="t")
    v3 = scbc.select(steps, d, r, c, n_boot=500)
    assert v1.params["studentize"] == "fixed" and v2.params["studentize"] == "t"
    assert v3.params["studentize"] == "perm"
    with pytest.raises(ValueError):
        scbc.select(steps, d, r, c, studentize="other")
    with pytest.raises(ValueError):
        scbc.select(steps, d, r, c, final="other")


# --------------------------------------------------------------- permutation max-T (v3)
def test_permutation_indices_are_permutations_within_each_seed() -> None:
    idx = scbc.permutation_indices(6, 4, 50, 0)
    assert idx.shape == (50, 6, 4)
    assert (np.sort(idx, axis=2) == np.arange(4)).all()
    assert np.array_equal(idx, scbc.permutation_indices(6, 4, 50, 0))


def test_permutation_band_matches_a_manual_computation() -> None:
    est = np.array([0.3, 0.1, 0.2])
    perm = np.array([[0.2, 0.2, 0.2], [0.25, 0.15, 0.2], [0.1, 0.3, 0.2], [0.2, 0.1, 0.3]])
    band = scbc.permutation_band(est, perm, 0.5)
    pairs = [(0, 1), (0, 2), (1, 2)]
    sd = {p: np.std([row[p[0]] - row[p[1]] for row in perm], ddof=1) for p in pairs}
    tmax = [max(abs(row[i] - row[j]) / sd[(i, j)] for i, j in pairs) for row in perm]
    assert band.q == pytest.approx(np.quantile(tmax, 0.5, method="higher"))
    assert band.se[0, 1] == pytest.approx(sd[(0, 1)])


def test_permutation_selection_eliminates_a_clearly_worse_checkpoint_and_is_deterministic() -> None:
    steps, d, r, c = world(k=4, comp=0.99)
    d[2] = np.floor(r[2] * 0.6)
    c[1] = c[1] + 30
    a = scbc.select(steps, d, r, c, n_boot=2000, seed=3)
    b = scbc.select(steps, d, r, c, n_boot=2000, seed=3)
    assert a.summary() == b.summary() and a.params["studentize"] == "perm"
    assert 2 not in a.eligible_completion and 1 not in a.eligible_conflicts


def test_permutation_null_rarely_eliminates() -> None:
    eliminated = 0
    for rep in range(60):
        rng = np.random.default_rng(rep)
        r = np.full((6, 20), 60.0)
        d = r - rng.binomial(60, 0.015, (6, 20))
        c = rng.poisson(2.0, (6, 20)).astype(float)
        sel = scbc.select(np.arange(6), d, r, c, n_boot=500, seed=rep)
        eliminated += len(sel.eligible_completion) < 6
    assert eliminated <= 9  # 5 % nominal; generous bound for 60 replicates


# ----------------------------------------------------------------------------- v4: sign flip
def manual_sign_flip(
    contrast: np.ndarray, eps: np.ndarray, alpha: float
) -> tuple[float, np.ndarray]:
    k, s = contrast.shape
    tmax, tobs = [], np.zeros((k, k))
    pairs = [(j, m) for j in range(k) for m in range(j + 1, k)]
    for j, m in pairs:
        d = contrast[j] - contrast[m]
        if d.std(ddof=1) == 0:
            continue
        tobs[j, m] = d.mean() / (d.std(ddof=1) / np.sqrt(s))
    for e in eps:
        best = 0.0
        for j, m in pairs:
            d = e * (contrast[j] - contrast[m])
            if (contrast[j] - contrast[m]).std(ddof=1) == 0:
                continue
            sd = d.std(ddof=1)
            if sd > 0:
                best = max(best, abs(d.mean()) / (sd / np.sqrt(s)))
        tmax.append(best)
    return float(np.quantile(tmax, 1 - alpha, method="higher")), tobs


def test_sign_flip_band_matches_a_loop_implementation() -> None:
    rng = np.random.default_rng(5)
    contrast = rng.normal(size=(4, 9)) * np.array([[0.5], [1.0], [2.0], [4.0]])
    contrast[3, :] = contrast[2, :]  # one degenerate pair (sd = 0)
    eps = scbc.sign_flips(9, 300, 2)
    band = scbc.sign_flip_band(contrast, eps, 0.1)
    q, tobs = manual_sign_flip(contrast, eps, 0.1)
    assert band.q == pytest.approx(q)
    iu = np.triu_indices(4, 1)
    ok = band.se[iu] > 0
    assert np.allclose((band.diff / np.where(band.se > 0, band.se, 1))[iu][ok], tobs[iu][ok])
    assert band.se[2, 3] == 0 and band.lower[2, 3] == band.diff[2, 3]
    assert np.allclose(band.se, band.se.T)


def test_sign_flip_exact_enumeration_matches_the_two_sample_reference_distribution() -> None:
    # K = 2, all 2^S sign vectors: q is the exact (1 − alpha) quantile of |t*|
    s = 6
    contrast = np.array([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [2.0, 6.0, 5.0, 3.0, 5.0, 8.0]])
    eps = np.array(list(itertools.product([-1.0, 1.0], repeat=s)))
    band = scbc.sign_flip_band(contrast, eps, 0.25)
    d = contrast[0] - contrast[1]
    ts = sorted(abs((e * d).mean()) / ((e * d).std(ddof=1) / np.sqrt(s)) for e in eps)
    # numpy 'higher': element ceil((n − 1) p) of the sorted values
    assert band.q == pytest.approx(ts[int(np.ceil((len(ts) - 1) * 0.75))])


def test_sign_flip_contrast_mean_is_the_pooled_completion() -> None:
    steps, d, r, c = world(k=5)
    sel = scbc.select(steps, d, r, c, n_boot=500, seed=1, studentize="signflip")
    assert np.allclose(sel.completion.estimate, d.sum(1) / r.sum(1))
    assert np.allclose(sel.conflicts.estimate, -c.mean(1))


def test_sign_flip_selection_eliminates_a_clearly_worse_checkpoint_and_is_deterministic() -> None:
    steps, d, r, c = world(k=4, s=20)
    d[1] = np.maximum(d[1] - 6, 0)
    a = scbc.select(steps, d, r, c, n_boot=2000, seed=3, studentize="signflip")
    b = scbc.select(steps, d, r, c, n_boot=2000, seed=3, studentize="signflip")
    assert a.summary() == b.summary()
    assert 1 not in a.eligible_completion and a.selected != 1


def test_sign_flip_is_exact_for_symmetric_heteroscedastic_contrasts() -> None:
    # equal means, Gaussian per-seed values with standard deviations 0.1-3 (symmetric contrasts):
    # the sign flip is exact here; the pooled-SD label permutation (v3) is not. Skewed, sparse
    # contrasts (conflicts, undelivered tasks) break the sign flip at 20 seeds (proposal §22).
    rng = np.random.default_rng(11)
    k, s, reps = 8, 20, 200
    sd = np.geomspace(0.1, 3.0, k)[:, None]
    hits4 = hits3 = 0
    for rep in range(reps):
        x = rng.normal(0, 1, (1, s)) + rng.normal(0, 1, (k, s)) * sd
        hits4 += int((scbc.sign_flip_band(x, scbc.sign_flips(s, 1000, rep), 0.05).lower > 0).any())
        idx = scbc.permutation_indices(s, k, 1000, rep)
        xp = x.T[np.arange(s)[None, :, None], idx].mean(1)
        hits3 += int((scbc.permutation_band(x.mean(1), xp, 0.05).lower > 0).any())
    assert hits4 / reps < 0.09  # nominal 0.05 (Monte Carlo SE about 0.015)
    assert hits3 / reps > hits4 / reps


def test_constrained_final_rule_uses_the_floor_and_falls_back_to_completion() -> None:
    steps = np.array([1.0, 2.0, 3.0])
    comp = np.array([0.995, 0.991, 0.985])
    cost = np.array([9.0, 2.0, 1.0])
    assert scbc.choose([0, 1, 2], steps, comp, cost, "constrained", 0.99) == 1
    assert scbc.choose([0, 1, 2], steps, comp, cost, "constrained", 0.999) == 0
    assert scbc.choose([0, 1, 2], steps, comp, cost, "constrained", 0.98) == 2
