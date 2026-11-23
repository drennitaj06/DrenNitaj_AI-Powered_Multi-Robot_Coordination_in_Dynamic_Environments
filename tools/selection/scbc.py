"""SCB-C checkpoint selection (Issue 6A, phase L1) — isolated analysis tooling.

**SCB-C = Simultaneous Comparison Band, then Conflicts.** Among the checkpoints of one training
run, it keeps every checkpoint that no other checkpoint is *shown* to beat in completion, with
simultaneous (family-wise) 95 % confidence over all pairs; among those, it keeps every checkpoint
that no other remaining checkpoint is shown to beat in conflicts per episode; then a pre-declared
final rule picks one (``final``: highest completion estimate — the default, provisional pending the
student's decision —, fewest conflicts, latest step, or fewest conflicts among those meeting an
engineering completion floor). No practical-equivalence margin is used.

Simultaneous procedure (``studentize``), all max-T over every pair of valid checkpoints:

* ``"signflip"`` (v4): studentised sign-flip max-T on the paired per-seed contrasts of every
  pair, each pair studentised by its own standard deviation — robust to unequal variances
  across checkpoints (L1 follow-up, proposal §22);
* ``"perm"`` (default, v3): within-seed permutation of checkpoint labels, pairs standardised by
  their permutation standard deviation (Westfall–Young). Exact only under exchangeability of all
  labels within a seed; anti-conservative when checkpoints differ in dispersion (§22);
* ``"t"`` (v2): seed bootstrap-t, delta-method standard error recomputed in every resample —
  valid but nearly powerless at 20 seeds with heavy-tailed conflicts (near-zero resample SEs);
* ``"fixed"`` (v1): seed bootstrap with one fixed standard error — anti-conservative at 20 seeds.

Inputs per checkpoint ``k`` and selection seed ``s`` (levels already summed / averaged per seed,
the seed being the independent unit):

* ``delivered[k, s]``, ``released[k, s]`` — task counts summed over the seed's levels;
* ``conflicts[k, s]`` — mean proposed conflicts per episode over the seed's levels;
* ``collisions[k, s]`` — physical collisions summed over the seed's levels;
* ``steps[k]`` — training step of the checkpoint.

Missing observations are ``NaN``. Definitions and the error interpretation are in
``docs/research/2026-11-20-issue-6a-proposal.md`` §21.

Pure NumPy; independent of the production PPO / MADQL code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]
MIN_SEEDS = 5


@dataclass(frozen=True)
class Band:
    """Simultaneous pairwise comparison of one metric (higher = better after orientation)."""

    estimate: F  # (K,) point estimates, oriented so that larger is better
    diff: F  # (K, K) diff[j, k] = estimate[j] - estimate[k]
    se: F  # (K, K) bootstrap standard error of diff[j, k]
    q: float  # max-T critical value
    lower: F  # (K, K) diff - q * se
    upper: F  # (K, K) diff + q * se


@dataclass(frozen=True)
class Selection:
    selected: int | None
    eligible_completion: tuple[int, ...]
    eligible_conflicts: tuple[int, ...]
    excluded: dict[int, str]
    completion: Band | None
    conflicts: Band | None
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        def band(b: Band | None) -> dict[str, Any] | None:
            if b is None:
                return None
            return {"estimate": b.estimate.tolist(), "q": b.q}

        return {
            "selected": self.selected,
            "eligible_completion": list(self.eligible_completion),
            "eligible_conflicts": list(self.eligible_conflicts),
            "excluded": {str(k): v for k, v in self.excluded.items()},
            "completion": band(self.completion),
            "conflicts": band(self.conflicts),
            "params": self.params,
        }


def validate(delivered: F, released: F, conflicts: F, collisions: F) -> dict[int, str]:
    """Checkpoints to exclude, with the reason (first matching rule wins).

    * ``incomplete`` — any missing (NaN) observation;
    * ``invalid`` — negative values, ``delivered > released``, or no released task at all;
    * ``collision`` — any physical collision (safety precedes performance).
    """
    out: dict[int, str] = {}
    for k in range(delivered.shape[0]):
        rows = (delivered[k], released[k], conflicts[k], collisions[k])
        if any(np.isnan(r).any() for r in rows):
            out[k] = "incomplete"
        elif (
            any((r < 0).any() for r in rows)
            or (delivered[k] > released[k]).any()
            or released[k].sum() <= 0
        ):
            out[k] = "invalid"
        elif collisions[k].sum() > 0:
            out[k] = "collision"
    return out


def resample_counts(n_seeds: int, n_boot: int, seed: int) -> npt.NDArray[np.int64]:
    """Multiplicity matrix ``(n_boot, n_seeds)`` of a seed bootstrap (each row sums to
    ``n_seeds``). One matrix is shared by every checkpoint, which keeps the pairing."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seeds, (n_boot, n_seeds))
    counts = np.zeros((n_boot, n_seeds), dtype=np.int64)
    np.add.at(counts, (np.arange(n_boot)[:, None], idx), 1)
    return counts


def max_t_band(estimate: F, boot: F, alpha: float) -> Band:
    """Simultaneous intervals for all pairwise differences of ``estimate`` (``K``) from its
    bootstrap replicates ``boot`` (``B × K``), by the studentised maximum statistic.

    ``T_b = max over pairs j < k with se > 0 of |(Δ*_jk − Δ_jk) / se_jk|``; ``q`` = the
    ``1 − alpha`` quantile of ``T`` (``numpy`` ``higher`` method, so ``P(T ≤ q) ≥ 1 − alpha``
    in the bootstrap distribution). Pairs with ``se = 0`` (no resampling variability) keep the
    degenerate interval ``[Δ, Δ]`` and do not enter ``T``.
    """
    k = estimate.size
    diff = estimate[:, None] - estimate[None, :]
    bdiff = boot[:, :, None] - boot[:, None, :]  # (B, K, K)
    se = bdiff.std(axis=0, ddof=1) if boot.shape[0] > 1 else np.zeros((k, k))
    iu = np.triu_indices(k, 1)
    ok = se[iu] > 0
    if k < 2 or not ok.any():
        q = 0.0
    else:
        dev = np.abs(bdiff[:, iu[0], iu[1]][:, ok] - diff[iu][ok]) / se[iu][ok]
        tmax = dev.max(axis=1)
        q = float(np.quantile(tmax, 1 - alpha, method="higher"))
    return Band(estimate, diff, se, q, diff - q * se, diff + q * se)


def pair_se(z: F, w: F) -> F:
    """Delta-method standard errors of all pairwise differences of per-seed means.

    ``z`` (``B × K × S``) are per-seed influence values (for a mean: the values themselves; for a
    pooled ratio ``Σ D / Σ R``: ``(D − ĉ R) / R̄``), ``w`` (``B × S``) the seed multiplicities
    of each resample (rows sum to ``S``). Returns ``SE`` (``B × K × K``) with
    ``SE²_jk = (v_j + v_k − 2 c_jk) / S`` from the weighted sample covariance ``c``."""
    s = z.shape[2]
    m = np.einsum("bks,bs->bk", z, w) / s
    m2 = np.einsum("bks,bs,bjs->bkj", z, w, z) / s
    cov = (m2 - m[:, :, None] * m[:, None, :]) * s / (s - 1)
    var = np.diagonal(cov, axis1=1, axis2=2)
    vd = var[:, :, None] + var[:, None, :] - 2 * cov
    se: F = np.sqrt(np.clip(vd, 0.0, None) / s)
    return se


def bootstrap_t_band(estimate: F, boot: F, se0: F, se_boot: F, alpha: float) -> Band:
    """Studentised max-T band with the standard error recomputed in every resample.

    ``T_b = max over pairs j < k of |Δ*_jk − Δ_jk| / SE*_jk`` (pairs with ``SE*_jk = 0`` in a
    resample contribute 0); ``q`` = the ``1 − alpha`` quantile (``higher``); intervals
    ``Δ_jk ± q · SE_jk`` with the original-sample ``SE`` (``se0``). Pairs with ``SE_jk = 0`` keep
    the degenerate interval ``[Δ, Δ]``."""
    k = estimate.size
    diff = estimate[:, None] - estimate[None, :]
    if k < 2:
        return Band(estimate, diff, se0, 0.0, diff.copy(), diff.copy())
    iu = np.triu_indices(k, 1)
    dev = np.abs((boot[:, :, None] - boot[:, None, :])[:, iu[0], iu[1]] - diff[iu])
    seb = se_boot[:, iu[0], iu[1]]
    t = np.divide(dev, seb, out=np.zeros_like(dev), where=seb > 0)
    q = float(np.quantile(t.max(axis=1), 1 - alpha, method="higher"))
    return Band(estimate, diff, se0, q, diff - q * se0, diff + q * se0)


def permutation_indices(n_seeds: int, k: int, n_perm: int, seed: int) -> npt.NDArray[np.int64]:
    """``(n_perm, n_seeds, k)``: an independent random permutation of the ``k`` checkpoint labels
    within every seed (the seed's cells are exchanged, so the pairing is kept)."""
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_perm, n_seeds, k)), axis=2)


def permutation_band(estimate: F, perm: F, alpha: float) -> Band:
    """Westfall–Young max-T band from within-seed label permutations (v3).

    ``perm`` (``P × K``) holds the estimates recomputed after permuting the checkpoint labels
    within every seed. Each pair's statistic is standardised by its permutation standard
    deviation ``sd_jk``; ``T_p = max over pairs with sd > 0 of |Δ^p_jk| / sd_jk``; ``q`` = the
    ``1 − alpha`` quantile (``higher``). ``j`` is shown better than ``k`` when
    ``Δ_jk − q · sd_jk > 0``. Under exchangeability of the labels within seeds (identical
    distributions), the family-wise error over all pairs is at most ``alpha`` (up to Monte Carlo
    error); ``lower`` / ``upper`` are test-inversion bounds, not confidence intervals for means."""
    k = estimate.size
    diff = estimate[:, None] - estimate[None, :]
    pdiff = perm[:, :, None] - perm[:, None, :]
    sd = pdiff.std(axis=0, ddof=1) if perm.shape[0] > 1 else np.zeros((k, k))
    iu = np.triu_indices(k, 1)
    ok = sd[iu] > 0
    if k < 2 or not ok.any():
        q = 0.0
    else:
        t = np.abs(pdiff[:, iu[0], iu[1]][:, ok]) / sd[iu][ok]
        q = float(np.quantile(t.max(axis=1), 1 - alpha, method="higher"))
    return Band(estimate, diff, sd, q, diff - q * sd, diff + q * sd)


def sign_flips(n_seeds: int, n_flip: int, seed: int) -> F:
    """``(n_flip, n_seeds)`` random signs ``±1``, one per seed, shared by every pair."""
    rng = np.random.default_rng(seed)
    out: F = rng.choice(np.array([-1.0, 1.0]), size=(n_flip, n_seeds))
    return out


def sign_flip_band(contrast: F, eps: F, alpha: float) -> Band:
    """Studentised sign-flip max-T over pairwise per-seed contrasts (v4).

    ``contrast`` (``K × S``) holds per-seed values whose pairwise differences are the contrasts
    (for completion: delivered counts / mean released per seed; for conflicts: minus the per-seed
    mean conflicts), oriented so that larger is better. For each pair ``j < k`` the per-seed
    differences ``d_s = contrast[j, s] − contrast[k, s]`` give ``t_jk = mean(d) / (sd(d) / √S)``.
    One sign ``ε_s ∈ {±1}`` per seed (a row of ``eps``, ``B × S``) is shared by every pair
    (keeping the dependence between pairs); in each draw ``t*_jk`` is recomputed — studentised — on ``ε d``, and
    ``T* = max |t*_jk|``; ``q`` = the ``1 − alpha`` quantile (``higher``). ``j`` is shown better
    than ``k`` when ``t_jk > q``.

    Null hypothesis per pair: ``d`` symmetric about 0 (exact); mean ``d`` = 0 with unequal
    variances (asymptotically, by studentisation). Unlike the label permutation (v3), each pair's
    reference distribution uses only that pair's data. Pairs with ``sd(d) = 0`` keep the
    degenerate interval and do not enter ``T``."""
    k, s = contrast.shape
    est = contrast.mean(1)
    diff = est[:, None] - est[None, :]
    iu = np.triu_indices(k, 1)
    d = (contrast[iu[0]] - contrast[iu[1]]).T  # (S, P)
    sd = d.std(axis=0, ddof=1)
    se = np.zeros((k, k))
    se[iu] = sd / np.sqrt(s)
    se[(iu[1], iu[0])] = se[iu]
    ok = sd > 0
    if k < 2 or not ok.any():
        return Band(est, diff, se, 0.0, diff.copy(), diff.copy())
    dd = d[:, ok]
    m = (eps @ dd) / s  # (B, P)
    ss = (dd**2).sum(0)
    var = (ss[None, :] - s * m**2) / (s - 1)
    t = np.divide(np.abs(m), np.sqrt(np.clip(var, 0.0, None) / s),
                  out=np.zeros_like(m), where=var > 0)  # fmt: skip
    q = float(np.quantile(t.max(axis=1), 1 - alpha, method="higher"))
    return Band(est, diff, se, q, diff - q * se, diff + q * se)


FINAL_RULES = ("completion", "conflicts", "latest", "constrained")


def choose(members: list[int], steps: F, comp: F, conf_cost: F, rule: str,
           floor: float = 0.99) -> int:  # fmt: skip
    """Final pick among the doubly eligible members (indices into the valid set).

    * ``completion`` — highest completion estimate, then fewer conflicts, then later step;
    * ``conflicts`` — fewest conflicts, then higher completion, then later step;
    * ``latest`` — latest training step (revision-2 proposal; v1 of the study);
    * ``constrained`` — fewest conflicts among members whose completion estimate is at least
      ``floor`` (an engineering requirement, not an equivalence margin), then higher completion,
      then later step; if no member meets the floor, the ``completion`` rule."""
    if rule == "constrained":
        ok = [i for i in members if comp[i] >= floor]
        if ok:
            return max(ok, key=lambda i: (-conf_cost[i], comp[i], steps[i]))
        rule = "completion"
    if rule == "completion":
        return max(members, key=lambda i: (comp[i], -conf_cost[i], steps[i]))
    if rule == "conflicts":
        return max(members, key=lambda i: (-conf_cost[i], comp[i], steps[i]))
    if rule == "latest":
        return max(members, key=lambda i: (steps[i], -i))
    raise ValueError(f"unknown final rule {rule!r}; expected one of {FINAL_RULES}")


def not_dominated(band: Band, members: list[int]) -> list[int]:
    """Members ``k`` for which no member ``j`` has a simultaneous lower bound ``> 0`` on
    ``estimate[j] − estimate[k]`` (no other member is shown better)."""
    return [k for k in members if not any(band.lower[j, k] > 0 for j in members if j != k)]


def select(
    steps: npt.ArrayLike,
    delivered: npt.ArrayLike,
    released: npt.ArrayLike,
    conflicts: npt.ArrayLike,
    collisions: npt.ArrayLike | None = None,
    *,
    alpha_completion: float = 0.05,
    alpha_conflicts: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
    studentize: str = "perm",
    final: str = "completion",
    completion_floor: float = 0.99,
) -> Selection:
    """SCB-C selection for one run (``K`` checkpoints × ``S`` seeds arrays)."""
    st = np.asarray(steps, dtype=np.float64)
    d = np.asarray(delivered, dtype=np.float64)
    r = np.asarray(released, dtype=np.float64)
    c = np.asarray(conflicts, dtype=np.float64)
    x = np.zeros_like(d) if collisions is None else np.asarray(collisions, dtype=np.float64)
    if not (d.shape == r.shape == c.shape == x.shape) or d.ndim != 2 or st.shape != (d.shape[0],):
        raise ValueError("inputs must be (K, S) arrays with steps of length K")
    if d.shape[1] < MIN_SEEDS:
        raise ValueError(f"at least {MIN_SEEDS} seeds are required, got {d.shape[1]}")
    for a in (alpha_completion, alpha_conflicts):
        if not 0 < a < 1:
            raise ValueError("alpha must be in (0, 1)")
    params = {"alpha_completion": alpha_completion, "alpha_conflicts": alpha_conflicts,
              "n_boot": n_boot, "seed": seed, "n_checkpoints": d.shape[0], "n_seeds": d.shape[1]}  # fmt: skip
    excluded = validate(d, r, c, x)
    valid = [k for k in range(d.shape[0]) if k not in excluded]
    if not valid:
        return Selection(None, (), (), excluded, None, None, params)
    dv, rv, cv = d[valid], r[valid], c[valid]
    if studentize not in ("signflip", "perm", "t", "fixed"):
        raise ValueError(
            "studentize must be 'signflip' (v4), 'perm' (v3), 't' (bootstrap-t, v2) or 'fixed' (v1)"
        )
    if final not in FINAL_RULES:
        raise ValueError(f"final must be one of {FINAL_RULES}")
    s = d.shape[1]
    w = resample_counts(s, n_boot, seed).astype(np.float64)  # (B, S), shared
    comp = dv.sum(1) / rv.sum(1)
    comp_b = (w @ dv.T) / (w @ rv.T)  # (B, K)
    conf = -cv.mean(1)  # oriented: larger is better
    conf_b = -(w @ cv.T) / s
    if studentize == "signflip":
        # released counts are shared by the checkpoints of a seed in the stored data; with
        # contrast D / mean(R) the per-checkpoint mean equals the pooled completion exactly
        eps = sign_flips(s, n_boot, seed)
        bc = sign_flip_band(dv / rv.mean(1)[:, None], eps, alpha_completion)
        bm = sign_flip_band(-cv, eps, alpha_conflicts)
    elif studentize == "perm":
        idx = permutation_indices(s, len(valid), n_boot, seed)  # (P, S, K)
        cols = np.arange(s)[None, :, None]
        dp, rp, cp = dv.T[cols, idx], rv.T[cols, idx], cv.T[cols, idx]  # (P, S, K)
        bc = permutation_band(comp, dp.sum(1) / rp.sum(1), alpha_completion)
        bm = permutation_band(conf, -cp.mean(1), alpha_conflicts)
    elif studentize == "fixed":
        bc = max_t_band(comp, comp_b, alpha_completion)
        bm = max_t_band(conf, conf_b, alpha_conflicts)
    else:
        one = np.ones((1, s))
        z0 = ((dv - comp[:, None] * rv) / rv.mean(1)[:, None])[None]
        rbar_b = (w @ rv.T) / s
        zb = (dv[None] - comp_b[:, :, None] * rv[None]) / rbar_b[:, :, None]
        bc = bootstrap_t_band(comp, comp_b, pair_se(z0, one)[0], pair_se(zb, w), alpha_completion)
        cz = -cv[None]
        bm = bootstrap_t_band(conf, conf_b, pair_se(cz, one)[0],
                              pair_se(np.broadcast_to(cz, (n_boot, *cv.shape)), w), alpha_conflicts)  # fmt: skip
    params |= {"studentize": studentize, "final": final}
    if final == "constrained":
        params["completion_floor"] = completion_floor
    local = list(range(len(valid)))
    e1 = not_dominated(bc, local)  # stage 1: completion (family: all valid pairs)
    e2 = not_dominated(bm, e1)  # stage 2: conflicts (family fixed in advance: all valid pairs)
    best = choose(e2, st[valid], comp, cv.mean(1), final, completion_floor)
    return Selection(
        valid[best],
        tuple(valid[i] for i in e1),
        tuple(valid[i] for i in e2),
        excluded,
        bc,
        bm,
        params,
    )


def old_rule(steps: npt.ArrayLike, delivered: npt.ArrayLike, released: npt.ArrayLike,
             base_return: npt.ArrayLike) -> int:  # fmt: skip
    """The historical (Issue #5/#6) rule, for comparison in simulation: lexicographic maximum of
    pooled completion, then mean return; ties to the earliest step (first maximum)."""
    d = np.asarray(delivered, dtype=np.float64)
    r = np.asarray(released, dtype=np.float64)
    ret = np.asarray(base_return, dtype=np.float64)
    comp = d.sum(1) / r.sum(1)
    keys = list(zip(comp, ret, strict=True))
    return max(range(len(keys)), key=lambda i: (keys[i][0], keys[i][1], -i))
