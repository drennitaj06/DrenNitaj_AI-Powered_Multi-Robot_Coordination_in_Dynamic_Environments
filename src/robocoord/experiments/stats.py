"""Statistics for experiment summaries (05 §B.3): bootstrap confidence intervals, paired
bootstrap of differences, Holm–Bonferroni, rank-biserial effect size, log–log exponent fits.

The Wilcoxon signed-rank p-value needs SciPy, which becomes a dependency in Issue #4; it is
added there. Everything here is NumPy only and deterministic for a given ``seed``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float
    n: int


def bootstrap_ci(
    values: Sequence[float] | FloatArray,
    stat: Literal["mean", "median"] = "mean",
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI of the mean or median (vectorised resampling)."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    sample = a[rng.integers(0, a.size, size=(n_boot, a.size))]
    if stat == "mean":
        point, boots = float(np.mean(a)), sample.mean(axis=1)
    else:
        point, boots = float(np.median(a)), np.median(sample, axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return Estimate(point, float(lo), float(hi), int(a.size))


def paired_bootstrap(
    x: Sequence[float], y: Sequence[float], *, n_boot: int = 10_000, seed: int = 0
) -> Estimate:
    """Mean of the paired differences ``x_s − y_s`` with a percentile bootstrap CI."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return bootstrap_ci(d, n_boot=n_boot, seed=seed)


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation of ``x − y`` (zero differences dropped)."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    order = np.argsort(np.abs(d), kind="stable")
    ranks = np.empty(d.size)
    absd = np.abs(d)[order]
    i = 0
    while i < d.size:  # average ranks for ties
        j = i
        while j + 1 < d.size and absd[j + 1] == absd[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    total = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


@dataclass(frozen=True)
class PowerFit:
    """``log y = a + b log x`` with a bootstrap CI for the exponent ``b``."""

    exponent: float
    low: float
    high: float
    intercept: float
    n: int
    rss: float = float("nan")  # residual sum of squares of the fit in log space


def loglog_fit(
    x: Sequence[float], y: Sequence[float], *, n_boot: int = 2000, seed: int = 0
) -> PowerFit:
    """Least-squares fit in log–log space; points with ``x ≤ 0`` or ``y ≤ 0`` are dropped."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    keep = (xa > 0) & (ya > 0)
    lx, ly = np.log(xa[keep]), np.log(ya[keep])
    if lx.size < 2 or np.ptp(lx) == 0:
        return PowerFit(float("nan"), float("nan"), float("nan"), float("nan"), int(lx.size))
    b, a = np.polyfit(lx, ly, 1)
    rss = float(np.sum((ly - (a + b * lx)) ** 2))
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, lx.size, size=lx.size)
        if np.ptp(lx[idx]) == 0:
            continue
        slopes.append(np.polyfit(lx[idx], ly[idx], 1)[0])
    lo, hi = np.quantile(slopes, [0.025, 0.975]) if slopes else (float("nan"), float("nan"))
    return PowerFit(float(b), float(lo), float(hi), float(a), int(lx.size), rss)


def polylog_fit(
    n: Sequence[float], y: Sequence[float], *, n_boot: int = 2000, seed: int = 0
) -> PowerFit:
    """Fit ``y ≈ a·(ln n)^c``, i.e. a log–log fit against ``ln n`` (points with ``n ≤ 1`` are
    dropped). Its ``rss`` is directly comparable with that of :func:`loglog_fit` on the same
    points, because both are least squares on ``log y`` with two parameters. A better fit over a
    finite range describes the data; it does not prove an asymptotic bound."""
    na = np.asarray(n, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    keep = na > 1
    return loglog_fit(np.log(na[keep]).tolist(), ya[keep].tolist(), n_boot=n_boot, seed=seed)
