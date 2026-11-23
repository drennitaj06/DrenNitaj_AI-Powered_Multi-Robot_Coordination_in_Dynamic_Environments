"""Block C development evaluation — the frozen analysis (Issue 6A; approved 2026-11-21).

Plan: ``docs/research/2026-11-21-block-c-analysis-plan.md``. This module and the plan were
hashed (``block_c.py freeze``) before any Block C episode was run; ``block_c.py run`` refuses
to start when either hash differs from the frozen value.

Question: for the four PPO runs whose pick changed, how does the SCB-C v5 checkpoint ("new")
compare with the Issue #6 checkpoint ("old") on the 120 Block C seeds × 3 levels, paired by
scenario instance? Block C is development data; nothing here feeds back into SCB-C v5.

Statistical unit: the scenario seed ``s`` (its three levels and both arms stay together).
Every outcome is a ratio of sums ``θ = Σ_s Y_s / Σ_s X_s`` over seeds (``Y_s``, ``X_s`` summed
over the levels in scope), and the effect is the paired difference ``Δ = θ_new − θ_old``:

=============================== ===================== ======================== =========
outcome                         Y (per episode)       X (per episode)          family
=============================== ===================== ======================== =========
``completion``                  delivered             released                 primary
``conflicts_per_episode``       conflicts_total       1                        primary
``conflicts_per_delivered``     conflicts_total       delivered                secondary
``conflicts_per_100_robot_steps`` 100·conflicts_total robots·steps             secondary
``base_return_per_episode``     base_return           1                        secondary
=============================== ===================== ======================== =========

Standard error (delta method, paired): ``u_cs = (Y_cs − θ_c X_cs) / X̄_c`` for arm ``c``,
``d_s = u_new,s − u_old,s``, ``SE = sd(d, ddof=1) / √n``.

Inference: seed-cluster bootstrap-t, symmetric. One index matrix ``(B, n)`` from
``default_rng(BOOT_SEED)`` resamples seeds with replacement and is shared by every run, arm,
level and outcome. Per resample ``t*_b = (Δ*_b − Δ̂) / SE*_b`` (``0/0 := 0``, ``x/0 := ±∞``, and
``|v| ≤ 10⁻¹²`` counts as 0 for an SE or a difference).
95 % CI ``Δ̂ ± q·SE`` with ``q`` the ``1 − α`` quantile of ``|t*|`` (method ``higher``);
two-sided ``p = (1 + #{|t*_b| ≥ |t_obs|}) / (B + 1)`` with ``t_obs = Δ̂ / SE`` (same
conventions). Holm step-down within each family (primary: 4 runs × 2 outcomes = 8; secondary:
4 runs × 3 outcomes = 12), each at family-wise α = 0.05. Per-level results, the across-run
mean and the safety tables are descriptive (unadjusted CIs, no tests).

Validity gate (before any statistic): every (run, arm, level, seed) has exactly one record;
every record is a completed episode (``status == "ok"``) with valid counts; old and new agree on
every paired field, including the scenario-instance hash, which also equals the hash computed by
``block_c.py check`` before the run. A record whose configuration differs from the frozen one
aborts the analysis. A run with a missing, failed or invalid episode is *not evaluable*: its
hypotheses enter Holm with ``p = 1`` and its estimates come from a complete-case analysis (the
seeds with every episode of the run valid), reported as descriptive only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]

BLOCK_C = range(11_000, 11_120)
LEVELS = ("static", "obstacles", "full")
RUNS = ("mappo_seed1", "mappo_seed2", "mappo_seed3", "ippo_seed1")
ARMS = ("old", "new")
ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 0
QUANTILE_METHOD: Final = "higher"

#: The frozen evaluation condition of every record.
EXPECTED: dict[str, Any] = {
    "controller": "ppo",
    "planner": "astar",
    "block": "C",
    "shield": "off",
    "allocator": "auction",
    "mode": "greedy",
    "replicate": 0,
    "rrt_iterations": 1000,
}
#: Fields that must be identical for the old and the new episode of one instance.
PAIRED = (*EXPECTED, "label", "level", "seed", "robots", "max_steps", "instance_sha256")
COUNTS = ("steps", "released", "delivered", "conflicts_total", "physical_collisions", "robots")

#: outcome -> (numerator field, denominator field, scale)
OUTCOMES: dict[str, tuple[str, str, float]] = {
    "completion": ("delivered", "released", 1.0),
    "conflicts_per_episode": ("conflicts", "episodes", 1.0),
    "conflicts_per_delivered": ("conflicts", "delivered", 1.0),
    "conflicts_per_100_robot_steps": ("conflicts", "robot_steps", 100.0),
    "base_return_per_episode": ("base_return", "episodes", 1.0),
}
PRIMARY = ("completion", "conflicts_per_episode")
SECONDARY = ("conflicts_per_delivered", "conflicts_per_100_robot_steps", "base_return_per_episode")
FIELDS = ("delivered", "released", "conflicts", "robot_steps", "base_return", "episodes")


class ConfigError(ValueError):
    """A record does not have the frozen evaluation configuration: the analysis is aborted."""


# --------------------------------------------------------------------------- statistics


#: Numerical zero of a standard error or of a difference ``Δ* − Δ̂`` (floating-point residue).
EPS = 1e-12


def _div(num: F, den: F) -> F:
    """Elementwise ``num / den`` with ``0/0 := 0`` and ``x/0 := ±∞``; ``|v| ≤ EPS`` is 0."""
    num = np.where(np.abs(num) <= EPS, 0.0, num)
    den = np.where(np.abs(den) <= EPS, 0.0, den)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.where(num == 0, 0.0, np.sign(num) * np.inf), num / den)
    return np.asarray(out, dtype=float)


def paired_ratio(yo: F, xo: F, yn: F, xn: F) -> tuple[F, F]:
    """``Δ = θ_new − θ_old`` and its paired delta-method SE over the last axis (seeds)."""
    n = yo.shape[-1]
    to = yo.sum(-1) / xo.sum(-1)
    tn = yn.sum(-1) / xn.sum(-1)
    uo = (yo - to[..., None] * xo) / xo.mean(-1)[..., None]
    un = (yn - tn[..., None] * xn) / xn.mean(-1)[..., None]
    se = (un - uo).std(-1, ddof=1) / math.sqrt(n)
    return np.asarray(tn - to, dtype=float), np.asarray(se, dtype=float)


def mean_of_pairs(arms: Sequence[tuple[F, F, F, F]]) -> tuple[F, F]:
    """The mean of several paired ratio differences on the same seeds and its SE."""
    n = arms[0][0].shape[-1]
    deltas, infl = [], []
    for yo, xo, yn, xn in arms:
        to = yo.sum(-1) / xo.sum(-1)
        tn = yn.sum(-1) / xn.sum(-1)
        deltas.append(tn - to)
        infl.append((yn - tn[..., None] * xn) / xn.mean(-1)[..., None]
                    - (yo - to[..., None] * xo) / xo.mean(-1)[..., None])  # fmt: skip
    d = np.mean(infl, axis=0)
    return np.asarray(np.mean(deltas, axis=0)), np.asarray(d.std(-1, ddof=1) / math.sqrt(n))


def boot_index(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> npt.NDArray[np.int64]:
    """The shared seed-resampling matrix ``(B, n)``."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


@dataclass(frozen=True)
class Inference:
    estimate: float
    se: float
    ci: tuple[float, float]
    ci_percentile: tuple[float, float]
    p: float
    p_mc_se: float
    q: float
    d_z: float | None
    n_seeds: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "se": self.se,
            "ci95": list(self.ci),
            "ci95_percentile": list(self.ci_percentile),
            "p": self.p,
            "p_mc_se": self.p_mc_se,
            "q_abs_t": self.q,
            "d_z": self.d_z,
            "n_seeds": self.n_seeds,
        }


def bootstrap_t(
    stat: Any, args: Sequence[F], idx: npt.NDArray[np.int64], alpha: float = ALPHA
) -> Inference:
    """Symmetric seed-cluster bootstrap-t for ``stat(*args) -> (Δ, SE)`` (seeds on axis −1)."""
    est_a, se_a = stat(*args)
    est, se = float(est_a), float(se_a)
    b_est, b_se = stat(*(a[..., idx] for a in args))
    t_star = np.abs(_div(np.asarray(b_est - est), np.asarray(b_se)))
    t_obs = abs(float(_div(np.asarray(est), np.asarray(se))))
    q = float(np.quantile(t_star, 1 - alpha, method=QUANTILE_METHOD))
    n_boot = idx.shape[0]
    p = (1 + int(np.sum(t_star >= t_obs))) / (n_boot + 1)
    if se <= EPS:
        se = 0.0
    half = q * se if se > 0 else 0.0
    lo, hi = np.quantile(b_est, [alpha / 2, 1 - alpha / 2])
    n = int(args[0].shape[-1])
    d_z = est / (se * math.sqrt(n)) if se > 0 else (0.0 if est == 0 else None)
    return Inference(
        estimate=est,
        se=se,
        ci=(est - half, est + half),
        ci_percentile=(float(lo), float(hi)),
        p=p,
        p_mc_se=math.sqrt(p * (1 - p) / n_boot),
        q=q,
        d_z=d_z,
        n_seeds=n,
    )


def holm(p: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (ties broken by position; monotone, capped at 1)."""
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------- data and gate


@dataclass
class Data:
    """Per run and arm, per field, an array (levels, seeds); and the gate report."""

    seeds: list[int]
    arrays: dict[str, dict[str, dict[str, F]]]
    evaluable: dict[str, bool]
    problems: dict[str, list[str]] = field(default_factory=dict)
    complete_seeds: dict[str, list[int]] = field(default_factory=dict)


def _is_count(v: Any) -> bool:
    return (isinstance(v, int) and not isinstance(v, bool) and v >= 0) or (
        isinstance(v, float) and math.isfinite(v) and v >= 0 and v.is_integer()
    )


def record_problem(r: Mapping[str, Any]) -> str | None:
    """Why an ``ok`` record cannot be used (``None`` when it is valid)."""
    for k in COUNTS:
        if not _is_count(r.get(k)):
            return f"bad count {k}={r.get(k)!r}"
    if r["delivered"] > r["released"]:
        return "delivered > released"
    if r["robots"] == 0 or r["steps"] == 0:
        return "no robots or no steps"
    if r["steps"] > r["max_steps"]:
        return "steps > max_steps"
    if r["terminated"] and r["delivered"] != r["released"]:
        return "terminated with undelivered tasks"
    br = r.get("base_return")
    if not isinstance(br, (int, float)) or isinstance(br, bool) or not math.isfinite(br):
        return f"bad base_return {br!r}"
    conf = r.get("conflicts")
    if not isinstance(conf, Mapping) or sum(conf.values()) != r["conflicts_total"]:
        return "conflict breakdown does not sum to conflicts_total"
    return None


def build(
    records: Iterable[Mapping[str, Any]],
    picks: Mapping[str, Mapping[str, str]],
    fingerprints: Mapping[str, Mapping[int, str]],
    seeds: Sequence[int] = tuple(BLOCK_C),
    levels: Sequence[str] = LEVELS,
    runs: Sequence[str] = RUNS,
) -> Data:
    """Apply the validity gate and arrange the records (see the module docstring)."""
    arm_of: dict[tuple[Any, Any], str] = {
        (run, ck): arm for run in runs for arm, ck in picks[run].items()
    }
    seen: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    problems: dict[str, list[str]] = {run: [] for run in runs}
    bad_cells: dict[str, set[tuple[str, int]]] = {run: set() for run in runs}
    seed_set, level_set = set(seeds), set(levels)
    for r in records:
        run, ck = r.get("label"), r.get("checkpoint")
        if (run, ck) not in arm_of:
            raise ConfigError(f"record of an unexpected run/checkpoint {run!r}/{ck!r}")
        for k, v in EXPECTED.items():
            if r.get(k) != v:
                raise ConfigError(f"{run}/{ck}: {k}={r.get(k)!r}, expected {v!r}")
        if r.get("level") not in level_set or r.get("seed") not in seed_set:
            raise ConfigError(f"{run}/{ck}: level/seed {r.get('level')}/{r.get('seed')}")
        assert isinstance(run, str)
        key = (run, arm_of[run, ck], str(r["level"]), int(r["seed"]))
        if key in seen:
            problems[run].append(f"duplicate {key}")
            bad_cells[run].add((key[2], key[3]))
            continue
        seen[key] = r
        if r["instance_sha256"] != fingerprints[key[2]][key[3]]:
            raise ConfigError(f"{key}: scenario instance differs from the pre-run check")
    for run in runs:
        for lvl in levels:
            for s in seeds:
                pair = [seen.get((run, arm, lvl, s)) for arm in ARMS]
                for arm, rr in zip(ARMS, pair, strict=True):
                    if rr is None:
                        problems[run].append(f"missing {arm}/{lvl}/{s}")
                    elif rr.get("status") != "ok":
                        problems[run].append(f"failed {arm}/{lvl}/{s}: {rr.get('error_type')}")
                    elif (why := record_problem(rr)) is not None:
                        problems[run].append(f"invalid {arm}/{lvl}/{s}: {why}")
                    else:
                        continue
                    bad_cells[run].add((lvl, s))
                if all(r is not None for r in pair):
                    a, b = pair
                    assert a is not None and b is not None
                    diff = [k for k in PAIRED if a.get(k) != b.get(k)]
                    if diff:
                        raise ConfigError(f"{run}/{lvl}/{s}: old and new differ in {diff}")
    si = {s: j for j, s in enumerate(seeds)}
    li = {lv: i for i, lv in enumerate(levels)}
    arrays: dict[str, dict[str, dict[str, F]]] = {}
    for run in runs:
        arrays[run] = {}
        for arm in ARMS:
            a = {k: np.zeros((len(levels), len(seeds))) for k in FIELDS}
            for (rn, am, lvl, s), r in seen.items():
                if rn != run or am != arm or r.get("status") != "ok" or record_problem(r):
                    continue
                i, j = li[lvl], si[s]
                a["delivered"][i, j] = r["delivered"]
                a["released"][i, j] = r["released"]
                a["conflicts"][i, j] = r["conflicts_total"]
                a["robot_steps"][i, j] = r["robots"] * r["steps"]
                a["base_return"][i, j] = r["base_return"]
                a["episodes"][i, j] = 1.0
            arrays[run][arm] = a
    bad_seeds = {run: {s for _, s in cells} for run, cells in bad_cells.items()}
    return Data(
        seeds=list(seeds),
        arrays=arrays,
        evaluable={run: not problems[run] for run in runs},
        problems=problems,
        complete_seeds={run: [s for s in seeds if s not in bad_seeds[run]] for run in runs},
    )


# --------------------------------------------------------------------------- analysis


def _args(
    data: Data, run: str, outcome: str, levels: Sequence[int], cols: npt.NDArray[np.int64]
) -> list[F]:
    num, den, scale = OUTCOMES[outcome]
    out = []
    for arm in ARMS:
        a = data.arrays[run][arm]
        out += [scale * a[num][list(levels)][:, cols].sum(0), a[den][list(levels)][:, cols].sum(0)]
    return [out[0], out[1], out[2], out[3]]


def _theta(y: F, x: F) -> float:
    return float(y.sum() / x.sum())


def analyse(
    data: Data, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = ALPHA
) -> dict[str, Any]:
    """The frozen analysis: primary and secondary families, per-level, across-run mean.

    An evaluable run uses all seeds. A run that is not evaluable is analysed on its complete
    seeds (descriptive only) and enters Holm with ``p = 1``. The across-run mean uses the seeds
    complete for every run. Each seed count has its own index matrix from ``default_rng(seed)``
    (with all runs evaluable there is one, shared by everything).
    """
    runs = list(data.arrays)
    pos = {s: j for j, s in enumerate(data.seeds)}
    cols = {
        run: np.array([pos[s] for s in data.complete_seeds[run]], dtype=np.int64) for run in runs
    }
    shared = sorted(set.intersection(*(set(data.complete_seeds[r]) for r in runs)))
    cols_all = np.array([pos[s] for s in shared], dtype=np.int64)
    cache: dict[int, npt.NDArray[np.int64]] = {}

    def ix(n: int) -> npt.NDArray[np.int64]:
        if n not in cache:
            cache[n] = boot_index(n, n_boot, seed)
        return cache[n]

    all_levels = list(range(len(LEVELS)))

    def one(run: str, outcome: str, lv: Sequence[int]) -> dict[str, Any]:
        yo, xo, yn, xn = _args(data, run, outcome, lv, cols[run])
        inf = bootstrap_t(paired_ratio, [yo, xo, yn, xn], ix(len(cols[run])), alpha)
        old, new = _theta(yo, xo), _theta(yn, xn)
        return {
            "old": old,
            "new": new,
            "relative_change": (new - old) / old if old != 0 else None,
            **inf.as_dict(),
        }

    families: dict[str, list[dict[str, Any]]] = {"primary": [], "secondary": []}
    for fam, outs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for run in runs:
            for outcome in outs:
                res = one(run, outcome, all_levels)
                res.update(run=run, outcome=outcome, evaluable=data.evaluable[run])
                res["p_for_holm"] = res["p"] if data.evaluable[run] else 1.0
                families[fam].append(res)
        adj = holm([h["p_for_holm"] for h in families[fam]])
        for h, a in zip(families[fam], adj, strict=True):
            h["p_holm"] = a
            h["reject"] = a <= alpha

    per_level = []
    for run in runs:
        for li, lvl in enumerate(LEVELS):
            for outcome in (*PRIMARY, *SECONDARY):
                res = one(run, outcome, [li])
                del res["p"], res["p_mc_se"]
                res.update(run=run, level=lvl, outcome=outcome)
                per_level.append(res)

    across = []
    for outcome in (*PRIMARY, *SECONDARY):
        flat = [x for run in runs for x in _args(data, run, outcome, all_levels, cols_all)]

        def stat(*a: F) -> tuple[F, F]:
            groups = [
                (a[4 * k], a[4 * k + 1], a[4 * k + 2], a[4 * k + 3]) for k in range(len(runs))
            ]
            return mean_of_pairs(groups)

        inf = bootstrap_t(stat, flat, ix(len(cols_all)), alpha).as_dict()
        del inf["p"], inf["p_mc_se"]
        across.append({"outcome": outcome, **inf})

    return {
        "families": families,
        "per_level": per_level,
        "across_run_mean": across,
        "evaluable": data.evaluable,
        "n_seeds_used": {run: len(cols[run]) for run in runs},
        "problems": {k: v[:50] for k, v in data.problems.items()},
        "n_problems": {k: len(v) for k, v in data.problems.items()},
        "config": {
            "alpha": alpha,
            "n_boot": n_boot,
            "boot_seed": seed,
            "quantile_method": QUANTILE_METHOD,
            "n_seeds": len(data.seeds),
            "levels": list(LEVELS),
            "runs": runs,
        },
    }


def safety(records: Iterable[Mapping[str, Any]], picks: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:  # fmt: skip
    """Descriptive safety and validity counts per run × arm × level (and ``all`` levels)."""
    arm_of: dict[tuple[Any, Any], str] = {
        (run, ck): arm for run, p in picks.items() for arm, ck in p.items()
    }
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        arm = arm_of.get((r.get("label"), r.get("checkpoint")))
        if arm is None:
            continue
        for lvl in (r["level"], "all"):
            k = (r["label"], arm, lvl)
            row = rows.setdefault(k, {
                "run": k[0], "arm": arm, "checkpoint": r["checkpoint"], "level": lvl,
                "episodes": 0, "completed": 0, "failed": 0, "error_types": {},
                "invalid_records": 0, "physical_collisions": 0, "invalid_moves_INV": 0,
                "terminated_all_delivered": 0, "horizon_with_undelivered": 0,
                "horizon_all_delivered": 0, "undelivered_tasks": 0,
                "conflict_types": {}, "mean_flow_time_mean": None, "_flow": [],
            })  # fmt: skip
            row["episodes"] += 1
            if r.get("status") != "ok":
                row["failed"] += 1
                et = str(r.get("error_type"))
                row["error_types"][et] = row["error_types"].get(et, 0) + 1
                continue
            row["completed"] += 1
            if record_problem(r) is not None:
                row["invalid_records"] += 1
                continue
            row["physical_collisions"] += r["physical_collisions"]
            row["invalid_moves_INV"] += r["conflicts"].get("INV", 0)
            und = r["released"] - r["delivered"]
            row["undelivered_tasks"] += und
            if r["terminated"]:
                row["terminated_all_delivered"] += 1
            elif und > 0:
                row["horizon_with_undelivered"] += 1
            else:
                row["horizon_all_delivered"] += 1
            for ct, v in r["conflicts"].items():
                row["conflict_types"][ct] = row["conflict_types"].get(ct, 0) + v
            if r.get("mean_flow_time") is not None:
                row["_flow"].append(r["mean_flow_time"])
    out = []
    for row in rows.values():
        flow = row.pop("_flow")
        row["mean_flow_time_mean"] = float(np.mean(flow)) if flow else None
        out.append(row)
    order = {lv: i for i, lv in enumerate((*LEVELS, "all"))}
    return sorted(out, key=lambda r: (r["run"], ARMS.index(r["arm"]), order[r["level"]]))
