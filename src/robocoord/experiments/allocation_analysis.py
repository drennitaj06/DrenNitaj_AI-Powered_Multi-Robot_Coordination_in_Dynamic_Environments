"""Summaries of the Issue #4 experiments (E3a, E3b, preliminary E5) as Markdown, computed only
from stored run records (registered in :mod:`robocoord.experiments.analysis`)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from robocoord.experiments.stats import (
    Estimate,
    bootstrap_ci,
    holm,
    loglog_fit,
    paired_bootstrap,
    rank_biserial,
    wilcoxon_p,
)

Record = dict[str, Any]
Metrics = dict[str, Any]
METHODS = ("auction_scaling", "auction_fixed1", "sequential", "hungarian")


def _f(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and abs(x) >= 1e4:
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def _ci(e: Estimate, nd: int = 2) -> str:
    return f"{_f(e.value, nd)} [{_f(e.low, nd)}, {_f(e.high, nd)}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _outcomes(records: Sequence[Record]) -> str:
    counts: dict[str, int] = {}
    for r in records:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def _paired(x: Sequence[float], y: Sequence[float]) -> tuple[Estimate, float, float]:
    return paired_bootstrap(x, y), wilcoxon_p(x, y), rank_biserial(x, y)


# ====================================================================== E3a

#: The theoretical bound compared with each measured exponent (05 §B.3: named bound and type).
E3A_BOUNDS = {
    "auction_scaling": "time O(k·A·log(kC)) = O(k³ log kC) for dense matrices, worst case "
    "(Bertsekas & Eckstein 1988, cited); bids O(k² log kC)",
    "auction_fixed1": "bids ≤ k(⌊C/ε⌋+1)+1 (derived in `04` §4.1, Guaranteed), i.e. O(kC); "
    "time O(k²C) worst case",
    "sequential": "time O(F log F) = O(k² log k) for F = k² feasible pairs (one sort and a scan)",
    "hungarian": "SciPy linear_sum_assignment, O(k³) worst case",
}


def _e3a_guarantees(ok: Sequence[Record]) -> list[list[object]]:
    rows: list[list[object]] = []
    for method in METHODS:
        for c in (100, 10000):
            sub: list[Metrics] = [
                r["metrics"]
                for r in ok
                if r["params"]["method"] == method and r["params"]["C"] == c
            ]
            if not sub:
                continue
            optimal = sum(1 for m in sub if m["gap"] == 0)
            card = sum(1 for m in sub if m["cardinality"] == m["max_cardinality"])
            bids_ok = all(m["bids"] <= m["k"] * (m["C"] + 1) + 1 for m in sub)
            rows.append(
                [
                    method,
                    c,
                    f"{optimal}/{len(sub)}",
                    max(m["gap"] for m in sub),
                    max(m["gap"] / m["k"] for m in sub),
                    _f(min(m["ratio"] for m in sub), 6),
                    "yes" if all(m["gap"] <= m["k"] for m in sub) else "NO",
                    "yes" if all(2 * m["value"] >= m["opt"] for m in sub) else "NO",
                    f"{card}/{len(sub)}",
                    ("yes" if bids_ok else "NO") if method == "auction_fixed1" else "—",
                ]
            )
    return rows


def _e3a_scaling(ok: Sequence[Record], metric: str, ks: Sequence[int]) -> list[list[object]]:
    rows: list[list[object]] = []
    for method in METHODS:
        if metric == "bids" and method == "hungarian":
            continue
        for c in (100, 10000):
            cells: list[str] = []
            xs: list[float] = []
            ys: list[float] = []
            for k in ks:
                vals = [
                    float(r["metrics"][metric])
                    for r in ok
                    if r["params"]["method"] == method
                    and r["params"]["C"] == c
                    and r["params"]["k"] == k
                ]
                med = float(np.median(vals)) if vals else float("nan")
                cells.append(_f(med * 1e3, 2) if metric == "time_s" else _f(med, 0))
                xs += [float(k)] * len(vals)
                ys += vals
            fit_all = loglog_fit(xs, ys)
            big = [(x, y) for x, y in zip(xs, ys, strict=True) if x >= 50]
            fit_big = loglog_fit([x for x, _ in big], [y for _, y in big])
            rows.append(
                [
                    method,
                    c,
                    *cells,
                    f"{_f(fit_all.exponent)} [{_f(fit_all.low)}, {_f(fit_all.high)}]",
                    f"{_f(fit_big.exponent)} [{_f(fit_big.low)}, {_f(fit_big.high)}]",
                ]
            )
    return rows


def summarize_e3a(records: Sequence[Record]) -> str:
    ok = [r for r in records if r["outcome"] == "done"]
    ks = sorted({int(r["params"]["k"]) for r in ok})
    header = [
        "method",
        "C",
        "optimal",
        "max gap",
        "max gap / k",
        "min V/OPT",
        "gap ≤ k",
        "2V ≥ OPT",
        "max cardinality",
        "bids ≤ k(C+1)+1",
    ]
    lines = [
        f"Solves: {len(records)} ({_outcomes(records)}).",
        "",
        "**Optimality and guarantees** (integer objective; OPT from the Hungarian oracle):",
        "",
        _table(header, _e3a_guarantees(ok)),
        "",
        "**Work and time vs k** (median over the 20 matrices; time = median of 5 repeats):",
        "",
    ]
    scale_header = ["method", "C", *[f"k={k}" for k in ks], "exponent, all k", "exponent, k ≥ 50"]
    for metric, label in (("time_s", "time (ms)"), ("bids", "bids")):
        lines += [f"*{label}*:", "", _table(scale_header, _e3a_scaling(ok, metric, ks)), ""]
    lines += [
        "**Bounds compared** (05 §B.3: a measured exponent below a worst-case bound does not "
        "contradict it; exponents describe the measured range only and are not proofs):",
        "",
    ]
    lines += [f"- `{m}`: {b}" for m, b in E3A_BOUNDS.items()]
    return "\n".join(lines)


# ====================================================================== E3b


def _level(m: Metrics, key: str, level: str) -> float | None:
    value = (m.get(key) or {}).get(level)
    return None if value is None else float(value)


Getter = Callable[[Metrics], float | None]


def _get(key: str) -> Getter:
    def get(m: Metrics) -> float | None:
        v = m.get(key)
        return None if v is None else float(v)

    return get


def _reassignments(m: Metrics) -> float:
    return float(m["allocation"]["preemptions"] + m["allocation"]["lost"])


#: The E3b family: (comparison, other variant, metric, getter). Lower is better for all.
E3B_FAMILY: list[tuple[str, str, str, Getter]] = [
    ("κ > 0 vs κ = 0", "kappa0", "F_w (priority-weighted flow)", _get("weighted_flow_time")),
    (
        "κ > 0 vs κ = 0",
        "kappa0",
        "flow time, top level 3",
        lambda m: _level(m, "flow_time_by_priority", "3"),
    ),
    ("κ > 0 vs κ = 0", "kappa0", "mean flow time (price)", _get("mean_flow_time")),
    (
        "κ > 0 vs κ = 0",
        "kappa0",
        "travel per delivered task (price)",
        _get("travel_per_delivered_task"),
    ),
    (
        "α > 0 vs α = 0",
        "alpha0",
        "waiting max, level 1",
        lambda m: _level(m, "waiting_max_by_priority", "1"),
    ),
    (
        "α > 0 vs α = 0",
        "alpha0",
        "waiting p95, level 1",
        lambda m: _level(m, "waiting_p95_by_priority", "1"),
    ),
    ("h > 0 vs h = 0", "h0", "re-assignments (pre-emptions + lost)", _reassignments),
    ("h > 0 vs h = 0", "h0", "travel per delivered task", _get("travel_per_delivered_task")),
]


def _e3b_variant_rows(
    by: dict[tuple[str, int], Metrics], seeds: Sequence[int]
) -> list[list[object]]:
    rows: list[list[object]] = []
    for variant in ("default", "kappa0", "alpha0", "h0"):
        ms = [by[(variant, s)] for s in seeds if (variant, s) in by]
        if not ms:
            continue  # variant not (yet) run, e.g. in a partial or smoke run
        flow_w = [float(m["weighted_flow_time"]) for m in ms if m["weighted_flow_time"] is not None]
        top = [v for m in ms if (v := _level(m, "flow_time_by_priority", "3")) is not None]
        low_wait = [v for m in ms if (v := _level(m, "waiting_max_by_priority", "1")) is not None]
        rows.append(
            [
                variant,
                f"{sum(m['tasks_delivered'] for m in ms)}/{sum(m['tasks_released'] for m in ms)}",
                _f(float(np.mean([m["completion_rate"] for m in ms])), 3),
                sum(m["waiting_censored"] for m in ms),
                _ci(bootstrap_ci(flow_w), 1),
                _ci(bootstrap_ci(top), 1),
                _ci(bootstrap_ci(low_wait), 1),
                _f(float(np.mean([m["utilisation"] for m in ms])), 2),
            ]
        )
    return rows


def summarize_e3b(records: Sequence[Record]) -> str:
    ok = [r for r in records if r["outcome"] == "done"]
    by: dict[tuple[str, int], Metrics] = {
        (str(r["params"]["variant"]), int(r["params"]["seed"])): r["metrics"] for r in ok
    }
    seeds = sorted({s for _, s in by})
    results = []
    for comparison, other, label, get in E3B_FAMILY:
        xs: list[float] = []
        ys: list[float] = []
        for s in seeds:
            if ("default", s) not in by or (other, s) not in by:
                continue
            a, b = get(by[("default", s)]), get(by[(other, s)])
            if a is not None and b is not None:
                xs.append(a)
                ys.append(b)
        est, p, r = _paired(xs, ys)
        results.append((comparison, label, len(xs), est, p, r))
    adjusted = holm([res[4] for res in results])
    rows: list[list[object]] = []
    for (comparison, label, n, est, p, r), p_adj in zip(results, adjusted, strict=True):
        direction = "lower (better)" if est.high < 0 else "higher (worse)" if est.low > 0 else ""
        if not direction:
            verdict = "no clear difference"
        elif p_adj < 0.05:
            verdict = f"default {direction}: CI excludes 0, Holm p < 0.05"
        else:
            verdict = f"default {direction} by CI only; Holm p ≥ 0.05 (weak)"
        rows.append([comparison, label, n, _ci(est), _f(p, 4), _f(p_adj, 4), _f(r), verdict])
    variant_header = [
        "variant",
        "delivered/released",
        "completion",
        "censored",
        "F_w",
        "flow level 3",
        "wait max level 1",
        "utilisation",
    ]
    return "\n".join(
        [
            f"Episodes: {len(records)} ({_outcomes(records)}); paired over {len(seeds)} seeds.",
            "",
            "**Per variant** (mean, bootstrap 95 % CI over seeds; flow times over delivered "
            "tasks):",
            "",
            _table(variant_header, _e3b_variant_rows(by, seeds)),
            "",
            "**Paired comparisons**, d = default − variant (negative: default lower; lower is "
            "better for every metric): paired bootstrap 95 % CI (10 000 resamples), two-sided "
            "Wilcoxon signed-rank p, Holm-adjusted over the family of 8, matched-pairs "
            "rank-biserial r (positive: default larger). The verdict states whether the CI "
            "excludes 0 and whether the Holm-adjusted p is below 0.05; both are needed for a "
            "clear difference.",
            "",
            _table(
                [
                    "comparison",
                    "metric",
                    "pairs",
                    "mean d [95 % CI]",
                    "Wilcoxon p",
                    "Holm p",
                    "r",
                    "verdict",
                ],
                rows,
            ),
        ]
    )


# ====================================================================== preliminary E5

CONFIGS: tuple[tuple[str, str, str], ...] = tuple(
    (p, a, d)
    for p in ("astar", "rrtstar")
    for a in ("auction", "sequential")
    for d in ("static", "obstacles", "full")
)


def _cfg(r: Record) -> tuple[str, str, str]:
    p = r["params"]
    return (str(p["planner"]), str(p["allocator"]), str(p["dynamics"]))


def _controller(records: Sequence[Record]) -> str:
    names = {str(r["metrics"].get("controller", "baseline")) for r in records}
    return ", ".join(sorted(names)) or "n/a"


def _e5_cell_rows(ok: Sequence[Record]) -> list[list[object]]:
    rows: list[list[object]] = []
    for cfg in CONFIGS:
        ms: list[Metrics] = [r["metrics"] for r in ok if _cfg(r) == cfg]
        if not ms:
            continue
        released = sum(m["tasks_released"] for m in ms)
        delivered = sum(m["tasks_delivered"] for m in ms)
        incomplete = sum(1 for m in ms if m["tasks_delivered"] < m["tasks_released"])
        flows = [float(m["mean_flow_time"]) for m in ms if m["mean_flow_time"] is not None]
        detours = [m["deadlock"]["detours_started"] for m in ms if m.get("deadlock")]
        rows.append(
            [
                *cfg,
                len(ms),
                _ci(bootstrap_ci([m["completion_rate"] for m in ms]), 3),
                f"{delivered}/{released} = {_f(delivered / released, 4)}",
                f"{incomplete}/{len(ms)}",
                _ci(bootstrap_ci(flows), 1),
                sum(m["conflicts"]["RRv"] + m["conflicts"]["RRs"] for m in ms),
                sum(m["conflicts"]["ROv"] + m["conflicts"]["ROs"] for m in ms),
                sum(m["physical_collisions"] for m in ms),
                sum(detours) if detours else "—",
                _f(float(np.median([m["wall_s_per_step"] for m in ms])) * 1e3, 1),
            ]
        )
    return rows


E5_HEADER = [
    "planner",
    "allocator",
    "dynamics",
    "episodes",
    "completion, mean of episodes [95 % CI]",
    "pooled delivered/released",
    "incomplete episodes",
    "mean flow [95 % CI]",
    "RR conflicts",
    "RO conflicts",
    "physical",
    "detours",
    "ms/step (median)",
]


def summarize_e5_prelim(records: Sequence[Record]) -> str:
    ok = [r for r in records if r["outcome"] == "done"]
    released = sum(r["metrics"]["tasks_released"] for r in ok)
    delivered = sum(r["metrics"]["tasks_delivered"] for r in ok)
    incomplete = sum(
        1 for r in ok if r["metrics"]["tasks_delivered"] < r["metrics"]["tasks_released"]
    )
    by: dict[tuple[str, str, str, int], Metrics] = {
        (*_cfg(r), int(r["params"]["seed"])): r["metrics"] for r in ok
    }
    seeds = sorted({k[3] for k in by})
    results = []
    for planner in ("astar", "rrtstar"):
        for dyn in ("static", "obstacles", "full"):
            for label, key in (("completion", "completion_rate"), ("mean flow", "mean_flow_time")):
                xs: list[float] = []
                ys: list[float] = []
                for s in seeds:
                    a = by.get((planner, "auction", dyn, s), {}).get(key)
                    b = by.get((planner, "sequential", dyn, s), {}).get(key)
                    if a is not None and b is not None:
                        xs.append(float(a))
                        ys.append(float(b))
                est, p, r = _paired(xs, ys)
                results.append((planner, dyn, label, len(xs), est, p, r))
    adjusted = holm([res[5] for res in results])
    pair_rows = [
        [pl, dyn, label, n, _ci(est, 3), _f(p, 4), _f(pa, 4), _f(r)]
        for (pl, dyn, label, n, est, p, r), pa in zip(results, adjusted, strict=True)
    ]
    recovery = [r["metrics"]["endpoint"] for r in ok if r["metrics"].get("endpoint")]
    recovery_line = (
        "Endpoint recovery: on. Counters (sums): "
        + ", ".join(f"{k} {sum(int(e.get(k, 0)) for e in recovery)}" for k in RECOVERY_COUNTERS)
        if recovery
        else "Endpoint recovery: off."
    )
    return "\n".join(
        [
            f"**Preliminary E5 slice, controller: {_controller(ok)} — not the full E5 "
            "experiment.**",
            recovery_line,
            f"Episodes: {len(records)} ({_outcomes(records)}). Pooled completion "
            f"{delivered}/{released} = {_f(delivered / released if released else None, 4)}; "
            f"incomplete episodes {incomplete}/{len(ok)}. 'Completion, mean of episodes' "
            "averages the per-episode rate delivered/released (episodes weighted equally); "
            "'pooled' divides total delivered by total released tasks.",
            "",
            _table(E5_HEADER, _e5_cell_rows(ok)),
            "",
            "**Paired auction − sequential** (same seeds; Holm over the family of 12; "
            "exploratory):",
            "",
            _table(
                [
                    "planner",
                    "dynamics",
                    "metric",
                    "pairs",
                    "mean d [95 % CI]",
                    "Wilcoxon p",
                    "Holm p",
                    "r",
                ],
                pair_rows,
            ),
        ]
    )


def compare_e5_controllers(baseline: Sequence[Record], reference: Sequence[Record]) -> str:
    """Paired baseline vs reference follower over identical (planner, allocator, dynamics,
    seed) instances. Each controller is reported separately; differences are d = baseline −
    reference per instance, with paired bootstrap CIs, Wilcoxon and Holm over the family."""
    b = {
        (*_cfg(r), int(r["params"]["seed"])): r["metrics"]
        for r in baseline
        if r["outcome"] == "done"
    }
    f = {
        (*_cfg(r), int(r["params"]["seed"])): r["metrics"]
        for r in reference
        if r["outcome"] == "done"
    }
    common = sorted(set(b) & set(f))
    side: list[list[object]] = []
    for cfg in CONFIGS:
        keys = [k for k in common if k[:3] == cfg]
        if not keys:
            continue
        for name, src in (("baseline", b), ("reference", f)):
            ms = [src[k] for k in keys]
            rel = sum(m["tasks_released"] for m in ms)
            dlv = sum(m["tasks_delivered"] for m in ms)
            rs = sum(m["robot_moves"] for m in ms)
            side.append(
                [
                    *cfg,
                    name,
                    f"{dlv}/{rel} = {_f(dlv / rel, 4)}",
                    f"{sum(1 for m in ms if m['tasks_delivered'] < m['tasks_released'])}/{len(ms)}",
                    _f(float(np.mean([m["conflicts_per_100"] for m in ms])), 2),
                    sum(m["conflicts"]["RRv"] + m["conflicts"]["RRs"] for m in ms),
                    sum(m["conflicts"]["ROv"] + m["conflicts"]["ROs"] for m in ms),
                    sum(m["physical_collisions"] for m in ms),
                    rs,
                ]
            )
    metrics: list[tuple[str, str]] = [
        ("completion", "completion_rate"),
        ("proposed conflicts per 100 robot-steps", "conflicts_per_100"),
    ]
    results = []
    for cfg in CONFIGS:
        for label, key in metrics:
            xs: list[float] = []
            ys: list[float] = []
            for k in common:
                if k[:3] != cfg:
                    continue
                x, y = b[k].get(key), f[k].get(key)
                if x is not None and y is not None:
                    xs.append(float(x))
                    ys.append(float(y))
            if xs:
                est, p, r = _paired(xs, ys)
                results.append((cfg, label, len(xs), est, p, r))
    adjusted = holm([res[4] for res in results])
    rows = [
        [*cfg, label, n, _ci(est, 3), _f(p, 4), _f(pa, 4), _f(r)]
        for (cfg, label, n, est, p, r), pa in zip(results, adjusted, strict=True)
    ]
    return "\n".join(
        [
            "## Preliminary E5 — baseline vs reference follower (paired; not the full E5)",
            "",
            f"Paired instances: {len(common)} (baseline records {len(b)}, reference records "
            f"{len(f)}). Baseline = right-of-way baseline with the deadlock resolver; reference "
            "= reference follower without it (documented defaults); shield off for both.",
            "",
            "**Per controller** (pooled over the 50 paired seeds of each configuration; robot "
            "moves = total cells moved):",
            "",
            _table(
                [
                    "planner",
                    "allocator",
                    "dynamics",
                    "controller",
                    "pooled delivered/released",
                    "incomplete episodes",
                    "conflicts / 100 robot-steps (mean)",
                    "RR conflicts",
                    "RO conflicts",
                    "physical",
                    "robot moves",
                ],
                side,
            ),
            "",
            "**Paired differences** d = baseline − reference (per instance); paired bootstrap 95 % "
            f"CI, Wilcoxon, Holm over the family of {len(results)}, rank-biserial r:",
            "",
            _table(
                [
                    "planner",
                    "allocator",
                    "dynamics",
                    "metric",
                    "pairs",
                    "mean d [95 % CI]",
                    "Wilcoxon p",
                    "Holm p",
                    "r",
                ],
                rows,
            ),
        ]
    )


#: Endpoint-recovery counters reported by :func:`compare_e5_recovery` (``04`` §5.6).
RECOVERY_COUNTERS: tuple[str, ...] = (
    "blockages_temporary",
    "blockages_persistent",
    "retreats_started",
    "retreats_succeeded",
    "retreats_timed_out",
    "retreats_aborted",
    "retreats_cancelled",
    "attempts_exhausted",
    "no_retreat_cell",
    "goal_reached_after_recovery",
    "parking_needed",
    "parking_started",
    "parking_completed",
    "parking_cancelled",
    "parking_aborted",
    "no_parking_cell",
    "parking_blocked_dynamic",
    "unresolved_obstacle_at_end",
    "unresolved_idle_robot_at_end",
    "idle_on_endpoint_at_end",
)


def compare_e5_recovery(off: Sequence[Record], on: Sequence[Record]) -> str:
    """Paired no-recovery vs endpoint-recovery runs of one controller over identical (planner,
    allocator, dynamics, seed) instances. d = recovery − no recovery per instance; paired
    bootstrap CIs, Wilcoxon, Holm over the family. Metric definitions are unchanged: mean flow
    time is still averaged over *delivered* tasks only (so it can rise when recovery delivers
    tasks that waited long)."""
    a = {(*_cfg(r), int(r["params"]["seed"])): r["metrics"] for r in off if r["outcome"] == "done"}
    b = {(*_cfg(r), int(r["params"]["seed"])): r["metrics"] for r in on if r["outcome"] == "done"}
    common = sorted(set(a) & set(b))
    side: list[list[object]] = []
    totals = {"off": [0, 0, 0], "on": [0, 0, 0]}
    for cfg in CONFIGS:
        keys = [k for k in common if k[:3] == cfg]
        if not keys:
            continue
        row: list[object] = list(cfg)
        for name, src in (("off", a), ("on", b)):
            ms = [src[k] for k in keys]
            rel = sum(m["tasks_released"] for m in ms)
            dlv = sum(m["tasks_delivered"] for m in ms)
            inc = sum(1 for m in ms if m["tasks_delivered"] < m["tasks_released"])
            totals[name][0] += dlv
            totals[name][1] += rel
            totals[name][2] += inc
            flows = [float(m["mean_flow_time"]) for m in ms if m["mean_flow_time"] is not None]
            row += [
                f"{dlv}/{rel} = {_f(dlv / rel, 4)}",
                f"{inc}/{len(ms)}",
                _f(float(np.mean(flows)) if flows else None, 1),
            ]
        row.append(sum(m["physical_collisions"] for m in (b[k] for k in keys)))
        side.append(row)
    results = []
    for cfg in CONFIGS:
        for label, key in (("completion", "completion_rate"), ("mean flow", "mean_flow_time")):
            xs: list[float] = []
            ys: list[float] = []
            for k in common:
                if k[:3] != cfg:
                    continue
                x, y = b[k].get(key), a[k].get(key)
                if x is not None and y is not None:
                    xs.append(float(x))
                    ys.append(float(y))
            if xs:
                est, p, r = _paired(xs, ys)
                results.append((cfg, label, len(xs), est, p, r))
    adjusted = holm([res[4] for res in results])
    rows = [
        [*cfg, label, n, _ci(est, 3), _f(p, 4), _f(pa, 4), _f(r)]
        for (cfg, label, n, est, p, r), pa in zip(results, adjusted, strict=True)
    ]
    counters = {
        c: sum(int((b[k].get("endpoint") or {}).get(c, 0)) for k in common)
        for c in RECOVERY_COUNTERS
    }
    controller = _controller([r for r in on if r["outcome"] == "done"])
    t_off, t_on = totals["off"], totals["on"]
    return "\n".join(
        [
            f"## Preliminary E5 — no recovery vs endpoint recovery, controller {controller} "
            "(paired; not the full E5)",
            "",
            f"Paired instances: {len(common)}. Pooled completion: no recovery "
            f"{t_off[0]}/{t_off[1]} = {_f(t_off[0] / t_off[1] if t_off[1] else None, 4)} "
            f"({t_off[2]} incomplete episodes); recovery {t_on[0]}/{t_on[1]} = "
            f"{_f(t_on[0] / t_on[1] if t_on[1] else None, 4)} ({t_on[2]} incomplete episodes).",
            "",
            _table(
                [
                    "planner",
                    "allocator",
                    "dynamics",
                    "off: pooled",
                    "off: incomplete",
                    "off: mean flow",
                    "on: pooled",
                    "on: incomplete",
                    "on: mean flow",
                    "on: physical",
                ],
                side,
            ),
            "",
            "**Endpoint recovery counters** (sums over the recovery runs; definitions in `04` "
            "§5.6): " + ", ".join(f"{k} {v}" for k, v in counters.items()),
            "",
            "**Paired differences** d = recovery − no recovery (per instance); paired bootstrap "
            f"95 % CI, Wilcoxon, Holm over the family of {len(results)}, rank-biserial r:",
            "",
            _table(
                [
                    "planner",
                    "allocator",
                    "dynamics",
                    "metric",
                    "pairs",
                    "mean d [95 % CI]",
                    "Wilcoxon p",
                    "Holm p",
                    "r",
                ],
                rows,
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m robocoord.experiments.allocation_analysis compare-e5 <baseline dir>
    <reference dir> <out.md>``: writes the paired controller comparison;
    ``compare-recovery <no-recovery dir> <recovery dir> <out.md>`` the paired recovery
    comparison of one controller."""
    import sys
    from pathlib import Path

    from robocoord.experiments.records import RecordStore

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4 or args[0] not in ("compare-e5", "compare-recovery"):
        print(
            "usage: compare-e5 <baseline dir> <reference dir> <out.md> | "
            "compare-recovery <no-recovery dir> <recovery dir> <out.md>"
        )
        return 2
    compare = compare_e5_controllers if args[0] == "compare-e5" else compare_e5_recovery
    text = compare(RecordStore(args[1]).read(), RecordStore(args[2]).read())
    Path(args[3]).write_text(text + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
