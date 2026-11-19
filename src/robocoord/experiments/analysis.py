"""Summaries of the planner experiments (E1a/b/c, E2) as Markdown tables, computed only from
stored run records::

    python -m robocoord.experiments.analysis results/experiments/e1b_neighbor_index

Every number in the documentation's results register comes from these summaries.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.experiments.records import RecordStore
from robocoord.experiments.stats import (
    bootstrap_ci,
    loglog_fit,
    paired_bootstrap,
    polylog_fit,
    rank_biserial,
)

Record = dict[str, Any]


def _f(x: float | None, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and abs(x) >= 1e4:
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _group(records: Iterable[Record], key: Callable[[Record], Any]) -> dict[Any, list[Record]]:
    out: dict[Any, list[Record]] = defaultdict(list)
    for r in records:
        out[key(r)].append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0]) if kv[0] is not None else ""))


def _median_iqr(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    q1, q2, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return f"{q2 * 1e3:.1f} [{q1 * 1e3:.1f}, {q3 * 1e3:.1f}]"


def _fit_str(x: Sequence[float], y: Sequence[float]) -> str:
    fit = loglog_fit(x, y)
    return f"{_f(fit.exponent)} [{_f(fit.low)}, {_f(fit.high)}]"


def outcomes(records: Sequence[Record]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r["outcome"]] += 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


# ====================================================================== E1b


def _per(counters: list[dict[str, Any]], key: str, denom: str, offset: int = 0) -> str:
    """Mean of ``(counter − offset) / denominator`` over runs; ``n/a`` if the counter is absent
    (records written before the counter existed). The offset (the root insertion) is only
    subtracted from non-zero counters, so indexes without rebuilds report 0."""
    if not counters or any(key not in x for x in counters):
        return "n/a"
    vals = [
        max(x[key] - offset, 0) / max(x[denom] - (1 if denom == "tree_size" else 0), 1)
        for x in counters
    ]
    return _f(float(np.mean(vals)), 2)


def summarize_e1b(records: Sequence[Record]) -> str:
    lines = [f"Runs: {len(records)} ({outcomes(records)}).", ""]
    rows = []
    for (family, n_max, index), rs in _group(
        records, lambda r: (r["params"]["family"], r["params"]["n_max"], r["params"]["index"])
    ).items():
        c = [r["metrics"]["counters"] for r in rs]
        ok = sum(r["outcome"] == "ok" for r in rs)
        rows.append(
            [
                family,
                n_max,
                index,
                f"{ok}/{len(rs)}",
                _median_iqr([r["metrics"]["time_s"] for r in rs]),
                _per(c, "nn_visited", "nn_queries"),
                _per(c, "nn_nodes", "nn_queries"),
                _per(c, "near_k", "radius_queries"),
                _per(c, "near_kprime", "radius_queries"),
                _per(c, "range_nodes", "radius_queries"),
                _per(c, "range_nodes_tested", "radius_queries"),
                _per(c, "propagation", "iterations"),
                # the root is inserted before the first iteration: exclude it (offset 1)
                _per(c, "rebuild_points", "tree_size", 1),
                _per(c, "rebuild_work", "tree_size", 1),
            ]
        )
    lines.append(
        _table(
            [
                "family",
                "n_max",
                "index",
                "success",
                "time ms median [IQR]",
                "NN points/query",
                "NN nodes/query",
                "k/query",
                "k′/query",
                "range nodes expanded/query",
                "range nodes examined/query",
                "s/iteration",
                "rebuild pts/insertion",
                "rebuild work/insertion",
            ],
            rows,
        )
    )
    lines += [
        "",
        "Counter definitions (`planning/kdtree.py`): *NN points* = points whose distance was "
        "evaluated; *NN nodes* = tree nodes examined, including internal and pruned nodes; *range "
        "nodes expanded* = nodes whose box intersects the query square; *range nodes examined* = "
        "including pruned nodes; *rebuild pts* = points placed into newly built static trees; "
        "*rebuild work* = points processed by those builds (Σ subtree sizes). Tree counters are 0 "
        "for the brute-force index.",
    ]
    # identical trees with both indexes
    by_query = _group(
        records,
        lambda r: (
            r["params"]["family"],
            r["params"]["map"],
            r["params"]["pair"],
            r["params"]["n_max"],
        ),
    )
    same = sum(
        1
        for rs in by_query.values()
        if len(rs) == 2
        and rs[0]["metrics"].get("length") == rs[1]["metrics"].get("length")
        and rs[0]["metrics"]["counters"]["near_k"] == rs[1]["metrics"]["counters"]["near_k"]
    )
    lines += [
        "",
        f"Index-independence check: {same}/{len(by_query)} query × budget pairs built "
        "identical trees with both indexes (same Σk and same path length).",
    ]
    # exponent fits over the tree size n, pooled from the largest-budget traces
    lines += [
        "",
        "Empirical exponents b in `quantity ≈ a·n^b` (log–log fit over the per-iteration "
        "counters binned by tree size n, n ≥ 64, largest-budget runs; 95 % bootstrap CI). "
        "Rebuild quantities are per insertion (bin mean divided by the bin's insertion rate):",
        "",
    ]
    fields = (
        "nn_visited",
        "nn_nodes",
        "range_nodes",
        "range_tested",
        "k",
        "kprime",
        "checks",
        "s",
        "rebuild",
        "rebuild_work",
    )
    per_insert = {"rebuild", "rebuild_work"}
    rows = []
    top = max(x["params"]["n_max"] for x in records)
    for (family, index), rs in _group(
        [r for r in records if r["params"]["n_max"] == top],
        lambda r: (r["params"]["family"], r["params"]["index"]),
    ).items():
        xs: dict[str, list[float]] = defaultdict(list)
        ys: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            tb = r["metrics"].get("trace_bins") or {}
            for i, n in enumerate(tb.get("n_mean", [])):
                if n < 64:
                    continue
                for f in fields:
                    if f not in tb:
                        continue
                    v = tb[f][i]
                    if f in per_insert:
                        added = tb["added"][i]
                        if added <= 0:
                            continue
                        v = v / added
                    xs[f].append(n)
                    ys[f].append(v)
        rows.append([family, index] + [_fit_str(xs[f], ys[f]) if xs[f] else "n/a" for f in fields])
    lines.append(
        _table(
            [
                "family",
                "index",
                "NN points",
                "NN nodes",
                "range expanded",
                "range examined",
                "k",
                "k′",
                "segment checks",
                "s",
                "rebuild pts/insert",
                "rebuild work/insert",
            ],
            rows,
        )
    )
    # direct comparison of power-law and polylog fits for the k-d nearest-neighbour work
    rows = []
    for (family, _index), rs in _group(
        [r for r in records if r["params"]["n_max"] == top and r["params"]["index"] == "kdtree"],
        lambda r: (r["params"]["family"], r["params"]["index"]),
    ).items():
        for f in ("nn_nodes", "nn_visited"):
            pts = [
                (n, tb[f][i])
                for r in rs
                for tb in [r["metrics"].get("trace_bins") or {}]
                if f in tb
                for i, n in enumerate(tb["n_mean"])
                if n >= 64
            ]
            if not pts:
                continue
            nx, ny = [p[0] for p in pts], [p[1] for p in pts]
            pw, pl = loglog_fit(nx, ny), polylog_fit(nx, ny)
            rows.append(
                [
                    family,
                    f,
                    f"{_f(pw.exponent)} [{_f(pw.low)}, {_f(pw.high)}]",
                    _f(pw.rss, 3),
                    f"{_f(pl.exponent)} [{_f(pl.low)}, {_f(pl.high)}]",
                    _f(pl.rss, 3),
                ]
            )
    if rows:
        lines += [
            "",
            "k-d nearest-neighbour work: power law `a·n^b` vs polylog `a·(ln n)^c` (both "
            "two-parameter least-squares fits on log y over the same binned points, n ≥ 64; "
            "RSS = residual sum of squares, lower is better). A better fit over this finite "
            "range describes the data; it does not prove an asymptotic bound:",
            "",
            _table(
                [
                    "family",
                    "counter",
                    "power b [95 % CI]",
                    "RSS power",
                    "polylog c [95 % CI]",
                    "RSS polylog",
                ],
                rows,
            ),
        ]
    # total time vs n_max
    rows = []
    for (family, index), rs in _group(
        records, lambda r: (r["params"]["family"], r["params"]["index"])
    ).items():
        rows.append(
            [
                family,
                index,
                _fit_str([r["params"]["n_max"] for r in rs], [r["metrics"]["time_s"] for r in rs]),
            ]
        )
    lines += [
        "",
        "Total time vs iteration budget N_it (exponent of `time ≈ a·N_it^b`):",
        "",
        _table(["family", "index", "b [95 % CI]"], rows),
    ]
    return "\n".join(lines)


def summarize_memory(records: Sequence[Record]) -> str:
    lines = [f"Runs: {len(records)} ({outcomes(records)}).", ""]
    rows = []
    for (family, n_max, index), rs in _group(
        records, lambda r: (r["params"]["family"], r["params"]["n_max"], r["params"]["index"])
    ).items():
        mem = [r["metrics"]["peak_memory_bytes"] / 2**20 for r in rs]
        per_node = [
            r["metrics"]["peak_memory_bytes"] / max(r["metrics"]["counters"]["tree_size"], 1)
            for r in rs
        ]
        rows.append(
            [
                family,
                n_max,
                index,
                _f(float(np.median(mem)), 2),
                _f(float(np.max(mem)), 2),
                _f(float(np.median(per_node)), 0),
            ]
        )
    lines.append(
        _table(
            [
                "family",
                "n_max",
                "index",
                "peak MiB (median)",
                "peak MiB (max)",
                "bytes per tree node (median)",
            ],
            rows,
        )
    )
    sizes: dict[str, list[float]] = defaultdict(list)
    peaks: dict[str, list[float]] = defaultdict(list)
    for r in records:
        sizes[r["params"]["index"]].append(r["metrics"]["counters"]["tree_size"])
        peaks[r["params"]["index"]].append(r["metrics"]["peak_memory_bytes"])
    lines += [
        "",
        "Peak memory vs tree size n (exponent b of `peak ≈ a·n^b`; O(n) predicts 1):",
        "",
        _table(
            ["index", "b [95 % CI]"], [[k, _fit_str(sizes[k], peaks[k])] for k in sorted(sizes)]
        ),
    ]
    return "\n".join(lines)


# ====================================================================== E1a


def summarize_e1a(records: Sequence[Record]) -> str:
    lines = [f"Runs: {len(records)} ({outcomes(records)}).", ""]
    rows = []
    for (family, size, planner), rs in _group(
        records, lambda r: (r["params"]["family"], r["params"]["size"], r["params"]["planner"])
    ).items():
        ok = [r for r in rs if r["outcome"] == "ok"]
        gaps = [r["metrics"]["step_gap"] for r in ok]
        lens = [
            r["metrics"]["length"] / r["metrics"]["euclid_lower_bound"]
            for r in ok
            if r["metrics"]["euclid_lower_bound"] > 0
        ]
        work = (
            [r["metrics"]["counters"]["expansions"] / r["metrics"]["free_cells"] for r in rs]
            if planner == "astar"
            else [
                r["metrics"]["counters"]["tree_size"]
                / max(r["metrics"]["counters"]["iterations"], 1)
                for r in rs
            ]
        )
        rows.append(
            [
                family,
                size,
                planner,
                f"{len(ok)}/{len(rs)}",
                sum(r["outcome"] == "timeout" for r in rs),
                _median_iqr([r["metrics"]["time_s"] for r in rs]),
                _f(float(np.mean(gaps))) if gaps else "n/a",
                _f(float(np.max(gaps))) if gaps else "n/a",
                _f(float(np.mean(lens))) if lens else "n/a",
                _f(float(np.mean(work)), 3),
            ]
        )
    lines.append(
        _table(
            [
                "family",
                "size",
                "planner",
                "success",
                "timeouts",
                "time ms median [IQR]",
                "mean steps/BFS",
                "max steps/BFS",
                "mean length/Euclid LB",
                "A*: expansions/free cells · RRT*: nodes/iteration",
            ],
            rows,
        )
    )
    # scaling exponents
    rows = []
    for (family, planner), rs in _group(
        records, lambda r: (r["params"]["family"], r["params"]["planner"])
    ).items():
        v = [r["metrics"]["free_cells"] for r in rs]
        if planner == "astar":
            work = [r["metrics"]["counters"]["expansions"] for r in rs]
            rows.append(
                [
                    family,
                    planner,
                    _fit_str(v, work),
                    _fit_str(v, [r["metrics"]["time_s"] for r in rs]),
                ]
            )
        else:
            rows.append([family, planner, "—", _fit_str(v, [r["metrics"]["time_s"] for r in rs])])
    lines += [
        "",
        "Scaling with the number of free cells V (exponent b of `y ≈ a·V^b`, 95 % CI). "
        "A* bound: expansions ≤ V, time O(V log V) worst case.",
        "",
        _table(["family", "planner", "A* expansions vs V", "time vs V"], rows),
    ]
    # success by family and budget (pooled sizes)
    rows = []
    for (family, planner), rs in _group(
        records, lambda r: (r["params"]["family"], r["params"]["planner"])
    ).items():
        by_size = {
            s: sum(r["outcome"] == "ok" for r in rs if r["params"]["size"] == s)
            for s in (16, 32, 64, 128)
        }
        rows.append([family, planner] + [f"{by_size[s]}/100" for s in (16, 32, 64, 128)])
    lines += [
        "",
        "Success (status ok within the budget) per size:",
        "",
        _table(["family", "planner", "16", "32", "64", "128"], rows),
    ]
    return "\n".join(lines)


# ====================================================================== E1c


def summarize_e1c(records: Sequence[Record]) -> str:
    lines = [f"Runs: {len(records)} ({outcomes(records)}).", ""]
    rows = []
    for family, rs in _group(records, lambda r: r["params"]["family"]).items():
        cps = sorted({int(k) for r in rs for k in r["metrics"].get("checkpoints", {})})
        for cp in cps:
            ratios_lb = []
            ratios_final = []
            solved = 0
            for r in rs:
                v = r["metrics"]["checkpoints"].get(str(cp))
                if v is None or v == "inf":
                    continue
                solved += 1
                ratios_lb.append(v / r["metrics"]["euclid_lower_bound"])
                final = r["metrics"].get("planner_cost")
                if final:
                    ratios_final.append(v / final)
            radius = rs[0]["metrics"]["radius_at"][str(cp)]
            rows.append(
                [
                    family,
                    cp,
                    f"{solved}/{len(rs)}",
                    _f(radius, 2),
                    _f(float(np.mean(ratios_lb)), 4) if ratios_lb else "n/a",
                    _f(float(np.max(ratios_lb)), 4) if ratios_lb else "n/a",
                    _f(float(np.mean(ratios_final)), 4) if ratios_final else "n/a",
                ]
            )
    lines.append(
        _table(
            [
                "family",
                "iteration n",
                "solved",
                "r_n",
                "mean cost / Euclid LB",
                "max cost / Euclid LB",
                "mean cost / final cost",
            ],
            rows,
        )
    )
    first = [r["metrics"]["counters"]["first_solution_iteration"] for r in records]
    lines += ["", f"First solution iteration: median {np.median(first):.0f}, max {max(first)}."]
    return "\n".join(lines)


# ====================================================================== E2

#: Conflicts per 100 robot-steps from which an episode is listed as a persistent-conflict episode.
PERSISTENT_CONFLICTS = 50.0


def _conflict_summary(values: Sequence[float]) -> str:
    """``mean / median / share > 0`` of per-episode conflict rates (heavy-tailed distribution)."""
    if not values:
        return "n/a"
    share = sum(v > 0 for v in values) / len(values)
    return f"{_f(float(np.mean(values)))} / {_f(float(np.median(values)))} / {share:.2f}"


def _persistent_conflicts_table(records: Sequence[Record]) -> str:
    rows = []
    hits = [
        r
        for r in records
        if r.get("metrics") and r["metrics"]["conflicts_per_100"] >= PERSISTENT_CONFLICTS
    ]
    for r in sorted(
        hits,
        key=lambda r: (
            r["params"]["team"],
            r["params"]["planner"],
            r["params"]["obstacles"],
            r["params"]["seed"],
            r["params"]["events"],
        ),
    ):
        p, m = r["params"], r["metrics"]
        kinds = {k: v for k, v in m["conflicts"].items() if v}
        dominant = max(kinds, key=lambda k: kinds[k]) if kinds else "—"
        rows.append(
            [
                p["planner"],
                p["team"],
                p["obstacles"],
                p["events"],
                p["seed"],
                _f(m["conflicts_per_100"], 1),
                dominant,
                _f(m["completion_rate"] or 0.0),
            ]
        )
    if not rows:
        return "none"
    return _table(
        [
            "planner",
            "team",
            "K",
            "events",
            "seed",
            "conflicts /100",
            "dominant kind",
            "completion",
        ],
        rows,
    )


def summarize_e2(records: Sequence[Record]) -> str:
    lines = [f"Episodes: {len(records)} ({outcomes(records)}).", ""]
    rows = []
    for (team, events, k, planner), rs in _group(
        records,
        lambda r: (
            r["params"]["team"],
            r["params"]["events"],
            r["params"]["obstacles"],
            r["params"]["planner"],
        ),
    ).items():
        m = [r["metrics"] for r in rs if r["metrics"]]
        comp = bootstrap_ci([x["completion_rate"] or 0.0 for x in m], n_boot=2000)
        calls = [x["planning"]["calls"] for x in m]
        trig: dict[str, float] = defaultdict(float)
        status: dict[str, float] = defaultdict(float)
        for x in m:
            for t, v in x["planning"]["calls_by_trigger"].items():
                trig[t] += v / len(m)
            tot = max(x["planning"]["calls"], 1)
            for s, v in x["planning"]["status_mix"].items():
                status[s] += v / tot / len(m)
        lat = [x["latency_s"]["median"] for x in m if x["latency_s"]["median"] is not None]
        p95 = [x["latency_s"]["p95"] for x in m if x["latency_s"]["p95"] is not None]
        ratio = [
            x["planning"]["leg_steps_over_reference"]
            for x in m
            if x["planning"]["leg_steps_over_reference"] is not None
        ]
        rows.append(
            [
                team,
                events,
                k,
                planner,
                f"{_f(comp.value)} [{_f(comp.low)}, {_f(comp.high)}]",
                _f(float(np.mean(calls)), 1),
                " ".join(f"{t}:{trig[t]:.1f}" for t in ("T1", "T2", "T3", "T4", "T5", "T6")),
                f"{status['ok']:.2f}/{status['ok_degraded']:.2f}/{status['unreachable']:.2f}/"
                f"{status['budget_exhausted']:.2f}",
                f"{np.median(lat) * 1e3:.2f}" if lat else "n/a",
                f"{np.median(p95) * 1e3:.2f}" if p95 else "n/a",
                _f(float(np.mean(ratio))) if ratio else "n/a",
                _conflict_summary([x["conflicts_per_100"] for x in m]),
            ]
        )
    lines.append(
        _table(
            [
                "team",
                "events",
                "K",
                "planner",
                "completion [95 % CI]",
                "calls/episode",
                "mean calls by trigger",
                "status ok/degraded/unreachable/budget",
                "latency ms (median)",
                "latency ms (p95)",
                "leg steps / first ref. length",
                "conflicts /100 robot-steps: mean / median / share of episodes > 0",
            ],
            rows,
        )
    )
    lines += [
        "",
        "Conflict rates are heavy-tailed: a few deadlocked episodes can dominate a cell mean. "
        "The median and the share of episodes with any conflict are therefore reported next to "
        "the mean.",
        "",
        f"Persistent-conflict episodes (≥ {PERSISTENT_CONFLICTS:g} proposed conflicts per 100 "
        "robot-steps, i.e. on at least every second robot-step on average):",
        "",
        _persistent_conflicts_table(records),
    ]
    # paired A* vs RRT* (same seed, K, events, team)
    pairs: dict[tuple[Any, ...], dict[str, Record]] = defaultdict(dict)
    for r in records:
        p = r["params"]
        pairs[(p["team"], p["events"], p["obstacles"], p["seed"])][p["planner"]] = r
    rows = []
    for team in sorted({k[0] for k in pairs}):
        a, b, la, lb = [], [], [], []
        for key, d in pairs.items():
            if key[0] != team or "astar" not in d or "rrtstar" not in d:
                continue
            ma, mb = d["astar"]["metrics"], d["rrtstar"]["metrics"]
            a.append(ma["completion_rate"] or 0.0)
            b.append(mb["completion_rate"] or 0.0)
            la.append(ma["latency_s"]["total"] / max(ma["steps"], 1))
            lb.append(mb["latency_s"]["total"] / max(mb["steps"], 1))
        d1 = paired_bootstrap(a, b)
        d2 = paired_bootstrap(la, lb)
        rows.append(
            [
                team,
                len(a),
                f"{_f(d1.value, 3)} [{_f(d1.low, 3)}, {_f(d1.high, 3)}]",
                _f(rank_biserial(a, b)),
                f"{d2.value * 1e3:.2f} [{d2.low * 1e3:.2f}, {d2.high * 1e3:.2f}]",
            ]
        )
    lines += [
        "",
        "Paired differences A* − RRT* over identical instances (paired bootstrap 95 % "
        "CI, 10 000 resamples; rank-biserial effect size):",
        "",
        _table(
            [
                "team",
                "pairs",
                "completion A*−RRT*",
                "rank-biserial",
                "planning ms per step A*−RRT*",
            ],
            rows,
        ),
    ]
    return "\n".join(lines)


def _issue4(name: str) -> Callable[[Sequence[Record]], str]:
    def run(records: Sequence[Record]) -> str:
        from robocoord.experiments import allocation_analysis as a

        fn: Callable[[Sequence[Record]], str] = getattr(a, name)
        return fn(records)

    return run


SUMMARIES: dict[str, Callable[[Sequence[Record]], str]] = {
    "e1a_planners_static": summarize_e1a,
    "e1b_neighbor_index": summarize_e1b,
    "e1b_memory": summarize_memory,
    "e1c_convergence": summarize_e1c,
    "e2_replanning": summarize_e2,
    "e3a_assignment": _issue4("summarize_e3a"),
    "e3b_priority": _issue4("summarize_e3b"),
    "e5_prelim": _issue4("summarize_e5_prelim"),
    "e5_prelim_reference": _issue4("summarize_e5_prelim"),
    "e5_recovery": _issue4("summarize_e5_prelim"),
    "e5_recovery_reference": _issue4("summarize_e5_prelim"),
}


def summarize(directory: str | Path) -> str:
    directory = Path(directory)
    records = RecordStore(directory).read()
    # the summary is chosen by the experiment name recorded in spec.json, so that a re-run written
    # to another directory (e.g. ``e1b_neighbor_index_r2``) is summarised the same way
    spec_file = directory / "spec.json"
    name = (
        json.loads(spec_file.read_text(encoding="utf-8")).get("name", directory.name)
        if spec_file.exists()
        else directory.name
    )
    if name not in SUMMARIES:
        raise SystemExit(f"no summary for {name!r}; known: {sorted(SUMMARIES)}")
    text = f"## {directory.name}\n\n" + SUMMARIES[name](records) + "\n"
    (directory / "summary.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":  # pragma: no cover
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for d in sys.argv[1:]:
        print(summarize(d))
