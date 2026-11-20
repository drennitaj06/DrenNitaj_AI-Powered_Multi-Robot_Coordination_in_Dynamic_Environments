"""Project overview figures, Issues #1-#6 (post-Issue-6 research phase).

Every figure is computed from the stored experiment records (never from summary tables). The
records are only read. Output: one SVG per figure, one CSV table view per figure and
``SOURCES.md`` (every input file with its SHA-256, and how each figure is computed).

Usage (from the repository root)::

    python tools/figures/overview.py --out results/figures/overview

Uncertainty: 95 % percentile bootstrap intervals (2 000 resamples, seed 0) over episodes or
queries, as stated per figure. Pooled completion = delivered / released over the resampled
episodes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from svgplot import AXIS, NEUTRAL, SERIES, TEXT_2, Axes, Figure

R = Path("results")
EXP = R / "experiments"
TRAIN = R / "training"
DEV5 = R / "dev5" / "comparison" / "records.jsonl"
DEV6 = R / "dev6"
N_BOOT = 2000

ALGO = {"madql": SERIES[0], "mappo": SERIES[1], "ippo": SERIES[2]}
ALGO_NAME = {"madql": "MADQL", "mappo": "MAPPO", "ippo": "IPPO"}
COMP = {  # non-learned comparators: neutral ink, identity by shape and label
    "baseline": (NEUTRAL[0], "square", False, "Baseline (resolver on)"),
    "baseline-noresolver": (NEUTRAL[0], "diamond", False, "Matched baseline (resolver off)"),
    "reference": (NEUTRAL[1], "triangle", False, "Reference follower"),
}
POLICIES = [f"{a}-seed{s}" for a in ("madql", "mappo", "ippo") for s in (1, 2, 3)]
LEVELS = ("static", "obstacles", "full")


# ------------------------------------------------------------------------------ io helpers
class Sources:
    def __init__(self) -> None:
        self.used: dict[str, set[str]] = defaultdict(set)
        self.hashes: dict[str, str] = {}
        self.text: dict[str, str] = {}

    def read(self, fig: str, path: Path) -> list[dict[str, Any]]:
        key = path.as_posix()
        self.used[fig].add(key)
        if key not in self.hashes:
            self.hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def read_csv(self, fig: str, path: Path) -> list[dict[str, str]]:
        key = path.as_posix()
        self.used[fig].add(key)
        if key not in self.hashes:
            self.hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def describe(self, fig: str, text: str) -> None:
        self.text[fig] = text


def write_table(out: Path, name: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in r])


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def boot(
    values: Sequence[float], stat: Callable[[np.ndarray], float] = np.mean
) -> tuple[float, float, float]:  # type: ignore[assignment]
    a = np.asarray(values, dtype=float)
    g = rng()
    idx = g.integers(0, len(a), size=(N_BOOT, len(a)))
    bs = np.array([stat(a[i]) for i in idx])
    return float(stat(a)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio(num: Sequence[float], den: Sequence[float]) -> tuple[float, float, float]:
    n, d = np.asarray(num, float), np.asarray(den, float)
    g = rng()
    idx = g.integers(0, len(n), size=(N_BOOT, len(n)))
    bs = n[idx].sum(1) / d[idx].sum(1)
    return float(n.sum() / d.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_paired(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    return boot(np.asarray(x, float) - np.asarray(y, float))


def save(out: Path, name: str, fig: Figure) -> None:
    (out / f"{name}.svg").write_text(fig.svg(), encoding="utf-8")
    print("wrote", name)


def label_of(p: str) -> str:
    a, s = p.split("-seed")
    return f"{ALGO_NAME[a]} {s}"


# ------------------------------------------------------------------------------ fig 01
def fig01_planners(out: Path, src: Sources) -> None:
    name = "fig01_planners_static_queries"
    recs = src.read(name, EXP / "e1a_planners_static_r3" / "records.jsonl")
    fams = ["open", "d10", "d20", "d30", "d35", "rooms", "warehouse"]
    fam_lab = ["open", "10 %", "20 %", "30 %", "35 %", "rooms", "ware-\nhouse"]
    planners = ["astar", "rrtstar-500", "rrtstar-2000", "rrtstar-5000"]
    pcol = {"astar": NEUTRAL[0], "rrtstar-500": SERIES[0], "rrtstar-2000": SERIES[1],
            "rrtstar-5000": SERIES[2]}  # fmt: skip
    plab = {"astar": "A*", "rrtstar-500": "RRT* n=500", "rrtstar-2000": "RRT* n=2 000",
            "rrtstar-5000": "RRT* n=5 000"}  # fmt: skip
    ok: dict[tuple[str, str], list[float]] = defaultdict(list)
    tm: dict[tuple[str, int], list[float]] = defaultdict(list)
    exp_ratio: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in recs:
        p, m = r["params"], r["metrics"]
        ok[(p["planner"], p["family"])].append(1.0 if m["status"] == "ok" else 0.0)
        tm[(p["planner"], int(p["size"]))].append(1000 * m["time_s"])
        if p["planner"] == "astar" and m["status"] == "ok" and m["path_steps"] > 0:
            exp_ratio[(p["family"], int(p["size"]))].append(
                m["counters"]["expansions"] / m["path_steps"]
            )
    fig = Figure(1180, 470, "Fig. 1 — A* and RRT* on 2 800 static queries (E1a, Issue #3)",
                 "Success within the iteration budget, planning time per query, and A* search "
                 "effort relative to the path length", legend=[(plab[p], pcol[p], "square" if p == "astar" else "circle") for p in planners])  # fmt: skip
    a = fig.add(Axes(80, 120, 300, 260, ylim=(0, 1.02), xcats=fam_lab, title="(a) Success rate by map family",
                     ylabel="success rate", xlabel="map family (density or structure)"))  # fmt: skip
    rows = []
    for j, p in enumerate(planners):
        off = (j - 1.5) * 0.17
        for i, f in enumerate(fams):
            v, lo, hi = boot(ok[(p, f)])
            a.vrange(i + off, lo, hi, pcol[p], width=1.5)
            a.point(i + off, v, pcol[p], r=3.6, shape="square" if p == "astar" else "circle",
                    title=f"{plab[p]}, {f}: {v:.3f} [{lo:.3f}, {hi:.3f}], n={len(ok[(p, f)])}")  # fmt: skip
            rows.append(["success", p, f, "", v, lo, hi, len(ok[(p, f)])])
    sizes = [16, 32, 64, 128]
    b = fig.add(Axes(470, 120, 300, 260, xlim=(13, 150), ylim=(0.05, 20000), xlog=True, ylog=True,
                     xticks=sizes, title="(b) Median time per query", xlabel="map side (cells)",
                     ylabel="milliseconds (log)"))  # fmt: skip
    for p in planners:
        med = [statistics.median(tm[(p, s)]) for s in sizes]
        b.line(sizes, med, pcol[p], title=plab[p])
        for s, v in zip(sizes, med, strict=True):
            b.point(s, v, pcol[p], r=3.6, shape="square" if p == "astar" else "circle",
                    title=f"{plab[p]}, {s}×{s}: median {v:.3g} ms (n={len(tm[(p, s)])})")  # fmt: skip
            rows.append(["median_ms", p, "", s, v, "", "", len(tm[(p, s)])])
    c = fig.add(Axes(860, 120, 300, 260, ylim=(0.8, 30), ylog=True, xcats=fam_lab,
                     title="(c) A* expansions per path step",
                     ylabel="expansions ÷ path length (log; 1 = ideal)", xlabel="map family"))  # fmt: skip
    for j, s in enumerate((32, 64, 128)):
        off = (j - 1) * 0.22
        for i, f in enumerate(fams):
            vals = exp_ratio[(f, s)]
            v = statistics.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            col = (SERIES[3], SERIES[6], SERIES[4])[j]
            c.vrange(i + off, float(q1), float(q3), col, width=1.5)
            c.point(
                i + off,
                v,
                col,
                r=3.4,
                title=f"{f}, {s}×{s}: median {v:.1f}, IQR [{q1:.1f}, {q3:.1f}], n={len(vals)}",
            )
            rows.append(
                ["astar_expansions_per_step", "astar", f, s, v, float(q1), float(q3), len(vals)]
            )
    c.hline(1, AXIS)
    fig.legend += [(f"A* effort, {s}×{s}", col, "circle")
                   for s, col in zip((32, 64, 128), (SERIES[3], SERIES[6], SERIES[4]), strict=True)]  # fmt: skip
    fig.notes = ["Error bars: (a) 95 % bootstrap CI over queries; (c) interquartile range. A* solved every query "
                 "(status ok in 2 800/2 800). Source: results/experiments/e1a_planners_static_r3/records.jsonl"]  # fmt: skip
    save(out, name, fig)
    write_table(
        out, name, ["quantity", "planner", "family", "size", "value", "low", "high", "n"], rows
    )
    src.describe(name, "Success = status 'ok' per query. Time = metrics.time_s. A* effort = "
                 "counters.expansions / path_steps for successful A* queries, by family and size.")  # fmt: skip


# ------------------------------------------------------------------------------ fig 02
def fig02_replanning(out: Path, src: Sources) -> None:
    name = "fig02_replanning"
    recs = src.read(name, EXP / "e2_replanning_r2" / "records.jsonl")
    ks = [0, 4, 8, 16]
    lat: dict[tuple[str, int], list[float]] = defaultdict(list)
    comp: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in recs:
        p, m = r["params"], r["metrics"]
        if m["latency_s"]["calls"]:
            lat[(p["planner"], int(p["obstacles"]))].append(1000 * m["latency_s"]["median"])
        if int(p["team"]) == 4:
            comp[(p["planner"], int(p["obstacles"]))].append(
                (m["tasks_delivered"], m["tasks_released"])
            )
    col = {"astar": NEUTRAL[0], "rrtstar": SERIES[0]}
    lab = {"astar": "A*", "rrtstar": "RRT* (n=2 000)"}
    fig = Figure(820, 440, "Fig. 2 — Reactive replanning (E2, Issue #3)",
                 "Per-call planning latency and 4-robot completion with the reference follower, "
                 "by number of moving obstacles K", legend=[(lab[p], col[p], "square" if p == "astar" else "circle") for p in col])  # fmt: skip
    a = fig.add(Axes(80, 120, 300, 250, xcats=[str(k) for k in ks], ylim=(0.05, 1000), ylog=True,
                     title="(a) Median latency per planner call", xlabel="moving obstacles K",
                     ylabel="ms per call (log)"))  # fmt: skip
    rows = []
    for j, p in enumerate(col):
        for i, k in enumerate(ks):
            v, lo, hi = boot(lat[(p, k)], np.median)  # type: ignore[arg-type]
            off = (j - 0.5) * 0.25
            a.vrange(i + off, lo, hi, col[p], width=1.5)
            a.point(i + off, v, col[p], shape="square" if p == "astar" else "circle",
                    title=f"{lab[p]}, K={k}: median of episode medians {v:.3g} ms [{lo:.3g}, {hi:.3g}]")  # fmt: skip
            rows.append(["latency_ms", p, k, v, lo, hi, len(lat[(p, k)])])
    b = fig.add(Axes(490, 120, 300, 250, xcats=[str(k) for k in ks], ylim=(0.5, 1.0),
                     title="(b) Completion, 4 robots", xlabel="moving obstacles K",
                     ylabel="delivered ÷ released"))  # fmt: skip
    for j, p in enumerate(col):
        for i, k in enumerate(ks):
            d = comp[(p, k)]
            v, lo, hi = boot_ratio([x[0] for x in d], [x[1] for x in d])
            off = (j - 0.5) * 0.25
            b.vrange(i + off, lo, hi, col[p], width=1.5)
            b.point(
                i + off,
                v,
                col[p],
                shape="square" if p == "astar" else "circle",
                title=f"{lab[p]}, K={k}: {v:.3f} [{lo:.3f}, {hi:.3f}], {len(d)} episodes",
            )
            rows.append(["completion_team4", p, k, v, lo, hi, len(d)])
    fig.notes = ["95 % bootstrap CIs over episodes (both event rates pooled). Source: results/experiments/"
                 "e2_replanning_r2/records.jsonl (test seeds; Issue #3 experiment, read only)."]  # fmt: skip
    save(out, name, fig)
    write_table(out, name, ["quantity", "planner", "K", "value", "low", "high", "n"], rows)
    src.describe(name, "(a) metrics.latency_s.median per episode with ≥1 call, then the median over "
                 "episodes. (b) pooled tasks_delivered / tasks_released, 4-robot episodes.")  # fmt: skip


# ------------------------------------------------------------------------------ fig 03
def fig03_allocation(out: Path, src: Sources) -> None:
    name = "fig03_assignment"
    recs = src.read(name, EXP / "e3a_assignment" / "records.jsonl")
    ks = [5, 10, 20, 50, 100, 200, 400]
    meth = ["hungarian", "auction_scaling", "auction_fixed1", "sequential"]
    col = {"hungarian": NEUTRAL[0], "auction_scaling": SERIES[0], "auction_fixed1": SERIES[1],
           "sequential": SERIES[2]}  # fmt: skip
    lab = {"hungarian": "Hungarian (SciPy)", "auction_scaling": "Auction, ε-scaling",
           "auction_fixed1": "Auction, fixed ε = 1", "sequential": "Sequential auction"}  # fmt: skip
    t: dict[tuple[str, int], list[float]] = defaultdict(list)
    ratio: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in recs:
        p, m = r["params"], r["metrics"]
        if int(p["C"]) != 10_000:
            continue
        t[(p["method"], int(p["k"]))].append(1000 * m["time_s"])
        ratio[(p["method"], int(p["k"]))].append(m["ratio"])
    fig = Figure(820, 450, "Fig. 3 — Assignment algorithms (E3a, Issue #4)",
                 "Cost range C = 10⁴, 20 random matrices per size k",
                 legend=[(lab[x], col[x], "square" if x == "hungarian" else "circle") for x in meth])  # fmt: skip
    a = fig.add(Axes(80, 125, 300, 250, xlim=(4, 500), ylim=(0.001, 100000), xlog=True, ylog=True,
                     xticks=ks, title="(a) Median solve time", xlabel="robots = tasks (k)",
                     ylabel="milliseconds (log)"))  # fmt: skip
    rows = []
    for x in meth:
        med = [statistics.median(t[(x, k)]) for k in ks]
        a.line(ks, med, col[x], title=lab[x])
        for k, v in zip(ks, med, strict=True):
            a.point(
                k,
                v,
                col[x],
                r=3.4,
                shape="square" if x == "hungarian" else "circle",
                title=f"{lab[x]}, k={k}: median {v:.3g} ms",
            )
            rows.append(["median_ms", x, k, v, "", "", len(t[(x, k)])])
    b = fig.add(Axes(490, 125, 300, 250, xlim=(4, 500), ylim=(0.5, 1.005), xlog=True, xticks=ks,
                     title="(b) Value ÷ optimum", xlabel="robots = tasks (k)",
                     ylabel="assignment value ÷ optimal value"))  # fmt: skip
    for x in ("auction_scaling", "auction_fixed1", "sequential"):
        mins = [min(ratio[(x, k)]) for k in ks]
        meds = [statistics.median(ratio[(x, k)]) for k in ks]
        b.line(ks, meds, col[x], title=f"{lab[x]} (median)")
        b.line(ks, mins, col[x], dash="4 3", width=1.5, title=f"{lab[x]} (minimum)")
        for k, v, mn in zip(ks, meds, mins, strict=True):
            b.point(k, v, col[x], r=3.2, title=f"{lab[x]}, k={k}: median {v:.4f}, min {mn:.4f}")
            rows.append(["ratio", x, k, v, mn, "", len(ratio[(x, k)])])
    b.hline(0.5, AXIS, label="sequential guarantee: ≥ 0.5")
    fig.notes = ["(b) solid line = median, dashed = minimum over the 20 matrices; the ε-scaling auction is optimal on every matrix (ratio 1) and its line lies under the fixed-ε line. Source: results/experiments/"
                 "e3a_assignment/records.jsonl (C = 10 000). Times: single-threaded Python, except SciPy."]  # fmt: skip
    save(out, name, fig)
    write_table(out, name, ["quantity", "method", "k", "value", "min", "", "n"], rows)
    src.describe(name, "metrics.time_s and metrics.ratio (value / optimum) for C = 10 000.")


# ------------------------------------------------------------------------------ fig 04
def fig04_system(out: Path, src: Sources) -> None:
    name = "fig04_nonlearned_system"
    recs = (src.read(name, EXP / "e5_recovery" / "records.jsonl")
            + src.read(name, EXP / "e5_recovery_reference" / "records.jsonl"))  # fmt: skip
    cats = [(d, al) for d in LEVELS for al in ("auction", "sequential")]
    short = {"static": "stat", "obstacles": "obst", "full": "full"}
    clab = [f"{short[d]}\n{al[:3]}" for d, al in cats]
    g: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        p, m = r["params"], r["metrics"]
        g[(m["controller"], p["planner"], p["dynamics"], p["allocator"])].append(m)
    ctrls = [
        ("baseline", NEUTRAL[0], "square", "Baseline + endpoint recovery"),
        ("reference", SERIES[3], "triangle", "Reference follower + endpoint recovery"),
    ]
    fig = Figure(1180, 520, "Fig. 4 — Non-learned system comparison (Issue #4, 50 test seeds per cell)",
                 "Completion and proposed conflicts by dynamics level, allocator and planner",
                 legend=[(c[3], c[1], c[2]) for c in ctrls])  # fmt: skip
    rows = []
    for col_i, planner in enumerate(("astar", "rrtstar")):
        a = fig.add(Axes(80 + col_i * 560, 120, 220, 290, xcats=clab, ylim=(0.6, 1.01),
                         title=f"({'ac'[col_i]}) Completion, {'A*' if planner == 'astar' else 'RRT*'}",
                         ylabel="delivered ÷ released"))  # fmt: skip
        b = fig.add(Axes(360 + col_i * 560, 120, 220, 290, xcats=clab, ylim=(0.01, 100), ylog=True,
                         title=f"({'bd'[col_i]}) Conflicts / 100 robot-steps, {'A*' if planner == 'astar' else 'RRT*'}",
                         ylabel="proposed conflicts (log)"))  # fmt: skip
        for j, (c, colr, shp, lab) in enumerate(ctrls):
            off = (j - 0.5) * 0.3
            for i, (d, al) in enumerate(cats):
                ms = g[(c, planner, d, al)]
                v, lo, hi = boot_ratio(
                    [m["tasks_delivered"] for m in ms], [m["tasks_released"] for m in ms]
                )
                a.vrange(i + off, lo, hi, colr, width=1.5)
                a.point(
                    i + off,
                    v,
                    colr,
                    shape=shp,
                    title=f"{lab}, {planner}, {d}, {al}: {v:.3f} [{lo:.3f}, {hi:.3f}]",
                )
                cv, clo, chi = boot([m["conflicts_per_100"] for m in ms])
                b.vrange(i + off, max(clo, 0.011), max(chi, 0.011), colr, width=1.5)
                b.point(
                    i + off,
                    max(cv, 0.011),
                    colr,
                    shape=shp,
                    title=f"{lab}, {planner}, {d}, {al}: {cv:.3f} per 100 [{clo:.3f}, {chi:.3f}]",
                )
                rows += [["completion", c, planner, d, al, v, lo, hi, len(ms)],
                         ["conflicts_per_100", c, planner, d, al, cv, clo, chi, len(ms)]]  # fmt: skip
    fig.notes = ["Levels: stat = static, obst = moving obstacles, full = obstacles + map events + arrivals; "
                 "auc = auction, seq = sequential allocator. "
                 "95 % bootstrap CIs over the 50 episodes of each cell. Zero conflict rates are drawn at the "
                 "axis floor (0.011). Sources: results/experiments/e5_recovery{,_reference}/records.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out,
        name,
        ["metric", "controller", "planner", "dynamics", "allocator", "value", "low", "high", "n"],
        rows,
    )
    src.describe(name, "Pooled completion (tasks_delivered / tasks_released) and mean conflicts_per_100 "
                 "per cell; endpoint recovery on (the configuration adopted in Issue #4).")  # fmt: skip


# ------------------------------------------------------------------------------ learning data
def runs(algo: str) -> list[str]:
    return [f"{algo}_seed{s}" for s in (1, 2, 3)]


def validation(src: Sources, fig: str, run: str) -> list[dict[str, Any]]:
    return [
        v for v in src.read(fig, TRAIN / run / "validation.jsonl") if v.get("planner") == "astar"
    ]


def best_step(run: str) -> int:
    return int(json.loads((TRAIN / run / "best.json").read_text("utf-8"))["env_steps"])


def fig05_learning(out: Path, src: Sources) -> None:
    name = "fig05_learning_curves"
    fig = Figure(1180, 480, "Fig. 5 — Learning curves on validation block A (E4; 60 A* episodes per point)",
                 "Mean of three training seeds (line) and the range across seeds (band); hollow marker = "
                 "selected checkpoint of each seed",
                 legend=[(ALGO_NAME[a], ALGO[a], "line") for a in ALGO] + [("seed range", TEXT_2, "band")])  # fmt: skip
    a = fig.add(Axes(80, 125, 300, 270, xlim=(0, 1_000_000), ylim=(0, 1.02),
                     xticks=[0, 250_000, 500_000, 750_000, 1_000_000],
                     xticklabels=["0", "250 k", "500 k", "750 k", "1 M"],
                     title="(a) Completion", xlabel="environment steps", ylabel="delivered ÷ released"))  # fmt: skip
    b = fig.add(Axes(470, 125, 300, 270, xlim=(0, 1_000_000), ylim=(-12, 4.5),
                     xticks=[0, 250_000, 500_000, 750_000, 1_000_000],
                     xticklabels=["0", "250 k", "500 k", "750 k", "1 M"],
                     title="(b) Mean discounted base return", xlabel="environment steps",
                     ylabel="return (default reward weights)"))  # fmt: skip
    c = fig.add(Axes(860, 125, 300, 270, xlim=(0, 2.2), ylim=(0, 1.02),
                     title="(c) Completion against training wall time", xlabel="hours (concurrent runs)",
                     ylabel="delivered ÷ released"))  # fmt: skip
    for ax in (a, b):
        ax.vline(200_000, AXIS, label="stage 2")
        ax.vline(500_000, AXIS, label="stage 3")
    rows = []
    for algo in ALGO:
        per_run = []
        for run in runs(algo):
            vs = validation(src, name, run)
            prog = {
                p["env_steps"]: p
                for p in src.read(name, TRAIN / run / "train_log.jsonl")
                if p.get("type") == "progress"
            }
            per_run.append((run, vs, prog))
        steps = [v["env_steps"] for v in per_run[0][1]]
        for key, ax in (("completion", a), ("base_return", b)):
            vals = np.array([[v["summary"]["overall"][key] for v in vs] for _, vs, _ in per_run])
            mean = vals.mean(0)
            ax.band(steps, vals.min(0), vals.max(0), ALGO[algo])
            ax.line(steps, mean, ALGO[algo], title=f"{ALGO_NAME[algo]} mean {key}")
            for run, vs, _ in per_run:
                bs = best_step(run)
                sel = next(v for v in vs if v["env_steps"] == bs)
                ax.point(
                    bs,
                    sel["summary"]["overall"][key],
                    ALGO[algo],
                    hollow=True,
                    r=4,
                    title=f"{run}: selected at {bs:,} steps, {key} {sel['summary']['overall'][key]:.3f}",
                )
            for i, s in enumerate(steps):
                rows.append(
                    [algo, key, s, float(mean[i]), float(vals.min(0)[i]), float(vals.max(0)[i])]
                )
        hrs = []
        comp = []
        for _run, vs, prog in per_run:
            hrs.append(
                [
                    prog[v["env_steps"]]["wall_s"] / 3600 if v["env_steps"] in prog else np.nan
                    for v in vs
                ]
            )
            comp.append([v["summary"]["overall"]["completion"] for v in vs])
        h = np.nanmean(np.array(hrs, float), 0)
        cm = np.array(comp).mean(0)
        c.line(
            list(h), list(cm), ALGO[algo], title=f"{ALGO_NAME[algo]} mean completion vs wall time"
        )
        for i, s in enumerate(steps):
            rows.append([algo, "hours_at_step", s, float(h[i]), "", ""])
    fig.notes = ["Validation every 50 000 environment steps; selection is lexicographic (completion, then return). "
                 "Wall time includes validation and was measured with 3 (MADQL) or 6 (PPO) runs sharing one machine.",
                 "Sources: results/training/{madql,mappo,ippo}_seed{1,2,3}/validation.jsonl, best.json, train_log.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out, name, ["algorithm", "quantity", "env_steps", "mean", "min_seed", "max_seed"], rows
    )
    src.describe(name, "validation.jsonl A* records: summary.overall completion and base_return per "
                 "step; mean and min-max over the three seeds. Wall time = train_log progress wall_s "
                 "at the validation step.")  # fmt: skip


def fig06_training(out: Path, src: Sources) -> None:
    name = "fig06_training_episodes"
    bins = np.arange(0, 1_000_001, 25_000)
    fig = Figure(820, 470, "Fig. 6 — Training episodes (E4): conflicts and completion during learning",
                 "Mean over 25 000-step windows of all completed training episodes; line = mean of the "
                 "three seeds, band = range of the seed means",
                 legend=[(ALGO_NAME[a], ALGO[a], "line") for a in ALGO])  # fmt: skip
    a = fig.add(Axes(80, 125, 300, 270, xlim=(0, 1_000_000), ylim=(0.05, 3), ylog=True,
                     xticks=[0, 250_000, 500_000, 750_000, 1_000_000],
                     xticklabels=["0", "250 k", "500 k", "750 k", "1 M"],
                     title="(a) Conflicts per 100 non-idle robot-steps", xlabel="environment steps",
                     ylabel="training-log rate (log)"))  # fmt: skip
    b = fig.add(Axes(490, 125, 300, 270, xlim=(0, 1_000_000), ylim=(0, 1.02),
                     xticks=[0, 250_000, 500_000, 750_000, 1_000_000],
                     xticklabels=["0", "250 k", "500 k", "750 k", "1 M"],
                     title="(b) Training-episode completion", xlabel="environment steps",
                     ylabel="per-episode completion (mean)"))  # fmt: skip
    for ax in (a, b):
        ax.vline(200_000, AXIS, label="stage 2")
        ax.vline(500_000, AXIS, label="stage 3")
    rows = []
    centers = (bins[:-1] + bins[1:]) / 2
    for algo in ALGO:
        conf, comp = [], []
        for run in runs(algo):
            eps = [
                e
                for e in src.read(name, TRAIN / run / "train_log.jsonl")
                if e.get("type") == "episode"
            ]
            idx = np.digitize([e["env_steps"] for e in eps], bins) - 1
            cs = np.full(len(centers), np.nan)
            ps = np.full(len(centers), np.nan)
            for k in range(len(centers)):
                sel = [e for e, i in zip(eps, idx, strict=True) if i == k]
                if sel:
                    cs[k] = np.mean([e["conflicts_per_100"] for e in sel])
                    ps[k] = np.mean([e["completion"] for e in sel])
            conf.append(cs)
            comp.append(ps)
        for key, arr, ax in (
            ("conflicts_per_100_nonidle", np.array(conf), a),
            ("completion", np.array(comp), b),
        ):
            ok = ~np.isnan(arr).any(0)
            x = centers[ok]
            m = arr[:, ok].mean(0)
            lo, hi = arr[:, ok].min(0), arr[:, ok].max(0)
            if key.startswith("conf"):
                m, lo, hi = np.maximum(m, 0.051), np.maximum(lo, 0.051), np.maximum(hi, 0.051)
            ax.band(list(x), list(lo), list(hi), ALGO[algo])
            ax.line(list(x), list(m), ALGO[algo], title=f"{ALGO_NAME[algo]} {key}")
            rows += [
                [algo, key, float(xx), float(mm), float(ll), float(hh)]
                for xx, mm, ll, hh in zip(x, m, lo, hi, strict=True)
            ]
    fig.notes = ["Training scenarios change with the curriculum (stage 1: 2–4 robots, 16×16; stage 3: dynamic, up to 8 robots), "
                 "so values are not comparable across stages or with validation. MADQL explores ε-greedily, PPO samples its policy.",
                 "The training log divides conflicts by NON-IDLE robot-steps (evaluation records divide by all robot-steps). "
                 "Sources: results/training/*/train_log.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out,
        name,
        ["algorithm", "quantity", "env_steps_bin_centre", "mean", "min_seed", "max_seed"],
        rows,
    )
    src.describe(name, "train_log.jsonl episode records binned by env_steps (25 000); per-seed bin means, "
                 "then mean and range over seeds.")  # fmt: skip


# ------------------------------------------------------------------------------ block B data
def block_b(src: Sources, fig: str) -> dict[str, list[dict[str, Any]]]:
    recs = (src.read(fig, DEV5) + src.read(fig, DEV6 / "ppo" / "records.jsonl")
            + src.read(fig, DEV6 / "baseline_matched" / "records.jsonl"))  # fmt: skip
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by[f"{r['label']}/{r['planner']}"].append(r)
    return by


ROWS = ["baseline", "baseline-noresolver", "reference", *POLICIES]


def row_label(k: str) -> str:
    return COMP[k][3] if k in COMP else label_of(k)


def row_style(k: str) -> tuple[str, str, bool]:
    if k in COMP:
        c, s, h, _ = COMP[k]
        return c, s, h
    return ALGO[k.split("-seed")[0]], "circle", False


def fig07_blockb(out: Path, src: Sources) -> None:
    name = "fig07_blockB_comparison"
    by = block_b(src, name)
    labels = [row_label(k) for k in ROWS]
    fig = Figure(1180, 560, "Fig. 7 — Development comparison on validation block B (greedy, auction, shield off)",
                 "Filled = A* (240 episodes), hollow = RRT* (60 episodes); 95 % bootstrap CI over episodes",
                 legend=[("MADQL", ALGO["madql"], "circle"), ("MAPPO", ALGO["mappo"], "circle"),
                         ("IPPO", ALGO["ippo"], "circle"), ("non-learned", NEUTRAL[0], "square"),
                         ("RRT* (hollow)", TEXT_2, "circle-hollow")])  # fmt: skip
    a = fig.add(Axes(250, 110, 260, 380, ycats=labels, xlim=(0.5, 1.0), title="(a) Completion",
                     xlabel="delivered ÷ released"))  # fmt: skip
    b = fig.add(Axes(570, 110, 260, 380, ycats=[""] * len(labels), xlim=(0.5, 1000), xlog=True,
                     title="(b) Proposed conflicts per episode", xlabel="conflicts (log)"))  # fmt: skip
    c = fig.add(Axes(890, 110, 260, 380, ycats=[""] * len(labels), xlim=(-22, 5),
                     title="(c) Mean discounted base return", xlabel="return (default weights)"))  # fmt: skip
    rows = []
    for i, k in enumerate(ROWS):
        color, shape, hollow = row_style(k)
        for j, planner in enumerate(("astar", "rrtstar")):
            rs = by[f"{k}/{planner}"]
            y = i + (j - 0.5) * 0.36
            v, lo, hi = boot_ratio([r["delivered"] for r in rs], [r["released"] for r in rs])
            cv, clo, chi = boot([r["conflicts_total"] for r in rs])
            bv, blo, bhi = boot([r["base_return"] for r in rs])
            for ax, (m, lo_, hi_), unit in ((a, (v, lo, hi), "completion"), (b, (max(cv, 0.51), max(clo, 0.51), max(chi, 0.51)), "conflicts/episode"),
                                       (c, (bv, blo, bhi), "base return")):  # fmt: skip
                ax.hrange(y, lo_, hi_, color, width=1.5)
                ax.point(
                    m,
                    y,
                    color,
                    shape=shape,
                    r=3.8,
                    hollow=(planner == "rrtstar") or hollow,
                    title=f"{row_label(k)}, {'A*' if planner == 'astar' else 'RRT*'}: {unit} {m:.3f} [{lo_:.3f}, {hi_:.3f}], n={len(rs)}",
                )
            rows.append([k, planner, len(rs), v, lo, hi, cv, clo, chi, bv, blo, bhi])
    fig.notes = ["MADQL, the baseline and the reference follower are the Issue #5 records (read only); PPO and the matched "
                 "baseline are Issue #6. Seeds 10 020–10 099 (A*) and 10 020–10 039 (RRT*); no test seed.",
                 "Sources: results/dev5/comparison/records.jsonl, results/dev6/{ppo,baseline_matched}/records.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(out, name, ["controller", "planner", "episodes", "completion", "c_low", "c_high", "conflicts_per_episode",
                            "k_low", "k_high", "base_return", "r_low", "r_high"], rows)  # fmt: skip
    src.describe(name, "Per controller and planner: pooled completion, mean conflicts_total and mean "
                 "base_return over block-B episodes, with episode-bootstrap CIs.")  # fmt: skip


def fig08_conflict_types(out: Path, src: Sources) -> None:
    name = "fig08_conflict_types"
    by = block_b(src, name)
    types = [
        ("RRv", "Robot–robot vertex"),
        ("RRs", "Robot–robot swap"),
        ("ROv", "Robot–obstacle vertex"),
    ]
    labels = [row_label(k) for k in ROWS]
    fig = Figure(1180, 565, "Fig. 8 — What kind of conflicts? Block B, A*, per episode",
                 "Proposed conflicts reaching the arbiter (all are cancelled; physical collisions = 0 everywhere). "
                 "Mean with 95 % bootstrap CI over 240 episodes",
                 legend=[("MADQL", ALGO["madql"], "circle"), ("MAPPO", ALGO["mappo"], "circle"),
                         ("IPPO", ALGO["ippo"], "circle"), ("non-learned", NEUTRAL[0], "square")])  # fmt: skip
    rows = []
    for t_i, (t, tl) in enumerate(types):
        ax = fig.add(Axes(250 + t_i * 320, 110, 260, 380, ycats=labels if t_i == 0 else [""] * len(labels),
                          xlim=(0.01, 1000), xlog=True, title=f"({'abc'[t_i]}) {tl}",
                          xlabel="conflicts per episode (log)"))  # fmt: skip
        for i, k in enumerate(ROWS):
            color, shape, hollow = row_style(k)
            rs = by[f"{k}/astar"]
            v, lo, hi = boot([r["conflicts"][t] for r in rs])
            ax.hrange(i, max(lo, 0.011), max(hi, 0.011), color, width=1.5)
            ax.point(
                max(v, 0.011),
                i,
                color,
                shape=shape,
                hollow=hollow,
                r=4,
                title=f"{row_label(k)}: {t} {v:.3f} [{lo:.3f}, {hi:.3f}] per episode",
            )
            rows.append([k, t, v, lo, hi, len(rs)])
    fig.notes = ["Zero means are drawn at the axis floor (0.011). Robot–obstacle swaps (ROs) were 0 for every controller. "
                 "Same sources as Fig. 7."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out, name, ["controller", "type", "mean_per_episode", "low", "high", "episodes"], rows
    )
    src.describe(name, "records 'conflicts' dict per episode (A*), mean and episode-bootstrap CI.")


def fig09_tradeoff(out: Path, src: Sources) -> None:
    name = "fig09_completion_conflict_tradeoff"
    by = block_b(src, name)
    stoch = src.read(name, DEV6 / "stochastic" / "records.jsonl")
    sg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in stoch:
        if r.get("mode") == "stochastic":
            sg[r["label"]].append(r)
    fig = Figure(1180, 500, "Fig. 9 — Completion against conflicts (block B, A*)",
                 "Each point is one controller; filled = greedy (primary), hollow = stochastic execution "
                 "(3 replicates, same instances); crosshairs = 95 % bootstrap CI. Panel (b) enlarges the "
                 "top of panel (a)",
                 legend=[("MADQL", ALGO["madql"], "circle"), ("MAPPO", ALGO["mappo"], "circle"),
                         ("IPPO", ALGO["ippo"], "circle"), ("baseline", NEUTRAL[0], "square"),
                         ("matched baseline", NEUTRAL[0], "diamond"),
                         ("reference follower", NEUTRAL[1], "triangle")])  # fmt: skip
    panels = [
        fig.add(Axes(90, 130, 470, 300, xlim=(0.5, 1000), ylim=(0.55, 1.005), xlog=True,
                     title="(a) All controllers", xlabel="proposed conflicts per episode (log)",
                     ylabel="completion (delivered ÷ released)")),
        fig.add(Axes(680, 130, 470, 300, xlim=(0.5, 100), ylim=(0.975, 1.0005), xlog=True,
                     title="(b) Completion ≥ 0.975", xlabel="proposed conflicts per episode (log)",
                     ylabel="completion")),
    ]  # fmt: skip
    rows = []
    for k in ROWS:
        color, shape, _hollow = row_style(k)
        rs = by[f"{k}/astar"]
        v, lo, hi = boot_ratio([r["delivered"] for r in rs], [r["released"] for r in rs])
        cv, clo, chi = boot([r["conflicts_total"] for r in rs])
        rows.append([k, "greedy", v, lo, hi, cv, clo, chi])
        stoch_pt = None
        if k in sg:
            ss = sg[k]
            sv, slo, shi = boot_ratio([r["delivered"] for r in ss], [r["released"] for r in ss])
            scv, sclo, schi = boot([r["conflicts_total"] for r in ss])
            stoch_pt = (sv, scv)
            rows.append([k, "stochastic", sv, slo, shi, scv, sclo, schi])
        for ax in panels:
            ax.hrange(v, max(clo, 0.51), chi, color, width=1.2)
            ax.vrange(max(cv, 0.51), lo, hi, color, width=1.2)
            ax.point(
                max(cv, 0.51),
                v,
                color,
                shape=shape,
                r=4.5,
                title=f"{row_label(k)} greedy: completion {v:.3f}, conflicts {cv:.1f}",
            )
            if stoch_pt is not None:
                sv, scv = stoch_pt
                ax.line([max(cv, 0.51), max(scv, 0.51)], [v, sv], color, width=1, dash="2 3")
                ax.point(
                    max(scv, 0.51),
                    sv,
                    color,
                    hollow=True,
                    r=4,
                    title=f"{row_label(k)} stochastic: completion {sv:.3f}, conflicts {scv:.1f}",
                )
    panels[0].label_at(3.0, 0.62, "MADQL (greedy)", size=10.5)
    panels[0].label_at(408, 0.911, "reference", dx=-10, dy=-10, anchor="end", size=10.5)
    panels[1].label_at(1.0, 0.9994, "baseline", dx=-10, dy=4, anchor="end", size=10.5)
    panels[1].label_at(1.0, 0.9856, "matched baseline", dx=9, dy=4, size=10.5)
    fig.notes = ["Stochastic: PPO samples its policy; MADQL uses ε-greedy with ε = 0.05 (diagnostic only; greedy is primary). "
                 "Sources: Fig. 7 sources and results/dev6/stochastic/records.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out,
        name,
        [
            "controller",
            "mode",
            "completion",
            "c_low",
            "c_high",
            "conflicts_per_episode",
            "k_low",
            "k_high",
        ],
        rows,
    )
    src.describe(name, "Pooled completion vs mean conflicts_total per controller (A*); stochastic = "
                 "mode 'stochastic' records of the diagnostic, all replicates pooled.")  # fmt: skip


def fig10_shield(out: Path, src: Sources) -> None:
    name = "fig10_predictive_shield"
    shield = src.read(name, DEV6 / "shield" / "records.jsonl")
    base = [r for r in src.read(name, DEV5) + src.read(name, DEV6 / "ppo" / "records.jsonl")
            if r["level"] == "full" and r["label"] in POLICIES and r["planner"] == "astar"]  # fmt: skip
    g: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in shield:
        if r["planner"] == "astar":
            g[(r["label"], r["allocator"], r["shield"])][int(r["seed"])] = r
    for r in base:
        g[(r["label"], "auction", "off")][int(r["seed"])] = r
    labels = [label_of(p) for p in POLICIES]
    fig = Figure(1180, 470, "Fig. 10 — Predictive shield on − off (full level, A*, 80 paired episodes per policy)",
                 "Paired mean difference with 95 % bootstrap CI; filled = auction allocator, hollow = sequential allocator",
                 legend=[("MADQL", ALGO["madql"], "circle"), ("MAPPO", ALGO["mappo"], "circle"), ("IPPO", ALGO["ippo"], "circle")])  # fmt: skip
    specs = [
        ("conflicts_total", "(a) Conflicts per episode", (-130, 20)),
        ("completion", "(b) Completion", (-0.03, 0.03)),
        ("wait_ratio", "(c) Wait ratio", (-0.03, 0.03)),
    ]
    rows = []
    for s_i, (metric, title, lim) in enumerate(specs):
        ax = fig.add(Axes(170 + s_i * 340, 110, 270, 290, ycats=labels if s_i == 0 else [""] * len(labels),
                          xlim=lim, title=title, xlabel="shield on − off"))  # fmt: skip
        ax.vline(0, AXIS, dash=None)
        for i, p in enumerate(POLICIES):
            color = ALGO[p.split("-seed")[0]]
            for j, alloc in enumerate(("auction", "sequential")):
                on, off = g[(p, alloc, "predictive")], g[(p, alloc, "off")]
                keys = sorted(set(on) & set(off))
                v, lo, hi = boot_paired(
                    [on[k][metric] for k in keys], [off[k][metric] for k in keys]
                )
                y = i + (j - 0.5) * 0.34
                ax.hrange(y, lo, hi, color, width=1.5)
                ax.point(
                    v,
                    y,
                    color,
                    hollow=(alloc == "sequential"),
                    r=3.8,
                    title=f"{label_of(p)}, {alloc}: {metric} {v:+.4f} [{lo:+.4f}, {hi:+.4f}], n={len(keys)}",
                )
                rows.append([p, alloc, metric, v, lo, hi, len(keys)])
    fig.notes = ["The auction / shield-off arm is the primary block-B comparison (Issue #5 records for MADQL, Issue #6 for PPO). "
                 "Holm-adjusted tests are in results/dev6/summary.md. Source: results/dev6/shield/records.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out, name, ["policy", "allocator", "metric", "mean_diff", "low", "high", "pairs"], rows
    )
    src.describe(name, "Per policy and allocator, paired (by seed) predictive − off differences on the "
                 "80 A* full-level instances; bootstrap over pairs.")  # fmt: skip


def fig11_stochastic(out: Path, src: Sources) -> None:
    name = "fig11_greedy_vs_stochastic"
    recs = src.read(name, DEV6 / "stochastic" / "records.jsonl")
    g: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in recs:
        g[(r["label"], r["mode"])][int(r["seed"]) * 10 + LEVELS.index(r["level"])].append(r)
    fig = Figure(820, 470, "Fig. 11 — Greedy against stochastic execution (block B, A*, 240 paired instances)",
                 "Stochastic: three replicates per instance, averaged per instance before pairing",
                 legend=[("MADQL (ε = 0.05)", ALGO["madql"], "line"), ("MAPPO (sampling)", ALGO["mappo"], "line"),
                         ("IPPO (sampling)", ALGO["ippo"], "line")])  # fmt: skip
    a = fig.add(Axes(80, 120, 300, 280, xcats=["greedy", "stochastic"], ylim=(0.5, 1.01),
                     title="(a) Completion per policy", ylabel="delivered ÷ released"))  # fmt: skip
    b = fig.add(Axes(490, 120, 300, 280, xcats=["greedy", "stochastic"], ylim=(0.5, 100), ylog=True,
                     title="(b) Conflicts per episode per policy", ylabel="proposed conflicts (log)"))  # fmt: skip
    rows = []
    for p in POLICIES:
        color = ALGO[p.split("-seed")[0]]
        gr, st = g[(p, "greedy")], g[(p, "stochastic")]
        comp = [sum(r["delivered"] for k in gr for r in gr[k]) / sum(r["released"] for k in gr for r in gr[k]),
                sum(r["delivered"] for k in st for r in st[k]) / sum(r["released"] for k in st for r in st[k])]  # fmt: skip
        conf = [float(np.mean([r["conflicts_total"] for k in gr for r in gr[k]])),
                float(np.mean([np.mean([r["conflicts_total"] for r in st[k]]) for k in st]))]  # fmt: skip
        for ax, vals, unit in ((a, comp, "completion"), (b, conf, "conflicts/episode")):
            ax.line(
                [0, 1],
                vals,
                color,
                width=1.8,
                title=f"{label_of(p)} {unit}: greedy {vals[0]:.3f} → stochastic {vals[1]:.3f}",
            )
            for x, v in enumerate(vals):
                ax.point(
                    x,
                    max(v, 0.51),
                    color,
                    r=3.6,
                    title=f"{label_of(p)} {['greedy', 'stochastic'][x]}: {unit} {v:.3f}",
                )
        rows.append([p, comp[0], comp[1], conf[0], conf[1]])
    a.label_at(0, 0.584, "MADQL greedy", anchor="end", dx=-10)
    fig.notes = ["Paired tests (Holm-adjusted) are in results/dev6/summary.md; every completion difference is positive. "
                 "Source: results/dev6/stochastic/records.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(
        out,
        name,
        [
            "policy",
            "completion_greedy",
            "completion_stochastic",
            "conflicts_greedy",
            "conflicts_stochastic",
        ],
        rows,
    )
    src.describe(name, "Pooled completion and mean conflicts_total per mode; stochastic conflicts averaged "
                 "per instance over the three replicates first.")  # fmt: skip


def fig12_ablation(out: Path, src: Sources) -> None:
    name = "fig12_reward_ablation"
    abl = src.read(name, DEV6 / "ablation" / "records.jsonl")
    ppo = [r for r in src.read(name, DEV6 / "ppo" / "records.jsonl") if r["planner"] == "astar"]
    inst: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for r in abl + ppo:
        inst[r["label"]][(r["level"], int(r["seed"]))] = r
    cells = [("mappo", "default (ζ=1, β=0.3, c_rr=1)"), ("mappo_zeta0", "ζ = 0 (no shaping)"),
             ("mappo_beta0", "β = 0 (individual reward)"), ("mappo_crr05", "c_rr = 0.5"), ("mappo_crr2", "c_rr = 2")]  # fmt: skip
    labels = [c[1] for c in cells]
    fig = Figure(1180, 470, "Fig. 12 — Reward ablation, Option C (MAPPO, run seeds 1 and 2)",
                 "Dots = the two run seeds of each cell; (a, b) block B, A*, 240 episodes per policy; "
                 "(c) block A, mean of the 11 validations from 500 k to 1 M steps (selection-independent)",
                 legend=[("run seed 1", SERIES[1], "circle"), ("run seed 2", SERIES[1], "diamond")])  # fmt: skip
    a = fig.add(Axes(250, 115, 260, 270, ycats=labels, xlim=(0.97, 1.0), title="(a) Completion, block B",
                     xlabel="delivered ÷ released"))  # fmt: skip
    b = fig.add(Axes(570, 115, 260, 270, ycats=[""] * len(labels), xlim=(1, 100), xlog=True,
                     title="(b) Conflicts per episode, block B", xlabel="conflicts (log)"))  # fmt: skip
    c = fig.add(Axes(890, 115, 260, 270, ycats=[""] * len(labels), xlim=(3.3, 3.8),
                     title="(c) Base return, block A late", xlabel="mean discounted base return"))  # fmt: skip
    rows = []
    for i, (cell, _) in enumerate(cells):
        for j, seed in enumerate((1, 2)):
            lab = f"{cell}-seed{seed}"
            rs = list(inst[lab].values())
            y = i + (j - 0.5) * 0.3
            v, lo, hi = boot_ratio([r["delivered"] for r in rs], [r["released"] for r in rs])
            cv, clo, chi = boot([r["conflicts_total"] for r in rs])
            shp = "circle" if seed == 1 else "diamond"
            a.hrange(y, lo, hi, SERIES[1], width=1.4)
            a.point(
                v,
                y,
                SERIES[1],
                shape=shp,
                r=3.8,
                title=f"{lab}: completion {v:.4f} [{lo:.4f}, {hi:.4f}]",
            )
            b.hrange(y, clo, chi, SERIES[1], width=1.4)
            b.point(
                cv,
                y,
                SERIES[1],
                shape=shp,
                r=3.8,
                title=f"{lab}: conflicts {cv:.2f} [{clo:.2f}, {chi:.2f}]",
            )
            run = f"{cell}_seed{seed}"
            late = [
                v["summary"]["overall"]["base_return"]
                for v in validation(src, name, run)
                if v["env_steps"] >= 500_000
            ]
            lv = float(np.mean(late))
            c.point(
                lv,
                y,
                SERIES[1],
                shape=shp,
                r=3.8,
                title=f"{run}: block-A late mean return {lv:.3f} (11 validations); selected at {best_step(run):,}",
            )
            c.label_at(lv, y, f"sel. {best_step(run) // 1000} k", dx=8, dy=4, size=9.5)
            rows.append([cell, seed, v, lo, hi, cv, clo, chi, lv, best_step(run)])
    fig.notes = ["The default cell's selected checkpoints (200 k and 300 k steps) carry most of its block-B conflicts; late in "
                 "training (c) all cells look alike. Cell − default statistics: results/dev6/summary.md.",
                 "Sources: results/dev6/{ablation,ppo}/records.jsonl, results/training/mappo*_seed{1,2}/validation.jsonl."]  # fmt: skip
    save(out, name, fig)
    write_table(out, name, ["cell", "run_seed", "completion", "c_low", "c_high", "conflicts_per_episode", "k_low", "k_high",
                            "blockA_late_mean_return", "selected_step"], rows)  # fmt: skip
    src.describe(name, "Block-B pooled completion / mean conflicts_total per policy; block-A mean "
                 "base_return over validations at ≥ 500 000 steps.")  # fmt: skip


def fig13_generalisation(out: Path, src: Sources) -> None:
    name = "fig13_blockA_vs_blockB_and_levels"
    by = block_b(src, name)
    fig = Figure(1180, 480, "Fig. 13 — Selection on block A, evaluation on block B, and difficulty levels (A*)",
                 "(a) block-A score of the selected checkpoint against block B (disjoint seeds); "
                 "(b) block-B completion per dynamics level",
                 legend=[("MADQL", ALGO["madql"], "circle"), ("MAPPO", ALGO["mappo"], "circle"),
                         ("IPPO", ALGO["ippo"], "circle"), ("non-learned", NEUTRAL[0], "square")])  # fmt: skip
    a = fig.add(Axes(80, 120, 300, 290, xlim=(0.5, 1.0), ylim=(0.5, 1.0),
                     title="(a) Block A (selection) vs block B", xlabel="block-A completion at the selected step",
                     ylabel="block-B completion"))  # fmt: skip
    a.line([0.5, 1.0], [0.5, 1.0], AXIS, width=1.2, dash="4 3")
    a.label_at(0.55, 0.53, "y = x", size=10.5)
    rows = []
    for p in POLICIES:
        run = p.replace("-", "_")
        bs = best_step(run)
        va = next(v for v in validation(src, name, run) if v["env_steps"] == bs)["summary"][
            "overall"
        ]["completion"]
        rs = by[f"{p}/astar"]
        vb, lo, hi = boot_ratio([r["delivered"] for r in rs], [r["released"] for r in rs])
        color = ALGO[p.split("-seed")[0]]
        a.vrange(va, lo, hi, color, width=1.4)
        a.point(
            va,
            vb,
            color,
            r=4,
            title=f"{label_of(p)}: block A {va:.3f} → block B {vb:.3f} [{lo:.3f}, {hi:.3f}]",
        )
        rows.append([p, "blockA_selected", "", va, "", ""])
        rows.append([p, "blockB", "", vb, lo, hi])
    b = fig.add(Axes(490, 120, 660, 290, xcats=list(LEVELS), ylim=(0.25, 1.01),
                     title="(b) Block-B completion by level", ylabel="delivered ÷ released"))  # fmt: skip
    groups = ["baseline", "baseline-noresolver", "reference", "madql", "mappo", "ippo"]
    for j, gname in enumerate(groups):
        off = (j - 2.5) * 0.13
        for i, lvl in enumerate(LEVELS):
            if gname in COMP:
                color, shape, hollow, lab = COMP[gname]
                rs = [r for r in by[f"{gname}/astar"] if r["level"] == lvl]
                v, lo, hi = boot_ratio([r["delivered"] for r in rs], [r["released"] for r in rs])
                b.vrange(i + off, lo, hi, color, width=1.4)
                b.point(
                    i + off,
                    v,
                    color,
                    shape=shape,
                    hollow=hollow,
                    r=4,
                    title=f"{lab}, {lvl}: {v:.3f} [{lo:.3f}, {hi:.3f}]",
                )
                rows.append([gname, "blockB_level", lvl, v, lo, hi])
            else:
                vals = []
                for s in (1, 2, 3):
                    rs = [r for r in by[f"{gname}-seed{s}/astar"] if r["level"] == lvl]
                    vals.append(sum(r["delivered"] for r in rs) / sum(r["released"] for r in rs))
                color = ALGO[gname]
                b.vrange(i + off, min(vals), max(vals), color, width=1.6)
                b.point(
                    i + off,
                    float(np.mean(vals)),
                    color,
                    r=4,
                    title=f"{ALGO_NAME[gname]}, {lvl}: seeds {', '.join(f'{v:.3f}' for v in vals)}",
                )
                rows.append(
                    [
                        gname,
                        "blockB_level_mean_of_seeds",
                        lvl,
                        float(np.mean(vals)),
                        min(vals),
                        max(vals),
                    ]
                )
    fig.notes = ["(b) learned: mean of 3 policies, bar = seed range; non-learned: pooled, bar = 95 % CI. Square = baseline (resolver on); diamond = matched baseline (resolver off); triangle = reference follower. "
                 "Sources: validation.jsonl + best.json of each run; block-B sources as in Fig. 7."]  # fmt: skip
    save(out, name, fig)
    write_table(out, name, ["controller", "quantity", "level", "value", "low", "high"], rows)
    src.describe(name, "(a) block-A overall completion at best.json step vs block-B pooled completion. "
                 "(b) block-B pooled completion per level.")  # fmt: skip


def fig14_runtime(out: Path, src: Sources) -> None:
    name = "fig14_runtime_and_planner_load"
    cells = src.read(name, R / "pilot_rrt" / "cells.jsonl")
    by = block_b(src, name)
    perf = src.read_csv(name, R / "perf_verify" / "nfr_perf1_samples.csv")
    fig = Figure(1180, 480, "Fig. 14 — Where the time goes: planner load and step-time composition",
                 "(a) RRT* training pilot; (b) planner calls caused by each controller (block B, A*); "
                 "(c) per-step time by component, non-learned system (NFR-PERF-1 verification)",
                 legend=[("A*", NEUTRAL[0], "square"), ("RRT*", SERIES[0], "circle")])  # fmt: skip
    a = fig.add(Axes(80, 120, 280, 280, xcats=["stage 1", "stage 2", "stage 3"], ylim=(0.1, 1000), ylog=True,
                     title="(a) Mean env-step time in PPO training", ylabel="ms per environment step (log)"))  # fmt: skip
    rows = []
    for c in cells:
        col = NEUTRAL[0] if c["planner"] == "astar" else SERIES[0]
        shape = "square" if c["planner"] == "astar" else "circle"
        hollow = c["source"] != "ppo"
        off = -0.12 if c["source"] == "ppo" else 0.12
        a.point(
            c["stage"] - 1 + off,
            c["step_mean_ms"],
            col,
            shape=shape,
            hollow=hollow,
            r=4,
            title=f"{c['planner']}/{c['source']} stage {c['stage']}: {c['step_mean_ms']:.2f} ms/step, planner share {c['planner_share']:.3f}",
        )
        rows.append(
            [
                "pilot_step_ms",
                c["planner"],
                c["source"],
                c["stage"],
                c["step_mean_ms"],
                c["planner_share"],
            ]
        )
    ctrl = ["baseline", "baseline-noresolver", "reference", "madql", "mappo", "ippo"]
    names = ["baseline", "matched", "reference", "MADQL", "MAPPO", "IPPO"]
    b = fig.add(Axes(470, 120, 280, 280, xcats=names, ylim=(0, 40),
                     title="(b) Planner calls per 100 robot-steps", ylabel="calls (A*)"))  # fmt: skip
    for i, cname in enumerate(ctrl):
        keys = [
            k
            for k in by
            if k.endswith("/astar")
            and (k.split("/")[0] == cname or k.split("/")[0].startswith(cname + "-seed"))
        ]
        rs = [r for k in keys for r in by[k]]
        rate = (
            100
            * sum(r["planner_calls"] for r in rs)
            / sum(r["steps"] * r.get("robots", 8) for r in rs)
        )
        dev = float(np.mean([r.get("deviation_rate_per_100") or 0 for r in rs]))
        color = ALGO.get(cname, NEUTRAL[0] if cname.startswith("baseline") else NEUTRAL[1])
        b.vbar(
            i,
            0,
            rate,
            color,
            thick=26,
            title=f"{names[i]}: {rate:.2f} calls / 100 robot-steps; off-path (T3) {dev:.1f} / 100",
        )
        b.label_at(i, rate, f"{rate:.1f}", dx=0, dy=-6, anchor="middle")
        rows.append(["planner_calls_per_100", cname, "astar", "", rate, dev])
    comps = [
        "allocation_ms",
        "planning_ms",
        "deadlock_ms",
        "endpoint_ms",
        "controller_ms",
        "engine_rest_ms",
    ]
    cn = ["allocation", "planning (A*)", "deadlock", "endpoint", "controller", "engine rest"]
    fast = [p for p in perf if p["impl"] == "fast"]
    c = fig.add(Axes(880, 120, 270, 280, xcats=["mean", "p99 step"], ylim=(0, 12),
                     title="(c) Step time by component", ylabel="milliseconds per step"))  # fmt: skip
    tot = np.array([float(p["total_ms"]) for p in fast])
    p99 = float(np.percentile(tot, 99))
    worst = [p for p in fast if float(p["total_ms"]) >= p99]
    palette = [SERIES[0], NEUTRAL[0], SERIES[4], SERIES[6], SERIES[2], NEUTRAL[2]]
    for x, sample in ((0, fast), (1, worst)):
        base = 0.0
        for comp, colr, lab in zip(comps, palette, cn, strict=True):
            v = float(np.mean([float(p[comp]) for p in sample]))
            c.vbar(
                x,
                base,
                base + v,
                colr,
                thick=46,
                title=f"{lab}: {v:.3f} ms ({'all steps' if x == 0 else 'slowest 1 %'})",
            )
            base += v
            rows.append(["perf_component_ms", comp, "fast", ["mean", "p99"][x], v, ""])
    fig.legend += [(lab, colr, "swatch") for lab, colr in zip(cn, palette, strict=True)]
    fig.notes = ["(a) filled = untrained PPO actor, hollow = reference-follower actions. (b) learned controllers leave the reference path (trigger T3) 11–36 times per 100 robot-steps; the baseline never does. "
                 "(c) workload R, optimised implementation, A*; 'p99 step' = mean composition of the slowest 1 % of steps.",
                 "Sources: results/pilot_rrt/cells.jsonl, block-B records (Fig. 7), results/perf_verify/nfr_perf1_samples.csv."]  # fmt: skip
    save(out, name, fig)
    write_table(out, name, ["quantity", "a", "b", "c", "value", "extra"], rows)
    src.describe(name, "(a) pilot cells step_mean_ms; (b) Σ planner_calls / Σ(steps·robots) per controller "
                 "group; (c) mean per-component ms over all optimised-implementation samples and over the "
                 "slowest 1 %.")  # fmt: skip


FIGURES = [fig01_planners, fig02_replanning, fig03_allocation, fig04_system, fig05_learning,
           fig06_training, fig07_blockb, fig08_conflict_types, fig09_tradeoff, fig10_shield,
           fig11_stochastic, fig12_ablation, fig13_generalisation, fig14_runtime]  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; figures are written to a new directory")
    out.mkdir(parents=True)
    src = Sources()
    for f in FIGURES:
        f(out, src)
    lines = ["# Overview figures — sources", "",
             "Generated by `tools/figures/overview.py`. Every figure is computed from the original records "
             "listed below; the records were only read. Uncertainty: 95 % percentile bootstrap, "
             f"{N_BOOT} resamples, seed 0 (unless a figure states an IQR or a seed range).", ""]  # fmt: skip
    for fig, files in src.used.items():
        lines += [f"## {fig}", "", src.text.get(fig, ""), "", f"Table view: `{fig}.csv`", ""]
        lines += [f"- `{p}` — SHA-256 `{src.hashes[p]}`" for p in sorted(files)]
        lines.append("")
    (out / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
    figs = sorted(p.name for p in out.glob("fig*.svg"))
    items = "\n".join(
        f'<section><img src="{f}" alt="{f}"><p><a href="{f[:-4]}.csv">table view (CSV)</a></p></section>'
        for f in figs
    )
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Overview figures</title><style>body{margin:16px;background:#f3f3f1;"
        "font-family:system-ui,sans-serif;color:#0b0b0b}section{margin:0 0 28px}"
        "img{max-width:100%;height:auto;border:1px solid #e6e5e1;background:#fcfcfb}"
        "a{color:#256abf}</style><h1>Overview figures, Issues #1–#6</h1>"
        "<p>Sources and SHA-256 of every input: <a href='SOURCES.md'>SOURCES.md</a></p>" + items,
        encoding="utf-8",
    )
    print("wrote SOURCES.md and index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
