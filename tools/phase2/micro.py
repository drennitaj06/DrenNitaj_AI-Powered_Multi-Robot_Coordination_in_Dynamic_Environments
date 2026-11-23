"""Issue 6A Phase 2 (P2, RRT\\* budget control): microbenchmarks on the E1a / E1c / E2 query sets.

Protocol and selection rule: ``docs/research/2026-11-22-issue-6a-phase2-rrt-budget.md`` §3
(written before any microbenchmark was run). All runs use the system budget of 1 000 iterations,
default RRT\\* options and no wall-clock limit.

Commands::

    python tools/phase2/micro.py e1a    --out results/dev6a/phase2/micro --workers 10
    python tools/phase2/micro.py e1c    --out results/dev6a/phase2/micro --workers 10
    python tools/phase2/micro.py select --out results/dev6a/phase2/micro
    python tools/phase2/micro.py timing --out results/dev6a/phase2/micro --workers 10
    python tools/phase2/micro.py e2     --out results/dev6a/phase2/micro --workers 10

``e1a`` and ``e1c`` run the full-budget reference and all 11 candidates on every query and check
the exact-prefix property of every candidate run against the reference. ``select`` applies the
pre-declared rule (§3.3) and writes ``selection.json``. ``timing`` re-times the reference and the
selected rule on the selection stratum without tracing. ``e2`` (reported comparison on reserved
test seeds, never used for selection) refuses to run before ``selection.json`` exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

BUDGET = 1000  # evaluation.RRT_ITERATIONS (the system budget)
REFINE = (0, 25, 50, 100, 200, 400)
PATIENCE = (25, 50, 100, 200, 400)
#: label -> (refine_iterations, patience); "full" is the full-budget reference
CONFIGS: dict[str, tuple[int | None, int | None]] = {
    "full": (None, None),
    **{f"k{k}": (k, None) for k in REFINE},
    **{f"m{m}": (None, m) for m in PATIENCE},
}
E1A_SPEC = Path("configs/experiments/e1a_planners_static.toml")
E1C_SPEC = Path("configs/experiments/e1c_convergence.toml")
E2_SPEC = Path("configs/experiments/e2_replanning.toml")
E1C_STORED = Path("results/experiments/e1c_convergence/records.jsonl")
STRATUM_SIZES = (16, 32)
TOLERANCE = 1.01  # quality screen of §3.3 (engineering tolerance; microbenchmark only)
CODE = (
    "src/robocoord/planning/rrtstar.py",
    "src/robocoord/planning/procedure.py",
    "src/robocoord/experiments/planning_runs.py",
    "tools/phase2/micro.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------- static queries
def _query(seed_base: int, family: str, size: int, map_index: int, pair: int) -> Any:
    from robocoord.experiments import planning_runs as pr
    from robocoord.planning import build_constraints

    grid = pr.make_map(family, size, seed_base, map_index)
    key = (seed_base, pr._FAMILY_INDEX[family], size, map_index, pair)
    start, goal = pr.make_pair(grid, key)
    return grid, start, goal, build_constraints(grid, start, goal), key


def _solve(grid: Any, start: Any, goal: Any, c: Any, key: Any, label: str, trace: bool) -> Any:
    from robocoord.experiments import planning_runs as pr
    from robocoord.planning import PlanQuery, make_planner, plan_with_procedure

    k, m = CONFIGS[label]
    planner = make_planner(
        "rrtstar", max_iterations=BUDGET, record_trace=trace, refine_iterations=k, patience=m
    )
    return plan_with_procedure(planner, PlanQuery(start, goal, c, pr._rng(*key, 7)))


def static_job(j: dict[str, Any]) -> dict[str, Any]:
    """All configurations on one static query, with the exact-prefix check against ``full``."""
    from robocoord.core.distance import bfs_distance_field
    from robocoord.planning.base import lattice_cost

    grid, start, goal, c, key = _query(j["seed_base"], j["family"], j["size"], j["map"], j["pair"])
    bfs = int(bfs_distance_field(grid, goal)[start[1], start[0]])
    out: dict[str, Any] = {**j, "bfs_steps": bfs, "configs": {}}
    ref: Any = None
    for label in CONFIGS:
        res = _solve(grid, start, goal, c, key, label, trace=True)
        st = res.stats
        trace: tuple[float, ...] = st.best_cost_trace or ()
        row: dict[str, Any] = {
            "status": res.status,
            "iterations": st.iterations,
            "first_solution_iteration": st.first_solution_iteration,
            "planner_cost": res.cost if res.path is not None else None,
            "path_steps": len(res.path) - 1 if res.path is not None else None,
            "lattice_cost": lattice_cost(res.path, c) if res.path is not None else None,
            "segment_checks": st.segment_checks,
            "time_s": st.time_s,
            "best_at_1000": None,
        }
        if label == "full":
            ref = (res, trace)
            row["best_at_1000"] = trace[-1] if trace else None
        else:
            fres, ftrace = ref
            n = st.iterations
            row["prefix_ok"] = bool(
                tuple(trace) == tuple(ftrace[:n])
                and res.status == fres.status
                and st.first_solution_iteration == fres.stats.first_solution_iteration
                and (res.path is None or res.cost == trace[-1])
                and (fres.path is not None or n == fres.stats.iterations)
            )
        out["configs"][label] = row
    return out


def _static_jobs(spec_path: Path) -> list[dict[str, Any]]:
    from robocoord.experiments.spec import load_spec

    spec = load_spec(spec_path)
    f = spec.factors
    return [
        {"seed_base": spec.seed_base, "family": fa, "size": int(s), "map": int(m), "pair": int(p)}
        for fa in f["family"]
        for s in f["size"]
        for m in f["map"]
        for p in f["pair"]
    ]


def _key(j: dict[str, Any]) -> tuple[Any, ...]:
    return (j["seed_base"], j["family"], j["size"], j["map"], j["pair"])


def _run_static(out: Path, name: str, jobs: list[dict[str, Any]], workers: int) -> int:
    target = out / f"{name}.jsonl"
    done = {_key(r) for r in _records(target)}
    todo = [j for j in jobs if _key(j) not in done]
    _log(out, f"{name}: {len(done)} done, {len(todo)} to do, workers {workers}")
    t0 = time.time()
    with target.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(static_job, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"{name}: {i}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    recs = _records(target)
    bad = [
        _key(r)
        for r in recs
        for lab, row in r["configs"].items()
        if lab != "full" and not row["prefix_ok"]
    ]
    _log(
        out,
        f"{name}: finished {len(todo)} new in {time.time() - t0:.0f}s; prefix failures {len(bad)}",
    )
    print(f"{name}: {len(recs)} queries; prefix failures {len(bad)}")
    return 1 if bad else 0


def _log(out: Path, msg: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.log").open("a", encoding="utf-8") as lf:
        lf.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


# ---------------------------------------------------------------------------- selection
def summarise(recs: Iterable[dict[str, Any]], sizes: Sequence[int] | None = None) -> dict[str, Any]:
    """Per family and pooled: success, Q (executed steps ratio vs full), iterations, time."""
    rows = [r for r in recs if sizes is None or r["size"] in sizes]
    fams = sorted({r["family"] for r in rows})
    out: dict[str, Any] = {}
    for label in CONFIGS:
        per: dict[str, Any] = {}
        for fam in [*fams, "all"]:
            rs = [r for r in rows if fam == "all" or r["family"] == fam]
            solved = [r for r in rs if r["configs"]["full"]["path_steps"] is not None]
            c = [r["configs"][label] for r in rs]
            num = sum(r["configs"][label]["path_steps"] for r in solved)
            den = sum(r["configs"]["full"]["path_steps"] for r in solved)
            pnum = sum(r["configs"][label]["planner_cost"] for r in solved)
            pden = sum(r["configs"]["full"]["planner_cost"] for r in solved)
            per[fam] = {
                "queries": len(rs),
                "solved": sum(x["status"] == "ok" for x in c),
                "success": sum(x["status"] == "ok" for x in c) / len(rs) if rs else None,
                "q_steps": num / den if den else None,
                "q_planner_cost": pnum / pden if pden else None,
                "share_steps_identical": (
                    sum(
                        r["configs"][label]["path_steps"] == r["configs"]["full"]["path_steps"]
                        for r in solved
                    )
                    / len(solved)
                    if solved
                    else None
                ),
                "max_step_increase": max(
                    (
                        r["configs"][label]["path_steps"] - r["configs"]["full"]["path_steps"]
                        for r in solved
                    ),
                    default=None,
                ),
                "step_gap_vs_bfs": (num / sum(r["bfs_steps"] for r in solved) if solved else None),
                "mean_iterations": sum(x["iterations"] for x in c) / len(rs) if rs else None,
                "mean_iterations_solved": (
                    sum(r["configs"][label]["iterations"] for r in solved) / len(solved)
                    if solved
                    else None
                ),
                "mean_segment_checks": sum(x["segment_checks"] for x in c) / len(rs)
                if rs
                else None,
                "median_first_solution": _median(
                    [r["configs"][label]["first_solution_iteration"] for r in solved]
                ),
                "median_time_ms_traced": _median([1000 * x["time_s"] for x in c]),
            }
        out[label] = per
    return out


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def select(out: Path) -> dict[str, Any]:
    recs = _records(out / "e1a.jsonl")
    if len(recs) != len(_static_jobs(E1A_SPEC)):
        raise SystemExit(f"E1a incomplete: {len(recs)} records")
    if any(
        not row["prefix_ok"] for r in recs for lab, row in r["configs"].items() if lab != "full"
    ):
        raise SystemExit("prefix check failed; selection refused")
    summ = summarise(recs, STRATUM_SIZES)
    fams = [f for f in summ["full"] if f != "all"]
    table = {}
    for label in CONFIGS:
        if label == "full":
            continue
        q = {f: summ[label][f]["q_steps"] for f in fams}
        table[label] = {
            "max_q_steps": max(q.values()),
            "q_steps_by_family": q,
            "eligible": max(q.values()) <= TOLERANCE,
            "mean_iterations": summ[label]["all"]["mean_iterations"],
        }
    elig = [lab for lab, v in table.items() if v["eligible"]]

    def order(lab: str) -> tuple[float, float, int]:
        k, m = CONFIGS[lab]
        param = k if k is not None else m
        assert param is not None
        # fewest iterations (rounded to 0.1), then the larger parameter, then refine over patience
        return (round(table[lab]["mean_iterations"], 1), -param, 0 if k is not None else 1)

    chosen = min(elig, key=order) if elig else None
    res = {
        "selected_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rule": "§3.3: stratum sizes 16/32; eligible if max_f Q_f <= 1.01; fewest mean iterations;"
        " ties: larger parameter, then refine over patience",
        "stratum_sizes": list(STRATUM_SIZES),
        "tolerance": TOLERANCE,
        "candidates": table,
        "full_mean_iterations": summ["full"]["all"]["mean_iterations"],
        "eligible": elig,
        "selected": chosen,
        "selected_config": (
            None
            if chosen is None
            else {"refine_iterations": CONFIGS[chosen][0], "patience": CONFIGS[chosen][1]}
        ),
        "e1a_sha256": sha256(out / "e1a.jsonl"),
        "code_sha256": {c: sha256(Path(c)) for c in CODE},
    }
    (out / "selection.json").write_text(json.dumps(res, indent=1), "utf-8")
    _log(out, f"selection: {chosen}")
    return res


def _selected(out: Path) -> str:
    p = out / "selection.json"
    if not p.exists():
        raise SystemExit(
            "selection.json missing: run `select` first (E2 is never used for selection)"
        )
    sel = json.loads(p.read_text("utf-8"))
    if sel["selected"] is None:
        raise SystemExit("no candidate selected")
    return str(sel["selected"])


# ---------------------------------------------------------------------------- timing (no trace)
def timing_job(j: dict[str, Any]) -> dict[str, Any]:
    grid, start, goal, c, key = _query(j["seed_base"], j["family"], j["size"], j["map"], j["pair"])
    out = {**j, "time_s": {}, "iterations": {}}
    for label in ("full", j["selected"]):
        best = math.inf
        for _ in range(3):  # min of 3: reduce scheduling noise (the work is identical)
            res = _solve(grid, start, goal, c, key, label, trace=False)
            best = min(best, res.stats.time_s)
        out["time_s"][label] = best
        out["iterations"][label] = res.stats.iterations
    return out


def timing(out: Path, workers: int) -> int:
    sel = _selected(out)
    jobs = [j | {"selected": sel} for j in _static_jobs(E1A_SPEC) if j["size"] in STRATUM_SIZES]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        recs = list(ex.map(timing_job, jobs, chunksize=4))
    (out / "e1a_timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), "utf-8")
    _log(out, f"timing: {len(recs)} queries in {time.time() - t0:.0f}s (selected {sel})")
    return 0


# ---------------------------------------------------------------------------- E2 (reported only)
def e2_job(j: dict[str, Any]) -> dict[str, Any]:
    from robocoord.experiments import planning_runs as pr
    from robocoord.experiments.spec import PlannerSetting, load_spec

    spec = load_spec(E2_SPEC)
    k, m = CONFIGS[j["config"]]
    options: dict[str, Any] = {"max_iterations": BUDGET}
    if k is not None:
        options["refine_iterations"] = k
    if m is not None:
        options["patience"] = m
    spec = replace(spec, planners={"rrtstar": PlannerSetting("rrtstar", "rrtstar", options)})
    params = {k2: j[k2] for k2 in ("obstacles", "events", "team", "seed")} | {"planner": "rrtstar"}
    t0 = time.perf_counter()
    outcome, metrics = pr.run_episode(spec, params)
    return {**j, "outcome": outcome, "metrics": metrics, "wall_s": time.perf_counter() - t0}


def e2(out: Path, workers: int) -> int:
    from robocoord.experiments.spec import load_spec

    sel = _selected(out)
    spec = load_spec(E2_SPEC)
    f = spec.factors
    jobs = [
        {"config": cfg, "obstacles": int(o), "events": str(e), "team": int(t), "seed": int(s)}
        for s in f["seed"]
        for o in f["obstacles"]
        for e in f["events"]
        for t in f["team"]
        for cfg in ("full", sel)
    ]
    target = out / "e2.jsonl"

    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (r["config"], r["obstacles"], r["events"], r["team"], r["seed"])

    done = {key(r) for r in _records(target)}
    todo = [j for j in jobs if key(j) not in done]
    _log(out, f"e2: {len(done)} done, {len(todo)} to do (selected {sel})")
    t0 = time.time()
    with target.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(workers) as ex:
        for rec in ex.map(e2_job, todo, chunksize=1):
            fh.write(json.dumps(rec) + "\n")
    _log(out, f"e2: finished {len(todo)} new in {time.time() - t0:.0f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("e1a", "e1c", "select", "timing", "e2"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "e1a":
        return _run_static(a.out, "e1a", _static_jobs(E1A_SPEC), a.workers)
    if a.cmd == "e1c":
        return _run_static(a.out, "e1c", _static_jobs(E1C_SPEC), a.workers)
    if a.cmd == "select":
        res = select(a.out)
        print(json.dumps({k: res[k] for k in ("eligible", "selected", "selected_config")}))
        return 0
    if a.cmd == "timing":
        return timing(a.out, a.workers)
    return e2(a.out, a.workers)


if __name__ == "__main__":
    sys.exit(main())
