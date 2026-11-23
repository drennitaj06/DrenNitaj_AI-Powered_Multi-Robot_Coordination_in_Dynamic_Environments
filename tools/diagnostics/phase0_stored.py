"""Issue 6A Phase 0 — read-only diagnostics from **stored** records (no simulation, no replay).

Approved scope (2026-11-21): replanning and RRT* diagnostics from what the stored records contain,
and an explicit inventory of what they do not contain. Nothing is simulated, trained or changed;
the only output is a new directory (default ``results/dev6a/phase0``). Deterministic: every
iteration is over sorted keys, and the JSON is written with sorted keys.

Sections of the output:

* ``replanning.system`` — learned and non-learned controllers in the block B / C / D evaluation
  records: planner calls per 100 robot-steps (pooled), the stored per-episode deviation rate
  (T3 calls per 100 *non-idle* robot-steps), incomplete episodes, calls in successful vs
  incomplete episodes, and a rank correlation of the per-episode call rate with undelivered tasks.
  These records do **not** store calls by trigger nor non-idle robot-steps, so the T3 share of
  calls is not reconstructable; only the upper bound ``deviation rate / call rate`` is given.
* ``replanning.by_trigger`` — exact calls by trigger where records store them: the Issue #4
  non-learned system comparison (``e5_*``), E3b and E2 (plan follower).
* ``replanning.failure_modes`` — the Phase 6A.0 replay of the 928 greedy block-B failures: call
  rate per failed episode against the number of oscillating / frozen stuck robots.
* ``rrt.static_queries`` — E1a (n_max 500 / 2 000 / 5 000) and E1b (n_max 1 000–10 000): first
  solution iteration, budget exhaustion and iterations after the first solution, per map family,
  size and budget. One call per query, static snapshot, Euclidean cost.
* ``rrt.convergence`` — E1c: cost at fixed iteration counts relative to the first solution.
* ``rrt.replanning`` — E2 (plan follower, n_max 1 000): status mix and iterations per call.
* ``rrt.learned_controllers`` — block B / C / D RRT* episodes: only calls and wall time exist.
* ``missing`` — the instrumentation needed for what cannot be answered.

Usage::

    python tools/diagnostics/phase0_stored.py --out results/dev6a/phase0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]

#: Block B / C / D system records (one record per episode).
SYSTEM_SOURCES = {
    "dev5_comparison": "results/dev5/comparison/records.jsonl",
    "dev6_baseline_matched": "results/dev6/baseline_matched/records.jsonl",
    "dev6_ppo": "results/dev6/ppo/records.jsonl",
    "dev6_stochastic": "results/dev6/stochastic/records.jsonl",
    "block_c": "results/dev6a/block_c/records.jsonl",
    "block_d": "results/dev6a/block_d/records.jsonl",
}
TRIGGER_SOURCES = {
    "e5_prelim": "results/experiments/e5_prelim/records.jsonl",
    "e5_prelim_reference": "results/experiments/e5_prelim_reference/records.jsonl",
    "e5_recovery": "results/experiments/e5_recovery/records.jsonl",
    "e5_recovery_reference": "results/experiments/e5_recovery_reference/records.jsonl",
    "e3b_priority": "results/experiments/e3b_priority/records.jsonl",
    "e2_replanning_r2": "results/experiments/e2_replanning_r2/records.jsonl",
}
REPLAY_6A0 = "results/dev6a/6a0_diagnostics/replay_episodes.jsonl"
E1A = "results/experiments/e1a_planners_static_r3/records.jsonl"
E1B = "results/experiments/e1b_neighbor_index_r2/records.jsonl"
E1C = "results/experiments/e1c_convergence/records.jsonl"
E2 = "results/experiments/e2_replanning_r2/records.jsonl"
#: Evaluation levels fix 8 robots (``marl.scenarios.LEVELS``); Issue #5 records omit the field.
EVAL_ROBOTS = 8
TRIGGERS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dist(values: Iterable[float]) -> dict[str, Any]:
    """Count, mean and fixed quantiles (NumPy ``linear``) of a sample; empty -> count 0."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"n": 0}
    qs = np.quantile(v, QUANTILES)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        **{f"q{int(q * 100):02d}": float(x) for q, x in zip(QUANTILES, qs, strict=True)},
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho, or ``None`` when undefined (fewer than 3 points or a constant input)."""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


# --------------------------------------------------------------------------- replanning


def system_key(source: str, r: Mapping[str, Any]) -> str:
    policy = r.get("label") or r.get("controller")
    arm = r.get("arm") or ""
    return "|".join(
        str(x)
        for x in (
            source,
            r.get("controller"),
            policy,
            arm,
            r.get("planner"),
            r.get("mode", "greedy"),
            r.get("shield", "off"),
            r.get("allocator", "auction"),
        )
    )


def system_replanning(sources: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Per configuration: call rate, stored deviation rate, and the association with incomplete
    episodes. ``robots`` defaults to :data:`EVAL_ROBOTS` where the record omits it."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source, recs in sources.items():
        for r in recs:
            if r.get("status", "ok") != "ok":
                continue
            groups[system_key(source, r)].append(r)
    out: dict[str, Any] = {}
    for key in sorted(groups):
        rs = groups[key]
        rate = [
            100.0 * r["planner_calls"] / (r.get("robots", EVAL_ROBOTS) * r["steps"]) for r in rs
        ]
        und = [r["released"] - r["delivered"] for r in rs]
        dev = [
            r["deviation_rate_per_100"] for r in rs if r.get("deviation_rate_per_100") is not None
        ]
        calls = sum(r["planner_calls"] for r in rs)
        rsteps = sum(r.get("robots", EVAL_ROBOTS) * r["steps"] for r in rs)
        pooled = 100.0 * calls / rsteps
        inc = [x for x, u in zip(rate, und, strict=True) if u > 0]
        suc = [x for x, u in zip(rate, und, strict=True) if u == 0]
        out[key] = {
            "episodes": len(rs),
            "incomplete": len(inc),
            "calls_per_100_robot_steps_pooled": pooled,
            "calls_per_100_robot_steps": dist(rate),
            "calls_per_100_robot_steps_successful": dist(suc),
            "calls_per_100_robot_steps_incomplete": dist(inc),
            "deviation_rate_per_100_nonidle": dist(dev),
            "t3_share_upper_bound": (
                min(1.0, float(np.mean(dev)) / pooled) if dev and pooled else None
            ),
            "spearman_call_rate_vs_undelivered": spearman(rate, und),
            "robots_field_present": all("robots" in r for r in rs),
        }
    return out


def _calls_by_trigger(r: Mapping[str, Any]) -> Mapping[str, int] | None:
    m = r.get("metrics", {})
    return m.get("calls_by_trigger") or (m.get("planning") or {}).get("calls_by_trigger")


def _controller(r: Mapping[str, Any]) -> str:
    c = r.get("metrics", {}).get("controller")
    if isinstance(c, Mapping):
        return str(c.get("controller"))
    return str(c or r.get("params", {}).get("controller") or "plan_follower")


def trigger_breakdown(sources: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Exact calls by trigger, pooled per source × controller × planner × dynamics."""
    acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    eps: dict[str, int] = defaultdict(int)
    for source, recs in sources.items():
        for r in recs:
            cbt = _calls_by_trigger(r)
            if cbt is None:
                continue
            p = r.get("params", {})
            dyn = p.get("dynamics") or f"obstacles={p.get('obstacles')},events={p.get('events')}"
            planner = p.get("planner") or r.get("metrics", {}).get("planner")
            key = "|".join(
                str(x)
                for x in (
                    source,
                    _controller(r),
                    planner,
                    dyn,
                    p.get("team", ""),
                    p.get("allocator", ""),
                )
            )
            eps[key] += 1
            for t, n in cbt.items():
                acc[key][t] += int(n)
    out = {}
    for key in sorted(acc):
        total = sum(acc[key].values())
        out[key] = {
            "episodes": eps[key],
            "calls": total,
            "calls_by_trigger": {t: acc[key].get(t, 0) for t in TRIGGERS if t in acc[key]},
            "share_by_trigger": {t: acc[key][t] / total for t in TRIGGERS if t in acc[key]}
            if total
            else {},
        }
    return out


def failure_mode_association(replay: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """6A.0 failed episodes: call rate against oscillating / frozen stuck robots, per algorithm."""
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in replay:
        by[str(r["algorithm"])].append(r)
    out = {}
    for alg in sorted(by):
        rs = by[alg]
        rate = [
            100.0 * r["replayed"]["planner_calls"] / (EVAL_ROBOTS * r["replayed"]["steps"])
            for r in rs
        ]
        osc = [sum(rb["class"] == "oscillating" for rb in r["robots"]) for r in rs]
        frz = [sum(rb["class"] == "frozen" for rb in r["robots"]) for r in rs]
        loop = [sum(rb["class"] == "conflict_loop" for rb in r["robots"]) for r in rs]
        out[alg] = {
            "failed_episodes": len(rs),
            "calls_per_100_robot_steps": dist(rate),
            "spearman_rate_vs_oscillating_robots": spearman(rate, osc),
            "spearman_rate_vs_frozen_robots": spearman(rate, frz),
            "spearman_rate_vs_conflict_loop_robots": spearman(rate, loop),
            "rate_with_any_oscillating": dist(x for x, o in zip(rate, osc, strict=True) if o),
            "rate_without_oscillating": dist(x for x, o in zip(rate, osc, strict=True) if not o),
        }
    return out


# --------------------------------------------------------------------------- RRT*


def static_queries(
    e1a: Sequence[Mapping[str, Any]], e1b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per family × size × budget: success, exhaustion, first-solution iteration and the share of
    the budget run after the first solution (every call runs its whole budget)."""
    rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in e1a:
        planner = str(r["params"]["planner"])
        if not planner.startswith("rrtstar"):
            continue
        n_max = int(r["metrics"]["options"]["max_iterations"])
        rows[f"e1a|{r['params']['family']}|{r['params']['size']}|{n_max}"].append(r)
    for r in e1b:
        p = r["params"]
        rows[f"e1b|{p['family']}|{p['size']}|{p['n_max']}|{p['index']}"].append(r)
    out = {}
    for key in sorted(rows, key=lambda k: [int(x) if x.isdigit() else x for x in k.split("|")]):
        rs = rows[key]
        n_max = int(key.split("|")[3])
        ok = [r for r in rs if r["metrics"]["status"] == "ok"]
        fsi = [r["metrics"]["counters"]["first_solution_iteration"] for r in ok]
        its = [r["metrics"]["counters"]["iterations"] for r in rs]
        after = [r["metrics"]["counters"]["iterations"] - f for r, f in zip(ok, fsi, strict=True)]
        out[key] = {
            "queries": len(rs),
            "ok": len(ok),
            "budget_exhausted": sum(r["metrics"]["status"] == "budget_exhausted" for r in rs),
            "other_status": sum(
                r["metrics"]["status"] not in ("ok", "budget_exhausted") for r in rs
            ),
            "all_ran_full_budget": all(i == n_max for i in its),
            "first_solution_iteration": dist(fsi),
            "iterations_after_first_solution": dist(after),
            "share_of_budget_after_first_solution": dist(a / n_max for a in after),
            "share_of_all_iterations_spent_without_any_solution": (
                sum(n_max for r in rs if r["metrics"]["status"] != "ok") / (n_max * len(rs))
            ),
        }
    return out


def convergence(e1c: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """E1c: best cost at fixed iteration counts over the cost at the first checkpoint with a
    solution (``inf`` checkpoints are before the first solution)."""
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in e1c:
        by[f"{r['params']['family']}|{r['params']['size']}"].append(r)
    out = {}
    for key in sorted(by):
        rs = by[key]
        fsi = [
            r["metrics"]["counters"]["first_solution_iteration"]
            for r in rs
            if r["metrics"]["counters"]["first_solution_iteration"] >= 0
        ]
        ratios: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            cps = {int(k): float(v) for k, v in r["metrics"]["checkpoints"].items()}
            finite = [k for k in sorted(cps) if np.isfinite(cps[k])]
            if not finite:
                continue
            first = cps[finite[0]]
            for k in sorted(cps):
                if np.isfinite(cps[k]):
                    ratios[str(k)].append(cps[k] / first)
        out[key] = {
            "queries": len(rs),
            "first_solution_iteration": dist(fsi),
            "cost_over_first_found_cost": {
                k: dist(v) for k, v in sorted(ratios.items(), key=lambda kv: int(kv[0]))
            },
        }
    return out


def replanning_rrt(e2: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """E2 RRT* episodes: per-call status mix and iterations per call (n_max = 1 000)."""
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in e2:
        p = r["params"]
        if p.get("planner") != "rrtstar":
            continue
        by[f"team={p['team']}|obstacles={p['obstacles']}|events={p['events']}"].append(r)
    out = {}
    for key in sorted(by):
        rs = by[key]
        mix: dict[str, int] = defaultdict(int)
        for r in rs:
            for s, n in r["metrics"]["planning"]["status_mix"].items():
                mix[s] += int(n)
        calls = sum(r["metrics"]["planning"]["calls"] for r in rs)
        its = sum(r["metrics"]["planning"]["work"]["iterations"] for r in rs)
        out[key] = {
            "episodes": len(rs),
            "calls": calls,
            "status_mix": dict(sorted(mix.items())),
            "budget_exhausted_share": mix.get("budget_exhausted", 0) / calls if calls else None,
            "iterations_per_call": its / calls if calls else None,
            "latency_median_s": dist(r["metrics"]["latency_s"]["median"] for r in rs),
        }
    return out


def learned_rrt(sources: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Block B / C / D RRT* episodes: what is stored (calls, wall time) per configuration."""
    by: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source, recs in sources.items():
        for r in recs:
            if r.get("planner") == "rrtstar" and r.get("status", "ok") == "ok":
                by[system_key(source, r)].append(r)
    return {
        k: {
            "episodes": len(v),
            "calls_per_episode": dist(r["planner_calls"] for r in v),
            "wall_s_per_episode": dist(r["wall_s"] for r in v if "wall_s" in r),
            "wall_s_per_call_upper_bound": dist(
                r["wall_s"] / r["planner_calls"]
                for r in v
                if r.get("planner_calls") and "wall_s" in r
            ),
        }
        for k, v in sorted(by.items())
    }


MISSING = {
    "madql_q_values": "No stored record holds observations, masks or Q-values; the 6A.0 replay "
    "stored only per-robot counts and classes. Needs a deterministic replay that evaluates the "
    "MADQL Q-network on each step's observations (no new episodes; the stored ones reproduced).",
    "calls_by_trigger_learned": "Block B/C/D records store planner_calls and the T3 deviation "
    "rate per 100 non-idle robot-steps only; neither calls_by_trigger nor non-idle robot-steps. "
    "The engine computes calls_by_trigger (extra['planning']) but run_episode does not store it.",
    "rrt_per_call_learned": "Block B/C/D RRT* records hold no per-call status, first-solution "
    "iteration or iteration counts.",
    "rrt_first_solution_in_replanning": "E2 stores per-episode status mix and total iterations, "
    "not the first-solution iteration of each call.",
    "unnecessary_replanning": "Whether a replan was necessary (whether the old path suffix was "
    "still valid and optimal from the new cell) was never recorded; it needs the replay.",
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    system = {k: read_jsonl(p) for k, p in SYSTEM_SOURCES.items()}
    trig = {k: read_jsonl(p) for k, p in TRIGGER_SOURCES.items()}
    used = sorted(
        {*SYSTEM_SOURCES.values(), *TRIGGER_SOURCES.values(), REPLAY_6A0, E1A, E1B, E1C, E2}
    )
    res = {
        "inputs": {p: {"sha256": sha256(p), "records": len(read_jsonl(p))} for p in used},
        "replanning": {
            "system": system_replanning(system),
            "by_trigger": trigger_breakdown(trig),
            "failure_modes": failure_mode_association(read_jsonl(REPLAY_6A0)),
        },
        "rrt": {
            "static_queries": static_queries(read_jsonl(E1A), read_jsonl(E1B)),
            "convergence": convergence(read_jsonl(E1C)),
            "replanning": replanning_rrt(read_jsonl(E2)),
            "learned_controllers": learned_rrt(system),
        },
        "missing": MISSING,
        "notes": {"eval_robots_default": EVAL_ROBOTS, "quantile_method": "numpy linear"},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stored_diagnostics.json").write_text(
        json.dumps(res, indent=1, sort_keys=True), "utf-8"
    )
    print(f"wrote {args.out / 'stored_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
