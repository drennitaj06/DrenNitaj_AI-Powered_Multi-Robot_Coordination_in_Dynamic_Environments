"""Issue 6A Phase 2 — analysis of the P2 Block C evaluation (protocol: the Phase 2 document §5).

The estimator, the families, α and the decision rules are those of Phase 1 (§3.2, §3.4), applied
to the arms ``full`` (1 000 iterations) and ``p2`` (the frozen stopping rule) in two conditions:
L (9 learned policies, R + S + D) and B (the non-learned baseline). Everything is computed by the
frozen ``block_c_analysis.py`` through the Phase 3 / Phase 1 helpers.

Usage::

    python tools/phase2/phase2_analysis.py --out results/dev6a/phase2/eval
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase1"))
sys.path.insert(0, str(HERE.parent / "phase3"))
sys.path.insert(0, str(HERE.parent / "selection"))
import block_c_analysis as bca  # noqa: E402
import phase1_analysis as p1a  # noqa: E402
import phase2 as p2  # noqa: E402
import phase3 as p3  # noqa: E402
import phase3_analysis as pa  # noqa: E402

ALPHA = p1a.ALPHA  # {"P1": 0.03, "P2": 0.005, "E": 0.01, "P4": 0.005}
PLANNER = "rrtstar"
BASE, NEW = "full", "p2"
CONDITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "L": dict(pa.FAMILY_RUNS),
    "B": {"baseline": (p2.BASELINE_RUN,)},
}


class Data(p1a.Data):
    def __init__(self, out: Path) -> None:
        self.recs = {}
        self.files = {}
        for s in p2.STAGES:
            path = out / f"records_{s}.jsonl"
            self.files[str(path)] = pa.sha256(path) if path.exists() else None
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    k = p3.key(r)
                    if k in self.recs:
                        raise SystemExit(f"duplicate record {k}")
                    self.recs[k] = r

    def recs_of(self, arm: str, runs: Sequence[str]) -> list[dict[str, Any]]:
        return [
            self.get(arm, run, PLANNER, lv, s)
            for run in runs
            for lv in p3.LEVELS
            for s in self.seeds(PLANNER)
        ]


# --------------------------------------------------------------------------- integrity
def completeness(d: Data) -> dict[str, Any]:
    expected = {
        (arm, run, PLANNER, lv, s)
        for st, runs in p2.STAGES.items()
        for run in runs
        for lv in p3.LEVELS
        for s in p3.BLOCK_C_RRT
        for arm in p2.ARMS
    }
    have = set(d.recs)
    return {
        "expected": len(expected),
        "present": len(have & expected),
        "missing": len(expected - have),
        "unexpected": len(have - expected),
        "complete": have == expected,
    }


def g0(d: Data, cond: str) -> dict[str, Any]:
    from robocoord.control.supervisor import frozen_params

    runs = [r for rs in CONDITIONS[cond].values() for r in rs]
    recs = [r for a in (BASE, NEW) for r in d.recs_of(a, runs)]
    ok = [r for r in recs if r["status"] == "ok"]
    sel = p2.selected()
    res: dict[str, Any] = {
        "episodes": len(recs),
        "errors": [p3.key(r) for r in recs if r["status"] != "ok"],
        "physical_collisions": sum(r["physical_collisions"] for r in ok),
        "inv_total": sum(r["conflicts"]["INV"] for r in ok),
        "invalid_reference_paths": sum(r["path_checks"]["invalid"] for r in ok),
        "reference_paths_checked": sum(r["path_checks"]["checked"] for r in ok),
        "p2_arm_rule_ok": all(
            r.get("rrt_patience") == sel["patience"] and r.get("rrt_refine") == sel["refine"]
            for r in ok
            if r["arm"] == NEW
        ),
        "full_arm_rule_absent": all(
            "rrt_patience" not in r and "rrt_refine" not in r for r in ok if r["arm"] == BASE
        ),
        "full_arm_never_stopped_early": all(
            r["rrt"]["stopped_early"] == 0 for r in ok if r["arm"] == BASE
        ),
        "repair_off": all("repair_radius" not in r for r in ok),
    }
    inst: dict[tuple[str, str, int], set[str]] = {}
    for r in ok:
        inst.setdefault((r["run"], r["level"], r["seed"]), set()).add(r["instance_sha256"])
    res["instances_identical_across_arms"] = all(len(v) == 1 for v in inst.values())
    if cond == "L":
        fp = frozen_params(PLANNER)
        frozen = {k: getattr(fp, k) for k in fp.__slots__}
        sup = [r for r in ok if r.get("supervisor_stats")]
        res.update(
            {
                "supervised_episodes": len(sup),
                "supervisor_mode_ok": all(
                    r.get("supervisor") == "R+S" and r["deadlock"] for r in ok
                ),
                "supervisor_introduced_invalid_moves": sum(r.get("sup_invalid", 0) for r in ok),
                "fairness_bound_ok": all(
                    r["supervisor_stats"]["max_consecutive_r_losses"] <= fp.fairness_losses
                    for r in sup
                ),
                "burst_bound_violations": [
                    p3.key(r)
                    for r in sup
                    if r["supervisor_stats"]["monitor"]["max_recoveries_in_a_leg"]
                    > math.ceil(r["steps"] / (min(fp.stall_window, fp.osc_window) + 1))
                ],
                "params_frozen": all(r["supervisor_stats"]["params"] == frozen for r in sup),
            }
        )
    else:
        res["no_supervisor"] = all("supervisor" not in r for r in ok)
        res["resolver_on"] = all(r["deadlock"] for r in ok)
    checks = [
        not res["errors"],
        res["physical_collisions"] == 0,
        res["inv_total"] == 0,
        res["invalid_reference_paths"] == 0,
        res["p2_arm_rule_ok"],
        res["full_arm_rule_absent"],
        res["full_arm_never_stopped_early"],
        res["repair_off"],
        res["instances_identical_across_arms"],
    ]
    if cond == "L":
        checks += [
            res["supervised_episodes"] == len(ok),
            res["supervisor_mode_ok"],
            res["supervisor_introduced_invalid_moves"] == 0,
            res["fairness_bound_ok"],
            not res["burst_bound_violations"],
            res["params_frozen"],
        ]
    else:
        checks += [res["no_supervisor"], res["resolver_on"]]
    res["pass"] = bool(all(checks))
    return res


def phase1_reproduction(d: Data) -> dict[str, Any]:
    """G0.1: the L ``full`` arm equals Phase 1 ``records_1b`` arm ``rsd_r0`` (non-timing fields)."""
    path, arm = p2.PHASE1_RECORDS
    skip = set(p2.TIMING) | {"stage", "arm", "planning"} | set(p2.ADDED)
    n = same = 0
    diffs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] != arm:
                continue
            mine = d.recs.get((BASE, r["run"], PLANNER, r["level"], int(r["seed"])))
            if mine is None:
                continue
            n += 1
            keys = (set(r) | set(mine)) - skip
            bad = sorted(k for k in keys if r.get(k) != mine.get(k))
            same += not bad
            if bad and len(diffs) < 10:
                diffs.append(
                    {"run": r["run"], "level": r["level"], "seed": r["seed"], "fields": bad}
                )
    return {"compared": n, "identical": same, "differences": diffs, "pass": n == 1080 and same == n}


# --------------------------------------------------------------------------- outcomes
def iterations_per_robot_step(r: Mapping[str, Any]) -> tuple[float, float]:
    return float(r["rrt"]["iterations"]), float(r["robots"] * r["steps"])


def analyse(d: Data, cond: str) -> dict[str, Any]:
    fams = CONDITIONS[cond]
    idx = bca.boot_index(len(d.seeds(PLANNER)))
    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA}
    for f, runs in fams.items():
        fam["P1"].append(
            pa._row(
                "completion", f, d.pair(BASE, NEW, runs, PLANNER, pa.simple(pa.completion)), idx
            )
        )
    if cond == "L":
        for run in p3.RUNS:
            fam["P2"].append(
                pa._row(
                    "completion",
                    run,
                    d.pair(BASE, NEW, [run], PLANNER, pa.simple(pa.completion)),
                    idx,
                )
            )
    for f, runs in fams.items():
        fam["E"].append(
            pa._row(
                "planner_calls_per_100_robot_steps",
                f,
                d.pair(BASE, NEW, runs, PLANNER, pa.simple(p1a.calls_per_100)),
                idx,
            )
        )
        fam["E"].append(
            pa._row(
                "rrt_iterations_per_robot_step",
                f,
                d.pair(BASE, NEW, runs, PLANNER, pa.simple(iterations_per_robot_step)),
                idx,
            )
        )
    for f, runs in fams.items():
        fam["P4"].append(
            pa._row(
                "conflicts_per_episode",
                f,
                d.pair(BASE, NEW, runs, PLANNER, pa.simple(p1a.conflicts)),
                idx,
            )
        )
        fam["P4"].append(
            pa._row(
                "top_priority_common_flow_time",
                f,
                d.pair(BASE, NEW, runs, PLANNER, pa.common_top_flow),
                idx,
            )
        )
        a = d.pair(BASE, NEW, runs, PLANNER, pa.simple(pa.top_undelivered))
        fam["P4"].append(
            pa._row("top_priority_undelivered", f, a if a[0].sum() + a[2].sum() > 0 else None, idx)
        )
    for k, rows in fam.items():
        if rows:
            pa.holm_family(rows, ALPHA[k])
    g = g0(d, cond)
    d1 = {
        f: 1.0 / (sum(r["released"] for r in d.recs_of(BASE, runs)) / len(d.recs_of(BASE, runs)))
        for f, runs in fams.items()
    }
    harm = [r["unit"] for r in fam["P1"] + fam["P2"] if r["rejected"] and r["inf"]["estimate"] < 0]
    lower = {r["unit"]: r["inf"]["ci95"][0] > -d1[r["unit"]] for r in fam["P1"]}
    veto = [
        f"{r['unit']}|{r['hypothesis']}"
        for r in fam["P4"]
        if r["rejected"] and r["inf"]["estimate"] > 0
    ]
    benefit = [
        f"{r['unit']}|{r['hypothesis']}"
        for r in fam["E"]
        if r["rejected"] and r["inf"]["estimate"] < 0
    ]
    if not g["pass"] or harm or veto:
        verdict = "reject"
    elif not all(lower.values()):
        verdict = "inconclusive"
    else:
        verdict = "adopt" if benefit else "no benefit"
    supportive: dict[str, Any] = {}
    for f, runs in fams.items():
        for name, fn in (
            ("undelivered_per_episode", pa.per_episode("undelivered")),
            ("stuck_total_per_episode", pa.stuck_count(None)),
            ("steps_per_episode", lambda r: (float(r["steps"]), 1.0)),
            (
                "rrt_iterations_per_call",
                lambda r: (float(r["rrt"]["iterations"]), float(r["rrt"]["calls"])),
            ),
            ("rrt_time_s_per_episode", lambda r: (float(r["rrt"]["time_s_sum"]), 1.0)),
            (
                "moves_per_successful_plan",
                lambda r: (
                    float(r["rrt"]["path_moves_ok_sum"]),
                    float(r["rrt"]["path_moves_ok_n"]),
                ),
            ),
            (
                "conflicts_per_100_robot_steps",
                lambda r: (100.0 * r["conflicts_total"], float(r["robots"] * r["steps"])),
            ),
            (
                "mean_flow_time_delivered",
                lambda r: ((r["mean_flow_time"] or 0.0) * r["delivered"], float(r["delivered"])),
            ),
            ("wait_ratio", lambda r: (float(r["wait_ratio"]), 1.0)),
        ):
            supportive[f"{f}|{name}"] = pa.infer(
                d.pair(BASE, NEW, runs, PLANNER, pa.simple(fn)), idx
            )
    return {
        "arms": [BASE, NEW],
        "g0": g,
        "delta1": d1,
        "families": {k: v for k, v in fam.items() if v},
        "decision": {
            "no_harm": g["pass"] and not harm and all(lower.values()) and not veto,
            "harm": harm,
            "lower_bound_ok": lower,
            "vetoes": veto,
            "benefit": benefit,
            "verdict": verdict,
        },
        "supportive_nominal": supportive,
        "per_level_nominal": per_level(d, fams, idx),
        "seed_heterogeneity": seed_heterogeneity(d, fams),
        "describe": {a: {f: describe(d, a, runs) for f, runs in fams.items()} for a in (BASE, NEW)},
        "breakdown": {
            a: breakdown(d, a, [r for rs in fams.values() for r in rs]) for a in (BASE, NEW)
        },
    }


def per_level(d: Data, fams: Mapping[str, Sequence[str]], idx: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seeds = list(d.seeds(PLANNER))
    for f, runs in fams.items():
        for lv in p3.LEVELS:
            a = np.zeros((4, len(seeds)))
            for j, s in enumerate(seeds):
                for run in runs:
                    a[:, j] += (
                        *pa.completion(d.get(BASE, run, PLANNER, lv, s)),
                        *pa.completion(d.get(NEW, run, PLANNER, lv, s)),
                    )
            inf = pa.infer(list(a), idx)
            out[f"{f}|{lv}"] = (
                None
                if inf is None
                else {k: inf[k] for k in ("estimate", "ci95", "theta_old", "theta_new")}
            )
    return out


def seed_heterogeneity(d: Data, fams: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    out = {}
    for run in [r for rs in fams.values() for r in rs]:
        better = worse = same = 0
        for s in d.seeds(PLANNER):
            do = sum(d.get(BASE, run, PLANNER, lv, s)["delivered"] for lv in p3.LEVELS)
            dn = sum(d.get(NEW, run, PLANNER, lv, s)["delivered"] for lv in p3.LEVELS)
            better += dn > do
            worse += dn < do
            same += dn == do
        out[run] = {"seeds_better": better, "seeds_worse": worse, "seeds_equal": same}
    return out


def _rrt(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = sum(r["rrt"]["calls"] for r in rs)
    ok_n = sum(r["rrt"]["path_moves_ok_n"] for r in rs)
    status: Counter[str] = Counter()
    for r in rs:
        status.update(r["rrt"]["status"])
    first_n = sum(r["rrt"]["first_solution_n"] for r in rs)
    return {
        "rrt_calls_per_episode": calls / len(rs),
        "rrt_iterations_per_call": sum(r["rrt"]["iterations"] for r in rs) / calls
        if calls
        else None,
        "share_ok_calls_stopped_early": sum(r["rrt"]["stopped_early"] for r in rs) / ok_n
        if ok_n
        else None,
        "status_share": {k: v / calls for k, v in sorted(status.items())} if calls else {},
        "budget_exhausted_calls": status.get("budget_exhausted", 0),
        "mean_first_solution_iteration": sum(r["rrt"]["first_solution_sum"] for r in rs) / first_n
        if first_n
        else None,
        "moves_per_successful_plan": sum(r["rrt"]["path_moves_ok_sum"] for r in rs) / ok_n
        if ok_n
        else None,
        "rrt_ms_per_call_mean": 1000 * sum(r["rrt"]["time_s_sum"] for r in rs) / calls
        if calls
        else None,
        "rrt_ms_per_call_median_of_episodes": statistics.median(
            r["rrt"]["time_ms_median"] for r in rs if r["rrt"]["time_ms_median"] is not None
        ),
        "rrt_s_per_episode_median": statistics.median(r["rrt"]["time_s_sum"] for r in rs),
        "rrt_s_total": sum(r["rrt"]["time_s_sum"] for r in rs),
        "tree_nodes_per_call": sum(r["rrt"]["tree_nodes_sum"] for r in rs) / calls
        if calls
        else None,
        "tree_nodes_max": max(r["rrt"]["tree_nodes_max"] for r in rs),
        "segment_checks_per_call": sum(r["rrt"]["segment_checks"] for r in rs) / calls
        if calls
        else None,
    }


def describe(d: Data, arm: str, runs: Sequence[str]) -> dict[str, Any]:
    rs = d.recs_of(arm, runs)
    out = p1a.describe(d, arm, PLANNER, runs)
    rsteps = sum(r["robots"] * r["steps"] for r in rs)
    delivered = sum(r["delivered"] for r in rs)
    out.update(_rrt(rs))
    out.update(
        {
            "delivered": delivered,
            "released": sum(r["released"] for r in rs),
            "conflicts_per_delivered": sum(r["conflicts_total"] for r in rs) / delivered
            if delivered
            else None,
            "conflicts_per_100_robot_steps": 100 * sum(r["conflicts_total"] for r in rs) / rsteps,
            "flow_time_delivered_mean": sum(
                (r["mean_flow_time"] or 0.0) * r["delivered"] for r in rs
            )
            / delivered
            if delivered
            else None,
            "top_priority_flow_time_mean": _top_flow(rs),
            "reference_paths_checked": sum(r["path_checks"]["checked"] for r in rs),
            "invalid_reference_paths": sum(r["path_checks"]["invalid"] for r in rs),
            "physical_collisions": sum(r["physical_collisions"] for r in rs),
            "invalid_moves": sum(r["conflicts"]["INV"] for r in rs),
            "errors": sum(r["status"] != "ok" for r in rs),
        }
    )
    return out


def _top_flow(rs: Sequence[Mapping[str, Any]]) -> float | None:
    v = [
        t[4] - t[2] for r in rs for t in r["tasks"] if t[1] == pa.TOP_PRIORITY and t[4] is not None
    ]
    return sum(v) / len(v) if v else None


def breakdown(d: Data, arm: str, runs: Sequence[str]) -> dict[str, Any]:
    """Per policy × level (descriptive)."""
    out: dict[str, Any] = {}
    for run in runs:
        for lv in p3.LEVELS:
            rs = [d.get(arm, run, PLANNER, lv, s) for s in d.seeds(PLANNER)]
            rel = sum(r["released"] for r in rs)
            rr = _rrt(rs)
            out[f"{run}|{lv}"] = {
                "completion": sum(r["delivered"] for r in rs) / rel,
                "undelivered": sum(r["undelivered"] for r in rs),
                "conflicts_per_episode": sum(r["conflicts_total"] for r in rs) / len(rs),
                "flow_time_delivered_mean": sum(
                    (r["mean_flow_time"] or 0.0) * r["delivered"] for r in rs
                )
                / max(1, sum(r["delivered"] for r in rs)),
                "steps_per_episode": sum(r["steps"] for r in rs) / len(rs),
                "rrt_iterations_per_call": rr["rrt_iterations_per_call"],
                "rrt_calls_per_episode": rr["rrt_calls_per_episode"],
                "rrt_s_per_episode_median": rr["rrt_s_per_episode_median"],
                "wall_s_per_episode_median": statistics.median(
                    r["wall_s"] - r.get("observer_s", 0.0) for r in rs
                ),
            }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = Data(args.out)
    comp = completeness(d)
    if not comp["complete"]:
        raise SystemExit(f"records incomplete: {comp}")
    res: dict[str, Any] = {
        "files_sha256": d.files,
        "completeness": comp,
        "g0_1_phase1_reproduction": phase1_reproduction(d),
        "selection": p2.selected(),
    }
    rp = args.out / "replay.json"
    if rp.exists():
        res["replay"] = json.loads(rp.read_text("utf-8"))
    for cond in CONDITIONS:
        res[cond] = analyse(d, cond)
        print(cond, json.dumps(res[cond]["decision"]))
    l_ok = res["L"]["decision"]["verdict"] == "adopt" and res["L"]["g0"]["pass"]
    res["recommendation"] = {
        "L": res["L"]["decision"]["verdict"],
        "B": res["B"]["decision"]["verdict"],
        "g0_1": res["g0_1_phase1_reproduction"]["pass"],
        "replay_identical": res.get("replay", {}).get("identical"),
        "p2_forward_default": bool(
            l_ok
            and res["B"]["decision"]["verdict"] != "reject"
            and res["g0_1_phase1_reproduction"]["pass"]
            and res.get("replay", {}).get("identical") is True
        ),
    }
    print("recommendation", json.dumps(res["recommendation"]))
    (args.out / "analysis.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
