"""Issue 6A Phase 1 — analysis of the P1 repair evaluation (protocol: the Phase 1 document §3).

Reuses the Phase 3 analysis helpers, which import the frozen ``block_c_analysis.py`` estimator
(paired ratio of sums, seed-cluster bootstrap-t B = 10 000, Holm). Conditions C1–C4 are analysed
separately; each compares repair 3 with repair 0 under identical settings.

Usage::

    python tools/phase1/phase1_analysis.py --out results/dev6a/phase1/eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase3"))
sys.path.insert(0, str(HERE.parent / "selection"))
import block_c_analysis as bca  # noqa: E402
import phase1 as p1  # noqa: E402
import phase3 as p3  # noqa: E402
import phase3_analysis as pa  # noqa: E402

ALPHA = {"P1": 0.03, "P2": 0.005, "E": 0.01, "P4": 0.005}
CONDITIONS = {
    "C1": ("astar", "rsd"),
    "C2": ("astar", "off"),
    "C3": ("rrtstar", "rsd"),
    "C4": ("rrtstar", "off"),
}


class Data(pa.Data):
    def __init__(self, out: Path) -> None:
        self.recs = {}
        self.files = {}
        for s in p1.STAGES:
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


def calls_per_100(r: Mapping[str, Any]) -> tuple[float, float]:
    return 100.0 * r["planner_calls"], float(r["robots"] * r["steps"])


def work(planner: str) -> Callable[[Mapping[str, Any]], tuple[float, float]]:
    field = "expansions" if planner == "astar" else "iterations"
    return lambda r: (float(r["planning"][field]), float(r["robots"] * r["steps"]))


def conflicts(r: Mapping[str, Any]) -> tuple[float, float]:
    return float(r["conflicts_total"]), 1.0


def g0(d: Data, planner: str, base: str, rep: str) -> dict[str, Any]:
    from robocoord.control.supervisor import frozen_params

    fp = frozen_params(planner)
    frozen = {k: getattr(fp, k) for k in fp.__slots__}
    recs = [
        d.get(a, run, planner, lv, s)
        for a in (base, rep)
        for run in p3.RUNS
        for lv in p3.LEVELS
        for s in d.seeds(planner)
    ]
    ok = [r for r in recs if r["status"] == "ok"]
    sup = [r for r in ok if r.get("supervisor_stats")]
    burst = [
        p3.key(r)
        for r in sup
        if r["supervisor_stats"]["monitor"]["max_recoveries_in_a_leg"]
        > math.ceil(r["steps"] / (min(fp.stall_window, fp.osc_window) + 1))
    ]
    inst: dict[tuple[str, str, int], set[str]] = {}
    for r in ok:
        inst.setdefault((r["run"], r["level"], r["seed"]), set()).add(r["instance_sha256"])
    res = {
        "episodes": len(recs),
        "errors": [p3.key(r) for r in recs if r["status"] != "ok"],
        "physical_collisions": sum(r["physical_collisions"] for r in ok),
        "inv_total": sum(r["conflicts"]["INV"] for r in ok),
        "supervisor_introduced_invalid_moves": sum(r.get("sup_invalid", 0) for r in ok),
        "max_consecutive_r_losses": max(
            (r["supervisor_stats"]["max_consecutive_r_losses"] for r in sup), default=0
        ),
        "fairness_bound_ok": all(
            r["supervisor_stats"]["max_consecutive_r_losses"] <= fp.fairness_losses for r in sup
        ),
        "burst_bound_violations": burst,
        "params_frozen": all(r["supervisor_stats"]["params"] == frozen for r in sup),
        "repair_arm_radius_ok": all(
            r.get("repair_radius") == p1.REPAIR_RADIUS for r in ok if r["arm"] == rep
        ),
        "base_arm_repair_off": all("repair_radius" not in r for r in ok if r["arm"] == base),
        "instances_identical_across_arms": all(len(v) == 1 for v in inst.values()),
    }
    res["pass"] = bool(
        not res["errors"]
        and res["physical_collisions"] == 0
        and res["inv_total"] == 0
        and res["supervisor_introduced_invalid_moves"] == 0
        and res["fairness_bound_ok"]
        and not burst
        and res["params_frozen"]
        and res["repair_arm_radius_ok"]
        and res["base_arm_repair_off"]
        and res["instances_identical_across_arms"]
    )
    return res


def phase3_reproduction(d: Data) -> dict[str, Any]:
    """G0.1: the repair-off arms equal the stored Phase 3 records (all non-timing fields)."""
    skip = set(p1.TIMING) | {"stage", "arm", "planning"}
    out = {}
    for (arm, planner), (path, p3arm) in p1.PHASE3_RECORDS.items():
        n = same = 0
        diffs: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["arm"] != p3arm:
                    continue
                mine = d.recs.get((arm, r["run"], planner, r["level"], int(r["seed"])))
                if mine is None:
                    continue
                n += 1
                bad = [k for k in r if k not in skip and r[k] != mine.get(k)]
                same += not bad
                if bad and len(diffs) < 10:
                    diffs.append(
                        {"run": r["run"], "level": r["level"], "seed": r["seed"], "fields": bad}
                    )
        out[f"{arm}|{planner}"] = {"compared": n, "identical": same, "differences": diffs}
    return out


def analyse_condition(d: Data, planner: str, sup: str) -> dict[str, Any]:
    base, rep = f"{sup}_r0", f"{sup}_r{p1.REPAIR_RADIUS}"
    idx = bca.boot_index(len(d.seeds(planner)))
    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA}
    fam_runs = pa.FAMILY_RUNS
    for f, runs in fam_runs.items():
        fam["P1"].append(
            pa._row(
                "completion", f, d.pair(base, rep, runs, planner, pa.simple(pa.completion)), idx
            )
        )
    for run in p3.RUNS:
        fam["P2"].append(
            pa._row(
                "completion", run, d.pair(base, rep, [run], planner, pa.simple(pa.completion)), idx
            )
        )
    for f, runs in fam_runs.items():
        fam["E"].append(
            pa._row(
                "planner_calls_per_100_robot_steps",
                f,
                d.pair(base, rep, runs, planner, pa.simple(calls_per_100)),
                idx,
            )
        )
        fam["E"].append(
            pa._row(
                "planner_work_per_robot_step",
                f,
                d.pair(base, rep, runs, planner, pa.simple(work(planner))),
                idx,
            )
        )
    for f, runs in fam_runs.items():
        fam["P4"].append(
            pa._row(
                "conflicts_per_episode",
                f,
                d.pair(base, rep, runs, planner, pa.simple(conflicts)),
                idx,
            )
        )
        fam["P4"].append(
            pa._row(
                "top_priority_common_flow_time",
                f,
                d.pair(base, rep, runs, planner, pa.common_top_flow),
                idx,
            )
        )
        a = d.pair(base, rep, runs, planner, pa.simple(pa.top_undelivered))
        fam["P4"].append(
            pa._row("top_priority_undelivered", f, a if a[0].sum() + a[2].sum() > 0 else None, idx)
        )
    for k, rows in fam.items():
        pa.holm_family(rows, ALPHA[k])
    g = g0(d, planner, base, rep)
    d1 = {}
    for f, runs in fam_runs.items():
        rs = [
            d.get(base, run, planner, lv, s)
            for run in runs
            for lv in p3.LEVELS
            for s in d.seeds(planner)
        ]
        d1[f] = 1.0 / (sum(r["released"] for r in rs) / len(rs))
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
    no_harm = g["pass"] and not harm and all(lower.values()) and not veto
    if not g["pass"] or harm or veto:
        verdict = "reject"
    elif not all(lower.values()):
        verdict = "inconclusive"
    else:
        verdict = "adopt" if benefit else "no benefit"
    supportive = {}
    for f, runs in fam_runs.items():
        for name, fn in (
            ("undelivered_per_episode", pa.per_episode("undelivered")),
            ("stuck_total_per_episode", pa.stuck_count(None)),
            (
                "t3_calls_per_100_robot_steps",
                lambda r: (
                    100.0 * r["planning"]["by_trigger"].get("T3", 0),
                    float(r["robots"] * r["steps"]),
                ),
            ),
            (
                "deviations_per_100_robot_steps",
                lambda r: (100.0 * r["planning"]["deviations"], float(r["robots"] * r["steps"])),
            ),
            ("steps_per_episode", lambda r: (float(r["steps"]), 1.0)),
        ):
            supportive[f"{f}|{name}"] = pa.infer(
                d.pair(base, rep, runs, planner, pa.simple(fn)), idx
            )
    return {
        "arms": [base, rep],
        "g0": g,
        "delta1": d1,
        "families": fam,
        "decision": {
            "no_harm": no_harm,
            "harm": harm,
            "lower_bound_ok": lower,
            "vetoes": veto,
            "benefit": benefit,
            "verdict": verdict,
        },
        "supportive_nominal": supportive,
        "per_level_nominal": pa.per_level(d, base, rep, planner, idx),
        "seed_heterogeneity": pa.seed_heterogeneity(d, base, rep, planner),
        "describe": {
            a: {f: describe(d, a, planner, runs) for f, runs in fam_runs.items()}
            for a in (base, rep)
        },
    }


def describe(d: Data, arm: str, planner: str, runs: Sequence[str]) -> dict[str, Any]:
    out = pa.describe_arm(d, arm, planner, runs)
    rs = [
        d.get(arm, run, planner, lv, s)
        for run in runs
        for lv in p3.LEVELS
        for s in d.seeds(planner)
    ]
    rsteps = sum(r["robots"] * r["steps"] for r in rs)
    trig: Counter[str] = Counter()
    for r in rs:
        trig.update(r["planning"]["by_trigger"])
    tot = sum(trig.values())
    dev = sum(r["planning"]["deviations"] for r in rs)
    rep = sum(r["planning"]["repairs"] for r in rs)
    per = np.array([(r["wall_s"] - r.get("observer_s", 0.0)) / r["steps"] for r in rs])
    plan = np.array([r["planning_s"] / r["steps"] for r in rs])
    med = float(np.median(per))
    outl = per > 50 * med
    out.update(
        {
            "planner_calls_per_100_robot_steps": 100 * sum(r["planner_calls"] for r in rs) / rsteps,
            "calls_by_trigger_share": {k: v / tot for k, v in sorted(trig.items())} if tot else {},
            "t3_calls_per_100_robot_steps": 100 * trig.get("T3", 0) / rsteps,
            "deviations_per_100_robot_steps": 100 * dev / rsteps,
            "repairs": rep,
            "repair_success_share": rep / dev if dev else None,
            "expansions_per_robot_step": sum(r["planning"]["expansions"] for r in rs) / rsteps,
            "iterations_per_robot_step": sum(r["planning"]["iterations"] for r in rs) / rsteps,
            "step_ms_median": 1000 * med,
            "step_ms_iqr": [
                1000 * float(np.quantile(per, 0.25)),
                1000 * float(np.quantile(per, 0.75)),
            ],
            "planning_ms_per_step_median": 1000 * float(np.median(plan)),
            "planning_share_of_step_time": float(plan[~outl].sum() / per[~outl].sum()),
            "timing_outliers_flagged": int(outl.sum()),
            "steps_per_episode": sum(r["steps"] for r in rs) / len(rs),
            # Per-episode totals (descriptive): rates per robot-step hide longer episodes.
            "planner_calls_per_episode": sum(r["planner_calls"] for r in rs) / len(rs),
            "planner_work_per_episode": sum(
                r["planning"]["expansions" if planner == "astar" else "iterations"] for r in rs
            )
            / len(rs),
            "repairs_per_episode": rep / len(rs),
            "supervisor_changes_per_episode": sum(r.get("sup_changed", 0) for r in rs) / len(rs),
            "planning_s_per_episode_median": float(np.median([r["planning_s"] for r in rs])),
            "wall_s_per_episode_median": float(
                np.median([r["wall_s"] - r.get("observer_s", 0.0) for r in rs])
            ),
            "terminated_share": sum(bool(r["terminated"]) for r in rs) / len(rs),
        }
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = Data(args.out)
    res: dict[str, Any] = {
        "inputs_sha256": d.files,
        "analysis_sha256": pa.sha256(Path(__file__)),
        "block_c_analysis_sha256": pa.sha256(HERE.parent / "selection" / "block_c_analysis.py"),
        "alpha": ALPHA,
        "n_boot": bca.N_BOOT,
        "repair_radius": p1.REPAIR_RADIUS,
    }
    rp = args.out / "replay.json"
    res["replay"] = json.loads(rp.read_text("utf-8")) if rp.exists() else None
    res["g0_1_phase3_reproduction"] = phase3_reproduction(d)
    for c, (planner, sup) in CONDITIONS.items():
        if d.has(f"{sup}_r0", planner) and d.has(f"{sup}_r{p1.REPAIR_RADIUS}", planner):
            res[c] = analyse_condition(d, planner, sup)
            print(c, json.dumps(res[c]["decision"]))
    (args.out / "analysis.json").write_text(
        json.dumps(res, indent=1, sort_keys=True, default=str), "utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
