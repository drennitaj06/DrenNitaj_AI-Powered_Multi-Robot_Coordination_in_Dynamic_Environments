"""Issue 6A Phase 4 — analysis of the MADQL-v2 Block C evaluation (protocol: the Phase 4 document
§5, written after the selection and before any Block C episode).

Arms ``hist`` (historical MADQL) and ``v2`` (the selected candidate), paired by instance and by
training seed (policy k of each arm), in the conditions FA (A\\*, R + S + D; primary), FR (RRT\\*,
R + S + D) and NA (A\\*, no supervisor). Estimator, bootstrap and Holm: the frozen
``block_c_analysis.py`` through the Phase 3 helpers (``phase3_analysis.infer``).

Usage::

    python tools/phase4/phase4_analysis.py --out results/dev6a/phase4/eval
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sub in ("phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))
import block_c_analysis as bca  # noqa: E402
import phase3 as p3  # noqa: E402
import phase3_analysis as pa  # noqa: E402
import phase4 as p4  # noqa: E402

ALPHA = {"P1": 0.03, "P2": 0.005, "E": 0.01, "P4": 0.005}
OLD, NEW = "hist", "v2"
FAMILY = "madql"
PairFn = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[float, float, float, float]]
F = Callable[[Mapping[str, Any]], tuple[float, float]]


# --------------------------------------------------------------------------- outcomes
def calls_per_100(r: Mapping[str, Any]) -> tuple[float, float]:
    return 100.0 * r["planner_calls"], float(r["robots"] * r["steps"])


def sup_per_100(r: Mapping[str, Any]) -> tuple[float, float]:
    return 100.0 * r.get("sup_changed", 0), float(r["robots"] * r["steps"])


def conflicts(r: Mapping[str, Any]) -> tuple[float, float]:
    return float(r["conflicts_total"]), 1.0


def madql_share(field: str) -> F:
    return lambda r: (float(r["madql"][field]), float(r["madql"]["nonidle_steps"]))


def common_flow(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Flow time (completion − release) of the tasks delivered in both arms (all priorities)."""
    da = {t[0]: t[4] - t[2] for t in a["tasks"] if t[4] is not None}
    db = {t[0]: t[4] - t[2] for t in b["tasks"] if t[4] is not None}
    common = sorted(set(da) & set(db))
    n = float(len(common))
    return float(sum(da[i] for i in common)), n, float(sum(db[i] for i in common)), n


def e_family(cond: str) -> list[tuple[str, PairFn]]:
    third: tuple[str, PairFn] = (
        ("supervisor_changes_per_100_robot_steps", pa.simple(sup_per_100))
        if cond != "NA"
        else ("stuck_robots_per_episode", pa.simple(pa.stuck_count(None)))
    )
    return [
        ("planner_calls_per_100_robot_steps", pa.simple(calls_per_100)),
        ("common_flow_time", common_flow),
        third,
    ]


P4: list[tuple[str, PairFn]] = [
    ("conflicts_per_episode", pa.simple(conflicts)),
    ("top_priority_common_flow_time", pa.common_top_flow),
    ("top_priority_undelivered", pa.simple(pa.top_undelivered)),
]
SUPPORTIVE: list[tuple[str, PairFn]] = [
    ("undelivered_per_episode", pa.simple(lambda r: (float(r["released"] - r["delivered"]), 1.0))),
    ("steps_per_episode", pa.simple(lambda r: (float(r["steps"]), 1.0))),
    ("wait_ratio", pa.simple(lambda r: (float(r["wait_ratio"]), 1.0))),
    ("conflicts_per_100_robot_steps", pa.simple(lambda r: (100.0 * r["conflicts_total"], float(r["robots"] * r["steps"])))),
    ("share_stay", pa.simple(madql_share("stay"))),
    ("share_self_loop_stay", pa.simple(madql_share("self_loop_stay"))),
    ("share_in_stay_spells_ge20", pa.simple(madql_share("spell_ge20_steps"))),
    ("share_follow_suggestion", pa.simple(madql_share("follow"))),
    ("mean_gap", pa.simple(lambda r: (float(r["madql"]["gap_sum"]), float(r["madql"]["gap_n"])))),
    ("inference_ms_per_step", pa.simple(lambda r: (1000.0 * r["madql"]["inference_s"], float(r["madql"]["inference_calls"])))),
    ("planning_s_per_episode", pa.simple(lambda r: (float(r["planning_s"]), 1.0))),
    ("wall_s_per_episode", pa.simple(lambda r: (float(r["wall_s"]), 1.0))),
]  # fmt: skip


# --------------------------------------------------------------------------- data
class Data(pa.Data):
    def __init__(self, out: Path) -> None:
        self.recs = {}
        self.files = {}
        path = out / "records_4c.jsonl"
        self.files[str(path)] = pa.sha256(path) if path.exists() else None
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    k = p3.key(r)
                    if k in self.recs:
                        raise SystemExit(f"duplicate record {k}")
                    self.recs[k] = r

    def arm_recs(self, arm: str, cond: str) -> list[dict[str, Any]]:
        planner = p4.CONDS[cond][0]
        return [
            self.get(f"{arm}:{cond}", run, planner, lv, s)
            for run in p4.RUNS
            for lv in p3.LEVELS
            for s in self.seeds(planner)
        ]


def completeness(d: Data) -> dict[str, Any]:
    expected = {
        (f"{a}:{c}", run, p4.CONDS[c][0], lv, s)
        for c in p4.STAGES["4c"]["conds"]
        for a in p4.STAGES["4c"]["arms"]
        for run in p4.RUNS
        for lv in p3.LEVELS
        for s in p4.seeds("4c", c)
    }
    have = set(d.recs)
    return {
        "expected": len(expected),
        "present": len(have & expected),
        "missing": len(expected - have),
        "unexpected": len(have - expected),
        "errors": sum(r["status"] != "ok" for r in d.recs.values()),
        "complete": have == expected,
    }


def phase1_reproduction(d: Data) -> dict[str, Any]:
    """G0.1: the ``hist`` arms equal the stored Phase 1 MADQL records (non-timing fields)."""
    skip = set(p4.TIMING) | {"stage", "arm", "policy_arm", "candidate", "cond", "madql", "rrt",
                             "path_checks", "observer_s"}  # fmt: skip
    res: dict[str, Any] = {}
    for cond, (path, arm) in p4.PHASE1_RECORDS.items():
        planner = p4.CONDS[cond][0]
        n = same = 0
        diffs: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["arm"] != arm or not r["run"].startswith("madql") or r["planner"] != planner:
                    continue
                mine = d.recs.get((f"{OLD}:{cond}", r["run"], planner, r["level"], int(r["seed"])))
                if mine is None:
                    continue
                n += 1
                bad = sorted(k for k in (set(r) | set(mine)) - skip if r.get(k) != mine.get(k))
                same += not bad
                if bad and len(diffs) < 10:
                    diffs.append(
                        {"run": r["run"], "level": r["level"], "seed": r["seed"], "fields": bad}
                    )
        want = 3 * 3 * len(p4.seeds("4c", cond))
        res[cond] = {
            "compared": n,
            "identical": same,
            "differences": diffs,
            "pass": n == want and same == n,
        }
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    return res


def g0(d: Data, cond: str) -> dict[str, Any]:
    from robocoord.control.supervisor import frozen_params

    planner = p4.CONDS[cond][0]
    recs = d.arm_recs(OLD, cond) + d.arm_recs(NEW, cond)
    ok = [r for r in recs if r["status"] == "ok"]
    inst: dict[tuple[str, str, int], set[str]] = {}
    for r in ok:
        inst.setdefault((r["run"], r["level"], r["seed"]), set()).add(r["instance_sha256"])
    res: dict[str, Any] = {
        "episodes": len(recs),
        "errors": [p3.key(r) for r in recs if r["status"] != "ok"],
        "physical_collisions": sum(r["physical_collisions"] for r in ok),
        "inv_total": sum(r["conflicts"]["INV"] for r in ok),
        "invalid_reference_paths": sum(r["path_checks"]["invalid"] for r in ok),
        "reference_paths_checked": sum(r["path_checks"]["checked"] for r in ok),
        "instances_identical_across_arms": all(len(v) == 1 for v in inst.values()),
        "repair_off": all("repair_radius" not in r for r in ok),
        "p2_off": all("rrt_patience" not in r and "rrt_refine" not in r for r in ok),
        "full_rrt_budget": all(r["rrt_iterations"] == 1000 for r in ok),
        "v2_candidate_ok": all(
            r["candidate"] == p4.selected() for r in ok if r["policy_arm"] == NEW
        ),
    }
    if cond != "NA":
        fp = frozen_params(planner)
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
                "o_unused": all(r.get("supervisor") == "R+S" for r in sup),
            }
        )
        res["pass"] = bool(
            not res["errors"]
            and res["physical_collisions"] == 0
            and res["inv_total"] == 0
            and res["invalid_reference_paths"] == 0
            and res["supervisor_introduced_invalid_moves"] == 0
            and res["fairness_bound_ok"]
            and not res["burst_bound_violations"]
            and res["params_frozen"]
            and res["supervisor_mode_ok"]
            and res["supervised_episodes"] == len(ok)
            and res["instances_identical_across_arms"]
            and res["repair_off"]
            and res["p2_off"]
            and res["full_rrt_budget"]
            and res["v2_candidate_ok"]
        )
    else:
        res["no_supervisor"] = all("supervisor" not in r for r in ok)
        res["resolver_off"] = all(r["deadlock"] is False for r in ok)
        res["pass"] = bool(
            not res["errors"]
            and res["physical_collisions"] == 0
            and res["inv_total"] == 0
            and res["invalid_reference_paths"] == 0
            and res["instances_identical_across_arms"]
            and res["no_supervisor"]
            and res["resolver_off"]
            and res["repair_off"]
            and res["v2_candidate_ok"]
        )
    return res


# --------------------------------------------------------------------------- inference
def est(r: Mapping[str, Any]) -> float:
    return float(r["inf"]["estimate"]) if r["inf"] else 0.0


def delta1(d: Data, cond: str) -> float:
    rs = d.arm_recs(OLD, cond)
    return float(1.0 / (sum(r["released"] for r in rs) / len(rs)))


def ceiling_bound(arrs: list[Any]) -> float:
    """Protocol §5.4 ceiling rule: −(Σ over seeds of max(0, old − new delivered)) / Σ released
    (old). Ignores every gain, so it cannot favour the new arm."""
    loss = np.maximum(0.0, np.asarray(arrs[0]) - np.asarray(arrs[2]))
    return float(-loss.sum() / np.asarray(arrs[1]).sum())


def lower_bound(row: dict[str, Any], arrs: list[Any]) -> dict[str, Any]:
    inf = row["inf"]
    lo = inf["ci95"][0]
    if math.isfinite(lo):
        return {"value": lo, "source": "bootstrap-t"}
    pc = inf.get("ci95_percentile", [lo, None])[0]
    cb = ceiling_bound(arrs)
    return {"value": min(pc, cb), "source": "ceiling rule (min of percentile and loss-only bound)",
            "percentile": pc, "loss_only": cb}  # fmt: skip


def analyse(d: Data, cond: str) -> dict[str, Any]:
    planner = p4.CONDS[cond][0]
    idx = bca.boot_index(len(d.seeds(planner)))
    g = g0(d, cond)
    o, n = f"{OLD}:{cond}", f"{NEW}:{cond}"
    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA}
    comp = pa.simple(pa.completion)
    p1_arrs = d.pair(o, n, p4.RUNS, planner, comp)
    fam["P1"].append(pa._row("completion", FAMILY, p1_arrs, idx))
    for run in p4.RUNS:
        fam["P2"].append(pa._row("completion", run, d.pair(o, n, (run,), planner, comp), idx))
    for name, fn in e_family(cond):
        fam["E"].append(pa._row(name, FAMILY, d.pair(o, n, p4.RUNS, planner, fn), idx))
    for name, fn in P4:
        fam["P4"].append(pa._row(name, FAMILY, d.pair(o, n, p4.RUNS, planner, fn), idx))
    for f, rows in fam.items():
        pa.holm_family(rows, ALPHA[f])
    d1 = delta1(d, cond)
    lb = lower_bound(fam["P1"][0], p1_arrs) if fam["P1"][0]["inf"] else {"value": -math.inf}
    harm = [r["unit"] for r in fam["P1"] + fam["P2"] if r["rejected"] and est(r) < 0]
    gain = [r["unit"] for r in fam["P1"] if r["rejected"] and est(r) > 0]
    vetoes = [r["hypothesis"] for r in fam["P4"] if r["rejected"] and est(r) > 0]
    e_better = [r["hypothesis"] for r in fam["E"] if r["rejected"] and est(r) < 0]
    e_worse = [r["hypothesis"] for r in fam["E"] if r["rejected"] and est(r) > 0]
    if not g["pass"] or harm or vetoes:
        verdict = "reject"
    elif lb["value"] <= -d1:
        verdict = "inconclusive"
    elif e_worse and (e_better or gain):
        verdict = "inconclusive (mixed)"
    elif e_worse:
        verdict = "no benefit (efficiency harm)"
    elif gain or e_better:
        verdict = "adopt"
    else:
        verdict = "no benefit"
    decision = {
        "g0_pass": g["pass"],
        "completion_harm": harm,
        "completion_gain": gain,
        "p4_vetoes": vetoes,
        "e_improved": e_better,
        "e_worsened": e_worse,
        "completion_lower_bound": lb,
        "delta1": d1,
        "verdict": verdict,
    }
    supportive = {
        name: pa.infer(d.pair(o, n, p4.RUNS, planner, fn), idx) for name, fn in SUPPORTIVE
    }
    per_level = {}
    for lv in p3.LEVELS:
        arrs = [np.zeros(len(d.seeds(planner))) for _ in range(4)]
        for j, s in enumerate(d.seeds(planner)):
            for run in p4.RUNS:
                v = comp(d.get(o, run, planner, lv, s), d.get(n, run, planner, lv, s))
                for k in range(4):
                    arrs[k][j] += v[k]
        per_level[lv] = pa.infer(arrs, idx)
    return {
        "g0": g,
        "families": fam,
        "decision": decision,
        "supportive_nominal": supportive,
        "per_level_nominal": per_level,
        "describe": {
            a: {run: describe(d.arm_recs(a, cond), run) for run in (*p4.RUNS, "all")}
            for a in (OLD, NEW)
        },
        "seeds": seed_counts(d, cond),
    }


def describe(recs: list[dict[str, Any]], run: str) -> dict[str, Any]:
    rs = [r for r in recs if (run == "all" or r["run"] == run) and r["status"] == "ok"]
    rel = sum(r["released"] for r in rs)
    dl = sum(r["delivered"] for r in rs)
    rsteps = sum(r["robots"] * r["steps"] for r in rs)
    m = [r["madql"] for r in rs]
    ns = sum(x["nonidle_steps"] for x in m)
    trig: Counter[str] = Counter()
    for r in rs:
        trig.update(r["planning"]["by_trigger"])
    tot = sum(trig.values())
    stuck: Counter[str] = Counter(s["class"] for r in rs for s in r["stuck"])
    types: Counter[str] = Counter()
    for r in rs:
        types.update(r["conflicts"])
    acts = np.sum([x["actions"] for x in m], axis=0)
    hist_stay = np.sum([x["gap_hist_stay"] for x in m], axis=0)
    flow_w = sum(
        r["mean_flow_time"] * r["delivered"] for r in rs if r["mean_flow_time"] is not None
    )
    tops = [t for r in rs for t in r["tasks"] if t[1] == pa.TOP_PRIORITY and t[5] != "unreleased"]
    return {
        "episodes": len(rs),
        "completion": dl / rel if rel else None,
        "delivered": dl,
        "released": rel,
        "undelivered_per_episode": (rel - dl) / len(rs),
        "successful_episodes": sum(r["delivered"] == r["released"] for r in rs),
        "top_priority_completion": sum(t[5] == "delivered" for t in tops) / len(tops) if tops else None,
        "flow_time_delivered_mean": flow_w / dl if dl else None,
        "makespan_mean": float(np.mean([r["makespan"] for r in rs if r["makespan"] is not None]))
        if any(r["makespan"] is not None for r in rs)
        else None,
        "steps_per_episode": float(np.mean([r["steps"] for r in rs])),
        "conflicts_per_episode": sum(r["conflicts_total"] for r in rs) / len(rs),
        "conflicts_per_delivered": sum(r["conflicts_total"] for r in rs) / dl if dl else None,
        "conflicts_per_100_robot_steps": 100 * sum(r["conflicts_total"] for r in rs) / rsteps,
        "conflicts_by_type": dict(sorted(types.items())),
        "wait_ratio_mean": float(np.mean([r["wait_ratio"] for r in rs])),
        "planner_calls_per_100_robot_steps": 100 * sum(r["planner_calls"] for r in rs) / rsteps,
        "calls_by_trigger_share": {k: v / tot for k, v in sorted(trig.items())} if tot else {},
        "deviations_per_100_robot_steps": 100 * sum(r["planning"]["deviations"] for r in rs) / rsteps,
        "supervisor_changes_per_100_robot_steps": 100 * sum(r.get("sup_changed", 0) for r in rs) / rsteps,
        "stuck_total": sum(stuck.values()),
        "stuck_by_class": dict(sorted(stuck.items())),
        "share_stay": sum(x["stay"] for x in m) / ns,
        "share_stay_free_cell": sum(x["stay_free"] for x in m) / ns,
        "share_stay_occupied_cell": sum(x["stay_occupied"] for x in m) / ns,
        "share_self_loop_stay": sum(x["self_loop_stay"] for x in m) / ns,
        "share_in_stay_spells_ge20": sum(x["spell_ge20_steps"] for x in m) / ns,
        "share_follow_suggestion": sum(x["follow"] for x in m) / ns,
        "action_shares": (acts / acts.sum()).tolist(),
        "gap_hist_stay_share": (hist_stay / max(1, hist_stay.sum())).tolist(),
        "mean_gap": sum(x["gap_sum"] for x in m) / max(1, sum(x["gap_n"] for x in m)),
        "mean_q_chosen": sum(x["q_chosen_sum"] for x in m) / ns,
        "inference_ms_per_step": 1000 * sum(x["inference_s"] for x in m) / max(1, sum(x["inference_calls"] for x in m)),
        "planning_s_per_episode_median": statistics.median(r["planning_s"] for r in rs),
        "wall_s_per_episode_median": statistics.median(r["wall_s"] for r in rs),
        "physical_collisions": sum(r["physical_collisions"] for r in rs),
        "invalid_moves": sum(r["conflicts"]["INV"] for r in rs),
        "invalid_reference_paths": sum(r["path_checks"]["invalid"] for r in rs),
        "reference_paths_checked": sum(r["path_checks"]["checked"] for r in rs),
        "errors": 0,
    }  # fmt: skip


def seed_counts(d: Data, cond: str) -> dict[str, dict[str, int]]:
    planner = p4.CONDS[cond][0]
    out = {}
    for run in p4.RUNS:
        b = w = e = 0
        for s in d.seeds(planner):
            x = sum(d.get(f"{OLD}:{cond}", run, planner, lv, s)["delivered"] for lv in p3.LEVELS)
            y = sum(d.get(f"{NEW}:{cond}", run, planner, lv, s)["delivered"] for lv in p3.LEVELS)
            b += y > x
            w += y < x
            e += y == x
        out[run] = {"seeds_better": b, "seeds_worse": w, "seeds_equal": e}
    return out


def recommendation(
    res: Mapping[str, Any], g01: Mapping[str, Any], rep: Mapping[str, Any] | None
) -> dict[str, Any]:
    v = {c: res[c]["decision"]["verdict"] for c in p4.STAGES["4c"]["conds"]}
    adopt = (
        v["FA"] == "adopt"
        and not v["FR"].startswith("reject")
        and not v["NA"].startswith("reject")
        and bool(g01["pass"])
        and bool(rep and rep["identical"])
    )
    return {**v, "g0_1": bool(g01["pass"]), "replay_identical": bool(rep and rep["identical"]),
            "madql_v2_forward_candidate": adopt}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = Data(args.out)
    comp = completeness(d)
    if not comp["complete"]:
        raise SystemExit(f"incomplete: {comp}")
    rep_path = args.out / "replay_4c.json"
    rep = json.loads(rep_path.read_text("utf-8")) if rep_path.exists() else None
    g01 = phase1_reproduction(d)
    res: dict[str, Any] = {
        "files_sha256": d.files,
        "completeness": comp,
        "selection": json.loads(p4.SELECTION.read_text("utf-8")),
        "g0_1_phase1_reproduction": g01,
        "replay": rep,
    }
    for cond in p4.STAGES["4c"]["conds"]:
        res[cond] = analyse(d, cond)
        print(cond, json.dumps(res[cond]["decision"]))
    res["recommendation"] = recommendation(res, g01, rep)
    print("recommendation", json.dumps(res["recommendation"]))
    (args.out / "analysis.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
