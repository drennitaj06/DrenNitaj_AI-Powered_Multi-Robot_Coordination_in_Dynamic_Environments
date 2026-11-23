"""Issue 6A Phase 6 — analysis of the confirmatory Block V run. Protocol (approved 2026-11-23):
``docs/research/2026-11-23-issue-6a-phase6-protocol.md`` (decisions D1–D13 of the decision review).

Three confirmatory comparison sets, each analysed per planner (H-L2, H-L12) or per condition
(H-v2), with the frozen estimator of ``block_c_analysis.py`` (ratio of sums, seed cluster,
bootstrap-t, B = 10 000, ``default_rng(0)``, Holm within each family, MC-sensitivity flag,
not-evaluable rows at p = 1) through the Phase 3 / Phase 4 helpers:

* **H-L2** (Phase 3 claim; proposal §12): R + S + D vs OFF, the same checkpoints (hist MADQL,
  SCB-C v5 IPPO and MAPPO). Decision: the approved Stage 3a rule (§6.3: 3b for A\\*, 3e for
  RRT\\*), with the robot–obstacle veto added to P4.
* **H-v2** (Phase 4 claim): MADQL-v2 n3 vs historical MADQL in FA (A\\*, R + S + D), FR (RRT\\*,
  R + S + D) and NA (A\\*, OFF). Decision: Phase 4 §5.4 (with the ceiling rule), the robot–obstacle
  veto added to P4.
* **H-L12** (proposal §12): the Issue 6A forward system (v5 PPO + R + S + D; MADQL-v2 n3 +
  R + S + D) vs the Issue #6 system (Issue #6 PPO picks, OFF; historical MADQL, OFF). Decision:
  the Stage 3a rule, as H-L2.

**Robot–obstacle veto** (the student's requirement, 2026-11-23): RO per 1 000 moves =
1 000 · Σ (ROv + ROs on the arbiter's input K^arb) / Σ moves proposed to the arbiter (robot-steps
whose ``a^arb`` is not STAY). A Holm-rejected increase in the P4 family is a veto.

Usage::

    python tools/phase6/phase6_analysis.py --out results/dev6a/phase6/eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sub in ("phase4", "phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))
import block_c_analysis as bca  # noqa: E402
import phase3 as p3  # noqa: E402
import phase3_analysis as pa  # noqa: E402
import phase4_analysis as p4a  # noqa: E402
import phase6 as p6  # noqa: E402

Arr = Any
PairFn = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[float, float, float, float]]
RunArm = Callable[[str], str]

#: α per family (union bound ≤ 0.05 per comparison unit; D5, D6), as Stage 3a §6.1 and Phase 4
#: §5.2. P3 is used in H-L2 (A\*) only; H-L12 reports its stuck categories nominally (D6).
ALPHA_L = {"P1": 0.03, "P2": 0.005, "P3": 0.01, "P4": 0.005}
ALPHA_V2 = dict(p4a.ALPHA)
FAMILIES = {"madql": p6.MADQL_RUNS, "ippo": p6.PPO_RUNS[:3], "mappo": p6.PPO_RUNS[3:]}
ALL_RUNS = (*p6.MADQL_RUNS, *p6.PPO_RUNS)


# --------------------------------------------------------------------------- outcomes


def ro_per_1000_moves(r: Mapping[str, Any]) -> tuple[float, float]:
    k = r["p6"]["k_arb"]
    return 1000.0 * (k["ROv"] + k["ROs"]), float(r["p6"]["moves_arb"])


def rr_ctrl_per_100(r: Mapping[str, Any]) -> tuple[float, float]:
    k = r["p6"]["k_ctrl"]
    return 100.0 * (k["RRv"] + k["RRs"]), float(r["robots"] * r["steps"])


def rr_arb_per_100(r: Mapping[str, Any]) -> tuple[float, float]:
    k = r["p6"]["k_arb"]
    return 100.0 * (k["RRv"] + k["RRs"]), float(r["robots"] * r["steps"])


def ro_nonidle_per_1000_moves(r: Mapping[str, Any]) -> tuple[float, float]:
    """Definition (ii) (secondary): RO events of non-idle robots / moves of non-idle robots."""
    return 1000.0 * r["p6"]["ro_arb_nonidle"], float(r["p6"]["moves_arb_nonidle"])


def ro_per_episode(r: Mapping[str, Any]) -> tuple[float, float]:
    k = r["p6"]["k_arb"]
    return float(k["ROv"] + k["ROs"]), 1.0


def conflicts_per_delivered(r: Mapping[str, Any]) -> tuple[float, float]:
    return float(r["conflicts_total"]), float(r["delivered"])


RO_VETO = ("ro_per_1000_moves", pa.simple(ro_per_1000_moves))
RO_OUTCOMES: dict[str, Callable[[Mapping[str, Any]], tuple[float, float]]] = {
    "ro_per_1000_moves": ro_per_1000_moves,
    "ro_nonidle_per_1000_moves": ro_nonidle_per_1000_moves,
    "ro_per_episode": ro_per_episode,
}
SECONDARY: list[tuple[str, PairFn]] = [
    ("rr_conflicts_ctrl_per_100_robot_steps", pa.simple(rr_ctrl_per_100)),
    ("rr_conflicts_arb_per_100_robot_steps", pa.simple(rr_arb_per_100)),
    ("conflicts_per_episode", pa.simple(p4a.conflicts)),
    ("flow_time_common", p4a.common_flow),
    ("undelivered_per_episode", pa.simple(lambda r: (float(r["released"] - r["delivered"]), 1.0))),
    ("planner_calls_per_100_robot_steps", pa.simple(p4a.calls_per_100)),
    ("wait_ratio", pa.simple(lambda r: (float(r["wait_ratio"]), 1.0))),
    ("ro_per_episode", pa.simple(ro_per_episode)),
    ("ro_nonidle_per_1000_moves", pa.simple(ro_nonidle_per_1000_moves)),
    ("conflicts_per_delivered", pa.simple(conflicts_per_delivered)),
]


# --------------------------------------------------------------------------- data

HORIZON = 400  # the scenario horizon (max_steps) of every Phase 6 instance
REQUIRED = ("released", "delivered", "steps", "terminated", "conflicts", "conflicts_total",
            "base_return", "physical_collisions", "robots", "wait_ratio", "planner_calls",
            "tasks", "stuck", "planning", "path_checks", "p6", "madql")  # fmt: skip


def invalid_reasons(r: Mapping[str, Any]) -> list[str]:
    """Block D §7 'invalid' classes for an ``ok`` record (empty list: valid)."""
    miss = [k for k in REQUIRED if r.get(k) is None]
    if miss:
        return [f"missing {miss}"]
    out = []
    counts = [r["released"], r["delivered"], r["steps"], r["conflicts_total"],
              r["physical_collisions"], *r["conflicts"].values()]  # fmt: skip
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in counts):
        out.append("counts not non-negative integers")
    if r["delivered"] > r["released"]:
        out.append("delivered > released")
    if r["steps"] > r.get("max_steps", HORIZON):
        out.append("steps > max_steps")
    if r["terminated"] and r["delivered"] != r["released"]:
        out.append("terminated with tasks undelivered")
    if not r["terminated"] and r["steps"] < r.get("max_steps", HORIZON):
        out.append("neither terminated nor at the horizon")
    if sum(r["conflicts"].values()) != r["conflicts_total"]:
        out.append("conflict breakdown does not sum to conflicts_total")
    if not math.isfinite(float(r["base_return"])):
        out.append("non-finite return")
    return out


def usable(r: Mapping[str, Any]) -> bool:
    """Protocol §5 / Block D §7: only an ``ok`` and valid record enters an estimate."""
    return r.get("status") == "ok" and not invalid_reasons(r)


class Data:
    def __init__(self, out: Path) -> None:
        self.recs: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
        path = out / "records.jsonl"
        self.files = {str(path): pa.sha256(path) if path.exists() else None}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        k = p6.key(r)
                        if k in self.recs:
                            raise SystemExit(f"duplicate record {k}")
                        self.recs[k] = r

    def get(self, arm: str, run: str, planner: str, level: str, seed: int) -> dict[str, Any]:
        return self.recs[(arm, run, planner, level, seed)]

    def seeds(self, planner: str) -> range:
        return p6.seeds(planner)

    def pair(self, old: RunArm, new: RunArm, runs: Sequence[str], planner: str,
             fn: PairFn) -> list[Arr] | None:  # fmt: skip
        ss = list(self.seeds(planner))
        out = np.zeros((4, len(ss)))
        for j, s in enumerate(ss):
            for run in runs:
                for lv in p6.LEVELS:
                    a = self.get(old(run), run, planner, lv, s)
                    b = self.get(new(run), run, planner, lv, s)
                    if not (usable(a) and usable(b)):
                        return None  # Block D §7: the unit is not evaluable
                    out[:, j] += fn(a, b)
        return [out[0], out[1], out[2], out[3]]

    def pair_groups(
        self,
        old: tuple[str, Sequence[str]],
        new: tuple[str, Sequence[str]],
        planner: str,
        f: Callable[[Mapping[str, Any]], tuple[float, float]],
    ) -> list[Arr] | None:
        """Seed-level sums for two arms with different run sets (paired by instance only)."""
        ss = list(self.seeds(planner))
        out = np.zeros((4, len(ss)))
        for j, s in enumerate(ss):
            for k, (arm, runs) in enumerate((old, new)):
                for run in runs:
                    for lv in p6.LEVELS:
                        r = self.get(arm, run, planner, lv, s)
                        if not usable(r):
                            return None
                        y, x = f(r)
                        out[2 * k, j] += y
                        out[2 * k + 1, j] += x
        return [out[0], out[1], out[2], out[3]]


def infer_or_none(arrs: list[Arr] | None, idx: Arr) -> dict[str, Any] | None:
    return None if arrs is None else pa.infer(arrs, idx)


def has_events(arrs: list[Arr] | None, both: bool = False) -> list[Arr] | None:
    """A category / count row is evaluable only when it has events (Stage 3a §6.4)."""
    if arrs is None:
        return None
    total = arrs[0].sum() + (arrs[2].sum() if both else 0)
    return arrs if total > 0 else None


def const(arm: str) -> RunArm:
    return lambda _run: arm


def historical_system(run: str) -> str:
    if run.startswith("madql"):
        return "hist:OFF"
    return "i6:OFF" if run in p6.CHANGED else "v5:OFF"


def forward_system(run: str) -> str:
    return "v2:RSD" if run.startswith("madql") else "v5:RSD"


def completeness(d: Data) -> dict[str, Any]:
    expected = {p6.key(j) for j in p6.jobs({"policies": {}})}
    have = set(d.recs)
    return {"expected": len(expected), "present": len(have & expected),
            "missing": len(expected - have), "unexpected": len(have - expected),
            "errors": sum(r["status"] != "ok" for r in d.recs.values()),
            "invalid": sum(r["status"] == "ok" and bool(invalid_reasons(r))
                           for r in d.recs.values()),
            "complete": have == expected}  # fmt: skip


# --------------------------------------------------------------------------- integrity (G0)


def _instances_match(recs: Sequence[Mapping[str, Any]], ic: Mapping[str, Any] | None) -> bool:
    """A1: every record's instance hash equals the fingerprint of the Block V instance check made
    after the execution authorization and before any episode."""
    if not ic or not ic.get("ok"):
        return False
    fp = ic["levels"]
    return all(
        r["instance_sha256"] == fp[r["level"]]["fingerprints"].get(str(r["seed"])) for r in recs
    )


def g0(d: Data, planner: str, ic: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from robocoord.control.supervisor import frozen_params

    recs = [r for k, r in d.recs.items() if k[2] == planner]
    ok = [r for r in recs if usable(r)]
    fp = frozen_params(planner)
    frozen = {k: getattr(fp, k) for k in fp.__slots__}
    inst: dict[tuple[str, int], set[str]] = {}
    for r in ok:
        inst.setdefault((r["level"], int(r["seed"])), set()).add(r["instance_sha256"])
    rsd = [r for r in ok if r["cond"] == "RSD"]
    res: dict[str, Any] = {
        "episodes": len(recs),
        "errors": [list(p6.key(r)) for r in recs if r["status"] != "ok"],
        "invalid_records": [{"key": list(p6.key(r)), "reasons": invalid_reasons(r)}
                            for r in recs if r["status"] == "ok" and invalid_reasons(r)],
        "physical_collisions": sum(r["physical_collisions"] for r in ok),
        "inv_total": sum(r["conflicts"]["INV"] for r in ok),
        "invalid_reference_paths": sum(r["path_checks"]["invalid"] for r in ok),
        "instances_identical_across_arms": all(len(v) == 1 for v in inst.values()),
        # A1: every record's instance equals the Block V instance check made before any episode
        "instances_match_pre_episode_check": _instances_match(ok, ic),
        "k_arb_equals_record_conflicts": all(
            r["p6"]["k_arb"] == {k: r["conflicts"][k] for k in r["p6"]["k_arb"]} for r in ok
        ),
        "k_ctrl_available_for_learned": all(
            r["p6"]["k_ctrl"] is not None for r in ok if r["policy"] != "base"
        ),
        "repair_off": all("repair_radius" not in r for r in ok),
        "p2_off": all(r.get("rrt_patience") is None and r.get("rrt_refine") is None for r in ok),
        "full_rrt_budget": all(r["rrt_iterations"] == 1000 for r in ok),
        "off_arms_plain": all(r.get("supervisor") in (None, "off") and r["deadlock"] is False
                              for r in ok if r["cond"] == "OFF"),
        "baseline_original": all(r.get("supervisor") in (None, "off") and r["deadlock"] is True
                                 for r in ok if r["cond"] == "ORIG"),
        "rsd_mode_ok": all(r.get("supervisor") == "R+S" and r["deadlock"] is True for r in rsd),
        "supervisor_introduced_invalid_moves": sum(r.get("sup_invalid", 0) for r in rsd),
        "fairness_bound_ok": all(r["supervisor_stats"]["max_consecutive_r_losses"]
                                 <= fp.fairness_losses for r in rsd),
        "burst_bound_violations": [
            list(p6.key(r)) for r in rsd
            if r["supervisor_stats"]["monitor"]["max_recoveries_in_a_leg"]
            > math.ceil(r["steps"] / (min(fp.stall_window, fp.osc_window) + 1))
        ],
        "params_frozen": all(r["supervisor_stats"]["params"] == frozen for r in rsd),
        "v2_label_ok": all(r["label"].startswith(f"madql-v2-{p6.V2_CANDIDATE}")
                           for r in ok if r["policy"] == "v2"),
    }  # fmt: skip
    res["pass"] = bool(
        not res["errors"]
        and not res["invalid_records"]
        and res["physical_collisions"] == 0
        and res["inv_total"] == 0
        and res["invalid_reference_paths"] == 0
        and res["supervisor_introduced_invalid_moves"] == 0
        and not res["burst_bound_violations"]
        and all(v for k, v in res.items() if isinstance(v, bool))
    )
    return res


# --------------------------------------------------------------------------- decisions


NOT_EVALUABLE_NOTE = (
    "not evaluable: no estimate, interval or p-value is computed or published (protocol §5, "
    "Block D §7); p = 1 enters Holm only as the protocol's ranking input"
)


def holm(rows: list[dict[str, Any]], alpha: float) -> None:
    """Holm within a family (the frozen procedure, ``phase3_analysis.holm_family``), then make a
    not-evaluable row explicit: ``evaluable`` False, ``holm_input_p`` 1 (the protocol's rule),
    and no ``p_holm``. A not-evaluable row is never rejected and never MC-sensitive."""
    pa.holm_family(rows, alpha)
    for r in rows:
        r["evaluable"] = r["inf"] is not None
        if not r["evaluable"]:
            r.update(holm_input_p=1.0, p_holm=None, rejected=False, mc_sensitive=False,
                     note=NOT_EVALUABLE_NOTE)  # fmt: skip


def lower_bound(row: dict[str, Any], arrs: list[Arr] | None) -> dict[str, Any]:
    """Phase 4 §5.4 ceiling rule: a finite bootstrap-t bound as is; an unbounded one is replaced
    by the smaller of the percentile bound and the loss-only bound."""
    if row["inf"] is None or arrs is None:
        return {"value": -math.inf, "source": "not evaluable"}
    return dict(p4a.lower_bound(row, arrs))


def est(r: Mapping[str, Any]) -> float:
    """The estimate, or NaN for a not-evaluable row (every comparison with NaN is false)."""
    return float(r["inf"]["estimate"]) if r["inf"] is not None else math.nan


def _harm(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{r['unit']}|{r['hypothesis']}" for r in rows
            if r["rejected"] and r["inf"]["estimate"] < 0]  # fmt: skip


def _veto(rows: list[dict[str, Any]]) -> list[str]:
    return [f"{r['unit']}|{r['hypothesis']}" for r in rows
            if r["rejected"] and r["inf"]["estimate"] > 0]  # fmt: skip


def system_change(d: Data, planner: str, old: RunArm, new: RunArm, *, oldarm_for_delta: RunArm,
                  categories: bool) -> dict[str, Any]:  # fmt: skip
    """Families P1–P4 for a system change (H-L2 or H-L12), Stage 3a §6.1, with RO in P4."""
    idx = bca.boot_index(len(d.seeds(planner)))
    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA_L}
    comp = pa.simple(pa.completion)
    arrs_p1: dict[str, list[Arr] | None] = {}
    for f, runs in FAMILIES.items():
        arrs_p1[f] = d.pair(old, new, runs, planner, comp)
        fam["P1"].append(pa._row("completion", f, arrs_p1[f], idx))
    for run in ALL_RUNS:
        fam["P2"].append(pa._row("completion", run, d.pair(old, new, (run,), planner, comp), idx))
    if categories:
        for f, runs in FAMILIES.items():
            for c in pa.CATEGORIES:
                a = d.pair(old, new, runs, planner, pa.simple(pa.stuck_count(c)))
                fam["P3"].append(pa._row(f"stuck_{c}", f, has_events(a), idx))
    else:
        fam.pop("P3")  # D6: no P3 family in H-L12; the categories are reported nominally
    for f, runs in FAMILIES.items():
        fam["P4"].append(pa._row("top_priority_common_flow_time", f,
                                 d.pair(old, new, runs, planner, pa.common_top_flow), idx))  # fmt: skip
        a = d.pair(old, new, runs, planner, pa.simple(pa.top_undelivered))
        fam["P4"].append(pa._row("top_priority_undelivered", f, has_events(a, both=True), idx))
        fam["P4"].append(pa._row(RO_VETO[0], f, d.pair(old, new, runs, planner, RO_VETO[1]), idx))
    for k, rows in fam.items():
        holm(rows, ALPHA_L[k])
    d1 = {}
    for f, runs in FAMILIES.items():
        rs = [r for run in runs for lv in p6.LEVELS for s in d.seeds(planner)
              if usable(r := d.get(oldarm_for_delta(run), run, planner, lv, s))]  # fmt: skip
        d1[f] = 1.0 / (sum(r["released"] for r in rs) / len(rs))
    lbs = {f: lower_bound(next(r for r in fam["P1"] if r["unit"] == f), arrs_p1[f])
           for f in FAMILIES}  # fmt: skip
    secondary = {f"{f}|{n}": infer_or_none(d.pair(old, new, runs, planner, fn), idx)
                 for f, runs in FAMILIES.items() for n, fn in SECONDARY}  # fmt: skip
    if not categories:
        for f, runs in FAMILIES.items():
            for cat in (*pa.CATEGORIES, None):
                a = d.pair(old, new, runs, planner, pa.simple(pa.stuck_count(cat)))
                secondary[f"{f}|stuck_{cat or 'total'}"] = infer_or_none(has_events(a), idx)
    return {"families": fam, "delta1": d1, "completion_lower_bounds": lbs,
            "secondary_nominal": secondary}  # fmt: skip


def decide_system(res: dict[str, Any], g: dict[str, Any], planner: str) -> dict[str, Any]:
    """Stage 3a §6.3: 3b for A\\*, 3e for RRT\\* (approved for Phase 3), RO veto in P4."""
    f = res["families"]
    p1 = {r["unit"]: r for r in f["P1"]}
    vetoes = _veto(f["P4"])
    if planner == "astar":
        m = p1["madql"]
        improves = bool(m["rejected"] and est(m) >= res["delta1"]["madql"] > 0)
        harm = [h for h in _harm(f["P1"] + f["P2"]) if not h.startswith("madql")]
        interpretable = {u: res["completion_lower_bounds"][u]["value"] > -res["delta1"][u]
                         for u in ("ippo", "mappo")}  # fmt: skip
        conds = {"G0": g["pass"], "madql_improves_by_one_task": improves, "no_ppo_harm": not harm,
                 "no_harm_interpretable": all(interpretable.values()), "no_veto": not vetoes}  # fmt: skip
        if all(conds.values()):
            verdict = "go"
        elif not g["pass"] or harm or est(m) <= 0:
            verdict = "no-go"
        else:
            verdict = "inconclusive"
        return {"conditions": conds, "harm": harm, "vetoes": vetoes, "verdict": verdict,
                "no_harm_lower_bounds": res["completion_lower_bounds"]}  # fmt: skip
    harm = _harm(f["P1"] + f["P2"])
    conds = {"G0": g["pass"], "no_harm": not harm,
             "madql_positive": est(p1["madql"]) > 0, "no_veto": not vetoes}  # fmt: skip
    return {"conditions": conds, "harm": harm, "vetoes": vetoes,
            "verdict": "go" if all(conds.values()) else "no-go"}  # fmt: skip


V2_CONDS = {"FA": ("astar", "RSD"), "FR": ("rrtstar", "RSD"), "NA": ("astar", "OFF")}


def v2_condition(d: Data, cond: str, g: dict[str, Any]) -> dict[str, Any]:
    """Phase 4 §5 families and §5.4 decision for one condition, RO veto added to P4."""
    planner, c = V2_CONDS[cond]
    idx = bca.boot_index(len(d.seeds(planner)))
    o, n = const(f"hist:{c}"), const(f"v2:{c}")
    runs = p6.MADQL_RUNS
    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA_V2}
    comp = pa.simple(pa.completion)
    p1_arrs = d.pair(o, n, runs, planner, comp)
    fam["P1"].append(pa._row("completion", "madql", p1_arrs, idx))
    for run in runs:
        fam["P2"].append(pa._row("completion", run, d.pair(o, n, (run,), planner, comp), idx))
    for name, fn in p4a.e_family("NA" if c == "OFF" else "FA"):
        fam["E"].append(pa._row(name, "madql", d.pair(o, n, runs, planner, fn), idx))
    for name, fn in [*p4a.P4, RO_VETO]:
        fam["P4"].append(pa._row(name, "madql", d.pair(o, n, runs, planner, fn), idx))
    for k, rows in fam.items():
        holm(rows, ALPHA_V2[k])
    rs = [r for run in runs for lv in p6.LEVELS for s in d.seeds(planner)
          if usable(r := d.get(f"hist:{c}", run, planner, lv, s))]  # fmt: skip
    d1 = 1.0 / (sum(r["released"] for r in rs) / len(rs))
    lb = lower_bound(fam["P1"][0], p1_arrs)
    harm = [r["unit"] for r in fam["P1"] + fam["P2"] if r["rejected"] and r["inf"]["estimate"] < 0]
    gain = [r["unit"] for r in fam["P1"] if r["rejected"] and r["inf"]["estimate"] > 0]
    vetoes = [r["hypothesis"] for r in fam["P4"] if r["rejected"] and r["inf"]["estimate"] > 0]
    better = [r["hypothesis"] for r in fam["E"] if r["rejected"] and r["inf"]["estimate"] < 0]
    worse = [r["hypothesis"] for r in fam["E"] if r["rejected"] and r["inf"]["estimate"] > 0]
    if not g["pass"] or harm or vetoes:
        verdict = "reject"
    elif lb["value"] <= -d1:
        verdict = "inconclusive"
    elif worse and (better or gain):
        verdict = "inconclusive (mixed)"
    elif worse:
        verdict = "no benefit (efficiency harm)"
    elif gain or better:
        verdict = "adopt"
    else:
        verdict = "no benefit"
    secondary = {nm: infer_or_none(d.pair(o, n, runs, planner, fn), idx) for nm, fn in SECONDARY}
    per_run = {run: infer_or_none(d.pair(o, n, (run,), planner, RO_VETO[1]), idx) for run in runs}
    return {"families": fam, "secondary_nominal": secondary, "ro_per_run_nominal": per_run,
            "decision": {"g0_pass": g["pass"], "completion_harm": harm, "completion_gain": gain,
                         "vetoes": vetoes, "e_improved": better, "e_worsened": worse,
                         "completion_lower_bound": lb, "delta1": d1, "verdict": verdict}}  # fmt: skip


def context_v2_vs_ppo(d: Data, planner: str) -> dict[str, Any]:
    """D4 context (pre-registered, nominal, never gating): MADQL-v2 (R + S + D) vs the forward
    PPO policies (v5, R + S + D) on the same instances; old = PPO, new = v2."""
    idx = bca.boot_index(len(d.seeds(planner)))
    return {
        name: infer_or_none(
            d.pair_groups(("v5:RSD", p6.PPO_RUNS), ("v2:RSD", p6.MADQL_RUNS), planner, f), idx
        )
        for name, f in RO_OUTCOMES.items()
    }


def describe_arms(d: Data) -> dict[str, Any]:
    """Descriptive summaries per planner and arm (and the RO rate per run)."""
    out: dict[str, Any] = {}
    for planner in p6.PLANNERS:
        out[planner] = {}
        for pol, cond in p6.MATRIX[planner]:
            arm = f"{pol}:{cond}"
            cell = [r for k, r in d.recs.items() if k[0] == arm and k[2] == planner]
            rs = [r for r in cell if usable(r)]
            if not rs:  # Block D §7.5: counts only, "data incomplete"
                out[planner][arm] = {"episodes": len(cell), "unusable_episodes": len(cell),
                                     "data_incomplete": True}  # fmt: skip
                continue
            desc = p4a.describe(rs, "all")
            desc["unusable_episodes"] = len(cell) - len(rs)
            q = [r["p6"] for r in rs]
            mv = sum(x["moves_arb"] for x in q)
            mvn = sum(x["moves_arb_nonidle"] for x in q)
            ro = sum(x["k_arb"]["ROv"] + x["k_arb"]["ROs"] for x in q)
            steps = sum(x["robot_steps"] for x in q)
            ctrl = [x["k_ctrl"] for x in q if x["k_ctrl"] is not None]
            by_run = {}
            for run in p6.POLICY_RUNS[pol]:
                rr = [r["p6"] for r in rs if r["run"] == run]
                m = sum(x["moves_arb"] for x in rr)
                by_run[run] = (
                    1000 * sum(x["k_arb"]["ROv"] + x["k_arb"]["ROs"] for x in rr) / m if m else None
                )
            desc.update(
                {
                    "ro_per_1000_moves": 1000 * ro / mv if mv else None,
                    "ro_nonidle_per_1000_moves": (
                        1000 * sum(x["ro_arb_nonidle"] for x in q) / mvn if mvn else None
                    ),
                    "moves_per_episode": mv / len(rs),
                    "idle_move_share": 1 - mvn / mv if mv else None,
                    "rr_arb_per_100_robot_steps": 100
                    * sum(x["k_arb"]["RRv"] + x["k_arb"]["RRs"] for x in q)
                    / steps,
                    "rr_ctrl_per_100_robot_steps": (
                        100 * sum(c["RRv"] + c["RRs"] for c in ctrl) / steps if ctrl else None
                    ),
                    "ro_per_1000_moves_by_run": by_run,
                }
            )
            out[planner][arm] = desc
    return out


UNITS = ("H-L2/astar", "H-L2/rrtstar", "H-L12/astar", "H-L12/rrtstar", "H-v2/FA", "H-v2/FR",
         "H-v2/NA")  # fmt: skip


def analyse(out: Path) -> dict[str, Any]:
    d = Data(out)
    comp = completeness(d)
    if not comp["complete"]:
        raise SystemExit(f"incomplete: {comp}")
    rep_path = out / "replay.json"
    rep = json.loads(rep_path.read_text("utf-8")) if rep_path.exists() else None
    res: dict[str, Any] = {"files_sha256": d.files, "completeness": comp, "replay": rep}
    ic_path = out / "instance_check.json"
    ic = json.loads(ic_path.read_text("utf-8")) if ic_path.exists() else None
    res["instance_check_sha256"] = pa.sha256(ic_path) if ic_path.exists() else None
    g = {p: g0(d, p, ic) for p in p6.PLANNERS}
    res["g0"] = g
    res["H-L2"], res["H-L12"] = {}, {}
    for p in p6.PLANNERS:
        cat = p == "astar"  # P3 in H-L2 on A* only (Stage 3a: 3e uses P1, P2 and P4)
        l2 = system_change(d, p, lambda r: "hist:OFF" if r.startswith("madql") else "v5:OFF",
                           lambda r: "hist:RSD" if r.startswith("madql") else "v5:RSD",
                           oldarm_for_delta=lambda r: "hist:OFF" if r.startswith("madql")
                           else "v5:OFF", categories=cat)  # fmt: skip
        l2["decision"] = decide_system(l2, g[p], p)
        res["H-L2"][p] = l2
        l12 = system_change(d, p, historical_system, forward_system,
                            oldarm_for_delta=historical_system, categories=False)  # fmt: skip
        l12["decision"] = decide_system(l12, g[p], p)
        res["H-L12"][p] = l12
    res["H-v2"] = {c: v2_condition(d, c, g[V2_CONDS[c][0]]) for c in V2_CONDS}
    res["context_v2_vs_forward_ppo"] = {p: context_v2_vs_ppo(d, p) for p in p6.PLANNERS}
    res["arms"] = describe_arms(d)
    replay_ok = bool(rep and rep["identical"])
    g_ok = all(g[p]["pass"] for p in p6.PLANNERS)
    v = {c: res["H-v2"][c]["decision"]["verdict"] for c in V2_CONDS}
    l2 = {p: res["H-L2"][p]["decision"]["verdict"] for p in p6.PLANNERS}
    l12 = {p: res["H-L12"][p]["decision"]["verdict"] for p in p6.PLANNERS}
    res["summary"] = {
        "g0_pass": {p: g[p]["pass"] for p in p6.PLANNERS},
        "replay_identical": replay_ok,
        "H-L2": l2,
        "H-L12": l12,
        "H-v2": v,
        # protocol §6.8 confirmation statements (conjunctions; G0 and the replay are part of each)
        "supervisor_confirmed": all(x == "go" for x in l2.values()) and replay_ok and g_ok,
        "madql_v2_confirmed": v["FA"] == "adopt"
        and not v["FR"].startswith("reject")
        and not v["NA"].startswith("reject")
        and replay_ok
        and g_ok,
        "integrated_system_confirmed": all(x == "go" for x in l12.values()) and replay_ok and g_ok,
        "error_control": {
            "per_unit_fwer": 0.05,
            "units": list(UNITS),
            "global_bound_any_false_rejection": 1 - 0.95 ** len(UNITS),
        },
    }
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/dev6a/phase6/eval"))
    args = ap.parse_args(argv)
    import os

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run the analysis with PYTHONHASHSEED=0 (byte-identical re-runs, §6.9)")
    p6.verify_freeze(args.out)
    res = analyse(args.out)
    res["freeze_sha256"] = pa.sha256(args.out / "plan_freeze.json")
    res["environment"] = p3._env()
    (args.out / "analysis.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
