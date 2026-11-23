"""Issue 6A Phase 3 — statistical analysis of Stages 3b, 3c and 3e.

Criteria: the Stage 3a freeze protocol §6 (``docs/research/2026-11-21-issue-6a-stage3a-calibration-
freeze.md``), *recommended* go / no-go rules (not formally approved; reported as such). Estimator
and bootstrap: imported unchanged from the frozen ``tools/selection/block_c_analysis.py``
(``paired_ratio``, ``mean_of_pairs``, ``boot_index`` B = 10 000 ``default_rng(0)``,
``bootstrap_t``, ``holm``). The statistical unit is the seed (its three levels summed); a family
estimate also sums the family's three policies per seed.

Usage::

    python tools/phase3/phase3_analysis.py --out results/dev6a/phase3/eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selection"))
import block_c_analysis as bca  # noqa: E402
import phase3 as p3  # noqa: E402

FAMILY_RUNS = {f: tuple(r for r in p3.RUNS if r.startswith(f)) for f in ("madql", "ippo", "mappo")}
CATEGORIES = ("frozen", "oscillating", "conflict_loop", "deadlock_cycle")
ALL_CLASSES = (
    "frozen",
    "oscillating",
    "conflict_loop",
    "deadlock_cycle",
    "blocked_by_idle",
    "mostly_stationary",
    "other_moving",
)
TOP_PRIORITY = 3
ALPHA = {"P1": 0.03, "P2": 0.005, "P3": 0.01, "P4": 0.005}
CELL = {c: a for a, (_, _, c) in p3.ARMS.items() if c is not None}
FACTORS = {"R": 0, "SO": 1, "D": 2}
TARGET = {
    "R": ("conflict_loop",),
    "SO": ("frozen", "oscillating"),
    "D": ("deadlock_cycle",),
    "O": ("oscillating",),
}
Arr = Any
PairFn = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[float, float, float, float]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- per-episode outcomes


def completion(r: Mapping[str, Any]) -> tuple[float, float]:
    return float(r["delivered"]), float(r["released"])


def per_episode(field: str) -> Callable[[Mapping[str, Any]], tuple[float, float]]:
    return lambda r: (float(r[field]), 1.0)


def stuck_count(cls: str | None) -> Callable[[Mapping[str, Any]], tuple[float, float]]:
    def f(r: Mapping[str, Any]) -> tuple[float, float]:
        return float(sum(cls is None or s["class"] == cls for s in r["stuck"])), 1.0

    return f


def top_undelivered(r: Mapping[str, Any]) -> tuple[float, float]:
    n = sum(
        1 for t in r["tasks"] if t[1] == TOP_PRIORITY and t[5] not in ("unreleased", "delivered")
    )
    return float(n), 1.0


def simple(f: Callable[[Mapping[str, Any]], tuple[float, float]]) -> PairFn:
    return lambda a, b: (*f(a), *f(b))


def common_top_flow(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    """Flow time (completion − release) of the highest-priority tasks delivered in both arms."""
    da = {t[0]: t[4] - t[2] for t in a["tasks"] if t[1] == TOP_PRIORITY and t[4] is not None}
    db = {t[0]: t[4] - t[2] for t in b["tasks"] if t[1] == TOP_PRIORITY and t[4] is not None}
    common = sorted(set(da) & set(db))
    return (
        float(sum(da[i] for i in common)),
        float(len(common)),
        float(sum(db[i] for i in common)),
        float(len(common)),
    )


# --------------------------------------------------------------------------- data


class Data:
    def __init__(self, out: Path) -> None:
        self.recs: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
        self.files = {}
        for s in p3.STAGES:
            path = out / f"records_{s}.jsonl"
            self.files[str(path)] = sha256(path) if path.exists() else None
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    k = p3.key(r)
                    if k in self.recs:
                        raise SystemExit(f"duplicate record {k}")
                    self.recs[k] = r

    def get(self, arm: str, run: str, planner: str, level: str, seed: int) -> dict[str, Any]:
        return self.recs[(arm, run, planner, level, seed)]

    def seeds(self, planner: str) -> range:
        return p3.BLOCK_C if planner == "astar" else p3.BLOCK_C_RRT

    def has(self, arm: str, planner: str) -> bool:
        return all(
            (arm, run, planner, lv, s) in self.recs
            for run in p3.RUNS
            for lv in p3.LEVELS
            for s in self.seeds(planner)
        )

    def pair(self, old: str, new: str, runs: Sequence[str], planner: str, fn: PairFn) -> list[Arr]:
        seeds = list(self.seeds(planner))
        out = np.zeros((4, len(seeds)))
        for j, s in enumerate(seeds):
            for run in runs:
                for lv in p3.LEVELS:
                    out[:, j] += fn(
                        self.get(old, run, planner, lv, s), self.get(new, run, planner, lv, s)
                    )
        return [out[0], out[1], out[2], out[3]]


# --------------------------------------------------------------------------- inference


def infer(arrs: Sequence[Arr], idx: Arr, groups: int = 1) -> dict[str, Any] | None:
    """Bootstrap-t of a paired difference (``groups`` = 1) or of the mean of ``groups`` paired
    differences (factorial main effect). ``None`` when not evaluable (an arm's X sums to 0)."""
    for g in range(groups):
        if arrs[4 * g + 1].sum() <= 0 or arrs[4 * g + 3].sum() <= 0:
            return None

    def stat(*flat: Arr) -> Any:
        if groups == 1:
            return bca.paired_ratio(*flat)
        return bca.mean_of_pairs([tuple(flat[4 * i : 4 * i + 4]) for i in range(groups)])

    off = sum(arrs[4 * g + 1].sum() for g in range(groups))
    theta_old = float(sum(arrs[4 * g].sum() for g in range(groups)) / off)
    est, se = stat(*arrs)
    if float(se) <= bca.EPS and abs(float(est)) <= bca.EPS:  # identical arms: nothing to test
        res: dict[str, Any] = {
            "estimate": 0.0,
            "se": 0.0,
            "ci95": [0.0, 0.0],
            "p": 1.0,
            "p_mc_se": 0.0,
        }
    else:
        res = bca.bootstrap_t(stat, list(arrs), idx).as_dict()
    res["theta_old"] = theta_old
    res["theta_new"] = theta_old + res["estimate"] if groups == 1 else None
    return res


def holm_family(rows: list[dict[str, Any]], alpha: float) -> None:
    """Holm within a family (not-evaluable rows: p = 1, kept); decision and MC sensitivity."""
    ps = [r["inf"]["p"] if r["inf"] is not None else 1.0 for r in rows]
    adj = bca.holm(ps)
    order = sorted(range(len(ps)), key=lambda i: (ps[i], i))
    m = len(ps)
    for rank, i in enumerate(order):
        r = rows[i]
        r["p_holm"] = adj[i]
        r["alpha_family"] = alpha
        thr = alpha / (m - rank)
        r["own_threshold"] = thr
        r["rejected"] = r["inf"] is not None and adj[i] <= alpha
        if r["inf"] is not None:
            p = r["inf"]["p"]
            r["mc_sensitive"] = abs(p - thr) <= 2 * math.sqrt(p * (1 - p) / bca.N_BOOT)
        else:
            r["mc_sensitive"] = False


# --------------------------------------------------------------------------- integrity (G0)


def g0(d: Data, planner: str, arms: Sequence[str]) -> dict[str, Any]:
    from robocoord.control.supervisor import frozen_params

    fp = frozen_params(planner)
    frozen = {k: getattr(fp, k) for k in fp.__slots__}
    recs = [
        d.get(a, run, planner, lv, s)
        for a in arms
        for run in p3.RUNS
        for lv in p3.LEVELS
        for s in d.seeds(planner)
    ]
    errors = [p3.key(r) for r in recs if r["status"] != "ok"]
    ok = [r for r in recs if r["status"] == "ok"]
    sup = [r for r in ok if r.get("supervisor_stats")]
    burst_viol = []
    for r in sup:
        w = min(fp.stall_window, fp.osc_window)
        if r["supervisor_stats"]["monitor"]["max_recoveries_in_a_leg"] > math.ceil(
            r["steps"] / (w + 1)
        ):
            burst_viol.append(p3.key(r))
    inv_excess = []
    for a in arms:
        if a == "off":
            continue
        for run in p3.RUNS:
            for lv in p3.LEVELS:
                for s in d.seeds(planner):
                    x, o = d.get(a, run, planner, lv, s), d.get("off", run, planner, lv, s)
                    if x["conflicts"]["INV"] > o["conflicts"]["INV"]:
                        inv_excess.append(p3.key(x))
    inst = defaultdict(set)
    for r in ok:
        inst[(r["run"], r["level"], r["seed"])].add(r["instance_sha256"])
    res = {
        "episodes": len(recs),
        "errors": errors,
        "physical_collisions": sum(r["physical_collisions"] for r in ok),
        "inv_total": sum(r["conflicts"]["INV"] for r in ok),
        "inv_on_exceeding_off": inv_excess,
        "supervisor_introduced_invalid_moves": sum(r.get("sup_invalid", 0) for r in ok),
        "max_consecutive_r_losses": max(
            (r["supervisor_stats"]["max_consecutive_r_losses"] for r in sup), default=0
        ),
        "fairness_bound_ok": all(
            r["supervisor_stats"]["max_consecutive_r_losses"] <= fp.fairness_losses for r in sup
        ),
        "burst_bound_violations": burst_viol,
        "params_frozen": all(r["supervisor_stats"]["params"] == frozen for r in sup),
        "escalation_off": all(not r["supervisor_stats"]["params"]["escalate"] for r in sup),
        "instances_identical_across_arms": all(len(v) == 1 for v in inst.values()),
        "checkpoint_sha256": sorted({(r["run"], r["checkpoint_sha256"]) for r in ok}),
    }
    res["pass"] = bool(
        not errors
        and res["physical_collisions"] == 0
        and res["supervisor_introduced_invalid_moves"] == 0
        and not inv_excess
        and res["fairness_bound_ok"]
        and not burst_viol
        and res["params_frozen"]
        and res["escalation_off"]
        and res["instances_identical_across_arms"]
    )
    return res


def block_c_reproduction(d: Data) -> dict[str, Any]:
    """G0.1: the OFF arm of the four changed PPO runs equals the stored Block C 'new' records."""
    skip = set(p3.TIMING) | {
        "label",
        "arm",
        "stage",
        "run",
        "checkpoint_sha256",
        "checkpoint_path",
        "stuck",
        "tasks",
        "sup_changed",
        "sup_invalid",
        "status",
        "supervisor",
        "deadlock",
        "max_steps",
    }
    n = same = 0
    diffs = []
    with p3.BLOCK_C_RECORDS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["arm"] != "new" or r["planner"] != "astar":
                continue
            mine = d.recs.get(("off", r["label"], "astar", r["level"], int(r["seed"])))
            if mine is None:
                continue
            n += 1
            bad = [k for k in r if k not in skip and r[k] != mine.get(k)]
            same += not bad
            if bad:
                diffs.append(
                    {"run": r["label"], "level": r["level"], "seed": r["seed"], "fields": bad}
                )
    return {"compared": n, "identical": same, "differences": diffs[:20]}


# --------------------------------------------------------------------------- descriptive


def describe_arm(d: Data, arm: str, planner: str, runs: Sequence[str]) -> dict[str, Any]:
    rs = [
        d.get(arm, run, planner, lv, s)
        for run in runs
        for lv in p3.LEVELS
        for s in d.seeds(planner)
    ]
    rel = sum(r["released"] for r in rs)
    out: dict[str, Any] = {
        "episodes": len(rs),
        "completion": sum(r["delivered"] for r in rs) / rel,
        "undelivered_per_episode": sum(r["undelivered"] for r in rs) / len(rs),
        "successful_episodes": sum(r["released"] == r["delivered"] and r["terminated"] for r in rs),
        "incomplete_episodes": sum(r["released"] > r["delivered"] for r in rs),
        "conflicts_per_episode": sum(r["conflicts_total"] for r in rs) / len(rs),
        "conflicts_by_type": {
            k: sum(r["conflicts"][k] for r in rs) for k in ("RRv", "RRs", "ROv", "ROs", "INV")
        },
        "wait_ratio_mean": float(np.mean([r["wait_ratio"] for r in rs])),
        "mean_flow_time_mean": float(
            np.mean([r["mean_flow_time"] for r in rs if r["mean_flow_time"] is not None])
        ),
        "makespan_mean": float(np.mean([r["makespan"] for r in rs if r["makespan"] is not None])),
        "planner_calls_per_100_robot_steps": 100
        * sum(r["planner_calls"] for r in rs)
        / sum(r["robots"] * r["steps"] for r in rs),
        "stuck_per_episode": {
            c: sum(s["class"] == c for r in rs for s in r["stuck"]) / len(rs) for c in ALL_CLASSES
        },
        "stuck_total": sum(len(r["stuck"]) for r in rs),
        "top_priority_undelivered": sum(top_undelivered(r)[0] for r in rs),
    }
    sup = [r["supervisor_stats"] for r in rs if r.get("supervisor_stats")]
    if sup:
        robot_steps = sum(s["robot_steps"] for s in sup)
        elig = sum(s["eligible_robot_steps"] for s in sup)
        mod = sum(s["modified_robot_steps"] for s in sup)
        rec: Counter[str] = Counter()
        end: Counter[str] = Counter()
        chg: Counter[str] = Counter()
        for s in sup:
            rec.update(s["recoveries_started"])
            end.update(s["recoveries_ended"])
            chg.update(s["changes"])
        mon = [s["monitor"] for s in sup]
        per_robot_rec = [x for m in mon for x in m["recoveries_by_robot"]]
        out["interventions"] = {
            "modified_share_of_robot_steps": mod / robot_steps,
            "modified_share_of_eligible_robot_steps": mod / elig if elig else None,
            "changes": dict(chg),
            "recoveries_started": dict(rec),
            "recoveries_ended": dict(end),
            "share_ended_by_progress": end["progress"] / sum(end.values())
            if sum(end.values())
            else None,
            "recovery_steps": sum(s["recovery_steps"] for s in sup),
            "max_recoveries_in_a_leg": _dist([m["max_recoveries_in_a_leg"] for m in mon]),
            "recoveries_per_robot": _dist(per_robot_rec),
            "max_robot_modified_share": _dist(
                [
                    m["max_robot_modified_share"]
                    for m in mon
                    if m["max_robot_modified_share"] is not None
                ]
            ),
            "max_steps_without_progress": _dist([m["max_steps_without_progress"] for m in mon]),
            "max_consecutive_r_losses": _dist([s["max_consecutive_r_losses"] for s in sup]),
            "r_left_to_arbiter": {
                k: sum(s["r_left_to_arbiter"][k] for s in sup) for k in ("occupied", "swap")
            },
        }
    return out


def _dist(v: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "q50": float(np.quantile(a, 0.5)),
        "q90": float(np.quantile(a, 0.9)),
        "q99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def runtime(d: Data, planner: str, arms: Sequence[str]) -> dict[str, Any]:
    """Per-step wall time without the observer; timing outliers (system pauses) flagged."""
    out: dict[str, Any] = {}
    for fam, runs in FAMILY_RUNS.items():
        for a in arms:
            rs = [
                d.get(a, run, planner, lv, s)
                for run in runs
                for lv in p3.LEVELS
                for s in d.seeds(planner)
            ]
            per = np.array([(r["wall_s"] - r.get("observer_s", 0.0)) / r["steps"] for r in rs])
            med = float(np.median(per))
            outl = per > 50 * med
            sup = np.array([r.get("supervisor_s", 0.0) / r["steps"] for r in rs])
            out[f"{fam}|{a}"] = {
                "ms_per_step_median": 1000 * med,
                "ms_per_step_iqr": [
                    1000 * float(np.quantile(per, 0.25)),
                    1000 * float(np.quantile(per, 0.75)),
                ],
                "timing_outliers_flagged": int(outl.sum()),
                "supervisor_ms_per_step_median": 1000 * float(np.median(sup)),
                "supervisor_share_of_step_time": float((sup[~outl]).sum() / per[~outl].sum()),
                "act_ms_per_step_median": 1000
                * float(np.median([r["act_s"] / r["steps"] for r in rs])),
            }
        if "off" in arms and "full" in arms:
            on, off = out[f"{fam}|full"], out[f"{fam}|off"]
            on["ratio_to_off_median"] = on["ms_per_step_median"] / off["ms_per_step_median"]
    return out


def per_level(d: Data, old: str, new: str, planner: str, idx: Arr) -> dict[str, Any]:
    """Exploratory per-level completion effects (nominal intervals, no tests)."""
    out = {}
    for fam, runs in FAMILY_RUNS.items():
        for lv in p3.LEVELS:
            seeds = list(d.seeds(planner))
            a = np.zeros((4, len(seeds)))
            for j, s in enumerate(seeds):
                for run in runs:
                    a[:, j] += (
                        *completion(d.get(old, run, planner, lv, s)),
                        *completion(d.get(new, run, planner, lv, s)),
                    )
            inf = infer(list(a), idx)
            out[f"{fam}|{lv}"] = (
                None
                if inf is None
                else {k: inf[k] for k in ("estimate", "ci95", "theta_old", "theta_new")}
            )
    return out


def seed_heterogeneity(d: Data, old: str, new: str, planner: str) -> dict[str, Any]:
    out = {}
    for run in p3.RUNS:
        better = worse = same = 0
        for s in d.seeds(planner):
            do = sum(d.get(old, run, planner, lv, s)["delivered"] for lv in p3.LEVELS)
            dn = sum(d.get(new, run, planner, lv, s)["delivered"] for lv in p3.LEVELS)
            better += dn > do
            worse += dn < do
            same += dn == do
        out[run] = {"seeds_better": better, "seeds_worse": worse, "seeds_equal": same}
    return out


# --------------------------------------------------------------------------- stage analyses


def _row(
    name: str, unit: str, arrs: list[Arr] | None, idx: Arr, groups: int = 1, **meta: Any
) -> dict[str, Any]:
    inf = None if arrs is None else infer(arrs, idx, groups)
    return {"hypothesis": name, "unit": unit, "inf": inf, **meta}


def on_off(
    d: Data, planner: str, old: str = "off", new: str = "full", categories: bool = True
) -> dict[str, Any]:
    idx = bca.boot_index(len(d.seeds(planner)))
    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA}
    for f, runs in FAMILY_RUNS.items():
        fam["P1"].append(
            _row("completion", f, d.pair(old, new, runs, planner, simple(completion)), idx)
        )
    for run in p3.RUNS:
        fam["P2"].append(
            _row("completion", run, d.pair(old, new, [run], planner, simple(completion)), idx)
        )
    if categories:
        for f, runs in FAMILY_RUNS.items():
            for c in CATEGORIES:
                arrs = d.pair(old, new, runs, planner, simple(stuck_count(c)))
                fam["P3"].append(_row(f"stuck_{c}", f, arrs if arrs[0].sum() > 0 else None, idx))
    else:
        fam.pop("P3")
    for f, runs in FAMILY_RUNS.items():
        fam["P4"].append(
            _row(
                "top_priority_common_flow_time",
                f,
                d.pair(old, new, runs, planner, common_top_flow),
                idx,
            )
        )
        arrs = d.pair(old, new, runs, planner, simple(top_undelivered))
        fam["P4"].append(
            _row(
                "top_priority_undelivered",
                f,
                arrs if arrs[0].sum() + arrs[2].sum() > 0 else None,
                idx,
            )
        )
    for k, rows in fam.items():
        holm_family(rows, ALPHA[k])
    supportive = {}
    for f, runs in FAMILY_RUNS.items():
        for name, fn in (
            ("undelivered_per_episode", per_episode("undelivered")),
            ("conflicts_per_episode", per_episode("conflicts_total")),
            ("stuck_total_per_episode", stuck_count(None)),
        ):
            supportive[f"{f}|{name}"] = infer(d.pair(old, new, runs, planner, simple(fn)), idx)
        supportive[f"{f}|conflicts_per_100_robot_steps"] = infer(
            d.pair(
                old,
                new,
                runs,
                planner,
                simple(lambda r: (100.0 * r["conflicts_total"], float(r["robots"] * r["steps"]))),
            ),
            idx,
        )
    return {
        "families": fam,
        "supportive_nominal": supportive,
        "per_level_nominal": per_level(d, old, new, planner, idx),
        "seed_heterogeneity": seed_heterogeneity(d, old, new, planner),
    }


def delta1(d: Data, planner: str) -> dict[str, float]:
    out = {}
    for f, runs in FAMILY_RUNS.items():
        rs = [
            d.get("off", run, planner, lv, s)
            for run in runs
            for lv in p3.LEVELS
            for s in d.seeds(planner)
        ]
        out[f] = 1.0 / (sum(r["released"] for r in rs) / len(rs))
    return out


def _find(rows: list[dict[str, Any]], unit: str, name: str = "completion") -> dict[str, Any]:
    return next(r for r in rows if r["unit"] == unit and r["hypothesis"] == name)


def decide_3b(res: dict[str, Any], g: dict[str, Any], d1: Mapping[str, float]) -> dict[str, Any]:
    f = res["families"]
    p1m = _find(f["P1"], "madql")
    c2 = p1m["rejected"] and p1m["inf"]["estimate"] > 0 and p1m["inf"]["estimate"] >= d1["madql"]
    harm = [
        r["unit"]
        for r in f["P1"] + f["P2"]
        if not r["unit"].startswith("madql") and r["rejected"] and r["inf"]["estimate"] < 0
    ]
    c4 = {u: _find(f["P1"], u)["inf"]["ci95"][0] > -d1[u] for u in ("ippo", "mappo")}
    veto = [
        f"{r['unit']}|{r['hypothesis']}"
        for r in f["P4"]
        if r["rejected"] and r["inf"]["estimate"] > 0
    ]
    conds = {
        "G0": g["pass"],
        "madql_improves_by_one_task": bool(c2),
        "no_ppo_harm": not harm,
        "no_harm_interpretable": all(c4.values()),
        "no_veto": not veto,
    }
    if all(conds.values()):
        verdict = "go"
    elif (not g["pass"]) or harm or (p1m["inf"]["estimate"] <= 0):
        verdict = "no-go"
    else:
        verdict = "inconclusive"
    return {
        "conditions": conds,
        "ppo_harm": harm,
        "no_harm_lower_bounds": c4,
        "vetoes": veto,
        "verdict": verdict,
        "status": "recommended criteria (Stage 3a §6), not formally approved",
    }


def decide_3e(res: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    f = res["families"]
    harm = [r["unit"] for r in f["P1"] + f["P2"] if r["rejected"] and r["inf"]["estimate"] < 0]
    veto = [
        f"{r['unit']}|{r['hypothesis']}"
        for r in f["P4"]
        if r["rejected"] and r["inf"]["estimate"] > 0
    ]
    madql_pos = _find(f["P1"], "madql")["inf"]["estimate"] > 0
    conds = {"G0": g["pass"], "no_harm": not harm, "madql_positive": madql_pos, "no_veto": not veto}
    return {
        "conditions": conds,
        "harm": harm,
        "vetoes": veto,
        "verdict": "go" if all(conds.values()) else "no-go",
        "status": "recommended criteria (Stage 3a §6), not formally approved",
    }


def factorial(d: Data) -> dict[str, Any]:
    """3c: main effects of R, S+O and D (mean of the 4 paired differences over the other
    factors' levels) and O's own effect (full minus S-without-O), with the §6.5 retention rule."""
    planner = "astar"
    idx = bca.boot_index(len(d.seeds(planner)))

    def pairs_of(factor: str) -> list[tuple[str, str]]:
        i = FACTORS[factor]
        out = []
        for c, arm in sorted(CELL.items()):
            if c[i] == 0:
                hi = list(c)
                hi[i] = 1
                out.append((arm, CELL[tuple(hi)]))  # type: ignore[index]
        return out

    def arrays(factor: str, runs: Sequence[str], fn: PairFn) -> tuple[list[Arr], int]:
        if factor == "O":
            return d.pair("R_S_D", "full", runs, planner, fn), 1
        flat: list[Arr] = []
        for old, new in pairs_of(factor):
            flat += d.pair(old, new, runs, planner, fn)
        return flat, 4

    fam: dict[str, list[dict[str, Any]]] = {k: [] for k in ALPHA}
    for factor in ("R", "SO", "D", "O"):
        for f, runs in FAMILY_RUNS.items():
            a, g = arrays(factor, runs, simple(completion))
            fam["P1"].append(_row("completion", f, a, idx, g, factor=factor))
        for run in p3.RUNS:
            a, g = arrays(factor, [run], simple(completion))
            fam["P2"].append(_row("completion", run, a, idx, g, factor=factor))
        for f, runs in FAMILY_RUNS.items():
            for c in TARGET[factor]:
                a, g = arrays(factor, runs, simple(stuck_count(c)))
                evaluable = sum(a[4 * k].sum() for k in range(g)) > 0
                fam["P3"].append(
                    _row(f"stuck_{c}", f, a if evaluable else None, idx, g, factor=factor)
                )
            a, g = arrays(factor, runs, common_top_flow)
            fam["P4"].append(_row("top_priority_common_flow_time", f, a, idx, g, factor=factor))
            a, g = arrays(factor, runs, simple(top_undelivered))
            fam["P4"].append(_row("top_priority_undelivered", f, a, idx, g, factor=factor))
    for k, rows in fam.items():
        holm_family(rows, ALPHA[k])
    total_stuck = {}
    for factor in ("R", "SO", "D", "O"):
        for f, runs in FAMILY_RUNS.items():
            a, g = arrays(factor, runs, simple(stuck_count(None)))
            total_stuck[f"{factor}|{f}"] = infer(a, idx, g)
    retention = {}
    for factor in ("R", "SO", "D", "O"):
        comp = [r for r in fam["P1"] + fam["P2"] if r["factor"] == factor]
        neg = [r["unit"] for r in comp if r["rejected"] and r["inf"]["estimate"] < 0]
        tgt = [r for r in fam["P3"] if r["factor"] == factor]
        target_shown = []
        for r in tgt:
            if r["rejected"] and r["inf"]["estimate"] < 0:
                fam_comp = next(
                    x for x in fam["P1"] if x["factor"] == factor and x["unit"] == r["unit"]
                )
                ts = total_stuck[f"{factor}|{r['unit']}"]
                if fam_comp["inf"]["estimate"] >= 0 and ts is not None and ts["estimate"] <= 0:
                    target_shown.append(f"{r['unit']}|{r['hypothesis']}")
        comp_pos = [
            r["unit"]
            for r in fam["P1"]
            if r["factor"] == factor and r["rejected"] and r["inf"]["estimate"] > 0
        ]
        keep = not neg and (bool(target_shown) or bool(comp_pos))
        retention[factor] = {
            "completion_harm": neg,
            "target_reduction_shown": target_shown,
            "completion_gain_families": comp_pos,
            "keep": keep,
        }
    cells = {}
    for arm in (*CELL.values(), "R_S_D"):
        cells[arm] = {
            f: describe_arm(d, arm, planner, runs)["completion"] for f, runs in FAMILY_RUNS.items()
        }
    return {
        "families": fam,
        "total_stuck_nominal": total_stuck,
        "retention": retention,
        "cell_completion": cells,
        "status": "recommended retention rule (Stage 3a §6.5), not formally approved",
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = Data(args.out)
    res: dict[str, Any] = {
        "inputs_sha256": d.files,
        "analysis_sha256": sha256(Path(__file__)),
        "block_c_analysis_sha256": sha256(HERE.parent / "selection" / "block_c_analysis.py"),
        "alpha": ALPHA,
        "n_boot": bca.N_BOOT,
        "boot_seed": bca.BOOT_SEED,
    }
    replay = args.out / "replay.json"
    res["replay"] = json.loads(replay.read_text("utf-8")) if replay.exists() else None
    if d.has("off", "astar") and d.has("full", "astar"):
        g = g0(d, "astar", ["off", "full"])
        r3b = on_off(d, "astar")
        d1 = delta1(d, "astar")
        res["3b"] = {
            "g0": g,
            "g0_1_block_c_reproduction": block_c_reproduction(d),
            "delta1": d1,
            "results": r3b,
            "decision": decide_3b(r3b, g, d1),
            "arms": {
                a: {f: describe_arm(d, a, "astar", runs) for f, runs in FAMILY_RUNS.items()}
                for a in ("off", "full")
            },
            "runtime": runtime(d, "astar", ["off", "full"]),
        }
    arms_3c = [a for a in p3.STAGES["3c"][1]]
    if all(d.has(a, "astar") for a in arms_3c) and "3b" in res:
        res["3c"] = {
            "g0": g0(d, "astar", ["off", *arms_3c, "full"]),
            "results": factorial(d),
            "arms": {
                a: {f: describe_arm(d, a, "astar", runs) for f, runs in FAMILY_RUNS.items()}
                for a in arms_3c
            },
            "runtime": runtime(d, "astar", arms_3c),
        }
    if d.has("off", "rrtstar") and d.has("full", "rrtstar"):
        g = g0(d, "rrtstar", ["off", "full"])
        r3e = on_off(d, "rrtstar", categories=False)
        res["3e"] = {
            "g0": g,
            "delta1": delta1(d, "rrtstar"),
            "results": r3e,
            "decision": decide_3e(r3e, g),
            "arms": {
                a: {f: describe_arm(d, a, "rrtstar", runs) for f, runs in FAMILY_RUNS.items()}
                for a in ("off", "full")
            },
            "runtime": runtime(d, "rrtstar", ["off", "full"]),
        }
    (args.out / "analysis.json").write_text(
        json.dumps(res, indent=1, sort_keys=True, default=str), "utf-8"
    )
    for s in ("3b", "3c", "3e"):
        if s in res:
            print(s, json.dumps(res[s].get("decision") or res[s]["results"].get("retention")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
