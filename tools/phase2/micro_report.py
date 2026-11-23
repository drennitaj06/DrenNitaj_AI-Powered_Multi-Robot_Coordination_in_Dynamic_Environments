"""Phase 2 microbenchmark summary: ``micro_summary.json`` and ``micro_tables.md`` from the E1a,
E1c, timing and E2 records (descriptive; the selection itself is ``micro.py select``).

Usage::

    python tools/phase2/micro_report.py --out results/dev6a/phase2/micro
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import micro  # noqa: E402

E1C_STORED = Path("results/experiments/e1c_convergence/records.jsonl")


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def e1c_summary(out: Path, sel: str) -> dict[str, Any]:
    stored: dict[tuple[str, int, int], dict[str, Any]] = {}
    for line in E1C_STORED.open(encoding="utf-8"):
        r = json.loads(line)
        p = r["params"]
        stored[(p["family"], p["map"], p["pair"])] = r["metrics"]
    rows = []
    for r in micro._records(out / "e1c.jsonl"):
        m = stored[(r["family"], r["map"], r["pair"])]
        cp = {k: float(v) for k, v in m["checkpoints"].items()}
        full, cand = r["configs"]["full"], r["configs"][sel]
        rows.append(
            {
                "family": r["family"],
                "full_1000": full["planner_cost"],
                "sel_cost": cand["planner_cost"],
                "sel_iterations": cand["iterations"],
                "stored_1000": cp["1000"],
                "stored_20000": cp["20000"],
                "euclid_lb": m["euclid_lower_bound"],
                "reproduced_1000": full["best_at_1000"] == cp["1000"]
                or (full["best_at_1000"] is None and math.isinf(cp["1000"])),
                "full_steps": full["path_steps"],
                "sel_steps": cand["path_steps"],
            }
        )
    out_s: dict[str, Any] = {
        "queries": len(rows),
        "reproduced_1000": sum(x["reproduced_1000"] for x in rows),
    }
    for fam in ("open", "d20"):
        rs = [x for x in rows if x["family"] == fam and x["full_1000"] is not None]
        out_s[fam] = {
            "solved_full": len(rs),
            "solved_selected": sum(x["sel_cost"] is not None for x in rs),
            "mean_iterations_selected": statistics.mean(x["sel_iterations"] for x in rs)
            if rs
            else None,
            "cost_ratio_sel_vs_full1000": sum(x["sel_cost"] for x in rs)
            / sum(x["full_1000"] for x in rs),
            "cost_ratio_full1000_vs_20000": sum(x["full_1000"] for x in rs)
            / sum(x["stored_20000"] for x in rs),
            "cost_ratio_sel_vs_20000": sum(x["sel_cost"] for x in rs)
            / sum(x["stored_20000"] for x in rs),
            "cost_ratio_sel_vs_lower_bound": sum(x["sel_cost"] for x in rs)
            / sum(x["euclid_lb"] for x in rs),
            "steps_ratio_sel_vs_full": sum(x["sel_steps"] for x in rs)
            / sum(x["full_steps"] for x in rs),
        }
    return out_s


def timing_summary(out: Path, sel: str) -> dict[str, Any]:
    recs = micro._records(out / "e1a_timing.jsonl")
    res: dict[str, Any] = {"queries": len(recs)}
    for fam in [*sorted({r["family"] for r in recs}), "all"]:
        rs = [r for r in recs if fam == "all" or r["family"] == fam]
        tf = [1000 * r["time_s"]["full"] for r in rs]
        ts = [1000 * r["time_s"][sel] for r in rs]
        res[fam] = {
            "median_ms_full": statistics.median(tf),
            "median_ms_selected": statistics.median(ts),
            "total_s_full": sum(tf) / 1000,
            "total_s_selected": sum(ts) / 1000,
            "time_ratio_total": sum(ts) / sum(tf),
            "iterations_ratio_total": sum(r["iterations"][sel] for r in rs)
            / sum(r["iterations"]["full"] for r in rs),
        }
    return res


def e2_summary(out: Path, sel: str) -> dict[str, Any]:
    recs = micro._records(out / "e2.jsonl")
    by: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {"full": {}, sel: {}}
    for r in recs:
        by[r["config"]][(r["obstacles"], r["events"], r["team"], r["seed"])] = r
    keys = sorted(set(by["full"]) & set(by[sel]))
    res: dict[str, Any] = {"pairs": len(keys), "outcomes": {}}
    for cfg in ("full", sel):
        rs = [by[cfg][k] for k in keys]
        m = [r["metrics"] for r in rs]
        work = [x["planning"]["work"]["iterations"] for x in m]
        calls = [x["planning"]["calls"] for x in m]
        res[cfg] = {
            "outcomes": dict(
                sorted(
                    {
                        o: sum(r["outcome"] == o for r in rs) for o in {r["outcome"] for r in rs}
                    }.items()
                )
            ),
            "delivered": sum(x["tasks_delivered"] for x in m),
            "released": sum(x["tasks_released"] for x in m),
            "completion": sum(x["tasks_delivered"] for x in m)
            / sum(x["tasks_released"] for x in m),
            "terminated": sum(bool(x["terminated"]) for x in m),
            "steps_mean": statistics.mean(x["steps"] for x in m),
            "conflicts_total": sum(
                sum(x["conflicts"].values()) if isinstance(x["conflicts"], dict) else x["conflicts"]
                for x in m
            ),
            "planner_calls_total": sum(calls),
            "iterations_total": sum(work),
            "iterations_per_call": sum(work) / sum(calls),
            "latency_median_ms_median": 1000
            * statistics.median(
                x["latency_s"]["median"] for x in m if x["latency_s"]["median"] is not None
            ),
            "planning_total_s": sum(x["latency_s"]["total"] for x in m),
            "wall_total_s": sum(r["wall_s"] for r in rs),
            "mean_flow_time_mean": statistics.mean(
                x["mean_flow_time"] for x in m if x["mean_flow_time"] is not None
            ),
            "path_efficiency_mean": statistics.mean(
                x["path_efficiency"] for x in m if x["path_efficiency"] is not None
            ),
            "status_mix": {
                k: sum(x["planning"]["status_mix"][k] for x in m)
                for k in m[0]["planning"]["status_mix"]
            },
        }
    diffs = {"delivered_better": 0, "delivered_worse": 0, "steps_longer": 0, "steps_shorter": 0}
    for k in keys:
        a, b = by["full"][k]["metrics"], by[sel][k]["metrics"]
        diffs["delivered_better"] += b["tasks_delivered"] > a["tasks_delivered"]
        diffs["delivered_worse"] += b["tasks_delivered"] < a["tasks_delivered"]
        diffs["steps_longer"] += b["steps"] > a["steps"]
        diffs["steps_shorter"] += b["steps"] < a["steps"]
    res["paired"] = diffs
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    selj = json.loads((a.out / "selection.json").read_text("utf-8"))
    sel = selj["selected"]
    e1a = micro._records(a.out / "e1a.jsonl")
    summ: dict[str, Any] = {
        "selection": {k: selj[k] for k in ("selected", "selected_config", "eligible")},
        "e1a_prefix_failures": sum(
            not row["prefix_ok"] for r in e1a for lab, row in r["configs"].items() if lab != "full"
        ),
        "e1a_stratum": micro.summarise(e1a, micro.STRATUM_SIZES),
        "e1a_by_size": {str(s): micro.summarise(e1a, (s,)) for s in (16, 32, 64, 128)},
        "e1c": e1c_summary(a.out, sel),
        "timing": timing_summary(a.out, sel),
        "e2": e2_summary(a.out, sel),
    }
    (a.out / "micro_summary.json").write_text(json.dumps(summ, indent=1), "utf-8")
    lines = ["# Phase 2 microbenchmark tables (generated from micro_summary.json)", ""]
    lines += [f"Selected: **{sel}** {selj['selected_config']}; eligible: {selj['eligible']}.", ""]
    lines += [
        "## E1a selection stratum (sizes 16 and 32; 1 400 queries; budget 1 000)",
        "",
        "| Config | Success | Q steps (all) | max_f Q steps | Q planner cost | Paths identical to full | Max step increase | Mean iterations | Mean segment checks | Median first solution |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    st = summ["e1a_stratum"]
    fams = [x for x in st["full"] if x != "all"]
    for lab, per in st.items():
        allr = per["all"]
        lines.append(
            f"| {lab} | {f(allr['success'], 3)} | {f(allr['q_steps'])} | "
            f"{f(max(per[x]['q_steps'] for x in fams))} | {f(allr['q_planner_cost'])} | "
            f"{f(allr['share_steps_identical'], 3)} | {allr['max_step_increase']} | "
            f"{f(allr['mean_iterations'], 1)} | {f(allr['mean_segment_checks'], 0)} | "
            f"{f(allr['median_first_solution'], 0)} |"
        )
    lines += ["", "## E1a by family (stratum): Q steps / mean iterations", ""]
    lines += ["| Config | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for lab, per in st.items():
        lines.append(
            f"| {lab} | "
            + " | ".join(
                f"{f(per[x]['q_steps'], 3)} / {f(per[x]['mean_iterations'], 0)}" for x in fams
            )
            + " |"
        )
    lines += ["", f"## E1a by size: full vs {sel} (all families)", ""]
    lines += [
        "| Size | Success (full = selected) | Q steps | Mean iterations full → selected | Solved-query iterations selected |",
        "|---|---|---|---|---|",
    ]
    for s, per in summ["e1a_by_size"].items():
        a0, a1 = per["full"]["all"], per[sel]["all"]
        lines.append(
            f"| {s} | {f(a0['success'], 3)} = {f(a1['success'], 3)} | {f(a1['q_steps'])} | "
            f"{f(a0['mean_iterations'], 1)} → {f(a1['mean_iterations'], 1)} | {f(a1['mean_iterations_solved'], 1)} |"
        )
    tm = summ["timing"]
    lines += ["", "## Per-call time, untraced (stratum; min of 3 repetitions)", ""]
    lines += [
        "| Family | Median ms full | Median ms selected | Total time ratio | Iteration ratio |",
        "|---|---|---|---|---|",
    ]
    for fam, v in tm.items():
        if fam == "queries":
            continue
        lines.append(
            f"| {fam} | {f(v['median_ms_full'], 2)} | {f(v['median_ms_selected'], 2)} | "
            f"{f(v['time_ratio_total'], 3)} | {f(v['iterations_ratio_total'], 3)} |"
        )
    c = summ["e1c"]
    lines += [
        "",
        f"## E1c (descriptive; {c['queries']} queries; stored best cost at 1 000 reproduced: {c['reproduced_1000']}/{c['queries']})",
        "",
        "| Family | Solved full / selected | Mean iterations selected | Cost sel / full@1000 | full@1000 / stored@20000 | sel / stored@20000 | sel / lower bound | Steps sel / full |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fam in ("open", "d20"):
        v = c[fam]
        lines.append(
            f"| {fam} | {v['solved_full']} / {v['solved_selected']} | {f(v['mean_iterations_selected'], 1)} | "
            f"{f(v['cost_ratio_sel_vs_full1000'])} | {f(v['cost_ratio_full1000_vs_20000'])} | "
            f"{f(v['cost_ratio_sel_vs_20000'])} | {f(v['cost_ratio_sel_vs_lower_bound'])} | {f(v['steps_ratio_sel_vs_full'])} |"
        )
    e = summ["e2"]
    lines += [
        "",
        f"## E2 replanning (reported comparison after selection; {e['pairs']} paired episodes, seeds 20 000–20 019)",
        "",
        "| Metric | full | selected |",
        "|---|---|---|",
    ]
    for k in (
        "outcomes",
        "completion",
        "delivered",
        "terminated",
        "steps_mean",
        "conflicts_total",
        "planner_calls_total",
        "iterations_total",
        "iterations_per_call",
        "latency_median_ms_median",
        "planning_total_s",
        "wall_total_s",
        "mean_flow_time_mean",
        "path_efficiency_mean",
        "status_mix",
    ):
        lines.append(f"| {k} | {f(e['full'][k], 4)} | {f(e[sel][k], 4)} |")
    lines += ["", f"Paired episodes: {json.dumps(e['paired'])}", ""]
    (a.out / "micro_tables.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(len(lines), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
