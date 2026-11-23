"""Render the Phase 2 ``analysis.json`` as Markdown tables (no new computation).

Usage::

    python tools/phase2/phase2_report.py --out results/dev6a/phase2/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3"))
from phase3_report import f, family_table  # noqa: E402

NAMES = {
    "L": "L: 9 learned policies, RRT\\*, supervisor R + S + D",
    "B": "B: baseline controller, RRT\\* (resolver on, no supervisor)",
}
DESC_SYSTEM = (
    ("completion", "Completion", 4),
    ("delivered", "Delivered", 0),
    ("undelivered_per_episode", "Undelivered / ep.", 3),
    ("successful_episodes", "Successful ep.", 0),
    ("conflicts_per_episode", "Conflicts / ep.", 2),
    ("conflicts_per_delivered", "Conflicts / delivered", 3),
    ("conflicts_per_100_robot_steps", "Conflicts / 100 robot-steps", 3),
    ("flow_time_delivered_mean", "Flow time (delivered)", 2),
    ("top_priority_flow_time_mean", "High-priority flow time", 2),
    ("wait_ratio_mean", "Wait ratio", 4),
    ("steps_per_episode", "Steps / ep.", 1),
    ("makespan_mean", "Makespan (mean)", 1),
    ("stuck_total", "Stuck robots", 0),
    ("supervisor_changes_per_episode", "Supervisor changes / ep.", 2),
    ("planner_calls_per_100_robot_steps", "Calls / 100 robot-steps", 2),
    ("deviations_per_100_robot_steps", "Deviations / 100 robot-steps", 2),
)
DESC_RRT = (
    ("rrt_calls_per_episode", "RRT\\* calls / ep.", 1),
    ("rrt_iterations_per_call", "Iterations / call", 1),
    ("share_ok_calls_stopped_early", "Share of solved calls stopped early", 3),
    ("budget_exhausted_calls", "Budget-exhausted calls", 0),
    ("mean_first_solution_iteration", "Mean first-solution iteration", 1),
    ("moves_per_successful_plan", "Moves / successful plan", 2),
    ("rrt_ms_per_call_mean", "ms / call (mean)", 1),
    ("rrt_ms_per_call_median_of_episodes", "ms / call (median of ep. medians)", 1),
    ("rrt_s_per_episode_median", "RRT\\* s / ep. (median)", 2),
    ("wall_s_per_episode_median", "Wall s / ep. (median)", 2),
    ("planning_share_of_step_time", "Planning share of step time", 3),
    ("tree_nodes_per_call", "Tree nodes / call", 1),
    ("tree_nodes_max", "Max tree nodes", 0),
    ("segment_checks_per_call", "Segment checks / call", 0),
    ("rrt_s_total", "RRT\\* s total", 0),
)
DESC_DICTS = (
    ("conflicts_by_type", "Conflicts by type (total)", 0),
    ("calls_by_trigger_share", "Planner calls by trigger (share)", 4),
    ("status_share", "RRT\\* call status (share)", 4),
    ("stuck_per_episode", "Stuck robots by category (per ep.)", 4),
)
DESC_SAFETY = (
    ("physical_collisions", "Physical collisions", 0),
    ("invalid_moves", "Invalid moves (INV)", 0),
    ("invalid_reference_paths", "Invalid reference paths", 0),
    ("reference_paths_checked", "Reference paths checked", 0),
    ("errors", "Errors", 0),
)


def desc_table(desc: dict[str, dict[str, Any]], rows: Sequence[tuple[str, str, int]]) -> list[str]:
    cols = [(a, fam) for a, fams in desc.items() for fam in fams]
    out = [
        "| Metric | " + " | ".join(f"{fam} {a}" for a, fam in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for key, name, nd in rows:
        out.append(
            f"| {name} | " + " | ".join(f(desc[a][fam].get(key), nd) for a, fam in cols) + " |"
        )
    return out


def dict_table(desc: dict[str, dict[str, Any]], key: str, nd: int) -> list[str]:
    cols = [(a, fam) for a, fams in desc.items() for fam in fams]
    subs = sorted({k for a, fam in cols for k in (desc[a][fam].get(key) or {})})
    out = [
        "| Category | " + " | ".join(f"{fam} {a}" for a, fam in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for sub in subs:
        out.append(
            f"| {sub} | "
            + " | ".join(f((desc[a][fam].get(key) or {}).get(sub), nd) for a, fam in cols)
            + " |"
        )
    return out


def sensitivity(b: dict[str, Any]) -> list[str]:
    """Not pre-registered: the frozen tool's stored percentile interval next to the decision bound."""
    out = [
        "| Family | Δ̂ | bootstrap-t 95 % CI | percentile 95 % CI | −δ₁ | percentile lower > −δ₁ |",
        "|---|---|---|---|---|---|",
    ]
    for r in b["families"]["P1"]:
        inf, u = r["inf"], r["unit"]
        pc = inf.get("ci95_percentile", inf["ci95"])
        out.append(
            f"| {u} | {f(inf['estimate'])} | [{f(inf['ci95'][0])}, {f(inf['ci95'][1])}] | "
            f"[{f(pc[0])}, {f(pc[1])}] | {f(-b['delta1'][u])} | {pc[0] > -b['delta1'][u]} |"
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    a = json.loads((args.out / "analysis.json").read_text("utf-8"))
    g1 = a["g0_1_phase1_reproduction"]
    lines = ["# Phase 2 evaluation tables (generated from analysis.json)", ""]
    lines += [
        f"Selected rule: {json.dumps(a['selection'])}. Completeness: {json.dumps(a['completeness'])}.",
        "",
        f"G0.1 (L `full` = Phase 1 `rsd_r0` RRT\\*): {g1['identical']} / {g1['compared']} identical.",
    ]
    if a.get("replay"):
        lines.append(
            f"G0.2 replay: {a['replay']['replayed']} replayed, identical: {a['replay']['identical']}."
        )
    lines += ["", f"Recommendation inputs: `{json.dumps(a['recommendation'])}`", ""]
    for c, name in NAMES.items():
        b = a[c]
        lines += [f"## {name}", "", "### G0", "", "```", json.dumps(b["g0"], indent=1), "```", ""]
        lines += [f"δ₁: {json.dumps({k: round(v, 4) for k, v in b['delta1'].items()})}", ""]
        for fam, rows in b["families"].items():
            lines += [f"### {fam}", "", *family_table(rows), ""]
        lines += ["### Decision", "", "```", json.dumps(b["decision"], indent=1), "```", ""]
        lines += ["### System (descriptive)", "", *desc_table(b["describe"], DESC_SYSTEM), ""]
        lines += [
            "### RRT\\* and runtime (descriptive)",
            "",
            *desc_table(b["describe"], DESC_RRT),
            "",
        ]
        for key, name, nd in DESC_DICTS:
            lines += [f"### {name}", "", *dict_table(b["describe"], key, nd), ""]
        lines += ["### Safety", "", *desc_table(b["describe"], DESC_SAFETY), ""]
        lines += [
            "### P1 interval sensitivity (not pre-registered; does not change the verdict)",
            "",
            *sensitivity(b),
            "",
        ]
        lines += [
            "### Supportive (nominal)",
            "",
            "| Outcome | Δ̂ | nominal 95 % CI | full θ | p2 θ |",
            "|---|---|---|---|---|",
        ]
        for k, v in b["supportive_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['estimate'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] | "
                    f"{f(v['theta_old'])} | {f(v['theta_new'])} |"
                )
        lines += [
            "",
            "### Per level, completion (nominal)",
            "",
            "| Family / level | full | p2 | Δ̂ | CI |",
            "|---|---|---|---|---|",
        ]
        for k, v in b["per_level_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['theta_old'])} | {f(v['theta_new'])} | {f(v['estimate'])} | "
                    f"[{f(v['ci95'][0])}, {f(v['ci95'][1])}] |"
                )
        lines += [
            "",
            "### Per policy × level (descriptive): full → p2",
            "",
            "| Policy / level | Completion | Undelivered | Conflicts / ep. | Flow time | Steps / ep. | Iterations / call | RRT\\* calls / ep. | RRT\\* s / ep. (median) | Wall s / ep. (median) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for k in b["breakdown"]["full"]:
            x, y = b["breakdown"]["full"][k], b["breakdown"]["p2"][k]
            lines.append(
                f"| {k} | {f(x['completion'])} → {f(y['completion'])} | {x['undelivered']} → {y['undelivered']} | "
                f"{f(x['conflicts_per_episode'], 2)} → {f(y['conflicts_per_episode'], 2)} | "
                f"{f(x['flow_time_delivered_mean'], 1)} → {f(y['flow_time_delivered_mean'], 1)} | "
                f"{f(x['steps_per_episode'], 1)} → {f(y['steps_per_episode'], 1)} | "
                f"{f(x['rrt_iterations_per_call'], 0)} → {f(y['rrt_iterations_per_call'], 0)} | "
                f"{f(x['rrt_calls_per_episode'], 1)} → {f(y['rrt_calls_per_episode'], 1)} | "
                f"{f(x['rrt_s_per_episode_median'], 2)} → {f(y['rrt_s_per_episode_median'], 2)} | "
                f"{f(x['wall_s_per_episode_median'], 2)} → {f(y['wall_s_per_episode_median'], 2)} |"
            )
        lines += [
            "",
            "### Seeds (completion: better / worse / equal)",
            "",
            "| Policy | better | worse | equal |",
            "|---|---|---|---|",
        ]
        for k, v in b["seed_heterogeneity"].items():
            lines.append(f"| {k} | {v['seeds_better']} | {v['seeds_worse']} | {v['seeds_equal']} |")
        lines.append("")
    (args.out / "tables.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(len(lines), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
