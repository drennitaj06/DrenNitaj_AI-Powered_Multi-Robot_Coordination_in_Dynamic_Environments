"""Render the Phase 1 ``analysis.json`` as Markdown tables (no new computation).

Usage::

    python tools/phase1/phase1_report.py --out results/dev6a/phase1/eval
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
    "C1": "C1: A\\*, supervisor R + S + D",
    "C2": "C2: A\\*, no supervisor",
    "C3": "C3: RRT\\*, supervisor R + S + D",
    "C4": "C4: RRT\\*, no supervisor",
}


def describe_table(desc: dict[str, dict[str, Any]]) -> list[str]:
    out = [
        "| Arm | Family | Completion | Undelivered / ep. | Successful ep. | Conflicts / ep. | Stuck robots | Calls / 100 robot-steps | T3 / 100 | Deviations / 100 | Repairs (success share) | Expansions / robot-step | Iterations / robot-step | Step ms (median) | Planning ms / step (median) | Planning share | Steps / ep. |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, fams in desc.items():
        for fam, d in fams.items():
            out.append(
                f"| {arm} | {fam} | {f(d['completion'])} | {f(d['undelivered_per_episode'], 3)} | "
                f"{d['successful_episodes']} | {f(d['conflicts_per_episode'], 2)} | {d['stuck_total']} | "
                f"{f(d['planner_calls_per_100_robot_steps'], 2)} | {f(d['t3_calls_per_100_robot_steps'], 2)} | "
                f"{f(d['deviations_per_100_robot_steps'], 2)} | {d['repairs']} ({f(d['repair_success_share'], 3)}) | "
                f"{f(d['expansions_per_robot_step'], 1)} | {f(d['iterations_per_robot_step'], 0)} | "
                f"{f(d['step_ms_median'], 2)} | {f(d['planning_ms_per_step_median'], 2)} | "
                f"{f(d['planning_share_of_step_time'], 3)} | {f(d['steps_per_episode'], 1)} |"
            )
    out += [
        "",
        "Per-episode totals (descriptive):",
        "",
        "| Arm | Family | Planner calls / ep. | Planner work / ep. | Repairs / ep. | Supervisor changes / ep. | Planning s / ep. (median) | Wall s / ep. (median) | Terminated share |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, fams in desc.items():
        for fam, d in fams.items():
            out.append(
                f"| {arm} | {fam} | {f(d['planner_calls_per_episode'], 1)} | "
                f"{f(d['planner_work_per_episode'], 0)} | {f(d['repairs_per_episode'], 1)} | "
                f"{f(d['supervisor_changes_per_episode'], 2)} | "
                f"{f(d['planning_s_per_episode_median'], 3)} | {f(d['wall_s_per_episode_median'], 3)} | "
                f"{f(d['terminated_share'], 3)} |"
            )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    a = json.loads((args.out / "analysis.json").read_text("utf-8"))
    lines = ["# Phase 1 evaluation tables (generated from analysis.json)", ""]
    lines += [
        "## G0.1 reproduction of Phase 3 records",
        "",
        "| Arm / planner | compared | identical |",
        "|---|---|---|",
    ]
    for k, v in a["g0_1_phase3_reproduction"].items():
        lines.append(f"| {k} | {v['compared']} | {v['identical']} |")
    if a.get("replay"):
        lines += [
            "",
            f"G0.2 replay: {a['replay']['replayed']} replayed, identical: {a['replay']['identical']}.",
        ]
    lines.append("")
    for c, name in NAMES.items():
        if c not in a:
            continue
        b = a[c]
        lines += [f"## {name}", "", "### G0", "", "```", json.dumps(b["g0"], indent=1), "```", ""]
        lines += [f"δ₁: {json.dumps({k: round(v, 4) for k, v in b['delta1'].items()})}", ""]
        for fam, rows in b["families"].items():
            lines += [f"### {fam}", "", *family_table(rows), ""]
        lines += ["### Decision", "", "```", json.dumps(b["decision"], indent=1), "```", ""]
        lines += ["### Arms", "", *describe_table(b["describe"]), ""]
        lines += [
            "### Supportive (nominal)",
            "",
            "| Outcome | Δ̂ | nominal 95 % CI | repair-off θ |",
            "|---|---|---|---|",
        ]
        for k, v in b["supportive_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['estimate'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] | {f(v['theta_old'])} |"
                )
        lines += [
            "",
            "### Per level (nominal)",
            "",
            "| Family / level | off | on | Δ̂ | CI |",
            "|---|---|---|---|---|",
        ]
        for k, v in b["per_level_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['theta_old'])} | {f(v['theta_new'])} | {f(v['estimate'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] |"
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
