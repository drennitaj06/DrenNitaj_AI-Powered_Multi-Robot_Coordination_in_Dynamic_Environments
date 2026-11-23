"""Render ``analysis.json`` of the Phase 3 evaluation as Markdown tables (no new computation).

Usage::

    python tools/phase3/phase3_report.py --out results/dev6a/phase3/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def family_table(rows: list[dict[str, Any]], with_factor: bool = False) -> list[str]:
    head = (
        "| "
        + ("Factor | " if with_factor else "")
        + "Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |"
    )
    out = [head, "|" + "---|" * (11 if with_factor else 10)]
    for r in rows:
        i = r["inf"]
        fac = f"{r.get('factor')} | " if with_factor else ""
        if i is None:
            out.append(
                f"| {fac}{r['unit']} | {r['hypothesis']} | — | not evaluable | — | — | 1 | {r['alpha_family']} | not rejected | — |"
            )
            continue
        dec = "**rejected**" if r["rejected"] else "not rejected"
        out.append(
            f"| {fac}{r['unit']} | {r['hypothesis']} | {f(i['theta_old'])} | {f(i['estimate'])} | "
            f"[{f(i['ci95'][0])}, {f(i['ci95'][1])}] | {f(i['p'], 5)} | {f(r['p_holm'], 5)} | "
            f"{r['alpha_family']} | {dec} | {f(r['mc_sensitive'])} |"
        )
    return out


def arms_table(arms: dict[str, dict[str, Any]]) -> list[str]:
    out = [
        "| Arm | Family | Completion | Undelivered / ep. | Successful | Incomplete | Conflicts / ep. | Stuck robots | frozen / osc. / loop / cycle per ep. | Top-priority undelivered | Wait ratio | Planner calls / 100 robot-steps |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, fams in arms.items():
        for fam, d in fams.items():
            s = d["stuck_per_episode"]
            out.append(
                f"| {arm} | {fam} | {f(d['completion'])} | {f(d['undelivered_per_episode'], 3)} | "
                f"{d['successful_episodes']} | {d['incomplete_episodes']} | {f(d['conflicts_per_episode'], 2)} | "
                f"{d['stuck_total']} | {f(s['frozen'], 3)} / {f(s['oscillating'], 3)} / "
                f"{f(s['conflict_loop'], 3)} / {f(s['deadlock_cycle'], 3)} | {f(d['top_priority_undelivered'], 0)} | "
                f"{f(d['wait_ratio_mean'], 3)} | {f(d['planner_calls_per_100_robot_steps'], 2)} |"
            )
    return out


def interventions_table(arms: dict[str, dict[str, Any]]) -> list[str]:
    out = [
        "| Arm | Family | Modified share (robot-steps) | (eligible) | R losses | Recovery changes | Recoveries S / O / D | Ended by progress | Max recoveries in a leg (q50 / q99 / max) | Recoveries per robot (mean / max) | Max per-robot modified share (q50 / max) | Max consecutive R losses (max) | Max steps without progress (q50 / max) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, fams in arms.items():
        for fam, d in fams.items():
            iv = d.get("interventions")
            if not iv:
                continue
            rs = iv["recoveries_started"]
            ml = iv["max_recoveries_in_a_leg"]
            pr = iv["recoveries_per_robot"]
            ms = iv["max_robot_modified_share"]
            sp = iv["max_steps_without_progress"]
            out.append(
                f"| {arm} | {fam} | {f(iv['modified_share_of_robot_steps'])} | "
                f"{f(iv['modified_share_of_eligible_robot_steps'])} | {iv['changes'].get('R_loss', 0)} | "
                f"{iv['changes'].get('recovery', 0)} | {rs.get('S', 0)} / {rs.get('O', 0)} / {rs.get('D', 0)} | "
                f"{f(iv['share_ended_by_progress'], 3)} | {f(ml.get('q50'), 0)} / {f(ml.get('q99'), 0)} / {f(ml.get('max'), 0)} | "
                f"{f(pr.get('mean'), 2)} / {f(pr.get('max'), 0)} | {f(ms.get('q50'), 3)} / {f(ms.get('max'), 3)} | "
                f"{f(iv['max_consecutive_r_losses'].get('max'), 0)} | {f(sp.get('q50'), 0)} / {f(sp.get('max'), 0)} |"
            )
    return out


def runtime_table(rt: dict[str, Any]) -> list[str]:
    out = [
        "| Family / arm | ms per step (median) | IQR | supervisor ms / step (median) | supervisor share of step time | ON / OFF median ratio | timing outliers flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in rt.items():
        out.append(
            f"| {k} | {f(v['ms_per_step_median'], 3)} | [{f(v['ms_per_step_iqr'][0], 3)}, {f(v['ms_per_step_iqr'][1], 3)}] | "
            f"{f(v['supervisor_ms_per_step_median'], 3)} | {f(v['supervisor_share_of_step_time'], 3)} | "
            f"{f(v.get('ratio_to_off_median'), 3)} | {v['timing_outliers_flagged']} |"
        )
    return out


def g0_table(g: dict[str, Any]) -> list[str]:
    keys = (
        "episodes",
        "errors",
        "physical_collisions",
        "inv_total",
        "inv_on_exceeding_off",
        "supervisor_introduced_invalid_moves",
        "max_consecutive_r_losses",
        "fairness_bound_ok",
        "burst_bound_violations",
        "params_frozen",
        "escalation_off",
        "instances_identical_across_arms",
        "pass",
    )
    out = ["| Check | Value |", "|---|---|"]
    for k in keys:
        v = g[k]
        out.append(f"| {k} | {len(v) if isinstance(v, list) else f(v)} |")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    a = json.loads((args.out / "analysis.json").read_text("utf-8"))
    lines: list[str] = ["# Phase 3 evaluation tables (generated from analysis.json)", ""]
    for s in ("3b", "3e"):
        if s not in a:
            continue
        b = a[s]
        lines += [f"## Stage {s}", "", "### G0", "", *g0_table(b["g0"]), ""]
        if "g0_1_block_c_reproduction" in b:
            r = b["g0_1_block_c_reproduction"]
            lines += [
                f"G0.1 Block C reproduction: {r['identical']} / {r['compared']} identical.",
                "",
            ]
        lines += [f"δ₁: {json.dumps({k: round(v, 4) for k, v in b['delta1'].items()})}", ""]
        for fam, rows in b["results"]["families"].items():
            lines += [f"### {fam}", "", *family_table(rows), ""]
        lines += ["### Decision", "", "```", json.dumps(b["decision"], indent=1), "```", ""]
        lines += ["### Arms", "", *arms_table(b["arms"]), ""]
        lines += ["### Interventions", "", *interventions_table(b["arms"]), ""]
        lines += ["### Runtime", "", *runtime_table(b["runtime"]), ""]
        lines += [
            "### Supportive (nominal)",
            "",
            "| Outcome | Δ̂ | nominal 95 % CI | OFF θ |",
            "|---|---|---|---|",
        ]
        for k, v in b["results"]["supportive_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['estimate'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] | {f(v['theta_old'])} |"
                )
        lines += [
            "",
            "### Per level (nominal)",
            "",
            "| Family / level | OFF | ON | Δ̂ | CI |",
            "|---|---|---|---|---|",
        ]
        for k, v in b["results"]["per_level_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['theta_old'])} | {f(v['theta_new'])} | {f(v['estimate'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] |"
                )
        lines += [
            "",
            "### Seeds (completion: seeds better / worse / equal)",
            "",
            "| Policy | better | worse | equal |",
            "|---|---|---|---|",
        ]
        for k, v in b["results"]["seed_heterogeneity"].items():
            lines.append(f"| {k} | {v['seeds_better']} | {v['seeds_worse']} | {v['seeds_equal']} |")
        lines.append("")
    if "3c" in a:
        c = a["3c"]
        lines += ["## Stage 3c", "", "### G0", "", *g0_table(c["g0"]), ""]
        for fam, rows in c["results"]["families"].items():
            lines += [f"### {fam} (main effects)", "", *family_table(rows, with_factor=True), ""]
        lines += [
            "### Total stuck robots, main effects (nominal)",
            "",
            "| Factor / family | Δ̂ | CI |",
            "|---|---|---|",
        ]
        for k, v in c["results"]["total_stuck_nominal"].items():
            if v:
                lines.append(
                    f"| {k} | {f(v['estimate'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] |"
                )
        lines += [
            "",
            "### Retention",
            "",
            "```",
            json.dumps(c["results"]["retention"], indent=1),
            "```",
            "",
        ]
        lines += ["### Cell completion", "", "| Arm | MADQL | IPPO | MAPPO |", "|---|---|---|---|"]
        for arm, v in c["results"]["cell_completion"].items():
            lines.append(f"| {arm} | {f(v['madql'])} | {f(v['ippo'])} | {f(v['mappo'])} |")
        lines += [
            "",
            "### Arms",
            "",
            *arms_table(c["arms"]),
            "",
            "### Interventions",
            "",
            *interventions_table(c["arms"]),
            "",
            "### Runtime",
            "",
            *runtime_table(c["runtime"]),
            "",
        ]
    (args.out / "tables.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(len(lines), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
