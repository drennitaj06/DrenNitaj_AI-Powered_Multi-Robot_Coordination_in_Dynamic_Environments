"""Render the Phase 4 JSON outputs as Markdown tables (no new computation).

Usage::

    python tools/phase4/phase4_report.py select --out results/dev6a/phase4/select
    python tools/phase4/phase4_report.py eval   --out results/dev6a/phase4/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase3"))
from phase3_report import f, family_table  # noqa: E402

ARMS_S = ("hist", "n3", "t8", "n3t8")
SEL_ROWS = (
    ("completion", "Completion", 4),
    ("undelivered_per_episode", "Undelivered / ep.", 3),
    ("successful_episodes", "Successful ep.", 0),
    ("flow_time_delivered", "Flow time (delivered)", 2),
    ("steps_per_episode", "Steps / ep.", 1),
    ("wait_ratio_mean", "Wait ratio", 4),
    ("conflicts_per_episode", "Conflicts / ep.", 2),
    ("calls_per_100_robot_steps", "Planner calls / 100 robot-steps", 2),
    ("supervisor_changes_per_100_robot_steps", "Supervisor changes / 100 robot-steps", 3),
    ("stuck_per_episode", "Stuck robots / ep.", 3),
    ("share_stay", "Share STAY", 4),
    ("share_stay_free_cell", "Share STAY, free suggested cell", 4),
    ("share_self_loop_stay", "Share self-loop STAY", 4),
    ("share_in_stay_spells_ge20", "Share in STAY spells ≥ 20", 4),
    ("share_follow_suggestion", "Share following the suggestion", 4),
    ("mean_gap", "Mean gap Q(STAY) − best move", 4),
    ("mean_q_chosen", "Mean Q of the chosen action", 3),
    ("inference_ms_per_step", "Inference ms / step", 3),
    ("physical_collisions", "Physical collisions", 0),
    ("invalid_moves", "Invalid moves", 0),
    ("invalid_reference_paths", "Invalid reference paths", 0),
)
EVAL_ROWS = (
    ("completion", "Completion", 4),
    ("delivered", "Delivered", 0),
    ("undelivered_per_episode", "Undelivered / ep.", 3),
    ("successful_episodes", "Successful ep.", 0),
    ("top_priority_completion", "High-priority completion", 4),
    ("flow_time_delivered_mean", "Flow time (delivered)", 2),
    ("makespan_mean", "Makespan (mean)", 1),
    ("steps_per_episode", "Steps / ep.", 1),
    ("conflicts_per_episode", "Conflicts / ep.", 2),
    ("conflicts_per_delivered", "Conflicts / delivered", 3),
    ("conflicts_per_100_robot_steps", "Conflicts / 100 robot-steps", 3),
    ("wait_ratio_mean", "Wait ratio", 4),
    ("planner_calls_per_100_robot_steps", "Planner calls / 100 robot-steps", 2),
    ("deviations_per_100_robot_steps", "Deviations / 100 robot-steps", 2),
    ("supervisor_changes_per_100_robot_steps", "Supervisor changes / 100 robot-steps", 3),
    ("stuck_total", "Stuck robots", 0),
    ("share_stay", "Share STAY", 4),
    ("share_stay_free_cell", "Share STAY, free suggested cell", 4),
    ("share_stay_occupied_cell", "Share STAY, occupied suggested cell", 4),
    ("share_self_loop_stay", "Share self-loop STAY", 4),
    ("share_in_stay_spells_ge20", "Share in STAY spells ≥ 20", 4),
    ("share_follow_suggestion", "Share following the suggestion", 4),
    ("mean_gap", "Mean gap Q(STAY) − best move", 4),
    ("mean_q_chosen", "Mean Q of the chosen action", 3),
    ("inference_ms_per_step", "Inference ms / step", 3),
    ("planning_s_per_episode_median", "Planning s / ep. (median)", 2),
    ("wall_s_per_episode_median", "Wall s / ep. (median)", 2),
    ("physical_collisions", "Physical collisions", 0),
    ("invalid_moves", "Invalid moves (INV)", 0),
    ("invalid_reference_paths", "Invalid reference paths", 0),
    ("reference_paths_checked", "Reference paths checked", 0),
    ("errors", "Errors", 0),
)
NAMES = {
    "FA": "FA: A\\*, R + S + D (primary)",
    "FR": "FR: RRT\\*, R + S + D",
    "NA": "NA: A\\*, no supervisor (policy's own decisions)",
}


def select_report(out: Path) -> list[str]:
    r = json.loads((out / "selection_report.json").read_text("utf-8"))
    sel = r["selection"]
    lines = ["# Phase 4 development selection tables (generated)", ""]
    lines += [f"Completeness: `{json.dumps(sel['completeness'])}`", ""]
    lines += ["## Selection (Block S, rule §3.5)", "",
              "| Candidate | F completion | N completion | F flow time | F within 0.005 | N improves | Eligible |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    h = sel["historical"]
    lines.append(
        f"| historical | {f(h['F_completion'])} | {f(h['N_completion'])} | — | — | — | — |"
    )
    for c, v in sel["candidates"].items():
        lines.append(
            f"| {c} | {f(v['F_completion'])} | {f(v['N_completion'])} | {f(v['F_flow_time'], 2)} | "
            f"{v['F_eligible']} | {v['N_improves']} | {v['eligible']} |"
        )
    lines += ["", f"**Selected: {sel['selected']}** (eligible: {sel['selected_eligible']}).", ""]
    for cond in ("FA", "NA"):
        lines += [f"## Block S, condition {cond} (descriptive)", "",
                  "| Metric | " + " | ".join(ARMS_S) + " |", "|---|" + "---|" * len(ARMS_S)]  # fmt: skip
        for k, name, nd in SEL_ROWS:
            lines.append(
                f"| {name} | "
                + " | ".join(f(r["table"][a][cond].get(k), nd) for a in ARMS_S)
                + " |"
            )
        lines.append("")
    lines += [
        "## Block S completion per policy",
        "",
        "| Arm | Cond | seed 1 | seed 2 | seed 3 |",
        "|---|---|---|---|---|",
    ]
    for a in ARMS_S:
        for c in ("FA", "NA"):
            vals = [f(r["per_policy"][f"{a}:{c}:madql_seed{k}"]["completion"]) for k in (1, 2, 3)]
            lines.append(f"| {a} | {c} | " + " | ".join(vals) + " |")
    lines += ["", "## Training (block A validation; training logs)", "",
              "| Run | best step | best completion | mean completion, last 5 | max swing after 500 k | late loss | wall h |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, v in r["training"].items():
        lines.append(
            f"| {k} | {v['best_step']} | {f(v['best_key'][0])} | {f(v['mean_completion_last5'])} | "
            f"{f(v['max_abs_change_after_500k'])} | {f(v['late_loss_mean'], 4)} | {f(v['wall_s'] / 3600, 2)} |"
        )
    lines += ["", "## Checkpoint churn, 600 k – 1 M (diagnostics sample)", "",
              r"| Run | greedy action changed (mean) | STAY/move flipped (mean) | median \|Δ gap\| (mean) | STAY preferred at 1 M |",
              "|---|---|---|---|---|"]  # fmt: skip
    for k, v in r["churn"].items():
        if k == "n_states":
            continue
        lines.append(
            f"| {k} | {f(v['greedy_changed_mean'])} | {f(v['stay_move_flip_mean'])} | "
            f"{f(v['gap_abs_change_median_mean'])} | {f(v['share_stay_preferred_at_1M'])} |"
        )
    return lines


def eval_report(out: Path) -> list[str]:
    a = json.loads((out / "analysis.json").read_text("utf-8"))
    lines = ["# Phase 4 Block C evaluation tables (generated from analysis.json)", ""]
    lines += [
        f"Selected candidate: **{a['selection']['selected']}**. Completeness: `{json.dumps(a['completeness'])}`.",
        "",
    ]
    g1 = a["g0_1_phase1_reproduction"]
    for c in ("FA", "NA", "FR"):
        lines.append(
            f"G0.1 {c} (`hist` = Phase 1 records): {g1[c]['identical']} / {g1[c]['compared']} identical."
        )
    if a.get("replay"):
        lines.append(
            f"G0.2 replay: {a['replay']['replayed']} replayed, identical: {a['replay']['identical']}."
        )
    lines += ["", f"Recommendation inputs: `{json.dumps(a['recommendation'])}`", ""]
    for c in ("FA", "FR", "NA"):
        b = a[c]
        lines += [
            f"## {NAMES[c]}",
            "",
            "### G0",
            "",
            "```",
            json.dumps(b["g0"], indent=1),
            "```",
            "",
        ]
        for fam, rows in b["families"].items():
            lines += [f"### {fam}", "", *family_table(rows), ""]
        lines += ["### Decision", "", "```", json.dumps(b["decision"], indent=1), "```", ""]
        cols = [("hist", "all"), ("v2", "all")] + [
            (x, f"madql_seed{k}") for k in (1, 2, 3) for x in ("hist", "v2")
        ]
        lines += ["### Descriptive (all policies, then per policy)", "",
                  "| Metric | " + " | ".join(f"{x} {run.replace('madql_', '')}" for x, run in cols) + " |",
                  "|---|" + "---|" * len(cols)]  # fmt: skip
        for k, name, nd in EVAL_ROWS:
            lines.append(
                f"| {name} | "
                + " | ".join(f(b["describe"][x][run].get(k), nd) for x, run in cols)
                + " |"
            )
        lines.append("")
        for key, title in (("conflicts_by_type", "Conflicts by type (total)"),
                           ("calls_by_trigger_share", "Planner calls by trigger (share)"),
                           ("stuck_by_class", "Stuck robots by class (total)")):  # fmt: skip
            subs = sorted(
                set(b["describe"]["hist"]["all"][key]) | set(b["describe"]["v2"]["all"][key])
            )
            lines += [f"### {title}", "", "| Category | hist | v2 |", "|---|---|---|"]
            for s in subs:
                lines.append(
                    f"| {s} | {f(b['describe']['hist']['all'][key].get(s), 4)} | {f(b['describe']['v2']['all'][key].get(s), 4)} |"
                )
            lines.append("")
        lines += ["### Action shares (STAY, up, down, left, right — non-idle robot-steps)", "",
                  "| Arm | " + " | ".join(str(i) for i in range(5)) + " |", "|---|---|---|---|---|---|"]  # fmt: skip
        for x in ("hist", "v2"):
            lines.append(
                f"| {x} | "
                + " | ".join(f(v, 4) for v in b["describe"][x]["all"]["action_shares"])
                + " |"
            )
        lines += [
            "",
            "### Supportive (nominal)",
            "",
            "| Outcome | Δ̂ | nominal 95 % CI | hist θ | v2 θ |",
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
            "| Level | hist | v2 | Δ̂ | CI |",
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
            "### Seeds (completion: better / worse / equal)",
            "",
            "| Policy pair | better | worse | equal |",
            "|---|---|---|---|",
        ]
        for k, v in b["seeds"].items():
            lines.append(f"| {k} | {v['seeds_better']} | {v['seeds_worse']} | {v['seeds_equal']} |")
        lines.append("")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("select", "eval"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    lines = select_report(args.out) if args.cmd == "select" else eval_report(args.out)
    name = "selection_tables.md" if args.cmd == "select" else "tables.md"
    (args.out / name).write_text("\n".join(lines) + "\n", "utf-8")
    print(len(lines), "lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
