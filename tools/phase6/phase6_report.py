"""Issue 6A Phase 6 — tables from ``analysis.json`` (protocol §6.12). Writes ``tables.md`` next
to it. Reads only the analysis output; computes no new statistic.

Usage::

    python tools/phase6/phase6_report.py --out results/dev6a/phase6/eval
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
import phase3_report as p3r  # noqa: E402

f = p3r.f
UNIT_TITLES = {
    ("H-L2", "astar"): "H-L2 (supervisor R + S + D vs OFF), A\\* — rule 3b",
    ("H-L2", "rrtstar"): "H-L2 (supervisor R + S + D vs OFF), RRT\\* — rule 3e",
    ("H-L12", "astar"): "H-L12 (Issue 6A system vs Issue #6 system), A\\* — rule 3b",
    ("H-L12", "rrtstar"): "H-L12 (Issue 6A system vs Issue #6 system), RRT\\* — rule 3e",
    ("H-v2", "FA"): "H-v2 (MADQL-v2 n3 vs historical MADQL), FA: A\\*, R + S + D — Phase 4 rule",
    ("H-v2", "FR"): "H-v2 (MADQL-v2 n3 vs historical MADQL), FR: RRT\\*, R + S + D — Phase 4 rule",
    ("H-v2", "NA"): "H-v2 (MADQL-v2 n3 vs historical MADQL), NA: A\\*, OFF — Phase 4 rule",
}
ARM_ROWS = (
    ("completion", "Completion", 4),
    ("released", "Released tasks", 0),
    ("delivered", "Delivered tasks", 0),
    ("undelivered_per_episode", "Undelivered / episode", 3),
    ("successful_episodes", "Successful episodes", 0),
    ("top_priority_completion", "High-priority completion", 4),
    ("flow_time_delivered_mean", "Flow time (delivered)", 2),
    ("makespan_mean", "Makespan (mean)", 1),
    ("steps_per_episode", "Steps / episode", 1),
    ("conflicts_per_episode", "Conflicts / episode", 3),
    ("conflicts_per_delivered", "Conflicts / delivered task", 4),
    ("ro_per_1000_moves", "RO / 1 000 moves (all robots; veto outcome)", 3),
    ("ro_nonidle_per_1000_moves", "RO / 1 000 moves (non-idle; secondary)", 3),
    ("moves_per_episode", "Moves / episode (a^arb)", 1),
    ("idle_move_share", "Share of moves by idle robots", 3),
    ("rr_ctrl_per_100_robot_steps", "RR / 100 robot-steps (K^ctrl)", 4),
    ("rr_arb_per_100_robot_steps", "RR / 100 robot-steps (K^arb)", 4),
    ("wait_ratio_mean", "Wait ratio", 4),
    ("planner_calls_per_100_robot_steps", "Planner calls / 100 robot-steps", 2),
    ("supervisor_changes_per_100_robot_steps", "Supervisor changes / 100 robot-steps", 3),
    ("stuck_total", "Stuck robots (total)", 0),
    ("share_stay", "Share STAY", 4),
    ("inference_ms_per_step", "Inference ms / step", 3),
    ("planning_s_per_episode_median", "Planning s / episode (median)", 2),
    ("wall_s_per_episode_median", "Wall s / episode (median)", 2),
    ("physical_collisions", "Physical collisions", 0),
    ("invalid_moves", "Invalid moves (INV)", 0),
    ("invalid_reference_paths", "Invalid reference paths", 0),
    ("reference_paths_checked", "Reference paths checked", 0),
)


def fmt(x: Any, nd: int) -> str:
    if isinstance(x, float) and nd == 0:
        return f"{x:.0f}"
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    return f(x, nd) if nd else f(x)


def family_table(rows: list[dict[str, Any]]) -> list[str]:
    """Confirmatory family table. A not-evaluable row shows no estimate, interval or p-value
    (protocol §5, Block D §7); only its Holm treatment is stated."""
    out = [
        "| Unit | Outcome | old θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        i = r["inf"]
        if i is None:
            out.append(
                f"| {r['unit']} | {r['hypothesis']} | **not evaluable** | — | — | — | — "
                f"(Holm input p = 1 by rule) | {r['alpha_family']} | not rejected | — |"
            )
            continue
        dec = "**rejected**" if r["rejected"] else "not rejected"
        out.append(
            f"| {r['unit']} | {r['hypothesis']} | {f(i['theta_old'])} | {f(i['estimate'])} | "
            f"[{f(i['ci95'][0])}, {f(i['ci95'][1])}] | {f(i['p'], 5)} | {f(r['p_holm'], 5)} | "
            f"{r['alpha_family']} | {dec} | {f(r['mc_sensitive'])} |"
        )
    return out


def inf_row(name: str, i: dict[str, Any] | None) -> str:
    if i is None:
        return f"| {name} | **not evaluable** (no estimate, interval or p-value) | — | — | — |"
    return (f"| {name} | {f(i['theta_old'])} | {f(i['estimate'])} | "
            f"[{f(i['ci95'][0])}, {f(i['ci95'][1])}] | {f(i['p'], 5)} |")  # fmt: skip


def nominal_table(title: str, rows: dict[str, Any]) -> list[str]:
    out = ["", f"**{title}** (nominal 95 % intervals, unadjusted; no decision)", "",
           "| Outcome | old θ | Δ̂ | nominal 95 % CI | p (nominal) |", "|---|---|---|---|---|"]  # fmt: skip
    out += [inf_row(k, v) for k, v in rows.items()]
    return out


def unit_section(name: str, key: str, u: dict[str, Any]) -> list[str]:
    lines = ["", f"## {UNIT_TITLES[(name, key)]}", ""]
    for fam, rows in u["families"].items():
        lines += ["", f"### {fam}", "", *family_table(rows)]
    lines += [
        "",
        "### Decision",
        "",
        "```",
        json.dumps(u["decision"], indent=1, default=str),
        "```",
    ]
    lines += nominal_table("Secondary outcomes", u["secondary_nominal"])
    if "ro_per_run_nominal" in u:
        lines += nominal_table("RO per 1 000 moves by policy pair", u["ro_per_run_nominal"])
    return lines


def arms_section(arms: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for planner, cells in arms.items():
        names = list(cells)
        lines += ["", f"## Descriptive per arm: {planner}", "",
                  "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]  # fmt: skip
        for k, label, nd in ARM_ROWS:
            lines.append(
                f"| {label} | " + " | ".join(fmt(cells[a].get(k), nd) for a in names) + " |"
            )
        lines += ["", "RO per 1 000 moves by run:", ""]
        for a in names:
            lines.append(f"- `{a}`: " + ", ".join(
                f"{r} {f(v, 3)}" for r, v in cells[a].get("ro_per_1000_moves_by_run", {}).items()))  # fmt: skip
        lines += ["", "Conflicts by type (totals):", ""]
        for a in names:
            lines.append(f"- `{a}`: {cells[a].get('conflicts_by_type', 'data incomplete')}")
        lines += ["", "Stuck robots by class (totals):", ""]
        for a in names:
            lines.append(f"- `{a}`: {cells[a].get('stuck_by_class', 'data incomplete')}")
    return lines


def render(a: dict[str, Any]) -> str:
    s = a["summary"]
    lines = [
        "# Issue 6A Phase 6: confirmatory tables (generated from analysis.json)",
        "",
        f"Completeness: `{json.dumps(a['completeness'])}`. Replay: `{json.dumps(a['replay'])}`.",
        f"Freeze SHA-256: `{a.get('freeze_sha256')}`. Environment: `{json.dumps(a.get('environment'))}`.",
        "",
        "## Summary of the pre-registered decisions",
        "",
        "| Unit | Verdict |",
        "|---|---|",
    ]  # fmt: skip
    for p, v in s["H-L2"].items():
        lines.append(f"| H-L2 {p} | **{v}** |")
    for p, v in s["H-L12"].items():
        lines.append(f"| H-L12 {p} | **{v}** |")
    for c, v in s["H-v2"].items():
        lines.append(f"| H-v2 {c} | **{v}** |")
    lines += [
        "",
        "| Confirmation statement (protocol §6.8) | Result |",
        "|---|---|",
        f"| Supervisor R + S + D confirmed (H-L2 go on A\\* and RRT\\*) | **{f(s['supervisor_confirmed'])}** |",
        f"| MADQL-v2 n3 confirmed as forward MADQL | **{f(s['madql_v2_confirmed'])}** |",
        f"| Integrated Issue 6A system confirmed (H-L12 go on both) | **{f(s['integrated_system_confirmed'])}** |",
        f"| G0 pass (A\\*, RRT\\*) | {s['g0_pass']} |",
        f"| Replay identical | {f(s['replay_identical'])} |",
        "",
        f"Error control: family-wise error ≤ 0.05 per unit over {len(s['error_control']['units'])} "
        f"units; global bound on any false rejection ≤ "
        f"{s['error_control']['global_bound_any_false_rejection']:.3f}.",
        "",
        "## G0 (integrity and safety)",
    ]  # fmt: skip
    for p, g in a["g0"].items():
        lines += ["", f"### {p}", "", "```", json.dumps(g, indent=1, default=str), "```"]
    for name in ("H-L2", "H-L12"):
        for p, u in a[name].items():
            lines += unit_section(name, p, u)
    for c, u in a["H-v2"].items():
        lines += unit_section("H-v2", c, u)
    lines += ["", "## Context (pre-registered, nominal, non-gating): MADQL-v2 vs forward PPO"]
    for p, rows in a["context_v2_vs_forward_ppo"].items():
        lines += nominal_table(f"{p}: old = v5 PPO (R + S + D), new = MADQL-v2 (R + S + D)", rows)
    lines += arms_section(a["arms"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/dev6a/phase6/eval"))
    args = ap.parse_args(argv)
    a = json.loads((args.out / "analysis.json").read_text("utf-8"))
    (args.out / "tables.md").write_text(render(a), "utf-8")
    print(f"{args.out / 'tables.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
