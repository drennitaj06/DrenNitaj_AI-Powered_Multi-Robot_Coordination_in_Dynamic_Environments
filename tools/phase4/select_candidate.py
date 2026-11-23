"""Issue 6A Phase 4 — development selection of the MADQL-v2 candidate (protocol §3.5).

Reads ``results/dev6a/phase4/select/records_4s.jsonl`` (Block S) and applies the pre-declared
rule:

1. eligible iff F completion ≥ historical F completion − 0.005 **and** N completion > historical
   N completion (pooled Σ delivered / Σ released over the 3 policies);
2. the eligible candidate with the highest N completion; within 0.005, the lower F mean flow time
   of delivered tasks;
3. none eligible → the highest N completion goes to Block C as "not eligible at selection".

Also reports (never used for the selection): MADQL decision statistics per arm and condition,
the training-log stability of every run, and the checkpoint churn of each candidate on the
diagnostics observation sample.

Usage::

    python tools/phase4/select_candidate.py --out results/dev6a/phase4/select
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase4 as p4  # noqa: E402

sys.path.insert(0, str(HERE.parent / "phase3"))
import phase3 as p3  # noqa: E402

TOL_F = 0.005
TIE = 0.005
CHURN_STEPS = tuple(range(600_000, 1_000_001, 50_000))


def records(out: Path) -> list[dict[str, Any]]:
    path = out / "records_4s.jsonl"
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def completeness(recs: list[dict[str, Any]]) -> dict[str, Any]:
    want = {
        (f"{a}:{c}", r, lv, s)
        for c in p4.STAGES["4s"]["conds"]
        for a in p4.STAGES["4s"]["arms"]
        for r in p4.RUNS
        for lv in ("static", "obstacles", "full")
        for s in p4.BLOCK_S
    }
    have = [(x["arm"], x["run"], x["level"], int(x["seed"])) for x in recs]
    return {
        "expected": len(want),
        "present": len(set(have) & want),
        "duplicates": len(have) - len(set(have)),
        "unexpected": len(set(have) - want),
        "errors": sum(x["status"] != "ok" for x in recs),
        "complete": set(have) == want and len(have) == len(set(have)),
    }


def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rs if r["status"] == "ok"]
    rel = sum(r["released"] for r in ok)
    dl = sum(r["delivered"] for r in ok)
    flow = [r["mean_flow_time"] for r in ok if r["mean_flow_time"] is not None]
    fl_w = sum(r["mean_flow_time"] * r["delivered"] for r in ok if r["mean_flow_time"] is not None)
    robot_steps = sum(r["robots"] * r["steps"] for r in ok)
    m = [r["madql"] for r in ok]
    ns = sum(x["nonidle_steps"] for x in m)
    out = {
        "episodes": len(rs),
        "completion": dl / rel if rel else None,
        "undelivered_per_episode": (rel - dl) / len(ok) if ok else None,
        "successful_episodes": sum(r["delivered"] == r["released"] for r in ok),
        "flow_time_delivered": fl_w / dl if dl else None,
        "mean_of_episode_flow": float(np.mean(flow)) if flow else None,
        "conflicts_per_episode": sum(r["conflicts_total"] for r in ok) / len(ok) if ok else None,
        "calls_per_100_robot_steps": 100 * sum(r["planner_calls"] for r in ok) / robot_steps,
        "wait_ratio_mean": float(np.mean([r["wait_ratio"] for r in ok])),
        "steps_per_episode": float(np.mean([r["steps"] for r in ok])),
        "supervisor_changes_per_100_robot_steps": 100
        * sum(r.get("sup_changed", 0) for r in ok)
        / robot_steps,
        "stuck_per_episode": sum(len(r["stuck"]) for r in ok) / len(ok),
        "physical_collisions": sum(r["physical_collisions"] for r in ok),
        "invalid_moves": sum(r["conflicts"]["INV"] for r in ok),
        "invalid_reference_paths": sum(r["path_checks"]["invalid"] for r in ok),
        "share_stay": sum(x["stay"] for x in m) / ns,
        "share_stay_free_cell": sum(x["stay_free"] for x in m) / ns,
        "share_self_loop_stay": sum(x["self_loop_stay"] for x in m) / ns,
        "share_in_stay_spells_ge20": sum(x["spell_ge20_steps"] for x in m) / ns,
        "share_follow_suggestion": sum(x["follow"] for x in m) / ns,
        "mean_gap": sum(x["gap_sum"] for x in m) / max(1, sum(x["gap_n"] for x in m)),
        "mean_q_chosen": sum(x["q_chosen_sum"] for x in m) / ns,
        "inference_ms_per_step": 1000
        * sum(x["inference_s"] for x in m)
        / max(1, sum(x["inference_calls"] for x in m)),
    }
    return out


def select(table: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    h = table["hist"]
    rows = {}
    for c in p4.CANDIDATES:
        f_ok = table[c]["FA"]["completion"] >= h["FA"]["completion"] - TOL_F
        n_ok = table[c]["NA"]["completion"] > h["NA"]["completion"]
        rows[c] = {
            "F_completion": table[c]["FA"]["completion"],
            "N_completion": table[c]["NA"]["completion"],
            "F_flow_time": table[c]["FA"]["flow_time_delivered"],
            "F_eligible": f_ok,
            "N_improves": n_ok,
            "eligible": f_ok and n_ok,
        }
    eligible = [c for c in p4.CANDIDATES if rows[c]["eligible"]]
    pool = eligible or list(p4.CANDIDATES)
    best = max(rows[c]["N_completion"] for c in pool)
    near = [c for c in pool if rows[c]["N_completion"] >= best - TIE]
    pick = min(near, key=lambda c: (rows[c]["F_flow_time"], p4.CANDIDATES.index(c)))
    return {
        "rule": "protocol §3.5",
        "historical": {
            "F_completion": h["FA"]["completion"],
            "N_completion": h["NA"]["completion"],
        },
        "candidates": rows,
        "eligible": eligible,
        "selected": pick,
        "selected_eligible": pick in eligible,
        "within_tie_of_best": near,
    }


def training_logs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    roots = {f"hist_seed{k}": Path(f"results/training/madql_seed{k}") for k in (1, 2, 3)}
    roots |= {
        f"{c}_seed{k}": p4.TRAINING / f"{c}_seed{k}" for c in p4.CANDIDATES for k in (1, 2, 3)
    }
    for name, root in roots.items():
        val = [json.loads(x) for x in (root / "validation.jsonl").read_text("utf-8").splitlines()]
        comp = [v["key"][0] for v in val if v["planner"] == "astar"]
        prog = [
            json.loads(x)
            for x in (root / "train_log.jsonl").read_text("utf-8").splitlines()
            if '"progress"' in x
        ]
        late = [p["loss_mean"] for p in prog if p["env_steps"] > 500_000]
        summ = json.loads((root / "summary.json").read_text("utf-8"))
        out[name] = {
            "best_step": summ["best_step"],
            "best_key": summ["best_key"],
            "validation_completion": comp,
            "mean_completion_last5": float(np.mean(comp[-5:])),
            "max_abs_change_after_500k": float(np.max(np.abs(np.diff(comp[9:])))),
            "late_loss_mean": float(np.mean(late)),
            "wall_s": summ["wall_s"],
        }
    return out


def churn() -> dict[str, Any]:
    import torch

    from robocoord.marl.checkpoint import load_inference
    from robocoord.marl.networks import masked_argmax

    z = np.load("results/dev6a/phase4/diagnostics/states.npz")
    obs = torch.from_numpy(z["samples"])
    full = torch.ones((obs.shape[0], 5), dtype=torch.bool)
    res: dict[str, Any] = {"n_states": int(obs.shape[0])}
    roots = {f"hist_seed{k}": Path(f"results/training/madql_seed{k}") for k in (1, 2, 3)}
    roots |= {
        f"{c}_seed{k}": p4.TRAINING / f"{c}_seed{k}" for c in p4.CANDIDATES for k in (1, 2, 3)
    }
    for name, root in roots.items():
        g, gap = [], []
        for st in CHURN_STEPS:
            net, _, _ = load_inference(root / "checkpoints" / f"step_{st:08d}.pt")
            with torch.no_grad():
                q = net.eval()(obs)
            g.append(masked_argmax(q, full).numpy())
            qn = q.numpy()
            gap.append(qn[:, 0] - qn[:, 1:].max(axis=1))
        res[name] = {
            "greedy_changed_mean": float(
                np.mean([(a != b).mean() for a, b in itertools.pairwise(g)])
            ),
            "stay_move_flip_mean": float(
                np.mean([((a >= 0) != (b >= 0)).mean() for a, b in itertools.pairwise(gap)])
            ),
            "gap_abs_change_median_mean": float(
                np.mean([np.median(np.abs(a - b)) for a, b in itertools.pairwise(gap)])
            ),
            "share_stay_preferred_at_1M": float((gap[-1] >= 0).mean()),
        }
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    recs = records(args.out)
    comp = completeness(recs)
    if not comp["complete"] or comp["errors"]:
        raise SystemExit(f"selection data incomplete or with errors: {comp}")
    table = {
        a: {c: summary([r for r in recs if r["arm"] == f"{a}:{c}"]) for c in ("FA", "NA")}
        for a in p4.STAGES["4s"]["arms"]
    }
    per_policy = {
        f"{a}:{c}:{run}": summary([r for r in recs if r["arm"] == f"{a}:{c}" and r["run"] == run])
        for a in p4.STAGES["4s"]["arms"]
        for c in ("FA", "NA")
        for run in p4.RUNS
    }
    sel = select(table)
    sel["completeness"] = comp
    sel["records_sha256"] = p3.sha256(args.out / "records_4s.jsonl")
    sel["protocol_sha256"] = p3.sha256(p4.PROTOCOL)
    (args.out / "selection.json").write_text(json.dumps(sel, indent=1), "utf-8")
    report = {
        "selection": sel,
        "table": table,
        "per_policy": per_policy,
        "training": training_logs(),
        "churn": churn(),
    }
    (args.out / "selection_report.json").write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps(sel, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
