"""L1 follow-up: out-of-sample comparison of checkpoint-selection rules (read only).

Data: the six main PPO runs, 20 step checkpoints each, on block A (20 seeds, the stored
validation logs) and on the new selection block S (80 seeds, ``selection_block.py``).

1. **A → S.** Each rule selects on block A; the pick is scored on all 80 block-S seeds.
2. **Split S.** 200 random splits of the 80 block-S seeds into a selection part
   (``n_sel`` = 20 or 40 seeds) and an evaluation part (the remaining seeds); each rule selects
   on the selection part, the pick is scored on the evaluation part.

Rules (final pick among the checkpoints left by the stages): ``old`` (Issues #5/#6:
completion, then mean return), ``completion`` (highest completion), ``conflicts`` (fewest
conflicts), ``constrained`` (fewest conflicts among completion ≥ 0.99, else highest completion),
``latest``. Stages: ``none`` (no elimination) or SCB-C with ``v2`` / ``v3`` / ``v4`` at
``alpha`` 0.05 + 0.05 or 0.025 + 0.025. The 0.99 floor is the project's completion objective,
used as an engineering requirement on the point estimate; it is not an equivalence margin.

Scores of a pick on the evaluation seeds: pooled completion, mean conflicts per episode,
``meets_099`` (completion ≥ 0.99), completion shortfall from the evaluation-best checkpoint,
and conflict excess over the fewest-conflict checkpoint that meets 0.99 on the evaluation seeds.

Usage::

    python tools/selection/rule_compare.py --out results/dev6a/l1_selection/followup --workers 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import calibrate
import scbc
import selection_block

FLOOR = 0.99
RULES = ("old", "completion", "conflicts", "constrained", "latest")
STAGES = ("none", "v2", "v3", "v4")
ALPHAS = {"a05": (0.05, 0.05), "a025": (0.025, 0.025)}
STUDENTIZE = {"v2": "t", "v3": "perm", "v4": "signflip"}
N_BOOT = 2000


def sub(tab: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: (v if k == "steps" else v[:, idx]) for k, v in tab.items()}


def picks(tab: dict[str, np.ndarray], seed: int) -> dict[str, int]:
    """Every (stage, alpha, rule) pick on one selection table."""
    st, d, r, c, x = (
        tab["steps"],
        tab["delivered"],
        tab["released"],
        tab["conflicts"],
        tab["collisions"],
    )
    comp = d.sum(1) / r.sum(1)
    conf = c.mean(1)
    out: dict[str, int] = {}
    out["old"] = scbc.old_rule(st, d, r, tab["ret"].mean(1))
    excluded = scbc.validate(d, r, c, x)
    valid = [k for k in range(len(st)) if k not in excluded]
    for rule in RULES[1:]:
        loc = scbc.choose(list(range(len(valid))), st[valid], comp[valid], conf[valid], rule, FLOOR)
        out[f"none_{rule}"] = valid[loc]
    for stage, stud in STUDENTIZE.items():
        for tag, (a1, a2) in ALPHAS.items():
            sel = scbc.select(st, d, r, c, x, alpha_completion=a1, alpha_conflicts=a2,
                              n_boot=N_BOOT, seed=seed, studentize=stud)  # fmt: skip
            e2 = [valid.index(i) for i in sel.eligible_conflicts]
            for rule in RULES[1:]:
                loc = scbc.choose(e2, st[valid], comp[valid], conf[valid], rule, FLOOR)
                out[f"{stage}_{tag}_{rule}"] = valid[loc]
            out[f"{stage}_{tag}_e1"] = len(sel.eligible_completion)
            out[f"{stage}_{tag}_e2"] = len(sel.eligible_conflicts)
    return out


def score(tab: dict[str, np.ndarray], pick: dict[str, int]) -> dict[str, Any]:
    comp = tab["delivered"].sum(1) / tab["released"].sum(1)
    conf = tab["conflicts"].mean(1)
    feas = comp >= FLOOR
    ref = conf[feas].min() if feas.any() else np.nan
    out: dict[str, Any] = {}
    for name, i in pick.items():
        if name.endswith(("_e1", "_e2")):
            out[name] = i
            continue
        out[name] = {
            "completion": float(comp[i]),
            "conflicts": float(conf[i]),
            "meets_099": bool(comp[i] >= FLOOR),
            "shortfall": float(comp.max() - comp[i]),
            "conf_excess_vs_feasible_min": float(conf[i] - ref) if feas.any() else None,
            "step": float(tab["steps"][i]),
        }
    return out


def split_job(args: tuple[str, dict[str, np.ndarray], int, list[int]]) -> list[dict[str, Any]]:
    run, tab, n_sel, reps = args
    n = tab["delivered"].shape[1]
    rows = []
    for rep in reps:
        rng = np.random.default_rng([n_sel, rep, sum(map(ord, run))])
        perm = rng.permutation(n)
        a, b = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
        rows.append(
            {"run": run, "n_sel": n_sel, "rep": rep} | score(sub(tab, b), picks(sub(tab, a), rep))
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [k for k, v in rows[0].items() if isinstance(v, dict)]
    out: dict[str, Any] = {}
    for nm in names:
        vals = [r[nm] for r in rows]
        ex = [
            v["conf_excess_vs_feasible_min"]
            for v in vals
            if v["conf_excess_vs_feasible_min"] is not None
        ]
        out[nm] = {
            "completion": float(np.mean([v["completion"] for v in vals])),
            "conflicts": float(np.mean([v["conflicts"] for v in vals])),
            "conflicts_median": float(np.median([v["conflicts"] for v in vals])),
            "meets_099": float(np.mean([v["meets_099"] for v in vals])),
            "shortfall": float(np.mean([v["shortfall"] for v in vals])),
            "shortfall_gt_0005": float(np.mean([v["shortfall"] > 0.005 for v in vals])),
            "conf_excess": float(np.mean(ex)) if ex else None,
            "conf_excess_median": float(np.median(ex)) if ex else None,
        }
    for k in [k for k in rows[0] if k.endswith(("_e1", "_e2"))]:
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


TAUS = (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 1.0)


def tolerance_picks(tab: dict[str, np.ndarray]) -> dict[str, int]:
    """Fewest conflicts among checkpoints whose completion estimate is within ``tau`` of the best
    estimate (``tau`` = 0: highest completion, ties to fewer conflicts; ``tau`` = 1: fewest
    conflicts). A policy trade-off curve, not an inference: ``tau`` is the completion the
    selector is willing to give up, on the point estimate, for fewer conflicts."""
    comp = tab["delivered"].sum(1) / tab["released"].sum(1)
    conf = tab["conflicts"].mean(1)
    out = {}
    for tau in TAUS:
        ok = np.flatnonzero(comp >= comp.max() - tau - 1e-12)
        out[f"tau_{tau}"] = int(ok[np.lexsort((-tab["steps"][ok], -comp[ok], conf[ok]))[0]])
    for mult in (1.0, 2.0):
        out[f"one_se_{mult}"] = one_se_pick(tab, mult)
    return out


def one_se_pick(tab: dict[str, np.ndarray], mult: float = 1.0) -> int:
    """One-standard-error rule (Breiman et al. 1984): ``j`` = the highest completion estimate;
    a checkpoint ``k`` is in the band when ``comp[j] − comp[k] ≤ mult · SE_jk``, with ``SE_jk``
    the standard error of the paired per-seed completion difference on the selection seeds;
    the pick is the fewest conflicts in the band (then higher completion, then later step).

    Engineering rationale (§22.7): inside one SE of the best, the selection data cannot order
    completion (out-of-sample rank correlation 0.14–0.18), but they do order conflicts
    (0.57–0.64). It is not an equivalence test: no claim is made that band members are equal."""
    d, r = tab["delivered"], tab["released"]
    comp = d.sum(1) / r.sum(1)
    conf = tab["conflicts"].mean(1)
    x = d / r.mean(1, keepdims=True)
    j = int(np.lexsort((-tab["steps"], conf, -comp))[0])
    se = (x[j][None, :] - x).std(axis=1, ddof=1) / np.sqrt(x.shape[1])
    ok = np.flatnonzero(comp[j] - comp <= mult * se + 1e-12)
    return int(ok[np.lexsort((-tab["steps"][ok], -comp[ok], conf[ok]))[0]])


def tolerance_job(args: tuple[str, dict[str, np.ndarray], int, int]) -> list[dict[str, Any]]:
    run, tab, n_sel, splits = args
    n = tab["delivered"].shape[1]
    rows = []
    for rep in range(splits):
        rng = np.random.default_rng([n_sel, rep, sum(map(ord, run))])  # the same splits as above
        perm = rng.permutation(n)
        a, b = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
        rows.append({"run": run, "n_sel": n_sel} | score(sub(tab, b), tolerance_picks(sub(tab, a))))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--s-block", type=Path, default=Path("results/dev6a/l1_selection/selection_block")
    )
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    s_tabs = selection_block.load(args.s_block)
    a_tabs = {run: calibrate.run_arrays(run) for run in selection_block.RUNS}
    res: dict[str, Any] = {"floor": FLOOR, "n_boot": N_BOOT}
    # 1. A -> S
    a2s = {}
    for run in selection_block.RUNS:
        assert np.array_equal(a_tabs[run]["steps"], s_tabs[run]["steps"]), run
        a2s[run] = score(s_tabs[run], picks(a_tabs[run], 0))
    res["a_to_s"] = {"per_run": a2s, "mean": aggregate(list(a2s.values()))}
    # per-run block-S description
    res["block_s_runs"] = {
        run: {
            "steps": t["steps"].tolist(),
            "completion": (t["delivered"].sum(1) / t["released"].sum(1)).tolist(),
            "conflicts": t["conflicts"].mean(1).tolist(),
            "collisions": t["collisions"].sum(1).tolist(),
        }
        for run, t in s_tabs.items()
    }
    # 2. splits within S
    jobs = []
    for n_sel in (20, 40):
        for run in selection_block.RUNS:
            for lo in range(0, args.splits, 20):
                jobs.append((run, s_tabs[run], n_sel, list(range(lo, min(lo + 20, args.splits)))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for chunk in pool.map(split_job, jobs) for r in chunk]
    res["splits"] = {}
    for n_sel in (20, 40):
        rs = [r for r in rows if r["n_sel"] == n_sel]
        res["splits"][f"n_sel_{n_sel}"] = {
            "all_runs": aggregate(rs),
            "per_run": {
                run: aggregate([r for r in rs if r["run"] == run]) for run in selection_block.RUNS
            },
        }
    # 3. tolerance curve (point-estimate rule, no inference), A -> S and splits
    tol: dict[str, Any] = {"taus": list(TAUS)}
    tol["a_to_s"] = aggregate(
        [score(s_tabs[run], tolerance_picks(a_tabs[run])) for run in selection_block.RUNS]
    )
    for n_sel in (20, 40):
        trs = [r for run in selection_block.RUNS
               for r in tolerance_job((run, s_tabs[run], n_sel, args.splits))]  # fmt: skip
        tol[f"n_sel_{n_sel}"] = aggregate(trs)
    res["tolerance_curve"] = tol
    res["wall_s"] = time.perf_counter() - t0
    (args.out / "rule_compare.json").write_text(json.dumps(res, indent=1), "utf-8")
    with open(args.out / "rule_compare_splits.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"done in {res['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
