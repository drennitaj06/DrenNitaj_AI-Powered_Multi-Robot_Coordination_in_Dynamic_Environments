"""L1 follow-up: power of the valid procedure (v2) on the real selection data, and the picks of
the candidate rules on block A, block S and block A ∪ S (informational; nothing is frozen).

Usage::

    python tools/selection/combined_power.py --out results/dev6a/l1_selection/followup
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import calibrate
import rule_compare
import scbc
import selection_block


def union(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    assert np.array_equal(a["steps"], b["steps"])
    return {k: (a[k] if k == "steps" else np.concatenate([a[k], b[k]], axis=1)) for k in a}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--s-block", type=Path, default=Path("results/dev6a/l1_selection/selection_block")
    )
    args = ap.parse_args(argv)
    s_tabs = selection_block.load(args.s_block)
    res: dict[str, Any] = {}
    for run in selection_block.RUNS:
        a = calibrate.run_arrays(run)
        tabs = {"A": a, "S": s_tabs[run], "A+S": union(a, s_tabs[run])}
        res[run] = {}
        for name, t in tabs.items():
            st, d, r, c, x = (
                t["steps"],
                t["delivered"],
                t["released"],
                t["conflicts"],
                t["collisions"],
            )
            comp, conf = d.sum(1) / r.sum(1), c.mean(1)
            row: dict[str, Any] = {"seeds": int(d.shape[1])}
            for stud in ("t", "signflip", "perm"):
                sel = scbc.select(st, d, r, c, x, alpha_completion=0.025, alpha_conflicts=0.025,
                                  n_boot=10_000, seed=0, studentize=stud)  # fmt: skip
                assert sel.completion is not None and sel.conflicts is not None
                row[stud] = {"e1": len(sel.eligible_completion), "e2": len(sel.eligible_conflicts),
                             "q1": sel.completion.q, "q2": sel.conflicts.q}  # fmt: skip
            picks = {"completion_first": int(np.lexsort((-st, conf, -comp))[0]),
                     "one_se": rule_compare.one_se_pick(t, 1.0),
                     "conflicts_first": int(np.lexsort((-st, -comp, conf))[0]),
                     "old": scbc.old_rule(st, d, r, t["ret"].mean(1))}  # fmt: skip
            row["picks"] = {k: {"step": int(st[i]), "completion": float(comp[i]),
                                "conflicts": float(conf[i])} for k, i in picks.items()}  # fmt: skip
            res[run][name] = row
    (args.out / "combined_power.json").write_text(json.dumps(res, indent=1), "utf-8")
    for run, rr in res.items():
        for name, row in rr.items():
            p = row["picks"]
            print(f"{run:12s} {name:4s} S={row['seeds']:3d} v2 e1/e2 {row['t']['e1']:2d}/{row['t']['e2']:2d} q {row['t']['q1']:.2f}/{row['t']['q2']:.2f} | v4 {row['signflip']['e1']:2d}/{row['signflip']['e2']:2d} | v3 {row['perm']['e1']:2d}/{row['perm']['e2']:2d} | "
                  + " ".join(f"{k}={v['step'] // 1000}k({v['completion']:.4f},{v['conflicts']:.1f})" for k, v in p.items()))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
