"""Issue 6A Phase 4 — post-hoc, descriptive analysis of the conflict types (NOT pre-registered).

Written after the Block C verdicts were known, because conflicts per episode rose nominally in
FA while the pre-registered P4 veto was not Holm-rejected. It never changes a verdict. Nominal
95 % intervals from the frozen estimator (``phase3_analysis.infer``: ratio of sums, seed cluster,
bootstrap-t, B = 10 000), no multiplicity correction.

Usage::

    python tools/phase4/posthoc_conflicts.py --out results/dev6a/phase4/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sub in ("phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))
import block_c_analysis as bca  # noqa: E402
import phase3_analysis as pa  # noqa: E402
import phase4 as p4  # noqa: E402
import phase4_analysis as p4a  # noqa: E402


def per_episode(kind: str) -> p4a.F:
    return lambda r: (float(r["conflicts"][kind]), 1.0)


def per_1000_moves(kind: str) -> p4a.F:
    def f(r: Mapping[str, Any]) -> tuple[float, float]:
        m = r["madql"]
        return 1000.0 * r["conflicts"][kind], float(m["nonidle_steps"] - m["stay"])

    return f


OUTCOMES = {
    f"{k}_{u}": fn(k)
    for k in ("ROv", "RRv")
    for u, fn in (("per_episode", per_episode), ("per_1000_moves", per_1000_moves))
}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    d = p4a.Data(args.out)
    res: dict[str, Any] = {"note": "post-hoc, descriptive, nominal; never used for a verdict",
                           "files_sha256": d.files}  # fmt: skip
    for cond in p4.STAGES["4c"]["conds"]:
        planner = p4.CONDS[cond][0]
        idx = bca.boot_index(len(d.seeds(planner)))
        o, n = f"{p4a.OLD}:{cond}", f"{p4a.NEW}:{cond}"
        res[cond] = {
            name: pa.infer(d.pair(o, n, p4.RUNS, planner, pa.simple(fn)), idx)
            for name, fn in OUTCOMES.items()
        }
    (args.out / "posthoc_conflicts.json").write_text(
        json.dumps(res, indent=1, default=float), "utf-8"
    )
    for cond in p4.STAGES["4c"]["conds"]:
        for name, inf in res[cond].items():
            print(cond, name, {k: inf[k] for k in ("estimate", "ci95") if k in inf},
                  "theta_old", inf.get("theta_old"))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
