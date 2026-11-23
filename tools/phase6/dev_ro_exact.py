"""Issue 6A Phase 6 decision review — exact robot–obstacle (RO) rates per move on **Block C
development data** (never confirmatory). Re-runs Block C episodes through the Phase 6 runner and
observer (Block C guard only; no other block can be generated) so that the RO-per-move outcome is
measured with the exact definitions the Phase 6 protocol could freeze:

* **(i) all robots:** RO events on K^arb / every move proposed to the arbiter (any robot);
* **(ii) non-idle robots:** RO events of non-idle robots / moves of non-idle robots.

Cells: historical MADQL and MADQL-v2 n3 (A\\* OFF and R + S + D; RRT\\* R + S + D) and the SCB-C v5
PPO policies (A\\* OFF and R + S + D): the cells of H-v2 and the A\\* cells of H-L2. Every re-run
record is also compared with its stored Phase 1 / Phase 4 counterpart (shared non-timing fields).

Usage::

    python tools/phase6/dev_ro_exact.py run --out results/dev6a/phase6/dev_ro_exact --workers 10
    python tools/phase6/dev_ro_exact.py analyse --out results/dev6a/phase6/dev_ro_exact
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sub in ("phase4", "phase2", "phase1", "phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))
import block_c_analysis as bca  # noqa: E402
import phase1 as p1  # noqa: E402
import phase3 as p3  # noqa: E402
import phase3_analysis as pa  # noqa: E402
import phase6 as p6  # noqa: E402

CELLS: tuple[tuple[str, str, str], ...] = (
    ("hist", "OFF", "astar"), ("hist", "RSD", "astar"), ("v2", "OFF", "astar"),
    ("v2", "RSD", "astar"), ("hist", "RSD", "rrtstar"), ("v2", "RSD", "rrtstar"),
    ("v5", "OFF", "astar"), ("v5", "RSD", "astar"),
)  # fmt: skip
STORED = {  # (policy, cond, planner) -> (file, arm) of the stored counterpart
    ("hist", "OFF", "astar"): ("results/dev6a/phase1/eval/records_1a.jsonl", "off_r0"),
    ("hist", "RSD", "astar"): ("results/dev6a/phase1/eval/records_1a.jsonl", "rsd_r0"),
    ("v5", "OFF", "astar"): ("results/dev6a/phase1/eval/records_1a.jsonl", "off_r0"),
    ("v5", "RSD", "astar"): ("results/dev6a/phase1/eval/records_1a.jsonl", "rsd_r0"),
    ("hist", "RSD", "rrtstar"): ("results/dev6a/phase1/eval/records_1b.jsonl", "rsd_r0"),
    ("v2", "OFF", "astar"): ("results/dev6a/phase4/eval/records_4c.jsonl", "v2:NA"),
    ("v2", "RSD", "astar"): ("results/dev6a/phase4/eval/records_4c.jsonl", "v2:FA"),
    ("v2", "RSD", "rrtstar"): ("results/dev6a/phase4/eval/records_4c.jsonl", "v2:FR"),
}


def jobs(ck: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for pol, cond, planner in CELLS:
        seeds = p3.BLOCK_C if planner == "astar" else p3.BLOCK_C_RRT
        for run in p6.POLICY_RUNS[pol]:
            info = ck["policies"][pol][run]
            for lv in p6.LEVELS:
                for s in seeds:
                    out.append({"stage": "devro", "arm": f"{pol}:{cond}", "policy": pol,
                                "cond": cond, "run": run, "planner": planner, "level": lv,
                                "seed": s, "checkpoint": info["path"],
                                "checkpoint_sha256": info["sha256"]})  # fmt: skip
    return sorted(out, key=lambda j: (j["planner"] != "rrtstar", j["seed"]))


def run(out: Path, workers: int) -> int:
    ck = p6.checkpoints()
    if not ck["ok"]:
        raise SystemExit("checkpoint verification failed")
    target = out / "records.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    done = {p6.key(r) for r in p1._records(target)}
    todo = [j for j in jobs(ck) if p6.key(j) not in done]
    t0 = time.time()
    with (
        target.open("a", encoding="utf-8") as fh,
        ProcessPoolExecutor(workers, initializer=p6._init_worker, initargs=("c", True)) as ex,
    ):
        for n, rec in enumerate(ex.map(p6._run_anchor, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + "\n")
            if n % 500 == 0:
                fh.flush()
                print(f"devro: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    return 0


def _rate_all(r: Mapping[str, Any]) -> tuple[float, float]:
    k = r["p6"]["k_arb"]
    return 1000.0 * (k["ROv"] + k["ROs"]), float(r["p6"]["moves_arb"])


def _rate_nonidle(r: Mapping[str, Any]) -> tuple[float, float]:
    return 1000.0 * r["p6"]["ro_arb_nonidle"], float(r["p6"]["moves_arb_nonidle"])


DEFS: dict[str, Callable[[Mapping[str, Any]], tuple[float, float]]] = {
    "all_robots": _rate_all,
    "non_idle": _rate_nonidle,
}


def analyse(out: Path) -> dict[str, Any]:
    recs = {p6.key(r): r for r in p1._records(out / "records.jsonl")}
    errors = sum(r["status"] != "ok" for r in recs.values())
    # reproduction of the stored records (shared non-timing fields)
    mism = 0
    compared = 0
    for (pol, cond, planner), (path, arm) in STORED.items():
        for r in p1._records(Path(path)):
            if r.get("arm") != arm or r["planner"] != planner:
                continue
            if r["run"] not in p6.POLICY_RUNS[pol]:
                continue
            mine = recs.get((f"{pol}:{cond}", r["run"], planner, r["level"], int(r["seed"])))
            if mine is None:
                continue
            compared += 1
            shared = {k for k in r if k in mine} - p6.ANCHOR_SKIP
            mism += any(r[k] != mine[k] for k in shared)

    def arrays(old: str, new: str, runs_old: Sequence[str], runs_new: Sequence[str],
               planner: str, f: Callable[[Mapping[str, Any]], tuple[float, float]]) -> list[Any]:  # fmt: skip
        seeds = list(p3.BLOCK_C if planner == "astar" else p3.BLOCK_C_RRT)
        a = np.zeros((4, len(seeds)))
        for j, s in enumerate(seeds):
            for k, (arm, runs) in enumerate(((old, runs_old), (new, runs_new))):
                for run in runs:
                    for lv in p6.LEVELS:
                        y, x = f(recs[(arm, run, planner, lv, s)])
                        a[2 * k, j] += y
                        a[2 * k + 1, j] += x
        return [a[0], a[1], a[2], a[3]]

    def summ(a: list[Any], n: int) -> dict[str, Any]:
        inf = pa.infer(a, bca.boot_index(n))
        assert inf is not None
        return {"old_rate": float(a[0].sum() / a[1].sum()), "new_rate": float(a[2].sum() / a[3].sum()),
                "estimate": inf["estimate"], "se": inf["se"], "ci95": inf["ci95"], "p_nominal": inf["p"]}  # fmt: skip

    m, ppo = p6.MADQL_RUNS, p6.PPO_RUNS
    res: dict[str, Any] = {
        "note": "Block C development data; nominal intervals; never confirmatory",
        "episodes": len(recs), "errors": errors,
        "stored_records_compared": compared, "stored_records_mismatched": mism,
        "idle_move_share": {
            arm: float(1 - sum(r["p6"]["moves_arb_nonidle"] for k, r in recs.items() if k[0] == arm)
                       / sum(r["p6"]["moves_arb"] for k, r in recs.items() if k[0] == arm))
            for arm in sorted({k[0] for k in recs})
        },
        "idle_ro_share": {
            arm: float(1 - sum(r["p6"]["ro_arb_nonidle"] for k, r in recs.items() if k[0] == arm)
                       / max(1, sum(r["p6"]["k_arb"]["ROv"] + r["p6"]["k_arb"]["ROs"]
                                    for k, r in recs.items() if k[0] == arm)))
            for arm in sorted({k[0] for k in recs})
        },
    }  # fmt: skip
    for name, f in DEFS.items():
        res[name] = {
            "H-v2": {
                "FA": summ(arrays("hist:RSD", "v2:RSD", m, m, "astar", f), 120),
                "FR": summ(arrays("hist:RSD", "v2:RSD", m, m, "rrtstar", f), 40),
                "NA": summ(arrays("hist:OFF", "v2:OFF", m, m, "astar", f), 120),
            },
            "H-L2_astar": {
                "madql": summ(arrays("hist:OFF", "hist:RSD", m, m, "astar", f), 120),
                "ippo": summ(arrays("v5:OFF", "v5:RSD", ppo[:3], ppo[:3], "astar", f), 120),
                "mappo": summ(arrays("v5:OFF", "v5:RSD", ppo[3:], ppo[3:], "astar", f), 120),
            },
            "H-L12_madql_astar": summ(arrays("hist:OFF", "v2:RSD", m, m, "astar", f), 120),
            "context_v2_vs_forward_ppo_astar_rsd": summ(
                arrays("v5:RSD", "v2:RSD", ppo, m, "astar", f), 120
            ),
        }
    (out / "dev_ro_exact.json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyse"))
    ap.add_argument("--out", type=Path, default=Path("results/dev6a/phase6/dev_ro_exact"))
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args.out, args.workers)
    res = analyse(args.out)
    print(json.dumps({k: res[k] for k in ("episodes", "errors", "stored_records_compared",
                                          "stored_records_mismatched")}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
