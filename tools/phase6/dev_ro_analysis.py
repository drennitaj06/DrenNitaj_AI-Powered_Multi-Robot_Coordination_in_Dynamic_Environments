"""Issue 6A Phase 6 decision review — robot–obstacle (RO) conflicts per move on **stored
development records only** (Blocks C and S). Never confirmatory evidence; runs no episode and
generates no scenario.

Move denominator: the non-idle robot-steps whose *requested* action (the supervisor's output, or
the policy's without a supervisor; the shield is off, so this is the arbiter's input) is not STAY:
``planning.active_robot_steps × (1 − wait_ratio)``. ``active_robot_steps`` equals the non-idle
robot-steps (checked on all 5 040 Phase 4 records), and ``wait_ratio`` = requested STAYs / non-idle
steps (``sim.metrics``). Idle robots' moves are excluded (the Phase 6 observer counts them too).

Estimator: the frozen ``block_c_analysis`` bootstrap-t through ``phase3_analysis.infer`` (ratio of
sums, seed cluster, B = 10 000, ``default_rng(0)``); intervals are nominal 95 %, unadjusted.

Usage::

    python tools/phase6/dev_ro_analysis.py --out results/dev6a/phase6/dev_ro
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
for _sub in ("phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))
import block_c_analysis as bca  # noqa: E402
import phase3_analysis as pa  # noqa: E402

LEVELS = ("static", "obstacles", "full")
P1A = Path("results/dev6a/phase1/eval/records_1a.jsonl")
P1B = Path("results/dev6a/phase1/eval/records_1b.jsonl")
P4C = Path("results/dev6a/phase4/eval/records_4c.jsonl")
P4S = Path("results/dev6a/phase4/select/records_4s.jsonl")
MADQL = ("madql_seed1", "madql_seed2", "madql_seed3")
IPPO = ("ippo_seed1", "ippo_seed2", "ippo_seed3")
MAPPO = ("mappo_seed1", "mappo_seed2", "mappo_seed3")


def moves(r: Mapping[str, Any]) -> float:
    a = r["planning"]["active_robot_steps"]
    return float(a - round(a * r["wait_ratio"])) if r["wait_ratio"] is not None else 0.0


def ro(r: Mapping[str, Any]) -> float:
    return float(r["conflicts"]["ROv"] + r["conflicts"]["ROs"])


def load(path: Path, keep: Callable[[Mapping[str, Any]], bool]) -> dict[tuple[str, ...], Any]:
    out: dict[tuple[str, ...], Any] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if keep(r):
                out[(r["arm"], r["run"], r["level"], str(r["seed"]))] = r
    return out


def arrays(
    recs: Mapping[tuple[str, ...], Any],
    old: tuple[str, Sequence[str]],
    new: tuple[str, Sequence[str]],
    seeds: Sequence[int],
    f: Callable[[Mapping[str, Any]], tuple[float, float]],
    levels: Sequence[str] = LEVELS,
) -> list[Any]:
    """Seed-level sums Y_old, X_old, Y_new, X_new (old/new: (arm, runs))."""
    out = np.zeros((4, len(seeds)))
    for j, s in enumerate(seeds):
        for k, (arm, runs) in enumerate((old, new)):
            for run in runs:
                for lv in levels:
                    y, x = f(recs[(arm, run, lv, str(s))])
                    out[2 * k, j] += y
                    out[2 * k + 1, j] += x
    return [out[0], out[1], out[2], out[3]]


def per_move(r: Mapping[str, Any]) -> tuple[float, float]:
    return 1000.0 * ro(r), moves(r)


def per_episode(r: Mapping[str, Any]) -> tuple[float, float]:
    return ro(r), 1.0


def moves_per_episode(r: Mapping[str, Any]) -> tuple[float, float]:
    return moves(r), 1.0


def summarise(a: list[Any], idx: Any) -> dict[str, Any]:
    inf = pa.infer(a, idx)
    yo, xo, yn, xn = (float(v.sum()) for v in a)
    d = (a[2] / np.maximum(a[3], 1) - a[0] / np.maximum(a[1], 1)) if a[1].sum() > 1 else None
    res = {
        "old_rate": yo / xo if xo else None,
        "new_rate": yn / xn if xn else None,
        "ratio": (yn / xn) / (yo / xo) if xo and yo else None,
        "estimate": inf["estimate"] if inf else None,
        "se": inf["se"] if inf else None,
        "ci95": inf["ci95"] if inf else None,
        "p_nominal": inf["p"] if inf else None,
    }
    if d is not None:
        res["seeds_new_higher"] = int((d > 0).sum())
        res["seeds_new_lower"] = int((d < 0).sum())
        res["seeds"] = len(d)
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    blk_c, blk_cr, blk_s = range(11_000, 11_120), range(11_000, 11_040), range(13_000, 13_080)
    idx = {n: bca.boot_index(n) for n in (120, 40, 80)}
    res: dict[str, Any] = {
        "note": "development data (Blocks C and S); nominal; never confirmatory",
        "denominator": "non-idle requested moves = active_robot_steps × (1 − wait_ratio)",
    }
    p1a = load(P1A, lambda r: r["arm"] in ("off_r0", "rsd_r0"))
    p1b = load(P1B, lambda r: r["arm"] in ("off_r0", "rsd_r0"))
    p4c = load(P4C, lambda r: True)
    p4s = load(P4S, lambda r: r["arm"].split(":")[0] in ("hist", "n3"))

    # H-v2: v2 vs hist, per condition (Block C) and the Block S replication
    hv2: dict[str, Any] = {}
    for cond, seeds, n in (("FA", blk_c, 120), ("FR", blk_cr, 40), ("NA", blk_c, 120)):
        o, nw = (f"hist:{cond}", MADQL), (f"v2:{cond}", MADQL)
        hv2[cond] = {
            "per_move": summarise(arrays(p4c, o, nw, seeds, per_move), idx[n]),
            "per_episode": summarise(arrays(p4c, o, nw, seeds, per_episode), idx[n]),
            "moves_per_episode": summarise(arrays(p4c, o, nw, seeds, moves_per_episode), idx[n]),
            "per_run": {
                run: summarise(
                    arrays(p4c, (o[0], (run,)), (nw[0], (run,)), seeds, per_move), idx[n]
                )
                for run in MADQL
            },
            "per_level": {
                lv: summarise(arrays(p4c, o, nw, seeds, per_move, (lv,)), idx[n])
                for lv in ("obstacles", "full")
            },
        }
    res["H-v2_block_C"] = hv2
    res["H-v2_block_S"] = {
        cond: {
            "per_move": summarise(
                arrays(p4s, (f"hist:{cond}", MADQL), (f"n3:{cond}", MADQL), blk_s, per_move),
                idx[80],
            ),
            "per_run": {
                run: summarise(
                    arrays(p4s, (f"hist:{cond}", (run,)), (f"n3:{cond}", (run,)), blk_s, per_move),
                    idx[80],
                )
                for run in MADQL
            },
        }
        for cond in ("FA", "NA")
    }

    # H-L2: supervisor OFF vs R + S + D, the 9 Phase 3 policies (Phase 1 records)
    hl2: dict[str, Any] = {}
    for planner, recs, seeds, n in (("astar", p1a, blk_c, 120), ("rrtstar", p1b, blk_cr, 40)):
        hl2[planner] = {
            fam: summarise(
                arrays(recs, ("off_r0", runs), ("rsd_r0", runs), seeds, per_move), idx[n]
            )
            for fam, runs in (("madql", MADQL), ("ippo", IPPO), ("mappo", MAPPO))
        }
        hl2[planner]["per_policy"] = {
            run: summarise(
                arrays(recs, ("off_r0", (run,)), ("rsd_r0", (run,)), seeds, per_move), idx[n]
            )
            for run in (*MADQL, *IPPO, *MAPPO)
        }
    res["H-L2_block_C"] = hl2

    # H-L12 (MADQL part, Block C): hist OFF vs v2 R + S + D
    res["H-L12_madql_block_C"] = {
        "astar": summarise(
            arrays(p4c, ("hist:NA", MADQL), ("v2:FA", MADQL), blk_c, per_move), idx[120]
        ),
    }

    # PPO context: rate levels, and v2 vs the forward PPO policies on the same instances
    def rate(rs: Sequence[Mapping[str, Any]]) -> float:
        ys = [per_move(r) for r in rs]
        return sum(y for y, _ in ys) / sum(x for _, x in ys)

    rates: dict[str, Any] = defaultdict(dict)
    for planner, recs in (("astar", p1a), ("rrtstar", p1b)):
        for fam, runs in (("ippo", IPPO), ("mappo", MAPPO), ("madql_hist", MADQL)):
            for arm in ("off_r0", "rsd_r0"):
                rates[planner][f"{fam}|{arm}"] = rate(
                    [r for k, r in recs.items() if k[0] == arm and k[1] in runs]
                )
    for planner, cond in (("astar", "FA"), ("rrtstar", "FR"), ("astar", "NA")):
        rates[planner][f"madql_v2|{cond}"] = rate(
            [r for k, r in p4c.items() if k[0] == f"v2:{cond}"]
        )
    res["rates_per_1000_moves"] = rates
    merged = {**{("ppo", *k[1:]): r for k, r in p1a.items() if k[0] == "rsd_r0"
                 and not k[1].startswith("madql")},
              **{("v2", *k[1:]): r for k, r in p4c.items() if k[0] == "v2:FA"}}  # fmt: skip
    res["v2_vs_forward_ppo_astar_rsd"] = summarise(
        arrays(merged, ("ppo", (*IPPO, *MAPPO)), ("v2", MADQL), blk_c, per_move), idx[120]
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dev_ro_analysis.json").write_text(
        json.dumps(res, indent=1, default=float), "utf-8"
    )
    print(json.dumps(res, indent=1, default=float)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
