"""Issue 6A.0 — seed-cluster re-analysis of the Issue #5/#6 pooled block-B statistics (read only).

For one scenario seed the three levels share map, robot starts and tasks, so the independent
unit is the seed. The historical analyses resampled (seed, level) instances. This tool:

1. recomputes every historical pooled estimate with the *original* instance bootstrap and checks
   it against the stored ``analysis.json`` (so differences below are due to the unit only);
2. recomputes it with a seed-cluster bootstrap (all levels of a sampled seed stay together;
   same 10 000 resamples, seed 0, percentile CI);
3. recomputes the Wilcoxon p-values on per-seed mean differences and Holm within the original
   families;
4. adds per-level estimates (seeds within one level are independent).

Shield rows use the full level only (one instance per seed) and are unaffected by construction.
Historical files are only read; output goes to ``--out``.

Usage::

    python tools/diagnostics/cluster_reanalysis.py --out results/dev6a/6a0_diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.experiments.stats import holm, paired_bootstrap, wilcoxon_p
from robocoord.marl import study
from robocoord.marl.devcompare import (
    algorithm_bootstrap,
    load_records,
    metric_value,
    two_level_bootstrap,
)

DEV5 = Path("results/dev5/comparison")
DEV6 = Path("results/dev6")
B, SEED = 10_000, 0
LEVELS = ("static", "obstacles", "full")
Key = tuple[str, str, int]


def by_label(records: Sequence[dict[str, Any]]) -> dict[str, dict[Key, dict[str, Any]]]:
    out: dict[str, dict[Key, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        out[r["label"]][(r["planner"], r["level"], int(r["seed"]))] = r
    return out


def pair(
    a: dict[Key, Any], b: dict[Key, Any], metric: str, planner: str
) -> tuple[list[Key], np.ndarray]:
    keys, d = [], []
    for k in sorted(k for k in a if k[0] == planner and k in b):
        va, vb = metric_value(a[k], metric), metric_value(b[k], metric)
        if va is None or vb is None:
            continue
        keys.append(k)
        d.append(va - vb)
    return keys, np.asarray(d, dtype=np.float64)


def seed_index(keys: Sequence[Key]) -> tuple[np.ndarray, int]:
    seeds = sorted({k[2] for k in keys})
    pos = {s: i for i, s in enumerate(seeds)}
    return np.array([pos[k[2]] for k in keys]), len(seeds)


def cluster_ci(keys: Sequence[Key], d: np.ndarray) -> tuple[float, float, float]:
    idx, s = seed_index(keys)
    sums = np.bincount(idx, weights=d, minlength=s)
    cnt = np.bincount(idx, minlength=s).astype(np.float64)
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, s, (B, s))
    boots = sums[pick].sum(1) / cnt[pick].sum(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def per_seed(keys: Sequence[Key], d: np.ndarray) -> np.ndarray:
    idx, s = seed_index(keys)
    return np.bincount(idx, weights=d, minlength=s) / np.bincount(idx, minlength=s)


def cluster_algorithm(
    x: np.ndarray, y: np.ndarray, keys: Sequence[Key]
) -> tuple[float, float, float]:
    idx, s = seed_index(keys)
    cnt = np.bincount(idx, minlength=s).astype(np.float64)
    sx = np.stack([np.bincount(idx, weights=row, minlength=s) for row in x])
    sy = np.stack([np.bincount(idx, weights=row, minlength=s) for row in y])
    rng = np.random.default_rng(SEED)
    boots = np.empty(B)
    for b in range(B):
        inst = rng.integers(0, s, s)
        px = rng.integers(0, x.shape[0], x.shape[0])
        py = rng.integers(0, y.shape[0], y.shape[0])
        n = cnt[inst].sum()
        boots[b] = sx[np.ix_(px, inst)].sum() / (len(px) * n) - sy[np.ix_(py, inst)].sum() / (
            len(py) * n
        )
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(x.mean() - y.mean()), float(lo), float(hi)


def cluster_two_level(diffs: Sequence[tuple[list[Key], np.ndarray]]) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    prepared = []
    for keys, d in diffs:
        idx, s = seed_index(keys)
        prepared.append(
            (
                np.bincount(idx, weights=d, minlength=s),
                np.bincount(idx, minlength=s).astype(float),
                s,
            )
        )
    boots = np.empty(B)
    for b in range(B):
        chosen = rng.integers(0, len(prepared), len(prepared))
        vals = []
        for k in chosen:
            sums, cnt, s = prepared[k]
            pick = rng.integers(0, s, s)
            vals.append(sums[pick].sum() / cnt[pick].sum())
        boots[b] = np.mean(vals)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean([d.mean() for _, d in diffs])), float(lo), float(hi)


def row(section: str, family: str, name: str, metric: str, planner: str, keys: list[Key], d: np.ndarray,
        stored: dict[str, Any] | None) -> dict[str, Any]:  # fmt: skip
    old = paired_bootstrap(d, np.zeros_like(d))
    new = cluster_ci(keys, d)
    ps = per_seed(keys, d)
    levels = {}
    for lvl in LEVELS:
        sel = [i for i, k in enumerate(keys) if k[1] == lvl]
        if len(sel) >= 2:
            kk = [keys[i] for i in sel]
            levels[lvl] = dict(
                zip(("mean", "low", "high"), cluster_ci(kk, d[sel]), strict=True)
            ) | {"n_seeds": len(sel)}
    out = {
        "section": section, "family": family, "comparison": name, "metric": metric, "planner": planner,
        "instances": len(d), "seeds": len(ps), "mean": float(d.mean()),
        "old_low": old.low, "old_high": old.high, "new_low": new[1], "new_high": new[2],
        "width_ratio": (new[2] - new[1]) / (old.high - old.low) if old.high > old.low else None,
        "old_p": wilcoxon_p(d, np.zeros_like(d)), "new_p": wilcoxon_p(ps, np.zeros_like(ps)),
        "per_level": levels,
    }  # fmt: skip
    if stored is not None:
        out["replicates_stored"] = bool(
            np.isclose(stored["ci_low"], old.low) and np.isclose(stored["ci_high"], old.high)
            and np.isclose(stored["mean_diff"], out["mean"])
        )  # fmt: skip
        out["stored_holm_p"] = stored.get("holm_p")
    return out


def finish_families(rows: list[dict[str, Any]]) -> None:
    fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "old_p" in r:
            fam[r["section"] + "|" + r["family"]].append(r)
    for rs in fam.values():
        for r, a, b in zip(
            rs, holm([x["old_p"] for x in rs]), holm([x["new_p"] for x in rs]), strict=True
        ):
            r["old_holm_p"], r["new_holm_p"] = a, b


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    target = args.out / "cluster_reanalysis.json"
    if target.exists():
        raise SystemExit(f"{target} exists; written once")
    rows: list[dict[str, Any]] = []
    alg: list[dict[str, Any]] = []
    # ---------------- Issue #5 development comparison (dev5/comparison/analysis.json)
    hist = load_records(DEV5 / "records.jsonl")
    a5 = json.loads((DEV5 / "analysis.json").read_text("utf-8"))
    g5 = by_label(hist)
    for s in a5["paired"]:
        keys, d = pair(g5[s["policy"]], g5[s["comparator"]], s["metric"], s["planner"])
        rows.append(
            row(
                "issue5",
                s["metric"],
                f"{s['policy']} − {s['comparator']}",
                s["metric"],
                s["planner"],
                keys,
                d,
                s,
            )
        )
    for s in a5["pooled"]:
        pols = sorted(p for p in g5 if p.startswith("madql"))
        diffs = [pair(g5[p], g5[s["comparator"]], s["metric"], s["planner"]) for p in pols]
        old = two_level_bootstrap([d for _, d in diffs])
        new = cluster_two_level(diffs)
        alg.append({"section": "issue5_pooled", "comparison": f"MADQL − {s['comparator']}", "metric": s["metric"],
                    "planner": s["planner"], "mean": new[0], "old_low": old["low"], "old_high": old["high"],
                    "new_low": new[1], "new_high": new[2],
                    "replicates_stored": bool(np.isclose(old["low"], s["low"]) and np.isclose(old["high"], s["high"])),
                    "width_ratio": (new[2] - new[1]) / (old["high"] - old["low"])})  # fmt: skip
    # ---------------- Issue #6 primary (dev6/analysis.json)
    a6 = json.loads((DEV6 / "analysis.json").read_text("utf-8"))
    prim = [
        *hist,
        *load_records(DEV6 / "baseline_matched" / "records.jsonl"),
        *load_records(DEV6 / "ppo" / "records.jsonl"),
    ]
    g6 = by_label(prim)
    for s in a6["primary"]["per_policy"]["paired"]:
        keys, d = pair(g6[s["policy"]], g6[s["comparator"]], s["metric"], s["planner"])
        rows.append(
            row(
                "issue6_primary",
                s["metric"],
                f"{s['policy']} − {s['comparator']}",
                s["metric"],
                s["planner"],
                keys,
                d,
                s,
            )
        )
    for s in a6["primary"]["algorithm_level"]:
        la = sorted(p for p in g6 if p.split("-seed")[0] == s["a"])
        lb = sorted(p for p in g6 if p.split("-seed")[0] == s["b"])
        pols = la + lb
        keys = sorted(set.intersection(*({k for k in g6[p] if k[0] == s["planner"]} for p in pols)))
        keys = [
            k for k in keys if all(metric_value(g6[p][k], s["metric"]) is not None for p in pols)
        ]
        x = np.array([[metric_value(g6[p][k], s["metric"]) for k in keys] for p in la], dtype=float)
        y = np.array([[metric_value(g6[p][k], s["metric"]) for k in keys] for p in lb], dtype=float)
        old = algorithm_bootstrap(x, y)
        new = cluster_algorithm(x, y, keys)
        alg.append({"section": "issue6_algorithm", "comparison": f"{s['a']} − {s['b']}", "metric": s["metric"],
                    "planner": s["planner"], "mean": new[0], "old_low": old["low"], "old_high": old["high"],
                    "new_low": new[1], "new_high": new[2],
                    "replicates_stored": bool(np.isclose(old["low"], s["low"]) and np.isclose(old["high"], s["high"])),
                    "width_ratio": (new[2] - new[1]) / (old["high"] - old["low"])})  # fmt: skip
    # ---------------- resolver effect, ablation, stochastic (paired rows stored with x / y labels)
    abl = [*load_records(DEV6 / "ablation" / "records.jsonl"),
           *[r for r in load_records(DEV6 / "ppo" / "records.jsonl") if r["label"] in ("mappo-seed1", "mappo-seed2")
             and r["planner"] == "astar"]]  # fmt: skip
    gab = by_label(abl)
    for s in a6["ablation"]["paired"]:
        keys, d = pair(gab[s["x"]], gab[s["y"]], s["metric"], s["planner"])
        rows.append(
            row(
                "issue6_ablation",
                s["metric"],
                s["comparison"],
                s["metric"],
                s["planner"],
                keys,
                d,
                s,
            )
        )
    for s in a6["ablation"]["cell_vs_default"]:
        labs = [f"{s['cell']}-seed1", f"{s['cell']}-seed2", "mappo-seed1", "mappo-seed2"]
        keys = sorted(set.intersection(*(set(gab[lab]) for lab in labs)))
        keys = [
            k
            for k in keys
            if all(metric_value(gab[lab][k], s["metric"]) is not None for lab in labs)
        ]
        x = np.array(
            [[metric_value(gab[lab][k], s["metric"]) for k in keys] for lab in labs[:2]],
            dtype=float,
        )
        y = np.array(
            [[metric_value(gab[lab][k], s["metric"]) for k in keys] for lab in labs[2:]],
            dtype=float,
        )
        old = algorithm_bootstrap(x, y)
        new = cluster_algorithm(x, y, keys)
        alg.append({"section": "issue6_ablation_cell", "comparison": f"{s['cell']} − mappo", "metric": s["metric"],
                    "planner": "astar", "mean": new[0], "old_low": old["low"], "old_high": old["high"],
                    "new_low": new[1], "new_high": new[2],
                    "replicates_stored": bool(np.isclose(old["low"], s["low"]) and np.isclose(old["high"], s["high"])),
                    "width_ratio": (new[2] - new[1]) / (old["high"] - old["low"])})  # fmt: skip
    gst = study.stochastic_groups(load_records(DEV6 / "stochastic" / "records.jsonl"))
    for s in a6["stochastic"]["paired"]:
        keys, d = pair(gst[s["x"]], gst[s["y"]], s["metric"], s["planner"])
        rows.append(
            row(
                "issue6_stochastic",
                s["metric"],
                s["comparison"],
                s["metric"],
                s["planner"],
                keys,
                d,
                s,
            )
        )
    for s in a6["resolver_effect"]:
        keys, d = pair(g6["baseline"], g6["baseline-noresolver"], s["metric"], s["planner"])
        rows.append(
            row(
                "issue6_resolver",
                s["metric"],
                s["comparison"],
                s["metric"],
                s["planner"],
                keys,
                d,
                s,
            )
        )
    finish_families(rows)
    for r in rows:
        r["old_ci_excludes_0"] = not (r["old_low"] <= 0 <= r["old_high"])
        r["new_ci_excludes_0"] = not (r["new_low"] <= 0 <= r["new_high"])
        r["old_holm_sig"] = r["old_holm_p"] < 0.05
        r["new_holm_sig"] = r["new_holm_p"] < 0.05
    for r in alg:
        r["old_ci_excludes_0"] = not (r["old_low"] <= 0 <= r["old_high"])
        r["new_ci_excludes_0"] = not (r["new_low"] <= 0 <= r["new_high"])
    args.out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"paired": rows, "pooled": alg}, indent=1), "utf-8")
    print("wrote", target, len(rows), "paired rows,", len(alg), "pooled rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
