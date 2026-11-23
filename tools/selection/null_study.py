"""L1 follow-up: error rates of the SCB-C simultaneous procedures under null worlds.

Methods (all max-T over every pair of checkpoints, two-sided, larger = better):

* ``v2`` seed bootstrap-t (``scbc.bootstrap_t_band``, SE recomputed per resample);
* ``v3`` within-seed label permutation, pooled permutation SD (``scbc.permutation_band``);
* ``v3s`` studentised within-seed label permutation (per-pair paired SD, recomputed in every
  permutation; the paired analogue of Chung & Romano's studentised permutation test);
* ``v4`` studentised sign-flip (``scbc.sign_flip_band``).

Null worlds (every pairwise null true, so any elimination is a false elimination):

* ``synthetic`` — the L1 generator (``simulate.synthetic_draw``): homoscedastic null,
  heteroscedastic conflicts null (NB size × 10 across checkpoints), heteroscedastic completion
  null (checkpoint-by-seed noise 0.2–1.2);
* ``plugin`` — the stored per-seed tables of a PPO run (block A, 20 seeds, or the new selection
  block S, 80 seeds) with each checkpoint's undelivered counts and conflicts rescaled
  multiplicatively to a common mean (shape, dispersion, skewness and the seed pairing of every
  checkpoint are kept). Seeds are drawn without replacement when fewer than the table's seeds are
  used, otherwise with replacement.

For each world, ``FWER`` is the rate of replicates with at least one elimination in a stage;
``joint`` is the rate with an elimination in stage 1 *or* stage 2 (stage 2 among the stage-1
survivors) — the overall error of the two-stage procedure. Each rate has a Wilson 95 % interval
and an exact two-sided binomial test against 0.05 (``p_vs_005``).

Usage::

    python tools/selection/null_study.py --out results/dev6a/l1_selection/followup --workers 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent))
import scbc
import selection_block
import simulate

METHODS = ("v2", "v3", "v3s", "v4")
N_RESAMPLE = 1000


# ------------------------------------------------------------------------------ methods
ALPHA_LEVELS = (0.05, 0.025)


def bands_v2(x: np.ndarray, seed: int) -> list[scbc.Band]:
    k, s = x.shape
    w = scbc.resample_counts(s, N_RESAMPLE, seed).astype(float)
    se0 = scbc.pair_se(x[None], np.ones((1, s)))[0]
    seb = scbc.pair_se(np.broadcast_to(x[None], (N_RESAMPLE, k, s)), w)
    boot = (w @ x.T) / s
    return [scbc.bootstrap_t_band(x.mean(1), boot, se0, seb, a) for a in ALPHA_LEVELS]


def bands_v3(x: np.ndarray, seed: int) -> list[scbc.Band]:
    k, s = x.shape
    idx = scbc.permutation_indices(s, k, N_RESAMPLE, seed)
    xp = x.T[np.arange(s)[None, :, None], idx].mean(1)  # (P, K)
    return [scbc.permutation_band(x.mean(1), xp, a) for a in ALPHA_LEVELS]


def bands_v3s(x: np.ndarray, seed: int) -> list[scbc.Band]:
    """Studentised within-seed label permutation: per-pair paired t, recomputed per permutation."""
    k, s = x.shape
    iu = np.triu_indices(k, 1)
    est = x.mean(1)
    diff = est[:, None] - est[None, :]
    d0 = (x[iu[0]] - x[iu[1]]).T  # (S, P)
    sd0 = d0.std(axis=0, ddof=1)
    se = np.zeros((k, k))
    se[iu] = sd0 / math.sqrt(s)
    se[(iu[1], iu[0])] = se[iu]
    ok = sd0 > 0
    if not ok.any():
        return [scbc.Band(est, diff, se, 0.0, diff.copy(), diff.copy()) for _ in ALPHA_LEVELS]
    idx = scbc.permutation_indices(s, k, N_RESAMPLE, seed)
    xp = x.T[np.arange(s)[None, :, None], idx]  # (P, S, K)
    tmax = np.zeros(N_RESAMPLE)
    a_all, b_all = iu[0][ok], iu[1][ok]
    for lo in range(0, a_all.size, 64):  # chunks of pairs keep memory small
        a, b = a_all[lo : lo + 64], b_all[lo : lo + 64]
        d = xp[:, :, a] - xp[:, :, b]
        sd = d.std(axis=1, ddof=1)
        t = np.divide(np.abs(d.mean(1)), sd / math.sqrt(s), out=np.zeros_like(sd), where=sd > 0)
        tmax = np.maximum(tmax, t.max(1))
    out = []
    for al in ALPHA_LEVELS:
        q = float(np.quantile(tmax, 1 - al, method="higher"))
        out.append(scbc.Band(est, diff, se, q, diff - q * se, diff + q * se))
    return out


def bands_v4(x: np.ndarray, seed: int) -> list[scbc.Band]:
    eps = scbc.sign_flips(x.shape[1], N_RESAMPLE, seed)
    return [scbc.sign_flip_band(x, eps, a) for a in ALPHA_LEVELS]


BANDS = {"v2": bands_v2, "v3": bands_v3, "v3s": bands_v3s, "v4": bands_v4}


# ------------------------------------------------------------------------------ worlds
def plugin_null(tab: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Completion and conflict contrasts (``K × S_total``) with every checkpoint rescaled to the
    run's median mean; checkpoints with zero undelivered or zero conflicts on every seed are
    dropped (they cannot be rescaled)."""
    rel = tab["released"]
    u = rel - tab["delivered"]
    c = tab["conflicts"]
    mu, mc = u.mean(1), c.mean(1)
    keep = (mu > 0) & (mc > 0)
    if keep.sum() < 2:
        return None
    u, c, rel, mu, mc = u[keep], c[keep], rel[keep], mu[keep], mc[keep]
    u = u * (np.median(mu) / mu)[:, None]
    c = c * (np.median(mc) / mc)[:, None]
    return -u / rel.mean(1, keepdims=True), -c


def world_draw(world: dict[str, Any], rep: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(world["wid"].encode()), rep]))
    if world["kind"] == "synthetic":
        sc = world["scenario"]
        d = simulate.synthetic_draw(sc, world["cal"], rng)
        rbar = d["released"].mean(1, keepdims=True)
        return d["delivered"] / rbar, -d["conflicts"]
    comp, conf = world["tables"]
    n, s = comp.shape[1], world["s"]
    idx = rng.choice(n, s, replace=s >= n)
    return comp[:, idx], conf[:, idx]


def one(world: dict[str, Any], rep: int, methods: Sequence[str]) -> dict[str, Any]:
    comp, conf = world_draw(world, rep)
    k = comp.shape[0]
    out: dict[str, Any] = {}
    for m in methods:
        bc, bm = BANDS[m](comp, rep), BANDS[m](conf, rep)
        for b1, b2, tag in zip(bc, bm, ("a05", "a025"), strict=True):
            e1 = scbc.not_dominated(b1, list(range(k)))
            all2 = scbc.not_dominated(b2, list(range(k)))
            e2 = scbc.not_dominated(b2, e1)
            out[f"{m}_{tag}_stage1"] = len(e1) < k
            out[f"{m}_{tag}_stage2_all"] = len(all2) < k
            out[f"{m}_{tag}_joint"] = len(e1) < k or len(e2) < len(e1)
            out[f"{m}_{tag}_elim_frac1"] = 1 - len(e1) / k
            out[f"{m}_{tag}_elim_frac2"] = 1 - len(all2) / k
            out[f"{m}_{tag}_q1"] = b1.q
            out[f"{m}_{tag}_q2"] = b2.q
    return out


def run_chunk(args: tuple[dict[str, Any], list[int], Sequence[str]]) -> list[dict[str, Any]]:
    world, reps, methods = args
    return [one(world, r, methods) for r in reps]


def rate(v: list[bool]) -> dict[str, float]:
    n, x = len(v), int(sum(v))
    p = x / n
    z = 1.959964
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": p, "lo": c - h, "hi": c + h, "n": n, "x": x,
            "p_vs_005": float(binomtest(x, n, 0.05).pvalue)}  # fmt: skip


def summarise(rows: list[dict[str, Any]], methods: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in methods:
        for tag in ("a05", "a025"):
            key = f"{m}_{tag}"
            out[key] = {
                "stage1": rate([r[f"{key}_stage1"] for r in rows]),
                "stage2_all": rate([r[f"{key}_stage2_all"] for r in rows]),
                "joint": rate([r[f"{key}_joint"] for r in rows]),
                "mean_elim_frac1": float(np.mean([r[f"{key}_elim_frac1"] for r in rows])),
                "mean_elim_frac2": float(np.mean([r[f"{key}_elim_frac2"] for r in rows])),
                "median_q1": float(np.median([r[f"{key}_q1"] for r in rows])),
                "median_q2": float(np.median([r[f"{key}_q2"] for r in rows])),
            }
    return out


def worlds(cal: dict[str, Any], reps_syn: int, reps_plug: int, s_block: Path | None) -> list[dict[str, Any]]:  # fmt: skip
    scs = {sc.sid: sc for sc in simulate.scenarios(cal)}
    out: list[dict[str, Any]] = []
    for sid in ("A_null_K20_S20", "N_hetero_conf_null", "N_hetero_comp_null"):
        for s in (20, 80):
            reps = reps_syn if s == 20 else max(1, reps_syn // 2)  # S = 80 is 3-4x costlier
            out.append({"wid": f"syn_{sid}_S{s}", "kind": "synthetic", "reps": reps,
                        "scenario": replace(scs[sid], s=s), "cal": cal, "s": s})  # fmt: skip
    import calibrate

    for run in selection_block.RUNS:
        tab = plugin_null(calibrate.run_arrays(run))
        if tab is not None:
            out.append({"wid": f"plugA_{run}_S20", "kind": "plugin", "reps": reps_plug,
                        "tables": tab, "s": 20})  # fmt: skip
    if s_block is not None and (s_block / "records.jsonl").exists():
        tabs = selection_block.load(s_block)
        for run in selection_block.RUNS:
            tab = plugin_null(tabs[run])
            if tab is None:
                continue
            for s in (20, 40, 80):
                out.append({"wid": f"plugS_{run}_S{s}", "kind": "plugin", "reps": reps_plug,
                            "tables": tab, "s": s})  # fmt: skip
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cal", type=Path, default=Path("results/dev6a/l1_selection/calibration.json"))
    ap.add_argument(
        "--s-block", type=Path, default=Path("results/dev6a/l1_selection/selection_block")
    )
    ap.add_argument("--reps-synthetic", type=int, default=2000)
    ap.add_argument("--reps-plugin", type=int, default=500)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--only", default="", help="comma-separated world-id prefixes")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(args.cal.read_text("utf-8"))
    methods = tuple(args.methods.split(","))
    ws = worlds(cal, args.reps_synthetic, args.reps_plugin, args.s_block)
    if args.only:
        ws = [w for w in ws if any(w["wid"].startswith(p) for p in args.only.split(","))]
    t0 = time.perf_counter()
    res: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for w in ws:
            chunks = [list(range(i, min(i + 25, w["reps"]))) for i in range(0, w["reps"], 25)]
            futs[w["wid"]] = [pool.submit(run_chunk, (w, c, methods)) for c in chunks]
        for w in ws:
            rows = [r for f in futs[w["wid"]] for r in f.result()]
            res[w["wid"]] = summarise(rows, methods) | {"reps": len(rows), "s": w["s"]}
            j = res[w["wid"]]
            print(w["wid"], " ".join(
                f"{m}:{j[m + '_a05']['stage1']['rate']:.3f}/{j[m + '_a05']['stage2_all']['rate']:.3f}"
                f"/J{j[m + '_a025']['joint']['rate']:.3f}" for m in methods), flush=True)  # fmt: skip
    meta = {"wall_s": time.perf_counter() - t0, "n_resample": N_RESAMPLE, "methods": methods,
            "reps_synthetic": args.reps_synthetic, "reps_plugin": args.reps_plugin}  # fmt: skip
    name = "null_study.json" if not args.only else f"null_study_{args.only.replace(',', '_')}.json"
    (args.out / name).write_text(json.dumps({"meta": meta, "worlds": res}, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
