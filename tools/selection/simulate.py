"""L1 simulation study of SCB-C (Issue 6A). Synthetic and plug-in scenarios; no robot simulation.

* **SYNTHETIC** scenarios draw per-seed counts from a parametric generator calibrated to block A
  (``calibration.json``): binomial undelivered tasks with a shared seed effect and a
  checkpoint × seed effect on the logit scale; negative-binomial conflicts with a shared seed
  effect, optionally correlated with the completion seed effect. True values are known by
  construction.
* **PLUG-IN** scenarios treat each stored run's block-A table (20 checkpoints × 20 seeds) as the
  population: true values are its 20-seed values, and each replicate resamples seeds with
  replacement. They keep the real dependence structure but cannot create new scenarios.

Neither kind is a robot-policy evaluation. Outputs go to ``--out``.

Usage::

    python tools/selection/simulate.py --cal results/dev6a/l1_selection --out results/dev6a/l1_selection/v2 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import scbc

N_BOOT = 2000


@dataclass(frozen=True)
class Scenario:
    sid: str
    kind: str  # "synthetic" | "plugin"
    question: str
    k: int = 20
    s: int = 20
    completion: tuple[float, ...] = ()  # true pooled completion per checkpoint (synthetic)
    conflicts: tuple[float, ...] = ()  # true mean conflicts per episode per checkpoint
    rho_seed: float | None = None  # corr(failure seed effect, conflict seed effect)
    reps: int = 500
    n_boot: int = N_BOOT
    run: str = ""  # plug-in world
    duplicate: tuple[int, int] | None = None  # make checkpoint b an exact copy of a
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------ generator
def expit(x: np.ndarray | float) -> Any:
    return 1 / (1 + np.exp(-x))


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)


def mean_completion(mu: float, sd: float) -> float:
    return float((GH_W * expit(mu + sd * GH_X)).sum() / GH_W.sum())


def logit_for(target: float, sd: float) -> float:
    lo, hi = -10.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mean_completion(mid, sd) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def synthetic_draw(
    sc: Scenario, cal: dict[str, Any], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    f = cal["fit_undelivered"]
    su, se = f["sigma_seed"], f["sigma_checkpoint_seed"]
    cc = cal["conflicts_per_episode_low_checkpoints"]
    phi, sw = cc["nb_size"], cc["sigma_seed_log"]
    rho = cal["seed_corr_undelivered_conflicts"] if sc.rho_seed is None else sc.rho_seed
    se_k = np.asarray(sc.extra.get("sigma_e", (se,) * sc.k), float)
    mu = np.array(
        [
            logit_for(c, math.sqrt(su**2 + s_e**2))
            for c, s_e in zip(sc.completion, se_k, strict=True)
        ]
    )
    cov = [[su**2, -rho * su * sw], [-rho * su * sw, sw**2]]  # failures ~ -u, conflicts ~ +w
    uw = rng.multivariate_normal([0, 0], cov, sc.s)
    u, w = uw[:, 0], uw[:, 1]
    e = rng.normal(0, 1, (sc.k, sc.s)) * se_k[:, None]
    p = expit(mu[:, None] + u[None, :] + e)
    rel = rng.choice(np.asarray(cal["released_per_seed"]["values"], float), sc.s)
    released = np.repeat(rel[None, :], sc.k, 0)
    undel = rng.binomial(released.astype(int), 1 - p)
    delivered = released - undel
    mean_c = np.asarray(sc.conflicts)[:, None] * np.exp(w[None, :] - sw**2 / 2)
    n_ep = 3  # per-seed mean over three level episodes: NB(3 m, 3 phi) / 3
    size = n_ep * np.asarray(sc.extra.get("nb_size", (phi,) * sc.k), float)[:, None]
    lam = rng.gamma(np.broadcast_to(size, mean_c.shape), (n_ep * mean_c) / size)
    conflicts = rng.poisson(lam) / n_ep
    ret = (
        3.5
        - 0.05 * conflicts
        + 3 * (delivered / released - 0.99)
        + rng.normal(0, 0.3, (sc.k, sc.s))
    )
    if sc.duplicate is not None:
        a, b = sc.duplicate
        delivered[b], conflicts[b], ret[b] = delivered[a], conflicts[a], ret[a]
    return {"delivered": delivered.astype(float), "released": released.astype(float),
            "conflicts": conflicts, "ret": ret, "steps": np.arange(1, sc.k + 1) * 50_000.0,
            "true_completion": np.asarray(sc.completion), "true_conflicts": np.asarray(sc.conflicts)}  # fmt: skip


def plugin_draw(
    world: dict[str, np.ndarray], s: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    n = world["delivered"].shape[1]
    idx = rng.integers(0, n, s)
    out = {k: world[k][:, idx] for k in ("delivered", "released", "conflicts", "ret")}
    out["steps"] = world["steps"]
    out["true_completion"] = world["delivered"].sum(1) / world["released"].sum(1)
    out["true_conflicts"] = world["conflicts"].mean(1)
    return out


# ------------------------------------------------------------------------------ one replicate
def band_metrics(
    sel: scbc.Selection, tc: np.ndarray, tm: np.ndarray, prefix: str
) -> dict[str, Any]:
    k = len(tc)
    best_c = tc.max()
    best_set = set(np.flatnonzero(np.isclose(tc, best_c, rtol=0, atol=1e-12)).tolist())
    e1, e2 = set(sel.eligible_completion), set(sel.eligible_conflicts)
    tmin_e1 = min(tm[j] for j in e1)
    near25 = [i for i in range(k) if best_c - tc[i] <= 0.0025 and i not in best_set]
    near50 = [i for i in range(k) if best_c - tc[i] <= 0.005 and i not in best_set]
    far = [i for i in range(k) if best_c - tc[i] > 0.005]
    b = sel.completion
    bi = next(iter(best_set))
    shortfall = (
        float(max(b.upper[j, bi] for j in range(k) if j != bi))
        if (b is not None and k > 1)
        else None
    )
    return {
        f"{prefix}e1": len(e1),
        f"{prefix}e2": len(e2),
        f"{prefix}any_elim_stage1": bool(len(e1) < k),
        f"{prefix}any_elim_stage2": len(e2) < len(e1),
        f"{prefix}best_eliminated": bool(best_set - e1),
        f"{prefix}conf_best_eliminated_stage2": bool(
            any(i not in e2 and np.isclose(tm[i], tmin_e1) for i in e1)
        ),
        f"{prefix}near_0025_elim_frac": float(np.mean([i not in e1 for i in near25]))
        if near25
        else None,
        f"{prefix}near_0050_elim_frac": float(np.mean([i not in e1 for i in near50]))
        if near50
        else None,
        f"{prefix}far_elim_frac": float(np.mean([i not in e1 for i in far])) if far else None,
        f"{prefix}q_completion": None if b is None else b.q,
        f"{prefix}q_conflicts": None if sel.conflicts is None else sel.conflicts.q,
        f"{prefix}best_shortfall_bound": shortfall,
        f"{prefix}median_halfwidth": float(np.median((b.q * b.se)[np.triu_indices(k, 1)]))
        if b is not None and k > 1
        else None,
    }


def one(sc: Scenario, rep: int, cal: dict[str, Any], world: dict[str, np.ndarray] | None,
        method: str = "t", compare: bool = True) -> dict[str, Any]:  # fmt: skip
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(sc.sid.encode()), rep]))
    data = synthetic_draw(sc, cal, rng) if sc.kind == "synthetic" else plugin_draw(world, sc.s, rng)  # type: ignore[arg-type]
    args = (data["steps"], data["delivered"], data["released"], data["conflicts"])
    v2 = scbc.select(*args, n_boot=sc.n_boot, seed=rep, studentize=method, final="completion")
    v1 = (
        scbc.select(*args, n_boot=sc.n_boot, seed=rep, studentize="fixed", final="latest")
        if compare
        else None
    )
    assert not v2.excluded and (v1 is None or not v1.excluded)
    tc, tm = data["true_completion"], data["true_conflicts"]
    best_set = set(np.flatnonzero(np.isclose(tc, tc.max(), rtol=0, atol=1e-12)).tolist())
    out: dict[str, Any] = {"rep": rep}
    out |= band_metrics(v2, tc, tm, "")
    if v1 is not None:
        out |= band_metrics(v1, tc, tm, "v1_")
    assert v2.completion is not None and v2.conflicts is not None
    comp_est = v2.completion.estimate
    conf_est = -v2.conflicts.estimate
    e2 = list(v2.eligible_conflicts)
    picks = {
        rule: scbc.choose(e2, data["steps"], comp_est, conf_est, rule) for rule in scbc.FINAL_RULES
    }
    if v1 is not None:
        assert v1.selected is not None
        picks["v1_latest"] = v1.selected
    picks["old"] = scbc.old_rule(
        data["steps"], data["delivered"], data["released"], data["ret"].mean(1)
    )
    for name, i in picks.items():
        out[f"pick_{name}_is_best"] = i in best_set
        out[f"pick_{name}_gap"] = float(tc.max() - tc[i])
        out[f"pick_{name}_gap_gt_0005"] = bool(tc.max() - tc[i] > 0.005)
        out[f"pick_{name}_conf"] = float(tm[i])
        out[f"pick_{name}_conf_excess"] = float(tm[i] - tm.min())
    return out


def run_chunk(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    sc, reps, cal, world, method, compare = args
    return [one(sc, r, cal, world, method, compare) for r in reps]


# ------------------------------------------------------------------------------ scenarios
def scenarios(cal: dict[str, Any]) -> list[Scenario]:
    c0 = cal["fit_undelivered"]["completion"]  # 0.988 (block-A calibrated)
    m0 = round(cal["conflicts_per_episode_low_checkpoints"]["mean"], 2)  # ~1.58
    out: list[Scenario] = []
    out.append(Scenario("A_null_K20_S20", "synthetic", "FWER under the global null (all checkpoints equal)",
                        completion=(c0,) * 20, conflicts=(m0,) * 20, reps=2000))  # fmt: skip
    for k in (5, 10, 40):
        out.append(Scenario(f"B_null_K{k}", "synthetic", "FWER vs number of checkpoints", k=k,
                            completion=(c0,) * k, conflicts=(m0,) * k, reps=1000))  # fmt: skip
    for s in (10, 40, 80):
        out.append(Scenario(f"C_null_S{s}", "synthetic", "FWER vs number of seeds", s=s,
                            completion=(c0,) * 20, conflicts=(m0,) * 20, reps=1000))  # fmt: skip
    for d in (0.0025, 0.005, 0.01, 0.02):
        comp = (0.985 + d,) + (0.985,) * 19
        out.append(Scenario(f"D_onebest_d{d}", "synthetic", "Sensitivity to a completion difference",
                            completion=comp, conflicts=(m0,) * 20))  # fmt: skip
    spread = tuple(np.round(np.linspace(0.975, 0.990, 20), 6))
    out.append(Scenario("E_spread_K20", "synthetic", "Realistic completion spread 0.975-0.990 (late training)",
                        completion=spread, conflicts=(m0,) * 20))  # fmt: skip
    out.append(Scenario("E_spread_K40", "synthetic", "Many checkpoints (spread, K = 40)", k=40,
                        completion=tuple(np.round(np.linspace(0.975, 0.990, 40), 6)), conflicts=(m0,) * 40))  # fmt: skip
    for s in (10, 40, 80):
        out.append(Scenario(f"E_spread_S{s}", "synthetic", "Spread vs number of seeds", s=s,
                            completion=spread, conflicts=(m0,) * 20))  # fmt: skip
    for dm in (0.5, 1.0, 2.0, 5.0, 20.0):
        out.append(Scenario(f"F_conf_dm{dm}", "synthetic", "Sensitivity to a conflict difference (completion tied)",
                            completion=(c0,) * 20, conflicts=(1.0,) + (1.0 + dm,) * 19))  # fmt: skip
    comp_g = tuple(np.round(np.linspace(0.980, 0.990, 20), 6))
    out.append(Scenario("G_aligned", "synthetic", "Completion and conflicts aligned (better completion, fewer conflicts)",
                        completion=comp_g, conflicts=tuple(np.round(np.linspace(5, 1, 20), 3))))  # fmt: skip
    perm = np.random.default_rng(3).permutation(np.round(np.linspace(1, 5, 20), 3))
    out.append(Scenario("G_independent", "synthetic", "Completion and conflicts unrelated",
                        completion=comp_g, conflicts=tuple(perm.tolist())))  # fmt: skip
    out.append(Scenario("G_tradeoff", "synthetic", "Trade-off: better completion, more conflicts (early-MAPPO-like)",
                        completion=comp_g, conflicts=tuple(np.round(np.geomspace(1, 40, 20), 3))))  # fmt: skip
    for rho in (-0.5, 0.0, 0.5):
        out.append(Scenario(f"H_tradeoff_rho{rho}", "synthetic", "Trade-off with seed-level correlation of failures and conflicts",
                            completion=comp_g, conflicts=tuple(np.round(np.geomspace(1, 40, 20), 3)), rho_seed=rho))  # fmt: skip
    out.append(Scenario("I_exact_tie", "synthetic", "Two identical best checkpoints (exact tie), others 0.01 worse",
                        completion=(0.99, 0.99) + (0.98,) * 18, conflicts=(m0,) * 20, duplicate=(0, 1)))  # fmt: skip
    out.append(Scenario("I_near_tie", "synthetic", "Near tie: best 0.001 above the runner-up, others 0.01 worse",
                        completion=(0.991, 0.990) + (0.981,) * 18, conflicts=(m0,) * 20))  # fmt: skip
    for b in (1000, 10_000):
        out.append(Scenario(f"L_boot_B{b}", "synthetic", "Bootstrap-size sensitivity (spread scenario)",
                            completion=spread, conflicts=(m0,) * 20, reps=200, n_boot=b))  # fmt: skip
    for s in (20, 40, 80, 160):
        out.append(Scenario(f"K_equiv_S{s}", "synthetic", "Can a 0.005 practical-equivalence margin be demonstrated?", s=s,
                            completion=(c0,) * 20, conflicts=(m0,) * 20, reps=300))  # fmt: skip
    out.append(Scenario("N_hetero_conf_null", "synthetic", "Heteroscedastic null: equal mean conflicts, dispersion varies 10x",
                        completion=(c0,) * 20, conflicts=(m0,) * 20, reps=1000,
                        extra={"nb_size": tuple(np.round(np.geomspace(0.05, 0.5, 20), 4))}))  # fmt: skip
    out.append(Scenario("N_hetero_comp_null", "synthetic", "Heteroscedastic null: equal mean completion, checkpoint noise varies",
                        completion=(c0,) * 20, conflicts=(m0,) * 20, reps=1000,
                        extra={"sigma_e": tuple(np.round(np.linspace(0.2, 1.2, 20), 3))}))  # fmt: skip
    for run in cal["plugin_worlds"]["ppo"] + cal["plugin_worlds"]["madql"]:
        out.append(
            Scenario(
                f"P_{run}",
                "plugin",
                "Plug-in world from the stored block-A table",
                run=run,
                reps=300,
            )
        )
    return out


def summarise(sc: Scenario, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> dict[str, float] | None:
        v = [float(r[key]) for r in rows if r.get(key) is not None]
        if not v:
            return None
        p, n = float(np.mean(v)), len(v)
        z = 1.959964
        den = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        return {"p": p, "low": max(centre - half, 0.0), "high": min(centre + half, 1.0), "n": n}

    def mean(key: str) -> float | None:
        v = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    out: dict[str, Any] = {"scenario": asdict(sc), "reps": len(rows)}
    for pre in ("", "v1_"):
        for key in (
            "any_elim_stage1",
            "any_elim_stage2",
            "best_eliminated",
            "conf_best_eliminated_stage2",
        ):
            out[pre + key] = rate(pre + key)
        for key in ("near_0025_elim_frac", "near_0050_elim_frac", "far_elim_frac", "e1", "e2", "q_completion",
                    "q_conflicts", "median_halfwidth"):  # fmt: skip
            out[pre + key] = mean(pre + key)
        sb = [
            r[pre + "best_shortfall_bound"]
            for r in rows
            if r.get(pre + "best_shortfall_bound") is not None
        ]
        out[pre + "equiv_0005_demonstrated"] = (
            float(np.mean([x < 0.005 for x in sb])) if sb else None
        )
    for name in (*scbc.FINAL_RULES, "v1_latest", "old"):
        if not any(f"pick_{name}_gap" in r for r in rows):
            continue
        out[f"pick_{name}"] = {"is_best": rate(f"pick_{name}_is_best"), "gap": mean(f"pick_{name}_gap"),
                               "gap_gt_0005": rate(f"pick_{name}_gap_gt_0005"), "conf": mean(f"pick_{name}_conf"),
                               "conf_excess": mean(f"pick_{name}_conf_excess")}  # fmt: skip
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cal",
        type=Path,
        required=True,
        help="directory with calibration.json and plugin_worlds.npz",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--method", choices=("signflip", "perm", "t", "fixed"), default="t")
    ap.add_argument(
        "--no-compare", action="store_true", help="skip the v1 comparator per replicate"
    )
    ap.add_argument("--scenarios", default="", help="comma-separated scenario ids (default: all)")
    args = ap.parse_args(argv)
    target = args.out / "simulation_results.json"
    if target.exists():
        raise SystemExit(f"{target} exists; written once")
    args.out.mkdir(parents=True, exist_ok=True)
    cal = json.loads((args.cal / "calibration.json").read_text("utf-8"))
    npz = np.load(args.cal / "plugin_worlds.npz")
    worlds: dict[str, dict[str, np.ndarray]] = {}
    for key in npz.files:
        run, name = key.split("__")
        worlds.setdefault(run, {})[name] = npz[key]
    scs = scenarios(cal)
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scs = [sc for sc in scs if sc.sid in keep]
    t0 = time.perf_counter()
    jobs = []
    for sc in scs:
        reps = list(range(sc.reps))
        for i in range(0, len(reps), 50):
            jobs.append(
                (sc, reps[i : i + 50], cal, worlds.get(sc.run), args.method, not args.no_compare)
            )
    with ProcessPoolExecutor(args.workers) as ex:
        chunks = list(ex.map(run_chunk, jobs))
    wall = time.perf_counter() - t0
    by: dict[str, list[dict[str, Any]]] = {}
    for (sc, *_), rows in zip(jobs, chunks, strict=True):
        by.setdefault(sc.sid, []).extend(rows)
    results = [summarise(sc, by[sc.sid]) for sc in scs]
    # single-call resource measurement (K = 20, S = 20, B = 10 000)
    sc0 = scs[0]
    data = synthetic_draw(sc0, cal, np.random.default_rng(0))
    tracemalloc.start()
    t1 = time.perf_counter()
    scbc.select(data["steps"], data["delivered"], data["released"], data["conflicts"], n_boot=10_000, seed=0,
                studentize=args.method)  # fmt: skip
    one_call = time.perf_counter() - t1
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    meta = {"wall_s": wall, "workers": args.workers, "method": args.method, "compare_v1": not args.no_compare, "replicates": sum(sc.reps for sc in scs),
            "single_select_K20_S20_B10000": {"seconds": one_call, "peak_traced_mb": peak / 2**20}}  # fmt: skip
    target.write_text(json.dumps({"meta": meta, "results": results}, indent=1), "utf-8")
    with (args.out / "simulation_reps.jsonl").open("w", encoding="utf-8") as fh:
        for sc in scs:
            for r in by[sc.sid]:
                fh.write(json.dumps({"sid": sc.sid, **r}) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
