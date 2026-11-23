"""L1 simulation calibration from the stored block-A validation logs (read only).

Writes ``calibration.json`` (moments and fitted generator parameters) and ``plugin_worlds.npz``
(per-run arrays of the 20 validation checkpoints × 20 block-A seeds) to ``--out``. Block-A logs
are the selection data of Issues #5/#6; no block-B, C or D data are read.

Usage::

    python tools/selection/calibrate.py --out results/dev6a/l1_selection
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

PPO_RUNS = [f"{a}_seed{s}" for a in ("mappo", "ippo") for s in (1, 2, 3)] + [
    f"mappo_{c}_seed{s}" for c in ("zeta0", "beta0", "crr05", "crr2") for s in (1, 2)
]
MADQL_RUNS = [f"madql_seed{s}" for s in (1, 2, 3)]


def run_arrays(run: str) -> dict[str, np.ndarray]:
    """``steps (K)``, ``delivered / released / conflicts / collisions / ret (K, S)`` per seed."""
    with open(f"results/training/{run}/validation.jsonl", encoding="utf-8") as fh:
        vals = [json.loads(line) for line in fh]
    vals = [v for v in vals if v.get("planner") == "astar"]
    seeds = sorted({r["seed"] for r in vals[0]["records"]})
    out = {
        k: np.zeros((len(vals), len(seeds)))
        for k in ("delivered", "released", "conflicts", "collisions", "ret")
    }
    for i, v in enumerate(vals):
        per: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in v["records"]:
            per[r["seed"]].append(r)
        for j, s in enumerate(seeds):
            rs = per[s]
            out["delivered"][i, j] = sum(r["delivered"] for r in rs)
            out["released"][i, j] = sum(r["released"] for r in rs)
            out["conflicts"][i, j] = np.mean([r["conflicts_total"] for r in rs])
            out["collisions"][i, j] = sum(r["physical_collisions"] for r in rs)
            out["ret"][i, j] = np.mean([r["base_return"] for r in rs])
    out["steps"] = np.array([v["env_steps"] for v in vals], dtype=float)
    return out


def simulate_undelivered(c: float, su: float, se: float, released: np.ndarray, k: int, s: int,
                         rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    u = rng.normal(0, su, s)
    e = rng.normal(0, se, (k, s))
    p = 1 / (1 + np.exp(-(np.log(c / (1 - c)) + u + e)))
    rel = rng.choice(released, s)
    return rng.binomial(rel.astype(int)[None, :].repeat(k, 0), 1 - p)


def corr_offdiag(m: np.ndarray) -> float:
    c = np.corrcoef(m)
    iu = np.triu_indices(c.shape[0], 1)
    return float(np.nanmean(c[iu]))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    worlds = {run: run_arrays(run) for run in PPO_RUNS + MADQL_RUNS}
    arrays: dict[str, Any] = {f"{run}__{k}": v for run, a in worlds.items() for k, v in a.items()}
    np.savez_compressed(args.out / "plugin_worlds.npz", **arrays)
    # ---- empirical moments from late PPO checkpoints (>= 200 k steps)
    und, rel, conf, corr_u, corr_c, corr_uc, conf_low = [], [], [], [], [], [], []
    for run in PPO_RUNS:
        a = worlds[run]
        late = a["steps"] >= 200_000
        u = a["released"][late] - a["delivered"][late]
        c = a["conflicts"][late]
        und.append(u)
        rel.append(a["released"][late])
        conf.append(c)
        corr_u.append(corr_offdiag(u))
        corr_c.append(corr_offdiag(c))
        corr_uc.append(np.nanmean([np.corrcoef(u[i], c[i])[0, 1] for i in range(u.shape[0])]))
        conf_low.append(c[c.mean(1) <= 5])
    und_all, rel_all = np.concatenate(und), np.concatenate(rel)
    conf_all, conf_lo = np.concatenate(conf), np.concatenate(conf_low)
    released_values = rel_all[0].tolist() + rel_all[1:].ravel().tolist()
    m_u, v_u, r_u = float(und_all.mean()), float(und_all.var()), float(np.nanmean(corr_u))
    # ---- fit (c, sigma_u, sigma_e) to mean, variance and cross-checkpoint correlation
    rng = np.random.default_rng(0)
    rel_pool = np.asarray(released_values)
    best: tuple[float, float, float, float] | None = None
    for su in np.arange(0.0, 1.61, 0.1):
        for se in np.arange(0.0, 1.61, 0.1):
            for comp in (0.982, 0.984, 0.986, 0.988, 0.99):
                sim = simulate_undelivered(comp, su, se, rel_pool, 12, 4000, rng)
                loss = (
                    ((sim.mean() - m_u) / m_u) ** 2
                    + ((sim.var() - v_u) / v_u) ** 2
                    + (corr_offdiag(sim) - r_u) ** 2 / 0.05**2
                )
                if best is None or loss < best[0]:
                    best = (float(loss), float(comp), float(su), float(se))
    assert best is not None
    # ---- conflicts: negative binomial size and shared seed effect (low-conflict checkpoints)
    mc, vc = float(conf_lo.mean()), float(conf_lo.var())
    phi = mc**2 / max(vc - mc, 1e-9)
    r_c = float(np.nanmean(corr_c))
    sigma_w = float(np.sqrt(max(np.log1p(r_c * (vc / mc**2)), 0.0))) if mc > 0 else 0.0
    cal = {
        "source": "results/training/<run>/validation.jsonl (block A, A*), PPO runs: "
        + ", ".join(PPO_RUNS),
        "late_checkpoints_from_step": 200_000,
        "released_per_seed": {
            "mean": float(rel_all.mean()),
            "min": float(rel_all.min()),
            "max": float(rel_all.max()),
            "values": sorted(set(released_values)),
        },
        "undelivered_per_seed": {"mean": m_u, "var": v_u, "cross_checkpoint_corr": r_u},
        "fit_undelivered": {
            "completion": best[1],
            "sigma_seed": best[2],
            "sigma_checkpoint_seed": best[3],
            "loss": best[0],
        },
        "conflicts_per_episode_low_checkpoints": {
            "mean": mc,
            "var": vc,
            "nb_size": phi,
            "cross_checkpoint_corr": r_c,
            "sigma_seed_log": sigma_w,
        },
        "conflicts_all_late": {
            "mean": float(conf_all.mean()),
            "median": float(np.median(conf_all)),
            "max": float(conf_all.max()),
        },
        "seed_corr_undelivered_conflicts": float(np.nanmean(corr_uc)),
        "plugin_worlds": {
            "ppo": PPO_RUNS,
            "madql": MADQL_RUNS,
            "checkpoints_per_run": 20,
            "seeds": 20,
        },
    }
    (args.out / "calibration.json").write_text(json.dumps(cal, indent=1), "utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in cal.items()
                if k not in ("released_per_seed", "plugin_worlds", "source")
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
