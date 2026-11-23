"""SCB-C v5 — the frozen checkpoint-selection procedure (Issue 6A; approved 2026-11-21).

Specification: ``docs/research/2026-11-20-issue-6a-proposal.md`` §23. Summary:

1. **Selection data**: block A (seeds 10 000–10 019) and block S (13 000–13 079), every seed at
   the three levels, A\\* planner, greedy mode, shield off: 100 seeds, 300 episodes per checkpoint.
2. **Exclusions** (first matching reason wins): ``invalid`` (a duplicate episode; a record of
   the checkpoint that is not a greedy A\\* PPO episode with the shield off on a selection seed
   and level; a negative or non-integer count; delivered > released in an episode), then
   ``incomplete`` (a seed × level episode missing, or a required field missing / null), then
   ``invalid`` (no released task at all), then ``collision`` (any physical collision). Nothing
   is imputed.
3. **Stage 1 (the only statistical stage)**: seed bootstrap-t max-T over all pairs of valid
   checkpoints on pooled completion (``scbc.bootstrap_t_band``, delta-method standard errors
   recomputed per resample), two-sided, α = 0.05, B = 10 000 resamples, NumPy ``default_rng(0)``,
   quantile method ``higher``. Checkpoint ``k`` is eliminated when some valid ``j`` has a lower
   bound ``> 0`` on ``comp_j − comp_k``. There is no conflict test.
4. **One-SE band** among the survivors: ``j*`` = the survivor with the highest completion
   estimate; ``k`` is in the band when ``comp_j* − comp_k ≤ SE_j*k``, the standard error of the
   paired per-seed difference ``D_j*s / R̄_j* − D_ks / R̄_k`` (sample SD, ddof 1, divided by
   √S). A selection heuristic, not an equivalence test.
5. **Pick**: the band member with the fewest conflicts per episode.
6. **Ties** (exact rational comparison of the counts): the latest training step, both for
   ``j*`` and for the pick.

No completion tolerance or floor is used. Pure NumPy; independent of the production code.

Usage::

    python tools/selection/scbc_frozen.py --out results/dev6a/l1_selection/frozen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent))
import scbc

F = npt.NDArray[np.float64]

SPEC_VERSION = "SCB-C v5 (frozen 2026-11-21)"
BLOCK_A = tuple(range(10_000, 10_020))
BLOCK_S = tuple(range(13_000, 13_080))
SEEDS = BLOCK_A + BLOCK_S
LEVELS = ("static", "obstacles", "full")
ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 0
REQUIRED = ("seed", "level", "checkpoint", "delivered", "released", "conflicts_total",
            "physical_collisions")  # fmt: skip
EXPECTED = {"controller": "ppo", "planner": "astar", "mode": "greedy", "shield": "off"}
RUNS = [f"{a}_seed{s}" for a in ("mappo", "ippo") for s in (1, 2, 3)]
BLOCK_OF = {s: "A" for s in BLOCK_A} | {s: "S" for s in BLOCK_S}


# ------------------------------------------------------------------------------ data
@dataclass(frozen=True)
class Table:
    """Per checkpoint ``k`` (sorted by step) and seed ``s`` (``SEEDS`` order): sums over the
    three levels of delivered / released / conflicts / collisions; ``NaN`` where incomplete."""

    checkpoints: tuple[str, ...]
    steps: tuple[int, ...]
    delivered: F
    released: F
    conflicts: F  # total conflicts over the seed's three episodes
    collisions: F
    problems: dict[int, str] = field(default_factory=dict)  # checkpoint index -> reason
    n_levels: int = len(LEVELS)  # episodes per seed


def _count(v: Any) -> int | None:
    """A non-negative integer count, or ``None`` when the value is not one."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and (not math.isfinite(v) or not v.is_integer()):
        return None
    return int(v) if v >= 0 else None


def build_table(records: Iterable[Mapping[str, Any]], checkpoints: Sequence[str],
                seeds: Sequence[int] = SEEDS, levels: Sequence[str] = LEVELS,
                block_of: Mapping[int, str] | None = BLOCK_OF) -> Table:  # fmt: skip
    """Strict per-seed table from episode records of the given checkpoints.

    A record of an unknown seed or level, of a seed whose ``block`` field does not match
    ``block_of``, or one that is not a greedy A* PPO episode with the shield off, makes its
    checkpoint ``invalid`` (records of other checkpoints are ignored); so does a duplicate (checkpoint, seed,
    level) or a bad count. A missing episode or a missing / null required field makes it
    ``incomplete``. A checkpoint with any invalid record is reported ``invalid`` (its skipped
    record also leaves a cell missing); otherwise a missing cell makes it ``incomplete``."""
    names = sorted(set(checkpoints), key=_step)
    ci = {c: i for i, c in enumerate(names)}
    si = {s: j for j, s in enumerate(seeds)}
    li = set(levels)
    k, n = len(names), len(seeds)
    cells: dict[tuple[int, int, str], tuple[int, int, int, int]] = {}
    bad: dict[int, set[str]] = {}
    for r in records:
        c = r.get("checkpoint")
        if c not in ci:
            continue  # another run's or another block's record: not part of this table
        i = ci[c]
        if any(r.get(f) is None for f in REQUIRED):
            bad.setdefault(i, set()).add("incomplete")
            continue
        if any(r.get(f) != v for f, v in EXPECTED.items()) or r["seed"] not in si \
                or r["level"] not in li                 or (block_of is not None and r.get("block") != block_of.get(r["seed"])):  # fmt: skip
            bad.setdefault(i, set()).add("invalid")
            continue
        raw = [
            _count(r[f])
            for f in ("delivered", "released", "conflicts_total", "physical_collisions")
        ]
        vals = [v for v in raw if v is not None]
        if len(vals) != len(raw):
            bad.setdefault(i, set()).add("invalid")
            continue
        d, rel, cf, col = vals
        if d > rel:
            bad.setdefault(i, set()).add("invalid")
            continue
        key = (i, si[r["seed"]], r["level"])
        if key in cells:
            bad.setdefault(i, set()).add("invalid")
            continue
        cells[key] = (d, rel, cf, col)
    arrs = {
        f: np.full((k, n), np.nan) for f in ("delivered", "released", "conflicts", "collisions")
    }
    for i in range(k):
        for j in range(n):
            got = [cells.get((i, j, lv)) for lv in levels]
            if any(g is None for g in got):
                bad.setdefault(i, set()).add("incomplete")
                continue
            for f, idx in (("delivered", 0), ("released", 1), ("conflicts", 2), ("collisions", 3)):
                arrs[f][i, j] = sum(g[idx] for g in got if g is not None)
    problems = {i: ("invalid" if "invalid" in r else "incomplete") for i, r in bad.items()}
    return Table(tuple(names), tuple(_step(c) for c in names), arrs["delivered"],
                 arrs["released"], arrs["conflicts"], arrs["collisions"], problems, len(levels))  # fmt: skip


def _step(name: str) -> int:
    """``step_00050000.pt`` -> 50 000."""
    stem = Path(name).stem
    if not stem.startswith("step_") or not stem[5:].isdigit():
        raise ValueError(f"not a step checkpoint name: {name!r}")
    return int(stem[5:])


def load_run(run: str, s_records: Path) -> Table:
    """Block-A records from ``results/training/<run>/validation.jsonl`` (A* entries) and block-S
    records of ``run`` from the selection-block ``records.jsonl``; the run's 20 step checkpoints."""
    ck = sorted(p.name for p in Path(f"results/training/{run}/checkpoints").glob("step_*[0-9].pt"))
    recs: list[dict[str, Any]] = []
    with open(f"results/training/{run}/validation.jsonl", encoding="utf-8") as fh:
        for line in fh:
            v = json.loads(line)
            if v.get("planner") == "astar":
                recs.extend(r for r in v["records"] if r.get("block") == "A")
    with open(s_records, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("label") == run and r.get("block") == "S":
                recs.append(r)
    return build_table(recs, ck)


# ------------------------------------------------------------------------------ procedure
@dataclass(frozen=True)
class Frozen:
    selected: int | None  # index into the table's checkpoints
    selected_step: int | None
    excluded: dict[int, str]
    survivors: tuple[int, ...]
    reference: int | None  # j*: the survivor with the highest completion estimate
    band: tuple[int, ...]
    completion: tuple[float, ...]  # pooled completion estimate per checkpoint (NaN if excluded)
    conflicts_per_episode: tuple[float, ...]
    se_to_reference: dict[int, float]
    q: float | None
    params: dict[str, Any]


def _exclusions(t: Table) -> dict[int, str]:
    out = dict(t.problems)
    rest = scbc.validate(t.delivered, t.released, t.conflicts, t.collisions)
    for i, reason in rest.items():
        out.setdefault(i, reason)
    return out


def _latest_max(
    items: Sequence[int], key: Mapping[int, Fraction | int], steps: Sequence[int]
) -> int:
    """The item with the largest ``key``; ties to the latest step (steps are unique)."""
    return max(items, key=lambda i: (key[i], steps[i]))


def select(t: Table, alpha: float = ALPHA, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Frozen:
    """The frozen SCB-C v5 selection on one run's table."""
    if len(set(t.steps)) != len(t.steps):
        raise ValueError("training steps must be unique")
    k, s = t.delivered.shape
    if s < scbc.MIN_SEEDS:
        raise ValueError(f"at least {scbc.MIN_SEEDS} seeds are required")
    params = {"spec": SPEC_VERSION, "alpha": alpha, "n_boot": n_boot, "seed": seed,
              "n_checkpoints": k, "n_seeds": s, "quantile": "higher", "studentize": "t (v2)"}  # fmt: skip
    excluded = _exclusions(t)
    valid = [i for i in range(k) if i not in excluded]
    comp = np.full(k, np.nan)
    cpe = np.full(k, np.nan)
    if not valid:
        return Frozen(None, None, excluded, (), None, (), tuple(comp), tuple(cpe), {}, None, params)
    d, r, c = t.delivered[valid], t.released[valid], t.conflicts[valid]
    exact_comp = {i: Fraction(int(t.delivered[i].sum()), int(t.released[i].sum())) for i in valid}
    total_conf = {i: int(t.conflicts[i].sum()) for i in valid}
    comp[valid] = d.sum(1) / r.sum(1)
    cpe[valid] = c.sum(1) / (s * t.n_levels)
    # ---- stage 1: v2 bootstrap-t on pooled completion over all valid pairs
    est = d.sum(1) / r.sum(1)
    w = scbc.resample_counts(s, n_boot, seed).astype(np.float64)
    est_b = (w @ d.T) / (w @ r.T)
    z0 = ((d - est[:, None] * r) / r.mean(1)[:, None])[None]
    rbar_b = (w @ r.T) / s
    zb = (d[None] - est_b[:, :, None] * r[None]) / rbar_b[:, :, None]
    band1 = scbc.bootstrap_t_band(est, est_b, scbc.pair_se(z0, np.ones((1, s)))[0],
                                  scbc.pair_se(zb, w), alpha)  # fmt: skip
    survivors = [valid[i] for i in scbc.not_dominated(band1, list(range(len(valid))))]
    # ---- one-SE band among the survivors
    j = _latest_max(survivors, exact_comp, t.steps)
    x = {i: t.delivered[i] / t.released[i].mean() for i in survivors}
    se = {i: float(np.std(x[j] - x[i], ddof=1) / math.sqrt(s)) for i in survivors}
    band = [i for i in survivors if float(exact_comp[j] - exact_comp[i]) <= se[i]]
    neg_conf = {i: -total_conf[i] for i in band}  # every valid checkpoint has S × n_levels episodes
    pick = _latest_max(band, neg_conf, t.steps)
    return Frozen(pick, t.steps[pick], excluded, tuple(sorted(survivors)), j, tuple(sorted(band)),
                  tuple(comp), tuple(cpe), se, float(band1.q), params)  # fmt: skip


# ------------------------------------------------------------------------------ CLI
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s-records", type=Path,
                    default=Path("results/dev6a/l1_selection/selection_block/records.jsonl"))  # fmt: skip
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "spec": SPEC_VERSION,
        "config": {
            "seeds": [SEEDS[0], BLOCK_A[-1], BLOCK_S[0], SEEDS[-1]],
            "n_seeds": len(SEEDS),
            "levels": LEVELS,
            "alpha": ALPHA,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "expected_record": EXPECTED,
        },
        "code": {
            p: sha256(Path(p))
            for p in ("tools/selection/scbc.py", "tools/selection/scbc_frozen.py")
        },
        "data": {"tools-selection-block-records": sha256(args.s_records)}
        | {
            f"{run}/validation.jsonl": sha256(Path(f"results/training/{run}/validation.jsonl"))
            for run in RUNS
        },
        "numpy": np.__version__,
        "runs": {},
    }
    for run in RUNS:
        t = load_run(run, args.s_records)
        f = select(t)
        out["runs"][run] = {
            "selected": None if f.selected is None else t.checkpoints[f.selected],
            "selected_step": f.selected_step,
            "excluded": {t.checkpoints[i]: why for i, why in f.excluded.items()},
            "survivors": [t.checkpoints[i] for i in f.survivors],
            "reference": None if f.reference is None else t.checkpoints[f.reference],
            "band": [t.checkpoints[i] for i in f.band],
            "q": f.q,
            "completion": dict(zip(t.checkpoints, f.completion, strict=True)),
            "conflicts_per_episode": dict(zip(t.checkpoints, f.conflicts_per_episode, strict=True)),
            "se_to_reference": {t.checkpoints[i]: v for i, v in f.se_to_reference.items()},
        }
        sel = f.selected
        print(f"{run:12s} pick {t.checkpoints[sel] if sel is not None else None}"
              f"  survivors {len(f.survivors)}/{len(t.checkpoints)}  band {[t.steps[i] // 1000 for i in f.band]}"
              f"  q {f.q:.3f}  excluded {len(f.excluded)}")  # fmt: skip
    (args.out / "selections.json").write_text(json.dumps(out, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
