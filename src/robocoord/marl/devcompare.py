"""Issue #5 development comparison on validation block B (not E5; E5/E6 belong to Issue #8).

Run once, after the configuration is frozen and each training run's checkpoint has been selected
on block A. Block B is never used for selection or tuning. Design (approved):

* controllers: ``baseline`` (resolver on, endpoint on), ``reference`` (endpoint on) and each
  selected MADQL policy (endpoint on, greedy); auction allocator, shield off;
* A*: seeds 10 020–10 099 × levels {static, obstacles, full} (240 episodes per controller);
* RRT* (``n_max`` = 1 000): seeds 10 020–10 039 × the three levels (60 per controller).

Statistics (``05`` B.3): per-instance paired differences ``MADQL − comparator`` with a paired
bootstrap 95 % CI, the Wilcoxon signed-rank p-value, the matched-pairs rank-biserial correlation,
Holm–Bonferroni within each metric's family, and a two-level bootstrap pooled over the policies
(resample policies, then instances). Every number comes from the stored records.

Issue #6 generalisation (additive; the Issue #5 analysis is reproduced unchanged by the defaults):

* learned policies are recognised by their label prefix (``madql``, ``mappo``, ``ippo``) and the
  comparators are the non-learned controllers present in the records (``baseline``,
  ``baseline-noresolver``, ``reference``);
* the absolute conflict metrics of D10 can be analysed as separate families
  (``conflicts_total`` per episode, ``conflicts_per_delivered``; the latter is derived for Issue #5
  records, and instances where it is undefined are excluded and counted);
* :func:`algorithm_bootstrap` compares two algorithms (or an algorithm and a non-learned
  controller) with a two-level bootstrap that resamples the **training seeds independently per
  algorithm** and the **instances jointly** (paired);
* :func:`load_records` reads stored ``records.jsonl`` files read-only, so historical records are
  reused, never rewritten.

Usage::

    python -m robocoord.marl.devcompare --policy results/training/madql_seed1/best.pt \\
        --policy … --out results/dev5/comparison --workers 9
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.experiments.stats import holm, paired_bootstrap, rank_biserial, wilcoxon_p
from robocoord.marl.evaluation import EvalTask, run_tasks, summarise
from robocoord.marl.scenarios import BLOCK_B, BLOCK_B_RRT, LEVELS, check_seeds

METRICS = ("completion", "base_return", "conflicts_per_100")
#: Issue #6 adds the absolute conflict metrics (D10) as separate families.
METRICS_ISSUE6 = (*METRICS, "conflicts_total", "conflicts_per_delivered")
COMPARATORS = ("baseline", "reference")
ALL_COMPARATORS = ("baseline", "baseline-noresolver", "reference")
LEARNED_PREFIXES = ("madql", "mappo", "ippo")


def metric_value(r: dict[str, Any], metric: str) -> float | None:
    """A record's metric; ``conflicts_per_delivered`` is derived when the record predates it."""
    if metric == "conflicts_per_delivered" and metric not in r:
        d = int(r["delivered"])
        return float(r["conflicts_total"]) / d if d else None
    v = r.get(metric)
    return None if v is None else float(v)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``records.jsonl`` file (read-only)."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tasks_for(controller: str, label: str, checkpoint: str | None = None) -> list[EvalTask]:
    a = [
        EvalTask(controller, lvl, s, "astar", "B", checkpoint, label=label)
        for lvl in LEVELS
        for s in check_seeds(BLOCK_B, "B")
    ]
    r = [
        EvalTask(controller, lvl, s, "rrtstar", "B", checkpoint, label=label)
        for lvl in LEVELS
        for s in check_seeds(BLOCK_B_RRT, "B")
    ]
    return a + r


def _key(r: dict[str, Any]) -> tuple[str, str, int]:
    return (r["planner"], r["level"], int(r["seed"]))


def two_level_bootstrap(
    diffs: Sequence[Sequence[float]], *, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean difference with a CI from resampling policies, then instances within each policy."""
    arr = [np.asarray(d, dtype=np.float64) for d in diffs]
    rng = np.random.default_rng(seed)
    p = len(arr)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, p, size=p)
        boots[b] = np.mean(
            [arr[k][rng.integers(0, arr[k].size, arr[k].size)].mean() for k in chosen]
        )
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(np.mean([a.mean() for a in arr])), "low": float(lo), "high": float(hi)}


def _is_learned(label: str) -> bool:
    return label.startswith(LEARNED_PREFIXES)


def paired_values(
    a: dict[tuple[str, str, int], dict[str, Any]],
    b: dict[tuple[str, str, int], dict[str, Any]],
    metric: str,
    planner: str,
) -> tuple[list[tuple[str, str, int]], list[float], list[float], int]:
    """Instances of ``planner`` present in both, with the metric defined in both; returns the
    keys, the two value lists and the number of instances excluded as undefined."""
    keys, x, y, excluded = [], [], [], 0
    for k in sorted(k for k in a if k[0] == planner and k in b):
        va, vb = metric_value(a[k], metric), metric_value(b[k], metric)
        if va is None or vb is None:
            excluded += 1
            continue
        keys.append(k)
        x.append(va)
        y.append(vb)
    return keys, x, y, excluded


def analyse(
    records: list[dict[str, Any]],
    metrics: Sequence[str] = METRICS,
    comparators: Sequence[str] | None = None,
) -> dict[str, Any]:
    by_label: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    for r in records:
        by_label.setdefault(r["label"], {})[_key(r)] = r
    policies = sorted(k for k in by_label if _is_learned(k))
    comps = [c for c in (comparators or ALL_COMPARATORS) if c in by_label]
    out: dict[str, Any] = {"summaries": {}, "paired": [], "pooled": []}
    for label, recs in sorted(by_label.items()):
        for planner in ("astar", "rrtstar"):
            rs = [r for k, r in recs.items() if k[0] == planner]
            if rs:
                out["summaries"][f"{label}/{planner}"] = summarise(rs)
    for metric in metrics:
        family: list[dict[str, Any]] = []
        for comp in comps:
            for planner in ("astar", "rrtstar"):
                pooled: list[list[float]] = []
                for pol in policies:
                    keys, x, y, excluded = paired_values(
                        by_label[pol], by_label[comp], metric, planner
                    )
                    if not keys:
                        continue
                    est = paired_bootstrap(x, y)
                    row = {
                        "metric": metric,
                        "policy": pol,
                        "comparator": comp,
                        "planner": planner,
                        "n": len(keys),
                        "mean_diff": est.value,
                        "ci_low": est.low,
                        "ci_high": est.high,
                        "wilcoxon_p": wilcoxon_p(x, y),
                        "rank_biserial": rank_biserial(x, y),
                    }
                    if excluded:
                        row["excluded_undefined"] = excluded
                    family.append(row)
                    pooled.append([a - b for a, b in zip(x, y, strict=True)])
                if pooled:
                    out["pooled"].append(
                        {
                            "metric": metric,
                            "comparator": comp,
                            "planner": planner,
                            "policies": len(pooled),
                            **two_level_bootstrap(pooled),
                        }
                    )
        for row, adj in zip(family, holm([r["wilcoxon_p"] for r in family]), strict=True):
            row["holm_p"] = adj
        out["paired"].extend(family)
    return out


def algorithm_bootstrap(
    x: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Algorithm X − algorithm Y on paired instances (Issue #6, ``05`` B.3).

    ``x[p][i]`` is the metric of X's policy (training seed) ``p`` on instance ``i``, likewise
    ``y``; all policies share the instance order. Each replicate resamples the instances once
    (shared, so the pairing is kept) and the training seeds of each algorithm independently. A
    non-learned comparator is an algorithm with one "policy". Returns the mean difference and the
    percentile 95 % CI.
    """
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    if ax.ndim != 2 or ay.ndim != 2 or ax.shape[1] != ay.shape[1]:
        raise ValueError("x and y must be (policies, instances) arrays over the same instances")
    rng = np.random.default_rng(seed)
    n = ax.shape[1]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        inst = rng.integers(0, n, n)
        px = rng.integers(0, ax.shape[0], ax.shape[0])
        py = rng.integers(0, ay.shape[0], ay.shape[0])
        boots[b] = ax[np.ix_(px, inst)].mean() - ay[np.ix_(py, inst)].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(ax.mean() - ay.mean()), "low": float(lo), "high": float(hi)}


def _fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Issue #5 development comparison (validation block B; not E5)",
        "",
        "Differences are MADQL − comparator per paired instance; positive is better for",
        "completion and base return, negative is better for conflicts per 100 robot-steps.",
        "",
        "## Pooled summaries",
        "",
        "| controller / planner | level | episodes | completion | base return | conflicts/100 |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in analysis["summaries"].items():
        for lvl, b in s.items():
            lines.append(
                f"| {name} | {lvl} | {b['episodes']} | {_fmt(b['completion'])} | "
                f"{_fmt(b['base_return'])} | {_fmt(b['conflicts_per_100'], 2)} |"
            )
    lines += ["", "## Paired comparisons per policy", "",
              "| metric | policy | vs | planner | n | mean diff | 95 % CI | Wilcoxon p | Holm p "
              "| r |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in analysis["paired"]:
        lines.append(
            f"| {r['metric']} | {r['policy']} | {r['comparator']} | {r['planner']} | {r['n']} | "
            f"{_fmt(r['mean_diff'])} | [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | "
            f"{r['wilcoxon_p']:.2g} | {r['holm_p']:.2g} | {r['rank_biserial']:.2f} |"
        )
    lines += ["", "## Pooled over policies (two-level bootstrap)", "",
              "| metric | vs | planner | policies | mean diff | 95 % CI |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for r in analysis["pooled"]:
        lines.append(
            f"| {r['metric']} | {r['comparator']} | {r['planner']} | {r['policies']} | "
            f"{_fmt(r['mean'])} | [{_fmt(r['low'])}, {_fmt(r['high'])}] |"
        )
    return "\n".join(lines) + "\n"


def run(policies: Sequence[Path], out: Path, workers: int) -> dict[str, Any]:
    if (out / "records.jsonl").exists():
        raise FileExistsError(f"{out} already holds a comparison; block B is evaluated once")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"block": "B", "policies": {}}
    tasks = tasks_for("baseline", "baseline") + tasks_for("reference", "reference")
    for path in policies:
        meta = json.loads(path.with_suffix(".json").read_text("utf-8"))
        label = f"madql-seed{meta['seeds']['run_seed']}"
        manifest["policies"][label] = {
            "checkpoint": str(path),
            "sha256": meta["sha256"],
            "config_hash": meta["config_hash"],
            "env_steps": meta["env_steps"],
            "block_a_validation": meta.get("validation"),
        }
        tasks += tasks_for("madql", label, str(path))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    records = run_tasks(tasks, workers)
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    analysis = analyse(records)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2), "utf-8")
    (out / "summary.md").write_text(report(analysis), "utf-8")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robocoord.marl.devcompare")
    ap.add_argument("--policy", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    run(args.policy, args.out, args.workers)
    # The summary contains "−"; a redirected Windows console (cp1252) cannot encode it, so
    # unencodable characters are replaced on output (the stored summary.md stays exact UTF-8).
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")
    print((args.out / "summary.md").read_text("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
