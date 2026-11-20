"""Issue #6 development experiments on validation block B (not E5/E6; those are Issue #8).

Each experiment writes its own directory (``manifest.json``, ``records.jsonl``) and is run once;
historical Issue #5 records (``results/dev5/comparison/records.jsonl``) are **read only** and
referenced by path and SHA-256, never copied over or rewritten. Experiments (approved Issue #6
design, Revision 2):

* ``matched-baseline`` — ``baseline-noresolver`` (resolver off; auction, endpoint on, shield off),
  A* 240 + RRT* 60 episodes: the controller-level comparator (D11);
* ``ppo`` — each selected MAPPO / IPPO policy, A* 240 + RRT* 60 (as MADQL in Issue #5);
* ``ablation`` — each Option-C reward-ablation policy, A* 240 (D3);
* ``shield`` — every learned policy on the full level (80 A* seeds, 20 RRT* seeds) with the
  predictive shield under both allocators, and shield off under the sequential allocator (the
  auction / shield-off arm is the main comparison);
* ``stochastic`` — every learned policy, A* 240 instances, greedy once and stochastic three times
  (seeded per instance and replicate, independent of the policy; D6), plus a field-by-field
  determinism check of the fresh greedy MADQL episodes against the historical records.

``analyse`` combines the stored records into ``results/dev6/analysis.json`` and ``summary.md``.
Every number comes from the stored records.

Usage::

    python -m robocoord.marl.study ppo --policy results/training/mappo_seed1/best.pt … \\
        --out results/dev6/ppo --workers 10
    python -m robocoord.marl.study analyse --root results/dev6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robocoord.experiments.stats import holm, paired_bootstrap, rank_biserial, wilcoxon_p
from robocoord.marl.devcompare import (
    METRICS_ISSUE6,
    algorithm_bootstrap,
    analyse,
    load_records,
    metric_value,
    tasks_for,
)
from robocoord.marl.evaluation import EvalTask, run_tasks
from robocoord.marl.scenarios import BLOCK_B, BLOCK_B_RRT, LEVELS, check_seeds

HISTORICAL = Path("results/dev5/comparison/records.jsonl")
REPLICATES = 3
#: Fields excluded from the determinism comparison (timing).
TIMING = ("wall_s",)


# ------------------------------------------------------------------------------------ helpers
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_label(checkpoint: Path) -> tuple[str, str, dict[str, Any]]:
    """``(controller, label, sidecar)``: ``madql-seedN``, ``mappo-seedN``, ``ippo-seedN`` or an
    ablation name such as ``mappo_zeta0-seedN``."""
    meta = json.loads(checkpoint.with_suffix(".json").read_text("utf-8"))
    seed = meta["seeds"]["run_seed"]
    if meta["algorithm"] == "madql":
        return "madql", f"madql-seed{seed}", meta
    name = meta.get("config", {}).get("name") or meta["variant"]
    return "ppo", f"{name}-seed{seed}", meta


def manifest_entry(checkpoint: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "sha256": meta["sha256"],
        "config_hash": meta["config_hash"],
        "env_steps": meta["env_steps"],
        "algorithm": meta["algorithm"],
        "variant": meta.get("variant"),
        "block_a_validation": meta.get("validation"),
    }


def _start(out: Path, manifest: dict[str, Any]) -> None:
    if (out / "records.jsonl").exists():
        raise FileExistsError(f"{out} already holds records; block B experiments run once")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


def _finish(out: Path, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    recs = list(records)
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def full_level_tasks(
    controller: str, label: str, checkpoint: str | None, **kw: Any
) -> list[EvalTask]:
    a = [EvalTask(controller, "full", s, "astar", "B", checkpoint, label=label, **kw)
         for s in check_seeds(BLOCK_B, "B")]  # fmt: skip
    r = [EvalTask(controller, "full", s, "rrtstar", "B", checkpoint, label=label, **kw)
         for s in check_seeds(BLOCK_B_RRT, "B")]  # fmt: skip
    return a + r


def astar_tasks(controller: str, label: str, checkpoint: str | None, **kw: Any) -> list[EvalTask]:
    return [EvalTask(controller, lvl, s, "astar", "B", checkpoint, label=label, **kw)
            for lvl in LEVELS for s in check_seeds(BLOCK_B, "B")]  # fmt: skip


# ------------------------------------------------------------------------------------ runs
def run_matched_baseline(out: Path, workers: int) -> list[dict[str, Any]]:
    manifest = {"experiment": "matched-baseline", "block": "B",
                "controller": "baseline-noresolver",
                "composition": {"allocator": "auction", "endpoint": True, "shield": "off",
                                "deadlock_resolver": False},
                "historical_baseline": _historical()}  # fmt: skip
    _start(out, manifest)
    return _finish(out, run_tasks(tasks_for("baseline-noresolver", "baseline-noresolver"), workers))


def _historical() -> dict[str, Any] | None:
    if not HISTORICAL.exists():
        return None
    return {"records": str(HISTORICAL), "sha256": sha256(HISTORICAL)}


def run_policies(
    out: Path, policies: Sequence[Path], workers: int, *, experiment: str
) -> list[dict[str, Any]]:
    manifest: dict[str, Any] = {"experiment": experiment, "block": "B", "policies": {}}
    tasks: list[EvalTask] = []
    for path in policies:
        controller, label, meta = policy_label(path)
        manifest["policies"][label] = manifest_entry(path, meta)
        if experiment == "ppo":
            tasks += tasks_for(controller, label, str(path))
        elif experiment == "ablation":
            tasks += astar_tasks(controller, label, str(path))
        elif experiment == "shield":
            for allocator, shield in (("auction", "predictive"), ("sequential", "off"),
                                      ("sequential", "predictive")):  # fmt: skip
                tasks += full_level_tasks(controller, label, str(path), allocator=allocator,
                                          shield=shield)  # fmt: skip
        elif experiment == "stochastic":
            tasks += astar_tasks(controller, label, str(path))
            for rep in range(1, REPLICATES + 1):
                tasks += astar_tasks(controller, label, str(path), mode="stochastic",
                                     replicate=rep)  # fmt: skip
        else:
            raise ValueError(f"unknown experiment {experiment!r}")
    if experiment == "stochastic":
        manifest["replicates"] = REPLICATES
        manifest["madql_epsilon"] = 0.05
        manifest["historical_greedy_madql"] = _historical()
    _start(out, manifest)
    recs = _finish(out, run_tasks(tasks, workers))
    if experiment == "stochastic" and HISTORICAL.exists():
        check = determinism_check(recs, load_records(HISTORICAL))
        (out / "determinism_check.json").write_text(json.dumps(check, indent=2), "utf-8")
    return recs


def determinism_check(
    fresh: Sequence[dict[str, Any]], historical: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare fresh greedy MADQL episodes with the historical block-B records field by field
    (the historical fields only, timing excluded). Historical files are never modified."""
    hist = {(r["label"], r["planner"], r["level"], r["seed"]): r for r in historical
            if r["label"].startswith("madql")}  # fmt: skip
    compared, mismatches = 0, []
    for r in fresh:
        if not r["label"].startswith("madql") or r.get("mode") != "greedy":
            continue
        h = hist.get((r["label"], r["planner"], r["level"], r["seed"]))
        if h is None:
            continue
        compared += 1
        diff = [k for k in h if k not in TIMING and r.get(k) != h[k]]
        if diff:
            mismatches.append({"key": [r["label"], r["planner"], r["level"], r["seed"]],
                               "fields": diff})  # fmt: skip
    return {"compared": compared, "mismatches": len(mismatches), "details": mismatches[:50],
            "excluded_fields": list(TIMING)}  # fmt: skip


# ------------------------------------------------------------------------------------ analysis
SUMMARY_FIELDS = ("steps", "mean_flow_time", "makespan", "conflicts_total", "conflicts_per_100",
                  "wait_ratio", "base_return", "shaped_return", "shield_interventions",
                  "planner_calls", "prepare_s", "act_s", "wall_s")  # fmt: skip


def summary_block(rs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled completion and conflicts per delivered task, means of the per-episode fields
    (``None`` when no record has the field, e.g. Issue #5 records)."""
    released = sum(int(r["released"]) for r in rs)
    delivered = sum(int(r["delivered"]) for r in rs)
    conflicts = sum(int(r["conflicts_total"]) for r in rs)
    out: dict[str, Any] = {
        "episodes": len(rs),
        "released": released,
        "delivered": delivered,
        "undelivered": released - delivered,
        "completion": delivered / released if released else None,
        "conflicts_per_delivered_pooled": conflicts / delivered if delivered else None,
        "terminated": sum(bool(r["terminated"]) for r in rs),
        "physical_collisions": sum(int(r["physical_collisions"]) for r in rs),
    }
    for f in SUMMARY_FIELDS:
        vals = [float(r[f]) for r in rs if r.get(f) is not None]
        out[f] = float(np.mean(vals)) if vals else None
    return out


def summaries(records: Sequence[dict[str, Any]], group: Sequence[str]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by["/".join(str(r.get(g, "")) for g in group)].append(r)
    out: dict[str, Any] = {}
    for key, rs in sorted(by.items()):
        out[key] = {"overall": summary_block(rs)}
        for lvl in LEVELS:
            sub = [r for r in rs if r["level"] == lvl]
            if sub:
                out[key][lvl] = summary_block(sub)
    return out


def _value(r: dict[str, Any], metric: str) -> float:
    v = metric_value(r, metric)
    if v is None:
        raise ValueError(f"{metric} is undefined for this record")
    return v


def _defined(
    groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    labels: Iterable[str],
    key: tuple[str, str, int],
    metric: str,
) -> bool:
    return all(metric_value(groups[lab][key], metric) is not None for lab in labels)


def _by_instance(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(r["planner"], r["level"], int(r["seed"])): r for r in records}


def algorithm_level(
    records: Sequence[dict[str, Any]],
    metrics: Sequence[str] = ("completion", "base_return", "conflicts_total"),
) -> list[dict[str, Any]]:
    """Two-level bootstrap comparisons between algorithms (training seeds resampled per
    algorithm, instances shared) and against the non-learned controllers. No p-values: with three
    training seeds per algorithm they would not be meaningful (``05`` B.3)."""
    labels: dict[str, list[str]] = defaultdict(list)
    for lab in sorted({r["label"] for r in records}):
        algo = lab.split("-seed")[0]
        labels[algo].append(lab)
    by_label = {lab: _by_instance(r for r in records if r["label"] == lab)
                for algos in labels.values() for lab in algos}  # fmt: skip
    learned = [a for a in ("mappo", "ippo", "madql") if a in labels]
    comparators = [c for c in ("baseline", "baseline-noresolver", "reference") if c in labels]
    pairs: list[tuple[str, str]] = [(a, b) for i, a in enumerate(learned) for b in learned[i + 1 :]]
    pairs += [(a, c) for a in learned for c in comparators]
    rows = []
    for metric in metrics:
        for planner in ("astar", "rrtstar"):
            for a, b in pairs:
                pols = labels[a] + labels[b]
                keys = sorted(set.intersection(*(
                    {k for k in by_label[p] if k[0] == planner} for p in pols)))  # fmt: skip
                keys = [k for k in keys if _defined(by_label, pols, k, metric)]
                if not keys:
                    continue
                x = [[_value(by_label[p][k], metric) for k in keys] for p in labels[a]]
                y = [[_value(by_label[p][k], metric) for k in keys] for p in labels[b]]
                rows.append({"metric": metric, "planner": planner, "a": a, "b": b,
                             "policies_a": len(x), "policies_b": len(y), "instances": len(keys),
                             "per_seed_a": [float(np.mean(v)) for v in x],
                             "per_seed_b": [float(np.mean(v)) for v in y],
                             **algorithm_bootstrap(x, y)})  # fmt: skip
    return rows


def paired_rows(
    groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    comparisons: Sequence[tuple[str, str, str]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Per-policy paired comparisons ``x − y`` with Holm within each metric family."""
    out: list[dict[str, Any]] = []
    for metric in metrics:
        family = []
        for name, gx, gy in comparisons:
            for planner in ("astar", "rrtstar"):
                keys = sorted(
                    k for k in groups.get(gx, {}) if k[0] == planner and k in groups.get(gy, {})
                )
                ok = [k for k in keys if _defined(groups, (gx, gy), k, metric)]
                if len(ok) < 2:
                    continue
                x = [_value(groups[gx][k], metric) for k in ok]
                y = [_value(groups[gy][k], metric) for k in ok]
                est = paired_bootstrap(x, y)
                family.append({"metric": metric, "comparison": name, "x": gx, "y": gy,
                               "planner": planner, "n": len(x),
                               "excluded_undefined": len(keys) - len(ok),
                               "mean_diff": est.value, "ci_low": est.low, "ci_high": est.high,
                               "wilcoxon_p": wilcoxon_p(x, y),
                               "rank_biserial": rank_biserial(x, y)})  # fmt: skip
        for row, adj in zip(family, holm([float(r["wilcoxon_p"]) for r in family]), strict=True):
            row["holm_p"] = adj
        out.extend(family)
    return out


def stochastic_groups(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[tuple[str, str, int], dict[str, Any]]]:
    """``label/greedy`` and ``label/stochastic`` instance maps; the stochastic replicates of an
    instance are averaged first (the instance is the analysis unit)."""
    groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = defaultdict(dict)
    reps: dict[tuple[str, tuple[str, str, int]], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r["planner"], r["level"], int(r["seed"]))
        if r.get("mode", "greedy") == "greedy":
            groups[f"{r['label']}/greedy"][key] = r
        else:
            reps[(r["label"], key)].append(r)
    for (label, key), rs in reps.items():
        avg: dict[str, Any] = {"label": label, "planner": key[0], "level": key[1], "seed": key[2]}
        for f in (
            "completion",
            "base_return",
            "conflicts_total",
            "delivered",
            "released",
            "wait_ratio",
            "steps",
        ):
            if all(x.get(f) is not None for x in rs):
                avg[f] = float(np.mean([float(x[f]) for x in rs]))
        avg["replicates"] = len(rs)
        groups[f"{label}/stochastic"][key] = avg
    return groups


ABLATION_CELLS = ("mappo_zeta0", "mappo_beta0", "mappo_crr05", "mappo_crr2")


def ablation_effects(
    groups: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    metrics: Sequence[str] = ("completion", "base_return", "conflicts_total"),
) -> list[dict[str, Any]]:
    """Each Option-C cell against the default cell, balanced on run seeds 1 and 2: a two-level
    bootstrap over the two training seeds of each cell and the shared block-B instances."""
    rows = []
    for cell in ABLATION_CELLS:
        labels = [f"{cell}-seed1", f"{cell}-seed2", "mappo-seed1", "mappo-seed2"]
        if not all(lab in groups for lab in labels):
            continue
        keys = sorted(set.intersection(*(set(groups[lab]) for lab in labels)))
        for m in metrics:
            ks = [k for k in keys if _defined(groups, labels, k, m)]
            if not ks:
                continue
            x = [[_value(groups[lab][k], m) for k in ks] for lab in labels[:2]]
            y = [[_value(groups[lab][k], m) for k in ks] for lab in labels[2:]]
            rows.append({"cell": cell, "metric": m, "instances": len(ks),
                         "per_seed_cell": [float(np.mean(v)) for v in x],
                         "per_seed_default": [float(np.mean(v)) for v in y],
                         **algorithm_bootstrap(x, y)})  # fmt: skip
    return rows


def analyse_all(root: Path) -> dict[str, Any]:
    """Combine every stored Issue #6 experiment (and the historical Issue #5 records)."""
    hist = load_records(HISTORICAL) if HISTORICAL.exists() else []
    out: dict[str, Any] = {"historical": _historical()}
    parts = {name: load_records(root / name / "records.jsonl")
             for name in ("baseline_matched", "ppo", "ablation", "shield", "stochastic")
             if (root / name / "records.jsonl").exists()}  # fmt: skip
    # primary comparison: MADQL, baseline and reference from Issue #5 (read-only), PPO and the
    # matched baseline from Issue #6; all greedy, auction, shield off, block B
    primary = [*hist, *parts.get("baseline_matched", []), *parts.get("ppo", [])]
    if primary:
        out["primary"] = {
            "summaries": summaries(primary, ("label", "planner")),
            "per_policy": analyse(primary, metrics=METRICS_ISSUE6),
            "algorithm_level": algorithm_level(primary),
        }
    if "ablation" in parts:
        defaults = ("mappo-seed1", "mappo-seed2")
        main_seeds = [
            r for r in parts.get("ppo", []) if r["label"] in defaults and r["planner"] == "astar"
        ]
        recs = [*parts["ablation"], *main_seeds]
        groups = {lab: _by_instance(r for r in recs if r["label"] == lab)
                  for lab in {r["label"] for r in recs}}  # fmt: skip
        comps = []
        for cell in ("mappo_zeta0", "mappo_beta0", "mappo_crr05", "mappo_crr2"):
            for seed in (1, 2):
                comps.append((f"{cell} vs default (seed {seed})", f"{cell}-seed{seed}",
                              f"mappo-seed{seed}"))  # fmt: skip
        out["ablation"] = {
            "summaries": summaries(recs, ("label", "planner")),
            "paired": paired_rows(groups, comps, ("completion", "base_return", "conflicts_total")),
            "cell_vs_default": ablation_effects(groups),
        }
    if "shield" in parts:
        base = [*hist, *parts.get("ppo", [])]
        # Issue #5 records predate the allocator / shield fields: they ran auction, shield off
        off_auction = [{**r, "allocator": r.get("allocator", "auction"),
                        "shield": r.get("shield", "off")}
                       for r in base if r["level"] == "full" and r["label"].startswith(
                           ("madql", "mappo", "ippo"))]  # fmt: skip
        recs = [*parts["shield"], *off_auction]
        groups = defaultdict(dict)
        for r in recs:
            groups[f"{r['label']}/{r.get('allocator', 'auction')}/{r.get('shield', 'off')}"][
                (r["planner"], r["level"], int(r["seed"]))
            ] = r
        labels = sorted({r["label"] for r in parts["shield"]})
        comps = []
        for lab in labels:
            for alloc in ("auction", "sequential"):
                comps.append((f"{lab} {alloc}: predictive − off", f"{lab}/{alloc}/predictive",
                              f"{lab}/{alloc}/off"))  # fmt: skip
        out["shield"] = {
            "summaries": summaries(recs, ("label", "allocator", "shield", "planner")),
            "paired": paired_rows(groups, comps, ("completion", "conflicts_total",
                                                  "conflicts_per_100", "wait_ratio")),
        }  # fmt: skip
    if "stochastic" in parts:
        recs = parts["stochastic"]
        groups = stochastic_groups(recs)
        labels = sorted({r["label"] for r in recs})
        comps = [(f"{lab}: stochastic − greedy", f"{lab}/stochastic", f"{lab}/greedy")
                 for lab in labels]  # fmt: skip
        check = root / "stochastic" / "determinism_check.json"
        out["stochastic"] = {
            "summaries": summaries(recs, ("label", "mode", "planner")),
            "paired": paired_rows(groups, comps, ("completion", "base_return", "conflicts_total")),
            "determinism_check": json.loads(check.read_text("utf-8")) if check.exists() else None,
        }
    if "baseline_matched" in parts and hist:
        orig = _by_instance(r for r in hist if r["label"] == "baseline")
        matched = _by_instance(parts["baseline_matched"])
        out["resolver_effect"] = paired_rows(
            {"baseline": orig, "baseline-noresolver": matched},
            [("baseline (resolver on) − baseline-noresolver", "baseline", "baseline-noresolver")],
            ("completion", "base_return", "conflicts_total", "conflicts_per_100"),
        )
    return out


# ------------------------------------------------------------------------------------ report
def _f(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None else f"{float(v):.{nd}f}"


def _summary_table(title: str, summ: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", "",
             "| group | level | episodes | completion | undelivered | base return | conflicts "
             "(total) | conflicts/100 | conflicts/delivered | wait ratio | collisions |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, blocks in summ.items():
        for lvl, b in blocks.items():
            lines.append(
                f"| {name} | {lvl} | {b['episodes']} | {_f(b['completion'])} | "
                f"{b['undelivered']} | {_f(b['base_return'])} | {_f(b['conflicts_total'], 2)} | "
                f"{_f(b['conflicts_per_100'], 2)} | {_f(b['conflicts_per_delivered_pooled'])} | "
                f"{_f(b['wait_ratio'])} | {b['physical_collisions']} |"
            )
    return [*lines, ""]


def _paired_table(title: str, rows: Sequence[dict[str, Any]], who: str = "comparison") -> list[str]:
    lines = [f"### {title}", "",
             "| metric | comparison | planner | n | mean diff | 95 % CI | Wilcoxon p | Holm p "
             "| r |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        name = r.get(who) or f"{r.get('policy')} − {r.get('comparator')}"
        lines.append(
            f"| {r['metric']} | {name} | {r['planner']} | {r['n']} | {_f(r['mean_diff'])} | "
            f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {r['wilcoxon_p']:.2g} | "
            f"{r['holm_p']:.2g} | {r['rank_biserial']:.2f} |"
        )
    return [*lines, ""]


def _bootstrap_table(title: str, rows: Sequence[dict[str, Any]], head: str) -> list[str]:
    lines = [f"### {title}", "", f"| metric | {head} | instances | per seed (x) | per seed (y) "
             "| mean diff | 95 % CI |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        if "cell" in r:
            name, px, py = f"{r['cell']} − mappo", r["per_seed_cell"], r["per_seed_default"]
        else:
            name, px, py = f"{r['a']} − {r['b']} ({r['planner']})", r["per_seed_a"], r["per_seed_b"]
        lines.append(
            f"| {r['metric']} | {name} | {r['instances']} | "
            f"{', '.join(_f(v) for v in px)} | {', '.join(_f(v) for v in py)} | "
            f"{_f(r['mean'])} | [{_f(r['low'])}, {_f(r['high'])}] |"
        )
    return [*lines, ""]


def report(res: dict[str, Any]) -> str:
    """``summary.md``: every table is rendered from ``analysis.json`` (no other source)."""
    hist = res.get("historical") or {}
    lines = ["# Issue #6 development experiments (validation block B; not E5/E6)", "",
             f"Historical Issue #5 records (read only): `{hist.get('records')}`, "
             f"SHA-256 `{hist.get('sha256')}`.", "",
             "Differences are x − y per paired instance. Higher is better for completion and "
             "base return; lower is better for every conflict metric. Conflicts/100 uses all "
             "robot-steps as its denominator (evaluation records); conflicts/delivered is pooled "
             "over the group.", ""]  # fmt: skip
    if "primary" in res:
        p = res["primary"]
        lines += ["## Primary comparison (greedy, auction, shield off)", ""]
        lines += _summary_table("Summaries", p["summaries"])
        lines += _bootstrap_table("Algorithm level (two-level bootstrap; no p-values)",
                                  p["algorithm_level"], "x − y")  # fmt: skip
        lines += _paired_table("Per policy against the comparators", p["per_policy"]["paired"])
    if "resolver_effect" in res:
        lines += ["## Deadlock resolver effect (non-learned baseline)", ""]
        lines += _paired_table("Resolver on − off", res["resolver_effect"])
    if "ablation" in res:
        a = res["ablation"]
        lines += ["## Reward ablation (Option C, MAPPO, A* only)", ""]
        lines += _summary_table("Summaries", a["summaries"])
        lines += _bootstrap_table("Cell − default (run seeds 1 and 2)", a["cell_vs_default"],
                                  "x − y")  # fmt: skip
        lines += _paired_table("Per seed", a["paired"])
    if "shield" in res:
        lines += ["## Predictive shield (full level)", ""]
        lines += _summary_table("Summaries", res["shield"]["summaries"])
        lines += _paired_table("Predictive − off", res["shield"]["paired"])
    if "stochastic" in res:
        s = res["stochastic"]
        check = s.get("determinism_check") or {}
        lines += ["## Stochastic diagnostic (A*; three replicates averaged per instance)", "",
                  f"Determinism check of fresh greedy MADQL against the historical records: "
                  f"{check.get('compared')} episodes compared, {check.get('mismatches')} "
                  f"mismatches.", ""]  # fmt: skip
        lines += _summary_table("Summaries", s["summaries"])
        lines += _paired_table("Stochastic − greedy", s["paired"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robocoord.marl.study")
    ap.add_argument("experiment", choices=("matched-baseline", "ppo", "ablation", "shield",
                                           "stochastic", "analyse"))  # fmt: skip
    ap.add_argument("--policy", action="append", type=Path, default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--root", type=Path, default=Path("results/dev6"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if args.experiment == "analyse":
        res = analyse_all(args.root)
        (args.root / "analysis.json").write_text(json.dumps(res, indent=2), "utf-8")
        (args.root / "summary.md").write_text(report(res), "utf-8")
        print(f"wrote {args.root / 'analysis.json'} and summary.md")
        return 0
    if args.out is None:
        ap.error("--out is required")
    if args.experiment == "matched-baseline":
        recs = run_matched_baseline(args.out, args.workers)
    else:
        if not args.policy:
            ap.error("--policy is required")
        recs = run_policies(args.out, args.policy, args.workers, experiment=args.experiment)
    print(f"{len(recs)} records written to {args.out / 'records.jsonl'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
