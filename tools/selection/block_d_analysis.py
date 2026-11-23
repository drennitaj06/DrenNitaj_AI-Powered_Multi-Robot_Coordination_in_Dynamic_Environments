"""Block D final evaluation of H-L1 — the analysis (Issue 6A; protocol revision 2, approved).

Protocol: ``docs/research/2026-11-21-block-d-protocol.md``. This module, the runner
``block_d.py`` and the protocol are hashed by ``block_d.py freeze`` before the first scenario
generation of the formal Block D execution; the runner refuses every later step if a hash differs.

**Statistical method — the frozen Block C method, unchanged.** The estimator, its standard error,
the bootstrap and Holm are imported from the frozen ``block_c_analysis.py`` (SHA-256 pinned in
:data:`BLOCK_C_ANALYSIS_SHA256`): ``paired_ratio`` (ratio of sums over seeds, paired delta-method
SE), ``bootstrap_t`` (symmetric seed-cluster bootstrap-t), ``boot_index`` (one ``(B, n)`` index
matrix from ``default_rng(0)``, shared by every arm of a planner) and ``holm``. The seed is the
unit; a seed's levels are summed before resampling; old and new are resampled together.

What differs from Block C (protocol §5–§7), and only that:

* three families with their own family-wise α — **F1** (A\\*, completion and conflicts per
  episode, 8 hypotheses) at 0.03, **F2** (A\\*, conflicts per delivered task, per 100
  robot-steps, base return, 12) at 0.01, **F3** (RRT\\*, completion and conflicts per episode,
  8) at 0.01; union bound 0.05. Decisions use Holm-adjusted p-values only;
* every interval is the **nominal 95 %** per-hypothesis bootstrap-t interval, labelled
  unadjusted (``bootstrap_t`` with α = 0.05); it is never a simultaneous interval;
* an **MC-sensitivity flag**: an evaluable hypothesis whose p lies within 2 Monte Carlo SE of
  its own Holm threshold (``α_F / (m − rank)``) is flagged;
* a run × planner with any unusable episode (failed, missing, duplicate, invalid) is **not
  evaluable**: its hypotheses stay in their family (``m`` unchanged) with ``p = 1``, are never
  rejected, and **no estimate, interval or p-value is computed or reported for them** — no
  complete-case analysis anywhere. Descriptive units (D2, D4) with an unusable episode are
  reported as counts only;
* the validity rules of Block C plus one: an episode that is neither terminated nor at its
  horizon is invalid.

Descriptive outputs (no p-values, nominal intervals only): per-level paired effects, the RRT\\*
versions of the F2 outcomes, each v5 candidate minus the matched baseline (same estimator, the
baseline in the "old" position), point summaries per arm, episode classes, safety counts,
conflict types and runtime.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).parent))
import block_c_analysis as bca

F = npt.NDArray[np.float64]

#: The frozen Block C analysis whose statistical functions are reused (protocol §5).
BLOCK_C_ANALYSIS = Path(__file__).parent / "block_c_analysis.py"
BLOCK_C_ANALYSIS_SHA256 = "3fe86a7795885ea17b445e3a24a8503cada31a2fa2da2bd85203e9d4560ebfec"

BLOCK_D = range(12_000, 12_200)
BLOCK_D_RRT = range(12_000, 12_060)
SEEDS: dict[str, range] = {"astar": BLOCK_D, "rrtstar": BLOCK_D_RRT}
PLANNERS = ("astar", "rrtstar")
LEVELS = bca.LEVELS
PAIR_RUNS = bca.RUNS  # mappo_seed1–3, ippo_seed1: the pick changed
UNCHANGED_RUNS = ("ippo_seed2", "ippo_seed3")
BASELINE = "matched_baseline"
PAIR_ARMS = ("old", "new")
V5_ARM = "v5"
BASELINE_ARM = "baseline"

N_BOOT = bca.N_BOOT  # 10 000
BOOT_SEED = bca.BOOT_SEED  # 0
CI_ALPHA = 0.05  # nominal, per hypothesis, unadjusted
FAMILY_ALPHA = {"F1": 0.03, "F2": 0.01, "F3": 0.01}
FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "F1": ("astar", bca.PRIMARY),
    "F2": ("astar", bca.SECONDARY),
    "F3": ("rrtstar", bca.PRIMARY),
}
MC_K = 2.0
OUTCOMES = bca.OUTCOMES
ALL_OUTCOMES = (*bca.PRIMARY, *bca.SECONDARY)

#: Condition shared by every record (protocol §2).
COMMON: dict[str, Any] = {
    "block": "D",
    "shield": "off",
    "allocator": "auction",
    "mode": "greedy",
    "replicate": 0,
    "rrt_iterations": 1000,
}
CONTROLLER = {"old": "ppo", "new": "ppo", V5_ARM: "ppo", BASELINE_ARM: "baseline-noresolver"}
#: Fields that must agree between the old and the new episode of one instance.
PAIRED = (*COMMON, "controller", "planner", "label", "level", "seed", "robots", "max_steps",
          "instance_sha256")  # fmt: skip
REQUIRED_OK = (*bca.COUNTS, "terminated", "max_steps", "base_return", "conflicts")
TIMING = ("wall_s", "act_s", "prepare_s")


class ConfigError(ValueError):
    """A record contradicts the frozen design: the whole analysis is aborted."""


def verify_frozen_dependency(path: Path = BLOCK_C_ANALYSIS) -> None:
    """Refuse to analyse if the reused Block C analysis is not the frozen file."""
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != BLOCK_C_ANALYSIS_SHA256:
        raise ConfigError(f"{path} has SHA-256 {h}, not the frozen {BLOCK_C_ANALYSIS_SHA256}")


# --------------------------------------------------------------------------- design


@dataclass(frozen=True)
class Design:
    """Checkpoint file names per run: the four pairs and the two unchanged candidates."""

    pairs: Mapping[str, Mapping[str, str]]
    unchanged: Mapping[str, str]

    def units(self, planner: str) -> list[tuple[str, str]]:
        """(unit, arm) cells evaluated with ``planner`` (protocol §2)."""
        out = [(run, arm) for run in self.pairs for arm in PAIR_ARMS]
        if planner == "astar":
            out += [(run, V5_ARM) for run in self.unchanged]
        return [*out, (BASELINE, BASELINE_ARM)]

    def arm_of(self, r: Mapping[str, Any]) -> tuple[str, str]:
        label, ck = str(r.get("label")), r.get("checkpoint")
        if label in self.pairs:
            for arm in PAIR_ARMS:
                if ck == self.pairs[label][arm]:
                    return str(label), arm
        elif label in self.unchanged and ck == self.unchanged[label]:
            return str(label), V5_ARM
        elif label == BASELINE and ck is None:
            return BASELINE, BASELINE_ARM
        raise ConfigError(f"record of an unexpected run/checkpoint {label!r}/{ck!r}")


# --------------------------------------------------------------------------- gate


def episode_problem(r: Mapping[str, Any]) -> str | None:
    """Why an episode is unusable (``None`` for a usable one: successful or incomplete)."""
    if r.get("status") != "ok":
        return f"failed: {r.get('error_type')}"
    missing = [k for k in REQUIRED_OK if r.get(k) is None]
    if missing:
        return f"invalid: missing {missing}"
    why = bca.record_problem(r)
    if why is not None:
        return f"invalid: {why}"
    if not r["terminated"] and r["steps"] != r["max_steps"]:
        return "invalid: neither terminated nor at the horizon"
    return None


def episode_class(r: Mapping[str, Any]) -> str:
    """successful / incomplete / horizon_all_delivered, or the unusable reason's class."""
    why = episode_problem(r)
    if why is not None:
        return why.split(":")[0]
    if r["terminated"]:
        return "successful"
    return "incomplete" if r["delivered"] < r["released"] else "horizon_all_delivered"


Key = tuple[str, str, str, str, int]  # planner, unit, arm, level, seed


@dataclass
class Data:
    design: Design
    arrays: dict[tuple[str, str, str], dict[str, F]]  # (planner, unit, arm) -> field -> (L, S)
    usable: dict[tuple[str, str], bool]  # (unit, planner)
    problems: dict[tuple[str, str], list[str]] = field(default_factory=dict)


def build(
    records: Iterable[Mapping[str, Any]],
    design: Design,
    fingerprints: Mapping[str, Mapping[int, str]],
    seeds: Mapping[str, Sequence[int]] | None = None,
) -> Data:
    """Apply the validity gate (protocol §7) and arrange the usable episodes."""
    seeds = seeds or {p: tuple(r) for p, r in SEEDS.items()}
    seen: dict[Key, Mapping[str, Any]] = {}
    problems: dict[tuple[str, str], list[str]] = {
        (u, p): [] for p in PLANNERS for u in {u for u, _ in design.units(p)}
    }
    level_set = set(LEVELS)
    for r in records:
        unit, arm = design.arm_of(r)
        planner = r.get("planner")
        if planner not in PLANNERS or (unit, arm) not in design.units(str(planner)):
            raise ConfigError(f"{unit}/{arm}: planner {planner!r} is not in the design")
        for fld, v in {**COMMON, "controller": CONTROLLER[arm]}.items():
            if r.get(fld) != v:
                raise ConfigError(f"{unit}/{arm}: {fld}={r.get(fld)!r}, expected {v!r}")
        if r.get("arm") != arm:
            raise ConfigError(f"{unit}: record arm {r.get('arm')!r} != {arm!r}")
        lvl, s = r.get("level"), r.get("seed")
        if lvl not in level_set or s not in set(seeds[str(planner)]):
            raise ConfigError(f"{unit}/{arm}/{planner}: level/seed {lvl}/{s} outside the design")
        assert isinstance(lvl, str) and isinstance(s, int)
        if r.get("instance_sha256") != fingerprints[lvl][s]:
            raise ConfigError(f"{planner}/{unit}/{arm}/{lvl}/{s}: instance differs from the check")
        key: Key = (str(planner), unit, arm, lvl, s)
        if key in seen:
            problems[unit, key[0]].append(f"duplicate {arm}/{lvl}/{s}")
            continue
        seen[key] = r
    bad: set[Key] = set()
    for planner in PLANNERS:
        for unit, arm in design.units(planner):
            for lvl in LEVELS:
                for s in seeds[planner]:
                    cell: Key = (planner, unit, arm, lvl, s)
                    rec = seen.get(cell)
                    if rec is None:
                        problems[unit, planner].append(f"missing {arm}/{lvl}/{s}")
                    elif (why := episode_problem(rec)) is not None:
                        problems[unit, planner].append(f"{why} ({arm}/{lvl}/{s})")
                    else:
                        continue
                    bad.add(cell)
        for run in design.pairs:
            for lvl in LEVELS:
                for s in seeds[planner]:
                    a, b = (seen.get((planner, run, arm, lvl, s)) for arm in PAIR_ARMS)
                    if a is not None and b is not None:
                        diff = [f for f in PAIRED if a.get(f) != b.get(f)]
                        if diff:
                            raise ConfigError(
                                f"{planner}/{run}/{lvl}/{s}: old/new differ in {diff}"
                            )
    arrays: dict[tuple[str, str, str], dict[str, F]] = {}
    li = {lv: i for i, lv in enumerate(LEVELS)}
    for planner in PLANNERS:
        si = {s: j for j, s in enumerate(seeds[planner])}
        for unit, arm in design.units(planner):
            a = {k: np.zeros((len(LEVELS), len(si))) for k in bca.FIELDS}
            for lvl in LEVELS:
                for s in seeds[planner]:
                    cell = (planner, unit, arm, lvl, s)
                    if cell in bad or cell not in seen:
                        continue
                    r = seen[cell]
                    i, j = li[lvl], si[s]
                    a["delivered"][i, j] = r["delivered"]
                    a["released"][i, j] = r["released"]
                    a["conflicts"][i, j] = r["conflicts_total"]
                    a["robot_steps"][i, j] = r["robots"] * r["steps"]
                    a["base_return"][i, j] = r["base_return"]
                    a["episodes"][i, j] = 1.0
            arrays[planner, unit, arm] = a
    usable = {k: not v for k, v in problems.items()}
    return Data(design=design, arrays=arrays, usable=usable, problems=problems)


# --------------------------------------------------------------------------- statistics


def _args(data: Data, planner: str, ref: tuple[str, str], arm: tuple[str, str], outcome: str,
          levels: Sequence[int]) -> list[F]:  # fmt: skip
    num, den, scale = OUTCOMES[outcome]
    out: list[F] = []
    for unit, a in (ref, arm):
        arr = data.arrays[planner, unit, a]
        out += [scale * arr[num][list(levels)].sum(0), arr[den][list(levels)].sum(0)]
    return out


def _effect(args: list[F], idx: npt.NDArray[np.int64], with_p: bool) -> dict[str, Any]:
    inf = bca.bootstrap_t(bca.paired_ratio, args, idx, CI_ALPHA).as_dict()
    old = float(args[0].sum() / args[1].sum())
    new = float(args[2].sum() / args[3].sum())
    res = {"ref": old, "arm": new, "relative_change": (new - old) / old if old else None, **inf}
    res["ci95_nominal"] = res.pop("ci95")
    res["ci95_percentile_nominal"] = res.pop("ci95_percentile")
    if not with_p:
        del res["p"], res["p_mc_se"]
    return res


def holm_thresholds(p: Sequence[float], alpha: float) -> list[float]:
    """Each hypothesis' own Holm threshold ``α / (m − rank)`` (rank by p, then position)."""
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    thr = [0.0] * m
    for rank, i in enumerate(order):
        thr[i] = alpha / (m - rank)
    return thr


def analyse(data: Data, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    """The pre-specified analysis (protocol §5–§7)."""
    d = data.design
    n_seeds = {p: next(iter(data.arrays[p, BASELINE, BASELINE_ARM].values())).shape[1]
               for p in PLANNERS}  # fmt: skip
    idx = {p: bca.boot_index(n_seeds[p], n_boot, seed) for p in PLANNERS}
    all_lv = list(range(len(LEVELS)))

    families: dict[str, list[dict[str, Any]]] = {}
    for fam, (planner, outs) in FAMILIES.items():
        rows: list[dict[str, Any]] = []
        for run in d.pairs:
            ok = data.usable[run, planner]
            for outcome in outs:
                row: dict[str, Any] = {"family": fam, "planner": planner, "run": run,
                                       "outcome": outcome, "evaluable": ok}  # fmt: skip
                if ok:
                    args = _args(data, planner, (run, "old"), (run, "new"), outcome, all_lv)
                    eff = _effect(args, idx[planner], with_p=True)
                    eff["old"], eff["new"] = eff.pop("ref"), eff.pop("arm")
                    row.update(eff)
                    row["p_for_holm"] = row["p"]
                else:
                    row["p_for_holm"] = 1.0
                rows.append(row)
        alpha = FAMILY_ALPHA[fam]
        p = [r["p_for_holm"] for r in rows]
        for r, adj, thr in zip(rows, bca.holm(p), holm_thresholds(p, alpha), strict=True):
            r["alpha_family"] = alpha
            r["holm_threshold"] = thr
            r["p_holm"] = adj
            r["reject"] = bool(r["evaluable"] and adj <= alpha)
            r["mc_sensitive"] = bool(r["evaluable"] and abs(r["p"] - thr) <= MC_K * r["p_mc_se"])
        families[fam] = rows

    per_level: list[dict[str, Any]] = []
    rrt_supportive: list[dict[str, Any]] = []
    for planner in PLANNERS:
        for run in d.pairs:
            if not data.usable[run, planner]:
                continue
            for li, lvl in enumerate(LEVELS):
                for outcome in ALL_OUTCOMES:
                    args = _args(data, planner, (run, "old"), (run, "new"), outcome, [li])
                    e = _effect(args, idx[planner], with_p=False)
                    per_level.append({"planner": planner, "run": run, "level": lvl,
                                      "outcome": outcome, **e})  # fmt: skip
            if planner == "rrtstar":
                for outcome in bca.SECONDARY:
                    args = _args(data, planner, (run, "old"), (run, "new"), outcome, all_lv)
                    e = _effect(args, idx[planner], with_p=False)
                    rrt_supportive.append({"run": run, "outcome": outcome, **e})

    vs_baseline: list[dict[str, Any]] = []
    for planner in PLANNERS:
        cands = [(run, "new") for run in d.pairs]
        if planner == "astar":
            cands += [(run, V5_ARM) for run in d.unchanged]
        for run, arm in cands:
            ok = data.usable[run, planner] and data.usable[BASELINE, planner]
            for scope, lv in (("all", all_lv), *((n, [i]) for i, n in enumerate(LEVELS))):
                for outcome in ALL_OUTCOMES:
                    row = {"planner": planner, "run": run, "arm": arm, "level": scope,
                           "outcome": outcome, "data_complete": ok}  # fmt: skip
                    if ok:
                        args = _args(data, planner, (BASELINE, BASELINE_ARM), (run, arm),
                                     outcome, lv)  # fmt: skip
                        e = _effect(args, idx[planner], with_p=False)
                        e["baseline"], e["candidate"] = e.pop("ref"), e.pop("arm")
                        row.update(e)
                    vs_baseline.append(row)

    summaries: list[dict[str, Any]] = []
    for planner in PLANNERS:
        for unit, arm in d.units(planner):
            ok = data.usable[unit, planner]
            arr = data.arrays[planner, unit, arm]
            for scope, lv in (("all", all_lv), *((n, [i]) for i, n in enumerate(LEVELS))):
                row = {"planner": planner, "unit": unit, "arm": arm, "level": scope,
                       "data_complete": ok}  # fmt: skip
                if ok:
                    for outcome, (num, den, scale) in OUTCOMES.items():
                        row[outcome] = float(scale * arr[num][lv].sum() / arr[den][lv].sum())
                summaries.append(row)

    return {
        "families": families,
        "per_level": per_level,
        "rrtstar_supportive_outcomes": rrt_supportive,
        "vs_matched_baseline": vs_baseline,
        "summaries": summaries,
        "usable": {f"{u}|{p}": v for (u, p), v in data.usable.items()},
        "problems": {f"{u}|{p}": v for (u, p), v in data.problems.items() if v},
        "config": {
            "n_boot": n_boot,
            "boot_seed": seed,
            "quantile_method": bca.QUANTILE_METHOD,
            "ci_alpha_nominal": CI_ALPHA,
            "family_alpha": FAMILY_ALPHA,
            "mc_k": MC_K,
            "n_seeds": n_seeds,
            "levels": list(LEVELS),
            "block_c_analysis_sha256": BLOCK_C_ANALYSIS_SHA256,
        },
        "labels": {
            "intervals": "nominal 95 % per hypothesis, unadjusted for multiplicity; not "
            "simultaneous. Inferential decisions use Holm-adjusted p only.",
            "mc_sensitive": "p within 2 Monte Carlo SE of its own Holm threshold; the formal "
            "decision stands but must be qualified.",
        },
    }


# --------------------------------------------------------------------------- descriptives


def descriptives(records: Iterable[Mapping[str, Any]], design: Design) -> list[dict[str, Any]]:
    """Episode classes, safety, conflict types and runtime per planner × unit × arm × level."""
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in records:
        unit, arm = design.arm_of(r)
        for lvl in (r["level"], "all"):
            k = (str(r["planner"]), unit, arm, lvl)
            row = rows.setdefault(k, {
                "planner": k[0], "unit": unit, "arm": arm, "level": lvl,
                "checkpoint": r.get("checkpoint"), "episodes": 0, "classes": {},
                "error_types": {}, "physical_collisions": 0, "invalid_moves_INV": 0,
                "undelivered_tasks": 0, "conflict_types": {}, "_t": {t: [] for t in TIMING},
                "_flow": [], "_calls": 0, "_robot_steps": 0,
            })  # fmt: skip
            row["episodes"] += 1
            cls = episode_class(r)
            row["classes"][cls] = row["classes"].get(cls, 0) + 1
            if r.get("status") != "ok":
                et = str(r.get("error_type"))
                row["error_types"][et] = row["error_types"].get(et, 0) + 1
                continue
            if cls not in ("successful", "incomplete", "horizon_all_delivered"):
                continue
            row["physical_collisions"] += r["physical_collisions"]
            row["invalid_moves_INV"] += r["conflicts"].get("INV", 0)
            row["undelivered_tasks"] += r["released"] - r["delivered"]
            for ct, v in r["conflicts"].items():
                row["conflict_types"][ct] = row["conflict_types"].get(ct, 0) + v
            for t in TIMING:
                if isinstance(r.get(t), (int, float)):
                    row["_t"][t].append(float(r[t]))
            if r.get("mean_flow_time") is not None:
                row["_flow"].append(r["mean_flow_time"])
            row["_calls"] += r.get("planner_calls") or 0
            row["_robot_steps"] += r["robots"] * r["steps"]
    out = []
    for row in rows.values():
        t = row.pop("_t")
        row["runtime"] = {k: ({"median": float(np.median(v)),
                               "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
                              if v else None) for k, v in t.items()}  # fmt: skip
        flow = row.pop("_flow")
        row["mean_flow_time_mean"] = float(np.mean(flow)) if flow else None
        calls, steps = row.pop("_calls"), row.pop("_robot_steps")
        row["planner_calls_per_100_robot_steps"] = 100 * calls / steps if steps else None
        out.append(row)
    order = {lv: i for i, lv in enumerate((*LEVELS, "all"))}
    return sorted(out, key=lambda r: (r["planner"], r["unit"], r["arm"], order[r["level"]]))
