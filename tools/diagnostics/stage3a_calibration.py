"""Issue 6A Phase 3, Stage 3a — calibration of the Execution Supervisor from stored block-B episodes.

Data boundary: **block B only** (seeds 10 020–10 099); the stored greedy learned-controller
episodes, nothing else:

* A\\*: the 2 160 greedy records of ``results/dev6/stochastic/records.jsonl`` (9 policies × 3
  levels × 80 seeds; the Phase 0b population);
* RRT\\*: the 180 MADQL records of ``results/dev5/comparison/records.jsonl`` and the 360 PPO
  records of ``results/dev6/ppo/records.jsonl`` (seeds 10 020–10 039).

Nothing from blocks C, D, S or the test block is read. No new experiment: every stored episode is
**replayed with the supervisor OFF** with the production components unchanged (as Phase 0b:
``evaluation._system``, ``Engine`` with shield off and ``path_limit`` 6, the stored ``best.pt``),
and the replay is checked field by field against the stored record. Mismatches are reported,
nothing is excluded silently.

Shadow mode (observation only): an ``ExecutionSupervisor("S+O")`` with unreachable trigger
thresholds sees every step; its output is asserted equal to the policy's proposal (it never acts),
and after each step its per-robot state — the implementation's own leg, progress and excused-step
rules — is read. Oscillation patterns and would-be right-of-way contests are computed from the same
trajectory with the module's pure functions (``is_oscillating``, ``right_of_way``,
``robot_priorities``).

Per robot-leg (a constant ``(task, mode, goal)`` interval) the replay records the peak counted
no-progress run, the first leg offset at which each candidate stall window ``W`` would have fired,
the first offset at which each candidate oscillation pattern ``(K, cells, m)`` would have fired,
the longest would-be R-loss streak, the remaining goal distance, and the outcome: ``completed``
(the robot was on the goal when the leg ended), ``reassigned`` or ``open`` (still running at the
end). Open legs of robots stuck at the end of a failed A\\* episode carry their Phase 6A.0 class.

Selection rules. Design §5 (declared before any data were analysed): the **reference** population
is every robot-leg of a **successful** episode (all released tasks delivered), per controller
family (MADQL, IPPO, MAPPO) × planner; a group needs at least ``MIN_LEGS`` legs (a 99th
percentile is not estimable from fewer).

* ``stall_window`` W: the smallest grid value such that, in every reference group, at most
  ``FP_TARGET`` (1 %) of the legs would have triggered S (the 99th-percentile rule).
* oscillation ``(osc_window K, osc_cells, osc_moves m)``: among the patterns with at most
  ``FP_TARGET`` of reference legs triggering in every group, the highest detection rate on stuck
  **oscillating** A\\* robots (6A.0 class); ties → smaller K, larger m, fewer cells.

Revisions made after the first look at the data, before any value was frozen (all recorded in the
Stage 3a protocol §3, with the literal-rule results kept in ``calibration.json``):

1. **Eligibility** — a reference group also needs successful episodes to be the family's typical
   outcome (share ≥ ``MIN_SUCCESS_SHARE``). MADQL succeeds in 10 / 720 A\\* and 2 / 180 RRT\\*
   episodes; its few successful episodes contain the very stalls S targets (99th percentile of the
   counted run 162 steps), so no grid value can satisfy the rule there (``literal_rule``).
2. **Per planner** — the rules are applied separately to A\\* and RRT\\* (the design's 3a
   population was A\\* only; RRT\\* plans are not shortest paths, so following them yields counted
   no-progress steps and about 2.5× the trigger rate at the same W).
3. **O must be faster than S** — a pattern is a candidate only if K < W (otherwise S fires first
   and O adds nothing).
4. **Extended O grid for A\\*** (``O_GRID_EXT``, K ∈ {14, 16, 20}), replayed after no pattern of
   the first grid met the target under A\\*.

Stability: an episode-cluster bootstrap (``BOOT`` replicates, seed ``BOOT_SEED``) re-applies the
rules to resampled reference episodes. ``hold``, ``progress_cells``, ``stay_limit`` and
``fairness_losses`` (and the resolver) govern behaviour that exists only with the supervisor ON,
so OFF data cannot calibrate them; the descriptive evidence (remaining distances, would-be loss
streaks, a burst projection) is reported against them.

Usage::

    python tools/diagnostics/stage3a_calibration.py replay  --out results/dev6a/phase3/stage3a --workers 10
    python tools/diagnostics/stage3a_calibration.py replay-o-ext --out results/dev6a/phase3/stage3a --workers 11
    python tools/diagnostics/stage3a_calibration.py analyse --out results/dev6a/phase3/stage3a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

A_STAR = Path("results/dev6/stochastic/records.jsonl")
RRT_MADQL = Path("results/dev5/comparison/records.jsonl")
RRT_PPO = Path("results/dev6/ppo/records.jsonl")
CLASSES_6A0 = Path("results/dev6a/6a0_diagnostics/replay_episodes.jsonl")
BLOCK_B = range(10_020, 10_100)
COMPARE = (
    "delivered",
    "released",
    "steps",
    "terminated",
    "conflicts",
    "conflicts_total",
    "physical_collisions",
    "wait_ratio",
    "planner_calls",
    "mean_flow_time",
)
W_GRID = (4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)
O_GRID = tuple(
    (k, c, m)
    for k in (4, 6, 8, 10, 12)
    for c in (2, 3)
    for m in sorted({math.ceil(k / 2), math.ceil(3 * k / 4)})
)
#: Extension of the oscillation grid, replayed for A\* only after the first grid had no
#: feasible pattern under A\* (recorded in the Stage 3a protocol, §3.3).
O_GRID_EXT = tuple(
    (k, c, m)
    for k in (14, 16, 20)
    for c in (2, 3)
    for m in sorted({math.ceil(k / 2), math.ceil(3 * k / 4)})
)
H_GRID = (4, 8, 12, 16)
FP_TARGET = 0.01
MIN_LEGS = 100  # a group enters the rules only with >= 100 reference legs
BOOT = 2000
BOOT_SEED = 20_261_121
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
NEVER = 10**9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def o_key(k: int, c: int, m: int) -> str:
    return f"{k}|{c}|{m}"


def population() -> list[dict[str, Any]]:
    """The stored greedy learned-controller block-B records (A\\* and RRT\\*)."""
    sources: tuple[tuple[Path, Callable[[dict[str, Any]], bool]], ...] = (
        (A_STAR, lambda r: r["mode"] == "greedy" and r["controller"] in ("madql", "ppo")),
        (RRT_MADQL, lambda r: r["controller"] == "madql" and r["planner"] == "rrtstar"),
        (RRT_PPO, lambda r: r["planner"] == "rrtstar"),
    )
    out = []
    for path, keep in sources:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if keep(r):
                    if r["seed"] not in BLOCK_B or r.get("block", "B") != "B":
                        raise SystemExit(f"non-block-B record in {path}: {r['seed']}")
                    out.append(r)
    return sorted(out, key=lambda r: (r["planner"], r["label"], r["level"], r["seed"]))


def classes_6a0() -> dict[tuple[str, str, int], dict[int, str]]:
    out: dict[tuple[str, str, int], dict[int, str]] = {}
    with CLASSES_6A0.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["label"], r["level"], int(r["seed"]))] = {
                int(rb["robot"]): str(rb["class"]) for rb in r["robots"]
            }
    return out


# --------------------------------------------------------------------------- pure helpers


def first_osc(
    hist: Sequence[tuple[int, int]], since_progress: Sequence[int], k: int, cells: int, m: int
) -> int | None:
    """First leg offset ``j`` at which the supervisor's O trigger would fire for ``(k, cells,
    m)``: no progress for ``>= k`` steps and :func:`is_oscillating` over positions ``j-k..j``."""
    from robocoord.control.supervisor import is_oscillating

    for j in range(k, len(hist)):
        if since_progress[j] >= k and is_oscillating(hist[j - k : j + 1], k, cells, m):
            return j
    return None


def first_at_least(values: Sequence[int], w: int) -> int | None:
    return next((j for j, v in enumerate(values) if v >= w), None)


def family(label: str) -> str:
    return label.split("-seed")[0]


def dist(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        **{f"q{q * 100:g}": float(np.quantile(a, q)) for q in QUANTILES},
    }


# --------------------------------------------------------------------------- replay worker


def replay(
    job: tuple[dict[str, Any], dict[int, str] | None]
    | tuple[dict[str, Any], dict[int, str] | None, tuple[tuple[int, int, int], ...]],
) -> dict[str, Any]:
    from robocoord.allocation.benefits import robot_priorities
    from robocoord.control.supervisor import (
        ExecutionSupervisor,
        SupervisorParams,
        filtered_suggestion,
        right_of_way,
    )
    from robocoord.core.types import Action
    from robocoord.marl.evaluation import EvalTask, _system
    from robocoord.marl.scenarios import LEVELS
    from robocoord.sim.engine import Engine

    stored, stuck = job[0], job[1]
    o_grid = job[2] if len(job) == 3 else O_GRID
    label = stored["label"]
    task = EvalTask(
        stored["controller"],
        stored["level"],
        stored["seed"],
        planner=stored["planner"],
        block="B",
        checkpoint=f"results/training/{label.replace('-', '_')}/best.pt",
        label=label,
    )
    coord, ctrl, extras = _system(task)
    eng = Engine(
        LEVELS[task.level].sample(task.seed), coord, shield=None, view_extras=extras, path_limit=6
    )
    ctrl.reset(len(eng.robots))
    n = len(eng.robots)
    shadow = ExecutionSupervisor(
        "S+O", SupervisorParams(stall_window=NEVER, osc_window=NEVER, osc_moves=1)
    )
    legs: list[dict[str, Any]] = []
    cur: dict[int, dict[str, Any]] = {}
    streak = [0] * n
    contest_steps = 0
    contest_robot_steps = 0
    t = 0
    while not eng.done:
        view = eng.prepare_step()
        acts = ctrl.act(view)
        prop = [acts[i] for i in range(n)]
        out, ev = shadow.apply(view, prop)
        if out != prop or ev:
            raise AssertionError("the shadow supervisor acted")
        prio = robot_priorities(view.robots, view.tasks, view.t, 0.01)
        _, lost, _ = right_of_way(
            [r.position for r in view.robots], prop, {i: (False, prio[i], 0, -i) for i in range(n)}
        )
        contest_steps += bool(lost)
        contest_robot_steps += len(lost)
        cells = [r.position for r in view.robots]
        obstacles = {o.position: o.velocity for o in view.obstacles}
        for r in view.robots:
            i = r.id
            st = shadow._robots[i]
            streak[i] = streak[i] + 1 if i in lost else 0
            c = cur.get(i)
            if c is not None and c["key"] != st.leg:
                c["end"] = t
                c["status"] = "completed" if r.position == c["key"][2] else "reassigned"
                legs.append(cur.pop(i))
                c = None
            if st.leg is not None and c is None:
                c = cur[i] = {"key": st.leg, "start": t, "hist": [], "since": [], "counter": []}
                c["lose_max"] = 0
                c["dev"] = []
                c["excused"] = 0
            if c is not None:
                filt, _ = filtered_suggestion(view, i, cells, obstacles, 3)
                c["dev"].append(int(filt is not Action.STAY and prop[i] != filt))
                c["excused"] += int(filt is Action.STAY)
                c["hist"].append(r.position)
                c["since"].append(view.t - st.last_progress_t)
                c["counter"].append(st.counter)
                c["lose_max"] = max(c["lose_max"], streak[i])
                c["best"] = st.best
        eng.apply_actions(acts)
        t += 1
    for i in sorted(cur):
        c = cur[i]
        c["end"] = t
        c["status"] = "open"
        c["stuck_class"] = (stuck or {}).get(i)
        legs.append(c)
    m = eng.summary().to_dict()
    planning = m["extra"].get("planning", {})
    replayed = {
        "delivered": m["tasks_delivered"],
        "released": m["tasks_released"],
        "steps": m["steps"],
        "terminated": eng.terminated,
        "conflicts": m["conflicts"],
        "conflicts_total": m["conflicts_total"],
        "physical_collisions": m["physical_collisions"],
        "wait_ratio": m["wait_ratio"],
        "planner_calls": planning.get("calls"),
        "mean_flow_time": m["mean_flow_time"],
    }
    mismatch = [f for f in COMPARE if f in stored and replayed[f] != stored[f]]
    out_legs = []
    for c in legs:
        first_o = {
            o_key(k, cl, mm): first_osc(c["hist"], c["since"], k, cl, mm) for k, cl, mm in o_grid
        }
        out_legs.append(
            {
                "status": c["status"],
                "length": c["end"] - c["start"],
                "stuck_class": c.get("stuck_class"),
                "max_counter": max(c["counter"]),
                "first_w": {str(w): first_at_least(c["counter"], w) for w in W_GRID},
                "first_o": first_o,
                "would_modify": {
                    f"{w}|{h}": sum(c["dev"][j : j + h])
                    for w in W_GRID
                    if (j := first_at_least(c["counter"], w)) is not None
                    for h in H_GRID
                },
                "would_modify_o": {
                    f"{key}|{h}": sum(c["dev"][j : j + h])
                    for key, j in first_o.items()
                    if j is not None
                    for h in H_GRID
                },
                "deviating_steps": sum(c["dev"]),
                "excused_steps": c["excused"],
                "lose_max": c["lose_max"],
                "remaining": c["best"],
            }
        )
    return {
        "label": label,
        "controller": stored["controller"],
        "planner": stored["planner"],
        "level": stored["level"],
        "seed": stored["seed"],
        "mismatch": mismatch,
        "failed": replayed["released"] > replayed["delivered"],
        "robots": n,
        "steps": replayed["steps"],
        "contest_steps": contest_steps,
        "contest_robot_steps": contest_robot_steps,
        "legs": out_legs,
    }


# --------------------------------------------------------------------------- analysis


PLANNERS = ("astar", "rrtstar")
MIN_SUCCESS_SHARE = 0.5  # successful episodes must be the family's typical outcome (§3.2)
STUCK_CLASSES = ("frozen", "oscillating", "other_moving", "conflict_loop", "deadlock_cycle")


class _Group:
    """Per-episode hit counts of one reference group (vectorised for the bootstrap)."""

    def __init__(self, eps: Sequence[Mapping[str, Any]], o_keys: Sequence[str]) -> None:
        self.episodes = len(eps)
        self.legs = np.array([len(e["legs"]) for e in eps], dtype=np.int64)
        self.w_hits = np.array(
            [
                [sum(lg["first_w"][str(w)] is not None for lg in e["legs"]) for w in W_GRID]
                for e in eps
            ],
            dtype=np.int64,
        ).reshape(len(eps), len(W_GRID))
        self.w_mods = np.array(
            [
                [sum(bool(lg["would_modify"].get(f"{w}|8", 0)) for lg in e["legs"]) for w in W_GRID]
                for e in eps
            ],
            dtype=np.int64,
        ).reshape(len(eps), len(W_GRID))
        self.o_keys = list(o_keys)
        self.o_hits = np.array(
            [[sum(lg["first_o"][k] is not None for lg in e["legs"]) for k in o_keys] for e in eps],
            dtype=np.int64,
        ).reshape(len(eps), len(o_keys))
        self.o_mods = np.array(
            [
                [
                    sum(bool(lg["would_modify_o"].get(f"{k}|8", 0)) for lg in e["legs"])
                    for k in o_keys
                ]
                for e in eps
            ],
            dtype=np.int64,
        ).reshape(len(eps), len(o_keys))

    def rates(self, idx: npt.NDArray[np.int64] | None = None) -> dict[str, npt.NDArray[np.float64]]:
        sel = slice(None) if idx is None else idx
        n = self.legs[sel].sum()
        return {
            "w": self.w_hits[sel].sum(axis=0) / n,
            "w_mod": self.w_mods[sel].sum(axis=0) / n,
            "o": self.o_hits[sel].sum(axis=0) / n,
            "o_mod": self.o_mods[sel].sum(axis=0) / n,
        }


def _merge_ext(eps: list[dict[str, Any]], ext: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """Add the extended A\\* oscillation grid to the A\\* legs (same deterministic replay, so the
    legs align one to one; checked). Returns the A\\* pattern keys."""
    keys = [o_key(*x) for x in O_GRID]
    if not ext:
        return keys
    by = {(e["label"], e["level"], e["seed"]): e for e in ext}
    for e in eps:
        if e["planner"] != "astar":
            continue
        x = by[(e["label"], e["level"], e["seed"])]
        if x["mismatch"] or len(x["legs"]) != len(e["legs"]):
            raise SystemExit(f"extension replay does not align: {e['label']} {e['seed']}")
        for lg, xl in zip(e["legs"], x["legs"], strict=True):
            if (lg["status"], lg["length"]) != (xl["status"], xl["length"]):
                raise SystemExit("extension replay legs differ")
            lg["first_o"] = {**lg["first_o"], **xl["first_o"]}
            lg["would_modify_o"] = {**lg["would_modify_o"], **xl["would_modify_o"]}
    return keys + [o_key(*x) for x in O_GRID_EXT]


def _pattern(key: str) -> tuple[int, int, int]:
    k, c, m = (int(v) for v in key.split("|"))
    return k, c, m


def _select_planner(
    groups: Mapping[str, _Group],
    rates: Mapping[str, Mapping[str, npt.NDArray[np.float64]]],
    o_keys: Sequence[str],
    osc_detect: Mapping[str, float],
) -> tuple[int | None, str | None]:
    """The two rules for one planner on its eligible groups (rates per group)."""
    if not groups:
        return None, None
    ok_w = np.all([rates[g]["w"] <= FP_TARGET for g in groups], axis=0)
    w_sel = next((w for w, ok in zip(W_GRID, ok_w, strict=True) if ok), None)
    ok_o = np.all([rates[g]["o"] <= FP_TARGET for g in groups], axis=0)
    cands = [
        k
        for k, ok in zip(o_keys, ok_o, strict=True)
        if ok and (w_sel is None or _pattern(k)[0] < w_sel)
    ]
    if not cands:
        return w_sel, None
    best = max(
        cands,
        key=lambda k: (osc_detect.get(k, 0.0), -_pattern(k)[0], _pattern(k)[2], -_pattern(k)[1]),
    )
    return w_sel, best


def analyse(
    eps: list[dict[str, Any]], ext: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    o_keys_astar = _merge_ext(eps, ext)
    o_keys = {"astar": o_keys_astar, "rrtstar": [o_key(*x) for x in O_GRID]}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in eps:
        by_group[f"{family(e['label'])}|{e['planner']}"].append(e)
    stuck: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for e in eps:
        for lg in e["legs"]:
            if lg["status"] == "open" and lg.get("stuck_class"):
                stuck[f"{family(e['label'])}|{lg['stuck_class']}"].append(lg)
                stuck[f"all|{lg['stuck_class']}"].append(lg)
    stuck = dict(sorted(stuck.items()))
    # ---- reference groups and their eligibility
    groups: dict[str, _Group] = {}
    elig: dict[str, dict[str, Any]] = {}
    for g, es in sorted(by_group.items()):
        planner = g.split("|")[1]
        succ = [e for e in es if not e["failed"]]
        groups[g] = _Group(succ, o_keys[planner])
        n_legs = int(groups[g].legs.sum())
        share = len(succ) / len(es)
        elig[g] = {
            "episodes": len(es),
            "successful_episodes": len(succ),
            "success_share": share,
            "reference_legs": n_legs,
            "eligible": n_legs >= MIN_LEGS and share >= MIN_SUCCESS_SHARE,
        }
    rates = {g: grp.rates() for g, grp in groups.items()}
    osc = stuck.get("all|oscillating", [])
    osc_detect = (
        {k: sum(lg["first_o"].get(k) is not None for lg in osc) / len(osc) for k in o_keys_astar}
        if osc
        else {}
    )
    selected: dict[str, Any] = {}
    use = {
        p: {g: groups[g] for g in groups if g.endswith(p) and elig[g]["eligible"]} for p in PLANNERS
    }
    for p in PLANNERS:
        w, o = _select_planner(use[p], rates, o_keys[p], osc_detect)
        selected[p] = {
            "stall_window": w,
            "osc": None
            if o is None
            else dict(zip(("osc_window", "osc_cells", "osc_moves"), _pattern(o), strict=True)),
        }
    # ---- the literal rule (every group with >= MIN_LEGS legs, no success-share condition)
    literal = {}
    for p in PLANNERS:
        lit = {
            g: groups[g] for g in groups if g.endswith(p) and elig[g]["reference_legs"] >= MIN_LEGS
        }
        w, o = _select_planner(lit, rates, o_keys[p], osc_detect)
        literal[p] = {"groups": sorted(lit), "stall_window": w, "osc": o}
    # ---- the modifying-trigger variant (sensitivity)
    mod_variant = {}
    for p in PLANNERS:
        ok = np.all([rates[g]["w_mod"] <= FP_TARGET for g in use[p]], axis=0) if use[p] else []
        mod_variant[p] = next((w for w, k in zip(W_GRID, ok, strict=True) if k), None)
    # ---- bootstrap of the selection (episodes resampled within each eligible group)
    rng = np.random.default_rng(BOOT_SEED)
    tally: dict[str, dict[str, Counter[str]]] = {
        p: {"w": Counter(), "o": Counter()} for p in PLANNERS
    }
    for _ in range(BOOT):
        r_b = {
            g: grp.rates(rng.integers(0, grp.episodes, grp.episodes))
            for g, grp in groups.items()
            if grp.episodes
        }
        for p in PLANNERS:
            w, o = _select_planner(use[p], r_b, o_keys[p], osc_detect)
            tally[p]["w"][str(w)] += 1
            tally[p]["o"][str(o)] += 1
    boot = {p: {k: dict(sorted(v.items())) for k, v in t.items()} for p, t in tally.items()}
    # ---- tables
    s_table = {}
    for j, w in enumerate(W_GRID):
        hits = {g: [lg for lg in v if lg["first_w"][str(w)] is not None] for g, v in stuck.items()}
        s_table[str(w)] = {
            "trigger_rate_reference": {g: float(rates[g]["w"][j]) for g in groups},
            "modifying_trigger_rate_reference_h8": {g: float(rates[g]["w_mod"][j]) for g in groups},
            "detected_stuck": {g: len(hits[g]) / len(v) for g, v in stuck.items()},
            "latency_leg_offset_stuck": {
                g: dist([lg["first_w"][str(w)] for lg in h]) for g, h in hits.items()
            },
            "steps_left_after_trigger_stuck": {
                g: dist([lg["length"] - lg["first_w"][str(w)] for lg in h]) for g, h in hits.items()
            },
        }
    o_table = {}
    for k in o_keys_astar:
        o_table[k] = {
            "trigger_rate_reference": {
                g: float(rates[g]["o"][groups[g].o_keys.index(k)])
                for g in groups
                if k in groups[g].o_keys
            },
            "modifying_trigger_rate_reference_h8": {
                g: float(rates[g]["o_mod"][groups[g].o_keys.index(k)])
                for g in groups
                if k in groups[g].o_keys
            },
            "detected_stuck": {
                g: sum(lg["first_o"].get(k) is not None for lg in v) / len(v)
                for g, v in stuck.items()
            },
        }
    # ---- O lead time over S on stuck oscillating legs (A* only: the 6A.0 labels are A* only)
    lead = {}
    for p in ("astar",):
        w, o = selected[p]["stall_window"], selected[p]["osc"]
        if w is None or o is None:
            continue
        k = o_key(o["osc_window"], o["osc_cells"], o["osc_moves"])
        d = [
            lg["first_w"][str(w)] - lg["first_o"][k]
            for lg in osc
            if lg["first_w"][str(w)] is not None and lg["first_o"].get(k) is not None
        ]
        lead[p] = {"pattern": k, "stall_window": w, "s_minus_o_first_trigger": dist(d)}
    # ---- burst projection (code-derived hypothesis for progress_cells; A* stuck legs only)
    bursts = {}
    for p in ("astar",):
        w = selected[p]["stall_window"]
        if w is None:
            continue
        rows = {}
        for c in ("frozen", "oscillating", "other_moving"):
            legs = [lg for lg in stuck.get(f"all|{c}", []) if lg["first_w"][str(w)] is not None]
            cover: dict[str, float | None] = {}
            for pc in (1, 2, 4, 8):
                step = min(pc, 8)
                fits = 0
                for lg in legs:
                    n_b = math.ceil((lg["remaining"] or 0) / step)
                    fits += n_b * step + max(n_b - 1, 0) * w <= lg["length"] - lg["first_w"][str(w)]
                cover[str(pc)] = fits / len(legs) if legs else None
            rows[c] = {"legs": len(legs), "share_covered": cover}
        bursts[p] = rows
    ref_legs = {g: [lg for e in by_group[g] if not e["failed"] for lg in e["legs"]] for g in groups}
    desc = {
        "max_counter_reference": {
            g: dist([lg["max_counter"] for lg in v]) for g, v in ref_legs.items()
        },
        "max_counter_stuck": {g: dist([lg["max_counter"] for lg in v]) for g, v in stuck.items()},
        "stuck_legs": {g: len(v) for g, v in stuck.items()},
        "leg_length_reference": {g: dist([lg["length"] for lg in v]) for g, v in ref_legs.items()},
        "excused_share_reference": {
            g: sum(lg["excused_steps"] for lg in v) / max(sum(lg["length"] for lg in v), 1)
            for g, v in ref_legs.items()
        },
        "deviating_share_reference": {
            g: sum(lg["deviating_steps"] for lg in v) / max(sum(lg["length"] for lg in v), 1)
            for g, v in ref_legs.items()
        },
        "remaining_distance_stuck": {
            g: dist([lg["remaining"] for lg in v if lg["remaining"] is not None])
            for g, v in stuck.items()
        },
        "would_lose_streak_reference": {
            g: dist([lg["lose_max"] for lg in v]) for g, v in ref_legs.items()
        },
        "would_lose_streak_all_legs": {
            g: dist([lg["lose_max"] for e in es for lg in e["legs"]])
            for g, es in sorted(by_group.items())
        },
        "would_lose_streak_stuck": {
            g: dist([lg["lose_max"] for lg in v]) for g, v in stuck.items()
        },
        "contest_robot_steps_per_100_robot_steps": {
            g: 100
            * sum(e["contest_robot_steps"] for e in es)
            / sum(e["robots"] * e["steps"] for e in es)
            for g, es in sorted(by_group.items())
        },
    }
    proj = {}
    for g, es in sorted(by_group.items()):
        p = g.split("|")[1]
        w, o = selected[p]["stall_window"], selected[p]["osc"]
        if w is None:
            continue
        ok = None if o is None else o_key(o["osc_window"], o["osc_cells"], o["osc_moves"])
        for cls_ in ("successful", "failed"):
            sub = [e for e in es if e["failed"] == (cls_ == "failed")]
            legs = [lg for e in sub for lg in e["legs"]]
            hit = sum(
                (lg["first_w"][str(w)] is not None)
                or (ok is not None and lg["first_o"].get(ok) is not None)
                for lg in legs
            )
            proj[f"{g}|{cls_}"] = {
                "episodes": len(sub),
                "legs": len(legs),
                "legs_with_a_first_trigger": hit,
                "share_of_legs": hit / len(legs) if legs else None,
            }
    return {
        "episodes": len(eps),
        "episodes_by_planner": {p: sum(e["planner"] == p for e in eps) for p in PLANNERS},
        "mismatching_episodes": [
            {
                "label": e["label"],
                "planner": e["planner"],
                "level": e["level"],
                "seed": e["seed"],
                "fields": e["mismatch"],
            }
            for e in eps
            if e["mismatch"]
        ],
        "rules": {
            "fp_target": FP_TARGET,
            "min_legs": MIN_LEGS,
            "min_success_share": MIN_SUCCESS_SHARE,
            "w_grid": list(W_GRID),
            "o_grid": [list(x) for x in O_GRID],
            "o_grid_ext_astar": [list(x) for x in O_GRID_EXT] if ext else [],
            "boot": BOOT,
            "boot_seed": BOOT_SEED,
        },
        "reference_groups": elig,
        "selected": selected,
        "literal_rule": literal,
        "modifying_trigger_variant_w": mod_variant,
        "bootstrap_selection": boot,
        "o_lead_over_s": lead,
        "s_table": s_table,
        "o_table": o_table,
        "burst_projection": bursts,
        "descriptive": desc,
        "projection_first_triggers": proj,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("replay", "replay-o-ext", "analyse"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N jobs only")
    args = ap.parse_args(argv)
    target = args.out / "replay_episodes.jsonl"
    ext = args.out / "replay_o_ext_astar.jsonl"
    if args.cmd == "replay-o-ext":
        if ext.exists():
            raise SystemExit(f"{ext} exists; the replay is written once")
        cls = classes_6a0()
        jobs_ext = [
            (r, cls.get((r["label"], r["level"], int(r["seed"]))), O_GRID_EXT)
            for r in population()
            if r["planner"] == "astar"
        ]
        print(len(jobs_ext), "A* episodes: extended oscillation grid", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            eps_ext = list(ex.map(replay, jobs_ext, chunksize=1))
        with ext.open("w", encoding="utf-8") as fh:
            for e in eps_ext:
                legs = [
                    {k: lg[k] for k in ("status", "length", "first_o", "would_modify_o")}
                    for lg in e["legs"]
                ]
                fh.write(
                    json.dumps(
                        {
                            **{k: e[k] for k in ("label", "planner", "level", "seed", "mismatch")},
                            "legs": legs,
                        }
                    )
                    + "\n"
                )
        print("mismatching:", sum(bool(e["mismatch"]) for e in eps_ext))
        return 0
    if args.cmd == "replay":
        if target.exists():
            raise SystemExit(f"{target} exists; the replay is written once")
        args.out.mkdir(parents=True, exist_ok=True)
        pop = population()
        cls = classes_6a0()
        jobs = [
            (
                r,
                cls.get((r["label"], r["level"], int(r["seed"])))
                if r["planner"] == "astar"
                else None,
            )
            for r in pop
        ]
        if args.limit is not None:
            jobs = jobs[: args.limit]
        print(len(jobs), "stored block-B episodes to replay (supervisor off, shadow)", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            eps = list(ex.map(replay, jobs, chunksize=1))
        with target.open("w", encoding="utf-8") as fh:
            for e in eps:
                fh.write(json.dumps(e) + "\n")
        inputs = {str(p): sha256(p) for p in (A_STAR, RRT_MADQL, RRT_PPO, CLASSES_6A0)}
        (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1), "utf-8")
        print("episodes", len(eps), "| mismatching:", sum(bool(e["mismatch"]) for e in eps))
        return 0
    with target.open(encoding="utf-8") as fh:
        eps = [json.loads(line) for line in fh]
    ext_eps = None
    if ext.exists():
        with ext.open(encoding="utf-8") as fh:
            ext_eps = [json.loads(line) for line in fh]
    res = analyse(eps, ext_eps)
    res["replay_sha256"] = sha256(target)
    res["replay_o_ext_astar_sha256"] = sha256(ext) if ext.exists() else None
    res["tool_sha256"] = sha256(Path(__file__))
    res["inputs"] = json.loads((args.out / "inputs.json").read_text("utf-8"))
    (args.out / "calibration.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    print(
        json.dumps(
            {"selected": res["selected"], "mismatching": len(res["mismatching_episodes"])}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
