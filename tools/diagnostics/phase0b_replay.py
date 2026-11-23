"""Issue 6A Phase 0b — deterministic replay of the stored greedy block-B learned A* episodes.

Approved scope (2026-11-21): (1) MADQL Q-gap analysis, (2) planner calls by trigger for every
greedy learned A* episode. No new experiment: the population is exactly the 2 160 stored greedy
records of ``results/dev6/stochastic/records.jsonl`` (720 MADQL + 1 440 PPO), replayed with the
production components unchanged, as in the Phase 6A.0 replay (``evaluation._system``, ``Engine``
with shield off and ``path_limit`` 6, the stored ``best.pt``).

Observation only, inside this tool's processes:

* the controller's ``choose`` is wrapped: after the original call (whose result is returned
  unchanged) the Q-network is evaluated once more on the same observations — a stateless,
  gradient-free forward pass — and the Q-values, masks and chosen actions are recorded;
* after the episode the monitor's call log (``coordinator.monitor.calls``) gives every planner
  call's robot, step and trigger, and ``monitor.active_robot_steps`` / ``deviations`` the exact
  denominators of the deviation rate.

Reproduction check (as 6A.0, plus the stored deviation rate): delivered, released, steps,
terminated, conflicts by type, conflicts total, physical collisions, wait ratio, planner calls,
mean flow time and deviation rate must equal the stored record. Every mismatch is written out;
nothing is excluded.

Q-gap of a robot-step: ``Q(STAY) − max_{a ∈ allowed moves} Q(a)`` (undefined when no move is
allowed). Greedy execution is a masked arg-max with ties to the lowest index, and STAY is index 0,
so STAY is chosen iff the gap is ``≥ 0``; at a gap of exactly 0 it is chosen by the tie-break.

State categories (non-idle robot-steps of MADQL episodes):

* ``frozen`` / ``oscillating`` / ``other_stuck`` — the last ``WINDOW`` steps of a robot whose
  6A.0 primary class (``results/dev6a/6a0_diagnostics/replay_episodes.jsonl``) is frozen /
  oscillating / anything else;
* ``moving`` — the robot chose a move and moved, outside those windows (successful movement);
* ``waiting`` — the robot chose STAY or its move was cancelled, outside those windows.

Usage::

    python tools/diagnostics/phase0b_replay.py --out results/dev6a/phase0b --workers 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

STORED = Path("results/dev6/stochastic/records.jsonl")
CLASSES_6A0 = Path("results/dev6a/6a0_diagnostics/replay_episodes.jsonl")
WINDOW = 100  # as 6A.0
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
    "deviation_rate_per_100",
)
TRIGGERS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
CATEGORIES = ("frozen", "oscillating", "other_stuck", "moving", "waiting")
CAT_CODE = {c: i for i, c in enumerate(CATEGORIES)}
#: Near-tie thresholds for |gap| (Q units; the per-step cost of the reward is 0.01).
TAUS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
STAY = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def population() -> list[dict[str, Any]]:
    """The stored greedy learned A* block-B records (replicate 0)."""
    out = []
    with STORED.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if (
                r["mode"] == "greedy"
                and r["controller"] in ("madql", "ppo")
                and r["planner"] == "astar"
            ):
                out.append(r)
    return sorted(out, key=lambda r: (r["label"], r["level"], r["seed"]))


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


def q_gap(q: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]) -> tuple[float, float, int]:
    """``(gap, best move Q, best move)``; the gap is ``nan`` and the move ``-1`` when no move is
    allowed. Ties among moves go to the lowest index, as in ``masked_argmax``."""
    moves = [a for a in range(1, len(q)) if mask[a]]
    if not moves:
        return math.nan, math.nan, -1
    best = max(moves, key=lambda a: (float(q[a]), -a))
    return float(q[STAY]) - float(q[best]), float(q[best]), best


def greedy_choice(q: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]) -> int:
    """Masked arg-max with ties to the lowest index (the production greedy rule)."""
    allowed = [a for a in range(len(q)) if mask[a]]
    return max(allowed, key=lambda a: (float(q[a]), -a))


def category(stuck_class: str | None, in_window: bool, chose_move: bool, moved: bool) -> str:
    if stuck_class is not None and in_window:
        return stuck_class if stuck_class in ("frozen", "oscillating") else "other_stuck"
    return "moving" if chose_move and moved else "waiting"


def dist(values: npt.NDArray[np.float64]) -> dict[str, Any]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0}
    qs = np.quantile(v, QUANTILES)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        **{f"q{q * 100:g}": float(x) for q, x in zip(QUANTILES, qs, strict=True)},
    }


def gap_summary(gap: npt.NDArray[np.float64], chosen: npt.NDArray[np.int64]) -> dict[str, Any]:
    """Distribution, sign shares, near-tie shares and how STAY was chosen."""
    fin = np.isfinite(gap)
    g = gap[fin]
    n = int(g.size)
    out: dict[str, Any] = {
        "robot_steps": int(gap.size),
        "no_allowed_move": int((~fin).sum()),
        "gap": dist(gap),
    }
    if n:
        out["share_gap_gt0"] = float((g > 0).mean())
        out["share_gap_eq0"] = float((g == 0).mean())
        out["share_gap_lt0"] = float((g < 0).mean())
        out["share_abs_gap_le"] = {f"{t:g}": float((np.abs(g) <= t).mean()) for t in TAUS}
        out["share_gap_gt"] = {f"{t:g}": float((g > t).mean()) for t in TAUS}
        stay = chosen[fin] == STAY
        out["stay_chosen"] = int(stay.sum())
        out["stay_chosen_by_tie_break"] = int((stay & (g == 0)).sum())
        out["stay_chosen_with_gap_gt"] = {f"{t:g}": int((stay & (g > t)).sum()) for t in TAUS}
    return out


# --------------------------------------------------------------------------- replay worker


def replay(job: tuple[dict[str, Any], dict[int, str] | None]) -> dict[str, Any]:
    import torch

    from robocoord.marl.evaluation import EvalTask, _system
    from robocoord.marl.scenarios import LEVELS
    from robocoord.sim.engine import Engine
    from robocoord.world.entities import RobotMode

    stored, stuck = job
    label = stored["label"]
    run = label.replace("-", "_")
    task = EvalTask(
        stored["controller"],
        stored["level"],
        stored["seed"],
        block="B",
        checkpoint=f"results/training/{run}/best.pt",
        label=label,
    )
    coord, ctrl, extras = _system(task)
    eng = Engine(
        LEVELS[task.level].sample(task.seed), coord, shield=None, view_extras=extras, path_limit=6
    )
    ctrl.reset(len(eng.robots))
    capture = stored["controller"] == "madql"
    rec: list[tuple[Any, Any, Any]] = []
    if capture:
        original = ctrl.choose

        def recording(obs: Any, masks: Any) -> Any:
            acts = original(obs, masks)
            with torch.no_grad():
                q = ctrl.net(torch.from_numpy(np.ascontiguousarray(obs))).numpy()
            rec.append(
                (
                    q.astype(np.float32),
                    np.asarray(masks, bool).copy(),
                    np.asarray(acts, np.int64).copy(),
                )
            )
            return acts

        ctrl.choose = recording
    n = len(eng.robots)
    rows: list[list[Any]] = []
    argmax_mismatch = 0
    k = 0
    while not eng.done:
        view = eng.prepare_step()
        acts = ctrl.act(view)
        idle = [r.mode is RobotMode.IDLE for r in eng.robots]
        pos = [r.position for r in eng.robots]
        eng.apply_actions(acts)
        new = [r.position for r in eng.robots]
        if capture:
            q, masks, chosen = rec[-1]
            for i in range(n):
                if idle[i]:
                    continue
                gap, _, best = q_gap(q[i], masks[i])
                if greedy_choice(q[i], masks[i]) != int(chosen[i]):
                    argmax_mismatch += 1
                sug = int(view.suggested[i]) if view.suggested is not None else -1
                gap_sug = (
                    float(q[i][STAY]) - float(q[i][sug]) if sug > 0 and masks[i][sug] else math.nan
                )
                rows.append([k, i, gap, gap_sug, int(chosen[i]), best, sug, int(pos[i] != new[i])])
        k += 1
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
        "deviation_rate_per_100": planning.get("deviation_rate_per_100"),
    }
    mismatch = [f for f in COMPARE if replayed[f] != stored[f]]
    mon = coord.monitor
    steps = int(m["steps"])
    by_robot: dict[int, Counter[str]] = defaultdict(Counter)
    by_robot_window: dict[int, Counter[str]] = defaultdict(Counter)
    for c in mon.calls:
        by_robot[c.robot][c.trigger] += 1
        if c.t >= steps - WINDOW:
            by_robot_window[c.robot][c.trigger] += 1
    states = None
    if capture:
        arr = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
        cats = np.array(
            [
                CAT_CODE[
                    category(
                        (stuck or {}).get(int(r[1])),
                        r[0] >= steps - WINDOW,
                        int(r[4]) != STAY,
                        bool(r[7]),
                    )
                ]
                for r in arr
            ],
            dtype=np.int8,
        )
        states = {
            "gap": arr[:, 2],
            "gap_sug": arr[:, 3],
            "chosen": arr[:, 4].astype(np.int64),
            "category": cats,
        }
    return {
        "label": label,
        "controller": stored["controller"],
        "level": stored["level"],
        "seed": stored["seed"],
        "mismatch": mismatch,
        "replayed": replayed,
        "calls_by_trigger": dict(planning.get("calls_by_trigger", {})),
        "active_robot_steps": int(mon.active_robot_steps),
        "deviations": int(mon.deviations),
        "robot_steps": n * steps,
        "calls_by_robot": {str(i): dict(v) for i, v in sorted(by_robot.items())},
        "calls_by_robot_last_window": {str(i): dict(v) for i, v in sorted(by_robot_window.items())},
        "stuck_classes": {str(i): c for i, c in sorted((stuck or {}).items())},
        "argmax_mismatch": argmax_mismatch,
        "states": states,
    }


# --------------------------------------------------------------------------- analysis


def trigger_tables(eps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exact calls by trigger per controller / policy / level, and per stuck class (last window)."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for e in eps:
        alg = e["label"].split("-seed")[0]
        for key in (
            f"{alg}|all|all",
            f"{alg}|{e['label']}|all",
            f"{alg}|all|{e['level']}",
            f"{alg}|{e['label']}|{e['level']}",
        ):
            groups[key].append(e)
    out: dict[str, Any] = {}
    for key in sorted(groups):
        es = groups[key]
        c: Counter[str] = Counter()
        for e in es:
            c.update(e["calls_by_trigger"])
        total = sum(c.values())
        rsteps = sum(e["robot_steps"] for e in es)
        active = sum(e["active_robot_steps"] for e in es)
        keys = [t for t in TRIGGERS if any(t in e["calls_by_trigger"] for e in es)]
        out[key] = {
            "episodes": len(es),
            "calls": total,
            "counts": {t: c.get(t, 0) for t in keys},
            "shares": {t: c.get(t, 0) / total for t in keys} if total else {},
            "calls_per_100_robot_steps": 100 * total / rsteps,
            "t3_per_100_robot_steps": 100 * c.get("T3", 0) / rsteps,
            "t3_per_100_nonidle_robot_steps": 100 * c.get("T3", 0) / active if active else None,
            "nonidle_share_of_robot_steps": active / rsteps,
            "failed_episodes": sum(
                e["replayed"]["released"] > e["replayed"]["delivered"] for e in es
            ),
        }
    per_class: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    robots: dict[str, Counter[str]] = defaultdict(Counter)
    for e in eps:
        alg = e["label"].split("-seed")[0]
        for rid, cls in e["stuck_classes"].items():
            per_class[alg][cls].update(e["calls_by_robot_last_window"].get(rid, {}))
            robots[alg][cls] += 1
    out_cls: dict[str, dict[str, Any]] = {}
    for alg in sorted(per_class):
        out_cls[alg] = {}
        for cls in sorted(per_class[alg]):
            c = per_class[alg][cls]
            tot = sum(c.values())
            out_cls[alg][cls] = {
                "stuck_robots": robots[alg][cls],
                "calls_last_window": tot,
                "calls_per_robot_per_100_steps": 100 * tot / (robots[alg][cls] * WINDOW),
                "t3_share": c.get("T3", 0) / tot if tot else None,
                "counts": dict(sorted(c.items())),
            }
    return {"groups": out, "stuck_robots_last_window": out_cls}


def qgap_tables(eps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per policy × level × category (and pooled): :func:`gap_summary` of the MADQL states."""
    parts: dict[str, list[tuple[Any, Any, Any]]] = defaultdict(list)
    for e in eps:
        s = e.get("states")
        if s is None:
            continue
        for pol in (e["label"], "all"):
            for lvl in (e["level"], "all"):
                for code, cat in enumerate(CATEGORIES):
                    sel = s["category"] == code
                    parts[f"{pol}|{lvl}|{cat}"].append(
                        (s["gap"][sel], s["gap_sug"][sel], s["chosen"][sel])
                    )
    out = {}
    for key in sorted(parts):
        gap = np.concatenate([p[0] for p in parts[key]])
        sug = np.concatenate([p[1] for p in parts[key]])
        ch = np.concatenate([p[2] for p in parts[key]])
        if gap.size == 0:
            continue
        res = gap_summary(gap, ch)
        res["gap_vs_suggested_move"] = dist(sug)
        res["share_suggested_move_available"] = float(np.isfinite(sug).mean())
        out[key] = res
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    target = args.out / "replay_episodes.jsonl"
    if target.exists():
        raise SystemExit(f"{target} exists; the replay is written once")
    args.out.mkdir(parents=True, exist_ok=True)
    pop = population()
    cls = classes_6a0()
    jobs = [(r, cls.get((r["label"], r["level"], int(r["seed"])))) for r in pop]
    print(len(jobs), "stored greedy learned A* episodes to replay", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        eps = list(ex.map(replay, jobs, chunksize=2))
    with target.open("w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps({k: v for k, v in e.items() if k != "states"}) + "\n")
    madql = [e for e in eps if e["states"] is not None]
    np.savez_compressed(
        args.out / "madql_states.npz",
        episode=np.concatenate([np.full(e["states"]["gap"].size, i) for i, e in enumerate(madql)]),
        gap=np.concatenate([e["states"]["gap"] for e in madql]),
        gap_sug=np.concatenate([e["states"]["gap_sug"] for e in madql]),
        chosen=np.concatenate([e["states"]["chosen"] for e in madql]),
        category=np.concatenate([e["states"]["category"] for e in madql]),
        labels=np.array([f"{e['label']}|{e['level']}|{e['seed']}" for e in madql]),
    )
    analysis = {
        "inputs": {str(STORED): sha256(STORED), str(CLASSES_6A0): sha256(CLASSES_6A0)},
        "episodes": len(eps),
        "mismatching_episodes": [
            {"label": e["label"], "level": e["level"], "seed": e["seed"], "fields": e["mismatch"]}
            for e in eps
            if e["mismatch"]
        ],
        "argmax_mismatches": sum(e["argmax_mismatch"] for e in eps),
        "failed_madql_without_6a0_classes": sum(
            1
            for e, j in zip(eps, jobs, strict=True)
            if e["controller"] == "madql"
            and e["replayed"]["released"] > e["replayed"]["delivered"]
            and j[1] is None
        ),
        "triggers": trigger_tables(eps),
        "qgap": qgap_tables(madql),
        "definitions": {
            "window": WINDOW,
            "taus": list(TAUS),
            "categories": list(CATEGORIES),
            "gap": "Q(STAY) - max over allowed moves of Q; STAY chosen iff gap >= 0",
        },
    }
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=1, sort_keys=True), "utf-8")
    print(
        "episodes",
        len(eps),
        "| mismatching:",
        len(analysis["mismatching_episodes"]),
        "| argmax mismatches:",
        analysis["argmax_mismatches"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
