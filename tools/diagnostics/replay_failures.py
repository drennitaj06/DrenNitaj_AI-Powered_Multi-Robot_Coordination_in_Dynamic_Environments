"""Issue 6A.0 — diagnostic replay of greedy block-B failures (read-only analysis tool).

Replays every greedy block-B episode (A*, auction, shield off, endpoint recovery on) that left
tasks undelivered, with the production evaluation components unchanged (``evaluation._system``,
``Engine``, the stored checkpoints), and records per step what the stuck robots did. Nothing in
``results/`` is modified: output goes to a new directory given by ``--out``.

Reproduction check: each replay's episode summary is compared field by field with the stored
greedy record (``results/dev6/stochastic/records.jsonl``, mode ``greedy``).

Usage (repository root)::

    python tools/diagnostics/replay_failures.py --out results/dev6a/6a0_diagnostics --workers 8

Definitions (also written to the summary):

* failed episode: a stored greedy record with ``released - delivered > 0``;
* stuck robot: a robot whose mode at the final step is not idle (it holds a task); its behaviour
  is described over the last ``WINDOW`` steps of the episode by flags (multi-label);
* wait-for graph (as ``planning/deadlock.py``): edge i -> j iff robot i is not idle, has a valid
  reference path with a next cell, and robot j stands on that cell; cycles via the production
  ``find_cycles``; a cycle is a *deadlock* when it is persistent (same members on the same cells
  for ``DETECT_STEPS`` consecutive steps) or recurrent (``RECUR_COUNT``-th new occurrence of the
  same member set within ``RECUR_WINDOW`` steps) — the resolver's default criteria.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

STORED = Path("results/dev6/stochastic/records.jsonl")
WINDOW = 100
DETECT_STEPS = 2  # DetourParams.detect_steps
RECUR_COUNT = 3  # DetourParams.recur_count
RECUR_WINDOW = 20  # DetourParams.recur_window
THRESH = 50  # steps of the window for the conflict / cycle / idle-block / oscillation flags
COMPARE = ("delivered", "released", "steps", "terminated", "conflicts", "conflicts_total",
           "physical_collisions", "wait_ratio", "planner_calls", "mean_flow_time")  # fmt: skip


def failures() -> list[dict[str, Any]]:
    out = []
    with STORED.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["mode"] == "greedy" and r["released"] > r["delivered"]:
                out.append(r)
    return out


def replay(stored: dict[str, Any]) -> dict[str, Any]:
    from robocoord.core.types import Action
    from robocoord.marl.evaluation import EvalTask, _system
    from robocoord.marl.scenarios import LEVELS
    from robocoord.planning.deadlock import find_cycles
    from robocoord.sim.engine import Engine
    from robocoord.world.entities import RobotMode

    label = stored["label"]
    run = label.replace("-", "_")
    task = EvalTask(stored["controller"], stored["level"], stored["seed"], block="B",
                    checkpoint=f"results/training/{run}/best.pt", label=label)  # fmt: skip
    coord, ctrl, extras = _system(task)
    eng = Engine(LEVELS[task.level].sample(task.seed), coord, shield=None, view_extras=extras,
                 path_limit=6)  # fmt: skip
    ctrl.reset(len(eng.robots))
    n = len(eng.robots)
    hist: deque[dict[str, Any]] = deque(maxlen=WINDOW)
    seen: dict[tuple[int, ...], deque[int]] = defaultdict(deque)
    prev_cycles: dict[tuple[int, ...], tuple[Any, int]] = {}
    deadlock_steps_total = Counter()
    while not eng.done:
        view = eng.prepare_step()
        acts = ctrl.act(view)
        pos = [r.position for r in eng.robots]
        idle = [r.mode is RobotMode.IDLE for r in eng.robots]
        cell_of = {p: i for i, p in enumerate(pos)}
        nxt: list[Any] = []
        succ: dict[int, int] = {}
        for i in range(n):
            path = view.reference_paths[i] if view.reference_paths else None
            c = path[1] if (path is not None and len(path) >= 2 and not idle[i]) else None
            nxt.append(c)
            if c is not None and c in cell_of and cell_of[c] != i:
                succ[i] = cell_of[c]
        t = eng.t if hasattr(eng, "t") else len(hist)
        cycles = [tuple(sorted(cy)) for cy in find_cycles(succ)]
        cur: dict[tuple[int, ...], tuple[Any, int]] = {}
        in_deadlock: set[int] = set()
        for mem in cycles:
            cells = tuple(pos[i] for i in mem)
            run_len = (
                prev_cycles[mem][1] + 1
                if mem in prev_cycles and prev_cycles[mem][0] == cells
                else 1
            )
            cur[mem] = (cells, run_len)
            if mem not in prev_cycles:  # a new occurrence
                q = seen[mem]
                q.append(t)
                while q and q[0] <= t - RECUR_WINDOW:
                    q.popleft()
            if run_len >= DETECT_STEPS or len(seen[mem]) >= RECUR_COUNT:
                in_deadlock.update(mem)
        prev_cycles = cur
        for i in in_deadlock:
            deadlock_steps_total[i] += 1
        out = eng.apply_actions(acts)
        new = [r.position for r in eng.robots]
        evs = [e.entities for e in out.events]
        hist.append({
            "pos": pos, "new": new, "idle": idle, "act": [int(acts[i]) for i in range(n)],
            "sug": [int(s) for s in view.suggested] if view.suggested is not None else None,
            "in_cycle": {i for mem in cycles for i in mem}, "deadlock": in_deadlock,
            "conf": {i for ents in evs for i in range(n) if f"r{i}" in ents},
            "next_occ": [None if nxt[i] is None else ("idle" if nxt[i] in cell_of and idle[cell_of[nxt[i]]]
                         else "active" if nxt[i] in cell_of else "free") for i in range(n)],
        })  # fmt: skip
    m = eng.summary().to_dict()
    replayed = {
        "delivered": m["tasks_delivered"], "released": m["tasks_released"], "steps": m["steps"],
        "terminated": eng.terminated, "conflicts": m["conflicts"],
        "conflicts_total": m["conflicts_total"], "physical_collisions": m["physical_collisions"],
        "wait_ratio": m["wait_ratio"], "planner_calls": m["extra"].get("planning", {}).get("calls"),
        "mean_flow_time": m["mean_flow_time"],
    }  # fmt: skip
    mismatch = [k for k in COMPARE if replayed[k] != stored[k]]
    final_idle = [r.mode is RobotMode.IDLE for r in eng.robots]
    robots = []
    for i in range(n):
        if final_idle[i]:
            continue
        h = list(hist)
        moves = sum(x["pos"][i] != x["new"][i] for x in h)
        props = sum(x["act"][i] != int(Action.STAY) for x in h)
        cells = len({x["pos"][i] for x in h})
        conf = sum(i in x["conf"] for x in h)
        cyc = sum(i in x["in_cycle"] for x in h)
        dead = sum(i in x["deadlock"] for x in h)
        idle_block = sum(x["next_occ"][i] == "idle" for x in h)
        nocc = Counter(x["next_occ"][i] for x in h)
        follow = sum(x["sug"] is not None and x["act"][i] == x["sug"][i] for x in h)
        flags = {
            "frozen": moves == 0 and props == 0,
            "oscillating": cells <= 3 and moves >= THRESH,
            "conflict_loop": conf >= THRESH,
            "deadlock_cycle": dead >= THRESH,
            "blocked_by_idle": idle_block >= THRESH,
            "mostly_stationary": moves < 10,
        }
        # revision-2 order (for verification of the revision-2 counts)
        if flags["frozen"]:
            occ = nocc.most_common(1)[0][0]
            rev2 = "frozen:" + {"free": "free", "idle": "robot_idle", "active": "robot_active",
                                None: "suggests_stay"}[occ]  # fmt: skip
        elif flags["oscillating"]:
            rev2 = "oscillating"
        elif flags["conflict_loop"]:
            rev2 = "conflict_loop"
        elif flags["mostly_stationary"]:
            rev2 = "mostly_stationary"
        else:
            rev2 = "other_moving"
        # 6A.0 primary class: genuine deadlocks first, then the order of revision 2
        if flags["deadlock_cycle"]:
            primary = "deadlock_cycle"
        elif flags["blocked_by_idle"]:
            primary = "blocked_by_idle"
        else:
            primary = rev2.split(":")[0] if rev2.startswith("frozen") else rev2
        robots.append({"robot": i, "moves": moves, "proposed_moves": props, "cells": cells,
                       "conflict_steps": conf, "cycle_steps": cyc, "deadlock_steps": dead,
                       "deadlock_steps_episode": deadlock_steps_total[i],
                       "idle_block_steps": idle_block, "follow_steps": follow,
                       "next_cell": dict(nocc), "flags": flags, "class_rev2": rev2,
                       "class": primary})  # fmt: skip
    held = len(robots)
    undelivered = stored["released"] - stored["delivered"]
    return {
        "label": label, "algorithm": label.split("-seed")[0], "level": stored["level"],
        "seed": stored["seed"], "undelivered": undelivered, "stuck_robots": held,
        "undelivered_not_held": max(undelivered - held, 0), "mismatch": mismatch,
        "replayed": replayed, "robots": robots,
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    target = args.out / "replay_episodes.jsonl"
    if target.exists():
        raise SystemExit(f"{target} exists; replays are written once")
    args.out.mkdir(parents=True, exist_ok=True)
    pop = failures()
    print(len(pop), "failed greedy episodes to replay")
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(replay, pop, chunksize=4))
    with target.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    print("wrote", target, "| mismatching episodes:", sum(bool(r["mismatch"]) for r in results))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.exit(main())
