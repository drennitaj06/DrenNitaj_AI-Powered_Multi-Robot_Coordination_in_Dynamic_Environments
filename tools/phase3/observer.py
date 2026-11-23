"""Read-only episode observer for the Phase 3 experiments (Issue 6A, Stages 3b, 3c and 3e).

Passed to ``evaluation.run_episode(task, observer=...)``; it never changes an action, the engine
or the record. It collects:

* **stuck-robot classes** at the end of the episode, with the Phase 6A.0 classifier
  (``tools/diagnostics/replay_failures.py``) ported verbatim: the same window (last 100 steps),
  flags, thresholds, wait-for-cycle deadlock criteria and class order. The one generalisation:
  the action that counts as "proposed" is the action *requested* of the arbiter's pipeline, i.e.
  the supervisor's output when it changed something, else the policy's (with the supervisor off
  this is exactly 6A.0's definition);
* **per-task outcomes** (priority, release, pickup and completion time, final status), for the
  high-priority flow-time and undelivered-task vetoes;
* **supervisor-introduced invalid moves**: robot-steps whose supervised action differs from the
  policy's and is not allowed by the action mask (must be 0; protocol G0.4);
* its own wall time (subtracted for the runtime comparison).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from typing import Any

WINDOW = 100  # as replay_failures.WINDOW
DETECT_STEPS = 2
RECUR_COUNT = 3
RECUR_WINDOW = 20
THRESH = 50


class Phase3Observer:
    def __init__(self) -> None:
        self.hist: deque[dict[str, Any]] = deque(maxlen=WINDOW)
        self.seen: dict[tuple[int, ...], deque[int]] = defaultdict(deque)
        self.prev_cycles: dict[tuple[int, ...], tuple[Any, int]] = {}
        self.deadlock_steps_total: Counter[int] = Counter()
        self.sup_invalid = 0
        self.sup_changed = 0
        self.seconds = 0.0
        self._pre: dict[str, Any] = {}
        self._engine: Any = None

    # ------------------------------------------------------------------ hooks
    def before(self, engine: Any, view: Any, actions: Any) -> None:
        from robocoord.planning.deadlock import find_cycles
        from robocoord.world.entities import RobotMode

        t0 = time.perf_counter()
        self._engine = engine
        n = len(engine.robots)
        pos = [r.position for r in engine.robots]
        idle = [r.mode is RobotMode.IDLE for r in engine.robots]
        cell_of = {p: i for i, p in enumerate(pos)}
        nxt: list[Any] = []
        succ: dict[int, int] = {}
        for i in range(n):
            path = view.reference_paths[i] if view.reference_paths else None
            c = path[1] if (path is not None and len(path) >= 2 and not idle[i]) else None
            nxt.append(c)
            if c is not None and c in cell_of and cell_of[c] != i:
                succ[i] = cell_of[c]
        t = engine.t if hasattr(engine, "t") else len(self.hist)
        cycles = [tuple(sorted(cy)) for cy in find_cycles(succ)]
        cur: dict[tuple[int, ...], tuple[Any, int]] = {}
        in_deadlock: set[int] = set()
        for mem in cycles:
            cells = tuple(pos[i] for i in mem)
            run_len = (
                self.prev_cycles[mem][1] + 1
                if mem in self.prev_cycles and self.prev_cycles[mem][0] == cells
                else 1
            )
            cur[mem] = (cells, run_len)
            if mem not in self.prev_cycles:
                q = self.seen[mem]
                q.append(t)
                while q and q[0] <= t - RECUR_WINDOW:
                    q.popleft()
            if run_len >= DETECT_STEPS or len(self.seen[mem]) >= RECUR_COUNT:
                in_deadlock.update(mem)
        self.prev_cycles = cur
        for i in in_deadlock:
            self.deadlock_steps_total[i] += 1
        self._pre = {
            "pos": pos,
            "idle": idle,
            "act": [int(actions[i]) for i in range(n)],
            "sug": [int(s) for s in view.suggested] if view.suggested is not None else None,
            "masks": view.masks,
            "in_cycle": {i for mem in cycles for i in mem},
            "deadlock": in_deadlock,
            "next_occ": [
                None
                if nxt[i] is None
                else "idle"
                if nxt[i] in cell_of and idle[cell_of[nxt[i]]]
                else "active"
                if nxt[i] in cell_of
                else "free"
                for i in range(n)
            ],
        }
        self.seconds += time.perf_counter() - t0

    def after(self, engine: Any, outcome: Any) -> None:
        t0 = time.perf_counter()
        pre = self._pre
        n = len(engine.robots)
        act = pre["act"]
        sup = outcome.supervised_actions
        if sup is not None:
            for i in range(n):
                if int(sup[i]) != act[i]:
                    self.sup_changed += 1
                    if not pre["masks"][i][sup[i]]:
                        self.sup_invalid += 1
            act = [int(a) for a in sup]
        evs = [e.entities for e in outcome.events]
        self.hist.append(
            {
                "pos": pre["pos"],
                "new": [r.position for r in engine.robots],
                "idle": pre["idle"],
                "act": act,
                "sug": pre["sug"],
                "in_cycle": pre["in_cycle"],
                "deadlock": pre["deadlock"],
                "conf": {i for ents in evs for i in range(n) if f"r{i}" in ents},
                "next_occ": pre["next_occ"],
            }
        )
        self.seconds += time.perf_counter() - t0

    # ------------------------------------------------------------------ results
    def stuck_robots(self) -> list[dict[str, Any]]:
        """The 6A.0 classification of every robot that is not idle at the end."""
        from robocoord.core.types import Action
        from robocoord.world.entities import RobotMode

        eng = self._engine
        out = []
        h = list(self.hist)
        for i, r in enumerate(eng.robots):
            if r.mode is RobotMode.IDLE:
                continue
            moves = sum(x["pos"][i] != x["new"][i] for x in h)
            props = sum(x["act"][i] != int(Action.STAY) for x in h)
            cells = len({x["pos"][i] for x in h})
            conf = sum(i in x["conf"] for x in h)
            dead = sum(i in x["deadlock"] for x in h)
            idle_block = sum(x["next_occ"][i] == "idle" for x in h)
            flags = {
                "frozen": moves == 0 and props == 0,
                "oscillating": cells <= 3 and moves >= THRESH,
                "conflict_loop": conf >= THRESH,
                "deadlock_cycle": dead >= THRESH,
                "blocked_by_idle": idle_block >= THRESH,
                "mostly_stationary": moves < 10,
            }
            if flags["frozen"]:
                rev2 = "frozen"
            elif flags["oscillating"]:
                rev2 = "oscillating"
            elif flags["conflict_loop"]:
                rev2 = "conflict_loop"
            elif flags["mostly_stationary"]:
                rev2 = "mostly_stationary"
            else:
                rev2 = "other_moving"
            if flags["deadlock_cycle"]:
                primary = "deadlock_cycle"
            elif flags["blocked_by_idle"]:
                primary = "blocked_by_idle"
            else:
                primary = rev2
            out.append({"robot": i, "class": primary, "moves": moves, "cells": cells})
        return out

    def tasks(self) -> list[list[Any]]:
        """``[id, priority, release, pickup_time, completion_time, status]`` per task."""
        return [
            [t.id, t.priority, t.release, t.pickup_time, t.completion_time, t.status.value]
            for _, t in sorted(self._engine.book.tasks.items())
        ]

    def result(self) -> dict[str, Any]:
        return {
            "stuck": self.stuck_robots(),
            "tasks": self.tasks(),
            "sup_changed": self.sup_changed,
            "sup_invalid": self.sup_invalid,
            "observer_s": self.seconds,
        }
