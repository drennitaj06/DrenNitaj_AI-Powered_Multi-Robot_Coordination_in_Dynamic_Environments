"""Issue 6A.0 — summary tables of the failure replay (reads ``replay_episodes.jsonl`` only).

Usage::

    python tools/diagnostics/summarise_replay.py --dir results/dev6a/6a0_diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

STORED = Path("results/dev6/stochastic/records.jsonl")
LEVELS = ("static", "obstacles", "full")
CLASSES = ("deadlock_cycle", "blocked_by_idle", "conflict_loop", "oscillating", "frozen",
           "mostly_stationary", "other_moving")  # fmt: skip
FLAGS = (
    "deadlock_cycle",
    "blocked_by_idle",
    "conflict_loop",
    "oscillating",
    "frozen",
    "mostly_stationary",
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args(argv)
    eps = [json.loads(line) for line in (args.dir / "replay_episodes.jsonl").open(encoding="utf-8")]
    greedy = Counter()
    with STORED.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["mode"] == "greedy":
                greedy[(r["label"], r["level"])] += 1
    s: dict[str, Any] = {
        "episodes_replayed": len(eps),
        "mismatching_episodes": sum(bool(e["mismatch"]) for e in eps),
    }
    cov = defaultdict(lambda: {"greedy_episodes": 0, "failed": 0, "seeds": []})
    for (lab, lvl), n in greedy.items():
        cov[f"{lab}|{lvl}"]["greedy_episodes"] = n
    for e in eps:
        c = cov[f"{e['label']}|{e['level']}"]
        c["failed"] += 1
        c["seeds"].append(e["seed"])
    for c in cov.values():
        c["seeds"] = sorted(c["seeds"])
    s["coverage"] = dict(sorted(cov.items()))
    robots = [(e, r) for e in eps for r in e["robots"]]

    def tab(key_fn, cls_key):
        t: dict[str, Counter] = defaultdict(Counter)
        for e, r in robots:
            t[key_fn(e)][r[cls_key]] += 1
        return {k: dict(v) for k, v in sorted(t.items())}

    s["robot_class_by_algorithm"] = tab(lambda e: e["algorithm"], "class")
    s["robot_class_by_policy_level"] = tab(lambda e: f"{e['label']}|{e['level']}", "class")
    s["robot_class_rev2_by_algorithm"] = tab(lambda e: e["algorithm"], "class_rev2")
    flags: dict[str, Counter] = defaultdict(Counter)
    co: dict[str, Counter] = defaultdict(Counter)
    for e, r in robots:
        on = [f for f in FLAGS if r["flags"][f]]
        for f in on:
            flags[e["algorithm"]][f] += 1
        flags[e["algorithm"]]["no_flag"] += not on
        co[e["algorithm"]]["+".join(on) or "none"] += 1
    s["flag_counts_by_algorithm"] = {k: dict(v) for k, v in flags.items()}
    s["flag_combinations_by_algorithm"] = {k: dict(v.most_common()) for k, v in co.items()}
    epi: dict[str, Counter] = defaultdict(Counter)
    for e in eps:
        kinds = sorted({r["class"] for r in e["robots"]})
        epi[e["algorithm"]][
            "no_stuck_robot" if not kinds else kinds[0] if len(kinds) == 1 else "mixed"
        ] += 1
    s["episode_class_by_algorithm"] = {k: dict(v) for k, v in epi.items()}
    s["stuck_robots_by_algorithm"] = dict(Counter(e["algorithm"] for e, _ in robots))
    s["failed_episodes_by_algorithm"] = dict(Counter(e["algorithm"] for e in eps))
    s["episodes_with_unheld_undelivered"] = dict(
        Counter(e["algorithm"] for e in eps if e["undelivered_not_held"])
    )
    s["unheld_undelivered_tasks"] = {
        a: sum(e["undelivered_not_held"] for e in eps if e["algorithm"] == a)
        for a in sorted({e["algorithm"] for e in eps})
    }
    s["undelivered_tasks"] = {
        a: sum(e["undelivered"] for e in eps if e["algorithm"] == a)
        for a in sorted({e["algorithm"] for e in eps})
    }
    s["episodes_with_genuine_deadlock_any_time"] = dict(
        Counter(
            e["algorithm"] for e in eps if any(r["deadlock_steps_episode"] > 0 for r in e["robots"])
        )
    )
    s["stuck_robots_with_deadlock_steps_in_window"] = dict(
        Counter(e["algorithm"] for e, r in robots if r["deadlock_steps"] > 0)
    )
    s["frozen_next_cell"] = {
        a: dict(
            Counter(
                r["class_rev2"]
                for e, r in robots
                if e["algorithm"] == a and r["class_rev2"].startswith("frozen")
            )
        )
        for a in ("madql", "mappo", "ippo")
    }
    (args.dir / "replay_summary.json").write_text(json.dumps(s, indent=1), "utf-8")
    print(
        json.dumps(
            {k: v for k, v in s.items() if k not in ("coverage", "robot_class_by_policy_level")},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
