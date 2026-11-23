"""L1 follow-up: evaluate stored PPO checkpoints on a new, separate selection block (read only).

Selection block **S = seeds 13 000–13 079** (80 seeds × the 3 levels, A* planner, greedy mode).
It is disjoint from training (0–9 999), validation A (10 000–10 019) and B (10 020–10 099),
the planned confirmation blocks C (11 000–11 119) and D (12 000–12 199), and the reserved test
seeds (20 000–20 999). ``check`` verifies, before any episode is run, that no stored record uses
a block-S seed and that every block-S scenario instance differs from every block A–D instance
(only the scenario generator is called for C and D; nothing is evaluated on them).

``run`` evaluates the 20 stored step checkpoints of each of the six main PPO runs with the
production episode runner :func:`robocoord.marl.evaluation.run_episode`. Its seed guard only
knows the Issue #5 blocks, so the workers replace it with a guard that accepts block S and
nothing else. No training, no checkpoint is written, no production file is changed.

Usage::

    python tools/selection/selection_block.py check --out results/dev6a/l1_selection/selection_block
    python tools/selection/selection_block.py run --out results/dev6a/l1_selection/selection_block --workers 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

SELECTION_BLOCK = range(13_000, 13_080)
OTHER_BLOCKS = {
    "train": range(0, 10_000),
    "A": range(10_000, 10_020),
    "B": range(10_020, 10_100),
    "C": range(11_000, 11_120),
    "D": range(12_000, 12_200),
    "test": range(20_000, 21_000),
}
RUNS = [f"{a}_seed{s}" for a in ("mappo", "ippo") for s in (1, 2, 3)]


def guard(seeds: Iterable[int], block: str) -> list[int]:
    """Replacement of ``check_seeds`` in the workers: block S only."""
    out = [int(s) for s in seeds]
    bad = [s for s in out if s not in SELECTION_BLOCK]
    if bad:
        raise ValueError(f"seeds {bad[:5]} are outside selection block S (13000–13079)")
    return out


def _init_worker() -> None:
    from robocoord.marl import evaluation

    setattr(evaluation, "check_seeds", guard)  # noqa: B010 (replaces the module global)


def _run(task: Any) -> dict[str, Any]:
    from robocoord.marl.evaluation import run_episode

    rec: dict[str, Any] = run_episode(task)
    return rec


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(spec: Any) -> str:
    """Hash of a scenario instance without its name and seed (map, robots, tasks, events)."""
    parts = [
        spec.grid._blocked.tobytes(),
        repr(spec.grid._blocked.shape).encode(),
        repr((spec.robots, spec.max_steps, spec.num_priorities, spec.obstacles, spec.tasks,
              spec.arrivals, spec.map_events, spec.script)).encode(),
    ]  # fmt: skip
    return hashlib.sha256(b"|".join(parts)).hexdigest()


def check(out: Path) -> dict[str, Any]:
    from robocoord.marl.scenarios import LEVELS

    for name, r in OTHER_BLOCKS.items():
        assert not set(r) & set(SELECTION_BLOCK), name
    # 1. no stored record mentions a block-S seed
    pat = re.compile(rb'"seed": ?(13\d{3})\b')
    hits: dict[str, int] = {}
    scanned = 0
    for p in Path("results").rglob("*"):
        if p.suffix not in (".json", ".jsonl") or out in p.parents:
            continue
        scanned += 1
        for m in pat.finditer(p.read_bytes()):
            if int(m.group(1)) in SELECTION_BLOCK:
                hits[str(p)] = hits.get(str(p), 0) + 1
    # 2. scenario instances: unique within S, disjoint from A–D
    fp: dict[str, dict[str, set[str]]] = {}
    for lvl, dist in LEVELS.items():
        fp[lvl] = {}
        for name in ("A", "B", "C", "D"):
            fp[lvl][name] = {fingerprint(dist.sample(s)) for s in OTHER_BLOCKS[name]}
        fp[lvl]["S"] = {fingerprint(dist.sample(s)) for s in SELECTION_BLOCK}
    report: dict[str, Any] = {
        "selection_block": [SELECTION_BLOCK.start, SELECTION_BLOCK.stop - 1],
        "files_scanned": scanned,
        "files_with_block_s_seeds": hits,
        "unique_instances_in_S": {lvl: len(v["S"]) for lvl, v in fp.items()},
        "overlap_with": {
            lvl: {b: len(v["S"] & v[b]) for b in ("A", "B", "C", "D")} for lvl, v in fp.items()
        },
        "note": "Blocks C and D: scenario generator only; no policy was evaluated on them.",
    }
    report["ok"] = (
        not hits
        and all(n == len(SELECTION_BLOCK) for n in report["unique_instances_in_S"].values())
        and all(x == 0 for v in report["overlap_with"].values() for x in v.values())
    )
    return report


def _key(t: Any) -> tuple[str, str, str, int]:
    return (t.label, os.path.basename(t.checkpoint), t.level, t.seed)


def tasks() -> list[Any]:
    from robocoord.marl.evaluation import EvalTask
    from robocoord.marl.scenarios import LEVELS

    out = []
    for run in RUNS:
        for ck in sorted(Path(f"results/training/{run}/checkpoints").glob("step_*[0-9].pt")):
            for lvl in LEVELS:
                for s in SELECTION_BLOCK:
                    out.append(EvalTask("ppo", lvl, s, "astar", "S", str(ck), label=run))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "check":
        rep = check(args.out)
        (args.out / "seed_check.json").write_text(json.dumps(rep, indent=1), "utf-8")
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 1
    rep = json.loads((args.out / "seed_check.json").read_text("utf-8"))
    if not rep["ok"]:
        raise SystemExit("seed check failed; not running")
    todo = tasks()
    rec_path = args.out / "records.jsonl"
    done: set[tuple[str, str, str, int]] = set()
    if rec_path.exists():
        for line in rec_path.read_text("utf-8").splitlines():
            r = json.loads(line)
            done.add((r["label"], r["checkpoint"], r["level"], r["seed"]))
    todo_left = [t for t in todo if _key(t) not in done]
    cks = sorted({t.checkpoint for t in todo})
    manifest = {
        "selection_block": [SELECTION_BLOCK.start, SELECTION_BLOCK.stop - 1],
        "runs": RUNS,
        "levels": sorted({t.level for t in todo}),
        "planner": "astar",
        "mode": "greedy",
        "episodes": len(todo),
        "checkpoints": {c.replace("\\", "/"): sha256(Path(c)) for c in cks},
        "code": {
            p: sha256(Path(p))
            for p in ("src/robocoord/marl/evaluation.py", "tools/selection/selection_block.py")
        },
        "workers": args.workers,
    }
    t0 = time.perf_counter()
    with (
        open(rec_path, "a", encoding="utf-8") as fh,
        ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool,
    ):
        for i, rec in enumerate(pool.map(_run, todo_left, chunksize=4), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 500 == 0:
                fh.flush()
                el = time.perf_counter() - t0
                print(f"{i}/{len(todo_left)} {el:.0f}s eta {el / i * (len(todo_left) - i):.0f}s",
                      flush=True)  # fmt: skip
    manifest["wall_s_this_invocation"] = time.perf_counter() - t0
    manifest["episodes_this_invocation"] = len(todo_left)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    with open(rec_path, encoding="utf-8") as fh:
        n = sum(1 for _ in fh)
    print(f"records: {n} / {len(todo)}")
    return 0 if n == len(todo) else 1


def load(out: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per run: ``steps (K)``, ``delivered / released / conflicts / collisions / ret (K, S)``
    with levels summed (counts) or averaged (conflicts, return) per seed — as in calibrate."""
    with open(out / "records.jsonl", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    seeds = list(SELECTION_BLOCK)
    res: dict[str, dict[str, np.ndarray]] = {}
    for run in RUNS:
        rr = [r for r in recs if r["label"] == run]
        cks = sorted({r["checkpoint"] for r in rr})
        a = {k: np.zeros((len(cks), len(seeds)))
             for k in ("delivered", "released", "conflicts", "collisions", "ret")}  # fmt: skip
        cnt = np.zeros((len(cks), len(seeds)))
        ci, si = {c: i for i, c in enumerate(cks)}, {s: j for j, s in enumerate(seeds)}
        for r in rr:
            i, j = ci[r["checkpoint"]], si[r["seed"]]
            a["delivered"][i, j] += r["delivered"]
            a["released"][i, j] += r["released"]
            a["conflicts"][i, j] += r["conflicts_total"]
            a["collisions"][i, j] += r["physical_collisions"]
            a["ret"][i, j] += r["base_return"]
            cnt[i, j] += 1
        a["conflicts"] /= cnt
        a["ret"] /= cnt
        a["steps"] = np.array([int(c[5:13]) for c in cks], dtype=float)
        res[run] = a
    return res


if __name__ == "__main__":
    sys.exit(main())
