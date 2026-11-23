"""Issue 6A Phase 2 — RRT\\* budget control (P2): Block C system-level evaluation runner.

Protocol: ``docs/research/2026-11-22-issue-6a-phase2-rrt-budget.md`` §5 (written after the
microbenchmark selection and before any Block C episode). The intervention is the RRT\\* stopping
rule frozen in ``results/dev6a/phase2/micro/selection.json`` (patience m = 400); the control is the
identical configuration with the full 1 000-iteration budget.

Stages (RRT\\*, Block C RRT\\* subset 11 000–11 039 × 3 levels; P1 repair off everywhere):

* ``2l`` — the 9 learned policies (Phase 3 checkpoints) with the approved supervisor
  configuration **R + S + D** (``supervisor="R+S"``, ``deadlock=True``, frozen Stage 3a
  parameters): arms ``full`` / ``p2``; 2 × 1 080 episodes;
* ``2b`` — the non-learned baseline controller (its original composition: resolver on, no
  supervisor): arms ``full`` / ``p2``; 2 × 120 episodes.

Commands::

    python tools/phase2/phase2.py check  --out results/dev6a/phase2/eval
    python tools/phase2/phase2.py run    --out results/dev6a/phase2/eval --stage 2b --workers 10
    python tools/phase2/phase2.py run    --out results/dev6a/phase2/eval --stage 2l --workers 10
    python tools/phase2/phase2.py replay --out results/dev6a/phase2/eval --workers 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase1"))
sys.path.insert(0, str(HERE.parent / "phase3"))
sys.path.insert(0, str(HERE.parent / "selection"))

import phase1 as p1  # noqa: E402
import phase3 as p3  # noqa: E402

SELECTION = Path("results/dev6a/phase2/micro/selection.json")
PROTOCOL = Path("docs/research/2026-11-22-issue-6a-phase2-rrt-budget.md")
BUDGET = 1000
ARMS = ("full", "p2")
BASELINE_RUN = "baseline"
STAGES: dict[str, tuple[str, ...]] = {"2l": p3.RUNS, "2b": (BASELINE_RUN,)}
REPLAY_STRIDE = 50
TIMING = p1.TIMING
CODE = (*p1.CODE, "src/robocoord/planning/rrtstar.py", "tools/phase2/phase2.py")
#: Phase 1 records that the full-budget learned arm must reproduce exactly (G0.1).
PHASE1_RECORDS = (Path("results/dev6a/phase1/eval/records_1b.jsonl"), "rsd_r0")
#: Fields this runner adds (absent from the Phase 1 records).
ADDED = ("rrt", "path_checks", "rrt_refine", "rrt_patience")


def selected() -> dict[str, Any]:
    sel = json.loads(SELECTION.read_text("utf-8"))
    cfg = sel["selected_config"]
    if cfg is None:
        raise SystemExit("no P2 candidate was selected; Block C is not run (protocol §3.3)")
    return {
        "label": sel["selected"],
        "refine": cfg["refine_iterations"],
        "patience": cfg["patience"],
    }


class Phase2Observer(p1.Phase1Observer):
    """The Phase 1 observer plus per-call RRT\\* statistics and a structural check of every new
    reference path (4-connected, in bounds, never through a permanent obstacle of the map)."""

    def __init__(self, base_blocked: Any) -> None:
        super().__init__()
        self._blocked = base_blocked
        self._seen: dict[int, tuple[int, int]] = {}
        self.paths_checked = 0
        self.paths_invalid = 0

    def after(self, engine: Any, outcome: Any) -> None:
        super().after(engine, outcome)
        t0 = time.perf_counter()
        mon = engine.coordinator.monitor
        h, w = self._blocked.shape
        for i, st in enumerate(mon.states):
            if st.path is None:
                continue
            ident = (st.planned_at, id(st.path))
            if self._seen.get(i) == ident:
                continue
            self._seen[i] = ident
            self.paths_checked += 1
            p = st.path
            ok = all(0 <= x < w and 0 <= y < h and not self._blocked[y, x] for x, y in p) and all(
                abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in itertools.pairwise(p)
            )
            self.paths_invalid += not ok
        self.seconds += time.perf_counter() - t0

    def result(self) -> dict[str, Any]:
        out = super().result()
        calls = [c for c in self._engine.coordinator.monitor.calls if c.stats.iterations > 0]
        ok = [c for c in calls if c.status == "ok"]
        status: Counter[str] = Counter(
            c.status + ("_degraded" if c.degraded else "") for c in calls
        )
        first = [
            c.stats.first_solution_iteration for c in ok if c.stats.first_solution_iteration > 0
        ]
        out["rrt"] = {
            "calls": len(calls),
            "status": dict(sorted(status.items())),
            "iterations": sum(c.stats.iterations for c in calls),
            "iterations_ok": sum(c.stats.iterations for c in ok),
            "stopped_early": sum(c.stats.iterations < BUDGET for c in ok),
            "budget_exhausted": status.get("budget_exhausted", 0),
            "first_solution_sum": sum(first),
            "first_solution_n": len(first),
            "path_moves_ok_sum": sum(c.path_length for c in ok),
            "path_moves_ok_n": len(ok),
            "tree_nodes_sum": sum(c.stats.tree_size for c in calls),
            "tree_nodes_max": max((c.stats.tree_size for c in calls), default=0),
            "segment_checks": sum(c.stats.segment_checks for c in calls),
            "time_s_sum": sum(c.stats.time_s for c in calls),
            "time_ms_median": 1000 * statistics.median(c.stats.time_s for c in calls)
            if calls
            else None,
        }
        out["path_checks"] = {"checked": self.paths_checked, "invalid": self.paths_invalid}
        return out


def jobs(stage: str, ck: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for seed in p3.BLOCK_C_RRT:
        for run in sorted(STAGES[stage], key=lambda r: (not r.startswith("madql"), r)):
            for level in p3.LEVELS:
                for arm in ARMS:
                    j = {
                        "stage": stage,
                        "arm": arm,
                        "run": run,
                        "planner": "rrtstar",
                        "level": level,
                        "seed": seed,
                        "checkpoint": None,
                        "checkpoint_sha256": None,
                    }
                    if run != BASELINE_RUN:
                        j["checkpoint"] = ck["runs"][run]["path"]
                        j["checkpoint_sha256"] = ck["runs"][run]["sha256"]
                    out.append(j)
    return out


def make_task(j: dict[str, Any], sel: dict[str, Any]) -> Any:
    from robocoord.marl.evaluation import EvalTask

    p2 = j["arm"] == "p2"
    common: dict[str, Any] = {
        "planner": "rrtstar",
        "block": "C",
        "rrt_iterations": BUDGET,
        "rrt_refine": sel["refine"] if p2 else None,
        "rrt_patience": sel["patience"] if p2 else None,
    }
    if j["run"] == BASELINE_RUN:  # original composition: resolver on, no supervisor
        return EvalTask("baseline", j["level"], int(j["seed"]), label="baseline", **common)
    return EvalTask(
        p3.FAMILIES[j["run"].split("_")[0]],
        j["level"],
        int(j["seed"]),
        checkpoint=j["checkpoint"],
        label=p3.label(j["run"]),
        supervisor="R+S",
        deadlock=True,
        **common,
    )


def _run(j: dict[str, Any], observe: bool = True) -> dict[str, Any]:
    import selection_block

    from robocoord.marl.evaluation import run_episode
    from robocoord.marl.scenarios import LEVELS as DISTS

    p3.guard("rrtstar", int(j["seed"]), "C")
    spec = DISTS[j["level"]].sample(int(j["seed"]))
    meta = {k: j[k] for k in ("stage", "arm", "run", "checkpoint_sha256")}
    meta["checkpoint_path"] = j["checkpoint"]
    meta["instance_sha256"] = selection_block.fingerprint(spec)
    task = make_task(j, selected())
    obs = Phase2Observer(spec.grid.blocked.copy()) if observe else None
    p3._SUP_TIMER[0] = 0.0
    try:
        rec: dict[str, Any] = run_episode(task, observer=obs)
    except Exception as exc:  # a failure is an outcome, never dropped
        rec = {k: v for k, v in asdict(task).items() if k != "checkpoint"}
        rec.update(status="error", error_type=type(exc).__name__, error_message=str(exc)[:500])
        return {**rec, **meta}
    rec = {**rec, **meta, "status": "ok", "supervisor_s": p3._SUP_TIMER[0]}
    if obs is not None:
        rec.update(obs.result())
    return rec


def _run_plain(j: dict[str, Any]) -> dict[str, Any]:
    return _run(j, observe=False)


def check(out: Path) -> dict[str, Any]:
    base = p3.check(out)  # Stage 3a freeze, frozen params, checkpoints, environment
    sel = selected()
    res = {
        **base,
        "phase": "Issue 6A Phase 2 (P2 RRT* budget control)",
        "selection": sel,
        "selection_sha256": p3.sha256(SELECTION),
        "protocol_sha256_at_check": p3.sha256(PROTOCOL),
        "budget": BUDGET,
        "arms": {"full": "RRT* 1000 iterations, no stopping rule", "p2": sel},
        "stages": {
            s: {"runs": list(r), "episodes": len(ARMS) * len(r) * 120} for s, r in STAGES.items()
        },
        "p1_repair": 0,
        "supervisor_learned": {"supervisor": "R+S", "deadlock": True},
        "code_sha256": {c: p3.sha256(Path(c)) for c in CODE},
    }
    res["ok"] = bool(base["ok"])
    (out / "check.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    return res


def _gate(out: Path) -> dict[str, Any]:
    chk: dict[str, Any] = json.loads((out / "check.json").read_text("utf-8"))
    if not chk["ok"]:
        raise SystemExit("check.json is not ok")
    now = {c: p3.sha256(Path(c)) for c in CODE}
    if now != chk["code_sha256"]:
        raise SystemExit(
            f"code changed since check: {[c for c in CODE if now[c] != chk['code_sha256'][c]]}"
        )
    if p3.sha256(SELECTION) != chk["selection_sha256"]:
        raise SystemExit("selection.json changed since check")
    return chk


def run(out: Path, stage: str, workers: int) -> int:
    chk = _gate(out)
    target = out / f"records_{stage}.jsonl"
    done = {p3.key(r) for r in p1._records(target)}
    todo = [j for j in jobs(stage, chk["checkpoints"]) if p3.key(j) not in done]
    ck_before = {r: p3.sha256(Path(v["path"])) for r, v in chk["checkpoints"]["runs"].items()}
    log = out / "run.log"
    with log.open("a", encoding="utf-8") as lf:
        lf.write(
            f"{dt.datetime.now().isoformat(timespec='seconds')} run {stage}: {len(done)} done, "
            f"{len(todo)} to do, workers {workers}\n"
        )
    t0, n = time.time(), 0
    with (
        target.open("a", encoding="utf-8") as fh,
        ProcessPoolExecutor(workers, initializer=p3._init_worker) as ex,
    ):
        for rec in ex.map(_run, todo, chunksize=1):
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 100 == 0:
                fh.flush()
                print(f"{stage}: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    ck_after = {r: p3.sha256(Path(v["path"])) for r, v in chk["checkpoints"]["runs"].items()}
    errors = sum(r["status"] != "ok" for r in p1._records(target))
    with log.open("a", encoding="utf-8") as lf:
        lf.write(
            f"{dt.datetime.now().isoformat(timespec='seconds')} run {stage} finished: {n} new "
            f"in {time.time() - t0:.0f}s; errors {errors}; checkpoints unchanged "
            f"{ck_before == ck_after}\n"
        )
    if ck_before != ck_after:
        raise SystemExit("CHECKPOINT HASH CHANGED DURING THE RUN")
    print(f"{stage}: {n} episodes, errors {errors}")
    return 0


def replay(out: Path, workers: int) -> dict[str, Any]:
    _gate(out)
    recs = [r for s in STAGES for r in p1._records(out / f"records_{s}.jsonl")]
    recs.sort(key=lambda r: (r["stage"], r["arm"], r["run"], r["level"], r["seed"]))
    sample = recs[::REPLAY_STRIDE]
    js = [
        {k: r[k] for k in ("stage", "arm", "run", "planner", "level", "seed", "checkpoint_sha256")}
        | {"checkpoint": r["checkpoint_path"]}
        for r in sample
    ]
    with ProcessPoolExecutor(workers, initializer=p3._init_plain) as ex:
        again = list(ex.map(_run_plain, js, chunksize=1))
    skip = set(TIMING) | {"stuck", "tasks", "sup_changed", "sup_invalid", "planning", *ADDED[:2]}
    diffs = []
    for a, b in zip(sample, again, strict=True):
        f = sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))
        if f:
            diffs.append({"key": list(p3.key(a)), "fields": f})
    res = {
        "replayed": len(sample),
        "stride": REPLAY_STRIDE,
        "differences": diffs,
        "identical": not diffs,
    }
    (out / "replay.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run", "replay"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=tuple(STAGES))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == "check":
        args.out.mkdir(parents=True, exist_ok=True)
        res = check(args.out)
        print(json.dumps({"ok": res["ok"], "selection": res["selection"]}))
        return 0 if res["ok"] else 1
    if args.cmd == "run":
        if args.stage is None:
            raise SystemExit("--stage is required")
        return run(args.out, args.stage, args.workers)
    res = replay(args.out, args.workers)
    print(json.dumps({k: res[k] for k in ("replayed", "identical")}), len(res["differences"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
