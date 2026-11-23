"""Issue 6A Phase 1 — replanning optimization (P1, repair before replanning): evaluation runner.

Protocol: ``docs/research/2026-11-22-issue-6a-phase1-replanning.md`` §3 (written before the run).
The intervention is the path monitor's T3 repair (``MonitorParams.repair_radius``; radius 3 =
the model's rejoin window ``H_r``); the baseline is the identical configuration with repair off.

Conditions × arms (repair 0 / 3), all other settings as Phase 3 (greedy, auction, shield off,
endpoint recovery on, ``path_limit`` 6, horizon 400, RRT\\* 1 000 iterations, the Phase 3
checkpoints, the Stage 3a frozen supervisor parameters):

* ``rsd`` — the approved supervisor configuration **R + S + D** (``supervisor="R+S"``,
  ``deadlock=True``; Phase 3 §11);
* ``off`` — no supervisor, resolver off (the historical learned-controller composition).

Stages: ``1a`` A\\* on Block C 11 000–11 119 × 3 levels; ``1b`` RRT\\* on the Block C RRT\\*
subset 11 000–11 039 × 3 levels. 9 policies. 12 960 + 4 320 episodes.

Commands::

    python tools/phase1/phase1.py check  --out results/dev6a/phase1/eval
    python tools/phase1/phase1.py run    --out results/dev6a/phase1/eval --stage 1a --workers 11
    python tools/phase1/phase1.py run    --out results/dev6a/phase1/eval --stage 1b --workers 11
    python tools/phase1/phase1.py replay --out results/dev6a/phase1/eval --workers 11
    python tools/phase1/phase1_analysis.py --out results/dev6a/phase1/eval
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
sys.path.insert(0, str(HERE.parent / "phase3"))
sys.path.insert(0, str(HERE.parent / "selection"))
import phase3 as p3  # noqa: E402
from observer import Phase3Observer  # noqa: E402

REPAIR_RADIUS = 3  # = MonitorParams.rejoin_window (H_r), model §11; not tuned
SUPERVISION: dict[str, tuple[str, bool]] = {"rsd": ("R+S", True), "off": ("off", False)}
ARMS: dict[str, tuple[str, int]] = {
    f"{s}_r{r}": (s, r) for s in SUPERVISION for r in (0, REPAIR_RADIUS)
}
STAGES: dict[str, tuple[str, range]] = {
    "1a": ("astar", p3.BLOCK_C),
    "1b": ("rrtstar", p3.BLOCK_C_RRT),
}
REPLAY_STRIDE = 50
TIMING = (*p3.TIMING, "planning_s")
CODE = (
    *p3.CODE,
    "src/robocoord/planning/monitor.py",
    "tools/phase1/phase1.py",
)
#: Phase 3 records that the repair-off arms must reproduce exactly (G0.1).
PHASE3_RECORDS = {
    ("rsd_r0", "astar"): (Path("results/dev6a/phase3/eval/records_3c.jsonl"), "R_S_D"),
    ("off_r0", "astar"): (Path("results/dev6a/phase3/eval/records_3b.jsonl"), "off"),
    ("off_r0", "rrtstar"): (Path("results/dev6a/phase3/eval/records_3e.jsonl"), "off"),
}


class Phase1Observer(Phase3Observer):
    """The Phase 3 observer plus deterministic planner work and per-call timings."""

    def result(self) -> dict[str, Any]:
        out = super().result()
        mon = self._engine.coordinator.monitor
        calls = mon.calls
        by = Counter(c.trigger for c in calls)
        out["planning"] = {
            "calls": len(calls),
            "by_trigger": dict(sorted(by.items())),
            "expansions": sum(int(c.stats.expansions) for c in calls),
            "iterations": sum(int(c.stats.iterations) for c in calls),
            "t3_expansions": sum(int(c.stats.expansions) for c in calls if c.trigger == "T3"),
            "t3_iterations": sum(int(c.stats.iterations) for c in calls if c.trigger == "T3"),
            "deviations": mon.deviations,
            "repairs": mon.repairs,
            "repair_failures": mon.repair_failures,
            "active_robot_steps": mon.active_robot_steps,
        }
        out["planning_s"] = sum(c.stats.time_s for c in calls)
        return out


def jobs(stage: str, ck: dict[str, Any]) -> list[dict[str, Any]]:
    planner, seeds = STAGES[stage]
    out = []
    for seed in seeds:
        for run in sorted(p3.RUNS, key=lambda r: (not r.startswith("madql"), r)):
            for level in p3.LEVELS:
                for arm in ARMS:
                    out.append(
                        {
                            "stage": stage,
                            "arm": arm,
                            "run": run,
                            "planner": planner,
                            "level": level,
                            "seed": seed,
                            "checkpoint": ck["runs"][run]["path"],
                            "checkpoint_sha256": ck["runs"][run]["sha256"],
                        }
                    )
    return out


def make_task(j: dict[str, Any]) -> Any:
    from robocoord.marl.evaluation import EvalTask

    sup, radius = ARMS[j["arm"]]
    mode, deadlock = SUPERVISION[sup]
    return EvalTask(
        p3.FAMILIES[j["run"].split("_")[0]],
        j["level"],
        int(j["seed"]),
        planner=j["planner"],
        block="C",
        checkpoint=j["checkpoint"],
        label=p3.label(j["run"]),
        supervisor=mode,
        deadlock=deadlock,
        repair=radius,
    )


def _run(j: dict[str, Any], observe: bool = True) -> dict[str, Any]:
    import selection_block

    from robocoord.marl.evaluation import run_episode
    from robocoord.marl.scenarios import LEVELS as DISTS

    p3.guard(j["planner"], int(j["seed"]), "C")
    spec = DISTS[j["level"]].sample(int(j["seed"]))
    meta = {k: j[k] for k in ("stage", "arm", "run", "checkpoint_sha256")}
    meta["checkpoint_path"] = j["checkpoint"]
    meta["instance_sha256"] = selection_block.fingerprint(spec)
    task = make_task(j)
    obs = Phase1Observer() if observe else None
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


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check(out: Path) -> dict[str, Any]:
    base = p3.check(out)  # Stage 3a freeze, frozen params, checkpoints, environment
    res = {
        **base,
        "phase": "Issue 6A Phase 1 (P1 repair)",
        "repair_radius": REPAIR_RADIUS,
        "arms": {
            a: {
                "supervision": s,
                "supervisor": SUPERVISION[s][0],
                "deadlock": SUPERVISION[s][1],
                "repair": r,
            }
            for a, (s, r) in ARMS.items()
        },
        "stages": {
            s: {"planner": p, "episodes": len(ARMS) * len(r) * 27} for s, (p, r) in STAGES.items()
        },
        "code_sha256": {c: p3.sha256(Path(c)) for c in CODE},
    }
    (out / "check.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    return res


def run(out: Path, stage: str, workers: int) -> int:
    chk = json.loads((out / "check.json").read_text("utf-8"))
    if not chk["ok"]:
        raise SystemExit("check.json is not ok")
    now = {c: p3.sha256(Path(c)) for c in CODE}
    if now != chk["code_sha256"]:
        raise SystemExit(
            f"code changed since check: {[c for c in CODE if now[c] != chk['code_sha256'][c]]}"
        )
    target = out / f"records_{stage}.jsonl"
    done = {p3.key(r) for r in _records(target)}
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
            if n % 500 == 0:
                fh.flush()
                print(f"{stage}: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    ck_after = {r: p3.sha256(Path(v["path"])) for r, v in chk["checkpoints"]["runs"].items()}
    errors = sum(r["status"] != "ok" for r in _records(target))
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
    recs = [r for s in STAGES for r in _records(out / f"records_{s}.jsonl")]
    recs.sort(key=lambda r: (r["stage"], r["arm"], r["run"], r["level"], r["seed"]))
    sample = recs[::REPLAY_STRIDE]
    js = [
        {k: r[k] for k in ("stage", "arm", "run", "planner", "level", "seed", "checkpoint_sha256")}
        | {"checkpoint": r["checkpoint_path"]}
        for r in sample
    ]
    with ProcessPoolExecutor(workers, initializer=p3._init_plain) as ex:
        again = list(ex.map(_run_plain, js, chunksize=1))
    skip = set(TIMING) | {"stuck", "tasks", "sup_changed", "sup_invalid", "planning"}
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
        print(json.dumps({"ok": res["ok"], "checkpoints_ok": res["checkpoints"]["ok"]}))
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
