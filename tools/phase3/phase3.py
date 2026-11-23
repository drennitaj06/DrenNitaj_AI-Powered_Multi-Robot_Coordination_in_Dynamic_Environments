"""Issue 6A Phase 3, Stages 3b, 3c and 3e — Execution Supervisor evaluation runner.

Protocol: design §10 (``docs/research/2026-11-21-issue-6a-phase3-execution-supervisor-design.md``)
and the Stage 3a freeze (``docs/research/2026-11-21-issue-6a-stage3a-calibration-freeze.md``,
§6 criteria). Results: ``docs/research/2026-11-21-issue-6a-phase3-evaluation.md``.

Condition of every arm (the existing evaluation configuration): greedy, auction allocator, shield
off, endpoint recovery on, ``path_limit`` 6, horizon 400, replicate 0, RRT\\* with 1 000
iterations; the production ``evaluation.run_episode`` with the Stage 3a frozen supervisor
parameters (``supervisor.frozen_params(planner)``); only ``supervisor`` and ``deadlock`` vary.

Arms (factor levels R, S+O, D; ``D`` = the existing deadlock resolver, ``EvalTask.deadlock``):

=========  ===========  =========  =====================================  ==============
arm        supervisor   resolver   factorial cell (R, S+O, D)             stages
=========  ===========  =========  =====================================  ==============
off        off          off        (0, 0, 0)                              3b, 3c, 3e
full       full         on         (1, 1, 1)                              3b, 3c, 3e
R          R            off        (1, 0, 0)                              3c
SO         S+O          off        (0, 1, 0)                              3c
D          off          on         (0, 0, 1)                              3c
R_SO       R+S+O        off        (1, 1, 0)                              3c
R_D        R            on         (1, 0, 1)                              3c
SO_D       S+O          on         (0, 1, 1)                              3c
R_S_D      R+S          on         "S without O" (full minus O)           3c
=========  ===========  =========  =====================================  ==============

Instances: A\\* (3b, 3c) Block C seeds 11 000–11 119 × 3 levels; RRT\\* (3e) the Block C RRT\\*
subset 11 000–11 039 × 3 levels (proposal §block definitions). Policies: MADQL seeds 1–3 (the
Issue #5 picks, ``best.pt``) and MAPPO / IPPO seeds 1–3 (the frozen SCB-C v5 picks; four of them
differ from the Issue #6 ``best.pt``). 3c reuses the 3b ``off`` and ``full`` episodes (same
configuration and code), so it runs only the 7 extra arms.

Commands::

    python tools/phase3/phase3.py check   --out results/dev6a/phase3/eval
    python tools/phase3/phase3.py run     --out results/dev6a/phase3/eval --stage 3b --workers 11
    python tools/phase3/phase3.py run     --out results/dev6a/phase3/eval --stage 3c --workers 11
    python tools/phase3/phase3.py run     --out results/dev6a/phase3/eval --stage 3e --workers 11
    python tools/phase3/phase3.py replay  --out results/dev6a/phase3/eval --workers 11
    python tools/phase3/phase3_analysis.py --out results/dev6a/phase3/eval

``run`` is resumable (completed tasks are skipped) and records failures instead of raising.
``replay`` re-runs every ``REPLAY_STRIDE``-th task in fresh processes **without** the observer
and the timing wrapper and compares every non-timing field (determinism and observer neutrality).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selection"))

BLOCK_C = range(11_000, 11_120)
BLOCK_C_RRT = range(11_000, 11_040)
LEVELS = ("static", "obstacles", "full")
FAMILIES = {"madql": "madql", "ippo": "ppo", "mappo": "ppo"}
RUNS = tuple(f"{f}_seed{s}" for f in ("madql", "ippo", "mappo") for s in (1, 2, 3))
CHANGED = ("mappo_seed1", "mappo_seed2", "mappo_seed3", "ippo_seed1")
ARMS: dict[str, tuple[str, bool, tuple[int, int, int] | None]] = {
    "off": ("off", False, (0, 0, 0)),
    "full": ("full", True, (1, 1, 1)),
    "R": ("R", False, (1, 0, 0)),
    "SO": ("S+O", False, (0, 1, 0)),
    "D": ("off", True, (0, 0, 1)),
    "R_SO": ("R+S+O", False, (1, 1, 0)),
    "R_D": ("R", True, (1, 0, 1)),
    "SO_D": ("S+O", True, (0, 1, 1)),
    "R_S_D": ("R+S", True, None),
}
STAGES: dict[str, tuple[str, tuple[str, ...], range]] = {
    "3b": ("astar", ("off", "full"), BLOCK_C),
    "3c": ("astar", ("R", "SO", "D", "R_SO", "R_D", "SO_D", "R_S_D"), BLOCK_C),
    "3e": ("rrtstar", ("off", "full"), BLOCK_C_RRT),
}
FORBIDDEN = {
    "train": range(0, 10_000),
    "A": range(10_000, 10_020),
    "B": range(10_020, 10_100),
    "D": range(12_000, 12_200),
    "S": range(13_000, 13_080),
    "test": range(20_000, 21_000),
}
TIMING = ("wall_s", "prepare_s", "act_s", "supervisor_s", "observer_s")
REPLAY_STRIDE = 50
STAGE3A_FREEZE = Path("results/dev6a/phase3/stage3a/freeze.json")
STAGE3A_FREEZE_SHA256 = "4c7b9dbccc9d1d87a115c3b98d535f7df1fb38b39bb8aad5dd4fa8a1866378fd"
BLOCK_C_RECORDS = Path("results/dev6a/block_c/records.jsonl")
CODE = (
    "src/robocoord/control/supervisor.py",
    "src/robocoord/marl/evaluation.py",
    "src/robocoord/marl/scenarios.py",
    "src/robocoord/marl/policy.py",
    "src/robocoord/sim/engine.py",
    "src/robocoord/sim/coordinator.py",
    "src/robocoord/planning/coordinator.py",
    "src/robocoord/planning/deadlock.py",
    "tools/phase3/phase3.py",
    "tools/phase3/observer.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(run: str) -> str:
    return run.replace("_", "-")


# --------------------------------------------------------------------------- guards


def guard(planner: str, seed: int, block: str) -> None:
    """Block C only (RRT\\*: its 40-seed subset only); raised before any scenario generation."""
    for name, r in FORBIDDEN.items():
        if seed in r:
            raise ValueError(f"block {name} seed {seed} is not allowed in Phase 3")
    allowed = BLOCK_C if planner == "astar" else BLOCK_C_RRT
    if block != "C" or seed not in allowed:
        raise ValueError(f"seed {seed} / block {block!r} not allowed for {planner}")


def _check_seeds(seeds: Any, block: str) -> list[int]:
    out = [int(s) for s in seeds]
    if block != "C" or any(s not in BLOCK_C for s in out):
        raise ValueError(f"seeds {out[:5]} / block {block!r} are outside Block C")
    return out


_SUP_TIMER = [0.0]


def _init_worker(timed: bool = True) -> None:
    """Replace the production seed check by the Block C check; optionally time the supervisor."""
    from robocoord.control import supervisor as sup
    from robocoord.marl import evaluation

    setattr(evaluation, "check_seeds", _check_seeds)  # noqa: B010 (module global)
    if timed and not getattr(sup.ExecutionSupervisor.apply, "phase3_timed", False):
        original = sup.ExecutionSupervisor.apply

        def apply(self: Any, *a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return original(self, *a, **k)
            finally:
                _SUP_TIMER[0] += time.perf_counter() - t0

        apply.phase3_timed = True  # type: ignore[attr-defined]
        sup.ExecutionSupervisor.apply = apply  # type: ignore[method-assign]


def _init_plain() -> None:
    _init_worker(timed=False)


# --------------------------------------------------------------------------- checkpoints


def checkpoints() -> dict[str, Any]:
    """Checkpoint identity and integrity (read-only). PPO: the frozen SCB-C v5 picks, verified by
    the Block D ``picks`` checks; MADQL: the Issue #5 ``best.pt`` (intact against ``best.json``)."""
    import block_d

    pk = block_d.picks()
    out: dict[str, Any] = {"ok": bool(pk["ok"]), "scbc_v5": pk, "runs": {}}
    for run in RUNS:
        if run.startswith("madql"):
            root = Path(f"results/training/{run}")
            path = root / "best.pt"
            best = json.loads((root / "best.json").read_text("utf-8"))
            h = sha256(path)
            ok = h == best["sha256"]
            out["ok"] &= ok
            out["runs"][run] = {
                "path": str(path).replace("\\", "/"),
                "sha256": h,
                "pick": "Issue #5 best.pt",
                "intact": ok,
            }
        else:
            info = pk["runs"][run]
            path = block_d.ckpt(run, info["new"])
            out["runs"][run] = {
                "path": str(path).replace("\\", "/"),
                "sha256": sha256(path),
                "pick": "SCB-C v5",
                "issue6_pick": info["old"],
                "issue6_best_pt_sha256": info["sha256"]["best.pt"],
                "changed_from_issue6": run in CHANGED,
                "differs_from_stage3a_calibration_checkpoint": info["old"] != info["new"],
            }
    return out


# --------------------------------------------------------------------------- tasks


def jobs(stage: str, ck: dict[str, Any]) -> list[dict[str, Any]]:
    planner, arms, seeds = STAGES[stage]
    out = []
    order = sorted(RUNS, key=lambda r: (not r.startswith("madql"), r))  # slow MADQL first
    for seed in seeds:
        for run in order:
            for level in LEVELS:
                for arm in arms:
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
    # interleave so that slow (MADQL) episodes are spread over the run
    return sorted(out, key=lambda j: (j["seed"], not j["run"].startswith("madql")))


def key(j: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (j["arm"], j["run"], j["planner"], j["level"], int(j["seed"]))


def make_task(j: dict[str, Any]) -> Any:
    from robocoord.marl.evaluation import EvalTask

    mode, deadlock, _ = ARMS[j["arm"]]
    return EvalTask(
        FAMILIES[j["run"].split("_")[0]],
        j["level"],
        int(j["seed"]),
        planner=j["planner"],
        block="C",
        checkpoint=j["checkpoint"],
        label=label(j["run"]),
        supervisor=mode,
        deadlock=deadlock,
    )


def _run(j: dict[str, Any], observe: bool = True) -> dict[str, Any]:
    from observer import Phase3Observer

    from robocoord.marl.evaluation import run_episode
    from robocoord.marl.scenarios import LEVELS as DISTS

    guard(j["planner"], int(j["seed"]), "C")
    import selection_block

    spec = DISTS[j["level"]].sample(int(j["seed"]))
    meta = {k: j[k] for k in ("stage", "arm", "run", "checkpoint_sha256")}
    meta["checkpoint_path"] = j["checkpoint"]
    meta["instance_sha256"] = selection_block.fingerprint(spec)
    task = make_task(j)
    obs = Phase3Observer() if observe else None
    _SUP_TIMER[0] = 0.0
    try:
        rec: dict[str, Any] = run_episode(task, observer=obs)
    except Exception as exc:  # a failure is an outcome (safety), never dropped
        rec = {k: v for k, v in asdict(task).items() if k != "checkpoint"}
        rec.update(status="error", error_type=type(exc).__name__, error_message=str(exc)[:500])
        return {**rec, **meta}
    rec = {**rec, **meta, "status": "ok", "supervisor_s": _SUP_TIMER[0]}
    if obs is not None:
        rec.update(obs.result())
    return rec


def _run_plain(j: dict[str, Any]) -> dict[str, Any]:
    return _run(j, observe=False)


# --------------------------------------------------------------------------- commands


def _env() -> dict[str, str]:
    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def check(out: Path) -> dict[str, Any]:
    """Verify the Stage 3a freeze, the frozen parameters, the checkpoints and the seed design."""
    from robocoord.control.supervisor import FROZEN_PARAMS

    freeze = json.loads(STAGE3A_FREEZE.read_text("utf-8"))
    frozen_ok = sha256(STAGE3A_FREEZE) == STAGE3A_FREEZE_SHA256 and all(
        freeze["frozen_params"][p] == {k: getattr(v, k) for k in v.__slots__}
        for p, v in FROZEN_PARAMS.items()
    )
    ck = checkpoints()
    res = {
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stage3a_freeze_ok": frozen_ok,
        "frozen_params": freeze["frozen_params"],
        "escalation_disabled": all(not v.escalate for v in FROZEN_PARAMS.values()),
        "checkpoints": ck,
        "seeds": {
            "astar": [BLOCK_C.start, BLOCK_C.stop - 1],
            "rrtstar": [BLOCK_C_RRT.start, BLOCK_C_RRT.stop - 1],
        },
        "arms": {a: {"supervisor": m, "deadlock": d, "cell": c} for a, (m, d, c) in ARMS.items()},
        "stages": {
            s: {"planner": p, "arms": list(a), "episodes": len(a) * len(r) * 27}
            for s, (p, a, r) in STAGES.items()
        },
        "code_sha256": {c: sha256(Path(c)) for c in CODE},
        "environment": _env(),
    }
    res["ok"] = bool(frozen_ok and ck["ok"] and res["escalation_disabled"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    return res


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(out: Path, stage: str, workers: int) -> int:
    chk = json.loads((out / "check.json").read_text("utf-8"))
    if not chk["ok"]:
        raise SystemExit("check.json is not ok; run `check` first")
    code_now = {c: sha256(Path(c)) for c in CODE}
    if code_now != chk["code_sha256"]:
        changed = [c for c in CODE if code_now[c] != chk["code_sha256"][c]]
        raise SystemExit(f"code changed since check: {changed}")
    target = out / f"records_{stage}.jsonl"
    done = {key({**r, "planner": r["planner"]}) for r in _records(target)}
    todo = [j for j in jobs(stage, chk["checkpoints"]) if key(j) not in done]
    ck_before = {r: sha256(Path(v["path"])) for r, v in chk["checkpoints"]["runs"].items()}
    log = out / "run.log"
    with log.open("a", encoding="utf-8") as lf:
        lf.write(
            f"{dt.datetime.now().isoformat(timespec='seconds')} run {stage}: "
            f"{len(done)} done, {len(todo)} to do, workers {workers}\n"
        )
    t0 = time.time()
    n = 0
    with (
        target.open("a", encoding="utf-8") as fh,
        ProcessPoolExecutor(workers, initializer=_init_worker) as ex,
    ):
        for rec in ex.map(_run, todo, chunksize=1):
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 500 == 0:
                fh.flush()
                print(f"{stage}: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    ck_after = {r: sha256(Path(v["path"])) for r, v in chk["checkpoints"]["runs"].items()}
    errors = sum(r["status"] != "ok" for r in _records(target))
    with log.open("a", encoding="utf-8") as lf:
        lf.write(
            f"{dt.datetime.now().isoformat(timespec='seconds')} run {stage} finished: "
            f"{n} new in {time.time() - t0:.0f}s; errors {errors}; "
            f"checkpoints unchanged {ck_before == ck_after}\n"
        )
    if ck_before != ck_after:
        raise SystemExit("CHECKPOINT HASH CHANGED DURING THE RUN")
    print(f"{stage}: {n} episodes, errors {errors}")
    return 0


def replay(out: Path, workers: int) -> dict[str, Any]:
    recs = [r for s in STAGES for r in _records(out / f"records_{s}.jsonl")]
    recs.sort(key=lambda r: (r["stage"], r["arm"], r["run"], r["planner"], r["level"], r["seed"]))
    sample = recs[::REPLAY_STRIDE]
    js = [
        {
            "stage": r["stage"],
            "arm": r["arm"],
            "run": r["run"],
            "planner": r["planner"],
            "level": r["level"],
            "seed": r["seed"],
            "checkpoint": r["checkpoint_path"],
            "checkpoint_sha256": r["checkpoint_sha256"],
        }
        for r in sample
    ]
    with ProcessPoolExecutor(workers, initializer=_init_plain) as ex:
        again = list(ex.map(_run_plain, js, chunksize=1))
    obs_fields = ("stuck", "tasks", "sup_changed", "sup_invalid")
    diffs = []
    for a, b in zip(sample, again, strict=True):
        fields = sorted(
            k
            for k in set(a) | set(b)
            if k not in TIMING and k not in obs_fields and a.get(k) != b.get(k)
        )
        if fields:
            diffs.append({"key": list(key(a)), "fields": fields})
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
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if args.cmd == "check":
        res = check(args.out)
        print(
            json.dumps(
                {
                    "ok": res["ok"],
                    "stage3a_freeze_ok": res["stage3a_freeze_ok"],
                    "checkpoints_ok": res["checkpoints"]["ok"],
                }
            )
        )
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
