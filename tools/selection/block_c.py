"""Block C development evaluation: Issue #6 picks ("old") vs SCB-C v5 picks ("new").

Plan: ``docs/research/2026-11-21-block-c-analysis-plan.md``; analysis: ``block_c_analysis.py``.
Scope (approved 2026-11-21): the four PPO runs whose pick changed (mappo_seed1–3, ippo_seed1),
both checkpoints, the 120 Block C seeds (11 000–11 119) × the 3 levels, greedy A*, shield off:
2 880 episodes. No training, no checkpoint written, no other run or controller, no Block D.

Commands (in order)::

    python tools/selection/block_c.py check   --out results/dev6a/block_c
    python tools/selection/block_c.py freeze  --out results/dev6a/block_c
    python tools/selection/block_c.py run     --out results/dev6a/block_c --workers 10
    python tools/selection/block_c.py replay  --out results/dev6a/block_c --workers 10
    python tools/selection/block_c.py analyse --out results/dev6a/block_c

``check`` — before any episode: no stored record uses a Block C seed; the 120 Block C instances
per level are unique and differ from every block A, B and S instance (Block D is not touched,
not even by the scenario generator); the old arm is the step checkpoint of the Issue #6 pick,
accepted only if the Issue #6 ``best.pt`` is intact and holds bit-identical tensors (see
``picks``); the new arm is the frozen SCB-C v5 pick, whose code hashes are current. ``freeze`` records the hashes of the plan, this file and the analysis
module; it refuses once records exist, and ``run`` refuses unless the hashes still match.
``run`` replaces the production seed guard in the workers with one that accepts Block C only,
records failed episodes (exception type and message) instead of stopping, and re-hashes every
checkpoint afterwards. ``replay`` re-runs every 60th task in fresh processes and compares every
non-timing field. ``analyse`` applies the frozen gate and analysis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import block_c_analysis as bca
import selection_block

BLOCK_C = bca.BLOCK_C
RUNS = bca.RUNS
ARMS = bca.ARMS
LEVELS = bca.LEVELS
#: Never evaluated here (and never passed to the scenario generator).
FORBIDDEN = {"D": range(12_000, 12_200), "test": range(20_000, 21_000)}
COMPARE_WITH = {
    "A": range(10_000, 10_020),
    "B": range(10_020, 10_100),
    "S": selection_block.SELECTION_BLOCK,
}
PLAN = Path("docs/research/2026-11-21-block-c-analysis-plan.md")
FROZEN = (PLAN, Path("tools/selection/block_c.py"), Path("tools/selection/block_c_analysis.py"))
SELECTIONS = Path("results/dev6a/l1_selection/frozen/selections.json")
SCBC_CODE = ("tools/selection/scbc.py", "tools/selection/scbc_frozen.py")
TIMING = ("wall_s", "prepare_s", "act_s")
REPLAY_STRIDE = 60
CODE = (
    "src/robocoord/marl/evaluation.py",
    "src/robocoord/marl/scenarios.py",
    "src/robocoord/marl/policy.py",
    "src/robocoord/sim/engine.py",
    *(str(p) for p in FROZEN[1:]),
)

sha256 = selection_block.sha256


def ckpt(run: str, name: str) -> Path:
    return Path(f"results/training/{run}/checkpoints/{name}")


def guard(seeds: Iterable[int], block: str) -> list[int]:
    """Replacement of ``check_seeds`` in the workers: Block C only."""
    out = [int(s) for s in seeds]
    for name, r in FORBIDDEN.items():
        if any(s in r for s in out):
            raise ValueError(f"block {name} seeds are not allowed here: {out[:5]}")
    bad = [s for s in out if s not in BLOCK_C]
    if bad or block != "C":
        raise ValueError(f"seeds {bad[:5]} / block {block!r} are outside Block C (11000–11119)")
    return out


def _init_worker() -> None:
    from robocoord.marl import evaluation

    setattr(evaluation, "check_seeds", guard)  # noqa: B010 (replaces the module global)


def _run(job: tuple[str, Any]) -> dict[str, Any]:
    """One episode; a failure is recorded (type and message) instead of raised."""
    from robocoord.marl.evaluation import run_episode
    from robocoord.marl.scenarios import LEVELS as DISTS

    arm, task = job
    guard([task.seed], task.block)
    spec = DISTS[task.level].sample(task.seed)
    extra = {
        "arm": arm,
        "instance_sha256": selection_block.fingerprint(spec),
        "max_steps": spec.max_steps,
    }
    try:
        rec: dict[str, Any] = run_episode(task)
    except Exception as exc:  # every failure is a reported outcome
        rec = {k: v for k, v in asdict(task).items() if k != "checkpoint"}
        rec.update(
            checkpoint=os.path.basename(task.checkpoint),
            robots=len(spec.robots),
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )
        return {**rec, **extra}
    return {**rec, **extra, "status": "ok"}


#: Sidecar fields that legitimately differ between ``best.json`` and the step's ``.json``.
SIDECAR_FILE_FIELDS = ("file_bytes", "sha256")


def same_weights(a: Path, b: Path) -> bool:
    """Both files hold the same tensors (names, dtypes, shapes, values) — the serialised archives
    differ in their internal folder name (``best/`` vs ``step_…/``), so their bytes differ."""
    import torch

    sa = torch.load(a, map_location="cpu", weights_only=True)
    sb = torch.load(b, map_location="cpu", weights_only=True)
    return sa.keys() == sb.keys() and all(
        sa[k].dtype == sb[k].dtype and sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k])
        for k in sa
    )


def picks() -> dict[str, Any]:
    """Old (Issue #6 pick) and new (SCB-C v5 pick) step checkpoints, with integrity checks.

    The old arm loads the step checkpoint of the Issue #6 pick. It is accepted only when the
    Issue #6 ``best.pt`` is intact (its SHA-256 equals ``best.json``), holds bit-identical
    tensors, and the two sidecars agree on every field except the file size and hash.
    """
    sel = json.loads(SELECTIONS.read_text("utf-8"))
    out: dict[str, Any] = {"runs": {}, "ok": True, "notes": []}
    for p in SCBC_CODE:
        if sel["code"][p] != sha256(Path(p)):
            out["ok"] = False
            out["notes"].append(f"{p} changed since the SCB-C v5 selection")
    for run in RUNS:
        root = Path(f"results/training/{run}")
        best = json.loads((root / "best.json").read_text("utf-8"))
        old = f"step_{int(best['env_steps']):08d}.pt"
        new = sel["runs"][run]["selected"]
        side = json.loads(ckpt(run, old).with_suffix(".json").read_text("utf-8"))
        meta_diff = sorted(
            k for k in set(best) | set(side)
            if k not in SIDECAR_FILE_FIELDS and best.get(k) != side.get(k)
        )  # fmt: skip
        h_best = sha256(root / "best.pt")
        h_old, h_new = sha256(ckpt(run, old)), sha256(ckpt(run, new))
        checks = {
            "best_pt_intact": h_best == best["sha256"],
            "step_file_hash_matches_its_sidecar": h_old == side["sha256"],
            "best_pt_tensors_identical_to_step_file": same_weights(
                root / "best.pt", ckpt(run, old)
            ),
            "sidecar_metadata_identical": not meta_diff,
            "pick_changed": old != new,
        }
        out["ok"] &= all(checks.values())
        out["runs"][run] = {
            "old": old,
            "new": new,
            "issue6_env_steps": best["env_steps"],
            "sha256": {"old": h_old, "new": h_new, "best.pt": h_best},
            "checks": checks,
            "sidecar_differences": meta_diff,
        }
    out["selections_sha256"] = sha256(SELECTIONS)
    return out


def fingerprints(block: range) -> dict[str, list[str]]:
    from robocoord.marl.scenarios import LEVELS as DISTS

    return {lvl: [selection_block.fingerprint(DISTS[lvl].sample(s)) for s in block]
            for lvl in LEVELS}  # fmt: skip


def check(out: Path) -> dict[str, Any]:
    for name, r in {**FORBIDDEN, **COMPARE_WITH}.items():
        assert not set(r) & set(BLOCK_C), name
    pat = re.compile(rb'"seed": ?(11\d{3})\b')
    hits: dict[str, int] = {}
    scanned = 0
    for p in Path("results").rglob("*"):
        if p.suffix not in (".json", ".jsonl") or out in p.parents:
            continue
        scanned += 1
        for m in pat.finditer(p.read_bytes()):
            if int(m.group(1)) in BLOCK_C:
                hits[str(p)] = hits.get(str(p), 0) + 1
    fc = fingerprints(BLOCK_C)
    other = {b: fingerprints(r) for b, r in COMPARE_WITH.items()}
    pk = picks()
    report: dict[str, Any] = {
        "block_c": [BLOCK_C.start, BLOCK_C.stop - 1],
        "files_scanned": scanned,
        "files_with_block_c_seeds": hits,
        "unique_instances_in_C": {lvl: len(set(v)) for lvl, v in fc.items()},
        "overlap_with": {
            lvl: {b: len(set(fc[lvl]) & set(other[b][lvl])) for b in other} for lvl in LEVELS
        },
        "block_d": "not touched (no episode, no scenario generation)",
        "picks": pk,
        "fingerprints": fc,
    }
    report["ok"] = (
        not hits
        and pk["ok"]
        and all(n == len(BLOCK_C) for n in report["unique_instances_in_C"].values())
        and all(x == 0 for v in report["overlap_with"].values() for x in v.values())
    )
    return report


def tasks(pk: dict[str, Any]) -> list[tuple[str, Any]]:
    from robocoord.marl.evaluation import EvalTask

    jobs = []
    for run in RUNS:
        for arm in ARMS:
            ck = str(ckpt(run, pk["runs"][run][arm]))
            for lvl in LEVELS:
                for s in BLOCK_C:
                    jobs.append((arm, EvalTask("ppo", lvl, s, "astar", "C", ck, label=run)))
    return jobs


def _key(job: tuple[str, Any]) -> tuple[str, str, str, int]:
    return (job[1].label, job[0], job[1].level, job[1].seed)


def frozen_hashes() -> dict[str, str]:
    return {str(p).replace("\\", "/"): sha256(p) for p in FROZEN}


def _env() -> dict[str, str]:
    import numpy
    import torch

    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "torch": torch.__version__, "platform": platform.platform()}  # fmt: skip


def _records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def run(out: Path, workers: int) -> int:
    rep = json.loads((out / "check.json").read_text("utf-8"))
    if not rep["ok"]:
        raise SystemExit("pre-run check failed; not running")
    freeze = json.loads((out / "plan_freeze.json").read_text("utf-8"))
    if freeze["hashes"] != frozen_hashes():
        raise SystemExit("plan / runner / analysis changed since the freeze; not running")
    pk = rep["picks"]
    jobs = tasks(pk)
    rec_path = out / "records.jsonl"
    old = _records(rec_path) if rec_path.exists() else []
    done = {(r["label"], r["arm"], r["level"], r["seed"]) for r in old}
    todo = [j for j in jobs if _key(j) not in done]
    cks = sorted({str(ckpt(run, pk["runs"][run][a])) for run in RUNS for a in ARMS})
    before = {c.replace("\\", "/"): sha256(Path(c)) for c in cks}
    manifest: dict[str, Any] = {
        "block_c": [BLOCK_C.start, BLOCK_C.stop - 1],
        "runs": list(RUNS),
        "arms": {run: {a: pk["runs"][run][a] for a in ARMS} for run in RUNS},
        "levels": list(LEVELS),
        "condition": bca.EXPECTED,
        "episodes": len(jobs),
        "checkpoints_before": before,
        "code": {p: sha256(Path(p)) for p in CODE},
        "frozen": freeze,
        "environment": _env(),
        "workers": workers,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    }
    t0 = time.perf_counter()
    with (
        open(rec_path, "a", encoding="utf-8") as fh,
        ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool,
    ):
        for i, rec in enumerate(pool.map(_run, todo, chunksize=2), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 240 == 0:
                fh.flush()
                el = time.perf_counter() - t0
                print(f"{i}/{len(todo)} {el:.0f}s eta {el / i * (len(todo) - i):.0f}s", flush=True)
    after = {c.replace("\\", "/"): sha256(Path(c)) for c in cks}
    manifest.update(
        checkpoints_after=after,
        checkpoints_unchanged=before == after,
        wall_s_this_invocation=time.perf_counter() - t0,
        episodes_this_invocation=len(todo),
        finished=dt.datetime.now().isoformat(timespec="seconds"),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    n = len(_records(rec_path))
    print(f"records: {n} / {len(jobs)}; checkpoints unchanged: {before == after}")
    return 0 if n == len(jobs) and before == after else 1


def replay(out: Path, workers: int) -> dict[str, Any]:
    pk = json.loads((out / "check.json").read_text("utf-8"))["picks"]
    jobs = tasks(pk)[::REPLAY_STRIDE]
    recs = _records(out / "records.jsonl")
    stored = {(r["label"], r["arm"], r["level"], r["seed"]): r for r in recs}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        fresh = list(pool.map(_run, jobs))
    diffs = []
    for job, r in zip(jobs, fresh, strict=True):
        s = stored[_key(job)]
        bad = sorted(k for k in set(r) | set(s) if k not in TIMING and r.get(k) != s.get(k))
        if bad:
            diffs.append({"key": list(_key(job)), "fields": bad})
    return {"stride": REPLAY_STRIDE, "episodes": len(jobs), "identical": len(jobs) - len(diffs),
            "differences": diffs, "ignored_fields": list(TIMING)}  # fmt: skip


def pairing(records: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["label"], r["arm"], r["level"], r["seed"]): r for r in records}
    n = same_instance = same_config = same_released = 0
    for run in RUNS:
        for lvl in LEVELS:
            for s in BLOCK_C:
                a, b = by.get((run, "old", lvl, s)), by.get((run, "new", lvl, s))
                if a is None or b is None:
                    continue
                n += 1
                same_instance += a["instance_sha256"] == b["instance_sha256"]
                same_config += all(a.get(k) == b.get(k) for k in bca.PAIRED)
                same_released += a.get("released") == b.get("released")
    return {"pairs": n, "expected_pairs": len(RUNS) * len(LEVELS) * len(BLOCK_C),
            "same_instance_sha256": same_instance, "same_paired_fields": same_config,
            "same_released_count": same_released,
            "paired_fields": list(bca.PAIRED)}  # fmt: skip


def analyse(out: Path) -> dict[str, Any]:
    check_rep = json.loads((out / "check.json").read_text("utf-8"))
    freeze = json.loads((out / "plan_freeze.json").read_text("utf-8"))
    records = _records(out / "records.jsonl")
    pk = {run: {a: check_rep["picks"]["runs"][run][a] for a in ARMS} for run in RUNS}
    fps = {lvl: dict(zip(BLOCK_C, v, strict=True)) for lvl, v in check_rep["fingerprints"].items()}
    data = bca.build(records, pk, fps)
    res = bca.analyse(data)
    res["safety"] = bca.safety(records, pk)
    res["pairing"] = pairing(records)
    res["picks"] = pk
    res["frozen_hashes_match"] = freeze["hashes"] == frozen_hashes()
    res["records_sha256"] = sha256(out / "records.jsonl")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "freeze", "run", "replay", "analyse"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "check":
        rep = check(args.out)
        (args.out / "check.json").write_text(json.dumps(rep, indent=1), "utf-8")
        print(json.dumps({k: v for k, v in rep.items() if k != "fingerprints"}, indent=1))
        return 0 if rep["ok"] else 1
    if args.cmd == "freeze":
        if (args.out / "records.jsonl").exists():
            raise SystemExit("records exist; the plan can no longer be frozen")
        rep = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
               "hashes": frozen_hashes()}  # fmt: skip
        (args.out / "plan_freeze.json").write_text(json.dumps(rep, indent=1), "utf-8")
        print(json.dumps(rep, indent=1))
        return 0
    if args.cmd == "run":
        return run(args.out, args.workers)
    if args.cmd == "replay":
        rep = replay(args.out, args.workers)
        (args.out / "replay.json").write_text(json.dumps(rep, indent=1), "utf-8")
        print(json.dumps(rep, indent=1))
        return 0 if not rep["differences"] else 1
    res = analyse(args.out)
    (args.out / "analysis.json").write_text(json.dumps(res, indent=1), "utf-8")
    print(json.dumps({k: res[k] for k in ("evaluable", "n_problems", "pairing",
                                          "frozen_hashes_match")}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
