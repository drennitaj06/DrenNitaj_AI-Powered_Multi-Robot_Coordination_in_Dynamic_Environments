"""Issue 6A Phase 6 — integrated independent confirmatory validation: the runner.

Protocol (approved by the student 2026-11-23; frozen only by ``freeze --authorized-freeze``):
``docs/research/2026-11-23-issue-6a-phase6-protocol.md``. The freeze also requires the two
verified checkpoint backups of decision D12 (``backup_status``).

**Nothing in this module may touch the held-out block before the protocol is approved and
frozen.** Every command that generates or evaluates a held-out scenario needs an explicit
authorization flag and a valid freeze record:

* ``freeze --authorized-freeze`` hashes the protocol, this runner, the analysis and every reused
  frozen file. It refuses once a check or record file exists.
* ``check`` (the pre-run check) needs ``--authorized-precheck`` and an unchanged freeze. It never
  generates, fingerprints or otherwise instantiates a Block V scenario (protocol amendment A1).
* ``run`` and ``replay`` need ``--authorized-execution`` (the student's separate execution
  authorization) and an unchanged freeze. ``run`` performs the Block V instance check first
  (``instance_check``: generation, uniqueness, disjointness, fingerprints) and aborts before any
  episode if it fails (amendment A1).
* ``anchor`` (no flag) re-runs a stride sample of **stored Block C development records** through
  this runner and requires them to be reproduced exactly. It never generates a held-out scenario.

Held-out block (proposed; reservation needs approval): **V = seeds 14 000–14 199** (A\\*), RRT\\*
subset **14 000–14 059**; 3 levels each. Every other block (train, A, B, C, D, S, test) is refused
inside every worker process before the scenario generator is called (three layers, as Block D §8).

Policy arms (record ``arm`` = ``"<policy>:<cond>"``; ``run`` = the training run):

========  =====================================================  ==========================
policy    checkpoints                                            runs
========  =====================================================  ==========================
hist      Issue #5 MADQL ``best.pt`` (reference)                 madql_seed1–3
v2        MADQL-v2 n3 ``best.pt`` (Phase 4 selection)            madql_seed1–3
v5        SCB-C v5 PPO picks (the forward PPO policies)          ippo_seed1–3, mappo_seed1–3
i6        Issue #6 PPO picks (historical system, H-L12)          the 4 runs whose pick changed
base      rule-based baseline, original composition              baseline
========  =====================================================  ==========================

Conditions: ``OFF`` (no supervisor, resolver off: the Issue #6 composition), ``RSD`` (R + S + D:
``supervisor="R+S"``, resolver on, frozen Stage 3a parameters, O off), ``ORIG`` (baseline only:
no supervisor, resolver on, the original rule). Every arm: greedy, auction, shield off, endpoint
recovery on, ``path_limit`` 6, horizon 400, P1 repair off, RRT\\* 1 000 iterations (P2 off).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
for _sub in ("phase4", "phase2", "phase1", "phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))

import phase1 as p1  # noqa: E402
import phase3 as p3  # noqa: E402
import phase4 as p4  # noqa: E402

BLOCK = "V"
BLOCK_V = range(14_000, 14_200)
BLOCK_V_RRT = range(14_000, 14_060)
FORBIDDEN = {
    "train": range(0, 10_000),
    "A": range(10_000, 10_020),
    "B": range(10_020, 10_100),
    "C": range(11_000, 11_120),
    "D": range(12_000, 12_200),
    "S": range(13_000, 13_080),
    "test": range(20_000, 21_000),
}
PLANNERS = ("astar", "rrtstar")
LEVELS = p3.LEVELS
MADQL_RUNS = ("madql_seed1", "madql_seed2", "madql_seed3")
PPO_RUNS = tuple(f"{f}_seed{s}" for f in ("ippo", "mappo") for s in (1, 2, 3))
CHANGED = p3.CHANGED  # the 4 PPO runs whose SCB-C v5 pick differs from the Issue #6 pick
BASELINE_RUN = "baseline"
V2_CANDIDATE = "n3"  # the student's decision (2026-11-23); verified against selection.json
CONDS: dict[str, tuple[str, bool | None]] = {
    "OFF": ("off", False),
    "RSD": ("R+S", True),
    "ORIG": ("off", None),
}
#: (policy, condition) cells per planner; each cell covers every run of the policy.
MATRIX: dict[str, tuple[tuple[str, str], ...]] = {
    "astar": (
        ("hist", "OFF"), ("hist", "RSD"), ("v2", "OFF"), ("v2", "RSD"),
        ("v5", "OFF"), ("v5", "RSD"), ("i6", "OFF"), ("base", "ORIG"),
    ),
    "rrtstar": (
        ("hist", "OFF"), ("hist", "RSD"), ("v2", "RSD"),
        ("v5", "OFF"), ("v5", "RSD"), ("i6", "OFF"), ("base", "ORIG"),
    ),
}  # fmt: skip
POLICY_RUNS: dict[str, tuple[str, ...]] = {
    "hist": MADQL_RUNS,
    "v2": MADQL_RUNS,
    "v5": PPO_RUNS,
    "i6": CHANGED,
    "base": (BASELINE_RUN,),
}
PROTOCOL = Path("docs/research/2026-11-23-issue-6a-phase6-protocol.md")
BACKUP_DIR = Path("results/dev6a/phase6/backup")
BACKUP_MANIFEST = BACKUP_DIR / "manifest_sha256.json"
BACKUP_RECORD = BACKUP_DIR / "backup_record.json"
BACKUP_WAIVER = BACKUP_DIR / "d12_waiver.json"  # owner-approved waiver (2026-11-23)
FREEZE_FILES = (
    "tools/phase6/phase6.py",
    "tools/phase6/phase6_analysis.py",
    "tools/phase6/phase6_report.py",
    "tools/phase6/backup_manifest.py",
    "docs/research/2026-11-23-issue-6a-phase6-decision-review.md",
    "tools/selection/block_c_analysis.py",
    "tools/selection/block_d.py",
    "tools/selection/block_c.py",
    "tools/selection/selection_block.py",
    "tools/phase3/phase3.py",
    "tools/phase3/observer.py",
    "tools/phase3/phase3_analysis.py",
    "tools/phase1/phase1.py",
    "tools/phase2/phase2.py",
    "tools/phase4/phase4.py",
    "tools/phase4/phase4_analysis.py",
    "tools/diagnostics/replay_failures.py",
    "tests/tools/test_phase6.py",
)
CODE = tuple(dict.fromkeys((*p4.CODE, "tools/phase6/phase6.py")))
REPLAY_STRIDE = 50
TIMING = p4.TIMING
OBS_FIELDS = (*p4.OBS_FIELDS, "p6")
META = ("stage", "arm", "policy", "cond", "run", "checkpoint_sha256", "checkpoint_path",
        "instance_sha256", "status", "supervisor_s")  # fmt: skip


# --------------------------------------------------------------------------- design


def seeds(planner: str) -> range:
    return BLOCK_V if planner == "astar" else BLOCK_V_RRT


def guard(planner: str, seed: int, block: str = BLOCK) -> None:
    """Block V only (RRT\\*: its 60-seed subset); raised before any scenario generation."""
    for name, r in FORBIDDEN.items():
        if seed in r:
            raise ValueError(f"block {name} seed {seed} is not allowed in Phase 6")
    if block != BLOCK or seed not in seeds(planner):
        raise ValueError(f"seed {seed} / block {block!r} is outside the Phase 6 design ({planner})")


def guard_task(task: Any) -> None:
    guard(task.planner, int(task.seed), task.block)


def _check_seeds_v(ss: Any, block: str) -> list[int]:
    out = [int(s) for s in ss]
    if block != BLOCK or any(s not in BLOCK_V for s in out):
        raise ValueError(f"seeds {out[:5]} / block {block!r} are outside Block V")
    return out


def _check_seeds_c(ss: Any, block: str) -> list[int]:
    out = [int(s) for s in ss]
    if block != "C" or any(s not in p3.BLOCK_C for s in out):
        raise ValueError(f"seeds {out[:5]} / block {block!r} are outside Block C (anchor)")
    return out


def anchor_guard(planner: str, seed: int, block: str) -> None:
    """The anchor check runs on Block C development data only (never on V or any other block)."""
    allowed = p3.BLOCK_C if planner == "astar" else p3.BLOCK_C_RRT
    if block != "C" or seed not in allowed:
        raise ValueError(f"anchor: seed {seed} / block {block!r} is not Block C")


# --------------------------------------------------------------------------- worker hooks


def _init_worker(mode: str = "v", observe: bool = True) -> None:
    """Install the seed guards (``mode`` "v": Block V; "c": the Block C anchor) and, if
    ``observe``, the read-only decision hook of Phase 4; time the supervisor."""
    from robocoord.marl import evaluation

    p3._init_worker(timed=True)  # supervisor timer (and a Block C check, replaced below)
    p4._init_worker("4c", observe)  # read-only decision hook (its seed check is replaced below)
    check = _check_seeds_v if mode == "v" else _check_seeds_c
    setattr(evaluation, "check_seeds", check)  # noqa: B010 (module global)
    if not getattr(evaluation.run_episode, "phase6_guarded", False):
        original = evaluation.run_episode
        gate = guard_task if mode == "v" else (lambda t: anchor_guard(t.planner, t.seed, t.block))

        def run_episode(task: Any, *a: Any, **k: Any) -> Any:
            gate(task)
            return original(task, *a, **k)

        run_episode.phase6_guarded = True  # type: ignore[attr-defined]
        evaluation.run_episode = run_episode


class Phase6Observer(p4.Phase4Observer):
    """The Phase 4 observer (stuck classes, tasks, supervisor checks, planner work, RRT\\* calls,
    reference-path checks, learned-controller decision statistics) plus the Phase 6 conflict
    denominators: moves proposed to the arbiter (``a^arb``, after supervisor and shield) and the
    conflict sets on ``a^arb`` (K^arb) and on the controller's own proposal (K^ctrl), by type."""

    def __init__(self, base_blocked: Any, n: int) -> None:
        super().__init__(base_blocked, n)
        self.p6: dict[str, Any] = {
            "moves_arb": 0,
            "moves_arb_nonidle": 0,
            "moves_ctrl": 0,
            "robot_steps": 0,
            "k_arb": Counter(),
            "k_ctrl": Counter(),
            "k_ctrl_available": True,
            "ro_arb_nonidle": 0,  # RO events on K^arb whose robot was non-idle at the step
        }

    def after(self, engine: Any, outcome: Any) -> None:
        from robocoord.core.types import Action

        super().after(engine, outcome)
        t0 = time.perf_counter()
        idle = self._pre["idle"]
        arb = (
            outcome.arbitrated_actions
            if outcome.arbitrated_actions is not None
            else outcome.actions
        )
        p = self.p6
        p["robot_steps"] += len(arb)
        for i, a in enumerate(arb):
            if a is not Action.STAY:
                p["moves_arb"] += 1
                p["moves_arb_nonidle"] += not idle[i]
        p["moves_ctrl"] += sum(a is not Action.STAY for a in outcome.actions)
        p["k_arb"].update(e.kind_code for e in outcome.events)
        for e in outcome.events:
            if e.kind_code in ("ROv", "ROs"):
                rob = [int(x[1:]) for x in e.entities if x.startswith("r")]
                p["ro_arb_nonidle"] += sum(not idle[i] for i in rob)
        if outcome.controller_events is None:
            p["k_ctrl_available"] = False
        else:
            p["k_ctrl"].update(e.kind_code for e in outcome.controller_events)
        self.seconds += time.perf_counter() - t0

    def result(self) -> dict[str, Any]:
        out = super().result()
        p = dict(self.p6)
        p["k_arb"] = {k: p["k_arb"].get(k, 0) for k in ("RRv", "RRs", "ROv", "ROs", "INV")}
        p["k_ctrl"] = (
            {k: p["k_ctrl"].get(k, 0) for k in ("RRv", "RRs", "ROv", "ROs", "INV")}
            if p["k_ctrl_available"]
            else None
        )
        out["p6"] = p
        return out


# --------------------------------------------------------------------------- checkpoints


def checkpoints() -> dict[str, Any]:
    """Every checkpoint identity, verified read-only:

    * hist: Issue #5 ``best.pt``, intact against ``best.json``;
    * v2: the selected Phase 4 candidate (must be ``n3``), ``best.pt`` intact against
      ``phase4_run.json``;
    * v5 / i6: the Block D ``picks`` checks (SCB-C v5 output and Issue #6 picks, every check);
    """
    import block_d

    pk = block_d.picks()
    sel = p4.selected()
    out: dict[str, Any] = {"ok": bool(pk["ok"]) and sel == V2_CANDIDATE, "v2_selected": sel,
                           "scbc_v5": pk, "policies": {}}  # fmt: skip
    for pol in ("hist", "v2", "v5", "i6"):
        out["policies"][pol] = {}
        for run in POLICY_RUNS[pol]:
            if pol == "hist":
                path = Path(f"results/training/{run}/best.pt")
                ref = json.loads(path.with_name("best.json").read_text("utf-8"))["sha256"]
            elif pol == "v2":
                root = p4.TRAINING / f"{V2_CANDIDATE}_seed{run[-1]}"
                path = root / "best.pt"
                ref = json.loads((root / "phase4_run.json").read_text("utf-8"))["best_sha256"]
            else:
                info = pk["runs"][run]
                path = block_d.ckpt(run, info["new" if pol == "v5" else "old"])
                ref = info["sha256"]["new" if pol == "v5" else "old"]
            h = p3.sha256(path)
            out["ok"] &= h == ref
            out["policies"][pol][run] = {
                "path": str(path).replace("\\", "/"),
                "sha256": h,
                "intact": h == ref,
            }
    return out


# --------------------------------------------------------------------------- tasks


def jobs(ck: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for planner in PLANNERS:
        for seed in seeds(planner):
            for pol, cond in MATRIX[planner]:
                for run in POLICY_RUNS[pol]:
                    for level in LEVELS:
                        info = ck["policies"].get(pol, {}).get(run)
                        out.append(
                            {
                                "stage": "6",
                                "arm": f"{pol}:{cond}",
                                "policy": pol,
                                "cond": cond,
                                "run": run,
                                "planner": planner,
                                "level": level,
                                "seed": seed,
                                "checkpoint": info["path"] if info else None,
                                "checkpoint_sha256": info["sha256"] if info else None,
                            }
                        )
    # slow RRT* first; then seed by seed with every arm of an instance adjacent (same CPU load)
    return sorted(out, key=lambda j: (j["planner"] != "rrtstar", j["seed"], j["level"]))


def key(j: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (j["arm"], j["run"], j["planner"], j["level"], int(j["seed"]))


def episode_counts() -> dict[str, int]:
    out = {
        p: sum(len(POLICY_RUNS[pol]) for pol, _ in MATRIX[p]) * len(LEVELS) * len(seeds(p))
        for p in PLANNERS
    }
    out["total"] = sum(out.values())
    return out


def make_task(j: dict[str, Any], block: str = BLOCK) -> Any:
    from robocoord.marl.evaluation import EvalTask

    mode, deadlock = CONDS[j["cond"]]
    if j["policy"] == "base":
        if j["cond"] != "ORIG":
            raise ValueError("the baseline runs only in its original composition (decision D9)")
        return EvalTask("baseline", j["level"], int(j["seed"]), planner=j["planner"],
                        block=block, label="baseline")  # fmt: skip
    if j["cond"] == "ORIG":
        raise ValueError("ORIG is the baseline's condition only")
    label = (
        f"madql-v2-{V2_CANDIDATE}-{j['run'][-5:]}" if j["policy"] == "v2" else p3.label(j["run"])
    )
    return EvalTask(
        p3.FAMILIES[j["run"].split("_")[0]],
        j["level"],
        int(j["seed"]),
        planner=j["planner"],
        block=block,
        checkpoint=j["checkpoint"],
        label=label,
        supervisor=mode,
        deadlock=deadlock,
    )


def _run(j: dict[str, Any], observe: bool = True, block: str = BLOCK) -> dict[str, Any]:
    import selection_block

    from robocoord.marl import evaluation
    from robocoord.marl.scenarios import LEVELS as DISTS

    if block == BLOCK:
        guard(j["planner"], int(j["seed"]))  # layer 1, before the sampler
    else:
        anchor_guard(j["planner"], int(j["seed"]), block)
    spec = DISTS[j["level"]].sample(int(j["seed"]))
    meta = {k: j[k] for k in ("stage", "arm", "policy", "cond", "run", "checkpoint_sha256")}
    meta["checkpoint_path"] = j["checkpoint"]
    meta["instance_sha256"] = selection_block.fingerprint(spec)
    task = make_task(j, block)
    obs = Phase6Observer(spec.grid.blocked.copy(), len(spec.robots)) if observe else None
    p3._SUP_TIMER[0] = 0.0
    try:
        rec: dict[str, Any] = evaluation.run_episode(task, observer=obs)  # layers 2 and 3
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


def _worker_probe(planner: str, seed: int, block: str) -> dict[str, str]:
    """Test support, run inside a real worker process: which guard layer stops a task. It never
    generates a scenario: for a task that passes every guard it returns without running it."""
    from robocoord.marl import evaluation
    from robocoord.marl.evaluation import EvalTask

    out: dict[str, str] = {"guards_installed": str(getattr(evaluation.run_episode,
                                                           "phase6_guarded", False))}  # fmt: skip
    task = EvalTask("baseline", "static", seed, planner=planner, block=block)
    for name, call in (
        ("check_seeds", lambda: getattr(evaluation, "check_seeds")([seed], block)),  # noqa: B009
        ("guard_task", lambda: guard_task(task)),
    ):
        try:
            call()
            out[name] = "passed"
        except ValueError as exc:
            out[name] = f"rejected: {exc}"
    if out["guard_task"] == "passed":
        return out  # an allowed task: never run it here
    try:
        evaluation.run_episode(task)
        out["run_episode"] = "ran"  # must never happen
    except ValueError as exc:
        out["run_episode"] = f"rejected: {exc}"
    return out


def _run_anchor(j: dict[str, Any]) -> dict[str, Any]:
    return _run(j, observe=True, block="C")


# --------------------------------------------------------------------------- freeze and gates


def _hashes() -> dict[str, str | None]:
    return {f: (p3.sha256(Path(f)) if Path(f).exists() else None) for f in (str(PROTOCOL),
            *FREEZE_FILES)}  # fmt: skip


def _waiver() -> dict[str, Any] | None:
    """The owner-approved D12 waiver for this Phase 6 execution, if recorded and well formed."""
    if not BACKUP_WAIVER.exists():
        return None
    w: dict[str, Any] = json.loads(BACKUP_WAIVER.read_text("utf-8"))
    ok = (w.get("owner_approved") is True and w.get("d12_satisfied") is False
          and w.get("scope") == "Issue 6A Phase 6 execution (this protocol version)")  # fmt: skip
    return w if ok else None


def backup_status() -> dict[str, Any]:
    """D12 gate. ``ok`` (the gate passes) iff the originals verify against the manifest **and**
    either two independent copies are recorded and verified (``d12_satisfied``) or the
    owner-approved waiver is recorded (``d12_waived``). A waiver never counts as D12 satisfied."""
    import backup_manifest as bm

    if not BACKUP_MANIFEST.exists():
        return {"ok": False, "reason": "no backup manifest"}
    man = json.loads(BACKUP_MANIFEST.read_text("utf-8"))
    msha = p3.sha256(BACKUP_MANIFEST)
    dests: list[dict[str, Any]] = []
    if BACKUP_RECORD.exists():
        dests = json.loads(BACKUP_RECORD.read_text("utf-8")).get("destinations", [])
    copies_ok = len(dests) >= 2 and all(
        d.get("manifest_sha256") == msha
        and d.get("verified", {}).get("ok")
        and d["verified"].get("checked") == man["file_count"]
        for d in dests
    )
    w = _waiver()
    orig = bm.verify(man, Path("."))
    return {
        "ok": bool(orig["ok"] and (copies_ok or w is not None)),
        "d12_satisfied": copies_ok,
        "d12_waived": w is not None and not copies_ok,
        "waiver_sha256": p3.sha256(BACKUP_WAIVER) if w is not None else None,
        "manifest_sha256": msha,
        "file_count": man["file_count"],
        "record_sha256": p3.sha256(BACKUP_RECORD) if BACKUP_RECORD.exists() else None,
        "destinations": [d.get("path") for d in dests],
        "copies_ok": copies_ok,
        "originals_ok": orig["ok"],
    }


def freeze(out: Path, authorized: bool) -> dict[str, Any]:
    if not authorized:
        raise SystemExit("freeze needs --authorized-freeze (the student's explicit approval)")
    if (out / "check.json").exists() or (out / "records.jsonl").exists():
        raise SystemExit("refusing to freeze: a check or record file already exists")
    h = _hashes()
    missing = [f for f, v in h.items() if v is None]
    if missing:
        raise SystemExit(f"cannot freeze, missing files: {missing}")
    bk = backup_status()
    if not bk["ok"]:
        raise SystemExit(
            f"cannot freeze: no verified backups and no owner-approved waiver (D12), or the "
            f"originals do not verify: {bk}"
        )
    res = {
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "files_sha256": h,
        "code_sha256": {c: p3.sha256(Path(c)) for c in CODE},
        "design": {"block": BLOCK, "astar_seeds": [BLOCK_V.start, BLOCK_V.stop - 1],
                   "rrtstar_seeds": [BLOCK_V_RRT.start, BLOCK_V_RRT.stop - 1],
                   "matrix": {p: [list(c) for c in MATRIX[p]] for p in PLANNERS},
                   "episodes": episode_counts(), "v2_candidate": V2_CANDIDATE},
        "environment": p3._env(),
        "backups": bk,
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan_freeze.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    return res


def verify_freeze(out: Path) -> dict[str, Any]:
    path = out / "plan_freeze.json"
    if not path.exists():
        raise SystemExit("no freeze record: the protocol must be approved and frozen first")
    fr: dict[str, Any] = json.loads(path.read_text("utf-8"))
    now = _hashes()
    bad = [f for f, v in fr["files_sha256"].items() if now.get(f) != v]
    bad += [c for c, v in fr["code_sha256"].items() if p3.sha256(Path(c)) != v]
    if bad:
        raise SystemExit(f"frozen files changed since the freeze: {bad}")
    return fr


def _authorize(flag: bool, name: str = "--authorized-execution") -> None:
    if not flag:
        raise SystemExit(f"this command needs {name} (separate student authorization)")


def scan_results(out: Path) -> dict[str, Any]:
    """Stored files (outside ``out``) with a Block V seed field. Read-only."""
    import re

    pat = re.compile(r'"seed"\s*:\s*(\d+)')
    hits: dict[str, int] = {}
    n = 0
    for root, _, files in os.walk("results"):
        if Path(root).resolve().is_relative_to(out.resolve()):
            continue
        for f in files:
            if not f.endswith((".json", ".jsonl", ".csv")):
                continue
            n += 1
            txt = Path(root, f).read_text("utf-8", errors="ignore")
            c = sum(int(m.group(1)) in BLOCK_V for m in pat.finditer(txt))
            if c:
                hits[str(Path(root, f))] = c
    return {"files_scanned": n, "files_with_block_v_seeds": hits, "clean": not hits}


def instance_check(execution_authorized: bool = False) -> dict[str, Any]:
    """The first generation of Block V (amendment A1: only inside ``run``, after the separate
    execution authorization): unique per level and disjoint from every A, B, C, D and S instance
    of the same level (fingerprints). Refuses before generating anything without the
    authorization."""
    _authorize(execution_authorized)
    import selection_block

    from robocoord.marl.scenarios import LEVELS as DISTS

    others = {k: v for k, v in FORBIDDEN.items() if k not in ("train", "test")}
    res: dict[str, Any] = {"levels": {}}
    ok = True
    for lv in LEVELS:
        v = {s: selection_block.fingerprint(DISTS[lv].sample(s)) for s in BLOCK_V}
        prev = {selection_block.fingerprint(DISTS[lv].sample(s)) for r in others.values()
                for s in r}  # fmt: skip
        uniq = len(set(v.values())) == len(v)
        overlap = len(set(v.values()) & prev)
        ok &= uniq and overlap == 0
        res["levels"][lv] = {"instances": len(v), "unique": uniq, "overlap_with_A_B_C_D_S": overlap,
                             "fingerprints": v}  # fmt: skip
    res["ok"] = ok
    res["training_disjointness"] = {
        "verified": False,
        "reason": "no training instance hashes are stored; training seeds 0–9 999 are a disjoint "
        "range and every instance is generated from its own seed (as Block D §8.6)",
    }
    return res


def check(out: Path, authorized: bool) -> dict[str, Any]:
    """The pre-run check (amendment A1): every requirement that does not involve Block V. No
    Block V scenario is generated, materialized or fingerprinted here."""
    _authorize(authorized, "--authorized-precheck")
    fr = verify_freeze(out)
    from robocoord.control.supervisor import FROZEN_PARAMS

    stage3a = json.loads(p3.STAGE3A_FREEZE.read_text("utf-8"))
    frozen_ok = p3.sha256(p3.STAGE3A_FREEZE) == p3.STAGE3A_FREEZE_SHA256 and all(
        stage3a["frozen_params"][p] == {k: getattr(v, k) for k in v.__slots__}
        for p, v in FROZEN_PARAMS.items()
    )
    scan = scan_results(out)
    ck = checkpoints()
    bk = backup_status()
    res: dict[str, Any] = {
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "freeze_sha256": p3.sha256(out / "plan_freeze.json"),
        "stage3a_freeze_ok": frozen_ok,
        "escalation_disabled": all(not v.escalate for v in FROZEN_PARAMS.values()),
        "results_scan": scan,
        "checkpoints": ck,
        "backups": bk,
        "block_v_instance_check": "deferred to run, after the execution authorization (A1)",
        "episodes": len(jobs(ck)),
        "expected_episodes": fr["design"]["episodes"]["total"],
        "environment": p3._env(),
    }
    res["ok"] = bool(frozen_ok and res["escalation_disabled"] and scan["clean"] and ck["ok"]
                     and bk["ok"] and res["episodes"] == res["expected_episodes"])  # fmt: skip
    (out / "check.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    return res


def _gate(out: Path) -> dict[str, Any]:
    verify_freeze(out)
    chk: dict[str, Any] = json.loads((out / "check.json").read_text("utf-8"))
    if not chk["ok"]:
        raise SystemExit("check is not ok")
    return chk


def block_v_gate(out: Path, authorized: bool) -> dict[str, Any]:
    """Amendment A1: the Block V instance check, once, after the execution authorization and
    before any episode. A failure aborts the run; nothing is executed."""
    _authorize(authorized)
    path = out / "instance_check.json"
    if path.exists():
        ic: dict[str, Any] = json.loads(path.read_text("utf-8"))
    else:
        if (out / "records.jsonl").exists():
            raise SystemExit("records exist without a Block V instance check: aborting")
        ic = instance_check(execution_authorized=True)
        ic["checked_at"] = dt.datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(ic, indent=1, sort_keys=True), "utf-8")
    if not ic.get("ok"):
        with (out / "run.log").open("a", encoding="utf-8") as lf:
            lf.write(f"{dt.datetime.now().isoformat(timespec='seconds')} ABORT: Block V instance "
                     "check failed; no episode executed\n")  # fmt: skip
        raise SystemExit("Block V instance check failed: aborted before any episode (A1)")
    return ic


def run(out: Path, workers: int, authorized: bool) -> int:
    _authorize(authorized)
    chk = _gate(out)
    block_v_gate(out, authorized)
    target = out / "records.jsonl"
    done = {key(r) for r in p1._records(target)}
    todo = [j for j in jobs(chk["checkpoints"]) if key(j) not in done]
    paths = {f"{p}/{r}": v["path"] for p, x in chk["checkpoints"]["policies"].items()
             for r, v in x.items()}  # fmt: skip
    before = {k: p3.sha256(Path(v)) for k, v in paths.items()}
    with (out / "run.log").open("a", encoding="utf-8") as lf:
        lf.write(f"{dt.datetime.now().isoformat(timespec='seconds')} run: {len(done)} done, "
                 f"{len(todo)} to do, workers {workers}\n")  # fmt: skip
    t0 = time.time()
    n = 0
    with (
        target.open("a", encoding="utf-8") as fh,
        ProcessPoolExecutor(workers, initializer=_init_worker, initargs=("v", True)) as ex,
    ):
        for rec in ex.map(_run, todo, chunksize=1):
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 250 == 0:
                fh.flush()
                print(f"phase 6: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    after = {k: p3.sha256(Path(v)) for k, v in paths.items()}
    errors = sum(r["status"] != "ok" for r in p1._records(target))
    with (out / "run.log").open("a", encoding="utf-8") as lf:
        lf.write(f"{dt.datetime.now().isoformat(timespec='seconds')} run finished: {n} new in "
                 f"{time.time() - t0:.0f}s; errors {errors}; checkpoints unchanged "
                 f"{before == after}\n")  # fmt: skip
    if before != after:
        raise SystemExit("CHECKPOINT HASH CHANGED DURING THE RUN")
    print(f"phase 6: {n} episodes, errors {errors}")
    return 0


def _compare(a: dict[str, Any], b: dict[str, Any], skip: set[str]) -> list[str]:
    return sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))


def replay(out: Path, workers: int, authorized: bool) -> dict[str, Any]:
    _authorize(authorized)
    _gate(out)
    if not (out / "instance_check.json").exists():
        raise SystemExit("no Block V instance check: run first")
    recs = sorted(p1._records(out / "records.jsonl"), key=key)
    sample = recs[::REPLAY_STRIDE]
    fields = ("stage", "arm", "policy", "cond", "run", "planner", "level", "seed")
    js = [{**{k: r[k] for k in fields}, "checkpoint": r["checkpoint_path"],
           "checkpoint_sha256": r["checkpoint_sha256"]} for r in sample]  # fmt: skip
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=("v", False)) as ex:
        again = list(ex.map(_run_plain, js, chunksize=1))
    skip = set(TIMING) | set(OBS_FIELDS)
    diffs = [{"key": list(key(a)), "fields": d} for a, b in zip(sample, again, strict=True)
             if (d := _compare(a, b, skip))]  # fmt: skip
    res = {"replayed": len(sample), "stride": REPLAY_STRIDE, "differences": diffs,
           "identical": not diffs}  # fmt: skip
    (out / "replay.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


# --------------------------------------------------------------------------- anchor (Block C)

#: Stored Block C development records that this runner must reproduce (non-timing fields).
ANCHORS: tuple[tuple[str, str, str, str, str], ...] = (
    # (records file, stored arm, Phase 6 policy, Phase 6 condition, planner)
    ("results/dev6a/phase1/eval/records_1a.jsonl", "off_r0", "hist|v5", "OFF", "astar"),
    ("results/dev6a/phase1/eval/records_1a.jsonl", "rsd_r0", "hist|v5", "RSD", "astar"),
    ("results/dev6a/phase1/eval/records_1b.jsonl", "off_r0", "hist|v5", "OFF", "rrtstar"),
    ("results/dev6a/phase1/eval/records_1b.jsonl", "rsd_r0", "hist|v5", "RSD", "rrtstar"),
    ("results/dev6a/phase4/eval/records_4c.jsonl", "v2:NA", "v2", "OFF", "astar"),
    ("results/dev6a/phase4/eval/records_4c.jsonl", "v2:FA", "v2", "RSD", "astar"),
    ("results/dev6a/phase4/eval/records_4c.jsonl", "v2:FR", "v2", "RSD", "rrtstar"),
    ("results/dev6a/block_c/records.jsonl", "old", "i6", "OFF", "astar"),
    ("results/dev6a/phase2/eval/records_2b.jsonl", "full", "base", "ORIG", "rrtstar"),
)
ANCHOR_SKIP = {*TIMING, *OBS_FIELDS, *META, "label", "block", "rrt", "path_checks",
               "rrt_refine", "rrt_patience", "repair", "observer_s", "error_type",
               "error_message", "checkpoint", "arm_label", "pair", "candidate", "policy_arm",
               "replicate_of"}  # fmt: skip


def anchor_jobs(ck: dict[str, Any], stride: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for path, arm, pols, cond, planner in ANCHORS:
        recs = [r for r in p1._records(Path(path))
                if r.get("arm") == arm and r["planner"] == planner]  # fmt: skip
        recs.sort(key=lambda r: (r.get("run", r.get("label", "")), r["level"], int(r["seed"])))
        for r in recs[::stride]:
            run = r.get("run") or r.get("label")
            if run == "baseline" or r.get("controller") == "baseline":
                pol, run = "base", BASELINE_RUN
            else:
                pol = next(p for p in pols.split("|") if run in POLICY_RUNS[p])
            info = ck["policies"].get(pol, {}).get(run)
            if info is not None and r.get("checkpoint_sha256") not in (None, info["sha256"]):
                continue  # a different checkpoint than the Phase 6 arm (not comparable)
            j = {"stage": "anchor", "arm": f"{pol}:{cond}", "policy": pol, "cond": cond,
                 "run": run, "planner": planner, "level": r["level"], "seed": int(r["seed"]),
                 "checkpoint": info["path"] if info else None,
                 "checkpoint_sha256": info["sha256"] if info else None}  # fmt: skip
            out.append((j, r))
    return out


def anchor(out: Path, workers: int, stride: int) -> dict[str, Any]:
    """Re-run a stride sample of stored Block C development records through this runner (with
    the observer on) and compare every shared non-timing field. Block C only."""
    ck = checkpoints()
    pairs = anchor_jobs(ck, stride)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=("c", True)) as ex:
        again = list(ex.map(_run_anchor, [j for j, _ in pairs], chunksize=1))
    rows = []
    by: Counter[str] = Counter()
    bad: Counter[str] = Counter()
    for (j, stored), new in zip(pairs, again, strict=True):
        shared = {k for k in stored if k in new} - ANCHOR_SKIP
        d = sorted(k for k in shared if stored[k] != new[k])
        cell = f"{j['arm']}|{j['planner']}"
        by[cell] += 1
        if d or new["status"] != "ok":
            bad[cell] += 1
            rows.append({"key": list(key(j)), "fields": d, "status": new["status"]})
        else:
            p6 = new["p6"]
            if p6["k_arb"] != {k: stored["conflicts"][k] for k in p6["k_arb"]}:
                bad[cell] += 1
                rows.append({"key": list(key(j)), "fields": ["p6.k_arb != conflicts"]})
    res = {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "stride": stride,
           "compared": dict(sorted(by.items())), "mismatched": dict(sorted(bad.items())),
           "differences": rows[:50], "identical": not rows,
           "block": "C (development data; not confirmatory evidence)"}  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "anchor.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("design", "anchor", "freeze", "check", "run", "replay"))
    ap.add_argument("--out", type=Path, default=Path("results/dev6a/phase6/eval"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--authorized-freeze", action="store_true")
    ap.add_argument("--authorized-precheck", action="store_true")
    ap.add_argument("--authorized-execution", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if args.cmd == "design":
        print(json.dumps({"episodes": episode_counts(),
                          "matrix": {p: [list(c) for c in MATRIX[p]] for p in PLANNERS}}))  # fmt: skip
        return 0
    if args.cmd == "anchor":
        res = anchor(args.out, args.workers, args.stride)
        print(json.dumps({k: res[k] for k in ("compared", "mismatched", "identical")}))
        return 0 if res["identical"] else 1
    if args.cmd == "freeze":
        freeze(args.out, args.authorized_freeze)
        return 0
    if args.cmd == "check":
        res = check(args.out, args.authorized_precheck)
        print(json.dumps({"ok": res["ok"], "episodes": res["episodes"]}))
        return 0 if res["ok"] else 1
    if args.cmd == "run":
        return run(args.out, args.workers, args.authorized_execution)
    res = replay(args.out, args.workers, args.authorized_execution)
    print(json.dumps({k: res[k] for k in ("replayed", "identical")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
