"""Issue 6A Phase 4 — MADQL-v2 evaluation runner (development selection on Block S, formal
evaluation on Block C). Protocol: ``docs/research/2026-11-22-issue-6a-phase4-madql.md`` §3.5
(selection) and §5 (Block C).

Conditions (greedy; auction; shield off; endpoint recovery on; ``path_limit`` 6; horizon 400;
P1 repair off; RRT\\* with its full 1 000-iteration budget, P2 off):

======  ========  ===============================================  =========================
cond    planner   composition                                      seeds
======  ========  ===============================================  =========================
FA      A\\*       R + S + D (``supervisor="R+S"``, resolver on)      S: 13 000–13 079; C: 11 000–11 119
NA      A\\*       no supervisor, resolver off (Issue #6)            S: 13 000–13 079; C: 11 000–11 119
FR      RRT\\*     R + S + D                                         C: 11 000–11 039 (Block C only)
======  ========  ===============================================  =========================

Policy arms: ``hist`` (the historical MADQL picks, Issue #5 ``best.pt`` of seeds 1–3) and the
MADQL-v2 candidates (``n3``, ``t8``, ``n3t8``: ``best.pt`` of each candidate's seeds 1–3); in
stage ``4c`` the selected candidate is the arm ``v2``. A record's ``arm`` is ``"<policy>:<cond>"``.

The observer adds MADQL decision statistics (``madql``): STAY decisions (with a free / occupied
suggested cell), self-loop STAYs (observation identical to the robot's previous one), STAY spells
of ≥ 20 steps, the action distribution, histograms of the gap Q(STAY) − best allowed move, and the
policy inference time. It wraps ``LearnedController.choose`` read-only (an extra forward pass that
is excluded from the inference timing); ``replay`` re-runs every 50th episode without it.

Commands::

    python tools/phase4/phase4.py check  --out results/dev6a/phase4/select --stage 4s
    python tools/phase4/phase4.py run    --out results/dev6a/phase4/select --stage 4s --workers 10
    python tools/phase4/phase4.py replay --out results/dev6a/phase4/select --stage 4s --workers 10
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sub in ("phase2", "phase1", "phase3", "selection"):
    sys.path.insert(0, str(HERE.parent / _sub))

import phase1 as p1  # noqa: E402
import phase2 as p2  # noqa: E402
import phase3 as p3  # noqa: E402

PROTOCOL = Path("docs/research/2026-11-22-issue-6a-phase4-madql.md")
TRAINING = Path("results/dev6a/phase4/training")
SELECTION = Path("results/dev6a/phase4/select/selection.json")
RUNS = ("madql_seed1", "madql_seed2", "madql_seed3")
CANDIDATES = ("n3", "t8", "n3t8")
BLOCK_S = range(13_000, 13_080)
CONDS: dict[str, tuple[str, str, bool]] = {
    "FA": ("astar", "R+S", True),
    "NA": ("astar", "off", False),
    "FR": ("rrtstar", "R+S", True),
}
STAGES: dict[str, dict[str, Any]] = {
    "4s": {"block": "S", "conds": ("FA", "NA"), "arms": ("hist", *CANDIDATES)},
    "4c": {"block": "C", "conds": ("FA", "NA", "FR"), "arms": ("hist", "v2")},
}
FORBIDDEN = {k: v for k, v in p3.FORBIDDEN.items() if k != "S"}
REPLAY_STRIDE = 50
TIMING = (*p2.TIMING, "inference_s")
OBS_FIELDS = (
    "stuck", "tasks", "sup_changed", "sup_invalid", "observer_s", "planning", "planning_s",
    "rrt", "path_checks", "madql",
)  # fmt: skip
CODE = (*p2.CODE, "src/robocoord/marl/madql.py", "src/robocoord/marl/replay.py",
        "src/robocoord/marl/trainer.py", "tools/phase4/phase4.py")  # fmt: skip
#: Phase 1 records that the historical arm must reproduce exactly in stage 4c (G0.1).
PHASE1_RECORDS = {
    "FA": (Path("results/dev6a/phase1/eval/records_1a.jsonl"), "rsd_r0"),
    "NA": (Path("results/dev6a/phase1/eval/records_1a.jsonl"), "off_r0"),
    "FR": (Path("results/dev6a/phase1/eval/records_1b.jsonl"), "rsd_r0"),
}
#: Gap histogram bin edges (Q units; the per-step cost is 0.01).
GAP_EDGES = (-0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
SPELL = 20


def seeds(stage: str, cond: str) -> range:
    if STAGES[stage]["block"] == "S":
        return BLOCK_S
    return p3.BLOCK_C if CONDS[cond][0] == "astar" else p3.BLOCK_C_RRT


def guard(stage: str, cond: str, seed: int) -> None:
    for name, r in FORBIDDEN.items():
        if seed in r:
            raise ValueError(f"block {name} seed {seed} is not allowed in Phase 4")
    if STAGES[stage]["block"] == "S" and seed in p3.BLOCK_C:
        raise ValueError("Block C is not allowed in the selection stage")
    if STAGES[stage]["block"] == "C" and seed in BLOCK_S:
        raise ValueError("Block S is not allowed in the Block C stage")
    if seed not in seeds(stage, cond):
        raise ValueError(f"seed {seed} not in the {stage} / {cond} design")


# --------------------------------------------------------------------------- worker hooks
_CAP: dict[str, Any] = {"q": None, "obs": None, "inference_s": 0.0, "calls": 0}


def _allowed_seeds(stage: str) -> set[int]:
    return {s for c in STAGES[stage]["conds"] for s in seeds(stage, c)}


def _init_worker(stage: str, observe: bool) -> None:
    from robocoord.marl import evaluation
    from robocoord.marl.policy import LearnedController

    allowed = _allowed_seeds(stage)

    def check(ss: Any, block: str) -> list[int]:
        out = [int(s) for s in ss]
        if block != STAGES[stage]["block"] or any(s not in allowed for s in out):
            raise ValueError(f"seeds {out[:5]} / block {block!r} outside the {stage} design")
        return out

    setattr(evaluation, "check_seeds", check)  # noqa: B010 (module global)
    if not observe or getattr(LearnedController.choose, "phase4", False):
        return
    import torch

    original = LearnedController.choose

    def choose(self: Any, obs: Any, masks: Any) -> Any:
        t0 = time.perf_counter()
        acts = original(self, obs, masks)
        _CAP["inference_s"] += time.perf_counter() - t0
        _CAP["calls"] += 1
        with torch.no_grad():
            _CAP["q"] = self.net(torch.from_numpy(np.ascontiguousarray(obs))).numpy()
        _CAP["obs"] = [zlib.crc32(np.ascontiguousarray(o).tobytes()) for o in obs]
        return acts

    choose.phase4 = True  # type: ignore[attr-defined]
    LearnedController.choose = choose  # type: ignore[method-assign]


class Phase4Observer(p2.Phase2Observer):
    """The Phase 2 observer (stuck classes, tasks, supervisor checks, planner work, RRT\\* calls,
    reference-path checks) plus MADQL decision statistics."""

    def __init__(self, base_blocked: Any, n: int) -> None:
        super().__init__(base_blocked)
        k = len(GAP_EDGES) + 1
        self.m: dict[str, Any] = {
            "nonidle_steps": 0, "stay": 0, "stay_free": 0, "stay_occupied": 0, "follow": 0,
            "self_loop_stay": 0, "spell_ge20_steps": 0, "spells_ge20": 0,
            "actions": [0] * 5, "gap_hist_all": [0] * k, "gap_hist_stay": [0] * k,
            "gap_hist_move": [0] * k, "gap_sum": 0.0, "gap_n": 0, "q_chosen_sum": 0.0,
        }  # fmt: skip
        self._prev_obs: list[int | None] = [None] * n
        self._spell = [0] * n
        self._last_t = [-2] * n
        self._t = 0
        _CAP.update(q=None, obs=None, inference_s=0.0, calls=0)

    def _close_spell(self, i: int) -> None:
        if self._spell[i] >= SPELL:
            self.m["spell_ge20_steps"] += self._spell[i]
            self.m["spells_ge20"] += 1
        self._spell[i] = 0

    def before(self, engine: Any, view: Any, actions: Any) -> None:
        super().before(engine, view, actions)
        t0 = time.perf_counter()
        q, oh = _CAP["q"], _CAP["obs"]
        pre = self._pre
        m = self.m
        for i in range(len(pre["act"])):
            same = oh is not None and self._prev_obs[i] == oh[i]
            if oh is not None:
                self._prev_obs[i] = oh[i]
            if pre["idle"][i]:
                self._close_spell(i)
                continue
            a = pre["act"][i]
            m["nonidle_steps"] += 1
            m["actions"][a] += 1
            sug = pre["sug"][i] if pre["sug"] is not None else -1
            m["follow"] += a == sug
            if q is not None:
                qi = q[i]
                allowed = [qi[b] for b in range(1, 5) if pre["masks"][i][b]]
                if allowed:
                    gap = float(qi[0] - max(allowed))
                    b = int(np.searchsorted(GAP_EDGES, gap, side="right"))
                    m["gap_hist_all"][b] += 1
                    m["gap_hist_stay" if a == 0 else "gap_hist_move"][b] += 1
                    m["gap_sum"] += gap
                    m["gap_n"] += 1
                m["q_chosen_sum"] += float(qi[a])
            if a == 0:
                m["stay"] += 1
                occ = pre["next_occ"][i]
                if sug > 0 and occ == "free":
                    m["stay_free"] += 1
                elif sug > 0 and occ in ("idle", "active"):
                    m["stay_occupied"] += 1
                m["self_loop_stay"] += same
                if self._last_t[i] == self._t - 1 and self._spell[i] > 0:
                    self._spell[i] += 1
                else:
                    self._close_spell(i)
                    self._spell[i] = 1
            else:
                self._close_spell(i)
            self._last_t[i] = self._t
        self._t += 1
        self.seconds += time.perf_counter() - t0

    def result(self) -> dict[str, Any]:
        out = super().result()
        for i in range(len(self._spell)):
            self._close_spell(i)
        out["madql"] = {
            **self.m,
            "gap_edges": list(GAP_EDGES),
            "inference_s": _CAP["inference_s"],
            "inference_calls": _CAP["calls"],
        }
        return out


# --------------------------------------------------------------------------- checkpoints


def selected() -> str:
    sel = json.loads(SELECTION.read_text("utf-8"))
    return str(sel["selected"])


def checkpoints(stage: str) -> dict[str, Any]:
    """Historical picks (intact against ``best.json``) and candidate ``best.pt`` (intact against
    the run's ``phase4_run.json``); read-only."""
    out: dict[str, Any] = {"ok": True, "arms": {}}
    arms = STAGES[stage]["arms"]
    for arm in arms:
        cand = selected() if arm == "v2" else arm
        out["arms"][arm] = {"candidate": None if arm == "hist" else cand, "runs": {}}
        for run in RUNS:
            k = run.split("seed")[1]
            if arm == "hist":
                root = Path(f"results/training/{run}")
                path = root / "best.pt"
                ref = json.loads((root / "best.json").read_text("utf-8"))["sha256"]
            else:
                root = TRAINING / f"{cand}_seed{k}"
                path = root / "best.pt"
                ref = json.loads((root / "phase4_run.json").read_text("utf-8"))["best_sha256"]
            h = p3.sha256(path)
            out["ok"] &= h == ref
            out["arms"][arm]["runs"][run] = {
                "path": str(path).replace("\\", "/"),
                "sha256": h,
                "intact": h == ref,
            }
    return out


# --------------------------------------------------------------------------- tasks


def jobs(stage: str, ck: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for cond in STAGES[stage]["conds"]:
        for seed in seeds(stage, cond):
            for run in RUNS:
                for level in p3.LEVELS:
                    for arm in STAGES[stage]["arms"]:
                        info = ck["arms"][arm]
                        out.append(
                            {
                                "stage": stage,
                                "arm": f"{arm}:{cond}",
                                "policy_arm": arm,
                                "candidate": info["candidate"],
                                "cond": cond,
                                "run": run,
                                "planner": CONDS[cond][0],
                                "level": level,
                                "seed": seed,
                                "checkpoint": info["runs"][run]["path"],
                                "checkpoint_sha256": info["runs"][run]["sha256"],
                            }
                        )
    # slow RRT* episodes first, then interleave by seed
    return sorted(out, key=lambda j: (j["planner"] != "rrtstar", j["seed"], j["cond"]))


def key(j: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (j["arm"], j["run"], j["planner"], j["level"], int(j["seed"]))


def make_task(j: dict[str, Any]) -> Any:
    from robocoord.marl.evaluation import EvalTask

    planner, mode, deadlock = CONDS[j["cond"]]
    lab = (
        p3.label(j["run"])
        if j["policy_arm"] == "hist"
        else f"madql-v2-{j['candidate']}-{j['run'][-5:]}"
    )
    return EvalTask(
        "madql",
        j["level"],
        int(j["seed"]),
        planner=planner,
        block=STAGES[j["stage"]]["block"],
        checkpoint=j["checkpoint"],
        label=lab,
        supervisor=mode,
        deadlock=deadlock,
    )


def _run(j: dict[str, Any], observe: bool = True) -> dict[str, Any]:
    import selection_block

    from robocoord.marl.evaluation import run_episode
    from robocoord.marl.scenarios import LEVELS as DISTS

    guard(j["stage"], j["cond"], int(j["seed"]))
    spec = DISTS[j["level"]].sample(int(j["seed"]))
    meta = {k: j[k] for k in ("stage", "arm", "policy_arm", "candidate", "cond", "run",
                              "checkpoint_sha256")}  # fmt: skip
    meta["checkpoint_path"] = j["checkpoint"]
    meta["instance_sha256"] = selection_block.fingerprint(spec)
    task = make_task(j)
    obs = Phase4Observer(spec.grid.blocked, len(spec.robots)) if observe else None
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


# --------------------------------------------------------------------------- commands


def check(out: Path, stage: str) -> dict[str, Any]:
    from robocoord.control.supervisor import FROZEN_PARAMS

    freeze = json.loads(p3.STAGE3A_FREEZE.read_text("utf-8"))
    frozen_ok = p3.sha256(p3.STAGE3A_FREEZE) == p3.STAGE3A_FREEZE_SHA256 and all(
        freeze["frozen_params"][p] == {k: getattr(v, k) for k in v.__slots__}
        for p, v in FROZEN_PARAMS.items()
    )
    ck = checkpoints(stage)
    res: dict[str, Any] = {
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "stage3a_freeze_ok": frozen_ok,
        "escalation_disabled": all(not v.escalate for v in FROZEN_PARAMS.values()),
        "checkpoints": ck,
        "conditions": {c: list(CONDS[c]) for c in STAGES[stage]["conds"]},
        "seeds": {
            c: [seeds(stage, c).start, seeds(stage, c).stop - 1] for c in STAGES[stage]["conds"]
        },
        "episodes": len(jobs(stage, ck)),
        "protocol_sha256_at_check": p3.sha256(PROTOCOL),
        "selection_sha256": p3.sha256(SELECTION) if stage == "4c" else None,
        "code_sha256": {c: p3.sha256(Path(c)) for c in CODE},
        "environment": p3._env(),
    }
    res["ok"] = bool(frozen_ok and ck["ok"] and res["escalation_disabled"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"check_{stage}.json").write_text(json.dumps(res, indent=1, sort_keys=True), "utf-8")
    return res


def _gate(out: Path, stage: str) -> dict[str, Any]:
    chk: dict[str, Any] = json.loads((out / f"check_{stage}.json").read_text("utf-8"))
    if not chk["ok"]:
        raise SystemExit("check is not ok")
    now = {c: p3.sha256(Path(c)) for c in CODE}
    if now != chk["code_sha256"]:
        raise SystemExit(
            f"code changed since check: {[c for c in CODE if now[c] != chk['code_sha256'][c]]}"
        )
    if stage == "4c" and p3.sha256(SELECTION) != chk["selection_sha256"]:
        raise SystemExit("selection.json changed since check")
    return chk


def run(out: Path, stage: str, workers: int) -> int:
    chk = _gate(out, stage)
    target = out / f"records_{stage}.jsonl"
    done = {key(r) for r in p1._records(target)}
    todo = [j for j in jobs(stage, chk["checkpoints"]) if key(j) not in done]
    paths = {
        f"{a}/{r}": v["path"]
        for a, x in chk["checkpoints"]["arms"].items()
        for r, v in x["runs"].items()
    }
    hist = {f"results/training/{r}/best.pt" for r in RUNS}
    before = {k: p3.sha256(Path(v)) for k, v in paths.items()}
    hist_before = {h: p3.sha256(Path(h)) for h in hist}
    with (out / "run.log").open("a", encoding="utf-8") as lf:
        lf.write(f"{dt.datetime.now().isoformat(timespec='seconds')} run {stage}: {len(done)} done, "
                 f"{len(todo)} to do, workers {workers}\n")  # fmt: skip
    t0 = time.time()
    n = 0
    with (
        target.open("a", encoding="utf-8") as fh,
        ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(stage, True)) as ex,
    ):
        for rec in ex.map(_run, todo, chunksize=1):
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 250 == 0:
                fh.flush()
                print(f"{stage}: {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    after = {k: p3.sha256(Path(v)) for k, v in paths.items()}
    unchanged = before == after and hist_before == {h: p3.sha256(Path(h)) for h in hist}
    errors = sum(r["status"] != "ok" for r in p1._records(target))
    with (out / "run.log").open("a", encoding="utf-8") as lf:
        lf.write(f"{dt.datetime.now().isoformat(timespec='seconds')} run {stage} finished: {n} new in "
                 f"{time.time() - t0:.0f}s; errors {errors}; checkpoints unchanged {unchanged}\n")  # fmt: skip
    if not unchanged:
        raise SystemExit("CHECKPOINT HASH CHANGED DURING THE RUN")
    print(f"{stage}: {n} episodes, errors {errors}")
    return 0


def replay(out: Path, stage: str, workers: int) -> dict[str, Any]:
    _gate(out, stage)
    recs = p1._records(out / f"records_{stage}.jsonl")
    recs.sort(key=lambda r: key(r))
    sample = recs[::REPLAY_STRIDE]
    fields = ("stage", "arm", "policy_arm", "candidate", "cond", "run", "planner", "level", "seed")
    js = [
        {
            **{k: r[k] for k in fields},
            "checkpoint": r["checkpoint_path"],
            "checkpoint_sha256": r["checkpoint_sha256"],
        }
        for r in sample
    ]
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(stage, False)) as ex:
        again = list(ex.map(_run_plain, js, chunksize=1))
    diffs = []
    for a, b in zip(sample, again, strict=True):
        bad = sorted(
            k
            for k in set(a) | set(b)
            if k not in TIMING and k not in OBS_FIELDS and a.get(k) != b.get(k)
        )
        if bad:
            diffs.append({"key": list(key(a)), "fields": bad})
    res = {
        "replayed": len(sample),
        "stride": REPLAY_STRIDE,
        "differences": diffs,
        "identical": not diffs,
    }
    (out / f"replay_{stage}.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "run", "replay"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=tuple(STAGES), required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    if args.cmd == "check":
        res = check(args.out, args.stage)
        print(json.dumps({"ok": res["ok"], "episodes": res["episodes"]}))
        return 0 if res["ok"] else 1
    if args.cmd == "run":
        return run(args.out, args.stage, args.workers)
    res = replay(args.out, args.stage, args.workers)
    print(json.dumps({k: res[k] for k in ("replayed", "identical")}), len(res["differences"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
