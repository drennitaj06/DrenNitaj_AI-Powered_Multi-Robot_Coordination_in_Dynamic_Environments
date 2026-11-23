"""Issue 6A Phase 4 — MADQL diagnostics on stored checkpoints (no training, read-only).

Development data only: Block B seeds 10 020–10 059 (the historical-reference block of Phase 0;
never Block C, D, S or the test seeds). Policies: the three historical MADQL picks (Issue #5
``best.pt``), A\\*, greedy, two conditions:

* ``N`` — the Issue #6 composition (no supervisor, resolver off): the policy's own decisions;
* ``F`` — the forward configuration R + S + D (frozen Stage 3a parameters).

Per non-idle robot-step it records the Q-values, the mask, the policy's action, the executed
action, the planner's suggestion and the occupancy of the suggested cell, whether the robot moved,
whether its observation row is identical to its previous one, and (condition ``N``) the realised
discounted team-mixed shaped return ``G_t`` of the reward the learner optimises (model §7, the
same ``step_rewards`` as ``marl.env``). ``churn`` evaluates a fixed sample of these observations
under every stored step checkpoint from 600 k to 1 M of each seed (read-only).

Commands::

    python tools/phase4/diagnose.py run    --out results/dev6a/phase4/diagnostics --workers 10
    python tools/phase4/diagnose.py churn  --out results/dev6a/phase4/diagnostics
    python tools/phase4/diagnose.py reference --out results/dev6a/phase4/diagnostics --workers 10
    python tools/phase4/diagnose.py report --out results/dev6a/phase4/diagnostics
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SEEDS = range(10_020, 10_060)
LEVELS = ("static", "obstacles", "full")
RUNS = ("madql_seed1", "madql_seed2", "madql_seed3")
CONDITIONS = {"N": ("off", None), "F": ("R+S", True)}
STAY = 0
#: Row layout of the per-robot-step array.
COLS = (
    "t", "robot", "q0", "q1", "q2", "q3", "q4", "m1", "m2", "m3", "m4", "chosen", "executed",
    "suggested", "sug_occ", "moved", "obs_same", "reward", "G", "tail", "T", "terminated",
)  # fmt: skip
C = {c: i for i, c in enumerate(COLS)}
OCC = {None: 0, "free": 1, "idle": 2, "active": 3}
SAMPLE_EVERY = 97  # observation sample for the churn analysis (deterministic stride)
CHURN_STEPS = tuple(range(600_000, 1_000_001, 50_000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs() -> list[dict[str, Any]]:
    return [
        {"run": r, "level": lv, "seed": s, "cond": c}
        for s in SEEDS
        for r in RUNS
        for lv in LEVELS
        for c in CONDITIONS
    ]


def episode(j: dict[str, Any]) -> dict[str, Any]:
    import torch

    from robocoord.control.supervisor import ExecutionSupervisor, frozen_params
    from robocoord.marl.config import RewardParams
    from robocoord.marl.evaluation import EvalTask, _phi, _system
    from robocoord.marl.reward import step_rewards
    from robocoord.marl.scenarios import LEVELS as DISTS
    from robocoord.sim.engine import Engine
    from robocoord.world.entities import RobotMode

    mode, deadlock = CONDITIONS[j["cond"]]
    task = EvalTask(
        j.get("controller", "madql"),
        j["level"],
        int(j["seed"]),
        block="B",
        checkpoint=j.get("checkpoint", f"results/training/{j['run']}/best.pt"),
        label=j["run"].replace("_", "-"),
        supervisor=mode,
        deadlock=deadlock,
    )
    coord, ctrl, extras = _system(task)
    sup = ExecutionSupervisor(mode, frozen_params("astar")) if mode != "off" else None
    eng = Engine(
        DISTS[task.level].sample(task.seed),
        coord,
        supervisor=sup,
        view_extras=extras,
        path_limit=6,
    )
    ctrl.reset(len(eng.robots))
    rec: list[tuple[Any, Any, Any]] = []
    original = ctrl.choose

    def recording(obs: Any, masks: Any) -> Any:
        acts = original(obs, masks)
        with torch.no_grad():
            q = ctrl.net(torch.from_numpy(np.ascontiguousarray(obs))).numpy()
        rec.append((q.astype(np.float32), np.asarray(obs, np.float32).copy(), acts.copy()))
        return acts

    ctrl.choose = recording
    reward = RewardParams()
    n = len(eng.robots)
    rows: list[list[float]] = []
    samples: list[npt.NDArray[np.float32]] = []
    rewards: list[npt.NDArray[np.float64]] = []
    prev_hash: list[int | None] = [None] * n
    pending: tuple[Any, Any, Any] | None = None
    phi_prev: Any = None
    k = 0
    while not eng.done:
        view = eng.prepare_step()
        phi = _phi(eng)
        if pending is not None:
            nonidle, outcome, prios = pending
            rewards.append(step_rewards(reward, nonidle, outcome, prios, phi_prev, phi).total)
        idle = [r.mode is RobotMode.IDLE for r in eng.robots]
        pos = [r.position for r in eng.robots]
        cell_of = {p: i for i, p in enumerate(pos)}
        acts = ctrl.act(view)
        q, obs, chosen = rec[-1]
        masks = view.masks
        outcome = eng.apply_actions(acts)
        executed = outcome.supervised_actions
        new = [r.position for r in eng.robots]
        for i in range(n):
            h = zlib.crc32(obs[i].tobytes())
            same = prev_hash[i] == h
            prev_hash[i] = h
            if idle[i]:
                continue
            path = view.reference_paths[i] if view.reference_paths else None
            c = path[1] if (path is not None and len(path) >= 2) else None
            occ = (
                None
                if c is None
                else ("idle" if idle[cell_of[c]] else "active")
                if c in cell_of and cell_of[c] != i
                else "free"
            )
            sug = int(view.suggested[i]) if view.suggested is not None else -1
            ex = int(executed[i]) if executed is not None else int(chosen[i])
            mask_bits = [float(masks[i][a]) for a in range(1, 5)]
            head = [k, i, *q[i].tolist(), *mask_bits, int(chosen[i]), ex, sug, OCC[occ]]
            rows.append([*head, float(pos[i] != new[i]), float(same), *[math.nan] * 5])
            if len(rows) % SAMPLE_EVERY == 0:
                samples.append(obs[i].copy())
        nonidle = [not x for x in idle]
        prios = {tid: eng.book.tasks[tid].priority for _, tid in outcome.delivered}
        pending = (nonidle, outcome, prios)
        phi_prev = phi
        k += 1
    phi_end = np.zeros(n) if eng.terminated else _phi(eng)
    assert pending is not None
    nonidle, outcome, prios = pending
    rewards.append(step_rewards(reward, nonidle, outcome, prios, phi_prev, phi_end).total)
    big_t = k
    r_arr = np.asarray(rewards)  # (T, n)
    g = np.zeros_like(r_arr)
    acc = np.zeros(n)
    for t in range(big_t - 1, -1, -1):
        acc = r_arr[t] + reward.gamma * acc
        g[t] = acc
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, len(COLS))
    if arr.size:
        ti, ri = arr[:, C["t"]].astype(int), arr[:, C["robot"]].astype(int)
        arr[:, C["reward"]] = r_arr[ti, ri]
        arr[:, C["G"]] = g[ti, ri]
        arr[:, C["tail"]] = reward.gamma ** (big_t - ti)
        arr[:, C["T"]] = big_t
        arr[:, C["terminated"]] = float(eng.terminated)
    m = eng.summary().to_dict()
    return {
        **j,
        "steps": big_t,
        "terminated": bool(eng.terminated),
        "released": m["tasks_released"],
        "delivered": m["tasks_delivered"],
        "rows": arr.astype(np.float32),
        "samples": np.asarray(samples, dtype=np.float32).reshape(-1, obs.shape[1]),
    }


# --------------------------------------------------------------------------- commands


def run(out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ck = {r: sha256(Path(f"results/training/{r}/best.pt")) for r in RUNS}
    js = jobs()
    rows: dict[str, list[Any]] = {c: [] for c in CONDITIONS}
    meta: list[dict[str, Any]] = []
    samples: list[Any] = []
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(episode, js, chunksize=1)):
            r = res.pop("rows")
            s = res.pop("samples")
            tag = np.full((len(r), 1), i, dtype=np.float32)
            rows[res["cond"]].append(np.hstack([tag, r]))
            if res["cond"] == "N":
                samples.append(s)
            meta.append(res)
            if (i + 1) % 60 == 0:
                print(f"{i + 1}/{len(js)}", flush=True)
    arrays: dict[str, Any] = {f"rows_{c}": np.vstack(v) for c, v in rows.items()}
    arrays["samples"] = np.vstack(samples)
    np.savez_compressed(out / "states.npz", **arrays)
    after = {r: sha256(Path(f"results/training/{r}/best.pt")) for r in RUNS}
    (out / "episodes.json").write_text(
        json.dumps(
            {
                "columns": ["episode", *COLS],
                "episodes": meta,
                "checkpoints": ck,
                "checkpoints_unchanged": ck == after,
            },
            indent=1,
        ),
        "utf-8",
    )


#: PPO references (frozen SCB-C v5 picks, read-only): only their realised returns are used, as an
#: empirical lower bound of the achievable value on the same instances.
REFERENCE = {
    "mappo_seed1": "results/training/mappo_seed1/checkpoints/step_00300000.pt",
    "ippo_seed1": "results/training/ippo_seed1/checkpoints/step_00500000.pt",
}


def reference(out: Path, workers: int) -> None:
    js = [
        {"run": r, "level": lv, "seed": s, "cond": "F", "controller": "ppo", "checkpoint": c}
        for s in SEEDS
        for r, c in REFERENCE.items()
        for lv in LEVELS
    ]
    before = {r: sha256(Path(c)) for r, c in REFERENCE.items()}
    res: dict[str, Any] = {}
    with ProcessPoolExecutor(workers) as ex:
        eps = list(ex.map(episode, js, chunksize=1))
    for r in REFERENCE:
        rows = np.vstack([e["rows"] for e in eps if e["run"] == r]).astype(np.float64)
        ok = (rows[:, C["tail"]] <= 0.134) | (rows[:, C["terminated"]] > 0.5)
        sel = ok & (rows[:, C["chosen"]] == rows[:, C["executed"]])
        g = rows[:, C["G"]]
        early = sel & (rows[:, C["t"]] < 100)
        res[r] = {
            "episodes": sum(e["run"] == r for e in eps),
            "completion": sum(e["delivered"] for e in eps if e["run"] == r)
            / sum(e["released"] for e in eps if e["run"] == r),
            "return": _dist(g[sel]),
            "return_t_lt_100": _dist(g[early]),
        }
    res["checkpoints_unchanged"] = before == {r: sha256(Path(c)) for r, c in REFERENCE.items()}
    (out / "reference.json").write_text(json.dumps(res, indent=1), "utf-8")


def churn(out: Path) -> None:
    import torch

    from robocoord.marl.checkpoint import load_inference
    from robocoord.marl.networks import masked_argmax

    z = np.load(out / "states.npz")
    obs = torch.from_numpy(z["samples"])
    # masks are not needed for the greedy move set: compare the unmasked STAY-vs-move ordering
    # and the masked arg-max with every move allowed except those blocked in the sample's mask
    res: dict[str, Any] = {"n_states": int(obs.shape[0]), "steps": list(CHURN_STEPS), "runs": {}}
    greedy: dict[str, dict[int, npt.NDArray[np.int64]]] = {}
    gaps: dict[str, dict[int, npt.NDArray[np.float32]]] = {}
    hashes: dict[str, str] = {}
    full = torch.ones((obs.shape[0], 5), dtype=torch.bool)
    for run in RUNS:
        greedy[run], gaps[run] = {}, {}
        for st in CHURN_STEPS:
            p = Path(f"results/training/{run}/checkpoints/step_{st:08d}.pt")
            hashes[str(p)] = sha256(p)
            net, _, _ = load_inference(p)
            with torch.no_grad():
                q = net.eval()(obs)
            greedy[run][st] = masked_argmax(q, full).numpy()
            qn = q.numpy()
            gaps[run][st] = qn[:, 0] - qn[:, 1:].max(axis=1)
        seq = [greedy[run][s] for s in CHURN_STEPS]
        res["runs"][run] = {
            "greedy_changed_between_consecutive": [
                float((a != b).mean()) for a, b in itertools.pairwise(seq)
            ],
            "stay_vs_move_flipped_between_consecutive": [
                float(((gaps[run][a] >= 0) != (gaps[run][b] >= 0)).mean())
                for a, b in itertools.pairwise(CHURN_STEPS)
            ],
            "gap_abs_change_median": [
                float(np.median(np.abs(gaps[run][a] - gaps[run][b])))
                for a, b in itertools.pairwise(CHURN_STEPS)
            ],
        }
    res["across_seeds_at_1M_greedy_agree_all_three"] = float(
        np.mean(
            (greedy[RUNS[0]][1_000_000] == greedy[RUNS[1]][1_000_000])
            & (greedy[RUNS[1]][1_000_000] == greedy[RUNS[2]][1_000_000])
        )
    )
    res["checkpoint_sha256"] = hashes
    (out / "churn.json").write_text(json.dumps(res, indent=1), "utf-8")


def _dist(x: npt.NDArray[Any]) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
    }


def stay_spells(
    ep: npt.NDArray[np.int64],
    rob: npt.NDArray[np.int64],
    t: npt.NDArray[np.int64],
    stay: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """For every STAY decision: the length of its spell (maximal run of STAY decisions at
    consecutive steps of one robot) and whether the spell ends with a move decision at the next
    step (resolved). Rows that are not STAY get length 0."""
    order = np.lexsort((t, rob, ep))
    length = np.zeros(len(t), dtype=np.int64)
    resolved = np.zeros(len(t), dtype=bool)
    i = 0
    while i < len(order):
        a = order[i]
        if not stay[a]:
            i += 1
            continue
        j = i
        while (
            j + 1 < len(order)
            and stay[order[j + 1]]
            and ep[order[j + 1]] == ep[a]
            and rob[order[j + 1]] == rob[a]
            and t[order[j + 1]] == t[order[j]] + 1
        ):
            j += 1
        last = order[j]
        nxt = order[j + 1] if j + 1 < len(order) else None
        ends = (
            nxt is not None
            and ep[nxt] == ep[a]
            and rob[nxt] == rob[a]
            and t[nxt] == t[last] + 1
            and not stay[nxt]
        )
        members = order[i : j + 1]
        length[members] = j - i + 1
        resolved[members] = ends
        i = j + 1
    return length, resolved


def report(out: Path) -> dict[str, Any]:
    z = np.load(out / "states.npz")
    res: dict[str, Any] = {}
    for cond in CONDITIONS:
        a = z[f"rows_{cond}"].astype(np.float64)
        cols = {c: a[:, i + 1] for i, c in enumerate(COLS)}
        q = a[:, 3:8]
        masks = np.hstack([np.ones((len(a), 1)), a[:, 8:12]]).astype(bool)
        moves = np.where(masks[:, 1:], q[:, 1:], -np.inf)
        best_move = moves.max(axis=1)
        gap = q[:, 0] - best_move
        chosen = cols["chosen"].astype(int)
        sug = cols["suggested"].astype(int)
        occ = cols["sug_occ"].astype(int)
        stay = chosen == STAY
        sug_move = sug > 0
        sug_q = np.where(sug_move, q[np.arange(len(a)), np.clip(sug, 0, 4)], np.nan)
        gap_sug = np.where(sug_move, q[:, 0] - sug_q, np.nan)
        same = cols["obs_same"] > 0.5
        # spell of consecutive STAY decisions per (episode, robot); resolved = robot later moves
        ep = a[:, 0].astype(int)
        rob = cols["robot"].astype(int)
        t = cols["t"].astype(int)
        spell_len, ends_moving = stay_spells(ep, rob, t, stay)
        nonidle_steps = len(a)
        free_wait = stay & sug_move & (occ == 1)
        blocked_wait = stay & sug_move & (occ >= 2)
        out_c: dict[str, Any] = {
            "nonidle_robot_steps": nonidle_steps,
            "share_stay": float(stay.mean()),
            "share_follow_suggestion": float((chosen == sug).mean()),
            "stay_with_free_suggested_cell": float(free_wait.mean()),
            "stay_with_occupied_suggested_cell": float(blocked_wait.mean()),
            "share_obs_identical_to_previous_when_staying": float(same[stay].mean()),
            "gap_all": _dist(gap),
            "gap_when_stay": _dist(gap[stay]),
            "gap_when_move": _dist(gap[~stay]),
            "gap_vs_suggested": _dist(gap_sug[sug_move]),
        }
        # aliasing: STAY with a free suggested cell, split by how the spell ends
        long = spell_len >= 20
        resolved = free_wait & ends_moving & ~long
        persistent = free_wait & long
        out_c["aliasing"] = {
            "free_cell_stay_short_spell_resolved": _dist(gap[resolved]),
            "free_cell_stay_long_spell_ge20": _dist(gap[persistent]),
            "occupied_cell_stay": _dist(gap[blocked_wait]),
            "share_obs_identical_in_long_spells": float(same[persistent].mean())
            if persistent.any()
            else None,
        }
        if cond == "N":
            ok = (cols["tail"] <= 0.134) | (cols["terminated"] > 0.5)
            qa = q[np.arange(len(a)), chosen]
            err = qa - cols["G"]
            out_c["calibration"] = {
                "rows": int(ok.sum()),
                "q_minus_return": _dist(err[ok]),
                "q_minus_return_when_stay": _dist(err[ok & stay]),
                "q_minus_return_long_free_stay": _dist(err[ok & persistent]),
                "abs_error_median": float(np.median(np.abs(err[ok]))),
                "return": _dist(cols["G"][ok]),
                "q_chosen": _dist(qa[ok]),
                "reward_when_stay": _dist(cols["reward"][stay]),
            }
            # upper bound of the true advantage of STAY when STAY returns the same state:
            # A*(s, STAY) <= r_stay - (1 - γ) V*(s) and V*(s) >= G_t (realised under the policy)
            ub = cols["reward"] - 0.01 * cols["G"]
            out_c["stay_advantage_upper_bound_self_loop"] = _dist(ub[ok & stay & same])
        # calibration on the steps where the executed action is the policy's (both conditions)
        own = (chosen == cols["executed"].astype(int)) & (
            (cols["tail"] <= 0.134) | (cols["terminated"] > 0.5)
        )
        qa_own = q[np.arange(len(a)), chosen]
        out_c["calibration_own_action"] = {
            "q": _dist(qa_own[own]),
            "return": _dist(cols["G"][own]),
            "q_minus_return": _dist((qa_own - cols["G"])[own]),
            "q_minus_return_t_lt_100": _dist((qa_own - cols["G"])[own & (t < 100)]),
            "return_t_lt_100": _dist(cols["G"][own & (t < 100)]),
        }
        res[cond] = out_c
    for extra in ("churn", "reference"):
        f = out / f"{extra}.json"
        if f.exists():
            res[extra] = json.loads(f.read_text("utf-8"))
    logs = {}
    for r in RUNS:
        prog = [
            json.loads(line)
            for line in Path(f"results/training/{r}/train_log.jsonl")
            .read_text("utf-8")
            .splitlines()
            if '"progress"' in line
        ]
        val = [
            json.loads(line)
            for line in Path(f"results/training/{r}/validation.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        comp = [v["key"][0] for v in val if v["planner"] == "astar"]
        late = [p["loss_mean"] for p in prog if p["env_steps"] > 500_000]
        logs[r] = {
            "late_loss_mean": float(np.mean(late)),
            "implied_td_rms_if_quadratic": float(math.sqrt(2 * np.mean(late))),
            "validation_completion": comp,
            "max_abs_change_between_validations_after_500k": float(
                np.max(np.abs(np.diff(comp[9:])))
            ),
        }
    res["training_logs"] = logs
    (out / "diagnostics.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "churn", "reference", "report"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args.out, args.workers)
    elif args.cmd == "churn":
        churn(args.out)
    elif args.cmd == "reference":
        reference(args.out, args.workers)
    else:
        print(json.dumps(report(args.out), indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
