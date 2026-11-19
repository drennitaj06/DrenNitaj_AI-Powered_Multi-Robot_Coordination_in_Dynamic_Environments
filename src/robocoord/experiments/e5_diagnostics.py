"""Why tasks stay undelivered in the preliminary E5 slice (Project Issue #4; `05` §B.6 E5).

Every incomplete episode of an E5 record store is replayed deterministically from its spec
(same builder as the executor, :func:`~robocoord.experiments.allocation_runs.system_episode_setup`),
the replay is checked against the recorded delivery count, and every undelivered task is
classified by the first matching rule below, evaluated on the final state:

1. ``unassigned_unavailable`` — released, nobody holds it, and it is not *available* (model §4:
   a map event closed its pickup/drop-off or cut them apart).
2. ``unassigned_available`` — released, available, but nobody holds it.
3. ``horizon`` — the holder moved within the last ``window`` steps: still progressing when the
   400-step cap ended the episode (censored by the horizon).
4. ``goal_idle_robot`` — the holder's goal cell is occupied by an idle robot (no parking
   behaviour exists; the idle robot never moves away).
5. ``goal_obstacle`` — the goal cell is occupied by a dynamic obstacle and the holder waits for
   it (robot–obstacle goal deadlock: obstacles never enter robot cells, so an obstacle whose
   next move is the robot's cell stays on the goal; the occupied-goal policy of model §5.3 is to
   wait). The obstacle's motion model is recorded.
6. ``goal_unreachable`` — the goal is statically unreachable from the holder.
7. ``no_plan`` — the holder has no valid reference plan.
8. ``route_idle_robot`` / ``route_busy_robot`` — the holder's next planned cell is occupied by an
   idle / a busy robot.
9. ``route_contested_cell`` — the holder's next planned cell is free, but it is also another
   robot's next planned cell: a robot–robot vertex contest on a free cell. A plain plan follower
   proposes it every step and the arbiter freezes all claimants (T4 needs an *occupied* cell,
   model §5.4); the baseline's right-of-way lets exactly one claimant move.
10. ``other`` — none of the above.

The deadlock resolver (`04` §5.5) only handles robot–robot wait-for *cycles*; the causes 4, 5,
8 (idle) and 9 are not cycles and are not detected by it. Endpoint recovery (`04` §5.6, when the
experiment enables it) addresses causes 4 and 5; for those items the holder's recovery state
(``recovery``: persistent blockage, retreat attempts, exhausted, no retreat cell) and, for 4,
the idle occupant's parking state (``occupant_parking``) are recorded, so an undelivered task is
never left without its recovery outcome.

Run: ``python -m robocoord.experiments.e5_diagnostics <experiment dir> <config.toml>``; writes
``undelivered.json`` and ``undelivered.md`` into the experiment directory (new files; the run
records are only read).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robocoord.control.base import Controller
from robocoord.experiments.allocation_runs import system_episode_setup
from robocoord.experiments.records import RecordStore
from robocoord.experiments.spec import ExperimentSpec, load_spec
from robocoord.planning.coordinator import AllocationCoordinator
from robocoord.sim.engine import Engine
from robocoord.world.entities import RobotMode, TaskStatus

CAUSES: tuple[str, ...] = (
    "unassigned_unavailable",
    "unassigned_available",
    "horizon",
    "goal_idle_robot",
    "goal_obstacle",
    "goal_unreachable",
    "no_plan",
    "route_idle_robot",
    "route_busy_robot",
    "route_contested_cell",
    "other",
)


def replay(
    spec: ExperimentSpec, params: dict[str, Any]
) -> tuple[Engine, AllocationCoordinator, dict[int, int]]:
    """Run one episode to its end; returns the engine, coordinator and each robot's last move t."""
    engine, coord, ctrl = system_episode_setup(spec, params)
    controller: Controller = ctrl
    controller.reset(len(engine.robots))
    last_move: dict[int, int] = {}
    while not engine.done:
        before = [r.position for r in engine.robots]
        engine.step(controller.act)
        for r, b in zip(engine.robots, before, strict=True):
            if r.position != b:
                last_move[r.id] = engine.t
    return engine, coord, last_move


def classify(
    engine: Engine, coord: AllocationCoordinator, last_move: dict[int, int], window: int = 5
) -> list[dict[str, Any]]:
    """Cause of every released, undelivered task in the final state (rules in the module doc)."""
    robots_at = {r.position: r for r in engine.robots}
    obstacles_at = {o.position: o for o in engine.obstacles}
    assert coord.monitor is not None
    out: list[dict[str, Any]] = []
    for task in engine.book.tasks.values():
        if task.status in (TaskStatus.UNRELEASED, TaskStatus.DELIVERED):
            continue
        holder = next((r for r in engine.robots if r.task == task.id), None)
        item: dict[str, Any] = {
            "task": task.id,
            "status": task.status.value,
            "release": task.release,
        }
        if holder is None:
            available = engine.task_available(task)
            item["cause"] = "unassigned_available" if available else "unassigned_unavailable"
            out.append(item)
            continue
        goal = task.pickup if holder.mode is RobotMode.TO_PICKUP else task.dropoff
        item.update({"robot": holder.id, "mode": holder.mode.value, "goal": list(goal)})
        on_goal = robots_at.get(goal)
        nxt = coord.monitor.next_cell(holder.id, holder.position)
        ahead = robots_at.get(nxt) if nxt is not None else None
        if engine.t - last_move.get(holder.id, 0) <= window:
            cause = "horizon"
        elif on_goal is not None and on_goal.id != holder.id and on_goal.task is None:
            cause = "goal_idle_robot"
        elif goal in obstacles_at:
            cause = "goal_obstacle"
            item["obstacle_model"] = obstacles_at[goal].model.name
        elif not engine.oracle.reachable(holder.position, goal):
            cause = "goal_unreachable"
        elif coord.monitor.states[holder.id].path is None:
            cause = "no_plan"
        elif ahead is not None and ahead.task is None:
            cause = "route_idle_robot"
        elif ahead is not None:
            cause = "route_busy_robot"
        elif nxt is not None and any(
            r.id != holder.id and coord.monitor.next_cell(r.id, r.position) == nxt
            for r in engine.robots
        ):
            cause = "route_contested_cell"
        else:
            cause = "other"
        item["cause"] = cause
        item["stalled_steps"] = engine.t - last_move.get(holder.id, 0)
        if coord.endpoint is not None and cause in ("goal_idle_robot", "goal_obstacle"):
            item["recovery"] = coord.endpoint.robot_report(holder.id)
            if cause == "goal_idle_robot" and on_goal is not None:
                item["occupant_parking"] = coord.endpoint.robot_report(on_goal.id)
        out.append(item)
    return out


#: Recovery outcomes of undelivered goal-occupancy items (``to_markdown``; recovery runs only).
RECOVERY_OUTCOMES: tuple[str, ...] = (
    "attempts_exhausted",
    "no_retreat_cell",
    "retreating",
    "blockage_persistent",
    "occupant_no_parking_cell",
    "occupant_parking_blocked_dynamic",
    "occupant_parking",
)


def _recovery_outcomes(item: dict[str, Any]) -> list[str]:
    rec = item.get("recovery") or {}
    out = [k for k in RECOVERY_OUTCOMES[:4] if rec.get(k)]
    occ = item.get("occupant_parking") or {}
    out += [
        f"occupant_{k}"
        for k in ("no_parking_cell", "parking_blocked_dynamic", "parking")
        if occ.get(k)
    ]
    return out


def diagnose(spec: ExperimentSpec, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Replay and classify every incomplete episode; check each replay against its record."""
    episodes = []
    mismatches = 0
    for rec in records:
        m = rec["metrics"]
        if rec["outcome"] != "done" or m["tasks_delivered"] == m["tasks_released"]:
            continue
        engine, coord, last_move = replay(spec, dict(rec["params"]))
        delivered = engine.summary().tasks_delivered
        if delivered != m["tasks_delivered"]:
            mismatches += 1
        episodes.append(
            {
                "params": rec["params"],
                "released": m["tasks_released"],
                "delivered": m["tasks_delivered"],
                "replay_delivered": delivered,
                "undelivered": classify(engine, coord, last_move),
            }
        )
    return {"experiment": spec.name, "replay_mismatches": mismatches, "episodes": episodes}


def to_markdown(result: dict[str, Any]) -> str:
    totals: Counter[str] = Counter()
    models: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    with_recovery = 0
    by_cfg: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for ep in result["episodes"]:
        p = ep["params"]
        key = (str(p["planner"]), str(p["allocator"]), str(p["dynamics"]))
        for item in ep["undelivered"]:
            totals[item["cause"]] += 1
            by_cfg[key][item["cause"]] += 1
            if item["cause"] == "goal_obstacle":
                models[item["obstacle_model"]] += 1
            if "recovery" in item:
                with_recovery += 1
                outcomes.update(_recovery_outcomes(item))
    lines = [
        f"## Undelivered tasks — {result['experiment']}",
        "",
        f"Incomplete episodes replayed: {len(result['episodes'])}; replays not reproducing the "
        f"recorded delivery count: {result['replay_mismatches']}.",
        "",
        "| cause | tasks |",
        "|---|---|",
    ]
    lines += [f"| {c} | {totals.get(c, 0)} |" for c in CAUSES]
    lines += [f"| **total** | **{sum(totals.values())}** |", ""]
    if models:
        lines += [
            "goal_obstacle by obstacle model: "
            + ", ".join(f"{k} {v}" for k, v in sorted(models.items())),
            "",
        ]
    if with_recovery:
        lines += [
            f"Endpoint recovery state of the {with_recovery} goal-occupancy items (an item can "
            "have several): " + ", ".join(f"{k} {outcomes.get(k, 0)}" for k in RECOVERY_OUTCOMES),
            "",
        ]
    lines += ["| planner | allocator | dynamics | " + " | ".join(CAUSES) + " |"]
    lines += ["|---|---|---|" + "---|" * len(CAUSES)]
    for key in sorted(by_cfg):
        lines.append(
            "| "
            + " | ".join(key)
            + " | "
            + " | ".join(str(by_cfg[key].get(c, 0)) for c in CAUSES)
            + " |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: python -m robocoord.experiments.e5_diagnostics <experiment dir> <config.toml>"
        )
        return 2
    directory, config = Path(args[0]), Path(args[1])
    spec = load_spec(config)
    result = diagnose(spec, RecordStore(directory).read())
    (directory / "undelivered.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = to_markdown(result)
    (directory / "undelivered.md").write_text(md + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
