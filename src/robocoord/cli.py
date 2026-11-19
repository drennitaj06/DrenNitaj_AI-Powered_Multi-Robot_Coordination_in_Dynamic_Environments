"""Command-line interface: ``python -m robocoord <command>`` (architecture §5.8).

Commands:

* ``simulate`` — run a scenario headless or as a terminal animation, print metrics, write a trace.
  ``--planner astar|rrtstar`` (Issue #3) attaches the path planner with its monitor.
  Without ``--allocator`` tasks come from the scenario's ``[script]`` fixture (Issue #3);
  ``--allocator auction|sequential`` (Issue #4) allocates them with the event-triggered auction
  or the sequential baseline. ``--controller reference|baseline`` (Issue #4) selects the
  reference follower or the right-of-way baseline (default with ``--allocator``: ``baseline``);
  ``--shield off|predictive`` the safety shield (default ``off``); ``--deadlock auto|on|off`` the
  deadlock-detour resolver (default ``auto`` = on exactly for ``baseline`` with an allocator);
  ``--endpoint on|off`` endpoint recovery (idle-robot parking and retreat from a persistently held
  task goal; default on with an allocator, for every controller; ``off`` restores the earlier
  behaviour). The 2 × 2 × 2 × 2 planner × allocator × controller × shield matrix is selectable
  here.
* ``validate`` — load and validate a scenario file.
* ``experiment`` — run an experiment specification (Issue #3: E1, E2; Issue #4: E3, E5 prelim.).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from robocoord import __version__
from robocoord.control.base import (
    Controller,
    RandomController,
    ScriptedController,
    StayController,
    SuggestionFollower,
)
from robocoord.core.errors import ConfigError
from robocoord.core.rng import make_streams
from robocoord.scenario import ScenarioError, load_scenario
from robocoord.scenario.spec import ScenarioSpec
from robocoord.sim.coordinator import Coordinator, NullCoordinator, ScriptedCoordinator
from robocoord.sim.engine import Engine, StepOutcome
from robocoord.sim.metrics import EpisodeMetrics
from robocoord.sim.text_render import enable_windows_ansi, render_frame
from robocoord.sim.trace import TraceWriter

CONTROLLERS = ("scripted", "random", "stay", "follow", "reference", "baseline", "madql")
ALLOCATORS = ("auction", "sequential")
PLAN_CONTROLLERS = ("follow", "reference", "baseline", "madql")  # need --planner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robocoord", description="Multi-robot coordination simulator"
    )
    parser.add_argument("--version", action="version", version=f"robocoord {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a scenario")
    sim.add_argument("--scenario", required=True, type=Path, help="scenario TOML file")
    sim.add_argument("--seed", type=int, default=None, help="override the scenario seed")
    sim.add_argument(
        "--controller",
        choices=CONTROLLERS,
        default=None,
        help="scripted (default if the scenario has a [script]), random (masked), stay, "
        "follow (plays the planner's suggested actions; default with --planner), "
        "reference (reference follower, Issue #4), baseline (right-of-way baseline, Issue #4; "
        "default with --allocator), madql (learned controller, Issue #5; needs --checkpoint)",
    )
    sim.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="inference checkpoint (.pt with its .json sidecar) for --controller madql",
    )
    sim.add_argument(
        "--planner",
        choices=("astar", "rrtstar"),
        default=None,
        help="attach a path planner (A* or RRT*) with the path monitor",
    )
    sim.add_argument(
        "--allocator",
        choices=ALLOCATORS,
        default=None,
        help="allocate tasks with the auction or the sequential baseline (needs --planner); "
        "without it tasks come from the scenario's [script] fixture",
    )
    sim.add_argument(
        "--shield",
        choices=("off", "predictive"),
        default="off",
        help="safety shield between controller and arbiter (default off)",
    )
    sim.add_argument(
        "--deadlock",
        choices=("auto", "on", "off"),
        default="auto",
        help="deadlock-detour resolver (needs --allocator); auto = on exactly for the baseline "
        "controller (default auto)",
    )
    sim.add_argument(
        "--endpoint",
        choices=("on", "off"),
        default=None,
        help="endpoint recovery: idle-robot parking and retreat from a persistently held task "
        "goal (needs --allocator; default on with an allocator, for every controller)",
    )
    sim.add_argument(
        "--rrt-iterations", type=int, default=None, help="RRT* iteration budget per call"
    )
    sim.add_argument("--max-steps", type=int, default=None, help="override scenario.max_steps")
    sim.add_argument("--delay", type=float, default=0.25, help="seconds between frames (animation)")
    sim.add_argument("--every", type=int, default=1, help="draw every N-th step")
    sim.add_argument(
        "--no-animate", action="store_true", help="print only the final state and metrics"
    )
    sim.add_argument("--no-clear", action="store_true", help="print frames one after another")
    sim.add_argument("--color", action="store_true", help="ANSI colours")
    sim.add_argument(
        "--trace", type=Path, default=None, help="write a JSON-Lines trace to this path"
    )
    sim.add_argument("--json", action="store_true", help="print metrics as JSON")
    sim.add_argument(
        "--run-to-max", action="store_true", help="do not stop when all tasks are delivered"
    )

    val = sub.add_parser("validate", help="validate a scenario file")
    val.add_argument("--scenario", required=True, type=Path)

    exp = sub.add_parser("experiment", help="run an experiment specification (TOML)")
    exp.add_argument("--config", required=True, type=Path, help="experiment TOML file")
    exp.add_argument("--out", type=Path, default=None, help="output directory override")
    exp.add_argument("--workers", type=int, default=None, help="parallel worker processes")
    exp.add_argument("--limit", type=int, default=None, help="run at most N new runs (smoke)")

    tr = sub.add_parser("train", help="train a MADQL policy (Issue #5; needs the marl extra)")
    tr.add_argument("--config", required=True, type=Path, help="training TOML file")
    tr.add_argument("--out", required=True, type=Path, help="new output directory")
    tr.add_argument("--seed", type=int, default=None, help="override run.run_seed")
    tr.add_argument("--total-steps", type=int, default=None, help="override run.total_steps")
    return parser


def resolve_deadlock(mode: str, controller: str, allocator: str | None) -> bool:
    """The documented default: ``auto`` enables the resolver exactly for the baseline controller
    with an allocator; ``on`` requires an allocator."""
    if mode == "on" and allocator is None:
        raise ConfigError("deadlock", "--deadlock on needs --allocator")
    if mode == "auto":
        return controller == "baseline" and allocator is not None
    return mode == "on"


def resolve_endpoint(mode: str | None, allocator: str | None) -> bool:
    """The documented default: endpoint recovery is on with an allocator (every controller);
    an explicit ``on`` requires an allocator."""
    if mode == "on" and allocator is None:
        raise ConfigError("endpoint", "--endpoint on needs --allocator")
    if mode is None:
        return allocator is not None
    return mode == "on"


def make_controller(
    name: str,
    spec: ScenarioSpec,
    seed: int,
    planner: str | None = None,
    rrt_iterations: int | None = None,
    allocator: str | None = None,
    deadlock: bool = False,
    endpoint: bool = False,
    checkpoint: Path | None = None,
) -> tuple[Controller, Coordinator]:
    if allocator is not None:
        from robocoord.allocation.auction import AuctionAllocator
        from robocoord.allocation.sequential import SequentialAllocator
        from robocoord.planning import make_planner
        from robocoord.planning.coordinator import AllocationCoordinator
        from robocoord.planning.deadlock import DetourParams
        from robocoord.planning.endpoint import EndpointParams

        if planner is None:
            raise ConfigError("allocator", "--allocator needs --planner")
        if name not in PLAN_CONTROLLERS:
            raise ConfigError(
                "controller", f"with --allocator choose one of {PLAN_CONTROLLERS}, got {name!r}"
            )
        options = (
            {}
            if rrt_iterations is None or planner != "rrtstar"
            else {"max_iterations": rrt_iterations}
        )
        alloc = AuctionAllocator() if allocator == "auction" else SequentialAllocator()
        acoord = AllocationCoordinator(
            make_planner(planner, **options),
            alloc,
            deadlock=DetourParams() if deadlock else None,
            endpoint=EndpointParams() if endpoint else None,
        )
        return _plan_controller(name, checkpoint), acoord
    if planner is not None:
        from robocoord.planning import make_planner
        from robocoord.planning.coordinator import PlanningCoordinator

        if spec.script is None:
            raise ScenarioError(
                "script", "--planner needs fixture task assignments ([script].assignments)"
            )
        options = (
            {}
            if rrt_iterations is None or planner != "rrtstar"
            else {"max_iterations": rrt_iterations}
        )
        coord = PlanningCoordinator(make_planner(planner, **options), spec.script.assignments)
        if name in PLAN_CONTROLLERS:
            return _plan_controller(name, checkpoint), coord
        if name == "random":
            return RandomController(make_streams(seed).generator("policy")), coord
        if name == "stay":
            return StayController(), coord
        return ScriptedController(spec.script.actions), coord
    if name in PLAN_CONTROLLERS:
        raise ScenarioError("controller", f"the {name} controller needs --planner")
    if name == "scripted":
        if spec.script is None:
            raise ScenarioError("script", "the scripted controller needs a [script] section")
        return ScriptedController(spec.script.actions), ScriptedCoordinator(spec.script.assignments)
    coordinator: Coordinator = (
        ScriptedCoordinator(spec.script.assignments)
        if spec.script is not None
        else NullCoordinator()
    )
    if name == "random":
        return RandomController(make_streams(seed).generator("policy")), coordinator
    return StayController(), coordinator


def _plan_controller(name: str, checkpoint: Path | None = None) -> Controller:
    if name == "madql":
        return _learned_controller(checkpoint)
    if name == "reference":
        from robocoord.control.reference import ReferenceFollower

        return ReferenceFollower()
    if name == "baseline":
        from robocoord.control.baseline import BaselineController

        return BaselineController()
    return SuggestionFollower()


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the installation
        raise ConfigError(
            "marl",
            'PyTorch is not installed; install the marl extra: pip install -e ".[marl]" '
            "--extra-index-url https://download.pytorch.org/whl/cpu",
        ) from None


def _learned_controller(checkpoint: Path | None) -> Controller:
    if checkpoint is None:
        raise ConfigError("checkpoint", "--controller madql needs --checkpoint FILE.pt")
    if not checkpoint.exists() or not checkpoint.with_suffix(".json").exists():
        raise ConfigError("checkpoint", f"{checkpoint} or its .json sidecar does not exist")
    _require_torch()
    from robocoord.marl.policy import LearnedController

    return LearnedController.from_checkpoint(checkpoint)


def format_metrics(m: EpisodeMetrics) -> str:
    def f(x: float | None, nd: int = 2) -> str:
        return "n/a" if x is None else f"{x:.{nd}f}"

    makespan = "n/a" if m.makespan is None else str(m.makespan)
    lines = [
        "=== episode metrics ===",
        f"steps {m.steps} | robots {m.robots}",
        f"tasks: delivered {m.tasks_delivered}/{m.tasks_released} released "
        f"(completion {f(m.completion_rate)}) | makespan {makespan}",
        f"flow time: mean {f(m.mean_flow_time)} | priority-weighted {f(m.weighted_flow_time)} | "
        "by priority " + ", ".join(f"p{k}={f(v)}" for k, v in m.flow_time_by_priority.items()),
        f"efficiency: path efficiency {f(m.path_efficiency)} over {m.legs_completed} legs | "
        f"utilisation {f(m.utilisation)} | wait ratio {f(m.wait_ratio)} | "
        f"throughput {f(m.throughput_per_100_steps)} per 100 steps",
        f"safety: proposed conflicts {m.conflicts_total} "
        f"(prevented by arbiter {m.conflicts_prevented}) "
        + str(m.conflicts)
        + f" | blocked moves {m.blocked_moves} | physical collisions {m.physical_collisions}",
        f"rates per 100 robot-steps: conflicts {f(m.conflicts_per_100_robot_steps)} | "
        f"blocked moves {f(m.blocked_moves_per_100_robot_steps)} | "
        f"blocked-goal robot-steps {m.blocked_goal_robot_steps}",
    ]
    if m.conflicts_avoided is None:
        lines.append(
            "collisions avoided vs reference plan: n/a (no planner attached until Issue #3)"
        )
    else:
        lines.append(
            f"vs reference plan: avoided {m.conflicts_avoided} | "
            f"introduced {m.conflicts_introduced} | persisted {m.conflicts_persisted} | "
            f"avoidance rate {f(m.avoidance_rate)}"
        )
    return "\n".join(lines)


def format_planning(p: dict[str, object], adherence: float | None) -> str:
    adh = "n/a" if adherence is None else f"{adherence:.2f}"
    return (
        f"planning ({p['planner']}): {p['calls']} planner calls | "
        f"by trigger {p['calls_by_trigger']}\n"
        f"  status mix {p['status_mix']} | plan adherence {adh} | "
        f"legs completed {p['legs_completed']} | releases {p['releases']}"
    )


def format_extra(extra: dict[str, object]) -> list[str]:
    """Allocation, deadlock, controller and shield statistics (Issue #4), when present."""
    lines = []
    planning = extra.get("planning")
    if isinstance(planning, dict):
        if "allocation" in planning:
            a = planning["allocation"]
            lines.append(
                f"allocation ({a['allocator']}): {a['auctions']} auctions | triggers "
                f"{a['triggers']} | assignments {a['assignments']} | pre-emptions "
                f"{a['preemptions']} | lost {a['lost']} | tasks reassigned {a['tasks_reassigned']}"
            )
        if "deadlock" in planning:
            d = planning["deadlock"]
            detected = f"{d['detected_persistent']}+{d['detected_recurrent']}"
            lines.append(
                f"deadlock resolver: detected {detected} | detours {d['detours_started']} | "
                f"released {d['released']} (timeout {d['released_timeout']}) | aborted "
                f"{d['aborted']} | no passing place {d['no_passing_place']} (dynamic "
                f"{d['no_passing_place_dynamic']}) | unresolved at end {d['unresolved_at_end']}"
            )
        if "endpoint" in planning:
            e = planning["endpoint"]
            lines.append(
                f"endpoint recovery: blockages temporary {e['blockages_temporary']} / persistent "
                f"{e['blockages_persistent']} | retreats {e['retreats_started']} (succeeded "
                f"{e['retreats_succeeded']}, timed out {e['retreats_timed_out']}, aborted "
                f"{e['retreats_aborted']}) | exhausted {e['attempts_exhausted']} | parking "
                f"{e['parking_started']} (completed {e['parking_completed']}) | no parking cell "
                f"{e['no_parking_cell']} | no retreat cell {e['no_retreat_cell']} | unresolved at "
                f"end {e['unresolved_obstacle_at_end'] + e['unresolved_idle_robot_at_end']}"
            )
    ctrl = extra.get("controller")
    if isinstance(ctrl, dict):
        lines.append(f"controller ({ctrl['controller']}): decisions {ctrl['decisions']}")
    shield = extra.get("shield")
    if isinstance(shield, dict):
        lines.append(
            f"shield ({shield['mode']}): interventions {shield['interventions']} | prevented "
            f"{shield['prevented']} | introduced {shield['introduced']}"
        )
    return lines


def cmd_simulate(args: argparse.Namespace) -> int:
    spec = load_scenario(args.scenario, seed=args.seed)
    if args.allocator is not None:
        controller_name = args.controller or "baseline"
    elif args.planner is not None:
        controller_name = args.controller or "follow"
    else:
        controller_name = args.controller or ("scripted" if spec.script is not None else "random")
    deadlock = resolve_deadlock(args.deadlock, controller_name, args.allocator)
    endpoint = resolve_endpoint(args.endpoint, args.allocator)
    controller, coordinator = make_controller(
        controller_name,
        spec,
        spec.seed,
        args.planner,
        args.rrt_iterations,
        args.allocator,
        deadlock,
        endpoint,
        args.checkpoint,
    )
    shield = None
    if args.shield != "off":
        from robocoord.control.shield import Shield

        shield = Shield(args.shield)
    stats = getattr(controller, "episode_stats", None)
    extras: frozenset[str] = frozenset()
    path_limit = 16
    if controller_name == "madql":
        from robocoord.marl.policy import LearnedController

        assert isinstance(controller, LearnedController)
        extras = frozenset({"paths", "goal_distance"})  # the learned observation needs them
        path_limit = controller.builder.params.path_cells
    trace = TraceWriter(args.trace) if args.trace else None
    ansi_ok = enable_windows_ansi()
    color = args.color and ansi_ok
    try:
        engine = Engine(
            spec,
            coordinator,
            trace=trace,
            stop_when_done=not args.run_to_max,
            max_steps=args.max_steps,
            shield=shield,
            controller_stats=stats
            if controller_name in ("reference", "baseline", "madql")
            else None,
            view_extras=extras,
            path_limit=path_limit,
        )
        controller.reset(len(engine.robots))
        animate = not args.no_animate and not args.json
        # Redraw in place only on a real terminal that understands ANSI; when the output is
        # redirected or piped, frames are printed one after another without escape codes.
        clear = "\x1b[2J\x1b[H" if (animate and not args.no_clear and ansi_ok and _is_tty()) else ""
        if animate:
            print(clear + render_frame(engine, color=color), flush=True)
            time.sleep(args.delay)

        def on_step(eng: Engine, out: StepOutcome) -> None:
            if animate and (eng.t % max(args.every, 1) == 0 or eng.done):
                print(clear + render_frame(eng, out, color=color), flush=True)
                if not eng.done:
                    time.sleep(args.delay)

        metrics = engine.run(controller.act, on_step=on_step)
    finally:
        if trace is not None:
            trace.close()
    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    else:
        if not animate:
            print(render_frame(engine, color=color))
        status = "all tasks delivered" if engine.terminated else "step limit reached"
        print(
            f"\nfinished: {status} (controller: {controller_name}, coordinator: {coordinator.name})"
        )
        print(format_metrics(metrics))
        planning = metrics.extra.get("planning")
        if planning:
            print(format_planning(planning, metrics.plan_adherence))
        for line in format_extra(metrics.extra):
            print(line)
        if isinstance(coordinator, NullCoordinator) and metrics.tasks_released:
            print(
                "note: no task coordinator is attached to this run, so tasks are never assigned "
                "or picked up (delivered 0 is expected). Use --planner with --allocator "
                "auction|sequential, or a scenario with a [script] section."
            )
        if args.trace:
            print(f"trace written to {args.trace}")
    return 0


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):  # pragma: no cover - exotic stdout replacements
        return False


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_scenario(args.scenario)
    print(
        f"OK: {spec.name} | map {spec.grid.width}x{spec.grid.height} "
        f"({spec.grid.num_free()} free) | "
        f"robots {len(spec.robots)} | obstacles {len(spec.obstacles)} | tasks {len(spec.tasks)} | "
        f"arrivals {spec.arrivals.kind} | map events {len(spec.map_events)} | "
        f"script {'yes' if spec.script else 'no'}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    _require_torch()
    from robocoord.marl.trainer import load_training_config, train

    cfg = load_training_config(args.config, run_seed=args.seed, total_steps=args.total_steps)
    summary = train(cfg, args.out)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "simulate":
            return cmd_simulate(args)
        if args.command == "experiment":
            from robocoord.experiments.runner import run_experiment_cli

            return run_experiment_cli(args.config, args.out, args.workers, args.limit)
        if args.command == "train":
            return cmd_train(args)
        return cmd_validate(args)
    except ScenarioError as exc:
        print(f"scenario error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
