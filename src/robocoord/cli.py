"""Command-line interface: ``python -m robocoord <command>`` (architecture §5.8).

Issue #2 commands:

* ``simulate`` — run a scenario headless or as a terminal animation, print metrics, write a trace.
* ``validate`` — load and validate a scenario file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from robocoord import __version__
from robocoord.control.base import Controller, RandomController, ScriptedController, StayController
from robocoord.core.rng import make_streams
from robocoord.scenario import ScenarioError, load_scenario
from robocoord.scenario.spec import ScenarioSpec
from robocoord.sim.coordinator import Coordinator, NullCoordinator, ScriptedCoordinator
from robocoord.sim.engine import Engine, StepOutcome
from robocoord.sim.metrics import EpisodeMetrics
from robocoord.sim.text_render import enable_windows_ansi, render_frame
from robocoord.sim.trace import TraceWriter

CONTROLLERS = ("scripted", "random", "stay")


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
        help="scripted (default if the scenario has a [script]), random (masked), stay",
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
    return parser


def make_controller(name: str, spec: ScenarioSpec, seed: int) -> tuple[Controller, Coordinator]:
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


def cmd_simulate(args: argparse.Namespace) -> int:
    spec = load_scenario(args.scenario, seed=args.seed)
    controller_name = args.controller or ("scripted" if spec.script is not None else "random")
    controller, coordinator = make_controller(controller_name, spec, spec.seed)
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
        if isinstance(coordinator, NullCoordinator) and metrics.tasks_released:
            print(
                "note: no task coordinator is attached to this run, so tasks are never assigned "
                "or picked up (delivered 0 is expected). The auction allocator arrives in "
                "Issue #4; use a scenario with a [script] section to see deliveries now."
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "simulate":
            return cmd_simulate(args)
        return cmd_validate(args)
    except ScenarioError as exc:
        print(f"scenario error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
