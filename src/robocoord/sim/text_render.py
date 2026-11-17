"""Plain-text (optionally ANSI-coloured) rendering of the simulation for the terminal demo.

Only ASCII characters are used so the output is safe on any Windows console or redirected file.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from robocoord.sim.conflicts import ConflictKind
from robocoord.world.entities import RobotMode, TaskStatus

if TYPE_CHECKING:
    from robocoord.sim.engine import Engine, StepOutcome

_ROBOT_CHARS = string.digits + string.ascii_lowercase

_RESET = "\x1b[0m"
_COLORS = {
    "wall": "\x1b[90m",
    "robot": "\x1b[1;96m",
    "robot_busy": "\x1b[1;93m",
    "obstacle": "\x1b[1;91m",
    "pickup": "\x1b[33m",
    "dropoff": "\x1b[32m",
}

LEGEND = (
    "legend: 0-9/a-z robot | x moving obstacle | P pickup of open task | "
    "D drop-off of carried task | # wall"
)


def robot_char(i: int) -> str:
    return _ROBOT_CHARS[i] if i < len(_ROBOT_CHARS) else "?"


def render_frame(engine: Engine, outcome: StepOutcome | None = None, *, color: bool = False) -> str:
    """Render the current state (after ``outcome``, if given) as a multi-line string."""
    grid = engine.grid
    w, h = grid.width, grid.height
    cells: dict[tuple[int, int], tuple[str, str]] = {}
    for tk in engine.book.tasks.values():
        if tk.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
            cells[tk.pickup] = ("P", "pickup")
    for tk in engine.book.tasks.values():
        if tk.status is TaskStatus.PICKED:
            cells[tk.dropoff] = ("D", "dropoff")
    for o in engine.obstacles:
        cells[o.position] = ("x", "obstacle")
    for r in engine.robots:
        cells[r.position] = (
            robot_char(r.id),
            "robot" if r.mode is RobotMode.IDLE else "robot_busy",
        )

    def paint(ch: str, kind: str) -> str:
        return f"{_COLORS[kind]}{ch}{_RESET}" if color else ch

    lines: list[str] = []
    title = (
        f"t={engine.t}/{engine.max_steps}  scenario: {engine.scenario.name}  seed: {engine.seed}"
    )
    lines.append(title)
    lines.append("     " + "".join(str(x % 10) for x in range(w)))
    for y in range(h - 1, -1, -1):
        row = []
        for x in range(w):
            if (x, y) in cells:
                row.append(paint(*cells[(x, y)]))
            elif grid.is_blocked((x, y)):
                row.append(paint("#", "wall"))
            else:
                row.append(".")
        lines.append(f"{y:>4} " + "".join(row))
    lines.append(LEGEND)
    lines.append("robots:")
    for r in engine.robots:
        desc = f"  {robot_char(r.id)}  at {r.position!s:<9} {r.mode.value:<11}"
        if r.task is not None:
            tk = engine.book.tasks[r.task]
            goal = tk.pickup if r.mode is RobotMode.TO_PICKUP else tk.dropoff
            desc += f"task {tk.id} (priority {tk.priority}) -> goal {goal}"
        lines.append(desc)
    counts = {s: 0 for s in TaskStatus}
    for tk in engine.book.tasks.values():
        counts[tk.status] += 1
    lines.append(
        "tasks:  "
        f"waiting release {counts[TaskStatus.UNRELEASED]} | pending {counts[TaskStatus.PENDING]} | "
        f"assigned {counts[TaskStatus.ASSIGNED]} | carried {counts[TaskStatus.PICKED]} | "
        f"delivered {counts[TaskStatus.DELIVERED]}"
    )
    if outcome is not None:
        lines.append("this step: " + describe_step(outcome))
    m = engine.metrics
    rr = m.conflict_counts["RRv"] + m.conflict_counts["RRs"]
    ro = m.conflict_counts["ROv"] + m.conflict_counts["ROs"]
    lines.append(
        f"totals: conflicts prevented {rr + ro + m.conflict_counts['INV']} "
        f"(robot-robot {rr}, robot-obstacle {ro}, invalid {m.conflict_counts['INV']}) | "
        f"blocked moves {m.blocked_moves} | physical collisions {m.physical_collisions}"
    )
    return "\n".join(lines)


def describe_step(out: StepOutcome) -> str:
    parts: list[str] = []
    if out.released:
        parts.append("released task " + ", ".join(map(str, out.released)))
    for r, j in out.assigned:
        parts.append(f"robot {robot_char(r)} assigned task {j}")
    for r, j in out.picked:
        parts.append(f"robot {robot_char(r)} picked up task {j}")
    for r, j in out.delivered:
        parts.append(f"robot {robot_char(r)} DELIVERED task {j}")
    for e in sorted(out.events):
        names = {
            ConflictKind.RR_VERTEX: "robots want the same cell",
            ConflictKind.RR_SWAP: "robots would swap through each other",
            ConflictKind.RO_VERTEX: "robot would hit an obstacle",
            ConflictKind.RO_SWAP: "robot would pass through an obstacle",
            ConflictKind.INVALID: "move into a wall",
        }
        parts.append(f"STOPPED {'/'.join(e.entities)}: {names[e.kind]}")
    if out.blocked:
        parts.append(
            "waited behind stopped robot: " + ", ".join(robot_char(i) for i in sorted(out.blocked))
        )
    for ev in out.map_events:
        parts.append(f"map: {ev.describe()}")
    return "; ".join(parts) if parts else "-"


def enable_windows_ansi() -> bool:
    """Enable ANSI escape processing on a Windows console.

    Returns True if ANSI sequences can be used (always on non-Windows platforms); False if the
    output is not a console (e.g. redirected) or the mode could not be set.
    """
    import os

    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 - windll exists only on Windows
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # pragma: no cover - platform specific
        return False
