"""MovingAI ``.map`` file format I/O (FR-ENV-9).

Format::

    type octile
    height H
    width W
    map
    <H rows of W characters>

Rows are written top-down; ``.``/``G``/``S`` are passable, ``@``/``O``/``T``/``W`` impassable.
We treat ``T`` (trees) and ``W`` (water) as impassable for ground robots. ``#`` is also accepted
for hand-written maps.
"""

from __future__ import annotations

from pathlib import Path

from robocoord.core.grid import OccupancyGrid


class MapFormatError(ValueError):
    """Raised for malformed map files."""


def parse_movingai(text: str) -> OccupancyGrid:
    """Parse the contents of a MovingAI map file."""
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    header: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "map":
            break
        parts = line.split()
        if len(parts) != 2:
            raise MapFormatError(f"line {i}: malformed header line {line!r}")
        header[parts[0]] = parts[1]
    else:
        raise MapFormatError("missing 'map' line")
    try:
        height = int(header["height"])
        width = int(header["width"])
    except KeyError as exc:
        raise MapFormatError(f"missing header field {exc.args[0]!r}") from None
    except ValueError:
        raise MapFormatError("height/width must be integers") from None
    rows = [ln for ln in lines[i : i + height]]
    if len(rows) != height:
        raise MapFormatError(f"expected {height} map rows, found {len(rows)}")
    for r, row in enumerate(rows):
        if len(row) != width:
            raise MapFormatError(f"map row {r} has length {len(row)}, expected {width}")
    try:
        return OccupancyGrid.from_rows(rows)
    except ValueError as exc:
        raise MapFormatError(str(exc)) from None


def format_movingai(grid: OccupancyGrid) -> str:
    """Serialise a grid in MovingAI format."""
    lines = ["type octile", f"height {grid.height}", f"width {grid.width}", "map", *grid.to_rows()]
    return "\n".join(lines) + "\n"


def load_movingai_map(path: str | Path) -> OccupancyGrid:
    return parse_movingai(Path(path).read_text(encoding="utf-8"))


def save_movingai_map(grid: OccupancyGrid, path: str | Path) -> None:
    Path(path).write_text(format_movingai(grid), encoding="utf-8")
