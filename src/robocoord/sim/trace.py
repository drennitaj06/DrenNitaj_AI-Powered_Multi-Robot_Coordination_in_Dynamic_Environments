"""JSON-Lines episode traces (ADR-7). Byte-identical for identical scenario + seed (test X-1).

Record types: ``header`` (once), ``step`` (one per simulation step), ``summary`` (once).
No wall-clock timings are written, so traces are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any


def dumps(record: dict[str, Any]) -> str:
    """Canonical serialisation: sorted keys, compact separators."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class TraceWriter:
    """Writes records to a file path or an open text stream."""

    def __init__(self, target: str | Path | IO[str]) -> None:
        if isinstance(target, str | Path):
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            self._stream: IO[str] = open(target, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
            self._owns = True
        else:
            self._stream = target
            self._owns = False

    def write(self, record: dict[str, Any]) -> None:
        self._stream.write(dumps(record))
        self._stream.write("\n")

    def close(self) -> None:
        if self._owns:
            self._stream.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
