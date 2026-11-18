"""Run records: one JSON line per run, appended as runs finish (crash-safe, resumable)."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import robocoord

#: Run outcomes. ``timeout`` = exceeded the per-run cap; ``error`` = exception (message kept).
OUTCOMES = ("ok", "unreachable", "budget_exhausted", "invalid_query", "timeout", "error", "done")


@dataclass
class RunRecord:
    run_id: str
    experiment: str
    params: Mapping[str, Any]
    outcome: str
    metrics: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(_plain(asdict(self)), sort_keys=True, separators=(",", ":"))


def _plain(x: Any) -> Any:
    """Convert NumPy scalars/arrays and non-finite floats to JSON-safe values."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


class RecordStore:
    """``records.jsonl`` in the experiment's output directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "records.jsonl"

    def completed_ids(self) -> set[str]:
        return {r["run_id"] for r in self.read()}

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # a line truncated by a crash: rerun that run
                    continue
        return out

    def append(self, records: Iterable[RunRecord]) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(r.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def environment_info() -> dict[str, Any]:
    """Hardware and software of the machine running the experiment (05 §B.2)."""
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "robocoord": robocoord.__version__,
    }
