"""Experiment specifications (TOML) and their expansion into individual runs.

File format::

    [experiment]
    name = "e1b_neighbor_index"     # output directory name
    kind = "planner_queries"         # executor: planner_queries | replanning
    description = "..."
    seed_base = 30000                # from the documented seed range (05 §B.1)
    timeout_s = 30.0                 # per-run cap; exceeding it is recorded as "timeout"
    workers = 8                      # parallel worker processes (default 1)

    [factors]                        # full Cartesian product, in this key order
    family = ["open", "d20"]
    size = [64]

    [fixed]                          # passed unchanged to every run
    record_trace = true

    [[planners]]                     # optional: labelled planner configurations (factor "planner")
    label = "rrtstar-1000"
    name = "rrtstar"
    options = { max_iterations = 1000 }
"""

from __future__ import annotations

import itertools
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robocoord.core.errors import ConfigError

KINDS = ("planner_queries", "replanning")


@dataclass(frozen=True)
class PlannerSetting:
    label: str
    name: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    """One run: a stable id (used for resuming) and its parameters."""

    run_id: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    kind: str
    description: str = ""
    seed_base: int = 30_000
    timeout_s: float = 30.0
    workers: int = 1
    factors: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    fixed: Mapping[str, Any] = field(default_factory=dict)
    planners: Mapping[str, PlannerSetting] = field(default_factory=dict)

    def runs(self) -> list[RunSpec]:
        """Cartesian product of the factors, in declaration order (deterministic)."""
        keys = list(self.factors)
        out: list[RunSpec] = []
        for values in itertools.product(*(self.factors[k] for k in keys)):
            params = dict(zip(keys, values, strict=True))
            run_id = "|".join(f"{k}={v}" for k, v in params.items())
            out.append(RunSpec(run_id, params))
        return out


def _require(table: Mapping[str, Any], key: str, typ: type, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"{where}.{key}", "missing")
    value = table[key]
    if not isinstance(value, typ) or (isinstance(value, bool) and typ is not bool):
        raise ConfigError(f"{where}.{key}", f"expected {typ.__name__}, got {value!r}")
    return value


def parse_spec(data: Mapping[str, Any]) -> ExperimentSpec:
    known = {"experiment", "factors", "fixed", "planners"}
    for key in data:
        if key not in known:
            raise ConfigError(key, f"unknown section; expected one of {sorted(known)}")
    exp = data.get("experiment")
    if not isinstance(exp, Mapping):
        raise ConfigError("experiment", "missing [experiment] table")
    name = _require(exp, "name", str, "experiment")
    kind = _require(exp, "kind", str, "experiment")
    if kind not in KINDS:
        raise ConfigError("experiment.kind", f"unknown kind {kind!r}; expected one of {KINDS}")
    factors_raw = data.get("factors", {})
    factors: dict[str, tuple[Any, ...]] = {}
    for k, v in factors_raw.items():
        if not isinstance(v, list) or not v:
            raise ConfigError(f"factors.{k}", "must be a non-empty list")
        factors[k] = tuple(v)
    planners: dict[str, PlannerSetting] = {}
    for i, p in enumerate(data.get("planners", [])):
        label = _require(p, "label", str, f"planners[{i}]")
        pname = _require(p, "name", str, f"planners[{i}]")
        if label in planners:
            raise ConfigError(f"planners[{i}].label", f"duplicate label {label!r}")
        planners[label] = PlannerSetting(label, pname, dict(p.get("options", {})))
    if "planner" in factors:
        for label in factors["planner"]:
            if label not in planners:
                raise ConfigError("factors.planner", f"no [[planners]] entry labelled {label!r}")
    workers = int(exp.get("workers", 1))
    if workers < 1:
        raise ConfigError("experiment.workers", "must be >= 1")
    timeout = float(exp.get("timeout_s", 30.0))
    if timeout <= 0:
        raise ConfigError("experiment.timeout_s", "must be > 0")
    return ExperimentSpec(
        name=name,
        kind=kind,
        description=str(exp.get("description", "")),
        seed_base=int(exp.get("seed_base", 30_000)),
        timeout_s=timeout,
        workers=workers,
        factors=factors,
        fixed=dict(data.get("fixed", {})),
        planners=planners,
    )


def load_spec(path: str | Path) -> ExperimentSpec:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError("file", f"experiment file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError("file", f"invalid TOML: {exc}") from None
    return parse_spec(data)
