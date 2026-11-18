"""Resumable experiment runner (05 §B.1–B.2).

* Every run has a stable id; finished runs are read back from ``records.jsonl`` and skipped, so
  an interrupted experiment continues where it stopped.
* Runs execute in worker processes (``spawn`` on Windows); records are appended as runs finish.
* Each run is capped at ``timeout_s`` and recorded as ``timeout`` when it exceeds the cap; an
  exception is recorded as ``error`` with its message (never silently dropped).
* ``env.json`` stores the hardware/software environment; ``spec.json`` the expanded spec.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robocoord.experiments.records import RecordStore, RunRecord, environment_info
from robocoord.experiments.spec import ExperimentSpec, RunSpec, load_spec

Executor = Callable[[ExperimentSpec, Mapping[str, Any]], tuple[str, dict[str, Any]]]


def _executor(kind: str) -> Executor:
    from robocoord.experiments import planning_runs

    if kind == "planner_queries":
        return planning_runs.run_query
    if kind == "replanning":
        return planning_runs.run_episode
    raise ValueError(f"unknown experiment kind {kind!r}")


def execute(spec: ExperimentSpec, run: RunSpec) -> RunRecord:
    """Run one experiment run in the current process (also the worker entry point)."""
    t0 = time.perf_counter()
    try:
        outcome, metrics = _executor(spec.kind)(spec, run.params)
        return RunRecord(
            run.run_id, spec.name, dict(run.params), outcome, metrics, time.perf_counter() - t0
        )
    except Exception as exc:  # recorded, not dropped (05 §B.1)
        return RunRecord(
            run.run_id,
            spec.name,
            dict(run.params),
            "error",
            {},
            time.perf_counter() - t0,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        )


def run_experiment(
    spec: ExperimentSpec,
    out_dir: str | Path,
    *,
    workers: int | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = print,
) -> RecordStore:
    """Execute every run not yet recorded in ``out_dir``. Returns the record store."""
    store = RecordStore(out_dir)
    (store.directory / "env.json").write_text(
        json.dumps(environment_info(), indent=2, sort_keys=True), encoding="utf-8"
    )
    spec_dict = asdict(spec)
    (store.directory / "spec.json").write_text(
        json.dumps(spec_dict, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    done = store.completed_ids()
    todo = [r for r in spec.runs() if r.run_id not in done]
    if limit is not None:
        todo = todo[:limit]
    total = len(todo)
    n_workers = workers or spec.workers
    if progress:
        progress(
            f"{spec.name}: {len(done)} runs already recorded, {total} to run "
            f"({n_workers} worker(s))"
        )
    t0 = time.perf_counter()
    finished = 0

    def report(batch: list[RunRecord]) -> None:
        nonlocal finished
        store.append(batch)
        finished += len(batch)
        if progress and (finished % max(1, total // 20) < len(batch) or finished == total):
            progress(f"  {finished}/{total} runs ({time.perf_counter() - t0:.0f} s)")

    if n_workers <= 1:
        for run in todo:
            report([execute(spec, run)])
        return store
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(execute, spec, run) for run in todo]
        batch: list[RunRecord] = []
        for fut in as_completed(futures):
            batch.append(fut.result())
            if len(batch) >= 10:
                report(batch)
                batch = []
        if batch:
            report(batch)
    return store


def run_experiment_cli(
    config: Path, out: Path | None, workers: int | None, limit: int | None
) -> int:
    spec = load_spec(config)
    out_dir = out or Path("results") / "experiments" / spec.name
    store = run_experiment(spec, out_dir, workers=workers, limit=limit)
    records = store.read()
    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"records: {len(records)} in {store.path} | outcomes {dict(sorted(outcomes.items()))}")
    return 0
