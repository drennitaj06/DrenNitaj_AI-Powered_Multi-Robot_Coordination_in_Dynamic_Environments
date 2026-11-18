"""Tiny end-to-end runs of every planner experiment kind, through the CLI and the summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from robocoord.cli import main
from robocoord.experiments.analysis import summarize
from robocoord.experiments.runner import run_experiment
from robocoord.experiments.spec import parse_spec

E1B = {
    "experiment": {"name": "e1b_neighbor_index", "kind": "planner_queries", "seed_base": 39_600},
    "factors": {
        "family": ["open"],
        "size": [10],
        "map": [0],
        "pair": [0, 1],
        "n_max": [100, 300],
        "index": ["kdtree", "brute"],
    },
    "fixed": {"record_trace": True},
}
MEM = {
    "experiment": {"name": "e1b_memory", "kind": "planner_queries", "seed_base": 39_600},
    "factors": {
        "family": ["open"],
        "size": [10],
        "map": [0],
        "pair": [0],
        "n_max": [100, 300],
        "index": ["kdtree", "brute"],
    },
    "fixed": {"measure_memory": True},
}
E1A = {
    "experiment": {"name": "e1a_planners_static", "kind": "planner_queries", "seed_base": 39_700},
    "factors": {
        "family": ["d20", "rooms", "warehouse"],
        "size": [16],
        "map": [0],
        "pair": [0],
        "planner": ["astar", "rrtstar-200"],
    },
    "planners": [
        {"label": "astar", "name": "astar"},
        {"label": "rrtstar-200", "name": "rrtstar", "options": {"max_iterations": 200}},
    ],
}
E1C = {
    "experiment": {"name": "e1c_convergence", "kind": "planner_queries", "seed_base": 39_800},
    "factors": {
        "family": ["open"],
        "size": [10],
        "map": [0],
        "pair": [0],
        "n_max": [300],
        "index": ["kdtree"],
    },
    "fixed": {"checkpoints": [1, 10, 100, 300, 1000]},
}
E2 = {
    "experiment": {"name": "e2_replanning", "kind": "replanning", "seed_base": 20_000},
    "factors": {
        "planner": ["astar", "rrtstar"],
        "obstacles": [0, 4],
        "events": ["low"],
        "team": [2],
        "seed": [20_000, 20_001],
    },
    "fixed": {"size": 12, "max_steps": 60, "tasks_per_robot": 1},
    "planners": [
        {"label": "astar", "name": "astar"},
        {"label": "rrtstar", "name": "rrtstar", "options": {"max_iterations": 150}},
    ],
}


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        (E1B, "Index-independence check: 4/4"),
        (E1A, "A* expansions vs V"),
        (E1C, "First solution iteration"),
        (MEM, "bytes per tree node"),
        (E2, "Paired differences"),
    ],
)
def test_run_and_summarize(tmp_path: Path, data: dict[str, object], needle: str) -> None:
    spec = parse_spec(data)
    out = tmp_path / spec.name
    store = run_experiment(spec, out, progress=None)
    records = store.read()
    assert len(records) == len(spec.runs())
    assert all(r["outcome"] != "error" for r in records), [r.get("error") for r in records]
    text = summarize(out)
    assert needle in text and (out / "summary.md").exists()


def test_experiment_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "tiny.toml"
    cfg.write_text(
        '[experiment]\nname = "tiny"\nkind = "planner_queries"\nseed_base = 39900\n'
        '[factors]\nfamily = ["open"]\nsize = [8]\nmap = [0]\npair = [0, 1]\n'
        'planner = ["a"]\n[[planners]]\nlabel = "a"\nname = "astar"\n',
        encoding="utf-8",
    )
    assert (
        main(["experiment", "--config", str(cfg), "--out", str(tmp_path / "o"), "--workers", "2"])
        == 0
    )
    assert "records: 2" in capsys.readouterr().out
    assert main(["experiment", "--config", str(tmp_path / "missing.toml")]) == 2
