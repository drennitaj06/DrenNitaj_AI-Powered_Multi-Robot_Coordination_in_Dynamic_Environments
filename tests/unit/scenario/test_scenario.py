from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from robocoord.core.distance import connected_components
from robocoord.core.grid import OccupancyGrid
from robocoord.core.mapio import save_movingai_map
from robocoord.core.types import Action
from robocoord.scenario import ScenarioError, build_scenario, load_scenario
from robocoord.scenario.generators import keep_largest_component, random_density, rooms, warehouse


def base() -> dict[str, Any]:
    return {
        "scenario": {"name": "t", "seed": 3, "max_steps": 50},
        "map": {"rows": ["......", "..@@..", "......"]},
        "robots": [{"start": [0, 0]}, {"start": [5, 0]}],
        "obstacles": [
            {"start": [0, 2], "model": "patrol", "waypoints": [[0, 2], [5, 2]]},
            {"start": [3, 0], "model": "bounce", "direction": "left"},
        ],
        "tasks": [
            {"pickup": [1, 0], "dropoff": [4, 2], "priority": 2, "release": 0},
            {"pickup": [5, 1], "dropoff": [0, 1], "priority": 3, "release": 4},
        ],
        "arrivals": {"kind": "bernoulli", "rate": 0.1, "priorities": [0.5, 0.3, 0.2], "until": 20},
        "map_events": [{"time": 5, "cell": [2, 0], "action": "block"}],
        "script": {"actions": ["RRR", "LL"], "assignments": [[0], [1]]},
    }


def test_valid_scenario_builds() -> None:
    spec = build_scenario(base())
    assert spec.name == "t" and spec.seed == 3 and spec.max_steps == 50
    assert spec.grid.is_blocked((2, 1)) and spec.robots == ((0, 0), (5, 0))
    assert spec.obstacles[1].direction is Action.LEFT
    assert spec.tasks[1].priority == 3 and spec.map_events[0].block
    assert spec.script is not None and spec.script.assignments == ((0,), (1,))
    assert build_scenario(base(), seed=11).seed == 11


def _set(d: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    d = copy.deepcopy(d)
    keys: list[Any] = [int(k) if k.isdigit() else k for k in path.split(".")]
    cur: Any = d
    for k in keys[:-1]:
        cur = cur[k]
    if value is _DELETE:
        del cur[keys[-1]]
    else:
        cur[keys[-1]] = value
    return d


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("scenario.name", "", "scenario.name"),
        ("scenario.max_steps", 0, "scenario.max_steps"),
        ("scenario.max_steps", "ten", "scenario.max_steps"),
        ("robots.1.start", [0, 0], "robots[1].start"),  # overlapping starts
        ("robots.1.start", [9, 0], "robots[1].start"),  # out of bounds
        ("robots.1.start", [2, 1], "robots[1].start"),  # static obstacle
        ("robots.0.start", [1, 2, 3], "robots[0].start"),  # malformed
        ("obstacles.0.start", [0, 0], "obstacles[0].start"),  # overlaps robot
        ("obstacles.0.model", "teleport", "obstacles[0].model"),
        ("obstacles.0.waypoints", [], "obstacles[0].waypoints"),
        ("obstacles.0.waypoints", [[2, 1]], "obstacles[0].waypoints[0]"),
        ("obstacles.1.direction", "north", "obstacles[1].direction"),
        ("tasks.0.pickup", [2, 1], "tasks[0].pickup"),
        ("tasks.0.dropoff", [1, 0], "tasks[0]"),  # same as pickup
        ("tasks.0.priority", 4, "tasks[0].priority"),
        ("tasks.0.release", -1, "tasks[0].release"),
        ("arrivals.kind", "storm", "arrivals.kind"),
        ("arrivals.rate", 2.0, "arrivals.rate"),
        ("arrivals.priorities", [0.5, 0.5], "arrivals.priorities"),
        ("arrivals.priorities", [0.5, 0.3, 0.3], "arrivals.priorities"),
        ("map_events.0.action", "explode", "map_events[0].action"),
        ("map_events.0.cell", [7, 7], "map_events[0].cell"),
        ("script.actions", ["RRX", "LL"], "script.actions[0]"),
        ("script.actions", ["RR"], "script.actions"),
        ("script.assignments", [[0], [0]], "script.assignments[1]"),
        ("script.assignments", [[5], []], "script.assignments[0]"),
        ("map", {"rows": ["..", "."]}, "map.rows"),
        ("map", {}, "map"),
        ("map", {"generator": {"kind": "spiral", "width": 4, "height": 4}}, "map.generator.kind"),
    ],
)
def test_invalid_scenarios_name_the_field(path: str, value: Any, field: str) -> None:
    with pytest.raises(ScenarioError) as info:
        build_scenario(_set(base(), path, value))
    assert info.value.field == field, str(info.value)


def test_unreachable_task_and_unknown_section() -> None:
    d = base()
    d["map"] = {"rows": ["..@..", "..@..", "..@.."]}
    d["obstacles"] = []
    d["map_events"] = []
    d["robots"] = [{"start": [0, 0]}]
    d["script"] = {"actions": [""], "assignments": [[]]}
    d["tasks"] = [{"pickup": [0, 1], "dropoff": [4, 1]}]
    with pytest.raises(ScenarioError, match="not reachable") as info:
        build_scenario(d)
    assert info.value.field == "tasks[0]"
    with pytest.raises(ScenarioError) as info2:
        build_scenario({**base(), "bogus": {}})
    assert info2.value.field == "bogus"


def test_no_robots() -> None:
    d = base()
    d["robots"] = []
    d.pop("script")
    with pytest.raises(ScenarioError) as info:
        build_scenario(d)
    assert info.value.field == "robots"


def test_load_from_file_and_map_file(tmp_path: Path) -> None:
    save_movingai_map(OccupancyGrid.from_rows(["....", ".@..", "...."]), tmp_path / "m.map")
    (tmp_path / "s.toml").write_text(
        '[scenario]\nname = "f"\n[map]\nfile = "m.map"\n[[robots]]\nstart = [0, 0]\n',
        encoding="utf-8",
    )
    spec = load_scenario(tmp_path / "s.toml")
    assert spec.grid.is_blocked((1, 1)) and spec.max_steps == 100
    with pytest.raises(ScenarioError) as info:
        load_scenario(tmp_path / "missing.toml")
    assert info.value.field == "file"
    (tmp_path / "bad.toml").write_text("[scenario\n", encoding="utf-8")
    with pytest.raises(ScenarioError):
        load_scenario(tmp_path / "bad.toml")


def test_generator_and_random_placement_are_seeded() -> None:
    d = {
        "scenario": {"name": "g", "seed": 5},
        "map": {"generator": {"kind": "random", "width": 12, "height": 9, "density": 0.25}},
        "placement": {
            "robots": 4,
            "obstacles": 3,
            "tasks": 6,
            "max_release": 10,
            "priorities": [0.5, 0.3, 0.2],
        },
    }
    a, b = build_scenario(d), build_scenario(d)
    assert a.grid == b.grid and a.robots == b.robots and a.tasks == b.tasks
    assert len(a.robots) == 4 and len(a.obstacles) == 3 and len(a.tasks) == 6
    c = build_scenario(d, seed=6)
    assert (c.grid, c.robots) != (a.grid, a.robots)
    too_many = copy.deepcopy(d)
    too_many["placement"]["robots"] = 500
    with pytest.raises(ScenarioError) as info:
        build_scenario(too_many)
    assert info.value.field == "placement"


def _connected(grid: OccupancyGrid) -> bool:
    labels = connected_components(grid)
    return int(labels.max(initial=-1)) <= 0


@given(st.integers(2, 20), st.integers(2, 20), st.floats(0.0, 0.6), st.integers(0, 10_000))
def test_random_density_is_connected(w: int, h: int, density: float, seed: int) -> None:
    grid = random_density(w, h, density, np.random.default_rng(seed))
    assert _connected(grid)


@given(st.integers(3, 30), st.integers(3, 30), st.integers(1, 6), st.integers(0, 1000))
def test_rooms_connected_and_keep_most_cells(w: int, h: int, room: int, seed: int) -> None:
    grid = rooms(w, h, room, np.random.default_rng(seed))
    assert _connected(grid)
    assert grid.num_free() > 0


@given(st.integers(3, 40), st.integers(3, 40), st.integers(1, 8))
def test_warehouse_connected(w: int, h: int, shelf: int) -> None:
    grid = warehouse(w, h, shelf)
    assert _connected(grid)


@given(st.integers(3, 40), st.integers(3, 40), st.integers(1, 8), st.integers(0, 1000))
def test_rooms_and_warehouse_are_connected_by_construction(
    w: int, h: int, room: int, seed: int
) -> None:
    """Review finding: the generators must not rely on the final pruning step. With pruning
    disabled the free space must already be one component, so no cells are silently dropped."""
    import robocoord.scenario.generators as gen

    original = gen.keep_largest_component
    gen.keep_largest_component = lambda grid: grid  # type: ignore[assignment]
    try:
        raw_rooms = gen.rooms(w, h, room, np.random.default_rng(seed))
        raw_wh = gen.warehouse(w, h, room)
    finally:
        gen.keep_largest_component = original
    assert _connected(raw_rooms) and _connected(raw_wh)


def test_keep_largest_component() -> None:
    g = OccupancyGrid.from_rows(["..@.", "..@.", "@@@."])
    kept = keep_largest_component(g)
    assert kept.num_free() == 4 and kept.is_free((0, 2)) and kept.is_blocked((3, 0))
