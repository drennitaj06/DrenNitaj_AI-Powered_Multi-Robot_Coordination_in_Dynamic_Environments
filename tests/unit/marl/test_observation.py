"""Observation builder on hand-built worlds (model §6.1; algorithms §8 tests 1 and 4)."""

from __future__ import annotations

import numpy as np
import pytest

from robocoord.core.grid import OccupancyGrid
from robocoord.core.types import Action, Cell
from robocoord.marl.config import (
    CH_DYN,
    CH_DYN_PRED,
    CH_GOAL,
    CH_PATH,
    CH_ROBOT,
    CH_ROBOT_PRIO,
    CH_STATIC,
    N_CHANNELS,
    ObsParams,
)
from robocoord.marl.observation import (
    S_DIST,
    S_GOAL_DX,
    S_GOAL_DY,
    S_MODE,
    S_PRIO,
    S_SUGGEST,
    S_VALID,
    ObservationBuilder,
    ObservationError,
)
from robocoord.sim.view import MapView, ObstacleView, RobotView, TaskView, WorldView, action_mask
from robocoord.world.entities import RobotMode, TaskStatus

R = 3
K = 2 * R + 1


def make_view(
    grid: OccupancyGrid,
    robots: list[RobotView],
    tasks: list[TaskView],
    obstacles: list[ObstacleView] = (),  # type: ignore[assignment]
    suggested: tuple[Action, ...] | None = None,
    paths: tuple[tuple[Cell, ...] | None, ...] | None = None,
    dist: tuple[int | None, ...] | None = None,
) -> WorldView:
    return WorldView(
        t=5,
        max_steps=100,
        map=MapView(grid),
        robots=tuple(robots),
        obstacles=tuple(obstacles),
        tasks=tuple(tasks),
        masks=tuple(action_mask(grid, r.position) for r in robots),
        suggested=suggested,
        reference_paths=paths,
        goal_distance=dist if dist is not None else tuple(None for _ in robots),
    )


def window(obs: np.ndarray, i: int) -> np.ndarray:
    return obs[i, : N_CHANNELS * K * K].reshape(N_CHANNELS, K, K)


def at(dx: int, dy: int) -> tuple[int, int]:
    return dy + R, dx + R


def scenario_view() -> WorldView:
    grid = OccupancyGrid.empty(10, 10)
    grid.set_blocked((8, 5), True)
    tasks = [
        TaskView(0, (7, 5), (7, 9), 2, 0, TaskStatus.ASSIGNED, 0, True),
        TaskView(1, (0, 0), (4, 9), 3, 0, TaskStatus.PICKED, 1, True),
    ]
    robots = [
        RobotView(0, (5, 5), RobotMode.TO_PICKUP, 0, (7, 5)),
        RobotView(1, (4, 6), RobotMode.TO_DROPOFF, 1, (4, 9)),
        RobotView(2, (1, 1), RobotMode.IDLE, None, None),
    ]
    obstacles = [ObstacleView(0, (5, 7), (0, -1)), ObstacleView(1, (2, 5), (0, 0))]
    paths: tuple[tuple[Cell, ...] | None, ...] = (((5, 5), (6, 5), (7, 5)), ((4, 6), (4, 7)), None)
    return make_view(
        grid,
        robots,
        tasks,
        obstacles,
        suggested=(Action.RIGHT, Action.UP, Action.STAY),
        paths=paths,
        dist=(2, 3, None),
    )


def test_dimension_is_356_by_default() -> None:
    assert ObsParams().dim == 356
    obs = ObservationBuilder().build(scenario_view())
    assert obs.shape == (3, 356) and obs.dtype == np.float32


def test_window_channels_of_robot_0() -> None:
    w = window(ObservationBuilder().build(scenario_view()), 0)
    assert w[CH_STATIC][at(3, 0)] == 1.0  # (8, 5) blocked
    assert w[CH_STATIC].sum() == 1.0  # nothing else blocked; window fully inside the map
    assert w[CH_DYN][at(0, 2)] == 1.0 and w[CH_DYN][at(-3, 0)] == 1.0
    assert w[CH_DYN].sum() == 2.0
    # constant-velocity predictions: (5,7)+(0,-1) = (5,6); v = 0 predicts the current cell
    assert w[CH_DYN_PRED][at(0, 1)] == 1.0 and w[CH_DYN_PRED][at(-3, 0)] == 1.0
    assert w[CH_DYN_PRED].sum() == 2.0
    assert w[CH_ROBOT][at(-1, 1)] == 1.0 and w[CH_ROBOT].sum() == 1.0  # robot 2 is out of range
    assert w[CH_ROBOT_PRIO][at(-1, 1)] == pytest.approx(1.0)  # robot 1 carries priority 3 = P
    assert w[CH_PATH][at(1, 0)] == 1.0 and w[CH_PATH][at(2, 0)] == 1.0
    assert w[CH_PATH].sum() == 2.0  # the own cell is not a path cell
    assert w[CH_GOAL][at(2, 0)] == 1.0 and w[CH_GOAL].sum() == 1.0


def test_scalars() -> None:
    obs = ObservationBuilder().build(scenario_view())
    s0 = obs[0, -13:]
    assert s0[S_GOAL_DX] == pytest.approx(2 / 10) and s0[S_GOAL_DY] == 0.0
    assert s0[S_DIST] == pytest.approx(2 / 20)
    assert list(s0[S_MODE : S_MODE + 3]) == [0.0, 1.0, 0.0]
    assert s0[S_PRIO] == pytest.approx(2 / 3)
    assert list(s0[S_SUGGEST : S_SUGGEST + 5]) == [0, 0, 0, 0, 1]  # RIGHT
    assert s0[S_VALID] == 1.0
    s2 = obs[2, -13:]  # idle robot: goal-dependent entries are 0, no plan
    assert s2[S_GOAL_DX] == s2[S_GOAL_DY] == s2[S_DIST] == s2[S_PRIO] == 0.0
    assert list(s2[S_MODE : S_MODE + 3]) == [1.0, 0.0, 0.0]
    assert s2[S_VALID] == 0.0 and list(s2[S_SUGGEST : S_SUGGEST + 5]) == [1, 0, 0, 0, 0]


def test_out_of_bounds_reads_as_static() -> None:
    w = window(ObservationBuilder().build(scenario_view()), 2)  # robot at (1, 1)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            outside = 1 + dx < 0 or 1 + dy < 0
            assert w[CH_STATIC][at(dx, dy)] == (1.0 if outside else 0.0)


def test_no_planner_gives_zero_suggestion_and_invalid_plan() -> None:
    v = scenario_view()
    v2 = WorldView(
        v.t,
        v.max_steps,
        v.map,
        v.robots,
        v.obstacles,
        v.tasks,
        v.masks,
        None,
        None,
        v.goal_distance,
    )
    obs = ObservationBuilder().build(v2)
    assert obs[:, -13 + S_SUGGEST : -13 + S_SUGGEST + 5].sum() == 0.0
    assert obs[:, -13 + S_VALID].sum() == 0.0
    assert window(obs, 0)[CH_PATH].sum() == 0.0


def test_view_without_extras_is_rejected() -> None:
    v = scenario_view()
    v2 = WorldView(v.t, v.max_steps, v.map, v.robots, v.obstacles, v.tasks, v.masks, None)
    with pytest.raises(ObservationError, match="view_extras"):
        ObservationBuilder().build(v2)


@pytest.mark.parametrize(("n", "size"), [(1, 6), (4, 12), (12, 40)])
def test_size_independent_of_team_and_map(n: int, size: int) -> None:
    grid = OccupancyGrid.empty(size, size)
    robots = [RobotView(i, (i % size, i // size), RobotMode.IDLE, None, None) for i in range(n)]
    obs = ObservationBuilder().build(make_view(grid, robots, []))
    assert obs.shape == (n, 356)


def test_path_channel_limited_to_l_o_cells() -> None:
    grid = OccupancyGrid.empty(20, 3)
    path = tuple((x, 1) for x in range(0, 15))
    robots = [RobotView(0, (0, 1), RobotMode.TO_PICKUP, 0, (14, 1))]
    tasks = [TaskView(0, (14, 1), (0, 0), 1, 0, TaskStatus.ASSIGNED, 0, True)]
    obs = ObservationBuilder(ObsParams(path_cells=2)).build(
        make_view(grid, robots, tasks, paths=(path,), dist=(14,), suggested=(Action.RIGHT,))
    )
    assert window(obs, 0)[CH_PATH].sum() == 2.0
