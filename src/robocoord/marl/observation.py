"""Egocentric per-robot observations (model §6.1; algorithms §8). NumPy only.

The observation of robot ``i`` is built from the :class:`~robocoord.sim.view.WorldView` it acts
on, so training (``marl.env``) and execution (``marl.policy.LearnedController``) share one code
path. The engine must be created with ``view_extras={"paths", "goal_distance"}``.

Layout of the flat ``float32`` vector (``d_o = 7·(2r+1)² + 13``, 356 for ``r = 3``):

* window channels ``[c, dy + r, dx + r]`` for the cell ``x_i + (dx, dy)``, flattened in C order:
  0 static obstacle or out of bounds; 1 dynamic obstacle now; 2 its constant-velocity prediction
  ``z + v`` (only for obstacles sensed in the window, only free in-bounds cells, as
  ``planning.constraints.sense``); 3 other robot; 4 the other robot's task priority ``π / P``
  (0 if idle); 5 own reference path, the next ``L_o`` cells; 6 own goal;
* 13 scalars: goal offset ``(g − x) / max(W, H)`` (2), ``D̄(x, g) / (W + H)`` (1), mode one-hot
  idle / to_pickup / to_dropoff (3), own task priority ``π / P`` (1), suggested-action one-hot
  (5; zeros without a planner), plan-valid flag (1). Goal-dependent entries are 0 when idle.

Nothing depends on the team size or the map size (FR-RL-5).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from robocoord.marl.config import (
    CH_DYN,
    CH_DYN_PRED,
    CH_GOAL,
    CH_PATH,
    CH_ROBOT,
    CH_ROBOT_PRIO,
    CH_STATIC,
    N_CHANNELS,
    N_SCALARS,
    ObsParams,
)
from robocoord.sim.view import WorldView
from robocoord.world.entities import RobotMode

Array = npt.NDArray[np.float32]

_MODE_INDEX = {RobotMode.IDLE: 0, RobotMode.TO_PICKUP: 1, RobotMode.TO_DROPOFF: 2}
# scalar offsets
S_GOAL_DX, S_GOAL_DY, S_DIST, S_MODE, S_PRIO, S_SUGGEST, S_VALID = 0, 1, 2, 3, 6, 7, 12


class ObservationError(ValueError):
    """The view lacks the data the observation needs (engine created without view extras)."""


class ObservationBuilder:
    """Builds the ``(N, d_o)`` observation matrix of a view. Time O(N (N + K) + N r²)."""

    def __init__(self, params: ObsParams | None = None) -> None:
        self.params = params or ObsParams()
        self._prio_scale = np.float32(self.params.max_priority)
        self._pad: tuple[int, int, int, npt.NDArray[np.bool_]] | None = None

    @property
    def dim(self) -> int:
        return self.params.dim

    def _padded_static(self, view: WorldView) -> npt.NDArray[np.bool_]:
        m = view.map
        key = (id(m), m.version, m.width)
        if self._pad is None or self._pad[:3] != key:
            r = self.params.radius
            padded = np.pad(np.asarray(m.blocked, dtype=bool), r, constant_values=True)
            self._pad = (*key, padded)
        return self._pad[3]

    def priority_value(self, level: int) -> np.float32:
        """``π / P`` computed in float32 (the codec in ``marl.replay`` inverts it exactly)."""
        return np.float32(level) / self._prio_scale

    def build(self, view: WorldView) -> Array:
        """Observation matrix, one row per robot in robot-id order."""
        if view.goal_distance is None:
            raise ObservationError(
                "the view has no goal distances: create the Engine with "
                "view_extras={'paths', 'goal_distance'}"
            )
        p = self.params
        r, k = p.radius, p.window
        n = len(view.robots)
        grid = np.zeros((n, N_CHANNELS, k, k), dtype=np.float32)
        scal = np.zeros((n, N_SCALARS), dtype=np.float32)
        padded = self._padded_static(view)
        width, height = view.map.width, view.map.height
        size_norm = np.float32(max(width, height))
        dist_norm = np.float32(width + height)
        prio = {tk.id: tk.priority for tk in view.tasks}
        robot_level = [
            0 if (rb.task is None or rb.mode is RobotMode.IDLE) else prio[rb.task]
            for rb in view.robots
        ]
        paths = view.reference_paths
        obstacles = [(o.position, o.velocity) for o in view.obstacles]

        for rb in view.robots:
            i = rb.id
            x, y = rb.position
            g = grid[i]
            g[CH_STATIC] = padded[y : y + k, x : x + k]
            for (ox, oy), (vx, vy) in obstacles:
                dx, dy = ox - x, oy - y
                if abs(dx) > r or abs(dy) > r:
                    continue
                g[CH_DYN, dy + r, dx + r] = 1.0
                px, py = ox + vx, oy + vy  # v = 0 predicts the current cell, as in ``sense``
                if abs(px - x) <= r and abs(py - y) <= r and view.map.is_free((px, py)):
                    g[CH_DYN_PRED, py - y + r, px - x + r] = 1.0
            for other in view.robots:
                if other.id == i:
                    continue
                dx, dy = other.position[0] - x, other.position[1] - y
                if abs(dx) <= r and abs(dy) <= r:
                    g[CH_ROBOT, dy + r, dx + r] = 1.0
                    g[CH_ROBOT_PRIO, dy + r, dx + r] = self.priority_value(robot_level[other.id])
            path = None if paths is None else paths[i]
            if path is not None:
                for cx, cy in path[1 : 1 + p.path_cells]:
                    dx, dy = cx - x, cy - y
                    if abs(dx) <= r and abs(dy) <= r:
                        g[CH_PATH, dy + r, dx + r] = 1.0
            s = scal[i]
            s[S_MODE + _MODE_INDEX[rb.mode]] = 1.0
            if rb.goal is not None:
                gx, gy = rb.goal
                if abs(gx - x) <= r and abs(gy - y) <= r:
                    g[CH_GOAL, gy - y + r, gx - x + r] = 1.0
                s[S_GOAL_DX] = np.float32(gx - x) / size_norm
                s[S_GOAL_DY] = np.float32(gy - y) / size_norm
                d = view.goal_distance[i]
                s[S_DIST] = np.float32(0 if d is None else d) / dist_norm
                s[S_PRIO] = self.priority_value(robot_level[i])
            if view.suggested is not None:
                s[S_SUGGEST + int(view.suggested[i])] = 1.0
            s[S_VALID] = 1.0 if path is not None else 0.0
        return np.concatenate([grid.reshape(n, -1), scal], axis=1)


def action_mask_array(view: WorldView) -> npt.NDArray[np.bool_]:
    """The static feasibility masks of the view as an ``(N, 5)`` boolean array (ADR-10)."""
    return np.asarray(view.masks, dtype=bool)
