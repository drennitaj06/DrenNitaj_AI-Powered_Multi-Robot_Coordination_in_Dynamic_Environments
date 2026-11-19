"""Fixed-size centralised state ``φ(s)`` for a CTDE critic (model §6.2). NumPy only.

Four channels — static obstacles, dynamic obstacles, robots, and open (unpicked, released) tasks'
pickups weighted by their aged weight ``w̃_j(t)`` — each average-pooled into ``G × G`` blocks,
followed by three scalars: fraction of released tasks delivered, ``t / T_max`` and open tasks
``/ N``. Size ``4G² + 3`` (259 for ``G = 8``), independent of ``N`` and of the map size. Built in
Issue #5 so that the environment API is complete; it is consumed by MAPPO in Issue #6.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from robocoord.allocation.benefits import effective_weight
from robocoord.sim.view import WorldView
from robocoord.world.entities import TaskStatus

#: Aging rate used for the pickup weights (the allocation default, model §11).
AGING_ALPHA = 0.01


def global_state_size(blocks: int) -> int:
    return 4 * blocks * blocks + 3


def _pool(layer: npt.NDArray[np.float64], blocks: int) -> npt.NDArray[np.float64]:
    """Mean over a ``blocks × blocks`` partition (``array_split`` bounds; empty blocks → 0)."""
    h, w = layer.shape
    rows = np.array_split(np.arange(h), blocks)
    cols = np.array_split(np.arange(w), blocks)
    out = np.zeros((blocks, blocks), dtype=np.float64)
    for a, rr in enumerate(rows):
        for b, cc in enumerate(cols):
            if rr.size and cc.size:
                out[a, b] = layer[rr[0] : rr[-1] + 1, cc[0] : cc[-1] + 1].mean()
    return out


def global_state(view: WorldView, horizon: int, blocks: int = 8) -> npt.NDArray[np.float32]:
    """``φ(s)`` of the view (time ``O(W H + N + M)``)."""
    w, h = view.map.width, view.map.height
    static = np.asarray(view.map.blocked, dtype=np.float64)
    dyn = np.zeros((h, w), dtype=np.float64)
    for o in view.obstacles:
        dyn[o.position[1], o.position[0]] = 1.0
    robots = np.zeros((h, w), dtype=np.float64)
    for r in view.robots:
        robots[r.position[1], r.position[0]] = 1.0
    pickups = np.zeros((h, w), dtype=np.float64)
    open_tasks = released = delivered = 0
    for tk in view.tasks:
        if tk.status is TaskStatus.UNRELEASED:
            continue
        released += 1
        if tk.status is TaskStatus.DELIVERED:
            delivered += 1
        if tk.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
            open_tasks += 1
            x, y = tk.pickup
            pickups[y, x] += effective_weight(tk.priority, tk.release, view.t, AGING_ALPHA)
    pooled = [_pool(layer, blocks).ravel() for layer in (static, dyn, robots, pickups)]
    scalars = np.array(
        [
            delivered / released if released else 0.0,
            view.t / horizon,
            open_tasks / max(1, len(view.robots)),
        ]
    )
    return np.concatenate([*pooled, scalars]).astype(np.float32)
