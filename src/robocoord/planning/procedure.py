"""The shared two-stage planning procedure and status mapping (model §5.3; algorithms §2.2).

Implemented once for every planner, so A* and RRT* cannot diverge in goal or obstacle policy:

1. ``invalid_query`` — start or goal out of bounds, or the start inside ``S(t)`` (the latter cannot
   happen in the simulator because map events are deferred while a cell is occupied).
2. A goal inside ``S(t)`` (only possible after a map event) is statically unreachable (step 4).
3. **Stage 1** on ``S(t) ∪ B_i`` with density ``ω_i``. Planners that do not decide reachability
   themselves (RRT*) run only after a BFS guard proves the goal reachable (Lemma 5.1).
4. **Stage 2** (static fallback) on ``S(t)`` with the same ``ω_i`` if stage 1 is infeasible; a
   success is marked ``degraded = True``.
5. ``unreachable`` if the goal is unreachable even on ``S(t)``.
6. ``budget_exhausted`` (RRT* only): reachable, but no path found within the budget.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace

from robocoord.core.types import Cell
from robocoord.planning.base import (
    PlanConstraints,
    Planner,
    PlanQuery,
    PlanResult,
    PlanStats,
    failure,
)


def reachable(constraints: PlanConstraints, start: Cell, goal: Cell) -> bool:
    """BFS on the 4-connected free cells of ``constraints`` (``S(t) ∪ B_i`` removed).

    Exact feasibility test for both planners (Lemma 5.1). Time: O(V) worst case, early exit.
    """
    w, h = constraints.width, constraints.height
    if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= goal[0] < w and 0 <= goal[1] < h):
        return False
    free = constraints.free_flat
    s = start[1] * w + start[0]
    g = goal[1] * w + goal[0]
    if not free[s] or not free[g]:
        return False
    if s == g:
        return True
    seen = bytearray(w * h)
    seen[s] = 1
    queue: deque[int] = deque([s])
    while queue:
        u = queue.popleft()
        x = u % w
        for v, ok in ((u + w, u + w < w * h), (u - w, u >= w), (u - 1, x > 0), (u + 1, x < w - 1)):
            if ok and free[v] and not seen[v]:
                if v == g:
                    return True
                seen[v] = 1
                queue.append(v)
    return False


def plan_with_procedure(
    planner: Planner,
    query: PlanQuery,
    *,
    static_reachable: Callable[[Cell, Cell], bool] | None = None,
) -> PlanResult:
    """Run ``planner`` through the shared procedure. Never raises for "no path".

    ``static_reachable`` may supply a cached static reachability test (e.g. the engine's
    :class:`~robocoord.core.distance.DistanceOracle`); by default a BFS on ``S(t)`` is used.
    """
    t0 = time.perf_counter()
    c = query.constraints
    start, goal = query.start, query.goal
    stats = PlanStats(planner=planner.name, stages=0)

    def finish(result: PlanResult, acc: PlanStats, degraded: bool = False) -> PlanResult:
        acc = replace(acc, time_s=time.perf_counter() - t0)
        return replace(result, stats=acc, degraded=degraded and result.ok)

    grid = c.static
    if not (grid.in_bounds(start) and grid.in_bounds(goal)) or grid.is_blocked(start):
        return finish(failure("invalid_query", stats), stats)
    if grid.is_blocked(goal):
        return finish(failure("unreachable", stats), stats)

    # ---- stage 1: S(t) ∪ B_i
    if planner.decides_reachability:
        r1 = planner.plan(query)
        stats = stats.merged(r1.stats)
        if r1.status != "unreachable":
            return finish(r1, stats)
    else:
        stats = replace(stats, guard_bfs=stats.guard_bfs + 1)
        if reachable(c, start, goal):
            r1 = planner.plan(query)
            return finish(r1, stats.merged(r1.stats))
    if not c.dynamic_blocked:  # stage 2 would be identical: statically unreachable
        return finish(failure("unreachable", stats), stats)

    # ---- stage 2: static fallback, same density
    q2 = replace(query, constraints=c.without_dynamic())
    if planner.decides_reachability:
        r2 = planner.plan(q2)
        stats = stats.merged(r2.stats)
        return finish(r2, stats, degraded=True)
    stats = replace(stats, guard_bfs=stats.guard_bfs + 1)
    ok = (
        static_reachable(start, goal)
        if static_reachable is not None
        else reachable(q2.constraints, start, goal)
    )
    if not ok:
        return finish(failure("unreachable", stats), stats)
    r2 = planner.plan(q2)
    return finish(r2, stats.merged(r2.stats), degraded=True)
