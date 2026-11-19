"""A* on the 4-connected grid with proximity-weighted cell costs (algorithms §2.3).

Guaranteed (G4): with the consistent Manhattan heuristic the returned path is a minimum
``κ_i``-cost 4-connected lattice path on the query snapshot, and ``unreachable`` is returned iff
no path exists in the constrained graph.
"""

from __future__ import annotations

import heapq
import math
import time

from robocoord.core.types import Cell
from robocoord.planning.base import PlanQuery, PlanResult, PlanStats, failure


class AStarPlanner:
    """Single-stage A*; call it through :func:`~robocoord.planning.procedure.plan_with_procedure`.

    Data structures (``04`` §2.3): flattened node ids ``y*W + x``; a binary heap of
    ``(f, −g, counter, node)`` with lazy deletion; ``g``, parent and closed arrays of size ``V``.
    The arrays are Python lists rather than NumPy arrays: the hot loop indexes single elements,
    which is several times faster on lists (same O(1) access and O(V) initialisation).

    Time: O(V log V) worst case (≤ |E| ≤ 2V pushes); O(V) initialisation. Space: O(V).
    """

    name = "astar"
    decides_reachability = True

    def plan(self, query: PlanQuery) -> PlanResult:
        t0 = time.perf_counter()
        c = query.constraints
        w, h = c.width, c.height
        (sx, sy), (gx, gy) = query.start, query.goal
        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return failure("invalid_query", PlanStats(planner=self.name))
        free = c.free_flat
        s, g = sy * w + sx, gy * w + gx
        if not free[s] or not free[g]:
            return failure(
                "unreachable", PlanStats(planner=self.name, time_s=time.perf_counter() - t0)
            )
        if s == g:
            return PlanResult(
                (query.start,),
                0.0,
                "ok",
                stats=PlanStats(planner=self.name, time_s=time.perf_counter() - t0),
            )
        penal = c.penalised_flat
        heavy = 1.0 + c.lambda_prox
        n = w * h
        inf = math.inf
        gcost = [inf] * n
        parent = [-1] * n
        closed = bytearray(n)
        gcost[s] = 0.0
        counter = 0
        open_: list[tuple[float, float, int, int]] = [
            (float(abs(sx - gx) + abs(sy - gy)), -0.0, 0, s)
        ]
        pushes, expansions, peak = 1, 0, 1
        found = False
        while open_:
            _, neg_g, _, u = heapq.heappop(open_)
            gu = -neg_g
            if closed[u] or gu > gcost[u]:
                continue  # stale entry (lazy deletion)
            if u == g:
                found = True
                break
            closed[u] = 1
            expansions += 1
            ux, uy = u % w, u // w
            # canonical neighbour order: UP, DOWN, LEFT, RIGHT (deterministic)
            for v, vx, vy, inside in (
                (u + w, ux, uy + 1, uy + 1 < h),
                (u - w, ux, uy - 1, uy > 0),
                (u - 1, ux - 1, uy, ux > 0),
                (u + 1, ux + 1, uy, ux + 1 < w),
            ):
                if not inside or not free[v] or closed[v]:
                    continue
                ng = gu + (heavy if v in penal else 1.0)
                if ng < gcost[v]:
                    gcost[v] = ng
                    parent[v] = u
                    counter += 1
                    heapq.heappush(open_, (ng + abs(vx - gx) + abs(vy - gy), -ng, counter, v))
                    pushes += 1
                    if len(open_) > peak:
                        peak = len(open_)
        stats = PlanStats(
            planner=self.name,
            expansions=expansions,
            pushes=pushes,
            peak_open=peak,
            time_s=time.perf_counter() - t0,
        )
        if not found:
            return failure("unreachable", stats)
        path: list[Cell] = []
        v = g
        while v != -1:
            path.append((v % w, v // w))
            v = parent[v]
        path.reverse()
        return PlanResult(tuple(path), gcost[g], "ok", stats=stats)
