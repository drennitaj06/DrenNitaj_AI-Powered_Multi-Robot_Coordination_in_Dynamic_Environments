"""RRT* in the continuous plane over the occupancy map (algorithms §3; Karaman & Frazzoli 2011).

Guaranteed (G5): every returned path is valid for the query snapshot (every tree edge passed the
supercover test and discretisation preserves freeness and 4-adjacency); the best goal cost is
non-increasing over iterations; the tree-cost invariant ``cost[q] = cost[parent] + c(parent, q)``
holds after every iteration (eager propagation).

Connection radius. The implemented near radius is the Karaman–Frazzoli form
``r_n = min(γ (ln n / n)^(1/d), η)`` with ``d = 2`` and ``γ = γ_f γ*`` (``γ_f ≥ 1``). Solovey et al.
(arXiv:1909.09688, ICRA 2020) found a logical gap in the original optimality proof and proved
asymptotic optimality for the corrected radius ``γ (log n / n)^(1/(d+1))`` (exponent reduced from
``1/d`` to ``1/(d+1)``, with a lower bound on ``γ``). The implemented exponent ``1/d`` does not meet
that condition for any constant, so the corrected guarantee is **not established** here.

Not claimed: asymptotic optimality for the implemented radius (G7; convergence is measured
empirically in E1c); success within a finite budget (G8); ``O(n log n)`` running time (G9).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from robocoord.core.errors import ConfigError
from robocoord.core.types import Cell
from robocoord.planning.base import PlanQuery, PlanResult, PlanStats, failure
from robocoord.planning.geometry import discretise, remove_loops, segment_cost
from robocoord.planning.neighbors import make_index


@dataclass(frozen=True, slots=True)
class RRTStarConfig:
    """RRT* parameters (``04`` §3.1).

    * ``max_iterations`` — iteration budget ``n_max`` (RRT* is anytime: all iterations are run).
    * ``eta`` — steer step ``η`` in cells; also the cap of the near radius ``r_n`` (default 2.0,
      chosen by a documented pilot, see ``04`` §3 implementation notes).
    * ``goal_bias`` — probability ``p_g`` of sampling uniformly inside the goal cell.
    * ``gamma_factor`` — ``γ_f`` in ``γ = γ_f γ*`` of the Karaman–Frazzoli radius (default 1.5,
      Issue #3 decision; ``γ_f ≥ 1`` keeps ``γ`` above their threshold ``γ*``). No value of ``γ_f``
      turns this radius into the corrected radius of Solovey et al. (see the module docstring).
    * ``neighbor_index`` — ``"brute"`` (default, chosen from E1b: faster at every measured
      budget up to 10 000 iterations) or ``"kdtree"`` (Bentley–Saxe logarithmic method).
    * ``time_limit_s`` — optional wall-clock budget; hitting it sets ``stats.timed_out``.
    * ``record_trace`` — keep the best cost and the work counters of every iteration.
    * ``check_invariants`` — verify the tree-cost invariant after every iteration (tests; O(n)).
    * ``stop_on_first_solution`` — stop as soon as the goal region is reached (RRT-like, no
      further improvement). Off by default; used where only feasibility matters (test X-11).
    """

    max_iterations: int = 2000
    eta: float = 2.0
    goal_bias: float = 0.1
    gamma_factor: float = 1.5
    neighbor_index: str = "brute"
    leaf_size: int = 8
    time_limit_s: float | None = None
    record_trace: bool = False
    check_invariants: bool = False
    stop_on_first_solution: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ConfigError("rrtstar.max_iterations", "must be >= 0")
        if not self.eta > 0:
            raise ConfigError("rrtstar.eta", "must be > 0")
        if not 0.0 <= self.goal_bias <= 1.0:
            raise ConfigError("rrtstar.goal_bias", "must be in [0, 1]")
        if self.gamma_factor < 1.0:
            raise ConfigError(
                "rrtstar.gamma_factor", "must be >= 1 (gamma above the Karaman–Frazzoli threshold)"
            )
        if self.neighbor_index not in ("kdtree", "brute"):
            raise ConfigError(
                "rrtstar.neighbor_index",
                f"unknown index {self.neighbor_index!r}; expected 'kdtree' or 'brute'",
            )


def gamma_star(free_area: float) -> float:
    """Karaman–Frazzoli threshold ``γ* = 2 (1 + 1/d)^{1/d} (μ(X_free)/ζ_d)^{1/d}`` for ``d = 2``
    (``ζ_2 = π``). It belongs to the original ``(log n / n)^{1/d}`` radius, not to the corrected
    radius of Solovey et al."""
    return 2.0 * math.sqrt(1.5) * math.sqrt(free_area / math.pi)


class TreeInvariantError(AssertionError):
    """The tree-cost invariant was violated (only raised with ``check_invariants``)."""


TRACE_FIELDS = (
    "n",
    "k",
    "kprime",
    "nn_visited",
    "nn_nodes",
    "range_nodes",
    "range_tested",
    "s",
    "rebuild",
    "rebuild_work",
    "checks",
    "added",
)


class RRTStarPlanner:
    """Single-stage RRT*; call it through the shared procedure (which runs the BFS guard)."""

    name = "rrtstar"
    decides_reachability = False

    def __init__(self, config: RRTStarConfig | None = None) -> None:
        self.config = config or RRTStarConfig()

    def plan(self, query: PlanQuery) -> PlanResult:
        cfg = self.config
        t0 = time.perf_counter()
        c = query.constraints
        w, h = c.width, c.height
        start, goal = query.start, query.goal
        if not (c.static.in_bounds(start) and c.static.in_bounds(goal)):
            return failure("invalid_query", PlanStats(planner=self.name))
        if start == goal:
            return PlanResult(
                (start,),
                0.0,
                "ok",
                stats=PlanStats(planner=self.name, time_s=time.perf_counter() - t0),
                waypoints=((start[0] + 0.5, start[1] + 0.5),),
            )
        rng = query.rng if query.rng is not None else np.random.default_rng(0)
        free = c.free_flat
        penal = c.penalised_flat
        heavy = 1.0 + c.lambda_prox
        gx, gy = goal
        eta = cfg.eta
        gamma = cfg.gamma_factor * gamma_star(max(c.free_area(), 1))

        index = make_index(cfg.neighbor_index, cfg.leaf_size)
        xs = [start[0] + 0.5]
        ys = [start[1] + 0.5]
        parent = [-1]
        cost = [0.0]
        ecost = [0.0]  # cost of the edge parent → node
        children: list[list[int]] = [[]]
        is_goal = [False]
        index.insert(xs[0], ys[0])

        best, best_node = math.inf, -1
        first_solution = -1
        nn_queries = radius_queries = near_k = near_kprime = 0
        rewires = propagation = checks = cells = 0
        trace_best: list[float] | None = [] if cfg.record_trace else None
        trace_cols: dict[str, list[int]] | None = (
            {f: [] for f in TRACE_FIELDS} if cfg.record_trace else None
        )
        timed_out = False
        it = 0
        for it in range(1, cfg.max_iterations + 1):
            if cfg.time_limit_s is not None and time.perf_counter() - t0 > cfg.time_limit_s:
                timed_out = True
                it -= 1
                break
            nn0, rn0, rb0 = index.nn_visited, index.range_nodes, index.rebuild_points
            nd0, rt0, rw0 = index.nn_nodes, index.range_nodes_tested, index.rebuild_work
            kp0, chk0 = index.range_points, checks
            k = s = 0
            added = 0
            n = len(xs)
            # ---- sample, nearest, steer
            if rng.random() < cfg.goal_bias:
                qx, qy = gx + rng.random(), gy + rng.random()
            else:
                qx, qy = rng.random() * w, rng.random() * h
            near = index.nearest(qx, qy)
            nn_queries += 1
            ddx, ddy = qx - xs[near], qy - ys[near]
            d = math.hypot(ddx, ddy)
            if d > 0.0:
                if d > eta:
                    qx, qy = xs[near] + ddx * (eta / d), ys[near] + ddy * (eta / d)
                c_near, nc = segment_cost((xs[near], ys[near]), (qx, qy), free, w, h, penal, heavy)
                checks += 1
                cells += nc
                if c_near != math.inf:
                    # ---- near set
                    r = eta if n < 2 else min(gamma * math.sqrt(math.log(n) / n), eta)
                    near_ids = index.radius(qx, qy, r).tolist()
                    radius_queries += 1
                    k = len(near_ids)
                    near_k += k
                    edge: dict[int, float] = {near: c_near}
                    for q in near_ids:
                        if q not in edge:
                            e, nc = segment_cost((xs[q], ys[q]), (qx, qy), free, w, h, penal, heavy)
                            checks += 1
                            cells += nc
                            edge[q] = e
                    # ---- choose parent (ties: lowest index)
                    q_min, c_min = -1, math.inf
                    for q in sorted(edge):
                        v = cost[q] + edge[q]
                        if v < c_min:
                            q_min, c_min = q, v
                    new = n
                    xs.append(qx)
                    ys.append(qy)
                    parent.append(q_min)
                    ecost.append(edge[q_min])
                    cost.append(c_min)
                    children.append([])
                    children[q_min].append(new)
                    in_goal = math.floor(qx) == gx and math.floor(qy) == gy
                    is_goal.append(in_goal)
                    index.insert(qx, qy)
                    added = 1
                    if in_goal and c_min < best:
                        best, best_node = c_min, new
                    # ---- rewire
                    for q in near_ids:
                        if q == q_min:
                            continue
                        e = edge[q]
                        if e == math.inf:
                            continue
                        cand = c_min + e
                        if cand < cost[q]:
                            children[parent[q]].remove(q)
                            parent[q] = new
                            ecost[q] = e
                            children[new].append(q)
                            rewires += 1
                            stack = [q]
                            while stack:  # eager propagation down the subtree
                                u = stack.pop()
                                cost[u] = cost[parent[u]] + ecost[u]
                                s += 1
                                if is_goal[u] and cost[u] < best:
                                    best, best_node = cost[u], u
                                stack.extend(children[u])
                    propagation += s
            if best_node >= 0 and first_solution < 0:
                first_solution = it
            if trace_best is not None and trace_cols is not None:
                trace_best.append(best)
                kp = index.range_points - kp0
                near_kprime += kp
                for name, val in (
                    ("n", n),
                    ("k", k),
                    ("kprime", kp),
                    ("nn_visited", index.nn_visited - nn0),
                    ("nn_nodes", index.nn_nodes - nd0),
                    ("range_nodes", index.range_nodes - rn0),
                    ("range_tested", index.range_nodes_tested - rt0),
                    ("s", s),
                    ("rebuild", index.rebuild_points - rb0),
                    ("rebuild_work", index.rebuild_work - rw0),
                    ("checks", checks - chk0),
                    ("added", added),
                ):
                    trace_cols[name].append(val)
            else:
                near_kprime += index.range_points - kp0
            if cfg.check_invariants:
                self._check_tree(xs, ys, parent, cost, ecost, free, w, h, penal, heavy)
            if cfg.stop_on_first_solution and best_node >= 0:
                break

        stats = PlanStats(
            planner=self.name,
            iterations=it,
            tree_size=len(xs),
            nn_queries=nn_queries,
            nn_visited=index.nn_visited,
            nn_nodes=index.nn_nodes,
            radius_queries=radius_queries,
            near_k=near_k,
            near_kprime=near_kprime,
            range_nodes=index.range_nodes,
            range_nodes_tested=index.range_nodes_tested,
            rewires=rewires,
            propagation=propagation,
            rebuild_points=index.rebuild_points,
            rebuild_work=index.rebuild_work,
            segment_checks=checks,
            segment_cells=cells,
            timed_out=timed_out,
            first_solution_iteration=first_solution,
            best_cost_trace=tuple(trace_best) if trace_best is not None else None,
            iteration_trace=(
                {k2: np.array(v, dtype=np.int64) for k2, v in trace_cols.items()}
                if trace_cols is not None
                else None
            ),
        )
        if best_node < 0:
            return failure("budget_exhausted", replace(stats, time_s=time.perf_counter() - t0))
        chain: list[int] = []
        u = best_node
        while u >= 0:
            chain.append(u)
            u = parent[u]
        chain.reverse()
        points = tuple((xs[i], ys[i]) for i in chain)
        # executable cell path: discretised polyline with revisits cut out (stays valid, `04` §3.3)
        path: tuple[Cell, ...] = tuple(remove_loops(discretise(points)))
        stats = replace(stats, time_s=time.perf_counter() - t0)
        return PlanResult(path, best, "ok", stats=stats, waypoints=points)

    @staticmethod
    def _check_tree(
        xs: list[float],
        ys: list[float],
        parent: list[int],
        cost: list[float],
        ecost: list[float],
        free: list[bool],
        w: int,
        h: int,
        penal: frozenset[int],
        heavy: float,
    ) -> None:
        if cost[0] != 0.0 or parent[0] != -1:
            raise TreeInvariantError("root must have cost 0 and no parent")
        for q in range(1, len(xs)):
            p = parent[q]
            e, _ = segment_cost((xs[p], ys[p]), (xs[q], ys[q]), free, w, h, penal, heavy)
            if e != ecost[q] or cost[q] != cost[p] + e:
                raise TreeInvariantError(f"node {q}: cost {cost[q]} != {cost[p]} + {e}")
