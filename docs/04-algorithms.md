# 04 — Algorithm Design and Complexity Analysis

| Field | Value |
|---|---|
| Document role | **Source of truth for algorithm design**: purpose, I/O, data structures, pseudocode, correctness arguments, theoretical complexity, edge cases and the tests of every algorithm. |
| Status | Revision 2 (Issue #1, after design review). Each section gets an **Implementation notes** subsection in the issue that implements it, recording deviations, measured behaviour and limitations. |
| Notation | As in `02-mathematical-model.md`. $V = WH$ (number of cells), $N$ robots, $K$ dynamic obstacles, $m$ tasks. |

**Complexity conventions.**
- Hash-map operations are $O(1)$ *expected*. Binary-heap operations are $O(\log n)$ worst case.
- **Worst case**, **expected** (with the probability model stated) and **amortised** bounds are always labelled.
- Because the grid is 4-connected, $|E| \le 2V$ and $O(|V| + |E|) = O(V)$.
- *Theoretical* bounds are statements about operation counts. *Measured* runtime and empirical scaling are reported separately in `05-testing-and-evaluation.md` and are never substituted for each other.

**Contents.** §1 Conflict detection and arbitration · §2 Grid search: BFS, shared planning constraints, A\* · §3 RRT\* · §4 Auction allocation · §5 Execution: priority handling, path monitor, controllers, shield · §6 MADQL · §7 PPO · §8 Observations and rewards · §9 Complexity summary.

---

## 1. Conflict detection and arbitration (simulator core)

**Purpose.** Apply a synchronous joint proposal so that the physical invariants hold (model §3, Lemma 3.1), and produce the exact conflict-event sets needed by rewards and metrics (model §7, §9.2). Two pure functions:
- `detect_conflicts(state, proposal) -> set[ConflictEvent]`. It is also used for the counterfactual reference proposal, so controller and reference conflicts are computed by *identical* code.
- `arbitrate(state, proposal, events) -> Resolution` (new positions, frozen set, blocked moves).

**Data structures.**
- `target: dict[Cell, list[int]]`: proposed cell → robots that want it.
- `occ: dict[Cell, int]`: robot occupying each cell at time $t$.
- `obs_now`, `obs_next: dict[Cell, int]`: obstacle positions at $t$ and $t+1$ (resolved).
- A FIFO worklist of newly frozen robots.

**Algorithm.**

```text
DETECT(x, x̂, z, z′, grid):
  E ← ∅
  for i in robots:
      if x̂_i not free in grid:                      E += (t, INV, {i})
      if a_i ≠ stay and x̂_i ∈ obs_next:            E += (t, ROv, {i, obs_next[x̂_i]})
      if x̂_i ∈ obs_now and z′[obs_now[x̂_i]] = x_i:  E += (t, ROs, {i, obs_now[x̂_i]})
  for each cell c with |target[c]| ≥ 2:             E += (t, RRv, {i, j}) for each pair in target[c]
  for i with x̂_i ≠ x_i:
      j ← occ.get(x̂_i);  if j exists and x̂_j = x_i and i < j:  E += (t, RRs, {i, j})
  return E

ARBITRATE(x, x̂, E):
  frozen ← all robots appearing in E;  worklist ← frozen (index order)
  while worklist not empty:
      j ← pop(worklist)
      for i in target[x_j] with i ∉ frozen:          # i wants the cell j now keeps
          BLOCKED += i; frozen += i; push(worklist, i)
  x_i(t+1) ← x_i if i ∈ frozen else x̂_i
  assert NO_PHYSICAL_COLLISION(x(t+1), z′)          # invariant, always checked
```

Note on `target` for staying robots: a robot with $a_i = \textsf{stay}$ is listed in `target[x_i]`. Hence a mover into its cell creates an $\textsf{RRv}$ event, as the model requires.

**Correctness (Lemma 3.1: no physical collision).** The frozen set only grows and each robot is pushed at most once, so the loop terminates. At termination, take two robots $i \ne j$:
- Both frozen: they occupy $x_i(t) \ne x_j(t)$, which were distinct by the invariant at time $t$.
- $i$ not frozen and $j$ frozen: if $\hat x_i = x_j(t)$, then $i \in$ `target`$[x_j]$, and processing $j$ would have frozen $i$. Contradiction.
- Both not frozen: $\hat x_i = \hat x_j$ is an $\textsf{RRv}$ event, which freezes both. Contradiction.
- Swaps are $\textsf{RRs}$ events and frozen.
- Robot–obstacle: a non-frozen robot entering $z'$ would be an $\textsf{ROv}$ event. A frozen robot stays at $x_i(t)$, which obstacles never enter (obstacle move rule). Passing through an obstacle is an $\textsf{ROs}$ event.
- Static cells: an invalid target is an $\textsf{INV}$ event, and frozen robots stay on free cells.

**Complexity.** Building `target` and `occ`: $O(N)$. Detection: $O(N + K)$ expected, plus the vertex-pair enumeration. That is $O(\sum_c \binom{|target[c]|}{2})$, at most $O(N)$ pairs in practice since at most 5 robots can propose one cell, so $O(N + K)$ expected overall. Arbitration: each robot enters the worklist at most once and appears in exactly one `target` list, so total $O(N)$. **Total $O(N + K)$ expected time and space per step.** The fixed-point formula in the model would cost $O(N^2)$ worst case if iterated literally; the worklist makes it linear. With the counterfactual detection for metrics, the cost is two detections per step, still $O(N + K)$.

**Edge cases.** All robots stay. A chain of followers behind a frozen head (all blocked, one pass). A rotation cycle of length ≥ 3 (allowed, no events). A two-robot swap. Three robots targeting one cell (three pair events). A robot staying while others move around it. A robot adjacent to an obstacle that moves away while the robot enters (a follow move, allowed).

**Tests.**
- Hand-built cases for every event type.
- **Property test** (Hypothesis, ≥ 500 cases): random grids, positions, obstacle moves and proposals. Check (a) no physical collision; (b) every non-frozen robot moved exactly to $\hat x_i$; (c) the event set equals that of an independent naive $O(N^2)$ reference implementation written in the test module; (d) blocked robots have no event of their own.
- Counterfactual: `detect_conflicts` on the reference proposal is deterministic, and the avoided/introduced set differences match hand-computed cases.

---

**Implementation notes (Issue #2).**
- Implemented in `src/robocoord/sim/conflicts.py` as designed (`detect_conflicts`, `arbitrate`, `physical_violations`).
- The engine checks the physical-collision invariant after every step and raises `PhysicalCollisionError` on violation (never observed in tests).
- The property test compares detection and arbitration against an independent naive O(N²) oracle on 600 generated situations (grids up to 6×6, up to 8 robots and 4 obstacles, unmasked random proposals, obstacle moves from the real move rule).
- Obstacle move rule: proposals are also rejected if an earlier obstacle already claimed the cell in the same step. This makes the fixed processing order explicit; model §2 is updated.
- **Path efficiency** is computed per completed *leg*: Σ optimal static distance at leg start / Σ moves made during the leg. Legs aborted by unassignment are excluded.
- Metrics added during the final review: `blocked_goal_robot_steps` (robots holding a task whose goal is statically unreachable, evaluated on the map after the step's map events), per-100-robot-step rates for blocked moves and for each conflict kind. The `physical_collisions` counter is 0 by construction because the engine raises on any violation.
- The scripted demo controller retries a statically feasible move that the arbiter cancelled, so scripted robots wait instead of skipping route steps. Symmetric head-on situations remain deadlocks for scripted robots, which is expected: no planner or learning is involved yet.

---

## 2. Grid search

### 2.1 BFS distance field

**Purpose.** Exact static distances $D(\cdot, v)$ for auction costs, reward potentials, path efficiency, reachability guards and test oracles.

**Algorithm.** BFS from $v$ over free cells with `collections.deque`. Distances go into an `int32` NumPy array, with −1 meaning unreachable.

**Correctness.** BFS visits vertices in non-decreasing distance order on unit-weight graphs.

**Complexity.** $O(V)$ time and space, worst case.

**Caching.** The `DistanceOracle` caches fields keyed by `(target, grid.version)` with LRU eviction (default 256 fields, ≈ 4 MB for 64×64).

**Tests.** Floyd–Warshall reference on small grids; unreachable components; cache invalidation after map events.

**Implementation notes (Issue #2).**
- `src/robocoord/core/distance.py`. Fields are cached read-only per `(target, grid.version)`.
- `capped_distance` implements $\bar D$.
- `connected_components` (also O(V)) is used by the map generators, which keep only the largest free component so that generated maps are connected (property-tested).
- The Floyd–Warshall comparison is a Hypothesis property test (200 random grids up to 6×6).

**Implementation notes (Project Issue #4, NFR-PERF-1 follow-up: O1, O2, symmetric reuse).**
- *O1, flat BFS.* Cells are indexed $i = yW + x$ and the traversal runs on Python lists instead of per-cell NumPy indexing. The oracle caches, per grid version, the list of free 4-neighbours of every free cell, so a field costs one pass over that list. The output is the same `int32` `[y, x]` array. BFS distances do not depend on the expansion order of neighbours, so the fields are identical; the earlier implementation is kept as `bfs_distance_field_legacy`.
- *O2, component reachability.* On the undirected 4-connected grid, $D_t(a,b) < \infty$ iff both cells are free and in the same connected component. `DistanceOracle.reachable` answers from component labels computed once per grid version ($O(V)$), not from a distance field per target. This serves task availability, the engine's per-step goal check, the planners' static-reachability guard, arrival sampling and scenario validation.
- *Symmetric reuse.* $D_t(a,b) = D_t(b,a)$. `distance(a, b)` reuses a cached field to $a$ when none to $b$ is cached, and the auction benefits read the leg $D(p_j, d_j)$ as $D(d_j, p_j)$, from the field to $p_j$ that the terms $D(x_i, p_j)$ need anyway: one field per task instead of two on a cold cache.
- *Invalidation.* Fields, neighbour lists and labels are all keyed by the grid version, which every map event changes, so nothing computed for an older map is ever used.
- *Not done:* no work was moved into engine construction or out of the timed step, and O3 (incremental field repair after a single-cell map event) was not implemented; see `05` §B.6 NFR-PERF-1 for why.
- *Equivalence tests* (`tests/unit/core/test_distance_equivalence.py`): flat BFS = legacy BFS for every target (blocked ones included) and flat labels = legacy labels on 80 random grids up to 9×9; fast and legacy oracles agree on fields, distances and reachability before and after random block/unblock sequences (50 grids); the `e5_scenario` maps and task fields are identical under both. `DistanceOracle(grid, legacy=True)` / `legacy_distance()` reproduces the earlier behaviour exactly and exists only for these tests and the paired before/after benchmark.

### 2.2 Shared planning constraints and procedure

One function, `build_constraints(world, robot) -> PlanConstraints`, computes $Z_i$, $\hat Z_i$, $B_i$ (goal and start removed), $Q_i$ and the density $\omega_i$ exactly as in model §5.2. When no obstacle is sensed, $Q_i = \varnothing$, $B_i = \varnothing$ and $\omega_i \equiv 1$. Both planners receive the same `PlanConstraints` object.

The two-stage procedure (constrained, then static fallback, then status mapping) of model §5.3 is implemented once, in `planning/procedure.py`, around any `Planner`. A\* and RRT\* therefore cannot diverge in goal or obstacle policy.

**Complexity.**
- Obstacle positions are indexed by cell, so $Z_i$ costs $O(|Z_i|)$ via a window scan of $O(r_s^2)$ cells.
- $Q_i$ costs $O(|Z_i|\, r_{\text{infl}}^2)$.
- The constrained-reachability BFS costs $O(V)$. It runs once per query; for A\* it is skipped because A\* detects unreachability itself.

**Tests.**
- Empty sensed set gives an all-ones density.
- The goal and start are never in $B_i$.
- An obstacle on the goal leaves the goal plannable.
- Dynamic blocking of a chokepoint triggers the stage-2 fallback with `degraded = true`.
- Static blocking of the goal returns `unreachable`.
- **Consistency test X-11:** on identical queries, A\* and RRT\* (generous budget) return the same status and the same `degraded` flag on ≥ 300 random queries with obstacles near or on goals and chokepoints.

**Implementation notes (Issue #3).**
- `planning/constraints.py`: `build_constraints(grid, start, goal, obstacles, params, extra_blocked)` is the only place where $Z_i$, $\hat Z_i$, $B_i$, $Q_i$ and $\omega_i$ are computed. `obstacles` maps each obstacle cell to its velocity. The sensing scan visits the $(2r_s+1)^2$ window or the obstacle list, whichever is smaller. `extra_blocked` carries the monitor's stall block list (T4); it is part of $B_i$, so the start and goal are removed from it and the stage-2 fallback drops it.
- `planning/procedure.py`: `plan_with_procedure(planner, query, static_reachable=...)`. A start inside $S(t)$ is reported as `invalid_query`, like an out-of-bounds cell. It cannot occur in the simulator (map events are deferred while a cell is occupied), but random test queries can produce it. When $B_i = \varnothing$ the procedure skips stage 2, which would be identical, and reports `unreachable` directly. `degraded` is only ever `True` together with `ok`.
- Planners declare `decides_reachability`. A\* (`True`) runs stage 1 and proves infeasibility itself. RRT\* (`False`) runs only after a BFS guard on the constrained grid, so its iteration budget is never spent on an unreachable goal (tested: `stats.iterations == 0`). The engine passes its cached `DistanceOracle.reachable` as the static guard.
- `PlanConstraints` caches flat row-major views (`free_flat`, `penalised_flat`) once per query, O(V). Both planners read the same views.
- Registry: `planning.make_planner("astar" | "rrtstar", **options)`. Unknown names or options raise `ConfigError` listing the valid choices.
- **X-11 result:** on 318 random queries (random maps with obstacles on and near goals; rooms maps with an obstacle on a door chokepoint; empty sensed sets; goals inside $S(t)$), A\* and RRT\* returned identical `status` and `degraded` for every query (318/318). Outcome mix: 268 `ok`, 27 `ok` + degraded, 23 `unreachable` (`tests/unit/planning/test_consistency.py`). The generous RRT\* budget there uses `stop_on_first_solution`, because only the status matters.

### 2.3 A\* search

**Purpose.** A minimum-$\kappa_i$-cost path on the 4-connected grid for one query snapshot.

**Input.** Start, goal, `PlanConstraints` (stage 1 or 2). **Output.** Path, cost, status, stats (expanded nodes, pushes, peak open size, time).

**Data structures and justification.**

| Structure | Implementation | Reason |
|---|---|---|
| Node ids | Flattened index `y*W + x` | Constant-time array indexing in the hot loop |
| Open list | Binary heap (`heapq`) of `(f, -g, counter, node)` | $O(\log n)$ push/pop. Ties go to larger $g$ (fewer expansions on equal-$f$ plateaus). `counter` makes ordering total and deterministic. |
| Decrease-key | **Lazy deletion**: duplicates pushed, stale entries skipped on pop | `heapq` has no decrease-key. Total pushes ≤ $\lvert E\rvert$, so asymptotically equivalent. |
| $g$, parent, closed | NumPy arrays of size $V$ (`float64`, `int32`, `bool`) | $O(1)$ access |

Array initialisation is $O(V)$ per call. A version-stamp scheme avoids it if profiling shows it matters.

**Pseudocode.**

```text
A*(start, goal, constraints):
  if start = goal: return [start], 0
  g[·] ← ∞; g[start] ← 0; push(open, (h(start), 0, 0, start))
  while open not empty:
      (f, −g_u, _, u) ← pop(open)
      if closed[u] or g_u > g[u]: continue               # stale entry
      if u = goal: return RECONSTRUCT(goal), g[goal]
      closed[u] ← true
      for v in N4(u) with v ∉ S(t) ∪ B_i:
          g' ← g[u] + κ_i(v)
          if g' < g[v]: g[v] ← g'; parent[v] ← u; push(open, (g' + h(v), −g', counter++, v))
  return UNREACHABLE
```

**Heuristic.** $h(v) = \|v - g\|_1$. *Admissible*: each step changes Manhattan distance by exactly 1 and costs ≥ 1. *Consistent*: $h(u) \le 1 + h(v) \le \kappa_i(v) + h(v)$ for every edge.

**Correctness (Guaranteed).** With a consistent heuristic, a node's $g$ is optimal when it is first popped (non-stale), so the closed set is safe and the first goal pop yields the optimal cost (Hart, Nilsson & Raphael, 1968). On a finite graph A\* terminates, and it returns UNREACHABLE iff no path exists in the constrained graph. **Scope:** optimality holds for the snapshot graph and cost $\kappa_i$ among 4-connected lattice paths. It does not hold for the executed trajectory (model §5.5).

**Complexity.**
- *Worst case:* each node expanded at most once and each edge relaxed at most once, so ≤ $\lvert E\rvert \le 2V$ pushes. Time $O(V \log V)$, space $O(V)$. This is reached for unreachable goals (the whole component is explored) and long detours.
- *Best case:* perfect guidance with tie-breaking to larger $g$ expands only nodes on one optimal path: $O(L \log L)$, plus $O(V)$ initialisation.
- *Typical:* depends on heuristic accuracy. Obstacles weaken Manhattan as a lower bound. E1 measures expansions vs density and map family. This is an empirical question, not a theorem.

**Edge cases.** start = goal; start or goal out of bounds (`invalid_query`); goal in $S(t)$ after a map event (`unreachable`, model §5.3 step 4); enclosed goal; 1×1 and 1×N grids; obstacle-free grid; equal-cost alternatives (deterministic choice); proximity penalties (optimal for $\kappa_i$, not shortest in steps).

**Tests.**
1. **Optimality property test:** ≥ 1 000 random grids. With $\omega \equiv 1$, A\* cost = BFS distance. With penalties, A\* cost = cost from an independent $O(V^2)$ Dijkstra in the test.
2. Validity: 4-adjacency, freeness w.r.t. $S \cup B_i$, endpoints, cost = $\sum \kappa_i$.
3. Statuses for unreachable and invalid queries.
4. Determinism.
5. Expanded nodes ≤ reachable cells.

**Implementation notes (Issue #3).**
- `planning/astar.py`. **Deviation from the table above:** the $g$, parent and closed arrays are Python lists and a `bytearray` instead of NumPy arrays. The hot loop indexes single elements, which is several times faster on lists, with the same O(1) access and O(V) initialisation. The asymptotic analysis is unchanged.
- The canonical neighbour order (UP, DOWN, LEFT, RIGHT), the heap key `(f, −g, counter, node)` and lazy deletion are as specified. In open space the tie-breaking towards larger $g$ expands only the nodes on one optimal path (tested: 12 expansions for a 12-step query on an empty 7×7 grid).
- **Oracle test result:** A\* matched an independent BFS on 600 unit-cost instances and an independent $O(V^2)$ array-scan Dijkstra on 600 proximity-weighted instances (1 200 random instances, grids up to 12×12, densities up to 45 %, random sensed obstacles, $\lambda \in \{0, 0.5, 2\}$). Every returned path was valid, its lattice cost equalled the reported cost, and expansions never exceeded the number of free cells.
- **Measured behaviour (E1a, `05` §B.6):** 2 800/2 800 static queries solved, median 0.2–3.2 ms per query. Expansions grow sub-linearly in $V$ (exponents 0.49–0.84 by family; bound: expansions ≤ $V$). Expansions per free cell rise super-linearly with obstacle density (128×128: 0.013 at 10 % to 0.111 at 35 %). The worst case (whole component explored) occurs only for unreachable goals, which E1a does not contain; the procedure tests cover it.

---

## 3. RRT\* (continuous sampling-based planner)

### 3.1 Purpose, input, output

**Purpose.** The second selectable planner: the RRT\* algorithm (designed for asymptotically optimal planning) in the continuous plane over the occupancy map (model §5.2). §3.4–3.5 state what the implemented radius does and does not establish. **Reference.** Karaman & Frazzoli, *Sampling-based algorithms for optimal motion planning*, IJRR 2011.

**Input.** Start and goal cell, `PlanConstraints`, RNG, and the configuration:
- `max_iterations` $n_{\max}$;
- steer step $\eta$ (in cells);
- goal bias $p_g$;
- radius safety factor $\gamma_f$;
- `neighbor_index` ∈ {`kdtree`, `brute`};
- optional time budget.

**Output.** `PlanResult`: cell path, cost, status, and stats. The stats are iterations, tree size, best cost per iteration, and the work counters of §3.6.

### 3.2 Neighbour index: design decision

RRT\* issues one nearest-neighbour query, one radius query and one insertion per iteration.

**Rejected: fixed-bucket uniform spatial hash.** The original draft used buckets of side $b = \eta$. In a bounded domain the number of buckets is fixed, $\Theta(WH/\eta^2)$, so the mean bucket occupancy grows **linearly in $n$**. A radius query then scans $\Theta(n)$ points in expectation as well as in the worst case, and the claimed $O(\log n)$ query cost did not hold. Shrinking buckets with $r_n$ restores small radius queries under uniform density but makes nearest-neighbour ring searches expensive in sparsely explored regions. Hashing gives no clean bound either way.

**Adopted: two interchangeable implementations behind one `NeighborIndex` protocol** (`insert`, `nearest`, `radius`), both written in Python/NumPy:

1. **`BruteForceIndex`**: points in a growing NumPy array. Nearest and radius queries are one vectorised distance computation. Simple, exact, and the **test oracle** for the other index.
2. **`LogKDTreeIndex`**: the Bentley–Saxe *logarithmic method* (static-to-dynamic transformation; Bentley & Saxe, 1980) over our own **static balanced 2-d trees**:
   - **Static tree.** Built from a point set by recursive median splits that **alternate between the x and y axes** (median via `numpy.argpartition`, $O(m)$ per level). Leaves hold ≤ $b_\ell$ points (default 8). Depth is $O(\log m)$ and build time $O(m \log m)$. Nodes are stored in flat arrays.
   - **Dynamic structure.** Slots $T_0, T_1, \dots$ where slot $i$ is either empty or holds a static tree of exactly $2^i$ points (the binary representation of $n$).
   - **Insert (rebuilding strategy).** Let $i$ be the lowest empty slot. Collect the new point and all points of slots $0..i-1$ (exactly $2^i$ points), build one static tree in slot $i$, and empty slots $0..i-1$.
   - **Queries.** Query every non-empty slot ($\le \lfloor \log_2 n \rfloor + 1$ trees). The nearest-neighbour search passes the best distance found so far between trees for pruning. The radius search prunes subtrees whose bounding rectangle does not intersect the query disk.

**Which index is used.** The default is chosen in Issue #3 from measurements (E1 includes the index as a factor) and recorded in the implementation notes. Every reported RRT\* complexity statement names the index it applies to.

### 3.3 Collision checking and weighted segment cost

**Supercover traversal** (Amanatides & Woo, 1987) enumerates, in order, the cells a segment touches, with the parametric entry and exit values $t_{\text{in}}, t_{\text{out}}$ per cell.
- *Collision check:* every touched cell must be outside $S(t) \cup B_i$. At an exact corner crossing all touching cells are checked, which is conservative and gives Lemma 5.1.
- *Weighted cost:* $\text{cost}(\overline{qq'}) = \|q' - q\|_2 \sum_{\text{cells } c} (t_{\text{out}} - t_{\text{in}})\,\omega_i(c)$, the exact integral of the piecewise-constant density along the segment.
- Both are computed in the same traversal: $O(\ell + 1)$ for a segment spanning $\ell$ cells (Guaranteed).

**Discretisation (polyline → executable cell path).** Walk each segment's supercover in order. The walk advances one axis at a time, so consecutive cells are 4-adjacent. At an exact corner, both side cells were checked free and the x-step side cell is inserted (deterministic). Consecutive duplicates are removed. Every produced cell was collision-checked, so the cell path is valid for the snapshot (Guaranteed).

**Loop removal (Issue #3 review, V-6).** Joining segment traversals can revisit a cell, typically as an A→B→A detour around a polyline vertex near a cell corner. Before the fix this happened in 8 of 176 sampled RRT\* paths on 16–32 maps (about 2 wasted steps each). In the E1a re-run it affected 18–25 % of all successful RRT\* paths (more on large maps), saving 2.9–3.3 steps on average and at most 14 (`05` §B.6, E1a). The executable path therefore goes through `remove_loops`: when a cell is visited again, the cells between the two visits are dropped. The result keeps the endpoints, uses only cells of the original path (all collision-checked), keeps 4-adjacency (the join point is the same cell), visits each cell once, and has a lattice cost $\sum\kappa$ that cannot increase. It is therefore valid under the same rules (Guaranteed; property test plus a regression test on a real looping query, 18 → 16 steps). The planner's reported cost and `waypoints` still describe the polyline, which is unchanged.

### 3.4 Pseudocode

```text
RRT*(q_start, goal_cell, constraints, n_max):          # called by the shared procedure (§2.2),
                                                         # which has already established reachability
  V ← {q_start}; cost[q_start] ← 0; index.insert(q_start)
  for it = 1 … n_max:
      q_rand ← SAMPLE()                      # with prob p_g: uniform in goal cell; else uniform in [0,W)×[0,H)
      q_near ← index.nearest(q_rand)
      q_new  ← STEER(q_near, q_rand, η)      # at most η towards q_rand; skip if zero-length
      if not FREE(q_near, q_new): continue
      n ← |V|;  r_n ← min(γ (ln n / n)^{1/2}, η) if n ≥ 2 else η
      Q ← index.radius(q_new, r_n)
      q_min ← argmin over q ∈ Q ∪ {q_near} with FREE(q, q_new) of cost[q] + c(q, q_new)   # choose parent
      add q_new: parent q_min, cost[q_new] ← cost[q_min] + c(q_min, q_new); index.insert(q_new)
      for q in Q \ {q_min}:                                                                  # rewire
          c′ ← cost[q_new] + c(q_new, q)
          if c′ < cost[q] and FREE(q_new, q):
              δ ← cost[q] − c′; reparent q to q_new
              PROPAGATE(q, δ)                # subtract δ from cost of every node in subtree(q) (iterative DFS)
      if q_new ∈ goal_cell and cost[q_new] < best_cost: best ← q_new
  if best undefined: return BUDGET_EXHAUSTED
  return DISCRETISE(path(root → best))
```

`c(·,·)` is the weighted segment cost of §3.3. Ties in argmin are broken by node index. The tree is stored in parallel NumPy arrays (`xy`, `parent`, `cost`, geometric growth) with per-node child lists, so reparenting costs $O(\deg)$ for removal from the old parent's list.

**Connection radius: what is implemented and what it does (not) establish.**
- **Implemented: the Karaman–Frazzoli radius.** $r_n = \min\big(\gamma (\ln n / n)^{1/d}, \eta\big)$ with exponent $1/d = 1/2$, $\gamma = \gamma_f \gamma^\*$, $\gamma_f = 1.5$ (`RRTStarConfig.gamma_factor`, must be ≥ 1), and $\gamma^\* = 2(1 + 1/d)^{1/d}\big(\mu(X_{\text{free}})/\zeta_d\big)^{1/d}$, with $\zeta_2 = \pi$ and $\mu(X_{\text{free}})$ = free area in cells. This is the radius of Karaman & Frazzoli (IJRR 2011). Their Theorem 38 states almost-sure convergence to the optimum for a radius of the form $\gamma^{KF}(\log n / n)^{1/d}$ with $\gamma^{KF}$ above their threshold.
- **The correction (Solovey, Janson, Schmerling, Frazzoli, Pavone, *Revisiting the Asymptotic Optimality of RRT\**, arXiv:1909.09688v2, to appear in ICRA 2020).** They identify a logical gap in the original optimality proof. Their Theorem 1 proves asymptotic optimality for the radius $r(n) = \gamma(\log n / n)^{1/(d+1)}$, i.e. with the **exponent reduced from $1/d$ to $1/(d+1)$** (a larger radius for large $n$), and a constant $\gamma \ge (2+\theta)\big(\frac{(1+\varepsilon/4)\,c^\*}{(d+1)\,\theta\,(1-\mu)}\cdot\frac{|F|}{\zeta_d}\big)^{1/(d+1)}$, $\theta \in (0, 1/4)$, $\mu, \varepsilon \in (0, 1)$. The conclusion is $\lim_{n\to\infty}\Pr[c(\sigma_n) \le (1+\varepsilon)c^\*] = 1$. Their algorithm also uses the steering step $\eta$ and the near radius $\min\{r(|V|), \eta\}$. The correction is to the **exponent**, not to the constant; an earlier version of this document wrongly described it as "a corrected, larger constant".
- **Consequence.** The implemented exponent $1/2$ is *not* the corrected exponent $1/3$ (for $d = 2$). For large $n$, $\gamma(\ln n/n)^{1/2}$ falls below every $\gamma'(\ln n/n)^{1/3}$, whatever the constants, so no choice of $\gamma_f$ satisfies Solovey et al.'s condition. **The implemented radius therefore does not establish the corrected asymptotic-optimality guarantee.** Solovey et al. show that the original proof is incomplete; that is not a proof that the $1/d$ radius fails. Asymptotic optimality of the implemented planner is therefore **not claimed** (model §12, G7). The radius was kept as approved, without a new option, and E1c reports its measured convergence.

### 3.5 Correctness

- **Validity (Guaranteed):** every tree edge passed `FREE`, and discretisation preserves freeness and 4-adjacency.
- **Monotone best cost (Guaranteed):** nodes are never removed and rewiring only lowers costs, so the best goal cost is non-increasing over iterations.
- **Tree-cost invariant (Guaranteed):** after every iteration, $\text{cost}[q] = \text{cost}[\text{parent}(q)] + c(\text{parent}(q), q)$ for all non-root $q$. Eager propagation (`PROPAGATE`) maintains it. Choose-parent and rewire decisions read exact costs, including for near nodes that are descendants of an earlier rewired node in the same iteration.
- **Asymptotic optimality (Not claimed for the implemented radius):**
  - The implementation uses the Karaman–Frazzoli radius, exponent $1/d$ (§3.4). Karaman & Frazzoli's optimality theorem for this radius rests on a proof that Solovey et al. (2020) showed to have a logical gap. The corrected proof (Solovey et al., Theorem 1) needs the exponent $1/(d+1)$, which is not implemented. So the corrected guarantee is **not established** here, for any $\gamma_f$.
  - It would also be restricted to the Euclidean cost ($Q_i = \varnothing$ or $\lambda_{\text{prox}} = 0$). For the weighted cost ($\lambda_{\text{prox}} > 0$) the density is bounded, additive and monotone but discontinuous at zone boundaries, so no optimality claim is made there either.
  - What *is* reported is empirical: the measured convergence of the best cost on open maps (E1c, `05` §B.6). That evidence describes finite budgets and proves no asymptotic property.
- **Completeness:** with a finite budget there is **no guarantee**. Reachability is decided exactly beforehand by the BFS guard (Lemma 5.1), so `budget_exhausted` always means "path exists, not found in budget".

### 3.6 Complexity

Let $n$ = tree size at the current iteration ($n \le$ iterations), $k$ = $|Q|$ (near-set size), $k'$ = number of points in the axis-aligned bounding square of the query disk, $\ell \le \eta$ = segment length in cells, and $s$ = number of cost updates performed by `PROPAGATE` in the iteration.

**Per-operation bounds**

| Operation | `BruteForceIndex` | `LogKDTreeIndex` |
|---|---|---|
| Insert | $O(1)$ amortised (array doubling) | **$O(\log^2 n)$ amortised**: each point is rebuilt into larger slots at most $\lfloor\log_2 n\rfloor$ times, at $O(\log n)$ per point per rebuild. **$O(n \log n)$ worst case** for a single insertion that triggers a full rebuild. |
| Nearest neighbour | $\Theta(n)$ worst and expected | **$O(n)$ worst case** (a k-d tree has no sublinear worst-case bound for nearest neighbour). **Expected $O(\log^2 n)$** under a random-point model in the spirit of Friedman, Bentley & Finkel (ACM TOMS 3(3):209–226, 1977): expected $O(\log m)$ per tree, $O(\log n)$ trees. This is assumption **R**; RRT\* node sets are not i.i.d., so R is a modelling assumption checked empirically. *Source status (`verification-log.md` V-6): the paper's own text could not be read. Its abstract, as summarised by a secondary index, says the number of records examined is independent of file size and descent time is proportional to $\log N$. Its tree construction and probabilistic assumptions were not verified, nor whether its analysis covers our alternating-axis trees. R is therefore stated as an assumption, not as a cited theorem about this tree.* |
| Radius query | $\Theta(n)$ worst and expected | **$O(\sqrt{n} + k')$ worst case**: for a balanced 2-d tree with alternating median splits, an orthogonal range query costs $O(\sqrt{m} + \text{output})$, the standard kd-tree bound (de Berg et al., *Computational Geometry*, Ch. 5; Lee & Wong, Acta Informatica 9:23–29, 1977). The implemented query traverses exactly the nodes whose box meets the bounding square, so its node count is that of the square query, and the disk is applied as a filter. Summing over slots, $\sum_i O(\sqrt{2^i}) = O(\sqrt{n})$. *Source status (V-6): the bound with output term, $O(n)$ storage and $O(n\log n)$ construction is confirmed by several university lecture notes based on de Berg et al. The textbook's lemma number and the exact statement in Lee & Wong (paywalled) were not verified. Constant factors from our leaf buckets ($b_\ell \le 8$) are not covered by the cited statement. The usual recurrence argument (quoted in those notes as $Q(n) = 2 + 2Q(n/4)$) relies only on alternating median splits, which our construction uses. That transfer is our own reasoning, not a cited statement.* |
| Collision checks and segment costs | $O((k+1)(\ell+1))$ | same |
| Cost propagation | $O(s)$, with $s \le k \cdot n$ | same |

The bound $s \le k n$ holds because each rewired near node's subtree has at most $n$ nodes. It is loose: subtrees are usually small, but no better worst-case bound is known for eager propagation. A node can be updated several times in one iteration if several of its ancestors are rewired.

**Totals for $n_{\max} = N_{\text{it}}$ iterations**

| Regime | `BruteForceIndex` | `LogKDTreeIndex` |
|---|---|---|
| **Unconditional worst case** ($k, k' \le n$, $s \le kn$) | $O(N_{\text{it}}^2 \eta + N_{\text{it}}^3)$ | $O(N_{\text{it}}^2 \eta + N_{\text{it}}^3)$ |
| **Conditional on U** (near-set and bounding-square sizes $k, k' = O(\log n)$), excluding propagation | $\Theta(N_{\text{it}}^2)$ queries $+ O(N_{\text{it}}\,\eta \log N_{\text{it}})$ checks | $O(N_{\text{it}}^{3/2})$ radius queries $+ O(N_{\text{it}} \log^2 N_{\text{it}})$ inserts (amortised) $+ O(N_{\text{it}}\,\eta\log N_{\text{it}})$ checks $+$ nearest-neighbour cost: $O(N_{\text{it}}^2)$ worst, $O(N_{\text{it}} \log^2 N_{\text{it}})$ expected under R |
| Cost propagation, total $S = \sum s$ | $O(N_{\text{it}}^2 \log N_{\text{it}})$ worst under U; **measured** | same |

*Assumption U* is the regime the radius $r_n \propto \sqrt{\ln n / n}$ is designed for: with roughly uniform node density over $X_{\text{free}}$, the expected number of nodes in the $r_n$-disk is $\Theta(\log n)$. It can fail early (when $r_n = \eta$ is still large) and in sparsely sampled narrow regions.

**What is and is not claimed.**
- We **do not claim $O(n \log n)$** total running time. Karaman & Frazzoli's $O(n \log n)$ result counts primitive-procedure calls and assumes neighbour structures with logarithmic query time (e.g. approximate nearest neighbour). It does not account for exact k-d tree worst cases or for descendant-cost propagation, so it does not apply verbatim to this implementation.
- What is claimed is the table above.
- E1 measures the per-iteration counters ($k$, $k'$, nodes visited per nearest query, $s$, rebuild work) and fits empirical exponents for both indexes. Whether U and R hold in practice is reported as a finding.

**Space.** $O(n)$ for the tree (arrays, child lists). $O(n)$ for either index (the logarithmic method stores each point once; temporary rebuild buffers are $O(n)$). Discretisation output is $O(\text{path cells})$.

### 3.7 Expected behaviour (hypotheses, not claims)

RRT\* should be slower per query than A\* on small grids, and much slower in narrow passages and mazes, where passages have tiny sampling probability. It may be competitive on large open maps. E1 tests this (hypothesis H1). For the small iteration counts used online, the vectorised brute-force index may be faster than the k-d tree despite worse asymptotics. That trade-off between constant factors and asymptotics is measured and reported, not assumed.

### 3.8 Edge cases

start = goal (immediate success); start inside the goal region; goal unreachable (handled by the procedure before RRT\* runs); width-1 corridors; paths tangent to obstacle corners (conservative rule); tiny $n_{\max}$; time budget hit mid-iteration; zero-length steer (skipped); duplicate points in the k-d tree (median ties broken by index); $n = 1$ radius rule.

### 3.9 Tests

1. Supercover traversal vs dense-sampling reference, including explicitly enumerated corner cases. The weighted segment cost is compared with numerical integration. **Lemma 5.1 cases:** a segment between diagonal cells through an exact corner is rejected when either side cell is blocked. Random segments are accepted only if their touched cells form a 4-connected free chain (property test).
2. **Index oracle tests:** `LogKDTreeIndex` vs `BruteForceIndex` on random insert/query sequences (property test). Nearest and radius results must be identical, with distance ties resolved by index.
3. Logarithmic-method invariants: slot sizes match the binary representation of $n$, and every point appears exactly once across slots.
4. Static k-d tree invariants: alternating split axes, median property, depth ≤ $\lceil\log_2(m/b_\ell)\rceil + 1$.
5. **Validity property test:** on random solvable maps, every returned path is free w.r.t. $S \cup B_i$, 4-connected, with correct endpoints.
6. Monotone best-cost trace; tree-cost invariant after every iteration (debug mode); determinism per seed.
7. Work counters are consistent: $k \le n$, $s \ge$ number of rewires.
8. Convergence sanity on an empty map: the Euclidean length approaches the straight-line distance as $n$ grows (statistical test over seeds, documented tolerance).

**Implementation notes (Issue #3).**
- Modules: `planning/geometry.py` (traversal, `segment_cost`, `discretise`), `planning/neighbors.py` (`NeighborIndex`, `BruteForceIndex`), `planning/kdtree.py` (`StaticKDTree`, `LogKDTreeIndex`), `planning/rrtstar.py` (`RRTStarConfig`, `RRTStarPlanner`).
- **Supercover rule, made precise.** The check set is every cell whose *closed* square meets the segment. Beyond the interior cells of the traversal this includes: all four cells at an exact corner crossing; the cells on both sides of a segment that lies on a grid line; and the cells touching an endpoint that lies on a cell boundary. Two crossings closer than $10^{-9}$ in the segment parameter count as a corner. This can only add cells, so it is conservative and keeps Lemma 5.1's "accepted ⇒ 4-connected free chain" direction. Tested against an exact rational-arithmetic (Liang–Barsky) oracle: exact equality on lattice and half-lattice endpoints (corners, grid-line segments, boundary endpoints); on random float segments, equality except extra cells within $10^{-7}$ of the segment (the tolerance).
- **Canonical direction.** Every segment is evaluated from its lexicographically smaller endpoint, so `segment_cost(p, q)` and `segment_cost(q, p)` are bitwise equal. Edge costs are therefore computed once per candidate and reused for choose-parent and rewire, and the tree-cost invariant is checked with **exact** equality (`check_invariants`).
- **Propagation.** Instead of subtracting $\delta$ from every descendant, `PROPAGATE` recomputes `cost[u] = cost[parent(u)] + c(parent(u), u)` top-down over the rewired subtree, from stored edge costs. The work is the same $O(s)$, and the invariant holds exactly in floating point instead of accumulating rounding drift.
- **Steer step $\eta = 2.0$ cells** (configurable). The documents left $\eta$ open. It was fixed by a small pilot on micro-benchmark seeds 39000–39009 (32×32 maps; open, 20 %, 30 % density and rooms; 2 000 iterations; $\eta \in \{1, 1.5, 2, 3\}$). $\eta = 2$ gave the best balance of success within budget (39/40 queries) and path quality. $\eta = 1$ lost success on 30 % maps (7/10); $\eta = 3$ lost success on 30 % and rooms maps (7/10, 8/10). Sampling: goal bias $p_g = 0.1$ (uniform in the goal cell), otherwise uniform in $[0,W)\times[0,H)$; $\gamma_f = 1.5$; $\mu(X_{\text{free}})$ = free cells w.r.t. $S(t) \cup B_i$.
- **Consequence for the near radius.** With $\eta = 2$, $r_n = \min(\gamma\sqrt{\ln n / n}, \eta)$ equals $\eta$ until $\gamma\sqrt{\ln n/n} < 2$, i.e. until $n \approx 10\,150$ on 32×32, $n \approx 47\,400$ on 64×64 and $n \approx 216\,000$ on 128×128 open maps. Below that, the expected near-set size $n\pi\eta^2/\mu$ grows **linearly** in $n$. Once the radius shrinks, $k \approx n\pi r_n^2/\mu = 6\gamma_f^2\ln n \approx 13.5\ln n$ under uniform density. Assumption U therefore describes the asymptotic regime only; E1 measures which regime the online budgets fall in.
- RRT\* runs its full iteration budget (anytime). `stop_on_first_solution` (off by default) is used only by the X-11 consistency test. `time_limit_s` is the per-run wall-clock cap of the experiments.
- **Index invariants and oracle tests:** k-d tree and brute force returned identical nearest and radius results on Hypothesis-generated insert/query sequences, including snapped coordinates with many exact ties and duplicates, and identical $k'$ totals. Slot sizes equal the binary representation of $n$. Every point is in exactly one slot. Static trees satisfy the alternating axes, the median property, tight bounding boxes, leaf size and depth ≤ $\lceil\log_2(m/b_\ell)\rceil + 1$. Because both indexes answer identically, RRT\* builds the **same tree** with either index for a given seed. This is tested, and E1b confirms it on every query.
- **Default neighbour index: `brute` (decided from E1b, `05` §B.6).** At every measured budget (1 000–10 000 iterations; 64×64 open and 20 % maps) the brute-force index was faster, by a factor of 1.2–2.1 in median time across the original run and the counter-corrected re-run, and used about 25 % less memory. The k-d tree evaluates 7–130× fewer distances per nearest-neighbour query, examines only 16–54 tree nodes, and has the lower time exponent, but its pure-Python traversal costs more than one vectorised NumPy scan of ≤ 10 000 points. `kdtree` remains implemented, tested and selectable. **Every complexity statement for the default configuration therefore refers to `BruteForceIndex`.**
- **Work counters (definitions in `planning/kdtree.py`, pinned by regression tests):**
  - `nn_visited`: points whose distance a nearest query evaluated.
  - `nn_nodes`: tree nodes examined, including internal nodes and nodes pruned by the box test.
  - `range_nodes`: nodes a radius query expanded (box intersects the query square); `range_nodes_tested`: nodes examined, including pruned ones.
  - `rebuild_points`: points placed into newly built static trees (per insertion the new slot size, inserted point included).
  - `rebuild_work`: points processed by those builds, Σ subtree sizes over all built nodes. This is the quantity the O(m log m) build and amortised O(log² n) insertion bounds are about.
  - The node and work counters were added in the Issue #3 review. Before that, `nn_visited` and `range_nodes` were the only tree counters, and they under-counted the work of the analysis (`05` §B.6, E1b review correction).
- **Measured behaviour vs §3.6 (E1b, counter-corrected re-run):** k-d nearest-neighbour **nodes examined** grow with exponent 0.32–0.35 in $n$. Over $n \in [64, 10^4]$ this matches the growth of $\log^2 n$ (local exponent ≈ 0.32). A direct fit $a(\ln n)^c$ gives $c = 2.14$ (open) and $2.29$ (d20), with lower residuals than the power law (`05` §B.6). This is consistent with R, without proving an asymptotic bound. The worst case $O(n)$ is not approached. Points evaluated grow with only 0.10–0.13. Range-query nodes (examined 0.51–0.76, expanded 0.57–0.81) stay within $O(\sqrt n + k')$. Rebuild work per insertion grows with exponent 0.17–0.19 (13–37 points), within amortised $O(\log^2 n)$. **Assumption U does not hold at online budgets:** $k$ and $k'$ grow linearly in $n$ (exponents 1.0–1.2), because $r_n = \eta$ throughout. Propagation work per iteration grows super-linearly (exponent 1.3–1.4), still within $s \le kn$. Total time grows with exponent 1.5–1.8 in $N_{\text{it}}$, within the unconditional $O(N_{\text{it}}^2\eta + N_{\text{it}}^3)$. Memory is linear (exponent 1.03–1.04). The conditional "under U" rows of the table do not apply to the measured range; this is reported, not hidden.
- **Measured behaviour (E1a, E1c):**
  - Success within budget is high on open maps (100 % at every size and budget) and falls sharply with density, size and corridor structure. At 128×128 with 5 000 iterations: 99 % (10 %), 69 % (20 %), 9 % (30 %), 3 % (35 %), 20 % (rooms), 64 % (warehouse). This is the finite-budget behaviour that G8 does not claim to avoid.
  - When it succeeds with enough iterations on open or small maps, its continuous *reference polyline* is shorter than A\*'s lattice path in Euclidean length (e.g. 1.01× vs 1.45× the straight-line bound at 16×16 open), and its discretised path is step-optimal or nearly so. Robots execute the discretised 4-connected path, which is never shorter than A\*'s. The shorter polyline is therefore not an execution-efficiency advantage (`05` §B.6, E1a).
  - E1c, open maps: best cost within 0.15 % of the exact optimum after 20 000 iterations (mean). This is finite-budget empirical convergence. It does not establish asymptotic optimality, which is not claimed for the implemented radius (§3.5, G7).
- **No $O(n\log n)$ claim** is made or supported (G9): the measured total-time exponents in $N_{\text{it}}$ are 1.5–1.8.
- RRT\* tests: validity, monotone best-cost trace, exact tree-cost invariant after every iteration (60 random solvable queries, both indexes, $\eta \in \{0.7, 1.5, 3\}$); determinism per seed; counter consistency ($k \le n$, $k \le k'$, $\sum s \ge$ rewires, trace sums equal totals); convergence sanity on an empty 16×16 map (mean cost ratio at 900 iterations < at 150 and ≤ 1.03, never below the exact optimum).

---

## 4. Market-based task allocation

### 4.1 Bertsekas' forward auction with ε-scaling (primary)

**Purpose.** Solve the assignment problem of model §8. **References.** Bertsekas, *The auction algorithm: a distributed relaxation method for the assignment problem*, Annals of OR 1988; Bertsekas & Eckstein, *Dual coordinate step methods for linear network flow problems*, Math. Programming 1988.

**Input.** Integer benefit matrix $A \in \mathbb{Z}_{\ge 0}^{k \times k}$ after padding.

**Padding rule.** Square size $k = \max(n, m)$. Dummy rows and columns, and infeasible pairs, get benefit 0, while every feasible benefit is $> 0$ (offset $M$).
- A padded perfect matching has the same value as the sub-matching of its positive-benefit pairs, which is a feasible partial matching.
- Conversely, every feasible partial matching extends to a padded perfect matching of equal value by completing it with zero-benefit pairs.
- Hence optima correspond (Guaranteed).

**Output.** The robot → task mapping, with dummy and infeasible pairs dropped, plus stats (phases, bids, final ε, time).

**Data structures.** NumPy benefit matrix; prices `q[k]`; `owner[k]`; `assigned[k]`; FIFO queue of unassigned persons.

**Pseudocode.**

```text
AUCTION(A, θ = 5):
  C ← max A;  q ← 0;  ε ← max(C/θ, 1)
  loop:                                    # ε-scaling phases; prices persist across phases
      clear assignments; queue ← all persons
      while queue not empty:               # Gauss–Seidel bidding
          i ← pop(queue)
          values ← A[i,:] − q              # O(k), vectorised
          j* ← argmax values (lowest index on ties); v* ← values[j*]
          w* ← second-largest value (w* := v* if k = 1)
          q[j*] ← q[j*] + (v* − w*) + ε
          if owner[j*] ≠ −1: push(queue, owner[j*])
          owner[j*] ← i; assigned[i] ← j*
      if ε < 1/k: return assigned
      ε ← max(ε/θ, 1/(k+1))
```

**Correctness (Guaranteed).**
- *ε-CS invariant:* after $i$'s bid, $a_{ij^\*} - q_{j^\*} = w^\* - \varepsilon \ge \max_{j\ne j^\*}(a_{ij} - q_j) - \varepsilon$. Prices only rise, so other holders keep ε-CS.
- *Termination of the first phase* (zero initial prices): while some object has received no bid, its price is 0 and its value is ≥ 0 for every bidder. So no object's price exceeds $2C + \varepsilon$, and each bid raises a price by ≥ ε. Hence $O(C/\varepsilon)$ bids per object, and the phase ends once every object has been bid on (square problem). For later phases (warm-started prices), termination and the bound below follow from the ε-scaling analysis of Bertsekas & Eckstein (1988).
- *Optimality bound:* if $S$ satisfies ε-CS with prices $q$, then for any assignment $S'$,
  $$\sum_i a_{iS(i)} \ge \sum_i\Big(\max_j(a_{ij} - q_j) - \varepsilon\Big) + \sum_j q_j \ge \sum_i a_{iS'(i)} - k\varepsilon .$$
  So $S$ is within $k\varepsilon$ of optimal, and **optimal** for integer benefits with final $\varepsilon < 1/k$.
- *Maximum cardinality:* with the offset $M$ (model §8), every optimal assignment has maximum feasible cardinality.

**Complexity.**
- One bid: $O(k)$.
- Without scaling: $O(k \cdot C/\varepsilon)$ bids, i.e. $O(k^2 C/\varepsilon)$ worst case, which is $O(k^3 C)$ at $\varepsilon \approx 1/k$ (pseudo-polynomial).
- With ε-scaling: $O(k \cdot A \log(kC))$ for $A$ arcs (Bertsekas & Eckstein 1988), i.e. $O(k^3 \log(kC))$ worst case for dense matrices (weakly polynomial).
- Space $O(k^2)$.
- Measured bids and time vs $k$ and $C$ are reported in E3 against SciPy's Hungarian implementation ($O(k^3)$).

**Benefit construction.** One BFS field per distinct pickup and drop-off cell ($O(V)$ on a cache miss) plus $O(1)$ per pair. Worst case $O(mV + nm)$ per auction; $O(nm)$ with a warm cache.

**Edge cases.** No eligible robots or no available tasks (no-op). $n \ne m$. All pairs infeasible (empty mapping). Ties (deterministic). $k = 1$. Large $C$: `int64` range validated.

**Implementation notes (Issue #4, stage 1: benefits).**
- `allocation/benefits.py`: `build_problem(robots, tasks, t, distance, BenefitConfig)` returns an `AllocationProblem`. It holds the real-valued `cost` ($c_{ij}$), `priority_value` ($\kappa\tilde w_j$), `value` ($\kappa\tilde w_j - c_{ij} + h\mathbb 1[\tau_i = j]$, $-\infty$ if infeasible) and the padded $k\times k$ int64 `matrix` used by the allocators.
- Defaults are those of model §11: $\kappa = 10$, $h = 2$, $\alpha = 0.01$, $s = 100$. $w(\pi) = 2^{\pi-1}$, and the age is clamped at 0 before release. $h$ must be an integer.
- **Integer form (a refinement of model §8, same formulas):**
  - Every term is multiplied by $s$. The integer benefit of a feasible pair is $A_{ij} = M_{\text{int}} + P_j - s\,c_{ij} + s\,h\,\mathbb 1[\tau_i = j]$, with the priority term $P_j = \mathrm{round}(s\kappa\tilde w_j)$, the **distance term** $s\,c_{ij} = s\,(D(x_i,p_j) + D(p_j,d_j))$ and the **hysteresis term** $s\,h$.
  - $c_{ij}$ (BFS distances) and $h$ are integers, so $s c_{ij}$ and $s h$ are exact. The only rounded quantity is $P_j$. Each integer benefit therefore differs from $s\,a_{ij}$ by at most $1/2$, i.e. by at most $1/(2s)$ in original units (model §8).
  - The offset uses the model's formula on the **scaled integer components**: $M_{\text{int}} = 1 + k(B_{\text{int}} + C_{\max})$ with $B_{\text{int}} = \max_j P_j + s h$ and $C_{\max} = s\,c_{\max}$. The model's maximum-cardinality argument then holds exactly in integers for every $\kappa$ and $k$.
  - Scaling a real-valued $M$ and rounding afterwards is not guaranteed to keep the strict inequality in general: the rounding error over a matching can reach about $k$ integer units, against a margin of $s(1+B)$. With the defaults ($B \ge \kappa = 10$) this would only fail for $k$ above about 2 000, and with $\kappa = 0$ nothing is rounded. So this is a robustness refinement, not a change of the model.
- **Tests** (`tests/unit/allocation/test_benefits.py`):
  - A hand-computed instance: every cost, value, offset and matrix entry.
  - Infeasible pairs (unreachable pickup; pickup → drop-off unreachable) are 0, every feasible entry is ≥ 1, and padding rows and columns are 0.
  - 300 random instances checked against independent brute force (all partial matchings; all $k!$ perfect assignments, $k \le 5$): the optimal padded assignment has maximum feasible cardinality, maximises the integer core among maximum-cardinality matchings, and equals the best partial matching in value (padding equivalence).
  - 300 instances for the rounding bound: $|{\rm core} - s\,a| \le 1/2$, identical down each task column because only $P_j$ is rounded.
  - 60 large instances ($k \le 120$, including the $\kappa = 0$ and $h = 0$ ablations) for the hardest margin inequality $k\cdot\min > (k-1)\cdot\max$, which implies $(q+1)\min > q\max$ for every $q \le k-1$.
  - Input validation and the int64 overflow guard.

**Implementation notes (Issue #4, stage 2: auction).**
- `allocation/auction.py`: `solve(matrix, scaling, epsilon, theta, time_limit_s, max_bids)` solves the square integer problem; `AuctionAllocator` (`name = "auction"`) applies it to an `AllocationProblem` and drops dummy and infeasible pairs. `allocation/base.py` holds the `Allocator` protocol, `SolveStats`, `SquareSolution` and `AllocationResult`.
- **Exact integer arithmetic.** Benefits, prices and ε are all int64; no floating-point price is ever compared.
  - *ε-scaling (production).* The benefits are multiplied by $k+1$. An integer ε of 1 then equals $\varepsilon = 1/(k+1) < 1/k$ in original units, which is the pseudocode's stopping condition: the final assignment is **optimal** for the integer benefits (Guaranteed, by the optimality bound above). The schedule is $\varepsilon_0 = \max(\lfloor C'/\theta \rfloor, 1)$, then $\varepsilon \leftarrow \max(\lfloor \varepsilon/\theta \rfloor, 1)$, prices kept between phases, stopping after the phase with integer ε = 1 ($C'$ = largest scaled benefit, θ = 5).
  - *Fixed ε (E3a only).* One phase from zero prices with an integer ε ≥ 1 in original units: $V \ge \mathrm{OPT} - k\varepsilon$ (Guaranteed). For $\varepsilon = 1$ this is the additive bound $V \ge \mathrm{OPT} - k$. No relative-error guarantee is claimed; E3a reports the measured gap against the Hungarian optimum.
- **Bid bound for fixed ε (derived here, Guaranteed).** Non-negative integer matrix with largest entry $C$, zero initial prices.
  - While some object $j_0$ has received no bid, $q_{j_0} = 0$ and every bidder values it at $a_{ij_0} \ge 0$.
  - A bid on $j^\* \ne j_0$ sets $q_{j^\*} = a_{ij^\*} - w^\* + \varepsilon \le C - 0 + \varepsilon$, since $w^\* \ge a_{ij_0} - q_{j_0} \ge 0$.
  - Each bid raises the price by at least ε, so each object receives at most $\lfloor C/\varepsilon \rfloor + 1$ bids before every object has been bid on.
  - In a square problem the phase ends as soon as every object has been bid on (each then has an owner). So the only bid not covered above is the single final bid on the last unbid object. Total: at most $k(\lfloor C/\varepsilon \rfloor + 1) + 1$ bids.
  - This sharpens the earlier $2C + \varepsilon$ price bound for the non-negative case. Warm-started ε-scaling phases are not covered by it; for those the Bertsekas & Eckstein (1988) analysis is cited, not re-derived.
- **Integer optimum vs real-valued value.** The auction is exact on the *integer* problem. Two integer-optimal matchings can still differ in original units, because only $P_j$ is rounded. With $q$ pairs, each within $1/(2s)$, the integer optimum $S$ and the real-valued optimum $S^\*$ (both of maximum cardinality) satisfy $\mathrm{real}(S) \ge \mathrm{real}(S^\*) - q/s$. This occurred in development (92.0 vs 92.1 on one instance), so the tests compare integer totals exactly and real values within $q/s$.
- **Time and bid caps.** `time_limit_s` and `max_bids` are checked every 256 bids, counted **cumulatively** across phases, and at every phase boundary. On a cap the solve returns no assignment (`timed_out = True`); what the coordinator does then is part of the allocation policy (stage 4).
  - The first version counted bids per phase. The timeout test showed that an ε-scaling solve with many short phases (k = 60: 11 phases, about 108 bids each) then never reached a check. Fixed before any results were recorded.
- **Tests** (`tests/unit/allocation/test_auction.py`; oracle `scipy.optimize.linear_sum_assignment`, independent of the auction):
  - 200 random square matrices, $k \in [1, 60]$, $C \in \{1, 3, 10, 10^3, 10^6\}$ (small $C$ gives many ties), 30 % with zero blocks like padding: ε-scaling value equals the Hungarian optimum; final ε < 1/k; ε-CS holds at termination.
  - 200 rectangular problems with infeasible pairs through `build_problem` (up to 12 × 12): integer total equals Hungarian's; same, maximum cardinality; real value within $q/s$; only feasible pairs; no robot or task twice.
  - 200 matrices × ε ∈ {1, 2, 7, 50, 1000} without scaling: $\mathrm{OPT} - k\varepsilon \le V \le \mathrm{OPT}$, ε-CS, the bid bound above, one phase.
  - The $k\varepsilon$ bound is not vacuous: a strictly suboptimal fixed-ε result is found.
  - Edge cases ($k = 0$, $k = 1$, all-equal ties, negative entries), determinism (same assignment and bid count), validation (non-square, θ < 2, ε < 1, int64 overflow, unknown mode), both caps, and a hand-built corridor instance.

**Tests.**
1. **Oracle property test:** random integer matrices, $k \in [1, 60]$, rectangular and with infeasible pairs. Total benefit equals the `scipy.optimize.linear_sum_assignment` optimum, and cardinality is maximum.
2. ε-CS holds at termination.
3. Fixed ε without scaling stays within $k\varepsilon$ of the optimum.
4. Padding and infeasibility handling; no dummy or infeasible pair in the output.
5. Determinism.

### 4.2 Sequential single-item auction (baseline)

**Algorithm.** While a free robot and a feasible task remain: every free robot bids its benefit for its best remaining task, the globally highest bid wins, and the pair is removed.

**Quality.** This is greedy maximum-weight matching: a **½-approximation** (Guaranteed), and not optimal in general. Because of the offset $M$ it also reaches maximum cardinality only if greedy happens to. Both shortfalls are measured.

**Complexity.** $O(\min(n,m)\cdot nm)$ naively. $O(nm \log m)$ with pre-sorted bid lists.

**Tests.** Optimality on instances where greedy is optimal. A constructed counter-example. Value ≥ ½ optimum on random instances (property test against the Hungarian oracle).

**Implementation notes (Issue #4, stage 3: sequential auction).**
- `allocation/sequential.py`: `solve(matrix)` on the same padded integer matrix as the auction (positive entry = feasible pair; 0 = dummy or infeasible); `SequentialAllocator` (`name = "sequential"`).
- **Tie rule (documented here; the spec leaves it open).** A robot bids for its lowest-index best task; among equal bids the lowest robot index wins.
- **Implementation.** The round winner is always the remaining feasible pair that comes first in the order $(-A_{ij}, i, j)$. So one sort of the feasible pairs and one scan that keeps each pair whose robot and task are both still free reproduce the round-by-round outcome exactly. Time $O(F \log F)$ for $F \le nm$ feasible pairs (the $O(nm\log m)$ variant above, up to the log factor); memory $O(F)$. Unmatched rows and columns are paired in index order to give a full permutation. Each such pair has benefit 0 (otherwise the matching would not be maximal), so the value is unchanged and the pairs are dropped as dummy or infeasible.
- **What the ½ guarantee is about.** It holds for the objective the allocator is given: the non-negative integer matrix, including the offset $M_{\text{int}}$: $2V \ge \mathrm{OPT}$ (Guaranteed). Each optimal pair loses, or ties under the rule, to a greedy pair sharing an endpoint, and each greedy pair shares an endpoint with at most two optimal pairs.
  - It does **not** transfer to original units: $\kappa\tilde w_j - c_{ij} + h\mathbb 1$ can be negative.
  - Constructed instance (8 × 1 corridor, κ = 4): greedy $-1$ vs optimum $+1$ in original units, while the integer bound holds. With the default κ = 10 the same instance gives 11 vs 13.
  - On benefit problems the offset gives a much stronger bound: $V \ge \frac{k-1}{k+1}\,\mathrm{OPT}$. Like the ½ bound, it is about the **offset-inclusive integer objective** $\sum A_{ij}$, *not* the original unshifted benefits $\kappa\tilde w_j - c_{ij} + h\mathbb 1$. It mostly measures $M$, so it says almost nothing about allocation quality. E3 therefore reports the gap in original units, not this ratio. Statement and proof below.
- **The connected-component assumption (CC)**, on which the maximum-cardinality result and the $(k-1)/(k+1)$ ratio rest.
  - *Statement.* For every robot $i$ and task $j$ of a problem, $(i, j)$ is feasible **iff** robot $i$'s cell and pickup $p_j$ lie in the same connected component of the free static grid $G(t)$, and $d_j$ is reachable from $p_j$.
  - *Why it holds in this code.* `build_problem` declares a pair feasible iff $D(x_i, p_j) < \infty$ and $D(p_j, d_j) < \infty$, with no other filter (no capacities, time windows or robot–task compatibility). `DistanceOracle` computes $D$ by BFS on the undirected 4-connected grid of static obstacles, so "$D < \infty$" is symmetric and transitive: it is "same connected component". A robot on a blocked cell, which map events do not create, would form a component of its own with no feasible pair, which is still consistent with CC. Dynamic obstacles never enter $D$ (model §4).
  - *Consequence.* The feasible pairs form one complete bipartite block $R_c \times J_c$ per component $c$, and blocks share no robot or task.
  - *When it would fail.* Any feasibility rule that depends on more than (robot component, task): directed or one-way aisles, distances that include dynamic obstacles or other robots, capacities, battery, robot–task compatibility. Only the general ½ guarantees would then remain.
- **Cardinality (refines the paragraph above).** On a general matrix, greedy is a *maximal* matching, so it gets at least half the maximum cardinality (Guaranteed). It can get exactly half: $[[2,2],[2,0]]$ gives one pair against two. Under CC, greedy always reaches **maximum** cardinality (Guaranteed under CC; tested on 200 benefit problems against Hungarian).
  - *Proof.* Greedy is maximal. If after greedy some block $c$ still had both a free robot and a free task, that pair would be feasible (CC) with positive benefit, contradicting maximality. So greedy matches $\min(|R_c|, |J_c|)$ in every block, and no matching can do better, because blocks share no endpoints.
  - On benefit problems the only shortfall of greedy is therefore in value.
- **The $(k-1)/(k+1)$ ratio on the integer objective (Guaranteed under the assumptions; tested on 200 benefit problems).**
  - *Assumptions.*
    - (i) The objective is the padded integer matrix $A$ of `build_problem`: feasible entries $A_{ij} = M_{\text{int}} + \mathrm{core}_{ij}$ with $\mathrm{core}_{ij} = P_j - s\,c_{ij} + s\,h\,\mathbb 1[\tau_i = j]$ and $M_{\text{int}} = 1 + k(B_{\text{int}} + C_{\max})$; dummy and infeasible entries are 0.
    - (ii) $\kappa \ge 0$ and $h \ge 0$ (enforced by `BenefitConfig`), so $0 \le P_j \le \max_j P_j$, and therefore $-C_{\max} \le \mathrm{core}_{ij} \le B_{\text{int}}$, with $B_{\text{int}} = \max_j P_j + s h$ and $C_{\max} = s\,c_{\max}$.
    - (iii) CC holds, so greedy $G$ and an optimum $O$ have the same cardinality $q$.
    - (iv) $k \ge 1$.
  - *Proof.*
    - If $q = 0$, both values are 0 and the bound holds trivially.
    - Otherwise $V(G) \ge q(M_{\text{int}} - C_{\max})$ and $\mathrm{OPT} \le q(M_{\text{int}} + B_{\text{int}})$, and $\mathrm{OPT} > 0$.
    - Then $(k+1)V(G) - (k-1)\mathrm{OPT} \ge q\big[2M_{\text{int}} - (k+1)C_{\max} - (k-1)B_{\text{int}}\big] = q\big[2 + (k+1)B_{\text{int}} + (k-1)C_{\max}\big] \ge 0$, substituting $M_{\text{int}} = 1 + kB_{\text{int}} + kC_{\max}$. ∎
  - *What it does not say.*
    - Nothing about the original benefits $\kappa\tilde w_j - c_{ij} + h\mathbb 1$, which have no offset and can be negative. On the κ = 4 corridor instance, greedy scores $-1$ against the optimum's $+1$: no ratio bound exists there.
    - The factor comes from $M_{\text{int}}$ dominating every core term, not from greedy being good.
    - Without CC, only the general ½ bound holds.
  - The policy's keep-current tie-break (§4.3 notes) runs greedy on $A' = (k+1)A + E$. Sorting by $A'$ is sorting by $A$ with ties broken towards current pairs, so this is greedy on $A$ under another tie rule. The ½ proof holds for any tie rule, maximality is unaffected, and so both bounds still hold for $A$.
- **Tests** (`tests/unit/allocation/test_sequential.py`; oracles independent of the module):
  - A literal round-by-round simulation of the rule above, used as the reference for 200 random matrices ($k \le 40$, tie-heavy small $C$, zero blocks): same pairs; value and round count match; $2V \ge \mathrm{OPT}$ and $V \le \mathrm{OPT}$ against Hungarian; maximality; cardinality ≥ ½ maximum.
    - Mutation check (scratchpad, not in the suite): flipping either tie rule changes the outcome on 99/200 and 102/200 of these instances, so the reference detects such errors.
  - 200 rectangular benefit problems through `build_problem`: integer ½ bound against Hungarian; maximum cardinality; only feasible pairs; reported value equals `matching_value`.
  - Greedy is optimal, checked by enumeration or Hungarian, on additive matrices $A_{ij} = u_i + v_j$ (every perfect matching has equal value) and on single-robot problems, plus a separated corridor instance where it matches the auction.
  - Counter-examples:
    - $[[2,2],[2,0]]$: ratio exactly ½ and cardinality 1 vs 2. $[[3,2],[2,0]]$: no ties, 3 vs 4.
    - The 8 × 1 corridor through the benefit model: greedy cost 9 vs optimum 7.
    - The κ = 4 variant: no ½ bound in original units.
  - Edge cases, determinism, validation (non-square, negative entries).

### 4.3 Allocation policy

- **Triggers:** task released; task delivered; robot became idle; a robot's plan status became `unreachable`; map event. At most one auction per step.
- **Eligible:** robots in mode idle or to_pickup; *available* tasks pending or assigned-but-not-picked (model §4).
- **Pre-emption:** only through the auction, with hysteresis $h$.
- **Output application:** a changed assignment changes the robot's goal, which is trigger T1 for replanning.

**Implementation notes (Issue #4, stage 4: allocation policy and coordinator).**
- `allocation/policy.py`: `AllocationPolicy(allocator, BenefitConfig)` with `step(view, api, distance)` and an `AllocationEvent` record per auction. `planning/coordinator.py`: `AllocationCoordinator(planner, allocator, benefit_config=…)`, named `"<allocator>+<planner>"`.
  - The path monitor, release/hold policy, suggestions and reporting now live in a shared base class. The Issue #3 fixture coordinator `PlanningCoordinator(planner, assignments)` keeps its name, signature and behaviour; the Issue #3 integration, regression and e2e tests pass unchanged.
- **Triggers and the one-auction rule.** Each step the policy compares the view with a snapshot from the previous step and collects, in this order: `released` (a task left `unreleased`), `delivered`, `idle` (a robot became idle other than through the policy itself), `unreachable`, `map_event` (grid version changed) and `retry`. If any fired, it runs **exactly one** auction for the step.
  - A release after an `unreachable` plan is detected by the monitor *after* the step's auction, so it is carried over and triggers the next step's auction.
  - Aging is not a trigger (as specified): it acts at the next allocation event.
  - No auction runs while nothing fires. This loses no assignment: after an auction the matching has maximum cardinality (model §8), and only the triggers can create a new feasible (idle robot, unassigned task) pair. Robot moves cannot, because under CC (§4.2 notes) feasibility depends only on components, which only map events change.
- **Eligibility.** $R_f$: robots in mode idle or to_pickup, so carried tasks are never pre-empted. $J_p$: tasks the engine reports as *available* (pending or assigned, both cells free, $D(p_j, d_j) < \infty$). The current task $\tau_i$ of a to_pickup robot gets $+h$.
  - A triggered step with no eligible robot, or with neither an available task nor a held one, is counted as `skipped`; no auction runs.
- **Application.** The new matching $\sigma$ is compared with $\tau$ in robot-id order. *All* robots whose task changes or disappears are unassigned first (the task returns to pending), then all new assignments are made. So a task can move between robots within one event, and the engine's state machine (assign only idle robots and pending tasks) is respected.
  - A robot whose task became unavailable and gets no new one is unassigned, and the task waits, pending, until it is available again (model §5.3 step 4).
  - The new goal $g_i$ is the pickup cell. The monitor sees $g_i \ne$ its stored goal and replans with T1 in the same step. A robot left without a task gets goal `None` and no call.
  - Not a replan: switching to another task *with the same pickup cell* leaves $g_i$ unchanged, so the existing path stays valid and T1 does not fire.
- **Pre-emption** exists only through this auction and only beyond the hysteresis margin, because $h$ is inside the benefits.
- **Ties keep the current assignment** (`prefer_current`, a refinement this stage adds). Model §8 says a robot keeps its task "unless another assignment is better by more than $h$", so at a gain of exactly $h$ it must keep it.
  - With integer distances such exact ties are reachable. Without a rule, the result depended on the task order: a scratchpad check showed both allocators switch at gain $= h$ whenever the other task is listed first.
  - The allocator is therefore given $A' = (k+1)A + E$, with $E_{ij} = 1$ on feasible current pairs. Every optimum of $A'$ is an optimum of $A$ that keeps as many current assignments as possible (exact lexicographic refinement; proof in `prefer_current`). Real-valued values are unchanged.
- **Release on `unreachable`** (model §5.3 step 4). Normally the `map_event` auction pre-empts it: map events are applied before the coordinator runs, so that auction already drops every pair that became infeasible, and the monitor then only plans reachable pickups. This was observed in the map-event integration test: `releases = 0`.
  - The path is reached when the map-event solve is capped (`retry`). The robot then keeps a cut-off pickup, the monitor's T2 call returns `unreachable`, and the robot releases the task. The next step's auction is triggered by `unreachable` and `retry`; an integration test covers this.
- **Statistics:** `extra["planning"]["allocation"]` holds auctions, skipped events, trigger counts, assignments, pre-emptions (robot switched task), lost tasks (robot left with none), tasks reassigned, timeouts and bids. Each auction also adds a deterministic `"alloc"` trace field (triggers, assigned, unassigned).
- **Not in this stage:** the CLI `--allocator` option (stage 6) and the baseline controller (stage 5). Tests use the existing Issue #3 plan follower (`SuggestionFollower`).

### 4.4 Priority semantics: what the tests check

The allocation implements **weighted priority** (model §8.1). Tests verify the *weighted* rule, not strict priority:
- **Threshold test:** two tasks of levels $\pi_{\text{hi}} > \pi_{\text{lo}}$ and one robot. The high-priority task is chosen iff $\kappa(\tilde w_{\text{hi}} - \tilde w_{\text{lo}}) > c_{\text{hi}} - c_{\text{lo}}$. Both sides of the threshold are checked, and exact equality is resolved by the documented tie rule.
- **Hysteresis test:** no switch when the gain is ≤ $h$; a switch when it is $> h$.
- **Aging property tests:** P1 (FIFO within a level), P2 (outranking newly released tasks after $A^\*$ steps), P3 (a younger higher-level task regains precedence), all on the priority term, with exact arithmetic.
- **Starvation mitigation** is *not* a unit-test property. It is measured in E3 (waiting-time tails per level, $\alpha > 0$ vs $\alpha = 0$).

**Implementation notes (Issue #4, stage 4: tests).**
- `tests/unit/allocation/test_priority.py`. With $\alpha = 1/100$ and $s = 100$, $P_j = \kappa\,w(\pi_j)(100 + a_j)$ is an exact integer, so expected outcomes are computed from the model's formulas with integers or `Fraction` only.
  - *Threshold:* 150 Hypothesis examples per allocator (κ, levels, ages, positions), on both sides of $\kappa(\tilde w_{\text{hi}} - \tilde w_{\text{lo}}) = c_{\text{hi}} - c_{\text{lo}}$; at exact equality, both choices are optimal and the result is deterministic. Also an explicit instance (a level-2 task buys a detour of fewer than 10 cells) and a check that priority is not strict.
  - *Hysteresis:* $h \in \{0, 2, 5\}$ and gains from $-2$ to $h+3$, both task orders, both allocators: no switch iff gain ≤ $h$. The equality case is shown to be an exact tie of the untie-broken matrix.
  - *`prefer_current`:* on random benefit problems, the result is optimal for the original integer matrix (Hungarian) and keeps at least as many current pairs.
  - *P1, P2, P3* on the priority term: FIFO within a level; $A^\* = 300$ for $P = 3$, tight at 299; P3's exact condition and that the advantage is only temporary.
- `tests/unit/allocation/test_policy.py`: hand-built views and a fake `AssignmentAPI` that enforces the engine's rules. Covers:
  - trigger detection and order, and one auction for a step with four triggers;
  - `unreachable` carried to the next step without a duplicate `idle`;
  - retry after a capped solve; skipped steps;
  - carried tasks never pre-empted; unavailable tasks excluded and their holder released;
  - a task moving between robots within one event (unassign before assign);
  - pre-emption by an urgent task, then no switch at gain $= h$;
  - determinism.
  All of it runs for both allocators.
- `tests/integration/test_allocation_integration.py`: the engine with `AllocationCoordinator` and the plan follower.
  - Planner {A\*, RRT\*} × allocator {auction, sequential}: delivery, one auction per step, T1 at $t = 0$.
  - Urgent-task pre-emption: switch at release, **T1 replan in the same step**, urgent task picked first, the low task re-auctioned and delivered, `"alloc"` trace field.
  - Hysteresis in simulation: no switch and no T1 at gain 0.8, a switch and a T1 at gain 2.8.
  - Carried task kept. Weighted priority ordering, both equal-travel and not strict.
  - Map-event re-allocation with `releases = 0`.
  - Determinism: byte-identical traces on two generated dynamic scenarios.
- **Aging in simulation (constructed scenarios, not guarantees):**
  - *Mitigation, no backlog:* one robot, a level-1 task at $x = 16$, level-2 tasks released every 2 steps (one always fresh when the robot frees). The low task is picked at $t = 126$ with $\alpha = 0$ (after the stream ends) and at $t = 18$ with $\alpha = 0.1$.
  - *Limitation, kept as a counter-example:* the same stream with the low task at $x = 25$ and level-3 tasks. With $\alpha = 0.1$ the low task wins an auction at $t = 42$, but is then pre-empted by an aged level-3 task from the backlog that formed during its trip, and is picked at $t = 135$, the same as with $\alpha = 0$.
  - This matches model §4 (P3, "Not claimed (i)"): multiplicative aging lets a backlog of higher levels outrank an old low task. It was found while building the first scenario, whose initial design assumed no backlog.

---

## 5. Execution: priority handling, path monitor, controllers, shield

### 5.1 Where priority acts

| Conflict | Mechanism | Guarantee status |
|---|---|---|
| Tasks compete for few robots | Weighted priority with aging in auction benefits | Optimal for the weighted objective (G10); no strict precedence (G11) |
| An urgent task appears | Event-triggered re-auction with pre-emption before pickup | Same as above |
| Robots compete for space | Right-of-way by effective task priority (baseline rule); neighbours' priorities observed by RL policies | Heuristic; evaluated empirically |

### 5.2 Path monitor

Implements model §5.4 for each robot after every step:
- **Rejoin:** check whether $x_i(t+1) \in \{c_{\ell_i}, \dots, c_{\ell_i+H_r}\}$ (scan ≤ $H_r + 1$ cells) and set $\ell_i$.
- **Invalidation (T2):** check the next $H_v$ cells against $S(t) \cup B_i(t)$.
- **Stall (T4):** maintain a counter and a temporary block list.
- **Retry (T5/T6):** cooldowns.

At most one planner call per robot per step. Calls go through the shared procedure (§2.2) and are logged with their trigger.

**Complexity.** $O(H_r + H_v)$ per robot per step, plus planner calls.

**Tests.** Rejoin advances the index without replanning. A deviation triggers T3. A map event on the path triggers T2. A stall triggers T4 with a detour around the blocked cell, but never blocks the goal. Cooldowns are respected. Trigger attribution is correct.

**Implementation notes (Issue #3).**
- `planning/monitor.py` (`PathMonitor`) and `planning/coordinator.py` (`PlanningCoordinator`). The coordinator runs the bookkeeping at the start of each step's pipeline step 3, on the positions at $t$. This is equivalent to "after each step" (architecture §2, step 9) and needs no extra engine phase.
- When several triggers fire, the single call is attributed to the first in the order T1, T2, T3, T4, T5, T6. Every call is logged with time, robot, trigger, status, `degraded` and path length. These fields, but no timings, are written to the trace (`"plans"`), so traces stay byte-identical (X-1 holds for both planners).
- **Interpretations of model §5.4, made explicit here and in the model:**
  1. *T2 and the goal cell.* "Excluding the goal" applies to the dynamic set (the goal is never in $B_i$). The goal is still checked against $S(t)$ at every step (O(1)). Otherwise a map event on the goal would never be detected, and the `unreachable` hold/release policy of model §5.3 step 4 could not apply.
  2. *T2 on degraded plans* checks $S(t)$ only. A degraded path crosses dynamically blocked cells by construction, so a dynamic check would fire on every step. Degraded plans are re-checked by T6 after $T_{\text{cd}}$.
  3. *T4 at the goal.* A stall in front of an occupied goal cell does not trigger a replan, and the goal is never added to the block list. Waiting there is the occupied-goal policy of model §5.3.
  4. *T4 needs an occupied cell.* Two robots that both keep proposing the same **free** cell are frozen by the arbiter every step without either next cell being occupied, so T4 does not fire. For a pure plan follower this is a deadlock. Resolving it is the job of the controllers (baseline right-of-way rule, learned policies; Issue #4+), not of the planner. E2 reports how often it limits completion. *Issue #4:* the baseline's right-of-way lets exactly one claimant move. Mutual blocking of robots in each other's cells (head-on, rotation) is handled by the opt-in deadlock resolver of §5.5 (trigger T7), not by T4.
  5. *T4 vs T6 on a degraded plan* (documented by `test_degraded_plan_and_stall_interaction`). With the defaults $T_{\text{cd}} = 3 < T_{\text{stall}} = 4$, a robot stalled in front of a dynamically blocked cell of a degraded path is replanned by T6 every 3 steps. Each replan resets the stall counter, so T4 does not fire while the plan stays degraded. T4 fires first only if $T_{\text{cd}} \ge T_{\text{stall}}$. This is the specified trigger order applied literally, not a separate rule.
- Release/hold policy (model §5.3 step 4): a robot whose pickup becomes statically unreachable releases the task (`AssignmentAPI.unassign_task`). **Issue #3 fixture logic only:** the fixture allocation re-offers a released task to the same robot only once that robot's static reachability *can* have changed, i.e. after a map event (grid version) or after the robot has moved, and only while the task is available. Re-offering every $T_{\text{cd}}$ steps in an unchanged situation repeated the same `unreachable` call and release every 3 steps (14 releases in 40 steps in the review probe; changed in V-6, regression test `test_release_is_not_reoffered_until_something_changes`). In Issue #4 the auction replaces this: a released task is re-auctioned and can go to another robot. A robot whose drop-off becomes unreachable holds the task, stays (suggested `stay`) and retries with T5 every $T_{\text{cd}}$ steps.
- Engine hooks: a coordinator that also implements `sim.coordinator.PlanReporting` (`plan_valid`, `trace_fields`, `episode_stats`, `finish`) gets plan adherence (model §9.3) recorded by the engine, its planning fields added to the trace, and its statistics added to `EpisodeMetrics.extra["planning"]`. This is an additive extension of the Issue #2 coordinator contract; earlier coordinators are unaffected.
- `control.base.SuggestionFollower` ("follow") plays the masked suggested action. It is the minimal plan follower used by tests and E2. The reference follower of Issue #4 takes over its role in the controller registry.
- **X-12 / X-5 results:** `tests/unit/planning/test_monitor.py` checks each trigger at its exact step: T1; rejoin within $H_r$ without replanning; T3 beyond $H_r$; T2 for an obstacle or a map event within $H_v$ and not beyond it; T4 exactly after $T_{\text{stall}} = 4$ steps with block expiry; no T4 at the goal; T5 every $T_{\text{cd}} = 3$ steps; T6. `tests/integration/test_planning_integration.py` runs the same checks inside the engine for both planners: a map event on the path gives T2 and a detour; a map event on the drop-off gives `unreachable`, hold, T5 retries at $t = 12, 15$, 11 blocked-goal robot-steps, then delivery; a map event on the pickup gives a release and a re-take; an obstacle crossing the path gives T2. Plan adherence and deviation rate are checked against hand computation.

### 5.3 Controllers (non-learning)

**Reference follower (diagnostic).** $a_i = \hat a^{\text{plan}}_i$ (stay if masked). It defines the counterfactual of model §9.2 and the episode-level comparison point. $O(N)$ per step.

*Implementation (Issue #4 stage 6):* `control/reference.py`, `ReferenceFollower` (`name = "reference"`). Its actions are exactly those of the Issue #3 plain follower `SuggestionFollower` (property-tested on 300 random worlds; the CLI's `follow` stays as an alias). It adds decision counts (`follow`, `no_plan`, `masked`). X-10 is checked in every scenario × planner × allocator × shield combination: $\mathcal{K}^{\text{ctrl}}_t = \mathcal{K}^{\text{ref}}_t$ at every step and $N^{\text{avoided}} = N^{\text{introduced}} = 0$.

**Rule-based baseline.** For each robot, with target $c = x_i + \Delta(\hat a^{\text{plan}}_i)$:

```text
if no valid plan or at goal:                                       stay
elif c ∈ occupied cells (robots, obstacles) or c ∈ Ẑ_i(t):         stay
elif ∃ robot j adjacent to c whose suggested target is c and
     (π̃_j > π̃_i or (π̃_j = π̃_i and j < i)):                      stay    # yield right-of-way
else:                                                              move to c
```

Here $\tilde\pi$ is the effective priority of the carried or assigned task, with idle lowest. The baseline reads neighbours' *suggested* actions, which is available because planning is centralised. It is still a heuristic, since neighbours controlled by learned policies may not follow their suggestions. ~~Deadlocks are handled by the controller-agnostic stall trigger T4, not by a baseline-specific rule.~~ *Revised in Issue #4:* T4 does not resolve mutual blocking: the symmetric replans of two blocked robots meet again (stage 5A measurement). Deadlocks are detected and, where a passing cell exists, resolved by the opt-in coordinator-level resolver of §5.5 (trigger T7), enabled for the baseline by default.

**Complexity.** $O(N)$ per step with the cell → robot map.

**Limitations (measured, not hidden).** Unpredictable obstacle turns can still produce $\textsf{RO}$ conflicts. ~~Corridor deadlocks are mitigated by T4, not eliminated.~~ *Revised:* corridor deadlocks are resolved only where a passing cell exists and under the progress assumptions of §5.5; otherwise they are reported as unresolved.

**Tests.** Yielding to a higher priority; tie-break by id; no $\textsf{RR}$ conflicts among baseline robots (property test); ~~head-on corridor case resolved via T4~~ *revised:* head-on corridor: resolved by the §5.5 detour with two lanes, reported unresolved with one lane.

**Implementation notes (Issue #4, stage 5A: baseline right-of-way).**
- `control/baseline.py`: `BaselineController(alpha=0.01, sensing_radius=3)`, `name = "baseline"`, implementing the `Controller` protocol (`reset`, `act`). `decide(view)` returns each robot's action with a decision reason.
  - The rule is the pseudocode above, applied in this order: no plan or at the goal; masked; target occupied by a robot or an obstacle; target in $\hat Z_i$; yield; move.
  - $\hat Z_i$ comes from `planning.constraints.sense`, still the only definition of $Z_i$ and $\hat Z_i$. Its map argument is now typed by a protocol (`FreeCellQuery`), so the read-only `MapView` can be passed; the behaviour is unchanged.
  - No deadlock-detour rule (stage 5B), no reference follower, no shield, no X-10.
- **Right-of-way.** Every unmasked, non-stay suggestion claims its target cell $c$. The winner of $c$ is the claimant with the largest $(\tilde\pi_j, -j)$: priority first, then the lower id. This is the pseudocode's "∃ $j$ adjacent to $c$ whose suggested target is $c$ and $(\tilde\pi_j > \tilde\pi_i$ or $(\tilde\pi_j = \tilde\pi_i$ and $j < i))$", since every claimant of $c$ is adjacent to it. A masked suggestion claims nothing; its target is a static obstacle, which every other claimant would see as masked too.
- **$\tilde\pi$ (interpretation, stated here).** The aged weight $w(\pi_j)(1 + \alpha\max(t - \rho_j, 0))$ of the robot's assigned or carried task, with the allocation's $\alpha$; idle robots get $0$, below every task, since $w \ge 1$.
  - The age keeps growing after pickup, as in the engine's `Task.age`. Model §4 defines the waiting age only until pickup, and the controller has no pickup time in its view.
- **Collision property (Guaranteed under the assumptions; property-tested).**
  - *Assumptions.*
    - (a) Every robot plays the baseline on the same view.
    - (b) Obstacle moves follow the simulator's rule: never into a cell occupied at $t$ by a robot or an obstacle, never a swap (model §2, `resolve_obstacle_moves`).
    - (c) For the RO-vertex part only: each obstacle either stays or moves by its velocity, and $r_s \ge 2$.
  - *Claim.* Under (a) and (b), the joint proposal contains no $\textsf{RRv}$, $\textsf{RRs}$, $\textsf{ROs}$ or invalid move. Under (a)–(c) it contains no conflict at all.
  - *Proof.*
    - Invalid moves: only unmasked suggestions are played.
    - A robot never moves into a cell occupied at $t$ by a robot or an obstacle. That excludes $\textsf{RRs}$, $\textsf{ROs}$, and an $\textsf{RRv}$ against a robot that stays.
    - Two robots moving into the same cell would both be claimants of it, but only the winner moves.
    - For $\textsf{ROv}$, an obstacle reaching $c$ at $t+1$ must come from an adjacent cell $z \neq c$. Under (c) that means $z + v = c$. Then $\|z - x_i\|_\infty \le 2 \le r_s$, so the obstacle is sensed and $c \in \hat Z_i$ ($c$ is free), and robot $i$ stays. ∎
  - *What it does not cover.* Obstacles that turn or start moving, and obstacles beyond $r_s$, can still give $\textsf{ROv}$. Teams with robots that do not follow their suggestions (learned controllers) break assumption (a). Conflict-freedom is not progress: see the deadlocks below.
- **Statistics:** `BaselineController.episode_stats()` reports decisions per reason (`move`, `no_plan`, `masked`, `robot_ahead`, `obstacle_ahead`, `predicted_obstacle`, `yield_priority`, `yield_tie`), yields, active robot-steps and `wait_fraction` (waiting share of active robot-steps).
  - The engine already records conflicts by type, blocked moves, plan adherence (waiting lowers it) and deviation (waiting on the path is not T3).
  - Writing the controller statistics into `EpisodeMetrics` and the CLI belongs to stage 6.
- **Limitations (measured, not hidden).**
  1. *No progress guarantee; head-on encounters are not resolved.*
     - Two robots facing each other both wait (`robot_ahead`), and so do robots in a cycle. With two robots carrying tasks to opposite ends of a corridor, nothing is delivered in 60 steps, **even with a second lane**.
     - T4 fires for *both* robots at the same step, both replan symmetrically and they meet again. The test this section expects, "head-on corridor case resolved via T4", is therefore **not met**; recorded in `tests/integration/test_baseline_integration.py`. T4 can resolve asymmetric cases: in `planning_detour` (A\*, auction) only one robot's T4 fired, and the robots passed.
     - A baseline-specific deadlock rule is open for stage 5B. T4 is not that rule. *Stage 5B (implemented, §5.5):* with the resolver on, the two-lane case is resolved; the single-lane case is reported as `no_passing_place` (no cell lies off the other robot's path). The Stage 5A observation (resolver off) is unchanged and still tested.
  2. *Conservative following.* A robot waits behind a robot that is leaving its cell, because the cell is occupied at $t$. Convoys therefore move with gaps, one step lost per robot. With map events timed against a convoy this changes outcomes: in `corridor_queue` (sequential) the deferred wall falls between the robots, giving 1/3 delivered against the follower's 3/3.
  3. *Idle robots never move aside* (no parking in the controller; endpoint recovery, §5.6, adds parking for active endpoints). An idle robot on a corridor or on another robot's goal blocks it until T4 finds a detour, if one exists.
     - An idle robot *on the goal cell* blocks that task for good: T4 never targets the goal, and the occupied-goal policy is to wait.
     - The deadlock resolver (§5.5) does not detect this: an idle robot has no plan, so it forms no wait-for cycle.
     - Measured in the Issue #4 system comparison (`05` §B.6): 28 of the baseline's 91 undelivered tasks.
     - *Addressed by endpoint recovery (§5.6)* for an idle robot on an active task endpoint (parking). An idle robot on a corridor cell that is no endpoint is still not moved.
  4. *$\textsf{ROv}$ from unpredicted obstacle moves:* 1–2 per episode in 4 of the 20 demo runs.
  5. *Waiting costs throughput.* In `planning_detour` (A\*, auction) the baseline delivers 1/2 in 80 steps, against 2/2 for the follower, which relied on 12 conflicts cancelled by the arbiter.
  6. *Robot–obstacle goal deadlock* (added after the Project Issue #4 audit; the same mechanism as E2 cause 2). A moving obstacle on the robot's goal whose next move is the robot's own cell never moves: obstacles never enter robot cells (model §2). The robot waits in front of it (`obstacle_ahead`), as the occupied-goal policy of model §5.3 prescribes. Neither moves again.
     - Neither the baseline nor the deadlock resolver handles this: it is not a robot–robot cycle.
     - Measured in the Issue #4 system comparison: 61 of the baseline's 91 undelivered tasks (54 with patrol obstacles, 7 with random-walk obstacles).
     - *Addressed by endpoint recovery (§5.6)*, the robot-side retreat; obstacle motion (A-5) is unchanged. With `--endpoint off` the limitation is as described here.
- **Tests.**
  - `tests/unit/control/test_baseline.py`: each rule on hand-built views (mask, occupied, predicted, sensing radius); vertex conflicts by priority, by id and idle-lowest; aging breaking equal levels; 3 and 4 robots competing for one cell; swap and 4-cycle; conservative following; statistics; validation.
    - Property tests against the simulator's own `detect_conflicts` and `resolve_obstacle_moves` (400 examples each): no conflict at all under (c); only $\textsf{ROv}$ with a turned obstacle otherwise. A non-vacuity check shows the plain follower does produce RR conflicts on the same worlds.
  - `tests/integration/test_baseline_integration.py`:
    - All 5 demo scenarios × {A\*, RRT\*} × {auction, sequential}: no $\textsf{RR}$, $\textsf{ROs}$ or invalid move.
    - A junction with 2 robots (priority, id tie) and with 3 robots: pass order equals right-of-way order; no conflicts; all delivered; adherence below 1; no T3.
    - Head-on in 1 and 2 lanes: conflict-free, not resolved, symmetric T4.
    - Byte-identical traces and statistics on repeated runs.

### 5.4 Safety shield (optional)

```text
SHIELD(a_i):  c ← x_i + Δ(a_i)
              if a_i ≠ stay and c ∈ Z_i(t) ∪ Ẑ_i(t): record (t, SH, i); return stay
              return a_i
```

**Properties.**
- **Guaranteed:** it never introduces a move, only replaces moves by $\textsf{stay}$, so it cannot create $\textsf{RR}$ or $\textsf{RO}$ conflicts in which the shielded robot is at fault.
- **Not claimed:** elimination of $\textsf{RO}$ conflicts, since predictions are one-step heuristics.
- Robot–robot conflicts are outside its scope.

**Complexity.** $O(1)$ per robot.

**Tests.** Replacement cases; a no-op on a safe action; interventions logged; disabled mode is the identity.

**Implementation notes (Issue #4 stage 6).**
- `control/shield.py`: `Shield(mode="off"|"predictive", sensing_radius=3)`.
  - `apply(view, proposal)` returns $a^{\text{arb}}$ and `ShieldEvent`s $(t, i, a_i, c, \text{reason})$, where the reason is `current` ($c \in Z_i$) or `predicted` ($c \in \hat Z_i \setminus Z_i$).
  - $Z_i, \hat Z_i$ come from `planning.constraints.sense`, the single shared definition.
- The engine applies it at pipeline step 5 (`Engine(..., shield=...)`, `SafetyLayer` protocol); `off` is the identity.
  - Conflicts are counted on $a^{\text{arb}}$. The counterfactual sets use $a^{\text{ctrl}}$, the raw proposal (model §9.2).
  - Shield statistics, under `extra["shield"]` only when a shield is configured: interventions, interventions per 100 robot-steps, prevented $\sum|\mathcal{K}^{\text{ctrl}} \setminus \mathcal{K}^{\text{arb}}|$, introduced $\sum|\mathcal{K}^{\text{arb}} \setminus \mathcal{K}^{\text{ctrl}}|$.
  - Trace fields `shield` (the interventions) and `arb_actions` appear only on steps with an intervention.
- **Properties, tested against the simulator's own detector:**
  - Every output action equals the input or `stay` (400 random worlds, arbitrary simultaneous proposals of all robots).
  - No shielded robot is ever `at_fault`. Its `stay` can only make it the *target* of another robot's move, which counts as introduced, at the mover's fault.
  - With constant-velocity obstacles and $r_s \ge 2$, the shielded plain follower proposes no RO conflict (400 worlds). With arbitrary obstacle turns, RO vertex conflicts still occur (counter-example found).
  - RR vertex and swap proposals and invalid moves pass unchanged, so masking defects stay visible.
- **With the baseline it never intervenes** (same $r_s$): the baseline already stays in front of current and predicted obstacle cells. Checked in all 5 demo scenarios × 2 planners × 2 allocators.
  - Demo `shield_corridor`: the reference follower proposes 37 RO conflicts with the shield off. With it on there are 37 interventions and 0 conflicts. The obstacle pins itself in the single lane, so nothing is delivered in either case.

### 5.5 Deadlock detection and detour resolution (Issue #4 stage 5B)

**Purpose.** Resolve mutual blocking that the right-of-way baseline makes conflict-free but not live: a head-on pair, or a rotation of robots each in the next one's way. T4 is a planning-monitor stall trigger, not a resolution rule: for two mutually blocked robots it fires symmetrically and their replans meet again (measured, stage 5A).

**Placement (approved design).** `planning/deadlock.py`, `DeadlockResolver(DetourParams)`, used by `AllocationCoordinator(..., deadlock=DetourParams())`.
- It is **opt-in** and off by default. The CLI enables it exactly for `--controller baseline` with an allocator (`--deadlock auto`).
- It never chooses actions. For one robot of a cycle it replaces the goal given to the path monitor by a *passing cell*. The planner (A\* or RRT\*) plans there, and the controller keeps all its rules.

**Definitions.**
- Robot $i$ is *active* if it is on a valid plan and not at its goal; its next planned cell is $n_i$.
- *Wait-for graph:* $i \to j$ iff $i$ is active and $n_i = x_j$. Out-degree is ≤ 1, so cycles are vertex-disjoint and found in $O(N)$. The 4-connected grid is bipartite by the parity of $x + y$, so every cycle has even length: a head-on pair, or a rotation of ≥ 4 robots.
- *Temporary waiting:* waiting that is not part of a detected deadlock. This includes queues, right-of-way yields, obstacle waits, and blocking by an idle robot or by a robot at its goal (not a cycle; idle robots on active endpoints are parked by endpoint recovery, §5.6, not by the resolver).
- *Repeated mutual blocking:* the same member set forms a new cycle again within `recur_window` steps.
- *Detected deadlock:* the same cycle (members and cells) for ≥ `detect_steps` consecutive steps (*persistent*), or the `recur_count`-th occurrence within the window (*recurrent*).
- *Unresolved:* no passing cell exists even with obstacles and robots outside the cycle ignored (`no_passing_place`), or the attempt budget is exhausted (`unresolved_retry_limit`). It is reported once and re-evaluated only after a map change or a member's goal change.

**Algorithm.**
1. *Yielder.* Members in ascending $(\tilde\pi_i, -i)$ order: lowest effective priority first; on a tie the higher id yields, as in the baseline. The first with a passing cell yields. A higher-priority robot can therefore yield when the lower one is boxed in; priority is a preference, not strict precedence.
2. *Passing cell* $s$: BFS from the yielder over static free cells, with every robot and obstacle cell blocked. Take the nearest cell that is (P1) on no other member's remaining path, (P2) not an obstacle's constant-velocity prediction, (P3) not reserved by another detour, and not the yielder's own task goal; ties by $(x, y)$.
3. *Detour* (state machine): PENDING → TO_WAYPOINT. The goal becomes $s$ (T7); the other members' cells are added as avoided cells for the detour plan; T4 is suspended for every member. The yielder reaches AT_WAYPOINT, then is **released**: the goal returns to the task goal (T7, normal replanning, the original route is not replayed). Release happens when every other member is at its goal, or is off $P_y^0$ (the yielder's route at detection) and its next `release_horizon` cells avoid $P_y^0$. Otherwise a timeout after `hold` $= 2|P_w| + $ `hold_extra` steps at $s$ releases it.
   - **Abort** (and re-detection while the cycle persists): $s$ blocked, another member's plan runs through $s$, the detour plan is `unreachable`, or no progress for `abort_steps`.
   - **Cancel:** the yielder's task goal changes (allocation pre-emption, pickup).
   - **Attempt bound:** at most 1 + `max_retries` detours per (member set, member task goals). A map change resets budgets and unresolved marks.
4. *Defaults* (`DetourParams`): `detect_steps` 2, `recur_count` 3, `recur_window` 20, `abort_steps` 4, `hold_extra` 4, `max_retries` 2, `release_horizon` 5, unbounded search radius, $\alpha$ 0.01.
   - The coordinator validates `detect_steps` < `stall_threshold` (4), so that detection precedes T4.

**Changes to the reviewed design, found during implementation (documented consequences).**
1. *Release horizon.* The reviewed rule required the winner's *whole* remaining path to avoid $P_y^0$. In `rooms_dynamic` this held a yielder for 16 steps after the conflict had dissolved, because both routes ended in the same region. The rule now looks only `release_horizon` cells ahead.
   - Consequence: a later crossing far ahead is a *new* encounter, detected and handled again, not covered by the first detour.
2. *Attempt bound counts starts.* The reviewed bound counted only aborts. With the earlier release, a detour could complete and the cycle re-form indefinitely.
   - Consequence: detours per (member set, goals) are bounded by $1 + $ `max_retries`, so livelock across detours is bounded per goal configuration. It is not excluded: new tasks give a fresh budget.
3. *Static vs dynamic absence of a passing place.* The reviewed design marked "no passing place" unresolved until a map or goal change. A dynamic obstacle temporarily occupying the only pocket then disabled resolution permanently (found in a constructed pocket scenario).
   - Now: if a cell exists with obstacles and non-member robots ignored, the event is `no_passing_place_dynamic`, re-checked every `abort_steps`; only static impossibility is marked unresolved.
4. *Release policy.* An `unreachable` result of a *detour* plan does not trigger the model §5.3 task release (it aborts the detour instead). Otherwise the yielder would lose its task.
5. *Trace stability.* `T7` appears in `calls_by_trigger` only when the resolver is enabled. With the resolver off, traces of 46 configurations are byte-identical to the pre-5B snapshot.

**Safety (G21, checked against the code).** The stage 5A proof (§5.3 notes) uses only the controller's rules 1–6, occupancy and predictions at $t$, and the single winner per contested cell. It holds for *arbitrary* suggested actions; the property tests draw them at random. The resolver changes only monitor goals, hence only suggestions, and never an action. So, under assumptions (a)–(c) of §5.3, the joint proposal still has no RR, ROs or invalid move, and none at all under (c). No assumption is added or weakened for safety. In tests: 0 RR / ROs / INV with the resolver on in every demo scenario × planner × allocator, and in all 600 Issue #4 system-comparison episodes.

**Progress (G22, conditional; two robots only).** Re-examined against the code in the Project Issue #4 audit; R7–R9 were implicit before and are now stated.

The claim: a *detected* deadlock of exactly two robots is resolved in finitely many steps when all of R1–R9 hold. The table gives what the code does when one fails.

| # | Assumption | If it fails, the resolver … |
|---|---|---|
| R1 | no map event during the detour | aborts if $s$ becomes blocked. Every map change clears attempt budgets and unresolved marks, and the cycle is detected again. |
| R2 | no obstacle enters $s$ or the yielder's route | lets the yielder wait (baseline `obstacle_ahead` / `predicted_obstacle`) and aborts after `abort_steps` without movement; re-detection follows |
| R3 | no robot outside the pair enters $s$ or the route | same as R2 (`robot_ahead`) |
| R4 | the winner's path is unchanged until it passes | aborts if the winner's new path crosses $s$; otherwise release uses the current paths |
| R5 | a passing cell exists | reports `no_passing_place` (static: once, until a map or goal change) or `no_passing_place_dynamic` (re-checked every `abort_steps`) |
| R6 | the winner's goal is not on the yielder's remaining route | *nothing*: once the winner is idle at its goal it has no plan, so no cycle exists. The yielder is then blocked by an idle robot, which is neither detected nor reported (limitation 3 of §5.3). |
| R7 | the planner finds a path to $s$ (A\*: guaranteed, complete on the snapshot; RRT\*: not, finite budget) | gives the yielder no plan: no movement, abort after `abort_steps`, re-detection; the attempt bound ends in `unresolved_retry_limit` |
| R8 | both robots run the baseline (assumption (a) of §5.3) | with other controllers the collision property G19 does not apply; physical collisions are still impossible (G1). The progress sketch below does not apply. |
| R9 | neither robot's task goal changes during the detour | cancels the detour; the new goals give a fresh attempt budget |

- *Sketch (under R1–R9).*
  - The yielder's next cell on its route to $s$ is never occupied: the winner is behind it or waiting (R2, R3). The two never target the same cell, because the winner targets the yielder's current cell. So the yielder reaches $s$ in $d$ steps (R7).
  - $s$ is off the winner's path (P1), so the winner then advances at least one cell every two steps (conservative following) and leaves $P_y^0$ (R4). Release follows within the horizon, and the yielder replans with the winner behind it.
- **Detection scope.** Only robot–robot wait-for **cycles** are detected. These are *not* detected and *not* reported:
  - a robot blocked by an idle robot, or by a robot at its goal;
  - a robot waiting for a dynamic obstacle on its goal (the robot–obstacle goal deadlock, limitation 6 of §5.3);
  - a chain ending at a robot that waits for an obstacle.

  So `unresolved_at_end` counts only unresolved robot–robot cycles. It is **not** a count of stuck robots: the Issue #4 system comparison had 0 unresolved cycles but 89 tasks stuck on idle robots or obstacles (`05` §B.6).
  The first two cases, when the blocked cell is the robot's *task goal*, are handled by endpoint recovery (§5.6), which reports its own unresolved cases.
- **Cases it cannot resolve (by design or by construction):**
  - (i) head-on encounters with no cell off the other robot's path, e.g. a single lane with goals at both ends: `no_passing_place`;
  - (ii) R6 and the idle-robot cases above (not detected);
  - (iii) robot–obstacle goal deadlocks (not detected);
  - (iv) an obstacle pinned permanently in the only passing place (found in a constructed pocket scenario): classified `no_passing_place_dynamic` and re-checked every `abort_steps` for the rest of the episode, never reported unresolved;
  - (v) a yielder that keeps *moving* without approaching $s$, e.g. alternating T2 replans: `abort_steps` measures movement, not distance to $s$, so no timeout applies before arrival. Theoretical; not observed in any test or experiment;
  - (vi) repeated re-formation of a cycle: bounded by 1 + `max_retries` detours per (member set, goals), but every task change gives a fresh budget.
- *Not claimed:* that every deadlock is resolvable; progress for cycles of ≥ 4 robots in general; absence of livelock beyond the attempt bound; completion.
- **What the 4-robot rotation tests establish, and only this:**
  - Configuration: one geometry, a 2 × 2 block of robots in an open 4 × 4 room, each robot's goal two cells straight ahead through the next robot's cell; the **auction allocator only**; A\* and RRT\* (400 iterations); no obstacles or map events; default `DetourParams`.
  - Priority patterns (1,1,1,1), (3,3,3,1) and (1,3,3,3): the 4-cycle is detected at $t = 2$, the expected lowest-$(\tilde\pi, -i)$ robot yields to the expected passing cell, and all four deliver with 0 conflicts.
  - Not established: other geometries (corridors, rooms with doors), cycles of 6 or more, simultaneous cycles plus third robots, dynamics, the sequential allocator, or the general draining argument.

**Complexity.** Detection $O(N)$ per step. Per new detour, one BFS $O(V)$ plus $O(L)$ path-set checks plus one ordinary planner call. Release checks $O(\text{release\_horizon})$ per member per step.

**Tests.**
- `tests/unit/planning/test_deadlock.py` (17):
  - cycle finding; passing-cell nearest, tie, blocked, forbidden, radius;
  - persistent threshold exactly; recurrent detection; yielder by priority, by id, with fallback;
  - predicted and reserved cells excluded; two simultaneous cycles get distinct cells;
  - release horizon; the four abort reasons; the attempt bound and a fresh budget on new goals;
  - cancellation; timeout; map-change clearing; validation, including `detect_steps` < `stall_threshold`.
- `tests/unit/planning/test_monitor_detour.py` (4): T7 in and out with one unsplit leg; cancellation to a new goal is T1 and a new leg; T4 held and resumed; avoided cells shape the plan.
- `tests/integration/test_deadlock_integration.py` (42):
  - two-lane head-on resolved for {A\*, RRT\*} × {auction, sequential} × {unequal, equal priority}: lower priority, or higher id, yields; T7 at $t = 6$ and $10$; no T4; 4 legs; 0 conflicts;
  - single lane: `no_passing_place`, nothing forced, no T7;
  - 4-robot rotation resolved for 3 priority patterns × both planners (one geometry, auction only; scope above);
  - map event on the passing cell: abort, re-detection, new cell, resolution;
  - an obstacle delaying the detour (2 waits, completed);
  - the collision property in 20 configurations; determinism; trace events; off-by-default.
- **Measured effect on the demo scenarios** (baseline, 20 runs): total deliveries 89 with the resolver and 89 without. Gains in `rooms_dynamic` RRT\*/sequential (4 → 6) and `random_warehouse` RRT\*/auction (9 → 10); losses in `rooms_dynamic` A\*/auction (7 → 5) and `planning_detour` A\*/auction (1 → 0), from time spent detouring within the step caps. Reported as measured; not tuned.

### 5.6 Endpoint recovery: idle-robot parking and retreat from a held goal (Project Issue #4 follow-up)

**Why.** The Issue #4 system comparison traced 89 of the baseline's 91 undelivered tasks to a task goal held *permanently* (`05` §B.6):
- **Robot–obstacle goal deadlock (61 tasks).** An obstacle stands on the goal, and its next move is the cell of the robot waiting next to it. In 54 cases this was a patrol obstacle, whose shortest-path step is fixed; in 7 cases a random walk on a dead-end goal whose only exit was the robot. Obstacles never enter robot cells (A-5), and the robot waits for its goal (model §5.3), so neither moves again.
- **Idle robot on the goal (28 tasks).** A robot that delivered there, or stopped there for another reason, stays forever: there is no parking.

The planner was correct: the goal is never dynamically blocked (`constraints.py`, model §5.2). Model §5.2 called an obstacle on a goal "transient", but nothing in the model made it so. That was a specification gap, now closed by a robot-side rule. Obstacle motion (A-5) is unchanged.

**Placement.** `planning/endpoint.py`, `EndpointRecovery(EndpointParams)`, used by `AllocationCoordinator(..., endpoint=EndpointParams())`.
- *Defaults:* on in the CLI with an allocator, for every controller (`--endpoint off` disables it), in the benchmark workloads, and in the recovery configurations of the Issue #4 system comparison (`e5_recovery*.toml`).
- *Opt-in constructor:* the bare constructor keeps it off, as with `deadlock=`, so that the stage-level tests keep pinning each stage's exact behaviour.
- *Mechanism:* like the resolver, it never chooses actions. It replaces the goal given to the path monitor by a temporary cell; this goal change is trigger **T8**, the current leg is not split, and the task goal is restored with T8.
- *Composition:* the resolver has precedence. A robot the resolver overrides is not in endpoint recovery, and its retreat or parking is cancelled. Resolver passing cells count as reserved cells here.

**Definitions.**
- *Active endpoints* $E(t)$: pickup and drop-off of released tasks that are pending or assigned, and drop-off of picked tasks.
- *Occupant* of a goal $g$: a dynamic obstacle on $g$, or an idle robot on $g$. A busy robot is not an occupant; robot–robot cycles belong to the resolver.
- *Blockage episode* of robot $i$: starts at the first step $i$ is adjacent to its task goal $g$ while $g$ has an occupant. It ends when that occupant leaves $g$ or $i$'s task goal changes.
  - It is **temporary** if it ends before $i$ has waited $W$ consecutive steps adjacent to $g$; otherwise it is **persistent**.

**Algorithm.**
1. *Parking.* An idle robot (after this step's allocation) standing on a cell of $E(t)$ gets a parking goal. The cell is the nearest one, by BFS from the robot over static free cells not occupied by a robot or an obstacle, that is:
   - (K1) not in $E(t)$;
   - (K2) not reserved (another parking or retreat cell, or a deadlock passing cell);
   - (K3) not a **cut vertex** of the free graph without the other stationary idle robots and the reserved cells. A parked robot never disconnects the map, and a single-access endpoint's entrance is never used;
   - (K4) not occupied.

   Ties are broken by $(x, y)$. The robot plans there with the configured planner (A\* or RRT\*), avoiding the cells of robots within distance 2 at the start (typically the ones waiting for the endpoint). Its priority stays 0, so every task-holding robot has right of way.
   - *Ends:*
     - **completed** at the cell;
     - **cancelled** by a task assignment or by the resolver;
     - **aborted** after `abort_steps` without progress, a statically blocked cell, a cell that became an active endpoint, or an `unreachable` plan. A new cell is searched `abort_steps` later.
   - *No cell:*
     - `no_parking_cell` when none satisfies K1–K3 in the robot's static component. Re-checked only after a map change or a change of $E(t)$;
     - `parking_blocked_dynamic` when such a cell exists but none can be reached now past robots and obstacles. Re-checked every step.
2. *Retreat.* When a blockage episode becomes persistent, the robot retreats.
   - *Retreat cell:* the nearest cell, by BFS from the robot over static free cells not occupied by robots or obstacles within `retreat_radius`, at Manhattan distance ≥ 2 from the occupant **and** from the robot's current cell. Those are the cells the occupant could step into next. The cell must also be outside $E(t)$, not reserved, and not an obstacle's constant-velocity prediction; ties by $(x, y)$.
   - *Movement:* the robot plans there (T8) with T4 suspended.
   - *Outcomes of an attempt:*
     - **succeeded** as soon as the occupant leaves $g$; the goal is re-assessed every step, and the task goal is restored (T8) and replanned;
     - **timed out** after $H$ steps at the retreat cell; the task goal is restored and the robot may wait again;
     - **aborted** after `abort_steps` without progress, a statically blocked retreat cell, or an `unreachable` plan (which does *not* release the task);
     - **cancelled** when the task goal changes or the resolver takes the robot over.
   - *Exhaustion:* at most $M$ attempts per blockage episode. When the robot has waited $W$ steps again after the last one, the episode is **exhausted**: the robot keeps waiting, and the episode is reported once.
3. *Parameters* (`EndpointParams`, fixed before any validation or test run; not tuned):
   - $W$ = `wait_steps` = 4, the monitor's $T_{\text{stall}}$, the model's existing bound for "not momentary";
   - $H$ = `hold_steps` = 8 = $2T_{\text{stall}}$;
   - $M$ = `max_attempts` = 3;
   - `retreat_radius` = 8;
   - `abort_steps` = 4, as the resolver.

**Counters** (`extra["planning"]["endpoint"]`; every event counted once):

| counter | counts |
|---|---|
| `blockages_temporary` | blockage episodes that ended before $W$ consecutive waiting steps (counted when they end) |
| `blockages_persistent` (`_obstacle`, `_robot`) | blockage episodes that reached $W$ (counted once per episode, by occupant kind) |
| `retreats_started` | retreat attempts, i.e. recovery initiated (one per attempt) |
| `retreats_succeeded` / `_timed_out` / `_aborted` / `_cancelled` | attempt outcomes; `retreats_started` = their sum + `retreats_active_at_end` |
| `attempts_exhausted` | persistent episodes whose $M$ attempts were used (once per episode) |
| `no_retreat_cell` | persistent episodes in which a retreat search found no cell (once per episode; the search is repeated every $W$ steps) |
| `goal_reached_after_recovery` | robots that reached the goal (pickup or delivery) after a successful retreat for it |
| `parking_needed` | stays of an idle robot on an active endpoint (once per stay) |
| `parking_started` / `_completed` / `_cancelled` / `_aborted` | parking trips; `parking_started` = completed + cancelled + aborted + `parkings_active_at_end` |
| `no_parking_cell`, `parking_blocked_dynamic` | stays without a safe cell (static) / without a reachable one (dynamic), once per stay each |
| `retreat_robot_steps`, `parking_robot_steps` | robot-steps spent with a retreat / parking goal |
| `unresolved_obstacle_at_end`, `unresolved_idle_robot_at_end` | persistent blockages still in place when the episode ends |
| `idle_on_endpoint_at_end` | idle robots standing on an active endpoint at the end |

The diagnostics of the Issue #4 system comparison (`experiments/e5_diagnostics.py`) attach the holder's recovery state to every undelivered `goal_obstacle` / `goal_idle_robot` item. For `goal_idle_robot` items they also attach the occupant's parking state. No undelivered task is left without its recovery outcome.

**Safety (G25, guaranteed, argued against the code).**
- Endpoint recovery changes only monitor goals, and therefore only suggested actions. It never changes an action, the occupancy rules, the arbiter or obstacle motion.
- The arbiter's Lemma 3.1 holds for any joint proposal, so no physical collision is possible.
- The baseline's collision property (G19, §5.3) holds for *arbitrary* suggested actions, so with the baseline there are still no RR, ROs or invalid proposals under its assumptions.
- The shield, when on, still only replaces moves by `stay`.
- *Tested:* 0 physical collisions in every endpoint test; 0 RR/ROs/INV for the baseline in every planner × allocator × shield combination of `tests/integration/test_endpoint_integration.py` and `test_cli_matrix.py`; the recovery runs of the Issue #4 system comparison (`05` §B.6).

**Progress (not claimed in general; conditional for the simplest cases).**
- *Parking:* an idle robot leaves an active endpoint if a safe cell exists and its route is not blocked. Its priority is 0, so busy robots pass first.
- *Retreat:* it removes the robot from the cells the occupant can enter next.
  - A *patrol* obstacle whose next shortest-path cell was the robot's then moves on. That cell is never the retreat cell, and is not adjacent to it.
  - A *random walk* then moves to the freed exit with probability ≥ 1/5 per step.
  - Neither implies delivery: the obstacle may re-enter the goal, block the corridor between the robot and the goal, or the robot may lose the race for the goal.
- **Unresolvable cases (reported, not hidden):**
  - (i) an occupant that never leaves, e.g. a one-waypoint patrol parked on the endpoint (exhausted after $M$ attempts, `unresolved_obstacle_at_end`);
  - (ii) no retreat cell within the radius, e.g. a 1-wide corridor whose far end is the goal (`no_retreat_cell`);
  - (iii) no safe parking cell, e.g. a small region whose free cells are all endpoints or cut vertices (`no_parking_cell`), or one reachable only through the waiting robot (`parking_blocked_dynamic`). If that robot also has no retreat cell, both stay (constructed test `NO_PARKING`);
  - (iv) after a successful retreat, an obstacle that moves into a 1-wide approach corridor and waits for the robot: a robot–obstacle head-on outside the goal, which neither the resolver nor this rule handles;
  - (v) idle robots blocking *routes* (not endpoints), and contests between busy robots on free cells, are out of scope.
- *Not claimed:* that every endpoint blockage is resolved, or that every task is delivered.

**Complexity.** Per step: $O(N + K + |J|)$ to build $E(t)$ and track blockages. Each parking search costs one BFS plus one cut-vertex computation (iterative Hopcroft–Tarjan, $O(V)$, cached per grid version and removed set). Each retreat search costs one BFS bounded by the radius. Searches run only on trigger events.

**Tests.**
- `tests/unit/planning/test_endpoint.py` (21): parameters and validation; active endpoints by status; nearest cell (ties, radius, blocking); cut vertices against a brute-force component count (Hypothesis, 60 grids) and by hand.
  - *Parking:* nearest safe cell, completion, cancellation, abort and retry, static and dynamic absence, stationary robots and reserved cells.
  - *Retreat:* temporary vs persistent occupancy, the expected retreat cell, success and restore (T8), timeouts, exhaustion and the end report, no retreat cell, cancellation (goal change, resolver), abort, an idle-robot occupant.
  - Counter keys and events; parking robot-steps excluded from the deviation-rate denominator (`count_active`).
- `tests/integration/test_endpoint_integration.py` (31): endpoint reuse after delivery (both controllers; off → `goal_idle_robot`, on → delivered and parked); shared endpoints between tasks; identical pickup/drop-off rejected by the model; no parking and no retreat cell reported; a patrol held on a pickup and on a drop-off (off → `goal_obstacle`, on → retreat succeeds, delivered); a crossing obstacle cleared without recovery; a never-leaving obstacle exhausted and reported; all planner × allocator × controller × shield combinations; `e5_scenario` development-seed episodes (both controllers, shield off and on): no collision, baseline RR = 0, counter identities; off = the earlier coordinator; determinism with RRT\*.
- `tests/integration/test_cli_matrix.py`: the 16 combinations with `--endpoint` on and off; option validation and text output.
- `tests/integration/test_endpoint_evidence.py` (7, one slow): the paired benchmark (identical trajectories, statistics, CLI validation, samples), the recovery configurations (`e5_recovery*.toml`), the paired recovery comparison, recovery-aware diagnostics.

**Metric definitions with recovery.** Unchanged. Their measured values can move as follows:
- `robot_moves` and path efficiency include retreat and parking moves; a retreat, like a T7 detour, stays inside its leg.
- Wait ratio counts holding at a retreat cell as waiting.
- Utilisation and plan adherence use robot modes, so parking (an idle robot) is excluded from both.
- The deviation rate keeps "per 100 non-idle robot-steps": parking steps and their T3 calls are excluded (`RobotInput.count_active`).
- Mean flow time is still averaged over delivered tasks only, so it can rise when recovery delivers tasks that had waited long.

---

## 6. MADQL: parameter-shared Double DQN for multiple agents

**Purpose.** The first required MARL algorithm (FR-RL-2). Design decision ADR-3. **Exact name of the variant:** *parameter-shared independent Double DQN with team-mixed rewards and action masking.*

**References.** Mnih et al. 2015 (DQN); van Hasselt et al. 2016 (Double DQN); Tampuu et al. 2017 and Gupta et al. 2017 (multi-agent DQN with parameter sharing); Foerster et al. 2017 (fingerprints).

**Model.** $Q_\theta : \mathbb{R}^{d_o} \to \mathbb{R}^{|\mathcal{A}|}$, an MLP $356 \to 256 \to 256 \to 5$ with ReLU (≈ 160 k parameters), shared by all robots. Target network $Q_{\theta^-}$.

**Loss.** Minibatch $\mathcal{B}$ of individual transitions $(o, a, r, o', m', d)$, where $m'$ is the next action mask and $d = 1$ only for **termination**, not truncation:

$$
y = r + \gamma (1 - d)\, Q_{\theta^-}\Big(o',\, \arg\max_{a' \in m'} Q_\theta(o', a')\Big), \qquad
\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{\mathcal{B}} \text{Huber}\big(y - Q_\theta(o, a)\big).
$$

**Algorithm.**

```text
initialise θ, θ⁻ ← θ, replay D, ε ← ε_start
for env step t = 1 … T_train (E vectorised envs, N robots each):
    every robot: a_i ← ε-greedy over masked Q_θ(o_i, ·)           # one batched forward pass
    step all envs (execution layers apply); store (o_i, a_i, r_i, o′_i, m′_i, terminated) per robot
    if |D| ≥ warmup and t mod train_freq = 0: minibatch → Double-DQN masked targets → Adam step (grad-clip 10)
    if t mod target_period = 0: θ⁻ ← θ
    decay ε linearly; periodic validation and checkpoint
```

The stored action is the controller's raw choice. If the shield is on and replaced it, the stored transition still records the raw action and the resulting reward. This keeps the learned values consistent with what the policy chose.

**Default hyper-parameters** (initial; final values recorded in Issue #5): lr 5e-4, batch 256, replay 200 k transitions, warmup 10 k, train_freq 4, target_period 2 000, $\gamma$ 0.99, ε 1.0 → 0.05 over 30 % of training.

**Correctness considerations.**
- Bootstrap at truncation (model §6.3).
- Masked max in the target.
- **Non-stationarity:** other agents' policies change, so replayed data becomes stale, and single-agent Q-learning convergence guarantees **do not transfer** (G16). Mitigations, not guarantees: parameter sharing, a moderate buffer, optional fingerprints.
- Correlated transitions: mitigated by $E \ge 4$ environments and a large buffer.

**Complexity.**
- Action selection: $O(EN \cdot P)$ per environment step for $P$ parameters.
- Update: $O(B \cdot P)$ per gradient step.
- Replay memory: $O(C_D \cdot d_o)$. Naively 200 k × 356 × 4 B × 2 ≈ 570 MB. Binary channels are stored as `uint8` and the remainder as `float16` where exact, with a target < 300 MB, decided in Issue #5.

**Tests.**
1. Replay buffer: FIFO overwrite, uniform sampling, dtypes, seeded determinism.
2. Target computation vs a manual NumPy computation, covering masks, termination vs truncation and Double-DQN selection.
3. **Learning sanity:** on a tiny chain MDP with known $Q^\*$, learned $Q$ approaches $Q^\*$.
4. Overfitting on a fixed batch.
5. Masked ε-greedy never selects masked actions (property test).
6. Short end-to-end training improves validation return over a random policy on a 2-robot 6×6 scenario, by a statistical check across seeds. Only this *sanity* claim is tested; performance claims come from experiments.

**Implementation notes (Issue #5).**
- `marl/madql.py`, `networks.py`, `replay.py`, `trainer.py`. The MLP has 158 469 parameters. Masked arg-max resolves ties to the lowest action index; masked ε-greedy draws uniformly over the *allowed* actions (D5).
- **Final hyper-parameters** = the initial values above, unchanged: lr 5e-4, batch 256, replay 200 k, warm-up 10 k stored transitions, train_freq 4 and target_period 2 000 *environment steps summed over the copies* (D3), $\gamma$ 0.99, ε 1.0 → 0.05 over 30 % of training, Huber loss, Adam, gradient-norm clip 10; $E$ = 4 copies. No hyper-parameter was tuned.
- **Replay layout:** window channels as `uint8` (the priority channel stores the level $\pi$), scalars as `float32`; the codec restores observations exactly (tested). About 0.8 kB per transition, ≈ 160 MB for 200 k (< 300 MB, tested).
- **Stored action:** the requested action $a^{\text{ctrl}}$, also when the shield or the arbiter changed or cancelled it, because the shield and the arbiter are part of the transition the learned values describe.
- **Step budget:** 1 000 000 environment steps per run, fixed from the measured throughput of this trainer (verification log V-14).
- **Tests:** `tests/unit/marl/test_madql.py` (tests 1–5; test 5 is a Hypothesis property test), `tests/integration/test_marl_learning_sanity.py` (test 6, marked `training`), `tests/integration/test_marl_training.py` (X-7, reproducibility, X-1 with a learned controller).

---

## 7. PPO: multi-agent PPO (MAPPO / IPPO)

**Purpose.** The second required MARL algorithm (FR-RL-3). Design decision ADR-4.

**References.** Schulman et al. 2017 (PPO); Schulman et al. 2016 (GAE); Yu et al. 2022 (MAPPO); de Witt et al. 2020 (IPPO).

**Model.**
- Actor $\pi_\phi(a \mid o_i)$: MLP $356 \to 256 \to 256 \to 5$ logits, with masked actions set to $-\infty$.
- Critic $V_\psi(s_i^c)$: MLP on $[o_i, \phi(s)]$ for MAPPO ($356 + 4G^2 + 3 = 615$ inputs for $G = 8$), or on $o_i$ for IPPO.
- Separate networks with orthogonal initialisation.

**GAE**, per robot stream:

$$
\delta_t = r_t + \gamma (1 - d_{t+1}) V(s_{t+1}) - V(s_t), \qquad
\hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l \prod_{u=1}^{l}(1 - d_{t+u})\,\delta_{t+l},
$$

computed backwards in $O(T)$, with $\hat R_t = \hat A_t + V(s_t)$. At truncation, $V(s_{t+1})$ is the critic's value of the true final state, stored before auto-reset.

**Objective.** With $\rho_t(\phi) = \pi_\phi(a_t \mid o_t)/\pi_{\phi_{\text{old}}}(a_t \mid o_t)$:

$$
\mathcal{L} = -\mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\; \text{clip}(\rho_t, 1-\epsilon_c, 1+\epsilon_c)\hat A_t\big)\Big]
+ c_v\,\mathbb{E}_t\big[(V_\psi - \hat R_t)^2\big] - c_e\,\mathbb{E}_t\big[\mathcal{H}(\pi_\phi(\cdot \mid o_t))\big].
$$

**Algorithm.** For each iteration:
1. Collect $T_r$ steps in $E$ environments with the shared actor (the raw chosen actions are stored, as in §6).
2. Compute GAE per (environment, robot) stream and normalise advantages.
3. Run $K$ epochs of minibatch Adam updates, with gradient-norm clip 0.5 and optional early stop on approximate KL.
4. Validate and checkpoint periodically.

**Default hyper-parameters** (initial; recorded in Issue #6): $T_r$ 128, $E$ 8, $K$ 4, 4 minibatches, lr 3e-4, $\gamma$ 0.99, $\lambda$ 0.95, $\epsilon_c$ 0.2, $c_v$ 0.5, $c_e$ 0.01, target_kl 0.02.

**Correctness considerations.**
- Per-robot GAE streams.
- Masks are stored with samples, so log-probabilities match between collection and update.
- Old log-probabilities are stored, never recomputed.
- On-policy: the buffer is discarded after each update.
- The MAPPO critic is training-only, so execution stays decentralised.
- PPO's clipping is a heuristic trust region; no monotonic-improvement guarantee is claimed for the implemented algorithm (G16).

**Complexity.**
- Collection: $O(E T_r N P_\pi)$.
- GAE: $O(E N T_r)$.
- Update: $O(K E N T_r (P_\pi + P_V))$.
- Memory: $O(E N T_r d_o)$, about 12 MB for the defaults.

**Tests.**
1. GAE vs a hand computation, including termination and truncation within a segment and stream separation.
2. Masked categorical: zero probability and no entropy contribution for masked actions.
3. Clipped-loss behaviour at $\rho = 1$ and outside the clip range.
4. **Learning sanity:** a bandit and the chain MDP.
5. The MAPPO/IPPO switch changes only the critic input.
6. Short end-to-end improvement check, as for MADQL.

**Implementation notes (Issue #6).**
- **Modules:**
  - `marl/rollout.py` (NumPy only): padded `[T_r, E, N_max]` buffer, GAE and the collection loop;
  - `marl/ppo.py`: actor, critic, masked categorical, loss and update;
  - `marl/ppo_trainer.py`: a separate trainer; the MADQL trainer is unchanged (D9);
  - `configs/training/{mappo,ippo,ppo_smoke}.toml` and the reward-ablation configurations `mappo_{zeta0,beta0,crr05,crr2}.toml`.
- **Actor:**
  - $356 \to 256 \to 256 \to 5$ (158 469 parameters), shared by all robots, with no robot identity;
  - orthogonal initialisation (hidden gain √2, output gain 0.01);
  - masked actions get the logit `finfo.min`; the entropy is summed over allowed actions only;
  - actions are sampled by inverse CDF from the float64 masked softmax with the run's `sample` stream; the stored log-probability is computed by the same function the update uses.
- **Critic:**
  - shared by all robots (orthogonal, output gain 1);
  - MAPPO: $[o_i, \phi(s)]$, 615 inputs, 223 745 parameters; $\phi(s)$ includes $t/T$;
  - IPPO: $o_i$, 356 inputs, 157 441 parameters;
  - the switch changes only the critic input and width; with the same seed both variants initialise identical actors (tested).
- **Rollout:** $E = 8$, $T_r = 125$, so 1 000 environment steps per iteration (D4; the documented 128 would not divide the 50 000-step validation interval). Robot slots beyond a copy's team size are invalid and excluded from every loss.
- **GAE** per (copy, slot) stream with $\gamma = 0.99$, $\lambda = 0.95$:
  - $V_\text{next} = 0$ after termination;
  - $V_\text{next} = V(\text{final observation}, \phi(s_T))$ after truncation, captured before the auto-reset (`SyncVectorEnv(with_state=True)`; tests R1–R6);
  - otherwise the next value of the same stream;
  - the recursion stops at termination **and** truncation, so it never crosses an auto-reset.
- **Update:**
  - $K = 4$ epochs × 4 shuffled minibatches;
  - advantages normalised per minibatch;
  - $\epsilon_c = 0.2$, $c_v = 0.5$ (MSE, no value clipping), $c_e = 0.01$;
  - Adam with lr 3e-4 and eps 1e-5 (one optimiser, actor and critic parameter groups);
  - gradient norms clipped to 0.5 **separately** for actor and critic (D5), so a critic loss of any scale leaves the actor's update unchanged (tested);
  - the iteration stops early once a minibatch's approximate KL $\mathbb{E}[(\rho - 1) - \log\rho]$ exceeds 0.02;
  - no learning-rate annealing.
- **Determinism:**
  - streams `sample`, `shuffle` and `torch` come from `SeedSequence(run_seed)`;
  - the scenario order is the fixed training order (sampler seed 0), as for MADQL;
  - deterministic torch with one thread;
  - the same configuration and seed give byte-identical checkpoints and logs, timings excepted (tested).
- **Checkpoints:**
  - inference export is the actor only, model specification `mlp-pi-v1`, next to `mlp-q-v1` in an additive version table;
  - the sidecar adds the variant, `evaluation_mode = greedy` and the critic specification (`included: false`);
  - training checkpoints `*.train.pt` hold actor, critic, optimiser, iteration, environment and robot steps, the scenario position, the NumPy and torch random states and the best selection so far.
- **Resume (D8):** at the latest validation point only. The episodes in progress there are discarded and restart from the next scenario seeds of the fixed order, so a resumed run is **not bitwise identical** to an uninterrupted one. The interrupted attempt's later log records are moved to `*.pre_resume_<n>.jsonl`, and the resume is recorded in the log and the summary (tested).
- **Learned controller:** the PPO actor runs through the existing `LearnedController` (masked arg-max = the policy's mode). Stochastic execution is an explicit evaluation mode, never the default: PPO samples from the masked policy, MADQL uses masked ε-greedy with ε = 0.05.
- **Tests:**
  - `tests/unit/marl/test_rollout.py` (test 1);
  - `test_ppo.py` (tests 2, 3, 5, separate clipping, determinism);
  - `test_ppo_sanity.py` (test 4: a masked bandit and the chain MDP, both variants);
  - `tests/integration/test_ppo_rollout.py` (R1–R6);
  - `test_ppo_training.py` (X-7, reproducibility, resume, CLI);
  - `test_ppo_learning_sanity.py` (test 6, marked `training`);
  - `tests/e2e/test_ppo_subprocess.py`;
  - `tests/integration/test_x4_matrix.py` (X-4).

---

## 8. Observations and rewards

**Observation builder.**
- Window slicing from padded layers: $O(N r^2)$ per step.
- Layer refresh: $O(V)$, or $O(N + K)$ incrementally.
- Path channel: $O(L_o)$ per robot.
- Global summary pooling: $O(V)$ per step.

**Reward function.** Base terms are $O(1)$ per robot, given the conflict events of §1 and the shield log.
- The potential $\Phi_i$ needs $\bar D(x_i, \cdot)$ from cached distance fields: one or two lookups, $O(1)$ on a cache hit and $O(V)$ on a miss.
- Team mixing: $O(N)$.

**Tests.**
1. Hand-constructed worlds with the expected channel contents.
2. Each base-reward term separately; at-fault attribution from event sets.
3. **Potential tests:**
   - $\Phi_i$ is unchanged across a pickup step when the robot stands still on $p_j$ (continuity).
   - The shaping sum over a recorded episode telescopes to $\gamma^T\Psi_i(s_T) - \Psi_i(s_0)$ (numerical check).
   - $\Psi_i = 0$ at termination.
   - The team-mixed shaping term equals the shaping of $\Psi_i$.
   - Capped distance when the goal becomes statically unreachable.
4. Observation size is independent of $N$ and the map.

**Implementation notes (Issue #5; `marl/observation.py`, `reward.py`, `env.py`).**
- Observation: $r = 3$, $L_o = 6$ path cells after the robot's own cell, $P = 3$. Channel 2 holds the constant-velocity prediction $z + v$ of every obstacle sensed in the window, kept only if it is a free in-bounds cell inside the window (as `planning.constraints.sense`; $v = 0$ predicts the current cell). Channel 4 and the own-priority scalar use the base level $\pi / P$ (D4), not the aged weight the baseline uses. Out-of-bounds cells read as static obstacles. Without a planner the suggested-action one-hot is all zero.
- Rewards attribute robot–robot and robot–obstacle conflicts through `ConflictEvent.at_fault` of the proposal reaching the arbiter (a robot that stayed is not at fault in a vertex conflict; both robots are at fault in a swap); invalid moves cannot occur under the mask. Idle robots pay no step cost and have $\Phi = 0$, so a waiting idle robot's own reward is exactly 0.
- Episode end: termination takes precedence; truncation at $T$ performs the normal preparation of time $T$ (events, arrivals, allocation, replanning) without acting, and the final observation, $\Phi(s_T)$ and the episode metrics come from that state. Training-episode metrics can therefore differ from an `Engine(max_steps = T)` run only in the counts this preparation adds (tasks released at $T$, planning counters). Validation and development comparisons run every controller on the plain engine.
- **Closed forms asserted by the tests** (`tests/unit/marl/test_reward.py`): the net discounted return of a $k$-step cycle is $(1-\gamma^k)\big(\zeta\Lambda/(W+H) - c_{\text{step}}/(1-\gamma)\big)$, i.e. $(1-\gamma^k)(\zeta\Lambda/(W+H) - 1)$ at the defaults, positive exactly when $\Lambda > (W+H)/\zeta$; a progress move beats `stay` by $\zeta\gamma/(W+H)$; finishing beats staying, and so does progress truncated before finishing. These are accounting identities; for terminating episodes the policy-invariance argument of model §7.1 applies, for truncated episodes it does not.
- Shaping spikes (capped distances when a goal becomes unreachable, assignment jumps) are not clipped; training logs record the maximum $|F|$ and the number of robot-steps with $|F| > 1$ per episode.

---

## 9. Complexity summary

| Algorithm | Time | Space | Status of the bound |
|---|---|---|---|
| Conflict detection + arbitration | $O(N + K)$ expected per step | $O(N + K)$ | Worst-case for the worklist, expected for hashing |
| BFS distance field | $O(V)$ | $O(V)$ | Worst case |
| A\* | $O(V \log V)$ worst; $O(L \log L)$ best (+$O(V)$ init) | $O(V)$ | Worst/best case; typical behaviour measured |
| RRT\*, unconditional | $O(N_{\text{it}}^2\eta + N_{\text{it}}^3)$ | $O(N_{\text{it}})$ | Worst case, both indexes |
| RRT\* + k-d index, under U | $O(N_{\text{it}}^{3/2} + N_{\text{it}}\eta\log N_{\text{it}})$ + NN ($O(N_{\text{it}}^2)$ worst, $O(N_{\text{it}}\log^2 N_{\text{it}})$ expected under R) + propagation $S$ (measured) | $O(N_{\text{it}})$ | Conditional |
| RRT\* + brute-force index, under U | $\Theta(N_{\text{it}}^2) + O(N_{\text{it}}\eta\log N_{\text{it}}) + S$ | $O(N_{\text{it}})$ | Conditional |
| Log-method k-d tree | insert $O(\log^2 n)$ amortised, $O(n\log n)$ worst; radius $O(\sqrt n + k')$ worst; NN $O(n)$ worst | $O(n)$ | Worst/amortised; NN expected under R |
| Supercover check / segment cost | $O(\ell + 1)$ | $O(1)$ | Worst case |
| Auction (ε-scaling) | $O(k^3 \log(kC))$ | $O(k^2)$ | Worst case (cited) |
| Auction (fixed ε) | $O(k^2 C/\varepsilon)$ | $O(k^2)$ | Worst case, pseudo-polynomial |
| Sequential auction | $O(nm \log m)$ | $O(nm)$ | Worst case; ½-approximation |
| Hungarian (oracle) | $O(k^3)$ | $O(k^2)$ | Worst case; testing and experiments only |
| Path monitor | $O(H_r + H_v)$ per robot per step | $O(L)$ per robot | Worst case, excluding planner calls |
| Baseline / reference follower / shield | $O(N)$ per step | $O(N)$ | Worst case (hash expected) |
| Observation build | $O(V + N r^2)$ per step | $O(V + N r^2)$ | Worst case |
| MADQL update | $O(B P)$ per gradient step | $O(C_D d_o + P)$ | Worst case |
| PPO iteration | $O(K E N T_r (P_\pi + P_V))$ | $O(E N T_r d_o + P)$ | Worst case |

Every bound in this table is paired with a measurement in `05-testing-and-evaluation.md` (E1–E7). Agreement or disagreement between predicted and measured scaling is reported explicitly.
