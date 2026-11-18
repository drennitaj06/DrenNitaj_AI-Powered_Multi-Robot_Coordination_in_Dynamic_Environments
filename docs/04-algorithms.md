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

### 4.3 Allocation policy

- **Triggers:** task released; task delivered; robot became idle; a robot's plan status became `unreachable`; map event. At most one auction per step.
- **Eligible:** robots in mode idle or to_pickup; *available* tasks pending or assigned-but-not-picked (model §4).
- **Pre-emption:** only through the auction, with hysteresis $h$.
- **Output application:** a changed assignment changes the robot's goal, which is trigger T1 for replanning.

### 4.4 Priority semantics: what the tests check

The allocation implements **weighted priority** (model §8.1). Tests verify the *weighted* rule, not strict priority:
- **Threshold test:** two tasks of levels $\pi_{\text{hi}} > \pi_{\text{lo}}$ and one robot. The high-priority task is chosen iff $\kappa(\tilde w_{\text{hi}} - \tilde w_{\text{lo}}) > c_{\text{hi}} - c_{\text{lo}}$. Both sides of the threshold are checked, and exact equality is resolved by the documented tie rule.
- **Hysteresis test:** no switch when the gain is ≤ $h$; a switch when it is $> h$.
- **Aging property tests:** P1 (FIFO within a level), P2 (outranking newly released tasks after $A^\*$ steps), P3 (a younger higher-level task regains precedence), all on the priority term, with exact arithmetic.
- **Starvation mitigation** is *not* a unit-test property. It is measured in E3 (waiting-time tails per level, $\alpha > 0$ vs $\alpha = 0$).

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
  4. *T4 needs an occupied cell.* Two robots that both keep proposing the same **free** cell are frozen by the arbiter every step without either next cell being occupied, so T4 does not fire. For a pure plan follower this is a deadlock. Resolving it is the job of the controllers (baseline right-of-way rule, learned policies; Issue #4+), not of the planner. E2 reports how often it limits completion.
  5. *T4 vs T6 on a degraded plan* (documented by `test_degraded_plan_and_stall_interaction`). With the defaults $T_{\text{cd}} = 3 < T_{\text{stall}} = 4$, a robot stalled in front of a dynamically blocked cell of a degraded path is replanned by T6 every 3 steps. Each replan resets the stall counter, so T4 does not fire while the plan stays degraded. T4 fires first only if $T_{\text{cd}} \ge T_{\text{stall}}$. This is the specified trigger order applied literally, not a separate rule.
- Release/hold policy (model §5.3 step 4): a robot whose pickup becomes statically unreachable releases the task (`AssignmentAPI.unassign_task`). **Issue #3 fixture logic only:** the fixture allocation re-offers a released task to the same robot only once that robot's static reachability *can* have changed, i.e. after a map event (grid version) or after the robot has moved, and only while the task is available. Re-offering every $T_{\text{cd}}$ steps in an unchanged situation repeated the same `unreachable` call and release every 3 steps (14 releases in 40 steps in the review probe; changed in V-6, regression test `test_release_is_not_reoffered_until_something_changes`). In Issue #4 the auction replaces this: a released task is re-auctioned and can go to another robot. A robot whose drop-off becomes unreachable holds the task, stays (suggested `stay`) and retries with T5 every $T_{\text{cd}}$ steps.
- Engine hooks: a coordinator that also implements `sim.coordinator.PlanReporting` (`plan_valid`, `trace_fields`, `episode_stats`, `finish`) gets plan adherence (model §9.3) recorded by the engine, its planning fields added to the trace, and its statistics added to `EpisodeMetrics.extra["planning"]`. This is an additive extension of the Issue #2 coordinator contract; earlier coordinators are unaffected.
- `control.base.SuggestionFollower` ("follow") plays the masked suggested action. It is the minimal plan follower used by tests and E2. The reference follower of Issue #4 takes over its role in the controller registry.
- **X-12 / X-5 results:** `tests/unit/planning/test_monitor.py` checks each trigger at its exact step: T1; rejoin within $H_r$ without replanning; T3 beyond $H_r$; T2 for an obstacle or a map event within $H_v$ and not beyond it; T4 exactly after $T_{\text{stall}} = 4$ steps with block expiry; no T4 at the goal; T5 every $T_{\text{cd}} = 3$ steps; T6. `tests/integration/test_planning_integration.py` runs the same checks inside the engine for both planners: a map event on the path gives T2 and a detour; a map event on the drop-off gives `unreachable`, hold, T5 retries at $t = 12, 15$, 11 blocked-goal robot-steps, then delivery; a map event on the pickup gives a release and a re-take; an obstacle crossing the path gives T2. Plan adherence and deviation rate are checked against hand computation.

### 5.3 Controllers (non-learning)

**Reference follower (diagnostic).** $a_i = \hat a^{\text{plan}}_i$ (stay if masked). It defines the counterfactual of model §9.2 and the episode-level comparison point. $O(N)$ per step.

**Rule-based baseline.** For each robot, with target $c = x_i + \Delta(\hat a^{\text{plan}}_i)$:

```text
if no valid plan or at goal:                                       stay
elif c ∈ occupied cells (robots, obstacles) or c ∈ Ẑ_i(t):         stay
elif ∃ robot j adjacent to c whose suggested target is c and
     (π̃_j > π̃_i or (π̃_j = π̃_i and j < i)):                      stay    # yield right-of-way
else:                                                              move to c
```

Here $\tilde\pi$ is the effective priority of the carried or assigned task, with idle lowest. The baseline reads neighbours' *suggested* actions, which is available because planning is centralised. It is still a heuristic, since neighbours controlled by learned policies may not follow their suggestions. Deadlocks are handled by the controller-agnostic stall trigger T4, not by a baseline-specific rule.

**Complexity.** $O(N)$ per step with the cell → robot map.

**Limitations (measured, not hidden).** Unpredictable obstacle turns can still produce $\textsf{RO}$ conflicts. Corridor deadlocks are mitigated by T4, not eliminated.

**Tests.** Yielding to a higher priority; tie-break by id; no $\textsf{RR}$ conflicts among baseline robots in static scenarios (property test); head-on corridor case resolved via T4.

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
