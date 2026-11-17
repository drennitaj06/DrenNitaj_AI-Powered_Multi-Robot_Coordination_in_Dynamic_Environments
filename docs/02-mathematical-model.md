# 02 — Mathematical Model

| Field | Value |
|---|---|
| Document role | **Source of truth for formal definitions**: notation, state, dynamics, conflicts, planning constraints, execution layers, objectives, RL formulation, assignment problem, metrics and the summary of guarantees (§12). Algorithms (`04-algorithms.md`) and code use these symbols. |
| Status | Revision 2 (Issue #1, after design review). Changes are listed in `verification-log.md` V-1. |

Notation conventions: sets are calligraphic or upper-case ($\mathcal{V}, R, J$); $\mathbb{1}[\cdot]$ is the indicator function; $\|\cdot\|_1$, $\|\cdot\|_2$, $\|\cdot\|_\infty$ are the Manhattan, Euclidean and Chebyshev norms. By convention $\min \varnothing = +\infty$.

Statements are labelled **[Req]** when they encode a project requirement and **[Asm]** when they encode a modelling assumption from `01-requirements.md` §7. Claims of mathematical properties are labelled **Guaranteed**, **Conditional** (holds under stated assumptions) or **Not claimed / empirical**. §12 collects them.

---

## 1. Time and space

**Time.** Discrete, synchronous steps $t \in \{0, 1, \dots, T_{\max}\}$. [Asm A-1]

**Grid.** A rectangular grid of $W \times H$ unit cells

$$
\mathcal{C} = \{(x, y) \in \mathbb{Z}^2 : 0 \le x < W,\; 0 \le y < H\}.
$$

Cell $(x,y)$ covers the closed-open square $[x, x+1) \times [y, y+1) \subset \mathbb{R}^2$; its centre is $(x+\tfrac12, y+\tfrac12)$. This embedding lets the continuous planner RRT\* and the discrete planner A\* work on the *same* map.

**Static obstacles.** $S(t) \subseteq \mathcal{C}$ is the set of blocked cells at time $t$. It is piecewise constant and changes only at **map events** $e = (t_e, c_e, b_e)$ with $b_e \in \{\text{block}, \text{unblock}\}$. [Req FR-ENV-3]
- A *block* event on a cell that is currently occupied by a robot or a dynamic obstacle is deferred until the cell is vacated, so a wall never appears on top of an agent.
- Map events are applied at the start of a step, before observations and action masks are computed (`03-architecture.md` §2). A masked controller therefore never chooses a move into a static obstacle.

**Free-space graph.** At time $t$, the free cells $\mathcal{V}(t) = \mathcal{C} \setminus S(t)$ form the undirected graph $G(t) = (\mathcal{V}(t), \mathcal{E}(t))$ with 4-neighbourhood edges

$$
\mathcal{E}(t) = \big\{\{u, v\} \subseteq \mathcal{V}(t) : \|u - v\|_1 = 1\big\}, \qquad |\mathcal{E}(t)| \le 2WH.
$$

So $|\mathcal{V}| = O(WH)$ and $|\mathcal{E}| = O(WH)$. Every graph algorithm here has $|E| = \Theta(|V|)$.

**Static shortest-path distance.** $D_t(u, v)$ is the number of edges on a shortest $u$–$v$ path in $G(t)$, or $+\infty$ if $v$ is unreachable. For a fixed target $v$, all values $D_t(\cdot, v)$ are computed by one breadth-first search (a *distance field*) in $O(WH)$.

**Capped distance.** $\bar D_t(u,v) = \min\big(D_t(u,v),\, D_{\text{cap}}\big)$ with $D_{\text{cap}} = WH$. Every finite distance is $< WH$, so the cap only replaces $+\infty$. It keeps potentials and features bounded (§7).

---

## 2. Dynamic obstacles

A finite set $K$ of dynamic obstacles. Obstacle $k$ has position $z_k(t) \in \mathcal{C}$ and a motion model $m_k$ that proposes a desired next cell $\tilde z_k(t+1)$:

| Model | Proposal |
|---|---|
| Random walk | $\tilde z_k(t+1) = z_k(t) + \delta$, with $\delta$ uniform over $\{(0,0), (\pm1,0), (0,\pm1)\}$ |
| Patrol | Next cell on a shortest path towards the current waypoint $w_{k,\ell}$; cycle $\ell \leftarrow \ell+1 \bmod L_k$ on arrival |
| Bounce | Keep the current direction $\delta_k$; reverse $\delta_k$ when the next cell is blocked |
| Data-driven (Issue #7, conditional) | Correlated random walk with parameters estimated from trajectory data |

**Obstacle move rule** [Asm A-5]. The proposal is accepted only if it is in bounds, not occupied at time $t$ by a static obstacle, a robot or another dynamic obstacle, and not already claimed by an earlier obstacle (lower index) in the same step. Otherwise $z_k(t+1) = z_k(t)$. Obstacles are processed in a fixed index order, so the outcome is deterministic. All randomness comes from the obstacle's own seeded generator. Obstacle moves at step $t$ do **not** depend on the robots' actions at step $t$, only on positions at time $t$. This makes counterfactual conflict evaluation (§9.2) well-defined.

**Velocity and prediction.** $v_k(t) = z_k(t) - z_k(t-1)$. The one-step constant-velocity prediction is $\hat z_k(t+1) = z_k(t) + v_k(t)$, kept only if it is a free in-bounds cell. It is a heuristic: obstacles may turn or stop.

---

## 3. Robots, actions, conflicts and arbitration

**Robots.** $R = \{1, \dots, N\}$. Robot $i$ has position $x_i(t) \in \mathcal{V}(t)$, a mode $\mu_i(t) \in \{\textsf{idle}, \textsf{to\_pickup}, \textsf{to\_dropoff}\}$, an assigned task $\tau_i(t) \in J \cup \{\varnothing\}$ and a goal cell

$$
g_i(t) = \begin{cases} p_{\tau_i} & \mu_i = \textsf{to\_pickup} \\ d_{\tau_i} & \mu_i = \textsf{to\_dropoff} \\ \varnothing & \mu_i = \textsf{idle}. \end{cases}
$$

**Actions.** $\mathcal{A} = \{\textsf{stay}, \textsf{up}, \textsf{down}, \textsf{left}, \textsf{right}\}$, with displacement $\Delta(a)$ (e.g. $\Delta(\textsf{up}) = (0,1)$).

**Execution layers.** An action passes through three layers before it changes the world. §5.5 gives the full specification.
1. **Action mask** $M_i(t) \subseteq \mathcal{A}$ (always on): removes actions whose target is out of bounds or in $S(t)$.
2. **Safety shield** (optional, configuration `shield`): may replace a chosen action by $\textsf{stay}$ (§5.5).
3. **Arbiter** (always on, the simulator's physical layer): cancels every move involved in a conflict (below).

**Proposed positions.** For the joint proposal $a = (a_1,\dots,a_N)$ that reaches the arbiter, $\hat x_i = x_i(t) + \Delta(a_i)$.

**Conflict types** (defined on *proposals*; nothing has happened physically yet). For robots $i \ne j$ and obstacle $k$, with $z_k' = z_k(t+1)$:

| Conflict | Condition |
|---|---|
| Robot–robot **vertex** $\textsf{RRv}$ | $\hat x_i = \hat x_j$ (includes moving into a robot that stays, since then $\hat x_j = x_j(t)$) |
| Robot–robot **swap** $\textsf{RRs}$ | $\hat x_i = x_j(t) \wedge \hat x_j = x_i(t) \wedge \hat x_i \ne x_i(t)$ |
| Robot–obstacle **vertex** $\textsf{ROv}$ | $\hat x_i = z_k'$ and $a_i \ne \textsf{stay}$ |
| Robot–obstacle **swap** $\textsf{ROs}$ | $\hat x_i = z_k(t) \wedge z_k' = x_i(t)$ |
| Invalid proposal $\textsf{INV}$ | $\hat x_i \notin \mathcal{V}(t)$. Only possible for controllers that ignore the mask. |

A staying robot can never be in an $\textsf{ROv}$ or $\textsf{ROs}$ conflict, because obstacles never enter robot-occupied cells. Following moves (robot $j$ enters the cell robot $i$ leaves in the same step) and rotations along cycles of length ≥ 3 are **not** conflicts.

**Conflict event.** A conflict event is the tuple $e = (t, \text{type}, \{\text{entity}_1, \text{entity}_2\})$, with an unordered pair of robot or obstacle identifiers, or a single robot for $\textsf{INV}$. The set of events of a joint proposal $a$ at state $s_t$ is $\mathcal{K}(s_t, a)$. It is computed by a pure function `detect_conflicts`. The **at-fault** robots of an event are those with $a_i \ne \textsf{stay}$.

**Arbitration (resolution operator)** [Asm A-8]. Let $\mathcal{F} \subseteq R$ be the set of robots whose move is cancelled ("frozen"; they stay at $x_i(t)$). $\mathcal{F}_0$ = all robots involved in any event of $\mathcal{K}(s_t, a)$ (vertex, swap, obstacle or invalid). Then

$$
\mathcal{F}_{n+1} = \mathcal{F}_n \cup \Big\{ i \notin \mathcal{F}_n : \exists j \ne i,\; \hat x_i = \text{pos}_n(j) \Big\},
\quad
\text{pos}_n(j) = \begin{cases} x_j(t) & j \in \mathcal{F}_n \\ \hat x_j & \text{otherwise}, \end{cases}
$$

until a fixed point $\mathcal{F}^\*$ is reached (at most $N$ rounds). Then $x_i(t+1) = x_i(t)$ for $i \in \mathcal{F}^\*$, else $\hat x_i$. Robots in $\mathcal{F}^\* \setminus \mathcal{F}_0$ were frozen only by the cascade: they wanted the cell of a robot that was itself frozen, without a conflict of their own. Each is counted as a **blocked move**, not as a conflict.

**Physical collision.** A *physical collision* is a post-resolution state violating occupancy: two robots in one cell, a robot in an obstacle cell, or two robots that exchanged cells. **Guaranteed (Lemma 3.1):** the arbiter output contains no physical collision, for any joint proposal; proof in `04-algorithms.md` §1. The simulator therefore models every proposed conflict as *prevented by an emergency stop*. The physical-collision count is an invariant check that must equal 0; a non-zero value aborts the run as a simulator defect. The safety performance of a controller is measured by its **proposed conflicts** (§9), not by physical collisions.

---

## 4. Tasks and priorities

**Tasks.** $J$ is the (possibly growing) set of tasks. Task $j$ is the tuple

$$
j = (p_j, d_j, \pi_j, \rho_j), \quad p_j, d_j \in \mathcal{C},\; \pi_j \in \{1, \dots, P\},\; \rho_j \in \mathbb{N},
$$

with pickup cell $p_j$, drop-off cell $d_j$, priority level $\pi_j$ (larger = more urgent) and release time $\rho_j$. Status $\sigma_j(t) \in \{\textsf{unreleased}, \textsf{pending}, \textsf{assigned}, \textsf{picked}, \textsf{delivered}\}$. The completion time $C_j$ is the step at which $\sigma_j$ becomes $\textsf{delivered}$. The waiting age is $a_j(t) = t - \rho_j$ for released, not yet picked tasks.

**Availability.** A released, unpicked task is **available** at $t$ iff $D_t(p_j, d_j) < \infty$ and $p_j, d_j \notin S(t)$. Map events can make a task temporarily unavailable. Scenario validation rejects tasks unavailable on the initial map. *Dynamic* obstacles never make a task unavailable (§5.2).

**Arrival processes** [Req FR-ENV-4]: *scheduled* (release times in the scenario); *Bernoulli* (one arrival per step with probability $\lambda$, i.e. geometric inter-arrival times); *non-homogeneous Poisson* (per-step count $\sim \text{Poisson}(\lambda(t))$, with $\lambda(t)$ estimated from data in Issue #7).

**Priority weight.** $w(\pi) = 2^{\pi - 1}$ by default (configurable), so $w_{\min} = 1$ and $w_{\max} = 2^{P-1}$.

**Aging** [Req FR-TASK-4]. A task's *effective* weight grows linearly with waiting age:

$$
\tilde w_j(t) = w(\pi_j)\,\big(1 + \alpha\, a_j(t)\big), \qquad \alpha \ge 0.
$$

**Properties of the aging rule.** Here "outranks" compares only the priority terms $\tilde w$. Allocation also involves travel cost (§8).

- **P1 (Guaranteed): FIFO within a level.** If $\pi_j = \pi_l$ and $\rho_j \le \rho_l$, then $\tilde w_j(t) \ge \tilde w_l(t)$ for all $t$.
- **P2 (Guaranteed): outranking newly released tasks.** If $\alpha > 0$ and $a_j(t) \ge A^\* := (w_{\max}/w_{\min} - 1)/\alpha$, then $\tilde w_j(t) \ge w(\pi_l) = \tilde w_l(t)$ for every task $l$ with $\rho_l = t$, whatever its priority. *Proof:* $w(\pi_j)(1+\alpha a_j) \ge w_{\min}(1 + w_{\max}/w_{\min} - 1) = w_{\max} \ge w(\pi_l)$.
- **P3 (Guaranteed, a limitation): no permanent overtaking of higher levels.** If $\pi_j < \pi_l$ and both tasks remain pending, then $\tilde w_j(t)/\tilde w_l(t) \to w(\pi_j)/w(\pi_l) < 1$ as $t \to \infty$. Any advantage of an older low-priority task over a younger high-priority one is temporary. Exactly: with $\rho_l = \rho_j + \Delta$, task $j$ outranks $l$ iff $\alpha\, w(\pi_l)\,\Delta \ge \big(w(\pi_l) - w(\pi_j)\big)\big(1 + \alpha a_j(t)\big)$, and the right-hand side grows with $t$. Multiplicative aging lets higher levels gain weight faster in absolute terms.
- **Not claimed: starvation freedom.** The rule does not guarantee that every task is eventually allocated. (i) P3 shows that a persistent backlog of higher-priority tasks can keep outranking it. (ii) The travel-cost and hysteresis terms of §8 can make a task unattractive even when its priority term is high. (iii) Under overload (arrival rate above service capacity) no allocation rule can serve every task in finite time. (iv) A task can stay unavailable because of map events.

Aging is therefore a **starvation-mitigation** mechanism. Its effect is an empirical question: maximum and high-percentile waiting time per priority level, measured with and without aging in E3 (`05-testing-and-evaluation.md`). *Documented alternative, not adopted:* additive aging $w(\pi_j) + \alpha a_j$ gives a time-invariant overtaking guarantee (task $j$ outranks every task released at least $(w_{\max} - w_{\min})/\alpha$ steps after it). It still does not give starvation freedom, for reasons (ii)–(iv), and it changes the requested design.

---

## 5. Path planning model

### 5.1 Reference paths

A planner returns, for robot $i$ with goal $g_i$, a cell sequence $P_i = (c_0 = x_i, c_1, \dots, c_L = g_i)$ with $\|c_{\ell+1} - c_\ell\|_1 = 1$, or a failure status. Planning is **reactive and spatial**: each query is solved on a *snapshot* of the constraints at its query time. There is no time dimension and no reservation of cells over time. Other robots are not planner constraints; they are handled by the controller and the arbiter. Space-time planning (e.g. space-time A\*, cooperative A\*) is out of scope (`01-requirements.md` §11).

### 5.2 Dynamic constraint set (shared by A\* and RRT\*)

For a query by robot $i$ at time $t$, both planners use the **same** constraint sets, built by one shared function:

- Sensed obstacles: $Z_i(t) = \{z_k(t) : \|z_k(t) - x_i(t)\|_\infty \le r_s\}$. Their predictions are $\hat Z_i(t) = \{\hat z_k(t+1) : z_k(t) \in Z_i(t)\}$.
- **Hard dynamic block set:** $B_i(t) = \big(Z_i(t) \cup \hat Z_i(t)\big) \setminus \{x_i(t),\, g_i\}$. The robot's own cell and its **goal cell are never dynamically blocked**: an obstacle on a pickup or drop-off cell is transient and must not make a task unreachable.
- **Proximity set:** $Q_i(t) = \{v \in \mathcal{C} : \min_{z \in Z_i(t)} \|v - z\|_1 \le r_{\text{infl}}\}$. If $Z_i(t) = \varnothing$, the minimum is $+\infty$ and $Q_i(t) = \varnothing$. This includes the goal cell when it is near an obstacle.
- **Cost density** over the plane: $\omega_i(p) = 1 + \lambda_{\text{prox}} \cdot \mathbb{1}[\text{cell}(p) \in Q_i(t)]$, so $1 \le \omega_i \le 1 + \lambda_{\text{prox}}$.

**A\* problem.** Graph $G(t)$ with cells in $B_i(t)$ removed. Entering cell $v$ costs $\kappa_i(v) = \omega_i(v)$, the density value of that cell.

**RRT\* problem.** Free space $X_{\text{free}} = [0,W)\times[0,H) \setminus \bigcup_{c \in S(t) \cup B_i(t)} \text{square}(c)$. Path cost is the **weighted length** $\int_{\text{path}} \omega_i(p)\, ds$, which equals the Euclidean length when $Q_i = \varnothing$ or $\lambda_{\text{prox}} = 0$. The start is the centre of $x_i$ and the goal region is $\text{square}(g_i)$. A segment is collision-free iff every cell its **supercover** touches is free; at an exact lattice-corner crossing all touching cells must be free.

Both planners therefore minimise the same proximity-weighted travel density under identical hard constraints. They differ only in their path spaces: 4-connected lattice paths vs polylines. **Lemma 5.1 (Guaranteed):** under the supercover rule, two free cells are connected in continuous free space iff they are connected in the 4-connected grid graph of free cells. A segment that crosses between diagonal cells passes through a side cell or an exact corner, and an exact corner requires all touching cells free. Consequently grid BFS decides RRT\* feasibility exactly.

### 5.3 Planning procedure and goal-availability policy

Identical for both planners:

1. **Invalid query** (start or goal out of bounds): status `invalid_query`. A goal inside $S(t)$ can only arise from a map event, because scenario validation rejects such tasks. It is handled as statically unreachable (step 4).
2. **Stage 1, constrained.** Plan with $S(t) \cup B_i(t)$ and density $\omega_i$. If BFS on the constrained grid shows the goal is unreachable, skip to stage 2 without spending RRT\* iterations (Lemma 5.1).
3. **Stage 2, static fallback.** If stage 1 is infeasible because of dynamic blocking, plan with $S(t)$ only, keeping $\omega_i$. If that succeeds, the result is marked `degraded = true`. The robot follows it; the controller, shield and arbiter prevent entering occupied cells. The monitor retries a non-degraded plan every $T_{\text{cd}}$ steps.
4. **Statically unreachable goal** ($D_t(x_i, g_i) = \infty$, e.g. a map event blocked the corridor or the goal cell): status `unreachable`.
   - In mode to_pickup, the robot releases its task. The task returns to pending and is excluded from auctions while unavailable (§4), and a re-auction is triggered.
   - In mode to_dropoff (task already picked), the robot keeps the task, stays in place (suggested action $\textsf{stay}$), and retries after $T_{\text{cd}}$ steps. This time is counted as *blocked-goal time*.
5. **Budget exhaustion (RRT\* only):** the goal is reachable by Lemma 5.1, but no path was found within the iteration budget. Status `budget_exhausted`: the robot stays and retries after $T_{\text{cd}}$.

**Occupied goal cells.** When the goal is currently or predictedly occupied by a dynamic obstacle, the goal stays valid. The robot approaches and waits until the cell is free. Waiting is enforced by the shield (if on), by the controller's own choice, or by the arbiter.

### 5.4 Path monitor and replanning

Robot $i$ follows $P_i$ with progress index $\ell_i$ (it is at $c_{\ell_i}$). After each step:

- **Advance / rejoin.** If $x_i(t+1) = c_{\ell'}$ for some $\ell' \in \{\ell_i, \dots, \ell_i + H_r\}$, set $\ell_i \leftarrow \ell'$ (no replanning).
- Otherwise the robot is **off-path**: trigger T3 below.

Replanning triggers (at most one planner call per robot per step):

| ID | Trigger | Attribution |
|---|---|---|
| T1 | Goal changed (new assignment, pickup completed) | allocation |
| T2 | One of the next $H_v$ path cells (excluding the goal) entered $S(t) \cup B_i(t)$ | environment |
| T3 | Robot off-path after its own action (deviation) | controller |
| T4 | **Stall**: valid plan, but $\ell_i$ has not advanced for $T_{\text{stall}}$ consecutive steps while $c_{\ell_i+1}$ is occupied. That cell, unless it is the goal, is added to a per-robot temporary block list for $T_{\text{stall}}$ steps, and the robot replans (detour). | interaction |
| T5 | Previous call failed and the cooldown $T_{\text{cd}}$ elapsed | retry |
| T6 | Current plan is `degraded` and $T_{\text{cd}}$ elapsed | retry |

The monitor is **controller-agnostic**: it applies identically to the baseline, reference-follower and learned controllers. Every planner call is logged with its trigger.

**Suggested action.** $\hat a_i^{\text{plan}}$ is the action moving from $x_i$ to $c_{\ell_i+1}$, or $\textsf{stay}$ if the robot has no valid plan or is at its goal.

### 5.5 Reference plans vs executed motion

**What the reference planners guarantee** (precise statements in `04-algorithms.md` §2–3):

| Property | A\* | RRT\* |
|---|---|---|
| Returned path valid for the **query snapshot** (free of $S(t) \cup B_i(t)$, 4-connected, correct endpoints) | Guaranteed | Guaranteed (discretisation lemma) |
| Decides reachability on the snapshot | Guaranteed (complete on a finite graph) | Guaranteed via BFS guard (Lemma 5.1); the RRT\* search itself is not complete with a finite budget |
| Optimal on the snapshot | Guaranteed: minimum $\kappa_i$-cost among 4-connected lattice paths | Conditional: asymptotically optimal ($n \to \infty$) in Euclidean length when $Q_i = \varnothing$ or $\lambda_{\text{prox}} = 0$, under the radius condition of `04` §3. Not claimed for the weighted cost or for finite budgets. Not step-optimal after discretisation. |
| Valid at later times | Not claimed: obstacles move, map events occur, other robots are not modelled | Not claimed (same reasons) |

**The executed trajectory is chosen by the controller.** MADQL, PPO, the baseline and the reference follower select $a_i(t)$ freely within the mask. They may follow, wait, yield, side-step or leave the reference path. Planner properties therefore **do not transfer** to executed trajectories or to task completion.
- An executed leg can be longer than the reference path.
- A goal reachable on every snapshot can still go unreached because of deadlock, livelock, persistent occupation, or a learned policy that never approaches it.
- The physical invariants (Lemma 3.1) hold for **every** controller, because they are enforced by the arbiter, not by the planner.

**Execution layers.**

| Layer | Always on? | Rule | Effect on an unsafe action |
|---|---|---|---|
| Action mask | Yes | $M_i(t)$ excludes out-of-bounds and $S(t)$ targets | Cannot be selected: probability 0 for PPO, excluded from the argmax for MADQL, never proposed by the rule-based controllers |
| Safety shield | Configurable: `off` (default in training and main experiments) or `predictive` | If the chosen target is in $Z_i(t) \cup \hat Z_i(t)$ (current or predicted obstacle cell), replace the action by $\textsf{stay}$ and log a shield intervention | Prevents predictable robot–obstacle conflicts. A heuristic, not a formal guarantee (obstacles can turn). Cannot address robot–robot conflicts, because decentralised controllers do not know other robots' choices. |
| Arbiter | Yes | Resolution of §3 | Cancels conflicting moves; the conflict is recorded and penalised (§7) |

The shield is off by default during training so that learned policies receive the conflict signal themselves. It is evaluated as an ablation in E5.

**Measuring how the controller uses the reference** (definitions in §9.3): plan adherence, deviation rate, replanning by trigger, executed-to-reference ratios per leg, and paired comparison against the **reference follower**. The reference follower is a diagnostic controller that always plays $\hat a_i^{\text{plan}}$.

---

## 6. Multi-agent decision problem (Dec-POMDP)

The low-level control problem is a decentralised partially observable Markov decision process

$$
\mathcal{M} = \langle R, \mathcal{S}, \{\mathcal{A}\}_{i\in R}, \mathcal{T}, \{r_i\}_{i\in R}, \{\Omega_i\}_{i \in R}, O, \gamma \rangle.
$$

- **State** $s_t \in \mathcal{S}$ includes:
  - robot positions, modes $\mu_i$, assignments $\tau_i$ and goals $g_i$;
  - obstacle positions, velocities and motion-model states;
  - the static map $S(t)$ (version) and scheduled future events;
  - task statuses, release times and ages;
  - the allocator's and planners' internal state (reference paths, progress indices, cooldowns, temporary block lists);
  - the generators' states.

  Assignment, task phase and map are part of the state. This matters for reward shaping (§7).
- **Transition** $\mathcal{T}(s_{t+1} \mid s_t, a_t)$ applies, in order: event application (task releases, map events), obstacle motion, allocation update, path monitoring and replanning, then the execution layers and arbitration of §3/§5.5, then pickups and deliveries. It is stochastic through arrivals, obstacle motion and RRT\* sampling.
- **Observation** $o_i = O_i(s_t)$: egocentric, local, fixed size (§6.1).
- **Discount** $\gamma \in (0,1)$, default 0.99.

The allocator and planners are **part of the environment** from the learner's perspective. The policy learns how to *execute* plans cooperatively. The responsibilities and limits of this split are given in §5.5. `03-architecture.md` ADR-2 records the alternatives.

### 6.1 Observation

Robot $i$ observes a $(2r+1) \times (2r+1)$ window centred on $x_i$ (Chebyshev radius $r$, default 3, i.e. 7×7). Cells outside the grid read as static obstacles. Channels ($C_o = 7$):

| Ch. | Content |
|---|---|
| 0 | Static obstacle or out of bounds |
| 1 | Dynamic obstacle (current) |
| 2 | Dynamic obstacle predicted next position $\hat z_k(t+1)$ |
| 3 | Other robot |
| 4 | Other robot's effective task priority $\pi / P$ (0 if idle) |
| 5 | Own reference path, next $L_o$ cells |
| 6 | Own goal cell (if within window) |

plus a scalar vector

$$
u_i = \Big[\tfrac{g_i - x_i}{\max(W,H)},\; \tfrac{\bar D(x_i, g_i)}{D_{\text{norm}}},\; \text{onehot}(\mu_i),\; \tfrac{\pi_{\tau_i}}{P},\; \text{onehot}(\hat a_i^{\text{plan}}),\; \mathbb{1}[\text{plan valid}]\Big] \in \mathbb{R}^{13},
$$

with goal-dependent entries set to 0 when idle. The sensing radius for channels 1–4 equals the window radius ($r_s = r$). The observation dimension is $d_o = C_o (2r+1)^2 + 13 = 356$ for $r = 3$. It is independent of $N$, $W$ and $H$ [Req FR-RL-5].

### 6.2 Centralised state for the critic (CTDE)

The MAPPO critic receives $s_i^c = [o_i,\, \phi(s_t)]$, where $\phi$ is a fixed-size global summary:
- channels {static obstacles, dynamic obstacles, robots, pending pickups weighted by $\tilde w$}, each average-pooled into $G \times G$ blocks (default $G = 8$);
- scalars {fraction delivered, $t/T_{\max}$, pending tasks$/N$}.

Size $4G^2 + 3$. IPPO uses $s_i^c = o_i$.

### 6.3 Episode end

- **Termination:** every released task is delivered and no future releases are scheduled. In this state all robots are idle.
- **Truncation:** $t = T_{\max}$. Truncation is a time limit, *not* a terminal state of the process. Value targets bootstrap from the value of the final state (`04-algorithms.md` §6–7).

---

## 7. Reward function

[Req FR-RL-4] **Base reward** (the objective the learner is meant to optimise). For robot $i$ at step $t$, let $j = \tau_i(t)$. Let $\mathbb{1}^{\text{pick}}_i$ and $\mathbb{1}^{\text{del}}_i$ indicate a pickup or delivery by $i$. Let $n^{\textsf{RR}}_i$, $n^{\textsf{RO}}_i$ and $n^{\textsf{SH}}_i$ count the robot–robot conflicts, robot–obstacle conflicts (after the shield) and shield interventions in which $i$ is at fault:

$$
r_i^{\text{base}} = -c_{\text{step}}\,\mathbb{1}[\mu_i \neq \textsf{idle}]
+ R_{\text{pick}} \mathbb{1}^{\text{pick}}_i + R_{\text{del}}\, w(\pi_j)\, \mathbb{1}^{\text{del}}_i
- c_{\text{rr}}\, n^{\textsf{RR}}_i - c_{\text{ro}}\, n^{\textsf{RO}}_i - c_{\text{sh}}\, n^{\textsf{SH}}_i .
$$

The terms are time pressure, task completion weighted by priority, and safety. Conflicts are penalised as proposed conflicts: the arbiter prevented them, but they would have been collisions without the emergency stop.

### 7.1 Potential-based shaping

**Phase-aware potential.** For each robot, define the remaining-route length in the current state:

$$
\Lambda_i(s) = \begin{cases}
\bar D(x_i, p_j) + \bar D(p_j, d_j) & \mu_i = \textsf{to\_pickup} \\
\bar D(x_i, d_j) & \mu_i = \textsf{to\_dropoff} \\
0 & \mu_i = \textsf{idle},
\end{cases}
\qquad
\Phi_i(s) = -\frac{\Lambda_i(s)}{D_{\text{norm}}},
$$

where $j = \tau_i$, $\bar D$ uses the static map $S(t)$ contained in $s$, and $D_{\text{norm}} = W + H$.
- $\Phi_i$ is a bounded function of the state alone: assignment, phase and map are state components (§6).
- It is **continuous across pickup**. At the pickup step $x_i = p_j$, so $\Lambda_i = 0 + \bar D(p_j, d_j)$ before and $\bar D(p_j, d_j)$ after.
- It jumps at assignment changes and at map events. This is allowed: those are state changes, and the shaping term below accounts for them exactly.
- Dynamic obstacles are deliberately excluded from $\Phi$. The potential does not depend on which planner is selected, which keeps the planner × learner comparison fair.

**Shaped reward.** $F_i(s_t, s_{t+1}) = \gamma\,\Phi_i(s_{t+1}) - \Phi_i(s_t)$. Both terms are evaluated on the actual states, with no goal held fixed. The individual reward is $r_i^{\text{ind}} = r_i^{\text{base}} + \zeta F_i$.

**Team mixing (collaboration).** With team-spirit coefficient $\beta \in [0,1]$:

$$
r_i = (1-\beta)\, r_i^{\text{ind}} + \frac{\beta}{N} \sum_{k \in R} r_k^{\text{ind}}
= \underbrace{(1-\beta)\, r_i^{\text{base}} + \frac{\beta}{N}\sum_k r_k^{\text{base}}}_{=: \; r_i^{\text{team-base}}}
+ \zeta\big(\gamma \Psi_i(s_{t+1}) - \Psi_i(s_t)\big),
\quad \Psi_i = (1-\beta)\Phi_i + \frac{\beta}{N}\sum_k \Phi_k .
$$

The shaping part of the mixed reward is again potential-based, with potential $\Psi_i$, a state function. $\beta = 0$ gives independent learners and $\beta = 1$ a fully shared team reward.

**Invariance argument and its scope.**
- *Telescoping (Guaranteed).* For any trajectory and any joint policy, including observation-based and stochastic ones, $\sum_{t=0}^{T-1} \gamma^t F(s_t, s_{t+1}) = \gamma^T \Psi_i(s_T) - \Psi_i(s_0)$.
- *Terminating episodes (Guaranteed).* At termination all robots are idle, so $\Psi_i(s_T) = 0$. The expected shaped return of agent $i$ therefore equals its expected base return minus $\zeta\,\mathbb{E}[\Psi_i(s_0)]$, a constant independent of the joint policy (the initial-state distribution is fixed by the scenario). Hence every agent's ranking of joint policies is unchanged. In particular, the set of Nash equilibria and, for $\beta = 1$, the set of team-optimal joint policies are preserved. This matches Ng, Harada & Russell (1999) for MDPs and Devlin & Kudenko (2011) for stochastic games, and the telescoping argument covers our partially observable, observation-based policies directly.
- *Truncated episodes (Conditional).* At $t = T_{\max}$ the boundary term $\gamma^{T}\Psi_i(s_T)$ is policy-dependent. Invariance then holds for the **continuing** objective that bootstrapped value targets estimate, treating the time limit as non-terminal. It does not hold for the literal finite-horizon return. Because the time limit is not part of the observation, this is the standard partial-episode-bootstrapping approximation, documented as a limitation.
- *Not claimed.* Invariance concerns which policies are optimal or equilibria. It says nothing about convergence of DQN or PPO with function approximation, nor about whether learning reaches those policies.

**Default weights** (initial; tuned in Issues #5–#6 and recorded there): $c_{\text{step}} = 0.01$, $\zeta = 1.0$, $R_{\text{pick}} = 0.5$, $R_{\text{del}} = 1.0$, $c_{\text{rr}} = 1.0$, $c_{\text{ro}} = 1.0$, $c_{\text{sh}} = 0.5$, $\beta = 0.3$. The trade-off between efficiency, safety and collaboration is examined by a reward ablation in E4.

---

## 8. Task allocation as an assignment problem

At each allocation event, let $R_f$ be the robots eligible for (re)assignment: modes idle or to_pickup, since carried tasks are never pre-empted. Let $J_p$ be the *available* tasks with status pending or assigned (not picked). The auction solves

$$
\max_{X} \sum_{i \in R_f} \sum_{j \in J_p} a_{ij}\, X_{ij}
\quad \text{s.t.} \quad
\sum_{j} X_{ij} \le 1 \;\; \forall i, \qquad
\sum_{i} X_{ij} \le 1 \;\; \forall j, \qquad
X_{ij} \in \{0, 1\},
$$

with **benefit**

$$
a_{ij}(t) = M + \kappa\, \tilde w_j(t) - c_{ij} + h\, \mathbb{1}[\tau_i = j], \qquad c_{ij} = D_t(x_i, p_j) + D_t(p_j, d_j),
$$

where pairs with $c_{ij} = \infty$ are infeasible. The terms:
- **Priority value** $\kappa\,\tilde w_j$, with aging (§4).
- **Travel cost** $c_{ij}$. Its robot-dependent part $D(x_i, p_j)$ encodes availability and proximity.
- **Switching hysteresis** $h \ge 0$: a robot keeps its current task unless another assignment is better by more than $h$.
- **Cardinality offset** $M = 1 + k(B + c_{\max})$, with $B = \max_j \kappa \tilde w_j + h$, $c_{\max} = \max_{ij: c_{ij}<\infty} c_{ij}$ and $k$ as below. It makes every feasible benefit positive and every larger-cardinality matching strictly better: for matchings of size $q+1$ and $q \le k-1$, $(q+1)(M - c_{\max}) > q(M + B)$ because $M > qB + (q+1)c_{\max}$. **Guaranteed:** the auction returns a **maximum-cardinality** feasible matching, so no eligible robot idles while a task it can reach is unassigned. Among maximum-cardinality matchings it maximises $\sum (\kappa \tilde w_j - c_{ij} + h\mathbb{1}[\tau_i = j])$.

### 8.1 Weighted priority vs strict priority

The design uses **weighted priority**. Priority is traded against travel cost at a fixed exchange rate $\kappa$ (distance units per unit of effective weight).
- *Single-robot view.* With other assignments fixed, robot $i$ prefers task $j$ to task $l$ iff $\kappa(\tilde w_j - \tilde w_l) > c_{ij} - c_{il}$, adjusted by $\pm h$ for its current task. A higher-priority task "buys" a detour of up to $\kappa(\tilde w_j - \tilde w_l)$ cells. With defaults $\kappa = 10$ and $w \in \{1,2,4\}$, a level-3 task outweighs a level-1 task until it is about 30 cells farther away, before aging.
- *Fleet view.* Among maximum-cardinality matchings, the auction maximises total weighted priority minus total travel. Under scarcity it chooses the set of served tasks and their robots jointly.
- **Not guaranteed:** that a higher-priority task is always assigned before a lower-priority one. A much closer low-priority task can be preferred. That is the intended efficiency trade-off.
- *Strict (lexicographic) priority*, which never serves a lower level while a higher one is unassigned, could be obtained with $\kappa$ larger than any possible travel-cost difference summed over a matching, or with sequential per-level auctions. It is **not adopted**: it can send a distant robot across the map for a marginally more urgent task and degrades throughput. The weighted design is validated empirically (E3: priority-sensitive metrics with $\kappa > 0$ vs $\kappa = 0$, and the travel-cost price paid).

**Rectangular case and integrality.** Pad to a square $k \times k$ problem with $k = \max(|R_f|, |J_p|)$. Dummy pairs *and infeasible pairs* get benefit 0 (`04-algorithms.md` §4.1 proves this preserves the optimum). Benefits are scaled by $s$ and rounded to integers. Bertsekas' auction with final $\varepsilon < 1/k$ then returns an **optimal** assignment for the integer benefits. The rounding changes each benefit by at most $1/(2s)$, so the total by at most $k/(2s)$, in original units.

**Duality view ("market-based").** Tasks are goods with prices $q_j$, and a robot's net value for good $j$ is $a_{ij} - q_j$. The auction maintains **ε-complementary slackness**: $a_{i j_i} - q_{j_i} \ge \max_{j}(a_{ij} - q_j) - \varepsilon$ for every assigned pair. Every robot holds a good within $\varepsilon$ of its best deal at current prices, which implies the optimality bound in `04-algorithms.md` §4.1.

**Pre-emption semantics** [Req FR-TASK-5]. A robot in mode to_pickup can lose or switch its task only through the auction and only beyond the hysteresis margin. Picked tasks are never reassigned.

---

## 9. Evaluation metrics

[Req FR-EXP-1] For an episode with released tasks $J_r$ and delivered tasks $J_d \subseteq J_r$.

### 9.1 Task completion and efficiency

| Metric | Definition | Spec metric |
|---|---|---|
| Completion rate | $\lvert J_d\rvert / \lvert J_r\rvert$ | Task completion |
| Makespan | $T_{\text{ms}} = \max_{j \in J_d} C_j$, reported only if $J_d = J_r$ | Task completion time |
| Mean flow time | $\bar F = \frac{1}{\lvert J_d\rvert}\sum_{j \in J_d}(C_j - \rho_j)$ | Task completion time |
| Flow time per priority level | $\bar F^{(\pi)}$, mean over delivered tasks of level $\pi$ | Priority |
| Priority-weighted flow time | $F_w = \sum_{j \in J_d} w(\pi_j)(C_j - \rho_j) / \sum_{j\in J_d} w(\pi_j)$ | Priority |
| Waiting-time tail per level | max and 95th percentile of pickup waiting time $(\text{pickup time} - \rho_j)$ per level, with undelivered tasks counted at their age at episode end (right-censored, flagged) | Starvation mitigation |
| Throughput | $\lvert J_d\rvert$ per 100 steps (continuous-arrival scenarios) | Efficiency |
| Path efficiency | $\eta_{\text{path}} = \sum_i L_i^\* / \sum_i \text{moves}_i$, where $L_i^\*$ is the sum of static shortest distances of the legs robot $i$ completed | Efficiency |
| Utilisation | fraction of robot-steps with $\mu_i \neq \textsf{idle}$ | Efficiency |
| Wait ratio | fraction of non-idle robot-steps with action $\textsf{stay}$ | Efficiency |
| Blocked moves | count of cascade freezes (§3) per 100 robot-steps | Efficiency |
| Blocked-goal time | robot-steps holding a task whose goal is statically unreachable | Dynamic handling |

"Overall efficiency" is the vector (completion rate, throughput per robot, $\eta_{\text{path}}$, utilisation). No arbitrarily weighted composite is used.

### 9.2 Safety and conflict metrics

All conflict metrics are computed from conflict-event **sets** keyed by $(t, \text{type}, \text{entity pair})$, not only from counts.

| Metric | Definition |
|---|---|
| Proposed robot–robot conflicts | $N^{\textsf{RR}} = \lvert\{e \in \mathcal{K}(s_t, a_t^{\text{arb}}) : \text{type} \in \{\textsf{RRv}, \textsf{RRs}\}\}\rvert$ summed over $t$, where $a^{\text{arb}}$ is the proposal reaching the arbiter. Reported per 100 robot-steps, split by vertex and swap. |
| Proposed robot–obstacle conflicts | $N^{\textsf{RO}}$, analogously for $\textsf{ROv}, \textsf{ROs}$ |
| Invalid proposals | $N^{\textsf{INV}}$ (0 for every masked controller; non-zero indicates a masking defect) |
| Shield interventions | $N^{\textsf{SH}}$: actions replaced by the shield. Each prevents a *predicted* robot–obstacle conflict; not all would have materialised, because predictions are heuristic. |
| Conflicts prevented by arbitration | $N^{\text{prev}} = N^{\textsf{RR}} + N^{\textsf{RO}} + N^{\textsf{INV}}$. Every proposed conflict reaching the arbiter is prevented by construction; reported to make the emergency-stop load explicit. |
| Physical collisions | $N^{\text{phys}}$: post-resolution occupancy violations. **Invariant = 0** (Lemma 3.1); checked every step. |

**Counterfactual comparison with the reference plan.** At each visited state $s_t$, let $a^{\text{ref}}_t = (\hat a^{\text{plan}}_1, \dots, \hat a^{\text{plan}}_N)$ be pure plan-following, with masked components replaced by $\textsf{stay}$. Let $a^{\text{ctrl}}_t$ be the controller's raw proposal before the shield. With $\mathcal{K}^{\text{ref}}_t = \mathcal{K}(s_t, a^{\text{ref}}_t)$ and $\mathcal{K}^{\text{ctrl}}_t = \mathcal{K}(s_t, a^{\text{ctrl}}_t)$:

$$
N^{\text{avoided}} = \sum_t \big|\mathcal{K}^{\text{ref}}_t \setminus \mathcal{K}^{\text{ctrl}}_t\big|, \qquad
N^{\text{introduced}} = \sum_t \big|\mathcal{K}^{\text{ctrl}}_t \setminus \mathcal{K}^{\text{ref}}_t\big|, \qquad
N^{\text{persisted}} = \sum_t \big|\mathcal{K}^{\text{ref}}_t \cap \mathcal{K}^{\text{ctrl}}_t\big|.
$$

Avoidance rate $= N^{\text{avoided}} / \sum_t |\mathcal{K}^{\text{ref}}_t|$ (undefined if the denominator is 0).

**"Collisions avoided" (specification metric) := $N^{\text{avoided}}$, reported together with the avoidance rate and $N^{\text{introduced}}$.** Why this is a defensible proxy:
- Obstacle moves at step $t$ do not depend on robot actions (§2), so $\mathcal{K}^{\text{ref}}_t$ is exactly the set of conflicts that faithful plan-following would have proposed *from the same state*.
- Its set difference with the controller's conflicts credits only conflicts the controller's decision removed. $N^{\text{introduced}}$ exposes conflicts the controller created that the plan would not have.

*Limitations:*
- The counterfactual is one step deep and evaluated on the controller's own state distribution.
- It does not credit avoiding states in which conflicts become likely, nor penalise leading the team into them.

It is therefore complemented by the **episode-level paired comparison** with the reference-follower controller on identical seeds (proposed conflicts per 100 robot-steps, flow times). That comparison is the primary safety comparison between controllers. For the reference follower itself, $N^{\text{avoided}} = N^{\text{introduced}} = 0$ by definition.

### 9.3 Plan-adherence metrics

| Metric | Definition |
|---|---|
| Plan adherence | fraction of non-idle robot-steps with a valid plan in which $a_i = \hat a^{\text{plan}}_i$ |
| Deviation rate | T3 (off-path) events per 100 non-idle robot-steps |
| Replans by trigger | counts of T1–T6 per episode (planner calls attributed to allocation, environment, controller, interaction, retry) |
| Leg execution ratio | for each completed leg, executed steps $/$ length of the first reference path of that leg; also executed moves $/$ static shortest distance |
| Planning status mix | fractions `ok`, `ok` + degraded, `unreachable`, `budget_exhausted` |

### 9.4 Computational metrics

Planner time per call, nodes expanded or iterations, neighbour-query and cost-propagation work (RRT\*), auction time and bids per call, policy inference time per step, wall time per simulation step, training environment steps and wall-clock time to a performance threshold.

---

## 10. Overall objective

Over the scenario distribution $\mathcal{D}$ the system aims at the multi-objective goal

$$
\min \; \mathbb{E}\big[F_w\big] \quad \text{while minimising } \mathbb{E}\big[N^{\textsf{RR}} + N^{\textsf{RO}}\big] \text{ and maximising the completion rate}.
$$

The RL base reward of §7 is a *scalarisation* of it. The evaluation reports the objectives separately so that trade-offs remain visible.

---

## 11. Symbol table

| Symbol | Meaning | Default |
|---|---|---|
| $W, H$ | grid width, height | scenario |
| $N = \lvert R\rvert$, $K$ | robots, dynamic obstacles | scenario |
| $P$ | priority levels | 3 |
| $T_{\max}$ | horizon | scenario |
| $r = r_s$ | observation window radius = sensing radius (Chebyshev) | 3 |
| $\lambda_{\text{prox}}, r_{\text{infl}}$ | proximity penalty, inflation radius (both planners) | 2.0, 1 |
| $H_v, H_r$ | validity look-ahead, rejoin window | 5, 3 |
| $T_{\text{cd}}, T_{\text{stall}}$ | replan cooldown, stall threshold | 3, 4 |
| $D_{\text{cap}}, D_{\text{norm}}$ | distance cap, normaliser | $WH$, $W+H$ |
| $\alpha$ | aging rate | 0.01 |
| $\kappa, h, s$ | priority scale, hysteresis, integer scale | 10, 2, 100 |
| $\gamma$ | discount | 0.99 |
| $\zeta, \beta$ | shaping scale, team-spirit | 1.0, 0.3 |

---

## 12. Summary of guarantees

| # | Property | Status | Basis |
|---|---|---|---|
| G1 | No physical collision after arbitration (any controller) | **Guaranteed** | Lemma 3.1, `04` §1 |
| G2 | Masked controllers never propose moves into static obstacles or out of bounds | **Guaranteed** | Mask definition, events before masks (§1) |
| G3 | Determinism for fixed scenario and seed | **Guaranteed** (by design; verified by test X-1) | ADR-8 |
| G4 | A\* path is $\kappa$-optimal and valid for the query snapshot; unreachability correctly reported | **Guaranteed** | `04` §2.2 |
| G5 | RRT\* path is valid for the query snapshot; best cost non-increasing in iterations | **Guaranteed** | `04` §3 |
| G6 | Continuous feasibility ⇔ 4-connected grid feasibility under the supercover rule | **Guaranteed** | Lemma 5.1 |
| G7 | RRT\* asymptotic optimality | **Conditional**: $n \to \infty$, Euclidean cost ($Q_i = \varnothing$ or $\lambda_{\text{prox}} = 0$), radius constant per `04` §3 | Karaman & Frazzoli 2011; Solovey et al. 2020 |
| G8 | RRT\* finds a path within a finite budget | **Not claimed** (measured: success rate) | — |
| G9 | RRT\* runtime $O(n \log n)$ | **Not claimed**. Unconditional and conditional bounds in `04` §3.6 | — |
| G10 | Auction returns a maximum-cardinality matching, optimal for integer benefits (within $k\varepsilon$ for fixed $\varepsilon$) | **Guaranteed** | `04` §4.1, §8 |
| G11 | Higher-priority tasks always assigned first | **Not claimed** (weighted priority, §8.1) | — |
| G12 | Aging properties P1, P2, P3 | **Guaranteed** | §4 |
| G13 | Starvation freedom | **Not claimed**; mitigation measured in E3 | §4 |
| G14 | Shaping preserves optimal policies and equilibria | **Guaranteed** for terminating episodes; **Conditional** (continuing objective) under truncation | §7.1 |
| G15 | Reference-planner optimality or completeness carries over to executed trajectories or task completion | **Not claimed** | §5.5 |
| G16 | Learned policies converge, cooperate, or outperform baselines | **Not claimed** a priori; established or refuted only by experiments | `05` Part B |
