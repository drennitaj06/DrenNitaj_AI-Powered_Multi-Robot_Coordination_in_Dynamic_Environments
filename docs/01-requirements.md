# 01 — Requirements Specification

| Field | Value |
|---|---|
| Project | AI-Powered Multi-Robot Coordination in Dynamic Environments |
| Course | Design and Analysis of Algorithms |
| Document role | **Source of truth for requirements.** Other documents reference requirement IDs from here and do not restate them. |
| Status | Revision 2 (Issue #1, after design review; changes in `verification-log.md` V-1) |

---

## 1. Purpose and scope

The system coordinates a team of mobile robots in a two-dimensional, obstacle-filled, **changing** environment. The robots must complete pickup-and-delivery tasks, avoid collisions and re-route when obstacles or new tasks appear. The system combines four algorithmic layers:

1. **Market-based task allocation**: an auction assigns tasks to robots according to priority and availability.
2. **Online path planning**: A\* or RRT\* computes a reference route and replans when the environment changes.
3. **Multi-agent reinforcement learning (MARL)**: MADQL or PPO learns cooperative, decentralised low-level motion decisions (follow, yield, sidestep, wait).
4. **Simulation**: a discrete-time environment with moving obstacles, map changes and task arrivals.

The deliverable is a **reproducible simulation and experimentation platform**. It demonstrates the algorithms, measures their behaviour and supports a defensible comparison of them. It does not control physical robots.

### 1.1 Original specification (verbatim summary)

| Item | Original text (condensed) |
|---|---|
| Requirement 1 | Implement a multi-agent reinforcement learning algorithm to train robots to cooperate. |
| Requirement 2 | Use an online path-planning algorithm (e.g. A\* or RRT\*) for dynamic obstacle avoidance. |
| Requirement 3 | Incorporate priority handling to resolve task conflicts dynamically. |
| Requirement 4 | Simulate the environment with changing conditions (moving obstacles, new tasks). |
| Training | MADQL or PPO to optimise objectives collaboratively. |
| Path planning | A\* or RRT\* with dynamic obstacle updates. |
| Task allocation | Market-based algorithm (e.g. auction-based assignment) by priority and robot availability. |
| Input | Map (incl. dynamic obstacles); initial robot positions and tasks; reward structure. |
| Output | Simulated execution of the robots achieving their tasks. |
| Metrics | Task completion time; collisions avoided; overall efficiency. |
| Challenges | Reward design (efficiency/safety/collaboration); high-dimensional state and real-time change; scalability. |

### 1.2 Professor's clarification (binding)

The "or" alternatives are **not** optional. **A\*, RRT\*, MADQL and PPO must all be implemented, be selectable at run time and be evaluated.** Deterministic baselines may be added only as extra comparison points. They never replace a required algorithm.

### 1.3 Project-level decisions (binding)

| ID | Decision |
|---|---|
| D-1 | Implementation language: **Python only**. No C++, Java, CMake or cross-language bindings. |
| D-2 | Kaggle datasets have a dedicated development phase and a genuine role in scenarios and experiments (see `06-kaggle-datasets.md`). |
| D-3 | The final `README.md` is written at the end of the project, once the system is verified. |
| D-4 | All Git/GitHub operations are performed by the student through GitHub Desktop / GitHub.com. |

---

## 2. Stakeholders and usage scenarios

| Stakeholder | Need |
|---|---|
| Student (author) | Implement, understand and academically defend every algorithm and decision. |
| Professor / evaluator | Verify that all required algorithms exist, work, are selectable, and are analysed theoretically and empirically. |
| Demo audience | See robots completing tasks, avoiding collisions and reacting to changes, with live metrics. |

**Usage scenarios**

- **US-1 Live demonstration.** The user selects a scenario, a planner (A\*/RRT\*), a controller (MADQL/PPO/baseline) and an allocator, then watches the simulation with overlays and live metrics.
- **US-2 Training.** The user trains a MADQL or PPO policy from a configuration file with a fixed seed. Checkpoints and learning curves are stored.
- **US-3 Experiment.** The user runs a predefined experiment suite headlessly. Raw results are written as CSV/JSON, and summary tables and plots are generated.
- **US-4 Verification.** The user runs the automated test suite and gets a structured report.

---

## 3. Functional requirements

Priority: **M** = mandatory (from the specification), **S** = should (needed for academic quality), **C** = could (only if time permits).

### 3.1 Environment and simulation (FR-ENV)

| ID | Requirement | Pri. | Source |
|---|---|---|---|
| FR-ENV-1 | Represent the environment as a 2-D 4-connected occupancy grid with static obstacles. | M | Input |
| FR-ENV-2 | Simulate **dynamic obstacles** that move every time step under configurable motion models (random walk, patrol along waypoints, bouncing linear motion). | M | Req. 4 |
| FR-ENV-3 | Support **map events**: cells that become blocked or unblocked at scheduled or random times. | M | Req. 4, Input |
| FR-ENV-4 | Support **task arrivals during execution** (scheduled list and stochastic arrival process). | M | Req. 4 |
| FR-ENV-5 | Advance time in discrete synchronous steps. All robots and dynamic obstacles act simultaneously. Joint proposals pass through the execution layers (action mask, optional safety shield, arbiter), and the arbiter resolves them deterministically. | M | Output |
| FR-ENV-6 | Detect every **proposed conflict** (robot–robot vertex/swap, robot–obstacle vertex/swap, invalid proposals) as an event (step, type, entities). Prevent all of them through arbitration, and verify every step that no **physical collision** occurs. Compute the same event sets for the reference-plan proposal (counterfactual) to measure conflicts avoided and introduced (model §9.2). | M | Metrics |
| FR-ENV-7 | Be fully deterministic for a given scenario and master seed. | M | NFR-REP |
| FR-ENV-8 | Load scenarios from human-readable configuration files and generate synthetic scenarios procedurally (random density, rooms, warehouse aisles, mazes). | M | Input |
| FR-ENV-9 | Load grid maps in the MovingAI `.map` text format (de-facto standard for grid path-finding benchmarks). | S | Experiments |
| FR-ENV-10 | Record an episode trace (per-step state and events) that can be replayed without re-running the algorithms. | S | Output |

### 3.2 Tasks and priorities (FR-TASK)

| ID | Requirement | Pri. | Source |
|---|---|---|---|
| FR-TASK-1 | A task has a pickup cell, a drop-off cell, an integer priority level, and a release time. | M | Input |
| FR-TASK-2 | Task lifecycle: `unreleased → pending → assigned → picked → delivered`. Each robot carries at most one task. | M | Output |
| FR-TASK-3 | Priority affects allocation through **weighted priority**: priority value is traded against travel cost at a configurable rate (model §8.1). Strict precedence of higher levels is not required. | M | Req. 3 |
| FR-TASK-4 | Priority **aging** grows a pending task's effective priority with waiting time as a **starvation-mitigation** mechanism. Starvation freedom is not guaranteed (model §4), and waiting-time tails are measured. | S | Req. 3 |
| FR-TASK-5 | **Dynamic conflict resolution.** A newly released, more urgent task may pre-empt an assignment whose pickup has not yet happened. Re-assignment is damped by a switching-hysteresis term. | M | Req. 3 |
| FR-TASK-6 | **Right-of-way.** When robots compete for space, the priority of the carried or assigned task is available to the controller: the baseline uses it directly, and the RL observation includes it. | M | Req. 3 |

### 3.3 Task allocation (FR-ALLOC)

| ID | Requirement | Pri. | Source |
|---|---|---|---|
| FR-ALLOC-1 | Implement a **market-based auction** allocator: Bertsekas' auction algorithm for the assignment problem with ε-scaling. | M | Alg. c |
| FR-ALLOC-2 | Bids (benefits) combine task priority, aging, and the robot's path distance to pickup plus pickup-to-drop-off distance. | M | Alg. c |
| FR-ALLOC-3 | Re-run the auction on events: task release, task delivery, robot becoming idle, a robot's goal becoming statically unreachable, and map change. Unavailable tasks are excluded until available again. | M | Req. 3 |
| FR-ALLOC-4 | Provide a **sequential single-item auction** (greedy) as a baseline allocator. | S | Comparison |
| FR-ALLOC-5 | Allocators are selectable by configuration. | M | Clarif. |

### 3.4 Path planning (FR-PLAN)

| ID | Requirement | Pri. | Source |
|---|---|---|---|
| FR-PLAN-1 | Implement **A\*** on the grid, with an admissible and consistent heuristic, over the shared dynamic-constraint set and proximity cost density (model §5.2). | M | Alg. b |
| FR-PLAN-2 | Implement **RRT\*** in the continuous plane over the same constraints: choose-parent and rewiring with a shrinking neighbourhood radius, an exact neighbour index, weighted segment costs, and conversion of the polyline into an executable cell sequence. | M | Alg. b |
| FR-PLAN-3 | Planners share a common interface and are selectable by configuration. | M | Clarif. |
| FR-PLAN-4 | **Online (reactive) replanning**: a controller-agnostic path monitor replans on goal change, path invalidation (map event, sensed or predicted obstacle), deviation, stall, and retry after failure (triggers T1–T6, model §5.4). *Issue #4 clarification, not a change to the requirement:* the same monitor also carries two goal-override triggers added in Issue #4, T7 (deadlock detour) and T8 (endpoint recovery), documented in model §5.4 and `04` §5.5–§5.6. Neither extends the events listed above. | M | Req. 2 |
| FR-PLAN-5 | Report failures explicitly (`invalid_query`, `unreachable`, `budget_exhausted`) and mark static-fallback plans as `degraded`. Never return an invalid path. | M | Correctness |
| FR-PLAN-6 | Expose instrumentation: nodes expanded, iterations, neighbour-query and cost-propagation work, path cost, planning time, trigger. | M | Complexity |
| FR-PLAN-7 | A\* and RRT\* apply **identical** dynamic-obstacle and goal-availability policies through one shared constraint builder and planning procedure (model §5.2–5.3). | M | Correctness |

### 3.5 Multi-agent reinforcement learning (FR-RL)

| ID | Requirement | Pri. | Source |
|---|---|---|---|
| FR-RL-1 | Provide a multi-agent environment interface (parallel, simultaneous actions) over the simulator, with local observations, action masks, per-agent rewards and a centralised state for CTDE critics. | M | Req. 1 |
| FR-RL-2 | Implement **MADQL**: parameter-shared deep Q-learning for all robots, with experience replay, target network, Double-DQN targets, and action masking. | M | Alg. a |
| FR-RL-3 | Implement **PPO** for multiple agents: shared actor; centralised critic (MAPPO) and decentralised critic (IPPO) as a configuration switch; GAE; clipped surrogate objective. | M | Alg. a |
| FR-RL-4 | The reward function balances task progress and completion (efficiency), proposed conflicts (safety) and team outcome (collaboration), with configurable weights. | M | Input, Challenge |
| FR-RL-5 | Observations have a fixed size independent of team size and map size, so a trained policy can run with different numbers of robots. | S | Challenge (scalability) |
| FR-RL-6 | Training is reproducible (seeded), checkpointed, and logs learning curves. Trained policies run as controllers in the simulator. | M | Experiments |
| FR-RL-7 | A deterministic **rule-based controller** (follow reference path, wait/yield by priority) is provided as a non-learning baseline, plus a diagnostic **reference follower** (always plays the planner's suggested action). | S | Comparison |
| FR-RL-8 | **Execution layers** apply to every controller: an always-on action mask (static feasibility), an optional predictive safety shield, and the always-on arbiter (model §5.5). Learned controllers may deviate from the reference path; adherence is measured. | M | Safety |

### 3.6 Interaction, output and experiments (FR-UI, FR-EXP)

| ID | Requirement | Pri. | Source |
|---|---|---|---|
| FR-UI-1 | Interactive 2-D visualisation of the simulation: grid, obstacles, robots, tasks, reference paths, assignments, priorities, live metrics. | M | Output |
| FR-UI-2 | The user can select scenario, planner, allocator and controller (incl. trained checkpoint), and can pause, step and change speed. | M | Clarif. |
| FR-UI-3 | Replay recorded traces. | S | Output |
| FR-UI-4 | Command-line interface for headless simulation, training and experiments. | M | US-2/3 |
| FR-EXP-1 | Compute all metrics defined in `02-mathematical-model.md` §9 for every episode. | M | Metrics |
| FR-EXP-2 | Run experiment suites from configuration files with explicit seed lists. Store raw per-run results and aggregated statistics. | M | Complexity |
| FR-EXP-3 | Produce comparison plots and tables: A\* vs RRT\*; MADQL vs PPO vs baseline; planner × controller combinations; scaling in robots, tasks, map size and obstacle density; static vs dynamic. | M | Clarif., Complexity |
| FR-EXP-4 | Generate scenarios from Kaggle-derived data (task spatial/temporal distributions, maze maps) and evaluate on them. | M | D-2 |

---

## 4. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-REP-1 | Reproducibility | Every stochastic component draws from a generator derived from one master seed via `numpy.random.SeedSequence`. The same seed, config and code version produce identical trajectories (except wall-clock timings). |
| NFR-REP-2 | Reproducibility | Every experiment result records config, seed, code version identifier and environment info (Python, library versions, CPU). |
| NFR-PERF-1 | Performance | A single headless simulation step for 8 robots on a 32×32 grid with A\* completes in < 20 ms on a typical laptop CPU, excluding RL inference. This is a target, measured in Issue #4. |
| NFR-PERF-2 | Performance | Interactive visualisation runs at ≥ 10 simulation steps/s for the demo scenarios. |
| NFR-PERF-3 | Performance | RL training of the reference configuration fits on a CPU-only laptop within a few hours per run. GPU is optional. |
| NFR-SCAL-1 | Scalability | Simulation, planning and allocation support at least 32 robots and 64×64 grids. Learned policies are evaluated at team sizes larger than in training. |
| NFR-MAINT-1 | Maintainability | Modular packages with explicit interfaces (`typing.Protocol`), type hints throughout, `ruff` lint/format clean, `mypy` clean on the library package. |
| NFR-MAINT-2 | Maintainability | No algorithm depends on the visualisation. Everything runs headless. |
| NFR-TEST-1 | Verification | Automated tests (pytest) for every algorithm and simulation rule, including property-based tests against independent oracles. See `05-testing-and-evaluation.md`. |
| NFR-TEST-2 | Verification | Test executions and outcomes are recorded in `verification-log.md`. |
| NFR-DOC-1 | Documentation | Documentation describes only implemented behaviour, apart from clearly-marked plans. It is updated within the issue that changes the behaviour. |
| NFR-PORT-1 | Portability | Supported Python versions: **3.12, 3.13 and 3.14** (`requires-python = ">=3.12,<3.15"`). A version counts as supported only once the CI matrix (`03-architecture.md` §3.3) has passed on it. Development machine: Python 3.14 on Windows 11. Primary CI: Linux; Windows is checked on 3.14. |
| NFR-HON-1 | Integrity | No fabricated data, benchmark numbers or test outcomes. RL claims distinguish *implemented*, *trained* and *empirically shown to perform*. |

---

## 5. Inputs and outputs

### 5.1 Inputs

| Input | Form | Content |
|---|---|---|
| Scenario file | TOML (`configs/scenarios/*.toml`) | Grid size or map file; static obstacles or generator parameters; dynamic obstacles (start, model, parameters); map events; robots (start cells); tasks (pickup, drop-off, priority, release time) or a task-arrival process; horizon $T_{\max}$; seed. |
| Map file | MovingAI `.map` text format | Static occupancy grid (from synthetic generators, Kaggle maze data or benchmark maps). |
| Reward configuration | TOML section | Weights of the reward terms (`02-mathematical-model.md` §7). |
| Algorithm selection | CLI flags / TOML / UI | Planner ∈ {`astar`, `rrtstar`}; allocator ∈ {`auction`, `sequential`}; controller ∈ {`madql`, `ppo`, `baseline`, `reference`} (+ checkpoint path); shield ∈ {`off`, `predictive`}. |
| Training configuration | TOML | Hyper-parameters, curriculum, seeds, budget. |
| Experiment configuration | TOML | Factor grid, seed list, metrics, repetitions. |
| Kaggle-derived parameters | Small processed files (`data/processed/`) | Empirical arrival intensities, spatial origin–destination samples, priority distributions, maze maps (see `06-kaggle-datasets.md`). |

### 5.2 Outputs

| Output | Form |
|---|---|
| Simulated execution | Live visualisation and/or recorded JSON-Lines trace. |
| Episode metrics | One record per episode (CSV/JSON): metrics of §9 of the model. |
| Training artefacts | Checkpoints (`.pt`), learning-curve CSV, config snapshot. |
| Experiment results | Raw CSV per run, aggregated tables, figures (PNG/SVG), summarised in the results register of `05-testing-and-evaluation.md` (§B.5) when produced. |
| Test reports | JUnit XML (generated, not committed) plus summary entries in `verification-log.md`. |

---

## 6. Constraints

| ID | Constraint |
|---|---|
| C-1 | Python only (D-1). Third-party libraries are allowed when justified (`03-architecture.md` §3). |
| C-2 | Commodity CPU hardware. A GPU must not be required. |
| C-3 | Limited project time. Scope is a technically strong academic system, not a research-grade robotics stack. |
| C-4 | Kaggle data must be legally usable for academic, non-commercial work. Raw and derived datasets are **not committed** by default. Only acquisition/preprocessing code, manifests, attribution and small synthetic test fixtures are committed (`06-kaggle-datasets.md` §4.1). |
| C-5 | The repository contains no generated artefacts (results, checkpoints, raw data) except small, documented, intentionally committed files. |

---

## 7. Assumptions (modelling choices, not requirements)

These assumptions make the problem well-defined. They can be changed only through an explicit, documented decision.

| ID | Assumption | Rationale |
|---|---|---|
| A-1 | The world is a discretised 2-D plane. Robots occupy one grid cell and move at most one cell (4-neighbourhood) per step. | Standard in multi-robot path-finding literature. Makes conflicts and actions exact and keeps a discrete action space for DQN. |
| A-2 | Robots are homogeneous: same speed, same capacity (one task), same sensing radius. | Enables parameter sharing in MARL. Heterogeneity is out of scope. |
| A-3 | Robots know the static map. They **sense** dynamic obstacles and other robots only within a local radius $r_s$. | Makes dynamic replanning meaningful, gives a partially observable RL setting, and keeps observation size bounded. |
| A-4 | The allocator is centralised (a "market server"). It knows robot positions and task states. | Typical auction-based multi-robot task allocation with an auctioneer. Its decisions are communicated to robots. |
| A-5 | Dynamic obstacles do not react to robots, except that they never move into a cell occupied by a robot or obstacle at time $t$. They never initiate a conflict with a stationary robot. | Conflicts are then attributable to robot decisions, so the safety metric measures the controllers. |
| A-6 | Pickup and drop-off are instantaneous on arrival at the cell. | Keeps the task model simple. A service time could be added as a constant without changing algorithms. |
| A-7 | Communication is perfect and instantaneous. | Out of scope: communication failures. |
| A-8 | Every robot has a low-level **emergency stop**, modelled by the arbiter. A proposed conflicting move is cancelled before contact (the robot stays in place), and the proposed conflict is recorded and penalised. Physical collisions therefore cannot occur in the model, and controller safety is measured by **proposed conflicts**. | Lets episodes continue, keeps the physical state valid, and gives an exact, controller-attributable safety measure (model §3, §9.2). |
| A-9 | Kaggle data drives *distributions* (where and when tasks appear, how urgent they are, what maps look like). Real-world coordinates are rescaled onto the simulation grid. | Real logistics data has no robot-grid semantics. The mapping is documented and validated in Issue #7. |
| A-10 | Path planning is **reactive and spatial**: each query is solved on a snapshot with a one-step obstacle prediction. There is no space-time search or cell reservation, and other robots are not planner constraints. | Keeps planners as specified (A\*, RRT\*). Inter-robot coordination is the controller's and arbiter's job (model §5.1). |

---

## 8. Expected system behaviour (summary)

1. At $t=0$ the scenario is loaded. Released tasks are auctioned, and robots with tasks obtain reference paths from the selected planner.
2. Each step: new tasks may be released, map events may occur and dynamic obstacles move. The allocator re-auctions on triggering events, and the path monitor triggers replanning where needed. Each robot's controller (MARL policy, baseline or reference follower) selects a masked action from its local observation, which includes the planner's suggested next move. The controller may follow, wait, yield or deviate. The optional shield may replace predictably unsafe moves by *stay*. The arbiter cancels conflicting moves and records proposed conflicts. Pickups, deliveries and metrics are then updated.
3. The episode ends when all tasks are delivered and no further arrivals are scheduled (termination), or at $T_{\max}$ (truncation).
4. Metrics are emitted, and the trace is optionally recorded.

---

## 9. Acceptance criteria (project level)

The project is complete when all of the following are demonstrably true.

| ID | Criterion | Verified by |
|---|---|---|
| AC-1 | A\*, RRT\*, MADQL, PPO and the auction allocator are implemented, and each can be selected via CLI and UI. | Tests + demo |
| AC-2 | A\* returns optimal paths: path length equals the BFS shortest-path length on ≥ 1 000 randomly generated grids (property test). | `tests/` + verification log |
| AC-3 | RRT\* returns only paths that are valid for the query snapshot. Its best cost is non-increasing over iterations of one run. The tree-cost invariant holds, and the k-d neighbour index agrees with the brute-force oracle. | Tests |
| AC-4 | The auction allocator's total benefit is within $k\varepsilon$ of the Hungarian optimum (and exactly optimal for integer benefits with $\varepsilon<1/k$), checked against `scipy.optimize.linear_sum_assignment`. | Tests |
| AC-5 | For every controller, no physical collision occurs: no shared cell, no swap-through, no robot in an obstacle. Every proposed conflict is recorded as an event (step, type, entities), and the conflict sets of the controller and of the reference proposal are both recorded. | Property tests (arbiter vs naive oracle) |
| AC-6 | Dynamic changes (moving obstacles, map events, new tasks) trigger replanning (with attributed triggers) and re-auctioning. In the designated dynamic demo scenarios, the **baseline** controller delivers every task that remains available. Completion rates of all controllers are reported. No end-to-end completion guarantee is claimed (model §12, G15). | Integration tests + demo |
| AC-7 | MADQL and PPO policies are trained with ≥ 3 seeds each. Learning curves are reported, and trained policies are evaluated against the baseline and the reference follower on held-out scenario seeds with paired statistics. Results are reported honestly, including the case where a learned policy does not beat the baseline. | Experiment reports |
| AC-8 | Theoretical time/space complexity is documented for every major algorithm, distinguishing worst-case, expected (with its probability model) and amortised bounds, and compared with measured scaling behaviour. | `04-algorithms.md` + experiment reports |
| AC-9 | At least two Kaggle datasets are integrated through a documented, reproducible preprocessing pipeline and used in evaluation scenarios. | Issue #7 deliverables |
| AC-10 | The full test suite passes, and results are recorded in `verification-log.md`. | Verification log |
| AC-11 | Documentation is consistent with the implementation, and the final README describes the verified system. | Final review |
| AC-12 | A\* and RRT\* return the same planning status and `degraded` flag on identical queries (shared constraints and goal policy). | Test X-11 |
| AC-13 | Every mathematical guarantee stated anywhere in the documentation appears in model §12 with its status (guaranteed, conditional, not claimed) and is backed by a proof sketch, a test, or a citation. Weighted priority and aging are validated by the E3 priority-sensitivity analysis, not by claims of strict precedence or starvation freedom. | Review + tests + E3 |

---

## 10. Traceability matrix

| Spec item | Requirements | Design | Issue |
|---|---|---|---|
| Req. 1 / Alg. a (MARL) | FR-RL-1…8 | Model §6–7, Algorithms §6–7 | #5, #6 |
| Req. 2 / Alg. b (planning) | FR-PLAN-1…7 | Model §5, Algorithms §2–3 | #3 |
| Req. 3 (priority) | FR-TASK-3…6, FR-ALLOC-3 | Model §4, §8, Algorithms §4–5 | #4 |
| Req. 4 (dynamic env.) | FR-ENV-2…5 | Model §2–3, Algorithms §1 | #2 |
| Alg. c (auction) | FR-ALLOC-1…5 | Model §8, Algorithms §4 | #4 |
| Input / Output | §5, FR-ENV-8…10, FR-UI | Architecture §5–6 | #2, #8 |
| Metrics | FR-EXP-1 | Model §9 | #2 (collection), #8 (analysis) |
| Challenges | FR-RL-4/5, NFR-SCAL-1 | Model §6–7, Architecture §7–8 | #5, #6, #8 |
| Kaggle (D-2) | FR-EXP-4 | `06-kaggle-datasets.md` | #7 |

---

## 11. Out of scope

- Physical robots, ROS integration, continuous dynamics, sensor noise models, communication failures.
- Optimal multi-agent path finding solvers (CBS, ICTS) and space-time planning or reservation schemes (space-time A\*, cooperative A\*). Mentioned as related work only.
- End-to-end guarantees of task completion or trajectory optimality for learned controllers (model §5.5).
- Incremental search variants (D\* Lite, LPA\*) and dynamic RRT variants (RRTˣ). These are discussed as alternatives, not implemented.
- Heterogeneous robots, multi-capacity robots, battery/charging.
