# 05 — Testing Strategy and Experimental Evaluation

| Field | Value |
|---|---|
| Document role | **Source of truth for how correctness is verified (tests) and how performance is measured (experiments)**, plus the results register. Per-algorithm test lists live in `04-algorithms.md` and are binding. This document defines the framework, cross-cutting tests, experiment protocol, budgets and statistics. |
| Results of test runs | `verification-log.md` |
| Status | Revision 2 (Issue #1). No tests or experiments have been run yet. |

---

## Part 0: Four kinds of evidence (kept separate)

| Level | Question | Evidence | Possible outcomes |
|---|---|---|---|
| **L1 Algorithmic correctness** | Does the implementation do exactly what the algorithm specifies, and do the guaranteed properties of model §12 hold? | Deterministic and property-based tests against independent oracles | Pass / fail. A failure blocks the issue. |
| **L2 Learning-implementation sanity** | Are the RL update rules implemented correctly? | Tiny MDPs with known solutions, hand-computed targets and advantages | Pass / fail |
| **L3 Learned-policy effectiveness** | Do trained MADQL/PPO policies perform well, cooperate, and beat baselines? | Experiments with multiple training seeds, held-out test seeds and paired statistics | Any outcome, **including negative**, reported as found |
| **L4 Complexity** | How do cost and quality scale? | Theory in `04` (worst, expected, amortised, conditional) vs measured operation counts and runtimes | Agreement or disagreement, both reported and explained |

A pass at L1 or L2 never implies anything about L3. Theoretical bounds (L4) are never presented as measurements, and measurements are never presented as bounds.

---

## Part A: Testing

### A.1 Principles

1. **Tests before "done".** Each issue lists its tests up front (`07-development-plan.md`). An issue is complete only when those tests exist, have been executed and pass, or failures are documented and justified.
2. **Independent oracles.** Algorithm tests compare against independent references, never against the implementation's own assumptions.

   | Under test | Oracle |
   |---|---|
   | A\* (unit costs) | BFS shortest distance |
   | A\* (proximity costs) | Simple $O(V^2)$ Dijkstra in the test module |
   | BFS distance field | Floyd–Warshall on small grids |
   | Auction allocator | `scipy.optimize.linear_sum_assignment` (Hungarian) |
   | Sequential auction | Hungarian optimum (½-approximation check), brute force on tiny instances |
   | Conflict detection and arbitration | Naive $O(N^2)$ detector and fixed-point arbiter in the test module |
   | `LogKDTreeIndex` | `BruteForceIndex` (identical nearest and radius results) |
   | Supercover traversal / weighted segment cost | Dense sampling and numerical integration |
   | Shaping potential | Recorded-episode telescoping identity |
   | GAE, DQN targets | Hand-computed NumPy values |
   | DQN / PPO learning sanity | Known optimal values and policies in tiny MDPs |

3. **Property-based testing** (Hypothesis) on random grids, matrices, joint proposals and insert/query sequences. Shrunk failures are kept as regression tests.
4. **Determinism.** Fixed seeds everywhere. Hypothesis uses a `derandomize=True` profile in CI.
5. **Test behaviour, not implementation details.**
6. **Test what is claimed, and only that.** Every *Guaranteed* item in model §12 has a test. *Not claimed* items (strict priority, starvation freedom, end-to-end completion, learned-policy superiority) are **not** asserted by tests. They are measured by experiments.

### A.2 Test levels and layout

| Level | Location | Scope |
|---|---|---|
| Unit | `tests/unit/<package>/test_*.py` | One function or class, mirroring `src/robocoord/` |
| Property | inside unit modules (`@given`) | Invariants and oracle equality |
| Integration | `tests/integration/` | Engine + coordinator + planners + allocator + controllers; env + agents |
| Simulation scenario | `tests/integration/scenarios/` | Small hand-designed scenarios with asserted outcomes, e.g. "planner replans after a map event", "deviation triggers T3", "weighted-priority threshold respected" |
| End-to-end | `tests/e2e/` | CLI `simulate`, `train` (tiny budget), `experiment` (tiny suite), `data prepare` on fixtures, trace replay |

pytest markers: `slow`, `training`, `e2e`.
- `pytest` runs the fast suite (target < 2 min); it is the one CI runs.
- `pytest -m "slow or training or e2e"` runs the long suite.
- `pytest -m ""` runs everything.

### A.3 Cross-cutting tests (in addition to `04-algorithms.md`)

| ID | Test | Issue |
|---|---|---|
| X-1 | **Determinism**: same scenario and seed give byte-identical traces (timings excluded) for every planner × allocator × controller combination, with shield on and off | #2, extended #3–#6 |
| X-2 | **Engine invariants** after every step in all integration tests: no physical collision (G1), legal task state machine, masked controllers produce no `INV` events (G2) | #2 |
| X-3 | **Scenario validation**: invalid files raise `ScenarioError` naming the field (overlapping starts, out-of-bounds cells, unavailable tasks, unknown algorithm names, infeasible density/robot-count combinations) | #2 |
| X-4 | **Combination matrix**: all 16 combinations {A\*, RRT\*} × {auction, sequential} × {reference, baseline, MADQL, PPO} run a 50-step episode without invariant violations (untrained networks allowed: this tests plumbing, not quality), plus each learned controller with the shield on | #6 |
| X-5 | **Dynamic-change handling**: a map event on the path triggers T2 and a detour; a map event on the goal gives `unreachable` and the release/hold policy (model §5.3); a new urgent task triggers a re-auction and pre-emption before pickup *when the weighted benefit favours it*; an obstacle crossing the path triggers T2 | #3, #4 |
| X-6 | **Metric correctness** on hand-computed episodes: flow times (overall, per level, weighted), right-censored waiting tails, path efficiency, blocked moves, conflict counts per type, and **event-set counterfactual metrics** (avoided / introduced / persisted, matched on step, type and entity pair) | #2 (conflicts), #4 (tasks) |
| X-7 | **Checkpoint round-trip**: save → load → identical actions on fixed observations. Committed checkpoints contain inference-only weights, and the sidecar metadata is complete (ADR-9) | #5, #6 |
| X-8 | **Kaggle pipeline** on small *synthetic* fixtures in each dataset's real schema; manifest reproducibility; fidelity checks (`06` §4.7) | #7 |
| X-9 | **Viewer smoke test** headless (`SDL_VIDEODRIVER=dummy`), live and replay | #8 |
| X-10 | **Reference-follower identities**: $N^{\text{avoided}} = N^{\text{introduced}} = 0$, and its conflict set equals $\mathcal{K}^{\text{ref}}$ at every step | #4 |
| X-11 | **Planner policy consistency**: A\* and RRT\* (generous budget) return identical `status` and `degraded` on ≥ 300 random queries with obstacles on and near goals and chokepoints, and with empty sensed sets | #3 |
| X-12 | **Path monitor**: triggers T1–T6 fire exactly when specified, rejoin advances without replanning, the goal is never added to the stall block list, trigger attribution is recorded; adherence metrics match hand computation | #3, #4 |
| X-13 | **Reward potential**: continuity at pickup, telescoping identity over recorded episodes, zero potential at termination, team-mixed shaping equals shaping of $\Psi_i$ | #5 |

### A.4 Quality gates

| Gate | Command | Requirement | Where |
|---|---|---|---|
| Lint, format | `ruff check .`; `ruff format --check .` | No findings | Local + CI |
| Types | `mypy src/robocoord` | No errors (strict on `core`, `planning`, `allocation`) | Local + CI |
| Fast tests | `pytest` | All pass on Python 3.12, 3.13, 3.14 (Linux) and 3.14 (Windows) | Local + CI matrix |
| Long tests | `pytest -m "slow or training or e2e"` | All pass before an issue is closed | Local |
| Coverage | `pytest --cov=robocoord --cov-report=term-missing` | Targets: ≥ 90 % lines in `core`, `sim`, `planning`, `allocation`; ≥ 80 % overall. Targets, not goals to be gamed. | Local |

### A.5 Execution and recording procedure (per issue)

1. Before coding, list the issue's test cases (`07`).
2. Implement functionality and tests together.
3. Run the relevant tests, investigate failures, fix, rerun.
4. Run the full fast suite and the issue's long tests:
   ```bash
   pytest --junitxml=results/test-reports/issue-<n>.xml --cov=robocoord --cov-report=term
   ```
   ```bash
   pytest -m "slow or training or e2e" --junitxml=results/test-reports/issue-<n>-long.xml
   ```
5. Record a `verification-log.md` entry from the actual output: commands, counts (passed / failed / skipped / errors), duration, coverage, environment, failures and resolutions, known limitations. CI results are recorded once the student reports them.
6. Update the affected documentation (implementation notes in `04`, the results register below).

### A.6 Definition of Done (every issue)

- [ ] Scope and acceptance criteria satisfied, checked item by item.
- [ ] Planned tests exist, ran and pass; failures resolved or documented with justification.
- [ ] Lint, format and type gates pass locally.
- [ ] `verification-log.md` entry written from real output.
- [ ] Documentation updated; nothing documented that is not implemented; no guarantee stated beyond model §12.
- [ ] The full fast suite passes, so integration with earlier issues is checked.
- [ ] Commit message(s) and GitHub Desktop steps given to the student.

---

## Part B: Experimental evaluation

### B.1 Principles

- **Controlled, paired comparisons.** Algorithms are compared on identical scenario instances. One scenario seed generates the same map, obstacles and tasks regardless of algorithm (ADR-8).
- **Seed separation.** Disjoint, documented ranges:

  | Use | Seeds |
  |---|---|
  | Training | 0–9 999 |
  | Validation (model selection, early stopping) | 10 000–10 099 |
  | Test (all reported comparisons) | 20 000–20 999 |
  | Algorithmic micro-benchmarks (E1 queries, E3 matrices) | 30 000–39 999 |

  Test seeds are never used for tuning.
- **Theory first, then measurement.** Each experiment states the theoretical prediction from `04` and its type (worst, expected, conditional), then reports measurements and explains agreement or disagreement.
- **Budgets fixed in advance** (B.4). A run exceeding its time cap is recorded as `timeout`, never silently dropped.
- **No fabricated or anticipated results.** Results enter this document only from stored run records with config and raw-data paths. Hypotheses (B.5) are written as questions, and negative results are reported as such.

### B.2 Measurement methodology

| Quantity | Method |
|---|---|
| Operation counts (primary complexity evidence) | Algorithm-intrinsic counters: A\* expansions; RRT\* iterations, $k$, $k'$, nodes visited per nearest-neighbour query, propagation updates $s$, rebuild work; auction bids and phases. Hardware-independent. |
| Wall time (secondary) | `time.perf_counter_ns` around the measured call only, warm-up discarded, **median and IQR** of repeats, `torch.set_num_threads(1)`, otherwise idle machine. Hardware and software recorded per run. |
| Memory | `tracemalloc` peak for planners and allocators (NumPy allocations are traced via its allocator hooks). Optional process RSS for training. |
| Learning progress | Validation metrics every fixed number of environment steps. Curves vs environment steps *and* vs wall-clock time. |

### B.3 Statistics

- **Paired comparisons** (same test seeds $s$, configurations X and Y, metric $m$): per-seed differences $d_s = m_X(s) - m_Y(s)$.
  - Report $\bar d$ with a **paired bootstrap 95 % CI** (resample seeds; 10 000 resamples; percentile method).
  - Report the **Wilcoxon signed-rank** p-value and the **matched-pairs rank-biserial correlation** as effect size, alongside $\bar d$ in metric units.
  - Families of comparisons use **Holm–Bonferroni** correction.
- **Learned policies** have two variance sources: training seed and test scenario. Each trained policy (≥ 3 training seeds) is evaluated on the same test seeds.
  - Per-policy results are reported.
  - Pooled estimates use a **two-level bootstrap**: resample training seeds, then test seeds.
  - With 3–5 training seeds, conclusions about algorithm-level differences are stated with corresponding caution.
- **Single-configuration summaries:** mean (quality) or median (time) with bootstrap 95 % CI.
- **Scaling:** fit $\log y = a + b \log x$ over the measured range with a bootstrap CI for $b$. Compare with the exponent of the *named* theoretical bound and state which bound (worst, expected, conditional) is compared. A measured exponent below a worst-case bound does not contradict it.
- **Ratios with zero denominators** (e.g. avoidance rate when $\sum|\mathcal{K}^{\text{ref}}| = 0$) are reported as undefined and excluded from means, with the count of exclusions.

### B.4 Experiment catalogue with budgets

| ID | Question | Design and budget | Metrics | Issue |
|---|---|---|---|---|
| **E1** Planners, static | How do A\* and RRT\* compare in quality, cost and scaling? How do the RRT\* neighbour indexes compare? | **E1a**: 7 synthetic map families (random density 0/10/20/30/35 %, rooms, warehouse) × sizes {16, 32, 64, 128} × 5 maps × 20 start–goal pairs = 2 800 queries; planners A\* and RRT\* (default index) with $n_{\max}$ ∈ {500, 2 000, 5 000}; 11 200 runs; cap 30 s/run. **E1b**: RRT\* index comparison: open and 20 % density 64×64 maps, 30 queries, $n_{\max}$ ∈ {1k, 2k, 5k, 10k}, both indexes: 480 runs. **E1c**: convergence: 20 queries, $n$ up to 20 000, cost trace per iteration. Issue #7 adds maze maps to E1a (same design, 5 sizes × 10 mazes × 20 pairs). | success, lattice $\kappa$-cost and Euclidean/weighted length of both planners' outputs, optimality gap vs BFS, expansions/iterations, $k$, $k'$, NN-visited nodes, $s$, time, peak memory, empirical exponents | #3 (#7) |
| **E2** Replanning | How costly and effective is reactive replanning? | Planner {A\*, RRT\*} × obstacles $K$ ∈ {0, 4, 8, 16} × map-event rate {0, low} × team {1 robot, 4 robots (reference follower)} × 20 seeds = 640 episodes | replans by trigger, status mix incl. `degraded`, planning latency, success, executed vs reference length | #3 |
| **E3** Allocation | Is the auction optimal and how does it scale? What do weighted priority, aging and hysteresis change? | **E3a** (algorithmic): $k$ ∈ {5, 10, 20, 50, 100, 200, 400} × $C$ ∈ {10², 10⁴} × 20 matrices × {auction ε-scaling, auction fixed ε, sequential, Hungarian} = 1 120 solves. **E3b** (in simulation, paired): scarce-robot suite (N = 4, 3 priority levels, high arrival rate) × variants {default, κ = 0, α = 0, h = 0} × 30 seeds, A\* + baseline = 120 episodes. | E3a: optimality ratio, cardinality, bids, time, exponent. E3b: $F_w$, $\bar F^{(\pi)}$ per level, mean flow, travel per task, waiting-time tails per level (max, p95, censored count), reassignments | #4 |
| **E4** Learning | How do MADQL and PPO (MAPPO / IPPO) compare in learning? What do the reward weights change? | MADQL × 3 seeds, MAPPO × 3 seeds (target 5 each), IPPO × ≥ 1 seed (target 3). Reward ablation $\beta$ ∈ {0, 0.3, 1} and $c_{\text{rr}}$ ∈ {0.5, 2} on MAPPO × 2 seeds. Same environment-step budget per run, fixed in Issue #5 from measured throughput. Each final policy evaluated on 100 test seeds. | curves (return, completion, conflicts per 100 robot-steps) vs steps and wall time; steps-to-threshold; seed variability; memory | #5, #6 |
| **E5** Full system | Which planner × controller × allocator combination performs best in static vs dynamic settings? Do learned controllers avoid or introduce conflicts relative to the reference? | Planner {A\*, RRT\*} × controller {reference, baseline, MADQL, PPO} × allocator {auction, sequential} × dynamics {static, moving obstacles, full (+ map events + arrivals)} × 50 test seeds. Each of ≥ 3 trained policies evaluated separately: 1 200 non-learned + 3 600 learned = 4 800 episodes. Shield ablation: {off, predictive} × learned controllers × full dynamics × A\* × auction × 50 seeds × 3 policies = 600. | all model §9 metrics: completion, flow times, conflicts per type, avoided / introduced / persisted, adherence, deviation rate, replans by trigger, shield interventions, compute per step | #6 (prelim.), #8 (final) |
| **E6** Scalability | How do runtime and quality scale with team size, task load, map size and density? Do policies generalise to larger teams? | **One-factor-at-a-time around a centre point** (N = 8, tasks = 2N, 32×32, density 15 %, moving obstacles): N ∈ {2, 4, 8, 16, 32}; tasks/N ∈ {1, 2, 4}; size ∈ {16, 32, 64}; density ∈ {0, 15, 30} % → 11 distinct cells. Plus a **2-factor grid** N ∈ {4, 16, 32} × size ∈ {16, 32, 64} (9 cells; overlaps reused). Controllers {baseline, MADQL, PPO}, using for each learned algorithm the training seed with **median validation score** (fixed before testing). Planner A\* everywhere; RRT\* additionally on the N-sweep. 20 test seeds per cell. ≈ 1 500 episodes; cap 10 min/episode. Infeasible cells (too few free cells for N robots) are rejected by validation and listed. | step-time breakdown (planning / allocation / inference / arbitration), quality metrics vs each factor, empirical exponents | #8 |
| **E7** Data-driven | Do conclusions hold on Kaggle-derived scenarios, out of distribution for training? | Scenario family {synthetic, Olist-derived, NYC-derived, maze maps} × {reference, baseline, best MADQL, best PPO} (best = median-validation policy) × A\* × auction × 50 test seeds = 800 episodes | as E5, plus generator fidelity checks | #7, #8 |

Total planned simulation budget is about 10 000 episodes plus about 13 000 planner queries and 1 120 assignment solves. This is feasible on one CPU over several days of background runs. Exact wall-clock budgets are set once throughput is measured in Issues #3–#5.

### B.5 Hypotheses (questions to test, not claims)

| ID | Hypothesis | Test |
|---|---|---|
| H1 | A\* expansions grow super-linearly with obstacle density. RRT\* success within budget drops sharply in mazes and narrow passages. Under the k-d index, RRT\* per-iteration query work grows sub-linearly in $n$, but at online budgets ($n \le 5 000$) the brute-force index may be faster. | E1 |
| H2 | Measured auction work grows well below the worst-case bound, and the auction is competitive with the Hungarian algorithm for the problem sizes that occur online. (Optimality itself is guaranteed and tested, not hypothesised.) | E3a |
| H3 | Weighted priority (κ > 0) lowers $F_w$ and the top-level flow time relative to κ = 0, at a measurable cost in travel per task and mean flow time. Aging (α > 0) reduces the waiting-time tail of the lowest level relative to α = 0, without bounding it. | E3b |
| H4 | Learned controllers propose fewer conflicts and blocked moves than the reference follower in dense dynamic scenarios. Their comparison with the rule-based baseline is open. | E5 |
| H5 | PPO is more stable across training seeds. MADQL is more sample-efficient per environment step but uses more memory. | E4 |
| H6 | The predictive shield reduces robot–obstacle conflicts of learned controllers at the cost of a higher wait ratio. | E5 shield ablation |

### B.6 Results register

Filled only from executed experiments. Status: *not run*, *running*, *complete* or *superseded*.

| ID | Status | Config | Raw data | Summary |
|---|---|---|---|---|
| E1 | not run | — | — | — |
| E2 | not run | — | — | — |
| E3 | not run | — | — | — |
| E4 | not run | — | — | — |
| E5 | not run | — | — | — |
| E6 | not run | — | — | — |
| E7 | not run | — | — | — |

Detailed result sections (tables, figures, interpretation, threats to validity) are appended below this register as experiments complete.
