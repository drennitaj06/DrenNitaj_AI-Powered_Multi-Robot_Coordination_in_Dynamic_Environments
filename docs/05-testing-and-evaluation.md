# 05 — Testing Strategy and Experimental Evaluation

| Field | Value |
|---|---|
| Document role | **Source of truth for how correctness is verified (tests) and how performance is measured (experiments)**, plus the results register. Per-algorithm test lists live in `04-algorithms.md` and are binding. This document defines the framework, cross-cutting tests, experiment protocol, budgets and statistics. |
| Results of test runs | `verification-log.md` |
| Status | Revision 2 (Issue #1). Results register updated with E1 and E2 (Issue #3). |

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
| E1 | complete (Issue #3, 2026-11-18) | `configs/experiments/e1a_planners_static.toml`, `e1b_neighbor_index.toml`, `e1b_memory.toml`, `e1c_convergence.toml` | `results/experiments/e1a_planners_static/`, `e1b_neighbor_index/`, `e1b_memory/`, `e1c_convergence/`, plus the counter-corrected re-run `e1b_neighbor_index_r2/` and the loop-removal re-run `e1a_planners_static_r3/` (local, git-ignored) | A\* solved 2 800/2 800 queries (median ≤ 3.2 ms). RRT\* success falls with density, size and corridor structure. RRT\*'s continuous reference polylines are shorter in Euclidean length on open and low-density maps, but its executed lattice paths are never shorter than A\*'s. Default index `brute`. U fails; R consistent (k-d NN nodes examined grow like log² n). Sections below. |
| E2 | complete (Issue #3, 2026-11-18) | `configs/experiments/e2_replanning.toml` | `results/experiments/e2_replanning/` and the loop-removal re-run `e2_replanning_r2/` (local, git-ignored) | 640 episodes, 0 timeouts. Completion shows no detectable difference between planners. RRT\* costs ≈ 140 ms per call vs ≈ 0.2 ms for A\*. Incomplete 4-robot episodes are dominated by robot–robot deadlocks of the plan follower. Section below. |
| E3 | not run | — | — | — |
| E4 | not run | — | — | — |
| E5 | not run | — | — | — |
| E6 | not run | — | — | — |
| E7 | not run | — | — | — |

Detailed result sections (tables, figures, interpretation, threats to validity) are appended below this register as experiments complete.

### E1b — RRT\* neighbour index (Issue #3, complete; counters corrected in review)

- **Config:** `configs/experiments/e1b_neighbor_index.toml` (timed, 480 runs, 1 worker) and `configs/experiments/e1b_memory.toml` (peak memory under `tracemalloc`, 80 runs on a subset of the same queries, 8 workers).
- **Raw data** (local, git-ignored, each with `env.json` and `summary.md`):
  - `results/experiments/e1b_neighbor_index_r2/records.jsonl`: the re-run with corrected counters. **The numbers below come from this run.**
  - `results/experiments/e1b_neighbor_index/records.jsonl`: the original run, kept unchanged.
  - `results/experiments/e1b_memory/records.jsonl`.
  - Summaries: `python -m robocoord.experiments.analysis <dir>`.
- **Design:** 64×64 maps, families open and 20 % density, 3 maps × 10 start–goal pairs, $n_{\max} \in \{1\,000, 2\,000, 5\,000, 10\,000\}$, both indexes, same seed per query. Seed base 31000 (micro-benchmark range). Default RRT\* parameters ($\eta = 2$, $p_g = 0.1$, $\gamma_f = 1.5$), static queries ($B_i = Q_i = \varnothing$). Per-iteration counters were recorded and averaged in $\log_2$ bins of the tree size $n$.
- **Environment:** Windows 11, CPython 3.14.3, NumPy 2.4.6, Intel 64 Family 6 Model 186 (12 logical CPUs). Timed runs were sequential so that wall time is not distorted by CPU contention. The processor mixes core types, so the operating system's core choice remains a source of timing noise; medians and IQRs are reported.
- **Outcome (both runs):** 440 `ok`, 40 `budget_exhausted` (all on 20 % maps at small budgets); 0 timeouts, 0 errors. **Both indexes built identical trees on 240/240 query × budget pairs** (same $\sum k$, same path), as the oracle tests predict. So only cost differs, not results.

**Review correction (2026-11-18).** The original run under-measured k-d tree work:
- `nn_visited` counted only the *leaf points* whose distance was evaluated, not the internal or pruned nodes a nearest query examines.
- `range_nodes` counted only nodes whose box intersects the query square, not the nodes examined and then pruned.
- `rebuild_points` (points placed into newly built static trees, inserted point included) was reported *per iteration* and labelled as re-insertion work. The O(m log m) build bound is about the points processed by the builds, and its natural unit is per insertion.

The fix keeps the existing counters and their meaning, documents them precisely, and adds `nn_nodes`, `range_nodes_tested` and `rebuild_work` (Σ subtree sizes of all built nodes). Regression tests pin all definitions. E1b was then re-run into a new directory.
- In the re-run, every counter that exists in both runs is **identical** (9 120 values over 480 runs), as are all statuses and path lengths. The fix changed measurement only, not behaviour.
- Wall times differ between the runs (machine noise, plus the extra increments now in the k-d traversal); the decision below holds in both runs.
- **Conclusions that changed:**
  1. k-d nearest-neighbour work grows with exponent **0.32–0.35** in $n$ when every examined node is counted, not 0.10–0.13. Evaluated points alone still grow with 0.10–0.13. Over $n \in [64, 10\,000]$ the local exponent of $\log^2 n$ is ≈ 0.32, so the node count matches the expected-case bound under R closely. The verdict "consistent with R, far below the $O(n)$ worst case" stands, now on the correct measure.
  2. The rebuild row is now per insertion. Its d20 exponent 0.30 was an artefact of dividing by iterations (fewer insertions per iteration on cluttered maps); per insertion it is 0.08.

| family | $n_{\max}$ | success | time ms, brute (median [IQR]) | time ms, k-d (median [IQR]) | NN points/query brute / k-d | NN nodes/query k-d | $k$/query | $k'$/query | range nodes k-d, expanded / examined | $s$/iteration | k-d rebuild per insertion: points / work |
|---|---|---|---|---|---|---|---|---|---|---|---|
| open | 1 000 | 30/30 | 62.3 [45.8, 82.1] | 110.2 [80.9, 138.2] | 500.5 / 30.3 | 27.1 | 6.1 | 6.8 | 15.7 / 27.5 | 1.55 | 5.06 / 15.8 |
| open | 2 000 | 30/30 | 161.8 [125.0, 190.8] | 228.2 [196.4, 306.2] | 1 000.5 / 33.1 | 34.7 | 12.6 | 13.5 | 23.1 / 38.9 | 4.78 | 5.56 / 19.9 |
| open | 5 000 | 30/30 | 657.9 [621.6, 805.9] | 953.5 [866.3, 1 060.3] | 2 500.5 / 35.7 | 44.1 | 32.1 | 34.1 | 36.4 / 58.1 | 13.22 | 6.67 / 30.9 |
| open | 10 000 | 30/30 | 2 545.2 [2 376.2, 2 624.7] | 3 152.9 [3 000.5, 3 356.8] | 5 000.5 / 38.6 | 53.6 | 64.2 | 68.2 | 53.8 / 81.6 | 17.77 | 7.17 / 36.5 |
| d20 | 1 000 | 18/30 | 24.4 [17.1, 32.7] | 52.3 [38.0, 68.0] | 172.8 / 23.6 | 16.1 | 6.5 | 7.1 | 9.4 / 15.9 | 0.26 | 4.68 / 13.0 |
| d20 | 2 000 | 24/30 | 69.2 [43.3, 95.1] | 114.8 [69.7, 158.1] | 402.8 / 28.0 | 23.5 | 13.3 | 14.0 | 16.6 / 27.3 | 1.15 | 5.44 / 18.9 |
| d20 | 5 000 | 29/30 | 396.6 [335.2, 479.2] | 585.4 [441.9, 668.9] | 1 306.9 / 33.1 | 35.4 | 33.5 | 34.9 | 31.3 / 49.1 | 5.71 | 6.20 / 25.8 |
| d20 | 10 000 | 29/30 | 1 739.6 [1 566.4, 1 884.0] | 2 225.3 [1 978.6, 2 429.4] | 3 101.6 / 37.0 | 46.6 | 69.2 | 72.1 | 50.4 / 75.0 | 12.17 | 6.67 / 30.7 |

**Empirical exponents** $b$ in $y \approx a\,n^b$ over the tree size $n$ (per-iteration counters, $n \ge 64$, the $n_{\max} = 10\,000$ runs of the re-run; 95 % bootstrap CI). For reference, over this range the local exponent of $\log n$ is ≈ 0.16 and that of $\log^2 n$ ≈ 0.32.

| quantity | open, k-d | d20, k-d | named bound (`04` §3.6) | agreement |
|---|---|---|---|---|
| NN **nodes examined** per query (incl. internal and pruned) | 0.32 [0.30, 0.33] | 0.35 [0.34, 0.36] | $O(n)$ worst (exponent 1); expected $O(\log^2 n)$ under R | Far below the worst case. Matches the growth of $\log^2 n$ over this range, so it is consistent with **assumption R** (an observation, not a proof). |
| NN points evaluated per query | 0.10 [0.09, 0.11] | 0.13 [0.12, 0.14] | (same) | Fewer than nodes: pruning removes most leaves. Brute force evaluates all $n$ points (exponent 1.00). |
| range-query nodes examined / expanded | 0.51 [0.50, 0.52] / 0.57 [0.56, 0.58] | 0.76 [0.73, 0.78] / 0.81 [0.78, 0.84] | $O(\sqrt n + k')$ worst | Within the bound. $k'$ itself grows linearly (next rows), so the $k'$ term dominates on d20. Examined ≤ 1 + 2·expanded per slot, so both counts have the same order. |
| near-set size $k$ | 1.04 [1.02, 1.06] | 1.18 [1.10, 1.26] | U assumes $k = O(\log n)$ | **U does not hold** at these budgets. $r_n$ is capped at $\eta$ until $n \approx 47\,400$ on 64×64 (`04` §3 notes), so $k$ grows linearly with node density, as predicted for the early regime. |
| points in bounding square $k'$ | 0.99 [0.97, 1.00] | 1.13 [1.06, 1.20] | U assumes $k' = O(\log n)$ | Same: linear, U fails. $k'/k \approx 1.06$, i.e. disk and square differ by little. |
| propagation updates $s$ per iteration | 1.27 [1.19, 1.35] | 1.44 [1.35, 1.53] | $s \le k n$ worst | Within the worst case ($k n$ grows with exponent ≈ 2). Super-linear growth: rewiring near the root moves large subtrees. |
| rebuild **work** per insertion (k-d) | 0.17 [0.14, 0.20] | 0.19 [0.17, 0.21] | amortised $O(\log^2 n)$ per insertion | Within the bound: between the growth of $\log n$ and $\log^2 n$; 13–37 points processed per insertion. |
| rebuild points per insertion (k-d) | 0.07 [0.06, 0.08] | 0.08 [0.07, 0.08] | $O(\log n)$ amortised points moved per insertion | Consistent: 4.7–7.2 points per insertion (cumulative mean; ≈ $\tfrac12\log_2 n + 1$ at $n = 10^4$). |

**Direct polylogarithmic fits (added in V-6).** Power-law exponents only give a local slope, so the k-d nearest-neighbour counters were also fitted directly with $y \approx a(\ln n)^c$. Both models have two parameters and are least-squares fits on $\log y$ over the same binned points ($n \ge 64$, $n_{\max} = 10\,000$, re-run data), so their residual sums of squares (RSS) are comparable.

| family | counter | power law $b$ [95 % CI] | RSS | polylog $c$ [95 % CI] | RSS |
|---|---|---|---|---|---|
| open | NN nodes examined | 0.32 [0.30, 0.33] | 3.589 | **2.14** [2.09, 2.20] | **1.701** |
| d20 | NN nodes examined | 0.35 [0.34, 0.36] | 1.463 | **2.29** [2.25, 2.32] | **0.421** |
| open | NN points evaluated | 0.10 [0.09, 0.11] | 1.307 | 0.68 [0.64, 0.72] | 0.977 |
| d20 | NN points evaluated | 0.13 [0.12, 0.14] | 1.167 | 0.87 [0.82, 0.93] | 0.895 |

The polylog model fits better in every case. For nodes examined its exponent is close to the 2 of the expected-case $O(\log^2 n)$ under R. This **describes** the data over $n \in [64, 10^4]$ (about 2.2 decades); it does **not** prove an asymptotic bound. Polylogarithmic and small-power growth are hard to tell apart on so short a range, and the RRT\* point sets are not the i.i.d. points R assumes. The CIs come from bootstrapping the binned points. A bootstrap that resamples whole runs gave intervals no wider (review probe, V-6).

**Total time** vs budget $N_{\text{it}}$ (exponent of time $\approx a N_{\text{it}}^b$, re-run): brute 1.66 [1.57, 1.74] (open), 1.83 [1.73, 1.92] (d20); k-d 1.51 [1.43, 1.59] (open), 1.65 [1.56, 1.73] (d20). The original run gave the same picture (brute 1.65 / 1.83, k-d 1.50 / 1.72). The unconditional worst case $O(N_{\text{it}}^2\eta + N_{\text{it}}^3)$ holds with room to spare. The conditional bounds do not apply, because U fails. The k-d tree's lower exponent shows its better asymptotics. In absolute terms it is still slower at every measured budget: the ratio of medians is 1.8–2.1× at 1 000 iterations and 1.2–1.3× at 10 000 in the re-run (1.6–1.7× and 1.2–1.3× in the original run).

**Peak memory** (`e1b_memory`, `tracemalloc`): brute 0.12–3.33 MiB, k-d 0.17–4.54 MiB at $n_{\max} = 1\,000$–$10\,000$; about 300–350 vs 420–490 bytes per tree node. Exponent vs tree size: brute 1.04 [1.03, 1.05], k-d 1.03 [1.01, 1.04]. This agrees with the $O(n)$ space bound. The memory run predates the counter correction; the added integer counters do not change memory measurably.

**Decision (recorded in `04` §3 notes; unchanged by the review):** the default neighbour index is **`brute`**. It was faster at every measured budget up to 10 000 iterations in both runs, which covers the online budgets (500–5 000), and it uses less memory. `kdtree` stays fully implemented and selectable (`neighbor_index = "kdtree"`). Extrapolating the measured exponents, it would overtake brute force only well beyond 10 000 iterations; that is a projection, not a measurement. This answers the index part of **H1**: at online budgets the vectorised brute-force index is faster despite worse asymptotics. The k-d tree evaluates 7–130× fewer point distances per nearest-neighbour query and examines only 16–54 tree nodes, but its per-node Python cost exceeds that of one vectorised NumPy scan of ≤ 10 000 points.

### E1a — A\* vs RRT\* on static queries (Issue #3, complete)

- **Config:** `configs/experiments/e1a_planners_static.toml`. **Raw data:** `results/experiments/e1a_planners_static/records.jsonl` (original run) and `results/experiments/e1a_planners_static_r3/records.jsonl` (re-run after the loop-removal fix of V-6, current code), each with `env.json` and `summary.md`. 11 200 runs (2 800 queries × {A\*, RRT\* with $n_{\max}$ = 500 / 2 000 / 5 000}), seed base 30000, cap 30 s/run, 10 worker processes. Wall time is secondary evidence here; counters are primary (§B.2).
- **Instances:** 7 families (random density 0/10/20/30/35 %, rooms with 5×5 rooms, warehouse) × sizes 16/32/64/128 × 5 maps × 20 start–goal pairs. Maps are the largest connected component, so **every query is solvable**. RRT\* uses the default `brute` index (E1b), $\eta = 2$, static queries ($\omega \equiv 1$, Euclidean cost).
- **Outcome:** 8 665 `ok`, 2 535 `budget_exhausted` (all RRT\*), **0 timeouts**, 0 errors.
- **Re-run after loop removal (V-6), compared run by run:**
  - Identical in both runs: every outcome, every A\* path, every RRT\* polyline length, and therefore the success table, all A\* figures and all length/LB values.
  - Changed: the executed RRT\* cell paths. 18.0–24.7 % of successful RRT\* runs (by budget) had a loop that is now cut, saving 2.9–3.3 steps on average (at most 14). No path became longer.
  - The steps/BFS column below and the execution statements use the re-run (current code). Wall times are from the original run; the re-run's medians differ by machine noise only.

**Success within budget** (out of 100 queries per cell; A\* is 100/100 in every cell):

| family | RRT\* 500: 16 / 32 / 64 / 128 | RRT\* 2 000 | RRT\* 5 000 |
|---|---|---|---|
| open | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| d10 | 99 / 98 / 87 / 60 | 100 / 100 / 100 / 89 | 100 / 100 / 100 / 99 |
| d20 | 98 / 70 / 47 / 9 | 100 / 100 / 83 / 29 | 100 / 100 / 98 / 69 |
| d30 | 84 / 28 / 9 / 2 | 100 / 76 / 26 / 4 | 100 / 97 / 54 / 9 |
| d35 | 76 / 20 / 8 / 2 | 98 / 51 / 16 / 2 | 100 / 71 / 27 / 3 |
| rooms | 96 / 51 / 8 / 6 | 100 / 91 / 24 / 12 | 100 / 100 / 78 / 20 |
| warehouse | 100 / 70 / 36 / 9 | 100 / 100 / 74 / 32 | 100 / 100 / 96 / 64 |

**Quality and cost (selected cells; full table in `summary.md`).** Two different path metrics are reported. They must not be confused:
- **"steps/BFS" (execution metric):** the length of the path the robot actually executes, as a 4-connected cell path in grid steps, over the optimal step count. For RRT\* this is the *discretised* path. It is the metric that determines travel time in the simulator.
- **"length/LB" (reference-geometry metric):** the Euclidean length of the planner's *reference curve* over the straight-line lower bound (centre of start to goal square). For RRT\* the reference curve is the continuous polyline; for A\* it is the lattice path, whose Euclidean length equals its step count. Robots cannot follow a continuous polyline: they move one cell per step on the 4-connected lattice. A shorter polyline is therefore **not** an execution-efficiency advantage.

| cell | A\* time ms (median) | RRT\* 5 000 time ms (median) | RRT\* steps/BFS (mean, max) | length/LB: A\* vs RRT\* 5 000 |
|---|---|---|---|---|
| open 16 | 0.2 | 4 540 | 1.00, 1.00 | 1.45 vs **1.01** |
| open 64 | 0.5 | 1 823 | 1.04, 1.29 | 1.32 vs **1.05** |
| open 128 | 1.1 | 1 505 | 1.18, 1.70 | 1.30 vs **1.19** |
| d20 32 | 0.4 | 2 011 | 1.01, 1.19 | 1.51 vs **1.10** |
| d20 128 | 1.7 | 367 | 1.65, 6.03 | 1.34 vs 1.80 |
| d35 32 | 0.4 | 697 | 1.01, 1.24 | 2.50 vs **1.66** |
| rooms 64 | 0.8 | 426 | 1.40, 3.69 | 1.44 vs 1.64 |
| warehouse 64 | 0.4 | 770 | 1.32, 2.76 | 1.32 vs 1.48 |

**Scaling of A\*** (exponent $b$ of expansions $\approx a V^b$ over free cells $V$, all sizes pooled, 95 % CI): open 0.50 [0.46, 0.54], d10 0.61 [0.56, 0.67], d20 0.70 [0.64, 0.76], d30 0.76 [0.69, 0.83], d35 0.84 [0.77, 0.91], rooms 0.78 [0.71, 0.85], warehouse 0.49 [0.44, 0.53].
- The named bound is expansions ≤ $V$ (exponent 1) and time $O(V\log V)$ worst case. Every family stays below exponent 1, consistent with the bound; a smaller exponent does not contradict a worst-case bound.
- Expansions per free cell grow with density: on 128×128 maps 0.005 (open), 0.013 (10 %), 0.035 (20 %), 0.068 (30 %), 0.111 (35 %). Density grows 3.5× from 10 % to 35 % while expansions per cell grow 8.5×.
- Time exponents (0.16–0.60) are lower than the expansion exponents, because fixed per-call overhead dominates at sub-millisecond times.

**Scaling of RRT\*.** Time per run *decreases* with map area at a fixed budget (exponents −0.1 to −0.75 vs $V$). With $r_n = \eta$ the near set holds $\approx n\pi\eta^2/\mu$ nodes, so larger free areas mean fewer candidates per iteration. In addition, many iterations on large cluttered maps add no node (rejected steer segments; nodes per iteration 0.02–0.07 on d30/d35/rooms at 128 vs ≈ 1.0 on open maps). The budget-to-time relationship is measured in E1b.

**Answer to H1 (E1a part).**
- A\* expansions grow super-linearly with obstacle density (above). RRT\* success within budget drops sharply with density, map size and corridor structure (rooms, warehouse, ≥ 30 % density): at 128×128 and 5 000 iterations it is 3 % (d35), 9 % (d30) and 20 % (rooms). The hypothesis is **supported** on these instances.
- **Where each planner is better** (measured, these instances, these budgets):
  - A\* is always step-optimal (steps/BFS = 1.00 by G4), always succeeds, and is 2–4 orders of magnitude faster (100× to more than 10 000× in median time).
  - **Execution (steps on the lattice):** RRT\*'s executed, discretised path is never shorter than A\*'s. A\* is step-optimal, so steps/BFS ≥ 1.00 for every planner. With enough iterations relative to the free area (open maps; low-to-moderate density at 16–32) RRT\*'s discretised paths come close (mean steps/BFS 1.00–1.05). On large or cluttered maps they are clearly longer (mean 1.18–1.65 in the cells above; 1.19–1.70 before loop removal), and RRT\* often fails (G8: not claimed).
  - **Reference geometry (continuous length):** RRT\*'s polyline is shorter in Euclidean length than A\*'s lattice path on open maps at every size and at low-to-moderate density at 16–32. This describes the shape of the reference curve, e.g. for a future any-angle or continuous-motion setting. In this simulator it gives **no** travel-time advantage, because execution is on the 4-connected lattice.
  - **Conclusion for this simulator:** on every measured instance A\* is at least as good in executed path length, more reliable, and 2–4 orders of magnitude cheaper. RRT\*'s measured advantage is confined to the continuous reference-curve metric. Whether that matters is a question for later experiments with controllers (E5), not something E1a shows.

### E1c — RRT\* convergence (Issue #3, complete)

- **Config:** `configs/experiments/e1c_convergence.toml`. **Raw data:** `results/experiments/e1c_convergence/records.jsonl`. 20 queries (32×32; open and 20 % density; 2 maps × 5 pairs), $n_{\max} = 20\,000$, best cost recorded every iteration. Seed base 32000. Outcome: 20/20 `ok`, 0 timeouts.
- **Open maps (exact Euclidean optimum known; Euclidean cost, $Q_i = \varnothing$):** mean best cost / optimum = 1.32 (100 iterations), 1.22 (500), 1.076 (1 000), 1.024 (2 000), 1.0052 (5 000), 1.0021 (10 000), **1.0015 (20 000)**; worst query 1.0045 at 20 000. All 10 queries were solved by iteration 100 (first solution: median 58 iterations over all 20 queries).
- **20 % maps** (true continuous optimum unknown, so ratios are to the straight-line lower bound, which obstacles make unreachable): 10/10 solved by 5 000 iterations; mean ratio 1.12 at 5 000 and 1.093 at 20 000. Relative to each run's final cost: 1.026 at 5 000, 1.006 at 10 000. Improvement continues but slows.
- **Radius:** $r_n$ stays at $\eta = 2$ until $n \approx 10\,150$ and is 1.48 (open) and 1.31 (20 %) at $n = 20\,000$. Only the last part of the runs is in the shrinking-radius regime that assumption U describes.
- **Interpretation (revised in V-6):** on open maps the best cost approaches the exact optimum closely within 20 000 iterations. This is **finite-budget empirical convergence** and the measured results above are unchanged. It does **not** establish asymptotic optimality: the implemented Karaman–Frazzoli radius (exponent $1/d$) does not satisfy the corrected condition of Solovey et al. (2020), which needs exponent $1/(d+1)$, so G7 is not claimed (`04` §3.4–3.5). The best cost is non-increasing in every trace (G5, also asserted by tests).

### E2 — Reactive replanning (Issue #3, complete)

- **Config:** `configs/experiments/e2_replanning.toml`. **Raw data:** `results/experiments/e2_replanning/records.jsonl` (+ `summary.md`). 640 episodes: planner {A\*, RRT\* with $n_{\max} = 1\,000$} × dynamic obstacles $K \in \{0, 4, 8, 16\}$ (alternating random walk / two-point patrol) × map events {none, low = 6 block events in $[5, 150)$, each lasting 20 steps} × team {1, 4} × test seeds 20000–20019. 24×24 maps with 15 % static density, 3 fixture tasks per robot, 300-step horizon, plan follower (`SuggestionFollower`), default monitor and constraint parameters. 10 worker processes. Instances are paired: map, robots and tasks depend only on (seed, team), obstacles and events on separate streams. Outcome: 640 `done`, **0 timeouts**, 0 errors.

| planner | team | mean completion | incomplete episodes | of which with robot–robot conflicts | calls (total) | status mix of calls | median call latency | planning time per step (mean) |
|---|---|---|---|---|---|---|---|---|
| A\* | 1 | 0.979 | 4 / 160 | 0 | 2 241 | 2 240 ok, 1 degraded | 0.20 ms | 0.02 ms |
| RRT\* | 1 | 0.958 | 8 / 160 | 0 | 1 854 | 1 841 ok, 13 budget_exhausted | 141 ms | 12.9 ms |
| A\* | 4 | 0.761 | 106 / 160 | 98 | 8 310 | 8 112 ok, 187 degraded, 11 unreachable | 0.19 ms | 0.05 ms |
| RRT\* | 4 | 0.780 | 100 / 160 | 92 | 7 596 | 7 389 ok, 28 degraded, 8 unreachable, 171 budget_exhausted | 138 ms | 28.3 ms |

**Re-run after loop removal (V-6)** into `results/experiments/e2_replanning_r2/` (current code; the table above is from the original run):
- A\* episodes are identical: same planning statistics and completion.
- RRT\*, 1 robot: completion unchanged in all 160 episodes; 1 848 calls instead of 1 854; conflicts per 100 robot-steps mean 4.39 instead of 4.45, median 0.
- RRT\*, 4 robots: 13 of 160 episodes end with a different completion; mean completion 0.783 instead of 0.780; 7 614 calls instead of 7 596; conflicts mean 20.18, median 19.29.
- Paired completion A\* − RRT\*: team 1 unchanged at +0.021 [−0.010, 0.054]; team 4 −0.021 [−0.069, 0.027] instead of −0.019. **No conclusion changes.**

**Paired A\* − RRT\*** (same instances; paired bootstrap 95 % CI, 10 000 resamples):
- Completion: team 1 +0.021 [−0.010, 0.054] (rank-biserial 0.39); team 4 −0.019 [−0.067, 0.029] (−0.08). Both CIs contain 0: **no detectable difference in completion**.
- Planning time per simulation step: −12.9 ms [−13.8, −12.0] (team 1) and −28.2 ms [−30.5, −26.0] (team 4). Per call A\* is about 700× cheaper (median 0.2 ms vs 140 ms).

**Replanning behaviour** (per-cell means in `summary.md`):
- Replans grow with $K$: single robot, A\*, no events: 6.0 calls/episode at $K = 0$ (only T1) and 20.8 at $K = 16$, of which 14.8 are T2.
- T3 never fired, as expected for a plan follower.
- T4 is frequent for 4 robots without obstacles (A\*: 12.9 per episode). Robots stall behind parked or idle robots.
- T5/T6 retries are rare (≤ 4.6 per episode).
- Map events added few calls at this "low" rate: blocked cells were mostly off the current paths.
- Executed leg length over the first reference path: 1.00 at $K = 0$ and 1.2–1.85 at $K = 16$. Detours and waiting lengthen the legs as obstacle density rises.

**Proposed conflicts per 100 robot-steps are heavy-tailed; the mean alone is not representative** (V-6 correction).

| planner | team | mean | median | share of episodes with any conflict | persistent-conflict episodes (≥ 50 per 100) |
|---|---|---|---|---|---|
| A\* | 1 | 0.91 | 0.00 | 0.19 | 2 / 160 |
| RRT\* | 1 | 4.45 | 0.00 | 0.29 | 8 / 160 |
| A\* | 4 | 21.17 | 19.17 | 0.96 | 19 / 160 |
| RRT\* | 4 | 20.38 | 19.46 | 0.95 | 18 / 160 |

- With one robot the median is 0 for both planners. The higher RRT\* mean comes from a few deadlocked episodes.
- **The two outliers of the cell RRT\*, 1 robot, $K = 4$:** seeds **20006** (82.0 conflicts per 100 robot-steps, 246 ROv, completion 1/3) and **20017** (92.3, 277 ROv, completion 0). Both are robot–obstacle deadlocks at an occupied goal (incompleteness cause 2 below), inspected step by step. In 20006 a patrol obstacle stays on the drop-off (19, 11) because its next cell (18, 11) holds the waiting robot. They appear twice because the "none" and "low" event variants of the same instance deadlock identically, before any map event matters. Without them the cell median is 0.0 and the mean 0.13–0.16; with them the mean is 8.84–8.86.
- `summary.md` now shows mean / median / share for every cell and lists every persistent-conflict episode with its dominant conflict kind. With 4 robots these are mostly robot–robot vertex conflicts (RRv).

**Why episodes stay incomplete (examined, not assumed):**
1. *4 robots:* 98/106 (A\*) and 92/100 (RRT\*) incomplete episodes contain robot–robot conflicts. The plan follower has no right-of-way rule, so two robots contending for the same free cell freeze each other indefinitely; T4 needs an *occupied* next cell (model §5.4 clarification). This is a controller limitation by design. The baseline right-of-way rule and the learned controllers (Issues #4–#6) are meant to address it, and Issue #4 will show whether they do.
2. *1 robot, robot–obstacle deadlock (seen with both planners):* a patrol obstacle waits for the cell the robot occupies while the robot waits for the obstacle to leave its goal cell. The occupied-goal policy (model §5.3) is to wait, and T4 never targets the goal. Inspected example: RRT\*, $K = 4$, seed 20017, 277 ROv conflicts at one cell pair.
3. *1 robot, replanning livelock (A\*):* A\*, $K = 4$, seed 20009: a two-cell patrol obstacle alternately blocks two routes, and A\*'s deterministic T2 replans flip the robot between them (147 T2 calls, no conflicts, no progress). RRT\* completed the same instance: its randomised paths broke the symmetry.

**Answer to E2's question.** Reactive replanning works as specified: triggers fire and are attributed, detours are taken, and blocked goals lead to hold or release with retries (also asserted by X-5/X-12 tests). Its cost is negligible for A\* and about 140 ms per call for RRT\* at 1 000 iterations. Completion is limited mainly by the plan follower's lack of coordination, not by the planner, and A\* vs RRT\* makes no statistically detectable difference to completion here. These are plan-follower results; how controllers change them is measured in E5.

**Threats to validity (E1, E2):**
- Synthetic map families and a single machine.
- Wall times come from a CPU with mixed core types; E1a and E2 ran 10 processes in parallel. Counters are hardware-independent; times are indicative.
- RRT\* parameters ($\eta = 2$, $p_g = 0.1$) were fixed by a small pilot on separate seeds; other settings shift the success/time trade-off.
- E2 uses fixture assignments and a plan follower, not the final controllers.
- The "low" map-event rate rarely hit active paths.
