# 05 — Testing Strategy and Experimental Evaluation

| Field | Value |
|---|---|
| Document role | **Source of truth for how correctness is verified (tests) and how performance is measured (experiments)**, plus the results register. Per-algorithm test lists live in `04-algorithms.md` and are binding. This document defines the framework, cross-cutting tests, experiment protocol, budgets and statistics. |
| Results of test runs | `verification-log.md` |
| Status | Revision 2 (Issue #1). Results register updated with E1 and E2 (Issue #3), E3, the Issue #4 non-learned system comparison (stored as `e5_*`) and NFR-PERF-1 (Issue #4), and with E4 (MADQL part) and the Issue #5 development comparison on validation block B (Issue #5). |

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
| X-4 | **Combination matrix**: all 16 combinations {A\*, RRT\*} × {auction, sequential} × {reference, baseline, MADQL, PPO} run a 50-step episode without invariant violations (untrained networks allowed: this tests plumbing, not quality), plus each learned controller with the shield on. *Implemented in Issue #6: `tests/integration/test_x4_matrix.py`, 16 combinations plus 8 shielded variants, all pass* | #6 |
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
  | Validation block A (periodic validation, checkpoint selection) | 10 000–10 019 |
  | Validation block B (development comparison in Issues #5/#6, after configuration freeze; never used for selection or tuning) | 10 020–10 099 |
  | Test (all reported comparisons; reserved for the official E5/E6 evaluation of Issue #8, including 20 200–20 249) | 20 000–20 999 |
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
| Conflict metrics (Issue #6, D10) | Proposed conflicts are counted on the proposal reaching the arbiter, before cancellation (model §9), by type. Reported as **conflicts per episode** (`conflicts_total`), **per delivered task** (`conflicts_per_delivered`; undefined when nothing is delivered, excluded with a count) and **per 100 robot-steps**. Denominators differ by source: evaluation records divide by all robot-steps, the training log by non-idle robot-steps. Rates per robot-step are confounded by episode length, so the absolute counts are reported alongside them. |

### B.3 Statistics

- **Paired comparisons** (same test seeds $s$, configurations X and Y, metric $m$): per-seed differences $d_s = m_X(s) - m_Y(s)$.
  - Report $\bar d$ with a **paired bootstrap 95 % CI** (resample seeds; 10 000 resamples; percentile method).
  - Report the **Wilcoxon signed-rank** p-value and the **matched-pairs rank-biserial correlation** as effect size, alongside $\bar d$ in metric units.
  - Families of comparisons use **Holm–Bonferroni** correction.
- **Learned policies** have two variance sources: training seed and test scenario. Each trained policy (≥ 3 training seeds) is evaluated on the same test seeds.
  - Per-policy results are reported.
  - Pooled estimates use a **two-level bootstrap**: resample training seeds, then test seeds.
  - With 3–5 training seeds, conclusions about algorithm-level differences are stated with corresponding caution.
- **Algorithm-level comparisons (Issue #6):** algorithm X − algorithm Y on shared instances with a two-level bootstrap that resamples each algorithm's training seeds independently and the instances jointly; a non-learned controller counts as an algorithm with one policy. No p-values are reported at this level: with three training seeds per algorithm they would not be meaningful.
- **Single-configuration summaries:** mean (quality) or median (time) with bootstrap 95 % CI.
- **Scaling:** fit $\log y = a + b \log x$ over the measured range with a bootstrap CI for $b$. Compare with the exponent of the *named* theoretical bound and state which bound (worst, expected, conditional) is compared. A measured exponent below a worst-case bound does not contradict it.
- **Ratios with zero denominators** (e.g. avoidance rate when $\sum|\mathcal{K}^{\text{ref}}| = 0$) are reported as undefined and excluded from means, with the count of exclusions.

### B.4 Experiment catalogue with budgets

| ID | Question | Design and budget | Metrics | Issue |
|---|---|---|---|---|
| **E1** Planners, static | How do A\* and RRT\* compare in quality, cost and scaling? How do the RRT\* neighbour indexes compare? | **E1a**: 7 synthetic map families (random density 0/10/20/30/35 %, rooms, warehouse) × sizes {16, 32, 64, 128} × 5 maps × 20 start–goal pairs = 2 800 queries; planners A\* and RRT\* (default index) with $n_{\max}$ ∈ {500, 2 000, 5 000}; 11 200 runs; cap 30 s/run. **E1b**: RRT\* index comparison: open and 20 % density 64×64 maps, 30 queries, $n_{\max}$ ∈ {1k, 2k, 5k, 10k}, both indexes: 480 runs. **E1c**: convergence: 20 queries, $n$ up to 20 000, cost trace per iteration. Issue #7 adds maze maps to E1a (same design, 5 sizes × 10 mazes × 20 pairs). | success, lattice $\kappa$-cost and Euclidean/weighted length of both planners' outputs, optimality gap vs BFS, expansions/iterations, $k$, $k'$, NN-visited nodes, $s$, time, peak memory, empirical exponents | #3 (#7) |
| **E2** Replanning | How costly and effective is reactive replanning? | Planner {A\*, RRT\*} × obstacles $K$ ∈ {0, 4, 8, 16} × map-event rate {0, low} × team {1 robot, 4 robots (reference follower)} × 20 seeds = 640 episodes | replans by trigger, status mix incl. `degraded`, planning latency, success, executed vs reference length | #3 |
| **E3** Allocation | Is the auction optimal and how does it scale? What do weighted priority, aging and hysteresis change? | **E3a** (algorithmic): $k$ ∈ {5, 10, 20, 50, 100, 200, 400} × $C$ ∈ {10², 10⁴} × 20 matrices × {auction ε-scaling, auction fixed ε, sequential, Hungarian} = 1 120 solves. **E3b** (in simulation, paired): scarce-robot suite (N = 4, 3 priority levels, high arrival rate) × variants {default, κ = 0, α = 0, h = 0} × 30 seeds, A\* + baseline = 120 episodes. | E3a: optimality ratio, cardinality, bids, time, exponent. E3b: $F_w$, $\bar F^{(\pi)}$ per level, mean flow, travel per task, waiting-time tails per level (max, p95, censored count), reassignments | #4 |
| **E4** Learning | How do MADQL and PPO (MAPPO / IPPO) compare in learning? What do the reward weights change? | MADQL × 3 seeds, MAPPO × 3 seeds (target 5 each), IPPO × ≥ 1 seed (target 3). Reward ablation $\beta$ ∈ {0, 0.3, 1} and $c_{\text{rr}}$ ∈ {0.5, 2} on MAPPO × 2 seeds (*Issue #6 ran Option C instead, approved D3: ζ = 0, β = 0, $c_{\text{rr}}$ = 0.5 and $c_{\text{rr}}$ = 2, one change at a time, MAPPO × 2 seeds; see the E4 reward-ablation section*). Same environment-step budget per run, fixed in Issue #5 from measured throughput. In Issues #5 and #6 each final policy is evaluated on validation block B (80 seeds); the official test-seed evaluation of the final policies belongs to E5/E6 in Issue #8. | curves (return, completion, conflicts per 100 robot-steps) vs steps and wall time; steps-to-threshold; seed variability; memory | #5, #6 |
| **E5** Full system | Which planner × controller × allocator combination performs best in static vs dynamic settings? Do learned controllers avoid or introduce conflicts relative to the reference? | Planner {A\*, RRT\*} × controller {reference, baseline, MADQL, PPO} × allocator {auction, sequential} × dynamics {static, moving obstacles, full (+ map events + arrivals)} × 50 test seeds. Each of ≥ 3 trained policies evaluated separately: 1 200 non-learned + 3 600 learned = 4 800 episodes. Shield ablation: {off, predictive} × learned controllers × full dynamics × A\* × auction × 50 seeds × 3 policies = 600. *Terminology: the non-learned runs of Project Issue #4 (GitHub issue #8) are the Issue #4 system comparison, not E5 (see the Issue #4 system comparison section below). Issues #5 and #6 report development comparisons on validation block B. E5 is run only in Issue #8.* | all model §9 metrics: completion, flow times, conflicts per type, avoided / introduced / persisted, adherence, deviation rate, replans by trigger, shield interventions, compute per step | #8 — plan numbering |
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

*Issue numbers here and in B.4 follow `07` (the project plan). Project Issue #4 is tracked on GitHub as **GitHub issue #8**. The plan's own Issue #8, the final evaluation (where E5 and E6 are run), is a different issue.*

| ID | Status | Config | Raw data | Summary |
|---|---|---|---|---|
| E1 | complete (Issue #3, 2026-11-18) | `configs/experiments/e1a_planners_static.toml`, `e1b_neighbor_index.toml`, `e1b_memory.toml`, `e1c_convergence.toml` | `results/experiments/e1a_planners_static/`, `e1b_neighbor_index/`, `e1b_memory/`, `e1c_convergence/`, plus the counter-corrected re-run `e1b_neighbor_index_r2/` and the loop-removal re-run `e1a_planners_static_r3/` (local, git-ignored) | A\* solved 2 800/2 800 queries (median ≤ 3.2 ms). RRT\* success falls with density, size and corridor structure. RRT\*'s continuous reference polylines are shorter in Euclidean length on open and low-density maps, but its executed lattice paths are never shorter than A\*'s. Default index `brute`. U fails; R consistent (k-d NN nodes examined grow like log² n). Sections below. |
| E2 | complete (Issue #3, 2026-11-18) | `configs/experiments/e2_replanning.toml` | `results/experiments/e2_replanning/` and the loop-removal re-run `e2_replanning_r2/` (local, git-ignored) | 640 episodes, 0 timeouts. Completion shows no detectable difference between planners. RRT\* costs ≈ 140 ms per call vs ≈ 0.2 ms for A\*. Incomplete 4-robot episodes are dominated by robot–robot deadlocks of the plan follower. Section below. |
| E3 | complete (Issue #4, 2026-11-18) | `configs/experiments/e3a_assignment.toml`, `e3b_priority.toml` | `results/experiments/e3a_assignment/`, `e3b_priority/` (local, git-ignored) | E3a: 1 120 solves, 0 timeouts. The ε-scaling auction is optimal in 280/280; every fixed-ε (ε = 1) gap ≤ k; the sequential auction satisfies 2V ≥ OPT in 280/280 but is optimal in only 10/280. Measured exponents are far below the worst-case bounds. The exact Python auction is 7–11× slower than SciPy's compiled Hungarian at k = 400, and more at small k. E3b: 120 paired episodes. κ > 0 lowers $F_w$ (−3.4 [−4.9, −2.0]) and top-level flow time (−10.3 [−14.2, −7.0]), at a mean-flow price (+2.0 [0.6, 3.7], Holm p = 0.059). α > 0 lowers the level-1 maximum wait (−20.3 [−34.0, −7.7]); h > 0 cuts re-assignments (−4.0). Sections below. |
| E4 | complete: MADQL part (Issue #5, 2026-11-19); MAPPO, IPPO and the Option C reward ablation (Issue #6, 2026-11-20) | `configs/training/madql.toml`, `mappo.toml`, `ippo.toml`, `mappo_{zeta0,beta0,crr05,crr2}.toml` (run seeds via `--seed`) | `results/training/{madql,mappo,ippo}_seed{1,2,3}/`, `mappo_{zeta0,beta0,crr05,crr2}_seed{1,2}/` (local, git-ignored) | 3 runs × 1 000 000 environment steps (7 176–7 581 s each, run concurrently). Checkpoints selected on block A: seed 1 at 800 k (completion 0.715, base return 1.590), seed 2 at 1 M (0.635, 1.566), seed 3 at 700 k (0.631, 1.692). Validation fluctuates by up to 0.25 between consecutive checks. The static level is the weakest. 0 physical collisions. Section below.<br>**PPO (Issue #6):** 6 runs × 1 M steps (MAPPO 6 517–6 520 s, IPPO 4 354–4 419 s). Block-A completion at selection 0.992–0.997 (MAPPO selected at 150–300 k, IPPO at 250–550 k); ≥ 0.95 within 50–100 k steps; consecutive validations differ by ≤ 0.07. H5: stability supported, MADQL's sample-efficiency advantage not supported. **Option C ablation:** no completion effect; return and conflict differences confounded with the selected step. Sections below. |
| Issue #5 development comparison (validation block B; not E5) | complete, run once (Issue #5, 2026-11-19) | `python -m robocoord.marl.devcompare` with the three selected checkpoints | `results/dev5/comparison/` (local, git-ignored) | 1 500 episodes (A\* 240 and RRT\* 60 per controller). **MADQL completion is lower** than the baseline (pooled −0.41 with A\*, −0.31 with RRT\*) and than the reference follower (−0.32, −0.21); every per-policy Holm p ≤ 4 × 10⁻⁷. MADQL proposes 13–16 fewer conflicts per 100 robot-steps than the reference follower. Against the baseline there is no clear difference in the per-step conflict rate, which is confounded by episode length. 0 physical collisions. Section below. |
| Issue #6 development comparison (validation block B; not E5) | complete, each experiment run once (Issue #6, 2026-11-20) | `python -m robocoord.marl.study {matched-baseline,ppo,ablation,shield,stochastic,analyse}` with the selected checkpoints | `results/dev6/` (local, git-ignored); historical `results/dev5/comparison/` read only | 15 360 new episodes plus the Issue #5 records. **PPO completion** 0.982–0.992 (A\*) and 0.983–0.995 (RRT\*): +0.40 / +0.30 over MADQL and +0.08 over the reference follower; level with the matched baseline (resolver off) on A\* and −0.007 to −0.010 on RRT\*; −0.012 against the original baseline (resolver on, A\*). PPO proposes more conflicts than both baselines, mostly from MAPPO's two earliest selected checkpoints (robot–obstacle), which the predictive shield removes. Stochastic execution raises completion for every policy (MADQL to 0.89–0.92). 0 physical collisions. Section below. |
| Issue #4 system comparison (non-learned; not E5) | Project Issue #4 = GitHub issue #8, 2026-11-18; MADQL/PPO and the shield ablation are not part of it | `configs/experiments/e5_prelim.toml` (baseline), `e5_prelim_reference.toml` (reference follower) | `results/experiments/e5_prelim/`, `e5_prelim_reference/`, `e5_prelim_comparison.md` (local, git-ignored) | 1 200 paired episodes without endpoint recovery (controller × planner × allocator × dynamics × 50 test seeds), plus 1 200 further episodes with endpoint recovery (`e5_recovery*.toml`): **2 400 runs in total**.<br>**Baseline:** pooled completion 0.9921 (91 of 11 488 tasks undelivered; 83 of 600 episodes incomplete), 0 RR conflicts, 0 physical collisions. 89 of the 91 undelivered tasks are permanent blockages: an obstacle or an idle robot on the goal.<br>**Reference follower:** pooled 0.8819 (404 of 600 episodes incomplete), 15–25 proposed conflicts per 100 robot-steps, dominated by contests on free cells.<br>**With endpoint recovery** (`e5_recovery*.toml`, same seeds, paired): baseline 0.9993 (8 of 600 incomplete; 6 tasks on never-leaving patrol obstacles, reported as exhausted), reference 0.8881 (384 incomplete). Section below. |
| NFR-PERF-1 (**A\*** requirement) | measured; before optimisation **not met** as a per-step bound; after O1/O2 + symmetric reuse **met for workload R** on AC power on this machine, in the original benchmark and in an independent Windows verification. RRT\* is outside the requirement and still 23–34 % of its workload-F steps exceed 20 ms | `python -m robocoord.experiments.perf --out <new dir> --workload R --impl legacy,fast` (paired); `--legacy` for the original workload | `results/perf/`, `results/perf_r2/` (before); `results/perf_r3_R/` (intermediate, battery); `results/perf_r4/` (original implementation benchmark); `results/perf_verify/` (independent Windows verification). All local, git-ignored | Before: 21–29 of ≈ 2 800 steps per run ≥ 20 ms (first and map-event steps). Original benchmark, workload R: max 9.53 ms (legacy 33.7–35.6 ms; 40 exceedances → 0). Independent verification, workload R: max 10.56 ms, 0 exceedances (legacy 40). Full workload F, A\*: max 15.43 ms, 0 exceedances. Battery run (intermediate): 3 of 40 first steps ≥ 20 ms. Section below. |
| E5 | not run (plan Issue #8) | — | — | — |
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

### E3a — Assignment algorithms (Issue #4, complete)

- **Config:** `configs/experiments/e3a_assignment.toml`. **Raw data:** `results/experiments/e3a_assignment/records.jsonl` (+ `summary.md`).
- **Design.** 1 120 solves: $k \in \{5, 10, 20, 50, 100, 200, 400\}$ × $C \in \{10^2, 10^4\}$ × 20 matrices × {auction ε-scaling, auction fixed ε = 1, sequential, Hungarian}.
  - Proposed parameters (Issue #4 review): dense matrices with entries uniform in $\{1, \dots, C\}$, seeded from the micro-benchmark range (30 400 –) by $(k, C, \text{index})$ only, so every method solves the identical matrix.
  - Wall time is the median of 5 repeats; one worker process; a 60 s cap per solve.
  - Outcome: 1 120 `done`, **0 timeouts**.
- **Theory first.** ε-scaling: exact optimum (G10). Fixed ε: $V \ge \mathrm{OPT} - k\varepsilon$ and at most $k(\lfloor C/\varepsilon\rfloor + 1) + 1$ bids. Sequential: $2V \ge \mathrm{OPT}$. Worst-case time: auction $O(k^3 \log kC)$ dense (cited), sequential $O(k^2 \log k)$, SciPy Hungarian $O(k^3)$.

| method | C | optimal | max gap (max gap / k) | min V/OPT | 2V ≥ OPT | fixed-ε bid bound |
|---|---|---|---|---|---|---|
| auction, ε-scaling | 10² / 10⁴ | 140/140 / 140/140 | 0 | 1 | yes | — |
| auction, fixed ε = 1 | 10² | 56/140 | 47 (0.30) | 0.99661 | yes | holds in all 140 |
| auction, fixed ε = 1 | 10⁴ | 121/140 | 15 (0.04) | 0.999996 | yes | holds in all 140 |
| sequential | 10² | 8/140 | 489 (19.2) | 0.758 | yes | — |
| sequential | 10⁴ | 2/140 | 45 743 (1 556) | 0.809 | yes | — |

- Every fixed-ε gap is ≤ k: the additive guarantee holds. **No relative-error guarantee is claimed.** The measured ratios are reported only as measurements.
- All solutions have maximum cardinality, but with dense matrices this is trivial. The cardinality property with infeasible (zero) pairs is covered by the unit tests (`04` §4.1–4.2).

**Scaling** (median over 20 matrices; log–log exponent with bootstrap 95 % CI for $k \ge 50$; full tables in `summary.md`):

| method | C | time k = 20 → 400 (ms) | time exponent | bids k = 20 → 400 | bids exponent |
|---|---|---|---|---|---|
| auction, ε-scaling | 10² | 0.67 → 52.2 | 1.56 [1.51, 1.62] | 170 → 12 700 | 1.53 [1.50, 1.56] |
| auction, ε-scaling | 10⁴ | 1.07 → 38.1 | 0.91 [0.79, 1.02] | 254 → 9 942 | 1.23 [1.21, 1.25] |
| auction, fixed ε = 1 | 10² | 0.19 → 3.42 | 0.88 [0.79, 0.98] | 42 → 752 | 0.81 [0.76, 0.87] |
| auction, fixed ε = 1 | 10⁴ | 0.31 → 16.9 | 0.92 [0.76, 1.09] | 74 → 4 230 | 1.22 [1.12, 1.33] |
| sequential | 10² / 10⁴ | 0.12 → 43.3 / 50.7 | 2.02 / 1.97 | k rounds | 1.00 |
| Hungarian (SciPy) | 10² / 10⁴ | 0.01 → 4.65 / 5.14 | 2.17 / 2.06 | — | — |

- Every measured exponent is far below its worst-case bound (auction time $\le 3 + o(1)$, bids $\le 2 + o(1)$; Hungarian ≤ 3). That is consistent with the bounds, but **not evidence of a better asymptotic bound**: these are fitted exponents over $k \le 400$ on one random-matrix family.
- The sequential exponent ≈ 2 matches its $O(k^2 \log k)$ sort.
- **H2:** "auction work grows well below the worst-case bound" is supported for this family. "Competitive with the Hungarian algorithm" is **not supported in wall time** for the exact (production) ε-scaling auction: the pure-Python implementation is 7–11× slower than SciPy's compiled one at $k = 400$, and ≈ 30–70× slower at $k \le 20$, where SciPy needs ≈ 10 µs. (The inexact fixed-ε auction is comparable at $k = 400$, $C = 10^2$: 3.4 vs 4.7 ms.) At the online problem sizes ($k$ ≈ 4–20) an auction costs 0.1–1 ms, which is small in absolute terms (see NFR-PERF-1).

### E3b — Priority sensitivity (Issue #4, complete; acceptance criterion 3)

- **Config:** `configs/experiments/e3b_priority.toml`. **Raw data:** `results/experiments/e3b_priority/records.jsonl` (+ `summary.md`).
- **Design.** 120 episodes: scarce-robot suite × variants {default (κ = 10, α = 0.01, h = 2), κ = 0, α = 0, h = 0} × test seeds 20100–20129, A\* + auction + baseline (resolver on).
  - Proposed parameters: 24×24 map, 15 % obstacles; N = 4; Bernoulli arrivals until t = 200 with levels 1/2/3 at 0.5/0.3/0.2; cap 400 steps.
  - **λ = 0.16 was chosen on validation seeds 10000–10004** (utilisation 0.78–0.90, all tasks delivered), before any test-seed run.
  - Paired: the map, robots and arrival stream depend only on the seed.
  - Outcome: 120 `done`; 997–998 of 999 released tasks delivered in every variant, 0 censored waits.
- **Statistics (B.3):** d = default − variant per seed; paired bootstrap 95 % CI (10 000 resamples); two-sided Wilcoxon signed-rank; Holm over the family of 8; matched-pairs rank-biserial r.

| comparison | metric | mean d [95 % CI] | Wilcoxon p | Holm p | r | reading |
|---|---|---|---|---|---|---|
| κ > 0 vs κ = 0 | $F_w$ | −3.41 [−4.89, −2.04] | < 0.0001 | 0.0002 | −0.80 | default lower (better), clear |
| κ > 0 vs κ = 0 | flow time, level 3 | −10.32 [−14.18, −6.97] | < 0.0001 | < 0.0001 | −0.99 | default lower (better), clear |
| κ > 0 vs κ = 0 | mean flow time (price) | +2.03 [0.55, 3.65] | 0.015 | 0.059 | 0.51 | higher by CI; not significant after Holm |
| κ > 0 vs κ = 0 | travel per delivered task (price) | +0.29 [−0.08, 0.66] | 0.21 | 0.35 | 0.27 | no clear difference |
| α > 0 vs α = 0 | max wait, level 1 | −20.33 [−34.00, −7.70] | 0.0075 | 0.038 | −0.62 | default lower (better), clear |
| α > 0 vs α = 0 | p95 wait, level 1 | −11.84 [−21.13, −3.26] | 0.11 | 0.32 | −0.36 | lower by CI only; weak |
| h > 0 vs h = 0 | re-assignments | −4.00 [−5.43, −2.70] | < 0.0001 | 0.0002 | −0.95 | default lower (better), clear |
| h > 0 vs h = 0 | travel per delivered task | +0.14 [−0.06, 0.35] | 0.17 | 0.35 | 0.30 | no clear difference |

- **Acceptance criterion 3:** κ > 0 improves both priority-sensitive metrics, with clear paired differences. The "stop and review the benefit design" condition is therefore **not** triggered. The price is a higher mean flow time (+2.0 steps), significant by CI but not after Holm, and no detectable travel cost.
- **H3** is supported on this suite for κ, and partly for α: the maximum lowest-level wait drops, but the p95 effect is weak. Aging is still **not** a starvation-freedom guarantee (model G13); see the constructed backlog counter-example in `04` §4.4.
- Threats: one synthetic suite, one arrival rate and N = 4. The default variant's p95 effect is small relative to seed variance. The episodes drain completely (no censoring), so overload behaviour is not covered.

### Issue #4 system comparison (non-learned; Project Issue #4 = GitHub issue #8; stored as `e5_*`; NOT E5)

- **Scope.**
  - GitHub issue #8 (Project Issue #4) asks for "preliminary non-learned E5 numbers". Under the approved experiment mapping, E5 and E6 belong exclusively to the plan's Issue #8 (final evaluation; not GitHub issue #8). The runs reported here are therefore called the *Issue #4 system comparison*. The historical term "preliminary E5" remains only in GitHub issue #8, in the verification log (V-8 to V-11) and in the stored `e5_*` file and directory names.
  - The non-learned system comparison run here covers **both non-learned controllers**: planner {A\*, RRT\* ($n_{\max}$ = 1 000)} × allocator {auction, sequential} × dynamics {static, moving obstacles, full} × controller {baseline, reference follower} × test seeds 20200–20249 = **1 200 paired episodes without endpoint recovery** (600 baseline + 600 reference follower). The same 1 200 instances were run again with endpoint recovery (`e5_recovery*.toml`, below), which adds **1 200 further episodes**. Total runs in this document: **2 400** (1 200 without recovery plus 1 200 with recovery).
  - The baseline half was run first (approved while the reference follower was out of scope). The reference half was added after the audit with the identical methodology.
  - **Not included:** MADQL/PPO and the shield ablation (plan Issues #5/#6), and E5/E6 (plan Issue #8). **This is not E5.**
- **Config and data:**
  - Baseline half: `configs/experiments/e5_prelim.toml` → `results/experiments/e5_prelim/`.
  - Reference half: `configs/experiments/e5_prelim_reference.toml` → `results/experiments/e5_prelim_reference/`.
  - Each directory has `records.jsonl`, `summary.md`, and `undelivered.json` / `.md` (cause diagnostics). The paired comparison is `results/experiments/e5_prelim_comparison.md`.
  - Proposed parameters: 32×32, 15 % obstacles, N = 8, 16 scheduled tasks; "obstacles" adds 8 moving obstacles (random walk / patrol); "full" adds 6 map events of 20 steps and Bernoulli arrivals (0.05 until t = 200); cap 400 steps; shield off.
  - Each controller uses its documented default: the **baseline with the deadlock resolver**, the **reference follower without it**. The comparison is therefore between these two *configurations*, not the right-of-way rule alone.
  - Outcome: 1 200 `done` (600 baseline + 600 reference follower; no recovery), 0 timeouts. Released counts are identical across all configurations of a (dynamics, seed), so the instances are paired.
- **Definitions.**
  - *Completion* of an episode = delivered / released (`metrics.py`); no episode had 0 released tasks.
  - *Mean of episodes* averages these rates with equal weight per episode. *Pooled* divides the total delivered by the total released over the 50 episodes of a configuration.
  - *Incomplete episodes* = episodes with at least one undelivered task.

**Per configuration** (pooled delivered/released, mean of episodes rounded half-up to 4 decimals, incomplete episodes of 50; recomputed from `records.jsonl`):

| planner | allocator | dynamics | baseline pooled | baseline mean | baseline incomplete | reference pooled | reference mean | reference incomplete |
|---|---|---|---|---|---|---|---|---|
| A\* | auction | static | 798/800 | 0.9975 | 2 | 672/800 | 0.8400 | 38 |
| A\* | auction | obstacles | 791/800 | 0.9888 | 8 | 716/800 | 0.8950 | 29 |
| A\* | auction | full | 1258/1272 | 0.9889 | 13 | 1106/1272 | 0.8709 | 39 |
| A\* | sequential | static | 797/800 | 0.9963 | 3 | 672/800 | 0.8400 | 39 |
| A\* | sequential | obstacles | 793/800 | 0.9913 | 7 | 717/800 | 0.8963 | 30 |
| A\* | sequential | full | 1262/1272 | 0.9922 | 8 | 1139/1272 | 0.8957 | 36 |
| RRT\* | auction | static | 798/800 | 0.9975 | 2 | 695/800 | 0.8688 | 37 |
| RRT\* | auction | obstacles | 791/800 | 0.9888 | 8 | 734/800 | 0.9175 | 20 |
| RRT\* | auction | full | 1265/1272 | 0.9944 | 6 | 1160/1272 | 0.9124 | 33 |
| RRT\* | sequential | static | 798/800 | 0.9975 | 2 | 660/800 | 0.8250 | 38 |
| RRT\* | sequential | obstacles | 790/800 | 0.9875 | 9 | 723/800 | 0.9038 | 28 |
| RRT\* | sequential | full | 1256/1272 | 0.9867 | 15 | 1137/1272 | 0.8949 | 37 |
| **all** | | | **11 397 / 11 488 = 0.9921** | | **83 / 600** | **10 131 / 11 488 = 0.8819** | | **404 / 600** |

- **Conflicts.**
  - Baseline: 0 RR conflicts and 0 physical collisions in all 600 episodes (G19, G21). RO conflicts occur only with moving obstacles: 59–107 per configuration, from obstacles that turn (G19 covers only constant velocity).
  - Reference follower: 19 698–35 965 proposed RR conflicts per configuration (15–25 per 100 robot-steps), every one prevented by the arbiter; 0 physical collisions (G1).
  - Paired baseline − reference: proposed conflicts per 100 robot-steps −14.7 to −24.4 and completion +0.071 to +0.172 in every configuration; all Holm-adjusted p < 0.001 over the family of 24. Full table in `e5_prelim_comparison.md`.
- **Why tasks stay undelivered** (every incomplete episode replayed deterministically and checked against its record: **0 mismatches** in 83 + 404 replays; rule order in `src/robocoord/experiments/e5_diagnostics.py`):

| cause (final state) | baseline | reference |
|---|---|---|
| robot–robot contest on a **free** next cell (follower proposes it every step; the arbiter freezes all claimants; no T4) | 0 | **1 138** |
| next cell occupied by a busy robot (head-on, rotation) | 0 | 59 |
| goal occupied by a **dynamic obstacle**: robot–obstacle goal deadlock (`04` §5.3 limitation 6) | **61** (54 patrol, 7 random walk) | 60 (49 patrol, 11 random walk) |
| goal occupied by an **idle robot** (no parking; `04` §5.3 limitation 3) | **28** | 19 |
| next cell occupied by an idle robot | 0 | 3 |
| no valid plan | 1 | 2 |
| still moving at the 400-step cap (censored by the horizon) | 1 | 65 |
| released and available, but unassigned (more open tasks than eligible robots) | 0 | 11 |
| **total undelivered** | **91** | **1 357** |

- **Reading the numbers.**
  - For the baseline, 89 of 91 undelivered tasks are *permanent* blockages: an obstacle or an idle robot on the goal. Neither the baseline nor the deadlock resolver handles these; the resolver detects only robot–robot wait-for cycles, and the baseline had 0 unresolved cycles at episode end.
  - Only 1 baseline task is censored by the horizon. The reference follower's failures are dominated by robot–robot contests that the baseline's right-of-way rule removes.
  - **What completion 0.99 means:** about 1 % of released tasks are never delivered, almost all because of a permanent blockage. It does **not** mean 99 % of episodes succeed: 83 of 600 baseline episodes (13.8 %; up to 15 of 50 in one configuration) leave at least one task undelivered.
  - Completion also says nothing about speed (flow times are averaged over delivered tasks only, so they are biased towards tasks that finished) or about conflicts. And it depends on the 400-step cap, which is generous: static baseline episodes finish in about 100 steps.
- **Paired auction − sequential, baseline half** (Holm over 12): no difference in completion or mean flow survives correction. The largest is RRT\*, obstacles, mean flow −1.38 [−2.32, −0.45], Holm p = 0.083.
- Wall times per step come from 10 parallel workers and are indicative only.

#### Issue #4 system comparison with endpoint recovery (Project Issue #4 endpoint follow-up; still NOT E5)

- **What changed.** Endpoint recovery (`04` §5.6): idle robots park off active task endpoints, and a robot retreats from a goal held for $W = 4$ steps. The parameters ($W = 4$, $H = 8$, $M = 3$, radius 8) were fixed from existing model constants before any run and were not tuned. Smoke checks during development used seeds 10000–10019, not the test seeds.
- **Configs and data.**
  - `configs/experiments/e5_recovery.toml` (baseline) and `e5_recovery_reference.toml` (reference follower): identical to `e5_prelim(.toml/_reference.toml)` in every factor, test seed (20200–20249), scenario parameter, metric and cap, plus `endpoint = "on"`.
  - Results in `results/experiments/e5_recovery/`, `e5_recovery_reference/`; comparisons in `e5_recovery_comparison.md`, `e5_recovery_reference_comparison.md`, `e5_recovery_controllers.md`.
  - The no-recovery runs above are the paired "before"; they were not re-run or changed.
  - Outcome: 1 200 `done` (600 baseline + 600 reference follower; recovery on). These are the same instances as the no-recovery runs above, so the two settings give **2 400 runs in total**. Released counts are identical across recovery off/on for every instance, so the comparison is paired.
- **Metric definitions are unchanged** (`04` §5.6 lists where recovery can move a value; mean flow is over delivered tasks only, and the deviation rate excludes parking steps).

| controller | recovery | pooled delivered/released | incomplete episodes | undelivered tasks |
|---|---|---|---|---|
| baseline | off | 11 397 / 11 488 = 0.9921 | 83 / 600 (13.8 %) | 91 |
| baseline | **on** | **11 480 / 11 488 = 0.9993** | **8 / 600 (1.3 %)** | **8** |
| reference | off | 10 131 / 11 488 = 0.8819 | 404 / 600 (67.3 %) | 1 357 |
| reference | **on** | **10 202 / 11 488 = 0.8881** | **384 / 600 (64.0 %)** | **1 286** |

- **Paired episodes** (same instance, off → on):
  - *Baseline:* 75 incomplete episodes became complete, **0 complete episodes became incomplete**, and no episode delivered fewer tasks.
  - *Reference:* 23 became complete, **3 became incomplete** (RRT\* seeds 20207 ×2 and 20223), and 9 episodes delivered fewer tasks. All undelivered tasks of those 3 are `route_contested_cell`, the plain follower's free-cell contest, not an endpoint cause. Recovery changed those trajectories so that such a contest formed.
- **Per configuration** (paired, Holm over 24): the completion difference is ≥ 0 in every baseline configuration (+0.003 to +0.013). Because only ≈ 1 % of tasks were affected, only RRT\*/sequential/full survives Holm correction (+0.013 [0.007, 0.020], Holm p = 0.015). No mean-flow difference survives (|d| ≤ 0.45 steps baseline, ≤ 1.8 reference). Tables: the comparison files.
- **Safety** (recovery on): 0 physical collisions in all 1 200 recovery episodes. The baseline has 0 RR, ROs and INV proposals. The reference follower's ROv proposals fall from 17 101 to 1 468, because a reference robot held next to an obstacle-occupied goal proposes entering it every step, and recovery ends most of those waits.

**Why tasks still stay undelivered with recovery** (all incomplete episodes replayed: 0 mismatches in 8 + 384 replays):

| cause | baseline off | baseline **on** | reference off | reference on |
|---|---|---|---|---|
| goal held by a dynamic obstacle | 61 | **6** (all patrol; all 6 `attempts_exhausted`, reported `unresolved_obstacle_at_end`) | 60 | 9 (2 exhausted; 1 random walk) |
| goal held by an idle robot | 28 | **0** | 19 | 1 (occupant still driving to its parking cell at the end) |
| contest on a free next cell | 0 | 0 | 1 138 | 1 149 |
| next cell held by a busy / idle robot | 0 / 0 | 0 / 0 | 59 / 3 | 54 / 3 |
| still moving at the 400-step cap | 1 | 1 | 65 | 57 |
| no valid plan | 1 | 1 | 2 | 2 |
| released, available, unassigned | 0 | 0 | 11 | 11 |
| **total** | **91** | **8** | **1 357** | **1 286** |

- **Recovery counters** (sums over 600 episodes; definitions in `04` §5.6):

| counter | baseline | reference |
|---|---|---|
| blockage episodes temporary / persistent | 197 / 107 | 151 / 100 |
| retreats started = succeeded + timed out + aborted + cancelled | 120 = 98 + 21 + 0 + 1 | 106 = 79 + 12 + 0 + 14 (+ 1 active at the end) |
| goal reached after a successful retreat | 91 | 73 |
| attempts exhausted / no retreat cell | 6 / 0 | 2 / 0 |
| parking needed / started / completed / cancelled / aborted | 35 / 39 / 34 / 4 / 1 | 36 / 47 / 33 / 2 / 11 (+ 1 active) |
| no parking cell / parking blocked (dynamic) | 0 / 0 | 0 / 1 |
| unresolved at the end: obstacle / idle robot; idle robots on endpoints | 6 / 0; 0 | 4 / 0; 1 |

  More parking starts than needs are restarts within one stay: after an abort, or after the resolver released the robot.
- **Reading.**
  - For the baseline, recovery removes the idle-robot cause entirely and 55 of 61 obstacle cases.
  - What remains is reported, not hidden: 6 patrol obstacles that never left the goal within 3 retreats (the obstacle's patrol path keeps returning to it), 1 task censored by the horizon, 1 without a plan.
  - For the reference follower, the endpoint causes were a small part of the failures. Its dominant cause, free-cell contests, is a controller property that recovery does not address.
  - **Recovery does not guarantee delivery (G26).**

### NFR-PERF-1 — Step time (Project Issue #4, measured; corrected after the audit)

- **Requirement** (`01`): "A single headless simulation step for 8 robots on a 32×32 grid with A\* completes in < 20 ms on a typical laptop CPU, excluding RL inference. This is a target, measured in Issue #4."
  - It names neither a statistic (every step or a typical step) nor the scenario's dynamics.
  - Project Issue #4's acceptance criterion 5 asks that it be "measured and reported, met or not".
- **What is timed:** one `Engine.step`:
  - events and arrivals; obstacle moves;
  - the coordinator: allocation, A\* planning and the deadlock resolver; three view builds;
  - the baseline controller; the predictive shield; conflict detection and arbitration; counterfactual detection; metrics; pickups and deliveries.
  - Excluded: engine construction and trace writing.
  - This is a **superset** of the requirement: the shield, the resolver, the counterfactual detection and the "full" dynamics are not required.
- **Workload and procedure** (`src/robocoord/experiments/perf.py`):
  - The `e5_scenario` "full" generator (a generator name from Issue #4): 32×32, 15 % obstacles, 8 robots, 16 scheduled tasks, 8 moving obstacles, 6 map events of 20 steps (block and unblock), arrivals.
  - Seeds 30500–30509 (micro-benchmark range).
  - One unmeasured warm-up episode, then one natural episode per seed (cap 400 steps) per allocator. `perf_counter` around each step; a single process.
  - Machine: Intel64 Family 6 Model 186 (12 logical CPUs), Windows 11, CPython 3.14.3, NumPy 2.4.6.
- **Runs:**
  - `results/perf/`: the original run; aggregates only.
  - `results/perf_r2/`: after the audit, same workload, with **raw per-step samples** in `nfr_perf1_samples.csv` and the breakdown below.
  - The original run is kept unchanged.

**Per-step distribution, run r2** (all values in ms):

| allocator | steps | n | median | p95 | p99 | max | < 20 ms |
|---|---|---|---|---|---|---|---|
| auction | all | 2 812 | 0.57 | 4.00 | 20.11 | 43.80 | 99.0 % |
| auction | first step of an episode (cold cache) | 10 | 37.59 | 42.31 | 43.50 | 43.80 | **0 %** |
| auction | steps with a map event | 116 | 11.28 | 28.99 | 32.87 | 35.34 | 83.6 % |
| auction | **steady state** (all other steps) | 2 686 | 0.55 | 2.39 | 3.62 | 6.68 | **100 %** |
| sequential | all | 2 762 | 0.59 | 4.25 | 20.53 | 47.67 | 99.0 % |
| sequential | first step | 10 | 36.76 | 44.21 | 46.98 | 47.67 | 0 % |
| sequential | map event | 116 | 10.78 | 27.56 | 32.57 | 36.65 | 83.6 % |
| sequential | steady state | 2 636 | 0.58 | 2.23 | 3.44 | 5.44 | 100 % |

- **Exceedances (≥ 20 ms):** run r2: 29 per allocator = **10 first steps + 19 map-event steps + 0 others**.
  - Original run: about 0.9 % of steps. Two audit re-measurements: 21–25 per allocator.
  - The count varies between runs (21–29) because many map-event steps lie near 20 ms. The categories never change, and in a repeat run 22 of 23 exceedances fell on the same steps, so they are workload-driven, not noise.
  - Disabling the garbage collector changed nothing.
- **Cause:** step time correlates 0.99 with the number of BFS distance fields computed in the step (≈ 1 ms each in pure Python).
  - First steps compute ≈ 35 fields: 16 tasks are released at once and the cache is cold.
  - Map-event steps compute ≈ 12 on average and up to ≈ 35: a map event changes the grid version and invalidates every cached field. The fields are then rebuilt by the availability checks, the auction benefits, the obstacle motion models and the metrics.
  - Steady-state steps compute 0.16 on average.
  - The workload's map events are listed per step in the samples file.
- **Without map events** (audit measurement, `static` and `obstacles` dynamics): only the 10 first steps exceed 20 ms. Every later step stays ≤ 2.7 ms (static) and ≤ 3.5 ms (obstacles).
- **Outcome (before optimisation; superseded by the paired measurement below).** NFR-PERF-1 is **not met as a per-step bound**: every episode's first step and 11–19 of the 116 map-event steps (9–16 %, depending on the run: 21, 23 and 29 exceedances = 10 first steps + 11, 13 and 19 map-event steps) exceed 20 ms. It is met for every steady-state step measured. The median and p99 alone do not establish the requirement.
  - At the time of this measurement no optimisation had been made (not required by acceptance criterion 5). The follow-up optimisation is reported in the next subsection.
  - A prototype that answers availability checks from connected-component labels (audit, not in the code) removed only a few exceedances, because the auction still needs the distance fields.
- **Component times** (median / p95): planning 0.04 / 0.34; controller 0.02 / 0.04; shield and resolver ≈ 0.007 / 0.02; allocation 0.01 / 1.5. The engine's remaining work (views, detection, metrics) is 0.45 / 2.3. RRT\* steps take ≈ 50–65 ms (Issue #4 system comparison), but NFR-PERF-1 is defined for A\*.
- *The runs above (r1, r2) are the pre-optimisation evidence and are kept unchanged. The optimisation and its paired measurements follow.*

#### NFR-PERF-1 after optimisation (Project Issue #4 follow-up): workloads R and F, paired before/after

- **Optimisations** (`04` §2.1; all exact, G27):
  - O1, a flat-list BFS over neighbour lists cached per grid version;
  - O2, reachability from per-version component labels instead of distance fields;
  - symmetric reuse of cached fields, with the auction benefits reading $D(p_j, d_j)$ from the pickup's field.
  - O3 (incremental repair after single-cell map events) was **not** implemented: after O1/O2 no A\* step exceeds 20 ms, including all map-event steps.
  - Nothing was moved into engine construction or out of the timed step. Every step is measured and kept.
- **Workloads** (`src/robocoord/experiments/perf.py`, `run_workloads`):
  - **R** (requirement-focused): 8 robots, 32×32 `e5_scenario` map (15 % obstacles), A\*, headless, the system defaults with an allocator. That means the baseline with the deadlock resolver and endpoint recovery, shield off; each allocator; without moving obstacles (`static`) and with 8 (`obstacles`).
  - **F** (full system): `full` dynamics (8 moving obstacles, 6 map events, Bernoulli arrivals), predictive shield, endpoint recovery, resolver as the CLI default; planner {A\*, RRT\* $n_{\max}$ = 1 000} × allocator × controller {baseline, reference}.
  - Seeds 30500–30509; one natural episode per seed, cap 400; one unmeasured warm-up per configuration and implementation.
- **Pairing.**
  - "Before" is the pre-optimisation distance code, kept selectable (`legacy_distance()`, used only by the benchmark and equivalence tests); "after" is the shipped code.
  - Both run **in one process, interleaved seed by seed** (alternating order), so both see the same machine state.
  - A digest of the state after every step shows that both followed **identical trajectories in all 12 configurations**, and the outcomes are identical.
- **Conditions of the original implementation benchmark** (`results/perf_r4/`, run by the author): this machine, **on AC power** (recorded at start and end), no other load started; `env.json` is in the folder.
- The independent Windows verification with the same command is reported separately, below (`results/perf_verify/`). The two runs are not merged or averaged.

| workload | configuration | before: max | before: steps ≥ 20 ms | after: median / p90 / p95 / p99 | **after: max** | after: steps ≥ 20 ms |
|---|---|---|---|---|---|---|
| R | static / auction (990 steps) | 33.66 | 10 (1.01 %) | 0.39 / 0.89 / 1.26 / 3.85 | **7.03** | **0** |
| R | static / sequential (1 324) | 35.32 | 10 (0.76 %) | 0.40 / 0.77 / 0.96 / 1.85 | **9.53** | **0** |
| R | obstacles / auction (1 027) | 35.59 | 10 (0.97 %) | 0.51 / 1.05 / 1.37 / 2.87 | **8.76** | **0** |
| R | obstacles / sequential (1 060) | 35.30 | 10 (0.94 %) | 0.51 / 0.89 / 1.09 / 2.37 | **7.46** | **0** |
| F | A\* / auction / baseline (2 427) | 47.17 | 25 (1.03 %) | 0.73 / 1.85 / 2.61 / 5.05 | 9.04 | 0 |
| F | A\* / auction / reference (3 328) | 39.21 | 42 (1.26 %) | 0.64 / 1.46 / 2.17 / 4.23 | 12.18 | 0 |
| F | A\* / sequential / baseline (2 400) | 69.40 | 33 (1.38 %) | 0.76 / 1.64 / 2.35 / 4.74 | 10.70 | 0 |
| F | A\* / sequential / reference (3 647) | 39.01 | 55 (1.51 %) | 0.67 / 1.19 / 1.54 / 3.78 | 15.43 | 0 |
| F | RRT\* / auction / baseline (2 528) | 478.79 | 778 (30.8 %) | 0.84 / 52.45 / 73.85 / 136.86 | 450.16 | 768 (30.4 %) |
| F | RRT\* / auction / reference (3 620) | 316.95 | 862 (23.8 %) | 0.66 / 36.38 / 58.79 / 90.96 | 296.56 | 843 (23.3 %) |
| F | RRT\* / sequential / baseline (2 280) | 414.69 | 784 (34.4 %) | 0.80 / 40.86 / 67.67 / 107.14 | 294.78 | 778 (34.1 %) |
| F | RRT\* / sequential / reference (3 428) | 533.95 | 862 (25.2 %) | 0.73 / 42.33 / 70.06 / 123.96 | 534.60 | 841 (24.5 %) |

(Original implementation benchmark, `results/perf_r4/`; all values in ms. Full per-kind tables, with first, map-event and steady steps for both implementations, are in `results/perf_r4/nfr_perf1.md`; every step is in `nfr_perf1_samples.csv`, with an `impl` column.)

- **Where the time went, and where it goes now** (mean BFS fields per step).
  - *Before:* first steps computed 31.4 (static) / 35.1 (obstacles) fields — availability checks 15.7, auction benefits 15.4, patrol obstacles 4 (profiled). Map-event steps computed 11.7–17.5.
  - *After:* first steps 16.0 / 19.8, map-event steps 6.4–8.3, each field 3–6× cheaper in the micro-benchmarks (five measurements, 3.0–5.7×). The largest remaining first-step part is allocation (≤ 6.4 ms max in A\* configurations).
  - The endpoint recovery's own cost is ≤ 3.8 ms per step at most (a cut-vertex computation on a parking trigger).
- **Outcome.**
  - **Scope.** NFR-PERF-1 is an **A\*** requirement. Workload R is the A\* requirement workload. The RRT\* configurations of workload F are outside the requirement and are reported separately.
  - **Workload R, original benchmark:** the maximum over all 4 401 measured steps per implementation is 9.53 ms (the legacy maximum over the same steps is 35.59 ms). In this run the strict per-step bound holds for R, under the conditions above (this machine, AC power). Every A\* configuration of workload F also stays below 20 ms (max 15.43 ms).
  - **RRT\* (limitation, not a failure of the A\* requirement):** RRT\* steps still exceed 20 ms in 23–34 % of workload-F steps. This document does not profile that cost and does not attribute it to the distance fields.
  - **Condition dependence (negative evidence kept):** an intermediate paired run of workload R on **battery power** (battery 14 %, Windows Balanced scheme; `results/perf_r3_R/`, with O1/O2 but without the symmetric reuse) had 3 of 40 first steps at or above 20 ms (fast max 24.83 ms; legacy max 66.01 ms). The final code was not measured on battery.

#### Independent Windows verification (`results/perf_verify/`)

The student ran the same command independently on Windows 11 (build 26300) with CPython 3.14.3, NumPy 2.4.6 and an Intel CPU with 12 logical CPUs. The machine was on AC power, with the battery at 100 % at start and end. The JSON, Markdown and per-step CSV are in `results/perf_verify/`. The figures below are read from its JSON (workload R, 4 401 measured steps per implementation):

| configuration | impl | n | median | p90 | p95 | p99 | max | ≥ 20 ms |
|---|---|---|---|---|---|---|---|---|
| static / auction | legacy | 990 | 0.41 | 0.98 | 1.44 | 8.73 | 47.13 | 10 (1.01 %) |
| static / auction | **fast** | 990 | 0.38 | 0.86 | 1.23 | 3.60 | **9.60** | **0** |
| static / sequential | legacy | 1 324 | 0.35 | 0.63 | 0.77 | 1.22 | 28.64 | 10 (0.76 %) |
| static / sequential | **fast** | 1 324 | 0.35 | 0.61 | 0.76 | 1.25 | **10.56** | **0** |
| obstacles / auction | legacy | 1 027 | 0.51 | 1.14 | 1.51 | 3.41 | 33.99 | 10 (0.97 %) |
| obstacles / auction | **fast** | 1 027 | 0.49 | 0.93 | 1.21 | 4.73 | **8.02** | **0** |
| obstacles / sequential | legacy | 1 060 | 0.48 | 0.84 | 1.10 | 2.34 | 32.73 | 10 (0.94 %) |
| obstacles / sequential | **fast** | 1 060 | 0.48 | 0.81 | 0.95 | 1.52 | **6.27** | **0** |

- **Legacy exceedances** are all first steps (10 per configuration, one per seed). The fast implementation has none.
- **Behaviour is identical:** the state digests match at every step in all four configurations. Fast deliveries are 160/160, 159/160, 160/160 and 160/160, the same as legacy, with 0 physical collisions.
- **Reading.** This run also meets the strict per-step bound for R, with a fast maximum of 10.56 ms, about 1.9× below the limit. The two runs on this machine agree on the conclusion, but not on the tail: the fast p99 in this run is up to 1.6× the original run's, and the maxima differ by about 1 ms. One run on one machine is therefore evidence, not a guarantee. The bound is established here for workload R on AC power on this Windows machine. It is not established on battery, on other hardware, or for RRT\*.

### E4 (MADQL part) — Training with A\* (Issue #5, complete; MAPPO and IPPO follow in Issue #6)

- **Config:** `configs/training/madql.toml` (1 000 000 environment steps per run; sampler seed 0). Run seeds 1, 2 and 3 are passed with `--seed`. **Raw data:** `results/training/madql_seed{1,2,3}/`. Each directory holds `config.json`, `train_log.jsonl`, `validation.jsonl`, `checkpoints/`, `best.pt` with its `best.json` sidecar, and `summary.json`; the stdout of each run is in `results/training/madql_seed{1,2,3}.log`. All local and git-ignored; no checkpoint is committed (ADR-9).
- **Command:** `python -m robocoord train --config configs/training/madql.toml --seed <s> --out results/training/madql_seed<s>`. The three runs ran concurrently on one machine.
- **Environment:** Windows 11 (build 26300), 12 logical CPUs, CPython 3.14.3, NumPy 2.4.6, torch 2.15.0+cpu. Torch used 1 thread in deterministic mode, and validation used 3 worker processes per run.
- **Design (as approved; `04` §6 and the implementation notes):**
  - Parameter-shared Double DQN, MLP 356→256→256→5 (158 469 parameters), 4 synchronous environment copies.
  - Planner A\* only during training, with the auction allocator, endpoint recovery on, resolver off and shield off.
  - Curriculum stages 1/2/3 take 20/30/50 % of the step budget.
  - Block-A validation every 50 000 steps: 60 A\* episodes (seeds 10 000–10 019 × {static, obstacles, full}).
  - RRT\* monitoring every 250 000 steps: 10 episodes (seeds 10 000–10 009, full level, $n_{\max}$ = 1 000). It is not used for selection.
  - Checkpoint selection is lexicographic on block A: overall completion first, then mean discounted base return.

**Cost per run**

| run seed | completed episodes | gradient updates | wall time | of which validation | training throughput (excluding validation) |
|---|---|---|---|---|---|
| 1 | 3 474 | 249 186 | 7 581 s | 2 114 s | 183 steps/s |
| 2 | 3 466 | 249 186 | 7 543 s | 2 170 s | 186 steps/s |
| 3 | 3 463 | 249 186 | 7 176 s | 1 995 s | 193 steps/s |

- The replay buffer holds 200 000 transitions in 160.2 MB (`ReplayBuffer.nbytes`). Peak process memory was **not measured**.
- One inference checkpoint takes 636 401 bytes. Each run directory, including every step checkpoint with its training state, takes 66.8 MB.

**Block-A A\* validation (60 episodes; completion / mean discounted base return)**

| run seed | selected step | overall | static | obstacles | full | final step (1 M), overall |
|---|---|---|---|---|---|---|
| 1 | 800 000 | **0.715 / 1.590** | 0.441 / 0.672 | 0.841 / 1.836 | 0.804 / 2.261 | 0.470 / 0.923 |
| 2 | 1 000 000 | **0.635 / 1.566** | 0.394 / 0.782 | 0.734 / 1.820 | 0.720 / 2.096 | 0.635 / 1.566 |
| 3 | 700 000 | **0.631 / 1.692** | 0.362 / 0.944 | 0.772 / 1.948 | 0.707 / 2.183 | 0.560 / 1.272 |

- **Learning curves** (overall block-A completion at 50 k / 250 k / 500 k / 750 k / 1 M steps):
  - seed 1: 0.065 / 0.238 / 0.328 / 0.571 / 0.470;
  - seed 2: 0.107 / 0.260 / 0.301 / 0.634 / 0.635;
  - seed 3: 0.083 / 0.241 / 0.143 / 0.357 / 0.560.
- **Steps to threshold:** the first validation with overall completion ≥ 0.5 was at 600 k for seeds 1 and 2 and 650 k for seed 3. Completion ≥ 0.6 was first reached at 700 k, 750 k and 700 k.
- Physical collisions were 0 in every validation episode and every training episode.
- **Instability between validations (negative evidence kept).** Consecutive validations differ by up to 0.25 in overall completion:
  - seed 1 fell from 0.715 at 800 k to 0.470 at 850 k;
  - seed 2 fell from 0.431 at 850 k to 0.248 at 900 k, then recovered to 0.543;
  - seed 3 fell from 0.631 at 700 k to 0.357 at 750 k.

  The final weights of seeds 1 and 3 are therefore worse on block A than the selected checkpoints. Selection on block A is what makes the selected policies usable, so their block-A scores are **optimistically biased** (the maximum of 20 noisy evaluations). Block B below is the unbiased development estimate.
- **Static level is the weakest** (0.36–0.44 completion against 0.71–0.84 for the dynamic levels at the selected steps). The cause has not been established. The block-A static episodes are the same 32×32, 8-robot, 16-task scenarios without obstacles.
- **RRT\* monitoring** (block A, full level, 10 episodes, completion / base return):

  | run seed | 250 k | 500 k | 750 k | 1 M |
  |---|---|---|---|---|
  | 1 | 0.298 / −0.17 | 0.415 / 0.26 | 0.732 / 1.62 | 0.526 / 1.32 |
  | 2 | 0.331 / −0.75 | 0.393 / 0.32 | 0.787 / 2.12 | 0.728 / 2.17 |
  | 3 | 0.368 / 0.60 | 0.121 / −0.36 | 0.312 / 0.40 | 0.662 / 1.70 |

  Policies trained only with A\* reference paths also run with RRT\* reference paths, with 0 physical collisions. The selected checkpoints of seeds 1 (800 k) and 3 (700 k) fall between RRT\* monitoring points; their RRT\* evaluation is the block-B comparison below.
- **Training episodes** (ε-greedy, the last 50 episodes of each stage; completion for runs 1 / 2 / 3):
  - stage 1: 0.90 / 0.94 / 0.89;
  - stage 2: 0.79 / 0.84 / 0.86;
  - stage 3: 0.92 / 0.90 / 0.91.

  These training scenarios include smaller maps and teams than the validation levels, so these figures are not comparable with block A.
- **Shaping magnitude** (logged per episode; approved design, no clipping):
  - |shaping| > 1 occurred in 1 911–2 003 robot-steps per run, about 0.04 % of about 4.68 M robot-steps.
  - In stages 1 and 2 (static and obstacle levels) the maximum was 1.55–1.98, from changes in the assigned route Λ.
  - Values up to 15.985 occur only at the full level, in 688–729 of about 1 300 stage-3 episodes. This equals the capped-distance jump when a goal becomes unreachable or reachable again after a map event ($W·H/(W+H)$ = 16 on 32×32 maps).
  - These spikes are a known property of $\bar D$ (model §7) and were not investigated further.
- **H5** (stability and sample efficiency against PPO) cannot be assessed until Issue #6. The between-seed spread of the selected policies is 0.631–0.715 in block-A completion.
- **Threats to validity:**
  - one machine, with three runs sharing it;
  - three training seeds;
  - one budget (1 M steps), set from measured throughput rather than from convergence (V-14);
  - hyper-parameters were not tuned;
  - block-A selection bias, as above;
  - the static-level gap has not been explained.

### Issue #5 development comparison (validation block B; NOT E5)

- **Command (run once, after all three trainings had finished and their checkpoints had been selected on block A):** `python -m robocoord.marl.devcompare --policy results/training/madql_seed1/best.pt --policy results/training/madql_seed2/best.pt --policy results/training/madql_seed3/best.pt --out results/dev5/comparison --workers 10`.
  - **Raw data:** `results/dev5/comparison/` (`manifest.json` with each checkpoint's SHA-256, configuration hash and block-A scores; `records.jsonl`; `analysis.json`; `summary.md`) and `results/dev5/comparison.log`. All local and git-ignored.
  - Nothing was tuned or re-selected afterwards.
- **Design (approved; `marl/devcompare.py`):**
  - Controllers: baseline (resolver on), reference follower, and the three selected MADQL policies (greedy). Every controller runs with the auction allocator, endpoint recovery on and shield off.
  - A\*: seeds 10 020–10 099 × {static, obstacles, full}, giving 240 paired episodes per controller.
  - RRT\* ($n_{\max}$ = 1 000): seeds 10 020–10 039 × the three levels, giving 60 per controller.
  - In total, 1 500 episodes in 2 109 s with 10 worker processes.
  - Seeds used are 10 020–10 099 only; no test seed was used.
- **Statistics (B.3):**
  - For each metric, policy, comparator and planner: the paired difference MADQL − comparator, with a paired bootstrap 95 % CI, the Wilcoxon signed-rank p, Holm–Bonferroni within the metric's family of 12, and the matched-pairs rank-biserial r.
  - The pooled estimate over the three policies is a two-level bootstrap (resample policies, then instances).
  - Completion is pooled per controller as delivered over released tasks.
- **Outcome:** 0 physical collisions in all 1 500 episodes.

**Pooled completion (delivered / released)**

| controller | A\* overall (240) | static | obstacles | full | RRT\* overall (60) | conflicts / 100 robot-steps (A\* / RRT\*) | base return (A\* / RRT\*) |
|---|---|---|---|---|---|---|---|
| baseline | 0.999 | 1.000 | 0.999 | 0.999 | 0.997 | 0.08 / 0.05 | 3.518 / 3.252 |
| reference follower | 0.911 | 0.866 | 0.933 | 0.925 | 0.904 | 13.51 / 16.31 | −16.561 / −20.643 |
| MADQL seed 1 (800 k) | 0.661 | 0.338 | 0.807 | 0.768 | 0.787 | 0.22 / 0.34 | 1.484 / 1.197 |
| MADQL seed 2 (1 M) | 0.584 | 0.336 | 0.727 | 0.647 | 0.609 | 0.08 / 0.07 | 1.460 / 1.366 |
| MADQL seed 3 (700 k) | 0.575 | 0.306 | 0.747 | 0.634 | 0.705 | 0.11 / 0.27 | 1.294 / 1.508 |

**Pooled over the three policies (two-level bootstrap; MADQL − comparator)**

| metric | vs | planner | mean diff | 95 % CI |
|---|---|---|---|---|
| completion | baseline | A\* | −0.408 | [−0.445, −0.364] |
| completion | baseline | RRT\* | −0.305 | [−0.391, −0.228] |
| completion | reference | A\* | −0.316 | [−0.353, −0.272] |
| completion | reference | RRT\* | −0.209 | [−0.295, −0.132] |
| base return | baseline | A\* | −2.106 | [−2.254, −1.993] |
| base return | baseline | RRT\* | −1.895 | [−2.223, −1.667] |
| base return | reference | A\* | +17.974 | [16.553, 19.447] |
| base return | reference | RRT\* | +22.000 | [18.291, 25.859] |
| conflicts / 100 | baseline | A\* | +0.056 | [−0.026, 0.165] |
| conflicts / 100 | baseline | RRT\* | +0.168 | [0.008, 0.403] |
| conflicts / 100 | reference | A\* | −13.371 | [−14.409, −12.365] |
| conflicts / 100 | reference | RRT\* | −16.087 | [−18.858, −13.416] |

Per-policy rows with Wilcoxon and Holm p-values and r are in `summary.md` / `analysis.json`. Their reading:

- **Completion: MADQL is worse than both non-learned controllers (negative result).**
  - All 12 per-policy comparisons are negative. The per-episode mean differences are −0.22 to −0.44 against the baseline and −0.13 to −0.34 against the reference. Every Holm p ≤ 4 × 10⁻⁷ and every |r| ≥ 0.80.
  - None of the three policies reaches the reference follower's completion on any planner.
- **Base return:**
  - Lower than the baseline in all 6 comparisons (≈ −1.7 to −2.2).
  - Much higher than the reference follower in all 6 (≈ +18 to +22). The reference follower's return is dominated by conflict penalties.
- **Conflicts against the reference follower:** MADQL proposes far fewer conflicts, 13–16 per 100 robot-steps fewer in all 6 comparisons (Holm p ≤ 1.4 × 10⁻¹⁰). This matches the first part of H4 on development data. It comes with the completion loss above. It is not E5 evidence.
- **Conflicts against the baseline: no clear difference, and confounded by episode length.**
  - On RRT\* every per-policy Holm p is 1.
  - On A\*, the Wilcoxon test favours MADQL (Holm p ≤ 1.4 × 10⁻⁴, r −0.39 to −0.70: 102–115 of 240 pairs lower and 32–44 higher). The mean differences, however, are close to zero (−0.004 to +0.145), and for seed 1 the difference is positive by CI.
  - The rate is per robot-step, and MADQL episodes run to the 400-step horizon, while the baseline usually finishes earlier (mean 100–226 steps per level). In **absolute** conflicts per episode, MADQL has 2.4–6.8 against the baseline's 1.0 on A\*, and 2.1–10.7 against 0.75 on RRT\*. The lower per-step rate therefore reflects longer episodes, not fewer conflicts.
- **Behaviour (descriptive):**
  - MADQL episodes almost never terminate: 0–6 of 80 per level on A\*, against 79–80 for the baseline and 27–47 for the reference follower.
  - Wait ratio: 0.10–0.19 for seed 1, 0.31–0.51 for seed 2 and 0.20–0.31 for seed 3, against ≈ 0.03–0.05 for the baseline.
  - MADQL deviates from the suggested action in 22–44 of every 100 robot-steps.
- **Static level is the weakest** (A\* completion 0.31–0.34), as on block A.
  - The cause is not established.
  - Recorded fact, not a tested explanation: the curriculum contains static scenarios only in stage 1 (2–4 robots, 16×16 maps). The obstacle level is equally absent from training at 32×32 with 8 robots, yet scores 0.73–0.81.
- **Selection bias:** block-B A\* completion (0.661 / 0.584 / 0.575) is 0.05–0.06 below the block-A scores of the same checkpoints (0.715 / 0.635 / 0.631). This is consistent with the optimistic bias of selecting the best of 20 validations. The instances differ, so this is not a paired test.
- **RRT\* against A\* for MADQL on the same 20 seeds (descriptive, not tested):** completion with RRT\* reference paths is not lower than with A\* (0.787 / 0.609 / 0.705 against 0.687 / 0.570 / 0.599).
- **Threats to validity:**
  - 80 block-B seeds (20 for RRT\*);
  - three policies from one configuration and one budget;
  - per-step conflict rates confounded by episode length, as above;
  - one machine.
- **Consequence for the plan:** MADQL at this budget and configuration does **not** outperform the rule-based baseline or the reference follower in task completion. Following ADR-3 in `03` (fallback), VDN *may be added* if MADQL learns poorly. That decision, and any change to the budget or hyper-parameters, is the student's and is **not** made here.

### E4 (PPO part) — MAPPO and IPPO training with A\* (Issue #6, complete)

- **Configs:** `configs/training/mappo.toml` and `ippo.toml`; run seeds 1, 2 and 3 via `--seed`. The two files differ only in the run name and `variant`. **Raw data:** `results/training/{mappo,ippo}_seed{1,2,3}/`, with the same layout as MADQL plus `*.train.pt` training checkpoints; stdout in `results/training/<run>.log`. All local and git-ignored; no checkpoint is committed (ADR-9).
- **Command:** `python -m robocoord train --config configs/training/<variant>.toml --seed <s> --out results/training/<variant>_seed<s>`. The six runs ran concurrently on one machine (environment as in E4, MADQL part).
- **Design (approved D1–D9; `04` §7 and its implementation notes):**
  - one PPO core; MAPPO critic on $[o_i, \phi(s)]$, IPPO critic on $o_i$; the shared actor is identical in both;
  - $E = 8$ environment copies × $T_r = 125$ steps = 1 000 environment steps per iteration; $K = 4$ epochs × 4 minibatches;
  - **shared with MADQL:** composition (A\* only, auction, endpoint recovery on, resolver off, shield off), `obs-v1`, `reward-v1`, curriculum, scenario order (sampler seed 0), the 1 000 000-environment-step budget, block-A validation every 50 000 steps, RRT\* monitoring every 250 000 steps and the lexicographic selection rule (completion first, then mean discounted base return).
- **Seed order:** training used seeds 0–9 999 only. PPO finishes far more episodes than MADQL, so the PPO trainer can restart the fixed order (V-16, deviation 1). **No run wrapped:** the runs used 0.938–0.970 of one pass.

**Cost per run**

| run | completed episodes | gradient updates | wall time | of which validation | training throughput (excluding validation) | peak RSS | rollout buffer |
|---|---|---|---|---|---|---|---|
| MAPPO 1 | 9 640 | 15 967 | 6 517 s | 837 s | 176 steps/s | 367.7 MB | 12.7 MB |
| MAPPO 2 | 9 672 | 15 969 | 6 517 s | 869 s | 177 steps/s | 368.7 MB | 12.7 MB |
| MAPPO 3 | 9 695 | 15 950 | 6 520 s | 884 s | 178 steps/s | 364.5 MB | 12.7 MB |
| IPPO 1 | 9 379 | 15 963 | 4 354 s | 951 s | 294 steps/s | 362.2 MB | 11.7 MB |
| IPPO 2 | 9 375 | 15 993 | 4 400 s | 995 s | 294 steps/s | 362.3 MB | 11.7 MB |
| IPPO 3 | 9 378 | 15 964 | 4 419 s | 998 s | 292 steps/s | 362.3 MB | 11.7 MB |

- MAPPO's lower throughput is the cost of computing $\phi(s)$ for every copy and step; it does not affect results.
- Each run reached 1 000 iterations (1 000 000 environment steps, about 4.69 M robot-steps). The approximate-KL early stop ended 2–11 of the 1 000 updates per run early.
- Peak RSS was measured for PPO only (MADQL's was not measured in Issue #5). Each run directory, with every step checkpoint and its training state, takes 91 MB (IPPO) or 106 MB (MAPPO); one inference checkpoint takes 636 401 bytes.

**Block-A A\* validation (60 episodes; completion / mean discounted base return)**

| run | selected step | overall | static | obstacles | full | final step (1 M), overall |
|---|---|---|---|---|---|---|
| MAPPO 1 | 200 000 | **0.997 / 1.612** | 0.991 / 2.454 | 1.000 / 1.294 | 0.998 / 1.090 | 0.991 / 3.62 |
| MAPPO 2 | 300 000 | **0.996 / 3.038** | 0.997 / 3.432 | 0.997 / 2.401 | 0.994 / 3.280 | 0.990 / 3.58 |
| MAPPO 3 | 150 000 | **0.992 / 2.565** | 0.991 / 3.433 | 0.988 / 2.256 | 0.996 / 2.005 | 0.985 / 3.62 |
| IPPO 1 | 550 000 | **0.993 / 3.612** | 0.991 / 3.532 | 1.000 / 3.334 | 0.991 / 3.970 | 0.976 / 3.57 |
| IPPO 2 | 250 000 | **0.992 / 3.086** | 0.988 / 3.173 | 1.000 / 2.998 | 0.991 / 3.087 | 0.982 / 3.65 |
| IPPO 3 | 500 000 | **0.993 / 3.482** | 0.988 / 3.238 | 0.997 / 3.362 | 0.994 / 3.844 | 0.986 / 3.60 |

- **Learning curves** (overall block-A completion at 50 k / 250 k / 500 k / 750 k / 1 M; base return at the same points):
  - MAPPO 1: 0.945 / 0.988 / 0.992 / 0.983 / 0.991 (return −8.73 / 2.62 / 3.61 / 3.61 / 3.62);
  - MAPPO 2: 0.937 / 0.976 / 0.988 / 0.973 / 0.990 (−8.90 / 3.32 / 3.66 / 3.58 / 3.58);
  - MAPPO 3: 0.962 / 0.991 / 0.991 / 0.985 / 0.985 (−3.83 / 3.15 / 3.68 / 3.34 / 3.62);
  - IPPO 1: 0.902 / 0.983 / 0.992 / 0.956 / 0.976 (−9.29 / 3.18 / 3.59 / 3.33 / 3.57);
  - IPPO 2: 0.921 / 0.992 / 0.985 / 0.976 / 0.982 (−10.06 / 3.09 / 3.55 / 3.53 / 3.65);
  - IPPO 3: 0.969 / 0.986 / 0.993 / 0.970 / 0.986 (−4.25 / 2.42 / 3.48 / 3.54 / 3.60).
- **Steps to threshold** (first validation with overall completion ≥ 0.95 / ≥ 0.98): MAPPO 100 k / 100 k, 100 k / 100 k, 50 k / 100 k; IPPO 100 k / 200 k, 100 k / 100 k, 50 k / 100 k. MADQL never exceeded 0.715 (E4, MADQL part).
- **Stability:** consecutive validations differ by at most 0.021–0.069 in overall completion (MADQL: up to 0.25). From 500 k on, overall completion stays within 0.956–0.993 and base return within 2.7–3.7 in every run.
- **Selection picks early checkpoints (negative evidence kept).** Completion is close to its ceiling from about 100 k steps on, so the completion-first rule chooses among checkpoints that differ by a few tasks out of 1 176 (0.005 ≈ 6 tasks). It selected 150 k–300 k for MAPPO, where base return is still rising (1.61 at 200 k for MAPPO 1 against 3.6 later). The rule was approved before training and is **not changed** here; its consequences appear on block B below.
- **RRT\* monitoring** (block A, full level, 10 episodes; completion at 250 k / 500 k / 750 k / 1 M): MAPPO 0.985–1.000, IPPO 0.978–0.996 at every point, base return 2.87–3.87. 0 physical collisions.
- **Training episodes** (stochastic policy, last 50 episodes of each stage): completion 0.994–1.000 in every stage and run.
- **Shaping magnitude:** the maximum |shaping| per robot-step was 15.92–15.99 in every run, the same capped-distance jump after map events as for MADQL.
- **0 physical collisions** in all training, validation and RRT\* monitoring episodes.
- **H5** ("PPO is more stable across training seeds; MADQL is more sample-efficient per environment step but uses more memory"), at this budget and configuration:
  - *stability:* supported. Selected block-A completion spans 0.992–0.997 across the six PPO runs against 0.631–0.715 for MADQL, and PPO's validation curves fluctuate far less;
  - *sample efficiency:* **not supported.** PPO reached 0.95 block-A completion within 50 k–100 k environment steps; MADQL did not reach 0.75 in 1 M;
  - *memory:* MADQL's replay buffer holds 160.2 MB against 11.7–12.7 MB for PPO's rollout buffer. Process peak RSS was measured only for PPO, so the full-process comparison is not available.
  - With three seeds per algorithm, these are statements about this configuration, not about the algorithms in general.
- **Threats to validity:** one machine with six runs sharing it; three seeds per variant; one budget; hyper-parameters not tuned; block-A completion near its ceiling, which limits its power to discriminate; block-A selection bias.

### E4 reward ablation — Option C (Issue #6, complete)

- **Design (approved D3).** The documented design ($\beta$ ∈ {0, 0.3, 1} × $c_{\text{rr}}$ ∈ {0.5, 2}) was compared with two alternatives before training. **Option C** was chosen: four one-change cells on MAPPO, two run seeds each (1 and 2), compared with the main MAPPO runs of the same seeds:

  | cell | change from `mappo.toml` | question |
  |---|---|---|
  | `mappo_zeta0` | ζ = 0 (no shaping) | does learning need the dense shaping signal? |
  | `mappo_beta0` | β = 0 (individual reward) | does team mixing matter? |
  | `mappo_crr05` | $c_{\text{rr}}$ = 0.5 | is the completion–conflict trade-off sensitive to the robot–robot conflict penalty (lower)? |
  | `mappo_crr2` | $c_{\text{rr}}$ = 2 | the same, higher |

  - *Why C:* it costs the same as the documented design (8 runs), keeps the $c_{\text{rr}}$ sweep, which tests the most concrete question left by Issue #5, and replaces β = 1 (a fully shared reward, the weakest motivation) with ζ = 0, which tests whether the comparison depends on shaping. It cannot estimate interactions, β = 1 or a base-reward-only cell.
  - Everything else is fixed, including the selection rule. **Validation, selection and every evaluation use the default reward weights,** so all cells are scored on the same base return.
- **Raw data:** `results/training/mappo_{zeta0,beta0,crr05,crr2}_seed{1,2}/`; block B in `results/dev6/ablation/` (A\* only, 240 episodes per policy). The eight runs ran concurrently with the block-B evaluations, 13 535–13 771 s each; all reached 1 M steps with 0 physical collisions and no wrap of the seed order (0.944–0.992 passes).
- **Selected steps** (block A): ζ = 0 at 750 k and 450 k; β = 0 at 700 k and 800 k; $c_{\text{rr}}$ = 0.5 at 250 k and 1 M; $c_{\text{rr}}$ = 2 at 350 k and 750 k. Block-A completion at selection 0.993–0.997 in every cell.

**Block B, cell − default** (two-level bootstrap over the two run seeds and the 240 shared A\* instances; no p-values)

| cell | completion | base return | conflicts per episode | per seed: conflicts per episode (cell / default) |
|---|---|---|---|---|
| ζ = 0 | −0.005 [−0.012, 0.001] | +1.02 [0.19, 1.84] | −30.1 [−60.2, −4.0] | 3.9, 10.0 / 56.6, 17.5 |
| β = 0 | +0.001 [−0.007, 0.008] | +1.27 [0.52, 2.06] | −33.2 [−62.7, −8.1] | 5.1, 2.6 / 56.6, 17.5 |
| $c_{\text{rr}}$ = 0.5 | 0.000 [−0.008, 0.007] | +1.23 [0.46, 2.04] | −33.0 [−62.4, −8.1] | 6.3, 1.7 / 56.6, 17.5 |
| $c_{\text{rr}}$ = 2 | 0.000 [−0.006, 0.005] | +1.25 [0.53, 2.03] | −33.8 [−63.5, −9.7] | 3.6, 3.0 / 56.6, 17.5 |

- **Completion:** no cell differs detectably from the default; every CI contains 0 and is within ±0.012.
- **Return and conflicts: not attributable to the reward weights.**
  - All four cells are "better" than the default by about the same amount, including $c_{\text{rr}}$ = 0.5 and $c_{\text{rr}}$ = 2, which change the penalty in opposite directions. A monotone effect of $c_{\text{rr}}$ would not produce this.
  - The default cell's policies are the checkpoints selected at 200 k and 300 k (E4, PPO part), and its conflicts are dominated by MAPPO seed 1 (56.6 per episode, mostly robot–obstacle; see the Issue #6 development comparison).
  - Descriptive check on block A, without new runs: averaged over the 11 validations from 500 k to 1 M steps, every run of every cell (default included) lies within completion 0.979–0.992, base return 3.46–3.65 and 0.08–0.20 conflicts per 100 robot-steps. No cell stands out.
  - The block-B differences are therefore **confounded with the selected training step.** The ablation does not show an effect of ζ, β or $c_{\text{rr}}$ on any metric at this budget.
- **Threats to validity:** two run seeds per cell; the confound above; one budget; A\* only.

### Issue #6 development comparison (validation block B; NOT E5)

- **Commands (each run once, after selection on block A; nothing tuned or re-selected afterwards):**
  - `python -m robocoord.marl.study matched-baseline --out results/dev6/baseline_matched --workers 10`
  - `python -m robocoord.marl.study ppo --policy results/training/{mappo,ippo}_seed{1,2,3}/best.pt … --out results/dev6/ppo --workers 3`
  - `python -m robocoord.marl.study ablation …`, `shield …` and `stochastic …` with the policies listed in each `manifest.json`
  - `python -m robocoord.marl.study analyse --root results/dev6` → `analysis.json`, `summary.md`
  - **Raw data:** `results/dev6/{baseline_matched,ppo,ablation,shield,stochastic}/` (`manifest.json` with checkpoint SHA-256, configuration hashes and block-A scores; `records.jsonl`), `results/dev6/analysis.json`, `summary.md`. All local and git-ignored.
- **Historical records are reused, read only (D11).** MADQL, the original baseline and the reference follower come from `results/dev5/comparison/records.jsonl`. Its SHA-256 (`53b4e114…`) is identical before and after Issue #6. The stochastic experiment reran the greedy MADQL episodes (720) and they match the historical records in every stored field except wall time (0 mismatches).
- **Design:** every controller with the auction allocator, endpoint recovery on and shield off; A\* on seeds 10 020–10 099 × {static, obstacles, full} (240 per policy), RRT\* ($n_{\max}$ = 1 000) on 10 020–10 039 × the three levels (60 per policy). Only seeds 10 020–10 099 were used; no test seed.
- **Two baselines (D11):**
  - the **original baseline** (deadlock resolver on) is the *system-level* benchmark;
  - the **matched baseline** `baseline-noresolver` (resolver off, otherwise identical) is the *controller-level* comparator: the learned controllers and the reference follower also run without the resolver.
- **Conflict metrics (D10):** `conflicts_total` (proposed conflicts per episode, all types), `conflicts_per_delivered` (per delivered task; undefined when nothing is delivered, excluded with a count) and `conflicts_per_100` (per 100 robot-steps). Conflicts are the proposals reaching the arbiter, before cancellation; physical collisions are counted separately.
- **Statistics:** per policy as in Issue #5 (paired bootstrap, Wilcoxon, Holm within each metric family, rank-biserial r). Algorithm-level comparisons use a two-level bootstrap with shared instances (B.3), without p-values.
- **Outcome:** 0 physical collisions in all 15 360 Issue #6 episodes.

**Pooled results (A\*: 240 episodes; RRT\*: 60)**

| controller | selected step | A\* completion | static | obstacles | full | RRT\* completion | base return (A\* / RRT\*) | conflicts per episode (A\* / RRT\*) | conflicts / 100 (A\* / RRT\*) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (resolver on; Issue #5) | — | 0.999 | 1.000 | 0.999 | 0.999 | 0.997 | 3.518 / 3.252 | 1.0 / 0.8 | 0.08 / 0.05 |
| baseline-noresolver (matched) | — | 0.986 | 0.967 | 0.995 | 0.991 | 0.998 | 3.419 / 3.245 | 1.0 / 0.8 | 0.08 / 0.06 |
| reference follower (Issue #5) | — | 0.911 | 0.866 | 0.933 | 0.925 | 0.904 | −16.561 / −20.643 | 408 / 509 | 13.51 / 16.31 |
| MAPPO 1 | 200 k | 0.988 | 0.981 | 0.994 | 0.989 | 0.990 | 1.532 / 0.764 | 56.6 / 73.5 | 2.66 / 3.12 |
| MAPPO 2 | 300 k | 0.988 | 0.979 | 0.991 | 0.992 | 0.991 | 2.553 / 2.602 | 17.5 / 13.9 | 0.75 / 0.63 |
| MAPPO 3 | 150 k | 0.988 | 0.978 | 0.988 | 0.994 | 0.995 | 1.501 / 1.457 | 53.1 / 56.7 | 2.31 / 2.47 |
| IPPO 1 | 550 k | 0.982 | 0.966 | 0.987 | 0.990 | 0.983 | 3.166 / 3.002 | 5.3 / 2.4 | 0.23 / 0.14 |
| IPPO 2 | 250 k | 0.989 | 0.984 | 0.988 | 0.993 | 0.990 | 2.838 / 2.310 | 12.3 / 27.7 | 0.53 / 0.99 |
| IPPO 3 | 500 k | 0.992 | 0.980 | 0.997 | 0.996 | 0.993 | 3.048 / 3.043 | 7.6 / 2.2 | 0.35 / 0.16 |
| MADQL 1–3 (Issue #5) | 800 k / 1 M / 700 k | 0.661 / 0.584 / 0.575 | 0.31–0.34 | 0.73–0.81 | 0.63–0.77 | 0.787 / 0.609 / 0.705 | 1.29–1.48 / 1.20–1.51 | 2.4–6.8 / 2.1–10.7 | 0.08–0.22 / 0.07–0.34 |

**Algorithm level (two-level bootstrap: training seeds resampled per algorithm, instances shared; x − y)**

| metric | comparison | A\* | RRT\* |
|---|---|---|---|
| completion | MAPPO − IPPO | +0.000 [−0.006, 0.007] | +0.004 [−0.005, 0.013] |
| completion | MAPPO − MADQL | +0.396 [0.348, 0.441] | +0.299 [0.216, 0.390] |
| completion | IPPO − MADQL | +0.396 [0.347, 0.440] | +0.296 [0.212, 0.385] |
| completion | MAPPO − baseline | −0.012 [−0.016, −0.008] | −0.006 [−0.012, −0.002] |
| completion | IPPO − baseline | −0.013 [−0.019, −0.007] | −0.010 [−0.017, −0.003] |
| completion | MAPPO − matched baseline | +0.003 [−0.004, 0.010] | −0.007 [−0.013, −0.002] |
| completion | IPPO − matched baseline | +0.002 [−0.006, 0.011] | −0.010 [−0.018, −0.004] |
| completion | MAPPO − reference | +0.080 [0.066, 0.094] | +0.090 [0.064, 0.116] |
| completion | IPPO − reference | +0.079 [0.065, 0.094] | +0.086 [0.060, 0.113] |
| base return | MAPPO − IPPO | −1.155 [−1.805, −0.462] | −1.178 [−2.380, −0.037] |
| base return | MAPPO − MADQL | +0.449 [−0.184, 1.126] | +0.251 [−0.942, 1.273] |
| base return | IPPO − MADQL | +1.604 [1.306, 1.894] | +1.428 [0.777, 1.934] |
| base return | MAPPO − matched baseline | −1.557 [−2.171, −0.883] | −1.638 [−2.750, −0.687] |
| base return | IPPO − matched baseline | −0.402 [−0.683, −0.165] | −0.460 [−1.122, −0.159] |
| conflicts per episode | MAPPO − IPPO | +34.0 [10.8, 56.6] | +37.3 [−0.2, 78.4] |
| conflicts per episode | MAPPO − MADQL | +38.2 [15.0, 60.2] | +40.9 [7.2, 81.1] |
| conflicts per episode | IPPO − MADQL | +4.2 [−2.0, 10.9] | +3.7 [−11.0, 23.9] |
| conflicts per episode | MAPPO − matched baseline | +41.4 [17.9, 63.4] | +47.2 [14.8, 85.7] |
| conflicts per episode | IPPO − matched baseline | +7.4 [2.6, 13.6] | +10.0 [1.0, 30.3] |
| conflicts per episode | MAPPO − reference | −365.5 [−428.5, −304.2] | −460.7 [−619.3, −310.4] |
| conflicts per episode | IPPO − reference | −399.6 [−458.4, −343.4] | −497.9 [−654.5, −351.8] |

Per-policy rows with Wilcoxon and Holm p-values and r, the remaining pairs and the per-delivered-task metric are in `summary.md` / `analysis.json`. Their reading:

- **Completion: PPO is far better than MADQL and the reference follower.** Every per-policy comparison with the reference is positive (A\*: +0.073 to +0.083, Holm p ≤ 2.5 × 10⁻¹⁹, r ≥ 0.89; RRT\*: +0.080 to +0.093, Holm p ≤ 4.5 × 10⁻⁵).
- **Against the baselines: slightly lower than the system benchmark, level with the matched baseline on A\*.**
  - Original baseline (resolver on): all 12 per-policy differences are negative, −0.009 to −0.019 on A\* (Holm p ≤ 3.8 × 10⁻⁴) and −0.003 to −0.015 on RRT\*. Part of this gap is the resolver itself (below).
  - Matched baseline: no detectable difference on A\* (−0.004 to +0.006; every Holm p ≥ 0.69). On RRT\* the per-policy differences are −0.003 to −0.016 (r = −1.00 in all six), but only one survives Holm (IPPO 1, p = 0.009); the algorithm-level CIs exclude 0. With 20 RRT\* seeds this is a small, uncertain deficit.
  - On the static level PPO and the matched baseline both complete less (0.966–0.984 and 0.967) than on the dynamic levels; the resolver removes this gap for the baseline. Descriptive, not tested.
- **MAPPO and IPPO complete the same** (algorithm-level CI within ±0.013).
- **Return and conflicts favour IPPO and the baselines.**
  - Every PPO policy proposes more conflicts than either baseline (per-policy Holm p ≤ 0.009) and far fewer than the reference follower.
  - MAPPO 1 and MAPPO 3, the two earliest selected checkpoints, propose 53–74 conflicts per episode. **Most are robot–obstacle vertex conflicts** (41–60 per episode, against 0.8–5.2 for the other PPO policies, MADQL and both baselines). These dominate MAPPO's algorithm-level conflict and return differences.
  - For the other four PPO policies, robot–robot vertex conflicts make up most of the total on A\* (4–16 per episode against 1–2 robot–obstacle). Every proposed conflict is cancelled by the arbiter; physical collisions stay at 0.
  - Base return is below the matched baseline for every PPO policy (IPPO −0.20 to −0.94, MAPPO −0.64 to −2.48).
- **Behaviour (descriptive):** PPO episodes usually terminate (182–213 of 240 on A\*, against 215 for the matched baseline, 238 for the original baseline and 1–7 for MADQL). Mean episode length 178–211 steps (A\*). Wait ratio 0.002–0.091.
- **Selection bias:** block-B A\* completion (0.982–0.992) is 0.001–0.011 below the block-A scores of the same checkpoints (0.992–0.997).
- **Deadlock resolver effect** (original − matched baseline, paired): completion +0.015 [0.009, 0.022] on A\* (Holm p = 2.7 × 10⁻⁵), no detectable effect on RRT\* (−0.001 [−0.002, 0.000]); base return +0.100 on A\*; conflicts unchanged.
- **Threats to validity:** 80 block-B seeds (20 for RRT\*); three policies per variant from one configuration and one budget; the early-selection effect above; completion near its ceiling for PPO and both baselines.

#### Predictive shield (full level; Issue #6)

- **Design:** every learned policy (MADQL 1–3, MAPPO 1–3, IPPO 1–3) on the full level, 80 A\* and 20 RRT\* seeds, with the predictive shield under both allocators and shield off under the sequential allocator. The auction / shield-off arm is the primary comparison above (2 700 new episodes). Paired predictive − off per policy and allocator, Holm within each metric family of 36.
- **Completion:** no detectable effect for any policy or allocator (|mean difference| ≤ 0.008; every Holm p ≥ 0.64).
- **Conflicts:**
  - **MAPPO 1 and 3:** the shield removes most of their robot–obstacle conflicts. Conflicts per episode fall by 60–91 on A\* (Holm p ≤ 2.7 × 10⁻⁸) and 41–78 on RRT\*; robot–obstacle vertex conflicts drop from 64–87 to 4–5 per episode on A\* (49–73 to 4–6 on RRT\*), with 73–87 shield interventions per episode on A\*.
  - Other policies: smaller reductions (about −0.2 to −3 per episode on A\*), significant after Holm for some (for example IPPO 2, MAPPO 2, MADQL 1–3 under the sequential allocator); IPPO 3 under the auction is +3.1 [−0.5, 10.1], not significant.
- **Wait ratio:** no detectable change for any policy (every Holm p = 1; |mean difference| ≤ 0.008).
- **H6** ("the predictive shield reduces robot–obstacle conflicts of learned controllers at the cost of a higher wait ratio"): the reduction is supported, strongly for the two policies with many robot–obstacle conflicts. A wait-ratio cost was **not detected**. Development evidence only; the E5 shield ablation belongs to Issue #8.
- **Sequential allocator (descriptive):** completion is within ±0.03 of the auction for every policy.

#### Stochastic execution diagnostic (D6; Issue #6)

- **Design:** every learned policy on the 240 A\* instances, greedy once and stochastic three times (PPO samples from its policy; MADQL uses masked ε-greedy with ε = 0.05). The random streams are seeded per instance and replicate, independently of the policy; replicates are averaged per instance before pairing (8 640 episodes). Greedy remains the primary mode; nothing above is replaced.

| policy | greedy completion | stochastic completion | difference [95 % CI] | conflicts per episode (greedy / stochastic) |
|---|---|---|---|---|
| MAPPO 1 / 2 / 3 | 0.988 / 0.988 / 0.988 | 0.999 / 0.999 / 0.999 | +0.011 to +0.013 | 56.6 / 15.5, 17.5 / 2.8, 53.1 / 13.4 |
| IPPO 1 / 2 / 3 | 0.982 / 0.989 / 0.992 | 0.999 / 0.999 / 0.999 | +0.008 to +0.019 | 5.3 / 1.8, 12.3 / 4.1, 7.6 / 2.1 |
| MADQL 1 / 2 / 3 | 0.661 / 0.584 / 0.575 | 0.918 / 0.888 / 0.908 | +0.280 to +0.347 | 6.8 / 6.2, 2.4 / 2.7, 3.5 / 2.7 |

- Every completion difference is positive (Holm p ≤ 3 × 10⁻⁵, r ≥ 0.91). Stochastic execution also raises base return for all policies (+0.21 to +1.43).
- For PPO, stochastic execution lowers conflicts per episode (−3.5 to −41.1). For MADQL the mean differences are close to 0, but most pairs have more conflicts with sampling (r +0.68 to +0.82); the means are dominated by a few greedy episodes with many conflicts.
- **Reading (not tested):** greedy execution of a policy without robot identity can hold symmetric situations indefinitely (`02` §6.4); a little randomness breaks them. This affects MADQL far more than PPO. MADQL's greedy shortfall (Issue #5) is therefore largely a property of greedy execution of its Q-function, not only of what it learned. The Issue #5 conclusions about greedy MADQL are unchanged.
- **Determinism check:** the 720 fresh greedy MADQL episodes reproduce the historical records field by field (wall time excluded); 0 mismatches.

#### Issue #6 summary

- PPO (both variants) learns this task at the 1 M-step budget: block-B completion 0.982–0.992 with A\* and 0.983–0.995 with RRT\*, against 0.575–0.661 and 0.609–0.787 for MADQL. The MADQL negative result of Issue #5 stands.
- PPO matches the controller-level (matched) baseline in completion with A\*, is slightly below it with RRT\*, and is below the system-level baseline with the deadlock resolver.
- PPO proposes more conflicts than either baseline, though far fewer than the reference follower. For MAPPO this comes mostly from early selected checkpoints with many robot–obstacle conflicts, which the predictive shield removes.
- The Option C reward ablation shows no effect of ζ, β or $c_{\text{rr}}$ on completion; its return and conflict differences are confounded with the selected training step.
- The completion-first selection rule loses discriminating power once completion saturates; whether to change it (for example a tolerance band on completion before comparing return) is a decision for the student and is **not** made here. RRT\* in training (V-17, borderline) is likewise left to the student.
