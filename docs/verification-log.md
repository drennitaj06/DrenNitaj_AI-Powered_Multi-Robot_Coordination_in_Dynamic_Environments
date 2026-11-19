# Verification Log

Central, chronological record of **executed** verification: automated test runs, quality gates and manual reviews. Each entry is written from the actual command output. Nothing is recorded that was not run. The procedure is defined in `05-testing-and-evaluation.md` §A.5.

## Entry template

```markdown
### V-<n> — Issue #<k>: <short title> (<YYYY-MM-DD>)
- **Environment:** OS, CPU, Python x.y.z, key library versions (numpy, torch, scipy, pygame-ce)
- **Commands:**
  - `pytest ...`
  - `ruff check .` · `ruff format --check .` · `mypy src/robocoord`
- **Results:** passed P · failed F · skipped S · errors E · duration T s
- **Long suite:** passed … (or "not applicable")
- **Coverage:** total X % (core Y %, sim Z %, planning …, allocation …)
- **Quality gates:** ruff ✔/✘ · format ✔/✘ · mypy ✔/✘
- **New/changed tests:** list of modules or test IDs
- **Failures and resolution:** what failed, root cause, fix, re-run result
- **Known limitations / open problems:** …
- **Raw report:** `results/test-reports/issue-<k>.xml` (local, not committed)
```

---

## Entries

### V-0 — Issue #1: Design documentation review (2026-11-17)
- **Scope:** documentation only. No application code exists, so **no automated tests were run**.
- **Checks performed:**
  - Cross-references between documents (file names and section references) checked with a text search over `docs/`.
  - Traceability: every specification item maps to requirements, design sections and an issue (`01-requirements.md` §10).
  - Required algorithms (A\*, RRT\*, MADQL, PPO, auction) present in requirements, model, architecture, algorithm design and issue plan.
- **Environment noted for later issues:** Windows 11, Python 3.14.3, pip 26.2.2. PyPI lists Python 3.14-compatible releases of torch, pygame-ce and numpy (checked 2026-11-17).
- **Open items:** decisions awaiting student approval (`07-development-plan.md` §5).

### V-1 — Issue #1: Design revision 2 and cross-document consistency review (2026-11-17)
- **Scope:** documentation only. **No application code, simulation code, tests or experiments were created, and no automated tests were run.** No Git or GitHub operations were performed.
- **Trigger:** the student's design review requested seven technical corrections and resolved the 12 approval decisions.
- **Changes by document:**
  - `02-mathematical-model.md` (rewritten): conflict taxonomy, arbitration and the physical-collision invariant (C7); aging properties P1–P3 and the starvation non-guarantee (C2); shared dynamic-constraint set, goal-availability policy, empty-set convention, reactive-planning statement (C5); path monitor T1–T6 and reference-vs-executed motion with execution layers (C4); phase-aware potential, invariance argument and its scope (C6); weighted vs strict priority (C3); conflict, counterfactual and adherence metrics (C7); summary of guarantees §12.
  - `04-algorithms.md` (rewritten): `detect_conflicts` / `arbitrate` split (C7); shared constraint builder and procedure (C5); RRT\* neighbour index redesign (brute force + Bentley–Saxe k-d tree), weighted segment cost, and complexity analysis split into worst, expected, amortised and conditional bounds, with no $O(n\log n)$ claim (C1); weighted-priority and aging tests (C2, C3); path monitor, reference follower and shield (C4); potential tests (C6).
  - `03-architecture.md` (rewritten): pipeline with shield, arbiter, counterfactual and monitor steps; CI matrix; spatial-hash rejection; ADR-2/5/9 updated; new ADR-10 (execution layers); interfaces (`PlanConstraints`, procedure, `NeighborIndex`, shield, reference follower); layout (`kdtree.py`, `neighbors.py`, `constraints.py`, `procedure.py`, `conflicts.py`, `control/reference.py`, `control/shield.py`, `data/manifests/`, `models/`).
  - `05-testing-and-evaluation.md` (rewritten): evidence levels L1–L4; new cross-cutting tests X-10…X-13; seed ranges; paired bootstrap, Wilcoxon, effect sizes, two-level bootstrap; explicit experiment budgets (incl. E6 one-factor-at-a-time design); revised hypotheses H1–H6.
  - `01-requirements.md` (edited): FR-ENV-5/6, FR-TASK-3/4, FR-ALLOC-3, FR-PLAN-1…7, FR-RL-7/8, NFR-PORT-1, C-4, A-1/5/8/10, AC-3/5/6/7/8, new AC-12/13, out-of-scope additions.
  - `06-kaggle-datasets.md` (edited): provisional status; explicit verified vs unverified facts; no-commit data policy; manifests; verification gate §5.1.
  - `07-development-plan.md` (rewritten): issue scopes and acceptance criteria updated for the corrections, same 8 issues and order; resolved decision table; new risks.
  - `docs/README.md`: index descriptions updated. `.gitignore`: `data/processed/` added.
- **Consistency checklist (manual review, text search across all `docs/` files):**
  - [x] Symbols: $B_i$, $Q_i$, $\omega_i$, $\Lambda_i$, $\Phi_i$, $\Psi_i$, $\tilde w_j$, conflict types (RRv, RRs, ROv, ROs, INV, SH) used identically in 02, 03, 04, 05.
  - [x] Algorithm descriptions match the model: one shared constraint and goal policy for A\* and RRT\*; arbiter semantics; potential definition.
  - [x] Complexity claims match the designed implementations (RRT\* table in `04` §3.6; no remaining $O(n\log n)$ claim; spatial hash only mentioned as rejected).
  - [x] No guarantee stated beyond model §12 (search for "starvation", "first", "optimal", "guarantee").
  - [x] Acceptance criteria reflect weighted priority, aging as mitigation, and no end-to-end completion guarantee.
  - [x] Every *Guaranteed* item in §12 has a test, and *Not claimed* items are measured, not tested.
  - [x] Metric definitions are identical in 02 §9, 03, 04, 05 (proposed conflicts, avoided, introduced, persisted, physical-collision invariant).
  - [x] 8 issues, unchanged order and dependencies; no milestones.
  - [x] Python-only constraint respected (own k-d tree in Python/NumPy; no compiled extensions).
  - [x] Cross-references (file names, section numbers) resolve.
- **Open items (unresolved, assigned to issues):**
  1. Citations used from memory that must be checked against the sources before the corresponding code is final:
     - the corrected RRT\* radius constant (Solovey et al. 2020) and hence the default $\gamma_f$ (Issue #3). **Closed in V-6:** verified from the paper; the correction is to the exponent, so G7 is not claimed for the implemented radius;
     - the ε-scaling bound $O(kA\log(kC))$ (Bertsekas & Eckstein 1988) (Issue #4);
     - the k-d tree range-query bound (Lee & Wong 1977; de Berg et al.) and the expected nearest-neighbour result (Friedman, Bentley & Finkel 1977) (Issue #3). **Closed in V-6 as "not verified against the originals"**: the sources were not accessible; secondary support and the exact status are recorded there, and `04` §3.6 is worded accordingly.
  2. Default RRT\* neighbour index: chosen from E1b measurements (Issue #3).
  3. Kaggle datasets: provisional until the verification gate (Issue #7).
  4. Python 3.12/3.13 support: claimed only after CI passes (Issue #2).
  5. Numerical defaults (κ, α, h, λ_prox, reward weights, hyper-parameters) are initial values; any change is recorded in the implementing issue.
  6. RL training budgets: fixed in Issue #5 from measured throughput. RRT\*-in-training decision: Issue #6.
  7. Known modelling limitations (documented, not to be fixed): shaping invariance under truncation is conditional; the counterfactual conflict metric is one step deep; planning is reactive (no space-time); no end-to-end completion guarantee.
- **Approval:** the student approved design revision 2 on 2026-11-17. Implementation (Issue #2) starts only when the student authorises it.

### V-2 — Issue #2: Simulation core, scenario system and project tooling (2026-11-17)
- **Environment:** Windows 11 Pro, Python 3.14.3 (venv), numpy 2.4.6, pytest 9.2.0, hypothesis 6.170.15, ruff 0.17.5, mypy 2.4.1 (full list in `requirements-lock.txt`).
- **Environment issue found:** with numpy 2.5.5, `import numpy.random` failed: *"DLL load failed while importing _philox: An Application Control policy has blocked this file."* This is a Windows security policy on this machine, not a code defect. numpy 2.4.6 loads normally and is pinned in the lock file. The package requirement stays `numpy>=2.0`. No security setting was changed.
- **Commands and results (actual output):**
  - `ruff check .` → All checks passed. `ruff format --check .` → 56 files already formatted.
  - `mypy` (strict, `src/robocoord`) → Success: no issues found in 28 source files.
  - `pytest --cov=robocoord` (fast suite) → **126 passed**, 3 deselected (e2e), 0 failed, 0 skipped, 3.54 s. Coverage (statements + branches) **92 % total**. Statements only: `core` ≈ 96 %, `sim` ≈ 94 %. Lowest modules: `sim/text_render.py` 77 % (Windows console code paths) and `sim/view.py` 87 %.
  - `pytest -m "slow or training or e2e"` → **3 passed** (CLI in a subprocess, cross-process byte-identical traces, error exit codes), 1.86 s.
  - Reports: `results/test-reports/issue-2.xml`, `issue-2-long.xml` (local, not committed).
- **Tests added:**
  - core: grid and MovingAI I/O round-trip; BFS vs Floyd–Warshall (property); rng streams.
  - world: exhaustive task-state transitions (25 pairs); motion models; obstacle move rule (property); arrival processes incl. a statistical rate check; map-event deferral.
  - scenario: 29 invalid-field cases (X-3); file and generator loading; generator connectivity (3 properties).
  - sim: conflict hand cases plus **600-case property test vs a naive oracle**.
  - integration: hand-computed episode metrics (X-6); counterfactual avoided/introduced/persisted sets with a stub planner (X-10 basis); invariants under random play on 40 generated scenarios (X-2); byte-identical traces for the same seed and different traces for a different seed (X-1); assignment-API state machine; CLI.
- **Failures during development and resolution:**
  1. The numpy DLL policy block (above).
  2. One hand-written test placed an obstacle inside a robot's current cell, a state the obstacle move rule makes impossible. The invariant checker correctly flagged it, and the test now checks that case at detection level only.
  3. The first demo script deadlocked: a random-walk obstacle pushed two scripted robots into a symmetric head-on wait. The scenario now uses a patrol obstacle in an unused column.
- **Demo result (scripted, `configs/scenarios/demo_delivery.toml`, seed 7):**
  - 4/4 tasks delivered, makespan 29 steps;
  - 1 robot–obstacle conflict proposed and prevented by the arbiter;
  - 0 blocked moves, 0 physical collisions;
  - map event: block at t = 15, unblock at t = 24.
- **Not yet verified:** the GitHub Actions matrix (Python 3.12/3.13 on Linux, 3.14 on Linux and Windows) runs only after the branch is pushed. Python 3.12/3.13 compatibility is therefore unverified until the student reports the CI result.
- **Known limitations:**
  - Without planners (Issue #3) the counterfactual "collisions avoided" metric is reported as n/a in CLI runs.
  - Scripted robots cannot resolve symmetric deadlocks.
  - The shield and path monitor are pipeline placeholders until Issues #3–#4.
- **Superseded by V-3:** the test counts and coverage above were taken before the final review; V-3 has the final numbers.

### V-3 — Issue #2: final verification and review fixes (2026-11-17)
- **Method:** source review of every module against the approved model, architecture and Issue #2 acceptance criteria, then re-execution of all gates, plus independent checks that do not reuse the code under test.
- **Environment:** Windows 11 Pro, Python 3.14.3, numpy 2.4.6, pytest 9.2.0, hypothesis 6.170.15, ruff 0.17.5, mypy 2.4.1. Only Python 3.14 is installed locally.
- **Gates (actual output):**
  - `ruff check .` → All checks passed. `ruff format --check .` → 57 files already formatted. `mypy` (strict) → no issues in 28 source files.
  - `pytest` (default, what CI runs) → **142 passed, 3 deselected**, 3.0 s.
  - `pytest -m e2e` → **3 passed** (the 3 deselected tests).
  - Complete suite (`-o addopts="-ra --strict-markers"`, fast + e2e) → **145 passed**, 0 failed, 0 skipped, 5.2 s. Report: `results/test-reports/issue-2-final.xml` (local).
- **Why 3 tests are deselected by default:** `pyproject.toml` sets `addopts = -m "not slow and not training and not e2e"` (testing doc §A.2). The three tests in `tests/e2e/test_cli_subprocess.py` are marked `e2e` because they start fresh interpreters. They must be run before an issue is closed (§A.5) and were: 3 passed. They are excluded from CI's fast job by design.
- **Coverage** (`--cov-branch`, whole suite): lines 1732/1828 = **94.7 %**; line+branch combined 93.2 %. By package (lines / branches): **core 96.0 % / 93.1 %**, **sim 95.6 % / 89.9 %**, world 96.9 % / 87.0 %, scenario 91.4 % / 83.7 %, control 94.5 % / 90.0 %, top-level 92.8 % / 88.5 %. Target (≥ 90 % lines in core and sim) met. Lowest modules: `scenario/loader.py` 85 % (error branches), `core/mapio.py` 86 %, `scenario/generators.py` 86 %, `sim/view.py` 87 %, `sim/text_render.py` 88 %. `__main__.py` shows 0 % because it only runs in the subprocess tests.
- **Independent checks (scratch scripts, not committed; actual results):**
  - Determinism: 10 repeated runs of each of 3 configurations → exactly 1 distinct trace digest each.
  - Seed sensitivity: `random_warehouse` with seeds 0–7 → 8 distinct traces. The scripted demo's step records are identical across seeds 7/8/9, as designed (patrol and bounce obstacles use no randomness; only the header's seed field differs).
  - Detector + arbiter vs the naive oracle: 30 000 random situations (separate from the 600-case Hypothesis test) → 0 mismatches, 0 physical violations.
  - Full engine with a controller that ignores the mask: 60 scenarios, 3 600 steps, 7 922 conflicts proposed (6 011 of them moves into walls) → 0 physical violations.
  - Trace audit (re-derives legality and 15 metrics from the raw JSONL only): demo and corridor_queue → every recomputed value equals the engine's; no jumps, wall entries, shared cells, swaps or obstacle overlaps.
- **Findings and fixes (all within Issue #2):**
  1. *Bug:* `ScriptedCoordinator` assigned tasks that were pending but unavailable (a map event had cut off the drop-off) and crashed with `EngineStateError`. Now waits for availability. Test added.
  2. *Gap vs acceptance criterion 4:* the model §9.1 metric "blocked-goal time" (needs no planner) was missing. Added `blocked_goal_robot_steps`, plus per-100-robot-step rates for blocked moves and for each conflict kind. Hand-computed test added.
  3. *Interface gap for the approved design:* coordinators had no access to the shared `DistanceOracle` (auction benefits, planner reachability) or the `planner` random stream (RRT\*). `AssignmentAPI` now exposes `oracle` and `streams`, and `Coordinator.reset` receives the API. Documented in architecture §5.3. Test added.
  4. *CLI defect:* the default animation emitted clear-screen escape codes into redirected output. Now only on a real ANSI terminal. Test added.
  5. *Ambiguity:* scenarios run without a coordinator report 0 deliveries. The CLI now says so explicitly. See the `random_warehouse` explanation below.
  6. *Scope gap:* only 2 example scenarios shipped (scope: 3–4). Added `corridor_queue.toml` and `rooms_dynamic.toml`. One test now loads, validates and runs every shipped example.
  7. *Test strengthening:* generators are verified to be connected without the pruning step (300 random sizes probed: 0 needed pruning); the no-task sandbox rule (runs to `max_steps`) is documented and tested.
- **`random_warehouse` delivering 0/N tasks is expected, not a defect.** That scenario has no `[script]`, so the CLI attaches `NullCoordinator`, which never assigns tasks. Robots stay in mode *idle* for the whole run (modes seen: only `I`; assignments made: 0), and pickup requires an assignment, so the random walkers crossed 6 of 10 pickup cells without picking anything up. The auction (Issue #4) fixes this by design.
- **Demonstration findings (honest):**
  - Dynamic obstacles do influence movement: the demo's single robot–obstacle conflict (t = 27) stops robot 1 for one step; without obstacles the makespan is 28 instead of 29.
  - The demo's map event is applied and visible but does not change any robot's behaviour (makespan 29 with or without it). Deferral and unavailability are demonstrated by `corridor_queue.toml` and covered by tests.
  - Scripted robots follow pre-computed routes around static walls. They do not avoid dynamic obstacles or each other on their own: the arbiter stops them and they retry. Avoidance by planning arrives in Issue #3.
- **Not verified / limitations:** the GitHub Actions matrix (Python 3.12 and 3.13 on Linux, 3.14 on Linux and Windows) has not run, so 3.12 and 3.13 compatibility is unverified. CI resolves current ruff/mypy versions, which may differ from the pinned local ones. Counterfactual "collisions avoided" is n/a until a planner exists. Scripted robots cannot resolve symmetric head-on deadlocks.

### V-4 — Issue #3: Online path planning, A\* and RRT\* with dynamic replanning (2026-11-18)
- **Scope:** `planning/` (constraints, procedure, A\*, geometry, `BruteForceIndex`, `StaticKDTree` + `LogKDTreeIndex`, RRT\*, path monitor, planning coordinator), the engine's optional `PlanReporting` hooks, plan adherence, `SuggestionFollower`, CLI `simulate --planner` and `experiment`, `experiments/` (spec, runner, records, stats, planning_runs, analysis), example scenario `planning_detour.toml`, experiment configs E1a, E1b, E1b-memory, E1c and E2.
- **Environment:** Windows 11 Pro, CPython 3.14.3, NumPy 2.4.6, pytest 9.2.0, Hypothesis 6.170.15, ruff 0.17.5, mypy 2.4.1. CPU: Intel 64 Family 6 Model 186, 12 logical CPUs. Only Python 3.14 is installed locally.
- **Gates (actual output):**
  - `ruff check .` → All checks passed. `ruff format --check .` → 89 files already formatted. `mypy` (strict) → no issues in 47 source files.
  - `pytest --junitxml=results/test-reports/issue-3.xml --cov=robocoord --cov-branch` (fast suite, what CI runs) → **218 passed, 3 deselected**, 16.4 s.
  - `pytest -m "slow or training or e2e" --junitxml=results/test-reports/issue-3-long.xml` → **3 passed** (the deselected e2e tests).
  - Complete suite (`-o addopts="-ra --strict-markers"`) → **221 passed**, 0 failed, 0 skipped, 17.7 s.
- **Coverage** (branch, complete suite): total lines 95 % (TOTAL 3 708 statements, 138 missed). By package (lines): **planning 98.2 %**, **core 96.4 %**, **sim 96.1 %**, experiments 97.0 %, world 96.9 %, control 93.7 %, top-level 91.9 %, scenario 91.4 %. The ≥ 90 % targets for core, sim and planning are met.
- **Acceptance-criterion evidence:**
  1. A\* oracle: 1 200 random instances (600 unit-cost vs BFS, 600 proximity-weighted vs an independent $O(V^2)$ Dijkstra); all equal, all paths valid (`tests/unit/planning/test_astar.py`).
  2. RRT\*:
     - Validity, monotone best cost and the exact tree-cost invariant after every iteration (60 Hypothesis queries, both indexes).
     - Determinism per seed; identical trees with both indexes.
     - k-d tree ≡ brute force on all generated insert/query sequences (200 Hypothesis cases, with ties and duplicates).
     - Slot and static-tree invariants; supercover vs an exact rational oracle; weighted cost vs numerical integration.
  3. X-11: 318/318 random queries gave identical `status` and `degraded` (268 ok, 27 ok + degraded, 23 unreachable).
  4. Both planners are selectable by name (`make_planner`, CLI `--planner`, experiment specs). Triggers T1–T6 fire at the exact specified steps and are attributed (`test_monitor.py`, `test_planning_integration.py`). X-1 is byte-identical for both planners on `planning_detour.toml`.
  5. E1 and E2 ran and are recorded in `05` §B.6, with counters and exponents compared to the named bounds. U: **does not hold** at the measured budgets ($k$, $k'$ linear in $n$). R: consistent (k-d nearest-neighbour work exponent ≈ 0.1 as first reported; **corrected in V-5** to 0.32–0.35 once every examined node is counted, which still matches the expected $O(\log^2 n)$).
- **Experiments (actual runs; raw data in `results/experiments/`, local and git-ignored; commands: `python -m robocoord experiment --config configs/experiments/<name>.toml`, then `python -m robocoord.experiments.analysis results/experiments/<name>`):**
  - E1b timed: 480 runs, 1 worker, 461 s → 440 ok, 40 budget_exhausted, 0 timeouts, 0 errors. E1b memory: 80 runs → 72 ok, 8 budget_exhausted. **Default index set to `brute`** from these results.
  - E1a: 11 200 runs, 10 workers → 8 665 ok, 2 535 budget_exhausted (RRT\* only), 0 timeouts, 0 errors.
  - E1c: 20 runs → 20 ok.
  - E2: 640 episodes, 10 workers, 151 s → 640 done, 0 timeouts, 0 errors.
- **Demonstration** (`python -m robocoord simulate --scenario configs/scenarios/planning_detour.toml --planner astar|rrtstar --no-animate`): both deliver 2/2 tasks, with 0 physical collisions and plan adherence 1.00.
  - A\*: makespan 73, 16 calls (T1 4, T2 7, T4 1, T5 4), 4 `unreachable` while the drop-off cell is blocked.
  - RRT\*: makespan 49, 18 calls (T1 4, T2 8, T4 2, T5 4), 4 `unreachable`.
  - The shorter RRT\* makespan comes from its different detour around the crossing obstacle in this single scenario; it is not a general result (see E2).
- **Findings and fixes during the issue:**
  1. *Gap found by the demo:* T2 skipped the goal cell entirely, so a map event on the goal was never noticed and the `unreachable` hold/release policy (model §5.3) could not trigger. Fixed: the goal is checked against $S(t)$. Clarification added to model §5.4 and `04` §5.2; covered by X-5 tests.
  2. *Last leg not counted:* the final delivery leg was not closed because the episode ends before the next monitor update. Fixed with the `finish` hook.
  3. *Conservative corner tolerance:* a Hypothesis case passed within ~1e-219 of a lattice corner; the implementation adds the side cell (safe), the exact oracle does not. The test now checks "superset of the exact set, extra cells within 1e-7", as documented.
  4. *E1b procedure:* the first E1b attempt also ran a second, untimed pass under `tracemalloc` for memory, which made it about 10× slower. It was stopped and its partial output discarded. Timing and memory were then measured separately (`e1b_neighbor_index`, `e1b_memory`); no timing was taken under `tracemalloc`.
  5. *Steer step:* $\eta$ was unspecified. A pilot on micro-benchmark seeds 39000–39009 chose $\eta = 2$ (`04` §3 notes).
- **Known limitations (honest):**
  - The plan follower has no right-of-way rule. Robot–robot contention for a free cell, a robot waiting at a goal occupied by a patrol obstacle that in turn waits for the robot's cell, and an A\* replanning livelock next to an oscillating obstacle all leave E2 episodes incomplete (examined in `05` §B.6). These are controller-level issues for Issues #4–#6. T4 is specified for occupied cells only.
  - RRT\* succeeds within budget far less often than A\* on large or cluttered maps (G8 not claimed). The conditional complexity rows of `04` §3.6 do not apply at online budgets because U fails.
  - Wall times are indicative (mixed core types; E1a and E2 ran in parallel). Counters are the primary complexity evidence.
  - The Wilcoxon p-value is not yet implemented (SciPy arrives in Issue #4); paired bootstrap CIs and rank-biserial are reported.
  - GitHub Actions has not run for this branch; Python 3.12/3.13 remain unverified until it does.
  - No Git commands were executed. One shell command accidentally contained `git diff --stat`; the permission system blocked it before it ran.

### V-5 — Issue #3 review: k-d tree work counters, demo trace analysis (2026-11-18)
- **Requested by the student before committing Issue #3:**
  1. verify the k-d tree instrumentation;
  2. explain the 12 blocked-goal robot-steps and the 4 `unreachable` calls in both planning demos;
  3. explain why RRT\* makes two more planner calls than A\*;
  4. add regression tests for any real defect.
- **1. Instrumentation: an under-measurement, now corrected.** The algorithms were not affected.
  - `nn_visited` counted only leaf points whose distance was evaluated, not the internal or pruned nodes a nearest query examines.
  - `range_nodes` counted only nodes whose box intersects the query square, not examined-then-pruned nodes.
  - `rebuild_points` (points placed into newly built static trees, inserted point included) was summarised per *iteration* and described as re-insertion work.
  - Fix: the existing counters keep their meaning with precise docstrings. Added `nn_nodes`, `range_nodes_tested`, `rebuild_work` (Σ subtree sizes of all built nodes) in `kdtree.py` and `neighbors.py`, with matching `PlanStats` fields and RRT\* trace columns. The E1b analysis now reports rebuild quantities per insertion and picks the summary by the experiment name in `spec.json`.
  - Regression tests (`tests/unit/planning/test_neighbors.py`):
    - `rebuild_points` equals $\sum_{t=1}^{n} 2^{\mathrm{ctz}(t)}$ exactly; `rebuild_work` equals the sum of each built tree's Σ subtree sizes, with $m \le$ work $\le m(\text{depth}+1)$.
    - An all-covering radius examines and expands every node; a distant radius examines exactly one root per slot and expands none; examined ≤ slots + 2·expanded.
    - Nearest examines ≥ one node per slot, including the internal root of a multi-leaf tree; a single-leaf tree examines exactly 1 node.
    - Brute-force counters.
  - `test_rrtstar.py`: trace sums equal the totals for the new counters.
- **E1b re-run** into `results/experiments/e1b_neighbor_index_r2/` (same config and seeds, 1 worker): 480 runs → 440 ok, 40 budget_exhausted, 0 timeouts, 0 errors, 7.8 min.
  - The original `results/experiments/e1b_neighbor_index/records.jsonl` is unchanged (MD5 checked before and after).
  - All 9 120 counter values present in both runs are identical, as are every status and path length.
  - **Changed conclusions** (`05` §B.6 E1b, `04` §3 notes): k-d nearest-neighbour nodes examined grow with exponent 0.32–0.35, not the 0.10–0.13 of evaluated points. This still matches $O(\log^2 n)$ under R (local exponent of $\log^2 n$ over $n \in [64, 10^4]$ ≈ 0.32), and is far below $O(n)$.
  - Range nodes examined: 0.51 / 0.76 (open / d20), within $O(\sqrt n + k')$.
  - Rebuild work per insertion: exponent 0.17–0.19 (13–37 points), within amortised $O(\log^2 n)$. The earlier d20 rebuild exponent 0.30 was an artefact of the per-iteration denominator; per insertion it is 0.08.
  - **Unchanged:** U fails; total-time exponents (1.51–1.83); brute force faster at every budget (1.2–2.1× in median time across both runs); default index `brute`.
- **2. Blocked-goal time and `unreachable` calls in `planning_detour.toml`: expected, identical for both planners.**
  - Robot 1 picks up task 1 at t = 4. The scenario's map events block its drop-off (2, 1) at t = 8 and unblock it at t = 20.
  - Blocked-goal robot-steps count steps 8–19 = **12** (robot 1 holds a picked task whose goal is in $S(t)$).
  - The monitor's goal check fires T2 at t = 8 → `unreachable`. The hold policy keeps robot 1 at (12, 1) and retries every $T_{\text{cd}} = 3$: T5 at t = 11, 14, 17 → `unreachable`; T5 at t = 20 → `ok`. Total: **4** `unreachable` calls.
  - Both runs are identical up to t = 11 and these numbers depend only on the event times.
- **3. RRT\* makes 18 calls vs 16 for A\*** (T2 8 vs 7, T4 2 vs 1; T1 4 and T5 4 in both). Traced call by call:
  - Every dynamic T2 in both runs is caused by the bouncing obstacle in column 7, and **no trigger cell was blocked when the previous plan was made**. So no plan was invalid at creation.
  - RRT\*'s polylines cut through the central gap cells (7, 2)/(7, 1) on the obstacle's track. This caused back-to-back T2s: robot 0 at t = 13, 14 and robot 1 at t = 22, 23.
  - At t = 24–26 the two RRT\* robots met head-on in the bottom corridor: 3 swap conflicts (RRs), then a stall of $T_{\text{stall}} = 4$ steps, so T4 for both at t = 27 and a detour.
  - A\* had one stall (vertex conflict, T4 at t = 59). It also had three extra T2s for robot 1 (t = 30, 36, 42) as it switched corridors around the obstacle, which is also why its makespan is longer (73 vs 49).
  - Path-dependent and expected under the reactive, spatial-only planning model (§5.1); not a defect.
- **4.** The only defect found was the instrumentation gap in point 1 (measurement only). Regression tests were added for it. No behavioural bug was found.
- **Gates after the review (actual output):**
  - `ruff check .` → All checks passed. `ruff format --check .` → 89 files already formatted. `mypy` → no issues in 47 source files.
  - Fast suite `pytest --junitxml=results/test-reports/issue-3-review.xml --cov=robocoord --cov-branch` → **221 passed, 3 deselected**. Long suite → **3 passed**. Complete suite → **224 passed**.
  - Coverage: total lines 95 %; `kdtree.py` and `neighbors.py` 100 %.
  - The fast suite took 55–61 s in this session (16 s in V-4). The three new tests account for ≈ 7 s; unchanged tests were also 2–3× slower than in V-4, which points to machine load or power state. It remains under the 2-minute target.

### V-6 — Issue #3 final-review corrections and reference verification (2026-11-18)
- **Scope:** the corrections requested after the independent final review of Issue #3. Required: 1 (RRT\* optimality claim) and 2 (close the V-1 reference items). Recommended: 3 (E1a metrics) and 4 (E2 conflicts). Optional: 5 (loops in RRT\* paths), 6 (fixture re-offer), 7 (polylog fits), 8 (missing tests). The approved architecture and algorithms are unchanged. The RRT\* radius code is unchanged and has no new option.
- **1. RRT\* asymptotic-optimality claim: corrected (was wrong).**
  - The documents said Solovey et al. (2020) derived "a corrected, larger constant" and that asymptotic optimality was claimed "under that corrected condition". This was false.
  - Verified against the paper's own text (arXiv:1909.09688v2, abstract and HTML body, sections III–IV, Theorem 1, Eq. 2–3, Algorithm 2; authors Solovey, Janson, Schmerling, Frazzoli, Pavone; arXiv comment "To appear in ICRA2020"):
    - The original Karaman–Frazzoli radius is $\gamma^{KF}(\log n/n)^{1/d}$, their Theorem 38.
    - Solovey et al. identify a logical gap in its optimality proof.
    - They prove asymptotic optimality for $r(n) = \gamma(\log n/n)^{1/(d+1)}$, i.e. the **exponent** is reduced from $1/d$ to $1/(d+1)$, with $\gamma \ge (2+\theta)\big(\frac{(1+\varepsilon/4)c^\*}{(d+1)\theta(1-\mu)}\frac{|F|}{\zeta_d}\big)^{1/(d+1)}$, $\theta \in (0, 1/4)$, $\mu, \varepsilon \in (0, 1)$. The conclusion is $\Pr[c \le (1+\varepsilon)c^\*] \to 1$.
    - Their algorithm uses $\min\{r(|V|), \eta\}$ with a fixed $\eta > 0$.
    - The main text does not say the $1/d$ radius is proven *not* optimal. The appendix with a counterexample (arXiv v2 only) was not read.
  - The implementation uses exponent $1/d = 1/2$. It therefore does not satisfy the corrected condition for any $\gamma_f$.
  - **G7 is now "Not claimed for the implemented radius"** in model §12 and §5.5, `04` §3.1/3.4/3.5/3.6 notes, the `rrtstar.py` module docstring, the `gamma_factor` docstring, `gamma_star`, the `gamma_factor` error message and the convergence-test docstring.
  - The E1c measurements are unchanged and are now described as finite-budget empirical convergence.
- **2. Reference items from V-1 open item 1: closed with the verification status stated exactly.**
  - *Solovey et al. (2020):* **verified** from the paper's own text (above).
  - *k-d tree orthogonal range-query bound $O(\sqrt n + k)$ in 2-D (de Berg, Cheong, van Kreveld & Overmars, *Computational Geometry*, 3rd ed., Ch. 5; Lee & Wong, *Acta Informatica* 9:23–29, 1977, DOI 10.1007/BF00263763):* **not verified against the originals.**
    - The textbook was not accessible. The Lee & Wong article is behind a login, its zbMATH record (0349.68016) returned 403, and the MaRDI record has metadata only (it lists "K. Appert" before Lee and Wong; not resolved).
    - A search-engine summary of an IBM research record states $O(k\cdot N^{1-1/k})$ for region queries without an output term, unconfirmed.
    - Several university lecture notes state $O(\sqrt n + k)$ query, $O(n)$ storage and $O(n\log n)$ construction for 2-D kd-trees, citing the recurrence $Q(n) = 2 + 2Q(n/4)$. These are secondary sources.
    - The lemma number "5.5" previously cited in `04` §3.6 was **removed**, because it could not be checked.
    - Status: a standard bound supported by secondary sources only; the original statement and numbering are unverified. `04` §3.6 now says so.
  - *Friedman, Bentley & Finkel, ACM TOMS 3(3):209–226, 1977, DOI 10.1145/355744.355745:* bibliographic data **verified** (UKY record). Content **not verified**:
    - The available scan's abstract and algorithm pages are image-only and could not be read; ACM DL and zbMATH returned 403.
    - A search-engine summary of the abstract says the number of records examined is independent of file size and descent time is proportional to $\log N$. Their tree construction (discriminator choice), probabilistic assumptions, and whether their analysis covers our alternating-axis trees were **not verified**.
    - `04` §3.6 now states R as a modelling assumption "in the spirit of" FBF, not as a cited theorem about this tree.
  - Open item V-1.1 is closed for Issue #3 with these statuses. The Bertsekas–Eckstein item belongs to Issue #4.
- **3. E1a interpretation (`05` §B.6 E1a, `04` §3 notes, results register).** The two metrics are now defined separately:
  - "steps/BFS" is the executed 4-connected path.
  - "length/LB" is the reference curve: RRT\*'s continuous polyline, or A\*'s lattice path.
  - The text states that a shorter polyline gives **no** travel-time advantage in this simulator, and that RRT\*'s executed path is never shorter than A\*'s.
- **4. E2 conflict reporting.**
  - `summarize_e2` now reports mean / median / share of episodes with any conflict per cell, and lists every episode with ≥ 50 conflicts per 100 robot-steps.
  - `05` §B.6 E2 adds a per-(planner, team) table and names the two outliers of the RRT\*, 1-robot, $K = 4$ cell: seeds 20006 and 20017. Both are robot–obstacle deadlocks at an occupied goal, inspected step by step. Without them the cell mean is 0.13–0.16 instead of 8.84–8.86 (median 0).
- **5. Loop removal (`geometry.remove_loops`, used by `RRTStarPlanner.plan`).**
  - When a cell is revisited, the cells between the two visits are dropped. The result keeps the endpoints, uses only already-checked cells, stays 4-connected, is simple, and its lattice cost cannot increase.
  - Tests: example cases, a Hypothesis property test (sub-sequence, simple, 4-connected), and a regression test on a real looping query (d20, 16×16, map 0, pair 6: 18 → 16 steps). The RRT\* property test now asserts simple paths.
  - **Measured effect** (re-runs into new directories; original records untouched):
    - *E1a* (`e1a_planners_static_r3`): 0 outcome changes; A\* paths and RRT\* polyline lengths identical. 18.0–24.7 % of successful RRT\* paths were shortened by 2.9–3.3 steps on average (max 14), none lengthened. Selected mean steps/BFS changed by ≤ 0.05; `05` E1a now uses the re-run for this column.
    - *E2* (`e2_replanning_r2`): A\* episodes identical. RRT\* completion unchanged with 1 robot and 0.780 → 0.783 with 4 robots. Paired A\*−RRT\* CIs still contain 0.
    - *E1b/E1c:* unaffected (same tree, polyline and cost).
    - **No conclusion changed.** The `planning_detour.toml` demo output is unchanged (A\* makespan 73, 16 calls; RRT\* 49, 18 calls).
- **6. Fixture re-offer after a release (`PlanningCoordinator`, Issue #3 fixture logic only).**
  - A released task is re-offered to the same robot only after a map event (grid version) or after the robot has moved, and only while the task is available. Previously it was re-offered every $T_{\text{cd}}$ steps.
  - Review probe (robot cut off from an available pickup, 40 steps): 14 releases and 14 `unreachable` T1 calls before; 1 and 1 after.
  - Regression test `test_release_is_not_reoffered_until_something_changes`: one release, re-offer exactly at the map event at $t = 12$, then delivery.
  - The coordinator docstring and `04` §5.2 notes keep the distinction from the Issue #4 auction explicit.
- **7. Polylog fits** (`stats.polylog_fit`, residual sums in `PowerFit.rss`, table in `summarize_e1b`, `05` E1b, `04` §3 notes).
  - k-d NN nodes examined: $(\ln n)^{2.14}$ [2.09, 2.20] (open), RSS 1.701 vs 3.589 for the power law; $(\ln n)^{2.29}$ [2.25, 2.32] (d20), RSS 0.421 vs 1.463.
  - Stated as a description over $n \in [64, 10^4]$, not a proof of an asymptotic bound. Unit test with exact $(\ln n)^2$ data.
- **8. Tests added:**
  - T5 retries after `budget_exhausted` (RRT\* with a zero budget; retries at $t = 3, 6, 9$).
  - Degraded-plan/stall interaction: with $T_{\text{cd}} = 3 < T_{\text{stall}} = 4$, T6 at 3, 6, 9 and no T4; with $T_{\text{cd}} = 6$, T4 at 4. Documented as `04` §5.2 note 5.
  - E2 episode timeout recorded as `timeout` with partial metrics.
  - The fixture re-offer and loop-removal tests above.
- **Also fixed:** control characters in `docs/04-algorithms.md`. An earlier edit script had turned `\text` into a tab in 5 places and `\varnothing` into a vertical tab in 1, corrupting $T_{\text{cd}}$, $T_{\text{stall}}$ and $B_i = \varnothing$. A scan of docs, sources, tests and configs now finds none.
- **Gates (actual output):**
  - `ruff check .` → All checks passed. `ruff format --check .` → 89 files already formatted. `mypy` → no issues in 47 source files.
  - Fast suite (`--junitxml=results/test-reports/issue-3-fixes.xml`) → **229 passed, 3 deselected**, 15.1 s.
  - `pytest -m e2e` → **3 passed**. Complete suite with coverage → **232 passed**.
  - Line coverage 95 % total: planning 98.2 %, core 96.4 %, sim 96.2 %, experiments 96.1 %.
- **Data integrity:** MD5 of all six pre-existing `records.jsonl` files (e1a, e1b_memory, e1b, e1b_r2, e1c, e2) verified unchanged after all work. New data: `results/experiments/e1a_planners_static_r3/` and `results/experiments/e2_replanning_r2/`. Derived `summary.md` files were regenerated for `e2_replanning` (new conflict columns), `e1b_neighbor_index_r2` (polylog table) and the two new directories.
- **Left open:** primary-source verification of the de Berg / Lee & Wong statement and of the FBF content (sources not accessible from this environment; stated as unverified above).

### V-7 — Issue #4: environment and stage 1, benefit construction (2026-11-18)
- **Branch:** `feature/task-allocation` at `b8e5fb1`, equal to `develop` and `origin/develop` (Issue #3 merged); verified from `.git` metadata only, no Git commands.
- **Environment changes (approved by the student):**
  - *Hypothesis pinned to 6.155.7* in `pyproject.toml` (`dev` extra, exact pin) and `requirements-lock.txt`.
    - **Why:** 6.170.15 is a compiled build (maturin wheel `cp314-win_amd64`) whose `hypothesis._native` module is blocked by Windows Application Control on the development machine. It imports `_native` with no pure-Python fallback, so `pytest` could not even load `conftest.py`.
    - **How the version was found:** a binary search over PyPI metadata shows 6.155.7 is the newest release with a pure-Python wheel (`hypothesis-6.155.7-py3-none-any.whl`). Every release from 6.156.1 on is compiled per platform, and 6.156.0 is not in the index.
    - **Compatibility:** `requires_python >= 3.10`, 3.14 classifier; only runtime dependency `sortedcontainers` (already pinned).
    - **Verified locally:** in a scratchpad install and then in `.venv`, complete suite 232 passed. `--hypothesis-show-statistics`: all 22 property tests ran 4 185 passing examples, 0 failing. No property test was disabled and no security exception was requested.
  - *SciPy 1.18.1* added to the `dev` extra and the lock file, as a **test and experiment oracle only** (`scipy.optimize.linear_sum_assignment`, `scipy.stats`); never the allocator.
    - Installed into `.venv` only (37.4 MB wheel). NumPy stayed at 2.4.6.
    - It loads despite Application Control. `linear_sum_assignment(maximize=True)` matched brute force on 300 random instances.
    - PyPI provides cp312, cp313 and cp314 wheels for manylinux x86_64 and win_amd64, which covers the CI matrix.
  - *CI resolution:* `pip install -e ".[dev]" --dry-run --report` (what CI runs) resolves Hypothesis 6.155.7 and SciPy 1.18.1, the same as the lock file. **Not yet verified:** an actual CI run with the new pins.
  - *Pre-existing difference, documented rather than changed:* CI resolves NumPy 2.5.5 (`numpy>=2.0`), while the lock file pins 2.4.6 (Issue #2 Application Control pin). Explained in `03-architecture.md` §3.3 and in the lock-file header; recorded experiments use the lock file and store their NumPy version in `env.json`.
- **Stage 1 — benefits** (`src/robocoord/allocation/{__init__,benefits}.py`, `tests/unit/allocation/{__init__,test_benefits}.py`): implementation notes in `04` §4.1. The only deviation from the model is the formulation of $M$ on scaled integer components, a robustness refinement that is exact for every $\kappa$ and $k$, documented there. Tests: hand-computed instance; infeasible pairs and padding; 300 brute-force instances for maximum cardinality, optimality among maximum-cardinality matchings and padding equivalence; 300 instances for the rounding bound; 60 large instances ($k \le 120$, including $\kappa = 0$ and $h = 0$) for the offset margin; validation and overflow guard.
- **Gates (actual output):** `ruff check .` passed; `ruff format --check .` 93 files formatted; `mypy` no issues in 49 source files. Fast suite **236 passed, 3 deselected**; e2e **3 passed**; complete suite **239 passed**. Coverage: `allocation/benefits.py` 99 %, total 95 %.
- **Stage 2 — auction** (`src/robocoord/allocation/{base,auction}.py`, `tests/unit/allocation/test_auction.py`): implementation notes in `04` §4.1.
  - Exact int64 auction. ε-scaling uses a $(k+1)$ benefit scale, so integer ε = 1 is $\varepsilon = 1/(k+1) < 1/k$ (exact optimum). Fixed ε gives $V \ge \mathrm{OPT} - k\varepsilon$ and at most $k(\lfloor C/\varepsilon\rfloor + 1) + 1$ bids, derived in `04`.
  - Tests against SciPy's Hungarian solver: 200 square matrices ($k \le 60$) for exact optimality, final ε < 1/k and ε-CS; 200 rectangular benefit problems for integer optimality, maximum cardinality and the $q/s$ real-value bound; 200 matrices × 5 values of ε for the additive and bid bounds; a non-vacuous suboptimal fixed-ε case; edge cases, determinism, validation, caps.
  - **Defect found by the tests and fixed:** the time and bid caps were checked every 256 bids *per phase*, so an ε-scaling solve with short phases never checked them. Now counted cumulatively and also checked at each phase boundary.
  - **Finding:** two integer-optimal matchings can differ in original units by up to $q/s$ (observed 92.0 vs 92.1). Tests compare integer totals exactly and real values within that bound.
  - **Gates (actual output):** `ruff check .` passed; `ruff format --check .` 96 files formatted; `mypy` no issues in 51 source files. Fast suite **243 passed, 3 deselected**; e2e **3 passed**. Coverage: `allocation/auction.py` 99 % (only the `k = 0` return of `epsilon_cs_violation` is not covered), `allocation` 99 %, total 95 %.
  - **Not yet verified:** a CI run with the new pins (needs a push by the student).
- **Stage 3 — sequential auction** (`src/robocoord/allocation/sequential.py`, `tests/unit/allocation/test_sequential.py`): implementation notes in `04` §4.2.
  - Greedy rule on the padded integer matrix, implemented as one sort on $(-A_{ij}, i, j)$. Tie rule documented: lowest task index within a bid, lowest robot index among equal bids.
  - **Oracles:** a literal round-by-round simulation written from the `04` §4.2 rule; SciPy's Hungarian solver; full enumeration. A scratchpad mutation check confirmed the literal oracle detects a flipped tie rule (99/200 and 102/200 instances differ).
  - **Results:** same pairs as the literal rule on 200 matrices; $2V \ge \mathrm{OPT}$ on 200 matrices and 200 benefit problems; maximality; optimal on additive and single-robot instances; constructed counter-examples (tight ½ ratio; 3 vs 4; corridor cost 9 vs 7).
  - **Findings:**
    - The ½ guarantee is for the integer objective including the offset. In original units it does not hold: on the κ = 4 corridor, greedy −1 vs optimum +1.
    - On benefit problems the offset gives a derived integer ratio $V/\mathrm{OPT} \ge (k-1)/(k+1)$, also tested on the 200 benefit problems. It mostly measures $M$, so E3 will report the gap in original units.
    - On benefit problems with grid distances greedy always reaches maximum cardinality (component-complete feasibility; proof in `04` §4.2). This refines the §4.2 sentence that greedy reaches maximum cardinality "only if greedy happens to", which remains true for general matrices.
  - **Gates (actual output):** `ruff check .` passed; `ruff format --check .` 98 files formatted; `mypy` no issues in 52 source files. Fast suite **251 passed, 3 deselected**; e2e **3 passed**. Coverage: `allocation/sequential.py` 100 %, total 95 %.
  - **Not in this stage:** allocation triggers and policy (stage 4), so the second approved commit message ("…and allocation triggers") should wait until stage 4.
- **Stage 4 — allocation policy and coordinator** (`src/robocoord/allocation/policy.py`, `src/robocoord/planning/coordinator.py`, `tests/unit/allocation/{test_priority,test_policy}.py`, `tests/integration/test_allocation_integration.py`): implementation notes in `04` §4.3–4.4. `03` §5.5 updated.
  - **Policy:** triggers released, delivered, idle, unreachable (carried to the next step), map_event and retry; at most one auction per step; eligibility per model §8; application unassigns before it assigns; pre-emption only through the auction; a new goal leads to T1 in the same step.
  - **Refactor:** the fixture `PlanningCoordinator` keeps its interface and behaviour (shared base class). The existing planning, integration and e2e tests pass unchanged (107 + 3 before the new tests were added).
  - **Refinement — keep-current tie-break (`prefer_current`):** model §8 requires that a robot keep its task at a gain of exactly $h$. A scratchpad check showed both allocators switching at gain $= h$ whenever the other task was listed first. The exact lexicographic matrix $(k+1)A + E$ fixes this (proof in the code) without changing the optimum of $A$. The hysteresis test now runs both task orders.
  - **Documentation requested by the student:** `04` §4.2 now states that the $(k-1)/(k+1)$ ratio holds for the offset-inclusive integer objective, not the original unshifted benefits, with its proof and assumptions (i)–(iv). It also states the connected-component assumption (CC), why it holds for `build_problem` with `DistanceOracle`, and when it would fail.
  - **Finding (aging):** in a constructed single-robot stream, aging moved the low task's pickup from $t = 126$ to $t = 18$ when no backlog forms. With a backlog of level-3 tasks, the low task won an auction at $t = 42$ but was pre-empted twice by aged higher-level tasks and picked at $t = 135$, the same as without aging. This is model §4 P3 / "Not claimed (i)" observed in simulation. Both are kept as tests; neither is a guarantee.
  - **Observation:** with the allocation coordinator, the `map_event` auction normally removes infeasible pairs before the monitor runs (`releases = 0` in the map-event test). The `unreachable` release path is reached only when that solve is capped. A test with an allocator capped at the map-event step covers it: one release, then an auction triggered by `unreachable` and `retry`.
  - **Gates (actual output):** `ruff check .` passed; `ruff format --check .` 102 files formatted; `mypy` no issues in 53 source files. Fast suite **309 passed, 3 deselected**; e2e **3 passed**. Coverage: `allocation/policy.py` 100 %, `allocation/sequential.py` 100 %, `planning/coordinator.py` 97 % (uncovered: the two abstract hooks, the default no-op `_on_release` of the base class, and the fixture's length check), total 95 %.
  - **Not yet verified:** a CI run with the new pins (needs a push by the student); the CLI `--allocator` option (stage 6).
- **Stage 5A — baseline right-of-way controller** (`src/robocoord/control/baseline.py`; `planning/constraints.py` type-only change; `tests/unit/control/{__init__,test_baseline}.py`; `tests/integration/test_baseline_integration.py`): implementation notes in `04` §5.3, `03` §5.6 updated. No deadlock detour, reference follower, shield or X-10.
  - **Rule:** the `04` §5.3 pseudocode. Right-of-way goes to the largest (aged task weight π̃, −id) among the robots claiming a cell; idle robots are lowest. $\hat Z_i$ comes from the shared `sense`. Decision reasons are recorded per robot and step.
  - **Collision property (proof in `04`):** with every robot on the baseline, no RR, RO-swap or invalid move is proposed; with constant-velocity obstacles and $r_s \ge 2$, no conflict at all. Property-tested against the simulator's own `detect_conflicts` and `resolve_obstacle_moves` (400 + 400 examples), with a non-vacuity check.
  - **Measured on the 5 demo scenarios × {A\*, RRT\*} × {auction, sequential} (20 runs, exploration script in the scratchpad, then asserted in the integration tests):**

    | | baseline | plain follower |
    |---|---|---|
    | RR conflicts (vertex + swap), total over 20 runs | **0** | 0 – 126 per run (17 of 20 runs > 0) |
    | RO vertex conflicts | 1–2 in 4 runs | 1–4 in 4 runs |
    | Runs where it delivered more / fewer tasks than the other | more in 8, fewer in 4, equal in 8 | |

    - Examples: `demo_delivery` 4/4 with 0 conflicts (follower 1/4, 80–121 RR conflicts); `random_warehouse` A\* 10/10 (follower 7/10, 48 RRv).
    - Fewer for the baseline:
      - `corridor_queue` with the sequential allocator, both planners (1/3 vs 3/3): conservative following lets a timed wall split the convoy.
      - `planning_detour` A\*/auction (1/2 vs 2/2): waiting plus repeated T2 detours around a shuttling obstacle; the follower relied on 12 conflicts cancelled by the arbiter.
      - `rooms_dynamic` RRT\*/auction (5/9 vs 6/9).
  - **Finding (head-on, for stage 5B):** two robots carrying tasks to opposite ends of a corridor never pass, in a single lane *and* in two lanes. T4 fires for both robots at the same step, and the symmetric replans make them meet again. The expected test in `04` §5.3, "head-on corridor case resolved via T4", is **not met**. T4 resolved an asymmetric encounter in `planning_detour` (only one robot's T4 fired). Recorded in a test and listed as limitation 1; not changed in this stage.
  - **Interpretation recorded:** π̃ uses the age $t - \rho_j$ also after pickup (the engine's `Task.age`), with the allocation's α.
  - **Gates (actual output):** `ruff check .` passed; `ruff format --check .` 106 files formatted; `mypy` no issues in 54 source files. Fast suite **361 passed, 3 deselected**; e2e **3 passed**. Coverage: `control/baseline.py` 100 %, `allocation/policy.py` 100 %, total 95 %.
  - **Not yet verified:** a CI run with the new pins; the controller statistics in `EpisodeMetrics` and the CLI (stage 6).

### V-8 — Issue #4: full-scope completion (stages 5B, 6, 7, 8) (2026-11-18)
- **Authorization:** the student's final authorization restored the full `07` scope (reference follower, shield, X-10), approved the stage 5B deadlock-detour design, and required E3, the preliminary E5 and NFR-PERF-1. No Git operation was run by the assistant.
- **Regression method:** before any stage 5B change, SHA-256 hashes of complete traces were taken for 46 configurations: 5 demo scenarios × planners × {fixture, auction, sequential} × {follow, baseline}.
  - After stage 5B: **0 of 46 differ** (resolver off).
  - After stage 6: 0 of 46 differ once the new per-step `suggested` field (X-10) is removed. It is the only trace change for existing configurations.
  - The final run confirms the same.
- **Stage 5B — deadlock detour** (`planning/deadlock.py`; `planning/monitor.py` T7 and optional `RobotInput` fields; `planning/coordinator.py` hooks; `allocation/benefits.robot_priorities`): design and proof in `04` §5.5.
  - Safety proof re-checked against the stage 5A code: it holds for arbitrary suggestions, and the resolver changes only goals.
  - **Three design corrections found during implementation**, each with documented consequences: release horizon instead of the whole-path test; the attempt bound counts starts; static vs dynamic `no_passing_place`. Plus the exclusion of detour plans from the task-release policy, and T7 listed only when the resolver is enabled (trace stability).
  - **Bug found by self-review and fixed before testing:** cycles still persisting after an abort or a map change were never handled again (detection used `count == detect_steps`).
  - Results:
    - two-lane head-on resolved for both planners × both allocators × {unequal, equal} priority;
    - single lane reported `no_passing_place`;
    - a 4-robot rotation resolved for 3 priority patterns;
    - a map event on the passing cell gives an abort, then resolution;
    - an obstacle delays a detour, which still completes;
    - 0 RR / ROs / INV with the resolver on in 20 demo configurations.
  - Demo deliveries: 89 with the resolver vs 89 without (gains and losses listed in `04` §5.5).
  - One Issue #3 test assertion updated: the trigger table now has T7; the base triggers are unchanged.
- **Stage 6 — reference follower, shield, X-10** (`control/reference.py`, `control/shield.py`, `sim/engine.py` step 5 and `SafetyLayer`, `sim/metrics.py` shield counters, trace fields `suggested` / `shield` / `arb_actions`):
  - The reference follower is action-identical to `SuggestionFollower` (300 random worlds).
  - X-10 identities hold in 40 configurations (5 scenarios × 2 planners × 2 allocators × shield on/off).
  - The shield only replaces moves by `stay` and is never at fault (400 worlds). It removes RO conflicts under constant velocity (400 worlds); a counter-example exists with turning obstacles.
  - The shield never intervenes for the baseline (20 configurations).
  - Each X-10 category is demonstrated on a constructed episode.
- **Stage 7 — CLI and demos** (`cli.py`; `configs/scenarios/{headon_two_lane,headon_single_lane,preemption,shield_corridor}.toml`):
  - 144 runs (9 scenarios × 16 combinations) exit 0 with 0 physical collisions.
  - The baseline delivers every task in `demo_delivery`, `random_warehouse`, `headon_two_lane` and `preemption` in all 8 of its combinations.
  - **A wrong demo claim was corrected:** the pre-empted task in `preemption` is re-auctioned to robot 1 at t = 6, not to robot 0.
  - `shield_corridor` delivers nothing for any controller (the obstacle pins itself in the single lane). It is documented as a shield demo only.
- **Stage 8 — experiments** (`experiments/{allocation_runs,allocation_analysis,perf}.py`, `configs/experiments/{e3a_assignment,e3b_priority,e5_prelim}.toml`):
  - E3a: 1 120 solves.
  - E3b: 120 episodes. λ = 0.16 was chosen on validation seeds 10000–10004 before the test run.
  - Preliminary E5: 600 baseline episodes.
  - NFR-PERF-1: median 0.56 ms, p99 18.6 ms, max 36.3 ms (target missed at the maximum; cause BFS-cache rebuilds after map events).
  - Results in `05` §B.6.
  - **Overstatements caught while writing them up and corrected:**
    - the E3b verdict column first used the CI alone (now CI and Holm);
    - the first NFR-PERF-1 harness included idle tail steps (now natural episodes plus a busy-step statistic);
    - "3–10× slower at every k" was replaced by the measured 7–11× at k = 400 and up to ≈ 70× at k ≤ 20.
- **X-6 (task part)** was added as a hand-computed multi-level test: $F_w$ = 8.4, flow times 6 / 18, waits 5 / 17.
- **Final gates (actual output):**
  - `ruff check .` passed; `ruff format --check .` 119 files formatted; `mypy src/robocoord` no issues in 60 source files.
  - Fast suite **550 passed, 35 deselected**; long suite (`-m "slow or e2e"`) **35 passed**; experiment smoke tests 12 passed.
  - Coverage total **95 %**: `planning/deadlock.py` 96 %, `control/{baseline,reference,shield}.py` 100 %, `allocation/*` 99–100 %, `sim/engine.py` 98 %, `cli.py` 94 %.
- **Not verified here:** a CI run (needs a push by the student; it would also confirm the Hypothesis/SciPy pins); behaviour on hardware other than the one machine recorded in `env.json`.

### V-9 — Project Issue #4 (GitHub issue #8): audit corrections, complete preliminary E5, performance evidence (2026-11-18)
- **Numbering:** Project Issue #4 = GitHub issue #8. The plan's own Issue #8 is the final evaluation, where the final E5 belongs.
- **Trigger:** the student's audit (investigation only; no files changed) found:
  - NFR-PERF-1 reported without the per-step exceedance structure, and without raw samples;
  - E5 incompleteness explained only for the static dynamics, with "no unresolved deadlock" true only for robot–robot cycles;
  - the reference-follower half of the *non-learned* preliminary E5 missing, although GitHub issue #8 asks for "preliminary non-learned E5 numbers";
  - detour assumptions R7–R9 and the detection scope undocumented;
  - the numbering ambiguity.

  The student accepted the findings and authorised the corrections.
- **Overwrite protection:** SHA-256 of all 68 files under `results/` was recorded before any change and compared at the end (result below).
- **Code changes (evidence tooling only; no algorithm, resolver or model change):**
  - `experiments/allocation_runs.py`: `system_episode_setup()` is shared by the executor and the replays. A new `fixed.controller` option defaults to `baseline`, so the existing `e5_prelim` configuration behaves identically. `reference` runs the reference follower with its documented defaults (resolver off, shield off).
  - New `configs/experiments/e5_prelim_reference.toml`: identical factors, seeds, scenario parameters, metrics and caps to `e5_prelim.toml`; only the controller differs. Separate output directory.
  - New `experiments/e5_diagnostics.py`: deterministic replay of every incomplete episode, checked against its record, plus a rule-ordered cause classification. It writes `undelivered.json` / `undelivered.md` as new files.
  - `experiments/allocation_analysis.py`: the E5 summary adds pooled completion and incomplete-episode counts and names the controller. New `compare_e5_controllers` writes a paired baseline-vs-reference report; each controller is shown separately, and differences are paired per instance.
  - `experiments/perf.py`: raw per-step samples (`nfr_perf1_samples.csv`) and first-step / map-event / steady-state statistics. The workload and timing are unchanged; the new reads happen outside the timed region. It refuses to overwrite an existing result without `--overwrite`.
  - `experiments/analysis.py`: registers `e5_prelim_reference`.
  - New `tests/integration/test_e5_evidence.py` (20 tests, 11 of them the parametrised cause-name documentation check): executor pairing; each diagnostic cause on a hand-derived episode (including the contested free cell); replays reproduce the records; comparison output; the diagnostics CLI; perf samples and the overwrite guard.
- **Runs:**
  - `e5_prelim_reference`: 600 `done`, 0 timeouts.
  - Diagnostics of the baseline half: 83 incomplete episodes, **0 replay mismatches**. 91 undelivered tasks = 61 obstacle on the goal (54 patrol, 7 random-walk), 28 idle robot on the goal, 1 no plan, 1 horizon. This equals the audit's independent classification exactly.
  - Diagnostics of the reference half: 404 incomplete episodes, **0 replay mismatches**. 1 357 undelivered tasks = 1 138 contested free next cell, 65 horizon, 60 obstacle on the goal (49 patrol, 11 random-walk), 59 busy robot on the route, 19 idle robot on the goal, 11 unassigned but available, 3 idle robot on the route, 2 no plan, **0 other**.
  - Rule refinement found while running: the first reference diagnostics left 1 138 tasks as `other`. A replay of a sample showed that in every case the holder's next cell was free but was also another robot's next cell. The plain follower proposes that cell every step and the arbiter freezes all the claimants; T4 does not fire because the cell is not occupied. A `route_contested_cell` rule was added (rule 9) together with a hand-derived junction test (the baseline delivers both tasks there). Both halves were then re-diagnosed; the baseline counts were unchanged.
  - Paired comparison: `results/experiments/e5_prelim_comparison.md`.
  - NFR-PERF-1 re-run into `results/perf_r2/`; the original `results/perf/` is kept.
- **Documentation corrected:** `05` (E5 and NFR-PERF-1 sections, register, numbering), `04` (§5.3 limitations 3 and 6; §5.5 R1–R9, detection scope, unresolvable cases, rotation-test scope), `02` (G20, G22), `07` (scope table, numbering).
- **Docs checked against the raw evidence** (every E5 number recomputed from both `records.jsonl` files; every NFR-PERF-1 r2 number from `nfr_perf1_samples.csv`). Pooled counts, incomplete episodes, conflict ranges, paired-difference ranges, cause counts and the deadlock counters (`unresolved_at_end` = 0 in all 600 baseline episodes) all matched. Two corrections were made:
  - mean-of-episodes column: baseline RRT\*/sequential/full was 0.9874, recomputed 0.9867; reference RRT\*/sequential/full was 0.894, recomputed 0.8949. The column is now 4 decimals, rounded half-up throughout;
  - map-event exceedances: the doc said "11–16 %"; the three runs give 11, 13 and 19 of 116 (9–16 %).
- **Overwrite check** (SHA-256, 68 files before, 80 after; repeated after the test suites): changed only `results/experiments/e5_prelim/summary.md`, regenerated on purpose in the new format from unchanged records. Missing: none. New: `e5_prelim/undelivered.{json,md}`, `e5_prelim_reference/*` (6 files), `e5_prelim_comparison.md`, `perf_r2/*` (3 files). `results/perf/`, `e3a/` and `e3b/` and all record files are byte-identical.
- **Final gates (actual output):**
  - `ruff check .` passed; `ruff format --check .` 121 files formatted; `mypy src/robocoord` no issues in 61 source files.
  - Fast suite **570 passed, 35 deselected** (550 + 20 new); long suite (`-m "slow or e2e"`) **35 passed**.
  - Coverage total **95 %**; `e5_diagnostics.py` 91 %, `perf.py` 92 %, `allocation_runs.py` 95 %, `allocation_analysis.py` 87 %.
- **Not verified here:** a CI run (needs a push by the student); other hardware; the timing numbers are from one machine and vary between runs (21–29 exceedances).

### V-10 — Project Issue #4 (GitHub issue #8): endpoint recovery and NFR-PERF-1 optimisation (2026-11-19)
- **Trigger:** the student's focused review after V-9:
  - 89 of 91 undelivered baseline E5 tasks were permanent goal blockages, against the model's "transient" wording;
  - NFR-PERF-1 was not met as a per-step bound.

  An investigation (no code changed) found:
  - *Robot–obstacle mutual waits* (61 tasks; every patrol case had another free exit, but its deterministic next step was the robot's cell; 7 random walks on dead-end goals).
  - *Idle robots left on endpoints* (28 tasks; 24 had delivered there).
  - *BFS-field construction dominating* first and map-event steps.
  - *Verdict:* the planner matched the model; the cause was a specification gap (model §5.2's "transient" was an unstated assumption that A-5 falsifies) plus missing parking.

  The student approved robot-side recovery (A-5 unchanged), on-demand parking, default-on for both controllers with `--endpoint off`, workloads R/F, and optimisations O1/O2 first (O3 only if needed).
- **Parameters fixed before any run** (`EndpointParams`): W = 4 = T_stall, H = 8, M = 3, radius 8, abort 4. Not tuned. Development smoke runs used seeds 10000–10019 only.
- **Code:**
  - New `planning/endpoint.py` (parking: nearest safe cell, K1–K4 with a cut-vertex rule; retreat: blockage episodes, W/H/M, outcomes; counters; end-of-episode report).
  - `planning/coordinator.py` composes it after the resolver.
  - `planning/monitor.py`: T8; `RobotInput.trigger` / `count_active`.
  - `planning/deadlock.py`: `Override` carries both.
  - `core/distance.py`: O1 flat BFS with per-version neighbour lists; O2 component reachability; symmetric field reuse; the legacy implementation selectable for tests and the paired benchmark.
  - `allocation/benefits.py`: the leg D(p, d) is read from the pickup's field.
  - `cli.py`: `--endpoint on|off`; recovery statistics line.
  - `experiments/`: `fixed.endpoint` (absent = off, so stored records replay), recovery-aware diagnostics, `compare-recovery`, workloads R/F with paired legacy/fast runs, p90, exceedance counts, a per-step state digest and the power state.
- **Found and fixed during the work:**
  1. The episode can end on the delivery step, so `goal_reached_after_recovery` is also counted in `finish()` (integration test).
  2. The deviation rate is defined per non-idle robot-step, so parking steps and their T3 calls are excluded (`count_active`; unit test).
  3. A robot held by the resolver had its parking restarted and cancelled every step, which double-counted events. Recovery is now not started for robots the resolver overrides (unit test).
  4. Workload F first used RRT\*'s library default budget; set to the E5 n_max = 1 000.

  Runs made before fixes 2–3 were moved out of `results/` to the scratchpad (not used), and every reported run uses the final code.
- **Tests:**
  - New: `test_endpoint.py` (21), `test_endpoint_integration.py` (31), `test_endpoint_evidence.py` (7, 1 slow), `test_distance_equivalence.py` (5), and 17 in `test_cli_matrix.py`.
  - Changed: one assertion in `test_monitor.py` now names the full trigger set including T8 (still an equality). No test was deleted, skipped or weakened, and Hypothesis settings were not reduced.
- **E5 with recovery** (`e5_recovery*.toml`, the same 1 200 paired instances; `05` §B.6):
  - *Baseline:* 91 → 8 undelivered, 83 → 8 incomplete episodes; 75 episodes fixed, 0 regressions. The remaining 8 are 6 never-leaving patrols (all `attempts_exhausted`, reported unresolved), 1 horizon and 1 no-plan.
  - *Reference:* 1 357 → 1 286; 23 fixed, 3 regressions (all free-cell contests).
  - 0 physical collisions; baseline RR/ROs/INV 0; diagnostics replays 0 mismatches (8 + 384).
- **NFR-PERF-1** (original implementation benchmark, `results/perf_r4/`, on AC, legacy and fast interleaved per seed, identical state digests in 12/12 configurations):
  - *Workload R:* max 33.7–35.6 ms → **7.0–9.5 ms**, 40 → 0 steps ≥ 20 ms.
  - *Workload F, A\*:* max ≤ 15.43 ms, 0 exceedances.
  - *Workload F, RRT\*:* still 23–34 % of steps ≥ 20 ms.
  - The intermediate battery run (`perf_r3_R/`, O1/O2 only) had 3/40 first steps ≥ 20 ms (max 24.83 ms).
  - O3 was not needed and not implemented.
- **Overwrite check** (SHA-256 against the 68-file snapshot taken before V-9): changed only `e5_prelim/summary.md` (the intentional V-9 regeneration). Missing: none. New: the V-9 files plus `e5_recovery/*`, `e5_recovery_reference/*`, three comparison files, `perf_r3_R/*` and `perf_r4/*`.
- **Final gates (actual output):**
  - `ruff check .` passed; `ruff format --check .` 126 files formatted; `mypy src/robocoord` no issues in 62 source files.
  - Fast suite **650 passed, 36 deselected**; long suite (`-m "slow or e2e"`) **36 passed**.
  - Coverage total **96 %**; `planning/endpoint.py` 98 %, `core/distance.py` 99 %, `perf.py` 93 %, `e5_diagnostics.py` 92 %.
- **Not verified here:**
  - a CI run;
  - other hardware, and the final code on battery power;
  - a full E5 (learned controllers, shield ablation).

### V-11 — Project Issue #4: independent Windows verification, documentation and audit (2026-11-19)
- **Scope:** documentation, audit and commit planning only. No source file, test, configuration or result was changed in this step. No Git operations.
- **Independent verification** (performed by the student on Windows 11, build 26300; CPython 3.14.3; NumPy 2.4.6; Intel CPU, 12 logical CPUs; AC power, battery 100 % at start and end). The gate results below are as reported by the student; they were not re-run in this step.
  - `ruff check .` passed; `ruff format --check .`: 126 files already formatted; `mypy src/robocoord`: no issues in 62 source files.
  - Fast suite: 650 passed, 36 deselected. Slow/E2E suite: 36 passed, 650 deselected.
  - **Distinct passing tests: 686** (650 + 36). The focused endpoint run, 52 passed in 34.78 s, is a **subset** of those 686, not extra tests. It is `test_endpoint.py` (21) plus `test_endpoint_integration.py` (31); these counts were checked by collection in this step.
  - Benchmark: `python -m robocoord.experiments.perf --out results/perf_verify --workload R --impl legacy,fast`. Workload R, 4 401 measured steps per implementation. Fast maximum 10.56 ms and 0 fast steps at or above 20 ms; legacy 40 steps at or above 20 ms, all first steps. Per-step digests identical in 4/4 configurations; deliveries identical; 0 physical collisions. The figures in `05` §B.6 were read from `results/perf_verify/nfr_perf1.json` and match the student's report.
- **Two evidence sets, kept apart:**
  - the **original implementation benchmark** is `results/perf_r4/` (fast max 9.53 ms);
  - the **independent Windows verification** is `results/perf_verify/` (fast max 10.56 ms).

  They are not merged or averaged. Their p99 values differ (fast p99 up to 1.6× between the runs), so the documentation treats one run as evidence and not as a guarantee.
- **Scope statements added or corrected:**
  - NFR-PERF-1 is an **A\*** requirement; workload R is the A\* requirement workload;
  - RRT\* steps in workload F still exceed 20 ms in 23–34 % of steps, reported as a limitation of RRT\* and not as a failure of the A\* requirement;
  - the bound is established for workload R on AC power on this Windows machine, and not on battery, on other hardware, or for RRT\*.
- **Corrections made while checking the documents against the evidence** (no measured value changed):
  1. Removed "on battery, step times varied 2–4× between consecutive runs": not measured in a controlled way.
  2. Removed "RRT\* cost is RRT\* planning, not distance fields": RRT\* was not profiled after the optimisation.
  3. "each field about 5× cheaper" → "3–6× in the micro-benchmarks (3.0–5.7×)".
  4. Relabelled `results/perf_r4/` as the original implementation benchmark everywhere (it was called "final").
  5. The endpoint test counts in `04` §5.6 are 21 and 31 (plus 7 in the evidence file), matching collection.
- **Inconsistency found and corrected (comment-only):** the header comment of `configs/experiments/e5_prelim.toml` still said the reference-follower half and all learned controllers "remain for Issues #6/#8" and that the slice was "baseline only". It now names the reference half (`e5_prelim_reference.toml`) and the recovery configs (`e5_recovery*.toml`), and states that endpoint recovery is off for this file. The parsed configuration is unchanged (SHA-256 `13b112ae…` before and after).
- **Repository-level dependency pins approved by the student (2026-11-19):** `scipy==1.18.1` (dev extra, test and experiment oracle only) and `hypothesis==6.155.7` (dev extra, Windows Application Control) in `pyproject.toml` and `requirements-lock.txt`. The approval covers the repository change, not only the local install.
- **Preliminary E5 status (unchanged):** the non-learned preliminary evaluation (baseline and reference, with and without endpoint recovery). It is **not** the full E5, which needs the learned controllers and the shield ablation.
- **Offline regression: endpoint off equals the stored pre-recovery records** (evidence outside the repository tests; the script is in the session scratchpad and is not committed). Every 12th stored record of each half (100 of the 1 200 no-recovery records) was replayed with the `endpoint` key absent. Steps, released and delivered tasks, robot moves, conflicts, physical collisions, mean flow time and makespan matched the stored values in all 100 episodes (0 mismatches). The repository test `test_endpoint_off_equals_absent_key_and_adds_no_recovery` checks only that explicit `"off"` equals the absent key, not this comparison.
- **Documentation cleanup (comments and documents only, no algorithm change):** stale T1–T6 and "no deadlock detour / no parking" statements corrected in `control/baseline.py`, `planning/{deadlock,monitor}.py`, `tests/unit/planning/test_monitor.py`, `tests/integration/test_baseline_integration.py`, `docs/{01,02,03}`; FR-PLAN-4 annotated with the Issue #4 triggers without changing the requirement; the E5 run totals (1 200 without recovery, 2 400 in all) made explicit in `docs/05` and `docs/07`; the endpoint-off test renamed and its docstring corrected.
- **Not verified here:** a CI run; other hardware; the final code on battery power.

### V-12 — Issue #5 preparation: optional PyTorch (`marl` extra), CPU-only installation and CI configuration (2026-11-19)
- **Scope:** dependency and CI configuration only. No source file, test, scenario, experiment configuration or result was changed. No Git operations were run by Claude.
- **Approvals (student, 2026-11-19, in chat):** (1) read-only investigation of PyTorch CPU wheel availability; (2) a trial in a disposable virtual environment outside the repository; (3) installation of `torch==2.15.0` CPU into the project `.venv` only; (4) the repository changes below, with a separate lock file for the MARL packages. The local installation and the repository change were approved separately.
- **Environment:** Windows 11 (build 26300), AMD64, CPython 3.14.3, pip 26.3 in the project `.venv`. Project `.venv` versions: numpy 2.4.6, scipy 1.18.1, hypothesis 6.155.7, torch 2.15.0+cpu.
- **Wheel availability** (read-only, `pip install --dry-run`, `--only-binary`):
  - CPU index, Windows: `torch-2.15.0+cpu-cp314-cp314-win_amd64.whl`.
  - CPU index, Linux x86_64: `torch-2.15.0+cpu-cp312/cp313/cp314-…-manylinux_2_28_x86_64.whl`.
  - Metadata `Requires-Python >=3.11`, compatible with `requires-python = ">=3.12,<3.15"`.
  - PyPI only, Linux: the wheel `torch-2.15.0-…-manylinux_2_28_x86_64.whl` (no `+cpu`). The PyPI metadata of torch 2.15.0 lists `cuda-toolkit==13.2.2`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`, `nvidia-cusparselt-cu13`, `cuda-bindings` and `triton` for Linux. This is why the CPU index is required (`03` §3.4).
- **Disposable trial** (virtual environment in the session scratchpad, outside the repository, deleted afterwards; pip 25.3). Installed: torch 2.15.0+cpu, filelock 4.2.12, fsspec 2026.9.0, Jinja2 3.1.6, MarkupSafe 3.0.4, mpmath 1.3.0, networkx 3.7, setuptools 84.2.1, sympy 1.14.0, typing_extensions 4.16.0 (numpy 2.4.6 added afterwards for the interoperability check).
  - `import torch` succeeded (Windows Application Control blocked nothing); `torch.cuda.is_available()` False, `torch.version.cuda` None.
  - Forward and backward pass of a 356→256→256→5 MLP on a batch of 32: output (32, 5), finite loss, non-zero gradients.
  - With `torch.set_num_threads(1)`, `set_num_interop_threads(1)` and `torch.use_deterministic_algorithms(True)`, two identical seeded 20-step Adam runs (Huber loss, gradient clip 10) produced bitwise-identical weights (same machine and process).
  - Warnings: "Failed to initialize NumPy" at import while the venv had no NumPy (expected; absent after numpy 2.4.6 was installed, and `from_numpy`/`.numpy()` then worked); a `UserWarning` from the test script's own `float()` on a tensor with `requires_grad`; a pip "new release available" notice.
  - The project's 159 files (excluding `.venv`, `.git`, caches, `results`) had identical SHA-256 hashes before and after the trial.
- **Installation in the project `.venv`:** `pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.15.0 numpy==2.4.6 scipy==1.18.1 hypothesis==6.155.7`. Nine packages were added (torch 2.15.0+cpu and the eight dependencies listed in `03` §3.4); `typing_extensions` was already present. A before/after `pip list` comparison shows **no existing package changed**. `pip check` reports no broken requirements. The import, CPU-only, NumPy-interoperability, forward/backward and deterministic one-thread checks were repeated in the project `.venv` with the same results and no import-time warnings.
- **Dependency configuration changes:**
  - `pyproject.toml`: new optional extra `marl = ["torch==2.15.0"]`; the base dependency stays `numpy>=2.0`; `dev`, `requires-python` and the numpy, scipy and hypothesis pins unchanged.
  - New `requirements-lock-marl.txt`: `--extra-index-url https://download.pytorch.org/whl/cpu`, `-r requirements-lock.txt`, `torch==2.15.0+cpu` and its eight dependencies.
  - `requirements-lock.txt`: two header comment lines pointing to the MARL lock; no pin changed.
  - `.github/workflows/tests.yml` (`tests` job): installs `.[dev,marl]` with the CPU index, then a new step runs `python -c "import torch; assert torch.version.cuda is None, torch.version.cuda; print(torch.__version__)"`. This **CPU-only guard** fails the job if a CUDA build of torch is installed. The `quality` job is unchanged.
- **Dry-run resolutions** (`pip install --dry-run --ignore-installed --report`, against a copy of the project in the scratchpad, Windows 11 / Python 3.14.3):

  | Command | Result |
  |---|---|
  | `pip install -e .` | 2 packages (numpy 2.5.5, robocoord); no torch, nvidia, cuda or triton |
  | `pip install -r requirements-lock.txt -e .` | 20 packages; numpy 2.4.6, scipy 1.18.1, hypothesis 6.155.7; no torch, nvidia, cuda or triton |
  | `pip install -e ".[dev,marl]" --extra-index-url <CPU index>` (the CI command) | 29 packages; `torch==2.15.0+cpu`; numpy 2.5.5; no nvidia, cuda or triton |
  | `pip install -r requirements-lock-marl.txt -e ".[marl]"` | 29 packages; `torch==2.15.0+cpu`; numpy 2.4.6; no nvidia, cuda or triton |
  | `pip install -e ".[marl]"` without the index (unsupported, shown for contrast) | 12 packages; `torch==2.15.0` from PyPI |

  - Linux, Python 3.12, 3.13 and 3.14: `pip install --dry-run --no-deps --platform manylinux_2_28_x86_64 --python-version <v> --target <scratch>` with the CPU index as extra index selected `torch-2.15.0+cpu-cp<v>-…-manylinux_2_28_x86_64.whl`; without it the plain PyPI wheel. The `.metadata` files of the three Linux CPU wheels were read: their unconditional dependencies are filelock, typing-extensions, setuptools, sympy, networkx, jinja2 and fsspec, with no CUDA, nvidia or triton entry. **These are wheel-selection and metadata checks. Nothing was installed on Linux and no Linux runtime test was run.**
- **Quality gates** (project `.venv`, after the configuration changes): `ruff check .` passed · `ruff format --check .`: 127 files already formatted (V-11 recorded 126; the difference of one file is not explained by a recorded cause; the file list of the current run is given below) · `mypy src/robocoord`: no issues in 62 source files.
  - File list of the 127 (`ruff format --check . --verbose`, ruff 0.17.5, run after the documentation edits of this entry): `format_path` was called for **116 Python files** (the number `ruff check --show-files .` also reports for `.py`) and **11 Markdown files** (the ten files in `docs/` and the root `README.md`). The 126 recorded in V-11 was not accompanied by a file list, so whether it also contained ten Markdown files was not checked; the cause of the difference between 126 and 127 is therefore **not established**.
- **Fast suite** (`pytest -q -p no:cacheprovider --cov=robocoord`): **650 passed, 36 deselected**, 676.72 s (0:11:16), coverage total **96 %** (6 419 statements, 211 missed). An earlier run on the same environment, before the configuration changes and after the torch installation, gave 650 passed, 36 deselected, 283.31 s, 96 %. The difference in duration was not investigated.
- **Long suite:** not re-run. No source or test file changed and no module imports torch yet. The last recorded result is 36 passed (V-11).
- **Git state:** `.git/HEAD` read as `refs/heads/develop`; no Git command was run by Claude.
- **Known limitations / open problems:**
  - The GitHub Actions workflow with the new install and guard steps has **not been run**.
  - Python 3.12 and 3.13 are not installed on the development machine. The full dependency resolution and the tests were **not** run on them. Only the Linux wheel selection and metadata above cover those versions.
  - Runtime compatibility of torch 2.15.0 with the NumPy that CI resolves (2.5.5) was **not tested**. The local trial and installation used NumPy 2.4.6.
  - Linux and Windows-on-3.12/3.13 installation and runtime behaviour were not tested.
  - The slow and e2e suites were not re-run because no source code changed.
  - Torch determinism was shown only as bitwise-identical repeated runs on this machine, with one thread, for one small network. Throughput of simulation plus training is unmeasured; the Double-DQN gather/scatter operations under deterministic mode are untested.

### V-13 — Documentation: experiment terminology and seed blocks for Issues #5 and #6 (2026-11-19)
- **Scope:** documentation only. No code, test, configuration or result was changed. No Git operations were run by Claude.
- **Clarification of historical terminology.** V-8 to V-11, GitHub issue #8 and the stored file and directory names call the non-learned runs of Project Issue #4 "preliminary E5" or "E5". Under the approved eight-issue experiment mapping, E5 and E6 belong exclusively to plan Issue #8. These runs are now called the **Issue #4 system comparison**. The V-8 to V-11 entries are historical records and are not rewritten. The names `e5_prelim*`, `e5_recovery*`, `e5_scenario` and `e5_diagnostics.py` are file, directory and function names, not experiment labels.
- **Seed blocks** (`05` §B.1): the validation range is split into block A (10 000–10 019, periodic validation and checkpoint selection) and block B (10 020–10 099, development comparisons in Issues #5 and #6 after configuration freeze, never used for selection or tuning). Test seeds 20 000–20 999, including 20 200–20 249, are reserved for the official E5 and E6 evaluation in Issue #8. The Issue #4 system comparison already used 20 200–20 249 (and E3b used 20 100–20 129) without tuning on them.
- **Plan changes:** acceptance criterion 3 of Issue #5 now reads "paired held-out development comparisons (validation block B)"; the E4 row in `05` §B.4 states that Issues #5 and #6 evaluate final policies on block B.
- **Decisions recorded for Issue #5 (student, 2026-11-19):** checkpoint selection over the 60 block-A A\* episodes is lexicographic — maximum overall task completion rate, then maximum mean discounted base return (potential-based shaping excluded); both metrics are reported overall and per scenario level. Training root seeds are 1, 2 and 3, with the same scenario-sampling sequence across runs and independent algorithmic random streams.
- **Files changed:** `docs/02`, `docs/03`, `docs/04`, `docs/05`, `docs/07`, this log.

### V-14 — Issue #5 implementation: multi-agent environment, MADQL, training pipeline and learned controller (2026-11-19)
- **Scope:** implementation and verification of Issue #5 as approved by the student on 2026-11-19 (design final; selection rule and run seeds as in V-13). PPO (Issue #6) and E5/E6 (Issue #8) are not part of it. No Git operations were run by Claude. `docs/SESSION_HANDOFF.md` was not modified: SHA-256 `ba4a439c…5fa3` before and after.
- **Environment:** Windows 11 (build 26300), 12 logical CPUs, CPython 3.14.3, NumPy 2.4.6, SciPy 1.18.1, Hypothesis 6.155.7, torch 2.15.0+cpu, ruff 0.17.5.
- **New code:** `src/robocoord/marl/`.
  - NumPy only: `config`, `observation`, `reward`, `global_state`, `scenarios`, `env`, `vector_env`, `replay`.
  - Torch: `networks`, `madql`, `checkpoint`, `policy`, `trainer`.
  - Torch imported lazily: `evaluation` (learned controllers only).
  - `devcompare`.
  - Configs: `configs/training/madql.toml` and `madql_smoke.toml`.
- **Changed code:**
  - `sim/view.py`: optional `reference_paths` and `goal_distance` fields.
  - `sim/coordinator.py`: `PathReporting` protocol.
  - `planning/coordinator.py`: `reference_paths(limit)`.
  - `sim/engine.py`: opt-in `view_extras` and `path_limit`.
  - `cli.py`: `--controller madql`, `--checkpoint`, and the `train` subcommand.
  - `.github/workflows/tests.yml`: the `quality` job installs `.[dev,marl]` with the CPU index, because mypy needs torch's types.
  - Without view extras the engine's traces are byte-identical to before (`tests/integration/test_view_extras.py`, A\* and RRT\*).
- **New tests:**
  - `tests/unit/marl/` (`test_observation.py`, `test_reward.py`, `test_madql.py`, `test_devcompare.py`);
  - `tests/integration/` (`test_view_extras.py`, `test_marl_env.py`, `test_marl_training.py`, `test_marl_learning_sanity.py`, marked `training`);
  - `tests/e2e/test_train_subprocess.py`.

  They cover:
  - `04` §6 tests 1–6 (test 6 is the chain-MDP sanity check plus the `training`-marked test against a random policy);
  - `04` §8 tests;
  - X-1 with the learned controller (A\* and RRT\* × shield off and predictive);
  - X-7 (round-trip, sidecar, tamper check);
  - X-13 (telescoping with β ∈ {0, 0.3, 1} on terminated and truncated episodes; zero potential at termination);
  - truncation look-ahead equivalence and termination precedence;
  - seed-block enforcement.

  No existing test was changed or weakened.
- **Step budget decision (cited in `04` §6):**
  - A 40 000-step trial of the full configuration ran alone on this machine (curriculum fractions as approved, validation off). It took 133.2 s wall time, 26.5 s of which was the final block-A validation, with 9 186 gradient updates.
  - Interval throughput fell from 554 steps/s (stage 1, before learning updates dominate) to 245–327 steps/s in stage 3.
  - With three runs sharing the machine, 1 000 000 environment steps per run was chosen. That is about 2 h per run at the measured rates, including periodic validation.
  - Measured afterwards in the real runs: 183–193 training steps/s per run (validation excluded) and 7 176–7 581 s wall time per run.
  - The trial configuration and output are in the session scratchpad and are not committed.
- **Quality gates** (after all changes): `ruff check .` passed. `ruff format --check .`: 153 files already formatted (Python and Markdown). `mypy`: no issues in 78 source files.
- **Fast suite** (`pytest -q -p no:cacheprovider --cov=robocoord`): **809 passed, 45 deselected**, 274.16 s, total coverage **95 %** (7 719 statements, 307 missed).
  - An earlier run of the same suite, while the three trainings ran in parallel, gave 808 passed, 43 deselected, 699.84 s, 94 %. That was before the two learning-sanity tests and the cp1252 regression test were added.
  - Before Issue #5 the suite gave 650 passed, 96 % (V-12).
  - Per-module coverage of the new torch modules in this run: `trainer` 85 %, `madql` 86 %, `evaluation` 87 %, `devcompare` 82 %, `checkpoint` 96 %, `policy` 98 %. Their long-run paths are exercised by the `slow`, `e2e` and `training` tests rather than by the fast suite.
- **Long suite** (`pytest -p no:cacheprovider -m "slow or e2e or training"`): **45 passed, 809 deselected** in 166.54 s. The learning-sanity test took 33.4 s for run seed 1 and 32.1 s for run seed 2.
- **Defects found and fixed during verification:**
  1. `python -m robocoord.marl.devcompare` exited with code 1 **after** writing every result file. Printing the summary's "−" to a redirected Windows (cp1252) console raised `UnicodeEncodeError`. The fix replaces unencodable characters on console output only. The regression test is `test_main_prints_the_summary_on_a_cp1252_console`.
  2. `docs/03` §5 contained a backspace character in place of `\b` in `$\bar D$`. It was introduced by an Issue #5 documentation edit and is now corrected. The same cause, an escape sequence interpreted by the editing script, had turned the opening backtick and the `r` of `ruff` in the V-12 file-list note into a carriage return. That note is restored to its intended text (`ruff format --check . --verbose`); its content is otherwise unchanged. A scan of 178 text files (`docs`, `src`, `tests`, `configs`, `README.md`, `pyproject.toml`, the MARL lock and the workflow) now finds no control character, tab or lone carriage return.
  3. `ruff format` reformats Python code blocks in Markdown and flagged a comment spacing in `docs/03` §6; corrected.
  4. `docs/03` §3.4 still said the torch-import rule was "not yet implemented"; it now describes the implementation.
- **Not verified here:**
  - a GitHub Actions run, including the changed `quality` job;
  - Linux, and Python 3.12 and 3.13;
  - peak process memory of training (the replay buffer alone is 160.2 MB by `ReplayBuffer.nbytes`);
  - determinism across machines (shown only on this machine with one torch thread).

### V-15 — Issue #5 results: MADQL training (E4, MADQL part) and the block-B development comparison (2026-11-19)
- **Training runs:**
  - Command: `for s in 1 2 3; do python -m robocoord train --config configs/training/madql.toml --seed $s --out results/training/madql_seed$s; done`, run concurrently.
  - All three reached 1 000 000 environment steps. Their logs contain no error or warning.
  - The background shell that launched them reported exit code 1. That came from a trailing `tail` display command in the wrapper, not from a training process: all three `summary.json` files, the checkpoints and the validation logs are complete.
- **Selection verified by recomputation:**
  - For each run, the lexicographic maximum over `validation.jsonl` equals the step recorded in `best.json`: 800 000, 1 000 000 and 700 000.
  - The SHA-256 of each `best.pt` matches its sidecar, and its weights are tensor-identical to the selected step checkpoint. The file bytes differ only through torch's archive serialisation.

  | run seed | selected step | `best.pt` SHA-256 | config hash |
  |---|---|---|---|
  | 1 | 800 000 | `dd9f088c2bfe807844579548e7603751b6041de10c74532d422a40b92f31ce83` | `3ffb73239e1a48d0` |
  | 2 | 1 000 000 | `c66bd8c2d606e70586e639027d5a15987b302e799bfe022e060034d9524f6856` | `59fee33cc03bf89a` |
  | 3 | 700 000 | `492f2a61d23a5ed1bbf0dfd5847c59988d87fed94108ee0890ac7ab67a657258` | `fa1ea5b4509dd8a0` |

  The configuration hashes differ only through `run_seed`, which is part of the hashed configuration.
- **Seed hygiene (checked from the stored records):**
  - Training episodes used seeds 6–9 993.
  - Validation used 10 000–10 019.
  - The development comparison used 10 020–10 099.
  - No seed in 20 000–20 999 appears in any record.
- **Development comparison:**
  - `python -m robocoord.marl.devcompare … --workers 10`, run **once** after selection: 1 500 episodes in 2 109 s.
  - Its process also exited with code 1, after writing `manifest.json`, all 1 500 records, `analysis.json` and `summary.md`. The cause is defect 1 of V-14; the results were not re-run or changed.
- **Results** (details in `05` §B.6, E4 and the Issue #5 development comparison):
  - Block-A selected completion: 0.715, 0.635 and 0.631.
  - On block B, MADQL completes fewer tasks than the baseline (pooled −0.41 with A\*, −0.31 with RRT\*) and than the reference follower (−0.32, −0.21). This is a **negative result** for completion.
  - It proposes 13–16 fewer conflicts per 100 robot-steps than the reference follower.
  - Its per-step conflict rate against the baseline is not clearly different and is confounded by episode length.
  - 0 physical collisions in all training, validation and comparison episodes.
- **Storage:** results (192 MB of training output, plus the comparison) are local and git-ignored. No checkpoint was committed or placed in `models/` (ADR-9; the final selection for `models/` is Issue #6).
