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
