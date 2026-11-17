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
     - the corrected RRT\* radius constant (Solovey et al. 2020) and hence the default $\gamma_f$ (Issue #3);
     - the ε-scaling bound $O(kA\log(kC))$ (Bertsekas & Eckstein 1988) (Issue #4);
     - the k-d tree range-query bound (Lee & Wong 1977; de Berg et al.) and the expected nearest-neighbour result (Friedman, Bentley & Finkel 1977) (Issue #3).
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
