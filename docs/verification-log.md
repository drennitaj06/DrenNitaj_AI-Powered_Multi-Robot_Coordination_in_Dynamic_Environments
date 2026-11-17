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
