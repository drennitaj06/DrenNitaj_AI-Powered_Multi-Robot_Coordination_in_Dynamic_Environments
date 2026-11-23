# Issue 6A: GitHub integration procedure (authoritative, 2026-11-23)

*For the owner, who performs every Git and GitHub step manually (GitHub Desktop and GitHub.com). No Git command and no GitHub action was run to prepare this document.*

**How the facts were established:**
- The repository's `.git` files (references, reflogs, loose commit and tree objects) were read as plain files with Python.
- Every working-tree file was hashed the way Git hashes blobs, then compared with the tree of the current commit.

---

## 0. Verified repository state

| Item | Value (verified) |
|---|---|
| Current branch | `feature/system-review` (created from `develop`; reflog: "branch: Created from develop") |
| `feature/system-review` = `develop` = `origin/develop` | `42515c3` "Merge pull request #14 from drennitaj06/feature/marl-ppo". The feature branch has **no commits yet** |
| `origin/main` | `7885906` "Merge pull request #10 from drennitaj06/develop", the post-Issue #4 checkpoint. There is no local `main` branch |
| Changes vs `42515c3` | **11 modified + 95 new = 106 files**, 0 deleted. Another 197 files differ only in CRLF vs LF line endings; Git normalizes them and they are not changes |
| Excluded from the 106 | `docs/SESSION_HANDOFF.md` (shown by GitHub Desktop; must stay unchecked) and 6 `src/robocoord.egg-info/*` files (matched by `.gitignore` `*.egg-info/`, so GitHub Desktop does not show them) |
| **Integration set** | 106 − 1 − 6 = **99 files**. GitHub Desktop shows **100** = these 99 + `docs/SESSION_HANDOFF.md` |
| Ignored and absent from GitHub Desktop | `results/` (except the already-tracked, unchanged `results/figures/overview/`), `.venv/`, caches, `.coverage`, `*.egg-info/`. `models/` does not exist |

**GitHub numbering:**
- Issues and PRs share one counter on GitHub. Project Issue #4 is GitHub issue #8 (`docs/05`, `docs/07`). Issue #5 merged as PR #12; Issue #6 merged as PR #14.
- **The GitHub issue number of project Issue 6 ("PPO (MAPPO/IPPO) and comparative learning study") is not recorded anywhere in the repository.** This document writes it as **`#<ISSUE6>`**.
  - Find it on GitHub.com → *Issues* (open and closed) → search "PPO (MAPPO/IPPO) and comparative learning study". Replace `#<ISSUE6>` before posting.
- The two new PRs get their numbers when they are created. They are written as **`#<PR-A>`** (feature → `develop`) and **`#<PR-B>`** (`develop` → `main`).

---

## 1. The 99-file inventory (no duplicates; every path verified in the working tree)

| Group | Files |
|---|---|
| A. Library source (9 modified, 1 new) | 10 |
| B. Library tests (new) | 8 |
| C. Model and algorithm docs (modified) | 2 |
| D. Research documents (new) | 21 |
| E. Phase 0 diagnostics and SCB-C selection tools (new) | 18 |
| F. Tool tests for E, plus `tests/tools/__init__.py` (new) | 9 |
| G. Phase 1–4 tools (new) | 20 |
| H. Tool tests for G (new) | 4 |
| I. Phase 6 tools (new) | 6 |
| J. Tool test for I (new) | 1 |
| **Total** | **99** |

The full paths are listed per commit in §2.

---

## 2. Commit plan: six commits, in this order

**Before each commit,** in GitHub Desktop's *Changes* list:
1. Untick everything (the checkbox at the top of the list).
2. Tick exactly the files listed for that commit, then check the count shown above the list.
3. **Never tick `docs/SESSION_HANDOFF.md`.**

**Message rules:** paste the *Summary* exactly. The *Description* is optional and can be pasted as shown. No AI attribution line.

The commits are ordered so that each one passes CI on its own:
- the library first;
- the documents before the tools (the tool test `test_block_d.py` lists frozen document paths);
- the tools in dependency order.

### Commit 1: library source and library tests (18 files)

**Summary:**
```
feat(issue-6a): add optional execution supervisor, replan repair, RRT* stopping rules and n-step MADQL
```
**Description:**
```
All options are off by default and byte-identical when off: EvalTask.supervisor="off",
MonitorParams.repair_radius=0, RRTStarConfig.refine_iterations=patience=None,
MADQLParams.n_step=1 (the Issue #5 configuration hash is unchanged).
```
**Files (18):**
```
src/robocoord/control/supervisor.py
src/robocoord/marl/evaluation.py
src/robocoord/marl/madql.py
src/robocoord/marl/replay.py
src/robocoord/marl/trainer.py
src/robocoord/planning/coordinator.py
src/robocoord/planning/monitor.py
src/robocoord/planning/rrtstar.py
src/robocoord/sim/coordinator.py
src/robocoord/sim/engine.py
tests/unit/control/test_supervisor.py
tests/unit/marl/test_madql_nstep.py
tests/unit/planning/test_monitor_repair.py
tests/unit/planning/test_rrtstar_budget.py
tests/integration/test_madql_nstep_training.py
tests/integration/test_rrt_budget_integration.py
tests/integration/test_supervisor_integration.py
tests/integration/test_supervisor_regression.py
```
**Why together:**
- This is every change to the installable `robocoord` package with the tests that cover it.
- The source files cannot be split by feature without hunk-level staging: `evaluation.py` carries the supervisor, repair and RRT\* options together, and `engine.py` and the planning files are shared.
- The tests that read `results/` skip when it is absent, as it is in CI.

### Commit 2: model and algorithm documentation (2 files)

**Summary:**
```
docs(model): document the execution supervisor, replan repair, RRT* stopping and n-step options
```
**Files (2):**
```
docs/02-mathematical-model.md
docs/04-algorithms.md
```
**Why together:** the specification documents that describe the library changes of commit 1.

### Commit 3: research documents (21 files)

**Summary:**
```
docs(research): add Issue 6A proposal, protocols, phase reports and Phase 6 confirmatory results
```
**Description:**
```
Proposal (§20–§36), Block C/D protocols, Phase 3 design and Stage 3a freeze, Phase 1–4 reports
and tables, Phase 6 readiness, decision review, frozen protocol (Amendment A1, Deviation W1),
results and tables, review, closure and this integration procedure.
```
**Files (21):**
```
docs/research/2026-11-20-issue-6a-proposal.md
docs/research/2026-11-21-block-c-analysis-plan.md
docs/research/2026-11-21-block-d-protocol.md
docs/research/2026-11-21-issue-6a-phase3-execution-supervisor-design.md
docs/research/2026-11-21-issue-6a-stage3a-calibration-freeze.md
docs/research/2026-11-22-issue-6a-phase1-replanning.md
docs/research/2026-11-22-issue-6a-phase1-replanning-tables.md
docs/research/2026-11-22-issue-6a-phase2-rrt-budget.md
docs/research/2026-11-22-issue-6a-phase2-rrt-budget-tables.md
docs/research/2026-11-22-issue-6a-phase3-evaluation.md
docs/research/2026-11-22-issue-6a-phase3-evaluation-tables.md
docs/research/2026-11-22-issue-6a-phase4-madql.md
docs/research/2026-11-22-issue-6a-phase4-madql-tables.md
docs/research/2026-11-23-issue-6a-phase6-readiness.md
docs/research/2026-11-23-issue-6a-phase6-decision-review.md
docs/research/2026-11-23-issue-6a-phase6-protocol.md
docs/research/2026-11-23-issue-6a-phase6-results.md
docs/research/2026-11-23-issue-6a-phase6-tables.md
docs/research/2026-11-23-issue-6a-review.md
docs/research/2026-11-23-issue-6a-closure.md
docs/research/2026-11-23-issue-6a-integration.md
```
**Why together:** the complete research record of the workstream, from proposal to closure. It contains no code. The Block D and Phase 6 protocols are referenced by the frozen tools committed later.

### Commit 4: Phase 0 diagnostics and SCB-C selection tools, with their tests (27 files)

**Summary:**
```
exp(issue-6a): add Phase 0 diagnostics and SCB-C checkpoint-selection tools
```
**Files (27):**
```
tools/diagnostics/cluster_reanalysis.py
tools/diagnostics/phase0_stored.py
tools/diagnostics/phase0b_replay.py
tools/diagnostics/replay_failures.py
tools/diagnostics/stage3a_calibration.py
tools/diagnostics/summarise_replay.py
tools/selection/block_c.py
tools/selection/block_c_analysis.py
tools/selection/block_d.py
tools/selection/block_d_analysis.py
tools/selection/calibrate.py
tools/selection/combined_power.py
tools/selection/null_study.py
tools/selection/rule_compare.py
tools/selection/scbc.py
tools/selection/scbc_frozen.py
tools/selection/selection_block.py
tools/selection/simulate.py
tests/tools/__init__.py
tests/tools/test_block_c.py
tests/tools/test_block_d.py
tests/tools/test_followup.py
tests/tools/test_phase0_stored.py
tests/tools/test_phase0b_replay.py
tests/tools/test_scbc.py
tests/tools/test_scbc_frozen.py
tests/tools/test_stage3a_calibration.py
```
**Why together:**
- The foundation tools: the Phase 0 diagnostics, the frozen Block C/D statistics and runners, and SCB-C selection.
- No dependency on any phase tool.
- `tests/tools/__init__.py` creates the tool-test package.

### Commit 5: Phase 1–4 evaluation tools, with their tests (24 files)

**Summary:**
```
exp(issue-6a): add Phase 1-4 evaluation tools (supervisor, replanning, RRT* budget, MADQL-v2)
```
**Files (24):**
```
tools/phase1/phase1.py
tools/phase1/phase1_analysis.py
tools/phase1/phase1_report.py
tools/phase2/micro.py
tools/phase2/micro_report.py
tools/phase2/phase2.py
tools/phase2/phase2_analysis.py
tools/phase2/phase2_report.py
tools/phase3/observer.py
tools/phase3/phase3.py
tools/phase3/phase3_analysis.py
tools/phase3/phase3_report.py
tools/phase4/diagnose.py
tools/phase4/phase4.py
tools/phase4/phase4_analysis.py
tools/phase4/phase4_report.py
tools/phase4/posthoc_conflicts.py
tools/phase4/repro.py
tools/phase4/select_candidate.py
tools/phase4/train.py
tests/tools/test_phase1.py
tests/tools/test_phase2.py
tests/tools/test_phase3.py
tests/tools/test_phase4.py
```
**Why together:**
- The development-block evaluation tools of Phases 3, 1, 2 and 4.
- They import each other: Phase 1 builds on Phase 3, Phase 2 on Phase 1, and Phase 4 on Phases 1–3. They also import the commit-4 selection tools.
- None imports Phase 6.

### Commit 6: Phase 6 confirmatory tools, with their test (7 files)

**Summary:**
```
exp(issue-6a): add the Phase 6 confirmatory runner, analysis, reporting and backup-manifest tools
```
**Description:**
```
Gated runner (freeze / pre-run check / execution authorizations; Block V generated only after the
execution authorization, Amendment A1), observer with robot-obstacle conflicts per move,
pre-registered analysis and report generator, checkpoint backup manifest, and the development-data
robot-obstacle analyses.
```
**Files (7):**
```
tools/phase6/backup_manifest.py
tools/phase6/dev_ro_analysis.py
tools/phase6/dev_ro_exact.py
tools/phase6/phase6.py
tools/phase6/phase6_analysis.py
tools/phase6/phase6_report.py
tests/tools/test_phase6.py
```
**Why together:** the Phase 6 runner and its analysis depend on every earlier tool, and its test covers all six files.

**Reconciliation:** 18 + 2 + 21 + 27 + 24 + 7 = **99**. After commit 6, GitHub Desktop's *Changes* list must show exactly **1** file, `docs/SESSION_HANDOFF.md`, and it stays uncommitted.

---

## 3. PR A: `feature/system-review` → `develop`

- **Base:** `develop` · **Compare:** `feature/system-review`.
- **Title:** `Issue 6A (workstream of Issue 6): system review, optimization and confirmatory validation`
- **Description** (replace `#<ISSUE6>`):

```markdown
**Issue 6A — the system-wide algorithm review, optimization and validation workstream of Issue 6 (#<ISSUE6>).**
Issue 6A is part of the existing Issue 6, not a separate (ninth) issue; the project keeps its eight
consolidated issues, and Issues 7 and 8 are untouched.

Part of #<ISSUE6> (the issue is closed after the `develop` → `main` checkpoint merge).

## Implementation (all options off by default; byte-identical when off)
- **Execution Supervisor** (`src/robocoord/control/supervisor.py`): right-of-way (R), stall breaker (S),
  oscillation recovery (O); an engine stage before the shield; used with the deadlock resolver (D).
- **Replan repair (P1)** — `MonitorParams.repair_radius` (0 = off).
- **RRT\* stopping rules (P2)** — `RRTStarConfig.refine_iterations` / `patience` (None = off).
- **n-step MADQL** — `MADQLParams.n_step` (1 = the Issue #5 one-step target; config hash unchanged).
- `EvalTask` fields `supervisor`, `deadlock`, `repair`, `rrt_refine`, `rrt_patience`.
- Model and algorithm docs (`docs/02`, `docs/04`) updated.

## Experimental work (tools in `tools/`, tests in `tests/tools/`, records in `docs/research/`)
| Phase | Outcome |
|---|---|
| 0 Comprehensive diagnostics (incl. H-L1 on Block D) | complete |
| 3 Execution Supervisor | R+S+D selected (O dropped) → **confirmed** in Phase 6 |
| 1 Replanning optimization (P1) | **rejected** (completion loss and high-priority flow-time veto) — stays off |
| 2 RRT\* budget control (P2) | **inconclusive** — stays off; full 1 000-iteration RRT\* |
| 4 MADQL optimization (MADQL-v2, n = 3) | candidate → **not confirmed** in Phase 6 |
| 5 A\* optimization | **not pursued** (no measurable benefit) |
| 6 Integrated confirmatory validation | **complete** |

**Phase 6** (held-out Block V, seeds 14 000–14 199 / RRT\* 14 000–14 059, 22 080 episodes, frozen protocol):
- H-L2 supervisor R+S+D vs none — **go / go → confirmed** (A\* completion MADQL 0.620 → 0.996, IPPO +0.014, MAPPO +0.011; no robot–obstacle increase).
- H-v2 MADQL-v2 vs historical MADQL — FA **reject**, FR **reject** (pre-registered robot–obstacle veto: +0.55 / +0.37 conflicts per 1 000 moves), NA adopt → **not confirmed**; the historical MADQL remains the forward MADQL.
- H-L12 Issue 6A system vs Issue #6 system — **go / go → confirmed** (as declared).
- Integrity: Block V unique and disjoint from all earlier blocks; 0 errors, 0 collisions, 0 invalid moves; G0 passed; replay 442/442 identical; analysis byte-identical.
- **Deviation:** the D12 checkpoint-backup prerequisite was waived by the owner for the Phase 6 run (**not satisfied**); post-run preservation copies are documented in the closure report.

**Forward configuration:** Execution Supervisor R+S+D (frozen Stage 3a parameters, O off), SCB-C v5 PPO picks,
historical MADQL, P1 and P2 off, full RRT\* budget. Library defaults are unchanged.

## Reports
- `docs/research/2026-11-23-issue-6a-phase6-results.md` (tables: `docs/research/2026-11-23-issue-6a-phase6-tables.md`)
- `docs/research/2026-11-23-issue-6a-closure.md`
- `docs/research/2026-11-20-issue-6a-proposal.md` (§20–§36)

## Testing
- Local (Windows 11, Python 3.14): fast suite **1 292 passed, 1 skipped** (POSIX-only), 0 failed;
  `ruff check .`, `ruff format --check .`, `mypy` clean.
- CI: quality job and fast tests on Ubuntu 3.12 / 3.13 / 3.14 and Windows 3.14 — see the checks below.
- Generated data and checkpoints (`results/`) are not committed; the reports reference them by path and SHA-256.
```

**Acceptance checklist** (tick on GitHub before merging):
- [ ] *Files changed* shows **99** files, consisting of 11 modified and 88 added. There are no `results/`, `models/`, `*.egg-info/` or `docs/SESSION_HANDOFF.md` files.
- [ ] *Commits* shows the **6** commits of §2, in order.
- [ ] CI is green: **lint, format, types**, plus **fast tests** on ubuntu 3.12, ubuntu 3.13, ubuntu 3.14 and windows 3.14.
- [ ] Defaults are unchanged in the diff: `supervisor: str = "off"`, `repair_radius: int = 0`, `refine_iterations` / `patience: … = None`, `n_step: int = 1`.
- [ ] No PPO, A\*, configuration, allocation or CLI file is in the diff (`ppo.py`, `ppo_trainer.py`, `astar.py`, `configs/`, `cli.py` are absent).
- [ ] The reported outcomes match `2026-11-23-issue-6a-closure.md`. The supervisor is confirmed; MADQL-v2 is **not** confirmed; P1 is rejected; P2 is inconclusive; Phase 5 was not pursued; D12 was waived.
- [ ] The PR says "Part of #<ISSUE6>" and nothing references a ninth issue.

**Merge method:** "Create a merge commit", as for PRs #12 and #14.

---

## 4. PR B: `develop` → `main` (the Issue #6 checkpoint)

**Verified content of this PR** (from the commit and tree objects):
- `main` (`7885906`) is the post-Issue #4 checkpoint. Its tree equals the tree of develop's commit `cf555f7` (PR #9), so **`main` contains no content that `develop` lacks**, and the merge will be clean.
- `develop` contains **16 commits not in `main`**: Issue #5 (MADQL, PR #12) and Issue #6 (PPO, comparative study, post-Issue-6 analysis, PR #14). After PR A it also contains the 6 Issue 6A commits.
- The current file difference `main` → `develop` is **101 files** (86 added, 15 modified, 0 deleted), before Issue 6A.
  - It covers the `marl` package, PPO, training configs, tests, lock files, the CI workflow, the docs, `tools/figures` and the 30 tracked overview figures in `results/figures/overview/`.
- **No Issue #7 or #8 content:** no Kaggle, data-driven scenario or visualisation work. **The merge is safe.**
- `docs/07` §3 defines the checkpoint "After #6: Both MARL algorithms implemented, trained and compared (E4, development comparison)". This PR is that checkpoint, extended by the Issue 6A workstream.

**Before opening PR B:**
1. PR A is merged. In GitHub Desktop, switch to `develop`, then *Fetch origin* / *Pull origin*.
2. CI on the `develop` merge commit is green.
3. *Recommended:* locally run the long suite, `pytest -m "slow or training or e2e"`, and record the result in the PR description. Locally means the project `.venv`, with `PYTHONPATH` not needed for these tests.
4. `models/` is still undecided. Issue #6 lists "candidate `models/` files" as a deliverable, and acceptance criterion 5 is "Committed checkpoints (**if any**) … approved by the student". No checkpoint is committed, which is permitted. Record the deferral in the PR, as below.

- **Base:** `main` · **Compare:** `develop`.
- **Title:** `Checkpoint: merge develop into main after Issue 6 (MARL algorithms and Issue 6A)`
- **Description** (replace `#<ISSUE6>` and `#<PR-A>`):

```markdown
Stable checkpoint after Issue 6 (`docs/07` §3: "Both MARL algorithms implemented, trained and compared").

Includes, since the previous checkpoint (PR #10, after Issue 4):
- Issue 5 — multi-agent RL environment and MADQL (PR #12)
- Issue 6 — PPO (MAPPO/IPPO) and the comparative learning study (PR #14) — #<ISSUE6>
- Issue 6A — the review, optimization and confirmatory-validation workstream of Issue 6 (#<PR-A>);
  not a separate issue

No Issue 7 or Issue 8 work is included.

**Forward configuration after Issue 6A:** Execution Supervisor R+S+D (frozen Stage 3a parameters, O off),
SCB-C v5 PPO picks, historical MADQL, P1/P2 off, full 1 000-iteration RRT\*. All new options are off by default.
MADQL-v2 (n = 3) was not confirmed (Phase 6 robot–obstacle veto) and remains a documented configuration.

**Open item, deferred:** no checkpoints are committed to `models/` (Issue 6 acceptance criterion 5 allows "if any";
the decision remains with the student).

Reports: `docs/research/2026-11-23-issue-6a-closure.md`, `docs/research/2026-11-23-issue-6a-phase6-results.md`.
```

**Release checklist:**
- [ ] *Commits* lists only PR #12, PR #14 and PR #A content, plus their commits. **No** Issue #7 or #8 branch or commit appears.
- [ ] *Files changed* contains no `kaggle`, scenario-ingestion or visualisation work, no `models/`, no `docs/SESSION_HANDOFF.md`, and no `results/` other than `results/figures/overview/`.
- [ ] GitHub reports no merge conflicts (expected: `main` has no unique content).
- [ ] All CI checks pass on the PR.
- [ ] The long suite has been run locally, or is explicitly noted as not run.
- [ ] Merge with "Create a merge commit", as for PR #10.

**After the merge:**
1. In GitHub Desktop, switch to `develop` → *Pull origin*.
2. Optionally create a local `main` from `origin/main` to check it.
3. Post the closure comment of §5 on `#<ISSUE6>`.

---

## 5. Final Issue 6 comment (post after PR A and PR B are merged)

Replace `#<PR-A>` and `#<PR-B>`. Post it on `#<ISSUE6>`, then close the issue if it is still open.

```markdown
### Issue 6A — complete and integrated (workstream of this issue)

Issue 6A, the system-wide review, optimization and confirmatory-validation workstream, was carried out
**within Issue 6** (this issue). It is **not** a separate, ninth issue: the project keeps its eight
consolidated issues, and Issues 7 and 8 are unchanged.

**Integration:** `feature/system-review` → `develop` in #<PR-A>; `develop` → `main` (Issue 6 checkpoint) in #<PR-B>.

**Confirmatory result** — Phase 6, held-out Block V (seeds 14 000–14 199; RRT\* 14 000–14 059), 22 080 episodes,
frozen protocol:
- **Execution Supervisor R+S+D — confirmed** (A\* and RRT\*): completion MADQL 0.620 → 0.996,
  IPPO +0.014, MAPPO +0.011 (A\*); all targeted stuck-robot classes reduced; no robot–obstacle increase detected.
- **MADQL-v2 (n-step, n = 3) — not confirmed:** with the supervisor it raises robot–obstacle conflicts per
  1 000 moves (A\* 0.67 → 1.22, RRT\* 0.55 → 0.93), triggering the pre-registered veto despite its efficiency
  gains. The **historical MADQL remains** the forward MADQL; MADQL-v2 stays a documented configuration.
- **Integrated Issue 6A system vs the Issue #6 system — confirmed** (both planners).
- Development phases: replan repair (P1) **rejected**; RRT\* stopping (P2) **inconclusive** (off; full
  1 000-iteration budget); A\* heuristic (Phase 5) **not pursued**.

**Forward configuration:** Execution Supervisor R+S+D (frozen Stage 3a parameters, O off), SCB-C v5 PPO picks,
historical MADQL, P1/P2 off. All new options are off by default in the library.

**Integrity:** Block V unique and disjoint from all earlier blocks; 0 errors, 0 collisions, 0 invalid moves;
G0 passed; replay 442/442 identical; analysis byte-identical; frozen Block C/D artifacts, historical results and
checkpoints unchanged.
**Deviation (disclosed):** the checkpoint-backup prerequisite (D12) was **waived by the owner** for the Phase 6
run and is **not satisfied**. After the run, one cloud copy (verified 1 216/1 216 files) and one same-disk copy
were made; they are documented as preservation copies, not as D12 backups.

**Reports**
- Phase 6 results: `docs/research/2026-11-23-issue-6a-phase6-results.md` (tables: `docs/research/2026-11-23-issue-6a-phase6-tables.md`)
- Closure: `docs/research/2026-11-23-issue-6a-closure.md`
- Proposal and phase history: `docs/research/2026-11-20-issue-6a-proposal.md` (§20–§36)
- Frozen protocol: `docs/research/2026-11-23-issue-6a-phase6-protocol.md`

The Issue 6A workstream is complete and formally integrated into `develop` and `main`.
```

---

## 6. Step-by-step procedure

**GitHub Desktop: commits** (current branch `feature/system-review`):
1. Confirm *Current Branch* = `feature/system-review` and that *Changes* shows **100** files.
2. For commits 1 → 6 (§2):
   - untick all, tick exactly the listed files, and check the count;
   - paste the *Summary* (and optionally the *Description*);
   - click *Commit to feature/system-review*.
3. After commit 6, *Changes* shows only `docs/SESSION_HANDOFF.md`. Leave it unchecked.
4. *History* shows the 6 new commits on top of `Merge pull request #14 …`.
5. *Push origin*. The branch already exists on `origin` at `42515c3`, so this is a normal push.

**GitHub.com: PR A:**

6. *Pull requests → New* → base `develop` ← compare `feature/system-review`. Paste the title and description of §3, replacing `#<ISSUE6>`. Create the PR and note its number as `#<PR-A>`.
7. Wait for CI. Work through the §3 checklist, review, then *Create a merge commit* → confirm.

**GitHub Desktop:**

8. Switch to `develop` → *Fetch origin* → *Pull origin*. Confirm that `develop` now contains the 6 Issue 6A commits and the PR A merge commit.

**Before PR B:**

9. Wait for CI on `develop` to pass. Run the long suite locally if possible. Work through the pre-PR-B items of §4.

**GitHub.com: PR B:**

10. *New pull request* → base `main` ← compare `develop`. Paste the title and description of §4, replacing the placeholders. Create the PR and note `#<PR-B>`.
11. Work through the §4 release checklist and wait for CI. *Create a merge commit* → confirm.

**GitHub.com: Issue 6:**

12. Open `#<ISSUE6>`, post the §5 comment with both PR numbers filled in, and close the issue if it is still open.

**GitHub Desktop:**

13. Switch to `develop` → *Pull origin*. Optionally delete the local `feature/system-review` after the merge. The remote branch can be kept, as earlier feature branches were.

---

## 7. Pre-merge safety checklist (both PRs)

- [ ] **Nothing excluded is committed:**
  - `results/` (except the existing, unchanged `results/figures/overview/`);
  - `models/`;
  - `docs/SESSION_HANDOFF.md`;
  - `*.egg-info/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.hypothesis/`, `.coverage`;
  - the backup folders (`D:\robocoord-phase6-backup`, `OneDrive\robocoord-phase6-backup`).
- [ ] **Exactly 99 files** in PR A; the six commits as specified.
- [ ] **Defaults off** (supervisor, P1, P2, n-step) and PPO / A\* / configs / allocation / CLI unchanged.
- [ ] **CI green** on every required check (the quality job, and fast tests on the 4 matrix entries).
- [ ] **No Issue #7 or #8 work** in either PR, and no ninth issue referenced.
- [ ] **Frozen artifacts untouched.** The Phase 6 freeze record and outputs live in the git-ignored `results/` and are not part of any commit. Do not "fix" or regenerate them.
- [ ] The **GitHub issue number** for Issue 6 has been looked up and substituted for `#<ISSUE6>` in all three texts.
- [ ] **D12** is stated as waived / not satisfied wherever backups are mentioned.

---

## 8. Discrepancies with the previous checklist (corrected here)

1. **Counts.** "100 in GitHub Desktop / 99 to integrate" is **confirmed** by the blob-hash comparison. The 6 `src/robocoord.egg-info/*` files are untracked as well, but git-ignored, so GitHub Desktop does not show them. They are now listed explicitly as excluded.
2. **Group 4 and 5 composition.** Unchanged (27 and 24), but now listed path by path. `tests/tools/__init__.py` belongs to commit 4.
3. **Commit order.** The research documents (commit 3) are placed **before** the tools (commits 4–6), unlike the earlier draft, because the tool tests reference frozen protocol paths.
4. **Issue number.** The earlier text wrote "Issue #6" as if it were the GitHub number. The GitHub number of project Issue 6 is **not recorded** in the repository (issues and PRs share one counter, and project Issue 4 = GitHub #8). Placeholders `#<ISSUE6>`, `#<PR-A>` and `#<PR-B>` are used, with lookup instructions.
5. **Target and checkpoint.** `develop` → `main` is now specified as the `docs/07` §3 checkpoint "after #6". Its content was verified: Issue #5, Issue #6 and Issue 6A only; `main` has no unique content.
6. **`models/`.** Recorded as a deferred open item. It is permitted by Issue 6's acceptance criterion 5 ("if any").
