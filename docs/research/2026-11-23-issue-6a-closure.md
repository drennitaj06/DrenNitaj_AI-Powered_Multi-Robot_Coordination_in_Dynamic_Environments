# Issue 6A: closure report (2026-11-23)

Issue 6A, the system-wide algorithm review, optimization and validation, is a **workstream within Issue #6**. It is not a separate GitHub issue. The project keeps exactly eight consolidated issues; Issues #7 and #8 were not touched.

**Evidence base:**
- the phase reports in `docs/research/`;
- the stored analyses in `results/dev6a/` (git-ignored, local);
- the verification review `2026-11-23-issue-6a-review.md`;
- the frozen Phase 6 run (`results/dev6a/phase6/eval/`).

No result in this report is restated from memory. Every number traces to one of these files.

---

## 1. Final outcome

| | Status |
|---|---|
| **Execution Supervisor R + S + D** (Phase 3) | **Confirmed on held-out data** (Phase 6 H-L2 go, A\* and RRT\*). It is part of the forward configuration; the library default stays `off` |
| **MADQL-v2 n3** (Phase 4) | **Not confirmed.** Phase 6 H-v2 FA and FR were rejected by the pre-registered robot–obstacle veto; NA was "adopt". The historical MADQL stays the forward MADQL. MADQL-v2 remains a documented configuration |
| **Integrated Issue 6A system** | **Confirmed as declared** (Phase 6 H-L12 go, both planners) against the Issue #6 system |
| **Forward configuration** | R + S + D (frozen Stage 3a parameters, O off); SCB-C v5 PPO picks; historical MADQL; P1 off; P2 off (full 1 000-iteration RRT\*); A\* unchanged |
| **D12 (checkpoint backups)** | **Not satisfied; waived by the owner** (Deviation W1). No independent backup is verified |

## 2. Phases 0–5 (final status; evidence verified in `2026-11-23-issue-6a-review.md`)

| Phase | Final status | Category | Key evidence |
|---|---|---|---|
| **0: Comprehensive Diagnostics** (6A.0, stored-data diagnostics, Phase 0b replay; H-L1 on Block D) | Complete | Investigation | The 928 block-B failures replayed with 0 mismatches. Phase 0b: 2 160 replays, 0 arg-max mismatches. MADQL's frozen behaviour is a learned STAY preference (median gap +0.044), not tie-breaking. T3 off-path replanning is 95 % (MADQL) and 62–65 % (PPO) of calls. H-L1 (Block D, single use): mappo_seed1–3 have fewer conflicts per episode (Holm); no completion loss detected |
| **3: Execution Supervisor** | Complete; **R + S + D** chosen as the forward configuration (O dropped); off by default in the library | Adopted, then **confirmed in Phase 6 (H-L2 go, both planners)** | Block C 3b / 3e "go": completion MADQL +0.37, IPPO +0.012, MAPPO +0.010 (A\*). 0 safety events. About 4 % step-time overhead |
| **1: Replanning optimization (P1)** | **Rejected** (C1–C4) | Rejected; optional, off (`repair_radius = 0`) | 59–86 % fewer planner calls, but a MADQL completion loss and a high-priority flow-time veto |
| **2: RRT\* budget control (P2)** | **Inconclusive** for the learned policies. The baseline condition's verdict label was "adopt", but `p2_forward_default = false` | Inconclusive; optional, off; the full 1 000-iteration RRT\* budget | RRT\* iterations −10 to −16 % (Holm). IPPO and MAPPO completion intervals unbounded at the ceiling |
| **4: MADQL optimization** | Complete; **MADQL-v2 n3** selected as the candidate (Block C "adopt" in FA / FR / NA, development) | Candidate, then **not confirmed in Phase 6** (H-v2 FA / FR rejected by the robot–obstacle veto; NA adopt) | n-step implementation with historical behaviour reproduced exactly; n3 selected on Block S. Robot–obstacle conflicts per move ×2.17 / ×1.83 / ×1.43 (exact definition, development) |
| **5: A\* optimization (P3)** | **Not pursued: not justified** | Deliberately not pursued | A\* is 6–13 % of episode time; its paths are optimal; no evidence of a bottleneck. A\* unchanged |

**Historical references kept:**
- Issue #5 MADQL;
- the Issue #6 PPO picks;
- the Issue #6 composition (supervisor off, resolver off for learned controllers);
- the baseline in its original composition.

**Options kept, off by default:** P1 repair, P2 stopping rules, the supervisor (library default `off`), and MADQL `n_step` (default 1).

## 3. Phase 6: confirmatory validation

**Design:**
- Held-out Block V: seeds 14 000–14 199; the RRT\* subset 14 000–14 059.
- 22 080 episodes; frozen protocol with Amendment A1 (Block V generated only after the execution authorization) and Deviation W1 (D12 waiver).
- Error-controlled at 0.05 per unit (7 units; global bound 0.302).

**Integrity:**
- The Block V instances are unique and disjoint from A, B, C, D and S.
- 0 errors, 0 collisions and 0 invalid moves; G0 pass.
- Replay 442 / 442 identical; analysis byte-identical.
- Checkpoints and every protected artifact are unchanged.

| Hypothesis | Verdict | Key numbers (Block V) |
|---|---|---|
| H-L2: supervisor vs none | **go / go** | A\* completion: MADQL 0.620 → 0.996; IPPO +0.014; MAPPO +0.011. All 12 stuck-category rows reduced. No RO increase rejected |
| H-v2: MADQL-v2 vs historical | FA **reject**, FR **reject**, NA **adopt** | RO per 1 000 moves FA 0.669 → 1.219 (Holm 0.0018) and FR 0.553 → 0.927 (Holm 0.0004): **veto**. Efficiency gains (flow time −16.5 / −21.1). NA completion 0.620 → 0.860 |
| H-L12: Issue 6A vs Issue #6 system | **go / go** | Completion gains in every family and planner; no veto |

**Context** (pre-registered, non-gating): MADQL-v2's robot–obstacle rate (1.22 per 1 000 moves, A\*) is below the forward PPO policies' rate (1.99).

Details: `2026-11-23-issue-6a-phase6-results.md` and `-tables.md`.

## 4. What Issue 6A does and does not establish

**Established, confirmatory:**
- On this scenario family, the execution supervisor R + S + D makes every learned controller near-complete without retraining. MADQL gains about 6 delivered tasks per episode.
- It does so without a detected safety cost.
- The Issue 6A system as a whole improves on the Issue #6 system.

**Established, development only:**
- P1 is harmful (rejected).
- P2 saves RRT\* work, but its effect on the learned policies is inconclusive.
- The Phase 0 / 4 diagnostic findings.

**Not established:**
- that MADQL-v2 is acceptable: its robot–obstacle hazard per move is ≈ 1.7–1.8× the historical MADQL's under supervision;
- generalization to other maps, sizes or robot counts (E5 / E6 belong to Issue #8);
- any equivalence claim.

## 5. Open items and limitations

1. **D12 is waived.** The checkpoints have no verified independent backup. Verify the owner's backup with `python tools/phase6/backup_manifest.py verify --root <copy>`; D12 is satisfied only by two recorded independent copies.
2. **MADQL-v2's robot–obstacle hazard.** Accepting it, or remedying it (for example a shield ablation or obstacle-aware training), would be a new, separately approved piece of work.
3. **Training-instance disjointness** is not verifiable.
4. **Fixed checkpoints, 3 seeds per configuration, one scenario family.**
5. **The `models/` decision** remains open (no checkpoint was copied).
6. **`docs/SESSION_HANDOFF.md`** was never modified. It is untracked in the working tree; whether to commit it is the owner's choice.

## 6. GitHub Issue #6 comment, and 7. Integration

**Superseded by the final integration checklist:** `2026-11-23-issue-6a-integration.md`. It contains:
- the exact 99-file inventory and the exclusions;
- the six-commit plan;
- the branch `feature/system-review`;
- the PR to `develop` (title, description, acceptance checklist);
- the `develop` → `main` plan;
- the final Issue #6 closure comment.

The earlier draft of these two sections was replaced so that only one version exists.

## 8. Post-run preservation copies (2026-11-23, after the Phase 6 run)

These copies were made **after** the Phase 6 run finished (21:26:05). They **do not change the run's record**:
- **D12 for the Phase 6 run remains NOT SATISFIED, WAIVED** (Deviation W1).
- The frozen Phase 6 files, `d12_waiver.json`, and the absence of a `backup_record.json` are unchanged.

| Copy | Verification against `manifest_sha256.json` (`87cb9ea6…`) | Independence |
|---|---|---|
| `C:\Users\Admin\OneDrive\robocoord-phase6-backup\2026-11-23` | **1 216 / 1 216 files match**; the manifest copy is identical. Before verification every file was an online-only placeholder, i.e. already uploaded. Every file was then read back from the cloud and matched | a **cloud** copy, independent of the internal disk. **Account:** the folder belongs to the sync root of the **personal** Microsoft account, the only account signed in to the OneDrive client. The owner described it as the university OneDrive; a university-account copy cannot be verified from this machine |
| `D:\robocoord-phase6-backup\2026-11-23` | **1 216 / 1 216 files match**; the manifest copy is identical | **NOT independent.** D: is partition 5 of physical disk 0, the internal NVMe SSD, the same disk as C: (partition 3) that holds the originals |
| Originals (`.`) | 1 216 / 1 216 | — |

**Record:** `results/dev6a/phase6/backup/post_run_preservation.json`. This is deliberately not a `backup_record.json`, so it can never be read as a D12 destination.

**Status today:** one verified independent (cloud) copy plus one same-disk copy. This is short of D12's two independent copies, and it applies only to preservation after the run.
