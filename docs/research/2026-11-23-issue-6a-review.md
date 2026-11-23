# Issue 6A: comprehensive review before the Phase 6 gates (2026-11-23)

*Scope:* Part A (Phases 0–5 and the integrity constraints) and Part B1–B2 (the Phase 6 design and implementation), checked against the repository, the stored result files and the integrity manifests, not against summaries.
- No Block V scenario was generated.
- Nothing was frozen, executed or committed.
- No Git command was run.

**Verification labels:**
- **[V]** re-verified in this review, from the stored artifact or the code;
- **[D]** taken from the phase report and consistent with its artifact, but not re-derived here;
- **[!]** an issue.

---

## 1. Issue 6A at a glance

Issue 6A is the system-wide algorithm review and optimization **workstream within Issue #6**. The project keeps exactly eight consolidated issues; see the inconsistency noted in §5, item 2.

| Phase (roadmap order) | Purpose | Final status | Category |
|---|---|---|---|
| **Phase 0: Comprehensive Diagnostics** (6A.0, stored-data diagnostics, Phase 0b replay; plus H-L1 on Block D) | Find where failures and cost come from before changing anything | **Complete** (§20, §26) | Investigation |
| **Phase 3: Execution Supervisor** | An execution-time mechanism for frozen, oscillating and conflict-loop robots | **Complete.** R + S + D adopted as the *forward configuration*; O dropped; development evidence only | Adopted (pending Phase 6 confirmation) |
| **Phase 1: Replanning optimization (P1)** | Bounded repair before T3 replanning | **Rejected** in all four conditions | Rejected; optional, off by default |
| **Phase 2: RRT\* budget control (P2)** | Early stopping of RRT\* refinement | **Inconclusive** for the learned policies (the baseline condition's verdict label was "adopt"; §9.2). **Not adopted:** P2 is off; the forward RRT\* budget is the full 1 000 iterations | Inconclusive; optional, off by default |
| **Phase 4: MADQL optimization (MADQL-v2)** | Fix MADQL's value-estimation weakness by retraining | **Complete.** MADQL-v2 n3 is the *candidate* forward MADQL; "adopt" on Block C (development) | Candidate, to be confirmed in Phase 6 |
| **Phase 5: A\* optimization (P3)** | Exact static heuristic | **Not pursued: not justified** (§34) | Deliberately not pursued |
| **Phase 6: Integrated confirmatory validation** | Independent held-out confirmation of H-L2, H-v2, H-L12 | **Prepared, not executed.** Protocol approved; implementation tested; waiting at the D12 backup gate | Pending |

**Historical references, kept unchanged:**
- the Issue #5 MADQL (`results/training/madql_seed1–3/best.pt`, `n_step` = 1);
- the Issue #6 PPO picks (H-L1 compared them with SCB-C v5 on Block D);
- the Issue #6 composition (supervisor off, resolver off for learned controllers);
- the baseline in its original composition.

---

## 2. Phase-by-phase verification (A1, A2)

### Phase 0: diagnostics and Block D

| Claim | Evidence | Check |
|---|---|---|
| The frozen Block D manifest is intact | `results/dev6a/block_d/plan_freeze.json`: all 35 hashed entries match; Block C: 3 / 3 | [V] |
| The Block D replay is identical | `block_d/replay.json`: 137 / 137 identical | [V] |
| H-L1 on Block D | `block_d/analysis.json` F1. mappo_seed1: completion +0.0049 (Holm-rejected, MC-sensitive). Conflicts per episode lower for mappo_seed1–3 (Holm). ippo_seed1: conflicts +7.2, not rejected, MC-sensitive. No other completion change detected | [V] |
| The 6A.0 replay of all block-B greedy failures is exact | `6a0_diagnostics/replay_summary.json`: 928 episodes replayed, 0 mismatches (MADQL 710, IPPO 118, MAPPO 100 failed episodes) | [V] |
| The Phase 0b replay is exact, and the Q-values reproduce every greedy action | `phase0b/analysis.json`: 2 160 episodes, 0 mismatching, **0 arg-max mismatches** | [V] |
| MADQL's frozen behaviour is a learned STAY preference, not tie-breaking | frozen-state gap Q(STAY) − best move: median **+0.0438**, 100 % positive, 0 exact ties (101 200 robot-steps) | [V] |
| T3 (off-path) dominates planner calls | T3 share: MADQL 0.946, IPPO 0.650, MAPPO 0.616 | [V] |
| Limitations | Block B development data; greedy; A\* replays; 6A.0 heuristic thresholds; no per-call RRT\* data for learned controllers (proposal §26.8) | documented |

### Phase 3: Execution Supervisor

| Claim | Evidence | Check |
|---|---|---|
| Architecture: an engine stage *between the policy and the shield / arbiter*, off by default and byte-identical when off | `sim/engine.py` (`sup_proposal` → shield → arbiter); `EvalTask.supervisor = "off"`, `Engine(supervisor=None)` | [V] |
| R + S + D selected, O dropped | `phase3/eval/analysis.json` 3c retention: R keep, S+O keep, D keep, **O keep = false**; decision §30.1. The forward mode is `supervisor="R+S"`, `deadlock=True` | [V] |
| Completion gains | 3b (A\*) P1: MADQL +0.373, IPPO +0.012, MAPPO +0.010. 3e (RRT\*): +0.301 / +0.008 / +0.004. All Holm-rejected; G0 pass; 3b and 3e "go" | [V] |
| Safety | 0 errors, collisions or supervisor-introduced invalid moves; fairness and burst bounds hold (§30) | [D] |
| Overhead and intervention | supervisor share of step time ≈ 3.9 % (IPPO, 3b) [V]; modified share 1.3 % (MADQL) and 0.15 % (PPO) [D] | [V]/[D] |
| Replay and historical compatibility | OFF reproduces 1 440 / 1 440 stored Block C records; 627 / 627 replays identical (§30) | [D] |
| Status | Development evidence on Block C. **Confirmation is H-L2 in Phase 6** | — |

### Phase 1: replanning optimization (P1)

| Claim | Evidence | Check |
|---|---|---|
| Fewer planner calls | E family Holm-rejected in every family and condition (calls −59 to −86 % with the supervisor) | [V] (rejections), [D] (percentages) |
| Rejected | Verdict **reject** in C1–C4 (`phase1/eval/analysis.json`) | [V] |
| Why | MADQL completion loss (C1 −0.0104, C3 −0.0383, Holm), and a high-priority flow-time veto in every family. Without the supervisor (C2 / C4), losses in every family | [V] |
| Status | `MonitorParams.repair_radius = 0` by default; the option stays implemented and off | [V] |

### Phase 2: RRT\* budget control (P2)

| Claim | Evidence | Check |
|---|---|---|
| Fewer RRT\* iterations | RRT\* iterations per robot-step: MADQL −16.2 %, IPPO −11.2 %, MAPPO −10.7 %, baseline −9.7 % (all Holm-rejected) | [V] |
| Verdict | L **inconclusive** (IPPO and MAPPO completion intervals unbounded at the ceiling). B: the condition-level verdict label "adopt" (§9.2). It is **not** an adoption into any configuration: `p2_forward_default: false` | [V] |
| Why not the default | The forward default needs L "adopt". The percentile reading was explicitly *not* substituted (Phase 2 §7.2) | [V] |
| Status | `refine_iterations = patience = None`; RRT\* keeps its full 1 000-iteration budget; P2 is excluded from Phase 6 (D10) | [V] |

### Phase 4: MADQL optimization

| Claim | Evidence | Check |
|---|---|---|
| Diagnosis | Weak correction at self-loop STAYs; TD noise larger than the decisive gaps; aliasing (Phase 4 §2) | [D] |
| The n-step implementation is compatible with the historical behaviour | `n_step = 1` by default and omitted from the configuration hash; `repro/madql_seed1/repro.json` **identical: true**; 19 tests | [V] |
| Selection | Block S rule §3.5 → **n3** (eligible: n3, t8, n3t8) (`select/selection.json`) | [V] |
| Block C | FA, FR and NA **adopt**; recommendation `madql_v2_forward_candidate: true` (`phase4/eval/analysis.json`) | [V] |
| Robot–obstacle (RO) concern | The Phase 4 post-hoc figure (1.9–2.5×; corrected from a misquote "2.1–2.5×", §9.1) used a mismatched denominator. Exact re-measurement: FA ×2.2, FR ×1.8, NA ×1.4 (n.s.) per 1 000 moves (`phase6/dev_ro_exact/dev_ro_exact.json`; 9 360 re-runs identical to their stored records) | [V] |
| Status | **The candidate for Phase 6, not a proven replacement.** The strict RO veto (D3) makes rejection likely if the Block C increase replicates | — |

### Phase 5: A\* optimization

| Claim | Evidence | Check |
|---|---|---|
| A deliberate, evidence-based scope decision | Proposal §34. A\* planning is 6–13 % of episode wall time (Phase 1 records); paths are already optimal; F8 concerns other maps; tie-breaking risk | [D] |
| No implementation | `planning/astar.py` last modified 2026-11-18, before Issue 6A Phase 4 / 5 | [V] |

---

## 3. Implementation and integrity constraints (A3)

| Constraint | Result |
|---|---|
| P1 and P2 optional and off by default | `repair_radius: int = 0`; `refine_iterations` and `patience` = None; `EvalTask.repair` = 0, `rrt_refine` and `rrt_patience` = None [V] |
| Supervisor off by default | `EvalTask.supervisor = "off"`, `Engine(supervisor=None)`; escalation `False` [V] |
| Forward supervisor mode R + S + D, O off | `supervisor="R+S"` (mechanisms parsed from the mode string; O only if listed) plus `deadlock=True`; Phase 6 G0 checks it on every record [V] |
| Historical MADQL available | Issue #5 checkpoints unchanged (in the 1 310-file manifest); `madql.toml` unmodified since 2026-11-19 [V] |
| MADQL-v2 = n3 | `selection.json` selects n3; the Phase 6 runner refuses another selection [V] |
| PPO unchanged by Phase 4 | `marl/ppo.py`, `ppo_trainer.py`, `ippo.toml` and `mappo.toml` last modified 2026-11-20, before Phase 4. None appears in the working-tree changes at the start of this session [V] |
| No A\* optimization | `astar.py` unchanged since 2026-11-18 [V] |
| No premature Phase 6 / Block V results | No `results/dev6a/phase6/eval/`; 0 Block V seed fields in 868 stored result files [V] |
| `models/` | does not exist [V] |
| Frozen and historical artifacts | Block C (3) and Block D (35) freeze hashes match; 1 310 historical files: 0 changed, 0 missing; frozen statistics `3fe86a77`, supervisor `83b84b8e`, Stage 3a `4c7b9dbc`; Phase 1–4 analyses unchanged [V] |
| `docs/SESSION_HANDOFF.md` | `ba4a439c`, last modified 2026-11-19 [V] |
| Issues #7 / #8 | No Issue #7 / #8 work in the change set. `docs/05` and `docs/07` still assign E5 / E6 to Issue #8 [V] |
| Eight consolidated issues | `docs/07` still lists the 8 issues [V]. **[!]** The Issue 6A proposal calls 6A a "ninth issue entry" (§5, item 2) |

---

## 4. Phase 6 design and implementation (B1, B2)

### 4.1 D1–D13 against the implementation

| Decision | Implemented as | Check |
|---|---|---|
| D1 Block V 14 000–14 199; RRT\* 14 000–14 059 | `phase6.BLOCK_V`, `BLOCK_V_RRT`; three-layer guard in every worker; disjointness test | [V] |
| D2 H-L2, H-v2, H-L12; 22 080 episodes | `MATRIX`, `episode_counts()` = 17 400 + 4 680 (test) | [V] |
| D3 Strict RO veto | `RO_VETO` (all-robot RO per 1 000 moves on K^arb) in every P4 family; a veto is a Holm-rejected increase (`_veto`, Phase 4 rule) | [V] |
| D4 Scope; PPO context non-gating | RO in the P4 of H-L2 (9), H-L12 (9) and H-v2 (4). `context_v2_vs_forward_ppo` is reported only; no decision reads it | [V] |
| D5 0.05 per unit, global bound | `UNITS` (7); `summary.error_control.global_bound` = 1 − 0.95⁷ = 0.302 | [V] |
| D6 Families and α; no P3 in H-L12 | `ALPHA_L`, `ALPHA_V2`; P3 only in H-L2 A\* (test) | [V] |
| D7 Ceiling rule | `lower_bound` (Phase 4 §5.4); also not evaluable → −∞ | [V] |
| D8 3b / 3e for H-L12 | `decide_system` for both H-L2 and H-L12 | [V] |
| D9 RR secondary | `rr_conflicts_ctrl/arb_per_100_robot_steps` nominal; the H-v2 conflicts-per-episode veto kept | [V] |
| D10 P2 off, 1 000 iterations | Tasks set `rrt_refine = rrt_patience = None`, `rrt_iterations = 1000` (test); G0 checks both | [V] |
| D11 No v2 RRT\* OFF | Not in `MATRIX["rrtstar"]` (test) | [V] |
| D12 Two verified independent backups | `freeze` refuses unless `backup_record.json` lists ≥ 2 destinations verified against the manifest, and the originals verify now (test) | [V] |
| D13 Separate gates | `freeze` needs `--authorized-freeze`; `check`, `run` and `replay` need `--authorized-phase6` and an unchanged freeze (tests) | [V] |

### 4.2 Implementation review: the twelve points

| # | Requirement | Result |
|---|---|---|
| 1 | Exact matrix | 22 080 unique jobs; RRT\* seeds = the subset (tests) |
| 2 | No overlap with historical or development data | Guard rejects train / A / B / C / D / S / test before generation, in real worker processes (spawn) |
| 3 | Units and multiplicity | 7 units, Holm within families, union ≤ 0.05 per unit |
| 4 | Consistent metrics | Shared ratio-of-sums functions; the observer's K^arb equals the recorded conflicts (G0 check); anchor 514 / 514 identical |
| 5 | RO veto | As D3 / D4 (tests, including an RO veto turning "go" into "inconclusive" (3b) / "no-go" (3e)) |
| 6 | Missing / invalid / duplicate / failed | **Defect found and fixed** (§5, item 1; not-evaluable output made explicit, §9.4). Now: a duplicate aborts; incomplete data aborts (resume first); an error or invalid record makes G0 fail and every row involving it "not evaluable" (p = 1, no estimate) |
| 7 | Replay before acceptance | Every confirmation requires `replay_identical`; a missing replay blocks confirmation (test) |
| 8 | Report from the analysis output | `phase6_report.render` reads only `analysis.json` |
| 9 | No confirmation without G0 and replay | Confirmation booleans are conjunctions with G0 (both planners) and the replay (tests) |
| 10 | `PYTHONHASHSEED=0` | Enforced in `phase6_analysis.main` (test) |
| 11 | No execution with an unapproved configuration | The freeze hashes the protocol, decision review, runner, analysis, report, backup tool, tests and reused frozen files; every later command re-verifies |
| 12 | Tests of the critical paths | 31 Phase 6 tests (design, guards, workers, freeze / backup gates, observer reproduction, end-to-end, failure paths, incomplete arm, replay, hash seed) |

---

## 5. Issues found

1. **[!] Defect: analysis crash on a failed or invalid episode** (fixed; no protocol change).
   - *Problem:* `phase6_analysis` read outcome fields from every record. A single exception record (status `error`) raised `KeyError`, so a real run with any failure would have produced no analysis.
   - *Required behaviour:* protocol §5 and Block D §7 require G0 failure and "not evaluable" rows.
   - *Fix:*
     - `usable()` / `invalid_reasons()` implement the Block D §7 classes;
     - `Data.pair` returns "not evaluable" for any unit involving an unusable episode;
     - G0 lists errors and invalid records and fails on either;
     - per-arm descriptions report "data incomplete";
     - the report tolerates it.
   - *Tests:* 3 new tests.
   - *Scope:* the matrix, thresholds, families and rules are unchanged.
2. **[!] Documentation inconsistency (not changed here).**
   - The Issue 6A proposal (§2, §18, §19 A.1) describes 6A as a "ninth issue entry" / "roadmap revision 3".
   - The project rule is exactly eight consolidated GitHub issues, with 6A a workstream under Issue #6. `docs/07` is consistent with eight.
   - *Proposed correction at closure:* a dated clarification note in the proposal. Do not rewrite the historical text.
3. **Corrected evidence (already documented):**
   - the Phase 4 post-hoc RO ratio, and readiness §10's PPO and supervisor RO figures, used inconsistent denominators. They are corrected in the decision review §3.6;
   - the Phase 4 report's Phase 5 figure (8–11 %, MADQL only) is narrower than the proposal §34 figure (6–13 %, all families). Both are correct for their data.
4. **Disclosures for Phase 6:**
   - The pre-run `check` regenerates A, B, C, D and S instances *in memory* for the disjointness fingerprints (as Block D's check did). No outcome is computed.
   - Training-instance disjointness is not verifiable.
5. **Remaining limitations:**
   - one scenario family; three training seeds per configuration;
   - the RO effect is heterogeneous across MADQL training seeds;
   - RRT\* RO rows for MADQL are low-power;
   - the global FWER bound is ≈ 0.30 for any false rejection (confirmations are conjunctions).

## 6. Tests and checks

| Check | Result |
|---|---|
| Full suite (`tests/unit tests/tools tests/integration`), after the follow-up patch | **1 286 passed**, 1 skipped (POSIX-only), 0 failed; the Phase 6 file: 32 / 32 |
| ruff (check, format), `mypy src`, `mypy` on the tools and tests | clean |
| Anchor | 514 / 514 stored Block C records reproduced (final runner) |
| Backup manifest | originals 1 216 / 1 216; D: staging 1 216 / 1 216 (not a D12 destination; not recorded) |

## 7. What remains before each gate

| Gate | Remaining |
|---|---|
| **D12 backups** | The student copies `D:\robocoord-phase6-backup\2026-11-23\` to (1) an external USB drive and (2) the personal OneDrive, then provides both paths. Then for each: check that it is accessible, physically independent of disk 0 and of the other copy, and (for OneDrive) inside the personal sync root and fully synchronized (not a cloud-only placeholder pending upload); `backup_manifest.py record --root <dest>` (1 216 / 1 216 plus the manifest); mark it read-only. Finally re-verify the originals |
| **Pre-freeze report** | Present the protocol SHA, code hashes, backup record, tests, anchor and integrity, then **stop** |
| **Freeze** | Separate authorization → `phase6.py freeze --authorized-freeze` |
| **Pre-run check** | After the freeze: `check --authorized-precheck`, all non-Block-V requirements, **no Block V generation** (Amendment A1). Then the pre-run report and **stop** |
| **Execution** | Separate authorization → `run --authorized-execution`: **first the Block V instance check (abort on failure)**, then the 22 080 episodes (≈ 5 h, 10 workers), `replay`, the `PYTHONHASHSEED=0` analysis (twice, byte-compared) and the final Phase 6 report |
| **Closure (Part C)** | End-to-end audit; the clarification of item 2; the closure report. No Git |

## 8. Current Phase 6 file hashes (pre-freeze, after Amendment A1)

| File | SHA-256 (prefix) |
|---|---|
| `docs/research/2026-11-23-issue-6a-phase6-protocol.md` (with Amendment A1) | `0c7dad5b43342b7e` |
| `docs/research/2026-11-23-issue-6a-phase6-decision-review.md` | `935c3ecc535e5cc3` |
| `tools/phase6/phase6.py` | `6bfa17cbff8928aa` |
| `tools/phase6/phase6_analysis.py` | `e34a4f25197d731c` |
| `tools/phase6/phase6_report.py` | `49c6832bfef00572` |
| `tools/phase6/backup_manifest.py` | `0040991bd8212f7e` |
| `tests/tools/test_phase6.py` | `709639ecf51dd5a2` |
| `results/dev6a/phase6/backup/manifest_sha256.json` | `87cb9ea65c651159` |

This report's own hash changes with every edit, so it is not listed. Tests after Amendment A1: full suite 1 291 passed, 1 skipped; Phase 6 file 37 / 37; anchor 514 / 514.

---

## 9. Reconciliations (follow-up, 2026-11-23)

### 9.1 The Phase 4 robot–obstacle ratio: "1.9–2.5×" vs "1.4–2.2×"

**Common to every figure below:**
- **Arms:** MADQL-v2 n3 vs the historical MADQL (`best.pt`, seeds 1–3).
- **Instances:** the same Block C instances. FA: A\*, R + S + D, 1 080 + 1 080 episodes. FR: RRT\*, R + S + D, 360 + 360. NA: A\*, OFF, 1 080 + 1 080.
- **Aggregation:** a ratio of sums over all episodes, pooled over the 3 policies, 3 levels and all seeds. The ratio is new rate / old rate. Intervals are the seed-cluster bootstrap-t of the difference.

| Source file | Numerator | Denominator (moves) | FA | FR | NA |
|---|---|---|---|---|---|
| `results/dev6a/phase4/eval/posthoc_conflicts.json` (Phase 4 §6.5, post-hoc) | ROv, all robots, on K^arb | MADQL policy's own non-STAY choices, **non-idle robots only** (`madql.nonidle_steps − madql.stay`) | 0.568 → 1.440 (**2.53**) | 0.611 → 1.278 (**2.09**) | 0.515 → 0.995 (**1.93**) |
| `results/dev6a/phase6/dev_ro/dev_ro_analysis.json` (stored-record approximation) | ROv + ROs, all robots | requested moves, non-idle robots (`active_robot_steps × (1 − wait_ratio)`) | 2.54 | 2.10 | 1.93 |
| `results/dev6a/phase6/dev_ro_exact/dev_ro_exact.json`, definition (i) | ROv + ROs, **all robots**, on K^arb | **every move reaching the arbiter, all robots** (`p6.moves_arb`) | 0.479 → 1.041 (**2.17**) | 0.518 → 0.948 (**1.83**) | 0.480 → 0.686 (**1.43**, n.s.) |
| same file, definition (ii) | ROv + ROs, non-idle robots | moves of non-idle robots | 2.10 | 1.62 | 1.15 (n.s.) |

**Why the figures differ:**
- The first two rows count every robot's conflicts but only non-idle robots' moves.
- MADQL-v2 moves idle robots far more often: 27–31 % of its moves against 7–15 % for the historical MADQL (exact re-run).
- The mismatch therefore inflates v2's rate more than the historical one.
- ROs is 0 throughout, so ROv vs ROv + ROs makes no difference.

**Which value is authoritative:**
- **For the approved Phase 6 outcome (D3, protocol §3.1):** definition (i), i.e. **×2.17 / ×1.83 / ×1.43**. These are still development (Block C) values. The confirmatory value will be Phase 6's own.
- **The Phase 4 figure 1.9–2.5× is correct for its post-hoc definition,** and its report keeps it.
- **The value "2.1–2.5×"** in the decision review (§1, §3.6) and in the first version of this report was **a misquote** of the Phase 4 figure.

**Documentation** (dated, nothing rewritten):
- an erratum in the decision review (§12);
- a clarification pointer in the Phase 4 report (§6.5);
- proposal §34.4;
- the correction above (this report).

### 9.2 Phase 2: "adopt" for the baseline

- **What the label means.** In Phase 2's pre-registered per-condition rule (§5.4), condition **B** (the rule-based baseline under RRT\*) received the verdict label **"adopt"**: no harm and a Holm-rejected reduction in RRT\* iterations.
- **That label does not adopt P2 anywhere.** The forward-default criterion was "L adopt **and** B not reject". L was "inconclusive", so `results/dev6a/phase2/eval/analysis.json` records `"p2_forward_default": false`.
- **Verified state [V]:**
  - `RRTStarConfig.refine_iterations` and `patience` are None by default;
  - `EvalTask.rrt_refine` and `rrt_patience` are None by default;
  - no configuration file and no Phase 6 code sets them;
  - Phase 6 excludes P2 (D10), and G0 checks `rrt_patience` / `rrt_refine` absent and `rrt_iterations` = 1000 on every record;
  - **patience = 400 was not adopted.** The approved RRT\* configuration is the full 1 000-iteration budget.

### 9.3 Block V generation timing: conflict with the approved protocol, awaiting the student's decision

**The approved texts:**
- **D13** (decision review §11): "Protocol approval → implementation → verified backups → separate freeze authorization → freeze → **pre-run check** → separate execution authorization → execution, replay, analysis, report".
- **Protocol §2:** "**Pre-run check** (the first generation, after the freeze; aborts on failure): 0 stored files with a Block V seed; the instances unique per level and disjoint from every A, B, C, D and S instance of that level".
- **Protocol §11, step 5:** "`check --authorized-phase6`: **the first Block V generation** and every pre-run check". Step 6: "**Separate authorization to execute**, then `run`, `replay` and `phase6_analysis.py`".
- **Protocol §2 (reservation):** Block V is "reserved at the approval of this protocol". This is a documentary designation in the approved text; no artifact exists.
- **Runner:** `instance_check()` inside `check` generates the 600 Block V instances in memory, computes their fingerprints and writes the fingerprints, plus uniqueness and overlap counts, to `check.json`. No policy, episode or outcome is involved. Block D followed the same sequence (Block D §9 steps 5–6).

**The conflict.** The student's instruction of 2026-11-23 is "do not generate, reserve, consume, or materialize any Block V scenarios before my explicit execution authorization". The approved protocol places the first Block V generation, and the storage of its fingerprints, **after the freeze but before** the execution authorization. Nothing has been changed; the student decides.

**Options:**
- **(A) Keep the approved protocol.** `check` generates the Block V instances in memory (fingerprints only) before the execution authorization.
- **(B) Amend before the freeze.**
  - Split the pre-run check. Everything that does not touch Block V runs before the execution authorization: the freeze verification, the results scan, the checkpoints, Stage 3a, the episode count, and the A–S fingerprints.
  - The Block V uniqueness and disjointness check runs **as the first step after** the execution authorization and aborts the run before any episode if it fails.
  - This needs a dated amendment to protocol §2 / §11 and a small runner change: move `instance_check` into a guarded pre-episode step of `run`, with tests. All before the freeze.

**Recommendation:** (B). It matches the student's instruction, loses no safeguard, and changes nothing scientific. The integrity check still precedes every episode.

**Decision (2026-11-23): option B. Implemented as protocol Amendment A1, before the freeze:**
- `check` (`--authorized-precheck`) no longer generates or fingerprints Block V.
- `run` (`--authorized-execution`) performs the Block V instance check first, once, and aborts before any episode on failure.
- G0 requires every record's instance hash to match that check.
- Five new tests prove it.

### 9.4 "Not evaluable" vs a p-value

- **Required by the protocol.** Protocol §5: "Not-evaluable rows: p = 1, kept in the family, never rejected, no estimate published (Block D §7)". Block D §7 rule 3: such a row "enters `holm` with p = 1 … never rejected"; rule 4: "no estimate, interval or p-value is computed or published anywhere".
- **So p = 1 is the protocol's Holm *input*, not a computed p-value.**
- **Implementation (this patch).** After the frozen Holm procedure, every not-evaluable row is stored with:
  - `evaluable: false`;
  - `holm_input_p: 1.0`;
  - `p_holm: null`;
  - `rejected: false` and `mc_sensitive: false`;
  - a note quoting the rule;
  - no estimate, interval or p.
- **Evaluable rows** carry `evaluable: true` and their computed p and Holm p.
- **Report.** The report prints "**not evaluable** … — (Holm input p = 1 by rule)" with no estimate, interval or p-value. Before this patch, the Phase 3 table helper printed "1" in the Holm-p column.
- **Effect on confirmation.** Any failed or invalid episode fails G0, and every confirmation statement requires G0 on both planners and an identical replay. Tested.

### 9.5 Changed-file inventory

**This follow-up patch:**

| File | Change |
|---|---|
| `tools/phase6/phase6_analysis.py` | `holm()` wrapper: explicit not-evaluable rows (§9.4) |
| `tools/phase6/phase6_report.py` | its own family table: no p-value for not-evaluable rows |
| `tests/tools/test_phase6.py` | +1 test (`test_not_evaluable_rows_never_carry_a_p_value`) |
| `docs/research/2026-11-23-issue-6a-review.md` | this section; corrected rows in §2, §4.2 and §7 |
| `docs/research/2026-11-23-issue-6a-phase6-decision-review.md` | §12 Errata, appended |
| `docs/research/2026-11-22-issue-6a-phase4-madql.md` | one dated clarification blockquote after §6.5's ratio sentence; no other change |
| `docs/research/2026-11-20-issue-6a-proposal.md` | §34.4, appended |

**The earlier review patch** (same day):
- `tools/phase6/phase6_analysis.py`: `usable` / `invalid_reasons`, not-evaluable pairs, G0 invalid records, incomplete arms;
- `tools/phase6/phase6_report.py`: incomplete arms;
- `tests/tools/test_phase6.py`: +4 tests and `_fixture`;
- `docs/research/2026-11-23-issue-6a-review.md`: new.

**Not changed:**
- the runner `tools/phase6/phase6.py`;
- the protocol `docs/research/2026-11-23-issue-6a-phase6-protocol.md`;
- the backup tool;
- D12;
- any frozen artifact, historical result or checkpoint;
- `docs/SESSION_HANDOFF.md`.
