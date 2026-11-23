# Issue 6A Phase 6: readiness assessment and draft confirmatory protocol

*Status: **DRAFT, not approved, not frozen.** No Phase 6 episode has been run, and no held-out scenario has been generated.*

**Readiness verdict: not ready to execute.** The design, the held-out block, the runner, the analysis and the integrity gates are prepared and tested. However, the approved documents leave several essential decisions open, and they are the student's to make (§13). Formal confirmatory runs stay blocked by code, because the runner refuses to work without a freeze record and an explicit authorization flag (§11).

Labels: **[A]** approved (with source); **[M]** measured; **[C]** follows from code or definitions; **[P]** projection; **[D]** a draft proposal that needs approval.

---

## 0. Summary

| Item | Status |
|---|---|
| Phase 6 scope | **Partly approved.** Roadmap: "All required algorithm × planner combinations, on held-out data" [A]. The hypotheses still to confirm are named in the proposal (§12, §19 D): H-L2 and H-L12 [A]. Phase 4 adds H-v2 [A, 2026-11-23]. No Phase 6 protocol exists yet; this document drafts it [D] |
| Held-out block | **Approved in principle, not reserved.** Phase 3 decision D7: "final validation in Phase 6 on a new block (14 000–14 199) … reserve the block before Phase 6" [A]. The RRT\* subset and the block name are not specified [D] |
| Data separation | **Verified, as far as possible before the freeze [M]** (§4). The instance-level disjointness check is part of the pre-run check, after the freeze, as in Block D |
| Configurations | Prepared (§5). They follow the approved forward configuration exactly |
| Outcomes, statistics, decisions | Drafted from the approved Stage 3a (§6) and Phase 4 (§5) rules, with **no new threshold** (§6–§8). Family structure and multiplicity scope need approval [D] |
| Robot–obstacle veto | Defined (§6.3). **Important consequence:** if the Block C effect replicates, the strict veto rejects MADQL-v2 with projected probability 0.98–1.0 in every condition (§10) |
| Runner, analysis, tests | Prepared and tested: 23 new tests; the anchor check reproduces 514 / 514 stored Block C records (§11) |
| Integrity | All frozen artifacts unchanged (§12) |

## 1. What was inspected

**Documents:**
- the Issue 6A proposal: §6, §11–§16, §19 (decisions), §23–§25 (SCB-C v5, Block C, Block D), §27 (roadmap), §30.1, §31–§34;
- the frozen Block D protocol (revision 2): the template for a confirmatory run;
- the Block C analysis plan;
- the Phase 3 design (decisions D1–D11, approved) and the Stage 3a freeze §6 (the go / no-go criteria, approved 2026-11-22);
- the Phase 3, 1, 2 and 4 reports;
- the post-Issue-6 analysis;
- `docs/05-testing-and-evaluation.md` (E5 / E6 belong to Issue #8).

**Code:**
- `marl/scenarios.py` (production seed blocks: train, A, B only) and `marl/evaluation.py` (`run_episode`, `EvalTask`, `check_seeds`);
- `sim/engine.py` (`StepOutcome`: `actions`, `arbitrated_actions`, `events` = K^arb, `controller_events` = K^ctrl);
- the frozen statistics `tools/selection/block_c_analysis.py`;
- the Block D tools (`block_d.picks`);
- the Phase 1–4 runners and observers.

**Data (read-only):**
- every `results/` JSON, JSONL and CSV file (863), scanned for held-out seeds;
- the Phase 1, 3 and 4 records, for unit costs, standard errors and descriptive conflict rates;
- the checkpoint manifests.

## 2. What is already approved and frozen [A]

| Item | Source | Content |
|---|---|---|
| Forward execution configuration | Proposal §30.1 (2026-11-22) | Execution Supervisor **R + S + D**, O dropped (arm `R_S_D`); Stage 3a parameters frozen (`freeze.json` `4c7b9dbc`) |
| P1 | §31 | Rejected; repair off |
| P2 | §32 and the student's decision of 2026-11-22 | Optional, **off by default**. m = 400 "may be considered during Phase 6 … must not be … silently enabled". The 2026-11-23 instruction: P2 disabled |
| MADQL | §34.1 (2026-11-23) | **MADQL-v2 n3** is the proposed configuration; the historical MADQL stays the reference; robot–obstacle conflicts per move are a pre-registered veto; Phase 4 Block C is not confirmation |
| Phase 5 | §34 | Not pursued; A\* unchanged |
| PPO policies | §23, Phase 3 decision D5 | The **SCB-C v5 picks** (frozen). H-L1 (v5 vs Issue #6 picks) was confirmed once on Block D (§25) and is not repeated |
| Supervisor scope | Phase 3 decision D9 | Learned controllers only; the baseline runs in its original composition |
| Held-out validation | Phase 3 decision D7 | A new block, **14 000–14 199**, to be reserved before Phase 6 |
| Confirmatory hypotheses | Proposal §12 and §19 D | **H-L1** (done on D), **H-L2** (supervisor vs none, same checkpoints) and **H-L12** (L1 + supervisor vs the historical system), "each with its metric family, margins and Holm correction" |
| Statistical machinery | Block C/D protocols; Stage 3a §6.1 (approved 2026-11-22) | Ratio of sums, seed cluster, bootstrap-t, B = 10 000, `default_rng(0)`, Holm within families, MC-sensitivity flag, not-evaluable rows at p = 1, decisions on Holm-adjusted p only, nominal intervals labelled unadjusted |
| Decision rules | Stage 3a §6.3 (approved for Phase 3); Phase 4 §5.4 (pre-registered for Phase 4) | The 3b / 3e go / no-go rules; the Phase 4 adopt / reject / inconclusive rules; δ₁ = one task per episode |
| Block roles | Proposal §12, §19 A.2 | B = historical reference; C = development; D = single-use held-out (used for H-L1); S = selection; test 20 000–20 999 reserved for Issue #8 |
| Single use | Block D protocol §13 | "Block D is single-use. A future L2 would need a new reserved block." |

**Out of Phase 6 scope** (approved elsewhere):
- E5 and E6, the system comparison and scaling on test seeds: Issue #8;
- stochastic execution, the shield and the sequential allocator: E5 ablations;
- P1, P2 and Phase 5;
- committing checkpoints to `models/`;
- Issues #7 and #8.

## 3. What Phase 6 must confirm

| Hypothesis | Development evidence (Block C) | Comparison on the held-out block |
|---|---|---|
| **H-L2** (Phase 3) | Supervisor completion gains: MADQL +0.37, IPPO +0.012, MAPPO +0.010 (A\*); the same pattern with RRT\* | OFF vs R + S + D for the 9 Phase 3 policies (historical MADQL; v5 IPPO / MAPPO), A\* and RRT\* |
| **H-v2** (Phase 4) | MADQL-v2 n3 "adopt" in FA, FR and NA | v2 vs historical MADQL in FA (A\*, R + S + D), FR (RRT\*, R + S + D) and NA (A\*, OFF) |
| **H-L12** (proposal §12) | Not measured as one comparison | The Issue 6A forward system (v5 PPO + R + S + D; MADQL-v2 + R + S + D) vs the Issue #6 system (Issue #6 PPO picks, OFF; historical MADQL, OFF), A\* and RRT\* |

The rule-based baseline is a descriptive comparator in its original composition (as Block D D4). It is never part of a hypothesis.

## 4. Held-out data verification [M]

| Check | Result |
|---|---|
| Proposed block | **V = seeds 14 000–14 199**, the 3 levels (static, obstacles, full); RRT\* subset **14 000–14 059** [D], mirroring Block D's 200 / 60 split |
| Range disjointness | V is disjoint from train (0–9 999), A, B, C (11 000–11 119), D (12 000–12 199), S (13 000–13 079) and test (20 000–20 999). Tested in `test_block_v_is_disjoint_from_every_other_block` |
| Stored data | **0 of 863** stored result files (`results/`, JSON / JSONL / CSV) contain a seed field in 14 000–14 199 |
| Code, tests and configurations | No reference to any seed in 14 000–14 199 in `src/`, `tools/`, `tests/` or `configs/` before this preparation. Only the earlier phase documents mention the range as a proposal |
| Prior generation | No code path generates these seeds. The new tests cannot: an autouse sentinel raises on any Block V generation, and the worker-process probe never runs an allowed task |
| Use in selection or tuning | None. PPO picks used A ∪ S; MADQL-v2 checkpoints used A (rule) and S (candidate selection); Stage 3a calibration used B; the supervisor, P1, P2 and Phase 4 were developed on C; Phase 4 diagnostics used B |
| Instance-level disjointness | **Not yet checked, by design.** It requires generating Block V instances. As in Block D §8, the first generation happens only in the pre-run `check`, after the freeze. The check requires the 600 instances to be unique per level and to coincide with no A, B, C, D or S instance |
| Training disjointness | **Not verifiable.** No training-instance hashes are stored (as Block D §8.6). Training seeds are a disjoint range, and every instance is generated from its own seed |

**Conclusion:** seeds 14 000–14 199 are genuinely unused. Before any generation, the block needs formal reservation (decision 1, §13).

## 5. Proposed configurations and sample sizes [D]

**Common settings for every arm** (the approved forward configuration):
- greedy; auction allocator; shield off; endpoint recovery on;
- `path_limit` 6; horizon 400;
- P1 repair off; RRT\* with 1 000 iterations (P2 off);
- the production `run_episode`;
- one episode per instance.

**Conditions:**
- **OFF:** no supervisor, resolver off (the Issue #6 composition).
- **RSD:** `supervisor="R+S"`, resolver on, the frozen Stage 3a parameters, O off.
- **ORIG:** baseline only; no supervisor, resolver on (the original rule).

**Policies and checkpoints** (all verified by `phase6.checkpoints()`):

| Policy | Runs | Checkpoint | Verification |
|---|---|---|---|
| `hist` | madql_seed1–3 | Issue #5 `best.pt` (`dd9f088c`, `c66bd8c2`, `492f2a61`) | equals `best.json` |
| `v2` | madql_seed1–3 | MADQL-v2 n3 `best.pt` (`652f2997`, `bb05d379`, `bdf781b9`) | `selection.json` selects n3; equals `phase4_run.json` |
| `v5` | ippo_seed1–3, mappo_seed1–3 | SCB-C v5 picks | every Block D `picks` check |
| `i6` | mappo_seed1–3, ippo_seed1 | Issue #6 picks (step files equal to `best.pt`) | every Block D `picks` check |
| `base` | — | rule-based baseline | — |

**Matrix:**

| Planner | Cells (policy × condition) | Instances per run | Episodes |
|---|---|---|---|
| A\* | hist × {OFF, RSD}; v2 × {OFF, RSD}; v5 × {OFF, RSD}; i6 × OFF; base × ORIG | 200 seeds × 3 levels = 600 | 29 × 600 = **17 400** |
| RRT\* | hist × {OFF, RSD}; v2 × RSD; v5 × {OFF, RSD}; i6 × OFF; base × ORIG | 60 seeds × 3 levels = 180 | 26 × 180 = **4 680** |
| **Total** | | | **22 080** |

**Which cell serves which hypothesis:**
- H-L2: hist and v5, OFF vs RSD.
- H-v2: hist vs v2 under RSD (FA, FR) and OFF (NA). v2 under RRT\* OFF is not a development claim and is not run.
- H-L12: historical (hist OFF; i6 OFF; v5 OFF for the unchanged ippo_seed2/3) vs forward (v2 RSD; v5 RSD).

**Projected cost [P]:** based on measured mean wall seconds per episode from the Phase 1 and 4 records on 10–11 workers.
- A\*: about 16 500 episode-seconds, ≈ 0.5 h.
- RRT\*: about 149 000 episode-seconds, ≈ 4.1–4.6 h. MADQL OFF with RRT\* (92 s per episode) dominates.
- Replay and checks: ≈ 0.3 h.
- **Total ≈ 5–5.5 h on 10 workers** (×1.1 margin; the machine otherwise idle).

## 6. Pre-registered outcomes [D, from approved definitions]

### 6.1 Integrity and safety (G0; hard gates, 0 required)

- physical collisions, `PhysicalCollisionError` and `InvalidTransitionError`, other exceptions (execution errors);
- invalid moves (INV);
- supervisor-introduced invalid moves;
- invalid reference paths.

Plus (approved Stage 3a G0.5–G0.7, and Phase 4 G0.8):
- the fairness and burst bounds;
- the frozen parameters;
- the composition of every condition;
- repair off, P2 off, 1 000 RRT\* iterations;
- instances identical across arms;
- K^arb counted by the observer equals the recorded conflicts;
- every checkpoint hash unchanged before and after the run;
- replay of every 50th record identical in fresh processes.

### 6.2 Primary, veto and secondary outcomes

| Role | Outcome | Definition (ratio of sums over seeds; levels summed per seed) |
|---|---|---|
| **Primary** | Task completion | Σ delivered / Σ released (P1 per family; P2 per policy) |
| **Veto** | High-priority flow time | Flow time of the highest-priority tasks delivered in both arms (Stage 3a §6.6) |
| **Veto** | High-priority undelivered | Undelivered highest-priority tasks per episode |
| **Veto (new, required)** | **Robot–obstacle conflicts per move** | §6.3 |
| **Veto (H-v2, as Phase 4)** | Conflicts per episode | `conflicts_total` per episode |
| **Targeted** (H-L2, H-L12, A\*) | Stuck robots by category | frozen, oscillating, conflict_loop, deadlock_cycle (Stage 3a §6.4, re-labelling rule) |
| **Efficiency** (H-v2, as Phase 4) | Planner calls per 100 robot-steps; flow time of common tasks; supervisor changes per 100 robot-steps (FA, FR) or stuck robots per episode (NA) | Phase 4 §5.2 |
| **Secondary, nominal** | Robot–robot conflicts (RRv + RRs) per 100 robot-steps, on the policy's own proposal K^ctrl and on the arbiter's input K^arb; conflicts per episode; flow time; undelivered; planner calls; wait ratio | Reported with nominal intervals. K^ctrl is the measure of record for the supervisor, because R lowers K^arb trivially (Phase 3 design §8) |
| **Descriptive** | Everything in the Phase 3 / Phase 4 reports | Episode classes, conflict types, triggers, stuck classes, MADQL decision statistics, runtime; by policy, level, planner and condition |

### 6.3 The robot–obstacle conflicts-per-move veto (the student's requirement)

**Definition [D]:**

> RO per 1 000 moves = 1 000 · Σ (ROv + ROs in K^arb) / Σ moves, where a *move* is a robot-step whose action reaching the arbiter (`a^arb`, after supervisor and shield) is not STAY.

- **Unit:** per instance, paired, ratio of sums, seed cluster.
- **Why this denominator:**
  - [C] A STAY can never be in an RO conflict (model §3), so moves are the exposure.
  - `a^arb` is the action set on which the recorded conflicts are counted, and it is controller-agnostic (MADQL, PPO, supervisor).
- **Difference from Phase 4:** the Phase 4 post-hoc analysis used the MADQL policy's own non-STAY choices. Without a supervisor the two coincide; with R + S + D they differ by the supervisor's changes (≈ 0.6–1 % of robot-steps).
- **Collection:** the Phase 6 observer (`p6.moves_arb`, `p6.k_arb`). Tested: K^arb equals the recorded conflicts, and moves are bounded by robot-steps.
- **Veto rule:** a **Holm-rejected increase** within the P4 family of the comparison. This is the existing veto convention for every P4 outcome (Stage 3a §6.6, Phase 4 §5.4). No margin is added, because none is approved (decision 3).
- **Where it applies [D]:** in all three comparison sets (decision 4).

## 7. Statistical design [D, from approved machinery]

**Estimator, bootstrap and inference (frozen):**
- `block_c_analysis.py` (`3fe86a77`) through the Phase 3 and Phase 4 helpers;
- paired by instance; seed-cluster bootstrap-t; B = 10 000; `default_rng(0)`; one index matrix per planner;
- two-sided p; Holm within each family; MC-sensitivity flag;
- not-evaluable rows enter Holm with p = 1 and are never rejected;
- decisions on Holm-adjusted p only; nominal 95 % intervals labelled unadjusted.

**Families and α** (the approved splits; union bound ≤ 0.05 per comparison set and planner / condition):

| Set | Family | Hypotheses | α |
|---|---|---|---|
| H-L2 and H-L12, per planner | P1 completion, per family (MADQL, IPPO, MAPPO) | 3 | 0.03 |
| | P2 completion, per policy | 9 | 0.005 |
| | P3 stuck categories × family (A\* only; as 3b) | 12 | 0.01 |
| | P4 vetoes × family: high-priority flow, high-priority undelivered, **RO per 1 000 moves** | 9 | 0.005 |
| H-v2, per condition (FA, FR, NA) | P1 completion (MADQL pooled) | 1 | 0.03 |
| | P2 per policy pair | 3 | 0.005 |
| | E efficiency (two-sided) | 3 | 0.01 |
| | P4 vetoes: conflicts per episode, high-priority flow, high-priority undelivered, **RO per 1 000 moves** | 4 | 0.005 |

**Multiplicity scope.** The approved Phase 3 and Phase 4 rules control the family-wise error at 0.05 *per stage* (here: per comparison set and planner / condition). Across the 7 analysis units of Phase 6 (H-L2 × 2, H-L12 × 2, H-v2 × 3) the global error is not controlled. Whether to keep per-unit control or to split a global 0.05 is decision 5.

**Completion ceiling (pre-registered; Phase 4 §5.4, the Phase 2 lesson):**
- If the bootstrap-t interval of a completion row is unbounded, the lower bound used by the decision rule is the **smaller** of the percentile bound and the loss-only bound −Σ_s max(0, delivered_old − delivered_new) / Σ released_old. The loss-only bound ignores every gain.
- An unbounded interval is **never** a rejection, and never evidence of equivalence.
- With a finite interval the rule does not apply.
- The same handling applies to the "no harm is interpretable" condition of the 3b rule.
- No other sensitivity analysis may replace the approved procedure.

**Practical-significance unit:** δ₁ = 1 / mean released tasks per old-arm episode (one task per episode; approved, Stage 3a §6.3).

## 8. Decision rules [A for the rules; D for their use in Phase 6]

**H-L2 and H-L12**, per planner (Stage 3a §6.3, approved), with the RO veto added to P4:
- **A\* ("3b"):**
  - "go" if G0, a MADQL gain that is Holm-rejected with Δ̂ ≥ δ₁, no Holm-rejected PPO loss (P1 or P2), PPO lower bounds > −δ₁ (ceiling rule), and no veto;
  - "no-go" if G0 fails, a PPO loss is rejected, or the MADQL Δ̂ ≤ 0;
  - otherwise "inconclusive".
  - **Note:** under this approved rule a veto alone gives "inconclusive", not "no-go".
- **RRT\* ("3e"):** "go" if G0, no rejected loss, MADQL Δ̂ > 0 and no veto; otherwise "no-go".

**H-v2**, per condition (Phase 4 §5.4), with the RO veto added to P4:
- "reject" if G0 fails, a completion loss is rejected, or a veto is triggered;
- "inconclusive" if the completion lower bound is ≤ −δ₁;
- "inconclusive (mixed)" or "no benefit (efficiency harm)" for a rejected E harm;
- "adopt" for a rejected completion gain or E improvement;
- otherwise "no benefit".

**Conclusions licensed by Phase 6 [D]:**
- **MADQL-v2 n3 is confirmed as the forward MADQL** only if H-v2 FA is "adopt", FR and NA are not "reject", and G0 and the replay hold. Otherwise the historical MADQL remains the forward MADQL, and v2 stays a documented, separately labelled configuration.
- **The supervisor R + S + D is confirmed** only if H-L2 is "go" on A\* and on RRT\*.
- **H-L12** is reported as the integrated Issue 6A vs Issue #6 comparison under the same rule.
- No result is re-analysed under another rule. A defect found after the run is reported, not fixed. A re-run needs approval and is labelled as such.

## 9. Execution sequence and authorization gates [D, Block D §9 as template]

1. Student approval of this protocol and of the decisions in §13 (with any changes).
2. Update of the runner and analysis to the approved decisions, with tests. No Block V generation.
3. **Separate approval to freeze**, then `phase6.py freeze --authorized-freeze`. This hashes this document, the runner, the analysis, every reused frozen file and the test suite, and refuses if a check or record exists.
4. `check --authorized-phase6`:
   - the first Block V generation;
   - the results scan, checkpoint verification, Stage 3a freeze check;
   - the instance uniqueness and disjointness check (A, B, C, D, S);
   - the episode count.
5. **Separate authorization to execute**, then `run`, `replay` and `phase6_analysis.py` (each verifies the freeze).
6. Report, as a new proposal section, then stop for review.

## 10. Projected precision and the consequence of the RO veto [P]

> **Correction (2026-11-23, decision review §3.6).** The RO rates and ratios below used inconsistent or approximate move denominators. The exact Block C values are in `2026-11-23-issue-6a-phase6-decision-review.md` §3.2–3.3, measured over all robots in both numerator and denominator:
> - MADQL-v2 vs historical: FA ×2.2, FR ×1.8, NA ×1.4 (NA not significant);
> - no supervisor effect on the RO rate in any family;
> - forward PPO rate 1.4–2.2 per 1 000 moves.
>
> The text below is kept unchanged for the record. The decision review supersedes it and the draft protocol of §6–§9.

**Method** (Block D §6):
- SE_V = SE_C · √(n_C / n_V), with A\* 120 → 200 seeds and RRT\* 40 → 60 seeds.
- MDE at 80 % power = (z₁₋α′/2 + 0.84) · SE_V.
- The worst case uses the first Holm step, the best case the last.
- Block C estimates are used only for these projections, never to change the design.

**H-L2 (from Phase 3 Block C):**
- Completion SEs are 0.0079 (MADQL) and 0.0012 (PPO) on A\*, and 0.0111 / 0.0009–0.0015 on RRT\*. Projected Block V SEs are about 0.006 / 0.0009 (A\*) and 0.009 / 0.001 (RRT\*).
- The development effects (+0.37 MADQL; +0.004 to +0.012 PPO) are therefore far above the MDE (≈ 0.003–0.004 for PPO on A\*).

**H-v2 (from Phase 4 Block C):**

| Condition | Completion SE_V / 80 % MDE | RO per 1 000 moves: Block C Δ̂, SE_V, 80 % MDE (worst–best) | **P(RO veto) if the Block C effect is real** |
|---|---|---|---|
| FA | 0.0004 / 0.0013 | +0.87, 0.12, 0.42–0.47 | **≈ 1.00** |
| FR | 0.0011 / 0.0034 | +0.67, 0.13, 0.46–0.52 | **≈ 0.98–0.99** |
| NA | 0.0059 / 0.018 | +0.48, 0.09, 0.33–0.37 | **≈ 0.98–0.99** |

- The existing P4 veto "conflicts per episode" (FA: Block C Δ̂ = +0.64, SE_V 0.20, MDE 0.72–0.80) would also probably be triggered in FA.

**Reading:**
- Under the strict veto, MADQL-v2 n3 is very likely to be **rejected** on Block V, unless the Block C robot–obstacle increase was a Block C artefact.
- For reference, the descriptive Block C rates give context. MADQL-v2 proposes about 1.4 RO conflicts per 1 000 moves (FA). That is roughly the level of the PPO policies already in the forward system: about 1.0–2.1, measured with an approximate move denominator, robot-steps × (1 − wait ratio), on Phase 3 data. The historical MADQL's 0.45–0.57 is unusually cautious.
- Phase 3 data also suggest, with the approximate denominator, that the supervisor raises the IPPO rate on A\* (about 1.64 → 2.09). If the veto applies to H-L2, it could bind there too.
- These are development observations, not tests. They are reported so that decision 3 is made knowingly.

## 11. Implementation and readiness of the tools [M]

| Component | Status |
|---|---|
| `tools/phase6/phase6.py` (new) | Design and matrix; three-layer Block V guard inside every worker (task function, wrapped `run_episode`, replaced `check_seeds`); checkpoint verification of all five policies; `Phase6Observer` (Phase 4 observer + moves to the arbiter + K^arb and K^ctrl by type); `freeze` (`--authorized-freeze`; refuses after a check); `check` / `run` / `replay` (`--authorized-phase6` plus an unchanged freeze); results scan; instance check; `anchor` (Block C only) |
| `tools/phase6/phase6_analysis.py` (new) | H-L2, H-v2 and H-L12 families, Holm, ceiling rule, decision rules, G0, completeness, summary; verifies the freeze before analysing |
| `tests/tools/test_phase6.py` (new) | 23 tests, below |
| Production code | **Unchanged.** No source, configuration, A\*, RRT\*, supervisor, PPO or MADQL change is needed for Phase 6. The production seed check stays train / A / B; Block V is admitted only inside the worker processes, as for Blocks C, D and S |
| Still to do after approval | Apply the decisions of §13 to the analysis constants; the report generator (tables); the freeze |

**Tests** (`tests/tools/test_phase6.py`, 23 passed):
- the sentinel itself;
- the design counts (22 080) and key uniqueness;
- block disjointness;
- 10 guard rejections and the accepted range;
- the anchor guard;
- the task encoding of every condition (supervisor, resolver, repair 0, P2 off, 1 000 iterations, greedy, auction, shield off; the baseline only in ORIG);
- refusal of every command without authorization;
- freeze refusal after a check, and freeze-change detection;
- the guards in real worker processes (spawn; fork where available): out-of-design tasks rejected at all three layers before any generation, and an allowed task reaching only the probe;
- the results scan;
- the observer on a Block C episode, reproducing the stored Phase 1 record exactly, with K^arb equal to the recorded conflicts and consistent move counts;
- the analysis maps and outcome functions;
- the ceiling rule;
- the system decision rules, including the RO veto.

**Anchor check** (development data, Block C only; never confirmatory evidence): **514 / 514 stored records reproduced exactly** in every shared non-timing field. In every one, the observer's K^arb equals the recorded conflicts.

- **Sources** (stride 25):
  - Phase 1 OFF / R + S + D, all 9 policies, A\* and RRT\*;
  - Phase 4 MADQL-v2 FA / NA / FR;
  - the Block C Issue #6 picks (A\*);
  - the Phase 2 baseline (RRT\*).
- **Coverage:** 13 of the 15 Phase 6 cells. The two without a stored Block C counterpart, baseline A\* and Issue #6 picks RRT\*, rely on the replay gate.
- **Output:** `results/dev6a/phase6/anchor/anchor.json` (`db0478f9`), produced by the final code.
- **Status:** Block C development data, not confirmatory evidence.

## 12. Tests and integrity [M]

| Check | Result |
|---|---|
| Full suite `tests/unit tests/tools tests/integration` | **1 277 passed**, 1 skipped (POSIX-only `getrusage` branch), 57 deselected (slow markers), 0 failed |
| New Phase 6 tests | 23 passed |
| Phase 4 default-off and reproduction guarantees | `n_step = 1` keeps the Issue #5 configuration hash (tests pass); `repro.json` byte-identical; the MADQL and PPO regression tests pass unchanged; the anchor reproduces the Phase 1 historical MADQL and PPO records |
| `ruff check`, `ruff format --check` (`tools/phase6`, `tests/tools/test_phase6.py`) | clean |
| `mypy src` | no issues (84 files) |
| `mypy` on `tools/selection`, `tools/phase1–4`, `tools/phase6` and the Phase 4 / 6 tool tests | no issues (36 files) |
| Block C / Block D `plan_freeze.json` hashes (3 + 35 entries) | all match |
| 1 310 historical files hashed before Phase 6A.0 (training runs and checkpoints, `SESSION_HANDOFF.md`, …) | 0 changed, 0 missing |
| Frozen statistics, supervisor, Stage 3a freeze | `3fe86a77`, `83b84b8e`, `4c7b9dbc` (unchanged) |
| Phase 1–4 records, analyses, selections, manifests (21 files checked) | unchanged |
| 9 MADQL-v2 candidate `best.pt` | intact against `phase4_run.json` |
| `models/` | does not exist |
| `docs/SESSION_HANDOFF.md` | untouched (`ba4a439c`, last modified 2026-11-19) |
| A\* implementation and defaults | unchanged (Phase 5 not pursued) |
| Phase 6 files (prepared, not frozen) | `phase6.py` `3b142512`, `phase6_analysis.py` `96a30827`, `test_phase6.py` `af044325` |

## 13. Decisions requiring the student's approval

| # | Decision | Options | Recommendation |
|---|---|---|---|
| **1** | **Reserve the held-out block** | (a) V = 14 000–14 199, RRT\* subset 14 000–14 059 (the D7 range, the Block D split) | (a). Reserve it now and register it as single-use |
| **2** | **Scope of the confirmatory comparisons** | (a) H-L2 + H-v2 + H-L12 (§3, 22 080 episodes, ≈ 5–5.5 h); (b) without H-L12 (−3 120 episodes); (c) A\* only for some sets | (a): all three are approved hypotheses (proposal §12, §34.1) |
| **3** | **The RO-per-move veto rule** | (a) strict: any Holm-rejected increase is a veto, the existing P4 convention (§10: likely rejects v2); (b) non-inferiority with a margin **chosen by the student** (e.g. in RO per 1 000 moves), tested one-sided; (c) a comparator-based bound, e.g. v2's rate not above the forward PPO policies' rate, which needs its own definition | Only (a) uses no new threshold. If (b) or (c) is preferred, it must be fixed now, before any Block V data, and never after |
| **4** | Where the RO veto applies | (a) all three sets (H-L2, H-v2, H-L12); (b) H-v2 only (the Phase 4 recommendation) | (a): the requirement was stated as a Phase 6 outcome, and the supervisor also changes moves (§10) |
| **5** | Multiplicity scope | (a) per comparison set and planner / condition, at 0.05 each (the approved Phase 3 / 4 practice); (b) a global 0.05 split over the 7 units (≈ 0.007 each, with lower power) | (a) for continuity with the approved rules, stated explicitly in the report |
| **6** | Families and α | As §7: the Stage 3a and Phase 4 splits, the RO veto added to P4 | As §7 |
| **7** | Ceiling handling | The Phase 4 §5.4 rule, applied to every completion bound | Approve as written |
| **8** | H-L12 decision rule | (a) the 3b / 3e rule (as H-L2); (b) descriptive only | (a) |
| **9** | Robot–robot conflicts | (a) pre-registered secondary with nominal intervals (K^ctrl and K^arb); (b) a tested family (needs α, which raises the union bound above 0.05) | (a) |
| **10** | P2 m = 400 | (a) excluded (the 2026-11-23 instruction: P2 disabled); (b) a separate secondary arm | (a) |
| 11 | MADQL-v2 RRT\* without the supervisor | (a) not run (not a development claim); (b) add 540 episodes (≈ +1.4 h) for a complete v2 description | (a) |
| 12 | Checkpoint preservation | The MADQL-v2 checkpoints live only in git-ignored `results/dev6a/phase4/training/` | Back them up outside the repository before Phase 6 (not to `models/`) |
| 13 | Authorization sequence | §9 | Approve the sequence; then the freeze and the execution need separate approvals |

## 14. Limitations

- **One held-out block, one scenario family** (32 × 32, density 0.15, 8 robots, 16 tasks). Phase 6 does not cover E5 / E6.
- **Fixed checkpoints, 3 seeds per configuration.** Statements are about these policies, not about the algorithms in general.
- **Training disjointness** is not verifiable at instance level.
- **The RO denominator** differs slightly from the Phase 4 post-hoc one (§6.3). The PPO and supervisor rates quoted in §10 use an approximate denominator.
- **Earlier phases had Block C expectations.** Phase 6 tests them on new instances, but the hypotheses were formed on Block C.
- **Runtime projections** are ±30 % (RRT\* tails).
