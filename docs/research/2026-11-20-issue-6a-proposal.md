# Issue 6A proposal — System-wide algorithm review, optimisation and validation (2026-11-20, revision 3)

**Status.**
- **Phase 6A.0** (instrumentation replay, seed-cluster re-analysis, evidence correction) was approved and is **complete**. Its verified results are in **§20**, and §3.1 and F15 are updated from them.
- **Phase L1** (SCB-C implementation and simulation study) is complete; results in **§21**. Its recommendation: **not yet ready for block C** (§21.9).
- **L1 follow-up** (validity of the simultaneous procedure, new selection block S = seeds 13 000–13 079, out-of-sample rule comparison, α allocation) is complete; results in **§22**. It corrects §21 on v3 (§22.2).
- **SCB-C v5 is approved and frozen** (2026-11-21): the one-SE rule on blocks A ∪ S, with v2 completion elimination at α = 0.05 and no conflict test. Specification in **§23**.
- **Block C development evaluation** (2026-11-21) of the four changed PPO picks, old vs new, is complete; results in **§24**. The plan was frozen before the run. There were no safety failures. Conflicts per episode are lower in two of four runs (Holm), and no completion difference was detected. SCB-C v5 is unchanged. Clarifications after review are in §24.7. **Block D has not been run.**
- **Block D final evaluation of H-L1** (2026-11-21) was run once under the frozen protocol; results in **§25**. All 8 220 episodes passed every validity check, and there were 0 safety events. On A\*, conflicts per episode are lower for mappo_seed1–3 (Holm, F1). No completion loss was detected; mappo_seed1's completion gain is MC-sensitive. ippo_seed1's conflict increase is not rejected and is MC-sensitive. Two deviations are documented (§25.7). The `models/` decision is still pending.
- **Phase 0: Comprehensive Diagnostics** (6A.0, stored-data diagnostics and the deterministic replay, consolidated by the student's decision) is **complete**; consolidated findings in **§26**.
- **Roadmap priority changed** (student decision, 2026-11-21; numbering kept, Issues #7 and #8 unchanged): Phase 3 (Execution Supervisor) is next, then Phases 1, 2, 4 (conditional), 5 (conditional) and 6. See **§27**. **Phase 3 is approved for design only**; the design is in `docs/research/2026-11-21-issue-6a-phase3-execution-supervisor-design.md`.
- **Phase 3, Execution Supervisor: implemented and software-tested** (2026-11-21; D1–D11 approved): engine stage before the shield, off by default and byte-identical when off; 61 new tests; full fast suite 1 170 passed. **Not yet evaluated**: calibration and Stages 3b–3f need approval. See **§28**.
- **Phase 3, Stage 3a calibration: complete and frozen** (2026-11-21): calibrated on 2 700 stored Block B greedy episodes only (0 replay mismatches); frozen per planner, A\* W = 24 and O (14, 3, 7), RRT\* W = 32 and O (8, 3, 4), the other parameters and escalation (off) unchanged. Go / no-go criteria for 3b, 3c and 3e are recommended and need approval. See **§29** and `docs/research/2026-11-21-issue-6a-stage3a-calibration-freeze.md`.
- **Phase 3, Stages 3b, 3c and 3e: executed and analysed** (2026-11-21/22; 31 320 episodes, 0 errors, 0 safety events, replay identical). Completion with the supervisor (A\*): MADQL 0.624 → 0.997, IPPO 0.987 → 0.999, MAPPO 0.988 → 0.998 (all Holm-rejected gains); RRT\* the same pattern. 3b and 3e **go** and 3c keeps R, S+O and D (O not shown) under the *recommended* criteria, which await approval. See **§30** and `docs/research/2026-11-22-issue-6a-phase3-evaluation.md`.
- **Phase 3 complete** (student decisions, 2026-11-22): forward configuration R + S + D; criteria approved; development evidence only. See **§30.1**.
- **Phase 1, Replanning optimization (P1): complete** (2026-11-22; 17 280 Block C episodes, 0 errors, G0 passed, replay identical). The bounded rejoin cuts planner calls by 59–86 % with the supervisor. However, MADQL completion falls (Holm) and high-priority flow time rises for every family (Holm veto), so the verdict is **reject** in all four conditions. Repair stays off by default. See **§31** and `docs/research/2026-11-22-issue-6a-phase1-replanning.md`.
- **Phase 2, RRT\* budget control (P2): complete, inconclusive** (2026-11-22). The rule is optional and off by default; a stopped run is an exact prefix of the full-budget run. The microbenchmark selected patience m = 400 as the only candidate within the 1 % path-quality screen. Block C (2 400 episodes, 0 errors, G0 passed, replay identical, 0 invalid reference paths) cuts RRT\* iterations per robot-step by 10–16 % (Holm) and RRT\* time by 14–24 %, with no Holm-detected harm. Verdicts: baseline **adopt**; learned policies **inconclusive**, because IPPO and MAPPO completion sits at the ceiling and the bootstrap-t intervals are degenerate. P2 is not the forward default; m = 400 is carried to Phase 6. See **§32** and `docs/research/2026-11-22-issue-6a-phase2-rrt-budget.md`.
- **Phase 4, MADQL optimization (MADQL-v2): complete** (2026-11-22/23). Diagnostics found a weak correction signal at self-loop STAY transitions, TD noise larger than the decisive Q gaps, and perceptual aliasing. A 2 × 2 design (n-step returns n = 3; target period ×4) was trained with 3 seeds each; Block S selected **n3**. Block C (5 040 episodes, 0 errors, G0 passed, `hist` identical to Phase 1, replay identical): completion without the supervisor 0.624 → 0.860 (Holm); with R + S + D, flow time −19 % (A\*) / −22 % (RRT\*), supervisor changes −40 % / −51 % and planner calls −9 % / −6 % (all Holm); RRT\* completion +0.0055 (Holm). Verdict **adopt** in all three conditions, so MADQL-v2 n3 is the forward MADQL candidate for Phase 6, with the historical MADQL kept as the reference. Cost: a post-hoc analysis shows about 2× more robot–obstacle conflicts per move (nominal; all prevented by the arbiter; 0 collisions), which Phase 6 must pre-register as a veto. Phase 5 is not justified by the evidence. See **§33** and `docs/research/2026-11-22-issue-6a-phase4-madql.md`.
- **Phase 4 decision and Phase 5** (student, 2026-11-23): MADQL-v2 n3 is the proposed MADQL configuration for Phase 6, with the historical MADQL as the reference. Phase 6 must evaluate robot–obstacle conflicts per move as a pre-registered veto, and the Phase 4 Block C results are not independent confirmation. **Phase 5 (A\* optimization, P3) is not pursued: not justified** on the Issue 6A evidence. The A\* implementation and its defaults are unchanged. See **§34**.
- **Phase 6 (integrated confirmatory validation): prepared, not ready to execute** (2026-11-23). The held-out block V = 14 000–14 199 (RRT\* subset 14 000–14 059) is verified unused, but not yet reserved. The draft protocol covers H-L2, H-v2 and H-L12: 22 080 episodes, about 5–5.5 h. The gated runner and analysis are prepared, with 23 tests, and the Block C anchor check reproduces 514 / 514 stored records. Student decisions are needed first, notably the robot–obstacle veto rule. See `docs/research/2026-11-23-issue-6a-phase6-readiness.md` §13. No Phase 6 episode has been run.
- **Phase 6, integrated confirmatory validation: complete** (2026-11-23; held-out Block V, 22 080 episodes, 0 errors, G0 passed, replay 442 / 442 identical, analysis byte-identical). **Supervisor R + S + D confirmed** (H-L2 go on A\* and RRT\*). **MADQL-v2 n3 not confirmed**: H-v2 FA and FR rejected by the pre-registered robot–obstacle veto (+0.55 and +0.37 per 1 000 moves); NA adopt. The forward MADQL therefore stays the historical MADQL. **Integrated Issue 6A system confirmed as declared** (H-L12 go on both planners). D12 waived by the owner (not satisfied). See **§35** and `docs/research/2026-11-23-issue-6a-phase6-results.md`.
- **Issue 6A closed** (2026-11-23): see **§36** and `docs/research/2026-11-23-issue-6a-closure.md`.
- **Everything else is still a proposal:** P3–P4.
- **Unchanged:** no source code, configuration, historical record, checkpoint or existing document was changed for this proposal or for 6A.0.

**Revision 2** answers the student's six clarification requests (2026-11-20):
1. diagnostic counts reconciled, with explicit definitions and denominators (§3.1);
2. L1 rebuilt on simultaneous inference with multiplicity control (§4);
3. seed blocks C and D defined, with independence checks (§12);
4. L2 isolated by a factorial design (§5);
5. costs separated by work item (§14);
6. decisions grouped by when they can be made (§19).

It also **corrects two errors of revision 1**:
- **Failure-mode counts.** The IPPO and MAPPO failure-mode counts were taken from a console printout truncated to its last 30 lines, which omitted three IPPO categories. The corrected counts are in §3.1.
- **MADQL sample.** The sample is one policy (MADQL seed 1), not "MADQL".

**Labels.** Every statement carries one label:
- **[Observed]**: directly supported by the code or by stored records (path given);
- **[Interpretation]**: an explanation consistent with the evidence, not yet established;
- **[Hypothesis]**: a proposed change whose effect must be tested.

**Evidence gathered for this proposal** (read-only; nothing written to `results/`):
1. Analyses of stored records: block-A validation logs, block-B records, Issue #3/#4 records.
2. (Revision 2) A diagnostic replay of every PPO greedy block-B episode with undelivered tasks (218 episodes), plus 30 MADQL episodes. Superseded by the 6A.0 replay of all 928 failures (§20). It ran under the unchanged evaluation code and stored checkpoints, in about 2 minutes, writing to the session scratchpad only. The replays reproduce the stored outcomes: termination agrees in 248 of 248 episodes.
3. Scenario-generation checks (generation only, no simulation): correlation across levels, uniqueness, and the seed usage of all stored records.

**Correction to the post-Issue-6 report.** That report attributed the greedy shortfall to "symmetric standoffs". The replays show these are one of three failure modes, and not the most frequent (§3.1).

---

## 1. Executive summary

- **[Observed]** The system is safe in every recorded episode and efficient at the planning level.
  - 0 physical collisions in all Issue #4–#6 episodes.
  - A\* takes 0.09 ms of a 0.54 ms mean step.
- **[Observed]** The remaining deficits sit at interfaces:
  - **Selection on noise.** Checkpoints are selected by completion differences below the measurement noise.
  - **Deterministic execution artefacts.** Greedy learned policies get stuck in oscillations, frozen states and mutual-cancellation loops, all broken by sampling.
  - **Robot-blind planning.** Planning ignores other robots.
  - **Replanning on every deviation.** Every policy deviation triggers a replan.
- **[Observed] A methodological finding.** The three levels of one scenario seed share map, robot starts and tasks. The Issue #5/#6 bootstraps resampled seed × level instances as independent, so the historical CIs were too narrow: by a median factor of 1.13 over 365 per-policy comparisons (§20.4). No headline conclusion changes. Per-level analyses are unaffected. 6A uses the seed as the statistical unit.
- **[Hypothesis]** The first two improvements need no retraining, and are developed and evaluated on a new development block C:
  - **L1:** a noise-aware checkpoint-selection rule with multiplicity control;
  - **L2:** an execution supervisor (ES).

  They are confirmed once, frozen, on an untouched block D.
- **Roadmap.** Issue 6A is a ninth issue entry, a **roadmap revision requiring the student's approval** (§18). Issues #7 and #8 keep their numbers, identities and scope.

## 2. Verified current-system assessment

Each subsystem: how it works, its interface, and where it is implemented.

| Subsystem | How it works [Observed, from the code] | Interface |
|---|---|---|
| Environment / engine | Per step: allocation → monitor (replanning) → view → controller proposal → optional shield → **arbiter** → motion; conflicts are detected on the proposal reaching the arbiter | `sim/engine.py`, `sim/view.py` |
| Arbiter | Emergency stop: cancels **every** move involved in a conflict event, plus a cascade into frozen cells; no right-of-way (Lemma 3.1, no physical collision) | `sim/conflicts.py:arbitrate` |
| A\* | 4-connected, Manhattan heuristic, heap ordered `(f, −g, counter)`, lazy deletion; cost 1 or 1 + λ near obstacles | `planning/astar.py` |
| Constraints | `B_i` = sensed obstacles ∪ their constant-velocity predictions ∪ the robot's own stall blocks and the resolver's `avoid` cells; **other robots are not included** | `planning/constraints.py` |
| Monitor | Replans on T1 goal change, T2 path invalid, T3 off-path, T4 stall (4 steps), T5/T6 retry, T7 deadlock detour, T8 endpoint; at most one call per robot per step | `planning/monitor.py` |
| RRT\* | Continuous plane; runs **all** `max_iterations` per call (anytime); brute-force neighbours (default, from E1b); BFS reachability guard first | `planning/rrtstar.py`, `procedure.py` |
| Allocation | Bertsekas ε-scaling auction or sequential auction; weighted priority with aging, hysteresis | `allocation/` |
| Baseline controller | Follows the suggestion unless the target cell is occupied or predicted to be, or another robot with higher `(π̃, −i)` claims it; guaranteed free of robot–robot conflicts; not a deadlock rule | `control/baseline.py` |
| Deadlock resolver | Wait-for graph (edge i → j if i's next planned cell holds j); persistent or recurrent cycles; the lowest-priority member detours (T7). **Enabled only for the baseline** in evaluation | `planning/deadlock.py`; `marl/evaluation.py:_system` |
| Endpoint recovery | Retreat and parking for idle robots on task endpoints (T8); on for every controller | `planning/endpoint.py` |
| Predictive shield | Replaces a move into a sensed or predicted obstacle cell by stay; robot–obstacle only | `control/shield.py` |
| Learned controllers | Masked arg-max with ties to the lowest index (greedy), or sampling / ε-greedy (stochastic); `obs-v1` (local window with obstacle predictions, other robots, own path and goal, suggested action); **no identity, no other robots' intents, no time or stall information** | `marl/policy.py`, `marl/observation.py` |
| Selection | Lexicographic on block A: overall completion, then mean discounted base return; every 50 k steps; all 20 step checkpoints stored per run | `marl/evaluation.py`, `ppo_trainer.py` |
| Scenario generation | `LEVELS[level].sample(seed)` → `e5_scenario(seed, dynamics, …)`: for one seed the **map, robot starts and scheduled tasks are identical across the three levels**; levels differ only in moving obstacles (obstacles, full) and arrivals and map events (full) | `marl/scenarios.py`, `experiments/allocation_runs.py` |

## 3. Evidence-backed findings and limitations

**Defect** = the code does something it should not; **algorithmic** = a property of the method; **environment-induced** = arises from the scenario dynamics.

| # | Finding | Label | Kind | Evidence | Enough to act? |
|---|---|---|---|---|---|
| F1 | Completion-first selection decides on noise and ignores conflicts | Observed | Evaluation protocol | Block-A 95 % half-width of pooled completion ≈ 0.0074 (one checkpoint), ≈ 0.0103 (paired difference of two). MAPPO picks differed from alternatives by ≈ 0.005. Completion and conflicts are uncorrelated across checkpoints after 150 k steps (median Spearman −0.03) | Yes (L1) |
| F2 | Greedy failures consist of three deterministic failure modes, broken by sampling | Observed | Execution × environment | §3.1 | Yes (L2) |
| F3 | Failures concentrate on the static level for PPO | Observed (counts); Interpretation (cause: repeated observations in static scenes) | Environment-induced | 57/240 IPPO and 36/240 MAPPO static instances fail greedily; all 93 complete in all three stochastic replicates | Supports L2 |
| F4 | The arbiter's cancel-all rule makes mutual proposals repeat | Observed (mechanism) | Interface | `arbitrate`; conflict loops: 68 of 168 MAPPO stuck robots | Yes (L2, right-of-way) |
| F5 | Planning is robot-blind | Observed | Design choice | `build_constraints` | Needs an experiment (P4) |
| F6 | Learned controllers multiply planner calls | Observed | Interface | 5.5 (baseline) vs 10.8–11.3 (PPO) and 33.1 (MADQL) calls per 100 robot-steps (block B, A\*) | Yes for RRT\* (P1) |
| F7 | RRT\* spends its full budget every call | Observed | Configuration | 125–173 ms per call (E2, n = 2 000); `stop_on_first_solution = False` | Yes (P2) |
| F8 | A\* expands extra nodes on dense and room maps | Observed | Heuristic | Expansions ÷ path length median 1.4–7.5 (64–128 cells), 1.0 on open/warehouse | Low priority |
| F9 | MADQL is unstable and below PPO even when sampled | Observed | Algorithm + training configuration | Validation jumps up to 0.25; stochastic 0.888–0.918 vs PPO 0.999 | Investigation first |
| F10 | MAPPO's early checkpoints are robot–obstacle-conflict-heavy | Observed | Training dynamics × F1 | 41–60 robot–obstacle conflicts per episode (MAPPO 1, 3) vs 1–5 elsewhere | Solved by L1 if confirmed |
| F11 | Option C reward weights do not change completion | Observed | — | Cell − default CIs within ±0.012 | No reward work justified |
| F12 | Allocation is not a measured bottleneck | Observed (Issue #4 data) | — | Auction vs sequential: travel per task 38.1 vs 38.8 | Measure first (G3) |
| F13 | The resolver helps the baseline | Observed | — | +0.015 completion (A\*), static level | Test for learned controllers (L2) |
| F14 | The shield removes robot–obstacle conflicts without detected cost | Observed | — | −60 to −91 conflicts per episode (MAPPO 1, 3); wait-ratio Holm p = 1 | A5 hypothesis only |
| **F15** | **Levels of one seed are correlated instances** | Observed | Evaluation protocol | Same map, starts and tasks across levels. Per-seed effects correlate 0.37–0.69 between levels. **6A.0 re-analysis (§20.4):** over all 365 historical per-policy comparisons, the seed-cluster CI is a median 1.13× the instance-bootstrap width (IQR 1.02–1.28; range 0.62–1.44). The 1.31× of revision 2 was one comparison (1.29× with the standard bootstrap) | Done (§20.4); seed is the unit in 6A |

**Measurement gaps** (to fill in 6A.0):
- **G1 — no deadlock measure without the resolver.** No deadlock frequency or duration is recorded for controllers without the resolver. The record field `deadlock` only says whether the resolver was enabled.
- **G2 — no record of why robots stand still.** Per-robot stall, oscillation and frozen events are not recorded.
- **G3 — no allocation statistics for learned controllers.**
- **G4 — planner calls are not split by trigger.**
- **G5 — no memory measurement for evaluation.**

### 3.1 Diagnostic reconciliation

**Superseded by the 6A.0 full replay (§20).** Revision 2 replayed all 218 PPO failures but only a sample of 30 MADQL episodes (MADQL seed 1 only). Phase 6A.0 replayed **all 928** greedy block-B failures (218 PPO + 710 MADQL, all nine policies) and added the genuine wait-for-cycle deadlock class.
- The revision-2 PPO counts are confirmed (§20.2), with one refinement: IPPO's 5 "other" robots are 4 that keep moving plus 1 mostly stationary.
- Definitions, rules and every table are in §20.

## 4. L1 — checkpoint-selection proposal

**Current rule [Observed].** Lexicographic maximum on block A (seeds 10 000–10 019 × three levels = 60 episodes): pooled completion, then mean discounted base return. It is preserved for every historical result: `best.pt` / `best.json` are never changed.

**Problems [Observed].**
1. Completion differences between checkpoints are below the noise.
2. Choosing the maximum of 20 noisy estimates is **optimistically biased**: the selected checkpoint's block-A score overstates its true completion.
3. Conflicts are ignored.
4. Treating the 60 episodes as independent understates uncertainty (F15): 20 seeds are the independent units.

**Methodology options compared** (K = 20 checkpoints per run, paired on the same block-A seeds):

| Approach | Controls multiplicity? | Handles "which is best" without optimism? | Practical equivalence? | Assessment |
|---|---|---|---|---|
| (a) Non-inferiority vs the observed best, one test per checkpoint (revision 1) | No (19 tests) | No (the best's estimate is inflated) | Only via a fixed δ | Rejected |
| (b) Bonferroni / Holm on (a) | Yes, conservatively | No | Fixed δ | Better, still anchored on an inflated best |
| (c) **Simultaneous CIs for all pairwise differences by a paired seed-cluster bootstrap, studentised max-T** | Yes (family-wise over all K(K−1)/2 pairs) | **Yes**: no single estimated best; a checkpoint is excluded only if some other checkpoint is shown better | Yes: equivalence is shown from the same simultaneous bounds | **Proposed** |
| (d) Hsu's multiple comparisons with the best (MCB), normal theory | Yes | Yes | Yes | Same logic as (c), but assumes normality; completion near 1 is skewed, so the bootstrap version (c) is preferred |
| (e) Bayesian ranking (posterior probability of being best) | Implicit | Yes | Yes | Valid but needs priors and modelling choices; not proposed |

**Proposed rule "SCB-C" (simultaneous comparison band, then conflicts) [Hypothesis].**

Per run, using only the stored block-A validation records (no new episodes):

1. **Units.** The 20 block-A seeds. For each seed s and checkpoint k: delivered and released summed over the three levels, and conflicts per episode averaged over the three levels.
2. **Completion uncertainty.** B = 10 000 paired bootstrap resamples of seeds, the same resample for all checkpoints, so pairing is kept. For every pair (j, k):
   - D_jk = c_j − c_k, the pooled-completion difference;
   - SE_jk from the bootstrap.

   The (1 − α) quantile q of T* = max over pairs |D*_jk − D_jk| / SE_jk gives simultaneous intervals D_jk ± q·SE_jk. α = 0.05.
3. **Eligible set E (no optimistic "best").** k ∈ E iff no checkpoint j has a simultaneous lower bound D_jk − q·SE_jk > 0, i.e. no other checkpoint is shown better than k. With probability ≥ 1 − α, the truly best checkpoint stays in E.
4. **Practical equivalence (secondary, hypothesis).** k is *practically equivalent* if the simultaneous upper bound D_jk + q·SE_jk < Δ for every j.
   - **Δ = 0.005 is a hypothesis, not an accepted threshold.** With 20 seeds, simultaneous bounds are expected to be wider than ±0.01, so equivalence at Δ = 0.005 will rarely be demonstrable on block A.
   - Δ is therefore **not** used for selection in this rule. It is examined on block C as a sensitivity analysis.
   - Alternatively, a larger selection set (§19 B) could make it usable.
5. **Conflicts as the secondary criterion.** Within E, the same simultaneous procedure is applied to per-seed mean conflicts per episode (lower is better). This gives E₂ ⊆ E: the checkpoints not shown worse in conflicts than any other member of E.
6. **Exclusion.** Any checkpoint with a physical collision on block A is excluded (never observed so far).
7. **Ties (deterministic, declared before use).** Within E₂, choose the **latest step**. Rationale: it does not re-use a noisy metric and prefers the most-trained policy.
   - Sensitivity analyses on C, not used for selection: minimum point-estimate conflicts; maximum return.
8. **Reporting.** The selected checkpoint's performance is never reported from block A, because of selection bias. Only block C or D estimates are reported.

**Validating the procedure before any C evaluation (cheap, no simulation episodes) [Hypothesis].** A simulation study on synthetic paired data generated from the empirical block-A seed-level distributions:
- **(i)** coverage: is the true best in E at least 95 % of the time?
- **(ii)** the size of E;
- **(iii)** how often a checkpoint 0.01 worse in completion is eligible.

This checks the procedure's statistical properties without looking at any new episodes.

**Freezing L1 before independent evaluation.**
1. Write the pre-registration: the rule as above, α, B, the bootstrap seed, the unit, the metrics and the tie order. Record its SHA-256 in the verification log.
2. Implement the rule with unit tests on synthetic tables and the simulation study.
3. Apply it to the stored block-A logs of all runs. Write the selected step per run, by reference to the existing `checkpoints/step_*.pt` with their SHA-256, to `results/dev6a/selection/`. Record that file's SHA-256.
4. Only then evaluate the old-rule and new-rule checkpoints on block C (development).
5. Any change after C is a new protocol version, documented, and the C results are labelled as development.
6. The frozen rule's final comparison is on D, once.

**Illustration only (in-sample, block A).** A simple δ = 0.008 band (not the proposed rule) changed 11 of 14 PPO selections, to checkpoints with 0.6–1.5 block-A conflicts per episode instead of up to 40. This motivates L1; it is not evidence that SCB-C works.

## 5. L2 — execution-supervisor proposal

**Why failures happen.**
- **[Observed + Interpretation] Oscillation (the most frequent mode).** A two- or three-cell limit cycle of a deterministic policy: observation A → move to B → observation B → move back.
- **[Observed + Interpretation] Frozen.** The policy repeats STAY although the suggested next cell is free. In 58 of 58 IPPO and 9 of 10 MAPPO frozen cases the suggested cell was free, so the planner is **not** the cause. While the robot waits, nothing in `obs-v1` changes: no clock, no stall counter, no identity.
- **[Observed] Conflict loop.** Two robots propose the same cell or a swap. The arbiter cancels both, the observations repeat, and so do the proposals. 40 % of MAPPO stuck robots.
- **[Observed] Does stochastic execution fix it consistently?**
  - All 93 PPO static-level failures complete in all three replicates.
  - On the dynamic levels: 23/26 and 32/38 MAPPO, and 24/26 and 27/35 IPPO failures complete in all replicates; 2–4 instances per group fail in every replicate.
  - MADQL: only 5–30 of 232–240 instances per level complete in all replicates.

**Where it belongs [Interpretation].** In the execution layer: policy → **ES** → shield → arbiter. The policy would need retraining. The arbiter's emergency-stop semantics, Lemma 3.1, stay untouched.

**Mechanism.** It is optional and off by default; with ES off, historical behaviour is byte-identical. All parameters are configuration values whose **values are not assumed** (setting them is in §19 C).

1. **R — right-of-way.** If several robots' proposed moves target the same cell, or two robots propose a swap, only the robot with the highest key keeps its move; the others' moves become **stay**.
   - **Key:** (π̃, fairness counter, −id). π̃ is the aged effective priority of the robot's task; an idle robot has π̃ = 0. The fairness counter is the number of consecutive R-losses; a robot that has lost `L_max` times in a row wins ties against robots with fewer losses. The final tie-break is the lower id.
   - **Starvation prevention:** aging raises π̃ with waiting time, and the fairness counter bounds consecutive losses among equal priorities.
2. **S — stall breaker.**
   - **Trigger:** a robot with a task whose static goal distance `D̄` has not decreased for `W` consecutive steps, and which was not an R-loser in those steps, is stalled. `W` is set from data on block C (proposal: the 99th percentile of no-progress run lengths in *successful* greedy episodes, so normal waiting rarely triggers it) and examined at a grid of values in the ablation.
   - **Action, for `H` steps:** the planner's suggested action, filtered by the baseline's rules 3–5 (occupied cell, predicted obstacle, lower right-of-way → stay).
   - **Variant S-rand:** with probability `p` per stalled step, a seeded sample from the policy's own masked distribution (PPO), or a uniform allowed action (MADQL). Stream seeded by (instance, robot, step, replicate).
3. **D — deadlock resolver.** The existing resolver, enabled for learned controllers by the evaluation switch `deadlock = True` (no new code). It handles wait-for cycles by T7 detours.

**Safety precedence [Observed design; property tests in §13].**
- ES only maps a proposal to **stay** (R, and the stay branch of S) or to the **baseline-filtered planner suggestion** (S). The arbiter (Lemma 3.1) still prevents any physical collision.
- **Pipeline:** policy proposal → ES → shield (if on) → arbiter → motion. Every ES output passes through the shield and the arbiter, so ES cannot bypass either.

**Logging.** Per step and robot:
- the original policy proposal (already `StepOutcome.actions`);
- the ES output (new field);
- the shield output (existing events);
- the arbitrated action (existing `arbitrated_actions`);
- the reason code: R-loss, S-trigger, S-hold, S-rand, D-detour.

Per episode: interventions by reason, stalled steps, R-losses per robot (fairness), cycles detected, resolved and their durations (G1).

**Experimental design (paired; no retraining; no reward change).**
- **Policies:** the 9 historical policies, plus the L1-reselected ones if L1 is approved for C.
- **Instances:** block C.
- **Minimum comparison set requested by the student:**

  | Arm | Execution | ES |
  |---|---|---|
  | G | greedy | off |
  | S | stochastic (3 replicates) | off |
  | G+ES | greedy | R + S + D |
  | S+ES | stochastic (3 replicates) | R + S + D |

  This 2 × 2 (mode × ES) estimates the ES effect under each mode and their interaction: does ES add anything once sampling is used?
- **Isolation of the three mechanisms.** A full 2³ factorial of {R, S, D} under greedy execution: 8 arms, G and G+ES included. It gives the main effects and the two- and three-way interactions. Interactions are expected: D only acts on cycles, which R can prevent, and S may pre-empt D.
  - A leave-one-out design (4 arms) would identify each mechanism's marginal contribution but not interactions.
  - Single-mechanism arms alone (3) would miss redundancy.
  - Recommended: the full factorial under greedy; under stochastic, only S and S+ES.
- **S-variant arm:** greedy + R + S-rand + D (3 replicates), to compare deterministic and randomised stall breaking.
- **Parameter grid** (development only, on C): `W` at three values and `H` at two, with the full ES under greedy.
- **Shield-on check:** G vs G+ES on the full level with the shield on, to show ES and the shield compose without loss.
- **Metrics per arm, policy, level and planner:**
  - completion and undelivered tasks;
  - physical collisions;
  - conflicts by type (vertex, swap, robot–obstacle);
  - ES, shield and resolver interventions;
  - stalls, oscillation and cycle counts and durations (G1/G2);
  - steps and termination;
  - wait ratio, flow time, planner calls by trigger;
  - runtime.

## 6. A\* and RRT\* improvement opportunities

| ID | Opportunity | Label | Evidence | Expected effect | Cost / risk |
|---|---|---|---|---|---|
| P1 | **Repair before replanning (T3)**: rejoin the remaining reference path within a bounded radius, or follow the static cost-to-go; replan only if repair fails or dynamic cells changed | Hypothesis | F6 | Fewer planner calls for learned controllers; less RRT\* time | Medium; changes monitor semantics (off by default) |
| P2 | **RRT\* budget control**: first solution plus a refinement budget, or a cost-improvement stopping rule; informed sampling after the first solution | Hypothesis | F7 | Large per-call savings | Medium; model G7–G9 restated if adopted |
| P3 | **Exact static heuristic for A\*** (cached `DistanceOracle` field; admissible and consistent: dynamic blocks only remove edges, costs ≥ 1) | Observed (argument) / Hypothesis (benefit) | F8 | Expansions → ≈ path length on static maps | Low; identical optimal costs (test) |
| P4 | **Space-time reservations** (windowed cooperative A\*, auction priorities) | Hypothesis | F5 | Fewer robot–robot conflicts and standoffs for every controller | High; incomplete; implies retraining |
| P5 | D\* Lite | Interpretation: low value | A\* 0.2–1.2 ms per query | Small | Defer |
| P6 | RRT\* tree reuse (RRTˣ-style) | Hypothesis | F6, F7 | Cheaper replans | High; only if P1 + P2 insufficient |

**RRT\*'s role [Observed].** On this lattice its executed paths are never shorter than A\*'s (E1a). It stays as the required comparator. 6A should make it affordable, not try to make it win.

## 7. MADQL, MAPPO and IPPO improvement opportunities

- **MADQL**
  - **[Observed]** Most of its gap is greedy-execution failure (0.58–0.66 → 0.89–0.92 with ε = 0.05).
  - **[Hypothesis]** ES raises greedy MADQL completion without retraining; the L2 factorial measures how much.
  - **I1 (no training):** Q-value gaps on stored block-A states.
  - **Retraining:** a new configuration "MADQL-v2", only if approved. Historical MADQL stays the reference.
- **MAPPO / IPPO**
  - **[Observed]** Stable learning; no reward-weight effect; explained variance 0.88–0.95; entropy 0.10–0.23.
  - **[Interpretation]** Retraining is not justified now: the deficits (F1, F2, F10) lie outside the learner.
  - **[Hypothesis, later]** Retrain only if P4 or an intent-sharing observation (obs-v2) is approved, as new configurations.

## 8. Task-allocation improvement opportunities

- **[Observed]** The auction is optimal (ε-scaling) or ≥ 0.76 of the optimum (sequential) on 280 matrices; in-system differences are small (F12).
- **[Hypothesis, measurement first]** Record allocation statistics for every controller (G3): reassignments, auctions, waiting censoring, goal clustering. Only then consider congestion-aware bidding.

## 9. Deadlock-resolution and collision-shielding opportunities

**Resolver.**
- **[Observed]** It removes the baseline's static-level failures.
- **[Observed]** It cannot see oscillations or frozen robots: they are not cycles. This is why D alone is expected to be insufficient for learned controllers (tested by the factorial).

To measure for every controller (G1):
- cycles, durations, resolution success and time;
- repeats;
- temporary waiting (blocked < T4) vs stall vs cycle.

**Shield.**
- **[Observed]** Robot–obstacle only; never creates a move.

To measure:
- **false interventions:** the intervened cell was not entered at t + 1;
- **missed conflicts:** robot–obstacle events with the shield on.

The robot–robot extension (A5) overlaps with R and is considered only after L2.

**Safety statement [Observed].** Zero observed collisions is evidence for these scenarios only. The arbiter's Lemma 3.1 is the proved property.

## 10. Architecture and interface improvements

| Interface | Issue [label] | Proposal [Hypothesis] |
|---|---|---|
| Policy → arbiter | No right-of-way for learned proposals; cancel-all repeats loops [Observed] | ES between policy and shield; arbiter unchanged |
| Planner → policy | Every deviation causes a replan [Observed] | Repair-first monitor option (P1) |
| Planner ↔ other robots | Robot-blind constraints [Observed] | Reservations (P4) |
| Deadlock ↔ controllers | Resolver bound to the controller name in `evaluation._system` [Observed] | Explicit configuration field for every controller |
| Metrics ↔ engine | Deadlock, stall, allocation and trigger data missing [Observed] | Additive record fields (G1–G4) |
| Duplicate computation | `sense()` can run several times per robot and step [Observed in the code]; total cost small | Per-step cache; low priority |

## 11. Proposed experimental sequence

| Step | ID | What | Data | Training |
|---|---|---|---|---|
| 0 | 6A-M | Instrumentation G1–G5; register blocks C and D; full replay classification of all 710 MADQL failures; non-destructive re-analysis of the Issue #5/#6 pooled CIs with the seed-cluster bootstrap (new output directory; historical analyses unchanged) | Stored records; replays | No |
| 1 | 6A-L1-dev | SCB-C simulation study → freeze v1 → re-selection → old vs new on C | Block A (selection), block C | No |
| 2 | 6A-L2-dev | 2 × 2 mode × ES; greedy 2³ factorial; S-rand; parameter grid; shield-on check | Block C | No |
| 3 | 6A-L12-dev | L1-selected policies × chosen ES configuration (2 × 2 mode × ES) | Block C | No |
| 4 | Freeze | L1 rule, ES configuration and parameters, arms, hypotheses, analysis code hash | — | — |
| 5 | 6A-confirm | Single pre-registered run on block D | Block D | No |
| 6 | 6A-P | P1–P3 development (block C, E1a/E2 query sets, pilot) | Block C | No |
| 7 | 6A-P4 | Only if approved | Block C (then D) | Only if approved |

## 12. Statistical methodology and data separation

**Seed blocks.**

| Block | Seeds | Purpose | Rules |
|---|---|---|---|
| Training | 0–9 999 | training | unchanged |
| A | 10 000–10 019 | **checkpoint selection only** | unchanged; never used to report performance |
| B | 10 020–10 099 | historical development comparison (Issues #5–#6) | inspected; **reference only**, not used to confirm 6A hypotheses |
| **C (new)** | **11 000–11 119** (A\*); RRT\* subset **11 000–11 039** | **development and validation**: protocol development and the independent development evaluation of L1, L2 and L1+L2; may be used repeatedly | every use logged (experiment ID, protocol version); results labelled "development" |
| **D (new)** | **12 000–12 199** (A\*); RRT\* subset **12 000–12 059** | **final held-out**: used only after L1, ES configuration, parameters, arms, hypotheses and analysis code are frozen; **once** | no tuning; any defect found after D is reported, and a re-run needs the student's approval and is labelled as such |
| Test | 20 000–20 999 | Issue #8 (E5/E6) | unchanged |

**Seed generation and independence.**
- **[Observed]** Seeds 11 000–12 199 occur in no stored record (scan of every `results/` JSONL and CSV file).
- **[Observed]** The static-level specifications of blocks A, B, C and D are all distinct: 0 duplicates and 0 overlaps among 420 seeds.
- **[Observed]** Parameters are drawn by `default_rng([crc32(level name), seed])` and maps by `e5_scenario(seed, …)`, so different seeds give separately seeded pseudo-random streams.
- **[Observed]** **Within a seed, the three levels share map, robots and tasks** (F15). Different seeds are therefore independent draws of the scenario family, while levels of one seed are correlated.
- **Consequences:**
  - the **seed is the statistical unit**, with levels nested within it;
  - pooled analyses use a seed-cluster bootstrap;
  - per-level analyses use seeds within the level;
  - fixed design factors (32 × 32 maps, density 0.15, 8 robots, 16 tasks, 400 steps) are the same for every seed and are part of the scenario definition, not a source of correlation between seeds.
- **Contiguous ranges** are acceptable because the generator hashes the seed. The 6A.0 tests re-check uniqueness and disjointness for C and D.

**Scenario composition and episode counts per policy-arm.**
- **C:** 120 seeds × 3 levels = 360 A\* episodes; 40 seeds × 3 = 120 RRT\* episodes.
- **D:** 200 seeds × 3 = 600 A\* episodes; 60 seeds × 3 = 180 RRT\* episodes.
- Non-learned comparators (baseline, matched baseline, reference) run on the same instances.

**Sample size [Observed noise; Hypothesis effects].**
- Per-seed paired completion differences have SD ≈ 0.026–0.029 (PPO greedy vs stochastic, block B), and conflicts per episode SD ≈ 28–35.
- For 80 % power at α = 0.05 with Holm over 6 policies (worst case α/6), n ≈ ((2.64 + 0.84)·SD/effect)². That is ≈ 80–100 seeds for a completion effect of 0.01, and ≈ 95–150 seeds for a conflict effect of 10 per episode. C's 120 seeds cover the completion target and the lower conflict SD; D's 200 seeds cover both.
- D's 200 seeds give a per-policy 95 % half-width of about ±0.004 for a completion difference.

**Statistical units and tests.**
- **Per-policy comparisons:** paired by seed (levels pooled per seed for overall effects; per-level analyses within level). Stochastic replicates are averaged per instance first.
- **Estimates and tests:** mean difference with a seed-cluster bootstrap 95 % CI; Wilcoxon signed-rank on per-seed differences; rank-biserial r; Holm within each metric family.
- **Algorithm level:** two-level bootstrap (training seeds × scenario seeds), no p-values.
- **Primary metrics** (declared per hypothesis): completion and conflicts per episode as co-primary, with completion analysed for non-inferiority where an intervention targets conflicts.
- **Report every result** per policy, planner, allocator and level, with no single aggregate score.

**Pre-registered confirmatory hypotheses on D** (final wording frozen after C):
- **H-L1:** frozen-L1 checkpoints vs historical checkpoints, same runs, greedy.
- **H-L2:** frozen ES vs no ES, same checkpoints.
- **H-L12:** L1 + ES vs the historical system.

Each with its metric family, margins and Holm correction.

## 13. Testing and regression strategy

| Change | Unit | Property-based | Integration / e2e | Regression | Performance |
|---|---|---|---|---|---|
| Instrumentation (G1–G5) | Cycle detector on hand-built wait-for graphs; stall/oscillation classifier on synthetic histories | Detector agrees with brute force on random successor maps | Fields present for every controller | **Byte-identical historical records** with it off; MADQL determinism check vs `dev5` | Step-time overhead (NFR-PERF-1 workload R) |
| Seed blocks C/D | Block membership, disjointness | Uniqueness of generated specs | `check_seeds` refuses mixing | Blocks A/B unchanged | — |
| SCB-C | Synthetic tables with a known best; ties; collision exclusion; latest-step tie | Invariance to row order; the truly best is never excluded when it dominates by a large margin | Reads only stored logs; writes a new directory | `best.json` / `best.pt` hashes unchanged | Bootstrap runtime |
| ES | R on contested cells and swaps; fairness counter; S trigger/hold; S-rand seeding; logging; original proposal preserved | **R never creates a robot–robot conflict among the robots it filters**; ES only maps move → stay or move → baseline-filtered suggestion; every ES output passes through shield and arbiter | X-4 matrix with ES; deterministic traces (fixed seeds) | ES off ⇒ traces identical to Issue #6 | Overhead per step |
| P1–P3 | as in revision 1 | as in revision 1 | as in revision 1 | defaults ⇒ identical | planner calls; time per call |

Cross-platform: the CI matrix (Ubuntu 3.12–3.14, Windows 3.14) runs the fast suite; platform-dependent code gets `sys.platform`-guarded tests.

Tests establish correctness properties only. Performance claims come only from pre-registered experiments.

## 14. Implementation and experiment costs

**Unit costs** (median wall seconds per episode from the stored records, measured under concurrent load; 8 workers assumed):

| Controller | A\* | RRT\* |
|---|---|---|
| PPO | ≈ 0.4–0.5 | ≈ 16–20 |
| MADQL | ≈ 1.9 | ≈ 120–125 |
| Baseline | ≈ 0.2 | ≈ 8 |

**Existing computation reused (no new cost):**
- all 14 PPO and 3 MADQL training runs and their 20 step checkpoints each;
- the block-A validation logs (all selection inputs);
- the block-B records (reference).

**No training in any item below.**

| Item | Engineering | Additional evaluation | Wall time (8 workers) | Notes |
|---|---|---|---|---|
| **Instrumentation (6A.0)** | 2–3 days (G1–G5, blocks C/D, tests) | Byte-identity replays: 9 policies × block B greedy (2 160 A\*, 540 RRT\*); MADQL full classification replay (710 A\*); cluster re-analysis (no episodes) | ≈ 1–1.5 h (RRT\* replays dominate; ≈ 10 min without them) | RRT\* replays could be limited to a subset if time matters |
| **L1 evaluation (C)** | 1 day (SCB-C, simulation study, tests) | Old and new checkpoints: ≤ 12 PPO + ≤ 6 MADQL policies × (360 A\* + 120 RRT\*) | A\* ≈ 15 min; RRT\* ≈ 1 h (PPO) + ≈ 3 h (MADQL) | MADQL RRT\* only if MADQL selections change |
| **L2 evaluation (C), 2 × 2 core** | 2–3 days (ES, logging, tests) | 9 policies × 8 episode sets (G, G+ES, S×3, S+ES×3) × 360 A\*; G and G+ES × 120 RRT\* | A\* ≈ 1 h; RRT\* ≈ 4 h (≈ 1 h PPO-only) | |
| **L2 ablations (C)** | — | Greedy 2³ factorial adds 6 arms; S-rand ×3; parameter grid 6 configs; shield-on check (full level) | ≈ 2–2.5 h (A\* only) | No RRT\* for ablations |
| **Combined L1+L2 (C)** | — | L1 policies (6 PPO + 3 MADQL) × 2 × 2 core × 360 A\*; G vs G+ES × 120 RRT\* | A\* ≈ 1 h; RRT\* ≈ 4 h (≈ 1 h PPO-only) | Skippable if L1 changes no selection |
| **Final held-out (D), once** | — | 9 policies × {historical greedy, L1 greedy, L1+ES greedy, L1+ES stochastic ×3} × 600 A\*; 3 greedy arms × 180 RRT\*; comparators | A\* ≈ 1–1.5 h; RRT\* ≈ 2.5 h PPO + ≈ 7 h MADQL | MADQL RRT\* on D is the largest single cost; restricting it is a decision (§19 B) |
| Storage | — | Records ≈ 5–10 MB per 10 000 episodes | < 300 MB in total | No new checkpoints |

## 15. Risks, limitations and dependencies

- **Protocol drift:** C may be reused for development; D is used once after the freeze. Test seeds stay untouched.
- **Classification heuristics in §3.1:** replaced by recorded events (G2) before any L2 conclusion.
- **MADQL evidence:** resolved in 6A.0; all 710 MADQL failures were replayed (§20).
- **ES changes the controller under test:** results are reported as "policy + ES".
- **Practical equivalence (Δ):** probably not demonstrable with 20 selection seeds. Either accept the statistical-indistinguishability rule (SCB-C), or enlarge the selection set (cost in §19 B).
- **Historical pooled CIs (F15):** probably too narrow by about 30 %. The re-analysis is non-destructive and reported separately; any change to an Issue #5/#6 conclusion would be flagged, not silently corrected.
- **Prioritised planning (P4):** incomplete, and implies retraining.
- **RRT\* guarantees (P2):** must be restated if adopted.
- **Dependencies:** 6A.0 precedes L1 and L2. L1 and L2 share block C. Freeze precedes D. P1 precedes any RRT\*-in-training revisit. Issue #8 uses the frozen 6A configuration.

## 16. Proposed implementation phases

1. **6A.0 Measurement:** G1–G5, blocks C/D, MADQL full replay, cluster re-analysis.
2. **6A.1 L1 development on C.**
3. **6A.2 L2 development on C.**
4. **6A.3 Combined development on C; freeze.**
5. **6A.4 Final confirmation on D (once).**
6. **6A.5 Planning interface (P1–P3) on C.**
7. **6A.6 Optional P4 / retraining (separate approval).**
8. **6A.7 Documentation and hand-over to Issue #8.**

Each phase ends with a report and the student's approval.

## 17. Draft GitHub issue

> **Issue 6A — System-wide Algorithm Review, Optimization and Validation**
>
> **Branch** `feature/system-review` · **Labels** `enhancement`, `research`, `evaluation`
>
> **Background and motivation.** Issue #6 delivered MAPPO/IPPO and the E4 comparison. The post-Issue-6 analysis and this proposal (`docs/research/`) found that the remaining deficits are at interfaces:
> - checkpoint selection decides on noise and ignores conflicts;
> - deterministic execution produces oscillations, frozen robots and mutual-cancellation loops;
> - planning ignores other robots and repeats on every deviation;
> - pooled statistics treated correlated levels as independent.
>
> 0 physical collisions were observed.
>
> **Objectives.**
> 1. Missing measurements and correct statistical units.
> 2. A multiplicity-controlled checkpoint-selection rule (L1).
> 3. A logged execution supervisor (L2).
> 4. Their independent development evaluation on block C and a single confirmation on block D.
> 5. Evidence-based planning-interface improvements (P1–P3).
>
> ≥ 99 % completion with substantially fewer conflicts is a research goal, not an acceptance criterion.
>
> **Scope.** In: 6A.0–6A.5 and 6A.7. Optional, separately approved: 6A.6 (P4, retraining).
>
> **Non-goals.**
> - No changes to Issue #5/#6 records, checkpoints or conclusions.
> - No reward changes.
> - No new learning algorithms (VDN deferred).
> - No Kaggle data (Issue #7).
> - No viewer or test-seed evaluation (Issue #8).
> - No renumbering of Issues #7 and #8.
>
> **Work packages.** 6A.0 Measurement · 6A.1 L1 (C) · 6A.2 L2 (C) · 6A.3 Combined (C) and freeze · 6A.4 Confirmation (D) · 6A.5 Planning interface · 6A.6 Optional · 6A.7 Documentation.
>
> **Dependencies.** #6 (complete). #7 (E7) and #8 (E5/E6) use the frozen 6A configuration.
>
> **Experimental requirements.**
> - Blocks C (11 000–11 119) and D (12 000–12 199) as in proposal §12.
> - Seed-cluster statistics.
> - Pre-registration and freeze before D.
> - Paired comparisons with Holm per metric family.
> - Results per policy, planner, allocator and level.
> - Required metrics: completion, delivered and undelivered tasks, physical collisions, conflicts by type, ES, shield and resolver interventions, deadlocks (frequency, duration), stalls and oscillations, flow time, path length, planner calls by trigger, wait ratio, runtime and memory.
>
> **Testing requirements.**
> - Unit, property-based and integration tests (proposal §13).
> - Byte-identical historical behaviour with every new option off.
> - MADQL / Issue #5 regression tests unchanged.
> - CI on Ubuntu and Windows.
>
> **Documentation deliverables.** `02`, `03` (ADR-10 extension, selection rule, seed blocks), `04`, `05`, `07`, verification-log entries.
>
> **Acceptance criteria.**
> 1. G1–G5 and blocks C/D implemented; historical replays byte-identical with them off.
> 2. SCB-C validated by simulation, frozen before C, evaluated on C, and reported whatever its direction.
> 3. ES implemented (off by default) with property tests; the 2 × 2 and the 2³ factorial evaluated on C and reported.
> 4. Configuration frozen and pre-registered; the D confirmation run once and reported whatever its direction.
> 5. All quality gates pass; no historical record or checkpoint changed.
> 6. The frozen configuration and selections for Issue #8 approved by the student.
>
> **Expected outputs.** Code options (off by default) and tests; `results/dev6a/` records with manifests; figures and tables; updated docs.
>
> **Risks and limitations.**
> - Protocol drift (C/D separation).
> - ES masks policy weaknesses (reported as policy + ES).
> - Practical equivalence not demonstrable with 20 selection seeds.
> - Prioritised planning is incomplete.
> - RRT\* guarantees must be restated.

## 18. Documentation change plan (to apply only after approval)

**Roadmap revision.** The approved plan has exactly eight issues (`07` §5, decision 12). Issue 6A is a **ninth entry, between #6 and #7, without renumbering #7 and #8**.

| File / location | Change |
|---|---|
| `07` header | "the 8 issues" → "the 8 numbered issues plus Issue 6A (roadmap revision 3, approved on <date>)"; Status → "Revision 3" |
| `07` §2 table | Insert after #6: `6A \| System-wide algorithm review, optimisation and validation \| feature/system-review \| #6`; #7 "Depends on" → `#2 (pipeline), #6A (for E7)`; #8 → `all` |
| `07` §2 Mermaid | `I6 --> I6A[#6A System review]`; `I6A --> I7`; `I6A --> I8` (replacing `I6 --> I7`, `I6 --> I8`) |
| `07` §3 | Row after #6A: "frozen execution, selection and planning configuration for E5–E7" |
| `07` §4 | New Issue 6A definition (§17 text in the §4 format) |
| `07` §5 | Decision 13: roadmap revision 3; #7 and #8 unchanged |
| `07` §6 | Risks: reuse of validation data (C/D); ES masking policy weaknesses; correlated instances |
| `docs/README.md` | "the 8 issues" → "the 8 issues plus Issue 6A" |
| `05` B.1 | Add blocks C (11 000–11 119; RRT\* 11 000–11 039) and D (12 000–12 199; RRT\* 12 000–12 059) with their rules; block B marked "historical development data" |
| `05` B.3 | Seed as the statistical unit (cluster bootstrap); simultaneous selection procedure; non-inferiority |
| `05` B.4 | Development experiments `6A-L1-dev`, `6A-L2-dev`, `6A-L12-dev`, `6A-confirm`; E5–E7 use the frozen 6A configuration |
| `05` B.2 | Deadlock, stall, oscillation and ES-intervention metrics |
| `05` B.6 | Register rows when run; a note on F15 for the Issue #5/#6 pooled CIs |
| `02` §5.5 / §6.4 / §9 | Execution layers with ES; metric definitions |
| `03` ADR-10, ADR-4 / §7 | ES layer and safety argument; selection rule after approval |
| `04` | ES algorithm; P1–P3 notes |
| `docs/research/2026-11-20-post-issue6-analysis.md` | Dated correction note (failure modes, F15) |
| Verification log | One entry per phase |
| `docs/SESSION_HANDOFF.md` | **Not modified** by this work; the student updates it if wanted |

## 19. Decisions

**A. Can be approved now** (no further evidence needed):
1. Roadmap revision 3: Issue 6A as a ninth entry between #6 and #7; #7 and #8 not renumbered; branch `feature/system-review`.
2. Block roles: C = development/validation (repeatable, logged); D = final held-out (once, after freeze); B = historical reference only.
3. Block ranges: C = 11 000–11 119 (RRT\* 11 000–11 039); D = 12 000–12 199 (RRT\* 12 000–12 059); including the code change to register them.
4. The seed as the statistical unit for 6A, and the non-destructive cluster re-analysis of the Issue #5/#6 pooled CIs.
5. Phase 6A.0 instrumentation (G1–G5) and the full MADQL replay classification.
6. The L1 methodology family (SCB-C: paired seed-cluster bootstrap, max-T simultaneous comparisons, conflicts as secondary, latest-step tie) for implementation and validation by simulation, without evaluating on C yet.
7. The L2 design structure: 2 × 2 mode × ES core, greedy 2³ factorial, S-rand arm, shield-on check.

**B. Require additional evidence first:**
1. Practical-equivalence margin Δ (0.005 remains a hypothesis): after the simulation study shows what block-A precision allows. Alternatively, approve a larger selection set: re-evaluating the 20 stored step checkpoints of the 6 main PPO runs on 40–80 additional selection seeds (a new selection block, not C or D), about 15–30 min of A\*-only evaluation.
2. Whether MADQL is re-selected under L1 and included in RRT\* evaluations on C and D: MADQL RRT\* episodes cost about 6× PPO's.
3. Whether to restrict RRT\* evaluation on D to greedy arms and PPO, given its cost.
4. Planning scope P1–P3, and whether P2 should target making RRT\* in training feasible.

**C. Defer until block-C results exist:**
1. Adoption of the L1 rule (v1 or a revised version).
2. ES parameter values: W, H, L_max, p for S-rand; which mechanisms to keep (from the factorial).
3. The default execution mode for learned controllers (greedy vs stochastic, with or without ES).
4. Whether P4 (reservations) or any retraining (obs-v2, MADQL-v2) is justified.
5. The `models/` candidates: the PPO selections may change under L1.

**D. Frozen until block D:**
1. The confirmatory hypotheses H-L1, H-L2 and H-L12, their arms, metrics, margins and multiplicity families.
2. The frozen L1 rule, the ES configuration and parameters, and the analysis code (hash recorded).
3. Any claim that 6A improves the system, and the configuration handed to Issue #8.
4. Committing final checkpoints to `models/`, after D confirms the selections.

---

## 20. Phase 6A.0 results (verified, 2026-11-20)

Approved scope: failure replay, seed-cluster re-analysis, evidence correction. **Not done in 6A.0** (by scope):
- no L1 rule implemented or evaluated;
- no execution supervisor;
- no block C or D episode;
- no training;
- no change to production code, configurations, historical records, checkpoints or existing documents.

### 20.1 Reproducible commands and outputs

All outputs are in `results/dev6a/6a0_diagnostics/` (git-ignored, separate from the authoritative results). The tools are in `tools/diagnostics/`, outside the package.

```text
python tools/diagnostics/replay_failures.py    --out results/dev6a/6a0_diagnostics --workers 8   # 2.6 min
python tools/diagnostics/cluster_reanalysis.py --out results/dev6a/6a0_diagnostics               # 2.0 min
python tools/diagnostics/summarise_replay.py   --dir results/dev6a/6a0_diagnostics
```

| File | Content |
|---|---|
| `replay_episodes.jsonl` | One record per replayed failure: replayed summary fields, mismatch list, per stuck robot the counts, flags and classes |
| `replay_summary.json` | Coverage (policy × level, failing seeds), all count tables |
| `cluster_reanalysis.json` | Every historical pooled row: old (instance) and new (seed-cluster) CI, width ratio, old/new Wilcoxon and Holm p, per-level estimates |
| `hashes_before.json`, `hashes_after.json` | SHA-256 of every historical file before and after 6A.0 |

The final, lint-formatted versions of the three tools were re-run into a scratch directory, and their outputs were compared byte for byte with the files above (§20.6).

### 20.2 Replay coverage and reproducibility

- **[Observed] Population:** every greedy block-B episode (A\*, auction, endpoint recovery on, shield off; seeds 10 020–10 099 × three levels) with `released − delivered > 0` in `results/dev6/stochastic/records.jsonl` (mode `greedy`). That is **928 of 2 160** greedy episodes: 218 PPO + 710 MADQL. Every one was replayed; none was excluded.
- **[Observed] Reproduction:** the replay uses the production `evaluation._system`, `Engine` and the stored `best.pt` of each policy. Each replayed episode was compared with its stored record on 10 fields: delivered, released, steps, terminated, conflicts by type, conflicts total, physical collisions, wait ratio, planner calls, mean flow time. **0 of 928 episodes differ in any field.**
- **[Observed] Coverage per policy** (failed / greedy episodes per level, 80 seeds each):

| Policy | static | obstacles | full | Total |
|---|---|---|---|---|
| IPPO 1 / 2 / 3 | 28 / 13 / 16 | 14 / 9 / 3 | 16 / 11 / 8 | 118 |
| MAPPO 1 / 2 / 3 | 11 / 12 / 13 | 7 / 8 / 11 | 16 / 12 / 10 | 100 |
| MADQL 1 / 2 / 3 | 80 / 80 / 80 | 74 / 79 / 79 | 79 / 80 / 79 | 710 |

- **MADQL policy and seed coverage [Observed]:**
  - all three MADQL policies are covered;
  - 710 of 720 greedy MADQL episodes fail;
  - the 10 that complete are MADQL 1: obstacles seeds 10 041, 10 052, 10 055, 10 063, 10 082, 10 087 and full seed 10 068; MADQL 2: obstacles 10 063; MADQL 3: obstacles 10 063 and full 10 063.

  Every static episode of every MADQL policy fails. The per-episode failing seed lists of every policy and level are in `replay_summary.json` (`coverage`).

### 20.3 Failure modes with explicit denominators

**Definitions (unchanged from revision 2).**
- **Failed episode:** as in §20.2. All 928 ran the full 400 steps.
- **Stuck robot:** not idle at the final step (it holds a task).
- **Unit of the tables:** stuck robots, observed over the **last 100 steps**.
- **Flags** (a robot can carry several):

  | Flag | Rule |
  |---|---|
  | `frozen` | 0 moves and 0 proposed moves |
  | `oscillating` | ≤ 3 distinct cells and ≥ 50 moves |
  | `conflict_loop` | the robot's proposal is in a conflict event reaching the arbiter in ≥ 50 steps |
  | `deadlock_cycle` | the robot is in a **genuine wait-for-cycle deadlock** in ≥ 50 steps (below) |
  | `blocked_by_idle` | its next reference cell holds an idle robot in ≥ 50 steps |
  | `mostly_stationary` | < 10 moves |

- **Genuine wait-for-cycle deadlock** (the resolver's own criteria, applied passively):
  - wait-for edge i → j when robot i is not idle and its next reference-path cell holds robot j;
  - cycles found with the production `find_cycles`;
  - a cycle counts as a deadlock when **persistent** (same members on the same cells for 2 consecutive steps) or **recurrent** (3rd new occurrence of the same member set within 20 steps).
- **Primary class**, exclusive, applied in order:
  1. `deadlock_cycle`
  2. `blocked_by_idle`
  3. `frozen`
  4. `oscillating`
  5. `conflict_loop`
  6. `mostly_stationary`
  7. `other_moving` (residual; reported)

**Primary class of stuck robots** (denominator = stuck robots of the row):

| Algorithm, level | Failed episodes | Undelivered tasks | Stuck robots | Deadlock cycle | Blocked by idle | Frozen | Oscillating | Conflict loop | Mostly stationary | Other moving |
|---|---|---|---|---|---|---|---|---|---|---|
| IPPO static | 57 | 89 | 89 | 10 | 1 | 29 | 35 | 14 | 0 | 0 |
| IPPO obstacles | 26 | 36 | 36 | 0 | 0 | 15 | 12 | 8 | 0 | 1 |
| IPPO full | 35 | 45 | 45 | 0 | 3 | 13 | 25 | 0 | 1 | 3 |
| **IPPO total** | **118** | **170** | **170** | **10 (6 %)** | **4 (2 %)** | **57 (34 %)** | **72 (42 %)** | **22 (13 %)** | **1** | **4 (2 %)** |
| MAPPO static | 36 | 79 | 79 | 6 | 1 | 4 | 43 | 24 | 0 | 1 |
| MAPPO obstacles | 26 | 35 | 35 | 2 | 0 | 1 | 15 | 16 | 1 | 0 |
| MAPPO full | 38 | 54 | 54 | 0 | 0 | 5 | 20 | 28 | 0 | 1 |
| **MAPPO total** | **100** | **168** | **168** | **8 (5 %)** | **1 (1 %)** | **10 (6 %)** | **78 (46 %)** | **68 (40 %)** | **1** | **2 (1 %)** |
| MADQL static | 240 | 2 586 | 1 850 | 30 | 1 | 567 | 1 232 | 4 | 0 | 16 |
| MADQL obstacles | 232 | 920 | 911 | 4 | 3 | 203 | 443 | 2 | 39 | 217 |
| MADQL full | 238 | 1 994 | 1 487 | 10 | 0 | 242 | 588 | 2 | 55 | 590 |
| **MADQL total** | **710** | **5 500** | **4 248** | **44 (1 %)** | **4 (0.1 %)** | **1 012 (24 %)** | **2 263 (53 %)** | **8 (0.2 %)** | **94 (2 %)** | **823 (19 %)** |

**Undelivered tasks not held by a stuck robot [Observed].**
- **PPO:** none. Every undelivered task is held by a stuck robot (170 / 170 IPPO, 168 / 168 MAPPO).
- **MADQL:** 1 252 of 5 500 undelivered tasks in 287 episodes were held by no robot. They were still pending because all robots were stuck holding other tasks: 185 static, 5 obstacles and 97 full-level episodes. These tasks are a *consequence* of the stuck robots, not a separate failure mode.

**Verification of the revision-2 counts [Observed].** The revision-2 rule order, recomputed on the full replay:

| Revision-2 class | IPPO | MAPPO |
|---|---|---|
| frozen (next cell free) | **58** (confirmed) | 9 |
| frozen (next cell: active robot) | 0 | 1 |
| oscillating | 85 | 87 |
| conflict loop | 22 | 68 |
| mostly stationary | **1** | 1 |
| other moving | **4** | 2 |

Revision 2 reported IPPO "other 5": that is these 4 + 1.

**Distinguishing the four behaviours [Observed].**
- **Oscillation** is the most common mode for every algorithm: 42–53 % of stuck robots.
- **Frozen** robots wait in front of a **free** suggested cell:
  - 57 / 57 IPPO;
  - 9 / 10 MAPPO, the tenth facing an active robot;
  - 1 010 of the 1 014 frozen MADQL robots (revision-2 order).

  The planner is not what blocks them.
- **Mutual-cancellation conflict loops are not wait-for cycles.** None of the 90 PPO or 8 MADQL conflict-loop robots carries the `deadlock_cycle` flag. These are contests over a **free** cell or a swap, cancelled by the arbiter every step.
- **Genuine wait-for-cycle deadlocks are rare,** and most come from oscillation:
  - 62 stuck robots in total, 1.4 % of 4 586 (IPPO 10, MAPPO 8, MADQL 44);
  - 59 of the 62 are also oscillating, so these are **recurrent** cycles that re-form as robots move back and forth;
  - only 3 are stationary, **persistent** cycles (2 MADQL, 1 IPPO);
  - episodes with a deadlock at any time: IPPO 8 of 118, MAPPO 14 of 100, MADQL 92 of 710.
- **Interpretation:** the existing resolver (cycles only) would address at most about 1.4 % of stuck robots. Oscillation and frozen behaviour, together 76 % of IPPO, 52 % of MAPPO and 77 % of MADQL stuck robots, need a different mechanism; for MAPPO, conflict loops (40 %) are the other large share. This supports the stall breaker of L2 (hypothesis, untested).

**Episode level** (an episode can contain several modes):

| | IPPO (118) | MAPPO (100) | MADQL (710) |
|---|---|---|---|
| Only oscillating | 50 | 39 | 64 |
| Only frozen | 40 | 6 | 7 |
| Only conflict loop | 9 | 37 | 0 |
| Only deadlock cycle | 3 | 2 | 0 |
| Only blocked by idle | 3 | 1 | 0 |
| Only other / stationary | 3 | 1 | 16 |
| Mixed | 10 | 14 | 623 |

### 20.4 Seed-cluster re-analysis of the historical statistics

**Method.**
- Every pooled historical estimate was recomputed twice, with identical inputs, 10 000 resamples, seed 0 and percentile CIs:
  - **(a)** the original instance bootstrap;
  - **(b)** a seed-cluster bootstrap, in which the three levels of a sampled seed stay together.
- Wilcoxon p-values were recomputed on per-seed mean differences, with Holm within the original metric families.
- **Algorithm-level and pooled-policy estimates:** instances were replaced by seed clusters inside the same two-level procedure.
- **Per-level estimates:** available separately for every row (`per_level` in `cluster_reanalysis.json`). Within one level each seed is one instance, so they are unaffected by the change of unit.
- **Shield rows:** use the full level only, one instance per seed, so they are unaffected by construction.

**Replication [Observed].**
- (a) reproduces **every** stored historical CI: 365 of 365 per-policy rows and 96 of 96 pooled/algorithm rows.
- The recomputed Holm p-values equal the stored ones (365 / 365).
- Every difference below is therefore due to the statistical unit alone.

**Width of the intervals [Observed]** (new width ÷ old width):

| Section | Rows | Median | Range |
|---|---|---|---|
| Issue #5 per policy, A\* / RRT\* | 18 / 18 | 1.06 / 1.14 | 0.68–1.34 / 0.62–1.42 |
| Issue #6 per policy, A\* / RRT\* | 135 / 135 | 1.16 / 1.13 | 0.68–1.36 / 0.62–1.44 |
| Issue #6 stochastic (A\*) | 27 | 1.07 | 0.65–1.25 |
| Issue #6 ablation per seed (A\*) | 24 | 1.06 | 0.95–1.26 |
| Resolver effect, A\* / RRT\* | 4 / 4 | 1.20 / 0.97 | 1.10–1.34 / 0.92–1.09 |
| Issue #5 pooled (two-level), A\* / RRT\* | 6 / 6 | 1.04 / 1.06 | 0.99–1.30 / 1.01–1.40 |
| Issue #6 algorithm level, A\* / RRT\* | 36 / 36 | 1.06 / 1.08 | 0.88–1.32 / 0.98–1.43 |
| Issue #6 ablation cell − default | 12 | 1.03 | 1.02–1.15 |
| **All per-policy rows** | **365** | **1.13** (IQR 1.02–1.28) | 0.62–1.44; 21 % of rows narrower |

**Impact of the unit [Interpretation].**
- **Wider:** where an effect is positively correlated across the levels of a seed (correlations 0.37–0.69 measured), the instance bootstrap counted the same scenario's information three times.
- **Narrower:** where within-seed level differences offset each other, the cluster mean has less variance, so the CI narrows.
- **The 1.31× of revision 2:** it came from one comparison and an ad-hoc resampling stream. The same row gives 1.29× with the standard bootstrap, near the 75th percentile of all rows; the typical factor is 1.13.

**Conclusions that change [Observed]** (CI-excludes-zero or Holm p < 0.05 status):
- **CI now includes 0 (2 of 461 rows):**
  - MADQL 1 − matched baseline, conflicts per 100 robot-steps, RRT\*: was [+0.011, +0.677], now [−0.002, +0.750].
  - Algorithm level, MAPPO − IPPO, base return, RRT\*: was [−2.380, −0.037], now [−2.630, +0.025].
- **Holm significance lost (18 of 365 per-policy rows; 299 → 281 significant):**
  - IPPO 1 − baseline and − matched baseline, completion, RRT\* (Holm 0.009 → 0.088). The per-policy statement "only IPPO 1 survives Holm on RRT\*" in `05` becomes "none survives". The algorithm-level CIs still exclude 0.
  - Several RRT\* conflict rows (IPPO 1, IPPO 3, MADQL 2) and MADQL 1 conflicts per 100 vs the baseline (A\*). The `05` statement "every PPO policy proposes more conflicts than either baseline (per-policy Holm p ≤ 0.009)" no longer holds for the RRT\* rows of IPPO 1 and IPPO 3; their CIs still exclude 0.
  - IPPO conflict reductions under stochastic execution (IPPO 1–3) and IPPO 2's return gain: not Holm-significant at the seed level. The CIs of IPPO 2 and 3 still exclude 0; IPPO 1's did not before either.
- **Unchanged [Observed]:**
  - PPO ≫ MADQL (A\*: +0.396, new CI [0.349, 0.436]);
  - PPO > reference follower;
  - PPO < resolver baseline with A\* (MAPPO −0.012 [−0.017, −0.008]);
  - PPO ≈ matched baseline with A\* (CI includes 0, as before);
  - PPO slightly below the matched baseline with RRT\* at the algorithm level (CI still excludes 0);
  - the MADQL negative result of Issue #5 (every pooled CI still excludes 0, width ratio 0.99–1.02);
  - the resolver's A\* effect;
  - the shield results (unaffected).

  The historical records and analyses are unchanged. Proposed annotations to `05` are listed in §20.8 and await approval.

### 20.5 Files created or modified in 6A.0

| Path | Status |
|---|---|
| `tools/diagnostics/replay_failures.py` | new |
| `tools/diagnostics/cluster_reanalysis.py` | new |
| `tools/diagnostics/summarise_replay.py` | new |
| `results/dev6a/6a0_diagnostics/` (6 files, §20.1) | new (git-ignored) |
| `docs/research/2026-11-20-issue-6a-proposal.md` | updated (this revision) |

Nothing else was created or modified.

### 20.6 Hash verification [Observed]

- **Coverage:** before any 6A.0 work, every file under `results/training`, `results/dev5`, `results/dev6`, `results/experiments`, `results/pilot_rrt` and `results/figures/overview` was hashed (SHA-256), plus `docs/SESSION_HANDOFF.md` and `results/dev5_records_sha_before_issue6.txt`. That is 1 310 files, including all 713 checkpoint files (`*.pt`) and all 46 training logs.
- **Result:** after 6A.0, **0 changed, 0 missing, 0 added**.
- **Spot values:** `results/dev5/comparison/records.jsonl` = `53b4e114…`; `docs/SESSION_HANDOFF.md` = `ba4a439c…`.
- **`models/`** does not exist; nothing was copied.
- **Tool reproducibility:** the final, lint-formatted tools were re-run into a scratch directory. `replay_episodes.jsonl` and `cluster_reanalysis.json` are byte-identical to the files above. `replay_summary.json` was identical in content but not in key order: one table iterated a Python set, whose order depends on string-hash randomisation. That loop now iterates in sorted order, the summary was regenerated from the unchanged replay records, and repeated runs give byte-identical files (SHA-256 `9000dfaa…`).

### 20.7 Remaining uncertainty and limitations

- **Thresholds.** The classification thresholds (100-step window; ≤ 3 cells; 50 of 100 steps) are fixed heuristics, not fitted. Moving them would shift robots between classes, especially oscillating ↔ other moving for MADQL. The flag combinations are in `replay_summary.json` for that reason.
- **Window.** Only the last 100 steps are classified. When a failure started, and whether it changed mode earlier, is not captured.
- **Wait-for graph.** It uses the reference path in the controller's view, the same source as the resolver, with the resolver's default persistence and recurrence parameters. Other parameters would give other deadlock counts.
- **Scope of the replay.** Block-B development data only, greedy A\* with the auction allocator. RRT\*, sequential-allocator and shielded failures were not replayed; their failure-mode mix is unknown.
- **Seed-level Wilcoxon.** It uses 80 seeds (20 for RRT\*) and does not use the within-seed level structure. RRT\* rows remain weakly powered.
- **Not re-analysed:** the Issue #3/#4 experiments (E1–E3, Issue #4 system comparison), which have different designs.

### 20.8 Recommendation

- **[Interpretation]** The evidence supports proceeding to the **L1 phase as scoped in §4**: implement the SCB-C procedure and its simulation study, without evaluating on block C until separately approved.
  - F1 (selection on noise, conflicts ignored) is unaffected by 6A.0.
  - F15 is now quantified, and SCB-C already uses the seed as its unit.
- **[Hypothesis]** 6A.0 strengthens the motivation for L2's stall breaker over the resolver alone: genuine cycles are about 1.4 % of stuck robots, while oscillation and frozen robots are 52–77 %, and MAPPO conflict loops a further 40 %. That is for the L2 phase, not L1.
- **Separate approvals still needed:**
  1. Annotating `05` (Issue #5 and #6 sections) with the seed-cluster results of §20.4 as dated notes, without changing the historical numbers.
  2. A verification-log entry (V-20) for 6A.0.
  3. The commit grouping for the new tools and this proposal.

---

## 21. Phase L1 results: SCB-C implementation and simulation study (verified, 2026-11-21)

Approved scope: implementation of the checkpoint-selection procedure in isolated tooling and its statistical simulation study. **Not done** (by scope):
- no block-C or block-D data read or evaluated;
- the procedure was **not applied** to the real block-A selection data to produce selections. Block A only calibrated the simulation (§21.3).
- no production code, checkpoint or historical record changed;
- no training;
- no L2 work.

### 21.1 Definition of SCB-C

**Name.** SCB-C stands for **Simultaneous Comparison Band, then Conflicts**.

**Data per run.**
- K candidate checkpoints: every stored validation checkpoint of the run, 20 per run, from 50 k to 1 M steps.
- S selection seeds: block A, 20 seeds. The seed is the unit, with the three levels pooled within a seed.
- For checkpoint k and seed s:
  - **D_ks, R_ks:** delivered and released tasks, summed over the seed's levels;
  - **C_ks:** mean proposed conflicts per episode over the seed's levels;
  - **X_ks:** physical collisions;
  - **step_k:** the checkpoint's training step.

**Validity (applied first).** A checkpoint is excluded, with the first matching reason, if:
- **incomplete:** any observation is missing (NaN). No imputation is ever done.
- **invalid:** any value is negative, D > R, or ΣR = 0.
- **collision:** any X_ks > 0. Safety precedes performance.

Further rules:
- If no checkpoint remains, there is no selection, and the exclusions are reported.
- Fewer than 5 seeds is an error.

**Estimates.**
- Completion: ĉ_k = Σ_s D_ks / Σ_s R_ks.
- Conflicts: m̂_k = mean_s C_ks.
- Pairwise differences: Δ_jk = ĉ_j − ĉ_k for completion, and Γ_jk = m̂_k − m̂_j for conflicts, oriented so that positive means "j is better".

**Comparison family.**
- **Completion family:** all K(K−1)/2 pairs of valid checkpoints.
- **Conflict family:** also all pairs of valid checkpoints, fixed in advance, so its simultaneous property also holds for any data-dependent subset.

**Simultaneous comparison (max-T), three versions studied.**
- **v1 — seed bootstrap with fixed SE.**
  - B resamples of the S seeds, one multiplicity matrix shared by all checkpoints, so the pairing is kept.
  - T_b = max over pairs of |Δ*_jk − Δ_jk| / SE_jk, with SE the bootstrap SD of Δ*_jk.
  - q = the (1 − α) quantile of T; "j shown better than k" when Δ_jk − q·SE_jk > 0.
- **v2 — seed bootstrap-t.**
  - As v1, but SE*_jk is recomputed inside each resample by the delta method: per-seed influence values (D − ĉR)/R̄ for the completion ratio and the per-seed values for conflicts, with the weighted covariance of each pair.
- **v3 — within-seed permutation max-T (Westfall–Young).**
  - P random permutations of the checkpoint labels within every seed (the seed's cells are exchanged, so the pairing is kept).
  - Δ^p_jk is recomputed for every permutation, and sd_jk is its permutation SD.
  - T_p = max over pairs with sd > 0 of |Δ^p_jk| / sd_jk; q = the (1 − α) quantile.
  - "j shown better than k" when Δ_jk − q·sd_jk > 0.
  - **Interpretation:** under exchangeability of the labels within seeds, i.e. identical distributions, P(any false "shown better" over all pairs) ≤ α, up to Monte Carlo error. Bounds are test inversions, not confidence intervals for means.

Implementation details for all versions:
- quantiles use the `higher` method;
- pairs with zero spread keep the degenerate interval [Δ, Δ] and do not enter T;
- α = 0.05 per stage, so the joint bound over both stages is ≤ 0.10 by Bonferroni.

**Elimination.**
- E₁ = valid checkpoints that no valid checkpoint is shown better than in completion.
- E₂ = members of E₁ that no member of E₁ is shown better than in conflicts.

Conflicts never override a completion elimination: stage 2 only removes from E₁.

**Final pick.** Applied within E₂ by a pre-declared rule, not a statistical step:

| Rule | Order of keys |
|---|---|
| `completion` (provisional default) | highest ĉ, then lowest m̂, then later step |
| `conflicts` | lowest m̂, then highest ĉ, then later step |
| `latest` (the revision-2 proposal) | later step |

**Practical-equivalence margin.** None is used. 0.005 remains a hypothesis (§21.4).

**Determinism.** The bootstrap matrix and the permutations are seeded (`seed`, default 0). Identical inputs give identical outputs (tested).

### 21.2 Implementation and files

| Path | Status | Role |
|---|---|---|
| `tools/selection/scbc.py` | new | Procedure (validation, bootstrap and permutation bands, elimination, final rules, historical rule for comparison); pure NumPy; no dependency on production PPO/MADQL code |
| `tools/selection/calibrate.py` | new | Reads block-A validation logs (read only); writes the calibration and plug-in worlds |
| `tools/selection/simulate.py` | new | Simulation study (synthetic and plug-in scenarios) |
| `tests/tools/__init__.py`, `tests/tools/test_scbc.py` | new | 30 tests (§21.5) |
| `results/dev6a/l1_selection/calibration.json`, `plugin_worlds.npz` | new (git-ignored) | Calibration; 17 runs × 20 checkpoints × 20 seeds |
| `results/dev6a/l1_selection/simulation_results.json`, `simulation_reps.jsonl` | new | **v1** study (fixed-SE bootstrap, latest-step rule) |
| `results/dev6a/l1_selection/v2/` | new | **v2** study (bootstrap-t) with v1 computed on the same replicates |
| `results/dev6a/l1_selection/v3/` | new | **v3** study (permutation) |
| `docs/research/2026-11-20-issue-6a-proposal.md` | updated | This section |

Nothing else was changed. The Phase 6A.0 outputs in `results/dev6a/6a0_diagnostics/` are untouched.

### 21.3 Simulation design

**Calibration [Observed], from the block-A A\* logs of the 14 PPO runs** (checkpoints from 200 k steps on):
- released tasks per seed: mean 58.8, range 54–66;
- undelivered per seed: mean 0.85, variance 1.73, cross-checkpoint correlation 0.155;
- conflicts per episode (low-conflict checkpoints): mean 1.58, variance 13.0;
- 0 incomplete validations; 0 collisions.

**Fitted generator.**
- Undelivered tasks: binomial, logit p = μ_k + u_s + e_ks.
  - **Shared seed effect** u_s, SD 0.5.
  - **Checkpoint × seed effect** e_ks, SD 0.6.
  - μ_k is set so that the *expected* pooled completion equals the scenario's true value, by quadrature.
- Conflicts: per-seed mean of three negative-binomial episodes (size 0.22) with a shared log-normal seed effect (SD 0.53).
- Correlation of the seed effects of failures and conflicts: 0.16 (default).

**Scenario kinds.**
- **SYNTHETIC:** true values set by construction. These are **not robot-policy results**.
- **PLUG-IN:** each stored run's block-A table (20 × 20) is treated as the population. True values are its 20-seed values, and each replicate resamples seeds with replacement. The plug-in worlds keep the real dependence structure, but their "truth" is itself a noisy 20-seed estimate.

Design:
- 300–2 000 replicates per scenario; B = P = 2 000 resamples/permutations (10 000 in the sensitivity check).
- Every replicate is seeded by (scenario id, replicate).
- The three versions were evaluated on **identical draws**.
- Monte Carlo uncertainty: Wilson 95 % intervals.

**Scenario groups:**
- **A:** global null (K = 20, S = 20).
- **B:** null with K = 5, 10, 40.
- **C:** null with S = 10, 40, 80.
- **D:** one checkpoint better by δ = 0.0025, 0.005, 0.01, 0.02.
- **E:** realistic spread of completion, 0.975–0.990 (also with K = 40 and S = 10, 40, 80).
- **F:** completion tied; one checkpoint with 0.5–20 fewer conflicts per episode.
- **G:** completion and conflicts aligned, unrelated, or in trade-off (higher completion with more conflicts, like early MAPPO).
- **H:** the trade-off with a seed-level failure–conflict correlation of −0.5, 0 or 0.5.
- **I:** an exact tie (duplicate data) and a near tie (0.001).
- **K:** equivalence demonstrability at S = 20, 40, 80, 160.
- **L:** bootstrap size, B = 1 000 vs 10 000.
- **N:** heteroscedastic nulls: equal means with checkpoint-specific noise (v3 only).
- **P:** 17 plug-in worlds (14 PPO, 3 MADQL).

### 21.4 Results by property

**Family-wise error under the null** (probability of any elimination; target ≤ 0.05; 1 000–2 000 replicates):

| Scenario | Stage | v1 fixed SE | v2 bootstrap-t | v3 permutation |
|---|---|---|---|---|
| K = 20, S = 20 | completion | **0.207** | 0.001 | 0.043 |
| K = 20, S = 20 | conflicts | **0.126** | 0.000 | 0.057 |
| K = 5 | completion | 0.096 | 0.016 | 0.048 |
| K = 10 | completion | 0.129 | 0.007 | 0.054 |
| K = 40 | completion | **0.323** | 0.000 | 0.050 |
| S = 10 | completion | **0.535** | 0.000 | 0.055 |
| S = 40 | completion | 0.109 | 0.015 | 0.044 |
| S = 80 | completion | 0.057 | 0.023 | 0.051 |
| S = 160 | completion | 0.050 | 0.033 | 0.037 |
| Heteroscedastic, completion | completion | — | — | 0.058 [0.045, 0.074] |
| Heteroscedastic, conflicts | conflicts | — | — | **0.083 [0.067, 0.102]** |

- **[Observed] v1 is anti-conservative at the available 20 seeds,** and worse with more checkpoints or fewer seeds. The revision-2 proposal (v1 with the latest-step tie) is therefore **withdrawn**.
- **[Observed] v2 controls the error but is extremely conservative.** Its critical values explode: 5.4–7.4 for completion and 7–63 for conflicts, against about 3.5 for v1/v3. In resamples concentrated on seeds where two checkpoints are almost identical, the standard error is near 0.
- **[Observed] v3 is at the nominal level** across K and S. It exceeds it for conflicts when checkpoints have equal means but different dispersion (0.083). It tests identical distributions, not equal means.
  - **Corrected in §22.2:** v3 does not control mean-comparison errors; in realistic null worlds its error is 0.07–0.21, and it grows with the number of seeds.

**False elimination of competitive checkpoints and of the true best** (v3):
- The true completion-best was eliminated in **0 %** of replicates in every non-null synthetic scenario and in all 17 plug-in worlds (upper Monte Carlo bound ≤ 0.8 % synthetic, ≤ 1.3 % plug-in).
- Checkpoints within 0.0025 of the best: eliminated in ≤ 0.4 % of cases.

**Power (v3, S = 20)** — probability that a given inferior checkpoint is eliminated:
- **One better checkpoint** (scenario D):

  | δ | Elimination probability |
  |---|---|
  | 0.0025 | 0.003 |
  | 0.005 | 0.004 |
  | 0.01 | 0.017 |
  | 0.02 | 0.091 |

- **Realistic spread** (scenario E), checkpoints more than 0.005 worse:

  | Seeds | Elimination probability |
  |---|---|
  | S = 10 | 0.017 |
  | S = 20 | 0.045 |
  | S = 40 | 0.138 |
  | S = 80 | 0.394 |

- **Conflicts with completion tied** (scenario F), probability that stage 2 eliminates anything:

  | Conflict difference per episode | P(any elimination) |
  |---|---|
  | 0.5 | 0.07 |
  | 2 | 0.18 |
  | 5 | 0.40 |
  | 20 | 0.62 |

- **Trade-off scenarios (G, H):** stage 2 eliminates in 99 % of replicates, regardless of the seed-level correlation (−0.5 / 0 / 0.5).
- **Plug-in PPO worlds:** stage 1 removed some checkpoints (mean |E₁| 18.4 of 20, mostly the 50 k checkpoints), and stage 2 a further mean of 1.0.

**[Observed] Conclusion:** with 20 selection seeds, a valid simultaneous procedure cannot separate most late-training checkpoints. The final rule therefore makes most of the decision.

**Probability of identifying the best, and the final rules (v3).** True completion gap of the pick, probability that the gap exceeds 0.005, and true conflicts per episode:

| Setting | `completion` | `conflicts` | `latest` | Historical rule |
|---|---|---|---|---|
| 14 PPO plug-in worlds (mean) | gap 0.0016; P(> 0.005) 0.11; conflicts 9.0 | 0.0100; 0.70; 1.2 | 0.0067; 0.71; 1.5 | 0.0017; 0.11; 9.2 |
| 3 MADQL plug-in worlds | 0.0021; 0.05; 2.7 | 0.108; 1.00; 0.7 | 0.086; 0.67; 1.1 | 0.0020; 0.05; 2.7 |
| Synthetic trade-off (G) | 0.0031; 0.19; excess 15.2 | 0.0096; 1.00; excess 0.2 | 0.0009; 0.00; excess 28.2 | 0.0020; 0.10; excess 22.2 |
| One best by 0.01 (D) | P(pick best) 0.87 | 0.04 | 0.00 | 0.86 |

- **[Observed]** `completion` behaves like the historical rule.
- **[Observed]** `conflicts` reduces conflicts strongly, but in the PPO plug-in worlds it accepts a true completion loss of about 0.010 on average.
- **[Observed]** `latest` (revision 2) depends on whether late checkpoints happen to be better, and fails badly when they are not (D: never picks the best).
- **[Interpretation]** At 20 seeds the choice between completion and conflicts is a **value judgement**, not a statistical finding.

**Ties and near ties (I).** Exact duplicates are never separated, and the final rule then decides. With `completion`, the duplicate pair is picked in 67 % of exact-tie replicates and the near-tie best in 49 %, the same as the historical rule.

**Number of checkpoints and seeds.** v3 keeps its level for K = 5–40 and S = 10–160. Power grows with S (see scenario E above).

**Bootstrap/permutation size (L).** P = 1 000 vs 10 000 gives the same decisions within Monte Carlo error. q ≈ 3.54 in both.

**Practical equivalence at 0.005 [Observed].** For the true best, an upper bound below 0.005 against every other checkpoint was obtained in at most a few per cent of replicates at any seed count:
- v2: 0 % at S = 20, 3.8 % at S = 80 (spread scenario);
- v1: 1.3 % even at S = 160.

**0.005 cannot be established as a practical-equivalence margin with the available selection data.** It stays provisional and unjustified.

### 21.5 Tests and checks

- `pytest tests/tools`: **30 passed.**

  | Area | What is tested |
  |---|---|
  | Determinism and pairing | the shared resample matrix; a precision gain of > 10× over unpaired resampling |
  | Independent checks of the max-T | loop reimplementations for v1, v2 and v3; the two-checkpoint case = bootstrap-t; **exact enumeration of all 5⁵ = 3 125 resamples** vs Monte Carlo (within 3 %) |
  | Delta-method SE | a loop implementation; agreement with the bootstrap SE at 400 seeds (within 10 %) |
  | Edge cases | identical checkpoints, clear dominance, conflicts breaking ties, conflicts never overriding completion, missing / invalid / collision exclusions, no valid checkpoint, input validation |
  | Final rules | each rule's pick |
  | Permutation sanity | permutation indices are valid permutations; a 60-replicate null check |
  | Property-based (Hypothesis) | E₂ ⊆ E₁ and the pick ∈ E₂ for v2 and v3; exact checkpoint-order invariance and shift invariance (seed bootstrap); a checkpoint best on every seed is never eliminated |

- **Full fast suite:** **917 passed, 1 skipped** (the POSIX-only `peak_rss` test on Windows), 62 deselected, 0 failed, 482 s; that is the 887 previous tests plus the 30 new ones.
- **Lint and types:** `ruff check` and `ruff format --check` (tools, tests) clean; `mypy --strict` on the three tools clean (`MYPYPATH=tools/selection`).
- **Cross-platform:** the tools are NumPy-only, with no platform-specific code. They were run on Windows; the CI matrix (Ubuntu, Windows) will run `tests/tools`.

### 21.6 Runtime and resources (Windows, 12 logical CPUs, 8 workers)

| Item | Measurement |
|---|---|
| One selection, K = 20, S = 20, 10 000 resamples/permutations | 0.14 s (v1), 0.51 s (v2), 0.50 s (v3) |
| Peak traced memory of that call | 66 MB (v1), 191 MB (v2), 189 MB (v3) |
| Study v1 | 25 700 replicates, 478 s |
| Study v2 | 25 700 replicates, both v1 and v2, 2 153 s |
| Study v3 | 27 700 replicates, 891 s |
| Calibration | 6 s |

### 21.7 Reproducibility

```text
python tools/selection/calibrate.py --out results/dev6a/l1_selection
python tools/selection/simulate.py --cal results/dev6a/l1_selection --out results/dev6a/l1_selection/v2 --workers 8 --method t
python tools/selection/simulate.py --cal results/dev6a/l1_selection --out results/dev6a/l1_selection/v3 --workers 8 --method perm --no-compare
```

(The v1 study was produced by the first version of `simulate.py`. v2 recomputes v1 on identical draws, column `v1_*`.)

- **Calibration:** re-running it after the final lint and type edits gives an identical `calibration.json` and identical plug-in arrays.
- **Simulations:** re-running subsets with the final code reproduces the stored v2 and v3 scenario summaries exactly, field for field.
  - v2: `B_null_K5`, `F_conf_dm5.0`, `P_mappo_seed1`.
  - v3: `B_null_K5`, `G_tradeoff`, `N_hetero_conf_null`, `P_ippo_seed2`, `P_mappo_seed2`.
- **Hashes:** all 1 310 historical files hashed before Phase 6A.0 (training runs and their 713 checkpoints, `dev5`, `dev6`, experiments, pilot, figures, `SESSION_HANDOFF.md`) are unchanged after L1 (0 changed); the 6A.0 outputs are untouched; `models/` does not exist

### 21.8 Limitations and unresolved decisions

**Limitations:**
- The synthetic generator is a calibrated approximation:
  - binomial-logit failures with normal random effects;
  - negative-binomial conflicts;
  - no time structure across checkpoints.
- Plug-in truth is the noisy 20-seed block A itself.
- v3 assumes exchangeability (identical distributions); under unequal dispersion its conflict-stage error was 0.083.
- The stage-wise α = 0.05 gives a joint bound of ≤ 0.10, not 0.05.
- All conclusions concern selection among the 20 stored checkpoints per run; no new episodes were used.

**Unresolved decisions (the student's):**
1. **Method:** v3 (permutation) as the simultaneous procedure. v1 is withdrawn; v2 is valid but powerless.
2. **α:** per stage 0.05/0.05 (joint ≤ 0.10), or 0.025/0.025 (joint ≤ 0.05; less power).
3. **The final rule.** This is the substantive choice, since the statistics rarely decide:
   - `completion` ≈ historical behaviour;
   - `conflicts` gives about 7.5× fewer conflicts in the PPO plug-in worlds, at about 0.008 more completion loss;
   - or an explicit, pre-declared constrained objective: for example, the fewest conflicts among checkpoints whose completion point estimate is within τ of the best, with τ justified as a **policy tolerance, not statistical equivalence**.
4. **More selection seeds:** whether to give selection more statistical power by evaluating the 20 stored step checkpoints of each run on an **enlarged selection set**. That needs new seeds outside C and D, and episodes, which are not authorised yet.
   - **Estimate:** 80 extra seeds × 3 levels × 20 checkpoints × 6 PPO runs ≈ 28 800 A\* episodes, about 3.6 CPU-hours, about 30 min with 8 workers. MADQL is about 4× per run.
   - **Benefit:** in the spread scenario, inferior checkpoints are eliminated about 40 % of the time instead of 4.5 %.

### 21.9 Recommendation on block C

**[Interpretation] SCB-C is not ready for block C as a statistically justified selector.**
- The implementation is correct and tested.
- v3 controls the family-wise error. *(Corrected in §22.2: only under exchangeability; not for mean comparisons.)*
- But with 20 selection seeds it rarely discriminates, so its result is set by the final rule.
- Evaluating it on C now would mainly test that rule (completion- vs conflict-first), not a statistical method.

**Before any block-C evaluation:**
1. Decide items 1–3 above. Either:
   - (a) accept SCB-C v3 with an explicit, justified final rule or constrained objective, labelled as a policy choice; or
   - (b) first approve an enlarged selection set (item 4).
2. Freeze the chosen version (rule text, α, P, seed, code hash).
3. Only then evaluate old vs new selections on C, under the pre-registration of §4 and §12.

Block C remains unconsumed. Block D remains reserved.

## 22. L1 follow-up: validity of the simultaneous procedure, a new selection block, and the selection rule (verified, 2026-11-21)

Approved scope (student, 2026-11-21): resolve the v3 heteroscedastic result; report Monte Carlo uncertainty; evaluate the six main PPO runs' stored checkpoints on 80 new selection seeds (about 30 min, no retraining, seeds separate from C and D); compare completion-first, conflicts-first and constrained selection; recommend an α allocation; update this proposal. **Not done** (by scope): no block-C or block-D episode; no training; no production code, configuration, checkpoint or historical record changed; no L2; `docs/SESSION_HANDOFF.md` not touched; no Git operation.

### 22.1 Answers in brief

| Question | Answer | Where |
|---|---|---|
| What does v3 test? | Exchangeability of the checkpoint labels within each seed (identical distributions), **not** equal means. | §22.2 |
| Does v3 control mean-comparison errors? | **No.** Inflated under unequal dispersion (0.09–0.12 synthetic; 0.10–0.20 in realistic worlds), and the inflation grows with the number of seeds. | §22.2, §22.4 |
| Does a heteroscedasticity-robust method fix it? | A studentised sign flip (v4) and a studentised permutation (v3s) fix *heteroscedasticity* (exact for symmetric contrasts) but fail badly on the *skewed, sparse* contrasts of this project (0.65–0.93 for conflicts). Only the seed bootstrap-t (v2) controls the error in every world tested (≤ 0.057). | §22.3, §22.4 |
| Monte Carlo uncertainty | Wilson 95 % intervals and exact binomial tests against 0.05 for every rate. | §22.4 |
| 80 new seeds | Block S = 13 000–13 079, verified unused and disjoint from A–D; 28 800 episodes, 49 min. | §22.5 |
| Which final rule? | Out of sample, completion-first and conflicts-first are the two ends of a monotone trade-off; a **one-standard-error rule** gives up 0.0017 completion for about half the conflicts. | §22.6, §22.7 |
| α allocation | **α = 0.05 to the completion stage (v2); no conflict-stage test.** Alternative if conflict claims are wanted: 0.025 + 0.025. | §22.8 |
| Ready for block C? | Ready **once the student approves §22.9**; nothing is frozen yet. | §22.9 |

### 22.2 What v3 tests, and why it does not control mean-comparison errors

**[Observed] Null hypothesis of v3.** v3 permutes the checkpoint labels independently within every seed. Its reference distribution is exact for

> H₀^exch: within each seed, the joint distribution of the K checkpoint values is invariant under every permutation of the labels,

that is, all checkpoints have the *same distribution* given the seed. SCB-C reports claims about **means**:

> H₀^jk: E[completion_j] = E[completion_k] (stage 1); E[conflicts_j] = E[conflicts_k] (stage 2), for every pair j, k.

Equal means with unequal dispersion satisfy H₀^jk but violate H₀^exch, so v3 has no guarantee for the reported claims.

**[Observed] Mechanism** (400 replicates of `N_hetero_conf_null`: equal mean conflicts, negative-binomial size 0.05–0.5 across the 20 checkpoints):
- Under permutation, every pair's difference mixes the dispersion of all K checkpoints, so v3 standardises every pair by roughly the same pooled spread.
- Median ratio of a pair's own standard error to its permutation SD:

  | Pair | Own SE / permutation SD | Effect |
  |---|---|---|
  | the two most dispersed checkpoints | 1.22 | anti-conservative |
  | mixed pairs | 0.92 | about right |
  | the two least dispersed checkpoints | 0.55 | conservative |

- False eliminations therefore concentrate on the most dispersed checkpoints.

**[Observed] v3 is not consistent for mean hypotheses.** The ratio does not shrink with more seeds, so the error grows with S instead of converging to α. Stage-2 rate of any elimination over all pairs, `N_hetero_conf_null`, α = 0.05, with Wilson 95 % intervals:

| Seeds S | Replicates | v3 | v4 |
|---|---|---|---|
| 20 | 2 000 | 0.091 [0.079, 0.104] | 0.285 [0.266, 0.306] |
| 80 | 1 000 | 0.121 [0.102, 0.143] | 0.168 [0.146, 0.192] |
| 320 (exploratory) | 200 | 0.14 | 0.105 |

The 0.083 of §21.4 is the same effect, measured inside the two-stage procedure.

**[Interpretation] Conclusion on v3.** v3 controls the family-wise error only when the checkpoints are exchangeable. The checkpoints of one training run are not: their conflict and failure distributions differ in shape, not only in mean (§22.4). **v3 must not be described as controlling mean-comparison errors.** The near-nominal rates of §21.4 hold for the homoscedastic synthetic nulls only. §21.4 and §21.9 are corrected accordingly.

### 22.3 Heteroscedasticity-robust alternatives

All four candidates are max-T over all pairs, two-sided, on the same per-seed contrasts:
- **Completion:** delivered / mean released per seed. Its mean equals the pooled completion exactly, because released counts are identical across the checkpoints of a seed; this was verified on all six runs, on blocks A and S.
- **Conflicts:** mean conflicts per episode, per seed.

| Method | Reference distribution | Exact for | Asymptotically valid for unequal-variance means? |
|---|---|---|---|
| v2 bootstrap-t (L1) | seed bootstrap, SE recomputed per resample | — (approximate) | yes; second-order accurate under skewness |
| v3 label permutation (L1) | within-seed label permutation, pooled SD | H₀^exch | **no** |
| v3s studentised label permutation (new, analysis only) | within-seed label permutation, per-pair paired t recomputed per permutation (paired analogue of Chung & Romano, 2013) | H₀^exch | yes |
| v4 studentised sign flip (new, `studentize="signflip"`) | one random sign per seed, shared by all pairs; per-pair t recomputed on the flipped differences | each pair's differences symmetric about 0, any variances | yes |

**How v4 works.** Each pair's reference uses only that pair's data, which removes v3's pooling. Sharing the signs across pairs keeps their dependence, so the max-T controls the family-wise error, strongly, because each pair's reference is unaffected by the other pairs.

**[Observed] v4 fixes heteroscedasticity.** In the test `test_sign_flip_is_exact_for_symmetric_heteroscedastic_contrasts`:
- design: K = 8, S = 20, Gaussian values with standard deviations 0.1–3, equal means, 200 replicates;
- result: v4 0.045, v3 0.125.

**[Observed] Skewness and sparsity are the larger problem.** Per seed, undelivered tasks and conflicts are mostly 0 or small, with rare large values (a permanent blockage, an oscillation episode). The difference between two such checkpoints is skewed and rests on a few seeds. The sample mean and SD of the difference are then strongly dependent, so the studentised statistic is asymmetric, and every method with a symmetric reference (v3s, v4) rejects too often.

**[Interpretation] This cannot be fixed by a better test.** It is a finite-sample property of mean comparisons: no test controls the size of a mean comparison uniformly over all distributions at a fixed sample size (Bahadur & Savage, 1956). Validity therefore has to be judged in realistic null worlds (§22.4). The bootstrap-t is valid there because its resamples reproduce the sparsity. This is also why its critical values are large (§21.4).

### 22.4 Error rates under null worlds, with Monte Carlo uncertainty

**Design** (`tools/selection/null_study.py`; 1 000 resamples per test).

| Element | Specification |
|---|---|
| Synthetic worlds | the L1 generator, at S = 20 (2 000 replicates) and S = 80 (1 000 replicates) |
| Plug-in null worlds | the real per-seed tables of each of the six runs, with every checkpoint's undelivered counts and conflicts rescaled multiplicatively to the run's median mean; dispersion, skewness, sparsity and seed pairing are kept, and every pairwise null is true |
| Block-A plug-in | 20 seeds drawn with replacement from the 20 block-A seeds; 500 replicates per run |
| Block-S plug-in | 80 seeds drawn with replacement from the 80 block-S seeds; 300 replicates per run |

**Rates reported**, per stage (any elimination), α = 0.05:
- **Joint** = an elimination in stage 1 or stage 2 (stage 2 among the stage-1 survivors), at α = 0.025 + 0.025. This is the overall error of the two-stage procedure.
- **Marks:** ↑ / ↓ = differs from 0.05, two-sided exact binomial test p < 0.05.

**Synthetic worlds.** Each cell gives the rate [Wilson 95 % interval].

| World | Stage | v2 | v3 | v3s | v4 |
|---|---|---|---|---|---|
| Homoscedastic null, S = 20 | completion | 0.002 [0.001, 0.004] ↓ | 0.058 [0.048, 0.069] | 0.053 [0.044, 0.064] | 0.055 [0.046, 0.066] |
| | conflicts | 0.000 [0.000, 0.002] ↓ | 0.051 [0.042, 0.061] | 0.044 [0.036, 0.054] | 0.042 [0.034, 0.051] |
| | joint (0.025 + 0.025) | 0.001 [0.000, 0.003] ↓ | 0.054 [0.045, 0.065] | 0.050 [0.041, 0.060] | 0.051 [0.042, 0.062] |
| Homoscedastic null, S = 80 | completion | 0.029 [0.020, 0.041] ↓ | 0.051 [0.039, 0.066] | 0.053 [0.041, 0.069] | 0.053 [0.041, 0.069] |
| | conflicts | 0.006 [0.003, 0.013] ↓ | 0.057 [0.044, 0.073] | 0.058 [0.045, 0.074] | 0.056 [0.043, 0.072] |
| | joint | 0.010 [0.005, 0.018] ↓ | 0.058 [0.045, 0.074] | 0.049 [0.037, 0.064] | 0.050 [0.038, 0.065] |
| Unequal conflict dispersion, S = 20 | conflicts | 0.000 [0.000, 0.002] ↓ | **0.091** [0.079, 0.104] ↑ | **0.302** [0.282, 0.322] ↑ | **0.285** [0.266, 0.306] ↑ |
| | joint | 0.001 [0.000, 0.003] ↓ | **0.082** [0.070, 0.094] ↑ | **0.237** [0.219, 0.256] ↑ | **0.225** [0.207, 0.244] ↑ |
| Unequal conflict dispersion, S = 80 | conflicts | 0.017 [0.011, 0.027] ↓ | **0.121** [0.102, 0.143] ↑ | **0.183** [0.160, 0.208] ↑ | **0.168** [0.146, 0.192] ↑ |
| | joint | 0.017 [0.011, 0.027] ↓ | **0.088** [0.072, 0.107] ↑ | **0.142** [0.122, 0.165] ↑ | **0.129** [0.110, 0.151] ↑ |
| Unequal completion noise, S = 20 | completion | 0.001 [0.000, 0.003] ↓ | **0.070** [0.060, 0.083] ↑ | **0.090** [0.079, 0.104] ↑ | **0.087** [0.076, 0.101] ↑ |
| | joint | 0.001 [0.000, 0.003] ↓ | **0.069** [0.059, 0.081] ↑ | **0.069** [0.058, 0.080] ↑ | **0.063** [0.053, 0.075] ↑ |
| Unequal completion noise, S = 80 | completion | 0.040 [0.030, 0.054] | **0.089** [0.073, 0.108] ↑ | **0.087** [0.071, 0.106] ↑ | **0.088** [0.072, 0.107] ↑ |
| | joint | 0.017 [0.011, 0.027] ↓ | **0.087** [0.071, 0.106] ↑ | **0.076** [0.061, 0.094] ↑ | **0.077** [0.062, 0.095] ↑ |

**Realistic plug-in null worlds.** Each cell gives the range over the six runs (mean). The Monte Carlo half-width at a rate of 0.05 is about 0.019 (block A, 500 replicates) and 0.025 (block S, 300 replicates).

| World | Stage | v2 | v3 | v3s | v4 |
|---|---|---|---|---|---|
| Block A, S = 20 | completion | 0.002–0.014 (0.008) | **0.102–0.156** (0.122) | **0.144–0.380** (0.234) | **0.154–0.362** (0.228) |
| | conflicts | 0.000–0.006 (0.002) | **0.148–0.204** (0.175) | **0.730–0.924** (0.816) | **0.694–0.904** (0.793) |
| | joint (0.025 + 0.025) | 0.000–0.004 (0.002) | **0.144–0.212** (0.183) | **0.688–0.912** (0.801) | **0.652–0.898** (0.778) |
| Block S, S = 80 | completion | 0.027–0.057 (0.045) | **0.070–0.123** (0.098) | 0.053–0.110 (0.080) | 0.050–0.110 (0.081) |
| | conflicts | 0.000–0.007 (0.004) | **0.093–0.203** (0.157) | **0.777–0.940** (0.855) | **0.727–0.927** (0.820) |
| | joint (0.025 + 0.025) | 0.017–0.040 (0.027) | **0.093–0.200** (0.161) | **0.743–0.923** (0.831) | **0.697–0.910** (0.793) |

The study also ran block-S worlds with S = 20 and 40 seeds drawn *without* replacement from the 80. Those are conservative by construction, because a subsample of a finite population varies less than the test assumes. They are stored, but are not used for conclusions.

**[Observed] Findings.**
1. **Only v2 controls the error in every world.** Its highest rates over all worlds:
   - single stage: 0.057 (completion, block-S world, ippo_seed1; not significantly above 0.05);
   - joint: 0.040.
2. **v3 is inflated in every realistic world**, at 0.07–0.21.
3. **v3s and v4 are valid for symmetric data but grossly invalid for conflicts here** (0.65–0.94), and also inflated for completion at 20 seeds.
   - The false eliminations are not random. They separate checkpoints with the same mean but different shape: consistently moderate conflicts versus mostly zero with rare bursts.
   - So v4 answers "fewer conflicts on a typical seed", not "fewer conflicts on average".
4. **Power of the valid method on the real data** (`combined_power.py`, v2 at α = 0.025 per stage, 10 000 resamples; checkpoints kept by stage 1, out of 20):

   | Selection data | Kept by stage 1 | Completion q | Conflict q | Conflict eliminations |
   |---|---|---|---|---|
   | A (20 seeds) | 19–20 | 5.8–8.0 | 36–104 | 0 |
   | S (80 seeds) | 7–14 | ≈ 4.0 | 15–30 | 0 |
   | A ∪ S (100 seeds) | 6–15 | ≈ 4.0 | 12–23 | 0, in every run |

   - **Stage 1 (completion)** has useful power from about 80 seeds.
   - **Stage 2 (conflicts)** has none even at 100 seeds, because conflicts per seed are heavy-tailed.

### 22.5 Selection block S: 80 new seeds

**[Observed] Seed separation** (`selection_block.py check`, `seed_check.json`):
- **Block S** is seeds 13 000–13 079. It is disjoint by construction from:
  - training 0–9 999;
  - A 10 000–10 019 and B 10 020–10 099;
  - C 11 000–11 119 and D 12 000–12 199;
  - the test seeds 20 000–20 999.
- **Unused:** none of the 537 stored `.json` / `.jsonl` result files contains a block-S seed.
- **Distinct instances:** all 3 × 80 scenario instances are unique, and none equals an instance of blocks A, B, C or D. The fingerprint covers map, robots, tasks, arrivals and events, excluding name and seed.
- **C and D:** only the scenario generator was called for them; no policy was run on them.

**Evaluation** (`selection_block.py run`):
- **Coverage:** all 20 stored step checkpoints (50 k–1 M) of mappo_seed1–3 and ippo_seed1–3 × 80 seeds × 3 levels: **28 800 A\* greedy episodes**.
- **Runner:** the production runner `run_episode`, unchanged. Its block guard only knows the Issue #5 blocks, so the worker processes replace it with a guard that accepts block S only; the test `test_guard_accepts_only_block_s` covers it.
- **Reproduction check:** the same path reproduces a stored block-A validation record exactly (delivered, released, conflicts, return, steps).
- **Manifest** (`manifest.json`): SHA-256 of all 120 checkpoints and of the code that ran.
  - `selection_block.py` received a lint/type-only edit afterwards; the manifest records both hashes.
- **Outcome:** 0 physical collisions. Released counts are identical across the checkpoints of every seed.
- **Wall time:** **49 min** with 10 workers, above the ~30 min estimate. For most of the run, two other analysis processes (3 cores) shared the 12 logical CPUs.

**[Observed] What block S shows** (pooled completion over 240 episodes per checkpoint):

| Run | Best block-S completion (step) | Late range (≥ 200 k) | Checkpoints ≥ 0.99 |
|---|---|---|---|
| mappo_seed1 | 0.9927 (300 k) | 0.9687–0.9927 | 1 |
| mappo_seed2 | 0.9896 (350 k) | 0.9640–0.9896 | 0 |
| mappo_seed3 | 0.9918 (350 k) | 0.9737–0.9918 | 2 |
| ippo_seed1 | 0.9868 (200 k) | 0.9493–0.9868 | 0 |
| ippo_seed2 | 0.9886 (250 k) | 0.9620–0.9886 | 0 |
| ippo_seed3 | 0.9868 (500 k) | 0.9575–0.9868 | 0 |

- **Agreement with block A:** across the 20 checkpoints, block-A and block-S estimates correlate 0.65–0.96 for completion and 0.88–0.95 for log conflicts.
- **Level shift:** block A is higher by 0.0043 on average (SD 0.0067). That is the shared seed effect of one 20-seed draw, not selection bias, since every checkpoint is included.
- **Resolution of a completion estimate** (from the 80-seed population, late checkpoints, medians):

  | Selection seeds | SE of one checkpoint | SE of a paired difference |
  |---|---|---|
  | 20 | ≈ 0.006 | ≈ 0.007 |
  | 40 | ≈ 0.004 | ≈ 0.005 |

- **[Interpretation]** With 20 seeds, many late checkpoints cannot be ordered by completion. Only 3 of 120 checkpoints reach the 0.99 objective on fresh seeds.

### 22.6 Out-of-sample comparison of the selection rules

**Design** (`rule_compare.py`):
1. **A → S:** each rule selects on block A (20 seeds) and is scored on all 80 block-S seeds; mean over the six runs.
2. **Splits of S:** 200 random splits per run of the 80 block-S seeds into a selection part (20 or 40 seeds) and an evaluation part (the other 60 or 40); mean over 6 × 200 splits.

**Scores of the pick on the evaluation seeds:** pooled completion; *shortfall* from the evaluation-best checkpoint; mean conflicts per episode (median in brackets); and the share of picks with completion ≥ 0.99. Each cell reads completion / shortfall / conflicts (median) / share ≥ 0.99.

| Rule | A → S (6 runs) | Split, 20 selection seeds | Split, 40 selection seeds |
|---|---|---|---|
| Historical (completion, then return) | 0.9872 / 0.0022 / 23.6 (13.6) / 0.00 | 0.9863 / 0.0035 / 13.4 (7.9) / 0.15 | 0.9868 / 0.0035 / 14.4 (10.3) / 0.20 |
| Completion-first (τ = 0) | 0.9872 / 0.0022 / 23.6 (13.6) / 0.00 | 0.9863 / 0.0035 / 13.5 (7.9) / 0.16 | 0.9868 / 0.0035 / 14.4 (10.3) / 0.20 |
| **One-SE rule** (§22.7) | 0.9852 / 0.0042 / **5.2** (4.4) / 0.00 | 0.9846 / 0.0051 / **6.5** (4.8) / 0.09 | 0.9851 / 0.0052 / **7.0** (5.3) / 0.12 |
| Two-SE rule | 0.9778 / 0.0116 / 2.1 (1.8) / 0.00 | 0.9803 / 0.0095 / 3.7 (2.6) / 0.01 | 0.9829 / 0.0074 / 3.8 (2.2) / 0.05 |
| Constrained: fewest conflicts with completion ≥ 0.99, else completion-first | 0.9844 / 0.0050 / 4.9 (4.0) / 0.00 | 0.9855 / 0.0042 / 10.0 (5.6) / 0.10 | 0.9863 / 0.0041 / 12.4 (7.6) / 0.14 |
| Completion-first after v2 stages (0.025 + 0.025) | 0.9872 / 0.0022 / 23.6 (13.6) / 0.00 | 0.9863 / 0.0035 / 13.5 (7.9) / 0.16 | 0.9868 / 0.0035 / 14.4 (10.3) / 0.20 |
| Completion-first after v4 stages (0.05 + 0.05) | 0.9866 / 0.0028 / 7.1 (4.9) / 0.00 | 0.9863 / 0.0035 / 11.7 (7.1) / 0.15 | 0.9869 / 0.0035 / 9.6 (8.5) / 0.22 |
| Conflicts-first among v2 stage-1 survivors | 0.9719 / 0.0175 / 3.2 (3.1) / 0.00 | 0.9700 / 0.0197 / 2.2 (1.6) / 0.00 | 0.9717 / 0.0186 / 2.3 (1.7) / 0.00 |
| Conflicts-first, no stage | 0.9719 / 0.0175 / 3.2 (3.1) / 0.00 | 0.9699 / 0.0198 / 2.2 (1.6) / 0.00 | 0.9684 / 0.0219 / 1.9 (1.2) / 0.00 |
| Latest step | 0.9766 / 0.0128 / 2.5 (2.4) / 0.00 | 0.9766 / 0.0131 / 2.5 (2.0) / 0.00 | 0.9766 / 0.0137 / 2.6 (2.1) / 0.00 |

**Tolerance curve:** the fewest conflicts among checkpoints within τ of the best completion *estimate*. This is a description of the trade-off, not a test.

| τ | Completion (20 sel.) | Conflicts (20 sel.) | Completion (40 sel.) | Conflicts (40 sel.) |
|---|---|---|---|---|
| 0 | 0.9863 | 13.5 | 0.9868 | 14.4 |
| 0.001 | 0.9860 | 11.4 | 0.9865 | 11.9 |
| 0.0025 | 0.9858 | 9.9 | 0.9858 | 9.4 |
| 0.005 | 0.9847 | 6.4 | 0.9844 | 5.8 |
| 0.0075 | 0.9832 | 4.8 | 0.9831 | 3.9 |
| 0.01 | 0.9817 | 4.0 | 0.9821 | 3.3 |
| 0.015 | 0.9781 | 3.2 | 0.9790 | 3.2 |
| 0.02 | 0.9754 | 3.0 | 0.9752 | 2.9 |
| ∞ (conflicts-first) | 0.9699 | 2.2 | 0.9684 | 1.9 |

**[Observed] Findings.**
1. **The final rule decides; the inference stages barely matter.**
   - The valid stages (v2) never change a pick at 20 or 40 seeds.
   - v4's stage 2 lowers conflicts with completion-first, but it is the invalid stage (§22.4).
2. **Completion-first and conflicts-first are the two ends of a monotone trade-off.**
   - Conflicts-first loses 0.017–0.022 completion, 5–8 times the shortfall of completion-first.
   - Completion-first pays 6–8× the conflicts.
3. **The ≥ 0.99 constraint rarely binds on fresh seeds.** Few checkpoints truly reach 0.99, and a point estimate ≥ 0.99 on 20–40 seeds is mostly noise: only 10–14 % of its picks meet 0.99 on the evaluation seeds. It behaves almost like completion-first.
4. **The one-SE rule is the best balanced.** Against completion-first it gives up 0.0017 completion (both seed counts) and halves conflicts (13.5 → 6.5 and 14.4 → 7.0); from A → S, 23.6 → 5.2. The direction is the same in every run:

   | Measure | Range over the six runs and both seed counts |
   |---|---|
   | completion change | −0.0008 to −0.0029 |
   | conflict change | −1.5 to −20.9 per episode in 11 of 12 cases; +0.04 in one |

5. **Uncertainty of these comparisons.** The splits reuse the same 80 seeds and six runs, so the spread across splits understates the real uncertainty. The block-S comparison is development evidence, not a confirmatory test. It shows the size and direction of the trade-off.

### 22.7 The one-standard-error rule: an engineering tolerance, justified

**Definition** (`rule_compare.one_se_pick`):
1. Let j be the checkpoint with the highest completion estimate on the selection seeds.
2. A checkpoint k is in the band if comp_j − comp_k ≤ SE_jk, where SE_jk is the standard error of the paired per-seed completion difference between j and k on the selection seeds.
3. Pick the band member with the fewest conflicts per episode; break ties by higher completion, then the later step.

This is the one-standard-error rule of Breiman, Friedman, Olshen & Stone (1984), used here for checkpoints instead of tree sizes.

**Why it is justified [Observed]** (200 splits per run of block S):
- **Inside the band, the selection data cannot order completion.**
  - Rank correlation of completion between selection and evaluation seeds, among band members: 0.14 (20 seeds), 0.18 (40 seeds).
  - Choosing the top completion estimate inside the band gains only 0.0010 / 0.0008 completion on the evaluation seeds over the band average.
- **They can still order conflicts:** the corresponding rank correlation is 0.64 / 0.57.
- So inside the band, ranking by completion is ranking by noise. Ranking by conflicts uses information that transfers to new seeds.

**What it is not.**
- It is **not a hypothesis test** and **not an equivalence margin**. It claims nothing about band members being equal, and it has no α.
- It introduces **no fixed number**: no 0.005, no 0.99. The band width is measured from the selection data and shrinks as seeds are added.
  - Median SE_jk: ≈ 0.007 with 20 seeds, ≈ 0.005 with 40, 0.0028 with 100 (A ∪ S).

**Cost.** 0.0017 completion on average, against roughly half the conflicts (§22.6). Whether that trade is acceptable is the student's decision.
- If completion must be maximised with no trade, use completion-first.
- The two-SE rule is shown for comparison only. It costs 0.007–0.012 and is not recommended.

### 22.8 Family-wise error allocation (overall α = 0.05)

**The error being controlled** is the probability of any false claim "checkpoint j is better than checkpoint k" made by the procedure: a false elimination at either stage.

**Bound.** With stage-wise levels α₁ (completion) and α₂ (conflicts):
- the overall error is at most α₁ + α₂ (union bound, valid under any dependence between the stages);
- each stage's family is fixed in advance (all valid pairs), so restricting stage 2 to the stage-1 survivors cannot raise its error.

| Option | α₁ | α₂ | Bound | Observed joint error, v2, realistic worlds | What it buys |
|---|---|---|---|---|---|
| **Recommended** | **0.05** | **— (no conflict test)** | **0.05** | stage 1 alone: 0.002–0.057, mean 0.045 at 80 seeds | all α on the only stage with power (§22.4, finding 4) |
| Equal split | 0.025 | 0.025 | 0.05 | 0.000–0.040 | conflict claims, which never occurred at 20–100 seeds |
| Stage-wise 0.05 (§21) | 0.05 | 0.05 | 0.10 | — | **does not meet 0.05**; withdrawn |

**Rationale for the recommendation.**
- v2's conflict stage made no elimination in any run, even at 100 seeds; α spent on it buys nothing.
- No valid alternative with power exists for mean conflicts at these sample sizes (§22.3).
- Conflicts therefore enter the selection through the pre-declared final rule (§22.7), not through a test.
- If the student wants conflict statements reported, use 0.025 + 0.025: the bound still holds, at a small cost in completion power.

**Where the confirmatory α goes.** The selection rule itself makes no inferential claim. The statement "the new selection is better / not worse than the old one" is tested on block C (development; reusable) under the pre-registration of §4 and §12, and confirmed once on block D (final evaluation). Each test has its own α, and the selection-stage α above does not consume it. *(Wording corrected 2026-11-21: block C is the development block, not a confirmation block.)*

### 22.9 Recommendation (not frozen; awaiting the student's approval)

*Approved 2026-11-21 with amendments: no conflict stage, ties to the latest step. Frozen as SCB-C v5 in §23, which supersedes the details below where they differ.*

**[Interpretation] Proposed selector for block C** ("SCB-C v5"):
1. **Data:** select on block A ∪ block S (100 seeds × 3 levels).
   - Block S is a selection block by design. It is now used and must never become a confirmation block.
   - Blocks C and D stay unconsumed.
2. **Exclusions:** as §21.1 (incomplete, invalid, any physical collision).
3. **Stage 1:** v2 seed bootstrap-t max-T on completion, α = 0.05, 10 000 resamples, seed 0. It reports which checkpoints are *shown* worse in completion, with family-wise error ≤ 0.05 in every world tested.
4. **No conflict-stage test** (§22.8).
5. **Final rule:** the one-SE rule (§22.7) among the stage-1 survivors.
   - On A ∪ S, stage 1 at α = 0.05 has q = 3.70–3.78. In all six runs the one-SE pick lies inside the stage-1 survivors.
   - Fallback, if the student prefers no completion trade: completion-first.
6. **Freeze before block C:** rule text, α, resample count and seed, code hashes of `scbc.py` and `rule_compare.py`, and the picks.

**What the rule would pick on A ∪ S** (`combined_power.json`; informational, not frozen; completion / conflicts are A ∪ S estimates):

| Run | Historical rule | Completion-first | One-SE rule |
|---|---|---|---|
| mappo_seed1 | 300 k (0.9928 / 15.6) | 300 k (0.9928 / 15.6) | 300 k (0.9928 / 15.6) |
| mappo_seed2 | 300 k (0.9905 / 11.3) | 300 k (0.9905 / 11.3) | 500 k (0.9881 / 1.9) |
| mappo_seed3 | 600 k (0.9904 / 3.9) | 600 k (0.9904 / 3.9) | 500 k (0.9888 / 3.3) |
| ippo_seed1 | 500 k (0.9878 / 6.7) | 500 k (0.9878 / 6.7) | 500 k (0.9878 / 6.7) |
| ippo_seed2 | 250 k (0.9893 / 15.2) | 250 k (0.9893 / 15.2) | 250 k (0.9893 / 15.2) |
| ippo_seed3 | 500 k (0.9881 / 3.7) | 500 k (0.9881 / 3.7) | 500 k (0.9881 / 3.7) |

For comparison, the stored Issue #6 selections used block A only.

**Withdrawn or not recommended:**
- **v3:** not valid for mean claims.
- **v4 and v3s:** valid only for symmetric contrasts, and invalid for this project's conflicts. v4 stays in `scbc.py` with that limitation documented.
- **Stage-wise 0.05 + 0.05:** joint bound 0.10.
- **Constrained ≥ 0.99:** almost never binds on fresh seeds.

**[Interpretation] Readiness.** The procedure is now statistically justified in the sense that matters for block C:
- its only inferential claims have verified error control;
- its final rule is an explicit, measured engineering choice.

It is ready for block C **once the student approves items 1–6**, or chooses the fallback. Block C remains unconsumed; block D remains reserved.

**Decisions for the student:**
1. The one-SE rule or completion-first.
2. Selection on A ∪ S (recommended) or on A only.
3. α: 0.05 on completion only (recommended) or 0.025 + 0.025.
4. Whether block S may be cited as development evidence in the report.

### 22.10 Files, tests, runtime and reproducibility

**Code** (isolated tooling; no production file changed):

| File | Change |
|---|---|
| `tools/selection/scbc.py` | added `sign_flips`, `sign_flip_band`, `studentize="signflip"` (v4), the `constrained` final rule with `completion_floor`; docstrings record v3's exchangeability null. Default unchanged (`perm`), so the L1 results stay reproducible. |
| `tools/selection/simulate.py` | `--method signflip` accepted |
| `tools/selection/selection_block.py` (new) | block-S seed checks, guarded evaluation runner, manifest, loader |
| `tools/selection/null_study.py` (new) | v2 / v3 / v3s / v4 under synthetic and plug-in nulls; Wilson intervals; binomial tests |
| `tools/selection/rule_compare.py` (new) | A → S and split comparisons, tolerance curve, one-SE rule |
| `tools/selection/combined_power.py` (new) | power on A, S and A ∪ S; informational picks |
| `tests/tools/test_scbc.py` | +6 tests: sign-flip loop check, exact enumeration of all 2⁶ sign vectors, contrast mean = pooled completion, elimination and determinism, symmetric-heteroscedastic exactness (v4 vs v3), constrained rule |
| `tests/tools/test_followup.py` (new) | 10 tests: block disjointness, seed guard, scenario fingerprint, plug-in null construction, v3s loop check, determinism, Wilson / binomial, rule-comparison picks and scores, one-SE rule |

**Outputs** (new; nothing earlier overwritten):
- `results/dev6a/l1_selection/selection_block/`: `seed_check.json`, `records.jsonl` (28 800 records), `manifest.json`;
- `results/dev6a/l1_selection/followup/`:
  - `null_study_syn_A_null_K20_S20_S20_syn_N_hetero_conf_null_S20_syn_N_hetero_comp_null_S20_plugA.json`;
  - `null_study_plugS_syn_A_null_K20_S20_S80_syn_N_hetero_conf_null_S80_syn_N_hetero_comp_null_S80.json`;
  - `rule_compare.json`, `rule_compare_splits.jsonl`, `combined_power.json`.

**Commands:**

```text
python tools/selection/selection_block.py check --out results/dev6a/l1_selection/selection_block
python tools/selection/selection_block.py run --out results/dev6a/l1_selection/selection_block --workers 10
python tools/selection/null_study.py --out results/dev6a/l1_selection/followup --workers 2 --only syn_A_null_K20_S20_S20,syn_N_hetero_conf_null_S20,syn_N_hetero_comp_null_S20,plugA
python tools/selection/null_study.py --out results/dev6a/l1_selection/followup --workers 9 --reps-plugin 300 --only plugS,syn_A_null_K20_S20_S80,syn_N_hetero_conf_null_S80,syn_N_hetero_comp_null_S80
python tools/selection/rule_compare.py --out results/dev6a/l1_selection/followup --workers 10
python tools/selection/combined_power.py --out results/dev6a/l1_selection/followup
```

**Runtime** (Windows, 12 logical CPUs):

| Run | Wall time | Workers |
|---|---|---|
| block-S evaluation | 2 952 s | 10 |
| null study, S = 20 and block A | 4 436 s | 2 |
| null study, S = 80 and block S | 3 978 s | 9 |
| rule comparison | 546 s | 10 |
| combined power | ≈ 150 s | 1 |

**Checks [Observed]:**
- **Tests:** `pytest tests/tools` 46 passed. Full fast suite: **933 passed, 1 skipped** (the POSIX-only `peak_rss` test), 62 deselected, 294 s; that is the 917 of §21.5 plus 16 new.
- **Lint and types:** `ruff check .` and `ruff format --check .` clean (187 files); `mypy --strict` on `tools/selection` clean (7 files, `MYPYPATH=tools/selection`); project `mypy` clean (83 files).
- **L1 reproduction:** re-running `B_null_K5` and `P_ippo_seed2` (v3) and `B_null_K5` (v2) with the modified code reproduces every stored summary field exactly. The only additions are the `pick_constrained` fields.
- **Earlier outputs untouched:** the L1 (`results/dev6a/l1_selection/` root, `v2/`, `v3/`) and 6A.0 outputs predate this follow-up.
- **Historical hashes:** all 1 310 files hashed before Phase 6A.0 are unchanged (0 changed, 0 missing).
- **Handoff file:** `docs/SESSION_HANDOFF.md` SHA-256 is still `ba4a439c…5fa3`.
- **`models/`:** does not exist.

### 22.11 Limitations

- **The plug-in null worlds are one realistic construction.** Multiplicative rescaling to equal means keeps each checkpoint's shape. Other constructions (additive shifts, quantile matching) could give somewhat different rates. The qualitative ranking (v2 valid; v3 inflated; v3s and v4 grossly inflated for conflicts) held in every world tested.
- **No uniform guarantee.** v2's validity is shown in these worlds, not proved (Bahadur & Savage). Its conservativeness at 20 seeds means its guarantee is cheap there, but so is its power.
- **Development evidence only.** The rule comparison rests on six runs and one 80-seed block. The 0.0017 / 50 % trade-off is an estimate from development data; block C is where any claim about the frozen selector is tested.
- **Block S is now used for selection.** It must never serve as a confirmation block.
- **Only greedy A\*, PPO.** MADQL runs and RRT\* were not evaluated on block S, and the recommendation is limited to the six main PPO runs. MADQL would need about 4× the episodes per run.
- **Over budget:** the evaluation took 49 min, against an approved estimate of about 30 min.

## 23. Frozen specification: SCB-C v5 (approved and frozen, 2026-11-21)

### 23.1 Approval and status

**Approved by the student (2026-11-21).**
1. **Selection rule:** the one-standard-error rule.
2. **Selection data:** blocks A and S combined (100 seeds).
3. **Statistical elimination:** v2 seed bootstrap-t, on completion only, α = 0.05.
4. **No conflict significance stage.** Conflicts are the secondary selection criterion.
5. **Procedure:** after the statistically supported completion exclusions, take the checkpoints within one standard error of the highest completion estimate, and select the one with the fewest conflicts per episode.
6. **Ties:** the latest training checkpoint.
7. **No fixed completion tolerance:** no Δ = 0.005 and no 0.99 floor.

**Status.** The specification below, its implementation (`tools/selection/scbc_frozen.py`) and its tests are **frozen**. Any change requires a new version number and the student's approval.
- **Not done** (by scope): no block-C or block-D evaluation, no PPO training, no L2 work, no production policy change, no Git operation.
- `docs/SESSION_HANDOFF.md` is not touched.

### 23.2 Roles of the seed blocks

| Block | Seeds | Role under SCB-C v5 |
|---|---|---|
| A | 10 000–10 019 | **Selection data.** Used for checkpoint selection since Issue #5. |
| S | 13 000–13 079 | **Selection data.** Created for the L1 selection study (§22.5) and now part of the frozen selector's input. |
| B | 10 020–10 099 | Development comparisons of Issues #5 and #6; not used by SCB-C. |
| C | 11 000–11 119 | Development and reusable experimental block. Not yet used. |
| D | 12 000–12 199 | **Untouched; reserved for the final evaluation.** |
| Test | 20 000–20 999 | Reserved for Issue #8; never used here. |

**Reporting rules (binding for every later report):**
- Blocks A and S are reported together as **selection data**, and always named as such.
- Block S may be discussed as part of the selection study (§22). It is **not** independent held-out validation. Its estimates, like block A's, must not be reported as the performance of the selected checkpoints, because the selection is optimistic on its own data.
- Performance of the selected checkpoints is reported only from blocks not used to select them: block C for development, block D for the final evaluation.
- The one-SE rule is described as a **selection heuristic, not a statistical equivalence test**.

### 23.3 Mathematical definition

**Notation.**

| Symbol | Meaning |
|---|---|
| K = 20 | the stored step checkpoints k of one training run |
| t_k | the training step of k; distinct within a run |
| 𝒮 | the selection seeds, 10 000–10 019 then 13 000–13 079, in that order; S = 100 |
| ℒ | the levels static, obstacles and full |
| e(k, s, ℓ) | one evaluation episode of k on seed s ∈ 𝒮 and level ℓ ∈ ℒ: A\* planner, greedy mode, shield off |
| d, r, c, x | an episode's delivered tasks, released tasks, proposed conflicts (`conflicts_total`) and physical collisions |

**Per-seed sums**, over the three levels of seed s:
- D_ks = Σ_ℓ d, R_ks = Σ_ℓ r, C_ks = Σ_ℓ c, X_ks = Σ_ℓ x.
- R̄_k = (1/S) Σ_s R_ks.

**E. Exclusions** (§23.4). 𝒱 is the set of valid checkpoints. If 𝒱 = ∅, no checkpoint is selected.

**Estimates** for k ∈ 𝒱:
- pooled completion ĉ_k = Σ_s D_ks / Σ_s R_ks, compared as an exact rational number;
- conflicts per episode m_k = Σ_s C_ks / (3S), compared through the integer total Σ_s C_ks, since every valid checkpoint has 3S = 300 episodes.

**Stage 1 — the only statistical stage: simultaneous completion comparison (v2 bootstrap-t).**
1. **Resamples.** w ∈ ℕ^{B×S} are seed-bootstrap multiplicities, row b counting how often each seed is drawn in S draws with replacement:
   - B = 10 000;
   - indices drawn by `numpy.random.default_rng(0).integers(0, S, (B, S))`;
   - one w shared by all checkpoints, which keeps the seed pairing.
2. **Resampled estimate:** ĉ*_bk = Σ_s w_bs D_ks / Σ_s w_bs R_ks.
3. **Standard errors (delta method).**
   - Influence values: z_ks = (D_ks − ĉ_k R_ks) / R̄_k, and in resample b, z*_bks = (D_ks − ĉ*_bk R_ks) / R̄*_bk with R̄*_bk = (1/S) Σ_s w_bs R_ks.
   - For each pair, SE²_jk = (v_j + v_k − 2 v_jk) / S, where v is the sample covariance matrix of z over the seeds (factor S/(S − 1)).
   - SE*_bjk is the same quantity computed from z*_b, weighted by w_b.
4. **Max-T statistic and critical value.**
   - Δ_jk = ĉ_j − ĉ_k.
   - T*_b = max over pairs j < k in 𝒱 with SE*_bjk > 0 of |Δ*_bjk − Δ_jk| / SE*_bjk.
   - q = the (1 − α) quantile of {T*_b}, NumPy method `higher`, α = 0.05.
5. **Elimination.**
   - Lower bound: L_jk = Δ_jk − q · SE_jk.
   - k is **eliminated** if L_jk > 0 for some j ∈ 𝒱 ("j is shown better than k").
   - The survivors 𝒱₁ = 𝒱 minus the eliminated checkpoints. 𝒱₁ is never empty: the checkpoint with the highest estimate cannot be shown worse.
   - The family is every pair of valid checkpoints. The family-wise error of the statement "every eliminated checkpoint is worse than some valid checkpoint in completion" is controlled at α in every world studied (§22.4), not uniformly.

**Stage 2 — the one-SE band (a heuristic, not a test).**
1. **Reference:** j* = the survivor with the largest ĉ; ties go to the largest t_k.
2. **Per-seed contrast:** for k ∈ 𝒱₁, δ_ks = D_j*s / R̄_j* − D_ks / R̄_k. Its mean over seeds equals ĉ_j* − ĉ_k exactly.
3. **Band standard error:** SE°_k = sd_s(δ_k) / √S, the sample SD with ddof = 1, computed in float64. SE°_j* = 0.
4. **Band:** ℬ = { k ∈ 𝒱₁ : ĉ_j* − ĉ_k ≤ SE°_k }.
   - The left side is the exact rational difference, rounded to float64; the comparison is in float64 with no tolerance.
   - j* ∈ ℬ always.

**Final pick.** k̂ = the member of ℬ with the smallest Σ_s C_ks (fewest conflicts per episode); ties go to the largest t_k.

**Output** (per run): k̂ and t_k̂, the exclusions with reasons, 𝒱₁, j*, ℬ, q, ĉ, m and SE°.

**No other parameter exists:**
- no completion tolerance, floor or equivalence margin;
- no conflict test;
- no return or step criterion other than the tie rule.

### 23.4 Exclusions, invalid and missing evaluations

Only the records of the run's 20 step checkpoints are read: block A from `results/training/<run>/validation.jsonl` (A\* entries), and block S from `results/dev6a/l1_selection/selection_block/records.jsonl`. Records of other checkpoints are ignored. The first matching rule wins:

| Order | Reason | Condition (any one suffices) |
|---|---|---|
| 1 | `invalid` | any record of the checkpoint with: a duplicate (checkpoint, seed, level); `controller` ≠ ppo, `planner` ≠ astar, `mode` ≠ greedy or `shield` ≠ off (a missing attribute counts as a mismatch); a seed outside 𝒮 or a level outside ℒ; a `block` field that does not match the seed (A for 10 000–10 019, S for 13 000–13 079); a count that is negative, non-integer, non-finite or boolean; delivered > released |
| 2 | `incomplete` | any of the 300 (seed, level) episodes missing, or a required field (seed, level, checkpoint, delivered, released, conflicts_total, physical_collisions) missing or null |
| 3 | `invalid` | no released task at all (Σ R = 0), or delivered > released after summation |
| 4 | `collision` | any physical collision in any episode (safety precedes performance) |

**Handling:**
- Nothing is imputed, repaired or re-run automatically.
- An excluded checkpoint takes no part in stage 1, the band or the pick, and it is listed with its reason.
- Re-running a missing episode is a new evaluation and needs the student's approval.

For the six main PPO runs, **all 120 checkpoints are complete and valid**: 0 exclusions, 0 collisions.

### 23.5 Bootstrap configuration and determinism

| Item | Frozen value |
|---|---|
| Resamples B | 10 000 |
| Generator | `numpy.random.default_rng(0)` (PCG64); NumPy 2.4.6 |
| Seed order | 𝒮 as listed in §23.3 (block A ascending, then block S ascending) |
| Quantile | `numpy.quantile(…, 0.95, method="higher")` |
| Interval | two-sided, max-T over all pairs of valid checkpoints |
| Arithmetic | completion ranking and ties: exact rationals; conflicts: integer totals; SEs, q and the band comparison: float64 |
| Tie-breaking | latest training step, for j* and for the pick; steps must be distinct (otherwise an error) |
| Record order | irrelevant (tested) |
| Checkpoint order | irrelevant: tables are sorted by step, and the max-T is order-invariant |

**Version caveat.** NumPy does not guarantee identical `Generator` streams across all future versions, so `selections.json` records the NumPy version. On a different version, a change of pick is possible only if a checkpoint lies within Monte Carlo error of the stage-1 boundary. On the frozen data, every band member is far from that boundary: band members lie within about 1 SE of j*, while elimination needs a lower bound above 0, about q ≈ 3.7 SE.

### 23.6 Rationale and limitations

**Rationale** (evidence in §22):
- **Stage 1.** v2 is the only simultaneous method whose error stayed at or below the target in every null world studied, at most 0.057 per stage. With 100 seeds it removes 6–14 of 20 checkpoints per run. Spending α on conflicts would have bought nothing: no conflict elimination was observed even at 100 seeds.
- **The one-SE band.** Inside one paired SE of the best completion estimate, the selection data do not order completion reliably: out-of-sample rank correlation 0.14–0.18, against 0.57–0.64 for conflicts. Ranking inside the band by conflicts therefore uses information that transfers, and ranking by completion would mostly use noise.
- **No fixed numbers.** The band width is measured from the selection data and shrinks with more seeds: median SE° ≈ 0.007 at 20 seeds, 0.0028 at 100. No margin had to be justified.

**Observed trade-off** (block-S splits, §22.6; development evidence, not a confirmatory result):

| Rule | Completion cost | Conflicts per episode |
|---|---|---|
| completion-first | — | 13.5 (20 seeds) / 14.4 (40 seeds) |
| one-SE rule | 0.0017 on average | 6.5 (20 seeds) / 7.0 (40 seeds) |

The direction is the same in all six runs:
- completion change −0.0008 to −0.0029;
- fewer conflicts in 11 of 12 run × seed-count cases.

**Limitations:**
- **What the rule is not.** It is a heuristic with no error guarantee and no equivalence claim: band members are not asserted to be equal in completion. The selected checkpoint can have slightly lower true completion than the best, about 0.0017 on average in the study.
- **What stage 1 guarantees.** Its error control is demonstrated in realistic null worlds, not proved uniformly (Bahadur & Savage, §22.3). It concerns only the eliminations, not the final pick.
- **Optimistic estimates.** The selection-data estimates of the picked checkpoint are optimistic (winner's curse). Their true performance is unknown until blocks C/D.
- **Same-data limitation.** The trade-off estimates come from the same six runs and block S that the selector now uses.
- **Scope.** Greedy A\* PPO only. MADQL, RRT\* and stochastic modes are not covered by this freeze.
- **Changing pick.** On mappo_seed2 and mappo_seed3, the one-SE pick is a different checkpoint from the completion-first pick. Elsewhere they coincide.

### 23.7 Frozen selections (selection data A ∪ S; **not** performance estimates)

`results/dev6a/l1_selection/frozen/selections.json`. In the table:
- **Issue #6 pick** is the historical rule on block A only (`best.json`).
- **ĉ / m** are A ∪ S selection-data estimates, given only to document the rule's decision.

| Run | Issue #6 pick | Kept by stage 1 | j* (ĉ / m) | Band (steps, k) | **SCB-C v5 pick** (ĉ / m) | SE° of the pick |
|---|---|---|---|---|---|---|
| mappo_seed1 | 200 k | 13/20 | 300 k (0.9928 / 15.6) | 200, 300 | **300 k** (0.9928 / 15.6) | 0 (= j*) |
| mappo_seed2 | 300 k | 11/20 | 300 k (0.9905 / 11.3) | 300, 350, 400, 500 | **500 k** (0.9881 / 1.9) | 0.0025 |
| mappo_seed3 | 150 k | 14/20 | 600 k (0.9904 / 3.9) | 150, 200, 300, 350, 500, 600 | **500 k** (0.9888 / 3.3) | 0.0020 |
| ippo_seed1 | 550 k | 7/20 | 500 k (0.9878 / 6.7) | 200, 350, 500 | **500 k** (0.9878 / 6.7) | 0 (= j*) |
| ippo_seed2 | 250 k | 10/20 | 250 k (0.9893 / 15.2) | 250 | **250 k** (0.9893 / 15.2) | 0 (= j*) |
| ippo_seed3 | 500 k | 6/20 | 500 k (0.9881 / 3.7) | 500 | **500 k** (0.9881 / 3.7) | 0 (= j*) |

- Critical values q = 3.70–3.78. There are no exclusions.
- The frozen picks differ from the Issue #6 picks in four runs (mappo_seed1–3, ippo_seed1) and agree in two (ippo_seed2, ippo_seed3).
- They equal the one-SE picks of §22.9 in all six runs. The tie rule and the restriction of the band to the survivors changed nothing here.

### 23.8 Implementation, tests and verification

**Code:**
- `tools/selection/scbc_frozen.py` (new) implements §23.3–23.5. It contains the constants, the strict loader `build_table` / `load_run`, the procedure `select`, and the CLI that writes `selections.json`.
- It reuses `scbc.resample_counts`, `pair_se`, `bootstrap_t_band`, `not_dominated` and `validate` from the L1 module. Both files are hashed into the output.
- No other tool and no production file was changed for the freeze.

**Tests:** `tests/tools/test_scbc_frozen.py`, 31 tests.

| Area | What is tested |
|---|---|
| Frozen constants | seeds A ∪ S = 100, disjoint from C, D and the test seeds; levels; α, B and the generator seed; expected record attributes; the block map |
| No forbidden parameters | `select` takes no tolerance, floor or conflict-α, and its source contains none |
| Loader | sums over levels; sorting by step; 14 cases of invalid or missing evaluations, each excluding the checkpoint with the specified reason and never selecting it; precedence of invalid over the missing cell it leaves; the block-field check; records of other checkpoints ignored; record-order invariance |
| Exclusions | released = 0 → invalid; a collision → excluded; all excluded → no selection |
| Procedure | a hand-computed band and pick (the fewest-conflict checkpoint outside the band is *not* chosen); conflict ties → latest step even against higher completion; reference ties → latest step; non-unique steps → error; exact rational completion ties; stage 1 identical to the study's v2 (`scbc.select(studentize="t")`: same survivors, same q); determinism |
| Property-based (Hypothesis, 40 cases) | pick ∈ band ⊆ survivors; j* has the highest survivor completion; the pick has the fewest band conflicts; ties go to the latest step |
| Regression on the frozen data | 20 × 100 complete tables and the six frozen picks. Skipped where `results/` is absent, e.g. on CI, because `results/` is not tracked. |

**Verification [Observed]:**
- `pytest tests/tools`: **77 passed** (46 from L1 and the follow-up, plus 31 new).
- Full fast suite: **964 passed, 1 skipped** (the POSIX-only `peak_rss` test), 62 deselected, 289 s; that is the 933 of §22.10 plus 31 new.
- `ruff check .`, `ruff format --check .`, `mypy --strict tools/selection` (`MYPYPATH=tools/selection`) and project `mypy`: clean.
- **Determinism:** two runs of `scbc_frozen.py` give byte-identical `selections.json` (5 s).
- **Hashes:** the 1 310 historical files are unchanged; the `SESSION_HANDOFF.md` SHA-256 is unchanged; `models/` does not exist.

**Frozen hashes (SHA-256):**

| File | SHA-256 |
|---|---|
| `tools/selection/scbc.py` | `b823445293aaaafa9f8c93d7029a836d4fd7fcfb5b1a261c11084077c6c55339` |
| `tools/selection/scbc_frozen.py` | `f913021f278577cf3a1335c69ba8f3e8ef6f5e2ab924748630384366668f641f` |

Input data hashes (the six `validation.jsonl` files and the block-S `records.jsonl`) are also in `selections.json`.

**Command:**

```text
python tools/selection/scbc_frozen.py --out results/dev6a/l1_selection/frozen
```

### 23.9 What is ready for the next approval

**Ready (frozen and verified):**
1. The SCB-C v5 specification (§23.3–23.5), its implementation and its 31 tests.
2. The frozen selections for the six main PPO runs (§23.7, `selections.json`), with code, data and NumPy hashes.
   - The set to compare on block C is the six Issue #6 checkpoints against the six SCB-C v5 checkpoints.
   - They differ in four runs (mappo_seed1–3, ippo_seed1), so block C has four distinct old/new pairs. ippo_seed2 and ippo_seed3 are unchanged.

**Needs the student's approval before it runs:**
1. **Block-C development evaluation** of the old and new selections, under the pre-registration of §4 and §12:
   - checkpoints: the four changed runs, both picks;
   - outcomes: completion, conflicts per episode, physical collisions and the secondary metrics, on all 120 block-C seeds × 3 levels, greedy A\*;
   - about 4 runs × 2 checkpoints × 360 episodes ≈ 2 900 episodes, a few minutes with 10 workers.
   - The paired analysis plan (seed-cluster bootstrap, Holm over the outcomes) must be confirmed or amended first.
2. **Whether to extend SCB-C v5 to MADQL**, which needs block-S episodes for the MADQL runs, and/or to RRT\*. Neither is covered by this freeze.
3. **L2**, the P items and **block D** remain out of scope, unchanged from §16 and §19.


## 24. Block C development evaluation: Issue #6 picks vs SCB-C v5 picks (verified, 2026-11-21)

### 24.1 Approval, scope and status of the evidence

Approved by the student on 2026-11-21. The comparison covers the four PPO runs whose pick changed under SCB-C v5. For each run, the old checkpoint is the Issue #6 pick and the new one is the SCB-C v5 pick. Both are evaluated on all 120 Block C seeds (11 000–11 119) × 3 levels, greedy A\*, shield off: 2 880 episodes.

Not evaluated:
- ippo_seed2 and ippo_seed3 (picks unchanged);
- MADQL and RRT\*;
- Block D, which was not passed even to the scenario generator.

Nothing was trained, no checkpoint was written, and no production file was changed.

**Status of the evidence.** Block C is the development / reusable block. The results below are development evidence about these four pairs of checkpoints. They are not a final or independent validation; that role is reserved for Block D. Block C played no part in the selection (the selection data were A ∪ S), so the estimates are not affected by selection optimism. **SCB-C v5 is unchanged and stays frozen.** Nothing here was used to tune it.

**Frozen beforehand.** The configuration and the analysis plan were written and hashed before any Block C episode was run:
- plan: `docs/research/2026-11-21-block-c-analysis-plan.md`, SHA-256 `42158054…6f615b7`;
- runner: `tools/selection/block_c.py`, `afc6f9dd…31a9b7`;
- analysis: `tools/selection/block_c_analysis.py`, `3fe86a77…60ebfec`;
- freeze time: 09:50:44, recorded in `results/dev6a/block_c/plan_freeze.json`.

The runner refuses to start if any of these hashes differs. They matched at the run and at the analysis. The plan defines:
- outcomes and families;
- the paired delta-method SE and the symmetric seed-cluster bootstrap-t (B = 10 000, `default_rng(0)`);
- Holm within each family;
- the validity gate;
- the safety reporting;
- the interpretation table.

They are not repeated here.

**One pre-run correction (before the freeze, before any episode).** The first `check` failed: each Issue #6 `best.pt` has a different SHA-256 from its step checkpoint.
- **Cause:** the two archives differ only in their internal folder name (`best/` vs `step_00200000/`, 428 bytes). The tensors are bit-identical, and the sidecars agree on every field except file size and hash.
- **Fix:** the byte-equality criterion was replaced by four checks:
  1. `best.pt` is intact (its SHA-256 equals `best.json`);
  2. `best.pt` holds bit-identical tensors (names, dtypes, shapes, values) to the step file;
  3. the sidecars agree on every field except file size and hash;
  4. the pick changed.

  All four pass for every run. The old arm loads the step file, so both arms use the same file format.
- **Documentation and test:** the plan states this rule, and a test covers it (`test_same_weights_ignores_archive_name_but_not_values`).

### 24.2 Validity, pairing and reproducibility

| Check | Result |
|---|---|
| Stored result files using a Block C seed (pre-run scan) | 0 of 546 |
| Block C instances unique per level / overlapping with A, B, S | 120 / 0 at each level |
| Block D | not touched |
| Old pick = Issue #6 `best.pt` (tensors, metadata), new pick = `selections.json` (SCB-C code hashes current) | pass, all four runs |
| Checkpoint SHA-256 before = after the run; equal to the block-S run's hashes | yes (8 of 8) |
| Records / expected | 2 880 / 2 880, all `status = ok`; gate: 0 problems, all four runs evaluable |
| Old/new pairs verified | 1 440 / 1 440: identical instance hash, identical paired fields (condition, run, level, seed, robots, `max_steps`), identical released-task count |
| Replay of every 60th task in fresh processes | 48 / 48 identical in every non-timing field |
| Analysis rerun | byte-identical `analysis.json` (`e8702b50…eb0b4`) |
| Historical files (`results/` except the new folder, `docs/`) | 1 387 files hashed before and after: unchanged; `SESSION_HANDOFF.md` hash `ba4a439c…` unchanged; `models/` absent |
| Wall time | 942 s on 10 workers |

Output folder: `results/dev6a/block_c/`. It contains:
- `check.json`, `plan_freeze.json`, `records.jsonl` (`bcef35d3…ea705`);
- `manifest.json`, with code and checkpoint hashes and the environment (Python 3.14.3, NumPy 2.4.6, torch 2.15.0+cpu);
- `replay.json`, `analysis.json`, `run.log`.

Monte Carlo SE of every reported p is at most 0.005. No Holm decision is within Monte Carlo reach of 0.05: the closest adjusted p-values are 0.106 and 0.122.

### 24.3 Results

Δ = new − old. CIs are 95 % symmetric bootstrap-t. `d_z` = Δ / (SE·√n). Holm is applied within each family at family-wise α = 0.05. "p < 0.0001" is the smallest p the bootstrap can give, 1/(B+1) ≈ 0.0001. The resampling unit, the pairing and the handling of levels are set out in §24.7.3. Every effect refers to one specific old/new checkpoint pair (§24.7.3).

#### Primary family (8 hypotheses, Holm)

| Run (old→new) | Outcome | Old | New | Δ (new − old) | Relative | 95 % CI (bootstrap-t) | d_z | p | p (Holm) | Holm decision |
|---|---|---|---|---|---|---|---|---|---|---|
| mappo_seed1 (200k→300k) | completion | 0.9870 | 0.9892 | +0.0022 | +0.2 % | [-0.0032, +0.0075] | +0.07 | 0.4373 | 1.0000 | not rejected |
| mappo_seed1 (200k→300k) | conflicts / episode | 65.92 | 13.98 | -51.94 | -78.8 % | [-68.64, -35.24] | -0.57 | <0.0001 | 0.0008 | **reject** (new lower) |
| mappo_seed2 (300k→500k) | completion | 0.9879 | 0.9877 | -0.0001 | -0.0 % | [-0.0050, +0.0048] | -0.01 | 0.9550 | 1.0000 | not rejected |
| mappo_seed2 (300k→500k) | conflicts / episode | 14.67 | 4.00 | -10.67 | -72.7 % | [-19.36, -1.97] | -0.24 | 0.0244 | 0.1220 | not rejected |
| mappo_seed3 (150k→500k) | completion | 0.9877 | 0.9871 | -0.0006 | -0.1 % | [-0.0059, +0.0047] | -0.02 | 0.8336 | 1.0000 | not rejected |
| mappo_seed3 (150k→500k) | conflicts / episode | 43.14 | 2.84 | -40.30 | -93.4 % | [-54.03, -26.56] | -0.56 | <0.0001 | 0.0008 | **reject** (new lower) |
| ippo_seed1 (550k→500k) | completion | 0.9821 | 0.9873 | +0.0052 | +0.5 % | [+0.0010, +0.0094] | +0.22 | 0.0176 | 0.1056 | not rejected |
| ippo_seed1 (550k→500k) | conflicts / episode | 3.38 | 4.25 | +0.88 | +26.0 % | [-2.19, +3.95] | +0.05 | 0.6214 | 1.0000 | not rejected |

#### Secondary family (12 hypotheses, Holm, separately)

| Run (old→new) | Outcome | Old | New | Δ (new − old) | Relative | 95 % CI (bootstrap-t) | d_z | p | p (Holm) | Holm decision |
|---|---|---|---|---|---|---|---|---|---|---|
| mappo_seed1 (200k→300k) | conflicts / delivered task | 3.477 | 0.736 | -2.742 | -78.8 % | [-3.632, -1.851] | -0.56 | <0.0001 | 0.0012 | **reject** (new lower) |
| mappo_seed1 (200k→300k) | conflicts / 100 robot-steps | 4.477 | 0.992 | -3.485 | -77.8 % | [-4.433, -2.538] | -0.67 | <0.0001 | 0.0012 | **reject** (new lower) |
| mappo_seed1 (200k→300k) | base return / episode | 1.363 | 2.838 | +1.475 | +108.3 % | [+0.995, +1.955] | +0.55 | <0.0001 | 0.0012 | **reject** (new higher) |
| mappo_seed2 (300k→500k) | conflicts / delivered task | 0.773 | 0.211 | -0.562 | -72.7 % | [-1.021, -0.103] | -0.24 | 0.0247 | 0.1235 | not rejected |
| mappo_seed2 (300k→500k) | conflicts / 100 robot-steps | 0.979 | 0.277 | -0.702 | -71.7 % | [-1.268, -0.136] | -0.24 | 0.0201 | 0.1206 | not rejected |
| mappo_seed2 (300k→500k) | base return / episode | 2.687 | 3.360 | +0.673 | +25.0 % | [+0.112, +1.234] | +0.28 | 0.0290 | 0.1235 | not rejected |
| mappo_seed3 (150k→500k) | conflicts / delivered task | 2.274 | 0.150 | -2.124 | -93.4 % | [-2.859, -1.389] | -0.55 | 0.0002 | 0.0014 | **reject** (new lower) |
| mappo_seed3 (150k→500k) | conflicts / 100 robot-steps | 2.911 | 0.192 | -2.718 | -93.4 % | [-3.522, -1.915] | -0.64 | <0.0001 | 0.0012 | **reject** (new lower) |
| mappo_seed3 (150k→500k) | base return / episode | 2.083 | 3.335 | +1.251 | +60.1 % | [+0.890, +1.613] | +0.63 | <0.0001 | 0.0012 | **reject** (new higher) |
| ippo_seed1 (550k→500k) | conflicts / delivered task | 0.179 | 0.224 | +0.045 | +25.3 % | [-0.118, +0.208] | +0.05 | 0.6291 | 1.0000 | not rejected |
| ippo_seed1 (550k→500k) | conflicts / 100 robot-steps | 0.200 | 0.269 | +0.069 | +34.5 % | [-0.115, +0.253] | +0.07 | 0.5108 | 1.0000 | not rejected |
| ippo_seed1 (550k→500k) | base return / episode | 3.325 | 3.189 | -0.136 | -4.1 % | [-0.339, +0.067] | -0.12 | 0.1988 | 0.5963 | not rejected |

#### Mean of the four run effects (descriptive, same seeds and resamples)

| Outcome | Mean Δ over the 4 runs | 95 % CI (bootstrap-t) | percentile CI |
|---|---|---|---|
| completion | +0.0017 | [-0.0006, +0.0039] | [-0.0006, +0.0039] |
| conflicts / episode | -25.51 | [-31.33, -19.69] | [-31.14, -20.08] |
| conflicts / delivered task | -1.346 | [-1.658, -1.033] | [-1.649, -1.055] |
| conflicts / 100 robot-steps | -1.709 | [-2.052, -1.366] | [-2.032, -1.376] |
| base return / episode | +0.816 | [+0.621, +1.011] | [+0.631, +1.008] |

#### Per level (descriptive; unadjusted 95 % bootstrap-t CIs, no tests)

| Run | Level | completion: old → new, Δ [95 % CI] | conflicts / episode: old → new, Δ [95 % CI] | conflicts / delivered task: old → new, Δ [95 % CI] | conflicts / 100 robot-steps: old → new, Δ [95 % CI] | base return / episode: old → new, Δ [95 % CI] |
|---|---|---|---|---|---|---|
| mappo_seed1 | static | 0.9833 → 0.9781, -0.0052 [-0.0171, +0.0067] | 27.94 → 16.38, -11.57 [-29.69, +6.56] | 1.776 → 1.046, -0.730 [-1.890, +0.431] | 2.386 → 1.318, -1.068 [-2.450, +0.315] | 2.015 → 2.513, +0.498 [-0.292, +1.287] |
| mappo_seed1 | obstacles | 0.9911 → 0.9948, +0.0036 [-0.0023, +0.0096] | 58.23 → 5.75, -52.48 [-73.90, -31.06] | 3.672 → 0.361, -3.311 [-4.681, -1.941] | 5.100 → 0.571, -4.529 [-5.788, -3.270] | 1.246 → 2.899, +1.653 [+1.120, +2.186] |
| mappo_seed1 | full | 0.9867 → 0.9925, +0.0059 [-0.0003, +0.0120] | 111.58 → 19.82, -91.76 [-122.88, -60.63] | 4.414 → 0.779, -3.635 [-4.889, -2.381] | 5.304 → 1.001, -4.302 [-5.610, -2.995] | 0.827 → 3.103, +2.275 [+1.561, +2.990] |
| mappo_seed2 | static | 0.9729 → 0.9740, +0.0010 [-0.0110, +0.0131] | 26.57 → 6.64, -19.93 [-38.62, -1.23] | 1.707 → 0.426, -1.280 [-2.489, -0.071] | 1.864 → 0.520, -1.344 [-2.643, -0.046] | 2.085 → 3.086, +1.001 [+0.134, +1.868] |
| mappo_seed2 | obstacles | 0.9953 → 0.9943, -0.0010 [-0.0062, +0.0041] | 7.53 → 3.00, -4.53 [-14.30, +5.23] | 0.473 → 0.189, -0.284 [-0.899, +0.330] | 0.719 → 0.285, -0.434 [-1.324, +0.455] | 2.661 → 3.175, +0.514 [-0.151, +1.178] |
| mappo_seed2 | full | 0.9925 → 0.9922, -0.0003 [-0.0050, +0.0044] | 9.92 → 2.37, -7.55 [-21.81, +6.71] | 0.390 → 0.093, -0.297 [-0.861, +0.267] | 0.490 → 0.118, -0.372 [-1.061, +0.317] | 3.314 → 3.818, +0.504 [+0.030, +0.977] |
| mappo_seed3 | static | 0.9760 → 0.9807, +0.0047 [-0.0070, +0.0164] | 12.90 → 3.26, -9.64 [-23.19, +3.91] | 0.826 → 0.208, -0.618 [-1.488, +0.252] | 0.967 → 0.268, -0.699 [-1.674, +0.277] | 2.690 → 3.220, +0.531 [-0.001, +1.062] |
| mappo_seed3 | obstacles | 0.9911 → 0.9911, +0.0000 [-0.0077, +0.0077] | 48.52 → 2.36, -46.16 [-69.40, -22.92] | 3.059 → 0.149, -2.911 [-4.388, -1.433] | 4.311 → 0.206, -4.104 [-5.676, -2.532] | 1.484 → 3.056, +1.573 [+0.951, +2.194] |
| mappo_seed3 | full | 0.9928 → 0.9886, -0.0042 [-0.0104, +0.0020] | 68.01 → 2.92, -65.09 [-85.26, -44.93] | 2.674 → 0.115, -2.559 [-3.372, -1.746] | 3.423 → 0.140, -3.283 [-4.185, -2.380] | 2.077 → 3.728, +1.651 [+1.236, +2.066] |
| ippo_seed1 | static | 0.9641 → 0.9750, +0.0109 [-0.0006, +0.0225] | 3.97 → 1.77, -2.20 [-21.18, +16.78] | 0.257 → 0.113, -0.144 [-1.389, +1.101] | 0.235 → 0.118, -0.116 [-1.106, +0.874] | 3.082 → 3.093, +0.011 [-0.353, +0.375] |
| ippo_seed1 | obstacles | 0.9922 → 0.9938, +0.0016 [-0.0038, +0.0070] | 2.04 → 2.89, +0.85 [-1.29, +2.99] | 0.129 → 0.182, +0.053 [-0.081, +0.187] | 0.172 → 0.254, +0.082 [-0.095, +0.258] | 3.129 → 3.023, -0.106 [-0.381, +0.169] |
| ippo_seed1 | full | 0.9870 → 0.9909, +0.0039 [-0.0010, +0.0088] | 4.12 → 8.10, +3.98 [-5.52, +13.49] | 0.163 → 0.319, +0.156 [-0.217, +0.529] | 0.188 → 0.383, +0.195 [-0.260, +0.651] | 3.763 → 3.450, -0.313 [-0.661, +0.035] |

#### Safety and validity (all levels; per-level rows in `analysis.json`)

Episode classes are defined in §24.7.1:
- **failed** is a runtime exception;
- **successful** is terminated with every task delivered;
- **incomplete** is reaching the horizon with tasks undelivered.

"Ran to the end" (`completed` in `analysis.json`) only means that no exception occurred. It does **not** mean that the tasks were delivered.

| Run | Arm | Checkpoint | Episodes | Ran to the end (no exception) | Failed (runtime exception) | Invalid records | Physical collisions | Invalid moves (INV) | **Successful**: terminated, all delivered | **Incomplete**: horizon, undelivered | Horizon, all delivered | Undelivered tasks | Mean flow time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ippo_seed1 | old | step_00550000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 272 | 88 | 0 | 124 | 50.80 |
| ippo_seed1 | new | step_00500000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 290 | 70 | 0 | 88 | 50.66 |
| mappo_seed1 | old | step_00200000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 299 | 61 | 0 | 90 | 49.69 |
| mappo_seed1 | new | step_00300000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 316 | 44 | 0 | 75 | 49.83 |
| mappo_seed2 | old | step_00300000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 304 | 56 | 0 | 84 | 50.03 |
| mappo_seed2 | new | step_00500000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 310 | 50 | 0 | 85 | 50.65 |
| mappo_seed3 | old | step_00150000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 306 | 54 | 0 | 85 | 50.64 |
| mappo_seed3 | new | step_00500000.pt | 360 | 360 | 0 | 0 | 0 | 0 | 305 | 55 | 0 | 89 | 51.33 |

#### Episodes reaching the horizon with undelivered tasks, per level (undelivered tasks in brackets)

| Run | Arm | static | obstacles | full |
|---|---|---|---|---|
| ippo_seed1 | old | 44 (69 tasks) | 12 (15 tasks) | 32 (40 tasks) |
| ippo_seed1 | new | 35 (48 tasks) | 12 (12 tasks) | 23 (28 tasks) |
| mappo_seed1 | old | 19 (32 tasks) | 15 (17 tasks) | 27 (41 tasks) |
| mappo_seed1 | new | 22 (42 tasks) | 7 (10 tasks) | 15 (23 tasks) |
| mappo_seed2 | old | 31 (52 tasks) | 8 (9 tasks) | 17 (23 tasks) |
| mappo_seed2 | new | 24 (50 tasks) | 8 (11 tasks) | 18 (24 tasks) |
| mappo_seed3 | old | 26 (46 tasks) | 12 (17 tasks) | 16 (22 tasks) |
| mappo_seed3 | new | 21 (37 tasks) | 12 (17 tasks) | 22 (35 tasks) |

#### Conflict types (all levels; RR = robot–robot, RO = robot–obstacle, v = vertex, s = swap; INV = invalid move)

These are *proposed* conflicts: the arbiter detects them and cancels the moves involved, so none is a collision. The per-level breakdown for mappo_seed1 and mappo_seed3 is in §24.7.4.

| Run | Arm | RRv | RRs | ROv | ROs | INV |
|---|---|---|---|---|---|---|
| ippo_seed1 | old | 855 | 18 | 342 | 0 | 0 |
| ippo_seed1 | new | 1082 | 14 | 435 | 0 | 0 |
| mappo_seed1 | old | 6365 | 375 | 16990 | 0 | 0 |
| mappo_seed1 | new | 3512 | 353 | 1168 | 0 | 0 |
| mappo_seed2 | old | 4663 | 69 | 550 | 0 | 0 |
| mappo_seed2 | new | 1127 | 2 | 312 | 0 | 0 |
| mappo_seed3 | old | 3840 | 166 | 11525 | 0 | 0 |
| mappo_seed3 | new | 670 | 19 | 335 | 0 | 0 |

**Post-hoc descriptives (not pre-specified; for interpretation only).**

| Run | Episode conflicts, median old → new | Share of episodes with 0 conflicts, old → new | Share of all conflicts in the top 5 % of episodes, old → new | Seeds with fewer / more / equal conflicts under new |
|---|---|---|---|---|
| mappo_seed1 | 10 → 1 | 0.24 → 0.36 | 0.37 → 0.81 | 108 / 11 / 1 |
| mappo_seed2 | 2 → 1 | 0.30 → 0.50 | 0.79 → 0.77 | 87 / 22 / 11 |
| mappo_seed3 | 6 → 0 | 0.25 → 0.55 | 0.47 → 0.69 | 113 / 7 / 0 |
| ippo_seed1 | 1 → 1 | 0.46 → 0.43 | 0.65 → 0.64 | 47 / 59 / 14 |

### 24.4 Interpretation (pre-specified table of the plan, §10)

| Run (old → new) | Conflicts / episode (primary, Holm) | Completion (primary, Holm) | Reading |
|---|---|---|---|
| mappo_seed1 (200k → 300k) | **lower**: −51.9 (−79 %), CI [−68.6, −35.2] | no detectable difference: +0.0022, CI [−0.0032, +0.0075] | intended trade-off; **no completion cost detected** |
| mappo_seed2 (300k → 500k) | no detectable difference after Holm: −10.7 (−73 %), CI [−19.4, −2.0], raw p 0.024, Holm 0.122 | no detectable difference: −0.0001, CI [−0.0050, +0.0048] | no conflict benefit established; the direction and the unadjusted CI favour the new pick |
| mappo_seed3 (150k → 500k) | **lower**: −40.3 (−93 %), CI [−54.0, −26.6] | no detectable difference: −0.0006, CI [−0.0059, +0.0047] | intended trade-off; **no completion cost detected** |
| ippo_seed1 (550k → 500k) | no detectable difference: +0.88 (+26 %), CI [−2.2, +4.0] | no detectable difference after Holm: +0.0052, CI [+0.0010, +0.0094], raw p 0.018, Holm 0.106 | no conflict benefit; adjacent checkpoints with low conflicts in both arms |

**What the results show.**

- **Safety and episode outcomes** (definitions in §24.7.1).
  - No physical collision, invalid move (INV) or illegal task transition occurred in any of the eight checkpoints.
  - No episode **failed** (raised a runtime exception) and none was rejected by the gate.
  - Of the 2 880 episodes, 2 402 were **successful** (every released task delivered, so the episode terminated before the horizon).
  - 478 were **incomplete**: they ran to the horizon with 1 or more tasks undelivered, 44–88 per arm.
  - No episode with an undelivered task was counted as successful. Incomplete episodes enter the completion outcome with their undelivered tasks.
- **Conflicts.** Two of the four primary conflict hypotheses are rejected: mappo_seed1 and mappo_seed3. In both runs:
  - the old pick was an early checkpoint (200 k and 150 k);
  - most of its proposed conflicts were robot–obstacle conflicts (ROv): 16 990 of 23 730 and 11 525 of 15 531;
  - the new pick removes almost all of them.

  Robot–obstacle conflicts account for 85 % and 77 % of the reduction; robot–robot conflicts fall too (§24.7.4).
  - The secondary family agrees for those two runs, all six of its hypotheses rejected. This is **supportive evidence**, not an independent confirmation (§24.7.2):
    - conflicts per delivered task and per 100 robot-steps fall by 78–93 %, but they re-express the same conflict counts over other denominators;
    - base return rises, but it includes the conflict penalties.
  - For mappo_seed2 the reduction is large in relative terms and consistent across seeds (87 of 120 lower), but it is not significant after Holm.
  - For ippo_seed1 the two checkpoints are neighbours (550 k and 500 k). Conflicts are low in both, and the difference is undetectable.
- **Completion.** No completion hypothesis is rejected. All four 95 % CIs lie within about ±0.0095 of zero. Three of them include 0 and are about ±0.005 wide. For ippo_seed1 the unadjusted CI is entirely above 0 (+0.0010 to +0.0094), but the hypothesis is not rejected after Holm.
  - Averaged over the four checkpoint pairs (descriptive), completion changes by +0.0017, CI [−0.0006, +0.0039]. Conflicts per episode change by −25.5, CI [−31.3, −19.7]. These CIs reflect only scenario-seed variation for these four fixed pairs. They do not estimate a general SCB-C effect (§24.7.3).
  - The selection study had predicted a completion cost of about 0.0017 for the one-SE rule relative to completion-first selection (§22, §23.6). That cost did not appear here; the CI excludes a mean cost larger than 0.0006.
  - One reading consistent with this (**[Interpretation]**): the Issue #6 picks were chosen on 20 block-A seeds by completion and then return. Their apparent completion edge did not carry over to new seeds.
- **Non-rejection is not equivalence.** For completion, the CIs above bound the differences compatible with these data. No equivalence margin was set, and none is claimed.
- **Per level** (descriptive, unadjusted).
  - The conflict reduction for mappo_seed1 and mappo_seed3 comes mainly from the `obstacles` and `full` levels.
  - On `static` the per-level CIs include 0 for these two runs and for ippo_seed1. For mappo_seed2, `static` is the only level whose CI excludes 0, and its point estimate is the largest of its three levels.
  - Every per-level completion CI includes 0. Two only barely: mappo_seed1 on `full`, [−0.0003, +0.0120], and ippo_seed1 on `static`, [−0.0006, +0.0225].

### 24.5 Limitations

1. **Development evidence on reusable data.** Block C will be reused. These results must not be presented as the final validation, and any later decision taken *because of* them makes Block C part of that decision's data.
2. **Four specific checkpoint pairs.** The result describes these picks. It is not the expected effect of SCB-C v5 over new training runs. The four pairs are not independent replications of a treatment, and the across-run mean is descriptive only (§24.7.3).
3. **Families.** Both families were defined in the frozen plan before any Block C episode (§24.7.2).
   - The primary family controls the family-wise error at 0.05.
   - The secondary family is controlled separately at 0.05.
   - Across both families, the union bound gives a family-wise error of at most **0.10**.
   - Per-level results, the across-run mean, the post-hoc descriptives and the conflict-category breakdown are unadjusted and exploratory.
4. **Heavy tails.** Conflicts are concentrated in a minority of episodes: for mappo_seed1 (new), the top 5 % of episodes carry 81 % of the conflicts. The bootstrap-t was chosen for this situation, and the percentile intervals agree closely. Even so, conflict-per-episode means depend strongly on rare bad episodes.
5. **Greedy mode, one instance per seed and level, A\* only.** Stochastic-policy behaviour, RRT\*, MADQL and the unchanged runs were outside the approved scope.
6. **Completion is near its ceiling** (about 0.98–0.99). Differences of a few thousandths are at the resolution of 120 seeds × 3 levels.
7. **Incomplete episodes remain common in both arms.** Between 44 and 88 of each arm's 360 episodes reach the horizon with tasks undelivered, 75–124 tasks in total. This is a property of the PPO controllers, not of the selection rule. It is unchanged from Issue #6, §20.

### 24.6 Recommendation (development evidence; for the student's review)

1. **Keep SCB-C v5 frozen and unchanged.** Nothing on Block C argues for changing it, and changing it on these data would invalidate Block C as independent development evidence.
2. **The v5 picks pass every safety and validity check on Block C.** On these four pairs they show a large, statistically supported conflict reduction in two runs. They show no detectable completion cost in any run, and a descriptively positive mean completion change. There is no safety or validity reason against carrying the v5 picks forward as the PPO checkpoints for the next phase.
3. **The claim to make is limited:** "for these four checkpoint pairs, lower conflicts per episode in two (mappo_seed1, mappo_seed3), no detectable completion difference in any, on development data." A claim of general superiority of the selection rule is not supported by this evidence. A final, confirmatory statement requires the Block D evaluation, which needs separate approval and its own pre-registered plan.
4. **Out of scope and not started:** L2, the P items, MADQL/RRT\* extension of SCB-C, PPO retraining, production policy changes and Block D.

**Files of this phase.**
- **Code:**
  - `tools/selection/block_c.py`: runner with check, freeze, run, replay and analyse.
  - `tools/selection/block_c_analysis.py`: the frozen analysis.
- **Tests:** `tests/tools/test_block_c.py`, 54 tests. They cover:
  - the guard (Block C only; D, test, A, B and S rejected);
  - the worker refusing Block D before generating a scenario;
  - the pairing of the 2 880 tasks;
  - Holm;
  - the delta-method SE;
  - the degenerate cases of the bootstrap;
  - calibration under a skewed paired null;
  - 12 not-evaluable and 11 abort cases of the gate;
  - the not-evaluable path;
  - the safety counts;
  - the weight-equality check.
- **Docs:** the plan (`docs/research/2026-11-21-block-c-analysis-plan.md`, frozen) and this section.
- **Results** (gitignored): `results/dev6a/block_c/`.

### 24.7 Clarifications after review (2026-11-21; documentation only)

The student asked for these clarifications after reviewing §24. No experiment was run for them, and no evaluation episode. The frozen plan, runner and analysis code were not modified. The stored outputs in `results/dev6a/block_c/` are unchanged: `analysis.json` is still `e8702b50…eb0b4` and `records.jsonl` is still `bcef35d3…ea705`.

The figures below are read from the stored records and from the stored p-values. Wherever a figure was not part of the frozen analysis, it is marked **post-hoc**.

#### 24.7.1 Failed, incomplete and successful episodes

Each episode falls into exactly one class. The classes are defined from the stored record fields.

| Class | Definition | Count (of 2 880) |
|---|---|---|
| **Failed** | The episode raised a runtime exception (`status = "error"`), so no metrics exist. This includes `PhysicalCollisionError`, which the engine raises on any collision, and `InvalidTransitionError`, which the task state machine raises on an illegal transition. | **0** |
| **Rejected by the gate** | A record without an exception that fails the validity checks of plan §7. Examples: delivered > released; terminated with an undelivered task; a conflict breakdown that does not sum to its total. | **0** |
| **Successful** | Ran without exception, `terminated = true`, and every released task delivered (`delivered = released`). The engine terminates only when every task has been delivered and no further arrival can occur, so termination itself means full delivery. | **2 402** |
| **Incomplete** | Ran without exception to the horizon (`steps = max_steps`), not terminated, with 1 or more released tasks undelivered. | **478** (44–88 per arm) |
| Reached the horizon with all tasks delivered | Ran to the horizon with every released task delivered. | 0 |

**Checks on the stored records** (post-hoc tabulation):
- **Undelivered tasks:** no episode with an undelivered task is terminated. The gate also rejects that combination.
- **Every non-terminated episode reached the horizon:** none stopped early without an exception.
- **Released tasks:** every episode released at least one task.

**No episode with undelivered tasks was counted as successful.** Incomplete episodes are not removed from any outcome. Completion is `Σ delivered / Σ released`, so each undelivered task lowers it. Conflicts and return include incomplete episodes in full.

**"Zero failed episodes" vs "478 incomplete episodes".** These are different classes:
- **failed** concerns software or safety failure (an exception);
- **incomplete** concerns task performance (work left at the horizon).

The column formerly headed "Completed" in the safety table of §24.3 is now "Ran to the end (no exception)". Its JSON key `completed` is unchanged in the frozen output.

#### 24.7.2 The two correction families

**Defined before the results.** Both families, their members and their Holm corrections are written in the frozen plan (§6 of `docs/research/2026-11-21-block-c-analysis-plan.md`). The sequence was:
1. The plan's SHA-256 was recorded at 09:50:44. `freeze` refuses to run once a records file exists, and none existed then.
2. The run started at 09:51:06 and finished at 10:06:49 (`manifest.json`).
3. The plan hash matched at the run and at the analysis, and still matches.

No family, member or ordering was changed after the records existed.

**Why two families.**
- **Primary family: completion and conflicts per episode, per run.** These are the two quantities SCB-C v5 explicitly trades. Completion is its statistical criterion; conflicts per episode is its secondary selection criterion (§23.3). They answer the decision question.
- **Secondary family: conflicts per delivered task, conflicts per 100 robot-steps, base return.**
  - The two conflict rates are the same conflict counts over different denominators. They were included to check that the conflict result is not an artefact of the denominator: for example, a policy that delivers less or finishes sooner has fewer delivered tasks or robot-steps.
  - Base return contains the conflict penalties, so it is not independent of the conflicts either.
  - Taken together, they bear on whether the primary result holds up, not on a separate claim.
- **Why not one family.** Pooling all 20 hypotheses would have reduced the power of the primary family to protect hypotheses that make no new claim. Keeping them apart limits the primary correction to 8 hypotheses.

**Error rates.**
- The primary family controls the family-wise error rate (FWER) at 0.05.
- The secondary family controls it at 0.05 separately.
- Across both families, the probability of at least one false rejection among the 20 hypotheses is at most **0.05 + 0.05 = 0.10**, by the union (Bonferroni) bound, whatever the dependence between them.
- 0.10 is the guaranteed bound. Strong positive dependence between the families would make the true rate lower, but that is not guaranteed and is not claimed.

**Post-hoc sensitivity** (not pre-specified; computed from the stored p-values; no new data). A single Holm correction over all 20 hypotheses at α = 0.05 rejects exactly the same 8 hypotheses as the two pre-specified families, with adjusted p ≤ 0.0026:
- mappo_seed1: all 4 conflict and return hypotheses;
- mappo_seed3: all 4 conflict and return hypotheses.

The closest non-rejected hypotheses would be at 0.21–0.24. The decisions therefore do not depend on the choice between two families and one. The pre-specified two-family analysis remains the reported one.

**Status of each kind of result.**

| Result | Correction | Status |
|---|---|---|
| Primary family (§24.3) | Holm, FWER 0.05 | the strongest statements of this phase, still development evidence: Block C is not confirmatory data (§24.1) |
| Secondary family | Holm, FWER 0.05 separately (0.10 combined) | supportive only |
| Per level, across-run mean, post-hoc descriptives, conflict-category breakdown (§24.7.4) | none | descriptive and exploratory; no inferential claim is made from them |

No secondary, per-level or descriptive result is used for a confirmatory claim in §24. Where §24.4 previously described the secondary agreement without qualification, it now says "supportive".

#### 24.7.3 Pairing, resampling unit, levels and the four checkpoint pairs

**Resampling unit.** The scenario seed is the only resampling unit. One matrix of seed indices, `(B = 10 000) × 120` from `default_rng(0)`, is drawn once. In every resample the same seed indices select:
- the old **and** the new observations of a run (`a[..., idx]` is applied to all four arrays `Y_old, X_old, Y_new, X_new` together);
- all four runs, for the across-run mean.

A resampled seed therefore always brings its old and new episodes together. The pairing is never broken, and the paired SE is recomputed in every resample from the per-seed paired influence differences `d_s = u_new,s − u_old,s`.

**Levels within a seed.**
- Each seed has exactly one episode per level per arm.
- For the overall analyses, each seed's numerator and denominator are summed over its three levels before resampling. A seed enters as one unit, carrying its `static`, `obstacles` and `full` episodes for both arms.
- Levels are fixed strata of the design: they are not resampled. Every resample keeps the one-episode-per-level balance, and any correlation between a seed's levels is preserved.
- Because every outcome is a ratio of sums, completion weights the levels by their released tasks. The per-episode outcomes weight each level equally, with one episode each.
- The per-level results use the same index matrix on one level at a time.

The independence assumption is between seeds only. Each seed generates its own instance; the pre-run check found the 120 instances of each level distinct, and none equal to an A, B or S instance of that level.

**The four checkpoint pairs are not replications.**
- Each per-run hypothesis concerns one fixed old/new pair, and its uncertainty is scenario-seed sampling for that pair alone.
- No test pools the runs or treats them as independent draws of a treatment.
- The across-run mean is the plain average of four pair-specific effects. Its CI reflects only scenario-seed variation, with the same seeds shared across runs. It does not reflect variation over training runs, over which checkpoint the old rule happened to pick, or over runs where the rule does not change the pick.
- The four runs are not a random sample: they are exactly the runs where SCB-C v5 picked a different checkpoint.
- The two unchanged runs (ippo_seed2, ippo_seed3) have Δ = 0 by construction and are not in the average.
- A general SCB-C treatment effect would need training runs as the replication unit, with new runs and the unchanged ones included. No such estimate is made.

**Conclusions that apply only to these four checkpoint pairs**, on Block C, greedy A\*, the three levels:
- every per-run estimate, CI and Holm decision in §24.3–24.4;
- the across-run means;
- the safety and episode-class counts;
- the conflict-category breakdown;
- the reading "conflicts per episode lower for mappo_seed1 (200 k → 300 k) and mappo_seed3 (150 k → 500 k); no completion difference detected for any of the four pairs".

None of these establishes a general property of the SCB-C v5 rule.

#### 24.7.4 Conflict categories for mappo_seed1 and mappo_seed3 (post-hoc, descriptive)

The figures are totals over the 120 seeds, with conflicts per episode in brackets.
- **RR (robot–robot)** = vertex (RRv) + swap (RRs).
- **RO (robot–obstacle)** = vertex (ROv) + swap (ROs).
- **Other** = invalid moves (INV).
- **Share of reduction** = (old − new) of the category / (old − new) of all conflicts. It is shown only where the total fell.

All counts are proposed conflicts that the arbiter cancelled, not collisions. ROs and INV were 0 in every episode of both arms, so RO equals ROv and Other is 0.

| Run | Level (episodes) | RR old → new | RO old → new | Other | Total old → new | RO share of reduction | RR share of reduction |
|---|---|---|---|---|---|---|---|
| mappo_seed1 (200k → 300k) | static (120) | 3 353 → 1 965 (27.94 → 16.38) | 0 → 0 | 0 → 0 | 3 353 → 1 965 (27.94 → 16.38) | 0 % | 100 % |
| | obstacles (120) | 561 → 440 (4.67 → 3.67) | 6 427 → 250 (53.56 → 2.08) | 0 → 0 | 6 988 → 690 (58.23 → 5.75) | 98 % | 2 % |
| | full (120) | 2 826 → 1 460 (23.55 → 12.17) | 10 563 → 918 (88.03 → 7.65) | 0 → 0 | 13 389 → 2 378 (111.58 → 19.82) | 88 % | 12 % |
| | **all (360)** | **6 740 → 3 865 (18.72 → 10.74)** | **16 990 → 1 168 (47.19 → 3.24)** | 0 → 0 | **23 730 → 5 033 (65.92 → 13.98)** | **85 %** | **15 %** |
| mappo_seed3 (150k → 500k) | static (120) | 1 548 → 391 (12.90 → 3.26) | 0 → 0 | 0 → 0 | 1 548 → 391 (12.90 → 3.26) | 0 % | 100 % |
| | obstacles (120) | 1 639 → 149 (13.66 → 1.24) | 4 183 → 134 (34.86 → 1.12) | 0 → 0 | 5 822 → 283 (48.52 → 2.36) | 73 % | 27 % |
| | full (120) | 819 → 149 (6.83 → 1.24) | 7 342 → 201 (61.18 → 1.68) | 0 → 0 | 8 161 → 350 (68.01 → 2.92) | 91 % | 9 % |
| | **all (360)** | **4 006 → 689 (11.13 → 1.91)** | **11 525 → 335 (32.01 → 0.93)** | 0 → 0 | **15 531 → 1 024 (43.14 → 2.84)** | **77 %** | **23 %** |

The RR split into vertex and swap conflicts (all levels):
- mappo_seed1: RRv 6 365 → 3 512, RRs 375 → 353;
- mappo_seed3: RRv 3 840 → 670, RRs 166 → 19.

**Reading.**
- **Robot–obstacle conflicts drive most of the reduction.** In both runs the aggregate reduction in conflicts per episode comes predominantly from robot–obstacle conflicts: 85 % for mappo_seed1 and 77 % for mappo_seed3.
  - RO conflicts per episode fall by 93 % and 97 %.
  - RO conflicts exist only on the `obstacles` and `full` levels; they are 0 on `static` in both arms. On those two levels RO accounts for 73–98 % of the reduction.
- **Robot–robot conflicts fall too, but are not the main driver.**
  - They fall by 43 % for mappo_seed1 and 83 % for mappo_seed3.
  - They are the whole reduction on `static`, and 15 % and 23 % of the total.
  - The total reduction is therefore not purely an obstacle effect. The newer checkpoints also propose fewer robot–robot conflicts.
- **The RO reduction is broad, not a few outlier seeds.** Of each run's 120 seeds:
  - the RO count over a seed's three levels is lower under the new pick in 114 (mappo_seed1) and 119 (mappo_seed3), and higher in 5 and 1;
  - the 10 seeds with the largest RO reductions account for 45 % and 52 % of the total RO reduction.
- **No uncertainty is attached.** These category figures carry no CI or test. They are a post-hoc decomposition of the primary conflict outcome, and they make no inferential claim of their own.

#### 24.7.5 Documentation changes made in this clarification

1. §24.3 introduction:
   - defines the floor of the bootstrap p ("p < 0.0001" = 1/(B+1));
   - points to §24.7.3 for pairing and levels;
   - states that effects are per checkpoint pair.
2. §24.3 safety table:
   - a definitions note above the table;
   - "Completed" renamed "Ran to the end (no exception)";
   - "Failed (type)" renamed "Failed (runtime exception)";
   - "Terminated (all delivered)" and "Horizon, undelivered" labelled **Successful** and **Incomplete**.

   The numbers are unchanged.
3. §24.3 conflict-type table: a note that these are proposed conflicts cancelled by the arbiter, not collisions.
4. §24.4 "What the results show":
   - the safety bullet now separates failed, successful and incomplete episodes, with counts;
   - the conflict bullet gives the RO and RR shares of the reduction;
   - the secondary-family agreement is labelled supportive, with the reason;
   - the across-run mean is qualified as pair-specific and descriptive.
5. §24.5 limitations:
   - item 2 states that the four pairs are not independent replications;
   - item 3 states that both families were pre-specified and that the combined bound is 0.10, and lists every uncorrected result as exploratory.
6. §24.6 recommendation 3: the claim is restated per checkpoint pair, naming the two runs.
7. New §24.7 (this subsection).
8. Status header at the top of the document: a pointer to §24.7.

**Not changed:**
- the frozen plan, runner and analysis code (their hashes still match `plan_freeze.json`);
- all stored outputs in `results/dev6a/block_c/`;
- SCB-C v5;
- every historical Issue #5 and Issue #6 artifact;
- `docs/SESSION_HANDOFF.md`.


## 25. Block D final evaluation of H-L1 (frozen protocol; executed 2026-11-21)

### 25.1 Status, authorization and scope

Block D was run once, under the frozen protocol `docs/research/2026-11-21-block-d-protocol.md` (revision 2). The student authorized each stage separately:
- the protocol;
- the implementation;
- the freeze, recorded in `plan_freeze.json` at 13:51:20, SHA-256 `33ac0d92…69b31`;
- the pre-run check, recorded in `check.json`, `11316f89…f977a`;
- this run with its replay and analysis.

**What was run.** The frozen matrix of 8 220 episodes, and nothing else:

| Part | Content |
|---|---|
| D1 | four PPO pairs, Issue #6 pick vs SCB-C v5 pick, A\*, seeds 12 000–12 199 × 3 levels |
| D2 | ippo_seed2 and ippo_seed3, A\* |
| D3 | the D1 pairs on RRT\*, seeds 12 000–12 059 × 3 levels |
| D4 | the matched baseline, descriptive only |

The condition was greedy, auction allocator and shield off, with the existing configuration.

**What was not done.** No training, no checkpoint changes, no SCB-C or L2 changes, no other experiment. Nothing was moved into `models/`.

**What Block D supports.** It is the final held-out data for H-L1, and it supports per-pair statements only:
- F1 is primary;
- F2 is supportive;
- F3 (RRT\*) is secondary, with no RRT\* development evidence, 60 seeds, and limited to the four tested pairs;
- per-level results are exploratory;
- D2 and D4 are descriptive.

None of these results is a claim about the SCB-C method in general or across planners.

### 25.2 Execution, validity and reproducibility

| Check | Result |
|---|---|
| Freeze record and every frozen file, selection-artifact, checkpoint and environment hash before the run | match |
| Pre-run check | passed. 600 instances, unique per level, 0 overlap with A, B, C and S; 0 stored files with a Block D seed; checkpoint integrity verified |
| Records | 8 220 / 8 220, one invocation, 5 651 s on 10 workers (projected 1.5–2 h); 0 duplicates |
| Scenario instances | every record's instance hash equals its pre-run hash (0 mismatches); 600 distinct instances used |
| Pairing | 3 120 / 3 120 episode pairs (A\* 2 400, RRT\* 720) identical in every paired field; 1 040 seed units |
| Gate | all 12 run × planner units usable; 0 failed, 0 missing, 0 duplicate, 0 invalid episodes |
| Checkpoints before = after | yes (manifest) |
| Historical files (`results/` except `block_d`, and `docs/`) before = after | yes (manifest snapshot, and an independent hash listing) |
| Replay: every 60th task, 137 episodes, fresh processes | 137 / 137 identical in every non-timing field |
| Analysis reproducibility | values identical across reruns. **Not byte-identical across processes** (deviation 2, §25.7); byte-identical under a fixed `PYTHONHASHSEED` |
| `docs/SESSION_HANDOFF.md` | unchanged (`ba4a439c…`) |

**Output files** in `results/dev6a/block_d/`:
- `records.jsonl` (`b2c9d39b…`)
- `analysis.json` (`b9514e6b…`), the official first analysis run
- `manifest.json`
- `replay.json`
- `check.json`
- `plan_freeze.json`
- `history_before.json`
- the logs

### 25.3 Episodes and safety

**Episode classes.** "Successful" means terminated with every task delivered. "Incomplete" means the episode reached the 400-step horizon with tasks undelivered.

| Planner | Run / arm | Episodes | Successful | Incomplete | Undelivered tasks |
|---|---|---|---|---|---|
| A\* | mappo_seed1 old / new | 600 / 600 | 506 / 541 | 94 / 59 | 140 / 83 |
| A\* | mappo_seed2 old / new | 600 / 600 | 508 / 514 | 92 / 86 | 126 / 138 |
| A\* | mappo_seed3 old / new | 600 / 600 | 520 / 522 | 80 / 78 | 132 / 127 |
| A\* | ippo_seed1 old / new | 600 / 600 | 466 / 465 | 134 / 135 | 183 / 199 |
| A\* | ippo_seed2 (v5, D2) | 600 | 501 | 99 | 144 |
| A\* | ippo_seed3 (v5, D2) | 600 | 534 | 66 | 104 |
| A\* | matched baseline (D4) | 600 | 522 | 78 | 165 |
| RRT\* | mappo_seed1 old / new | 180 / 180 | 165 / 170 | 15 / 10 | 21 / 16 |
| RRT\* | mappo_seed2 old / new | 180 / 180 | 164 / 166 | 16 / 14 | 22 / 16 |
| RRT\* | mappo_seed3 old / new | 180 / 180 | 168 / 169 | 12 / 11 | 17 / 11 |
| RRT\* | ippo_seed1 old / new | 180 / 180 | 130 / 137 | 50 / 43 | 59 / 52 |
| RRT\* | matched baseline (D4) | 180 | 178 | 2 | 2 |
| | **Total** | **8 220** | **7 046** | **1 174** | |

- **Other classes:** failed 0, invalid 0, missing 0, duplicate 0, "horizon with everything delivered" 0, and terminated with an undelivered task 0.
- **Incomplete episodes** remain in every outcome.
- **Safety:** 0 physical collisions, 0 invalid moves (INV), 0 illegal task transitions and 0 exceptions, in every arm.

**Conflict types** (proposed conflicts, cancelled by the arbiter; A\*, all levels, totals over 600 episodes):

| Run | old RRv / RRs / ROv | new RRv / RRs / ROv |
|---|---|---|
| mappo_seed1 | 10 362 / 74 / 30 237 | 5 005 / 45 / 1 006 |
| mappo_seed2 | 6 717 / 180 / 1 771 | 1 220 / 6 / 589 |
| mappo_seed3 | 5 459 / 592 / 19 407 | 1 917 / 24 / 896 |
| ippo_seed1 | 608 / 12 / 592 | 4 699 / 158 / 689 |

ROs and INV were 0 everywhere. The matched baseline had RR 0 (by construction) and ROv 590 (A\*) and 189 (RRT\*).

### 25.4 Confirmatory results: old vs new

Definitions used in all three tables:
- Δ = new − old.
- "Nominal 95 % CI" is the per-hypothesis symmetric bootstrap-t interval, **unadjusted for multiplicity**.
- Decisions use the Holm-adjusted p at the family α only.
- "MC-sens." means p lies within 2 Monte Carlo SE of its own Holm threshold. The formal decision stands but is qualified.
- "<0.0001" is the bootstrap floor, 1/(B+1).

**F1: primary, A\*, α = 0.03 (8 hypotheses)**

| Run (old → new) | Outcome | Old | New | Δ | Rel. | Nominal 95 % CI | d_z | p | Holm thr. | p (Holm) | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mappo_seed1 (200k → 300k) | completion | 0.9879 | 0.9928 | +0.0049 | +0.5 % | [+0.0014, +0.0084] | +0.20 | 0.0049 | 0.0060 | 0.0245 | **reject (new higher)** | **yes** |
| | conflicts / episode | 67.79 | 10.09 | −57.69 | −85.1 % | [−70.46, −44.93] | −0.64 | <0.0001 | 0.00375 | 0.0008 | **reject (new lower)** | no |
| mappo_seed2 (300k → 500k) | completion | 0.9891 | 0.9881 | −0.0010 | −0.1 % | [−0.0045, +0.0024] | −0.04 | 0.5617 | 0.0150 | 1.0000 | not rejected | no |
| | conflicts / episode | 14.45 | 3.02 | −11.42 | −79.1 % | [−17.21, −5.63] | −0.29 | 0.0021 | 0.0050 | 0.0126 | **reject (new lower)** | no |
| mappo_seed3 (150k → 500k) | completion | 0.9886 | 0.9891 | +0.0004 | +0.0 % | [−0.0037, +0.0045] | +0.01 | 0.8396 | 0.0300 | 1.0000 | not rejected | no |
| | conflicts / episode | 42.43 | 4.73 | −37.70 | −88.9 % | [−46.92, −28.49] | −0.58 | <0.0001 | 0.00429 | 0.0008 | **reject (new lower)** | no |
| ippo_seed1 (550k → 500k) | completion | 0.9842 | 0.9828 | −0.0014 | −0.1 % | [−0.0051, +0.0023] | −0.05 | 0.4665 | 0.0100 | 1.0000 | not rejected | no |
| | conflicts / episode | 2.02 | 9.24 | +7.22 | +358 % | [+2.76, +11.69] | +0.26 | 0.0087 | 0.0075 | 0.0348 | not rejected | **yes** |

**F2: supportive, A\*, α = 0.01 (12 hypotheses)**

| Run | Outcome | Old → new | Δ | Nominal 95 % CI | p | p (Holm) | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|
| mappo_seed1 | conflicts / delivered | 3.549 → 0.526 | −3.023 | [−3.697, −2.348] | <0.0001 | 0.0012 | reject | no |
| | conflicts / 100 robot-steps | 4.615 → 0.748 | −3.866 | [−4.599, −3.133] | <0.0001 | 0.0012 | reject | no |
| | base return | 1.187 → 2.909 | +1.722 | [+1.298, +2.146] | <0.0001 | 0.0012 | reject | no |
| mappo_seed2 | conflicts / delivered | 0.755 → 0.158 | −0.597 | [−0.902, −0.292] | 0.0021 | 0.0084 | reject | **yes** |
| | conflicts / 100 robot-steps | 0.974 → 0.209 | −0.765 | [−1.138, −0.392] | 0.0014 | 0.0070 | reject | **yes** |
| | base return | 2.826 → 3.356 | +0.531 | [+0.282, +0.779] | 0.0009 | 0.0054 | reject | no |
| mappo_seed3 | conflicts / delivered | 2.220 → 0.247 | −1.972 | [−2.456, −1.489] | <0.0001 | 0.0012 | reject | no |
| | conflicts / 100 robot-steps | 2.966 → 0.328 | −2.638 | [−3.209, −2.067] | <0.0001 | 0.0012 | reject | no |
| | base return | 1.988 → 3.230 | +1.242 | [+0.943, +1.541] | <0.0001 | 0.0012 | reject | no |
| ippo_seed1 | conflicts / delivered | 0.106 → 0.486 | +0.380 | [+0.144, +0.616] | 0.0090 | 0.0152 | not rejected | **yes** |
| | conflicts / 100 robot-steps | 0.122 → 0.559 | +0.437 | [+0.175, +0.700] | 0.0076 | 0.0152 | not rejected | no |
| | base return | 3.336 → 2.988 | −0.348 | [−0.538, −0.157] | 0.0035 | 0.0105 | not rejected | **yes** |

**F3: secondary, RRT\*, α = 0.01 (8 hypotheses; no RRT\* development evidence; 60 seeds; limited to these four pairs)**

| Run | Outcome | Old → new | Δ | Nominal 95 % CI | p | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| mappo_seed1 | completion | 0.9940 → 0.9954 | +0.0014 | [−0.0020, +0.0049] | 0.4243 | 1.0000 | not rejected |
| | conflicts / episode | 46.77 → 16.91 | −29.86 | [−48.92, −10.79] | 0.0110 | 0.0770 | not rejected |
| mappo_seed2 | completion | 0.9937 → 0.9954 | +0.0017 | [−0.0017, +0.0051] | 0.3241 | 1.0000 | not rejected |
| | conflicts / episode | 17.08 → 1.41 | −15.67 | [−25.58, −5.76] | 0.0111 | 0.0770 | not rejected |
| mappo_seed3 | completion | 0.9951 → 0.9968 | +0.0017 | [−0.0019, +0.0054] | 0.3505 | 1.0000 | not rejected |
| | conflicts / episode | 45.68 → 1.44 | −44.24 | [−61.35, −27.14] | 0.0007 | 0.0056 | **reject (new lower)** |
| ippo_seed1 | completion | 0.9830 → 0.9851 | +0.0020 | [−0.0029, +0.0069] | 0.4043 | 1.0000 | not rejected |
| | conflicts / episode | 4.23 → 4.23 | 0.00 | [−5.79, +5.79] | 1.0000 | 1.0000 | not rejected |

No F3 hypothesis is MC-sensitive. For ippo_seed1, the two arms have exactly the same conflict total (762), hence Δ = 0 and p = 1.

**Supportive outcomes on RRT\*** (descriptive): all reductions for the three MAPPO pairs, no difference for ippo_seed1.

| Run | Conflicts per delivered task | Conflicts per 100 robot-steps |
|---|---|---|
| mappo_seed1 | 2.434 → 0.879 | 3.270 → 1.231 |
| mappo_seed2 | 0.889 → 0.073 | 1.145 → 0.098 |
| mappo_seed3 | 2.375 → 0.075 | 3.199 → 0.102 |
| ippo_seed1 | 0.223 → 0.222 | 0.221 → 0.228 |

### 25.5 Per-level results (exploratory; nominal intervals, no tests)

All CIs below are nominal 95 %.

- **A\*, mappo_seed1 and mappo_seed3.**
  - The conflict reductions come from `obstacles` and `full`: −57 and −102 per episode for mappo_seed1; −42 and −66 for mappo_seed3.
  - On `static` the reductions (−14, −5) have CIs that include 0.
  - mappo_seed1's completion gain is on `obstacles` (+0.0059, CI [+0.0018, +0.0101]) and `full` (+0.0046, CI [+0.0009, +0.0083]).
- **A\*, mappo_seed2.** Conflicts are lower on every level. The CIs exclude 0 on `static` and `full`.
- **A\*, ippo_seed1.** The conflict increase is largest on `static`: 0.52 → 13.04, Δ +12.5, CI [+2.2, +22.9]. It consists mainly of robot–robot vertex conflicts: RRv 608 → 4 699 over all levels.
- **RRT\*.**
  - mappo_seed3's point estimates are lower on every level. The CIs exclude 0 on `obstacles` and `full`.
  - One per-level interval is **unstable**: ippo_seed1, `static`, CI [−351, +339], bootstrap-t q = 55. A single old episode with 375 conflicts (seed 12 012) dominates that level, and resamples that omit it have a near-zero SE.
  - On A\*, ippo_seed1 `full` (q = 4.5) and mappo_seed2 RRT\* `full` (q = 4.0) also have inflated q. These are descriptive per-level intervals only; no inference uses them.

### 25.6 Descriptive results: D2 and D4

**Point summaries, all levels:**

| Planner | Arm | Completion | Conflicts / episode | Conflicts / 100 robot-steps | Base return |
|---|---|---|---|---|---|
| A\* | ippo_seed2 (v5) | 0.9876 | 10.94 | 0.705 | 2.815 |
| A\* | ippo_seed3 (v5) | 0.9910 | 8.66 | 0.618 | 3.001 |
| A\* | matched baseline | 0.9858 | 0.98 | 0.069 | 3.442 |
| RRT\* | matched baseline | 0.9994 | 1.05 | 0.081 | 3.136 |

**Each v5 candidate minus the matched baseline** (descriptive; nominal CIs, no tests):

| Planner | Candidate | Completion Δ | Conflicts / episode Δ |
|---|---|---|---|
| A\* | mappo_seed1 | +0.0071 [+0.0032, +0.0109] | +9.11 [+5.11, +13.11] |
| A\* | mappo_seed2 | +0.0023 [−0.0021, +0.0067] | +2.04 [−1.77, +5.85] |
| A\* | mappo_seed3 | +0.0033 [−0.0010, +0.0076] | +3.75 [+0.09, +7.40] |
| A\* | ippo_seed1 | −0.0029 [−0.0075, +0.0017] | +8.26 [+3.76, +12.76] |
| A\* | ippo_seed2 | +0.0018 [−0.0027, +0.0063] | +9.95 [+5.90, +14.00] |
| A\* | ippo_seed3 | +0.0053 [+0.0009, +0.0096] | +7.68 [+3.39, +11.96] |
| RRT\* | mappo_seed1 | −0.0040 [−0.0072, −0.0009] | +15.86 [+6.46, +25.27] |
| RRT\* | mappo_seed2 | −0.0040 [−0.0066, −0.0014] | +0.36 [−0.17, +0.88] |
| RRT\* | mappo_seed3 | −0.0026 [−0.0046, −0.0005] | +0.39 [−0.10, +0.87] |
| RRT\* | ippo_seed1 | −0.0144 [−0.0186, −0.0101] | +3.18 [−0.46, +6.83] |

**Runtime** (median wall time per episode under 10 concurrent workers; hardware- and load-dependent):

| Arm | A\* | RRT\* |
|---|---|---|
| PPO | 0.46–0.76 s | 12.4–15.9 s |
| Baseline | 0.31 s | 6.5 s |

Planner calls per 100 robot-steps:

| Arm | A\* | RRT\* |
|---|---|---|
| PPO | 9.5–12.8 | 8.5–11.9 |
| Baseline | 6.2 | 5.4 |

### 25.7 Deviations and warnings

1. **Manifest timestamp.** The runner writes the `started` field of the invocation record when the run finishes (15:50:05), not when it starts. The recorded wall time (5 651 s) is correct. This is a labelling defect in the frozen runner and was not repaired.
2. **Byte-reproducibility of `analysis.json` (acceptance criterion 8 not met as written).**
   - **What happened:** a second analysis process produced a file that differs only in the order of the 12 keys of the `usable` listing. Every value is identical, and the two files are equal after canonical key sorting.
   - **Cause:** `block_d_analysis.build` iterates a set of strings, and Python randomizes string hashing per process.
   - **Under a fixed hash seed:** `PYTHONHASHSEED=0` gives byte-identical outputs, equal in content to the official file.
   - **Official file:** the first analysis run (`b9514e6b…`). The second run had overwritten it, and it was restored by copying the first-run file back.
   - **Not repaired:** the frozen code was not changed. A fix (sorting the keys) would need a protocol revision.
3. **Unstable per-level intervals** (§25.5), for exploratory per-level rows dominated by one extreme episode. No inference uses them.
4. **MC-sensitive decisions** in F1 and F2 (tables above). They are reported with that qualification.
5. **Wall time** was 94 min against a projection of about 1.5–2 h. No timeouts occurred.

### 25.8 Limitations

- Four fixed checkpoint pairs: the results are not a method-level effect.
- Block D re-tests the Block C pairs on new instances; the hypotheses were fixed before Block C.
- **RRT\* is secondary.** There is no development evidence, only 60 seeds, and the variability assumption was untested before the run. RRT\* does not support "validated across planners".
- No equivalence margin: completion results are bounded by intervals, not claims of "no cost".
- Completion is near its ceiling, and conflicts are heavy-tailed.
- Only one configuration was tested: greedy, auction allocator, shield off.
- Nominal intervals are not simultaneous.
- **Training-instance disjointness is not verified.** No scenario-instance hashes are stored for training, and training scenarios were not regenerated.
- **Earlier Block D generations.** Two earlier in-memory Block D generations happened for disjointness checks; they stored overlap counts only.
- **Block D is now used.** Any later re-analysis or re-run must be labelled as such.

### 25.9 Reading under the pre-specified table (protocol §11)

These readings are per pair, A\*, F1 after Holm.

| Run (old → new) | Conflicts per episode | Completion | Reading |
|---|---|---|---|
| mappo_seed1 (200k → 300k) | lower (−57.7) | higher (+0.0049), **MC-sensitive** | the new pick is better on both outcomes. The completion part is qualified as MC-sensitive and must not be stated without that qualification |
| mappo_seed2 (300k → 500k) | lower (−11.4) | no detectable difference; nominal CI [−0.0045, +0.0024] | fewer conflicts; no completion difference detected |
| mappo_seed3 (150k → 500k) | lower (−37.7) | no detectable difference; nominal CI [−0.0037, +0.0045] | fewer conflicts; no completion difference detected |
| ippo_seed1 (550k → 500k) | no detectable difference after Holm. Δ +7.2, nominal CI [+2.8, +11.7] excludes 0; p 0.0087 vs threshold 0.0075; **MC-sensitive** | no detectable difference; nominal CI [−0.0051, +0.0023] | no conflict difference detected under the frozen procedure. The direction is unfavourable to the new pick, and the result lies at the decision boundary |

**Supporting evidence.**
- **F2 (supportive)** supports the three MAPPO conflict reductions, with two MC-sensitive decisions for mappo_seed2. It rejects none of the ippo_seed1 hypotheses.
- **F3 (secondary, RRT\*)** confirms only the mappo_seed3 conflict reduction. No RRT\* completion difference was detected.

**What this does not establish.**
- These results do not show that SCB-C is a better selection method in general, or across planners.
- They do not show that the v5 picks match the matched baseline on conflicts. Descriptively, every candidate proposes more conflicts than the baseline, and on RRT\* the baseline's completion is higher.

The decision about `models/` remains deferred until the student has reviewed these results.


## 26. Phase 0: Comprehensive Diagnostics (consolidated; complete, 2026-11-21)

### 26.1 Scope and status

The student decided (2026-11-21) to consolidate three pieces of diagnostic work into **one** roadmap phase, **Phase 0: Comprehensive Diagnostics**, now **complete**:
1. **Phase 6A.0:** failure classification of the 928 greedy block-B failures (§20).
2. **Stored-data diagnostics:** replanning and RRT\* evidence from stored records only, with no simulation.
3. **Deterministic replay (Phase 0b):** the MADQL Q-gap analysis, and exact planner calls by trigger for the learned controllers.

They are not separate phases.

**What Phase 0 did not do.** It changed no algorithm behaviour, trained nothing and ran no new experiment. The only executions were the two deterministic replays of *stored* episodes (6A.0 and Phase 0b), both reproduced exactly. Historical records, checkpoints and frozen protocols are unchanged, `models/` does not exist, and `docs/SESSION_HANDOFF.md` is unchanged.

**Labels used in this section.**
- **[M]** measured from records or exact replays.
- **[C]** read from the code.
- **[H]** a hypothesis.
- **[R]** a recommendation.

**Sources and tools.**

| Item | Path | SHA-256 |
|---|---|---|
| 6A.0 replay | `tools/diagnostics/replay_failures.py` → `results/dev6a/6a0_diagnostics/replay_episodes.jsonl` | `c2d586fc…` |
| Stored-data diagnostics | `tools/diagnostics/phase0_stored.py` → `results/dev6a/phase0/stored_diagnostics.json` (16 input files, hashes inside) | `49c4f5ba…` |
| Replay (Phase 0b) | `tools/diagnostics/phase0b_replay.py` → `results/dev6a/phase0b/{analysis.json, replay_episodes.jsonl, madql_states.npz}` | `ce9aaa59…`, `bddf499e…`, `6570d6e3…` |
| Tests | `tests/tools/test_phase0_stored.py` (10), `tests/tools/test_phase0b_replay.py` (7) | — |

### 26.2 Replay integrity [M]

- **Population:** Phase 0b replayed all **2 160** stored greedy block-B learned A\* episodes, 720 MADQL and 1 440 PPO (`results/dev6/stochastic/records.jsonl`, mode greedy).
- **Production code:** the replay used the production `evaluation._system`, `Engine` (shield off, `path_limit` 6) and the stored `best.pt`.
- **Exact reproduction: 2 160 / 2 160.** Each episode was compared on delivered, released, steps, termination, conflicts by type and total, physical collisions, wait ratio, planner calls, mean flow time and the stored deviation rate. There were 0 mismatches.
- **Q-value capture:** a stateless extra forward pass recorded the Q-values. They reproduce every greedy MADQL action: 0 arg-max mismatches across about 2.0 M non-idle robot-steps.
- **Determinism:** both result tables, recomputed offline from the saved outputs, are identical.
- **Runtime:** 212 s on 10 workers.

### 26.3 Failure modes (from §20, unchanged) [M]

**Stuck robots** (holding a task at the end of a failed episode) are classified over their last 100 steps:

| Class | Share of stuck robots |
|---|---|
| Oscillating | 42–53 % |
| Frozen (waiting in front of a **free** suggested cell) | IPPO 34 %, MADQL 24 %, MAPPO 6 % |
| Conflict loop | MAPPO 40 %, IPPO 13 % |
| Genuine wait-for-cycle deadlock | 1.4 % overall, 59 of these 62 robots also oscillating |

- **Episodes:** 710 of 720 greedy MADQL episodes fail; 218 of 1 440 PPO episodes fail.

### 26.4 MADQL Q-gap [M]

**Definitions.**
- **Gap:** Q(STAY) − the highest-valued *allowed* move.
- **Greedy rule:** a masked arg-max with ties to the lowest index. STAY is index 0, so STAY is chosen exactly when gap ≥ 0. A gap of exactly 0 would be decided by the tie-break.
- **Reference scale:** the per-step reward cost is 0.01.

**Results** (all policies and levels pooled):

| State (non-idle robot-steps) | n | Median gap | 10th–90th percentile | gap > 0 | gap = 0 | \|gap\| ≤ 0.01 | gap > 0.1 |
|---|---|---|---|---|---|---|---|
| Frozen (6A.0 class, last 100 steps) | 101 200 | **+0.044** | +0.004 … +0.216 | **100 %** | **0** | 23 % | 31 % |
| Oscillating | 226 300 | −0.069 | −0.173 … −0.021 | 0.4 % | 0 | 3 % | 0.1 % |
| Other stuck | 97 300 | −0.041 | −0.128 … +0.055 | 26 % | 0 | 9 % | 6 % |
| Moving successfully | 1 219 680 | −0.069 | −0.153 … −0.024 | 0 % | 0 | 3 % | 0 % |
| Ordinary waiting | 356 232 | +0.039 | +0.003 … +0.204 | 99 % | 0 | 23 % | 27 % |

**Breakdowns.**
- **Per policy (frozen states):**

  | Policy | Median gap | \|gap\| ≤ 0.01 |
  |---|---|---|
  | MADQL-1 | +0.060 | 14 % |
  | MADQL-2 | +0.018 | 36 % |
  | MADQL-3 | +0.099 | 9 % |

  Every policy is 100 % positive.
- **Per level (frozen states):** median +0.041 (static), +0.043 (obstacles), +0.050 (full).
- **Against the planner's suggested move**, frozen robots prefer STAY by a median of **+0.134**.

**Findings.**
- **[M]** MADQL's frozen behaviour is a **learned preference for STAY, not tie-breaking.** There is not a single exact tie in 101 200 frozen robot-steps, and STAY is never chosen by the tie-break anywhere.
- **[M]** The preference is often weak: 23 % of frozen steps are within 0.01, and the median is about four step-costs.
- **[H]** This explains the earlier result that ε = 0.05 execution raises MADQL completion from 0.58–0.66 to 0.89–0.92: random actions easily overcome a small STAY preference, but greedy execution never does.
- **[M]** Oscillating robots prefer moving by the same margin as successfully moving robots. 6A.0 shows they follow the planner's suggestion in a median of 50 of their last 100 steps.
- **[H]** Oscillation is therefore a problem of *which move* the policy chooses, not of STAY.

### 26.5 Planner calls by trigger [M]

**Learned controllers** (Phase 0b, exact; T7 is 0 because the resolver is off for learned controllers):

| Controller | Calls | T1 goal | T2 environment | **T3 off-path** | T4–T6 | T8 | Calls per 100 robot-steps | T3 per 100 non-idle robot-steps |
|---|---|---|---|---|---|---|---|---|
| MADQL | 758 823 | 3.5 % | 1.8 % | **94.6 %** | < 0.1 % | 0.1 % | 33.1 | 35.9 |
| IPPO | 123 728 | 24.5 % | 9.5 % | **65.0 %** | 0.2 % | 0.9 % | 11.3 | 13.8 |
| MAPPO | 112 603 | 26.8 % | 10.3 % | **61.6 %** | 0.3 % | 1.0 % | 10.8 | 12.0 |

- **Per policy, T3 share of calls:** MADQL 94.0–94.9 %, IPPO 63.0–66.4 %, MAPPO 59.6–63.3 %.
- **Static level, T3 share:** MADQL 97.9 %, IPPO 75.1 %, MAPPO 73.1 %.

**Stuck robots, last 100 steps** [M]:

| Class | Calls per robot per 100 steps | Share that are T3 |
|---|---|---|
| Oscillating | 54–59, a replan about every other step | ≥ 99.8 % |
| Frozen | 0 | — |
| Conflict loop | about 0 | — |

**Non-learned controllers** (stored Issue #4 / E2 / E3b records; stored-data diagnostics) [M]:
- T3 is **0 %**.
- With A\*: T1 49–67 % and T2 21–25 %; T4 is 1.5 % (baseline) and 29 % (reference follower).
- With RRT\*: T5 retries are 4.8–5.3 %, against 0.1–0.2 % with A\*.

**Association with failure** [M]:
- **MADQL:** episodes with undelivered tasks have *more* planner calls (Spearman ρ +0.27 to +0.78).
- **PPO:** incomplete episodes have *fewer* calls (ρ −0.04 to −0.67).
- **Within failed episodes:** the call rate rises with the number of oscillating robots (ρ 0.70–0.86) and falls with frozen robots.

### 26.6 RRT\* (stored data only) [M]

- **Every RRT\* call that runs uses its whole budget:**
  - E1a: every query ran its full budget;
  - E2: on average 991–1 000 iterations per call, at a median of 111–207 ms per call;
  - learned-controller RRT\* episodes: about 107–156 ms of wall time per call, an upper bound.
- **Most of each call comes after the solution is found** (static 32×32 queries at 2 000 iterations):

  | Map | Median first-solution iteration | Budget run afterwards |
  |---|---|---|
  | Open | 40 | 98 % |
  | 10 % obstacles | 72 | 96 % |
  | 20 % obstacles | 183 | 91 % |
  | Warehouse | 268 | 87 % |
  | Rooms | 446 | 78 % |

- **Dense and corridor maps fail often instead:** at 500 iterations, 30–92 % of queries never find a path. Each such failure spends the whole budget.
- **In replanning (E2):** 185 of 9 462 calls (2.0 %) exhaust the budget.
- **Not reconstructable from stored data:** per-call RRT\* data inside learned-controller episodes (not instrumented). Phase 0b did not capture it; the student did not approve it.

### 26.7 Implications [H unless marked]

1. **Completion failures need an execution-layer mechanism, not faster replanning.**
   - Frozen robots, a learned STAY preference, make no planner calls.
   - Conflict-loop robots make about none.
   - Oscillating robots replan constantly but ignore half of the suggestions they receive.
   - Replanning optimization would mainly cut computation: up to 95 % of MADQL calls and about 62–65 % of PPO calls. It would not remove the failures.
2. **[R] The Execution Supervisor (L2, §5) is the most direct lever.**
   - Its stall breaker targets frozen robots and oscillation.
   - Its right-of-way rule targets conflict loops.
   - The existing resolver covers the rare genuine cycles.
   - Effectiveness is unproven until it is measured (Phase 3 design: `docs/research/2026-11-21-issue-6a-phase3-execution-supervisor-design.md`).
3. **MADQL retraining is not yet justified.** A small learned STAY preference may be addressable at execution time. Retraining is reconsidered in Phase 4 after the supervisor results.
4. **RRT\* budget control (P2) remains well supported for runtime**, and replan repair (P1) for runtime, especially with RRT\*. Both follow Phase 3.
5. **A\* stays unchanged.**

### 26.8 Limitations

- **Coverage:** block-B development data; greedy mode; A\* for the replays; auction allocator; shield off; Issue #5/#6 checkpoint picks.
- **Classifier:** the stuck classes are the 6A.0 heuristic thresholds.
- **Necessity of replans:** whether each T3 replan produced the path the robot already had was not measured.
- **RRT\*:** no per-call RRT\* data exists for learned controllers.

## 27. Updated Issue 6A roadmap (student decision, 2026-11-21)

This changes **priority only**. The phase numbers are kept, no phase is removed, and **Issues #7 and #8 keep their scope and order**.

| Priority | Phase | Content | Status |
|---|---|---|---|
| 1 | **Phase 0: Comprehensive Diagnostics** | 6A.0 + stored-data diagnostics + replay (§20, §26) | **Complete** |
| 2 | **Phase 3: Execution Supervisor** | Stall breaker, oscillation recovery, right-of-way, resolver integration (L2) | **Complete** (2026-11-22). Configuration going forward: R + S + D (§30, §30.1) |
| 3 | **Phase 1: Replanning optimization** | Replan repair (P1) | **Complete** (2026-11-22). P1 rejected as default; repair stays off (§31) |
| 4 | **Phase 2: RRT\* optimization** | Budget control (P2) | **Complete** (2026-11-22). Inconclusive; P2 stays optional and off by default; m = 400 is carried to Phase 6 (§32) |
| 5 | **Phase 4: MADQL optimization** | Depends on Phases 3, 1 and 2, and on any remaining evidence of learning weaknesses | **Complete** (2026-11-23). MADQL-v2 n3 (n-step returns) adopted on Block C in all conditions; forward MADQL candidate for Phase 6; historical MADQL kept as the reference; robot–obstacle conflicts to be a Phase 6 veto (§33) |
| 6 | Phase 5: A\* optimization | Exact heuristic (P3) | **Not pursued: not justified** (student decision, 2026-11-23; §34). A\* unchanged |
| 7 | **Phase 6: Integrated validation** | All required algorithm × planner combinations, on held-out data | **Complete** (2026-11-23). Supervisor confirmed; MADQL-v2 not confirmed (RO veto); integrated system confirmed as declared (§35) |

**Already complete and not repeated:** PPO checkpoint selection (L1 / SCB-C v5, §21–§23) and its Block C and Block D evaluations (§24–§25).

Each phase needs its own approval before implementation, training or expensive experiments.


## 28. Phase 3: Execution Supervisor, implemented and software-tested (2026-11-21; not yet evaluated)

### 28.1 Status

The student approved design decisions D1–D11 and authorized implementation and software testing (2026-11-21).

- **Implemented:** the supervisor with R, S, O and the D2b detour recovery; the engine stage before the shield; the explicit resolver switch; the additive record fields.
- **Not done (not authorized):** calibration (Stage 3a), any Block C or Block D simulation, any evaluation run, any training.
- **No claim is made** that the supervisor improves completion or reduces conflicts. That requires the Phase 3 experiments.

The full specification, the implemented behaviour, the deviations and the tests are in `docs/research/2026-11-21-issue-6a-phase3-execution-supervisor-design.md` (§14 records the implementation).

### 28.2 Architecture (as implemented) [C]

`policy proposal → Execution Supervisor (engine stage 5a; off by default) → predictive shield (5b) → arbiter (6) → motion`

- **Output per robot:** exactly one of the policy's action, STAY, or the planner's suggested action (mask-valid).
- **Untouched:** the arbiter, shield, resolver, endpoint recovery, planners, policies, the training environment and rewards.
- **Metrics:** the policy's own proposal remains `StepOutcome.actions`, and the conflict set of the raw proposal (K^ctrl) and the adherence metrics use it. The supervisor's output is recorded separately (`supervised_actions`, `supervisor_events`, `extra["supervisor"]`).
- **Resolver switch:** `EvalTask.deadlock` (default `None`) keeps the original rule, which is on for the baseline only, and can now enable the resolver for any controller.
- **Supervisor switch:** `EvalTask.supervisor` (default `"off"`) applies to learned controllers only.

**Interaction of the mechanisms:**
- **D** changes goals upstream (the existing resolver). Its detour members are excused from S and O. A member that ignores its detour gets a bounded D2b recovery.
- **S and O** act per robot, with bounded recoveries that execute the filtered suggestion.
- **R** settles vertex contests for free cells, with a fairness-bounded priority key.
- **Left to the arbiter:** moves into a staying robot's cell and head-on swaps (identical motion; counted only). Persisting swaps are resolved by D.

### 28.3 Deviations from the approved design, found in testing [M on scripted scenarios]

1. **The anti-backtrack rule is replaced by a recovery-cycle guard.** The rule blocked legitimate path moves; a recovery now ends if it revisits a cell.
2. **R changes only free-cell contests.** Rewriting moves the arbiter cancels anyway inflated the override statistics and hid the policy's conflicts.
3. **Escalation of repeated stalls is an option, off by default.** Enabling it would loosen the D11 override bound, so it awaits approval.

### 28.4 Verification [M]

- **Tests:** 61 new tests: 31 unit and property, 17 integration, 13 regression against stored records.
- **Full fast suite:** **1 170 passed**, 1 skipped (POSIX only), 62 deselected (slow / training / e2e markers), 0 failures, no warnings, 5 min 45 s.
- **Ruff and mypy:** clean (84 source files, plus the new tests).
- **Disabled supervisor:**
  - byte-identical traces;
  - stored block-B records reproduced exactly by `run_episode`, all non-timing fields **and field order** (MADQL greedy, IPPO greedy, MAPPO stochastic, matched baseline), plus the resolver-on baseline's values.
- **Safety (property tests):**
  - the output alphabet holds and every output is mask-valid;
  - no invalid move is introduced;
  - the arbiter and the post-step check never report a physical violation;
  - R leaves no shared free target and is idempotent;
  - the fairness bound holds;
  - the supervisor is deterministic.
- **Scripted behaviour** (not evaluation evidence):
  - a frozen robot, an oscillating robot and a free-cell conflict loop are each stuck without the supervisor and delivered with it;
  - a two-lane head-on swap is resolved only with the resolver (15 steps, with or without the supervisor);
  - a one-lane swap stays unresolved;
  - a robot waiting legitimately is never overridden.
- **Overhead:** about 0.04 ms per step against a 0.82 ms step (8 robots, full dynamics).
- **Unchanged:**
  - frozen Block C/D files (hashes verified);
  - historical results, checkpoints and `models/` (absent);
  - `docs/SESSION_HANDOFF.md`;
  - Issues #7 and #8.

### 28.5 Readiness and open decisions

Phase 3 is **ready for experimental evaluation at software level**. The following need approval first:
1. **Stage 3a calibration** from a replay of the stored block-B greedy episodes, with the parameters frozen afterwards.
2. **The escalation option:** an ablation arm, enabled by default, or neither.
3. **The pre-registered go / no-go rule** (D10), and approval of Stages 3b, 3c and 3e.


## 29. Phase 3, Stage 3a: calibration and freeze (2026-11-21)

**Full record:** `docs/research/2026-11-21-issue-6a-stage3a-calibration-freeze.md`. Freeze record: `results/dev6a/phase3/stage3a/freeze.json`.

- **Data [M].**
  - 2 700 stored greedy Block B episodes: 2 160 A\* and 540 RRT\*.
  - Replayed with the supervisor off and shadow detectors that never act; 0 mismatches, plus 0 in the 2 160-episode A\* extension.
  - No Block C/D data, no new experiment, no training.
- **Rule.** Design §5: ≤ 1 % of the legs of successful episodes may trigger. Disclosed revisions:
  - a group is eligible only where success is typical, which excludes MADQL (10 / 720 successful; its successful episodes contain the targeted stalls);
  - the rule is applied per planner;
  - an O pattern qualifies only with K < W;
  - an extended O grid for A\*.
- **Frozen** (`supervisor.FROZEN_PARAMS`, pinned by test; used by every supervised `run_episode`):

  | Planner | W | O (K, cells, m) |
  |---|---|---|
  | A\* | 24 | (14, 3, 7) |
  | RRT\* | 32 | (8, 3, 4) |

  - Unchanged: H 8, Δ_prog 2, H_stay 3, L_max 3, resolver defaults.
  - Escalation stays off.
  - The design values W = 12 and (8, 2, 4) fail the approved rule (S: 2.0–2.1 % A\*, 5.6 % RRT\*).
- **Evidence [M].**
  - Stuck A\* robots are detected at ≈ 100 % (frozen) and 99.5 % (oscillating) at any W ≤ 64, so W trades false triggers only against latency.
  - O fires a median of 10 steps before S.
  - Bootstrap stability: A\* W 89 %, A\* O 62 %, RRT\* W 67 %.
  - Burst bound [C]: H / (H + min(W, K)) = 36 % (A\*), 50 % (RRT\*).
- **Recommended go / no-go** (protocol §6, needs approval):
  - hard integrity and safety gates;
  - the primary completion criterion: MADQL improves by ≥ one task per episode (Holm), no PPO family or policy is harmed, and a "no harm" interval that excludes a one-task loss;
  - targeted categories with a rule against re-labelling;
  - factorial retention rules;
  - priority-fairness vetoes;
  - interventions, bursts and overhead reported.
- **Open decisions:**
  - the checkpoint identity of four PPO runs: the stored Block B data are their Issue #6 picks, not the SCB-C v5 picks;
  - approval of the rule revisions;
  - the go / no-go criteria (D10);
  - the 3e seed subset.

  Stages 3b, 3c and 3e were **not** run.


## 30. Phase 3, Stages 3b, 3c and 3e: evaluation results (2026-11-21/22)

**Full record:** `docs/research/2026-11-22-issue-6a-phase3-evaluation.md`; generated tables `docs/research/2026-11-22-issue-6a-phase3-evaluation-tables.md`; outputs `results/dev6a/phase3/eval/`.

- **Design [C]:**
  - Block C, A\* 11 000–11 119 × 3 levels and RRT\* 11 000–11 039 × 3 levels;
  - 9 policies: MADQL Issue #5 picks and the PPO SCB-C v5 picks (4 of which differ from the Stage 3a calibration checkpoints; parameters not re-calibrated);
  - the frozen Stage 3a parameters;
  - arms OFF / full (3b, 3e) and the 2³ factorial over {R, S+O, D} plus "S without O" (3c);
  - 6 480 + 22 680 + 2 160 episodes.
- **Integrity [M]:**
  - 0 errors, collisions or supervisor-introduced invalid moves;
  - R-loss streak ≤ 1;
  - no burst-bound violation;
  - OFF reproduces 1 440 / 1 440 stored Block C records;
  - 627 / 627 replays identical;
  - checkpoints unchanged.
- **3b (A\*) [M]:**
  - Completion Δ (full − OFF), with all three families Holm-rejected:

    | Family | Δ | 95 % CI |
    |---|---|---|
    | MADQL | +0.373 | [0.358, 0.389] |
    | IPPO | +0.012 | |
    | MAPPO | +0.010 | |

  - Undelivered tasks per episode: MADQL 7.22 → 0.05.
  - Frozen and oscillating stuck robots almost eliminated, with total stuck robots down in every family.
  - High-priority flow time and undelivered tasks improve or are unchanged.
  - Supervisor cost about 4 % of step time; modified share 1.3 % (MADQL) and 0.15 % (PPO).
  - **Go** under the recommended criteria.
- **3c [M]:**
  - **S+O main effect:** MADQL +0.370, IPPO +0.010, MAPPO +0.008.
  - **R:** +0.001 to +0.002 (IPPO and MAPPO Holm-rejected). R interacts positively with S+O and reduces S/O's interventions.
  - **D:** +0.000 to +0.002 (MAPPO rejected); it reduces MADQL deadlock cycles.
  - **O over S:** no completion or category effect, a faster MADQL high-priority delivery (−4 steps), and 20–80 % more overrides.
  - No mechanism shows harm.
  - **Rule outcome:** keep R, S, D; drop O. Recommended Phase 6 configuration: R + S + D (the student's decision).
- **3e (RRT\*) [M]:**
  - Completion: MADQL 0.695 → 0.996, IPPO 0.992 → 1.000, MAPPO 0.996 → 1.000 (Holm-rejected gains).
  - No harm or veto.
  - **Go.**
- **Acceptance:**
  - implementation and experiments are complete;
  - "go towards Phase 6" holds under the recommended criteria, which need formal approval (D10), as does the O decision;
  - the confirmatory test is reserved for Phase 6 on a new held-out block.


### 30.1 Student decisions (2026-11-22); Phase 3 complete

1. **Configuration going forward:** R + S + D, with O dropped. This is arm `R_S_D` with the frozen Stage 3a parameters. All Phase 3 results, including those for the full supervisor, are kept unchanged.
2. **Criteria approved.** The Stage 3a go / no-go criteria are approved for interpreting the Phase 3 development-block results.
   - Verdicts: 3b **go**, 3e **go**, 3c R/S+O/D retained.
   - This is development evidence. Phase 6 must confirm it on a new held-out block.
3. **Phase 3 is complete.** Phase 1 (replanning optimization) is next.

## 31. Phase 1: Replanning optimization (P1), evaluated (2026-11-22)

Full record: `docs/research/2026-11-22-issue-6a-phase1-replanning.md` (protocol §1–3 pre-registered before the run) and `2026-11-22-issue-6a-phase1-replanning-tables.md`.

**Implementation:**
- A bounded rejoin repair on T3 deviations: BFS over free cells outside `B_i ∪ Q_i`, radius `H_r` = 3, accepted only if it passes the T2 check, otherwise the T3 replan.
- Files: `MonitorParams.repair_radius`, `PathMonitor._repair` and `EvalTask.repair`.
- Off by default, and byte-identical when off.

**Evaluation:**
- Block C, 4 conditions (A\* and RRT\*, each with R + S + D and with no supervisor), repair 0 vs 3, 17 280 episodes.
- The approved Phase 3 criteria structure was used, with δ₁ = one task per episode.

**Results [M]:**
- Planner calls and planner work fall in every family and condition (Holm).
  - With the supervisor, calls per 100 robot-steps fall by 59–86 %.
  - RRT\* wall time per episode falls 2.0–4.8×.
  - A\* wall time per episode *rises* about 20 %, because episodes are longer.
- MADQL completion falls with the supervisor (Holm):
  - A\*: −0.010;
  - RRT\*: −0.038.
- PPO completion shows no detected change with the supervisor.
- Without the supervisor, completion falls 7–29 points for every family.
- High-priority flow time rises 13–43 % for every family (Holm veto).
- Conflicts show no change, and there were 0 safety events.

**Verdict:**
- **Reject** in C1–C4. P1 is not adopted as the forward default.
- The forward configuration remains R + S + D with T3 replanning.
- No parameter was re-tuned.

**Next:** Phase 2 (RRT\* budget control, P2) is independent of P1 and needs its own approval.

## 32. Phase 2: RRT\* budget control (P2), evaluated (2026-11-22)

Full record: `docs/research/2026-11-22-issue-6a-phase2-rrt-budget.md` and `2026-11-22-issue-6a-phase2-rrt-budget-tables.md`. Write order:
- the candidates and the selection rule (§1–3) were fixed before the microbenchmarks;
- the Block C protocol (§5) was written after the selection and before any episode.

**Implementation:**
- `RRTStarConfig.refine_iterations` (k): stop k iterations after the first solution.
- `RRTStarConfig.patience` (m): stop after m iterations without a strict decrease of the best cost.
- Both are optional, None by default, and act only after a solution exists.
- A stopped run is therefore an **exact prefix** of the full-budget run, with the same random stream. Hence:
  - a valid path whenever one is returned;
  - the best cost never increases;
  - status and fallback unchanged;
  - deterministic stopping, by iteration count rather than wall clock;
  - byte-identical behaviour when off.
- `EvalTask.rrt_refine` and `rrt_patience` pass the rule through.
- Informed sampling and tree reuse were not implemented.
- Tests: 8 unit and property tests (Hypothesis), 3 integration tests, 5 tool tests.

**Microbenchmarks [M]:**
- E1a, 2 800 queries × 12 configurations:
  - early stopping lengthens executed paths markedly: first solution +15 %, k = 400 +5 %, m = 200 +2 %;
  - only **m = 400** met the 1 % screen: Q 1.0007, 5.6 % fewer iterations, untraced time 0.90×.
- E2 (a reported comparison only): −5 % iterations and −7 % planning time, with no sign of harm.

**Block C [M]:**
- 9 learned policies with R + S + D (L), and the baseline (B); RRT\* seeds 11 000–11 039 × 3 levels; full vs m = 400; 2 400 episodes.
- G0 passed: G0.1 1 080 / 1 080 identical to Phase 1; replay identical; 0 invalid reference paths in 521 346.
- RRT\* iterations per robot-step: −16.2 % MADQL, −11.2 % IPPO, −10.7 % MAPPO, −9.7 % baseline (all Holm).
- RRT\* time: −14 to −24 %. Episode wall time: −15 to −24 %.
- Planner calls per robot-step unchanged.
- Completion, conflicts and high-priority flow time: no Holm-detected change.
- Flow time +0.3 to +1.3 % (descriptive).

**Verdict:**
- B **adopt**. L **inconclusive**: the IPPO and MAPPO family completion intervals are (−∞, ∞), because the full arm is at the ceiling (IPPO 1.0000).
- A percentile sensitivity analysis, not pre-registered, gives lower bounds ≥ −0.0007, against −δ₁ = −0.052.
- **P2 is not the forward default.** It stays an option, off by default. m = 400 is carried, untuned, to Phase 6, which must pre-register how degenerate intervals are handled.
- Development evidence only.

**Next:** Phase 4 (MADQL optimization, conditional) needs its own approval.


## 33. Phase 4: MADQL optimization (MADQL-v2), evaluated (2026-11-22/23)

Full record: `docs/research/2026-11-22-issue-6a-phase4-madql.md` and `2026-11-22-issue-6a-phase4-madql-tables.md`. Write order:
- the diagnostics, hypotheses, candidates, training protocol and selection rule (§1–3) were fixed before any training;
- the Block C protocol (§5) was written after the Block S selection and before any Block C episode (document SHA `8405d73c` at the check).

**Diagnosis [M]** (Block B, 720 MADQL episodes):
- MADQL's waiting is not congestion: 24.6 % of robot-steps are STAY with a *free* suggested cell, 0.06 % with an occupied one. 83 % of STAYs repeat the same observation.
- The decisive Q gaps (median +0.038) are smaller than the TD noise (≈ 0.115 RMS), and the greedy action ranking churns by 35–44 % per 50 k steps.
- [C] At a self-loop STAY, the one-step TD correction is only r − (1 − γ)Q ≈ −0.04; an n-step return makes it about n times larger.
- Aliasing (obs-v1 has no stall information) is also present; its direct remedy, obs-v2, is outside scope.

**Implementation:**
- `MADQLParams.n_step` (default 1, omitted from the configuration dictionary at 1, so the Issue #5 hash is unchanged).
- `NStepAccumulator` in `marl/replay.py`; per-row γ^h in the Double-DQN target.
- The historical configuration reproduces byte-identically (50 k checkpoint and log).
- 19 MADQL tests, including a Hypothesis property test, and 16 tool tests.
- PPO, allocation, planners, supervisor, reward and observation are unchanged.

**Training and selection [M]:**
- 3 candidates (n3; t8 = target period 8 000; n3t8) × seeds 1–3, 1 M steps each, under the unchanged historical rules.
- Block A best completion: historical 0.63–0.72; n3 0.84–0.89; n3t8 0.83–0.87; t8 0.67–0.75.
- Block S (5 760 episodes): all three eligible. **n3** had the highest raw (no-supervisor) completion, 0.861 vs 0.635 historical, and was selected by the pre-declared rule.

**Block C [M]** (FA = A\* + R + S + D; FR = RRT\* + R + S + D; NA = A\*, no supervisor; 5 040 episodes):

| | FA | FR | NA |
|---|---|---|---|
| Completion hist → v2 | 0.9970 → 0.9977 (n.s.) | 0.9914 → 0.9968 (Holm) | 0.624 → 0.860 (Holm) |
| Flow time, common tasks | −16.6 (Holm) | −21.9 (Holm) | −28.5 (Holm) |
| Planner calls / 100 robot-steps | −1.82 (Holm) | −1.08 (Holm) | −9.3 (Holm) |
| Supervisor changes / 100 robot-steps, or stuck robots / episode (NA) | −0.42 (Holm) | −0.36 (Holm) | −3.21 (Holm) |
| P4 vetoes | none | none | none |
| Verdict | **adopt** | **adopt** | **adopt** |

- STAY share falls from 13 % to 5 % (FA) and from 23 % to 10 % (NA). Spells of 20 or more steps fall by about 60 %.
- Inference cost is unchanged. RRT\* episode time is about 25 % shorter.
- 0 collisions, 0 invalid moves, 0 invalid reference paths.

**Cost [M, post-hoc]:**
- Robot–obstacle vertex conflicts per move are 1.9–2.5× the historical rate (nominal intervals exclude zero in all conditions). All of them were prevented by the arbiter.
- The pre-registered conflicts-per-episode veto was not Holm-rejected (FA Holm p 0.028 at α 0.005).
- This changes no verdict.

**Decision (pre-registered rule):**
- MADQL-v2 n3 is the **forward MADQL candidate for Phase 6**.
- The historical MADQL stays the reference; its checkpoints and results are unchanged.
- **Recommended for approval:** Phase 6 pre-registers the robot–obstacle conflict rate as a veto outcome for MADQL-v2.
- Development evidence only. H4.2 (slower target updates) is not supported as a stand-alone change; aliasing is reduced but not removed.

**Phase 5 (A\* exact heuristic, P3): not justified.**
- A\* planning is 8–11 % of episode wall time.
- A\* paths are already optimal.
- No result points at A\* as a cause of failures or delays.

**Next:** Phase 6 (integrated validation) needs its own approval and a pre-registered protocol.


## 34. Phase 4 decision and Phase 5 (A\* optimization): not pursued (2026-11-23)

### 34.1 Student decisions

1. **Phase 4.**
   - MADQL-v2 n3 (3-step returns) is the **proposed MADQL configuration to carry into Phase 6**. The historical MADQL is retained as the reference.
   - Phase 6 must evaluate **robot–obstacle conflicts per move as a pre-registered veto outcome**.
   - The Phase 4 Block C results are development evidence, not independent confirmation. Phase 6 must provide the independent held-out validation.
2. **Phase 5.** Not implemented at this stage. The conclusion was to be verified against this proposal and the roadmap criterion, and recorded as not pursued only if the evidence supports it.

### 34.2 Verification against the proposal and the roadmap criterion

**The criterion** (roadmap §27, row 6): "Conditional; only if a measurable benefit is justified".

**What P3 is** (§6): an exact static heuristic for A\*, the cached `DistanceOracle` field. It is admissible and consistent.
- Expected effect: "Expansions → ≈ path length on static maps".
- Evidence: F8.
- Cost and risk: "Low; identical optimal costs".

**What F8 measured** (§1; post-Issue-6 analysis, Fig. 1c):
- Expansions ÷ path length is a median of 1.4–7.5 on **dense random and room maps of 64–128 cells**.
- It is exactly 1.0 on open and warehouse maps.
- It was rated low priority.
- The post-Issue-6 analysis (L3) gives the main motivation as E6 (128 × 128 maps, up to 32 robots), which belongs to Issue #8.

**Evidence from the Issue 6A evaluations [M]:**

| Question | Evidence | Reading |
|---|---|---|
| How much time could a faster A\* save? | A\* planning is **6–13 %** of episode wall time for every controller family on Block C with A\*: Phase 1 `records_1a`, OFF and R + S + D, 9 policies (IPPO 0.060–0.064, MAPPO 0.059–0.066, MADQL 0.090–0.126). Phase 4: 0.076–0.109 | Even an infinitely fast A\* saves at most this share. P3 removes only part of it |
| Could path quality improve? | No. A\* already returns optimal paths, and P3 keeps the optimal costs (§6) | No task or coordination benefit is possible from path cost |
| Is A\* a cause of failures, conflicts or delays? | No Phase 0, 3, 1, 2 or 4 result attributes failures, conflicts or waiting to A\*. The remaining weaknesses are in the learned policies (Phase 4 §7) and in coordination (P4, reservations) | No mechanism to improve |
| Does F8 apply to the Issue 6A scenario family? | The 6A blocks use 32 × 32 maps at density 0.15. F8's large ratios are for dense and room maps of 64–128 cells, which are not in this family | The measured inefficiency is outside the Issue 6A scope |
| Risk | A different heuristic can change tie-breaking among equal-cost paths, and so the paths and observations seen by the learned policies [C]. Equivalence would then have to be re-established for every downstream result | A cost with no measurable 6A benefit |

### 34.3 Record

- **Phase 5 is not pursued in Issue 6A: not justified** under the roadmap criterion. On the Issue 6A evidence there is no measurable benefit to justify it.
- No optimization was created to complete the phase number.
- The A\* implementation, its heuristic and its default configuration are unchanged.
- **Open, outside Issue 6A:** L3's rationale for large maps (E6, Issue #8) is not evaluated here. If Issue #8's runtime on 128 × 128 maps makes A\* a bottleneck, P3 could be proposed there, with its own approval and an equivalence check on paths.

### 34.4 Clarification of the Phase 4 robot–obstacle figure (2026-11-23)

§33 ("1.9–2.5× the historical rate") and the status bullet ("about 2×") report the Phase 4 post-hoc definition: ROv only, counted over all robots, divided by the MADQL policy's own non-STAY choices of **non-idle** robots only (source: `results/dev6a/phase4/eval/posthoc_conflicts.json`). Idle robots move often under MADQL-v2 (27–31 % of all moves; 7–15 % for the historical MADQL), so the numerator and denominator covered different robots. With the definition approved for Phase 6 (ROv + ROs over all robots, divided by every move reaching the arbiter), the same Block C instances give FA ×2.17, FR ×1.83 and NA ×1.43 (NA not significant) (`results/dev6a/phase6/dev_ro_exact/dev_ro_exact.json`). §33 and the status bullet are unchanged.

## 35. Phase 6: integrated confirmatory validation, results (2026-11-23)

Full record:
- `docs/research/2026-11-23-issue-6a-phase6-results.md` (results);
- `2026-11-23-issue-6a-phase6-tables.md` (tables);
- `2026-11-23-issue-6a-phase6-protocol.md` (frozen protocol, with Amendment A1 and Deviation W1);
- `2026-11-23-issue-6a-phase6-decision-review.md` (D1–D13).

**Design:**
- Held-out **Block V**: 14 000–14 199; the RRT\* subset 14 000–14 059.
- 22 080 episodes, with the forward configuration (greedy, auction, shield off, P1 / P2 off, 1 000 RRT\* iterations).
- Three confirmatory hypotheses, error-controlled at 0.05 per unit (7 units; global bound 0.302).

**Integrity:**
- Frozen before any Block V generation; the Block V instances are unique and disjoint from A, B, C, D and S.
- 0 errors and 0 safety events; G0 pass; replay 442 / 442; analysis byte-identical.
- Checkpoints unchanged; protected artifacts unchanged.
- **D12 not satisfied (waived by the owner).**

| Hypothesis | Verdict | Main evidence |
|---|---|---|
| **H-L2** (supervisor vs none) | **go** (A\*) / **go** (RRT\*) → **confirmed** | Completion, A\*: MADQL 0.620 → 0.996, IPPO +0.014, MAPPO +0.011 (Holm). All stuck categories reduced. No RO increase rejected |
| **H-v2** (MADQL-v2 n3 vs historical) | FA **reject**, FR **reject**, NA **adopt** → **not confirmed** | **RO per 1 000 moves:** FA 0.669 → 1.219 (Holm p 0.0018), FR 0.553 → 0.927 (Holm p 0.0004), so the pre-registered veto applies. The efficiency gains and NA completion +0.239 do not override it |
| **H-L12** (Issue 6A system vs Issue #6 system) | **go** / **go** → **confirmed as declared** | Completion gains in every family and planner (Holm); no veto |

**Forward configuration** (protocol §8):
- execution supervisor R + S + D (frozen Stage 3a, O off);
- the SCB-C v5 PPO picks;
- **the historical MADQL.**

MADQL-v2 n3 remains a documented, separately labelled configuration. Whether its robot–obstacle hazard (below the PPO policies' rate in the pre-registered, non-gating context comparison) is acceptable is a decision for the owner. It is not part of the confirmatory result.

## 36. Issue 6A closure (2026-11-23)

**Final status of every phase:**
- Phase 0: complete.
- Phase 3: confirmed in Phase 6.
- Phase 1: rejected.
- Phase 2: inconclusive (P2 off).
- Phase 4: MADQL-v2 not confirmed.
- Phase 5: not pursued.
- Phase 6: complete.

The closure report is `docs/research/2026-11-23-issue-6a-closure.md`.

**Clarification of the project structure (dated; the earlier text is unchanged):**
- §2, §18 and §19 A.1 describe Issue 6A as a "ninth issue entry" / "roadmap revision 3".
- The project keeps **exactly eight consolidated GitHub issues**. **Issue 6A is a workstream within Issue #6**, not a separate (ninth) GitHub issue.
- `docs/07-development-plan.md` is unchanged and still lists eight issues.
- Issues #7 and #8 are unchanged and untouched.
