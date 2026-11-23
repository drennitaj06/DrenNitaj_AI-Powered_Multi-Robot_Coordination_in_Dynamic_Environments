# Block D final evaluation of H-L1: protocol (revision 2, approved; not yet frozen)

*Issue 6A.*

**Status and history.**
- Revision 2 was approved by the student on 2026-11-21, together with the final corrections A–D and the decisions recorded in §14.
- Implementation and tests were authorized at the same time.
- **This document is not frozen.** The student must separately approve the freeze, and separately authorize Block D execution.

**Files hashed at the freeze.** `block_d.py freeze --authorized-freeze` hashes:
- this file;
- `tools/selection/block_d.py` and `tools/selection/block_d_analysis.py`;
- the reused frozen files `tools/selection/block_c_analysis.py`, `tools/selection/block_c.py` and `tools/selection/selection_block.py`;
- the frozen SCB-C v5 selector, `tools/selection/scbc_frozen.py` and `tools/selection/scbc.py`;
- the verification suite, `tests/tools/test_block_d.py`.

Every later step refuses to run if any of these hashes has changed.

**What else the freeze records** (`freeze_manifest`, read only):
- whether the SCB-C code hashes equal those recorded in `selections.json`, and whether `selections.json` itself has its recorded hash. The freeze refuses if either differs;
- the hashes of the selection artifacts: `selections.json`, and the Block C check and manifest and the Block S manifest;
- the hashes of `best.pt`, `best.json` and the selected step checkpoints of all six runs;
- the environment: Python, NumPy, PyTorch and platform.

**Labels.**
- **[M]** a value measured read-only from stored records, files or checkpoints;
- **[P]** a projection from measured values;
- **[A]** an assumption.

## 1. Objective and the claims it can support

Block D tests **H-L1 only**: the frozen SCB-C v5 checkpoint pick against the Issue #6 pick, run by run, on held-out instances. It has no L2 hypotheses and makes no further confirmatory or exploratory claims.

**What it can support:**
- per-pair statements on A\* from family F1;
- supportive A\* statements from F2;
- secondary RRT\* statements from F3, limited to the four tested checkpoint pairs;
- descriptive figures for the two unchanged candidates and the matched baseline;
- safety counts.

**What it cannot support:**
- that SCB-C is a better selection method in general;
- that SCB-C is validated across planners;
- "no completion cost": no equivalence margin is set;
- anything about MADQL, stochastic execution, the shield or the allocators.

Block D is not E5 or E6. Those remain in Issue #8, on test seeds 20 000–20 999.

## 2. Experiment matrix

**Condition for every arm** (the existing evaluation configuration):
- greedy mode, auction allocator, shield off;
- resolver off, endpoint recovery on;
- `path_limit` 6, horizon `max_steps` of the scenario (400);
- one episode per instance, replicate 0;
- RRT\* with 1 000 iterations;
- the production `run_episode`, unchanged.

| ID | Arm | Planner | Seeds × levels | Episodes | Role |
|---|---|---|---|---|---|
| D1 | mappo_seed1, mappo_seed2, mappo_seed3, ippo_seed1 × {old = Issue #6 pick, new = SCB-C v5 pick} | A\* | 12 000–12 199 × 3 | **4 800** | F1 primary, F2 supportive |
| D2 | ippo_seed2, ippo_seed3 (v5 pick = Issue #6 pick) | A\* | 12 000–12 199 × 3 | **1 200** | descriptive |
| D3 | the four D1 pairs × {old, new} | RRT\* | 12 000–12 059 × 3 | **1 440** | F3 secondary |
| D4 | matched baseline (`baseline-noresolver`) | A\* + RRT\* | 200 × 3 + 60 × 3 | **780** (600 + 180) | descriptive only |
| | **Total** | | | **8 220** (A\* 6 600, RRT\* 1 620) | |

**Excluded:**
- the original baseline (resolver on), the reference follower and MADQL;
- the unchanged runs on RRT\*;
- stochastic mode, the shield and the sequential allocator;
- every other checkpoint and run.

**Pairs and statistical units.** An *episode pair* is the old and the new episode of one instance (run, planner, level, seed). The *seed unit* is the statistical unit: one seed with its three levels, for one run and one planner.

| | Episode pairs | Seed units |
|---|---|---|
| D1 (A\*) | 4 × 200 × 3 = **2 400** | 4 × 200 = **800** |
| D3 (RRT\*) | 4 × 60 × 3 = **720** | 4 × 60 = **240** |
| **Total** | **3 120** | **1 040** |

Inference resamples seed units. Episode pairs are only checked for identical configuration and instance.

## 3. Checkpoint identities and integrity

[M] SHA-256 prefixes (read-only, 2026-11-21). In all six runs, `best.pt` is intact and its tensors are identical to the step file.

| Run | Old (Issue #6 `best.pt` = step file) | New (SCB-C v5) |
|---|---|---|
| mappo_seed1 | step_00200000 `1cf6d409…` | step_00300000 `e1b93a11…` |
| mappo_seed2 | step_00300000 `7d4f8a86…` | step_00500000 `4cef9b23…` |
| mappo_seed3 | step_00150000 `3a5f1244…` | step_00500000 `6b00a5a5…` |
| ippo_seed1 | step_00550000 `43f4f625…` | step_00500000 `19095537…` |
| ippo_seed2 | step_00250000 `ab628b98…` | same file |
| ippo_seed3 | step_00500000 `63214e5f…` | same file |

**Checks in `check` (`block_d.picks`).** The run is refused if any check fails.
1. `selections.json` has SHA-256 `8eaa148a6165eb34d51ac8e2e23994ad931a84f759cbb0558d753d41ceeb93e8`, and the SCB-C code hashes it records are current.
2. The new pick's hash equals the Block S manifest hash. For the four pairs, both hashes also equal the Block C manifest hashes.
3. The old pick is the step of `best.json` and satisfies all of:
   - `best.pt` is intact: its hash equals `best.json`;
   - its tensors (names, dtypes, shapes, values) are identical to the step file;
   - the step file's hash equals its sidecar's;
   - the sidecars agree on every field except file size and hash.
4. The pick changed for D1 and D3, and is unchanged for D2.

**During and after the run.**
- Every checkpoint is loaded the same way: the production loader, greedy mode, one PyTorch thread per worker.
- Every checkpoint hash is recomputed after the run and must be unchanged.
- Nothing is written to `results/training/` and nothing is moved into `models/`. That decision is deferred until after the results are reviewed.
- SCB-C v5 and every checkpoint identity stay frozen whatever Block D shows.

## 4. Outcomes

**Definitions.**
- Every outcome is a ratio of sums over seeds: θ = Σ_s Y_s / Σ_s X_s.
- `Y_s` and `X_s` are summed over the levels in scope.
- The effect is Δ = θ_new − θ_old.

| Family | Outcome | Y per episode | X per episode |
|---|---|---|---|
| **F1 primary** (A\*) | task completion | delivered | released |
| | conflicts per episode (absolute) | `conflicts_total` | 1 |
| **F2 supportive** (A\*) | conflicts per delivered task (pooled ratio) | `conflicts_total` | delivered |
| | conflicts per 100 robot-steps (all robot-steps, as `sim.metrics` counts them) | 100 · `conflicts_total` | robots · steps |
| | base return per episode | `base_return` | 1 |
| **F3 secondary** (RRT\*) | task completion; conflicts per episode | as F1 | as F1 |

**Rules for the outcomes.**
- The three conflict measures are always reported apart and never substituted for one another.
- The training-log denominator (non-idle robot-steps) is not used.
- F2 is supportive only. Its two rates re-express the same conflict counts, and base return contains the conflict penalties.

**Safety** (counts, not tested):
- physical collisions;
- `PhysicalCollisionError` and `InvalidTransitionError`;
- other exceptions;
- invalid moves (INV);
- unusable records (§7).

**Descriptive and exploratory** (no p-values, nominal intervals only):
- per-level paired effects;
- the F2 outcomes on RRT\*;
- each v5 candidate minus the matched baseline, using the same estimator with the baseline in the "old" position;
- point summaries per arm;
- episode classes;
- undelivered tasks;
- conflict types (RRv, RRs, ROv, ROs, INV);
- mean flow time;
- planner calls per 100 robot-steps;
- runtime: wall time, `act_s` and `prepare_s`, as median and IQR.

The mean over the four pairs used in Block C is not computed.

## 5. Statistical method

**Estimator and bootstrap (unchanged from the frozen Block C analysis).** `block_d_analysis.py` imports these functions from the frozen `block_c_analysis.py`, whose SHA-256 `3fe86a77…60ebfec` is pinned and checked before analysis:
- `paired_ratio`: the ratio-of-sums difference and its paired delta-method SE, `u_cs = (Y_cs − θ_c X_cs) / X̄_c`, `d_s = u_new,s − u_old,s`, `SE = sd(d, ddof = 1) / √n`;
- `bootstrap_t`: the symmetric seed-cluster bootstrap-t, with the SE recomputed in every resample;
- `boot_index`: one `(B, n)` index matrix per planner, with B = 10 000, drawn by `default_rng(0)` and shared by every arm of that planner, so old and new are always resampled together;
- `holm`.

**How the bootstrap is applied.**
- Each seed's three levels are summed before resampling. Levels are fixed strata.
- The quantile method is `higher`.
- p = (1 + #{|t\*| ≥ |t_obs|}) / (B + 1), two-sided, with Monte Carlo SE √(p(1−p)/B).

**Families, α and Holm.**

| Family | Hypotheses | Family-wise α | Holm thresholds (smallest p → largest) |
|---|---|---|---|
| F1 | 4 runs × 2 = 8 | **0.03** | 0.00375, 0.00429, 0.0050, 0.0060, 0.0075, 0.0100, 0.0150, 0.0300 |
| F2 | 4 × 3 = 12 | **0.01** | 0.00083, 0.00091, 0.0010, 0.00111, 0.00125, 0.00143, 0.00167, 0.0020, 0.0025, 0.00333, 0.0050, 0.0100 |
| F3 | 4 × 2 = 8 | **0.01** | 0.00125, 0.00143, 0.00167, 0.0020, 0.0025, 0.00333, 0.0050, 0.0100 |

- Across all three families the union bound gives a family-wise error of at most **0.03 + 0.01 + 0.01 = 0.05**, under any dependence.
- All hypotheses are two-sided: H₀: Δ = 0.
- Holm runs within each family. Ties are broken by listing order: runs mappo_seed1, mappo_seed2, mappo_seed3, ippo_seed1, then outcomes in table order.
- A hypothesis is rejected when its Holm-adjusted p ≤ its family's α.
- No equivalence margin is used.

**Intervals and decisions.**
- Every hypothesis gets the **nominal 95 % bootstrap-t interval** Δ̂ ± q₀.₉₅ · SE, plus the percentile interval as a sensitivity check. Both are labelled *nominal 95 %, per hypothesis, unadjusted for multiplicity*. They are not simultaneous family-wise intervals.
- **All inferential decisions are based solely on Holm-adjusted p-values.**
- Every Holm rejection has raw p ≤ 0.03, so its nominal interval excludes 0, up to resampling discreteness. The converse does not hold: a nominal interval that excludes 0 is not a rejection.
- No Bonferroni simultaneous intervals are added (decision 3).

**Monte Carlo sensitivity (decision 1).**
- B stays at 10 000 and is not raised automatically.
- An evaluable hypothesis is **MC-sensitive** when |p − its own Holm threshold| ≤ 2 · √(p(1−p)/B). Its own threshold is α_F / (m − rank), where rank is its position in that family's Holm ordering.
- The formal decision under the frozen procedure is reported, but an MC-sensitive result is always qualified, and no unqualified inferential claim is made from it.
- [M/P] At the smallest thresholds the Monte Carlo SE of p is about 0.00061 (F1), 0.00029 (F2) and 0.00035 (F3).
- The smallest attainable p is 1/10 001 ≈ 0.0001, below every threshold.

## 6. Projected precision and power under the revised α

**Basis of the projections.**
- [M] Paired SEs measured on Block C (n = 120).
- [A] Block D has the same per-seed variability as Block C.
- [P] SE_D = SE_C · √(120/200) for A\*.
- [P] Minimum detectable effect at 80 % power: MDE = (z₁₋α′/2 + z₀.₈) · SE_D, a normal approximation. The measured Block C bootstrap-t q₀.₉₅ was about 2.0.
- "Worst" uses α′ = α_F / m (the first Holm step); "best" uses α′ = α_F (the last step).
- The Block C point estimates are not used to size or alter the design.

**F1 (A\*, α = 0.03):**

| Run | Outcome | SE_C [M] | SE_D [P] | Nominal 95 % half-width [P] | 80 % MDE worst / best [P] |
|---|---|---|---|---|---|
| mappo_seed1 | completion | 0.0028 | 0.0021 | ±0.0042 | 0.0080 / 0.0064 |
| | conflicts / episode | 8.38 | 6.49 | ±12.7 | 24.3 / 19.5 |
| mappo_seed2 | completion | 0.0025 | 0.0019 | ±0.0038 | 0.0073 / 0.0058 |
| | conflicts / episode | 4.04 | 3.13 | ±6.1 | 11.7 / 9.4 |
| mappo_seed3 | completion | 0.0027 | 0.0021 | ±0.0041 | 0.0078 / 0.0063 |
| | conflicts / episode | 6.55 | 5.08 | ±10.0 | 19.0 / 15.3 |
| ippo_seed1 | completion | 0.0021 | 0.0016 | ±0.0032 | 0.0061 / 0.0049 |
| | conflicts / episode | 1.57 | 1.22 | ±2.4 | 4.6 / 3.7 |

**F2 (A\*, α = 0.01, 12 hypotheses):**

| Run | MDE worst–best [P]: conflicts / delivered | conflicts / 100 robot-steps | base return |
|---|---|---|---|
| mappo_seed1 | 1.45–1.19 | 1.53–1.25 | 0.79–0.65 |
| mappo_seed2 | 0.69–0.57 | 0.86–0.70 | 0.71–0.58 |
| mappo_seed3 | 1.14–0.93 | 1.25–1.02 | 0.59–0.48 |
| ippo_seed1 | 0.27–0.22 | 0.31–0.25 | 0.33–0.27 |

**F3 (RRT\*, α = 0.01, n = 60).**

- No RRT\* development evidence exists for the v5 picks, so no RRT\* paired SE has been measured.
- **[A] Central assumption:** the RRT\* paired SE equals the A\* paired SE scaled to 60 seeds, SE_C · √(120/60), a variability multiplier of 1.0.
- **Indicative sensitivity check only.** The per-seed SD of RRT\* relative to A\* for the Issue #6 picks was measured on Block B seeds 10 020–10 039 [M]. These ratios are:
  - 20 seeds and a **single arm**;
  - completion 0.47–1.10 (median 0.87);
  - conflicts 0.06–3.63 (median 1.19).

  They do **not** estimate the paired old/new RRT\* SE. They only show that the conflict variability on RRT\* is highly uncertain.

| Run | Completion: half-width / 80 % MDE worst–best [P] | Conflicts: half-width / 80 % MDE worst–best [P] |
|---|---|---|
| mappo_seed1 | ±0.0076 / 0.016–0.013 | ±23 / 48–41 |
| mappo_seed2 | ±0.0069 / 0.014–0.012 | ±11 / 23–20 |
| mappo_seed3 | ±0.0075 / 0.016–0.013 | ±18 / 38–32 |
| ippo_seed1 | ±0.0059 / 0.012–0.010 | ±4.4 / 9.1–7.6 |

**Reading the projections.**
- On A\*, the conflict tests can detect large differences.
- The completion tests can detect differences of only about 0.005–0.008.
- F3 is roughly 1.8× less precise, or more if RRT\* conflicts vary more.
- RRT\* results stay secondary and are limited to the four tested checkpoint pairs.

## 7. Incomplete, failed, missing and invalid episodes

**Episode classes** (`block_d_analysis.episode_class` / `episode_problem`).

| Class | Definition | Treatment |
|---|---|---|
| Successful | ran without exception; terminated with every released task delivered | used in every outcome |
| **Incomplete (time limit)** | ran without exception to `max_steps`, not terminated, with 1 or more tasks undelivered | **kept in every applicable outcome.** Undelivered tasks lower completion; conflicts and return count in full |
| Reached the horizon, all delivered | ran to `max_steps` with every task delivered | used in every outcome |
| Failed | an exception in the episode, including `PhysicalCollisionError` and `InvalidTransitionError` | unusable |
| Missing | no record after the run is complete | unusable |
| Duplicate | more than one record for a cell | unusable |
| Invalid | a required field missing or null; counts not non-negative integers; delivered > released; steps > `max_steps`; terminated with tasks undelivered; conflict breakdown not summing to `conflicts_total`; non-finite return; **neither terminated nor at the horizon** | unusable |

**Rules.**
1. **Nothing is dropped or imputed.** Every unusable episode is listed with its planner, run, arm, level, seed, class and exception type or reason.
2. **Unit of evaluability: run × planner.** A single unusable episode in either arm makes that run's hypotheses for that planner *not evaluable*:
   - for A\*: 2 in F1 and 3 in F2;
   - for RRT\*: 2 in F3.
3. **Holm treatment (p = 1, exactly):**
   - A not-evaluable hypothesis keeps its place in its family, so m is not reduced and the other thresholds are unchanged.
   - It enters `holm` with p = 1, so it is ranked last.
   - It is never rejected, and never flagged MC-sensitive.
4. **No substitute estimate (decision 2).** For a not-evaluable pair, no estimate, interval or p-value is computed or published anywhere:
   - not in the families;
   - not in the per-level results;
   - not in the RRT\* supportive outcomes;
   - not in the comparison with the baseline;
   - not in any supplementary appendix.

   Its row reads "not evaluable", with the list of unusable episodes.
5. **Descriptive units (D2, D4)** with an unusable episode are reported as counts only. Their summaries and their difference from the baseline are marked "data incomplete", with no estimates.
6. **Abort conditions.** The whole analysis is aborted and reported, and no re-run happens without approval, on any of:
   - a record from an unexpected run, checkpoint, controller, planner, level or seed;
   - a field that differs from the frozen condition;
   - a record's `arm` disagreeing with its checkpoint;
   - an instance hash differing from the pre-run check;
   - old and new differing in any paired field;
   - a checkpoint hash changing during the run;
   - a frozen-hash mismatch.
7. **Interruptions are not failures.** If the process dies, the deterministic run is resumed before any analysis, and every invocation is logged. No outcome is inspected before the run is complete and the replay has passed.
8. **Safety events** are always reported as safety failures, whether or not the run is evaluable.

## 8. Seed and scenario-generation safeguards

**Disclosure of the two earlier in-memory generations** [M].
- Block D scenarios **have been generated before**, in memory only, twice. Both times it was solely for disjointness checks:
  1. the static-level check in Issue 6A proposal revision 2 (§12);
  2. the L1 follow-up's `selection_block.check` (Block S against blocks A–D, all three levels).
- Only overlap counts were stored, all 0 (`results/dev6a/l1_selection/selection_block/seed_check.json`). No scenario content and no outcome was stored.
- No policy, baseline or planner was ever run on a Block D instance, and the Block C run did not touch Block D.
- A scan of 570 stored result files found no Block D seed field.

**Limitation.** These generations exposed no outcome, so they could not inform tuning. They do mean that "no Block D scenario has ever been generated" would be false, and it is never claimed.

**Safeguards for the formal execution.**
1. **Freeze first.** The protocol, runner and analysis are frozen and hashed **before the first scenario generation of the formal Block D execution.**
   - `freeze` refuses once `check.json` or `records.jsonl` exists.
   - `check` and `run` refuse unless the freeze hashes still match.
2. **Separate authorization flags.**
   - `freeze` needs `--authorized-freeze`.
   - `check`, `run` and `replay` need `--authorized-block-d`.
   - Running a command without its flag exits before doing anything.
3. **Seed guard** (`guard_task`), enforced **inside every worker process**, spawned or forked, before the scenario generator is called:
   - A\* accepts 12 000–12 199 only;
   - RRT\* accepts 12 000–12 059 only;
   - every other block label, and the training, A, B, C, S and test seeds, are rejected.

   There are three layers, all set up in the worker by the worker initializer and the task function:
   1. the runner's task function `_run` calls `guard_task` before it calls the sampler;
   2. the worker initializer wraps the production `run_episode`, so `guard_task` runs first even if `run_episode` is called directly;
   3. the worker initializer replaces the production `check_seeds`, which `run_episode` calls first, with a Block D-only check.

   The production source code is not changed; the replacements live only inside the worker processes.
4. **Tests never generate Block D scenarios.**
   - In the test process, an autouse sentinel makes the scenario sampler raise on any Block D seed.
   - Every check that hashes Block D instances runs on synthetic fingerprints.
   - Guard tests call the guards alone.
   - **Real worker processes.** An integration test starts a real process pool with every available start method (spawn on Windows; spawn and fork on Linux), using the runner's own initializer and then a probe sampler that raises and never generates a scenario. In the worker it shows that:
     - the guards are installed;
     - out-of-range tasks are rejected through `_run` and through `run_episode` called directly, before the sampler;
     - an in-range task reaches the probe sampler only after the guards.
5. **Pre-run check** (`check`), the first Block D generation of the formal execution:
   - scan `results/` (output folder excluded) for Block D seed fields;
   - verify checkpoint integrity (§3);
   - regenerate the A, B, C and S instance hashes and check that the regenerated C hashes equal those stored by the Block C check;
   - hash the 600 Block D instances (200 seeds × 3 levels; RRT\* uses the same instances of its 60 seeds);
   - check that the instances are unique per level and coincide with no A, B, C or S instance of that level.
6. **Disjointness from training scenarios: not verified** (correction C).
   - A read-only inspection found **no stored scenario-instance hashes for training**: the training logs record only the seed and a scenario name.
   - Training scenarios are not regenerated to make this check.
   - `check` therefore reports `training_disjointness: verified = false` with this reason. If training hashes were stored, it would compare them.
   - Scenario-level disjointness against training instances is **not** claimed. Training seeds (0–9 999) and Block D seeds are disjoint ranges, and the generator seeds each instance from its own seed.
7. **The test seeds (20 000–20 999) are never generated.**
8. **Pairing.**
   - Every arm of an instance shares every task field except the checkpoint or controller.
   - All records of one instance must carry the pre-run instance hash.
   - All 3 120 episode pairs of D1 and D3 are checked for identical paired fields.
9. **Interleaved order.** Tasks run seed by seed (then level, then planner), with every arm of an instance adjacent, so all arms run under the same CPU load.

## 9. Execution sequence and authorization gates (decision 5)

1. Protocol approval: done.
2. Implementation and tests without any Block D generation or evaluation: done.
3. Implementation artifacts presented for review.
4. **Separate explicit approval to freeze**, then `freeze --authorized-freeze`.
5. Pre-run check, `check --authorized-block-d`, under the freeze.
6. **Separate authorization to execute Block D**, then `run`, `replay` and `analyse`.
7. Report (a new §25 of the Issue 6A proposal), then stop for review.

## 10. Reproducibility and acceptance criteria

**Block D is accepted as complete only if all of the following hold:**
1. The frozen hashes match at `check`, `run`, `replay` and `analyse`.
2. The reused Block C analysis hash matches its pin, and the SCB-C code and `selections.json` match their recorded hashes at the freeze.
3. `check` passed.
4. All 8 220 records are present.
5. The checkpoint integrity checks pass, and every checkpoint hash is unchanged before and after the run.
6. Pairing is verified on all 3 120 episode pairs.
7. The replay is identical in every non-timing field: every 60th task, 137 episodes, re-run in fresh processes.
8. `analyse` has run once on the complete records, and a rerun gives a byte-identical `analysis.json`.
9. The hashes of every file under `results/` (output folder excluded) and `docs/` are unchanged before and after the run. This includes Issue #5 and #6, Block C, the L1 outputs and `docs/SESSION_HANDOFF.md`.
10. The manifest records the code hashes, the environment (Python, NumPy, PyTorch, platform), workers, wall time and every invocation.

**After the run.**
- A defect found after the run is reported, not fixed.
- A re-run needs the student's approval and is labelled as a re-run.
- SCB-C v5 and the checkpoint identities stay frozen whatever the outcome.

## 11. Reporting and interpretation

**Report** (a new §25):
- F1, F2 and F3 tables with estimates, nominal intervals, raw p, Holm-adjusted p, family α, own Holm threshold, decision and MC-sensitivity flag; "not evaluable" rows;
- a forest plot of Δ;
- per-level tables;
- episode-class, safety and conflict-type tables;
- D2 and D4 descriptives;
- runtime;
- reproducibility evidence;
- limitations.

**Per-pair reading** (F1, after Holm):

| Conflicts per episode | Completion | Reading |
|---|---|---|
| lower | no detectable difference | fewer conflicts; no completion difference detected. The nominal interval shows the range compatible with the data |
| lower | lower | fewer conflicts, with a detected completion cost (its size is reported) |
| lower | higher | the new pick is better on both outcomes |
| no detectable difference | any | no conflict difference detected |
| higher | any | unfavourable to the new pick on conflicts |

**Wording rules.**
- An MC-sensitive decision is always reported with that qualification.
- Not rejecting H₀ is not evidence of equality.
- F2 results are described as supportive.
- F3 results appear under the heading *"secondary; no RRT\* development evidence; 60 seeds; limited to the four tested pairs"*. They never support a claim that SCB-C is validated across planners.

## 12. Cost

| Item | Episodes | Basis | Wall time on 10 workers |
|---|---|---|---|
| D1 | 4 800 | [M] Block C: 2 880 PPO A\* episodes in 942 s | [P] ≈ 26 min |
| D2 | 1 200 | same | [P] ≈ 7 min |
| D3 | 1 440 | [M] Issue #6 PPO RRT\* records, mean 17.4–25.7 s per episode (worker count not recorded) | [P] ≈ 46–68 min |
| D4 | 780 | [M] Issue #6 matched-baseline records (A\* 0.37 s, RRT\* 8.3 s mean) | [P] ≈ 3 min |
| Check, replay, analysis | — | [M] the Block C equivalents took minutes | [P] ≈ 5–10 min |
| **Total** | **8 220** | | [P] ≈ **1.5–2 h** |

**How the cost was projected.**
- Formula: mean per-episode wall time × episodes ÷ workers × 1.1.
- Check on Block C: the formula predicts 953 s against the measured 942 s.
- [A] RRT\* timing is uncertain to about ±50 %.
- The machine must be otherwise idle.

## 13. Limitations

- **Four fixed checkpoint pairs.** The result is not a method-level effect, and the pairs are not replications.
- **The same pairs as Block C.** Block D re-tests those pairs on new instances, so expectations formed on Block C cannot be fully excluded. The hypotheses were fixed before Block C (Issue 6A proposal §12).
- **RRT\* is secondary.** There is no RRT\* development evidence, it uses only 60 seeds, its variability is an assumption, and the Block B ratios are indicative only.
- **No equivalence margin.** Completion results are bounded by intervals; they are not claims of "no cost".
- **Ceilings and tails.** Completion is near its ceiling, and conflicts are heavy-tailed.
- **One configuration.** Greedy mode, auction allocator, shield off.
- **Intervals and MC sensitivity.** Nominal intervals are not simultaneous. Holm decisions at B = 10 000 can be MC-sensitive near a threshold.
- **Earlier Block D generations.** Two in-memory Block D generations happened earlier (§8).
- **Training disjointness.** Scenario-level disjointness from training instances is not verified (§8).
- **Block D is single-use.** A future L2 would need a new reserved block.
- **Runtime figures** depend on hardware and load.

## 14. Decisions recorded (student, 2026-11-21)

1. B = 10 000 is kept, with the MC-sensitivity flag. B is not raised automatically. MC-sensitive results are qualified.
2. For not-evaluable pairs, no complete-case estimates, intervals or p-values are published anywhere.
3. No supplementary Bonferroni simultaneous intervals. Nominal 95 % intervals are labelled unadjusted, and decisions use Holm-adjusted p.
4. The freeze precedes the first scenario generation of the formal execution, and the two earlier in-memory generations stay disclosed.
5. The sequence of §9.
6. Scope D1–D4. O1, O2 and O4 are excluded, D4 is descriptive only, and H-L1 is the only hypothesis.
7. α: F1 0.03, F2 0.01, F3 0.01.
8. Moving checkpoints into `models/` is deferred until after the results are reviewed.
