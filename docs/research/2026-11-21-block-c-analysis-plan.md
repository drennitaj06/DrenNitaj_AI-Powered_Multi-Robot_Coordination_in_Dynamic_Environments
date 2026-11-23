# Block C development evaluation — frozen configuration and analysis plan

*Issue 6A, approved 2026-11-21. Written and hashed before any Block C episode was run.
Hashes of this file, `tools/selection/block_c.py` and `tools/selection/block_c_analysis.py` are
recorded by `block_c.py freeze` in `results/dev6a/block_c/plan_freeze.json`, and `block_c.py run`
refuses to start if any of them has changed. Results are reported in the Issue 6A proposal §24,
not in this file.*

## 1. Purpose and status of the evidence

The comparison is between the Issue #6 checkpoint pick ("old") and the frozen SCB-C v5 pick
("new", proposal §23) for each PPO run whose pick changed. Block C is the **development /
reusable** block. Its results are development evidence about the trade-off SCB-C v5 makes. They
are not a final, independent validation; that role belongs to Block D, which stays untouched.

**SCB-C v5 stays frozen whatever Block C shows.** No Block C result may change the selector, its
parameters or its picks. If the results raise a concern, it is reported for the user's
decision. It is never fixed by re-selecting on Block C and then presenting Block C as
validation.

## 2. Evaluation configuration (frozen)

| Item | Value |
|---|---|
| Runs | `mappo_seed1`, `mappo_seed2`, `mappo_seed3`, `ippo_seed1` (the four whose pick changed) |
| Old pick (Issue #6 `best.pt`) | 200 k, 300 k, 150 k, 550 k env steps |
| New pick (SCB-C v5, `selections.json`) | 300 k, 500 k, 500 k, 500 k env steps |
| Not evaluated | `ippo_seed2`, `ippo_seed3` (unchanged picks), all MADQL runs, RRT\*, every other controller |
| Seeds | Block C, 11 000–11 119 (120 seeds) |
| Levels | `static`, `obstacles`, `full` |
| Episodes | 4 runs × 2 arms × 3 levels × 120 seeds = **2 880** |
| Episode runner | production `robocoord.marl.evaluation.run_episode`, unchanged |
| Condition | controller `ppo`, planner A\*, greedy mode, shield off, auction allocator, replicate 0, deadlock resolver off, endpoint recovery on, `path_limit` 6 (as in Issue #6 and blocks A and S) |
| Seed guard | the workers replace `check_seeds` with a guard that accepts Block C only and rejects Block D and the test seeds explicitly |
| Checkpoints | loaded read-only; SHA-256 recorded before and after the run. The old arm loads the step checkpoint of the Issue #6 pick, accepted only if (i) the Issue #6 `best.pt` is intact (its SHA-256 equals `best.json`), (ii) `best.pt` holds bit-identical tensors (names, dtypes, shapes, values), and (iii) `best.json` and the step's sidecar agree on every field except file size and hash. The bytes of the two files differ only because the archive's internal folder name differs (`best/` vs `step_…/`). Both arms therefore load files of the same format |
| Failures | an exception in an episode is recorded (`status = "error"`, type and message); the run continues |
| Training | none; no checkpoint or production file is written |

**Pairing.** For each run, level and seed, the old and new episodes are the same task except for
the checkpoint. The scenario is `LEVELS[level].sample(seed)`, and greedy PPO and the A\* planner
are deterministic, so both arms face the same instance. Each record carries `instance_sha256`, a
hash of the instance computed in the worker with its name and seed excluded. Before the run,
`check` computes the 360 Block C hashes independently and verifies:

- no stored result file uses a Block C seed;
- the 120 instances are unique per level;
- no instance coincides with any instance of blocks A, B or S.

Block D is not passed to the scenario generator.

## 3. Statistical unit and data structure

The **scenario seed** is the independent unit. A seed carries six episodes per run (three levels,
two arms) and stays whole in every resample, so the pairing across arms and across levels is
preserved. Runs share seeds. One resampling matrix is used for every run, which keeps the
dependence between runs when they are combined.

## 4. Outcomes

For arm `c` and seed `s`, `Y_cs` and `X_cs` are sums over the levels in scope. Every outcome is
the ratio of sums `θ_c = Σ_s Y_cs / Σ_s X_cs`. The effect is `Δ = θ_new − θ_old`.

| Outcome | `Y` per episode | `X` per episode | Family |
|---|---|---|---|
| Task completion (pooled) | delivered | released | **primary** |
| Conflicts per episode (absolute) | `conflicts_total` | 1 | **primary** |
| Conflicts per delivered task | `conflicts_total` | delivered | secondary |
| Conflicts per 100 robot-steps (`sim.metrics` denominator: all robot-steps, idle included) | 100 · `conflicts_total` | robots · steps | secondary |
| Base return per episode | `base_return` | 1 | secondary |

The three conflict measures are different quantities and are never merged or substituted for
one another:

- *Per episode* counts proposed conflicts per instance.
- *Per delivered task* divides by the work done. It is a pooled ratio, not the mean of the
  per-episode `conflicts_per_delivered` field, which is undefined when nothing was delivered.
- *Per 100 robot-steps* divides by the episode length times the team size, so a policy that
  finishes sooner has a smaller denominator.

The training-log rate, whose denominator is non-idle robot-steps, is not used.

## 5. Inference

**Standard error.** The paired delta-method SE is computed as follows:

- `u_cs = (Y_cs − θ_c X_cs) / X̄_c`
- `d_s = u_new,s − u_old,s`
- `SE = sd(d, ddof = 1) / √n`

For outcomes with `X = 1` this reduces to the ordinary paired SE.

**Seed-cluster bootstrap-t (symmetric).**

- Resampling: `B = 10 000` resamples of the `n = 120` seeds with replacement, drawn by
  `numpy.random.default_rng(0).integers(0, n, (B, n))`. This one matrix is shared by every run,
  outcome and level.
- Per resample: `t*_b = (Δ*_b − Δ̂) / SE*_b`, with `SE*` recomputed in the resample.
- Conventions: `0/0 := 0` and `x/0 := ±∞`. An SE or a difference `Δ* − Δ̂` with absolute
  value ≤ 10⁻¹² counts as 0. This only absorbs floating-point residue; it is not a tolerance on
  effects.
- Confidence interval: `Δ̂ ± q · SE`, where `q` is the 95 % quantile of `|t*|` (NumPy method
  `higher`).
- Two-sided p-value: `p = (1 + #{|t*_b| ≥ |t_obs|}) / (B + 1)`, where `t_obs = Δ̂ / SE`.
- Monte Carlo SE of `p`: `√(p(1−p)/B)`, reported with each p-value.
- Sensitivity: the percentile interval of `Δ*` is reported too. The bootstrap-t interval is the
  primary one.

This is the studentized seed bootstrap that the L1 null study found valid in every simulated
world, including skewed, sparse conflicts. Sign-flip and label-permutation tests are not used
because they failed on the conflicts (proposal §22).

**Effect sizes.**

- `Δ̂` in natural units: completion fraction, conflicts per episode, and so on.
- The relative change `Δ̂ / θ_old`.
- The standardised paired effect `d_z = Δ̂ / (SE · √n)`.

## 6. Hypotheses and multiplicity

All hypotheses are two-sided, `H₀: Δ = 0`.

| Family | Hypotheses | Correction |
|---|---|---|
| **Primary** | 4 runs × {completion, conflicts per episode} = 8 | Holm step-down, family-wise α = 0.05 |
| **Secondary** | 4 runs × {conflicts per delivered task, conflicts per 100 robot-steps, base return} = 12 | Holm step-down, family-wise α = 0.05, separately |
| Descriptive | per level (4 runs × 3 levels × 5 outcomes), the mean of the four run effects (5 outcomes), the safety and validity tables, the conflict-type breakdown, mean flow time | none; unadjusted 95 % CIs, no p-values |

Holm ties are broken by position, in the order runs × outcomes listed above. Adjusted p-values
are monotone and capped at 1. A hypothesis is rejected when its Holm-adjusted p ≤ 0.05. Claims
from the secondary family are supportive only. The primary family alone controls the
family-wise error at 0.05; across both families the bound is 0.10.

A non-rejected hypothesis is **not** evidence of equality. The confidence interval states which
differences are compatible with the data. No equivalence margin is used.

## 7. Validity gate (applied before any statistic)

1. **Configuration.** Every record must be one of the eight run/checkpoint combinations, with
   the frozen condition (§2), a Block C seed and a known level. Otherwise the analysis is
   **aborted** and the cause reported.
2. **Instance.** Every record's `instance_sha256` must equal the pre-run hash for its level and
   seed. Old and new must agree on every paired field: the condition, run, level, seed, number
   of robots, `max_steps` and the instance hash. Any mismatch aborts the analysis.
3. **Completeness and validity.** Each run × arm × level × seed must have exactly one completed
   episode with valid counts:
   - non-negative integers;
   - delivered ≤ released;
   - steps ≤ `max_steps`;
   - a terminated episode has no undelivered task;
   - the conflict breakdown sums to `conflicts_total`;
   - the return is finite.

   A run with any missing, duplicated, failed or invalid episode is **not evaluable**. Its
   primary and secondary hypotheses enter Holm with `p = 1` (never rejected). Its estimates are
   computed on the complete seeds (every one of the six episodes valid) and reported as
   descriptive only. Nothing is imputed.

## 8. Safety and validity reporting (every run × arm × level)

Each of the following is reported as a count:

- **Physical collisions:** the `physical_collisions` field. The engine also raises
  `PhysicalCollisionError`, so a collision also shows up as a failed episode of that type.
- **Invalid moves:** proposed moves into a blocked or invalid cell, the `INV` conflict type.
  They are cancelled by the arbiter.
- **Illegal task transitions:** the task state machine raises `InvalidTransitionError`, so these
  are counted as failed episodes of that type.
- **Other failures:** every other exception type, with its message.
- **Episode end:** terminated (every task delivered); reached the horizon with undelivered tasks
  (incomplete); reached the horizon with every task delivered. Termination before the horizon
  means all tasks were delivered. Any other early stop is an exception and is counted as a
  failure.
- **Missing records,** and records rejected by the gate.
- **Extras:** undelivered tasks, the conflict-type breakdown (RRv, RRs, ROv, ROs, INV), and the
  mean of the episodes' mean flow time.

**Pairing evidence:**

- the number of verified pairs out of 1 440;
- the pairs with identical instance hashes and identical paired fields;
- the pairs with an identical released-task count (expected, since arrivals do not depend on
  the policy).

## 9. Reproducibility

- `check.json` holds the seed scan, instance hashes, picks and checkpoint hashes, and the
  SCB-C v5 code-hash check.
- `plan_freeze.json` holds the hashes of this plan, the runner and the analysis module, taken
  before the run.
- `manifest.json` records the code hashes, the checkpoint hashes before and after, the
  environment and wall time.
- `replay`: every 60th task (48 episodes) is re-run in fresh processes. All non-timing fields
  must match the stored records.
- `analysis.json` stores the result together with the SHA-256 of `records.jsonl`. The analysis
  is deterministic given the records.

## 10. Pre-specified interpretation

For each run, the primary family gives a direction for each outcome after Holm: "lower",
"higher" or "no detectable difference". The run is then classified:

| Conflicts per episode | Completion | Reading |
|---|---|---|
| lower | no detectable difference | intended trade-off; no completion cost detected on Block C (the CI bounds the possible cost) |
| lower | lower | intended trade-off, with a detectable completion cost (its size is reported against the CI) |
| lower | higher | the new pick is better on both outcomes |
| no detectable difference | any | no conflict benefit detected on Block C |
| higher | any | unfavourable to the new pick on conflicts |

**Recommendation scope.**

- A safety failure of a new pick blocks a recommendation to carry the v5 picks forward. A
  safety failure means a collision, an illegal transition, any other exception, or a gate
  failure. It is reported for the user's decision.
- Otherwise, the recommendation summarises the trade-off on development data and its
  uncertainty.
- It does not propose changes to SCB-C v5. Block D remains the only data for a final claim.

**Expectation from the selection study (stated in advance, not a hypothesis).** Out of sample,
the one-SE rule had about half the conflicts per episode of completion-first selection, at about
0.0017 lower completion (proposal §22). The old picks are not exactly completion-first picks, so
this is only a rough guide.

## 10a. Limitations known in advance

- **Four runs and one fixed pair of checkpoints each.** The result describes these picks. It is
  not the expected behaviour of the selection rule over training runs.
- **Block C is reusable development data.** Results on it inform the next steps but are not
  confirmatory.
- **Completion is near its ceiling** (about 0.99). Differences of a few thousandths are near the
  resolution of 120 seeds × 3 levels.
- **One instance per seed and level, greedy mode only.** Stochastic-policy behaviour is not
  assessed.
