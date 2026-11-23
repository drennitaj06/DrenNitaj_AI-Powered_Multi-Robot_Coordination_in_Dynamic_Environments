# Issue 6A Phase 4: MADQL optimization (MADQL-v2)

*Write order:*
- **§1–3** were written after the diagnostics (§2) and before any candidate was trained. §3 fixes the candidates, the training protocol and the development selection rule.
- **§4** (training and the selection) was added after training.
- **§5** (the Block C protocol) was written after the selection and before any Block C episode.
- **§6 onward** (results) were added after the Block C run.

Labels: **[M]** measured; **[C]** follows from the code or the mathematics; **[I]** interpretation; **[H]** hypothesis; **[A]** taken from an approved document.

The forward system configuration is unchanged throughout:
- Execution Supervisor R + S + D, with the frozen Stage 3a parameters and O off;
- P1 repair off;
- P2 off (the full 1 000-iteration RRT\* budget).

---

## 1. Motivation and approved scope

### 1.1 What Phase 4 was meant to do [A]

| Source | Text |
|---|---|
| Roadmap, proposal §27 | "Phase 4: MADQL optimization — depends on Phases 3, 1 and 2, and on any remaining evidence of learning weaknesses. Conditional" |
| Proposal §7 | "**Retraining:** a new configuration 'MADQL-v2', only if approved. Historical MADQL stays the reference." "I1 (no training): Q-value gaps on stored block-A states." |
| Proposal §19 C.4 | "Whether P4 (reservations) or any retraining (obs-v2, MADQL-v2) is justified" (deferred until Block C results exist) |
| Proposal §26.7.3 | "MADQL retraining is not yet justified. A small learned STAY preference may be addressable at execution time. Retraining is reconsidered in Phase 4 after the supervisor results." |
| Post-Issue-6 analysis, I2 | "MADQL fixes: would **n-step returns, a slower target update, prioritised replay or VDN (deferred)** stabilise it?" |
| Post-Issue-6 analysis, B6 | "MADQL learning instability: ±0.25 validation swings; 0.08–0.11 below PPO even when sampled" |

**The approved Phase 4 scope [A]:**
- Decide, on the remaining evidence, whether a retrained **MADQL-v2** is justified.
- If it is, train it as a new configuration, with the stabilisation methods listed in I2.
- Evaluate it against the historical MADQL, which stays the reference.

**Approved and unchanged:**
- the MADQL algorithm family: parameter-shared independent Double DQN with masking;
- the observation obs-v1, the five actions and their semantics, and the action mask;
- the reward of model §7 (weights, shaping and γ);
- the curriculum, the training budget and the checkpoint rule;
- PPO, allocation, the planners, the supervisor, P1 and P2.

**Not Phase 4:**
- **P4** of the proposal (space-time reservations) is a planning-interface item, not "Phase 4".
- **obs-v2** (intent sharing or a stall feature) is a separate approval item (§19 C.4).
- **VDN** is explicitly deferred in I2.

### 1.2 Remaining MADQL weakness after Phases 3, 1 and 2 [M]

With the forward configuration (Block C, Phase 1 `rsd_r0` and Phase 2 `full` records):

| | MADQL | IPPO | MAPPO |
|---|---|---|---|
| Completion, A\* (Phase 3 R_S_D) | 0.9970 | 0.9985 | 0.9982 |
| Completion, RRT\* | 0.9914 | 1.0000 | 0.9993 |
| Mean flow time, delivered (RRT\*) | 100.7 | 61.4 | 60.0 |
| Wait ratio (RRT\*) | 0.155 | 0.015 | 0.011 |
| Planner calls / 100 robot-steps (RRT\*) | 19.0 | 12.1 | 11.3 |
| Steps / episode (RRT\*) | 269.9 | 166.6 | 168.7 |
| Robot-steps changed by the supervisor (A\*) | ≈ 1 in 80 | ≈ 1 in 700 | ≈ 1 in 700 |
| Greedy completion without the supervisor (A\*, Phase 3 OFF) | 0.624 | 0.987 | 0.988 |

**[I] What this shows:**
- The supervisor turns MADQL into a near-complete controller.
- The policy itself is still weak and slow. It waits far more, replans about 1.6× as often per robot-step, and depends on recovery about 9× as often as PPO.
- Without the supervisor it completes 62 % of tasks.
- This is the "remaining evidence of learning weaknesses" the roadmap asks about.

## 2. Diagnostic evidence (development data; no training) [M unless marked]

**Tool:** `tools/phase4/diagnose.py`. Outputs are in `results/dev6a/phase4/diagnostics/`.

**Data:**
- Block B seeds 10 020–10 059 × 3 levels × the 3 historical MADQL picks (Issue #5 `best.pt`), A\*, greedy.
- Two conditions: **N**, no supervisor (the policy's own decisions), and **F**, R + S + D.
- 720 episodes, plus 2 × 120 PPO reference episodes (SCB-C v5 `mappo_seed1` and `ippo_seed1`, condition F), used only for their realised returns.
- The checkpoint stride sample for the churn analysis: 10 051 observations from N.
- Block B is the development / historical-reference block. No Block C, D or S seed and no test seed was used.

### 2.1 The STAY decision

| Condition N (999 980 non-idle robot-steps) | Value |
|---|---|
| Share of STAY decisions | 24.7 % |
| STAY with the suggested next cell **free** | 24.6 % of all steps |
| STAY with the suggested cell occupied (another robot) | **0.06 %** |
| Policy follows the planner's suggestion | 40.1 % |
| Gap Q(STAY) − best allowed move, when STAY is chosen | median **+0.038** (10–90 %: +0.004 … +0.196) |
| Gap, when a move is chosen | median −0.068 |
| Observation identical to the robot's previous one, when STAY is chosen | **83 %** (88 % in spells of ≥ 20 steps) |

**Findings:**
- MADQL's waiting is **not** congestion-driven: it almost never waits for an occupied cell.
- It is not tie-breaking either: Phase 0 found no exact ties.
- A waiting robot usually sees exactly the same observation again, so the greedy policy is at a fixed point [C].

### 2.2 Aliasing: waits that resolve vs waits that persist

**Spell classes** (STAY with a free suggested cell, condition N):
- resolved: the spell is shorter than 20 steps and ends in a move;
- persistent: the spell lasts 20 steps or more.

| STAY class | n | Median gap | p25 … p75 |
|---|---|---|---|
| Short spell, resolved by a move | 21 858 | +0.032 | +0.010 … +0.085 |
| Persistent spell (≥ 20 steps) | 223 737 | +0.039 | +0.012 … +0.110 |

**[M]** The Q-function assigns practically the same STAY preference to waits that end and waits that never end.

**[I]** obs-v1 has no clock, stall counter or memory, so the network cannot tell the two situations apart. This is perceptual aliasing; Phase 0 noted the same absence of information. Fixing aliasing *directly* needs an observation change (obs-v2). That is outside the approved Phase 4 scope (§1.1) and is recorded as an open question (§10).

### 2.3 Instability of the action ranking

**Churn:** greedy actions of the 10 051 sampled observations, evaluated under each stored checkpoint from 600 k to 1 M, 50 k steps apart (read-only).

| | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| Greedy action changes between consecutive checkpoints | 38–43 % | 37–44 % | 35–42 % |
| STAY-vs-move decision flips between consecutive checkpoints | 9–16 % | 11–17 % | 10–16 % |
| Median absolute change of the STAY gap per 50 k steps | 0.028–0.046 | 0.033–0.045 | 0.031–0.050 |

**Other measures:**
- Greedy actions agree across all three seeds at 1 M on only **31 %** of the sample.
- Block A validation completion still swings by up to 0.24 / 0.29 / 0.27 between consecutive validations after 500 k steps.
- Late training Huber loss is 0.0066–0.0069. If quadratic, that is a TD error of about **0.115 RMS**.

**[M]** Between consecutive checkpoints, the STAY gap moves by about as much as its own median value (0.03–0.05). The decisions that make MADQL slow sit inside the estimation noise.

### 2.4 Calibration: what the Q-function believes vs what the greedy policy achieves

Here G is the realised discounted return of the reward the learner optimises: model §7, team-mixed, with shaping and γ = 0.99. Only steps with an exact or nearly exact return are used: terminated episodes, or a remaining tail weight γ^(T−t) ≤ 0.134.

| | Q(s, a) of the chosen action, median | Realised G, median | Q − G, median |
|---|---|---|---|
| MADQL, N (no supervisor) | 2.99 | −0.16 | **+2.65** |
| MADQL, F (R + S + D) | 2.78 | 1.54 | **+1.14** |
| PPO references, F (same instances; completion 0.999) | — | 2.35–2.38 | — |

**[I] Interpretation:**
- MADQL's Q-values are about the realised value of good behaviour on these instances (PPO: 2.4).
- The greedy MADQL policy realises far less: 1.54 with recovery, −0.16 without.
- The values are not absurd. What is wrong is the action ranking in the states that matter: the greedy policy keeps choosing STAY in states where moving would realise the value the network expects.

### 2.5 Why one-step DQN corrects these states slowly [C]

**Setting:** a STAY that returns the same observation, the fixed point of §2.1. For such a transition, the one-step TD error of the action the greedy policy keeps choosing is

> δ = r + γ max Q(s, ·) − Q(s, STAY) = r − (1 − γ) Q(s, STAY),

using max Q(s, ·) = Q(s, STAY) whenever STAY is greedy.

**Size of the correction:**
- With the measured values r ≈ −0.008 (median team-mixed reward of a STAY step) and Q ≈ 3, this gives δ ≈ −0.04.
- That is the only learning signal pushing Q(s, STAY) down. It is about **a third of the typical TD noise** (0.115 RMS, §2.3), and about the size of the gap itself.

**With n-step returns,** for an n-step return over a spell that repeats the same state, the error becomes

> δ_n = r (1 + γ + … + γ^(n−1)) − (1 − γ^n) Q.

For n = 3 that is ≈ 2.97 × δ.

**Consequences:**
- The self-correction per update grows about **n-fold**.
- A delivery reward is propagated n steps per backup instead of one.
- n-step returns therefore target exactly the mechanism of §2.1–2.4. They do not remove aliasing (§2.2).

### 2.6 Causes: supported and not supported

| Candidate cause | Evidence | Verdict |
|---|---|---|
| Action-selection tie-breaking | 0 exact ties (Phase 0); gaps are positive, not zero | **not supported** |
| Congestion (waiting for an occupied cell) | 0.06 % of steps | **not supported** |
| Reward structure (STAY rewarded) | The STAY reward is negative (median −0.008). The shaping is potential-based, so the optimal policy is invariant (Ng et al. 1999) [C]. The Q-values match good-behaviour returns (§2.4) | **not the primary cause**; the reward is also out of scope |
| Value estimation: weak correction signal at self-loops plus TD noise ≫ decisive gaps | §2.3–2.5 | **supported** |
| Training instability, non-convergence at 1 M | churn, validation swings (B6), loss still rising | **supported** |
| Perceptual aliasing (no stall information in obs-v1) | §2.2 | **supported**, but its direct remedy (obs-v2) is outside scope |
| Coordination / non-stationarity of independent learners | 31 % cross-seed agreement is consistent with it; not separable from the above with this data | open |

## 3. Hypotheses, candidates, training protocol and selection rule (pre-declared)

### 3.1 Hypotheses [H]

- **H4.1 (n-step returns).** The learning signal at self-loop STAY transitions grows about n-fold, and reward propagates faster. Expected effects:
  - fewer persistent STAY spells with a free suggested cell;
  - higher greedy completion without the supervisor;
  - fewer supervisor interventions, less waiting and shorter flow times with R + S + D.
- **H4.2 (slower target update).** A target period of 8 000 instead of 2 000 environment steps (2 000 instead of 500 gradient updates) lowers the churn of the action ranking (§2.3). Expected effects: a more stable greedy policy, with the same downstream effects as H4.1.
- **Null.** Neither changes the decisive STAY decisions, because aliasing (§2.2) dominates. In that case, no candidate improves the raw policy.

### 3.2 Candidates (fixed before any training)

The design is a 2 × 2 factorial whose fourth cell is the historical MADQL (n = 1, period 2 000):

| Candidate | `n_step` | `target_period` (env steps) | Gradient updates between syncs |
|---|---|---|---|
| historical (reference) | 1 | 2 000 | 500 |
| **n3** | 3 | 2 000 | 500 |
| **t8** | 1 | 8 000 | 2 000 |
| **n3t8** | 3 | 8 000 | 2 000 |

**Why n = 3 and ×4:**
- n = 3 is the Rainbow default (Hessel et al. 2018). Larger n adds more off-policy bias under ε-greedy data.
- A ×4 slower sync brings the period into the usual DQN range relative to the replay ratio.
- Neither value is tuned on any data.

**Not implemented, and why:**
- **Prioritised replay.**
  - It reweights towards large-|δ| transitions, which here are mostly pickup, delivery and conflict events, not the small-gap STAY states of §2.5.
  - It adds two hyper-parameters and an importance-sampling bias.
  - No diagnostic points at it.
- **VDN.** Deferred by I2. It changes the algorithm (value decomposition).
- **obs-v2 / a stall feature.** A different observation and approval item. It would be the direct test of aliasing; see §10.
- **A STAY margin at execution, a STAY penalty, a lower γ.**
  - These change the greedy action semantics or the reward objective.
  - The supervisor's S already provides execution-time recovery.
- **More training steps.** Budget parity with the historical MADQL and with PPO (1 M) is part of the approved configuration.

### 3.3 Implementation [C]

| File | Change |
|---|---|
| `marl/madql.py` | `MADQLParams.n_step` (default 1; validated as an integer ≥ 1). Omitted from `to_dict()` at 1, so the Issue #5 configuration dictionary and hash are unchanged. For n > 1, `loss` uses `γ^h` per row in the Double-DQN target |
| `marl/replay.py` | `NStepAccumulator`: one queue per vectorised copy. Every one-step transition is emitted exactly once, in time order, as the n-step transition. At an episode end, the remaining steps are flushed with h = the number of steps left; d = 1 only for termination, and truncation bootstraps from the look-ahead observation. The buffer stores h (1 by default); `Batch.horizons` is optional |
| `marl/trainer.py` | Uses the accumulator only when `n_step` > 1. With `n_step` = 1 the loop is exactly the Issue #5 one |

Inference is unchanged: the same network and the `mlp-q-v1` checkpoint format. MADQL-v2 checkpoints therefore run through the existing `LearnedController` and evaluation without modification.

**Reproduction check [M]** (`tools/phase4/repro.py`): the historical configuration, run seed 1, current code, the first 50 000 environment steps.
- `step_00050000.pt` and `step_00050000.train.pt` are **byte-identical** to the stored Issue #5 files (SHA-256).
- The configuration and its hash are equal, and so is every training-log line.
- The 50 k validation summary, the selection key and every field of all 60 Issue #5 validation records are equal. The current evaluation only adds the Issue #6 fields.

**Tests** (`tests/unit/marl/test_madql_nstep.py`, `tests/integration/test_madql_nstep_training.py`; 19 pass):
- default and validation; the Issue #5 dictionary;
- the Hypothesis property test (150 examples): every step emitted once, in order, with the exact n-step return, horizon, next observation and termination flag;
- n = 1 equals the raw transitions; copies are independent;
- the tensor-γ target equals the scalar and manual ones; the n-step loss with h = 1 equals the one-step loss;
- an n-step learner refuses batches without horizons; buffer horizons;
- the ε-greedy property: never a masked action, and exact arg-max at ε = 0;
- explicit `n_step = 1` training byte-identical to the default;
- n-step training deterministic and different from n = 1;
- an n-step checkpoint through the production `run_episode`, with and without R + S + D: deterministic, 0 collisions, 0 invalid moves.

### 3.4 Training protocol

**Configuration:**
- `configs/training/madql.toml` unchanged except for the candidate's two fields: 1 M environment steps, 4 copies, sampler seed 0, the approved curriculum, A\*, auction, endpoint recovery on, resolver off, shield off, obs-v1, the model §7 reward.
- Validation worker processes 2 instead of 3. Results keep their order, so this does not affect learning.

**Seeds and runs:**
- Run seeds **1, 2, 3**, the historical seeds. Each candidate run starts from the same initial network, ε-greedy stream, replay stream and scenario order as its historical counterpart.
- 9 runs (3 candidates × 3 seeds), run concurrently, one torch thread each.

**Checkpoint rule (per run):** the historical rule, unchanged: block A, 60 greedy A\* episodes every 50 k steps, lexicographic (completion, then mean discounted base return) → `best.pt`.

**Provenance:**
- `results/dev6a/phase4/training/manifest.json`: every run's configuration, the code hashes and the environment. `train.py run` refuses to start if the code or configuration changed.
- Per run: `config.json`, `train_log.jsonl`, `validation.jsonl`, every step checkpoint, `best.pt` and `phase4_run.json` (including the `best.pt` SHA-256).

**Separation:**
- No historical checkpoint is read or written by training.
- Nothing is copied to `models/`.
- New models are labelled `madql-v2-<candidate>`.

### 3.5 Development selection rule (Block S; fixed before any training result was seen)

**Data:**
- Block S, seeds 13 000–13 079 × 3 levels: the L1 selection block, never used for any MADQL decision before.
- For each candidate (its 3 `best.pt`) and for the historical MADQL (its 3 `best.pt`), A\*, greedy:
  - condition **F** (R + S + D);
  - condition **N** (no supervisor; the Issue #6 composition).
- 4 × 3 × 240 × 2 = 5 760 episodes.
- Block S is used only for this selection, never for Block C conclusions.

**Score:** pooled completion Σ delivered / Σ released over the 3 policies.

**Eligibility.** A candidate is eligible if both hold:
1. F completion ≥ historical F completion − 0.005 (no material system-level loss).
2. N completion > historical N completion (the raw policy improves).

**Selection:**
- Among the eligible candidates, the one with the highest N completion is selected.
- If two are within 0.005, the one with the lower F mean flow time of delivered tasks wins.

**If no candidate is eligible:**
- The candidate with the highest N completion still goes to Block C, recorded as "not eligible at selection", so that the negative result is confirmed under the formal protocol.
- It cannot be recommended unless the formal verdict is "adopt".

**Reported, never used for selection:** MADQL behaviour metrics (STAY share, gaps, churn), conflicts and planner calls.

Block C is not touched before §5 is written.

## 4. Training results and development selection (written after training and Block S, before Block C)

### 4.1 Training [M]

- All 9 runs completed 1 000 000 environment steps with no error: 249 184 (n = 3) or 249 186 (n = 1) gradient updates, 3 466–4 263 training episodes.
- Wall time was 3.6–3.8 h per run, all 9 concurrently on the i7-1355U. Each run used one torch thread and 2 validation workers. The historical runs took 2.0–2.1 h; they are not directly comparable, because they ran under a different machine load.
- Every `phase4_run.json` records the `best.pt` SHA-256. `train.py run` verified the code and configuration against `manifest.json` before each run.
- The protocol SHA-256 at launch (`protocol_at_launch.json`) and at selection (`selection.json`) is the same: `4e38d24b…`, §1–3 unchanged.

**Block A validation** (60 greedy A\* episodes per checkpoint, the checkpoint rule; Block A is the validation block of Issue #5):

| Run | `best.pt` step | Best completion | Mean of the last 5 validations | Largest change between consecutive validations after 500 k |
|---|---|---|---|---|
| historical, seeds 1 / 2 / 3 | 800 k / 1 M / 700 k | 0.715 / 0.635 / 0.631 | 0.550 / 0.478 / 0.571 | 0.245 / 0.294 / 0.274 |
| **n3** | 600 k / 850 k / 950 k | 0.856 / 0.844 / 0.889 | 0.781 / 0.779 / 0.830 | 0.197 / 0.260 / 0.188 |
| t8 | 850 k / **50 k** / 700 k | 0.723 / 0.672 / 0.745 | 0.633 / 0.605 / 0.673 | 0.344 / 0.253 / 0.382 |
| n3t8 | 700 k / 950 k / 900 k | 0.871 / 0.832 / 0.864 | 0.734 / 0.810 / 0.824 | 0.226 / 0.136 / 0.127 |

**Training logs:**
- Late Huber loss (after 500 k): about 0.0067 for n = 1 and 0.023–0.025 for n = 3.
- [C] The two are not comparable: an n-step target sums n rewards and bootstraps from a value n steps away, so its variance is larger by construction. The loss is not used for any decision.
- t8 seed 2 never exceeded its first validation (0.672 at 50 k, when the target network was still close to its initialisation). Its `best.pt` is therefore an early, barely trained network. This follows from the unchanged checkpoint rule and was not altered.

### 4.2 Stability of the action ranking [M]

Same method as §2.3: the 10 051 diagnostic observations (Block B), the stored checkpoints 600 k … 1 M, 50 k apart, read-only. Values are the mean over the 8 consecutive pairs; per run in `selection_tables.md`.

| | historical | n3 | t8 | n3t8 |
|---|---|---|---|---|
| Greedy action changed | 0.39–0.40 | **0.33** | 0.37–0.39 | 0.32–0.33 |
| STAY-vs-move decision flipped | 0.12–0.14 | **0.05–0.07** | 0.08–0.11 | 0.05–0.07 |
| Median \|Δ gap\| | 0.034–0.038 | 0.038–0.040 | 0.030–0.034 | 0.032–0.039 |
| Share of the sample where STAY is greedy at 1 M | 0.08–0.16 | **0.024–0.028** | 0.07–0.10 | 0.014–0.058 |

**[I] Reading:**
- n-step returns roughly halve the STAY/move flips and move far fewer of these states to STAY.
- The magnitude of the gap changes is the same for all four. Gaps remain noisy, but in n3 / n3t8 they sit further from the decision boundary.
- The slower target alone (t8) reduces the flips only modestly and does not reduce the validation swings (§4.1); with n = 3 (n3t8) it lowers the late swings for two seeds.

**Development evidence on the hypotheses** (not a formal test):
- **H4.1:** supported.
- **H4.2 as a stand-alone change:** not supported.

### 4.3 Block S selection [M]

5 760 episodes (stage `4s`): 0 errors; replay of every 50th record (116) identical; 0 physical collisions, 0 invalid moves and 0 invalid reference paths in every arm.

| Candidate | F completion (R + S + D) | N completion (no supervisor) | F flow time | F ≥ hist − 0.005 | N > hist | Eligible |
|---|---|---|---|---|---|---|
| historical | 0.9976 | 0.6348 | 86.74 | — | — | — |
| **n3** | 0.9974 | **0.8610** | 70.00 | yes | yes | yes |
| t8 | 0.9974 | 0.7060 | 78.98 | yes | yes | yes |
| n3t8 | 0.9958 | 0.8424 | 71.24 | yes | yes | yes |

**Selection by rule §3.5:**
- All three candidates are eligible. **n3** has the highest N completion.
- No other candidate is within 0.005 of it, so the tie-break did not apply.
- **Selected: MADQL-v2 n3** (`n_step` 3, `target_period` 2 000), label `madql-v2-n3`.
- Recorded in `results/dev6a/phase4/select/selection.json` (records SHA `54194c66…`, protocol SHA `4e38d24b…`).

**Descriptive Block S observations** (reported, not used for the selection; full tables in `selection_tables.md`):

| | hist F | n3 F | hist N | n3 N |
|---|---|---|---|---|
| STAY share | 0.134 | 0.051 | 0.223 | 0.102 |
| Share of non-idle robot-steps in STAY spells ≥ 20 | 0.085 | 0.029 | 0.198 | 0.089 |
| Wait ratio | 0.129 | 0.049 | 0.219 | 0.095 |
| Steps / episode | 234.1 | 200.2 | 399.8 | 372.9 |
| Planner calls / 100 robot-steps | 21.3 | 19.3 | 34.1 | 23.9 |
| Supervisor changes / 100 robot-steps | 1.05 | 0.66 | — | — |
| Stuck robots / episode | 0.05 | 0.05 | 5.83 | 2.59 |
| Conflicts / episode (RRv, ROv) | 2.09 (0.89, 1.18) | 1.82 (0.19, 1.61) | 4.58 (3.36, 1.22) | 4.72 (2.51, 2.11) |
| Mean Q of the chosen action | 2.86 | 2.38 | 3.02 | 2.36 |

**Points carried into §5 (known before Block C, not used to change anything):**
- n3 has more robot–obstacle conflicts (ROv) per episode than the historical MADQL in both conditions, while its robot–robot conflicts fall. Conflicts per episode are a P4 veto outcome in §5.
- n3 seed 1 in N has 9.7 conflicts per episode (historical seed 1: 5.2).
- t8 seed 2 (the 50 k `best.pt`) produces 139.6 conflicts per episode in N, and t8 has 12.0 conflicts per episode in F. t8 was not selected.
- [I] n3's mean Q of the chosen action (≈ 2.37) is close to the PPO reference realised return (2.35–2.38, §2.4). The historical value is 2.9–3.0. This suggests less over-estimation; it was not tested separately.

## 5. Block C protocol (pre-registered; written after §4 and before any Block C episode)

### 5.1 Arms, conditions and episodes

**Arms:**
- **`hist`:** the historical MADQL, Issue #5 `best.pt` of seeds 1–3, the reference.
- **`v2`:** MADQL-v2 **n3**, the `best.pt` of its seeds 1–3, frozen by `selection.json` (SHA-256 recorded in `check_4c.json`).
  - `n_step` 3, `target_period` 2 000; everything else is the historical configuration (§3.4).
  - `best.pt` SHA-256: seed 1 `652f2997…`, seed 2 `bb05d379…`, seed 3 `bdf781b9…` (full values in `phase4_run.json`; re-verified by `check_4c.json` before and after the run).

**Pairing:**
- By instance: the same `(level, seed)` and instance fingerprint.
- By training seed: `v2` seed k is compared with `hist` seed k. Both share the initial network, the exploration stream and the scenario order.

| Condition | Planner | Composition | Seeds × levels | Episodes |
|---|---|---|---|---|
| **FA (primary)** | A\* | R + S + D, frozen Stage 3a parameters, O off | Block C 11 000–11 119 × {static, obstacles, full} | 2 × 3 × 360 = 2 160 |
| **FR** | RRT\* (full 1 000 iterations; P2 off) | R + S + D | Block C RRT\* subset 11 000–11 039 × 3 levels | 2 × 3 × 120 = 720 |
| **NA** | A\* | no supervisor, resolver off (the Issue #6 composition: the policy's own decisions) | Block C 11 000–11 119 × 3 levels | 2 160 |
| **Total** | | | | **5 040** |

**Shared settings:**
- greedy; auction; shield off; endpoint recovery on; `path_limit` 6; horizon 400;
- P1 repair off; P2 off;
- the production `run_episode`.

**Reuse:**
- The `hist` arms are re-run, because the observer collects the MADQL decision statistics, which no stored record has.
- G0.1 requires them to equal the stored Phase 1 records exactly:
  - FA = `records_1a` `rsd_r0`;
  - NA = `records_1a` `off_r0`;
  - FR = `records_1b` `rsd_r0`.

**Pre-registered expectations (H4.1, §3.1), stated before any Block C episode:**
- NA: higher completion (P1, P2), fewer stuck robots and fewer planner calls.
- FA and FR: completion not materially lower; lower flow time on common tasks, fewer planner calls and fewer supervisor changes (E).
- Risk known from Block S (§4.3): more robot–obstacle conflicts. This is tested by the P4 conflict veto and is not reinterpreted afterwards.

### 5.2 Outcomes and families

The methodology is Phase 1 §3.2 unchanged:
- the frozen `block_c_analysis.py` estimator, ratio of sums, seed cluster (the three levels and three policies of a seed together);
- B = 10 000, `default_rng(0)`, bootstrap-t;
- Holm within each family; the MC-sensitivity flag and the not-evaluable rule.

Δ = `v2` − `hist`.

| Family | Hypotheses per condition | α |
|---|---|---|
| **P1** | completion, MADQL family (3 policies pooled) | 0.03 |
| **P2** | completion per policy pair (seed 1, 2, 3) | 0.005 |
| **E** (efficiency and coordination, two-sided) | planner calls per 100 robot-steps; flow time of the tasks delivered in both arms; supervisor changes per 100 robot-steps (FA, FR) or stuck robots per episode (NA) | 0.01 |
| **P4** (vetoes) | conflicts per episode; high-priority flow time on commonly delivered tasks; high-priority undelivered tasks | 0.005 |

δ₁ = 1 / mean released tasks per `hist` episode of the condition (one task per episode).

**Descriptive, reported by condition, arm, policy and level (no decisions):**
- **Task:** completion, delivered / undelivered, successful episodes, high-priority completion, flow time, makespan, steps.
- **Coordination:** conflicts per episode / per delivered / per 100 robot-steps, conflict types, wait ratio, planner calls by trigger, deviations, supervisor changes, stuck robots by class (deadlock cycles, conflict loops, frozen, oscillating, …).
- **MADQL behaviour:** STAY share; STAY with a free or occupied suggested cell; self-loop STAYs; the share of robot-steps in STAY spells of ≥ 20 steps; following the suggestion; action shares; the gap Q(STAY) − best move (mean and histogram); mean Q of the chosen action.
- **Safety:** collisions, invalid moves, invalid reference paths, errors.
- **Runtime:** inference ms per step, planning and wall time per episode (medians).

Training stability and convergence (§4) are reported from the training logs and the churn analysis; Block C data are not used for them.

### 5.3 Integrity gates (G0)

| Gate | Requirement |
|---|---|
| G0.1 | `hist` = the Phase 1 records above, exactly (all non-timing fields): FA 1 080 / 1 080, NA 1 080 / 1 080, FR 360 / 360 |
| G0.2 | every 50th record replays identically in fresh processes without the observer |
| G0.3 | 0 errors, 0 physical collisions, 0 INV |
| G0.4–G0.6 | (FA, FR) 0 supervisor-introduced invalid moves; fairness and burst bounds; frozen parameters; mode R + S (O unused) with the resolver on |
| G0.7 | 0 invalid reference paths |
| G0.8 | repair off, P2 off, 1 000 RRT\* iterations; NA has no supervisor and the resolver off |
| — | instances identical across arms; every `v2` record carries the selected candidate; checkpoint hashes unchanged before and after; the code, selection and protocol gates hold |

### 5.4 Decision per condition

1. **"reject"** if G0 fails, or a completion loss (P1 or P2) is Holm-rejected, or a P4 increase is Holm-rejected.
2. **"inconclusive"** if the P1 lower bound of Δ completion is ≤ −δ₁.
3. **"inconclusive (mixed)"** if an E outcome is Holm-rejected in the harmful direction while there is also a benefit. **"no benefit (efficiency harm)"** if only harm is rejected.
4. **"adopt"** if there is a benefit: P1 completion gain Holm-rejected, or at least one E outcome Holm-rejected with Δ̂ < 0.
5. Otherwise **"no benefit"**.

**Ceiling rule (pre-registered; the lesson of Phase 2):**
- *If* the bootstrap-t interval of the P1 row is unbounded, the lower bound used in step 2 is the **smaller** of:
  - the frozen tool's stored percentile lower bound;
  - the loss-only bound −Σ_seeds max(0, delivered_hist − delivered_v2) / Σ released_hist, which ignores every gain.
- An unbounded interval never counts as a rejection, and never as evidence of equivalence beyond this rule.
- With a finite interval the rule does not apply.

**Recommendation:** MADQL-v2 **n3** becomes the forward MADQL candidate for Phase 6 **only if** all of the following hold:
- FA is "adopt";
- neither FR nor NA is "reject";
- G0.1 and the replay hold.

Otherwise the historical MADQL stays the forward MADQL. In every case MADQL-v2 stays a documented, separately labelled configuration.

### 5.5 Anti-tuning

- No Block C result may change the candidate, its checkpoints, the selection, the training, the supervisor configuration or its parameters.
- Block C is the development block. Its results are development evidence; confirmation belongs to Phase 6 (proposed block 14 000–14 199).

---

## 6. Block C results (written after the run) [M]

Full tables: `2026-11-22-issue-6a-phase4-madql-tables.md`, generated from `results/dev6a/phase4/eval/analysis.json`. Δ = `v2` (MADQL-v2 n3) − `hist`, ratio of sums, seed cluster.

### 6.1 Completeness and integrity (G0)

| Gate | Result |
|---|---|
| Records | 5 040 / 5 040, 0 errors, 0 duplicates, 0 unexpected |
| Protocol at check | SHA-256 `8405d73c…` = the document after §5 was appended (§1–5 unchanged since) |
| G0.1 `hist` = Phase 1 records | FA 1 080 / 1 080, NA 1 080 / 1 080, FR 360 / 360 identical |
| G0.2 replay (every 50th record, fresh processes, no observer) | 101 / 101 identical |
| G0.3 | 0 errors, 0 physical collisions, 0 INV in every condition and arm |
| G0.4–G0.6 (FA, FR) | 0 supervisor-introduced invalid moves; fairness and burst bounds hold; parameters equal the Stage 3a freeze; mode R + S with the resolver on, O unused |
| G0.7 | 0 invalid reference paths (FA 759 343, FR 253 633, NA 1 926 252 checked) |
| G0.8 | repair off; P2 off; every RRT\* call at 1 000 iterations; NA has no supervisor and the resolver off |
| Other | instances identical across arms; every `v2` record carries `n3`; the 6 checkpoint hashes are unchanged after the run |

### 6.2 Pre-registered families

| Condition | P1 completion: hist → v2 (Δ̂, 95 % CI, Holm) | E (all Holm-rejected, Δ̂ < 0) | P4 vetoes | Verdict |
|---|---|---|---|---|
| **FA** (A\*, R + S + D, primary) | 0.9970 → 0.9977 (+0.0008, [−0.0004, +0.0019], p 0.18, not rejected). Lower bound −0.0004 > −δ₁ = −0.052; finite bootstrap-t interval, so the ceiling rule did not apply | planner calls / 100 robot-steps 21.29 → 19.47 (−1.82); common flow time 85.9 → 69.3 (−16.6); supervisor changes / 100 robot-steps 1.04 → 0.62 (−0.42) | none rejected (conflicts / episode +0.64, [+0.13, +1.16], Holm p 0.028 > 0.005) | **adopt** (by E) |
| **FR** (RRT\*, R + S + D) | 0.9914 → 0.9968 (+0.0055, [+0.0025, +0.0084], Holm p 0.002, **rejected: gain**). P2: seed 2 +0.0133 rejected (MC-sensitive); seeds 1 and 3 not rejected | calls 18.98 → 17.89 (−1.08); common flow 100.6 → 78.8 (−21.9); supervisor changes 0.70 → 0.34 (−0.36) | none (conflicts −0.26, n.s.; high-priority undelivered +0.006, interval (−∞, ∞), not rejected) | **adopt** |
| **NA** (A\*, no supervisor) | 0.624 → 0.860 (+0.236, [+0.220, +0.251], Holm p 0.0001, **rejected: gain**). P2: +0.18 / +0.24 / +0.28, all rejected | calls 33.5 → 24.2 (−9.3); common flow 94.3 → 65.8 (−28.5); stuck robots / episode 5.76 → 2.55 (−3.21) | none (conflicts +1.09, [−0.70, +2.89], n.s.) | **adopt** |

**Holm-rejected P4 outcomes in the beneficial direction:**
- high-priority flow time on commonly delivered tasks is lower in all three conditions: −7.6 (FA), −10.5 (FR), −14.7 (NA);
- high-priority undelivered tasks fall in NA (−0.87 per episode).

These are not vetoes; a veto is an increase.

**No completion loss is Holm-rejected anywhere.** The only nominal loss is FA policy seed 1: −0.0014, nominal CI [−0.0028, −0.0001], Holm p 0.11, not rejected. Seed 1 is also the policy whose `best.pt` was the earliest (600 k).

### 6.3 Task and coordination metrics (descriptive)

| | FA hist | FA v2 | FR hist | FR v2 | NA hist | NA v2 |
|---|---|---|---|---|---|---|
| Completion | 0.9970 | 0.9977 | 0.9914 | 0.9968 | 0.6240 | 0.8595 |
| Undelivered / episode | 0.058 | 0.044 | 0.167 | 0.061 | 7.22 | 2.70 |
| Successful episodes | 1 026 / 1 080 | 1 039 / 1 080 | 315 / 360 | 340 / 360 | 15 / 1 080 | 207 / 1 080 |
| High-priority completion | 0.9992 | 0.9997 | 1.0000 | 0.9991 (2 of 2 175 undelivered) | 0.778 | 0.919 |
| Flow time, delivered | 85.95 | 69.42 | 100.68 | 79.33 | 94.87 | 72.54 |
| Makespan (mean) | 223.3 | 189.2 | 251.3 | 208.5 | 288.6 | 236.9 |
| Steps / episode | 232.2 | 197.7 | 269.9 | 219.4 | 398.5 | 368.9 |
| Wait ratio | 0.125 | 0.047 | 0.155 | 0.060 | 0.226 | 0.090 |
| Conflicts / episode | 1.27 | 1.91 | 1.81 | 1.55 | 2.97 | 4.06 |
| Conflicts / delivered task | 0.066 | 0.100 | 0.094 | 0.080 | 0.248 | 0.246 |
| Planner calls / 100 robot-steps | 21.29 | 19.47 | 18.98 | 17.89 | 33.54 | 24.22 |
| Deviations / 100 robot-steps | 17.8 | 15.2 | 15.4 | 13.4 | 31.7 | 21.6 |
| Supervisor changes / 100 robot-steps | 1.04 | 0.62 | 0.70 | 0.34 | — | — |
| Stuck robots (total) | 63 | 47 | 60 | 22 | 6 219 | 2 753 |

**Per level** (NA completion): static 0.362 → 0.676, obstacles 0.762 → 0.929, full 0.702 → 0.931. In FA and FR the per-level differences are within ±0.003, except FR full (+0.010, nominal).

**Per seed** (NA, seeds with more deliveries / fewer / equal, of 120): 113 / 7 / 0, 117 / 3 / 0, 119 / 0 / 1.

**Planner triggers** (share of calls, FA; algorithms §4):
- v2 has more T1 calls (goal change: 0.136 vs 0.106), consistent with completing legs faster.
- It has more T2 calls (path invalidated by an obstacle or map event: 0.055 vs 0.045), consistent with §6.5.
- It has fewer T3 calls (deviation: 0.80 vs 0.84).

**Stuck robots by class** (NA):
- oscillating 3 443 → 1 999;
- frozen 1 391 → 413;
- other moving 1 174 → 226;
- deadlock cycles 60 → 72;
- blocked by an idle robot 5 → 18.

### 6.4 MADQL decision behaviour (descriptive)

| | FA hist | FA v2 | FR hist | FR v2 | NA hist | NA v2 |
|---|---|---|---|---|---|---|
| STAY share | 0.131 | 0.049 | 0.164 | 0.063 | 0.231 | 0.101 |
| STAY with the suggested cell free | 0.129 | 0.048 | 0.161 | 0.057 | 0.230 | 0.100 |
| STAY with the suggested cell occupied | 0.0022 | 0.0014 | 0.0014 | 0.0006 | 0.0017 | 0.0012 |
| Self-loop STAY (same observation) | 0.093 | 0.033 | 0.124 | 0.045 | 0.187 | 0.083 |
| Robot-steps in STAY spells ≥ 20 | 0.082 | 0.028 | 0.112 | 0.039 | 0.207 | 0.088 |
| Follows the planner's suggestion | 0.586 | 0.700 | 0.574 | 0.706 | 0.399 | 0.541 |
| Mean gap Q(STAY) − best move | −0.056 | −0.080 | −0.052 | −0.079 | −0.047 | −0.075 |
| Mean Q of the chosen action | 2.89 | 2.41 | 2.84 | 2.37 | 3.06 | 2.36 |
| Inference ms / step | 0.637 | 0.639 | 0.612 | 0.609 | 0.634 | 0.628 |

The four move actions are used almost equally by both arms (0.21–0.25 each). The decrease is entirely in STAY.

**Across seeds:** the historical policies differ strongly (FA STAY share 0.082 / 0.198 / 0.107). v2 is more uniform in FA and FR (0.062 / 0.060 / 0.023). The training-time churn (§4.2) is consistent with this.

### 6.5 Safety, including a post-hoc conflict analysis

**Physical safety [M]:** 0 physical collisions, 0 invalid moves, 0 invalid reference paths and 0 errors in all 5 040 episodes.

**Proposed conflicts.** The model's safety metric is the *proposed* conflicts that the arbiter prevents (model §3 and §9.2).
- The pre-registered P4 test of conflicts per episode was not Holm-rejected in any condition.
- The descriptive table, however, shows a rise in FA (1.27 → 1.91) and NA (2.97 → 4.06).
- The rise comes from **robot–obstacle vertex conflicts (ROv)**: FA 652 → 1 473, FR 264 → 490, NA 1 172 → 1 724.
- Robot–robot vertex conflicts fall in FA (708 → 567) and FR (385 → 48) and rise in NA (2 021 → 2 627).

Because this was not foreseen as a separate outcome, it was examined **after** the verdicts with a post-hoc tool (`tools/phase4/posthoc_conflicts.py`, output `posthoc_conflicts.json`). The tool uses the frozen estimator with nominal intervals and no multiplicity correction, and it is **never used for a verdict**:

| Post-hoc, nominal | FA | FR | NA |
|---|---|---|---|
| ROv / episode, hist → Δ̂ [95 % CI] | 0.60 → +0.76 [+0.32, +1.20] | 0.73 → +0.63 [+0.28, +0.98] | 1.09 → +0.51 [−0.25, +1.27] |
| ROv / 1 000 moves, hist → Δ̂ [95 % CI] | 0.57 → +0.87 [+0.40, +1.34] | 0.61 → +0.67 [+0.35, +0.99] | 0.51 → +0.48 [+0.17, +0.79] |
| RRv / 1 000 moves, hist → Δ̂ [95 % CI] | 0.62 → −0.06 [−0.54, +0.42] | 0.89 → −0.77 [−2.28, +0.74] | 0.89 → +0.63 [−0.39, +1.64] |

**Readings:**
- **[C]** A staying robot can never be in an ROv conflict (model §3). Fewer STAY decisions therefore raise the *exposure* to ROv.
- **[M]** Exposure does not explain the effect. v2 makes **fewer** moves in total (FA 1.02 M vs 1.15 M), because episodes are shorter. Yet its ROv rate *per move* is 1.9–2.5× the historical rate (FA 2.5×, FR 2.1×, NA 1.9×), with nominal intervals that exclude zero.

> **Clarification (2026-11-23; see `2026-11-23-issue-6a-review.md` §9.1).** The 1.9–2.5× above is correct for its post-hoc definition: ROv only, counted over all robots, divided by the MADQL policy's own non-STAY choices of **non-idle** robots only (source: `results/dev6a/phase4/eval/posthoc_conflicts.json`). Idle robots move often under MADQL-v2 (27–31 % of all moves; 7–15 % for the historical MADQL), so the numerator and denominator covered different robots. With the definition approved for Phase 6 (ROv + ROs over all robots, divided by every move reaching the arbiter), the same Block C instances give FA ×2.17, FR ×1.83 and NA ×1.43 (NA not significant) (`results/dev6a/phase6/dev_ro_exact/dev_ro_exact.json`). The text above is unchanged.

- **[I]** v2 is more willing to step into a cell that a dynamic obstacle enters in the same step. The historical STAY bias appears to have acted partly as implicit obstacle caution.
- **[H]** One possible cause: the n-step target spreads the one-step conflict penalty (c_ro = 1, model §7) over a return that also contains the next two steps' progress terms, so the immediate penalty weighs relatively less. This was not tested.
- ROv conflicts occur only on the obstacles and full levels (0 on static), as expected.
- In absolute terms, v2 proposes about 1.4 ROv conflicts per 1 000 moves (FA). All of them were prevented by the arbiter.

**This is the main cost of MADQL-v2.** It does not change the pre-registered verdicts (§5.5: no reinterpretation). §8 and §10 carry it forward.

### 6.6 Runtime and computational work (descriptive)

| | FA hist | FA v2 | FR hist | FR v2 | NA hist | NA v2 |
|---|---|---|---|---|---|---|
| Wall s / episode (median) | 0.95 | 0.78 | 43.1 | 32.5 | 1.83 | 1.61 |
| Planning s / episode (median) | 0.08 | 0.07 | 42.3 | 31.6 | 0.20 | 0.14 |
| Reference paths checked (the episodes' total path work) | 426 960 | 332 383 | 143 710 | 109 923 | 1 154 487 | 771 765 |

- Inference is unchanged (same network): 0.61–0.64 ms per step.
- With RRT\*, the shorter episodes and fewer planner calls cut the median episode time by about 25 %.
- **Training cost:** the same 1 M environment steps and the same number of gradient updates. The n-step bookkeeping is O(n) per transition and negligible next to the update. The wall times in §4.1 reflect 9 concurrent runs and are not a like-for-like comparison.

### 6.7 Position relative to PPO (descriptive; same Block C instances; PPO values from §1.2)

| | hist MADQL | MADQL-v2 n3 | IPPO | MAPPO |
|---|---|---|---|---|
| Completion, A\*, R + S + D | 0.9970 | 0.9977 | 0.9985 | 0.9982 |
| Completion, RRT\* | 0.9914 | 0.9968 | 1.0000 | 0.9993 |
| Flow time, RRT\* | 100.7 | 79.3 | 61.4 | 60.0 |
| Wait ratio, RRT\* | 0.155 | 0.060 | 0.015 | 0.011 |
| Steps / episode, RRT\* | 269.9 | 219.4 | 166.6 | 168.7 |
| Completion without the supervisor (A\*) | 0.624 | 0.860 | 0.987 | 0.988 |

**[I]** MADQL-v2 closes about half of the gap to PPO in flow time and steps (RRT\*), and about two thirds of the gap in raw completion. It remains clearly behind PPO, and it waits four to five times as much.

## 7. Mechanism and hypotheses (interpretation)

- **H4.1 (n-step returns) — supported by development evidence.**
  - Every predicted effect appears, in the predicted direction: fewer STAY spells with a free suggested cell, higher raw completion, fewer supervisor interventions, less waiting, and shorter flow times.
  - The mechanism-level measures move together. Self-loop STAYs fall by about 60 %, STAY/move flips during training about halve (§4.2), and the STAY gap moves further from zero.
  - This is consistent with §2.5: a larger correction signal at self-loop transitions.
- **H4.2 (slower target updates) — not supported as a stand-alone change** (Block S and the training logs only; t8 was not taken to Block C).
  - The churn reduction was modest.
  - The validation swings were not reduced.
  - One run's checkpoint rule selected a 50 k network.
  - Combined with n = 3 (n3t8) it was not better than n3 on the selection criterion.
- **Aliasing (§2.2) is reduced but not removed.**
  - v2 still chooses STAY on about 10 % of non-idle robot-steps without the supervisor, 8.8 % of them inside spells of 20 or more steps.
  - It still leaves 2 753 stuck robots in NA.
  - This is what §2.5 predicted: n-step returns strengthen the correction but do not give the network the missing information.
- **Calibration:** the mean Q of the chosen action falls from about 2.9 to about 2.4, close to the PPO reference realised return (§2.4). This is consistent with less over-estimation. It was not separately tested on v2 realised returns.
- **New side effect:** more robot–obstacle conflicts per move (§6.5). See [H] there.

## 8. Decision under the pre-registered rules

| Rule (§5.4) | Result |
|---|---|
| FA verdict | **adopt**: no harm, no veto, three E improvements; completion bound −0.0004 > −0.052 |
| FR verdict | **adopt**: completion gain and three E improvements |
| NA verdict | **adopt**: completion gain (+0.24) and three E improvements |
| G0.1 and the replay | pass |
| **Recommendation** | **MADQL-v2 n3 becomes the forward MADQL candidate for Phase 6** |

**What this does and does not mean:**
- Block C is the development block. This is development evidence, not confirmation. Phase 6 (proposed block 14 000–14 199) is where it is confirmed or not.
- **The historical MADQL stays the reference and is unchanged.** Its checkpoints, records and every earlier result remain valid and are not replaced. MADQL-v2 n3 is a separately labelled configuration (`madql-v2-n3`) with its own checkpoints in `results/dev6a/phase4/training/n3_seed{1,2,3}/`. Nothing was copied to `models/`.
- **Required condition for Phase 6, recommended for approval:** pre-register the robot–obstacle conflict rate (ROv per 1 000 moves, or proposed conflicts per episode) as a primary veto outcome for MADQL-v2. The Block C evidence for an increase is nominal and post-hoc (§6.5). It must be tested formally, not assumed away.
- Not decided here: whether the predictive shield (off in the approved configuration) should be part of a v2 configuration. That would be a separate, approval-dependent ablation.
- Nothing else in the forward configuration changes: R + S + D, O off, P1 off, P2 off, the Stage 3a parameters frozen.

## 9. Limitations and deviations

- **Development evidence only.** Block C has been used in earlier phases. The selection used Block S, not Block C, so the Block C estimates are not affected by selection optimism. Repeated use of Block C across phases is still a reason to confirm on a new block.
- **Three training seeds per configuration.** Training-seed variability is large for MADQL (§2.3, §4.1). P2 tests the three policy pairs separately.
- **The FA benefit comes from efficiency outcomes**, not from completion, which sits near the ceiling (0.997). The FA completion interval was finite, so the pre-registered ceiling rule was not needed.
- **FR high-priority undelivered:** v2 failed to deliver 2 of 2 175 high-priority tasks; hist failed to deliver none. The interval is (−∞, ∞) because the `hist` arm is at 0. By §5.4 an unbounded interval is neither a rejection nor evidence of equivalence. Reported as is.
- **FR P2 seed 2** (a completion gain) carries the MC-sensitivity flag.
- **The conflict analysis in §6.5 is post-hoc.** It is labelled as such, uses nominal intervals, and changes no verdict.
- **Deviations from the plan:**
  1. Validation used 2 worker processes instead of 3 (declared in §3.4; result order and values are unaffected).
  2. After the selection, a table header in `phase4_report.py` was fixed: an unescaped `|` in the churn table. The regenerated `selection_tables.md` has the same SHA-256 as the first generation (`ee750df1`), and no number changed.
  3. `tools/phase4/posthoc_conflicts.py` was added after the verdicts (§6.5).
  4. In stage 4c the RRT\* condition ran first. This affects only wall-time order, not results.
- **Not done** (outside scope or not justified by the evidence): prioritised replay, VDN, obs-v2, a STAY margin or penalty, a different γ, longer training, retraining PPO.

## 10. Open questions

1. **Aliasing.** A stall feature or intent sharing (obs-v2) would be the direct test of §2.2. It is a separate approval item (proposal §19 C.4).
2. **Robot–obstacle conflicts.** What causes the higher ROv rate per move (§6.5 [H])? A Block B replay of the ROv events, recording the obstacle's prediction and the robot's Q-values, would answer it without new training.
3. **Shield ablation.** Would the predictive shield remove the ROv increase at a small cost? It needs approval, because the shield is off in the approved configuration.
4. **t8 and the checkpoint rule.** A 50 k network was selected by Block A for t8 seed 2. Whether the checkpoint rule should require a minimum training fraction is a question for any future training; it was not changed here.

## 11. Tests, lint and types [M]

| Check | Result |
|---|---|
| Full suite `tests/unit tests/tools tests/integration` (default markers) | **1 254 passed**, 1 skipped (POSIX-only `getrusage` branch), 57 deselected (slow markers), 0 failed |
| New MADQL tests (`test_madql_nstep.py`, `test_madql_nstep_training.py`) | 19 passed, including the Hypothesis property test (150 examples) |
| Phase 4 tool tests (`tests/tools/test_phase4.py`) | 16 passed (design, guards, observer neutrality, spell logic, selection rule, ceiling rule, families, post-hoc outcomes) |
| Historical reproduction (`repro.py`) | byte-identical 50 k checkpoint, log and validation |
| PPO / MADQL regression | the existing MADQL and PPO tests pass unchanged; G0.1 reproduces 2 520 stored Phase 1 MADQL records exactly |
| `ruff check`, `ruff format --check` (`src`, `tools/phase4`, the new tests) | clean |
| `mypy src` | no issues (84 files) |
| `mypy` on `tools/selection`, `tools/phase1–4` and the new tests | clean, except the pre-existing `tests/unit/marl/test_madql.py:123` (not changed in Phase 4) |
| Weakened or removed tests | none |

## 12. Provenance and integrity

| Artifact | SHA-256 (prefix) |
|---|---|
| `diagnostics/diagnostics.json`, `churn.json`, `episodes.json`, `reference.json`, `states.npz` | `cb8f8132`, `f920386b`, `c25ae83a`, `d4218d79`, `e3c6ce26` |
| `repro/madql_seed1/repro.json` | `77ff3c5a` |
| `training/manifest.json`, `protocol_at_launch.json` | `0468af8c`, `8a05ed53` |
| n3 `best.pt`, seeds 1 / 2 / 3 | `652f2997`, `bb05d379`, `bdf781b9` |
| t8 `best.pt`, seeds 1 / 2 / 3 | `ee52d87f`, `69a248c6`, `9f8f2c01` |
| n3t8 `best.pt`, seeds 1 / 2 / 3 | `a7e8abed`, `fb343c04`, `ef1ee4b1` |
| `select/check_4s.json`, `records_4s.jsonl`, `replay_4s.json` | `bbab62e2`, `54194c66`, `cdb74d39` |
| `select/selection.json`, `selection_report.json`, `selection_tables.md` | `3fb6a2e4`, `ad519a62`, `ee750df1` |
| `eval/check_4c.json`, `records_4c.jsonl`, `replay_4c.json` | `e9df6aa8`, `e66e90c8`, `e24bff4a` |
| `eval/analysis.json`, `tables.md`, `posthoc_conflicts.json` | `276f8dd1`, `e6124ffc`, `9921699f` |
| This document: at training launch (§1–3) / at the Block C check (§1–5) | `4e38d24b` / `8405d73c` |

All paths are relative to `results/dev6a/phase4/`. `results/` is git-ignored, so the checkpoints (580 MB) and records stay local.

**Reproduction** (with `PYTHONPATH="src;tools/selection;tools/phase3;tools/phase1;tools/phase2;tools/phase4"`):

```
python tools/phase4/diagnose.py {run,churn,reference,report}
python tools/phase4/repro.py
python tools/phase4/train.py manifest --out results/dev6a/phase4/training
python tools/phase4/train.py run --out results/dev6a/phase4/training --candidate <c> --seed <k>   # 9 runs
python tools/phase4/phase4.py {check,run,replay} --out results/dev6a/phase4/select --stage 4s --workers 10
python tools/phase4/select_candidate.py --out results/dev6a/phase4/select
python tools/phase4/phase4_report.py select --out results/dev6a/phase4/select
python tools/phase4/phase4.py {check,run,replay} --out results/dev6a/phase4/eval --stage 4c --workers 10
python tools/phase4/phase4_analysis.py --out results/dev6a/phase4/eval
python tools/phase4/phase4_report.py eval --out results/dev6a/phase4/eval
python tools/phase4/posthoc_conflicts.py --out results/dev6a/phase4/eval
```

**Final integrity check [M]:**

| Check | Result |
|---|---|
| Block C / Block D `plan_freeze.json` file hashes (3 + 35 entries) | all match |
| 1 310 historical files hashed before Phase 6A.0 (training runs and their checkpoints, `dev5`, `dev6`, experiments, pilot, figures, `SESSION_HANDOFF.md`) | 0 changed, 0 missing |
| Frozen statistics `block_c_analysis.py` | `3fe86a77` (unchanged) |
| Supervisor `supervisor.py` | `83b84b8e` (unchanged) |
| Stage 3a `freeze.json` | `4c7b9dbc` (unchanged) |
| Phase 3 records 3b / 3c / 3e, `analysis.json`, `check.json`, `replay.json`, `tables.md` | `a6ed0179`, `09db9fe7`, `41ef734a`, `62180f3f`, `4890381c`, `db992ef5`, `65e89784` (unchanged) |
| Phase 1 `check.json`, `records_1a`, `records_1b`, `replay.json`, `analysis.json`, `tables.md` | `c1276332`, `8c4c6c31`, `60621df0`, `513635fe`, `24e53297`, `6fbb263d` (unchanged) |
| Phase 2 `selection.json`, `check.json`, `records_2b`, `records_2l`, `replay.json`, `analysis.json`, `tables.md` | `224fc343`, `ad101150`, `6f0e1f7a`, `83b04d00`, `e02956a4`, `97a83148`, `7bdaa8db` (unchanged) |
| Historical MADQL `best.pt` (used in Block S and Block C) | match `best.json` before and after both runs |
| `models/` | does not exist |
| `docs/SESSION_HANDOFF.md` | untouched (last modified 2026-11-19) |

## 13. Status

**Phase 4 complete.**
- MADQL-v2 n3 (n-step returns, n = 3) is implemented, tested, trained (3 seeds) and evaluated under a pre-registered protocol.
- The verdict is **adopt** in all three conditions, so it is the recommended forward MADQL candidate for Phase 6.
- The historical MADQL stays the reference.
- The rise in robot–obstacle conflicts per move (§6.5, post-hoc) must be a pre-registered veto in Phase 6.
- Development evidence only.

**Phase 5 (A\* optimization, P3 exact heuristic): not justified by the current evidence.** Recommendation: do not run it as a full phase. At most, a microbenchmark if runtime becomes a goal.
- **Size of the possible benefit:** A\* planning is 7.6–11 % of the episode wall time in every A\* condition here (FA 0.076–0.079, NA 0.096–0.109). Even an infinitely fast A\* would save at most that share.
- **No path-quality benefit:** A\* already returns optimal paths, and P3 keeps the optimal costs (proposal §6, P3 / F8).
- **No observed bottleneck:** no Phase 3, 1, 2 or 4 result points at A\* as a cause of failures, conflicts or delays. The remaining MADQL weakness is in the policy (§7).
- **A cost:** a new heuristic can change tie-breaking among equal-cost paths. The learned policies' observations would then change, so equivalence would have to be re-established.
