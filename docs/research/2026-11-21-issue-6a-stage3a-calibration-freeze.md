# Issue 6A Phase 3, Stage 3a: Execution Supervisor calibration and freeze

*Stage 3a, executed 2026-11-21. The configuration is calibrated on stored Block B episodes only and frozen in code (`robocoord.control.supervisor.FROZEN_PARAMS`). The go / no-go criteria of §6 are **recommended** and need the student's approval before Stage 3b. Stages 3b, 3c and 3e have **not** been run.*

Evidence labels: **[M]** measured on the Block B replay; **[C]** code-derived (follows from the implementation); **[H]** hypothesis or projection (not measured).

---

## 0. Summary

| | |
|---|---|
| Data | 2 700 stored greedy Block B episodes (2 160 A\*, 540 RRT\*), replayed with the supervisor **off**; **0 of 2 700** differ from their stored record in any compared field, and **0 of 2 160** in the A\* extension replay |
| Method | shadow-mode detectors (the implementation's own counters, never acting) on every robot-leg; the design §5 rule (≤ 1 % of legs in successful episodes trigger), per planner |
| Frozen, A\* | W = **24**, oscillation (K, cells, m) = **(14, 3, 7)**; H 8, Δ_prog 2, H_stay 3, L_max 3 unchanged; escalation off |
| Frozen, RRT\* | W = **32**, oscillation = **(8, 3, 4)**; the rest as A\* |
| Rejected | the design values W = 12, (8, 2, 4): under the approved rule they trigger on 2.0–2.1 % (A\*) and 5.6 % (RRT\*) of normal legs (S), and 1.4 % for IPPO A\* (O) |
| Protected | Block C/D artifacts, historical results, checkpoints, `docs/SESSION_HANDOFF.md`, Issues #7/#8: unchanged (hashes verified, §9) |

---

## 1. Scope and data boundary

**Used (Block B, seeds 10 020–10 099, development data on which calibration is allowed):**

| Source | Filter | Episodes |
|---|---|---|
| `results/dev6/stochastic/records.jsonl` | `mode == greedy`, `controller ∈ {madql, ppo}` (A\*) | 2 160 (9 policies × 3 levels × 80 seeds) |
| `results/dev5/comparison/records.jsonl` | `controller == madql`, `planner == rrtstar` | 180 (3 × 3 × 20, seeds 10 020–10 039) |
| `results/dev6/ppo/records.jsonl` | `planner == rrtstar` | 360 (6 × 3 × 20, seeds 10 020–10 039) |
| `results/dev6a/6a0_diagnostics/replay_episodes.jsonl` | the Phase 6A.0 stuck-robot classes of the failed A\* episodes | labels only |

**Not used:** Blocks C, D and S, the test block, any Block C or Block D result or episode. No new episode was generated (the replays reproduce stored episodes), nothing was trained, no checkpoint was changed or copied.

---

## 2. Consistency check (before calibration)

The design (§4, §5, §14), the proposal (§26–28), `supervisor.py`, its 61 tests and the stored data were checked against each other:
- the implemented trigger, excuse, termination and R-key rules match design §14.2;
- `alpha` (0.01) and `sensing_radius` (3) equal the allocation's and the baseline's;
- the resolver is off for learned controllers unless `EvalTask.deadlock` enables it (§14.1);
- the stored records carry `rrt_iterations` 1 000, as `EvalTask`'s default.

**No defect blocking calibration was found.** One limitation surfaced (§8, R1): the stored Block B episodes of four PPO runs were produced by their Issue #6 checkpoints, not the SCB-C v5 picks named for Phase 3 (design D5).

---

## 3. Methodology

### 3.1 Replay and shadow detectors [C]

`tools/diagnostics/stage3a_calibration.py`:
- **Replay.** Each stored episode is replayed with the production path (`evaluation._system`, `Engine` with shield off and `path_limit` 6, the stored `best.pt`) and the **supervisor off**. Ten fields are compared with the record: delivered, released, steps, terminated, conflicts by type, conflicts total, physical collisions, wait ratio, planner calls and mean flow time. **[M] 0 / 2 700 mismatches.**
- **Shadow.** An `ExecutionSupervisor("S+O")` with unreachable thresholds sees every step. Its output is asserted equal to the policy's proposal, so it never acts. After each step its per-robot leg, best D̄, last-progress time and **counted** stall counter (the implementation's own excused-step rule) are read.
- **Oscillation.** The supervisor's rule (`is_oscillating` over the last K steps, and no progress for ≥ K steps) is applied to the same per-leg positions for every grid pattern.
- **Right of way.** `right_of_way` with the baseline priority π̃ is applied to the proposals, which gives the would-be R losses per step.
- **Per robot-leg** (a constant (task, mode, goal) interval), the tool records:
  - the first offset at which each W and each (K, cells, m) would fire;
  - the would-be modified actions in the following H steps (policy action ≠ movable filtered suggestion);
  - the longest would-be R-loss streak;
  - the remaining distance;
  - the outcome: completed, reassigned, or open at the end, with its 6A.0 class if stuck.

**Exactness of the shadow.** Up to the first changed action, the ON trajectory equals the OFF one. So the first-trigger offsets, and whether the first burst would change any action, are **exact** for an ON run. Everything after the first changed action is not observable from OFF data (§3.5).

### 3.2 Selection rules

**As approved (design §5, fixed before the data were analysed):**
- **Reference population:** every robot-leg of a **successful** episode, per controller family × planner.
- **W:** the smallest grid value with at most 1 % of reference legs triggering S in every group.
- **Oscillation pattern:** the same 1 % target. Among the feasible patterns, the highest detection on stuck oscillating robots; ties → smaller K, larger m, fewer cells.
- **Sample size:** a group needs ≥ 100 legs.

**Revisions after the first look at the data, before any value was frozen.** These are disclosed as such. The literal-rule results are kept in `calibration.json` (`literal_rule`).

1. **Eligibility: success must be typical.** A reference group also needs successful episodes to make up at least half of its episodes.
   - *Reason [M]:* MADQL succeeds in 10 / 720 A\* episodes and 2 / 180 RRT\* episodes. Its few successful episodes contain the very stalls S targets: the counted-run 99th percentile is 162 steps, and 5.7 % of legs still trigger at W = 64.
   - Under the literal rule **no** W in the grid is admissible for A\* (`literal_rule.astar.stall_window = null`). The design's premise, that successful episodes show normal behaviour, holds for PPO (84–88 % successful) but not for MADQL.
   - MADQL's figures are reported, not used for selection.
2. **Per planner.**
   - The design's 3a population was A\* only (design §10: "2 160 replays"); RRT\* was added for robustness.
   - *[M]* At the same W, RRT\* triggers about 2.5× as often (W = 24: A\* 0.86–0.87 %, RRT\* 1.51–1.63 %). *[C]* RRT\* paths are not shortest paths, so following them produces counted no-progress steps.
   - One value for both planners would force W = 32 on A\* and slow every A\* recovery for no A\* benefit.
3. **O must be faster than S.** A pattern qualifies only if K < W; otherwise S fires first and O adds nothing [C].
4. **Extended O grid for A\*.**
   - The first grid (K ≤ 12) had no feasible pattern for IPPO A\*: the best reached 1.03 %.
   - K ∈ {14, 16, 20} was added (`O_GRID_EXT`) and replayed for the 2 160 A\* episodes. [M] 0 mismatches.

**Stability:** an episode-cluster bootstrap, 2 000 replicates with seed 20 261 121, resamples the reference episodes within each group and re-applies the rules.

### 3.3 Grids

- **W:** {4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64}.
- **Oscillation patterns:** K ∈ {4, 6, 8, 10, 12} (+ {14, 16, 20} for A\*); cells ∈ {2, 3}; m ∈ {⌈K/2⌉, ⌈3K/4⌉}.
- **H for the would-modify count:** {4, 8, 12, 16}.

### 3.4 Reference groups [M]

| Group | Episodes | Successful | Share | Reference legs | Eligible |
|---|---|---|---|---|---|
| IPPO A\* | 720 | 602 | 0.84 | 25 421 | yes |
| MAPPO A\* | 720 | 620 | 0.86 | 25 941 | yes |
| MADQL A\* | 720 | 10 | 0.014 | 371 | no |
| IPPO RRT\* | 180 | 150 | 0.83 | 6 428 | yes |
| MAPPO RRT\* | 180 | 159 | 0.88 | 6 668 | yes |
| MADQL RRT\* | 180 | 2 | 0.011 | 68 | no (also < 100 legs) |

Stuck A\* robots with a 6A.0 class:

| Class | Robots |
|---|---|
| frozen | 1 079 |
| oscillating | 2 413 (2 401 with a counted leg) |
| other_moving | 829 |
| conflict_loop | 98 |
| deadlock_cycle | 62 |
| mostly_stationary | 96 |
| blocked_by_idle | 9 |

### 3.5 What OFF data cannot show

The following exist only with the supervisor ON, so 3a cannot calibrate them:
- how the policy behaves after a recovery: whether it stalls again, and how quickly;
- recovery termination: hold, progress, stay limit and cycle;
- the effect of R decisions on later steps;
- detour membership: the resolver is off for learned controllers in the stored data.

The parameters that govern these keep their approved values unless the descriptive evidence contradicts them (§4). **No Block C/D or ON data were used to fill this gap.**

---

## 4. Frozen values and evidence

### 4.1 Stall window W

| | A\* | RRT\* |
|---|---|---|
| **Frozen** | **24** | **32** |
| Trigger rate on reference legs at the frozen W [M] | IPPO 0.86 %, MAPPO 0.87 % | IPPO 0.87 %, MAPPO 0.88 % |
| Next smaller grid value [M] | W = 20: 1.09 % / 1.09 % (fails) | W = 24: 1.51 % / 1.63 % (fails) |
| Design value W = 12 [M] | 2.14 % / 2.02 % (fails) | 5.58 % / 5.61 % (fails) |
| Bootstrap selection [M] | 24 in 89.2 %; 32 in 7.8 %; 20 in 3.1 % | 32 in 67.2 %; 48 in 32.9 % |
| Variant: count only triggers that would change an action within H = 8 [M] | W = 20 | W = 32 |
| Per policy, at the frozen W [M] | 0.61–1.13 % | 0.51–1.45 % |

**Detection of stuck A\* robots** [M]. At W = 24 the counted stall reaches W on:

| Class | Detected |
|---|---|
| frozen | 100 % |
| oscillating | 99.5 % |
| deadlock_cycle | 100 % |
| conflict_loop | 92.9 % |
| other_moving | 83.5 % |

Detection is almost flat in W (frozen still 99.9 % at W = 64), because stuck legs are long: their median counted run is 329 steps. **W therefore trades only false triggers against latency.** At W = 24, the median first trigger comes 42 leg-steps after the leg starts (frozen), with a median of 312 steps left in the episode. At W = 12 these are 29 and 326.

**Expected effect on interventions:**
- [M] A first S trigger on 0.86–0.87 % of normal A\* legs (RRT\*: 0.87–0.88 %).
- [C] The worst-case per-leg override share of repeated S bursts is H / (H + W): 8/32 = 25 % for A\* and 8/40 = 20 % for RRT\*. The design's value was 8/20 = 40 %.
- [H] Frozen MADQL robots get recovered more slowly than at W = 12: one burst per about 26 instead of 14 counted steps.

**Uncertainty:**
- The A\* choice is stable.
- The RRT\* choice is not: a third of the bootstrap replicates select 48. The RRT\* reference is 309 successful episodes on 20 seeds.
- Under RRT\*, the detection of stuck robots is **not measured** (the 6A.0 labels exist for A\* only).

**Alternatives rejected:**
- W = 12 (design): it violates the approved rule 2–5×.
- W = 20 for A\* (the would-change variant): this variant is a post-hoc rule; the frozen value follows the approved trigger definition.
- One W = 32 for both planners: no A\* rule requires it.
- W = 48 for RRT\*: the smallest admissible value is the rule.

### 4.2 Oscillation (osc_window K, osc_cells, osc_moves m)

| | A\* | RRT\* |
|---|---|---|
| **Frozen** | **(14, 3, 7)** | **(8, 3, 4)** |
| Trigger rate on reference legs [M] | IPPO 0.96 %, MAPPO 0.71 % | IPPO 0.34 %, MAPPO 0.31 % |
| Design value (8, 2, 4) [M] | IPPO 1.41 % (fails), MAPPO 0.95 % | 0.34 % / 0.31 % (feasible) |
| Bootstrap [M] | (14, 3, 7) 62.3 %; (16, 3, 8) 26.6 %; (20, 3, 10) 5.8 %; (12, 3, 6) 5.3 % | (8, 3, 4) 71.2 %; (10, 3, 5) 28.9 % |

**Detection on stuck oscillating A\* robots** [M]:
- (14, 3, 7): 99.9 %. Every feasible pattern detects 99.3–99.9 %, so the tie-break order decided the choice.
- O also fires on 15 % of frozen robots, 72 % of other_moving robots and 8 % of conflict-loop robots. All of these are stuck, so these triggers are not false.

**Lead over S [M]:** on stuck oscillating A\* robots, (14, 3, 7) fires a median of **10 steps before** S at W = 24 (q90 10, q99 41).

**Expected effect:**
- [M] A first trigger by S or O on **1.25 %** (IPPO) and **1.21 %** (MAPPO) of normal A\* legs, and on 1.6 % under RRT\*. Each mechanism alone stays ≤ 1 %.
- [C] An O recovery can re-trigger after K steps of fresh history, so the worst-case per-leg override share of repeated bursts is **H / (H + min(W, K))**:

  | | Value |
  |---|---|
  | A\* | 8/22 = 36 % |
  | RRT\* | 8/16 = 50 % |
  | Design values | 8/16 = 50 % |

  The D11 bound is therefore tightened for A\* and unchanged for RRT\*. (§14 of the design stated H / (H + W); with O enabled, the binding term is min(W, K).)

**Uncertainty:**
- IPPO A\* sits at 0.96 %, near the target.
- The A\* pattern is moderately stable (62 % bootstrap).
- Under RRT\*, detection was taken from the A\* labels (a proxy).

**Alternatives rejected:**
- (8, 2, 4) (design): it fails for IPPO A\*.
- Larger K for A\*: feasible but slower. The rule takes the most detecting, then smallest K.
- Disabling O for A\*: a feasible pattern with a 10-step lead exists.
- Patterns with K ≥ W: these are redundant.

### 4.3 Recovery hold H, progress Δ_prog, stay limit H_stay (kept: 8, 2, 3)

- **Not calibratable from OFF data [C].** These parameters act only after a recovery has started.
- **Evidence for H = 8:**
  - [M] The first burst on a normal leg would change a mean of 3.7 (IPPO A\*) and 3.4 (MAPPO A\*) actions within 8 steps at W = 24.
  - [M] On MADQL legs it would change 4.4 actions.
  - So H = 8 rarely binds on normal legs; it bounds the worst case.
  - The design's H grid {4, 8} called for "sensitivity analysis on Block C". **That is no longer admissible for tuning (§7);** any H sensitivity in 3c is descriptive only.
- **Δ_prog = 2:**
  - [H] Burst projection: assume the robot freezes again right after each recovery, and each recovery step gains one cell. With W = 24, Δ_prog = 2 would cover the remaining distance within the remaining time for **94 %** of frozen and **95 %** of oscillating stuck A\* robots, against 99 % at Δ_prog = 4.
  - [M] The remaining distance has a median of 1–2 cells and a q90 of 15.
  - A larger Δ_prog would hold control longer on every recovery, raising overrides on robots that resume by themselves. The 4–5 point gain is a projection under an optimistic assumption, so it does not justify a change before ON data exist.
- **H_stay = 3:** [M] the filtered suggestion is STAY on only 1.6–2.4 % of reference robot-steps. A recovery that meets 3 consecutive STAYs is blocked, and ending it then is the design's intent. There is no evidence for any other value.

### 4.4 Right-of-way fairness L_max (kept: 3) and priority (kept: π̃, α = 0.01)

- [M] Would-be R contests are rare: 0.05–0.86 losing robot-steps per 100 robot-steps (MADQL A\* 0.05, MAPPO A\* 0.86).
- [M] On reference legs, the longest would-be loss streak is 0 at the 99th percentile in every group. The 99.9th percentile is 4 (IPPO A\*) and 23–30 (MAPPO A\*/RRT\*).
- [M] Conflict-loop robots: IPPO's have a median streak of 162 (max 378). This is the case L_max exists for.
- [C] With L_max = 3, priority decides every contest of ≤ 3 steps (≥ 99 % of normal legs never reach one). In a persistent contest, the low-priority robot wins one step in four: no starvation, and the bound holds by construction (property test).
- A larger L_max would delay fairness in exactly the conflict loops R targets. A smaller one would let the low-priority robot override priority after one or two losses. The OFF data cannot measure the ON contest dynamics. **Kept; uncertainty [H].**
- π̃ and α = 0.01 are tied to the allocation (code-derived consistency) and are not tunable.

### 4.5 Deadlock resolver (kept: `DetourParams` defaults)

The stored learned episodes run without the resolver, so its parameters cannot be calibrated here [C]. [M] The deadlock_cycle class has 62 robots. All are detected by S within W, and their would-be loss streak is ≈ 0, as expected: R leaves swaps to the arbiter. Unchanged.

### 4.6 Other parameters that affect interventions

| Parameter | Value | Status |
|---|---|---|
| `escalate` | **False** | disabled, not a formal arm (instruction); pinned by test |
| `repeat_window` | 4 | inert while escalation is off |
| `sensing_radius` | 3 | equals the baseline filter [C] |
| `alpha` | 0.01 | equals the allocation [C] |
| progress metric D̄ | static goal distance | design; under RRT\* it is the reason for the higher W (§3.2) |

---

## 5. Freeze location and reproducibility

**Frozen in code:** `src/robocoord/control/supervisor.py`, `FROZEN_PARAMS` / `frozen_params(planner)`.
- `evaluation.run_episode` uses it for every supervised run.
- `SupervisorParams()` keeps the design values for the scripted software tests.
- `tests/unit/control/test_supervisor.py::test_stage3a_frozen_parameters_are_pinned` fails if any value changes.
- The regression test checks that a supervised `run_episode` record carries exactly the frozen values.

**Freeze record:** `results/dev6a/phase3/stage3a/freeze.json`, SHA-256 `4c7b9dbc…866378fd`, written once. It holds the frozen parameters, the filters, seeds, episode counts, input and output hashes, checkpoint hashes, code hashes, environment and commands.

| Artifact | SHA-256 |
|---|---|
| `results/dev6a/phase3/stage3a/freeze.json` | `4c7b9dbccc9d1d87a115c3b98d535f7df1fb38b39bb8aad5dd4fa8a1866378fd` |
| `results/dev6a/phase3/stage3a/replay_episodes.jsonl` | `43c707cce945bcc626cf39609b6049f133c8988deb5db30cc80ff0c26f4b9dfd` |
| `results/dev6a/phase3/stage3a/replay_o_ext_astar.jsonl` | `47336ee3d00422fee0acc187a6e8c1f3711c9141b1b2784447ea7053f5385c82` |
| `results/dev6a/phase3/stage3a/calibration.json` | `25624de4e3c9b63465cb7b85fbb46032a4eac7263238a1ec45b1a0d63c65b7e4` |
| `results/dev6a/phase3/stage3a/inputs.json` | `c0edf8004f8a6dc4bab907a413eb5c1c3669d13a16ffb903efe6f8d9999db74c` |
| `tools/diagnostics/stage3a_calibration.py` | `9e0c708030f69f67ae435dc7f81a365197144874ca8320b9e17ed30f2f545f8f` |
| `src/robocoord/control/supervisor.py` (with the freeze) | `83b84b8e0efdc1e7bb532bb3f3f6b3a1e437105c66293dc5ad84e4a23b8ee33d` |
| `src/robocoord/marl/evaluation.py` (with the freeze) | `bc65a7bae058a4dd5d96c97eaa6cf4db3d563d8c19cff6e2cd3b34161eaf1f59` |
| `tools/diagnostics/replay_failures.py` (6A.0 classifier, for §6.4) | `1d994db5705361d80dd25f317435374f76d3380a02f3b83e9530dfb414424bf3` |
| `tools/selection/block_c_analysis.py` (frozen estimator, for §6.1) | `3fe86a7795885ea17b445e3a24a8503cada31a2fa2da2bd85203e9d4560ebfec` |
| Input `results/dev6/stochastic/records.jsonl` | `1d64866e5d78a6785f34cc8d7e0f5df9bfe4fcfa3dd681a98b52c070a36cbcd1` |
| Input `results/dev5/comparison/records.jsonl` | `53b4e114583b4811aeba21e0deddb3210a79400b8c481f3dd088fd2841318e4b` |
| Input `results/dev6/ppo/records.jsonl` | `5f43c2157a71c411092bc1898ca1854a44f05536fea4adb20474fcdf15425852` |
| Input `results/dev6a/6a0_diagnostics/replay_episodes.jsonl` | `c2d586fc6b3319e6980101e1d0c84db00c7e7b978dada0627b62d2fa4d089c82` |

**Checkpoints** (`results/training/<run>/best.pt`; full hashes in `freeze.json`):
- madql_seed1–3: `dd9f088c…`, `c66bd8c2…`, `492f2a61…`;
- ippo_seed1–3: `3706f6c6…`, `47d849e4…`, `d350e417…`;
- mappo_seed1–3: `e0979974…`, `9ebc8afd…`, `a0af8b8a…`.

All equal the Block D freeze's `best.pt` hashes.

**Reproduce** (repository root, `.venv`; outputs are deterministic):
```
python tools/diagnostics/stage3a_calibration.py replay       --out <new dir> --workers 11
python tools/diagnostics/stage3a_calibration.py replay-o-ext --out <new dir> --workers 11
python tools/diagnostics/stage3a_calibration.py analyse      --out <new dir>
```
- The replay takes about 55 min on 11 workers (the RRT\* episodes dominate); the extension about 10 min; the analysis about 3 s.
- The replay and analysis commands refuse to overwrite existing outputs.

**Code-state note.**
- During the main replay the additive monitoring counters (§6.7) were added to `supervisor.py`. They only count and never change an action. The shadow was asserted never to act, and both replays reproduced every stored record.
- `FROZEN_PARAMS` was added after both replays. Neither path is read by the shadow, which passes its own `SupervisorParams`.

---

## 6. Go / no-go criteria for Stages 3b, 3c and 3e (recommended; pre-registration needs approval)

These criteria are fixed **before** any Phase 3 evaluation episode exists. They use only quantities that 3b, 3c and 3e record anyway, the frozen statistical machinery of Blocks C/D, and no threshold that is not derived from the problem itself. Where no non-arbitrary threshold exists, the quantity is **reported, not gated**, and this is stated.

### 6.1 Common machinery (unchanged from the frozen Block C/D analysis)

- **Instances:**
  - 3b and 3c: Block C, seeds 11 000–11 119 × 3 levels (the design's development block, §10), 9 policies.
  - 3e: the design's 40-seed RRT\* subset, proposed as 11 000–11 039 × 3 levels (to be confirmed at 3e approval).
  - Block C serves here to *evaluate* a frozen configuration. Nothing measured there may change §4 (§7).
- **Estimator:**
  - Every outcome is a ratio of sums, θ = Σ_s Y_s / Σ_s X_s, with levels summed per seed (fixed strata).
  - The effect is Δ = θ_on − θ_off, **paired by instance** (same seed, level, policy and planner).
  - `paired_ratio`, `bootstrap_t`, `boot_index` and `holm` are imported from the frozen `block_c_analysis.py` (hash in §5). B = 10 000, `default_rng(0)`, one index matrix per planner shared by every arm.
- **Family pooling:** a family estimate (MADQL, IPPO, MAPPO) sums Y and X over the family's 3 policies per seed before resampling, so the seed remains the cluster.
- **Tests and intervals:**
  - two-sided bootstrap-t p-values, with Holm within each hypothesis family;
  - nominal 95 % bootstrap-t intervals, labelled *per hypothesis, unadjusted*;
  - decisions on Holm-adjusted p only;
  - the MC-sensitivity flag: |p − own threshold| ≤ 2·√(p(1−p)/B).
  - Not-evaluable hypotheses enter Holm with p = 1 (Block D §7).
- **Episode classes, unusable episodes, abort conditions:** as Block D §7. Nothing is dropped or imputed, and an exception is a safety failure.

**Hypothesis families and α per stage** (union bound ≤ 0.05 per stage):

| Family | Content | Hypotheses | α |
|---|---|---|---|
| P1 primary | completion, ON − OFF, per controller family | 3 | 0.03 |
| P2 per policy | completion, ON − OFF, per policy | 9 | 0.005 |
| P3 targeted | stuck-robot categories (§6.4), ON − OFF, per family | 4 × 3 = 12 | 0.01 |
| P4 veto | common-delivered high-priority flow time; undelivered high-priority tasks; per family | 2 × 3 = 6 | 0.005 |

- **3c** uses the same four families, but for each factor's **main effect** (§6.5) instead of ON − OFF.
- **3e** uses P1, P2 and P4 only. RRT\* is a robustness stage, and its categories are descriptive.

### 6.2 G0: integrity and safety (hard gates; any failure is "no-go", whatever the effect)

| Check | How it is computed | Required |
|---|---|---|
| G0.1 OFF reproduces production | every OFF episode equals the `run_episode` record with default fields, field by field (non-timing fields and order, as the Phase 3 regression tests) | 100 % |
| G0.2 Determinism | a full second run (or replay) of every arm gives identical records | 100 % |
| G0.3 Safety | physical collisions, `PhysicalCollisionError`, `InvalidTransitionError` and other exceptions, in every arm | 0 |
| G0.4 No introduced invalid move | INV conflicts per paired instance, ON minus OFF; every instance with ON > OFF is listed and traced to the policy or the supervisor | 0 attributable to the supervisor |
| G0.5 Fairness bound | `max_consecutive_r_losses` ≤ `fairness_losses` in every episode | 100 % |
| G0.6 Burst bound | per episode, `monitor.max_recoveries_in_a_leg` ≤ ⌈steps / (min(W, K) + 1)⌉: a new S recovery needs W counted steps, and a new O recovery K steps of fresh history | 100 % |
| G0.7 Frozen configuration | every record's `supervisor_stats.params` equals §4 for its planner (escalation `False`); code hashes equal §5 | 100 % |

### 6.3 G1: task execution (the primary criterion; 3b and 3e)

**Outcome:** completion = Σ delivered / Σ released (P1 per family, P2 per policy).

**Practical-significance unit (derived, not chosen).** An episode has 16 scheduled tasks, so the smallest change in an episode's result is one task. The unit is **δ₁ = 1 / (mean released tasks per OFF episode)** in completion (≈ 1/16 = 0.0625): one more (or one fewer) delivered task per episode, on average.

**3b is "go" only if all of the following hold:**
1. G0 passes.
2. **MADQL improves.** P1(MADQL) is Holm-rejected with Δ̂ > 0, **and** Δ̂ ≥ δ₁: at least one additional delivered task per episode, on average.
3. **No PPO family is harmed.** Neither P1(IPPO) nor P1(MAPPO) is Holm-rejected with Δ̂ < 0, and no P2 hypothesis of the 6 PPO policies is Holm-rejected with Δ̂ < 0.
4. **"No harm" must be interpretable.** For each PPO family the nominal 95 % lower bound of Δ is > −δ₁. If it is ≤ −δ₁, the result is **inconclusive**, not "go": a loss of one task per episode cannot be excluded.
5. **No veto (§6.6)** is triggered.
6. **Episode classes** (successful / incomplete per family) are reported. They are not gated separately, because completion contains them.

**3b outcomes:**
- **"No-go"** if condition 2 fails with Δ̂ ≤ 0, or if any harm in condition 3 is Holm-rejected.
- **"Inconclusive"** when any other condition fails.

**3e is "go"** if all of the following hold:
- G0 passes;
- no P1 or P2 hypothesis is Holm-rejected with Δ̂ < 0;
- the MADQL family has Δ̂ > 0;
- no veto is triggered.

A Holm-rejected MADQL improvement under RRT\* is reported but not required, because the RRT\* subset has a third of 3b's instances.

### 6.4 G2: targeted failure categories (3b P3; 3c attribution)

**Outcome per episode:** the number of robots of category k at the end of an incomplete episode, classified by the frozen 6A.0 classifier (`tools/diagnostics/replay_failures.py`, hash in §5; last 100 steps).
- Ratio of sums with X = 1, i.e. the mean per episode.
- Successful episodes contribute 0.

| Mechanism | Targeted category | Expected sign |
|---|---|---|
| S | frozen | ↓ |
| O | oscillating | ↓ |
| R | conflict_loop | ↓ |
| D | deadlock_cycle | ↓ |

**Rule against re-labelling.** A category reduction counts only together with both of:
- (a) a non-negative completion effect in the same family;
- (b) no increase in the **total** number of stuck robots per episode (reported with its interval).

A robot that moves from "frozen" to "other_moving" without delivering is not an improvement.

**Evaluability:**
- A category with 0 robots in a family's OFF arm is not evaluable there (p = 1, no estimate).
- Rare categories will often be under-powered: conflict_loop and deadlock_cycle have only 98 and 62 robots in the whole Block B 6A.0 population. A non-rejection is then reported as "not shown", never as "no effect".

### 6.5 G3: mechanism attribution (3c)

- **Main effect of a factor M ∈ {R, S+O, D}:** per instance, the mean of the 4 paired differences (M on − M off) over the levels of the other two factors; then the frozen estimator.
- **O's own effect:** full minus "S without O".

**A mechanism is kept iff both hold:**
1. Its completion main effect is not Holm-rejected negative in any family (P1/P2 analogues).
2. Either:
   - its targeted category's main effect is Holm-rejected with Δ̂ < 0 in at least one family where the category is evaluable (§6.4, with the re-labelling rule); **or**
   - if the target is not shown, its completion main effect is Holm-rejected positive in at least one family.

Otherwise the mechanism is dropped (design §10, criterion 4: simplicity).

The retained configuration's estimate is read from its own 3c arm and labelled *selected on Block C*. Its confirmatory test belongs to Phase 6, on a new held-out block.

### 6.6 Vetoes (P4), fairness and starvation

| Quantity | Computation | Veto if |
|---|---|---|
| High-priority flow time | flow time of tasks delivered **in both arms** (paired by task id within an instance), highest priority class only; ratio of sums | Holm-rejected increase in any family |
| High-priority undelivered tasks | undelivered tasks of the highest priority class per episode | Holm-rejected increase in any family |
| R fairness | `max_consecutive_r_losses` (G0.5) and the distribution of `monitor.r_losses_by_robot` | G0.5 only (the bound holds by construction) |
| Starvation | `monitor.max_steps_without_progress`; the per-episode share of robots that end with a task but never progressed in their last leg | reported, not gated: there is no non-arbitrary threshold, and OFF has no supervisor counters |
| Per-robot load | the distribution of per-robot wait ratio and delivered tasks, ON vs OFF | reported |

The common-delivered set avoids a selection effect: comparing flow times over different task sets would make more completed hard tasks look like a fairness loss.

### 6.7 Interventions, overrides and repeated bursts (reported; bounded by G0)

Reported per policy × level × arm:
- the **override share**, `modified_robot_steps` / non-idle robot-steps, by reason (`recovery`, `R_loss`);
- `monitor.max_robot_modified_share`;
- recoveries started by kind (S, O, D) and ended by reason, and the **share ended by progress**;
- **repeated bursts:** the distribution of recoveries per robot (`monitor.recoveries_by_robot`) and `monitor.max_recoveries_in_a_leg`, against the [C] bound H / (H + min(W, K)) (§4.2);
- **efficiency:** modified robot-steps per additional delivered task (Σ modified / Σ Δdelivered, only when Σ Δdelivered > 0);
- **calibration transfer:** in instances the OFF arm completes, the share of robot-legs with a first trigger, compared with Stage 3a's [M] prediction (A\* 1.2–1.3 %, RRT\* 1.5–1.6 %). A value outside the range is flagged for review. It does **not** re-open calibration on Block C (§7).

No gate is placed on the override share itself. A high share on a policy that otherwise freezes is the intended behaviour, and G1 already requires that task execution improves.

### 6.8 Runtime overhead (reported)

- **Per arm:** median and IQR of wall time per step (`wall_s / steps`), of `act_s` and of `prepare_s`; the supervisor's own time from a profiled 1 % sample.
- **Overhead ratio** ON / OFF, with its interval.
- **Not gated:** the project has no real-time budget from which a threshold could be derived. The reference is the software-level measurement (0.04 ms of a 0.82 ms step).

### 6.9 What "go" licenses

- **3b go** → request approval for 3c (and 3e).
- **3c** → determines the retained configuration (§6.5); no parameter of §4 changes.
- **3e go** → the retained configuration is not harmful under RRT\*.
- **Phase 3 is "go towards Phase 6"** only if 3b is go, 3e is go, every retained mechanism passed §6.5, and G0 passed in every arm. All Block C results are development results; the claim is tested only on the new held-out block.

---

## 7. Anti-tuning rules for the later stages

1. §4's values are final for Phase 3. No 3b, 3c or 3e result may change W, K, cells, m, H, Δ_prog, H_stay, L_max or the resolver parameters. This includes the "H sensitivity on Block C" originally planned in design §5, which may only be reported.
2. If a stage reveals a defect (software, safety), the fix is reported as a deviation. Any re-calibration needs a new approval and uses Block B only. The affected stage is then re-run in full, never partially.
3. Every Phase 3 run records `supervisor_stats.params` (G0.7), so a record produced with any other configuration is detectable.
4. Escalation stays disabled and is not an evaluation arm.

---

## 8. Risks and unresolved decisions

| # | Risk / decision | Status |
|---|---|---|
| R1 | **Checkpoint identity.** The stored Block B episodes of mappo_seed1–3 and ippo_seed1 come from their Issue #6 `best.pt`. The SCB-C v5 picks named for Phase 3 (design D5) are other checkpoints of the same runs, and no stored Block B episode of them exists. ippo_seed2/3 and MADQL are unaffected. [M] Per policy at the frozen values, the trigger rates span 0.61–1.13 % (A\*), and the two unchanged IPPO runs lie inside that range. | **Decision needed before 3b:** (a) accept, as documented; (b) authorize an OFF Block B run of the four v5 picks, used *only* to check the §6.7 transfer rate (no tuning); or (c) evaluate the Issue #6 picks in 3b |
| R2 | Per-planner configuration: a revision of the approved rule (§3.2, items 1–4) | needs explicit approval of the revisions |
| R3 | RRT\* W is unstable (32 vs 48 in the bootstrap), and RRT\* detection is unmeasured | accepted for a robustness stage; reported |
| R4 | A\* O sits near the target for IPPO (0.96 %) | reported; §6.7 checks the transfer |
| R5 | H, Δ_prog, H_stay and L_max are not calibratable from OFF data | design values kept; [H] projections only |
| R6 | Repeated bounded bursts (no hard cap, D11): frozen robots are recovered in bursts | monitored (§6.7, G0.6); A\* bound tightened to 36 % |
| R7 | The go / no-go criteria (§6), including δ₁ and the α split | **need approval** (D10) before 3b |
| R8 | The 3e seed subset | to be confirmed at 3e approval |
| R9 | MADQL has no normal-behaviour reference of its own | inherent; MADQL's interventions are expected to be frequent (31 % of its A\* legs would see a first trigger [M]), and G1 judges them by completion |

---

## 9. Tests and checks run (Stage 3a only)

- **Targeted suites:** `tests/unit/control/test_supervisor.py` (33, two of them new: monitor statistics, frozen pin), `tests/integration/test_supervisor_integration.py` (17), `tests/integration/test_supervisor_regression.py` (13, extended to check the frozen params and the monitor), `tests/tools/test_stage3a_calibration.py` (4, new), `tests/tools/test_phase0b_replay.py`. **74 passed**, 0 failed, 0 skipped. The full suite was not rerun (not required).
- **Ruff** (format and check) and **mypy:** clean on every changed file.
- **Replays:** 2 700 + 2 160 episodes reproduce their stored records exactly.
- **Protected artifacts, verified unchanged:**
  - the 9 frozen hashes of the Block D plan;
  - the Block D `plan_freeze`, `check`, `records` and `analysis`;
  - the Phase 0b outputs;
  - `block_c_analysis.py` (3fe86a77…).
- **Untouched:** `docs/SESSION_HANDOFF.md` (last modified 2026-11-19), and `models/`, which does not exist.
