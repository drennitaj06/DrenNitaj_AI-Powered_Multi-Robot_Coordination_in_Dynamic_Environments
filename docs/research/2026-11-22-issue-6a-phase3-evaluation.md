# Issue 6A Phase 3: Execution Supervisor evaluation (Stages 3b, 3c, 3e)

> **Status (2026-11-22): Phase 3 complete.** Student decisions recorded in **§11**:
> - The Stage 3a go / no-go criteria are **approved** for interpreting these development-block results.
> - The configuration going forward is **R + S + D** (O dropped).
> - The verdicts below stand under the approved criteria.
>
> These are development-block results, **not** independent confirmation; Phase 6 must validate on a new held-out block.
> The original text below is unchanged. The word "recommended" in it refers to the criteria's status *before* this approval.

*Executed 2026-11-21/22 under the student's consolidated authorization of Stages 3b, 3c and 3e. All 31 320 planned episodes ran: 0 errors, 0 safety events, determinism replay identical. The decisions below use the **recommended** go / no-go criteria of the Stage 3a freeze (§6 of `2026-11-21-issue-6a-stage3a-calibration-freeze.md`); those criteria have **not** been formally approved, and every verdict is labelled accordingly. Full generated tables: `2026-11-22-issue-6a-phase3-evaluation-tables.md`.*

Labels: **[M]** measured; **[C]** follows from the code; **[I]** interpretation.

---

## 0. Summary

| | 3b (A\*, OFF vs full) | 3c (A\*, factorial) | 3e (RRT\*, OFF vs full) |
|---|---|---|---|
| Episodes | 6 480 | 22 680 (+ the 3b OFF/full arms) | 2 160 |
| Errors / collisions / introduced invalid moves | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| MADQL completion, OFF → full | 0.624 → **0.997** (Δ +0.373, CI [0.358, 0.389]) | S+O main effect **+0.370** | 0.695 → **0.996** (Δ +0.301) |
| IPPO completion | 0.987 → **0.999** (Δ +0.012) | S+O +0.010, R +0.002 | 0.992 → **1.000** (Δ +0.008) |
| MAPPO completion | 0.988 → **0.998** (Δ +0.010) | S+O +0.008, R +0.001, D +0.002 | 0.996 → **1.000** (Δ +0.004) |
| Verdict (recommended criteria) | **go** | keep **R, S+O, D**; **O not shown** (drop under the rule) | **go** |

All completion gains in the table are Holm-rejected at their family α. No family, policy, planner or level shows a completion loss; 6 of 4 320 paired A\*/RRT\* episodes deliver one or two tasks fewer with the supervisor, against 1 766 that deliver more (§5.4).

**Phase 3 acceptance:** implementation complete; experiments complete; acceptance **"go towards Phase 6" under the recommended criteria**, pending (a) the student's approval of those criteria and of the 3c retention outcome for O, and (b) confirmation on the new held-out block in Phase 6 (all Block C results are development results).

---

## 1. Preparation and verification

### 1.1 Stage 3a freeze and configuration [M]

- `results/dev6a/phase3/stage3a/freeze.json` SHA-256 `4c7b9dbc…866378fd` verified; its `frozen_params` equal `supervisor.FROZEN_PARAMS` (A\*: W 24, O (14, 3, 7); RRT\*: W 32, O (8, 3, 4); H 8, Δ_prog 2, H_stay 3, L_max 3; escalation off). `src/robocoord/control/supervisor.py` is byte-identical to the freeze (`83b84b8e…`).
- Every supervised record carries `supervisor_stats.params` equal to its planner's frozen values; escalation is off in every record (G0.7, §3).

### 1.2 Checkpoints (SCB-C v5 picks; read-only; verified by the Block D `picks` checks) [M]

| Run | Phase 3 checkpoint | SHA-256 | Issue #6 / Stage 3a checkpoint |
|---|---|---|---|
| madql_seed1 | `results/training/madql_seed1/best.pt` (Issue #5 pick) | `dd9f088c2bfe…` | same |
| madql_seed2 | `…/madql_seed2/best.pt` | `c66bd8c2d606…` | same |
| madql_seed3 | `…/madql_seed3/best.pt` | `492f2a61d23a…` | same |
| ippo_seed1 | `…/ippo_seed1/checkpoints/step_00500000.pt` | `1909553728a9…` | **differs**: step_00550000 (`best.pt` `3706f6c6…`) |
| ippo_seed2 | `…/ippo_seed2/checkpoints/step_00250000.pt` | `ab628b9847dd…` | same weights as `best.pt` |
| ippo_seed3 | `…/ippo_seed3/checkpoints/step_00500000.pt` | `63214e5f6767…` | same weights as `best.pt` |
| mappo_seed1 | `…/mappo_seed1/checkpoints/step_00300000.pt` | `e1b93a11fc9e…` | **differs**: step_00200000 |
| mappo_seed2 | `…/mappo_seed2/checkpoints/step_00500000.pt` | `4cef9b237088…` | **differs**: step_00300000 |
| mappo_seed3 | `…/mappo_seed3/checkpoints/step_00500000.pt` | `6b00a5a5cbcb…` | **differs**: step_00150000 |

- All checks of the Block D protocol §3 pass (selections hash, sidecars, tensor identity, Block S/C manifest hashes). Full hashes are in `results/dev6a/phase3/eval/check.json`.
- Every checkpoint hash was recomputed after each stage: unchanged. Nothing was written to `results/training/` or `models/`.
- **Checkpoint mismatch with the Stage 3a calibration data (documented, as instructed):** the Stage 3a reference legs of mappo_seed1–3 and ippo_seed1 came from their Issue #6 `best.pt`; Phase 3 evaluates their SCB-C v5 picks. The frozen parameters were **not** adjusted. Transfer check (§5.3): in OFF-complete instances, the ON first-trigger behaviour is consistent with Stage 3a (no evidence that the mismatch distorts the calibration), but the per-policy false-trigger rate of the v5 picks was not measured on Block B.

### 1.3 Seeds and arms [C]

- A\* (3b, 3c): Block C 11 000–11 119 × {static, obstacles, full}. RRT\* (3e): the Block C RRT\* subset **11 000–11 039** (proposal §block definitions: "RRT\* subset 11 000–11 039"). Workers replace the production seed check with a Block C guard (RRT\*: the subset only); Blocks train, A, B, D, S and test are refused before any scenario generation.
- Arms (supervisor mode, resolver): off (off, off); full (R+S+O, on); R; SO (S+O); D (resolver only); R_SO; R_D; SO_D; R_S_D ("S without O": R+S, on). 3c reuses the 3b off and full episodes (identical configuration and code hashes).
- Every condition otherwise as the established evaluation: greedy, auction, shield off, endpoint recovery on, `path_limit` 6, horizon 400, replicate 0, RRT\* 1 000 iterations.

---

## 2. Implementation

| Item | File | Notes |
|---|---|---|
| Read-only per-step hook | `src/robocoord/marl/evaluation.py` | `run_episode(..., observer=None)`: `before(engine, view, actions)` / `after(engine, outcome)`. Additive; without an observer the code path and records are unchanged (regression tests; §3) |
| Observer | `tools/phase3/observer.py` (new) | 6A.0 stuck-robot classifier ported verbatim (with the supervisor on, the *requested* action counts as "proposed"); per-task outcomes; supervisor-introduced invalid moves; its own time |
| Runner | `tools/phase3/phase3.py` (new) | `check` (freeze, params, checkpoints, code hashes), `run --stage 3b/3c/3e` (resumable, failures recorded, checkpoint hashes before/after, code-hash gate), `replay` (every 50th task, fresh processes, no observer, no timing wrapper) |
| Analysis | `tools/phase3/phase3_analysis.py` (new) | frozen Block C estimator; families P1–P4; G0; factorial main effects; retention rule; descriptive tables |
| Tables | `tools/phase3/phase3_report.py` (new) | renders `analysis.json`; no computation |
| Tests | `tests/tools/test_phase3.py` (new, 15) | design completeness, stage sizes, factorial pairs, guards, flow / undelivered outcomes, inference edge cases, Holm with not-evaluable rows, **observer neutrality** |

**Validation of the classifier port [M]:** on 116 stored Block B failed episodes (every 8th of the 6A.0 population), the observer reproduces the stored 6A.0 class of **every** stuck robot (116 / 116 episodes identical). This used Block B only as a tool check; nothing was tuned.

No production algorithm, parameter, policy, planner, shield, arbiter or resolver was changed. No defect blocking execution was found.

---

## 3. Integrity and safety (G0; hard gates) [M]

| Check | 3b | 3c (29 160 episodes incl. 3b arms) | 3e |
|---|---|---|---|
| G0.1 OFF reproduces production: OFF episodes of the 4 changed PPO runs vs stored Block C "new" records (all non-timing fields) | **1 440 / 1 440 identical** | — | — |
| G0.2 determinism and observer neutrality: replay of every 50th task (627 across stages) without observer | **627 / 627 identical** | (included) | (included) |
| G0.3 errors / physical collisions / transition errors | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| G0.4 INV conflicts (total; ON > OFF instances); supervisor-introduced invalid moves | 0; 0; 0 | 0; 0; 0 | 0; 0; 0 |
| G0.5 max consecutive R losses (bound 3) | 1 | 1 | 1 |
| G0.6 burst bound violations | 0 | 0 | 0 |
| G0.7 frozen params / escalation off / instances identical across arms | yes / yes / yes | yes / yes / yes | yes / yes / yes |
| **G0** | **pass** | **pass** | **pass** |

---

## 4. Stage 3b: core evaluation, A\* (OFF vs full)

**Families** (Holm within family; α as recommended: P1 0.03, P2 0.005, P3 0.01, P4 0.005). θ = completion = Σ delivered / Σ released; Δ = ON − OFF; nominal 95 % bootstrap-t CI; B = 10 000.

### 4.1 Completion (P1 families, P2 policies) [M]

| Unit | OFF | Full | Δ̂ | 95 % CI | Holm p | Decision |
|---|---|---|---|---|---|---|
| **MADQL** | 0.6240 | 0.9974 | **+0.3734** | [0.3578, 0.3890] | 0.0003 | rejected (gain) |
| **IPPO** | 0.9870 | 0.9987 | **+0.0117** | [0.0093, 0.0141] | 0.0003 | rejected (gain) |
| **MAPPO** | 0.9880 | 0.9982 | **+0.0102** | [0.0078, 0.0126] | 0.0003 | rejected (gain) |
| madql_seed1 / 2 / 3 | 0.673 / 0.596 / 0.603 | | +0.325 / +0.401 / +0.395 | all exclude 0 | ≤ 0.0009 | rejected |
| ippo_seed1 / 2 / 3 | 0.987 / 0.986 / 0.988 | | +0.012 / +0.013 / +0.011 | all exclude 0 | ≤ 0.0009 | rejected |
| mappo_seed1 / 2 / 3 | 0.989 / 0.988 / 0.987 | | +0.009 / +0.010 / +0.012 | all exclude 0 | ≤ 0.002 | rejected |

- **Practical significance.** δ₁ = 1 / mean released per OFF episode = **0.052** (mean 19.2 released tasks; the full level has 28). MADQL's gain is **7.2 tasks per episode** (+0.373 / 0.052): far above one task. The PPO gains are 0.20–0.22 tasks per episode, i.e. about one more delivered task per five episodes.
- **Undelivered tasks per episode [M]:** MADQL 7.22 → 0.05; IPPO 0.25 → 0.03; MAPPO 0.23 → 0.03. **Successful episodes:** MADQL 15 → 1 032 of 1 080; IPPO 897 → 1 055; MAPPO 931 → 1 047.
- **Per level (nominal):** MADQL static 0.362 → 0.998 (largest), obstacles 0.762 → 0.998, full 0.702 → 0.997; IPPO static 0.971 → 0.999; MAPPO static 0.978 → 0.999. Every level × family interval excludes 0.
- **Seeds:** every MADQL seed improves (120 / 120 × 3 policies). PPO: 29–49 seeds better, 0–1 worse, the rest equal per policy.

### 4.2 Targeted failure categories (P3; stuck robots per episode, 6A.0 classes) [M]

| Family | frozen | oscillating | conflict_loop | deadlock_cycle | total stuck (nominal) |
|---|---|---|---|---|---|
| MADQL | 1.288 → 0.009 (rej.) | 3.188 → 0.013 (rej.) | 0.008 → 0.001 (not rej.) | 0.056 → 0.000 (rej.) | 5.76 → 0.05 |
| IPPO | 0.031 → 0.002 (rej.) | 0.141 → 0.008 (rej.) | 0.039 → 0.004 (rej., MC-sens.) | 0.017 → 0.002 (not rej.) | 0.25 → 0.03 |
| MAPPO | 0.026 → 0.002 (not rej., MC-sens.; raw p 0.003) | 0.111 → 0.012 (rej.) | 0.020 → 0.001 (not rej.; raw p 0.03) | 0.046 → 0.006 (rej.) | 0.23 → 0.03 |

Every category falls; the total number of stuck robots falls in every family, so the reductions are not re-labelling. Residual stuck robots with the full supervisor (A\*, 3 240 episodes): 118 robots in 106 episodes (other_moving 45, oscillating 36, frozen 14, deadlock_cycle 8, blocked_by_idle 6, conflict_loop 6, mostly_stationary 3).

### 4.3 Vetoes (P4) and fairness [M]

| Family | High-priority flow time, common-delivered tasks (steps) | High-priority undelivered per episode |
|---|---|---|
| MADQL | 67.6 → 48.8 (Δ −18.8, rejected: **faster**) | 1.368 → 0.004 (rejected: fewer) |
| IPPO | 39.2 → 38.3 (Δ −0.9, not rejected) | 0.057 → 0.002 (rejected: fewer) |
| MAPPO | 39.1 → 38.2 (Δ −0.9, not rejected) | 0.056 → 0.004 (rejected: fewer) |

No veto: no family shows a high-priority flow-time increase or more undelivered high-priority tasks; every significant change is an improvement. R fairness: the longest R-loss streak in any episode is 1 (bound 3).

### 4.4 Conflicts (supportive, nominal) [M]

Conflicts per episode (arbiter-level, after the supervisor): MADQL 2.97 → 1.55, IPPO 11.48 → 2.67, MAPPO 6.94 → 1.58; per 100 robot-steps IPPO 0.75 → 0.22, MAPPO 0.48 → 0.13, MADQL unchanged (0.09 → 0.09, longer active time). **Caveat [C]:** these are conflicts of the proposal reaching the arbiter; R turns contested moves into STAY and so lowers them mechanically. The policy's own-proposal conflicts (K^ctrl) are not stored in the evaluation records (limitation, §9). R's own changes are few (122–237 R losses per 1 080 episodes), so most of the reduction comes from S/O resolving stuck robots that otherwise conflict every step.

### 4.5 Interventions, recovery bursts and starvation [M]

| Family (full, A\*) | Modified share of robot-steps (of eligible) | R losses | Recoveries S / O / D | Ended by progress | Recoveries per robot: mean / max | Max recoveries in a leg: q50 / q99 / max | Max per-robot modified share: q50 / max |
|---|---|---|---|---|---|---|---|
| MADQL | 1.26 % (1.91 %) | 122 | 5 450 / 11 930 / 3 | 37.8 % | 2.01 / 16 | 2 / 9 / 15 | 3.5 % / 17.0 % |
| IPPO | 0.14 % (0.24 %) | 237 | 150 / 768 / 0 | 52.1 % | 0.11 / 18 | 0 / 10 / 18 | 0 % / 13.5 % |
| MAPPO | 0.16 % (0.27 %) | 151 | 127 / 775 / 1 | 34.2 % | 0.10 / 20 | 0 / 13 / 20 | 0 % / 17.0 % |

- The supervisor changes ≈ 1 in 80 robot-steps for MADQL and ≈ 1 in 700 for PPO. The largest per-robot override share in any episode is 17 %, below the [C] bound H / (H + min(W, K)) = 36 % (A\*).
- **Repeated bursts:** the median robot never needs a recovery (PPO) or needs 2 (MADQL); the worst leg had 20 recoveries (MAPPO). No burst-bound violation (G0.6).
- **Efficiency:** modified robot-steps per additional delivered task (Σ modified / Σ Δ delivered): MADQL 3.0, IPPO 7.7, MAPPO 9.8.
- **Starvation:** the longest no-progress time of any robot is median 11–25 steps per episode; the maximum (348–390) belongs to the residual stuck robots of §4.2. No R starvation (streak ≤ 1).

### 4.6 Runtime [M]

Per-step wall time without the observer (median, A\*): OFF 3.88–4.00 ms, full 4.10–4.20 ms → **ratio 1.03–1.07**; the supervisor itself takes 0.16–0.18 ms per step (**3.9–4.2 %** of step time). 1–5 episodes per arm were flagged as timing outliers (wall time > 50 × the median; a batch near seed 11 005 includes a ≈ 78-minute system pause) and excluded from the share only; medians are unaffected.

### 4.7 Decision 3b (recommended criteria, §6.3 of the freeze)

| Condition | Result |
|---|---|
| G0 | pass |
| MADQL improves, Holm-rejected, Δ̂ ≥ δ₁ | yes (+0.373 ≥ 0.052) |
| No PPO family / policy harmed | yes (no rejected loss) |
| "No harm" interpretable (PPO CI lower bound > −δ₁) | yes (lower bounds +0.009, +0.008) |
| No veto | yes |
| **Verdict** | **go** |

---

## 5. Stage 3c: mechanism ablation (2³ factorial + "S without O")

### 5.1 Cell completion [M]

| Arm (R, S+O, D) | MADQL | IPPO | MAPPO |
|---|---|---|---|
| off (0,0,0) | 0.6240 | 0.9870 | 0.9880 |
| R (1,0,0) | 0.6242 | 0.9891 | 0.9890 |
| D (0,0,1) | 0.6268 | 0.9879 | 0.9907 |
| R_D (1,0,1) | 0.6272 | 0.9899 | 0.9916 |
| SO (0,1,0) | 0.9946 | 0.9973 | 0.9964 |
| SO_D (0,1,1) | 0.9947 | 0.9973 | 0.9971 |
| R_SO (1,1,0) | 0.9971 | 0.9987 | 0.9980 |
| full (1,1,1) | **0.9974** | **0.9987** | **0.9982** |
| R_S_D = full without O | 0.9970 | 0.9985 | 0.9982 |

### 5.2 Main effects (mean of the 4 paired differences over the other factors) [M]

| Factor | Family | Completion Δ̂ [95 % CI] | Holm decision | Targeted category Δ̂ (Holm) |
|---|---|---|---|---|
| **S+O** | MADQL | **+0.3704** [0.3550, 0.3857] | rejected | frozen −1.288, oscillating −3.166 (both rejected) |
| | IPPO | **+0.0095** [0.0075, 0.0116] | rejected | frozen −0.029, oscillating −0.126 (both rejected) |
| | MAPPO | **+0.0076** [0.0056, 0.0097] | rejected | oscillating −0.087 (rejected); frozen −0.024 (raw p 0.002, not rejected) |
| **R** | MADQL | +0.0014 [0.0004, 0.0025] | not rejected (MC-sens.) | conflict_loop −0.004 (not rejected) |
| | IPPO | **+0.0017** [0.0008, 0.0026] | rejected | conflict_loop −0.027 (raw p 0.007, not rejected) |
| | MAPPO | **+0.0011** [0.0005, 0.0017] | rejected | conflict_loop −0.017 (raw p 0.012, not rejected) |
| **D** | MADQL | +0.0016 [0.0002, 0.0030] | not rejected | deadlock_cycle −0.018 (**rejected**, MC-sens.) |
| | IPPO | +0.0004 [−0.0001, 0.0010] | not rejected | −0.004 (not rejected) |
| | MAPPO | **+0.0015** [0.0008, 0.0023] | rejected | −0.018 (raw p 0.003, not rejected) |
| **O** (full − S without O) | MADQL | +0.0005 [−0.0003, 0.0013] | not rejected | oscillating +0.007 (not rejected) |
| | IPPO | +0.0001 | not rejected | +0.001 (not rejected) |
| | MAPPO | +0.0000 | not rejected | −0.001 (not rejected) |

- **No mechanism has a negative completion main effect** in any family or policy (no rejection, every point estimate ≥ −0.0004).
- Per policy (P2, α 0.005 over 36 hypotheses): S+O rejected for all 3 MADQL and 2 IPPO policies (all MC-sensitive at p = 1/10 001); no other per-policy rejection. Several per-policy O intervals are unbounded ([−∞, ∞]): with sparse differences (a handful of differing seeds) some resamples have zero SE, which the frozen bootstrap-t maps to |t\*| = ∞. These rows are not rejected and their percentile intervals are in `analysis.json`.
- **Total stuck robots** fall with every factor except O (nominal; S+O −5.67 / −0.18 / −0.15; R −0.03 / −0.03 / −0.02; D −0.009 / −0.008 / −0.030), so the targeted reductions are not re-labelling.
- **P4 (vetoes):** no increase anywhere. S+O reduces high-priority undelivered tasks (MADQL −1.34, IPPO −0.05, rejected) and MADQL's high-priority flow time (−18.5 steps, rejected). **O reduces MADQL's high-priority flow time by 4.0 steps** (rejected, MC-sensitive) and IPPO/MAPPO's by 0.3 (raw p 0.0005 / 0.003, not rejected): O makes recovery faster without changing completion.

### 5.3 Interactions and interventions [M, I]

- **R × S+O (positive interaction).** R alone adds ≈ 0.0002–0.002; with S+O present it adds 0.0025 (MADQL), 0.0014 (IPPO), 0.0016 (MAPPO) (SO → R_SO). R also *reduces* S+O's interventions: modified share SO 1.63 % / 0.27 % / 0.41 % → R_SO 1.34 % / 0.18 % / 0.22 % (MADQL / IPPO / MAPPO): settling free-cell contests prevents stalls that S/O would otherwise have to recover. [I] The mechanisms are complementary, not redundant.
- **D.** Small additive gains on top of R (R → R_D) and of R+S+O (R_SO → full: +0.0003 / 0.0000 / +0.0002); D's own effect is concentrated in deadlock cycles (MADQL rejected).
- **O vs S.** S alone already detects stuck oscillating robots (Stage 3a: 99.5 %). O fires earlier (Stage 3a: median 10 steps), which shows as faster MADQL delivery, but completion is at the ceiling (0.997–0.999) once S and R are present. O raises interventions: modified share full 1.26 % / 0.14 % / 0.16 % vs R_S_D 1.04 % / 0.10 % / 0.09 % (+21 % / +40 % / +78 %), and recoveries per robot 2.01 vs 1.83 (MADQL).
- **Calibration transfer (Stage 3a §6.7):** Stage 3a predicted a first S/O trigger on 1.2–1.3 % of normal A\* legs (≈ 42 legs per successful episode, so ≈ 0.51 first triggers per episode) and 1.5–1.6 % under RRT\* (≈ 0.69). Observed in the instances the OFF arm completes, recoveries per full-arm episode (first triggers *and* repeats, so an upper bound): A\* IPPO 0.55, MAPPO 0.36; RRT\* IPPO 0.82, MAPPO 0.63. Consistent with the prediction, including for the four v5 picks whose calibration data came from other checkpoints; no flag.

### 5.4 Retention under the recommended rule (§6.5 of the freeze)

| Mechanism | Completion harm | Target reduction shown (with re-labelling rule) | Completion gain (Holm) | **Rule outcome** |
|---|---|---|---|---|
| S+O | none | MADQL frozen + oscillating, IPPO frozen + oscillating, MAPPO oscillating | all three families | **keep** |
| R | none | not shown after Holm (raw p 0.007–0.012) | IPPO, MAPPO | **keep** (via the completion-gain branch) |
| D | none | MADQL deadlock_cycle (MC-sensitive) | MAPPO | **keep** |
| O (within S+O) | none | not shown | none | **drop** |

**Episodes worse with the supervisor [M]:** A\* full vs OFF: 5 of 3 240 paired episodes deliver 1–2 tasks fewer (ippo_seed2 full 11 033 and 11 054, ippo_seed3 full 11 046, ippo_seed1 static 11 071, mappo_seed2 full 11 113); RRT\*: 1 of 1 080 (mappo_seed1 full 11 038). Against 1 350 A\* and 416 RRT\* paired episodes that improve. None changes any decision; they are listed for transparency.

---

## 6. Stage 3e: RRT\* (OFF vs full, frozen RRT\* parameters)

| Unit | OFF | Full | Δ̂ [95 % CI] | Holm decision |
|---|---|---|---|---|
| **MADQL** | 0.6951 | 0.9964 | **+0.3013** [0.2785, 0.3241] | rejected (gain) |
| **IPPO** | 0.9921 | 1.0000 | **+0.0079** [0.0047, 0.0111] | rejected (gain) |
| **MAPPO** | 0.9961 | 0.9997 | **+0.0036** [0.0016, 0.0056] | rejected (gain) |
| per policy | | | MADQL 1–3 +0.23 / +0.36 / +0.31 (rejected); ippo_seed1 +0.013 (rejected, MC-sens.); the other 5 PPO +0.002–+0.007 (not rejected) | |

- **Successful episodes:** MADQL 5 → 336 of 360, IPPO 323 → 360, MAPPO 335 → 358. Every MADQL seed improves (40 / 40 × 3).
- **Categories (descriptive under RRT\*):** MADQL frozen 1.80 → 0.003, oscillating 1.86 → 0.000 per episode; remaining stuck robots: 26 other_moving, 1 frozen.
- **Vetoes:** none; MADQL high-priority flow time −17.0 steps and high-priority undelivered −0.875 per episode (both rejected improvements).
- **Interventions:** modified share MADQL 1.17 %, IPPO 0.12 %, MAPPO 0.11 %; recoveries per robot 1.94 / 0.11 / 0.10; worst leg 24 recoveries (MADQL); max per-robot share 17.7 % (bound 50 % [C]).
- **Runtime:** supervisor 0.18–0.19 ms per step, 0.1 % of an RRT\* step (120–280 ms). Per-step time ON / OFF: PPO 0.99–1.04; **MADQL 0.66**: planner calls fall from 24.8 to 16.8 per 100 robot-steps once robots stop oscillating and replanning off-path [M], which makes the average RRT\* step cheaper [I]. The RRT\* stage's wall clock (13.4 h) includes long system pauses; 2–13 timing outliers per arm are flagged and medians used.
- **Decision 3e (recommended criteria): go** (G0 pass; no rejected loss; MADQL Δ̂ > 0; no veto).

**A\* vs RRT\*.** The effect pattern is the same: very large for MADQL, small but significant for PPO, no harm. MADQL's RRT\* OFF baseline is higher (0.695 vs 0.624, a different seed subset and planner) and its gain correspondingly smaller (+0.30 vs +0.37). Results are reported per planner and not pooled.

---

## 7. Recommendations [I]

1. **Retain R, S and D.** Each has a non-negative completion effect everywhere, and each reduces its target failure mode or measurably raises completion. R and S+O interact positively, and R lowers S/O's intervention load.
2. **O: reject as a default component (rule outcome), or keep as an explicit efficiency option.** O adds no completion and no oscillation reduction beyond S once S and R are present, and it raises the override share by 20–80 %. It does speed up MADQL's high-priority deliveries by 4 steps. Recommended Phase 6 configuration: **R + S + D** (arm R_S_D: completion 0.9970 / 0.9985 / 0.9982 on Block C, with the frozen Stage 3a parameters unchanged and O's parameters unused). If delivery speed is valued over simplicity, keep the full configuration. This choice is the student's (the retention rule is a recommendation).
3. **Do not modify the frozen parameters.** Phase 3 gives no reason to, and Phase 6 must test them as frozen.
4. **Report results as "policy + supervisor"** next to the unsupervised results. The supervisor does not fix the policy (MADQL alone remains at 0.62 completion) but executes it reliably.

---

## 8. Phase 3 acceptance assessment

| Aspect | Status |
|---|---|
| Implementation | **complete**: runner, observer, analysis, tests; one additive production hook |
| Experiments | **complete**: 3b 6 480, 3c 22 680, 3e 2 160 episodes; 0 errors; replay identical |
| Safety | **pass** in every arm (0 collisions, 0 errors, 0 introduced invalid moves; bounds hold) |
| 3b | **go** (recommended criteria) |
| 3c | R, S+O, D retained; O dropped under the rule (decision pending, §7.2) |
| 3e | **go** (recommended criteria) |
| **Phase 3 overall** | **go towards Phase 6 under the recommended criteria.** Formal acceptance requires the student's approval of the criteria (D10) and of the O decision; the confirmatory test belongs to Phase 6 on the new held-out block (proposed 14 000–14 199, not yet reserved) |

The project is ready to move on to the next Issue 6A phase (roadmap §27: Phase 1 next).

---

## 9. Limitations and open questions

1. **Criteria are recommendations.** The §6 go / no-go criteria and α split were proposed in Stage 3a and not formally approved; the results are far from every decision boundary except a few MC-sensitive category rows, so the verdicts are not sensitive to reasonable alternative α splits [I].
2. **Development block.** Block C was used before (Block C evaluation of the v5 picks). The supervisor's parameters were not tuned on it, but the 3c configuration choice is "selected on Block C"; confirmation needs Phase 6.
3. **Checkpoint mismatch** with the Stage 3a calibration data for 4 PPO runs (§1.2); not re-calibrated, by design.
4. **K^ctrl not recorded.** Conflict reductions are arbiter-level; R lowers these mechanically. Completion and the stuck categories are the primary evidence.
5. **6A.0 classifier with the supervisor on** uses the requested (supervised) action as "proposed"; validated exactly with the supervisor off only.
6. **Ceiling effects.** With S present, completion is 0.997–0.999; further mechanism effects (O, D) can only be small on these instances.
7. **Bootstrap-t with sparse differences** yields unbounded nominal intervals for some per-policy rows (§5.2); decisions are unaffected (not rejected).
8. **Runtime** wall times include system pauses (≈ 78 min in 3b, longer in 3e); medians and the supervisor's own timer are used. Absolute times are from one Windows machine.
9. **Residual failures** (118 stuck robots in 106 of 3 240 A\* full episodes, mostly other_moving/oscillating) are not addressed by Phase 3.
10. Stochastic mode (3d) and the shield check (3f) were optional and not run.

---

## 10. Provenance and reproduction

**Outputs** (`results/dev6a/phase3/eval/`):

| File | SHA-256 |
|---|---|
| `check.json` | `4890381c41ba705ac09c8bdd55bc3a2ee0f336d07d59bc68fa7cc5048aecbf24` |
| `records_3b.jsonl` (6 480) | `a6ed0179436a6685fee285a3f2ffcd4ea6ec2bb6489e44230b8d19d742802ae7` |
| `records_3c.jsonl` (22 680) | `09db9fe7573af8c3bc6f3c1769640312d0dbac5d66f7f7cbba6dc0a39d48486a` |
| `records_3e.jsonl` (2 160) | `41ef734a7d6b54d4067f45a429732f66517674037c7052800a377f9e6e49c255` |
| `replay.json` | `db992ef5dfca0ef27c91d0ad4506e10408f0e433d199ed1fe577a105cd4ecf02` |
| `analysis.json` | `62180f3fc99d2be29b3c3854a5bdb3d4dca3a5b130b0832c6af0633455842096` |
| `tables.md` | `65e89784e4ca9372ed9b95c5cd6cc45d4f7c9e6da447747651f2fec7779b8e01` |
| `run.log` | run start / end times, errors, checkpoint-hash checks |

**Code** (hashes gated by `run`; equal to `check.json`): `supervisor.py` `83b84b8e…` (= Stage 3a freeze), `evaluation.py` `61abb6e8…` (the freeze's `bc65a7ba…` plus the additive observer hook), `engine.py` `afc744a8…`, `tools/phase3/phase3.py` `ee8ca82f…`, `observer.py` `d8444ea9…`; analysis `phase3_analysis.py` `e7d1f6f0…`, `phase3_report.py` `a3a6e428…`; frozen estimator `block_c_analysis.py` `3fe86a77…`.

**Environment:** Python 3.14, NumPy and PyTorch as recorded in `check.json`; Windows 11; 11 worker processes, one PyTorch thread each.

**Reproduce** (repository root, `.venv`; a new `--out` directory; deterministic except timing fields):

```
python tools/phase3/phase3.py check  --out <dir>
python tools/phase3/phase3.py run    --out <dir> --stage 3b --workers 11
python tools/phase3/phase3.py run    --out <dir> --stage 3c --workers 11
python tools/phase3/phase3.py run    --out <dir> --stage 3e --workers 11
python tools/phase3/phase3.py replay --out <dir> --workers 11
python tools/phase3/phase3_analysis.py --out <dir>
python tools/phase3/phase3_report.py   --out <dir>
```

Measured wall time: 3b 1.5 h (incl. a 78-min pause), 3c 0.7 h, 3e 13.4 h (MADQL RRT\* episodes and pauses), replay and analysis < 15 min.


---

## 11. Student decisions (2026-11-22) and final status

1. **Supervisor configuration going forward: R + S + D, O dropped.**
   - *Basis:* the 3c factorial. O added no measurable completion benefit over S and increased the intervention frequency (§5.2–5.4).
   - *Effect:* the configuration corresponds to arm `R_S_D`, i.e. `EvalTask(supervisor="R+S", deadlock=True)`, with the frozen Stage 3a parameters unchanged. O's parameters are unused.
   - *Not changed:* all Phase 3 results, including the full-supervisor results and every ablation finding, are kept as recorded.
2. **Evaluation criteria approved.** The Stage 3a go / no-go criteria (freeze §6) are approved for interpreting the Phase 3 development-block results. Under them:

   | Stage | Verdict |
   |---|---|
   | 3b | **go** |
   | 3e | **go** |
   | 3c | R, S+O and D retained; O dropped (now adopted, item 1) |

   These criteria and results are **development evidence on Block C, not independent confirmation**. Phase 6 must perform the confirmatory validation on a new held-out block.
3. **Phase 3 status: complete.** Implementation, experiments, analysis and documentation are done. The next phase is Phase 1, replanning optimization (roadmap §27).
