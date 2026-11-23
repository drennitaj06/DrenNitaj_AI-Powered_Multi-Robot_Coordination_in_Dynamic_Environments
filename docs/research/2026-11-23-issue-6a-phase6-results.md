# Issue 6A Phase 6: integrated confirmatory validation, results (2026-11-23)

*This is the confirmatory evaluation of Issue 6A on the held-out Block V.*
- Protocol: `2026-11-23-issue-6a-phase6-protocol.md`, including Amendment A1 and Deviation W1. It was frozen 2026-11-23 15:33:05; `plan_freeze.json` has SHA-256 `db5bc079…`.
- Decisions: D1–D13, recorded in `2026-11-23-issue-6a-phase6-decision-review.md`.
- Generated tables: `2026-11-23-issue-6a-phase6-tables.md`. They are a copy of `results/dev6a/phase6/eval/tables.md`, produced from `analysis.json` by `phase6_report.py`.
- Every number below comes from the frozen analysis output. No analysis outside the protocol was run.

---

## 1. Outcome

| Pre-registered statement (protocol §8) | Result | Basis |
|---|---|---|
| **The Execution Supervisor R + S + D is confirmed** (H-L2 "go" on A\* and on RRT\*) | **CONFIRMED** | 3b go and 3e go; no harm, no veto, G0 and the replay pass |
| **MADQL-v2 n3 is confirmed as the forward MADQL** (H-v2 FA "adopt", FR and NA not "reject") | **NOT CONFIRMED** | **FA "reject" and FR "reject", both by the robot–obstacle (RO) veto**; NA "adopt" |
| **The integrated Issue 6A system is confirmed** (H-L12 "go" on A\* and on RRT\*) | **CONFIRMED, for the system as declared** (v5 PPO + R + S + D; MADQL-v2 + R + S + D) | 3b go and 3e go, against the Issue #6 system |

**Forward configuration under the frozen rules:**
- **Execution:** R + S + D, with the frozen Stage 3a parameters and O off.
- **PPO:** the SCB-C v5 picks.
- **MADQL: the historical (Issue #5) MADQL.** MADQL-v2 failed its pre-registered condition, so §8 keeps the historical MADQL. MADQL-v2 n3 stays a documented, separately labelled configuration.
- **What supports the forward system:** the decision review §7, fixed before the run, states that this system "is described by H-L2's MADQL family and H-L12's PPO families" and "is not re-tested as a new hypothesis". Both pass (§4.1, §4.3).

**Error control (D5):**
- The family-wise error is ≤ 0.05 for each of the 7 units.
- The global bound on any false rejection is **0.302**.
- Every confirmation is a conjunction of several conditions.

---

## 2. Gates, integrity and data

| Gate | Result |
|---|---|
| Freeze (`--authorized-freeze`) | 2026-11-23 15:33:05. It hashes 19 protocol, tool and test files and 19 code files |
| Pre-run check (`--authorized-precheck`) | ok. 869 stored files with 0 Block V seeds; 16 checkpoints intact; Stage 3a frozen; escalation off; 22 080 episodes. **No Block V generated** (Amendment A1) |
| Execution authorization | given by the student |
| **Block V instance check** (first step of `run`, A1) | ok. 200 instances per level, **unique**, and **0 overlap** with every Block A, B, C, D and S instance (`instance_check.json`, `4cee5fd9…`). Training-instance disjointness: not verifiable (no stored hashes), as disclosed |
| Run | **22 080 / 22 080 episodes, 0 errors** (5 h 51 min on 10 workers); checkpoint hashes identical before and after |
| G0, A\* and RRT\* | **pass**:<br>• 0 failed and 0 invalid records;<br>• 0 physical collisions, 0 invalid moves (INV), 0 supervisor-introduced invalid moves, 0 invalid reference paths;<br>• fairness and burst bounds hold; parameters frozen; R + S mode with D;<br>• OFF and ORIG compositions correct;<br>• repair off; P2 off; 1 000 RRT\* iterations;<br>• instances identical across arms and equal to the pre-episode check;<br>• K^arb equals the recorded conflicts;<br>• MADQL-v2 labels correct |
| Replay (every 50th record, fresh processes, no observer) | **442 / 442 identical** (`replay.json`) |
| Analysis | run twice with `PYTHONHASHSEED=0`: **byte-identical** `analysis.json` (`996bc888…`) |
| Post-run integrity | the freeze re-verified; the 1 216 original checkpoint files verify against the manifest; Block C (3) and D (35) freeze hashes match; 1 310 historical files unchanged; frozen statistics, supervisor and Stage 3a freeze unchanged; `docs/SESSION_HANDOFF.md` unchanged; no `models/` |
| **D12** | **NOT SATISFIED: WAIVED by the owner** (Deviation W1; `d12_waiver.json`, `9598c0c1…`). No independent backup has been verified or recorded |

**Records:** `results/dev6a/phase6/eval/records.jsonl` (`ea02baee…`; 17 400 A\* and 4 680 RRT\* episodes).

---

## 3. Results by arm (descriptive)

**A\*** (Block V, 200 seeds × 3 levels per run):

| Arm | Completion | High-priority completion | Flow time | Conflicts / episode | **RO / 1 000 moves** | RR / 100 robot-steps (K^ctrl) | Wait ratio | Stuck robots | Wall s / episode |
|---|---|---|---|---|---|---|---|---|---|
| hist MADQL, OFF | 0.6203 | 0.7828 | 95.7 | 4.85 | 0.804 | 0.095 | 0.225 | 10 459 | 2.04 |
| hist MADQL, R + S + D | 0.9962 | 0.9986 | 86.4 | 1.72 | 0.669 | 0.050 | 0.130 | 131 | 1.08 |
| MADQL-v2, OFF | 0.8596 | 0.9094 | 73.4 | 5.89 | 0.872 | 0.131 | 0.090 | 4 592 | 1.75 |
| MADQL-v2, R + S + D | 0.9970 | 0.9986 | 69.9 | 2.38 | **1.219** | 0.060 | 0.049 | 104 | 0.89 |
| v5 PPO, OFF | 0.9859 | 0.9907 | 50.8 | 9.45 | 1.787 | 0.473 | 0.020 | 976 | 0.69 |
| v5 PPO, R + S + D | 0.9984 | 0.9992 | 50.3 | 2.17 | 1.988 | 0.029 | 0.010 | 108 | 0.61 |
| Issue #6 PPO picks, OFF | 0.9847 | 0.9900 | 50.6 | 35.05 | 20.99 | 0.760 | 0.032 | 707 | 0.70 |
| baseline (original) | 0.9979 | 0.9994 | 49.4 | 1.02 | 1.499 | 0.000 | 0.045 | 24 | 0.31 |

**RRT\*** (60 seeds × 3 levels per run):

| Arm | Completion | High-priority completion | Flow time | Conflicts / episode | **RO / 1 000 moves** | RR / 100 robot-steps (K^ctrl) | Wait ratio | Wall s / episode |
|---|---|---|---|---|---|---|---|---|
| hist MADQL, OFF | 0.7064 | 0.8582 | 102.1 | 7.02 | 0.876 | 0.165 | 0.292 | 101.5 |
| hist MADQL, R + S + D | 0.9937 | 0.9990 | 99.8 | 1.58 | 0.553 | 0.041 | 0.155 | 49.4 |
| MADQL-v2, R + S + D | 0.9962 | 1.0000 | 78.9 | 2.49 | **0.927** | 0.075 | 0.067 | 37.3 |
| v5 PPO, OFF | 0.9934 | 0.9975 | 59.6 | 8.44 | 1.671 | 0.430 | 0.024 | 19.0 |
| v5 PPO, R + S + D | 0.9983 | 1.0000 | 59.5 | 2.48 | 1.915 | 0.036 | 0.013 | 18.8 |
| Issue #6 PPO picks, OFF | 0.9924 | 0.9961 | 59.5 | 27.07 | 16.92 | 0.518 | 0.033 | 17.9 |
| baseline (original) | 0.9994 | 1.0000 | 59.4 | 1.02 | 1.271 | 0.000 | 0.044 | 8.6 |

---

## 4. Pre-registered tests

Notation: Δ̂ = new − old, then [nominal 95 % CI], then raw p / Holm p, then α. **R** = Holm-rejected. MC = MC-sensitive.

### 4.1 H-L2: supervisor R + S + D vs OFF (the same checkpoints)

**A\* (rule 3b): GO.**

| Family | Row | Old | Δ̂ [95 % CI] | p / Holm p | α | Result |
|---|---|---|---|---|---|---|
| P1 | MADQL completion | 0.620 | +0.376 [+0.364, +0.388] | 0.0001 / 0.0003 | 0.03 | **R** |
| P1 | IPPO completion | 0.985 | +0.014 [+0.012, +0.016] | 0.0001 / 0.0003 | 0.03 | **R** |
| P1 | MAPPO completion | 0.987 | +0.011 [+0.009, +0.013] | 0.0001 / 0.0003 | 0.03 | **R** |
| P2 | all 9 policies | — | gains | — | 0.005 | **R** (all 9); no loss |
| P3 | all 12 stuck-robot rows (frozen, oscillating, conflict loop, deadlock cycle × 3 families) | — | reductions | — | 0.01 | **R** (all 12); 3 MC |
| P4 | high-priority flow time | — | MADQL −17.2 **R**; IPPO −0.76 and MAPPO −0.76 not rejected | — | 0.005 | decreases, no veto |
| P4 | high-priority undelivered | — | all decreased; **R** in every family | — | 0.005 | no veto |
| P4 | **RO / 1 000 moves** | — | MADQL −0.14 [−0.67, +0.40]; IPPO +0.40 [−0.01, +0.80] (p 0.055, Holm 0.16); MAPPO +0.03 [−0.37, +0.43] | — | 0.005 | **no veto** |

**RRT\* (rule 3e): GO.**
- P1: MADQL +0.287 [+0.268, +0.307], IPPO +0.007, MAPPO +0.003, all **R**.
- P2: 4 policies **R**; no loss.
- P4: no increase rejected. RO: MADQL −0.32, IPPO +0.47 and MAPPO +0.03, none rejected.
- MAPPO's high-priority-undelivered row has an unbounded interval and is not rejected (§7 rule).

### 4.2 H-v2: MADQL-v2 n3 vs historical MADQL (Phase 4 rule)

| Condition | Completion (P1) | Efficiency (E) | P4 vetoes | **Verdict** |
|---|---|---|---|---|
| **FA** (A\*, R + S + D) | 0.9962 → 0.9970, Δ̂ +0.0008 [−0.0001, +0.0017], Holm p 0.099 (not R); lower bound −0.0001 > −δ₁ | calls −1.54, common flow −16.5, supervisor changes −0.42 (all R) | **RO / 1 000 moves 0.669 → 1.219, Δ̂ +0.550 [+0.211, +0.889], p 0.0006, Holm p 0.0018 ≤ 0.005: VETO.** Conflicts per episode +0.67 [+0.02, +1.32], Holm 0.088 (not R). High-priority flow −7.1 (R, decrease) | **REJECT** |
| **FR** (RRT\*, R + S + D) | 0.9937 → 0.9962, Δ̂ +0.0025 [+0.0004, +0.0046], Holm p 0.018 (**R, gain**); P2 seed 2 +0.009 R (MC) | calls −1.60, common flow −21.1, supervisor changes −0.35 (all R) | **RO 0.553 → 0.927, Δ̂ +0.374 [+0.237, +0.511], p 0.0001, Holm p 0.0004: VETO.** High-priority undelivered: unbounded interval, not R | **REJECT** |
| **NA** (A\*, OFF) | 0.620 → 0.860, Δ̂ +0.239 [+0.228, +0.251] (**R**); P2 all 3 **R** | calls −9.2, common flow −29.0, stuck robots −3.26 (all R) | RO +0.068 [−0.52, +0.66], Holm 0.77 (no veto); high-priority flow −14.7 and undelivered −0.74 (R, decreases) | **ADOPT** |

**RO by policy pair** (nominal, not a test):
- FA: seed 1 +0.94 [+0.52, +1.37]; seed 2 +0.48 [+0.35, +0.61]; seed 3 +0.19 [−0.67, +1.06].
- FR: +0.70, +0.32 and +0.07, in the same order.

The increase is driven by policies 1 and 2, as on the development blocks.

**Secondary (nominal) in FA:**
- RO per episode +0.75 [+0.29, +1.21];
- RO per 1 000 moves under definition (ii, non-idle) +0.54 [−0.01, +1.09];
- RR conflicts (K^ctrl) +0.010 [−0.024, +0.044], no detectable change;
- conflicts per delivered task +0.035 [+0.001, +0.069];
- wait ratio −0.081.

### 4.3 H-L12: Issue 6A system vs Issue #6 system

The two systems compared:
- **Issue #6 system:** the Issue #6 PPO picks, OFF; the historical MADQL, OFF.
- **Issue 6A system:** v5 PPO + R + S + D; MADQL-v2 + R + S + D.

**A\* (3b): GO.**
- P1: MADQL +0.377, IPPO +0.014, MAPPO +0.012, all **R**. P2: all 9 policies **R**, no loss.
- P4, MADQL:
  - RO 0.804 → 1.219, Δ̂ +0.415 [−0.10, +0.93], Holm 0.225: **not rejected, so no veto**;
  - high-priority flow −19.9 (R).
- P4, PPO:
  - RO for MAPPO **falls** from 26.4 to 1.4 per 1 000 moves (R, a decrease), because the Issue #6 MAPPO picks propose very many RO conflicts. IPPO +0.12, not rejected;
  - high-priority undelivered falls in every family (R; the MAPPO row is MC-sensitive).

**RRT\* (3e): GO.**
- P1: +0.290 / +0.007 / +0.004, all **R**.
- P4: no increase rejected. RO for MADQL +0.05; the MAPPO RO decrease is not rejected (Holm 0.034 > 0.005).

**Why H-L12's MADQL RO row did not veto although H-v2 FA did:**
- The reference arms differ.
- H-L12 compares MADQL-v2 + R + S + D with the historical MADQL **without** the supervisor, whose Block V RO rate (0.804) is higher than with the supervisor (0.669, the H-v2 reference).
- This is the pre-registered design. Neither result is reinterpreted.

### 4.4 Context, pre-registered and non-gating: MADQL-v2 vs the forward PPO policies (both R + S + D)

| | A\* | RRT\* |
|---|---|---|
| RO / 1 000 moves, PPO → v2 | 1.988 → 1.219: Δ̂ −0.77 [−1.30, −0.24] | 1.915 → 0.927: Δ̂ −0.99 [−4.08, +2.11] |
| RO per episode | −0.41 [−0.94, +0.12] | −0.86 [−4.59, +2.88] |

MADQL-v2's per-move RO hazard is **lower** than that of the PPO policies in the forward system. By D3 / D4 this comparison does not gate anything, and the veto stands.

---

## 5. Interpretation (within the confirmatory scope)

1. **Supervisor (confirmed):**
   - Without changing any checkpoint, R + S + D raises completion from 0.620 to 0.996 (MADQL) and to 0.998 (PPO) on A\*.
   - It removes almost all frozen, oscillating, conflict-loop and deadlock-cycle robots.
   - It improves high-priority service.
   - It does **not** detectably raise the robot–obstacle hazard. The largest point estimate is IPPO +0.40 per 1 000 moves (not rejected); the smallest increase the test could reliably detect is ≈ 0.9.
   - It costs about 1 % of robot-steps changed (MADQL) and ≈ 0.1 % (PPO).
   - **Practical size:** about 6 more delivered tasks per episode for MADQL, and ≈ 0.2 for PPO.
2. **MADQL-v2 (not confirmed):**
   - The retrained MADQL is more efficient: with the supervisor it has 16–21 fewer flow-time steps, fewer replans, less than half the waiting (wait ratio 0.130 → 0.049) and fewer interventions.
   - Without the supervisor it completes far more: 0.86 vs 0.62.
   - But with the supervisor its robots step into dynamic obstacles' paths **≈ 1.7–1.8× as often per move** as the historical MADQL's (+0.55 and +0.37 per 1 000 moves). This is detected with high confidence on held-out data, so the pre-registered veto applies.
   - Every such conflict was stopped by the arbiter (0 collisions).
   - The increase is concentrated in two of the three training seeds.
   - The pre-registered context comparison shows the rate is still below the PPO policies' rate.
   - Whether that hazard level is acceptable was deliberately **not** made part of the rule. It is a decision for the owner after Phase 6, not a re-reading of this result.
3. **Integrated system (confirmed as declared):**
   - Issue 6A's combination is clearly better than the Issue #6 system on completion, for every family and planner, and on high-priority service, with no veto.
   - The PPO part combines the confirmed H-L1 picks with the confirmed supervisor, and it removes the Issue #6 MAPPO picks' very high RO and conflict rates.
   - Because MADQL-v2 is not confirmed, the forward system uses the historical MADQL + R + S + D. For that MADQL component, H-L2 gives +0.376 completion with no veto.

**What the evidence does not support:**
- that MADQL-v2 is safe or acceptable in the RO sense;
- that the forward system's MADQL component was tested as part of H-L12 (it was tested in H-L2);
- generalization beyond this scenario family (32 × 32 maps, density 0.15, 8 robots, 16 tasks), these checkpoints and three training seeds per configuration;
- equivalence for any non-rejected row.

---

## 6. Limitations and deviations

- **D12 is not satisfied (owner waiver W1).** The checkpoints the result depends on have no verified independent backup. Losing them would prevent reproduction or a re-run.
- **Amendment A1** moved the first Block V generation behind the execution authorization. This was a timing change only.
- **Instance-level disjointness from training maps** is not verifiable.
- **Multiplicity:** per-unit control. The global bound is 0.302, and confirmations are conjunctions.
- **MC-sensitive rows:**
  - 3 rejected P3 rows (H-L2 A\*);
  - the rejected MAPPO high-priority-undelivered row (H-L12 A\*);
  - FR P2 seed 2;
  - the non-rejected MAPPO high-priority-flow row (H-L2 A\*).

  None changes a verdict.
- **Unbounded intervals:** the high-priority-undelivered rows of FR and of the MAPPO RRT\* family. They are not rejected and not read as equivalence.
- **Power:** the MADQL RO rows on RRT\* are low-power. The IPPO RO row under the supervisor (A\* +0.40, p 0.055) is not significant and should not be read as "no effect".
- **RRT\*:** 60 seeds, so its rows are less precise than A\*'s.

---

## 7. Reproduction

All commands use `PYTHONPATH="src;tools/selection;tools/phase3;tools/phase1;tools/phase2;tools/phase4;tools/phase6"`.

```
python tools/phase6/phase6.py freeze --authorized-freeze      # 2026-11-23 15:33:05
python tools/phase6/phase6.py check  --authorized-precheck
python tools/phase6/phase6.py run    --workers 10 --authorized-execution
python tools/phase6/phase6.py replay --workers 10 --authorized-execution
PYTHONHASHSEED=0 python tools/phase6/phase6_analysis.py      # twice; byte-identical
python tools/phase6/phase6_report.py
```

| Output (`results/dev6a/phase6/eval/`) | SHA-256 (prefix) |
|---|---|
| `plan_freeze.json` | `db5bc0799aeb2a75` |
| `check.json` | `04e805418ac70fb4` |
| `instance_check.json` | `4cee5fd9dd211988` |
| `records.jsonl` | `ea02baeebb99b69f` |
| `replay.json` | `4d09b180919d3e5d` |
| `analysis.json` | `996bc8886fa385b2` |
| `tables.md` | `f78deb522fb9bae4` |
