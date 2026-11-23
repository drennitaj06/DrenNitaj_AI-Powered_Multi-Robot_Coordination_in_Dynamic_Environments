# Issue 6A Phase 1: Replanning optimization (P1, repair before replanning)

*Sections 1–3 were written **before** any formal Phase 1 episode was run (pre-registration). The results are in §4 onward, added after the run.*

**Status: complete (2026-11-22).**
- Verdict: **reject** in all four conditions (§6). The cuts in planner calls and work are large, but MADQL completion falls and high-priority flow time rises for every family.
- The repair stays in the library, off by default.
- Tables: `2026-11-22-issue-6a-phase1-replanning-tables.md`.

Labels: **[M]** measured; **[C]** follows from the code; **[I]** interpretation; **[A]** taken from an approved document.

---

## 1. Approved scope

**[A] Sources:**
- proposal §6 (P1), §10, §11 step 6, §13, §26.5 and §26.7;
- the roadmap (§27, priority 3: "Phase 1: Replanning optimization | Replan repair (P1)");
- the post-Issue-6 analysis, item A2.

| Element | What the approved documents say |
|---|---|
| Intervention | "**Repair before replanning (T3)**: rejoin the remaining reference path within a bounded radius, or follow the static cost-to-go; replan only if repair fails or dynamic cells changed" (P1). "Rejoin the reference path by a bounded local search" (A2) |
| Default | "changes monitor semantics (**off by default**)"; "defaults ⇒ identical" (§13) |
| Objective | "**Fewer planner calls** for learned controllers; **less RRT\* time**" (P1); "2–6× fewer planner calls" (A2). Phase 0: T3 is 62–95 % of learned-controller calls, and replanning optimization "would mainly cut computation … it would not remove the failures" (§26.7) |
| Validation | "Call counts and step time … **unchanged completion within a pre-set margin**" (A2); "planner calls; time per call" (§13); data: Block C (§11 step 6) |
| Not in scope | P2 (RRT\* budget control, Phase 2), P3 (A\* heuristic, Phase 5), P4 (reservations), retraining |

**What the documents leave open, and how it is resolved here** (no new objective or threshold is introduced):
- **Which repair option.** The bounded rejoin is the first-listed P1 option and the one named in A2, so it is the one implemented. "Follow the static cost-to-go" is not implemented.
- **The radius.** It is set to `H_r` = 3, the model's existing rejoin window (model §11, `MonitorParams.rejoin_window`), with no tuning. A path within `H_r` cells ahead already counts as "on the path"; the repair extends that tolerance to connectors of the same length.
- **The "pre-set margin".** δ₁ = one task per episode is used. It is the practical-significance unit of the Stage 3a criteria the student approved for Phase 3 (freeze §6.3).
- **Statistics and decision structure.** These are the approved Stage 3a / Phase 3 criteria (G0 gates, families with Holm, the no-harm rule, vetoes), applied to the Phase 1 objective (§3.4).
- **The supervisor.** The approved forward configuration **R + S + D** (Phase 3 §11) is the primary condition. The unsupervised composition, where F6 was measured, is secondary.

---

## 2. Evidence and implementation

### 2.1 Bottleneck evidence [M]

**Phase 0** (Block B, no supervisor):
- T3 (off-path) is 94.6 % of MADQL planner calls and 62–65 % of PPO calls.
- MADQL makes 33.1 calls per 100 robot-steps; PPO makes 10.8–11.3.
- Oscillating robots replan about every other step.

**Phase 3** (Block C records, with the supervisor):

| | T3 share of planner calls |
|---|---|
| A\*, R + S + D (arm R_S_D) | MADQL 84 %, IPPO 60 %, MAPPO 59 % |
| RRT\*, full supervisor | 55–77 % |

- Total calls per 100 robot-steps (A\*, R_S_D): MADQL 21.3, PPO 12.0–12.2.
- An RRT\* call uses its whole 1 000-iteration budget, at 107–156 ms per call (§26.6).
- So T3 replanning remains the largest share of planning work even under the supervisor.

### 2.2 Implementation [C]

| Change | File | Default |
|---|---|---|
| `MonitorParams.repair_radius` (≥ 0; 0 = off); `PathMonitor._repair`; counters `repairs` and `repair_failures` | `src/robocoord/planning/monitor.py` | off |
| Additive planning statistics `repairs` and `repair_failures`, only when repair is on | `src/robocoord/planning/coordinator.py` | — |
| `EvalTask.repair` (non-record field); it sets the coordinator's monitor parameters. Additive record fields `repair_radius`, `repairs`, `repair_failures` and `calls_by_trigger`, only when on | `src/robocoord/marl/evaluation.py` | 0 |

**Repair algorithm.** On a T3 deviation, and only when the plan is not degraded:
1. A breadth-first search runs from the robot over statically free cells, outside the robot's dynamic block set `B_i(t)` and proximity set `Q_i(t)` (the planner's own constraint snapshot, `build_constraints`).
2. It searches up to `repair_radius` moves for any cell of the remaining reference path.
3. It takes the shortest connector; ties go to the cell furthest along the path.
4. The new path is the connector plus the rest of the old path.
5. The new path is accepted only if it passes the T2 check, i.e. the next `H_v` cells are free of `S(t) ∪ B_i(t)`.
6. Otherwise the original T3 planner call is made.

Bookkeeping and properties:
- A repaired deviation still counts as a deviation (`deviations`, so the model §9.3 deviation rate keeps its meaning), but it is not a planner call.
- Neighbour order is canonical, so the repair is deterministic.
- Planner RNG streams are keyed by (t, robot), so skipped calls do not shift any other call's randomness.

**Tests:** `tests/unit/planning/test_monitor_repair.py` (8):
- validation; one-step repair; radius 0; the fallback when the path is too far;
- obstacle-blocked repair; the zero-length repair beyond `H_r`; degraded plans;
- a property test (150 examples): paths stay 4-connected over free cells and end at the goal, every T3 is either repaired or replanned, and the monitor is deterministic.

The existing monitor, planning and Phase 3 regression suites pass unchanged (repair off ⇒ identical).

**Pre-run smoke check [M]** (one episode each; not evidence about effects):
- The repair-off arms reproduce stored Phase 3 records exactly (A\* R_S_D, A\* off, RRT\* off).
- One unsupervised MADQL episode delivered 12 / 28 with repair against 23 / 28 without.
- This is why the protocol judges completion, not planner calls alone.

---

## 3. Protocol (pre-registered)

### 3.1 Conditions and arms

| Condition | Planner | Supervision | Seeds × levels | Arms | Episodes |
|---|---|---|---|---|---|
| **C1 (primary)** | A\* | R + S + D (`supervisor="R+S"`, `deadlock=True`, frozen Stage 3a parameters) | Block C 11 000–11 119 × 3 | repair 0 (`rsd_r0`) / 3 (`rsd_r3`) | 2 × 3 240 |
| C2 | A\* | none, resolver off | same | `off_r0` / `off_r3` | 2 × 3 240 |
| **C3 (primary)** | RRT\* | R + S + D | Block C RRT\* subset 11 000–11 039 × 3 | `rsd_r0` / `rsd_r3` | 2 × 1 080 |
| C4 | RRT\* | none | same | `off_r0` / `off_r3` | 2 × 1 080 |
| **Total** | | | | | **17 280** |

**Shared settings:**
- the 9 policies and checkpoints of Phase 3 (MADQL Issue #5 picks and the PPO SCB-C v5 picks; hashes in `check.json`);
- greedy mode, auction allocator, shield off, endpoint recovery on;
- `path_limit` 6, horizon 400, RRT\* with 1 000 iterations;
- the production `run_episode`;
- the stage-1a runner job order is interleaved by seed.

### 3.2 Outcomes (ratio of sums over seeds; seed is the unit; family estimates pool the family's 3 policies)

| Family | Outcome | Hypotheses per condition | α |
|---|---|---|---|
| P1 | completion (Σ delivered / Σ released), per controller family | 3 | 0.03 |
| P2 | completion per policy | 9 | 0.005 |
| E | planner calls per 100 robot-steps; planner work per robot-step (A\*: node expansions; RRT\*: iterations; deterministic and hardware-independent), per family | 6 | 0.01 |
| P4 (vetoes) | conflicts per episode; high-priority flow time on common-delivered tasks; high-priority undelivered tasks, per family | 9 | 0.005 |

- The union bound is ≤ 0.05 per condition.
- Estimator, bootstrap and tests: the frozen `block_c_analysis.py` (B = 10 000, `default_rng(0)`, bootstrap-t, Holm within each family), with the MC-sensitivity flag and the not-evaluable rule, as in Phase 3.
- **Descriptive:**
  - calls by trigger, the repair success share, and the deviation rate;
  - planner wall time per robot-step (median and IQR; not tested, because wall time is affected by system load) and step time;
  - stuck-robot categories, undelivered tasks, successful episodes and supervisor interventions;
  - per level, and seed heterogeneity.

### 3.3 Integrity gates (G0, as Phase 3)

Each must hold:
- **G0.1:** the repair-off arms reproduce the stored Phase 3 records exactly for A\* R_S_D, A\* off and RRT\* off. RRT\* R + S + D has no stored counterpart.
- **G0.2:** every 50th task replays identically without the observer.
- **G0.3:** 0 errors, collisions and transition errors.
- **G0.4:** 0 supervisor-introduced invalid moves.
- **G0.5:** the fairness bound holds.
- **G0.6:** the burst bound holds.
- **G0.7:** the frozen supervisor parameters are used.
- Checkpoint hashes are unchanged.

### 3.4 Decision per condition

1. **No harm** requires all of the following:
   - G0 passes;
   - no P1 or P2 completion hypothesis is Holm-rejected with Δ̂ < 0;
   - the nominal 95 % lower bound of Δ completion is > −δ₁ for every family (δ₁ = 1 / mean released per repair-off episode);
   - no P4 veto (a Holm-rejected increase).
2. **Benefit:** at least one family's E hypothesis (calls or work) is Holm-rejected with Δ̂ < 0.
3. **Verdict per condition:**
   - **"adopt"** if there is no harm and a benefit;
   - **"no benefit"** if there is no harm but no benefit;
   - **"reject"** if a completion loss or a veto is Holm-rejected, or G0 fails;
   - **"inconclusive"** otherwise (for example, a lower bound ≤ −δ₁).
4. **Phase 1 recommendation:** P1 is recommended as the forward default only if **C1 and C3 are "adopt"**. C2 and C4 tell whether repair is safe without the supervisor; they are reported but do not decide the default, because the approved forward configuration includes the supervisor. Either way the repair stays a configuration option, off by default in the library, and nothing is re-tuned.

### 3.5 Anti-tuning

- The radius (3) is fixed. No Phase 1 result may change it, the supervisor parameters or the Phase 3 configuration.
- Block C is the development block. A result here is development evidence; its confirmation belongs to Phase 6.

---

## 4. Execution and integrity [M]

**Runs (2026-11-22):**
- `check`: environment, checkpoints, frozen parameters, Stage 3a freeze and code hashes were recorded in `check.json`.
- Stage 1a (A\*, C1 and C2): 12 960 episodes in 1 163 s, 0 errors.
- Stage 1b (RRT\*, C3 and C4): 4 320 episodes in 8 629 s, 0 errors.
- Checkpoint hashes were identical before and after each stage.
- The code-hash gate held: all 12 gated files still match `check.json` after the run.

| Gate | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| G0.1: repair-off arms reproduce Phase 3 | 3 240 / 3 240 identical (3c `R_S_D`) | 3 240 / 3 240 (3b `off`) | no stored counterpart (§3.3) | 1 080 / 1 080 (3e `off`) |
| G0.2: replay without the observer (every 50th task, both stages) | 346 / 346 identical | ← | ← | ← |
| G0.3: errors, physical collisions, transition errors | 0 | 0 | 0 | 0 |
| G0.4: supervisor-introduced invalid moves | 0 | — | 0 | — |
| G0.5 / G0.6: fairness and burst bounds | hold | hold | hold | hold |
| G0.7: frozen parameters; repair radius 3 on repair arms only | yes | yes | yes | yes |
| Instances identical across arms | yes | yes | yes | yes |

**G0 passes in every condition.**

Two numerical notes:
- Some per-policy RRT\* rows (C3 P2: five policies; C3 P4: MAPPO high-priority undelivered) have a repair-off value of exactly 1.0 or 0. Bootstrap resamples with no failures give se = 0, so the frozen bootstrap-t interval is (−∞, ∞). The p-value remains valid (conservative). None of these rows is rejected, and none decides a verdict. The frozen tool also stores the percentile interval in `analysis.json`.
- The makespan of C4 `off_r3` MADQL is undefined, because that arm has no successful episode.

## 5. Results [M]

Δ = repair on (radius 3) − repair off. Completion uses the ratio of sums with seed as the cluster. Intervals are nominal 95 % bootstrap-t intervals. "Holm" means rejected within the family at its pre-registered α. All tables are in `2026-11-22-issue-6a-phase1-replanning-tables.md`.

### 5.1 Efficiency (family E): large reductions, all Holm-rejected

| Condition | Family | Calls / 100 robot-steps, off → on | Work / robot-step, off → on | T3 share of calls, off → on |
|---|---|---|---|---|
| C1 (A\*, R + S + D) | MADQL | 21.29 → 3.06 | 3.60 → 0.93 expansions | 84 % → 13 % |
| | IPPO | 12.24 → 4.51 | 3.05 → 1.37 | 60 % → 6 % |
| | MAPPO | 11.99 → 4.60 | 2.90 → 1.39 | 59 % → 6 % |
| C3 (RRT\*, R + S + D) | MADQL | 18.98 → 2.78 | 190 → 28 iterations | 82 % → 11 % |
| | IPPO | 12.12 → 4.31 | 121 → 43 | 58 % → 5 % |
| | MAPPO | 11.34 → 4.61 | 113 → 46 | 55 % → 5 % |

- The repair succeeds on 98–99 % of T3 deviations (99 % without the supervisor).
- C2 and C4 show the same pattern, with calls per 100 robot-steps falling by 75–95 %.
- All 6 E hypotheses are Holm-rejected in every condition (p_Holm ≤ 0.0006). **The benefit criterion is met everywhere.**

### 5.2 Completion (P1, P2)

| Condition | MADQL | IPPO | MAPPO |
|---|---|---|---|
| **C1** | 0.9970 → 0.9865; Δ −0.0104 [−0.0127, −0.0081], **Holm** | 0.9985 → 0.9980; Δ −0.0005 [−0.0012, 0.0002] | 0.9982 → 0.9981; Δ −0.0001 [−0.0008, 0.0006] |
| C2 | 0.624 → 0.446; Δ −0.178, **Holm** | 0.987 → 0.886; Δ −0.101, **Holm** | 0.988 → 0.902; Δ −0.086, **Holm** |
| **C3** | 0.9914 → 0.9531; Δ −0.0383 [−0.0455, −0.0311], **Holm** | 1.0000 → 0.9987; Δ −0.0013 [−0.0037, 0.0012] | 0.9993 → 0.9986; Δ −0.0007 [−0.0020, 0.0006] |
| C4 | 0.695 → 0.410; Δ −0.286, **Holm** | 0.992 → 0.897; Δ −0.095, **Holm** | 0.996 → 0.923; Δ −0.073, **Holm** |

**With the supervisor (C1, C3):**
- MADQL loses completion in every policy. All three P2 hypotheses are Holm-rejected in both conditions.
- In C1, successful episodes fall from 1 026 to 888 of 1 080. Undelivered tasks per episode rise from 0.058 to 0.258, and stuck robots from 63 to 279.
- In C3, successful episodes fall from 315 to 193 of 360.
- The PPO families show no detected completion difference. Their lower bounds are well above −δ₁ = −0.052.
- Per level (C1, nominal), the MADQL loss is largest on *full* (−0.019) and is also present on *static* (−0.005).

**Without the supervisor (C2, C4):**
- The losses are large for every family and every policy (all Holm-rejected).
- They exceed δ₁ for all families.
- PPO successful episodes fall from about 900 to 308–362 of 1 080 on A\*.

### 5.3 Vetoes (P4)

**High-priority flow time on commonly delivered tasks rises in every family and every condition (Holm, p_Holm ≤ 0.0009):**

| Family | C1 | C3 |
|---|---|---|
| MADQL | +17.3 steps on 57.5 (+30 %) | +27.5 on 64.4 (+43 %) |
| IPPO | +9.1 on 38.9 (+23 %) | +10.0 on 45.6 (+22 %) |
| MAPPO | +7.3 on 38.7 (+19 %) | +5.8 on 44.6 (+13 %) |

- Episodes are correspondingly longer. With the supervisor, PPO episodes take +23 to +41 steps and MADQL +74 to +86.
- High-priority undelivered tasks increase in C2 and C4 for all families (Holm). They do not increase in C1 or C3.
- **Conflicts per episode:** no change is detected in any condition. All 12 conflict hypotheses are not rejected, with wide intervals.
- There were no safety events in any arm (G0.3).

### 5.4 Runtime and overhead (descriptive; wall time is not tested)

Medians per episode, excluding observer time:

| Condition | Family | Planner work / ep., off → on | Planning s / ep. | Wall s / ep. |
|---|---|---|---|---|
| C1 (A\*) | MADQL | 6 684 → 2 277 expansions | 0.080 → 0.019 | 0.811 → **0.977** |
| | IPPO | 3 681 → 2 025 | 0.034 → 0.016 | 0.453 → **0.558** |
| | MAPPO | 3 507 → 2 006 | 0.033 → 0.016 | 0.452 → **0.555** |
| C3 (RRT\*) | MADQL | 409 250 → 78 917 iterations | 41.0 → 7.7 | 41.8 → 8.7 |
| | IPPO | 161 211 → 70 692 | 15.7 → 6.9 | 16.3 → 7.5 |
| | MAPPO | 152 308 → 69 750 | 15.1 → 7.1 | 15.6 → 7.6 |

- **A\*:** planning is only 6–9 % of step time, so the cut in planning work cannot pay for the longer episodes. Wall time per episode *rises* by about 20–23 %.
- **RRT\*:** planning is 97–98 % of step time, so wall time per episode falls 2.0–4.8×, even though episodes are longer.
- The repair itself is cheap: step time per step falls in every repair arm.

### 5.5 Mechanism (descriptive [M]; interpretation [I])

**[M]:**
- With the supervisor, repairs per episode (PPO 144–194, MADQL 478–610) far exceed the T3 calls they replace (PPO 84–94 per episode, MADQL 333–335).
- The deviation rate rises by 1.9–6.2 deviations per 100 robot-steps.
- Supervisor interventions rise from 0.5–1.2 to 4.3–8.4 per episode for PPO, and from 15–19 to 44–45 for MADQL.
- Without the supervisor, the PPO deviation rate more than doubles (MADQL: +15 %), and repairs reach 370–1 170 per episode.

**[I]** A fresh T3 plan starts from where the controller has taken the robot. The repair instead sends the robot back to the remaining old reference path, often a step behind or to the side. The controllers see the reference path in their observation, so they deviate from it again. Each repair ends one deviation only to set up the next.

The result is a repeat cycle of deviation → repair → deviation. It shows up as:
- more deviations;
- longer routes and flow times;
- more stall and oscillation interventions;
- without the supervisor, more stuck robots.

MADQL deviates most (Phase 0, F6), so it is affected most. The supervisor absorbs much of the damage for PPO but not for MADQL. This interpretation is consistent with the counters, but it was not tested separately.

## 6. Decisions under the pre-registered rules (§3.4)

| Condition | G0 | Completion harm (Holm, Δ̂ < 0) | Lower bound > −δ₁ | P4 veto | Benefit (E) | **Verdict** |
|---|---|---|---|---|---|---|
| **C1** (primary) | pass | MADQL family + 3 policies | all families | high-priority flow, all 3 families | all 6 | **reject** |
| C2 | pass | all families + all 9 policies | none | high-priority flow and undelivered, all families | all 6 | **reject** |
| **C3** (primary) | pass | MADQL family + 3 policies | all families | high-priority flow, all 3 families | all 6 | **reject** |
| C4 | pass | all families + all 9 policies | none | high-priority flow and undelivered, all families | all 6 | **reject** |

**Phase 1 recommendation:** P1 (bounded rejoin, radius `H_r` = 3) is **not recommended as the forward default**.
- C1 and C3 are "reject", not "adopt".
- The reductions in planner calls and planner work are real and large, but they come with a completion loss for MADQL and longer high-priority flow times for every controller family.
- On A\*, total episode wall time also increases.

As pre-registered:
- The repair stays in the library as an option, **off by default** (`repair_radius = 0`; behaviour is byte-identical when off, G0.1).
- Nothing was re-tuned. The radius, the supervisor parameters and the R + S + D configuration are unchanged.
- No other radius or repair variant was tried on Block C (§3.5).

This is development evidence on Block C. A rejection does not need Phase 6 confirmation to be acted on, because the forward configuration simply keeps repair off.

## 7. Limitations

- **One repair variant and one radius.** The second P1 option ("follow the static cost-to-go") and forward-only rejoining were not evaluated. §5.5 suggests that the backwards pull of the stale path, not the idea of avoiding replans, is what causes harm. A different variant would need its own pre-registered protocol and approval. It must not be tuned on Block C after seeing these results.
- **Development block.** Block C has been used repeatedly (Phase 3, Phase 1). These estimates are not confirmatory.
- **RRT\* subset.** C3 and C4 use 40 seeds. The high RRT\* PPO completion near 1.0 makes some per-policy bootstrap-t intervals degenerate (§4).
- **Wall time** is measured on one machine under background load and was not tested. The deterministic planner-work counts are the efficiency evidence.
- **C3 baseline.** The C1 repair-off arm is exactly the Phase 3 `R_S_D` arm (G0.1). The C3 repair-off arm has no stored counterpart, because Phase 3 ran RRT\* only with the full supervisor and without one.
- **Mechanism (§5.5)** is an interpretation of counters, not a tested hypothesis.
- **Post-run addition.** The per-episode totals in §5.4–§5.5 (planner work, wall time, repairs and supervisor changes per episode) were added to the descriptive output of `phase1_analysis.py` after the run. They are descriptive only: no test, α or decision depends on them.

## 8. Recommendations

1. Keep `repair_radius = 0` (off) as the forward default. The approved forward configuration stays **R + S + D with T3 replanning**.
2. Do not use P1 to reduce computation in Phase 6 or later phases.
3. RRT\* computation remains the cost bottleneck: 97–98 % of step time and 104–174 ms per step with the supervisor. That is the subject of **Phase 2 (P2, RRT\* budget control)**, which does not depend on P1.
4. A forward-only or cost-to-go repair could be proposed later as a separate, pre-registered item, if the student decides to. It is not part of Phase 1.

## 9. Reproduction and provenance

```
python tools/phase1/phase1.py check  --out results/dev6a/phase1/eval
python tools/phase1/phase1.py run    --out results/dev6a/phase1/eval --stage 1a --workers 11
python tools/phase1/phase1.py run    --out results/dev6a/phase1/eval --stage 1b --workers 11
python tools/phase1/phase1.py replay --out results/dev6a/phase1/eval --workers 11
python tools/phase1/phase1_analysis.py --out results/dev6a/phase1/eval
python tools/phase1/phase1_report.py   --out results/dev6a/phase1/eval
```

`run` is resumable. It refuses to continue if any gated code file changed since `check`.

| Artifact (`results/dev6a/phase1/eval/`) | SHA-256 |
|---|---|
| `check.json` | `c1276332eacf34bf6b303f28321a18648821d8fe0bc2f49439f74314afc052a4` |
| `records_1a.jsonl` (12 960) | `8c4c6c31fbb77b4856dd936644d759a21ae31284753bb5e8bfb772beaf0d709a` |
| `records_1b.jsonl` (4 320) | `60621df00f21bbc5f53265687134fbfd66a52d57527eb90dc97b2d07b61f21ae` |
| `replay.json` | `513635feec114132294e2377bae1a6a660d1a36c1daef727fa4ea3e8543ffb8f` |
| `analysis.json` | `24e5329754ac426180eb84855fd06481ae87c95065a28a298431cd4ea636d64b` |
| `tables.md` | `6fbb263d17651b5865ccc47620ec667e29d811506e54e864c34c8777cae2dbe0` |

**Gated code at run time:**
- `monitor.py`: `f9abfd52…`
- `coordinator.py`: `731ec221…`
- `evaluation.py`: `67b3b0ac…`
- `supervisor.py`: `83b84b8e…`, unchanged since the Stage 3a freeze
- `phase1.py`: `51283e33…`

The full list is in `check.json`.

**Seeds and other inputs:**
- Block C 11 000–11 119 (A\*); the RRT\* subset 11 000–11 039.
- 3 levels and 9 policies.
- Checkpoint hashes are in `check.json`.

## 10. Status

Phase 1 is complete (2026-11-22). The protocol was executed as pre-registered, G0 passed, and the verdict is **reject** in all four conditions. The repair is kept as a configuration option, off by default.
