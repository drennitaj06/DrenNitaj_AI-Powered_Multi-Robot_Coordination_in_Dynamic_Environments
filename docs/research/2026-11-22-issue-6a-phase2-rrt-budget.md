# Issue 6A Phase 2: RRT\* budget control (P2)

*Write order:*
- **§1–3** were written before any Phase 2 microbenchmark was run. §3 is the candidate set and the selection rule.
- **§4** (microbenchmark results and the selection) was added afterwards.
- **§5** (the Block C protocol) was written after the selection and before any Block C episode.
- **§6 onward** (results) were added after the Block C run.

Labels: **[M]** measured; **[C]** follows from the code; **[I]** interpretation; **[A]** taken from an approved document.

---

## 1. Motivation, approved scope and hypothesis

**[A] Sources:**
- proposal §6 (P2), F7, §13, §15 ("RRT\* guarantees (P2): must be restated if adopted"), §26.6 and §26.7 item 4;
- the post-Issue-6 analysis A3 / B5;
- the roadmap (§27, priority 4: "Phase 2: RRT\* optimization | Budget control (P2)");
- the student's Phase 2 approval (2026-11-22).

| Element | What the approved documents say |
|---|---|
| Intervention | "**RRT\* budget control**: first solution plus a refinement budget, **or** a cost-improvement stopping rule; informed sampling after the first solution" (P2). "Early termination (first solution + bounded refinement or a cost-improvement criterion), informed sampling after the first solution, tree reuse across replans" (A3) |
| Evidence | F7: every call spends the whole budget (125–173 ms per call, E2). §26.6: the first solution comes at a median of iteration 40–446 on 32×32 maps, so 78–98 % of the budget is spent after it. Phase 1 §5.4: RRT\* planning is 97–98 % of step time in learned-controller episodes |
| Expected effect | "Large per-call savings"; "path cost may rise slightly; anytime-optimality claims must be restated" (A3) |
| Validation | "E1a success and cost vs budget; pilot projection" (A3); "planner calls; time per call" and "defaults ⇒ identical" (§13); data: Block C, E1a/E2 query sets (§11 step 6) |
| Not in scope | tree reuse (P6, "only if P1 + P2 insufficient"); P1 (rejected in Phase 1, stays off); A\* changes (P3); retraining |

**Hypothesis.** A deterministic early stop after the first solution cuts RRT\* iterations and planning time per call substantially. Path cost rises only slightly, and system-level completion, delivery times, coordination and safety stay unchanged.

**Scope decisions [C/A].**
- **Implemented: both stopping rules the proposal names as alternatives.**
  - (a) first solution plus a refinement budget k (`refine_iterations`);
  - (b) a no-improvement rule (`patience` m).
  - Both are deterministic iteration counts, never wall time. One configuration is selected for the system by the pre-declared rule of §3.
- **Not implemented: informed sampling.**
  - It is an optional add-on in P2, and no evidence motivates it here: E1a shows the cost gap is small and the waste is the iterations themselves.
  - It changes the sampling distribution, so a stopped run would no longer be a prefix of the full run. That would weaken the guarantees in §2.2.
  - It interacts with the penalised lattice cost, because the Euclidean ellipse is not a bound on proximity-penalised costs.
- **Not implemented: tree reuse.** That is P6, which is deferred.

## 2. Implementation and guarantees

### 2.1 Changes [C]

| Change | File | Default |
|---|---|---|
| `RRTStarConfig.refine_iterations` (int ≥ 0 or `None`) and `RRTStarConfig.patience` (int ≥ 1 or `None`); the stop check at the end of an iteration | `src/robocoord/planning/rrtstar.py` | `None` (off) |
| `EvalTask.rrt_refine` and `EvalTask.rrt_patience` (non-record fields). They set the rules on the coordinator's planner, which is the object the path monitor uses. Additive record fields `rrt_refine` and `rrt_patience`, only when set | `src/robocoord/marl/evaluation.py` | `None` |

`make_planner("rrtstar", refine_iterations=…, patience=…)` works through the existing option registry.

**Rules.** Let `i₁` be the first iteration with a solution and `b(i)` the best goal cost after iteration i.
- **Refinement budget k:** stop after iteration `i₁ + k`. With k = 0 this equals the existing `stop_on_first_solution`.
- **Patience m:** stop after the first iteration i ≥ `i₁` with `i − ℓ(i) ≥ m`, where `ℓ(i)` is the last iteration ≤ i at which b strictly decreased (`ℓ(i₁) = i₁`).
- Both rules are capped by `max_iterations`. If both are set, the first to trigger stops the run.

### 2.2 Guarantees (restated; `04` §3 and model G5–G9 updated)

The check runs only after a solution exists and only ends the loop. The random stream, sampling, tree construction and every iteration before the stop are unchanged.

A run with a stopping rule is therefore an **exact prefix** of the full-budget run on the same query and stream. This holds [C] and is tested on every random query in §2.3. It gives:
- **G5 kept.** The returned path is the best path of the prefix tree, so it is valid for the snapshot. The best cost is non-increasing over the iterations run, and the tree-cost invariant holds after every iteration.
- **No worse or invalid result because of the budget.** The returned cost equals the full run's best cost at the stop iteration. It is ≥ the full run's final cost and ≤ the first-solution cost. Stopping never returns a path that the full run would not also have held at that iteration.
- **Success and failure unchanged.** A query without a solution never triggers a rule. It runs the whole budget and fails with `budget_exhausted`, exactly as before. The two-stage procedure (constrained, then static fallback, then status mapping) and the BFS guard are unchanged.
- **Determinism.** Each planner call's stream is keyed by `(t, robot)` (`monitor.py`), so stopping one call early does not shift any other call's randomness.
- **Off ⇒ identical.** With both rules `None`, the code path is the original one.
- **G7–G9.** Asymptotic optimality (G7) was already not claimed. Under a stopping rule the run is additionally no longer anytime: it ends at a finite, data-dependent iteration, so convergence claims of E1c type do not apply to it. G8 (success within a finite budget) is still not claimed, and success is unchanged versus the same budget. G9 is unchanged.

### 2.3 Tests [M]

**`tests/unit/planning/test_rrtstar_budget.py` (8 tests):**
- default off and validation;
- disabled rules reproduce the full run exactly (path, cost, counters, trace);
- a Hypothesis property over random queries, both indexes, η ∈ {0.7, 1.5, 3} and k, m ∈ grids (80 examples):
  - the stopped run's best-cost trace equals the full trace's prefix;
  - its iteration count equals an independent re-derivation of the stop;
  - the first-solution iteration and the status are unchanged;
  - the path is valid; the cost is ≥ the full cost and non-increasing;
  - the tree invariant is checked every iteration;
  - the run is deterministic;
  - k = 0 equals `stop_on_first_solution`;
- edge cases:
  - no solution: the whole budget is run and the failure is unchanged;
  - a late first solution: capped by the budget; i₁ + 5;
  - repeated improvement: patience restarts after each strict decrease;
  - no improvement: stops exactly m after the last improvement;
  - both rules set.

**`tests/integration/test_rrt_budget_integration.py` (3 tests):**
- the `run_episode` switch is additive and off by default (records equal);
- the rule reaches the monitor's planner, and every call stops at `min(budget, i₁ + k)` or runs the whole budget when there is no solution;
- the switch requires RRT\*.

## 3. Microbenchmark protocol and selection rule (pre-declared)

### 3.1 Query sets and roles

| Set | Instances | Role in Phase 2 |
|---|---|---|
| **E1a** (`configs/experiments/e1a_planners_static.toml`, seed base 30 000) | 7 families × sizes {16, 32, 64, 128} × 5 maps × 20 pairs = 2 800 static queries, generated by the E1a functions (`make_map`, `make_pair`, planner stream `(key, 7)`) | **Selection** (§3.3) and the full microbenchmark report |
| **E1c** (`e1c_convergence.toml`, seed base 32 000) | 20 queries, 32×32 open and 20 % maps | Descriptive: the cost at the stop versus the stored E1c best cost at 1 000 and 20 000 iterations, and the exact optimum on open maps. Not used for selection |
| **E2** (`e2_replanning.toml`, seeds 20 000–20 019) | 320 RRT\* replanning episodes (plan follower, 24×24 maps, dynamic obstacles {0, 4, 8, 16} × events {none, low} × team {1, 4}) | **Run only after the selection is frozen**, as a reported comparison (full budget vs the selected rule). Its seeds lie in the reserved test block, which "is never used for tuning" (`05` §B.1; proposal §12, §15), so E2 plays no part in the choice |

**Budget:** all microbenchmarks use `max_iterations` = 1 000, the system's RRT\* budget (`evaluation.RRT_ITERATIONS`). The other options are the defaults: brute-force index, η = 2, goal bias 0.1, γ_f = 1.5. There is no wall-clock limit.

### 3.2 Candidates

| Rule | Values |
|---|---|
| Refinement budget k | 0, 25, 50, 100, 200, 400 |
| Patience m | 25, 50, 100, 200, 400 |

That is 11 candidates plus the full-budget reference, all run on the same queries and streams. Every candidate run is checked against the reference for the exact-prefix property (§2.2): trace prefix, iterations, first solution and status. Any mismatch is an integrity failure.

### 3.3 Selection rule (fixed before any microbenchmark result was seen)

1. **Stratum:** E1a sizes 16 and 32, all 7 families (1 400 queries). This is the map-size regime of the system: Block C maps are 32×32 at 15 % density. Larger sizes are reported but do not enter the choice.
2. **Quality screen:** for each family f, Q_f = Σ executed path steps (candidate) / Σ executed path steps (full), over the queries the full run solves (the same set for every candidate, §2.2). A candidate is eligible if **max_f Q_f ≤ 1.01**.
   - This 1 % is an engineering screening tolerance for the microbenchmark only. It is not a Phase 2 acceptance threshold.
   - Block C mean flow times are 40–65 steps per task, so 1 % is below one lattice step per task.
   - Acceptance is decided only at system level by the approved criteria (§5).
3. **Economy:** among eligible candidates, choose the one with the fewest mean iterations per query over the stratum. Failed queries count their whole budget.
4. **Ties** (equal mean iterations to 0.1 iterations): the larger parameter value, then the refinement rule over patience.
5. **If no candidate is eligible,** P2 stops at the microbenchmark stage with "no candidate" and Block C is not run.

The selected configuration is recorded with hashes in `results/dev6a/phase2/micro/selection.json` before §5 is written. No Block C outcome is examined before then, and no Block C outcome may change it.

---

## 4. Microbenchmark results and the selection [M]

Runs (2026-11-22, 10 workers; records in `results/dev6a/phase2/micro/`; tables in `micro_tables.md`):

| Run | Content | Integrity |
|---|---|---|
| E1a | 2 800 queries × 12 configurations = 33 600 RRT\* runs | **0 exact-prefix failures** (trace prefix, iterations, first solution, status, returned cost) |
| E1c | 20 queries × 12 configurations | 0 prefix failures. The full run's best cost at iteration 1 000 equals the stored E1c checkpoint value on **20 / 20** queries |
| Selection | `micro.py select`, rule §3.3 | `selection.json` (SHA-256 `224fc343…`), written before §5 |
| Timing | stratum, untraced, full vs selected, minimum of 3 repetitions | — |
| E2 | 320 + 320 episodes, run after the selection | The full-budget arm reproduces the historical E2 call totals exactly (9 462 calls, 185 budget-exhausted, proposal §26.6) |

### 4.1 E1a selection stratum (sizes 16 and 32, budget 1 000)

| Config | Q steps, pooled (max over families) | Executed paths identical to full | Mean iterations (vs 1 000) | Eligible |
|---|---|---|---|---|
| k0 (first solution) | 1.152 (1.222) | 43 % | 273 | no |
| k100 | 1.114 (1.176) | 55 % | 360 | no |
| k400 | 1.049 (1.100) | 77 % | 611 | no |
| m50 | 1.109 (1.170) | 56 % | 361 | no |
| m200 | 1.021 (1.055) | 90 % | 753 | no |
| **m400** | **1.0007 (1.0018)** | **99.3 %** | **944** | **yes (the only one)** |

Success is identical for every configuration (0.874), as §2.2 guarantees. The median first solution is at iteration 66. All 11 candidates are in `micro_tables.md`.

**Selected: patience m = 400** (`refine_iterations = None`, `patience = 400`).

**Findings [M]:**
- **Early stopping costs far more executed-path quality than "slightly" (A3).**
  - Stopping at the first solution makes executed lattice paths 15 % longer on average (up to 22 % on 10 % density maps, and up to +34 steps on one query).
  - Even k = 400 (about 40 % of the budget saved) loses 5–10 % by family.
- **[I] Why:** the iterations after the first solution are where the executed path is shortened. The continuous cost keeps improving in small steps, and these improvements change the discretised path.
- **The only candidate within the 1 % screen saves little.** m = 400 cuts mean iterations by 5.6 % in the stratum, and by 4–7 % by size.
  - Untraced per-call time falls by 10 %: median 104 → 89 ms (pooled 0.90×, 0.89–0.95× by family).
  - The no-improvement window rarely ends a run early, because the continuous best cost keeps decreasing by small amounts until late. That is also why m ≤ 200 lose quality.
- **All sizes:** with m = 400, Q is 1.0003–1.0016 and success equals the full budget at every size.

### 4.2 E1c (descriptive)

| Family | m = 400 cost / full@1 000 | full@1 000 / stored@20 000 | Executed steps m = 400 / full |
|---|---|---|---|
| open | 1.0000; never stopped early (1 000 iterations) | 1.078 | 1.000 |
| d20 | 1.0062; mean 897 iterations | 1.305 | 1.0095 |

### 4.3 E2 replanning (reported comparison; not used for selection)

Full budget → m = 400, over 320 paired episodes:

| Metric | full | m = 400 |
|---|---|---|
| Iterations per call | 998 | 946 (−5.2 %) |
| Total planning time | 1 301 s | 1 206 s (−7.3 %) |
| Median per-call latency | 138 ms | 133 ms |
| Completion | 0.818 | 0.830 |
| Flow time | 82.5 | 82.5 |
| Path efficiency | 0.862 | 0.857 |
| Budget-exhausted calls | 185 | 159 |

- Trajectories diverge after the first differing path, so the per-episode differences go both ways:
  - delivered: 22 episodes better, 11 worse;
  - steps: 38 episodes longer, 21 shorter.
- E2 has no inferential test (reserved seeds, plan follower). It shows no sign of harm, and a computation saving of about 5–7 %.

**Expected Block C effect [I]:** a small saving of about 5–10 % in iterations and RRT\* time. The protocol below tests whether that saving holds without system-level harm.

## 5. Block C protocol (pre-registered; written after §4 and before any Block C episode)

### 5.1 Conditions and arms

| Condition | Controllers | Supervision | Seeds × levels | Arms | Episodes |
|---|---|---|---|---|---|
| **L (primary)** | 9 policies: MADQL Issue #5 picks; IPPO and MAPPO SCB-C v5 picks (Phase 3 checkpoints) | **R + S + D** (`supervisor="R+S"`, `deadlock=True`, frozen Stage 3a parameters; O not used) | Block C RRT\* subset 11 000–11 039 × {static, obstacles, full} | `full` (1 000 iterations, no rule) / `p2` (patience 400) | 2 × 1 080 |
| **B** | non-learned baseline controller | its original composition: resolver on, no supervisor (the supervisor applies to learned controllers only) | same | `full` / `p2` | 2 × 120 |
| **Total** | | | | | **2 400** |

**Shared settings:**
- RRT\* only: A\* does not use the changed code, so no A\* runs are needed.
- P1 repair off everywhere (`repair = 0`).
- Greedy mode, auction allocator, shield off, endpoint recovery on, `path_limit` 6, horizon 400.
- The production `run_episode`.
- Paired by instance: the same `(run, level, seed)` in both arms, with the instance fingerprint checked.

**Reuse:** the existing Phase 1 RRT\* `rsd_r0` records are the reference for the `full` arm of L, not a substitute for it.
- The `full` arm is re-run, because the new observer collects per-call RRT\* data that Phase 1 did not record.
- G0.1 then requires the `full` arm to equal those Phase 1 records exactly.
- No stored baseline Block C RRT\* records exist, so B has no reproduction counterpart.

### 5.2 Outcomes

The methodology is Phase 1 §3.2 unchanged: the frozen `block_c_analysis.py` estimator, ratio of sums, seed cluster, family pooling of 3 policies, B = 10 000, `default_rng(0)`, bootstrap-t, Holm within each family, the MC-sensitivity flag and the not-evaluable rule.

| Family | Outcome | L: hypotheses | B: hypotheses | α |
|---|---|---|---|---|
| P1 | completion per controller family | 3 | 1 | 0.03 |
| P2 | completion per policy | 9 | — | 0.005 |
| E | planner calls per 100 robot-steps; RRT\* iterations per robot-step | 6 | 2 | 0.01 |
| P4 (vetoes) | conflicts per episode; high-priority flow time on commonly delivered tasks; high-priority undelivered tasks | 9 | 3 | 0.005 |

**Descriptive, reported by policy, family and level:**
- **Planner:** RRT\* calls, iterations per call, the share stopped early, status mix (ok / degraded / budget-exhausted / unreachable), mean first-solution iteration, executed moves per successful plan, per-call time (median), total planner time per episode, and tree nodes per call (the planner memory proxy, O(n)).
- **System:** delivered and undelivered tasks, successful episodes, conflicts per episode, per delivered task and per 100 robot-steps, conflict types, calls by trigger, deviation rate, wait ratio, mean flow time, makespan, steps per episode, supervisor interventions, and stuck-robot categories.
- **Safety:** physical collisions, invalid moves, invalid reference paths, errors.
- **Runtime:** episode wall time and planning time, both median.

The first-solution *cost* is not recorded in episodes, because that would need tracing on every call. It is measured in E1a (the k0 column).

### 5.3 Integrity gates (G0, as Phase 1, plus G0.8)

| Gate | Requirement |
|---|---|
| G0.1 | L `full` = Phase 1 `records_1b` arm `rsd_r0`, exactly (all non-timing fields), 1 080 / 1 080 |
| G0.2 | every 50th record replays identically without the observer |
| G0.3 | 0 errors, 0 physical collisions, 0 transition errors (INV) |
| G0.4 | 0 supervisor-introduced invalid moves |
| G0.5 / G0.6 | fairness and burst bounds hold |
| G0.7 | frozen supervisor parameters; `p2` records carry `rrt_patience = 400`, and `full` records carry no P2 field |
| G0.8 | 0 invalid reference paths: every new path is 4-connected, in bounds, and avoids permanent obstacles |
| — | instances identical across arms; checkpoint hashes unchanged before and after; the code-hash and `selection.json` gates hold |

### 5.4 Decision

Phase 1 §3.4 applies unchanged to each condition:
- **"reject"** if G0 fails, or a completion loss (P1 / P2) or a P4 increase is Holm-rejected;
- **"inconclusive"** if a family's nominal lower bound of Δ completion is ≤ −δ₁ (one task per episode);
- otherwise **"adopt"** if at least one E hypothesis is Holm-rejected with Δ̂ < 0, else **"no benefit"**.

**P2 recommendation:** P2 is recommended as the forward default only if **L is "adopt" and B is not "reject"**. RRT\* is shared by every controller, so the change must not harm the baseline comparator. Either way, the rule stays an option, off by default in the library.

### 5.5 Anti-tuning

- m = 400 is fixed. No Block C result may change it, the candidate set, the supervisor configuration or its parameters.
- Block C is the development block. A result here is development evidence; confirmation belongs to Phase 6 (proposed block 14 000–14 199).

---

## 6. Block C execution and integrity [M]

Run on 2026-11-22 with 10 workers, `tools/phase2/phase2.py`. Records are in `results/dev6a/phase2/eval/`.

| Step | Result |
|---|---|
| `check` | ok; `selection.json` SHA and code hashes recorded in `check.json` (the runner's own SHA matches) |
| Stage 2b (B) | 240 / 240 episodes in 169 s; 0 errors; checkpoints unchanged before and after |
| Stage 2l (L) | 2 160 / 2 160 episodes in 4 907 s; 0 errors; checkpoints unchanged before and after |
| Completeness | 2 400 expected, 2 400 present, 0 missing, 0 unexpected, 0 duplicates (the loader refuses duplicates) |
| **G0.1** | L `full` = Phase 1 `records_1b` `rsd_r0`: **1 080 / 1 080 identical** (all non-timing fields) |
| **G0.2** | 48 records (every 50th) replayed without the observer: **identical** |
| G0.3 | 0 errors, 0 physical collisions, 0 INV, in both conditions and both arms |
| G0.4–G0.6 | 0 supervisor-introduced invalid moves; fairness and burst bounds hold (L, 2 160 supervised episodes) |
| G0.7 | frozen supervisor parameters; every `p2` record carries `rrt_patience = 400` and no `rrt_refine`; no `full` record carries a P2 field; no `full` call stopped early |
| **G0.8** | **0 invalid reference paths** out of 521 346 checked (L 505 016, B 16 330) |
| Instances | identical across arms for every `(run, level, seed)` |
| Repair | off in every record |

G0 passes in both conditions.

G0.1 also answers a question about the `evaluation.py` change (§2.1).
- The two new `EvalTask` fields change the file hash that the Phase 3 and Phase 1 runners gate on, so those runners now refuse to resume.
- With the fields at their defaults, the new code reproduces all 1 080 Phase 1 RRT\* records exactly.
- The change is therefore behaviour-preserving, and the frozen Phase 1 and 3 records stay valid. They were not touched (§12).

## 7. Results [M]

Δ = `p2` − `full`. The estimator is a ratio of sums, with seed as the cluster, 40 seeds and a nominal 95 % bootstrap-t interval. "Holm" means rejected within its family at the pre-registered α. Full tables are in `2026-11-22-issue-6a-phase2-rrt-budget-tables.md`.

### 7.1 Confirmatory families

**L (9 learned policies, R + S + D)**

| Family | Outcome | Unit | full θ | Δ̂ | 95 % CI | Holm p | Decision |
|---|---|---|---|---|---|---|---|
| P1 | completion | MADQL | 0.9914 | +0.0019 | [−0.0005, +0.0042] | 0.358 | not rejected |
| P1 | completion | IPPO | 1.0000 | −0.0001 | **[−∞, ∞]** | 0.449 | not rejected |
| P1 | completion | MAPPO | 0.9993 | −0.0003 | **[−∞, ∞]** | 0.358 | not rejected |
| P2 | completion per policy | 9 policies | — | −0.0009 … +0.0056 | 4 of 9 unbounded | ≥ 0.39 | none rejected |
| E | RRT\* iterations / robot-step | MADQL | 189.6 | **−30.7 (−16.2 %)** | [−34.0, −27.4] | 0.0006 | **rejected** |
| E | RRT\* iterations / robot-step | IPPO | 121.0 | **−13.5 (−11.2 %)** | [−15.6, −11.5] | 0.0006 | **rejected** |
| E | RRT\* iterations / robot-step | MAPPO | 112.8 | **−12.1 (−10.7 %)** | [−13.7, −10.4] | 0.0006 | **rejected** |
| E | planner calls / 100 robot-steps | MADQL / IPPO / MAPPO | 19.0 / 12.1 / 11.3 | +0.11 / −0.06 / −0.03 | all contain 0 | 1.0 | not rejected |
| P4 | conflicts / episode | MADQL / IPPO / MAPPO | 1.81 / 1.96 / 1.99 | −0.27 / −0.16 / −0.53 | all contain 0 | 1.0 | not rejected |
| P4 | high-priority common flow time | MADQL / IPPO / MAPPO | 64.7 / 45.7 / 44.6 | +0.25 / +0.12 / +0.36 | all contain 0 | 1.0 | not rejected |
| P4 | high-priority undelivered | all | — | not evaluable (0 in both arms) | — | — | not rejected |

**B (baseline controller)**

| Family | Outcome | full θ | Δ̂ | 95 % CI | Holm p | Decision |
|---|---|---|---|---|---|---|
| P1 | completion | 1.0000 | 0 | [0, 0] | 1.0 | not rejected |
| E | RRT\* iterations / robot-step | 54.6 | **−5.3 (−9.7 %)** | [−6.4, −4.2] | 0.0002 | **rejected** |
| E | planner calls / 100 robot-steps | 5.47 | 0.00 | [−0.12, +0.12] | 1.0 | not rejected |
| P4 | conflicts / episode | 1.13 | +0.21 | [−0.02, +0.44] | 0.207 | not rejected |
| P4 | high-priority common flow time | 44.2 | +0.77 | [−0.40, +1.94] | 0.370 | not rejected |
| P4 | high-priority undelivered | — | not evaluable | — | — | not rejected |

All E rejections are at p = 1 / 10 001, the bootstrap floor. None is flagged MC-sensitive.

### 7.2 Decision (pre-registered rule, §5.4)

| Condition | G0 | Completion harm (Holm) | Lower bound > −δ₁ (δ₁ = 0.0517) | P4 veto | Benefit (E) | **Verdict** |
|---|---|---|---|---|---|---|
| L | pass | none | MADQL yes; **IPPO and MAPPO no (−∞)** | none | iterations / robot-step, all 3 families | **inconclusive** |
| B | pass | none | yes | none | iterations / robot-step | **adopt** |

**P2 forward default** (requires L "adopt" and B not "reject"): **no.**

**Why L is "inconclusive" [M]:**
- The verdict comes from the interval estimator, not from an observed loss.
- IPPO completion with the full budget is 1.0000: every task was delivered in all 360 episodes. MAPPO is at 0.9993.
- With `p2`, IPPO left 1 task undelivered in 360 episodes, and MAPPO net 2 more than `full`.
- With such sparse differences, many bootstrap resamples have SE = 0. The frozen bootstrap-t maps these to |t\*| = ∞, so the interval is (−∞, ∞).
- The pre-registered rule treats a lower bound of −∞ as ≤ −δ₁, so the verdict is "inconclusive".
- Phase 1 §4 and Phase 3 §5.2 flagged the same mechanism for per-policy rows. Here it reaches the family rows, because IPPO completion sits exactly at the ceiling.

**Sensitivity (not pre-registered; it does not change the verdict):**
- The frozen tool also stores the percentile interval.
- For the P1 rows the percentile intervals are:
  - IPPO: [−0.0004, 0.0000];
  - MAPPO: [−0.0007, 0.0000];
  - MADQL: [−0.0004, +0.0041].
- All three lower bounds are about two orders of magnitude above −δ₁ = −0.0517.
- Under that reading L would be "adopt".
- This is reported for transparency. The pre-registered verdict stands.

### 7.3 Completion, quality and conflicts (descriptive)

| Metric | MADQL full → p2 | IPPO full → p2 | MAPPO full → p2 | Baseline full → p2 |
|---|---|---|---|---|
| Completion | 0.9914 → 0.9933 | 1.0000 → 0.9999 | 0.9993 → 0.9990 | 1.0000 → 1.0000 |
| Delivered (of released) | 6 909 → 6 922 (6 969) | 6 969 → 6 968 (6 969) | 6 964 → 6 962 (6 969) | 2 323 → 2 323 |
| Successful episodes | 315 → 321 / 360 | 360 → 359 | 355 → 353 | 120 → 120 |
| Flow time (delivered) | 100.7 → 101.0 | 61.4 → 61.6 | 60.0 → 60.8 (nominal CI +0.39 … +1.19) | 60.0 → 60.7 (nominal CI +0.04 … +1.26) |
| Executed moves per successful plan | 12.29 → 12.47 | 17.67 → 17.67 | 17.59 → 17.72 | 18.72 → 18.96 (nominal CI excludes 0) |
| Steps / episode | 269.9 → 266.8 | 166.6 → 166.6 | 168.7 → 169.5 | 161.2 → 161.5 |
| Wait ratio | 0.155 → 0.155 | 0.015 → 0.015 | 0.011 → 0.013 | 0.046 → 0.043 |
| Conflicts / episode | 1.81 → 1.54 | 1.96 → 1.80 | 1.99 → 1.47 | 1.13 → 1.33 |
| Conflicts / delivered | 0.094 → 0.080 | 0.101 → 0.093 | 0.103 → 0.076 | 0.058 → 0.069 |
| Conflicts / 100 robot-steps | 0.084 → 0.072 | 0.147 → 0.135 | 0.148 → 0.108 | 0.087 → 0.103 |
| Planner calls / 100 robot-steps | 18.98 → 19.09 | 12.12 → 12.07 | 11.34 → 11.31 | 5.47 → 5.47 |
| Stuck robots (total) | 60 → 47 | 0 → 1 | 5 → 7 | 0 → 0 |

**Conflict types:**
- RRv / ROv in L: MADQL 385 / 264 → 320 / 232; IPPO 126 / 539 → 29 / 577; MAPPO 207 / 488 → 29 / 479.
- B (ROv only): 135 → 160.
- The L conflict counts are dominated by a few episodes. For example, MAPPO seed 3 full level goes from 6.22 to 2.33 per episode.
- None of these conflict differences is significant (P4 above).

**Calls by trigger:**
- The trigger shares are unchanged to within 0.003. For example, T3 (deviation) is 0.817 → 0.816 for MADQL and 0.580 → 0.579 for IPPO.
- The Block A smoke-test concern (§5, more calls under P2) did not appear at scale: E calls per 100 robot-steps are not rejected in any family, and the point estimates are within ±0.6 %.

**By level** (nominal; tables file):
- No level shows a completion lower bound near −δ₁.
- The largest MADQL gain is on the full level: +0.0035.

**By policy:**
- MADQL seed 2 has the largest point gain: +0.0056, nominal p = 0.043, not Holm-rejected.
- Policy-level seed counts (better / worse / equal) are balanced, with the largest imbalance for MADQL seed 2 at 14 / 7 / 19.

### 7.4 Planner work and runtime (descriptive)

| Metric | MADQL full → p2 | IPPO full → p2 | MAPPO full → p2 | Baseline full → p2 |
|---|---|---|---|---|
| RRT\* calls / episode | 409.2 → 407.1 | 161.2 → 160.4 | 152.3 → 152.6 | 70.5 → 70.6 |
| Iterations / call | 1 000 → 832.7 | 1 000 → 892.6 | 1 000 → 895.0 | 1 000 → 903.2 |
| Share of solved calls stopped early | 0 → 0.515 | 0 → 0.354 | 0 → 0.349 | 0 → 0.321 |
| Mean first-solution iteration | 103.6 → 104.2 | 141.1 → 140.4 | 142.0 → 142.4 | 150.2 → 148.6 |
| Call status ok / degraded / budget-exhausted | 0.974 / 0.001 / 0.026 → 0.974 / 0.000 / 0.026 | 0.968 / 0.001 / 0.031 → 0.969 / 0.001 / 0.031 | 0.961 / 0.001 / 0.038 → 0.956 / 0.003 / 0.041 | see tables |
| ms / call (mean) | 102.3 → 78.1 (−23.6 %) | 100.7 → 85.2 (−15.4 %) | 99.7 → 84.6 (−15.1 %) | 98.4 → 84.9 (−13.8 %) |
| RRT\* s / episode (median) | 37.3 → 28.1 | 14.7 → 12.1 | 13.6 → 11.5 | 6.25 → 5.07 |
| RRT\* s total | 15 069 → 11 451 (−24.0 %) | 5 842 → 4 921 (−15.8 %) | 5 464 → 4 646 (−15.0 %) | 832 → 719 (−13.7 %) |
| Episode wall s (median) | 38.2 → 28.9 (−24.4 %) | 15.1 → 12.5 (−17.3 %) | 14.1 → 12.0 (−15.1 %) | 6.52 → 5.35 (−17.9 %) |
| Planning share of step time | 0.980 → 0.973 | 0.967 → 0.961 | 0.965 → 0.959 | 0.954 → 0.947 |
| Tree nodes / call (memory proxy) | 695 → 564 (−18.8 %) | 692 → 608 (−12.1 %) | 691 → 608 (−11.9 %) | 686 → 612 (−10.7 %) |
| Max tree nodes | 841 → 832 | 827 → 827 | 827 → 821 | 819 → 812 |

Supportive nominal intervals for RRT\* time per episode all exclude 0:
- MADQL −10.0 s, [−10.7, −9.3];
- IPPO −2.56 s;
- MAPPO −2.27 s;
- baseline −0.95 s.

**Memory:**
- The peak tree size bounds RRT\* memory, and P2 never raises it, because a stopped run is a prefix of the full run.
- Process peak RSS was not recorded per episode. The workers are separate processes, and the RRT\* tree is at most about 840 nodes, so the tree is not a memory concern at these sizes.

### 7.5 Safety [M]

| | MADQL | IPPO | MAPPO | Baseline |
|---|---|---|---|---|
| Physical collisions | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Invalid moves (INV) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Invalid reference paths (checked) | 0 / 0 (143 710 / 142 864) | 0 / 0 (56 367 / 56 104) | 0 / 0 (53 015 / 52 956) | 0 / 0 (8 112 / 8 218) |
| Errors | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Supervisor-introduced invalid moves | 0 | 0 | 0 | — |

Deadlocks and stalls: no `deadlock_cycle` or `conflict_loop` stuck robots in either arm. The remaining stuck categories are listed in the tables file.

## 8. Mechanism and interpretation

**[M] The saving is larger in episodes than in the microbenchmark.**
- Iterations: −10 to −16 % in episodes, against −5.6 % in E1a.
- RRT\* time: −14 to −24 %, against −10 %.
- In episodes, queries are replans from the robot's current cell to a nearby goal. Executed plans are 12–19 moves long, against a broader length mix in E1a.
- The first solution comes early: mean iteration 104–150.
- On short queries the best cost settles sooner, so a 400-iteration window without improvement ends the run more often. The stopped share is 32–52 %.
- MADQL replans most and over the shortest distances (12.3 moves per plan, 409 calls per episode), so it saves the most.

**[M] Per-call time falls more than iterations do.** For MADQL, iterations fall 16.7 % and time falls 23.6 %.
- Late iterations are the expensive ones. With a larger tree, the near-neighbour search and rewiring touch more nodes, so cutting the tail removes more than its share of time.
- The tree is 19 % smaller for MADQL.

**[M] Quality cost: small, and in line with E1a.**
- Executed moves per successful plan rise 0–1.5 %. E1a gave 0.07 % pooled and up to 0.18 % per family on the stratum.
- The episode figure is larger, consistent with the higher stopped share.
- Mean flow time rises 0.3–1.3 %. MAPPO and the baseline have nominal intervals excluding 0, but neither is a pre-registered family.
- High-priority flow time on commonly delivered tasks (P4) is not rejected in any family.

**[I] Completion is unaffected within the resolution of 40 seeds.**
- The point estimates go both ways: MADQL +0.0019, IPPO −0.0001, MAPPO −0.0003, baseline 0.
- They are 1–2 tasks out of about 7 000, and no Holm test is near rejection.
- Trajectories diverge after the first differing plan, so these differences are episode-level chaos rather than a systematic effect of the stopping rule.

**[I] No system-level feedback loop.**
- The concern was that slightly worse plans would trigger more deviation replans (T3) and cancel the saving.
- Calls per robot-step and the trigger mix are unchanged, so the saving per call carries through to the episode.

## 9. Decision and recommendation

| Question | Answer |
|---|---|
| Implementation correct and guarantees hold? | **Yes** [C, M]: exact-prefix property on 33 600 + 240 E1a/E1c runs and in property tests; 0 invalid paths in 521 346 checked; byte-identical behaviour when off (G0.1 1 080 / 1 080; integration test) |
| Microbenchmark selection | m = 400 was the only eligible candidate (§4) |
| Block C, B | **adopt**: −9.7 % iterations, −13.7 % RRT\* time, no harm |
| Block C, L | **inconclusive** (pre-registered rule): −11 to −16 % iterations and −15 to −24 % RRT\* time with no Holm-detected harm. The verdict is driven by degenerate bootstrap-t intervals at completion ≈ 1.0 for IPPO / MAPPO |
| **Overall** | **Inconclusive.** The P2 forward-default criterion is not met |

**Recommendation:**
1. **Keep P2 as an implemented option, off by default** (`refine_iterations = patience = None`). This is the library state now. Do not make m = 400 the forward default on this evidence.
2. **Carry m = 400 as the candidate into the confirmation phase** (Phase 6, block 14 000–14 199). Do not retune it.
   - Before that run, pre-register how a degenerate bootstrap-t family interval is handled. For example: use the stored percentile interval, or a one-sided exact bound on the count of lost tasks, whenever the full arm is at the ceiling.
   - That is a protocol decision for Phase 6. It is not applied retroactively here.
3. **If the user accepts the sensitivity reading (§7.2) as sufficient**, P2 m = 400 can be enabled explicitly for experiments where RRT\* runtime matters, as a documented engineering option.
   - The quality cost is about 1 % in flow time, and the saving is about 15–24 % of RRT\* time.
   - That choice belongs to the user, not to this phase's rule.
4. **Not pursued:** informed sampling (§1), k-type rules (they failed the screen), and tree reuse (P6, deferred).

## 10. Limitations and deviations

**Limitations:**
- **Development evidence only.** Block C is the development block; confirmation belongs to Phase 6.
- **40-seed RRT\* subset.** Completion near 1.0 leaves the tests with little resolution. The degenerate intervals (§7.2) are a direct consequence.
- **Single budget.** Only the system budget of 1 000 iterations and only m = 400 were tested in episodes. The result does not transfer to other budgets without new microbenchmarks.
- **First-solution cost** is not recorded in episodes (§5.2). It is measured in E1a (k0).
- **E2** uses reserved test seeds (20 000–20 019). It was used only as a reported comparison after the selection, never for selection or tuning.
- **The 1 % screen** in §3.3 is an engineering tolerance, not a statistical bound.
- **Timing** was measured on one machine (i7-1355U, 10 workers in parallel). Absolute times are indicative; the paired ratios are the robust figures.
- **Memory** is measured through the tree-size proxy, not process RSS.

**Deviations from §5 (none affects a verdict):**
- After the results were seen, `phase2_report.py` gained more descriptive tables (conflict types, calls by trigger, call status, stuck categories, makespan, supervisor changes, segment checks) and the P1 sensitivity table.
  - These render values that `phase2_analysis.py` had already computed and stored.
  - `analysis.json` is unchanged by them.
  - The sensitivity table is labelled as not pre-registered.
- The analysis is run with `PYTHONPATH=src` (§12), like the runners. This is an invocation detail and not a change.
- Because of the `evaluation.py` change, the Phase 3 and Phase 1 runners' code gates no longer pass. Their frozen records remain valid (G0.1), but resuming those runners now needs a new `check`.

## 11. Tests, lint and types [M]

| Suite | Result |
|---|---|
| `tests/unit/planning/test_rrtstar_budget.py` | 8 passed (including the Hypothesis prefix property, 80 examples) |
| `tests/integration/test_rrt_budget_integration.py` | 3 passed |
| `tests/tools/test_phase2.py` | 5 passed |
| Full suite (`tests/unit tests/tools tests/integration`, includes the supervisor and Phase 1 suites) | **1 219 passed**, 1 skipped (POSIX-only `getrusage` branch), 57 deselected (slow markers), 0 failed |
| `ruff check` / `ruff format --check` on `src`, `tools/phase1–3`, `tests` | all checks passed; 189 files already formatted |
| `mypy src` | no issues (84 files) |
| `mypy` on `tools/phase1–3` and the Issue 6A tests (supervisor, repair, P2) | no issues (19 files) |

`mypy` over the whole `tests` tree reports 104 errors, all in test files that no Phase 2 change touched (for example `test_ppo_training.py` and `test_conflicts.py`). They predate this phase and are left as they are.

## 12. Provenance and integrity

| Artifact | SHA-256 (prefix) |
|---|---|
| `results/dev6a/phase2/micro/selection.json` | `224fc343` |
| `results/dev6a/phase2/eval/check.json` | `ad101150` |
| `results/dev6a/phase2/eval/records_2b.jsonl` | `6f0e1f7a` |
| `results/dev6a/phase2/eval/records_2l.jsonl` | `83b04d00` |
| `results/dev6a/phase2/eval/replay.json` | `e02956a4` |
| `results/dev6a/phase2/eval/analysis.json` | `97a83148` |
| `results/dev6a/phase2/eval/tables.md` | `7bdaa8db` |
| This document, at the start of the Block C run (§1–5) | `2d95c48f` |

**Reproduction:**

```
PYTHONPATH=src python tools/phase2/micro.py {e1a,e1c,select,timing,e2} --out results/dev6a/phase2/micro --workers 10
PYTHONPATH=src python tools/phase2/micro_report.py --out results/dev6a/phase2/micro
PYTHONPATH=src python tools/phase2/phase2.py check --out results/dev6a/phase2/eval
PYTHONPATH=src python tools/phase2/phase2.py run --stage 2b --out results/dev6a/phase2/eval --workers 10
PYTHONPATH=src python tools/phase2/phase2.py run --stage 2l --out results/dev6a/phase2/eval --workers 10
PYTHONPATH=src python tools/phase2/phase2.py replay --out results/dev6a/phase2/eval --workers 10
PYTHONPATH=src python tools/phase2/phase2_analysis.py --out results/dev6a/phase2/eval
PYTHONPATH=src python tools/phase2/phase2_report.py --out results/dev6a/phase2/eval
```

**Final integrity check [M]:**

| Check | Result |
|---|---|
| Block C / Block D `plan_freeze.json` file hashes (3 + 9 files) | all match; records, analyses and replays last modified 2026-11-21 |
| Frozen statistics `block_c_analysis.py` | `3fe86a77` (unchanged) |
| Supervisor `supervisor.py` | `83b84b8e` (unchanged) |
| Stage 3a `freeze.json` | `4c7b9dbc` (unchanged) |
| Phase 3 records 3b / 3c / 3e, `analysis.json`, `check.json`, `replay.json`, `tables.md` | `a6ed0179`, `09db9fe7`, `41ef734a`, `62180f3f`, `4890381c`, `db992ef5`, `65e89784` (unchanged) |
| Phase 1 `check.json`, `records_1a`, `records_1b`, `replay.json`, `analysis.json`, `tables.md` | `c1276332`, `8c4c6c31`, `60621df0`, `513635fe`, `24e53297`, `6fbb263d` (unchanged) |
| 9 policy checkpoints | unchanged (hashes in `check.json` re-verified after the run) |
| `models/` | does not exist |
| `docs/SESSION_HANDOFF.md` | untouched (last modified 2026-11-19) |
| Analysis environment | `analysis.json` is byte-identical whether it is produced with the project `.venv` or the system interpreter |

## 13. Status

**Phase 2 complete, with an inconclusive result.**
- P2 is implemented, tested and documented.
- It is off by default.
- m = 400 is the frozen candidate for Phase 6 confirmation.
