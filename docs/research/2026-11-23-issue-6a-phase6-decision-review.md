# Issue 6A Phase 6: decision review and final draft confirmatory protocol

*Status: **decisions D1–D13 approved by the student (2026-11-23; §11). The approved protocol is `2026-11-23-issue-6a-phase6-protocol.md`. Not yet frozen.***
- No scenario from seeds 14 000–14 199 has been generated.
- No Phase 6 episode has been run.
- It supersedes the draft protocol in `2026-11-23-issue-6a-phase6-readiness.md` §6–§9 and corrects that report's §10 (see §3.6).

**Labels:**
- **[A]** previously approved or frozen, with its source;
- **[U]** unresolved: needs a decision;
- **[R]** a recommendation derived from existing evidence;
- **[N]** a new methodological proposal that needs explicit approval;
- **[M]** measured on development data (Blocks B, C and S only; never confirmatory);
- **[C]** follows from code or definitions;
- **[P]** a projection.

---

## 1. Executive summary

1. **The 13 decisions of readiness §13 are reviewed individually in §2.**
   - Eleven can be settled by extending already-approved rules, with no new threshold.
   - Two need a genuine choice from the student: the robot–obstacle (RO) veto rule (D3) and its scope (D4).
2. **RO veto: recommendation (strict, as every existing veto).**
   - **Outcome:** RO conflicts per 1 000 moves, measured over all robots.
   - **Rule:** a veto is a Holm-rejected increase, the convention that has governed every P4 veto in this project (Stage 3a §6.6, Phase 1 §3.4, Phase 4 §5.4).
   - **Reference:** MADQL-v2 is compared with the historical MADQL.
   - **Scope:** the veto applies to every confirmatory comparison (H-v2, H-L2, H-L12).
   - **Context only:** the comparison with the forward PPO policies is pre-registered, but never gates a decision.
   - **Why not a margin:** a non-inferiority margin or a PPO-referenced bound is not recommended. Both would be chosen with knowledge of the development data. Both would loosen the criterion only for the case where we already expect an increase.
3. **The development evidence was re-measured exactly (§3).** Block C, 9 360 episodes re-run through the Phase 6 observer, all 9 360 identical to their stored records.
   - **Phase 4 overstated the effect.** Its post-hoc figure (2.1–2.5× per move) counted idle robots' conflicts but only non-idle moves.
   - **With the consistent definition** (all robots in numerator and denominator), MADQL-v2 vs historical MADQL is:
     - FA ×2.2, +0.56 per 1 000 moves, nominal 95 % CI [0.23, 0.89];
     - FR ×1.8, +0.43 [0.19, 0.67];
     - NA ×1.4, +0.21 [−0.13, 0.54], not significant.
   - **The supervisor does not raise the RO rate** in any family: MADQL −0.00, IPPO +0.18, MAPPO −0.05, all n.s. Readiness §10's statement about IPPO came from a wrong approximate denominator, and is corrected here.
   - **PPO context:** the forward PPO policies run at 1.79 per 1 000 moves (A\*, R + S + D); MADQL-v2 at 1.04.
4. **Consequence, stated before any held-out data.**
   - If the Block C effect is real, the strict veto rejects MADQL-v2 with projected probability ≈ 1.0 in FA, 0.88–0.94 in FR and 0.26–0.41 in NA.
   - The independent development block S shows the same direction but a weaker and noisier effect, so the outcome is genuinely uncertain.
   - Phase 6 is therefore a real test of the safety claim, not a formality.
5. **Scope.** Keep H-L2, H-v2 and H-L12 (22 080 episodes, ≈ 5 h on 10 workers; re-checked against measured parallel efficiency).
   - H-L12 is the only test of the integrated claim "Issue 6A improves the system", which proposal §19 D.3 froze "until block D".
6. **Multiplicity.** Family-wise error at 0.05 *per comparison unit* (7 units), as in the original proposal §12 ("each with its metric family … and Holm correction") and in Phases 1–4. A global split would be a protocol deviation. Confirmation claims are conjunctions, which protects them against false positives (§4.3).
7. **Design flaws found and corrected** (§4.1):
   - the RO denominator definition;
   - the wrong RO approximations in readiness §10;
   - an unnecessary P3 family in H-L12;
   - the PPO context comparison, which existed only in development code.
8. **Checkpoint backup.** A SHA-256 manifest of the 1 216 files Phase 6 depends on (1.45 GB) was written read-only and verified. The copy procedure is in §6.10.
9. **Nothing in the protected set changed** (§10). No held-out scenario or episode was generated (§9).

---

## 2. Review of the 13 decisions (readiness §13)

### D1: Reserve the held-out block

**Question.** Is seeds 14 000–14 199 (RRT\*: 14 000–14 059) the right block, and should it be reserved now as single-use?

**Evidence:**
- **[A]** Phase 3 design decision D7 (approved): "final validation in Phase 6 on a new block (14 000–14 199) … reserve the block before Phase 6".
- **[A]** Block D protocol §13: "Block D is single-use. A future L2 would need a new reserved block". Block D was used once, for H-L1 (§25).
- **[M]** 0 of 863 stored result files contain a seed field in the range. No code, test or configuration referenced it before Phase 6 preparation.
- **[C]** The range is disjoint from train 0–9 999, A 10 000–10 019, B 10 020–10 099, C 11 000–11 119, D 12 000–12 199, S 13 000–13 079 and test 20 000–20 999 (unit-tested).
- **[M]** No selection, calibration or tuning step used it:
  - PPO picks: A ∪ S;
  - MADQL-v2 checkpoint rule: A; candidate selection: S;
  - Stage 3a calibration: B;
  - supervisor, P1, P2, Phase 4 evaluation: C;
  - Phase 4 diagnostics: B.

**Options:**
- **(a)** 14 000–14 199 with RRT\* subset 14 000–14 059.
- **(b)** Another unused range.
- **(c)** A larger block.

**Assessment:**
- (a) follows D7 exactly and mirrors Block D's approved 200 / 60 split. The approved precision projections used that split (Block D §6).
- (b) gains nothing.
- (c) raises cost, and the projections show 200 / 60 is adequate for every primary effect (§4.4).
- **Remaining limitations:**
  1. **Instance-level disjointness** from A, B, C, D and S can only be checked by generating Block V instances. As in Block D §8, this happens inside the post-freeze `check`, and the run is refused unless all 600 instances per level are unique and disjoint.
  2. **Disjointness from training maps is not verifiable.** Training logs store only a seed and a scenario name, never an instance fingerprint, and regenerating 10 000 training scenarios to compare would not prove that the stored checkpoints were trained on exactly those (training samples scenarios from a seed stream).
     - Mitigation: training seeds 0–9 999 are a disjoint range, and each instance is generated from its own seed via `default_rng([crc32(level), seed])`.
     - The same limitation was accepted for Block D (§8.6).

**Recommendation [R]:** (a). Reserve it as **Block V**, single-use, at the approval of this protocol. Register it in the Phase 6 runner only (the production seed registry stays train / A / B, as for C, D and S).

**Approval required:** **yes**, a reservation of data.

### D2: Scope of the confirmatory comparisons

**Question.** Which hypotheses does Phase 6 confirm?

**Evidence:**
- **[A]** Proposal §12: the confirmatory hypotheses are H-L1, H-L2 and H-L12, "each with its metric family, margins and Holm correction". §19 D froze them, and any claim that 6A improves the system, "until block D".
- **[A]** Block D tested H-L1 only (§25).
- **[A]** §34.1 adds MADQL-v2 (H-v2).
- **[A]** The roadmap says Phase 6 covers "all required algorithm × planner combinations": MADQL, IPPO and MAPPO, the baseline, and A\* and RRT\* (FR-EXP-3).

**Options:**
- **(a)** H-L2 + H-v2 + H-L12, 22 080 episodes.
- **(b)** Without H-L12: 18 960 episodes, ≈ 0.4 h less.
- **(c)** A\* only for some sets.

**What H-L12 adds** (see also §4.2):
- H-L2 tests the supervisor with fixed checkpoints. H-v2 tests the MADQL retraining under fixed execution.
- Neither tests the *combination* that Issue 6A hands to Issue #8 against the system Issue #6 left. For PPO, that is new picks + supervisor vs Issue #6 picks without supervisor. For MADQL, it is v2 + supervisor vs historical MADQL without supervisor.
- Chaining H-L1 (Block D) with H-L2 (Block V) across blocks does not test the combination: effects need not add, and the two blocks differ.
- H-L12 is the only direct test of the integrated claim §19 D.3 reserved for held-out data.
- It adds 3 120 episodes (the Issue #6 picks, OFF) and two analysis units. With per-unit error control (D5) it does not change any other unit's thresholds.

**Consequences of (b):**
- Issue 6A could then claim only component effects. The integrated statement would remain an untested inference.

**Consequences of (c):**
- RRT\* is a required planner. Dropping it would leave Phase 3e and Phase 4 FR without confirmation.

**Recommendation [R]:** (a).

**Approval required:** **yes**. The scope of a confirmatory run is the student's decision.

### D3: The robot–obstacle veto rule (highest priority; full analysis in §3)

**Question.** By what rule does an increase in RO conflicts per move block a confirmation?

**Options:**
- **(a) Strict:** a Holm-rejected increase in the P4 family is a veto.
- **(b) Non-inferiority (NI):** the confirmation requires the upper confidence bound of Δ to lie below a margin M.
- **(c) A PPO-referenced bound:** MADQL-v2's rate must not exceed the forward PPO policies' rate.
- **(d) Descriptive only.**

**Assessment** (details in §3.4–3.5):
- **(a)**
  - is the existing, approved convention for every P4 outcome, and is internally consistent;
  - uses no new threshold;
  - is not hypersensitive: at Block V sample sizes the smallest detectable increase is 0.31–0.40 per 1 000 moves, about +65–77 % of the historical rate;
  - its weakness is that a statistically detected but practically small increase also vetoes. At these sample sizes that would need a ≥ +65 % relative increase, which is not small.
- **(b)** needs a margin M that no project document provides.
  - On Block C, MADQL-v2 would pass FA only with M ≥ ≈ 0.77 per 1 000 moves, i.e. accepting up to +160 % of the historical rate.
  - Any M chosen now is chosen with knowledge of that number. A small M makes the criterion stricter than (a), with the burden of proof on v2. A large M is a weakening chosen after seeing the development effect.
- **(c)** changes the reference from a paired, same-architecture control to a cross-algorithm one. It turns PPO's rate into an implicit safety standard that was never approved.
  - Development data show that v2 (1.04) would pass a PPO bound (1.79) easily, so choosing (c) now is outcome-informed.
  - It also makes the criterion depend on PPO behaviour, a moving reference.
- **(d)** contradicts the student's instruction (a pre-registered veto).

**Recommendation [R]:** (a), with the PPO comparison pre-registered as contextual, nominal and non-gating (D4).

**Approval required:** **yes**. This is the one decision whose outcome is likely to determine the MADQL-v2 verdict.

### D4: Where the RO veto applies

**Question.** To which comparisons does the RO veto apply?

**Evidence [M]** (Block C, exact definition):
- Supervisor OFF → R + S + D changes the RO rate by −0.00 (MADQL), +0.18 (IPPO) and −0.05 (MAPPO), all n.s.
- H-L12 (MADQL) +0.56 [0.29, 0.83].
- **[C]** The supervisor's S and R change which robots move, so the supervisor could in principle change the RO rate.

**Options:**
- **(a)** All three sets.
- **(b)** H-v2 only.
- **(c)** H-v2 and H-L12.

**Assessment:**
- The RO rate is a property of the executed system, not of MADQL-v2 alone.
- A safety criterion applied to one intervention but not to another is inconsistent, and would let the supervisor pass a test that v2 must pass.
- The development data show no supervisor effect, so (a) costs the supervisor nothing in expectation. The decision is made on consistency, not on that expectation.
- H-L12 inherits the MADQL-v2 effect by construction.

**Recommendation [R]:** (a).

**Context comparisons, pre-registered and non-gating:**
- **[N]** MADQL-v2 (R + S + D) vs the forward PPO policies (v5, R + S + D) on the same instances, per planner.
- The RO rate of every arm, including the baseline.

**Approval required:** **yes**.

### D5: Multiplicity scope

**Question.** Family-wise error at 0.05 per comparison unit, or one global 0.05 across all units?

**Evidence:**
- **[A]** Proposal §12: Holm within each metric family, families per hypothesis.
- **[A]** Block D: a union bound ≤ 0.05 across its families for its single hypothesis.
- **[A]** Stage 3a §6.1 and Phase 1 §3.2: union bound ≤ 0.05 "per stage" / "per condition".
- **[A]** Phase 4 §5.2: per condition.

**Options:**
- **(a)** ≤ 0.05 per unit. The units are H-L2 × {A\*, RRT\*}, H-L12 × {A\*, RRT\*} and H-v2 × {FA, FR, NA}: 7 in all.
- **(b)** A global 0.05 split over the 7 units (≈ 0.0071 each).

**Assessment (a):**
- Consistent with every approved protocol, so it is not a deviation.
- Under a complete global null, the chance of at least one false rejection somewhere is at most 1 − 0.95⁷ ≈ 0.30.
- However, every confirmation statement in §6.8 is a **conjunction**:
  - the supervisor needs "go" on A\* *and* RRT\*;
  - MADQL-v2 needs FA "adopt" *and* no "reject" in FR and NA;
  - each "go" needs several simultaneous conditions.
- Under the null, a false *confirmation* is therefore much rarer than a false rejection.
- More tests make false **vetoes** more likely, which errs on the side of caution.

**Assessment (b):**
- A new procedure, i.e. a deviation from the approved methodology.
- It cuts power. At the first Holm step of P1 the critical z rises from 2.58 to 3.19, so the completion MDE grows by about 18 %. RO veto MDEs grow similarly, which makes harm harder to detect.
- It does not make the conjunction claims materially safer.

**Recommendation [R]:** (a), stated explicitly in the report together with the 0.30 bound.

**Approval required:** **yes**. The student should knowingly accept per-unit control.

### D6: Families and α within each unit

**Question.** Which hypothesis families and α splits does each unit use?

**Evidence:**
- **[A]** Stage 3a: P1 0.03, P2 0.005, P3 0.01, P4 0.005.
- **[A]** Phase 4: P1 0.03, P2 0.005, E 0.01, P4 0.005.

**Options:**
- **(a)** As approved, with the RO outcome added to each P4 family at the same α.
- **(b)** A separate RO family with its own α, which raises the union bound above 0.05 or forces other α down.

**Assessment of (a):** the other P4 hypotheses get slightly stricter Holm thresholds (m 6 → 9 in H-L2), so the existing vetoes become slightly harder to trigger. That is a small, conservative side effect.

**Finding:** H-L12 does not need P3. P3 (stuck categories) serves mechanism attribution, which H-L12 does not claim, and the 3b decision does not use P3.
- **Recommendation [R]:** drop P3 from H-L12 and report its categories nominally. This removes 12 hypotheses that have no decision role.
- Keep P3 in H-L2, as in approved Stage 3a 3b.

**Recommendation [R]:** (a), with P3 removed from H-L12.

**Approval required:** **yes**, for the RO addition and the H-L12 change. The remaining structure is approved.

### D7: Completion-ceiling handling

**Question.** How are degenerate completion intervals handled?

**Evidence:**
- **[A]** Phase 2 §9: "pre-register how a degenerate bootstrap-t family interval is handled".
- **[A]** The student, 2026-11-22: unbounded intervals "must not be treated as evidence of equivalence or automatically interpreted in favor of an optimization".
- **[A]** Phase 4 §5.4 pre-registered the rule below and applied it (it was not needed there).

**The rule:** when the bootstrap-t interval is unbounded, the lower bound is min(percentile bound, loss-only bound). An unbounded interval is never a rejection and never evidence of equivalence.

**Consistency check [C]:**
- The rule only lowers a bound. It can turn "go" / "adopt" into "inconclusive", never the reverse.
- It applies to the 3b "no harm interpretable" condition and to the Phase 4 lower-bound condition.
- It never applies to vetoes. An unbounded veto row is never rejected (not-evaluable rule), which is conservative for adoption and is how Phases 3–4 treated it.

**Options:**
- **(a)** The Phase 4 rule.
- **(b)** The bare bootstrap-t: an unbounded bound means "inconclusive".
- **(c)** The percentile interval alone, which was rejected by the student as a favourable substitution.

**Recommendation [R]:** (a).

**Approval required:** **yes**. It was pre-registered for Phase 4, but its use in a confirmatory run should be explicitly approved.

### D8: H-L12 decision rule

**Question.** What decision rule does H-L12 use?

**Options:**
- **(a)** The approved Stage 3a 3b (A\*) / 3e (RRT\*) rule.
- **(b)** Descriptive only.
- **(c)** A new rule.

**Assessment:**
- H-L12 is a system-change comparison of exactly the kind 3b / 3e were written for: a MADQL gain is required; no PPO loss, harm or veto is allowed.
- (b) would make the integrated claim untestable.
- (c) would be a new rule without justification.

**Recommendation [R]:** (a).

**Approval required:** **yes**.

### D9: Robot–robot conflicts

**Question.** Are robot–robot (RR) conflicts tested or reported?

**Evidence:**
- **[A]** Phase 3 design §8: R lowers K^arb trivially; effectiveness is judged on completion and K^ctrl.
- **[A]** Conflicts per episode is a P4 veto in Phase 4 (H-v2).

**Options:**
- **(a)** RR conflicts on K^ctrl and K^arb per 100 robot-steps as pre-registered secondary outcomes with nominal intervals. The existing H-v2 veto "conflicts per episode" (RR + RO) stays.
- **(b)** A tested RR family, which needs extra α.

**Assessment:**
- (a) keeps the approved union bound.
- (b) would make K^arb RR look better by construction under R. K^ctrl is a valid measure but has no approved role as a gate.

**Recommendation [R]:** (a).

**Approval required:** **yes**. It confirms that RR conflicts are not a gate outside H-v2's existing veto.

### D10: P2 (m = 400)

**Question.** Is P2 (RRT\* early stopping, patience m = 400) part of Phase 6?

**Evidence:**
- **[A]** 2026-11-22: P2 is optional and off by default; m = 400 "may be considered" in Phase 6 and must not be silently enabled.
- **[A]** 2026-11-23: "P1 and P2 disabled".

**Options:**
- **(a)** Excluded.
- **(b)** A secondary arm, about +1 h of RRT\*, two extra units, and a pre-registered degenerate-interval rule.

**Recommendation [R]:** (a), following the latest instruction. P2 remains an untested engineering option.

**Approval required:** yes, a one-word confirmation.

### D11: MADQL-v2 with RRT\* and no supervisor

**Question.** Is MADQL-v2 run with RRT\* and the supervisor OFF?

**Options:**
- **(a)** Not run.
- **(b)** Run it: +540 episodes, ≈ +1–1.4 h.

**Assessment:**
- Phase 4 made no RRT\*-without-supervisor claim (FR only). No confirmatory hypothesis needs the cell.
- (b) would only describe v2 more completely.

**Recommendation [R]:** (a).

**Approval required:** yes, a confirmation.

### D12: Checkpoint backup

**Question.** How are the checkpoints that exist only in git-ignored `results/` protected?

**Evidence [M]:** the 9 MADQL-v2 runs (580 MB) and the 9 historical and PPO runs Phase 6 uses (`results/training/…`) exist only locally.

**Recommendation [R]:** the procedure in §6.10:
- a SHA-256 manifest, already written read-only and verified against the originals;
- a copy to two locations outside the repository and never into `models/`;
- verification of each copy against the manifest;
- re-verification of the originals;
- recording the manifest hash in the freeze.

**Approval required:** **yes** for the destinations. The copy is made by the student, or by me on explicit instruction.

### D13: Authorization sequence

**Question.** In what order are approval, freeze and execution authorized?

**Evidence [A]:** Block D §9 (approval → implementation → separate freeze approval → check → separate execution authorization → report).

**Recommendation [R]:** the same sequence (§6.11). The runner already enforces it: `freeze` needs `--authorized-freeze`; `check`, `run` and `replay` need `--authorized-phase6` and an unchanged freeze.

**Approval required:** **yes**.

---

## 3. Robot–obstacle conflict veto: detailed analysis

### 3.1 Definitions [C]

| Term | Definition (model §3, §9.2; engine) |
|---|---|
| Move | A robot-step whose action submitted to the arbiter (`a^arb`: the policy's action after the supervisor and the shield; the shield is off in every arm) is not STAY. **All robots count, idle or not** |
| RO conflict | An event in K(s_t, a^arb) of type **ROv** (the robot moves into the cell z_k(t+1) that obstacle k enters; a STAY can never be one) or **ROs** (robot and obstacle swap cells). It is a *proposed* conflict: the arbiter cancels the move, so no collision occurs (Lemma 3.1). The model defines safety by proposed conflicts, not by collisions (§3, §9.2) |
| RO per 1 000 moves | 1 000 · Σ (ROv + ROs on K^arb) / Σ moves, a ratio of sums over the seed's 3 levels and the arm's runs, paired by instance |
| Conflicts per episode | All proposed conflicts (RRv + RRs + ROv + ROs + INV) per episode |

**Why normalize by moves:**
- RO conflicts can only occur on moves. Per episode they mix two quantities: how often a controller moves (exposure) and how risky each move is (hazard).
- MADQL-v2 moves less per episode (−11 % in FA, −24 % in NA) because it finishes faster, and more often while idle.
- Per episode, its RO count therefore understates the per-move hazard. In Block C NA, per episode is ×1.47 (n.s.) while per move is ×1.93 under the Phase 4 definition.
- Per move answers "is each move more likely to step into an obstacle's path?", which is what a safety criterion for a policy should ask. Per episode answers "how many emergency stops per episode", the arbiter's load. Both are reported; the veto uses per move, as the student required.

**Why all robots (definition (i)) and not only non-idle robots (ii) [N]:**
- Idle robots are a large and policy-dependent share of motion [M, Block C]:

  | Arm | Moves by idle robots | RO events by idle robots |
  |---|---|---|
  | Historical MADQL, OFF | 6.7 % | 9.0 % |
  | Historical MADQL, R + S + D | 15.2 % | 17.1 % |
  | MADQL-v2, OFF | 31.0 % | 46.0 % |
  | MADQL-v2, R + S + D | 27.1 % | 32.7 % |
  | v5 PPO, OFF | 34.6 % | 13.4 % |
  | v5 PPO, R + S + D | 29.7 % | 6.8 % |

- An idle robot that steps into an obstacle's path is as much a safety event as a busy one.
- Numerator and denominator must count the same robots. Phase 4's post-hoc analysis did not: idle robots' conflicts were counted against non-idle moves. That overstated MADQL-v2's increase (§3.6).
- Definition (i) is the one the Phase 6 runner records (`p6.k_arb`, `p6.moves_arb`). Definition (ii) is recorded too (`p6.ro_arb_nonidle`, `p6.moves_arb_nonidle`) and is reported as a secondary.

### 3.2 Development evidence, exact definition [M]

**Source:** Block C, 9 360 episodes re-run through the Phase 6 runner and observer (`tools/phase6/dev_ro_exact.py`). All 9 360 equal their stored Phase 1 / Phase 4 records in every shared non-timing field. The values are nominal; the estimator is the frozen one.

| Comparison (Block C) | Old rate | New rate | Ratio | Δ (per 1 000 moves) | Nominal 95 % CI |
|---|---|---|---|---|---|
| **H-v2 FA** (hist → v2, A\* + R + S + D) | 0.48 | 1.04 | 2.17 | **+0.56** | [0.23, 0.89] |
| **H-v2 FR** (RRT\* + R + S + D) | 0.52 | 0.95 | 1.83 | **+0.43** | [0.19, 0.67] |
| **H-v2 NA** (A\*, OFF) | 0.48 | 0.69 | 1.43 | +0.21 | [−0.13, 0.54] |
| H-L2 MADQL (OFF → R + S + D) | 0.48 | 0.48 | 1.00 | −0.00 | [−0.31, 0.31] |
| H-L2 IPPO | 1.97 | 2.15 | 1.09 | +0.18 | [−0.50, 0.86] |
| H-L2 MAPPO | 1.48 | 1.43 | 0.97 | −0.05 | [−1.37, 1.27] |
| H-L12 MADQL (hist OFF → v2 R + S + D) | 0.48 | 1.04 | 2.17 | +0.56 | [0.29, 0.83] |
| *Context:* forward PPO → v2 (both R + S + D, same instances) | 1.79 | 1.04 | 0.58 | −0.75 | [−1.77, 0.27] |

**Under definition (ii) (non-idle robots only):** H-v2 FA +0.51 [0.00, 1.02]; FR +0.32 [0.07, 0.57]; NA +0.07 [−0.30, 0.43]. The supervisor rows are unchanged in sign and significance.

**Robustness** (non-idle denominator, the stored-record approximation; `dev_ro_analysis.json`):
- **Across seeds:** in Block C FA, 109 of 120 seeds show a higher v2 rate.
- **Across training seeds:** the increase is large for policy seed 1 (×3.0), smaller for seed 2 (×1.5), and uncertain for seed 3 (wide interval).
- **Independent development block S** (80 seeds; v2 = n3; selected on completion, not on conflicts):
  - FA ×1.53, +0.58 [−0.26, 1.42], nominal p 0.19;
  - NA ×2.31, +0.74 [−0.48, 1.97], nominal p 0.13;
  - 69 of 80 seeds higher in both;
  - driven by policy seed 1 (×3.0–4.6); seed 3 shows no increase.
- **Reading:** the *direction* replicates. The *size* is uncertain and heterogeneous across training seeds. The Block C effect should not be assumed to generalize at its Block C size.

### 3.3 What a strict veto implies on Block V [P]

**Method:**
- SE_V = SE_C · √(n_C / n_V).
- Two-sided Holm in a P4 family of 4: the worst threshold is 0.005 / 4, the best 0.005.

| Comparison | SE_V | 80 % MDE (per 1 000 moves) | MDE relative to the historical rate | P(veto) if the Block C effect is true |
|---|---|---|---|---|
| H-v2 FA | 0.085 | 0.31–0.35 | +65–72 % | ≈ 1.00 |
| H-v2 FR | 0.098 | 0.36–0.40 | +69–77 % | 0.88–0.94 |
| H-v2 NA | 0.080 | 0.29–0.33 | +61–68 % | 0.26–0.41 |
| H-L12 MADQL (A\*, family of 9) | 0.112 | — | — | 0.94–0.99 |
| H-L2 (all families) | 0.08–0.22 | 0.33–0.92 | +47–68 % | < 0.03 |

**Reading:**
- **If the Block C effect is real,** the strict veto almost certainly rejects MADQL-v2 in FA, and therefore overall (FA must be "adopt").
- **If the true effect is closer to Block S's,** the FA outcome is open.
- **The veto cannot be triggered by small increases:** the smallest detectable change is about +65 % of the historical rate.
- **The supervisor** is very unlikely to be vetoed on the development evidence.

### 3.4 The options compared

| Rule | What it asks | Advantages | Disadvantages | Validity, power and interpretation |
|---|---|---|---|---|
| **(a) Strict (Holm-rejected increase)** | "Is there clear evidence that v2 raises the RO hazard?" | The approved convention for every veto (Stage 3a, Phase 1, which rejected P1 by such a veto despite large benefits, Phase 4); no new number; the burden of proof is on harm; MDE ≈ +65–77 %, so not hypersensitive | A large, real but tolerable increase still vetoes. Non-rejection is **not** evidence of safety (no equivalence claim) | Valid and consistent. A veto means "not confirmed under the pre-registered safety criterion", not "unsafe" |
| (b) Non-inferiority, margin M | "Can we show the increase is below M?" | Gives a positive safety statement if passed | M is not grounded in any approved document. On Block C, v2 passes FA only for M ≳ 0.77 per 1 000 moves (+160 %). An M chosen now knows that number. A small M is stricter than (a) (noise alone fails it) | Defensible only with an M justified independently of the data (for example a safety standard). None exists here |
| (c) PPO-referenced bound | "Is v2 no worse than the PPO policies already in the forward system?" | A system-level argument: the forward system already admits PPO's rate | A cross-algorithm, unapproved reference that moves with PPO's behaviour. Development data show v2 passes easily (1.04 vs 1.79), so choosing it now is outcome-informed. It changes the hypothesis from "did retraining make MADQL riskier?" to "is it within the fleet's range?" | It weakens the criterion in exactly the case at issue. Contradicts "do not weaken a safety criterion to make v2 pass" |
| (d) Descriptive only | — | — | Contradicts the student's instruction | — |

### 3.5 Recommendation and its consequences

**Recommendation [R]: (a) strict, MADQL-v2 vs historical MADQL, in every confirmatory set (D4).** In addition, pre-register as contextual and non-gating:
- MADQL-v2 vs the forward PPO policies (per planner);
- the RO rate of every arm;
- RO per episode;
- definition (ii).

**Implications:**
- **Rejecting v2 whenever an increase is detected** means that its completion and efficiency gains, however large, cannot make it the forward MADQL if it is detectably riskier per move than the model it replaces. This is the same trade-off the project accepted for P1.
- **Allowing a margin** would require the student to state an acceptable RO hazard independently of these data (for example from an external safety requirement). Without such a source, a margin is a tuning parameter.
- **Comparing with the historical MADQL** is the paired, controlled comparison. It answers whether Phase 4's change raised the hazard. The historical MADQL's rate is low partly because it waits and moves idle robots rarely. Per-move normalization removes the exposure part but not the behavioural difference. The comparison is still the right question for "did the retraining make MADQL riskier?".
- **Comparing with PPO** tells the reader where v2 stands in the fleet: below both PPO families on Block C. That is essential context for a veto. If v2 is vetoed, the report states that its per-move RO hazard is higher than the historical MADQL's but below the forward PPO policies'. Any decision to accept that is a *new*, explicit decision by the student after Phase 6, not part of the confirmatory rule.

### 3.6 Correction to the readiness report (§10)

- The readiness report quoted MADQL-v2's increase as 2.1–2.5× (Phase 4 post-hoc), the PPO rates as 1.0–2.1, and a supervisor effect on IPPO of 1.64 → 2.09.
- All of these used inconsistent or approximate denominators.
- The exact values are those in §3.2. The supervisor shows no RO effect. The PPO rates are 1.4–2.2 per 1 000 moves (A\*; IPPO 1.97 / 2.15, MAPPO 1.48 / 1.43, OFF / R + S + D).
- The readiness report keeps its text and carries a pointer to this correction.

---

## 4. Statistical and experimental-design review

### 4.1 Flaws found and proposed corrections

| # | Finding | Correction | Approval |
|---|---|---|---|
| F1 | The RO-per-move numerator and denominator must count the same robots. The Phase 4 post-hoc and readiness §10 did not | Definition (i), all robots, recorded by the observer (§3.1) | yes (part of D3) |
| F2 | Readiness §10's PPO and supervisor RO rates came from an approximate denominator | Corrected (§3.6); exact rates from the Phase 6 observer | — |
| F3 | H-L12 carried a P3 family with no decision role (12 hypotheses) | Drop P3 from H-L12; report the categories nominally (D6) | yes |
| F4 | The PPO-referenced context comparison existed only in development code | Add it to `phase6_analysis.py` as a nominal secondary before the freeze | part of the implementation step |
| F5 | The 3b rule maps a veto to "inconclusive", 3e to "no-go" and Phase 4 to "reject" | No change: these are approved rules. Every confirmation requires "go" / "adopt", so a veto blocks confirmation under all three. The report states this explicitly | — |
| F6 | The runner's `PROTOCOL` path points to the readiness report | Point it to the approved protocol document before the freeze | implementation |

No flaw was found in the pairing, the seed allocation, the arm composition or the outcome recording.
- Every arm of an instance shares the instance hash.
- Each comparison pairs the same (run, level, seed) or the same instance across runs (H-L12 PPO, context).
- Every required field is recorded: completion, tasks, flow time, conflicts by type on K^arb and K^ctrl, moves, stuck classes, supervisor statistics, planner work, safety counts, errors and runtime.

### 4.2 Do the arms answer the hypotheses?

| Hypothesis | Comparison | Pairing | Cells used |
|---|---|---|---|
| H-L2 | OFF vs R + S + D, the same checkpoint | (run, level, seed) | hist × {OFF, RSD}; v5 × {OFF, RSD}; A\* and RRT\* |
| H-v2 | hist vs v2 under the same condition | (training-seed pair k, level, seed) | {hist, v2} × RSD (A\*, RRT\*); {hist, v2} × OFF (A\*) |
| H-L12 | Issue #6 system vs Issue 6A system | (run, level, seed). For PPO: Issue #6 pick OFF → v5 pick RSD (ippo_seed2/3 unchanged picks: v5 OFF → v5 RSD). For MADQL: hist OFF → v2 RSD | i6 × OFF; v5 × {OFF, RSD}; hist × OFF; v2 × RSD |
| Context | v2 RSD vs v5 RSD | (level, seed); runs pooled per arm | existing cells, no new episodes |
| Baseline | descriptive | — | base × ORIG |

**No cell is unnecessary.**
- v2 × OFF (A\*) serves H-v2 NA.
- i6 serves only H-L12.
- The baseline is the approved descriptive comparator (Block D D4, D9).

### 4.3 Multiplicity, consistency and error control

**Per unit:**
- The union bound is ≤ 0.05: H-L2 and H-L12 use P1 0.03 + P2 0.005 + (P3 0.01) + P4 0.005; H-v2 uses P1 0.03 + P2 0.005 + E 0.01 + P4 0.005.
- Holm runs within each family. Not-evaluable rows enter at p = 1. Decisions use Holm-adjusted p only. Intervals are labelled nominal.
- Removing P3 from H-L12 leaves its union bound at 0.04.

**Across units:**
- There is no global control (D5); the bound is 1 − 0.95⁷ ≈ 0.30 for at least one false rejection under a complete null.
- Confirmation claims are conjunctions (§6.8), so a false confirmation is far less likely than this bound.

**Consistency checks:**
1. **Paired estimator:** a ratio of sums, the seed as cluster, levels summed (frozen).
2. **Intervals:**
   - nominal 95 % bootstrap-t, with the percentile interval stored;
   - the ceiling rule only lowers bounds (D7);
   - unbounded intervals are never rejections and never equivalence.
3. **No equivalence claim anywhere.**
   - The 3b condition "lower bound > −δ₁" and the Phase 4 condition "lower bound ≤ −δ₁ → inconclusive" use δ₁ = one task per episode, an approved, derived unit.
   - They are interpretability conditions, not equivalence tests. Non-rejection is reported as "not detected".
4. **Stage 3a criteria:**
   - the RO veto enters P4 exactly as the existing vetoes do (a Holm-rejected increase);
   - G0 is unchanged plus the Phase 6 checks;
   - no threshold is new except the inclusion of the RO outcome itself (required).
5. **MC sensitivity:** flagged as approved; an MC-sensitive decision is qualified, and B is not raised.

### 4.4 Sample size and power [P]

**Completion:**
- H-L2: MADQL SE_V ≈ 0.006 (A\*) / 0.009 (RRT\*), PPO ≈ 0.0009 / 0.001. The development effects (+0.37; +0.004 to +0.012) are many SE above the MDE (PPO A\* ≈ 0.003).
- H-v2 FA: SE_V 0.0004 against a δ₁ of 0.06. The FA completion bound is interpretable.
- H-v2 NA: SE_V 0.006, Block C effect +0.24.

**Efficiency (H-v2 E):** the development effects are 4–30 SE.

**RO veto:** as in §3.3.

**RRT\* on 60 seeds:**
- adequate for H-v2 FR (completion SE_V 0.0011) and the PPO RRT\* families;
- MADQL RRT\* RO rates are noisy (one policy has a heavy tail), so the H-L2 MADQL RO row on RRT\* has low power (MDE ≈ 2.2). Stated as a limitation.

### 4.5 Runtime [P, checked]

**Sum of per-episode wall times** (means from Phase 1 / 4 records):
- A\* ≈ 16 400 s;
- RRT\* ≈ 149 000 s.

**Earlier runs:** effective parallelism was 9.8–11.0 workers on 10–11 (Phase 1 1a / 1b, Phase 4 4s / 4c). So 165 500 s ÷ 10 ≈ 4.6 h. With the job-tail margin, the replay (≈ 2 %) and the pre-run check, the total is **≈ 5 h**. The estimate is consistent.

**Not included:** system pauses, ±30 % (RRT\* tails).

---

## 5. Recommended decision package

| ID | Decision | Existing approved position | Recommendation | Reason | Explicit approval |
|---|---|---|---|---|---|
| D1 | Held-out block | D7: 14 000–14 199, to be reserved | Reserve **Block V** 14 000–14 199, RRT\* 14 000–14 059, single-use | Unused, disjoint, matches the Block D split | **Yes** |
| D2 | Scope | §12 / §19 D: H-L2, H-L12; §34.1: v2 | H-L2 + H-v2 + H-L12 (22 080 episodes) | H-L12 is the only test of the integrated claim | **Yes** |
| D3 | RO veto rule | Student: RO per move is a pre-registered veto (rule open) | **Strict:** a Holm-rejected increase in P4, vs the matched old arm; no margin | The approved veto convention; no new threshold; MDE +65–77 %; the alternatives are outcome-informed or ungrounded | **Yes** |
| D4 | RO veto scope | Open | All of H-v2, H-L2 and H-L12. PPO context comparison and per-episode RO pre-registered, non-gating | One safety criterion for every system change | **Yes** |
| D5 | Multiplicity | Per stage / condition (Phases 1–4); per hypothesis (§12) | ≤ 0.05 per unit (7 units), global bound stated | Consistent with every approved protocol; conjunctions protect confirmations | **Yes** |
| D6 | Families / α | Stage 3a and Phase 4 splits | As approved + RO in P4; drop P3 from H-L12 | Union bound unchanged; removes 12 tests with no decision role | **Yes** |
| D7 | Ceiling | Phase 4 §5.4 (pre-registered there) | Phase 4 rule for every completion bound | Conservative; follows Phase 2 and the student's instruction | **Yes** |
| D8 | H-L12 rule | Stage 3a 3b / 3e | 3b (A\*), 3e (RRT\*) | Same kind of claim | **Yes** |
| D9 | RR conflicts | Phase 3 §8; Phase 4 veto (conflicts per episode) | Secondary, nominal (K^ctrl, K^arb); H-v2 keeps its veto | Keeps the union bound; avoids the K^arb artefact | **Yes** |
| D10 | P2 | Off; "may be considered"; 2026-11-23: disabled | Excluded | Latest instruction | Yes (confirm) |
| D11 | v2 RRT\* OFF | — | Not run | No hypothesis needs it | Yes (confirm) |
| D12 | Backups | — | Manifest (done) + 2 verified external copies, never `models/` | Single local copy today | **Yes** (destinations) |
| D13 | Sequence | Block D §9 | Approval → implementation → freeze approval → check → execution authorization → report | Enforced by the runner | **Yes** |

---

## 6. Complete draft confirmatory protocol (Phase 6)

Each element is marked **[A]** (already approved or frozen) or **[N]** (needs approval in this package).

### 6.1 Objective [A / N]

Confirm on held-out instances:
- **H-L2:** the supervisor R + S + D vs none, the same checkpoints [A, §12];
- **H-v2:** MADQL-v2 n3 vs the historical MADQL [A, §34.1];
- **H-L12:** the Issue 6A forward system vs the Issue #6 system [A, §12].

The baseline is descriptive [A, Block D D4].

**What Phase 6 cannot support:**
- method-level claims about SCB-C, DQN or PPO in general;
- other maps, sizes or robot counts (E5 / E6, Issue #8);
- stochastic execution, the shield or the sequential allocator;
- equivalence ("no cost") claims.

### 6.2 Held-out data [N]

- **Block V:** seeds **14 000–14 199** × {static, obstacles, full} = 600 instances (A\*); RRT\* subset **14 000–14 059** × 3 = 180 instances.
- **Single-use.** It is reserved at the approval of this protocol and never used for tuning.
- Every worker refuses every other block before the scenario generator, at three layers [C].
- **Pre-run check** (the first generation, after the freeze; aborts on failure):
  - 0 stored files with a Block V seed;
  - the instances unique per level and disjoint from every A, B, C, D and S instance of that level.
- **Training-instance disjointness:** reported as *not verifiable* (§2 D1).

### 6.3 Arms and sample sizes [N, from approved components]

**Common settings for every arm** [A]:
- greedy, auction allocator, shield off, endpoint recovery on;
- `path_limit` 6, horizon 400;
- P1 off, P2 off, RRT\* with 1 000 iterations;
- the production `run_episode`, one episode per instance;
- the frozen Stage 3a supervisor parameters (`freeze.json` `4c7b9dbc`).

**Conditions:**
- **OFF:** no supervisor, resolver off (the Issue #6 composition).
- **RSD:** `supervisor="R+S"`, resolver on (D), O off.
- **ORIG:** baseline only; no supervisor, resolver on.

**Policies and checkpoints** [A]:

| Policy | Runs | Checkpoint (SHA-256 prefix) |
|---|---|---|
| hist | madql_seed1–3 | Issue #5 `best.pt` (`dd9f088c`, `c66bd8c2`, `492f2a61`) |
| v2 | madql_seed1–3 | MADQL-v2 n3 `best.pt` (`652f2997`, `bb05d379`, `bdf781b9`) |
| v5 | ippo_seed1–3, mappo_seed1–3 | SCB-C v5 picks (verified by the Block D `picks` checks) |
| i6 | mappo_seed1–3, ippo_seed1 | Issue #6 picks |

**Matrix:**

| Planner | Cells | Instances per run | Episodes |
|---|---|---|---|
| A\* | hist {OFF, RSD}, v2 {OFF, RSD}, v5 {OFF, RSD}, i6 OFF, base ORIG | 600 | 17 400 |
| RRT\* | hist {OFF, RSD}, v2 RSD, v5 {OFF, RSD}, i6 OFF, base ORIG | 180 | 4 680 |
| **Total** | | | **22 080** (≈ 5 h on 10 workers) |

### 6.4 Outcomes [A unless marked]

- **Primary:**
  - completion = Σ delivered / Σ released (P1 per family; P2 per policy).
- **Vetoes (P4):**
  - high-priority flow time on tasks delivered in both arms;
  - high-priority undelivered tasks per episode;
  - **RO conflicts per 1 000 moves** [N, §3.1 definition (i)];
  - H-v2 also keeps conflicts per episode (Phase 4).
- **Targeted (P3, H-L2 A\* only):**
  - stuck robots per episode: frozen, oscillating, conflict_loop, deadlock_cycle (6A.0 classifier, re-labelling rule).
- **Efficiency (E, H-v2):**
  - planner calls per 100 robot-steps;
  - flow time of commonly delivered tasks;
  - supervisor changes per 100 robot-steps (FA, FR) or stuck robots per episode (NA).
- **Secondary, nominal** [N]:
  - RR conflicts (RRv + RRs) per 100 robot-steps on K^ctrl and on K^arb;
  - RO per episode;
  - RO per 1 000 moves under definition (ii);
  - the **context comparison** MADQL-v2 vs forward PPO (RSD) per planner;
  - conflicts per delivered task;
  - flow time, makespan, wait ratio, undelivered tasks, successful episodes;
  - planner calls by trigger;
  - stuck classes (also P3 for H-L12, nominal);
  - MADQL decision statistics.
- **Safety gates (G0), 0 required:**
  - physical collisions;
  - `PhysicalCollisionError`, `InvalidTransitionError` and other exceptions;
  - invalid moves;
  - supervisor-introduced invalid moves;
  - invalid reference paths.
- **Descriptive:** runtime (median and IQR of wall, planning and inference time); per level, per policy, per planner, per condition.

### 6.5 Statistical procedure [A]

- **Estimator:** the frozen `block_c_analysis.py` (`3fe86a77`): `paired_ratio`, symmetric seed-cluster `bootstrap_t`, `boot_index` (B = 10 000, `default_rng(0)`, one matrix per planner shared by every arm), `holm`.
- **Ratio of sums:** levels are fixed strata, summed per seed; family rows pool the family's runs per seed.
- **Tests:** two-sided p = (1 + #{|t\*| ≥ |t|}) / (B + 1); Holm within each family; decisions on Holm-adjusted p only.
- **Intervals:** nominal 95 % bootstrap-t (and the stored percentile interval), labelled "per hypothesis, unadjusted".
- **MC-sensitivity flag:** |p − own threshold| ≤ 2√(p(1 − p)/B); MC-sensitive decisions are qualified.
- **Not-evaluable rows:** p = 1, kept in the family, never rejected, no estimate published (Block D §7).

### 6.6 Families, α and multiplicity [A + N]

| Unit | P1 | P2 | P3 | E | P4 (vetoes) |
|---|---|---|---|---|---|
| H-L2 A\* | 3 (0.03) | 9 (0.005) | 12 (0.01) | — | 9 (0.005): high-priority flow, high-priority undelivered, **RO** × 3 families |
| H-L2 RRT\* | 3 (0.03) | 9 (0.005) | — | — | 9 (0.005) |
| H-L12 A\* / RRT\* | 3 (0.03) | 9 (0.005) | — [N] | — | 9 (0.005) |
| H-v2 FA / FR / NA | 1 (0.03) | 3 (0.005) | — | 3 (0.01) | 4 (0.005): conflicts per episode, high-priority flow, high-priority undelivered, **RO** |

Family-wise error ≤ 0.05 per unit [N, D5]. There is no global correction across the 7 units, and the report states the 0.30 bound.

### 6.7 Completion ceiling [A for Phase 4; N for Phase 6]

- **Unbounded bootstrap-t interval:** the lower bound used = min(percentile bound, loss-only bound −Σ_s max(0, d_old − d_new) / Σ released_old).
- **Never** a rejection, never equivalence; a finite interval is used as is.
- **Vetoes:** an unbounded row is never rejected.
- **δ₁** = 1 / mean released tasks per old-arm episode.

### 6.8 Decision rules [A] and confirmation statements [N]

**H-L2 and H-L12** (Stage 3a §6.3):
- **A\* ("3b"):**
  - "go" if:
    - G0 passes;
    - the MADQL P1 gain is Holm-rejected with Δ̂ ≥ δ₁;
    - no PPO P1 / P2 loss is rejected;
    - the PPO lower bounds are > −δ₁ (§6.7);
    - there is no veto.
  - "no-go" if G0 fails, a PPO loss is rejected, or the MADQL Δ̂ ≤ 0.
  - Otherwise "inconclusive".
- **RRT\* ("3e"):** "go" if G0 passes, no loss is rejected, the MADQL Δ̂ > 0 and there is no veto; otherwise "no-go".

**H-v2** (Phase 4 §5.4):
- "reject" if G0 fails, a completion loss is rejected, or a veto fires.
- "inconclusive" if the P1 lower bound is ≤ −δ₁.
- "inconclusive (mixed)" if a harmful E outcome is rejected alongside a benefit; "no benefit (efficiency harm)" if only a harmful E outcome is rejected.
- "adopt" if there is a rejected completion gain or E improvement.
- Otherwise "no benefit".

**Confirmation statements** (conjunctions):
- **The supervisor R + S + D is confirmed** iff H-L2 is "go" on A\* **and** RRT\*.
- **MADQL-v2 n3 is confirmed as the forward MADQL** iff H-v2 FA is "adopt", FR and NA are not "reject", and G0 and the replay pass. Otherwise the historical MADQL remains the forward MADQL, and v2 stays a documented, separately labelled configuration.
- **The integrated Issue 6A system is confirmed** iff H-L12 is "go" on A\* **and** RRT\*.
- **A veto blocks the corresponding confirmation under every rule**, whatever the other outcomes.

### 6.9 Integrity gates (G0) [A + N]

1. **Records:** 22 080 of 22 080; 0 duplicates; 0 unexpected.
2. **Safety:** 0 collisions; 0 errors; 0 invalid moves; 0 supervisor-introduced invalid moves; 0 invalid reference paths.
3. **Supervisor:** fairness and burst bounds hold; the parameters equal the Stage 3a freeze; the mode is R + S with the resolver in RSD; OFF has no supervisor and the resolver off; ORIG is the baseline with the resolver on.
4. **Configuration:** repair off, P2 off, 1 000 iterations.
5. **Instances:** identical across arms.
6. **Observer:** K^arb equals the recorded conflicts in every record [N]; K^ctrl is present for every learned arm.
7. **Checkpoints:** every hash is unchanged before and after the run.
8. **Replay:** every 50th record is identical in fresh processes, without the observer.
9. **Freeze:** the hashes match at check, run, replay and analysis. `analysis.json` is byte-identical on re-run under a fixed `PYTHONHASHSEED`.
10. **Protected files:** every file under `results/` (output excluded) and `docs/` is unchanged before and after the run.

### 6.10 Checkpoint backup and integrity [N]

**Already done [M]:**
- `tools/phase6/backup_manifest.py manifest` wrote `results/dev6a/phase6/backup/manifest_sha256.json` (SHA `87cb9ea6`). It holds 1 216 files, 1.45 GB:
  - the 9 MADQL-v2 run directories with every step checkpoint and log;
  - the Phase 4 selection;
  - the 9 historical MADQL / IPPO / MAPPO run directories.
- The 9 MADQL-v2 `best.pt` hashes equal `phase4_run.json`.
- `verify --root .` passed: 1 216 / 1 216 files.

**Procedure (by the student, or on explicit instruction):**
1. **First copy** to an external drive, outside the repository and never into `models/`:
   ```
   robocopy <repo>\results\dev6a\phase4\training <DEST>\results\dev6a\phase4\training /E /COPY:DT /R:1 /W:1
   ```
   Do the same for `results\dev6a\phase4\select` and the nine `results\training\<run>` directories.
2. **Verify the copy:**
   ```
   python tools/phase6/backup_manifest.py verify --root <DEST>
   ```
   Every file must match.
3. **Second copy** (cloud or another disk), verified the same way. Store the manifest with each copy.
4. **Protect:** mark the copies read-only. Never modify or re-save a checkpoint.
5. **Re-verify the originals** with `verify --root .` before the freeze. The freeze records the manifest SHA-256.
6. **After the run:** re-verify the originals (also G0.7).

### 6.11 Execution sequence and authorization [A, Block D §9]

1. The student approves this protocol and the decisions in §5, with any amendments.
2. Implementation of the approved items: the analysis constants, P3 removed from H-L12, the PPO context row, the protocol path, the report generator. This step comes with tests and **no Block V generation**.
3. Backups made and verified (§6.10).
4. **Separate approval to freeze**, then `phase6.py freeze --authorized-freeze`. The freeze hashes this protocol, the runner, the analysis, the reused frozen files and the tests.
5. `check --authorized-phase6`: the first Block V generation and every pre-run check.
6. **Separate authorization to execute**, then `run`, `replay` and `phase6_analysis.py`.
7. The report as a new proposal section, then stop.

**Rules for the run:**
- No re-analysis under another rule.
- A defect is reported, not fixed.
- A re-run needs approval and is labelled as such.

### 6.12 Reproducibility and audit [A]

- **Freeze record:** the hashes of the protocol, the code, the reused frozen files and the tests; the environment (Python, NumPy, PyTorch, platform); the design and episode counts; the backup manifest hash.
- **Manifest and logs:** every invocation is logged; checkpoint hashes are taken before and after the run.
- **Analysis:** `analysis.json` re-computed and compared byte for byte.
- **Report:** every table carries the estimate, nominal CI, raw p, Holm p, family α, own threshold, decision and MC flag. "Not evaluable" rows are listed with their unusable episodes.

---

## 7. Expected interpretations

| Outcome | Legitimate conclusion | Not a legitimate conclusion |
|---|---|---|
| **MADQL-v2 passes every gate** (FA "adopt", FR / NA not "reject", no veto) | MADQL-v2 n3 is confirmed, on held-out instances of this scenario family, as the forward MADQL. Its gains (completion without the supervisor, efficiency) are confirmed at their estimated sizes; no RO increase was detected (MDE ≈ +65–77 %) | That v2 has no RO cost. Non-rejection is not equivalence, and increases below the MDE are not excluded. Nor are claims about other maps or n-step DQN in general |
| **v2 improves completion but fails the RO veto** | v2 is **not** confirmed. The historical MADQL stays the forward MADQL. The report states both the confirmed gains and the confirmed RO increase (size, CI, the ratio to the historical rate and the PPO context). It also states that every RO conflict was prevented by the arbiter (0 collisions). Further work (for example a shield ablation or obstacle-aware training) needs a new approval | That v2 is unsafe in absolute terms, or that the veto was "too strict". The rule was fixed before the data and is not revisited |
| **Completion inconclusive** (lower bound ≤ −δ₁, or not evaluable) | The data cannot exclude a loss of one task per episode. The intervention is not confirmed for that unit | That there is "no difference", or "equivalence" |
| **Supervisor improves completion but increases a conflict metric** | If the increase is an RO veto (H-L2 P4), the supervisor is not confirmed for that planner, and the report says which family and by how much. If it is RR conflicts (secondary), it is reported nominally with the K^arb artefact noted. It does not block confirmation | Reading K^arb RR reductions as a safety gain (R lowers them mechanically) |
| **Integrated system shows no benefit over Issue #6** (H-L12 not "go") | The integrated claim is not confirmed. The component results (H-L2, H-v2) stand on their own, and the report shows which family or veto failed. If the failure is v2's RO veto, the fallback system (historical MADQL + R + S + D) is described by H-L2's MADQL family and H-L12's PPO families. It is not re-tested as a new hypothesis | Claiming the integrated benefit by chaining components, or re-defining the forward system after seeing the data |
| **All safety and performance gates pass** | Issue 6A's forward configuration is confirmed on held-out data: the supervisor, MADQL-v2 and the integrated system, at the estimated effect sizes. Practical importance is judged by those sizes (for example tasks per episode and flow time in steps), not by p-values | Statistical significance as practical importance. Generalization beyond the scenario family, the three training seeds per configuration and these checkpoints |

---

## 8. Risks, limitations and unresolved questions

1. **One block, one scenario family, fixed checkpoints, 3 seeds per configuration.**
2. **The RO effect is heterogeneous across training seeds** (§3.2). A pooled veto may be driven by one policy (seed 1). P2-like per-policy RO rows are reported nominally.
3. **The historical MADQL's low RO rate reflects its waiting style.** The veto answers "did retraining raise the hazard", not "is the hazard acceptable".
4. **Power:**
   - the H-L2 MADQL RRT\* RO row has low power (MDE ≈ 2.2 per 1 000 moves);
   - H-v2 NA's RO effect may go undetected (P ≈ 0.26–0.41).
5. **Per-unit error control:** global FWER ≤ 0.30 for at least one false rejection (D5).
6. **Training-instance disjointness** is not verifiable.
7. **Hypotheses were formed on Block C,** and Block S was seen. Block V is new, but expectations exist.
8. **Runtime ±30 %.**
9. **Unresolved** (outside Phase 6):
   - whether an increased per-move RO hazard below the PPO fleet level is acceptable. That is a policy decision for the student after Phase 6, not a criterion inside it;
   - the `models/` decision;
   - E5 / E6 (Issue #8).

---

## 9. Confirmation: no held-out generation or execution

- No scenario from seeds 14 000–14 199 was generated.
  - The development tools are restricted to Block C by the anchor guard: `dev_ro_exact.py` runs `_run_anchor` with the Block C worker guard.
  - `dev_ro_analysis.py` reads stored Block C and Block S records only.
  - `backup_manifest.py` reads checkpoints only.
- No Phase 6 episode was run.
- No freeze or check record exists (`results/dev6a/phase6/eval/` does not exist).
- The block is not reserved.
- The autouse test sentinel raises on any Block V generation. The full test suite passed with it.

## 10. Confirmation: protected artifacts unchanged

| Check | Result |
|---|---|
| Block C / Block D `plan_freeze.json` file hashes (3 + 35 entries) | all match |
| 1 310 historical files hashed before Phase 6A.0 (training runs and checkpoints, Issue #5 / #6 results, `SESSION_HANDOFF.md`) | 0 changed, 0 missing |
| Frozen statistics `block_c_analysis.py`, supervisor, Stage 3a freeze | `3fe86a77`, `83b84b8e`, `4c7b9dbc` (unchanged) |
| Phase 1–4 records, analyses, selections, manifests and the Phase 4 report (23 files checked) | unchanged |
| Checkpoints (MADQL-v2 ×9, historical MADQL, PPO) | the backup manifest verifies 1 216 / 1 216 files |
| `models/` | does not exist |
| `docs/SESSION_HANDOFF.md` | unchanged (`ba4a439c`, last modified 2026-11-19) |
| Block V seed fields in stored results | 0 in 868 files |
| Phase 6 freeze / check / record files | none (`results/dev6a/phase6/eval/` does not exist) |
| Tests | full suite **1 279 passed**, 1 skipped (POSIX-only), 0 failed; Phase 6 tool tests 25 passed |
| `ruff`, `mypy src`, `mypy` on the tools and tests | clean |
| Git | no Git or GitHub operation was performed |

**What this task added or changed** (all reversible, none protected):
- `tools/phase6/dev_ro_analysis.py`, `dev_ro_exact.py` and `backup_manifest.py` (new).
- `tools/phase6/phase6.py`: one observer field, `ro_arb_nonidle`. Authorization safeguards untouched.
- `tests/tools/test_phase6.py`: 2 tests added.
- Development outputs:
  - `results/dev6a/phase6/dev_ro/` (`77d7fdb8`);
  - `results/dev6a/phase6/dev_ro_exact/` (records `371265b5`, analysis `10c32956`);
  - `results/dev6a/phase6/backup/manifest_sha256.json` (`87cb9ea6`).
- This document, and a correction pointer in the readiness report.

---

## 11. Student decisions (2026-11-23)

The student approved the recommended decision package of §5 without change:

| ID | Approved decision |
|---|---|
| D1 | Seeds 14 000–14 199 reserved as single-use **Block V**; RRT\* subset 14 000–14 059 |
| D2 | All three confirmatory hypotheses: H-L2, H-v2, H-L12 |
| D3 | Strict RO veto: a Holm-rejected increase in RO conflicts per 1 000 moves (all robots) is a veto |
| D4 | The RO veto applies to H-v2, H-L2 and H-L12. The MADQL-v2 vs forward PPO comparison is contextual, pre-registered and non-gating |
| D5 | Family-wise error 0.05 per comparison unit over the seven units; the global bound is reported explicitly |
| D6 | The approved family structure and α; RO added to the P4 families; P3 removed from H-L12 |
| D7 | The Phase 4 completion-ceiling rule. Unbounded intervals are never equivalence and never evidence for an optimization |
| D8 | H-L12 uses the Stage 3a 3b (A\*) and 3e (RRT\*) rules |
| D9 | RR conflicts on K^ctrl and K^arb as secondary nominal outcomes; the H-v2 conflicts-per-episode veto kept |
| D10 | P2 excluded; RRT\* at the full 1 000-iteration budget |
| D11 | MADQL-v2 with RRT\* and supervisor OFF not run |
| D12 | Two SHA-256-verified backup copies (one external drive, one cloud-synchronized or another physical drive), never in `models/` |
| D13 | Protocol approval → implementation → verified backups → separate freeze authorization → freeze → pre-run check → separate execution authorization → execution, replay, analysis, report |

**Authoritative protocol:** `docs/research/2026-11-23-issue-6a-phase6-protocol.md`, which is §6 of this document, recorded as approved.

## 12. Errata (2026-11-23; no decision changed)

1. **§1, item 3 and §3.6: "2.1–2.5×" is a misquote.**
   - The Phase 4 report (§6.5) and the proposal (§33) state **1.9–2.5×**: FA 2.53, FR 2.09, NA 1.93, from `results/dev6a/phase4/eval/posthoc_conflicts.json`.
   - The readiness report did not quote a ratio.
   - The exact values of §3.2 (×2.17 / ×1.83 / ×1.43) and the conclusions are unaffected.
   - Full reconciliation: `2026-11-23-issue-6a-review.md` §9.1.
2. **Not-evaluable rows (protocol §5).**
   - "p = 1" is the protocol's Holm input for a not-evaluable row. It is not a computed p-value.
   - The analysis stores such a row with `evaluable: false`, `holm_input_p: 1` and no estimate, interval or Holm-adjusted p.
   - The report prints "not evaluable" with no p-value (Block D §7 rule 4).
3. **D13 timing amended (Amendment A1, 2026-11-23, the student's decision; before the freeze).**
   - The first Block V generation moves from the pre-run check to the first step of the run, after the separate execution authorization.
   - The pre-run check (`--authorized-precheck`) validates only the non-Block-V requirements.
   - A failed Block V check aborts before any episode.
   - Every other decision is unchanged. The text is in the protocol under "Amendment A1".
4. **D12 waived for this Phase 6 execution (Deviation W1, 2026-11-23, owner-approved; before the freeze).**
   - The two-copy backup prerequisite is replaced, for this execution only, by the recorded waiver `results/dev6a/phase6/backup/d12_waiver.json`.
   - D12 is **not satisfied**, and no backup record exists.
   - The originals must still verify at every gate.
   - Risk and text: protocol, "Deviation W1".
