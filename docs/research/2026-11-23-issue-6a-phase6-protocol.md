# Issue 6A Phase 6: confirmatory protocol (approved)

## Status

- **Approved by the student on 2026-11-23.** Decisions D1–D13 were approved as recommended in `2026-11-23-issue-6a-phase6-decision-review.md` (§5, and the record in §11 there). Every element marked [N] below is therefore approved. The marks are kept for traceability.
- **Not yet frozen.** The freeze requires its own explicit authorization (D13). At the freeze, `phase6.py freeze --authorized-freeze` hashes this document together with the runner, the analysis, the report generator, the tests, the reused frozen files and the backup record.
- **The text below is §6 of the decision review, unchanged.** Section numbers drop the "6." prefix. Interpretation guidance is in §7 of the decision review.
- **The analysis is run once,** with `PYTHONHASHSEED=0`, and must be byte-identical on re-run (§9).

Labels: **[A]** approved before this protocol; **[N]** approved with this protocol (D1–D13, 2026-11-23); **[M]** measured; **[C]** follows from code or definitions; **[P]** projection.

---

Each element is marked **[A]** (already approved or frozen) or **[N]** (needs approval in this package).

### 1 Objective [A / N]

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

### 2 Held-out data [N]

- **Block V:** seeds **14 000–14 199** × {static, obstacles, full} = 600 instances (A\*); RRT\* subset **14 000–14 059** × 3 = 180 instances.
- **Single-use.** It is reserved at the approval of this protocol and never used for tuning.
- Every worker refuses every other block before the scenario generator, at three layers [C].
- **Pre-run check** (the first generation, after the freeze; aborts on failure):
  - 0 stored files with a Block V seed;
  - the instances unique per level and disjoint from every A, B, C, D and S instance of that level.
- **Training-instance disjointness:** reported as *not verifiable* (§2 D1).

### 3 Arms and sample sizes [N, from approved components]

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

### 4 Outcomes [A unless marked]

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

### 5 Statistical procedure [A]

- **Estimator:** the frozen `block_c_analysis.py` (`3fe86a77`): `paired_ratio`, symmetric seed-cluster `bootstrap_t`, `boot_index` (B = 10 000, `default_rng(0)`, one matrix per planner shared by every arm), `holm`.
- **Ratio of sums:** levels are fixed strata, summed per seed; family rows pool the family's runs per seed.
- **Tests:** two-sided p = (1 + #{|t\*| ≥ |t|}) / (B + 1); Holm within each family; decisions on Holm-adjusted p only.
- **Intervals:** nominal 95 % bootstrap-t (and the stored percentile interval), labelled "per hypothesis, unadjusted".
- **MC-sensitivity flag:** |p − own threshold| ≤ 2√(p(1 − p)/B); MC-sensitive decisions are qualified.
- **Not-evaluable rows:** p = 1, kept in the family, never rejected, no estimate published (Block D §7).

### 6 Families, α and multiplicity [A + N]

| Unit | P1 | P2 | P3 | E | P4 (vetoes) |
|---|---|---|---|---|---|
| H-L2 A\* | 3 (0.03) | 9 (0.005) | 12 (0.01) | — | 9 (0.005): high-priority flow, high-priority undelivered, **RO** × 3 families |
| H-L2 RRT\* | 3 (0.03) | 9 (0.005) | — | — | 9 (0.005) |
| H-L12 A\* / RRT\* | 3 (0.03) | 9 (0.005) | — [N] | — | 9 (0.005) |
| H-v2 FA / FR / NA | 1 (0.03) | 3 (0.005) | — | 3 (0.01) | 4 (0.005): conflicts per episode, high-priority flow, high-priority undelivered, **RO** |

Family-wise error ≤ 0.05 per unit [N, D5]. There is no global correction across the 7 units, and the report states the 0.30 bound.

### 7 Completion ceiling [A for Phase 4; N for Phase 6]

- **Unbounded bootstrap-t interval:** the lower bound used = min(percentile bound, loss-only bound −Σ_s max(0, d_old − d_new) / Σ released_old).
- **Never** a rejection, never equivalence; a finite interval is used as is.
- **Vetoes:** an unbounded row is never rejected.
- **δ₁** = 1 / mean released tasks per old-arm episode.

### 8 Decision rules [A] and confirmation statements [N]

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

### 9 Integrity gates (G0) [A + N]

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

### 10 Checkpoint backup and integrity [N]

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

### 11 Execution sequence and authorization [A, Block D §9]

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

### 12 Reproducibility and audit [A]

- **Freeze record:** the hashes of the protocol, the code, the reused frozen files and the tests; the environment (Python, NumPy, PyTorch, platform); the design and episode counts; the backup manifest hash.
- **Manifest and logs:** every invocation is logged; checkpoint hashes are taken before and after the run.
- **Analysis:** `analysis.json` re-computed and compared byte for byte.
- **Report:** every table carries the estimate, nominal CI, raw p, Holm p, family α, own threshold, decision and MC flag. "Not evaluable" rows are listed with their unusable episodes.

---

---

## Amendment A1 (2026-11-23): timing of the first Block V generation

**Authority:** the student chose option B of `2026-11-23-issue-6a-review.md` §9.3 on 2026-11-23, before the freeze.

**Scope:** timing only. D1–D13 are otherwise unchanged. Specifically unchanged:
- the reserved seed designation (Block V 14 000–14 199; RRT\* 14 000–14 059);
- the scenario design, arms, sample sizes (22 080 episodes);
- the hypotheses, outcomes, statistical plan, thresholds, vetoes, decision rules and evaluation conditions;
- the backup requirement (D12).

**What it supersedes.** The following sentences are replaced by the rules below. Their original text is kept above, unchanged.
- §2: "Pre-run check (the first generation, after the freeze; aborts on failure): …".
- §11, step 5: "`check --authorized-phase6`: the first Block V generation and every pre-run check".
- §11, step 6: "Separate authorization to execute, then `run`, `replay` and `phase6_analysis.py`".

**Amended rules:**
1. **The pre-run check never touches Block V.**
   - It runs after the freeze, with `--authorized-precheck`.
   - It validates every requirement that does not involve Block V:
     - the freeze hashes;
     - 0 stored files with a Block V seed;
     - checkpoint identity and integrity;
     - the backup record and the originals (D12);
     - the Stage 3a freeze and escalation off;
     - the episode count.
   - It does **not** generate, materialize, fingerprint or otherwise instantiate any Block V scenario.
2. **The student's separate execution authorization** (`--authorized-execution`) is required before any Block V generation.
3. **The Block V instance check** is the first step of `run`, after that authorization and before any episode.
   - It generates the reserved Block V instances and fingerprints them.
   - It requires all of:
     - the instances to be unique per level;
     - the instances to be disjoint from every A, B, C, D and S instance of that level;
     - the protocol's design checks.
   - It is recorded once in `instance_check.json`. A resumed run re-uses it and never regenerates.
4. **If any Block V check fails, the run aborts before any episode,** and nothing is executed.
5. **The evaluation proceeds only if every gate passes:** the freeze, the pre-run check, the execution authorization and the Block V instance check.
6. **G0 addition** (consequential to rule 3): every record's instance hash must equal the fingerprint of the Block V instance check made before any episode (`instances_match_pre_episode_check`). This is the Block D §7 abort condition "an instance hash differing from the pre-run check", moved to the new check.

**The amended execution sequence** (replacing §11 steps 5–6):
1. **Pre-run check:** `check --authorized-precheck`. No Block V. The pre-run report follows, then **stop**.
2. **Separate execution authorization**, then `run --authorized-execution`. Its steps, in order:
   - the Block V instance check (abort on failure);
   - the 22 080 episodes;
   - `replay --authorized-execution`;
   - `phase6_analysis.py` with `PYTHONHASHSEED=0`;
   - the report.

---

## Deviation W1 (2026-11-23): owner-approved waiver of the D12 backup prerequisite

**Authority:** the project owner (student), on 2026-11-23, before the freeze. The owner's words: "We are explicitly skipping the backup procedure (D12) for now. I will personally create and maintain an independent backup … Record this as an explicit owner-approved waiver of the backup prerequisite for the current Phase 6 execution, with the associated risk documented."

**What it changes:** only the D12 gate of the freeze and the pre-run check.
- The gate passes with **either** two recorded, verified independent copies (D12 satisfied) **or** this recorded waiver.
- In both cases the original checkpoints must still verify against the manifest (1 216 files, `manifest_sha256.json`).
- The record is `results/dev6a/phase6/backup/d12_waiver.json`. The freeze and the pre-run check store its hash, with `d12_satisfied: false` and `d12_waived: true`.

**What it does not change:**
- D12 is **not satisfied** and is never reported as satisfied.
- No backup record exists, and no backup is marked verified.
- The D: staging copy remains a same-disk copy, not an independent backup.
- Every other decision, the amendment A1 and the statistical plan are unchanged.

**Risk (accepted by the owner):**
- The originals and the only verified copy are on the same physical disk. A disk failure would lose the MADQL-v2 and historical checkpoints, and with them the ability to reproduce or re-run Phase 6.
- Modification during the run is still detected: checkpoint hashes are taken before and after the run, and the originals are verified at the gates. It cannot be repaired without a copy.

**Reporting:** the Phase 6 report states "D12 waived by the owner; not satisfied".
