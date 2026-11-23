# Issue 6A Phase 3: Execution Supervisor design

*Approved for design (2026-11-21), then **implemented and software-tested** on the same day (D1–D11 approved). **Not experimentally evaluated:** calibration and evaluation need separate approval. The implementation, its deviations from §4 and the verification are in **§14**. **Stage 3a calibration is complete and frozen (§15); Stages 3b, 3c and 3e are executed and analysed (§16).** Sections 1–13 are the approved design record.*

**Labels.**
- **[M]** measured: from stored records or exact replays, Phase 0 (Issue 6A proposal §20 and §26).
- **[C]** read from the code.
- **[H]** a hypothesis that this design's experiments must test.
- **[D]** a design decision proposed here.
- **[A]** awaiting the student's approval (listed in §12).

Nothing in this document assumes that any mechanism will work. Each one is tied to a measured failure mode, and each has its own test of effectiveness (§9–§10).

## 1. Goal and the evidence it rests on

**Goal.** Raise multi-robot **task completion and reliability** for the learned controllers (MADQL, MAPPO, IPPO; A\* and RRT\*) at execution time, **without retraining**. Computation time is a secondary concern.

**Failure modes and evidence** (greedy block-B failures; Phase 0):

| Failure mode | Measured evidence | What any mechanism has to do |
|---|---|---|
| **Frozen** | IPPO 34 %, MADQL 24 %, MAPPO 6 % of stuck robots. The suggested next cell is free in 57/57, 1 010/1 014 and 9/10 cases. MADQL: Q(STAY) exceeds the best allowed move in **100 %** of 101 200 frozen robot-steps, with **0** exact ties and a median margin of +0.044. Frozen robots make **0** planner calls | Make the robot move along its (free) reference path when it does not make progress. Better tie-breaking cannot help: there are no ties |
| **Oscillating** | 42–53 % of stuck robots: at most 3 cells and at least 50 moves in the last 100 steps. They follow the suggestion in a median of 50 of 100 steps. They prefer moving, by the same margin as successful movement (median gap −0.069). About 54–59 planner calls per robot per 100 steps, ≥ 99.8 % of them off-path replans (T3) | Detect the cycle and make the robot follow the path long enough to leave it |
| **Conflict loop** | MAPPO 40 %, IPPO 13 % of stuck robots: in a conflict event in at least 50 of 100 steps. Contests over a **free** cell or a swap, cancelled by the arbiter every step. About 0 planner calls. None of them is a wait-for cycle | Let one contender go, deterministically and fairly |
| **Genuine wait-for cycle** | 1.4 % of stuck robots, 59 of these 62 also oscillating. Only 3 are stationary, persistent cycles | Use the existing resolver |

**[M] Why replanning optimization is not the primary lever.** Frozen and conflict-loop robots make almost no planner calls. Oscillating robots already receive suggestions and ignore half of them. Faster or fewer replans therefore cannot remove these failures (proposal §26.7).

## 2. What exists and what is new

| Component | Status | Where | Relevance |
|---|---|---|---|
| Arbiter (emergency stop): cancels every move in a conflict event, plus a cascade into frozen cells; no right-of-way | **Exists**, unchanged | `sim/conflicts.py: detect_conflicts, arbitrate` | Final safety layer. Guarantees no physical collision (Lemma 3.1) and is checked after every step (`physical_violations` raises `PhysicalCollisionError`) |
| Predictive shield: robot–obstacle only; replaces a move by STAY | **Exists**, unchanged | `control/shield.py`, engine pipeline step 5 | Applied after the supervisor (§3) |
| Deadlock resolver: wait-for cycles; T7 detours via a goal change | **Exists**, unchanged | `planning/deadlock.py`, `planning/coordinator.py` | Mechanism D (§4.5). Today it is enabled only for the baseline (`marl/evaluation.py: _system`, `deadlock=task.controller == "baseline"`) |
| Endpoint recovery: retreat and parking (T8) | **Exists**, unchanged, on for every controller | `planning/endpoint.py` | Handles robots blocked by idle robots on endpoints |
| Right-of-way rule (π̃, −i) and the "occupied / predicted obstacle → stay" filters | **Exists inside the baseline controller only** | `control/baseline.py` (rules 1–6), `allocation/benefits.py: robot_priorities` | Reused by R (§4.4) and by the recovery action of S (§4.2) |
| Static goal distance D̄ in the view | **Exists** (view extra `goal_distance`, on for learned controllers) | `sim/engine.py: _build_view`, `sim/view.py` | S's progress measure |
| Planner suggestion and reference path in the view | **Exists** | `WorldView.suggested`, `reference_paths` | S's recovery action |
| **Execution Supervisor layer** (state, R, S, O, logging) | **New** | proposed `control/supervisor.py` | — |
| Engine stage for the supervisor, before the shield | **New** (small, optional) | `sim/engine.py` | — |
| Explicit `deadlock` switch for any controller | **New** (configuration only) | `marl/evaluation.py: EvalTask`, `_system` | — |
| Additive record fields (supervisor statistics, calls by trigger) | **New** (only when the supervisor is configured) | `marl/evaluation.py`, `sim/metrics.py` | — |

## 3. Architecture

**Pipeline.** Steps 5 and 6 are unchanged; the supervisor stage is new:

```
policy proposal a_pol ──► Execution Supervisor ──► shield (optional) ──► arbiter ──► motion
                         (stage 4b, new; off by default)  (step 5, unchanged)  (step 6, unchanged)
```

- **[D] Engine stage, not a controller wrapper.** The supervisor is a separate engine stage with the same contract as the shield: `apply(view, proposal) → (proposal', events)`.
  - The engine keeps the raw policy proposal as `actions`. The conflict sets of the raw proposal (K^ctrl) and the adherence metrics therefore still describe **the policy**.
  - A new optional `supervised_actions` field records the supervisor's output.
  - A controller wrapper would hide the policy's own proposal from the metrics. The alternative is listed as decision D1.
- **Read-only inputs:** the `WorldView` (positions, modes, task priorities via `robot_priorities`, masks, `suggested`, `reference_paths`, `goal_distance`), the obstacle sensing used by the baseline (`planning/constraints.sense`), and, for D, the coordinator's detour membership (a new read-only accessor).
- **Output** for each robot, exactly one of: the policy's action, **STAY**, or the planner's **suggested** action. Nothing else can be produced.
- **State** per robot and episode: the best D̄ in the current leg, the no-progress counter, a short position history (K_osc), the recovery state and remaining hold, consecutive R-losses, and the current leg identifier (the goal). Reset at every episode start.
- **Determinism:** no randomness. Robots, events and cells are processed in fixed orders, so the supervisor's output is a pure function of the view history.
- **Off by default.** With the supervisor off, every record and trace is byte-identical to the current code. This is a regression requirement (§9).
- **Not used in training.** The policies are unchanged, so results are reported as "policy + supervisor".

## 4. Mechanisms

### 4.1 Common definitions

- **Eligible robot:** not idle, with a task goal `g_i`, where D̄_i = `goal_distance[i]` is defined.
- **Leg:** the interval with a constant task goal `g_i`. A goal change starts a new leg and resets the S/O state.
- **Progress:** D̄_i(t) < best_i, the smallest D̄_i reached so far in the leg. Each time this happens, best_i is updated and the no-progress counter is set to 0.
- **Excused step:** a step that neither counts towards a stall nor resets the counter, because waiting there is legitimate or handled elsewhere. It holds when any of the following is true:
  1. the suggestion is STAY (no valid plan, or at the goal);
  2. the suggested target cell is occupied by a robot or a dynamic obstacle, or is a predicted obstacle cell (baseline rules 3–4). This is interaction waiting, handled by trigger T4 and the resolver;
  3. the robot lost an R contest in the previous step;
  4. the robot is a member of an active resolver detour (D);
  5. the robot is in recovery.
- **No-progress counter `u_i`:** the number of consecutive non-excused, non-progress steps.

### 4.2 S: stall breaker (targets frozen robots; also catches slow oscillation)

**Activation (start of step t):**
- the robot is eligible;
- it is not in recovery or a detour;
- `u_i ≥ W` (stall window).

**Recovery action, for up to `H` steps:** the **baseline-filtered suggestion**:
- **STAY** if the suggestion is STAY, is masked, targets an occupied cell, or targets a predicted obstacle cell (baseline rules 1–4, reused);
- otherwise the suggested move.

Contests with other robots are settled by R (§4.4).

**[D] Why the suggestion.** In the frozen cases the suggested cell is free (≥ 99 %, [M]) and leads along the shortest reference path. The policy's own STAY preference is what blocks it ([M], 100 % positive gap).

**Termination**, at the first of:
1. `H` steps have passed;
2. progress of at least `Δ_prog` cells since the recovery started;
3. the goal is reached, the goal changes, or the robot becomes idle;
4. the robot becomes a detour member (control passes to D);
5. the suggestion is STAY for `H_stay` consecutive recovery steps (no plan; nothing to execute).

Afterwards `u_i = 0`, and the policy is in control again.

**Bounded override.** Each new recovery needs `W` further non-excused no-progress steps. The share of a robot's steps under S is therefore at most H / (H + W) between progress events. With H = 8 and W = 12 that is 40 %; it is reported per episode.

**What S cannot fix.** A robot whose suggested cell stays occupied is excused and left to T4, the resolver and endpoint recovery.

**[H] Expected effect:**
- frozen robots move along their path;
- since 23 % of MADQL frozen steps have a STAY preference within 0.01, the policy may continue moving once displaced;
- the alternative is re-freezing after the hold, which the re-trigger count measures.

### 4.3 O: oscillation detector (fast trigger for S)

**Activation:** the robot is eligible, not in recovery or a detour, and over its last `K_osc` steps:
- it occupied at most 2 distinct cells;
- it made at least `m_osc` moves;
- it made no progress.

O then **starts an S recovery immediately**, without waiting for `W`.

**[D] Why a separate detector.** In an A↔B cycle D̄ alternates, so best_i stops improving and `u_i` grows. S alone would catch it, but only after `W` steps. O catches the classic two-cell limit cycle within `K_osc` steps.

**Anti-backtrack during an O-triggered recovery:** if the filtered suggestion would step back onto the cell occupied at t − 1, and that cell is not the next path cell, STAY instead. This should not happen with a fresh path; it is a guard against flip-flopping between replans.

**[H] Expected effect:** oscillating robots follow the path for `H` steps, leave the cycle and reduce their T3 replans, since a robot on its path makes no off-path calls. This is a side benefit for RRT\* runtime.

**Risk:** false triggers on legitimate local manoeuvres. Calibration (§5) uses the false-trigger rate in successful episodes.

### 4.4 R: right-of-way for robot–robot conflicts (targets conflict loops)

**Input:** the joint proposal after the S/O substitutions.

**Detection:** robot–robot conflicts (`RRv`, `RRs`) found by the production `detect_conflicts` on that proposal. The robot–obstacle and invalid-move kinds are left to the shield and the arbiter.

**Rules** (deterministic, applied to a fixed point):
1. **Vertex contest on a free cell.** If at least 2 movers target the same cell and no robot stays on it, the mover with the highest key keeps its move and the others → STAY (R-loss).
2. **Move into a staying robot's cell:** every such mover → STAY. The arbiter would cancel it anyway, so motion is unchanged; it is logged.
3. **Swap (head-on exchange):** both → STAY, which is the arbiter's outcome.
   - R **cannot** resolve a swap: letting one robot move into the other's still-occupied cell is a vertex conflict.
   - A persisting swap is a two-robot wait-for cycle, so it is **D's** case (§4.5). This dependency is explicit and tested.
4. **Repeat** detection after each change. Turning a move into STAY can create a new rule-2 conflict for a robot following into that cell. Each iteration converts at least one move into STAY, so there are at most N iterations.

**[A] Key.** Proposed: `(f_i ≥ L_max, π̃_i, f_i, −i)`, compared lexicographically.
- `f_i` is the robot's number of consecutive R-losses.
- `π̃_i` is the effective (aged) task priority from `robot_priorities`: 0 for an idle robot, ≥ 1 otherwise.
- **[D] Starvation bound:** a robot that has lost `L_max` times in a row outranks any robot below that count, whatever its priority. In pairwise contests, consecutive losses are therefore bounded by L_max + 1 (property test, §9).
- This differs from the priority-first key of proposal §5, whose aging alone gives no explicit bound. Decision D3.
- `f_i` resets to 0 on any step without an R-loss.

**Minimal override.** R changes only robots involved in a robot–robot conflict. Every other robot's action passes through unchanged.

**[H] Expected effect:** in a conflict loop over a free cell, one robot passes each time and the loop dissolves. This is tested on conflict-loop robots (§10).

### 4.5 D: deadlock recovery (existing resolver)

- **Existing mechanism, used unchanged:** wait-for cycle detection (persistent: 2 steps; recurrent: 3rd occurrence within 20 steps) and passing-cell detours via goal changes (T7), with release, abort and retry limits (`planning/deadlock.py`).
- **[C]** It never chooses actions. It changes only the goal the monitor plans to, so the suggestion and reference path point to the passing cell.
- **New:** only the explicit switch that enables it for learned controllers (`EvalTask.deadlock`, defaulting to the current behaviour), plus a read-only accessor for detour membership.
- **Interaction with S:**
  - detour members are excused from S's stall counting, because D̄ to the *task* goal rises during a detour;
  - S's filtered-suggestion logic may be applied to a detour member that fails to follow its detour suggestion for W steps ([A], decision D2b), so that the detour is actually executed.
- **Scope [M]:** genuine cycles are 1.4 % of stuck robots. D's main role here is to resolve the swaps that R deliberately leaves standing (§4.4, rule 3).
- **[H] Known limitation:** the observations of learned controllers show the passing cell as the goal, which they never saw in training. The factorial (§10) measures whether D helps or hurts them.

### 4.6 Order within a step (per robot, then joint)

```
for i in ascending id:                        # state update from the history up to t-1
    update best_i, u_i, position history, leg; mark excused conditions
    if eligible and not in recovery and not detour member:
        if O(i) fires: start recovery (reason O)
        elif u_i >= W: start recovery (reason S)
    a[i] = recovery_action(i) if in recovery else a_pol[i]
a = R(a)                                      # fixed point over robot–robot conflicts
log reasons; return a  →  shield  →  arbiter
```

**Precedence:** D (goals, upstream in the coordinator) → S/O (per robot) → R (joint) → shield → arbiter. The arbiter always has the last word.

## 5. Parameters and how they are set

| Parameter | Meaning | Proposed starting point [A] | How to set it |
|---|---|---|---|
| `W` | Stall window (non-excused no-progress steps) | 12, grid {8, 12, 16} | The 99th percentile of no-progress run lengths in **successful** greedy episodes, per controller family, so that normal behaviour rarely triggers S |
| `K_osc`, `m_osc` | Oscillation window and minimum moves | 8, 4 | False-trigger rate in successful episodes, target ≤ 1 % of robot-legs |
| `H` | Recovery hold | 8, grid {4, 8} | Sensitivity analysis on Block C |
| `Δ_prog` | Progress that ends a recovery early | 2 cells | Fixed (design) |
| `H_stay` | Stop a recovery after consecutive STAY suggestions | 3 | Fixed (design) |
| `L_max` | R fairness threshold | 3 | Property test plus fairness metrics |
| D parameters | Resolver defaults (`DetourParams`) | Unchanged | Unchanged |

**[A] Calibration data.** Calibrate W, K_osc and m_osc from a deterministic **replay of the stored Block B greedy episodes** (development data, no new experiment, as in Phase 0b), using the no-progress statistics of robots in *successful* episodes. Then freeze the values before Block C. The Phase 3 evaluation on Block C then does not tune on itself, and the final validation (Phase 6) uses a new held-out block.

## 6. Interaction with controllers and planners

- **MADQL:** frozen (24 %) and oscillating (53 %) robots dominate its failures, and 710 of 720 greedy episodes fail.
  - [H] S and O have the largest expected effect here.
  - R matters little: conflict loops are 0.2 % of its stuck robots.
- **MAPPO:** conflict loops (40 %) and oscillation (46 %). [H] R and O are the main levers.
- **IPPO:** frozen (34 %) and oscillation (42 %). [H] S and O.
- **The policy keeps control except** during a bounded recovery or an R-loss; both are logged per step with their reason. The share of non-idle robot-steps that the supervisor modifies is a reported metric (§8).
- **A\* and RRT\*:** S follows the suggestion from either planner unchanged.
  - With RRT\*, the reference path can change at every replan because each call is reseeded.
  - S's filters and the anti-backtrack guard apply equally.
  - A budget failure gives a STAY suggestion; the step is excused, and `H_stay` ends the recovery.
  - [H] Fewer off-path steps reduce RRT\* calls, a computational side benefit that is measured, not claimed.
- **Baseline and reference follower:** the supervisor is **not** applied by default ([A], decision D9). The baseline already plays the right-of-way rule and has no stall failures of this kind.
- **Training:** unchanged; the policies never saw the supervisor.

## 7. Safety

**Properties** (to be proved or property-tested, §9):
1. **Output alphabet:** each robot's supervisor action is the policy action, STAY, or the suggested action.
2. **No invalid moves introduced:** STAY is always allowed, and the recovery action is used only if it passes the static mask (baseline rule 2). An `INV` event on the supervisor's output can therefore only come from the policy's own proposal passing through unchanged.
3. **No physical collisions:** every supervisor output passes through the optional shield and the unchanged arbiter, and Lemma 3.1 holds for any proposal. The engine's post-step check raises on any violation, so a violation could not go unnoticed.
4. **No illegal task transitions:** the supervisor never touches tasks, allocation or goals. D changes only monitor goals, through the existing resolver.
5. **R reduces conflicts and adds none:** it only turns moves into STAY, and it iterates to a fixed point with no rule-1 or rule-2 conflict among its robots.
6. **Bounded intervention:** each recovery lasts at most `H` steps, re-triggering needs `W` new steps, and in pairwise contests consecutive R-losses are bounded by L_max + 1.
7. **Oscillation control:** recovery ends on progress, after H steps, or when there is no plan, and the anti-backtrack guard applies. The supervisor cannot create an endless override.
8. **Off means unchanged:** byte-identical behaviour when the supervisor is off.

The arbiter, shield, resolver and endpoint recovery are not modified.

## 8. Metrics

**Every arm, per policy × planner × level:**
- task completion (primary);
- undelivered tasks, and successful / incomplete / failed episodes;
- physical collisions, invalid moves (INV), exceptions and illegal transitions;
- **conflicts on the policy's own proposal (K^ctrl)** and on the proposal reaching the arbiter (K^arb), each by type.
  - **Warning:** R trivially lowers K^arb, because it turns cancelled moves into STAY, so K^arb alone would overstate its benefit. Effectiveness is judged by completion and K^ctrl.
- conflicts per episode, per delivered task and per 100 robot-steps, kept apart as in Block C/D;
- wait ratio, flow time (also by task priority, for fairness), makespan;
- planner calls by trigger (T1–T8) per 100 robot-steps;
- supervisor interventions by reason (S, O, recovery hold, R-loss vertex, R-loss swap, R-cascade); the share of non-idle robot-steps modified; recoveries that ended with progress; re-triggers per leg;
- fairness: the maximum and distribution of consecutive R-losses, and waits per robot;
- stuck-robot classes at the end of failed episodes (the 6A.0 classifier), to show **which failure mode moved**;
- step-time overhead of the supervisor.

## 9. Testing strategy

- **Unit tests:**
  - the progress tracker and the excused-step rules;
  - S triggering exactly at `W`;
  - O on synthetic A↔B and three-cell trajectories, and not on straight runs;
  - every recovery termination condition;
  - R winner selection, rules 1–3 and the fixed point;
  - the key and fairness counter;
  - reset per episode and per leg;
  - determinism.
- **Property tests** (Hypothesis, random grids and joint proposals):
  - the output alphabet;
  - no invalid move introduced;
  - after R, no rule-1 or rule-2 robot–robot conflict among robots that R touched;
  - the R-loss bound in pairwise contests;
  - the arbiter's physical-collision check never fires on any supervisor output;
  - supervisor off ⇒ identity.
- **Integration scenarios** (scripted controllers, no training):
  - a "frozen" controller that always stays in front of a free cell → S delivers the task;
  - a scripted A↔B oscillator → O triggers and the robot progresses;
  - two robots repeatedly contesting one free cell → R lets one pass and both deliver;
  - a head-on corridor swap → R stays both, D detours, both deliver (supervisor + resolver);
  - a robot waiting behind an occupied cell → excused, with no S trigger.
- **Regression:**
  - supervisor off ⇒ byte-identical records and traces, both from the existing test suite and from a deterministic replay of stored Block B episodes (as in Phase 0b);
  - MADQL and Issue #5 regression tests unchanged.
- **Performance:** per-step overhead measured on the NFR-PERF-1 workload.

## 10. Experiments (each stage needs separate approval [A])

**Development block:** Block C (reusable, every use logged).

**Policies:**
- MADQL 1–3 (the Issue #5 picks);
- MAPPO 1–3 and IPPO 1–3 (the frozen SCB-C v5 picks).

**Statistics:** the established method (seed-cluster bootstrap-t, paired by instance), with results labelled as development.

**Projected costs** below use measured unit costs: PPO A\* about 0.33 s of wall time per episode on 10 workers (Block C); MADQL A\* about 1.6–1.8 s per episode, run in parallel.

| Stage | Content | Episodes | Projected wall time |
|---|---|---|---|
| 3a Calibration | Replay of stored Block B greedy episodes, recording no-progress and oscillation statistics in successful episodes | 0 new (2 160 replays) | ≈ 4 min |
| 3b Core | Greedy A\*, supervisor off vs full (R + S + O + D), 9 policies × 360 | 6 480 | ≈ 0.5–0.7 h |
| 3c Factorial | Greedy A\*, 2³ over {R, S+O, D} (6 extra arms) plus one "S without O" arm | +22 680 | ≈ 2–2.5 h |
| 3d Stochastic (optional) | Supervisor off / full × 3 replicates | +19 440 | ≈ 2 h |
| 3e RRT\* | Off vs full on the 40-seed RRT\* subset × 3 levels | 2 160 | PPO ≈ 0.6 h; MADQL ≈ 2 h |
| 3f Shield check (optional) | Full level, supervisor × shield on | +2 160 | ≈ 0.3 h |

**Primary development questions** (per policy, Holm across the 9 policies):
1. Does the full supervisor change completion relative to off?
2. Do the targeted failure modes shrink: frozen and oscillating robots under S/O, conflict-loop robots under R?

**The factorial** gives each mechanism's main effect and their interactions. Expected interactions: D only acts on cycles, which R can prevent or leave standing (swaps), and S can pre-empt D.

**Acceptance criteria for Phase 3** (to be pre-registered before Stage 3b) [A]:
1. **Safety:** 0 physical collisions, 0 exceptions, 0 illegal transitions, and 0 invalid moves introduced by the supervisor, in every arm.
2. **Regression and determinism:** supervisor off ⇒ byte-identical; replays identical.
3. **Effect:** completion differences with confidence intervals, per policy and planner. A "go" towards Phase 6 needs a decision rule fixed in advance (decision D10). The proposal is an improvement for MADQL and no detected completion reduction for any PPO policy, with no hidden margin.
4. **Mechanism attribution:** a mechanism is kept only if its factorial main effect on completion is non-negative **and** it measurably reduces its targeted failure mode. Otherwise it is dropped, which keeps the design simple.
5. **Costs reported:** override share, wait ratio, flow time by priority, R-loss fairness and step-time overhead. No acceptance criterion hides a cost.

**Final, independent validation** belongs to Phase 6, on a new held-out block (proposal: seeds 14 000–14 199, not yet reserved).

## 11. Possible negative side effects

1. **Forced moves into congestion.** The policy's STAY may have been right, for example waiting for a robot to pass. The filters cover only occupied and predicted cells. Risk: more conflicts or longer waits after recovery. Measured by re-triggers, waits and conflicts on the policy proposal.
2. **False S triggers on legitimate detours,** when D̄ temporarily rises. Mitigated by the excused-step rules and the calibration.
3. **R starvation and unfairness.** Strict priority favours high-priority tasks, and idle robots always yield. The fairness override bounds consecutive losses. Flow time by priority is reported.
4. **R is conservative in following chains,** because rule 2 stops the follower. This may slow traffic. Measured by wait ratio and makespan.
5. **Swaps depend on D.** Without D, R leaves head-on swaps unresolved; the factorial shows this explicitly.
6. **Resolver goals unseen in training** may confuse learned policies. Measured by the D main effect.
7. **The policy's weaknesses are masked, not fixed.** Results are reported as "policy + supervisor", always next to the unsupervised results.
8. **Metric artefact:** K^arb drops trivially under R (§8).
9. **Distribution shift after recovery** may create new oscillations (re-trigger counts).
10. **RRT\* path variability** (reseeded replans) may reduce the value of following the suggestion.

## 12. Decisions requiring the student's approval

| # | Decision | Recommendation |
|---|---|---|
| D1 | Integration point: an engine stage before the shield, or a controller wrapper | Engine stage (keeps the policy proposal and its metrics) |
| D2 | Mechanism set: R, S, O, plus the existing D; the S-rand (randomized) variant excluded | As stated; S-rand only if requested |
| D2b | Apply S's filtered suggestion to detour members that ignore their detour | Yes, after `W` steps |
| D3 | R key: fairness override `(f ≥ L_max, π̃, f, −id)`, or the priority-first key of §5 | Fairness override (explicit starvation bound) |
| D4 | Calibration from a deterministic replay of stored Block B episodes (Stage 3a) | Yes |
| D5 | Policies: the MADQL Issue #5 picks and the PPO SCB-C v5 picks | Yes |
| D6 | Evaluation stages and budget (3b–3f), including MADQL with RRT\* | 3b + 3c + 3e first; 3d and 3f optional |
| D7 | Development on Block C; final validation in Phase 6 on a new block (14 000–14 199) | Yes; reserve the block before Phase 6 |
| D8 | Additive record fields only when the supervisor is configured | Yes (historical byte-identity) |
| D9 | Apply the supervisor to learned controllers only | Yes |
| D10 | Go / no-go rule for Phase 3, and whether to pre-register a completion margin | CI-based, fixed before Stage 3b |
| D11 | Cap on recoveries per leg | No hard cap; the H / (H + W) bound plus logging |

## 13. Implementation plan (implemented; see §14)

**Files to create or change:**
- `src/robocoord/control/supervisor.py` (new): `SupervisorParams`, `ExecutionSupervisor` (state; S, O, R), `SupervisorEvent`, `stats()`.
- `src/robocoord/sim/engine.py`: optional supervisor stage before the shield; `StepOutcome.supervised_actions`; `extra["supervisor"]`. Without a supervisor, the code path is identical.
- `src/robocoord/marl/evaluation.py`: `EvalTask.supervisor = "off"` and `EvalTask.deadlock: bool | None = None` (None means the current rule); additive record fields only when the supervisor is on.
- `src/robocoord/planning/coordinator.py`: a read-only accessor for detour membership.
- Tests under `tests/unit/control/`, `tests/integration/`, plus property tests and a regression replay.
- **Documentation after approval:** `02` (execution layers), `03` (ADR-10 extension), `04` (algorithm), `05` (experiments).

**Estimated engineering:** 2–3 days, including tests [projection].


## 14. Implementation status (2026-11-21): implemented and tested; not yet evaluated

The student approved D1–D11 and authorized implementation and software testing on 2026-11-21. Experimental evaluation (Stages 3a–3f) is **not** authorized yet, and **no Block C or Block D simulation, training or evaluation run was made.**

Everything below is software-level evidence: correctness, safety, determinism, behaviour on hand-built scenarios, and overhead. **Nothing here shows that the supervisor improves completion or reduces conflicts on the evaluation distribution.** That requires Phase 3 experiments.

### 14.1 What was implemented

| Item | Where | Notes |
|---|---|---|
| Supervisor: R, S, O and the D2b detour recovery; statistics; events | `src/robocoord/control/supervisor.py` (new) | `ExecutionSupervisor(mode, SupervisorParams)`; modes `off` / `full` / `R+S+O` subsets |
| Engine stage 5a (supervisor) before stage 5b (shield) | `src/robocoord/sim/engine.py` | `SupervisorLayer` protocol; `StepOutcome.supervised_actions`, `supervisor_events`; trace keys `supervisor`, `sup_actions` (only when it changed something); `extra["supervisor"]` |
| Detour membership accessor | `src/robocoord/sim/coordinator.py` (`DetourReporting`), `src/robocoord/planning/coordinator.py` (`detour_members()`) | Read-only; empty without the resolver |
| Explicit resolver switch; supervisor in evaluation | `src/robocoord/marl/evaluation.py` | `EvalTask.supervisor = "off"`, `EvalTask.deadlock: bool | None = None`; `resolver_enabled()`. The supervisor is for learned controllers only (D9). The additive record fields `supervisor`, `supervisor_stats` and `calls_by_trigger` appear only when the supervisor is on (D8) |

**Not changed:**
- the arbiter, the shield, the resolver, endpoint recovery and the planners;
- MADQL and PPO, and the policy class;
- the training environment and the rewards;
- the allocation;
- configurations, checkpoints and records.

### 14.2 How the four mechanisms interact (as implemented)

**Per step, inside the supervisor**, after D has changed goals upstream in the coordinator:
1. For each robot in ascending id: update the leg, the best D̄ and the stall counter; check the recovery termination conditions; check the triggers (D2b for detour members, otherwise O, then S); during a recovery, substitute the filtered suggestion.
2. Then R on the joint proposal.

The supervised proposal then goes to the shield and the arbiter, which are unchanged and keep the last word.

**Implemented definitions:**
- **Eligible:** the robot is not idle, has a goal, and has a D̄.
- **Excused** (neither counted nor reset): the filtered suggestion is STAY (no plan, masked, robot or obstacle ahead, predicted obstacle); the robot lost an R contest in the previous step; or it is a detour member.
- **S:** a stall counter ≥ `stall_window`.
- **O:** the last `osc_window` steps cover ≤ `osc_cells` cells with ≥ `osc_moves` moves, and there has been no progress for ≥ `osc_window` steps.
- **D2b:** a detour member whose policy declines a movable filtered suggestion for `stall_window` consecutive steps gets a recovery of kind `D`, which ends when the detour ends.
- **Recovery ends** on `progress_cells` of progress (S / O), `hold` steps, `stay_limit` consecutive STAYs, a **revisited cell** (`cycle`), a leg change, or joining or leaving a detour.
- **R key:** `(losses ≥ L_max, π̃, losses, −id)`, with π̃ from `robot_priorities`, as for the baseline.

### 14.3 Deviations from §4, found in testing and fixed

1. **Anti-backtrack replaced by a recovery-cycle guard.**
   - *Found:* in the scripted oscillation scenario, the design's rule ("during an O recovery, a step back onto the cell of t − 1 becomes STAY") blocked the planner's legitimate suggestion twice. After an off-path move, the path often leads back through the previous cell, so the rule only delayed recovery.
   - *Fix:* a recovery of any kind now ends as soon as the robot moves onto a cell it already occupied during that recovery. This still prevents a recovery from oscillating (for example under alternating RRT\* replans) without blocking path moves. The scripted case recovered in 44 instead of 46 steps.
2. **R changes only free-cell contests.**
   - *Found:* in the legitimate-waiting scenario, R's rules 2 and 3 rewrote 36 robot-steps (moves into a staying robot's cell). The motion was identical, because the arbiter cancels them, but the rewrites inflated the override statistics and removed the policy's conflicts from the arbiter's record.
   - *Fix:* moves into a staying occupant's cell and head-on swaps are now **left to the arbiter** and only counted (`r_left_to_arbiter`). R changes robots only in vertex contests for a free cell.
   - One pass suffices: turning movers into STAY can never create a new free-cell contest.
   - Swaps are still not resolved by R; the resolver resolves them, as designed and tested.
3. **Optional escalation, added off by default (needs approval).**
   - *Found:* a persistently frozen policy is recovered in short bursts. Each S recovery ends after 2 cells of progress, then the policy stalls again for a full `stall_window`. In the scripted corridor it delivered in 54 steps.
   - *Option:* `SupervisorParams(escalate=True)` makes a robot that stalls again in the same leg re-trigger after `repeat_window` (4) and hold the full `hold`. Measured: 34 steps, and an override share of 24 % against 15 %.
   - This loosens the approved worst-case override bound (D11) from H / (H + W) to H / (H + repeat_window). It is therefore **not enabled** and is proposed for the ablation (§14.6).
4. **Metric conventions** (not in §4):
   - `StepOutcome.actions` and the adherence and K^ctrl metrics use the **policy's** proposal.
   - The wait-ratio counters (`record_step`) use the **supervised** proposal, the requested actions before the shield, as they use the controller's proposal before the shield today.
   - `supervisor_stats` reports `policy_stays_nonidle` and `executed_stays_nonidle` separately.

### 14.4 Tests and results

| Suite | Tests | Result |
|---|---|---|
| `tests/unit/control/test_supervisor.py`: unit tests of S, O, R, D2b, the termination conditions, fairness, events and statistics, configuration, escalation, plus **property tests** (Hypothesis): R leaves no shared free target, is idempotent and only stops movers; the supervisor's output is ⊆ {policy, STAY, suggestion} and mask-valid; arbiter + post-step check never report a violation; determinism | 31 | pass |
| `tests/integration/test_supervisor_integration.py`: frozen robot (A\*, RRT\*); oscillation; conflict loop over a free cell; head-on swap (R alone unresolved; resolver resolves; one-lane unresolved); legitimate waiting not overridden; MADQL-style and PPO-style `LearnedController` × A\*/RRT\* × shield off/predictive (output alphabet, masks, no introduced invalid moves, no collision); policy proposal kept, K^ctrl recorded; trace keys; detour accessor; **byte-identical traces with the supervisor off** (3 scenarios) | 17 | pass |
| `tests/integration/test_supervisor_regression.py`: `run_episode` with default fields **reproduces stored block-B records exactly**, all non-timing fields **and field order** (MADQL greedy, IPPO greedy, MAPPO stochastic, matched baseline); the resolver-on baseline's values; the additive fields when on; learned-only rule; explicit resolver switch | 13 | pass (skipped without `results/`) |
| Full fast suite | see the Issue 6A proposal §28 | — |

**Scripted outcomes** (software behaviour on hand-built scenarios, not evaluation evidence):

| Scenario | Off | On |
|---|---|---|
| Frozen robot | 0 / 1 delivered | delivered |
| Oscillating robot | 0 / 1 | delivered |
| Two plan-followers contesting a free crossing cell | 0 / 2, 58 conflicts | 2 / 2 in 7 steps, 1 R decision |
| Head-on swap, 2 lanes | — | resolved only with the resolver, in 15 steps with or without the supervisor |
| Head-on swap, 1 lane | unresolved | unresolved |

**Overhead** (an 8-robot full-dynamics instance, plan-followers, 400 steps): the supervisor takes about 0.04 ms of a 0.82 ms step. Episode time is unchanged within measurement noise.

### 14.5 Limitations still open

- **Parameters are design values, not calibrated:** `stall_window`, `hold`, `osc_*`, `L_max`. Calibration is Stage 3a.
- **False triggers.** A robot following a long detour, where D̄ does not improve, can trigger S. Its recovery action then equals its own (correct) action, so nothing changes, but the recovery is counted. A legitimate side-step manoeuvre can trigger O. Both are measured by the calibration statistics.
- **R and the shield.** R decides before the shield. If a shield then stops R's winner, the losers waited one step needlessly. This is rare (the predicted-obstacle filter uses the same `sense` and radius) and is not mitigated.
- **Head-on swaps in a single lane** stay unresolved: no passing place exists.
- **Resolver goals** shown to learned policies were never seen in training (the D main effect, Stage 3c).
- **Results must be reported as "policy + supervisor"**, next to the unsupervised results.

### 14.6 Ready for Phase 3 experimental evaluation?

**Yes, at software level.** The implementation is complete, off by default and byte-identical when off, safe by construction and by test, deterministic, and cheap.

**Before Stage 3b, the following need approval:**
- **Stage 3a calibration:** a replay of the stored block-B greedy episodes; parameters frozen afterwards.
- **Escalation:** whether it is included as a factorial arm, or enabled by default.
- **The pre-registered go / no-go rule** (D10).

No claim about effectiveness is made until those experiments have been run.


## 15. Stage 3a calibration (2026-11-21): frozen

The parameters of §5 were calibrated on stored Block B episodes only and frozen. The full record (data, method, evidence, alternatives, freeze hashes, recommended go / no-go criteria, risks) is `docs/research/2026-11-21-issue-6a-stage3a-calibration-freeze.md`.

| Parameter | Design (§5) | Frozen, A\* | Frozen, RRT\* | Basis |
|---|---|---|---|---|
| W | 12 | **24** | **32** | §5 rule, per planner, PPO reference (MADQL ineligible: success not typical) [M] |
| K_osc, osc_cells, m_osc | 8, 2, 4 | **14, 3, 7** | **8, 3, 4** | ≤ 1 % trigger rate, maximum detection, K < W [M] |
| H, Δ_prog, H_stay | 8, 2, 3 | 8, 2, 3 | 8, 2, 3 | not calibratable from OFF data; descriptive evidence does not contradict [C/H] |
| L_max | 3 | 3 | 3 | normal legs show no would-be contest at q99 [M]; bound by construction [C] |
| Resolver | defaults | defaults | defaults | not calibratable (off in the stored data) |
| escalate | False | False | False | disabled by instruction |

**Changes to §14:**
- The class defaults of `SupervisorParams` remain the design values, and the scripted software tests use them. Evaluation uses `FROZEN_PARAMS` by planner.
- With O enabled, the worst-case per-leg override share of repeated bursts is H / (H + min(W, K)): 36 % for A\* and 50 % for RRT\*. §14.3 stated it for S only.
- `stats()` gained an additive `monitor` block: per-robot modifications, eligible steps, recoveries and R losses; the largest per-robot override share; recoveries per leg; and the longest time without progress.


## 16. Stages 3b, 3c and 3e (2026-11-21/22): executed

Results, statistics, provenance and recommendations: `docs/research/2026-11-22-issue-6a-phase3-evaluation.md`.

In brief [M]:
- The full supervisor raises completion for every policy and planner with no safety event:

  | Family | A\* | RRT\* |
  |---|---|---|
  | MADQL | 0.624 → 0.997 | 0.695 → 0.996 |
  | PPO | 0.987–0.988 → 0.998–0.999 | → 1.000 |

- In the factorial, S+O carries almost all of the effect. R and D add small, significant gains, and R reduces S/O's interventions. O adds nothing measurable over S except a faster MADQL delivery, at a higher override share.
- Under the recommended retention rule, R, S and D are kept and O is dropped.

**Changes to the design as implemented:**
- `evaluation.run_episode` gained an optional read-only `observer` hook (additive).
- §8's K^ctrl conflict metric is not stored in evaluation records; conflicts are reported at the arbiter level, with that caveat.

**Status (2026-11-22): Phase 3 complete.**
- The student approved the Stage 3a criteria for the development-block results.
- The configuration going forward is **R + S + D**, with O dropped. O remains implemented and in the factorial record.
- Phase 6 confirms on a new held-out block.
