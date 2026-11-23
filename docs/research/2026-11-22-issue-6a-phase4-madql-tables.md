# Issue 6A Phase 4: tables

Generated tables, copied verbatim from `results/dev6a/phase4/select/selection_tables.md` (Part A) and `results/dev6a/phase4/eval/tables.md` (Part B). Part C is formatted from `results/dev6a/phase4/eval/posthoc_conflicts.json`. The narrative is in `2026-11-22-issue-6a-phase4-madql.md`.

---

# Part A: training and development selection (Block S)

Completeness: `{"expected": 5760, "present": 5760, "duplicates": 0, "unexpected": 0, "errors": 0, "complete": true}`

## Selection (Block S, rule §3.5)

| Candidate | F completion | N completion | F flow time | F within 0.005 | N improves | Eligible |
|---|---|---|---|---|---|---|
| historical | 0.9976 | 0.6348 | — | — | — | — |
| n3 | 0.9974 | 0.8610 | 70.00 | True | True | True |
| t8 | 0.9974 | 0.7060 | 78.98 | True | True | True |
| n3t8 | 0.9958 | 0.8424 | 71.24 | True | True | True |

**Selected: n3** (eligible: True).

## Block S, condition FA (descriptive)

| Metric | hist | n3 | t8 | n3t8 |
|---|---|---|---|---|
| Completion | 0.9976 | 0.9974 | 0.9974 | 0.9958 |
| Undelivered / ep. | 0.047 | 0.050 | 0.050 | 0.082 |
| Successful ep. | 691 | 688 | 689 | 664 |
| Flow time (delivered) | 86.74 | 70.00 | 78.98 | 71.24 |
| Steps / ep. | 234.1 | 200.2 | 216.9 | 213.0 |
| Wait ratio | 0.1286 | 0.0493 | 0.0831 | 0.0622 |
| Conflicts / ep. | 2.09 | 1.82 | 11.97 | 2.30 |
| Planner calls / 100 robot-steps | 21.31 | 19.34 | 22.52 | 19.30 |
| Supervisor changes / 100 robot-steps | 1.047 | 0.655 | 0.950 | 0.735 |
| Stuck robots / ep. | 0.047 | 0.050 | 0.050 | 0.082 |
| Share STAY | 0.1340 | 0.0514 | 0.0830 | 0.0630 |
| Share STAY, free suggested cell | 0.1319 | 0.0496 | 0.0812 | 0.0600 |
| Share self-loop STAY | 0.0956 | 0.0341 | 0.0595 | 0.0463 |
| Share in STAY spells ≥ 20 | 0.0849 | 0.0288 | 0.0524 | 0.0410 |
| Share following the suggestion | 0.5840 | 0.6983 | 0.6140 | 0.6708 |
| Mean gap Q(STAY) − best move | -0.0543 | -0.0792 | -0.0853 | -0.0686 |
| Mean Q of the chosen action | 2.860 | 2.377 | 2.080 | 2.284 |
| Inference ms / step | 0.621 | 0.621 | 0.621 | 0.618 |
| Physical collisions | 0 | 0 | 0 | 0 |
| Invalid moves | 0 | 0 | 0 | 0 |
| Invalid reference paths | 0 | 0 | 0 | 0 |

## Block S, condition NA (descriptive)

| Metric | hist | n3 | t8 | n3t8 |
|---|---|---|---|---|
| Completion | 0.6348 | 0.8610 | 0.7060 | 0.8424 |
| Undelivered / ep. | 7.054 | 2.685 | 5.679 | 3.044 |
| Successful ep. | 3 | 116 | 13 | 81 |
| Flow time (delivered) | 95.04 | 72.85 | 83.76 | 72.52 |
| Steps / ep. | 399.8 | 372.9 | 397.5 | 382.3 |
| Wait ratio | 0.2190 | 0.0948 | 0.1690 | 0.1253 |
| Conflicts / ep. | 4.58 | 4.72 | 52.40 | 5.35 |
| Planner calls / 100 robot-steps | 34.08 | 23.90 | 32.51 | 23.84 |
| Supervisor changes / 100 robot-steps | 0.000 | 0.000 | 0.000 | 0.000 |
| Stuck robots / ep. | 5.826 | 2.590 | 4.918 | 2.908 |
| Share STAY | 0.2227 | 0.1020 | 0.1745 | 0.1292 |
| Share STAY, free suggested cell | 0.2213 | 0.1011 | 0.1732 | 0.1281 |
| Share self-loop STAY | 0.1805 | 0.0808 | 0.1397 | 0.1097 |
| Share in STAY spells ≥ 20 | 0.1983 | 0.0886 | 0.1583 | 0.1157 |
| Share following the suggestion | 0.4027 | 0.5416 | 0.4233 | 0.5161 |
| Mean gap Q(STAY) − best move | -0.0457 | -0.0746 | -0.0699 | -0.0616 |
| Mean Q of the chosen action | 3.020 | 2.361 | 2.212 | 2.264 |
| Inference ms / step | 0.624 | 0.617 | 0.621 | 0.617 |
| Physical collisions | 0 | 0 | 0 | 0 |
| Invalid moves | 0 | 0 | 0 | 0 |
| Invalid reference paths | 0 | 0 | 0 | 0 |

## Block S completion per policy

| Arm | Cond | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| hist | FA | 0.9987 | 0.9974 | 0.9965 |
| hist | NA | 0.6864 | 0.5953 | 0.6227 |
| n3 | FA | 0.9959 | 0.9981 | 0.9983 |
| n3 | NA | 0.8581 | 0.8408 | 0.8842 |
| t8 | FA | 0.9968 | 0.9981 | 0.9974 |
| t8 | NA | 0.6719 | 0.6948 | 0.7513 |
| n3t8 | FA | 0.9965 | 0.9942 | 0.9965 |
| n3t8 | NA | 0.8691 | 0.8164 | 0.8417 |

## Training (block A validation; training logs)

| Run | best step | best completion | mean completion, last 5 | max swing after 500 k | late loss | wall h |
|---|---|---|---|---|---|---|
| hist_seed1 | 800000 | 0.7151 | 0.5502 | 0.2449 | 0.0068 | 2.11 |
| hist_seed2 | 1000000 | 0.6352 | 0.4781 | 0.2942 | 0.0066 | 2.10 |
| hist_seed3 | 700000 | 0.6310 | 0.5711 | 0.2738 | 0.0069 | 1.99 |
| n3_seed1 | 600000 | 0.8563 | 0.7811 | 0.1973 | 0.0241 | 3.78 |
| n3_seed2 | 850000 | 0.8444 | 0.7794 | 0.2602 | 0.0245 | 3.57 |
| n3_seed3 | 950000 | 0.8895 | 0.8304 | 0.1879 | 0.0238 | 3.73 |
| t8_seed1 | 850000 | 0.7228 | 0.6333 | 0.3435 | 0.0066 | 3.81 |
| t8_seed2 | 50000 | 0.6718 | 0.6051 | 0.2526 | 0.0068 | 3.80 |
| t8_seed3 | 700000 | 0.7449 | 0.6730 | 0.3818 | 0.0068 | 3.83 |
| n3t8_seed1 | 700000 | 0.8707 | 0.7342 | 0.2262 | 0.0231 | 3.79 |
| n3t8_seed2 | 950000 | 0.8325 | 0.8100 | 0.1361 | 0.0231 | 3.65 |
| n3t8_seed3 | 900000 | 0.8639 | 0.8238 | 0.1267 | 0.0232 | 3.63 |

## Checkpoint churn, 600 k – 1 M (diagnostics sample)

| Run | greedy action changed (mean) | STAY/move flipped (mean) | median \|Δ gap\| (mean) | STAY preferred at 1 M |
|---|---|---|---|---|
| hist_seed1 | 0.4047 | 0.1175 | 0.0336 | 0.1405 |
| hist_seed2 | 0.4016 | 0.1373 | 0.0360 | 0.1606 |
| hist_seed3 | 0.3906 | 0.1255 | 0.0379 | 0.0819 |
| n3_seed1 | 0.3287 | 0.0568 | 0.0386 | 0.0250 |
| n3_seed2 | 0.3280 | 0.0716 | 0.0396 | 0.0243 |
| n3_seed3 | 0.3280 | 0.0476 | 0.0382 | 0.0280 |
| t8_seed1 | 0.3703 | 0.1111 | 0.0323 | 0.0805 |
| t8_seed2 | 0.3905 | 0.1109 | 0.0299 | 0.0950 |
| t8_seed3 | 0.3714 | 0.0792 | 0.0335 | 0.0685 |
| n3t8_seed1 | 0.3326 | 0.0515 | 0.0336 | 0.0167 |
| n3t8_seed2 | 0.3219 | 0.0711 | 0.0385 | 0.0141 |
| n3t8_seed3 | 0.3230 | 0.0597 | 0.0324 | 0.0583 |

---

# Part B: Block C evaluation (pre-registered, §5)

Selected candidate: **n3**. Completeness: `{"expected": 5040, "present": 5040, "missing": 0, "unexpected": 0, "errors": 0, "complete": true}`.

G0.1 FA (`hist` = Phase 1 records): 1080 / 1080 identical.
G0.1 NA (`hist` = Phase 1 records): 1080 / 1080 identical.
G0.1 FR (`hist` = Phase 1 records): 360 / 360 identical.
G0.2 replay: 101 replayed, identical: True.

Recommendation inputs: `{"FA": "adopt", "NA": "adopt", "FR": "adopt", "g0_1": true, "replay_identical": true, "madql_v2_forward_candidate": true}`

## FA: A\*, R + S + D (primary)

### G0

```
{
 "episodes": 2160,
 "errors": [],
 "physical_collisions": 0,
 "inv_total": 0,
 "invalid_reference_paths": 0,
 "reference_paths_checked": 759343,
 "instances_identical_across_arms": true,
 "repair_off": true,
 "p2_off": true,
 "full_rrt_budget": true,
 "v2_candidate_ok": true,
 "supervised_episodes": 2160,
 "supervisor_mode_ok": true,
 "supervisor_introduced_invalid_moves": 0,
 "fairness_bound_ok": true,
 "burst_bound_violations": [],
 "params_frozen": true,
 "o_unused": true,
 "pass": true
}
```

### P1

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | completion | 0.9970 | 0.0008 | [-0.0004, 0.0019] | 0.17698 | 0.17698 | 0.03 | not rejected | no |

### P2

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql_seed1 | completion | 0.9980 | -0.0014 | [-0.0028, -0.0001] | 0.03520 | 0.10559 | 0.005 | not rejected | no |
| madql_seed2 | completion | 0.9958 | 0.0025 | [0.0001, 0.0048] | 0.04070 | 0.10559 | 0.005 | not rejected | no |
| madql_seed3 | completion | 0.9971 | 0.0013 | [-0.0005, 0.0031] | 0.15568 | 0.15568 | 0.005 | not rejected | no |

### E

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | planner_calls_per_100_robot_steps | 21.2872 | -1.8162 | [-2.1812, -1.4512] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |
| madql | common_flow_time | 85.8912 | -16.5894 | [-17.5564, -15.6223] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |
| madql | supervisor_changes_per_100_robot_steps | 1.0408 | -0.4213 | [-0.4541, -0.3885] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |

### P4

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | conflicts_per_episode | 1.2657 | 0.6444 | [0.1309, 1.1580] | 0.01390 | 0.02780 | 0.005 | not rejected | no |
| madql | top_priority_common_flow_time | 57.7250 | -7.6496 | [-8.6123, -6.6870] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |
| madql | top_priority_undelivered | 0.0046 | -0.0028 | [-0.0092, 0.0036] | 0.33087 | 0.33087 | 0.005 | not rejected | no |

### Decision

```
{
 "g0_pass": true,
 "completion_harm": [],
 "completion_gain": [],
 "p4_vetoes": [],
 "e_improved": [
  "planner_calls_per_100_robot_steps",
  "common_flow_time",
  "supervisor_changes_per_100_robot_steps"
 ],
 "e_worsened": [],
 "completion_lower_bound": {
  "value": -0.00036923041794453804,
  "source": "bootstrap-t"
 },
 "delta1": 0.052068267283772056,
 "verdict": "adopt"
}
```

### Descriptive (all policies, then per policy)

| Metric | hist all | v2 all | hist seed1 | v2 seed1 | hist seed2 | v2 seed2 | hist seed3 | v2 seed3 |
|---|---|---|---|---|---|---|---|---|
| Completion | 0.9970 | 0.9977 | 0.9980 | 0.9965 | 0.9958 | 0.9983 | 0.9971 | 0.9984 |
| Delivered | 20679 | 20695 | 6900 | 6890 | 6885 | 6902 | 6894 | 6903 |
| Undelivered / ep. | 0.058 | 0.044 | 0.039 | 0.067 | 0.081 | 0.033 | 0.056 | 0.031 |
| Successful ep. | 1026 | 1039 | 346 | 337 | 337 | 351 | 343 | 351 |
| High-priority completion | 0.9992 | 0.9997 | 1.0000 | 1.0000 | 0.9986 | 0.9991 | 0.9991 | 1.0000 |
| Flow time (delivered) | 85.95 | 69.42 | 82.09 | 70.56 | 90.02 | 71.62 | 85.75 | 66.07 |
| Makespan (mean) | 223.3 | 189.2 | 212.0 | 193.2 | 236.6 | 195.6 | 221.5 | 178.8 |
| Steps / ep. | 232.2 | 197.7 | 219.5 | 207.0 | 247.1 | 201.1 | 230.0 | 184.9 |
| Conflicts / ep. | 1.27 | 1.91 | 2.38 | 2.71 | 0.85 | 1.01 | 0.56 | 2.02 |
| Conflicts / delivered | 0.066 | 0.100 | 0.124 | 0.141 | 0.045 | 0.053 | 0.029 | 0.105 |
| Conflicts / 100 robot-steps | 0.068 | 0.121 | 0.136 | 0.163 | 0.043 | 0.063 | 0.031 | 0.136 |
| Wait ratio | 0.1252 | 0.0468 | 0.0812 | 0.0605 | 0.1900 | 0.0573 | 0.1045 | 0.0226 |
| Planner calls / 100 robot-steps | 21.29 | 19.47 | 23.00 | 17.78 | 19.97 | 19.59 | 21.06 | 21.24 |
| Deviations / 100 robot-steps | 17.78 | 15.19 | 19.11 | 13.79 | 16.76 | 15.53 | 17.61 | 16.39 |
| Supervisor changes / 100 robot-steps | 1.041 | 0.620 | 0.959 | 0.670 | 1.127 | 0.643 | 1.026 | 0.538 |
| Stuck robots | 63 | 47 | 14 | 24 | 29 | 12 | 20 | 11 |
| Share STAY | 0.1309 | 0.0491 | 0.0822 | 0.0621 | 0.1981 | 0.0604 | 0.1065 | 0.0228 |
| Share STAY, free suggested cell | 0.1285 | 0.0476 | 0.0814 | 0.0597 | 0.1945 | 0.0596 | 0.1040 | 0.0215 |
| Share STAY, occupied suggested cell | 0.0022 | 0.0014 | 0.0008 | 0.0023 | 0.0036 | 0.0007 | 0.0022 | 0.0012 |
| Share self-loop STAY | 0.0929 | 0.0328 | 0.0596 | 0.0434 | 0.1400 | 0.0387 | 0.0751 | 0.0150 |
| Share in STAY spells ≥ 20 | 0.0817 | 0.0279 | 0.0502 | 0.0362 | 0.1233 | 0.0333 | 0.0680 | 0.0132 |
| Share following the suggestion | 0.5860 | 0.6997 | 0.6159 | 0.7025 | 0.5317 | 0.6864 | 0.6149 | 0.7110 |
| Mean gap Q(STAY) − best move | -0.0555 | -0.0804 | -0.0610 | -0.0721 | -0.0409 | -0.0797 | -0.0657 | -0.0903 |
| Mean Q of the chosen action | 2.887 | 2.411 | 2.897 | 2.282 | 2.814 | 2.450 | 2.956 | 2.506 |
| Inference ms / step | 0.637 | 0.639 | 0.638 | 0.634 | 0.633 | 0.641 | 0.639 | 0.641 |
| Planning s / ep. (median) | 0.08 | 0.07 | 0.08 | 0.07 | 0.08 | 0.07 | 0.08 | 0.06 |
| Wall s / ep. (median) | 0.95 | 0.78 | 0.90 | 0.80 | 1.00 | 0.80 | 0.92 | 0.73 |
| Physical collisions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid moves (INV) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid reference paths | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Reference paths checked | 426960 | 332383 | 145381 | 105925 | 142093 | 113394 | 139486 | 113064 |
| Errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Conflicts by type (total)

| Category | hist | v2 |
|---|---|---|
| INV | 0 | 0 |
| ROs | 0 | 0 |
| ROv | 652 | 1473 |
| RRs | 7 | 23 |
| RRv | 708 | 567 |

### Planner calls by trigger (share)

| Category | hist | v2 |
|---|---|---|
| T1 | 0.1059 | 0.1358 |
| T2 | 0.0445 | 0.0550 |
| T3 | 0.8428 | 0.8013 |
| T4 | 0.0006 | 0.0007 |
| T5 | 0.0001 | 0.0001 |
| T6 | 0.0010 | 0.0011 |
| T7 | 0.0026 | 0.0033 |
| T8 | 0.0026 | 0.0027 |

### Stuck robots by class (total)

| Category | hist | v2 |
|---|---|---|
| blocked_by_idle | 4 | 2 |
| conflict_loop | — | 1 |
| deadlock_cycle | 2 | 4 |
| frozen | 7 | 5 |
| mostly_stationary | 1 | 2 |
| oscillating | 6 | 15 |
| other_moving | 43 | 18 |

### Action shares (STAY, up, down, left, right — non-idle robot-steps)

| Arm | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| hist | 0.1309 | 0.2107 | 0.2150 | 0.2213 | 0.2222 |
| v2 | 0.0491 | 0.2374 | 0.2439 | 0.2346 | 0.2350 |

### Supportive (nominal)

| Outcome | Δ̂ | nominal 95 % CI | hist θ | v2 θ |
|---|---|---|---|---|
| undelivered_per_episode | -0.0148 | [-0.0367, 0.0071] | 0.0583 | 0.0435 |
| steps_per_episode | -34.5315 | [-39.3287, -29.7342] | 232.2222 | 197.6907 |
| wait_ratio | -0.0785 | [-0.0831, -0.0739] | 0.1252 | 0.0468 |
| conflicts_per_100_robot_steps | 0.0526 | [0.0213, 0.0840] | 0.0681 | 0.1208 |
| share_stay | -0.0818 | [-0.0866, -0.0769] | 0.1309 | 0.0491 |
| share_self_loop_stay | -0.0601 | [-0.0643, -0.0560] | 0.0929 | 0.0328 |
| share_in_stay_spells_ge20 | -0.0538 | [-0.0584, -0.0493] | 0.0817 | 0.0279 |
| share_follow_suggestion | 0.1136 | [0.1087, 0.1185] | 0.5860 | 0.6997 |
| mean_gap | -0.0249 | [-0.0261, -0.0237] | -0.0555 | -0.0804 |
| inference_ms_per_step | 0.0018 | [-0.0011, 0.0047] | 0.6367 | 0.6385 |
| planning_s_per_episode | -0.0164 | [-0.0186, -0.0142] | 0.0909 | 0.0745 |
| wall_s_per_episode | -0.1625 | [-0.1817, -0.1434] | 1.1482 | 0.9856 |

### Per level, completion (nominal)

| Level | hist | v2 | Δ̂ | CI |
|---|---|---|---|---|
| static | 0.9986 | 0.9979 | -0.0007 | [-0.0025, 0.0011] |
| obstacles | 0.9972 | 0.9986 | 0.0014 | [-0.0004, 0.0032] |
| full | 0.9958 | 0.9971 | 0.0013 | [-0.0005, 0.0031] |

### Seeds (completion: better / worse / equal)

| Policy pair | better | worse | equal |
|---|---|---|---|
| madql_seed1 | 3 | 11 | 106 |
| madql_seed2 | 14 | 4 | 102 |
| madql_seed3 | 12 | 6 | 102 |

## FR: RRT\*, R + S + D

### G0

```
{
 "episodes": 720,
 "errors": [],
 "physical_collisions": 0,
 "inv_total": 0,
 "invalid_reference_paths": 0,
 "reference_paths_checked": 253633,
 "instances_identical_across_arms": true,
 "repair_off": true,
 "p2_off": true,
 "full_rrt_budget": true,
 "v2_candidate_ok": true,
 "supervised_episodes": 720,
 "supervisor_mode_ok": true,
 "supervisor_introduced_invalid_moves": 0,
 "fairness_bound_ok": true,
 "burst_bound_violations": [],
 "params_frozen": true,
 "o_unused": true,
 "pass": true
}
```

### P1

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | completion | 0.9914 | 0.0055 | [0.0025, 0.0084] | 0.00240 | 0.00240 | 0.03 | **rejected** | no |

### P2

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql_seed1 | completion | 0.9953 | 0.0026 | [-0.0012, 0.0064] | 0.11489 | 0.22978 | 0.005 | not rejected | no |
| madql_seed2 | completion | 0.9824 | 0.0133 | [0.0069, 0.0198] | 0.00110 | 0.00330 | 0.005 | **rejected** | yes |
| madql_seed3 | completion | 0.9966 | 0.0004 | [-0.0031, 0.0039] | 0.82392 | 0.82392 | 0.005 | not rejected | no |

### E

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | planner_calls_per_100_robot_steps | 18.9765 | -1.0815 | [-1.5797, -0.5833] | 0.00060 | 0.00060 | 0.01 | **rejected** | no |
| madql | common_flow_time | 100.6452 | -21.8813 | [-23.3352, -20.4274] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |
| madql | supervisor_changes_per_100_robot_steps | 0.7030 | -0.3587 | [-0.4000, -0.3174] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |

### P4

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | conflicts_per_episode | 1.8083 | -0.2583 | [-1.8627, 1.3460] | 0.62884 | 0.62884 | 0.005 | not rejected | no |
| madql | top_priority_common_flow_time | 64.6977 | -10.5191 | [-12.3178, -8.7203] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |
| madql | top_priority_undelivered | 0.0000 | 0.0056 | [-inf, inf] | 0.14519 | 0.29037 | 0.005 | not rejected | no |

### Decision

```
{
 "g0_pass": true,
 "completion_harm": [],
 "completion_gain": [
  "madql"
 ],
 "p4_vetoes": [],
 "e_improved": [
  "planner_calls_per_100_robot_steps",
  "common_flow_time",
  "supervisor_changes_per_100_robot_steps"
 ],
 "e_worsened": [],
 "completion_lower_bound": {
  "value": 0.002459479706418232,
  "source": "bootstrap-t"
 },
 "delta1": 0.05165733964700818,
 "verdict": "adopt"
}
```

### Descriptive (all policies, then per policy)

| Metric | hist all | v2 all | hist seed1 | v2 seed1 | hist seed2 | v2 seed2 | hist seed3 | v2 seed3 |
|---|---|---|---|---|---|---|---|---|
| Completion | 0.9914 | 0.9968 | 0.9953 | 0.9978 | 0.9824 | 0.9957 | 0.9966 | 0.9970 |
| Delivered | 6909 | 6947 | 2312 | 2318 | 2282 | 2313 | 2315 | 2316 |
| Undelivered / ep. | 0.167 | 0.061 | 0.092 | 0.042 | 0.342 | 0.083 | 0.067 | 0.058 |
| Successful ep. | 315 | 340 | 110 | 115 | 92 | 112 | 113 | 113 |
| High-priority completion | 1.0000 | 0.9991 | 1.0000 | 1.0000 | 1.0000 | 0.9972 | 1.0000 | 1.0000 |
| Flow time (delivered) | 100.68 | 79.33 | 95.75 | 78.62 | 108.18 | 82.67 | 98.21 | 76.70 |
| Makespan (mean) | 251.3 | 208.5 | 240.6 | 206.6 | 276.2 | 223.1 | 241.4 | 195.9 |
| Steps / ep. | 269.9 | 219.4 | 253.9 | 214.8 | 305.1 | 234.9 | 250.7 | 208.4 |
| Conflicts / ep. | 1.81 | 1.55 | 3.77 | 2.25 | 0.66 | 1.38 | 0.99 | 1.02 |
| Conflicts / delivered | 0.094 | 0.080 | 0.196 | 0.116 | 0.035 | 0.071 | 0.051 | 0.053 |
| Conflicts / 100 robot-steps | 0.084 | 0.088 | 0.186 | 0.131 | 0.027 | 0.073 | 0.049 | 0.061 |
| Wait ratio | 0.1546 | 0.0595 | 0.0846 | 0.0595 | 0.2516 | 0.0918 | 0.1276 | 0.0273 |
| Planner calls / 100 robot-steps | 18.98 | 17.89 | 21.20 | 16.19 | 16.60 | 17.81 | 19.62 | 19.74 |
| Deviations / 100 robot-steps | 15.40 | 13.43 | 17.28 | 11.82 | 13.46 | 13.75 | 15.87 | 14.73 |
| Supervisor changes / 100 robot-steps | 0.703 | 0.344 | 0.617 | 0.302 | 0.879 | 0.407 | 0.576 | 0.317 |
| Stuck robots | 60 | 22 | 11 | 5 | 41 | 10 | 8 | 7 |
| Share STAY | 0.1642 | 0.0627 | 0.0876 | 0.0608 | 0.2590 | 0.0970 | 0.1299 | 0.0276 |
| Share STAY, free suggested cell | 0.1607 | 0.0572 | 0.0856 | 0.0525 | 0.2531 | 0.0905 | 0.1276 | 0.0259 |
| Share STAY, occupied suggested cell | 0.0014 | 0.0006 | 0.0005 | 0.0009 | 0.0024 | 0.0005 | 0.0010 | 0.0004 |
| Share self-loop STAY | 0.1236 | 0.0451 | 0.0665 | 0.0436 | 0.1961 | 0.0688 | 0.0960 | 0.0208 |
| Share in STAY spells ≥ 20 | 0.1121 | 0.0391 | 0.0583 | 0.0367 | 0.1782 | 0.0611 | 0.0886 | 0.0179 |
| Share following the suggestion | 0.5737 | 0.7061 | 0.6189 | 0.7385 | 0.5082 | 0.6647 | 0.6048 | 0.7179 |
| Mean gap Q(STAY) − best move | -0.0520 | -0.0786 | -0.0607 | -0.0732 | -0.0360 | -0.0736 | -0.0619 | -0.0893 |
| Mean Q of the chosen action | 2.844 | 2.367 | 2.870 | 2.246 | 2.748 | 2.402 | 2.930 | 2.451 |
| Inference ms / step | 0.612 | 0.609 | 0.618 | 0.603 | 0.602 | 0.606 | 0.619 | 0.619 |
| Planning s / ep. (median) | 42.33 | 31.55 | 44.03 | 28.60 | 41.85 | 32.52 | 41.26 | 33.71 |
| Wall s / ep. (median) | 43.11 | 32.51 | 44.86 | 29.31 | 42.66 | 33.44 | 42.16 | 34.60 |
| Physical collisions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid moves (INV) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid reference paths | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Reference paths checked | 143710 | 109923 | 50403 | 32360 | 47355 | 39094 | 45952 | 38469 |
| Errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Conflicts by type (total)

| Category | hist | v2 |
|---|---|---|
| INV | 0 | 0 |
| ROs | 0 | 0 |
| ROv | 264 | 490 |
| RRs | 2 | 20 |
| RRv | 385 | 48 |

### Planner calls by trigger (share)

| Category | hist | v2 |
|---|---|---|
| T1 | 0.1039 | 0.1360 |
| T2 | 0.0492 | 0.0605 |
| T3 | 0.8167 | 0.7706 |
| T4 | 0.0003 | 0.0001 |
| T5 | 0.0251 | 0.0271 |
| T6 | 0.0003 | 0.0010 |
| T7 | 0.0013 | 0.0018 |
| T8 | 0.0032 | 0.0029 |

### Stuck robots by class (total)

| Category | hist | v2 |
|---|---|---|
| blocked_by_idle | 1 | — |
| frozen | 3 | 1 |
| mostly_stationary | 1 | 1 |
| oscillating | 2 | 2 |
| other_moving | 53 | 18 |

### Action shares (STAY, up, down, left, right — non-idle robot-steps)

| Arm | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| hist | 0.1642 | 0.1812 | 0.1840 | 0.2343 | 0.2363 |
| v2 | 0.0627 | 0.2162 | 0.2201 | 0.2502 | 0.2508 |

### Supportive (nominal)

| Outcome | Δ̂ | nominal 95 % CI | hist θ | v2 θ |
|---|---|---|---|---|
| undelivered_per_episode | -0.1056 | [-0.1647, -0.0464] | 0.1667 | 0.0611 |
| steps_per_episode | -50.5222 | [-58.1332, -42.9112] | 269.8750 | 219.3528 |
| wait_ratio | -0.0951 | [-0.1050, -0.0852] | 0.1546 | 0.0595 |
| conflicts_per_100_robot_steps | 0.0046 | [-0.0700, 0.0792] | 0.0838 | 0.0883 |
| share_stay | -0.1014 | [-0.1123, -0.0905] | 0.1642 | 0.0627 |
| share_self_loop_stay | -0.0786 | [-0.0883, -0.0688] | 0.1236 | 0.0451 |
| share_in_stay_spells_ge20 | -0.0730 | [-0.0819, -0.0640] | 0.1121 | 0.0391 |
| share_follow_suggestion | 0.1324 | [0.1267, 0.1381] | 0.5737 | 0.7061 |
| mean_gap | -0.0265 | [-0.0287, -0.0244] | -0.0520 | -0.0786 |
| inference_ms_per_step | -0.0028 | [-0.0077, 0.0021] | 0.6121 | 0.6093 |
| planning_s_per_episode | -11.0492 | [-12.2079, -9.8906] | 46.6541 | 35.6048 |
| wall_s_per_episode | -11.2538 | [-12.4351, -10.0726] | 47.8112 | 36.5574 |

### Per level, completion (nominal)

| Level | hist | v2 | Δ̂ | CI |
|---|---|---|---|---|
| static | 0.9964 | 0.9974 | 0.0010 | [-0.0020, 0.0041] |
| obstacles | 0.9953 | 0.9979 | 0.0026 | [-0.0003, 0.0055] |
| full | 0.9859 | 0.9958 | 0.0099 | [0.0053, 0.0145] |

### Seeds (completion: better / worse / equal)

| Policy pair | better | worse | equal |
|---|---|---|---|
| madql_seed1 | 5 | 1 | 34 |
| madql_seed2 | 20 | 2 | 18 |
| madql_seed3 | 3 | 3 | 34 |

## NA: A\*, no supervisor (policy's own decisions)

### G0

```
{
 "episodes": 2160,
 "errors": [],
 "physical_collisions": 0,
 "inv_total": 0,
 "invalid_reference_paths": 0,
 "reference_paths_checked": 1926252,
 "instances_identical_across_arms": true,
 "repair_off": true,
 "p2_off": true,
 "full_rrt_budget": true,
 "v2_candidate_ok": true,
 "no_supervisor": true,
 "resolver_off": true,
 "pass": true
}
```

### P1

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | completion | 0.6240 | 0.2355 | [0.2202, 0.2507] | 0.00010 | 0.00010 | 0.03 | **rejected** | no |

### P2

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql_seed1 | completion | 0.6733 | 0.1796 | [0.1580, 0.2012] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |
| madql_seed2 | completion | 0.5957 | 0.2430 | [0.2183, 0.2677] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |
| madql_seed3 | completion | 0.6031 | 0.2838 | [0.2616, 0.3059] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |

### E

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | planner_calls_per_100_robot_steps | 33.5365 | -9.3201 | [-10.1392, -8.5011] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |
| madql | common_flow_time | 94.2962 | -28.5343 | [-30.9458, -26.1228] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |
| madql | stuck_robots_per_episode | 5.7583 | -3.2093 | [-3.3621, -3.0564] | 0.00010 | 0.00030 | 0.01 | **rejected** | no |

### P4

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | conflicts_per_episode | 2.9667 | 1.0926 | [-0.7027, 2.8879] | 0.22728 | 0.22728 | 0.005 | not rejected | no |
| madql | top_priority_common_flow_time | 67.7062 | -14.6885 | [-16.9687, -12.4084] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |
| madql | top_priority_undelivered | 1.3676 | -0.8676 | [-0.9764, -0.7588] | 0.00010 | 0.00030 | 0.005 | **rejected** | no |

### Decision

```
{
 "g0_pass": true,
 "completion_harm": [],
 "completion_gain": [
  "madql"
 ],
 "p4_vetoes": [],
 "e_improved": [
  "planner_calls_per_100_robot_steps",
  "common_flow_time",
  "stuck_robots_per_episode"
 ],
 "e_worsened": [],
 "completion_lower_bound": {
  "value": 0.22021245263885872,
  "source": "bootstrap-t"
 },
 "delta1": 0.052068267283772056,
 "verdict": "adopt"
}
```

### Descriptive (all policies, then per policy)

| Metric | hist all | v2 all | hist seed1 | v2 seed1 | hist seed2 | v2 seed2 | hist seed3 | v2 seed3 |
|---|---|---|---|---|---|---|---|---|
| Completion | 0.6240 | 0.8595 | 0.6733 | 0.8529 | 0.5957 | 0.8387 | 0.6031 | 0.8869 |
| Delivered | 12944 | 17828 | 4655 | 5897 | 4119 | 5799 | 4170 | 6132 |
| Undelivered / ep. | 7.220 | 2.698 | 6.275 | 2.825 | 7.764 | 3.097 | 7.622 | 2.172 |
| Successful ep. | 15 | 207 | 11 | 55 | 0 | 56 | 4 | 96 |
| High-priority completion | 0.7784 | 0.9190 | 0.8083 | 0.9109 | 0.7754 | 0.8947 | 0.7516 | 0.9514 |
| Flow time (delivered) | 94.87 | 72.54 | 94.88 | 73.91 | 94.52 | 74.61 | 95.21 | 69.27 |
| Makespan (mean) | 288.6 | 236.9 | 288.2 | 249.9 | — | 238.7 | 289.8 | 228.5 |
| Steps / ep. | 398.5 | 368.9 | 396.6 | 377.3 | 400.0 | 375.0 | 398.8 | 354.4 |
| Conflicts / ep. | 2.97 | 4.06 | 3.97 | 5.11 | 2.36 | 3.34 | 2.57 | 3.73 |
| Conflicts / delivered | 0.248 | 0.246 | 0.307 | 0.312 | 0.206 | 0.207 | 0.222 | 0.219 |
| Conflicts / 100 robot-steps | 0.093 | 0.138 | 0.125 | 0.169 | 0.074 | 0.111 | 0.081 | 0.131 |
| Wait ratio | 0.2261 | 0.0903 | 0.1481 | 0.1138 | 0.3178 | 0.1140 | 0.2124 | 0.0432 |
| Planner calls / 100 robot-steps | 33.54 | 24.22 | 35.46 | 23.30 | 30.78 | 24.47 | 34.39 | 24.92 |
| Deviations / 100 robot-steps | 31.69 | 21.59 | 33.50 | 20.86 | 29.03 | 22.17 | 32.57 | 21.74 |
| Supervisor changes / 100 robot-steps | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Stuck robots | 6219 | 2753 | 1872 | 959 | 2214 | 1036 | 2133 | 758 |
| Share STAY | 0.2313 | 0.1007 | 0.1532 | 0.1215 | 0.3209 | 0.1233 | 0.2141 | 0.0505 |
| Share STAY, free suggested cell | 0.2297 | 0.0995 | 0.1528 | 0.1198 | 0.3178 | 0.1228 | 0.2127 | 0.0489 |
| Share STAY, occupied suggested cell | 0.0017 | 0.0012 | 0.0004 | 0.0017 | 0.0031 | 0.0005 | 0.0013 | 0.0016 |
| Share self-loop STAY | 0.1869 | 0.0830 | 0.1264 | 0.1016 | 0.2636 | 0.1023 | 0.1659 | 0.0393 |
| Share in STAY spells ≥ 20 | 0.2068 | 0.0877 | 0.1358 | 0.1058 | 0.2861 | 0.1069 | 0.1932 | 0.0445 |
| Share following the suggestion | 0.3990 | 0.5406 | 0.4397 | 0.5325 | 0.3489 | 0.5264 | 0.4114 | 0.5664 |
| Mean gap Q(STAY) − best move | -0.0467 | -0.0748 | -0.0531 | -0.0667 | -0.0348 | -0.0750 | -0.0527 | -0.0836 |
| Mean Q of the chosen action | 3.059 | 2.363 | 3.043 | 2.264 | 2.983 | 2.447 | 3.151 | 2.376 |
| Inference ms / step | 0.634 | 0.628 | 0.638 | 0.631 | 0.631 | 0.627 | 0.633 | 0.625 |
| Planning s / ep. (median) | 0.20 | 0.14 | 0.22 | 0.14 | 0.19 | 0.15 | 0.21 | 0.13 |
| Wall s / ep. (median) | 1.83 | 1.61 | 1.84 | 1.65 | 1.80 | 1.64 | 1.83 | 1.55 |
| Physical collisions | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid moves (INV) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid reference paths | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Reference paths checked | 1154487 | 771765 | 405010 | 253238 | 354597 | 264201 | 394880 | 254326 |
| Errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Conflicts by type (total)

| Category | hist | v2 |
|---|---|---|
| INV | 0 | 0 |
| ROs | 0 | 0 |
| ROv | 1172 | 1724 |
| RRs | 11 | 33 |
| RRv | 2021 | 2627 |

### Planner calls by trigger (share)

| Category | hist | v2 |
|---|---|---|
| T1 | 0.0343 | 0.0575 |
| T2 | 0.0173 | 0.0257 |
| T3 | 0.9474 | 0.9147 |
| T4 | 0.0001 | 0.0002 |
| T5 | 0.0000 | 0.0000 |
| T6 | 0.0003 | 0.0003 |
| T8 | 0.0006 | 0.0016 |

### Stuck robots by class (total)

| Category | hist | v2 |
|---|---|---|
| blocked_by_idle | 5 | 18 |
| conflict_loop | 9 | 10 |
| deadlock_cycle | 60 | 72 |
| frozen | 1391 | 413 |
| mostly_stationary | 137 | 15 |
| oscillating | 3443 | 1999 |
| other_moving | 1174 | 226 |

### Action shares (STAY, up, down, left, right — non-idle robot-steps)

| Arm | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| hist | 0.2313 | 0.1846 | 0.1867 | 0.1987 | 0.1987 |
| v2 | 0.1007 | 0.2274 | 0.2301 | 0.2211 | 0.2207 |

### Supportive (nominal)

| Outcome | Δ̂ | nominal 95 % CI | hist θ | v2 θ |
|---|---|---|---|---|
| undelivered_per_episode | -4.5222 | [-4.8325, -4.2119] | 7.2204 | 2.6981 |
| steps_per_episode | -29.5583 | [-34.7704, -24.3463] | 398.4528 | 368.8944 |
| wait_ratio | -0.1357 | [-0.1454, -0.1261] | 0.2261 | 0.0903 |
| conflicts_per_100_robot_steps | 0.0445 | [-0.0149, 0.1039] | 0.0931 | 0.1375 |
| share_stay | -0.1306 | [-0.1411, -0.1201] | 0.2313 | 0.1007 |
| share_self_loop_stay | -0.1038 | [-0.1141, -0.0935] | 0.1869 | 0.0830 |
| share_in_stay_spells_ge20 | -0.1191 | [-0.1295, -0.1087] | 0.2068 | 0.0877 |
| share_follow_suggestion | 0.1416 | [0.1344, 0.1488] | 0.3990 | 0.5406 |
| mean_gap | -0.0280 | [-0.0309, -0.0252] | -0.0467 | -0.0748 |
| inference_ms_per_step | -0.0062 | [-0.0084, -0.0039] | 0.6341 | 0.6279 |
| planning_s_per_episode | -0.0528 | [-0.0582, -0.0473] | 0.2086 | 0.1559 |
| wall_s_per_episode | -0.2898 | [-0.3133, -0.2664] | 1.9057 | 1.6159 |

### Per level, completion (nominal)

| Level | hist | v2 | Δ̂ | CI |
|---|---|---|---|---|
| static | 0.3620 | 0.6759 | 0.3139 | [0.2884, 0.3394] |
| obstacles | 0.7622 | 0.9286 | 0.1665 | [0.1537, 0.1793] |
| full | 0.7015 | 0.9310 | 0.2296 | [0.2072, 0.2519] |

### Seeds (completion: better / worse / equal)

| Policy pair | better | worse | equal |
|---|---|---|---|
| madql_seed1 | 113 | 7 | 0 |
| madql_seed2 | 117 | 3 | 0 |
| madql_seed3 | 119 | 0 | 1 |

---

# Part C: post-hoc conflict analysis (not pre-registered; never used for a verdict)

Nominal intervals from the frozen estimator (ratio of sums, seed cluster, bootstrap-t, B = 10 000), with no multiplicity correction. Moves = non-idle robot-steps that are not STAY.

| Condition | Outcome | hist θ | Δ̂ | nominal 95 % CI |
|---|---|---|---|---|
| FA | ROv_per_episode | 0.604 | +0.760 | [+0.318, +1.202] |
| FA | ROv_per_1000_moves | 0.568 | +0.872 | [+0.401, +1.343] |
| FA | RRv_per_episode | 0.656 | -0.131 | [-0.593, +0.332] |
| FA | RRv_per_1000_moves | 0.617 | -0.063 | [-0.545, +0.419] |
| FR | ROv_per_episode | 0.733 | +0.628 | [+0.280, +0.976] |
| FR | ROv_per_1000_moves | 0.611 | +0.666 | [+0.346, +0.987] |
| FR | RRv_per_episode | 1.069 | -0.936 | [-2.769, +0.897] |
| FR | RRv_per_1000_moves | 0.892 | -0.766 | [-2.278, +0.745] |
| NA | ROv_per_episode | 1.085 | +0.511 | [-0.247, +1.269] |
| NA | ROv_per_1000_moves | 0.515 | +0.481 | [+0.166, +0.795] |
| NA | RRv_per_episode | 1.871 | +0.561 | [-1.169, +2.291] |
| NA | RRv_per_1000_moves | 0.887 | +0.629 | [-0.385, +1.643] |
