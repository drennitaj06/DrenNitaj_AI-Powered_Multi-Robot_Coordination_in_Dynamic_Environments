# Issue 6A Phase 2: tables

Generated tables, copied verbatim from `results/dev6a/phase2/micro/micro_tables.md` (Part A) and `results/dev6a/phase2/eval/tables.md` (Part B). The narrative is in `2026-11-22-issue-6a-phase2-rrt-budget.md`.

---

# Part A: microbenchmarks


Selected: **m400** {'refine_iterations': None, 'patience': 400}; eligible: ['m400'].

### E1a selection stratum (sizes 16 and 32; 1 400 queries; budget 1 000)

| Config | Success | Q steps (all) | max_f Q steps | Q planner cost | Paths identical to full | Max step increase | Mean iterations | Mean segment checks | Median first solution |
|---|---|---|---|---|---|---|---|---|---|
| full | 0.874 | 1.0000 | 1.0000 | 1.0000 | 1.000 | 0 | 1000.0 | 11321 | 66 |
| k0 | 0.874 | 1.1519 | 1.2219 | 1.2253 | 0.433 | 34 | 273.1 | 455 | 66 |
| k25 | 0.874 | 1.1402 | 1.2074 | 1.1964 | 0.469 | 34 | 294.9 | 502 | 66 |
| k50 | 0.874 | 1.1311 | 1.1979 | 1.1788 | 0.496 | 32 | 316.6 | 562 | 66 |
| k100 | 0.874 | 1.1141 | 1.1756 | 1.1524 | 0.546 | 26 | 359.7 | 722 | 66 |
| k200 | 0.874 | 1.0825 | 1.1399 | 1.1094 | 0.646 | 24 | 445.1 | 1207 | 66 |
| k400 | 0.874 | 1.0486 | 1.0998 | 1.0611 | 0.774 | 24 | 610.7 | 2862 | 66 |
| m25 | 0.874 | 1.1361 | 1.2023 | 1.1878 | 0.482 | 34 | 307.4 | 540 | 66 |
| m50 | 0.874 | 1.1090 | 1.1700 | 1.1470 | 0.556 | 24 | 360.8 | 788 | 66 |
| m100 | 0.874 | 1.0648 | 1.1226 | 1.0851 | 0.704 | 24 | 500.9 | 2172 | 66 |
| m200 | 0.874 | 1.0212 | 1.0552 | 1.0279 | 0.902 | 24 | 753.3 | 6108 | 66 |
| m400 | 0.874 | 1.0007 | 1.0018 | 1.0017 | 0.993 | 2 | 943.9 | 9927 | 66 |

### E1a by family (stratum): Q steps / mean iterations

| Config | d10 | d20 | d30 | d35 | open | rooms | warehouse |
|---|---|---|---|---|---|---|---|
| full | 1.000 / 1000 | 1.000 / 1000 | 1.000 / 1000 | 1.000 / 1000 | 1.000 / 1000 | 1.000 / 1000 | 1.000 / 1000 |
| k0 | 1.222 / 76 | 1.136 / 219 | 1.114 / 473 | 1.129 / 542 | 1.187 / 32 | 1.128 / 351 | 1.125 / 219 |
| k25 | 1.207 / 101 | 1.122 / 243 | 1.107 / 491 | 1.124 / 558 | 1.171 / 57 | 1.120 / 372 | 1.113 / 243 |
| k50 | 1.198 / 126 | 1.109 / 267 | 1.099 / 509 | 1.118 / 573 | 1.159 / 82 | 1.107 / 393 | 1.112 / 267 |
| k100 | 1.176 / 175 | 1.096 / 314 | 1.075 / 544 | 1.104 / 604 | 1.138 / 132 | 1.098 / 434 | 1.096 / 315 |
| k200 | 1.140 / 274 | 1.060 / 407 | 1.039 / 614 | 1.082 / 665 | 1.108 / 232 | 1.075 / 515 | 1.062 / 409 |
| k400 | 1.100 / 472 | 1.037 / 587 | 1.023 / 744 | 1.028 / 779 | 1.070 / 432 | 1.042 / 670 | 1.026 / 591 |
| m25 | 1.202 / 116 | 1.117 / 257 | 1.105 / 500 | 1.119 / 567 | 1.166 / 70 | 1.116 / 384 | 1.110 / 258 |
| m50 | 1.170 / 171 | 1.083 / 319 | 1.068 / 543 | 1.103 / 604 | 1.136 / 135 | 1.090 / 438 | 1.097 / 315 |
| m100 | 1.123 / 337 | 1.045 / 484 | 1.031 / 647 | 1.041 / 697 | 1.093 / 314 | 1.058 / 558 | 1.045 / 468 |
| m200 | 1.055 / 660 | 1.013 / 744 | 1.012 / 822 | 1.012 / 850 | 1.033 / 671 | 1.013 / 760 | 1.004 / 767 |
| m400 | 1.001 / 929 | 1.001 / 939 | 1.000 / 952 | 1.001 / 975 | 1.002 / 935 | 1.001 / 937 | 1.000 / 941 |

### E1a by size: full vs m400 (all families)

| Size | Success (full = selected) | Q steps | Mean iterations full → selected | Solved-query iterations selected |
|---|---|---|---|---|
| 16 | 0.981 = 0.981 | 1.0003 | 1000.0 → 927.6 | 926.3 |
| 32 | 0.766 = 0.766 | 1.0010 | 1000.0 → 960.2 | 948.0 |
| 64 | 0.511 = 0.511 | 1.0016 | 1000.0 → 945.1 | 892.6 |
| 128 | 0.329 = 0.329 | 1.0006 | 1000.0 → 960.6 | 880.1 |

### Per-call time, untraced (stratum; min of 3 repetitions)

| Family | Median ms full | Median ms selected | Total time ratio | Iteration ratio |
|---|---|---|---|---|
| d10 | 140.19 | 116.78 | 0.897 | 0.929 |
| d20 | 105.46 | 86.03 | 0.893 | 0.939 |
| d30 | 59.72 | 51.02 | 0.894 | 0.952 |
| d35 | 55.73 | 49.51 | 0.954 | 0.975 |
| open | 142.58 | 115.83 | 0.894 | 0.935 |
| rooms | 86.61 | 69.93 | 0.895 | 0.937 |
| warehouse | 122.51 | 81.41 | 0.889 | 0.941 |
| all | 103.83 | 89.06 | 0.899 | 0.944 |

### E1c (descriptive; 20 queries; stored best cost at 1 000 reproduced: 20/20)

| Family | Solved full / selected | Mean iterations selected | Cost sel / full@1000 | full@1000 / stored@20000 | sel / stored@20000 | sel / lower bound | Steps sel / full |
|---|---|---|---|---|---|---|---|
| open | 10 / 10 | 1000 | 1.0000 | 1.0780 | 1.0780 | 1.0793 | 1.0000 |
| d20 | 7 / 7 | 897.4 | 1.0062 | 1.3047 | 1.3127 | 1.3508 | 1.0095 |

### E2 replanning (reported comparison after selection; 320 paired episodes, seeds 20 000–20 019)

| Metric | full | selected |
|---|---|---|
| outcomes | {'done': 320} | {'done': 320} |
| completion | 0.8179 | 0.8296 |
| delivered | 1963 | 1991 |
| terminated | 211 | 213 |
| steps_mean | 191.7031 | 189.6219 |
| conflicts_total | 40045 | 35108 |
| planner_calls_total | 9462 | 9513 |
| iterations_total | 9446000 | 9003520 |
| iterations_per_call | 998.3090 | 946.4438 |
| latency_median_ms_median | 138.4082 | 133.0711 |
| planning_total_s | 1300.8350 | 1206.0847 |
| wall_total_s | 1355.9210 | 1260.5306 |
| mean_flow_time_mean | 82.4816 | 82.5089 |
| path_efficiency_mean | 0.8619 | 0.8571 |
| status_mix | {'ok': 9240, 'ok_degraded': 29, 'unreachable': 8, 'budget_exhausted': 185, 'invalid_query': 0} | {'ok': 9318, 'ok_degraded': 28, 'unreachable': 8, 'budget_exhausted': 159, 'invalid_query': 0} |

Paired episodes: {"delivered_better": 22, "delivered_worse": 11, "steps_longer": 38, "steps_shorter": 21}


---

# Part B: Block C evaluation


Selected rule: {"label": "m400", "refine": null, "patience": 400}. Completeness: {"expected": 2400, "present": 2400, "missing": 0, "unexpected": 0, "complete": true}.

G0.1 (L `full` = Phase 1 `rsd_r0` RRT\*): 1080 / 1080 identical.
G0.2 replay: 48 replayed, identical: True.

Recommendation inputs: `{"L": "inconclusive", "B": "adopt", "g0_1": true, "replay_identical": true, "p2_forward_default": false}`

## L: 9 learned policies, RRT\*, supervisor R + S + D

### G0

```
{
 "episodes": 2160,
 "errors": [],
 "physical_collisions": 0,
 "inv_total": 0,
 "invalid_reference_paths": 0,
 "reference_paths_checked": 505016,
 "p2_arm_rule_ok": true,
 "full_arm_rule_absent": true,
 "full_arm_never_stopped_early": true,
 "repair_off": true,
 "instances_identical_across_arms": true,
 "supervised_episodes": 2160,
 "supervisor_mode_ok": true,
 "supervisor_introduced_invalid_moves": 0,
 "fairness_bound_ok": true,
 "burst_bound_violations": [],
 "params_frozen": true,
 "pass": true
}
```

δ₁: {"madql": 0.0517, "ippo": 0.0517, "mappo": 0.0517}

### P1

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | completion | 0.9914 | 0.0019 | [-0.0005, 0.0042] | 0.11919 | 0.35756 | 0.03 | not rejected | no |
| ippo | completion | 1.0000 | -0.0001 | [-inf, inf] | 0.44946 | 0.44946 | 0.03 | not rejected | no |
| mappo | completion | 0.9993 | -0.0003 | [-inf, inf] | 0.15218 | 0.35756 | 0.03 | not rejected | no |

### P2

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql_seed1 | completion | 0.9953 | 0.0009 | [-0.0030, 0.0047] | 0.58094 | 1.00000 | 0.005 | not rejected | no |
| madql_seed2 | completion | 0.9824 | 0.0056 | [0.0002, 0.0110] | 0.04340 | 0.39056 | 0.005 | not rejected | no |
| madql_seed3 | completion | 0.9966 | -0.0009 | [-0.0030, 0.0012] | 0.43466 | 1.00000 | 0.005 | not rejected | no |
| ippo_seed1 | completion | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.00000 | 1.00000 | 0.005 | not rejected | no |
| ippo_seed2 | completion | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.00000 | 1.00000 | 0.005 | not rejected | no |
| ippo_seed3 | completion | 1.0000 | -0.0004 | [-inf, inf] | 0.44946 | 1.00000 | 0.005 | not rejected | no |
| mappo_seed1 | completion | 0.9991 | 0.0000 | [-0.0011, 0.0011] | 1.00000 | 1.00000 | 0.005 | not rejected | no |
| mappo_seed2 | completion | 0.9991 | -0.0004 | [-inf, inf] | 0.44636 | 1.00000 | 0.005 | not rejected | no |
| mappo_seed3 | completion | 0.9996 | -0.0004 | [-inf, inf] | 0.74883 | 1.00000 | 0.005 | not rejected | no |

### E

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | planner_calls_per_100_robot_steps | 18.9765 | 0.1138 | [-0.2310, 0.4586] | 0.51035 | 1.00000 | 0.01 | not rejected | no |
| madql | rrt_iterations_per_robot_step | 189.5553 | -30.7167 | [-34.0360, -27.3974] | 0.00010 | 0.00060 | 0.01 | **rejected** | no |
| ippo | planner_calls_per_100_robot_steps | 12.1242 | -0.0590 | [-0.2472, 0.1292] | 0.52055 | 1.00000 | 0.01 | not rejected | no |
| ippo | rrt_iterations_per_robot_step | 120.9729 | -13.5227 | [-15.5954, -11.4501] | 0.00010 | 0.00060 | 0.01 | **rejected** | no |
| mappo | planner_calls_per_100_robot_steps | 11.3386 | -0.0305 | [-0.2253, 0.1643] | 0.74303 | 1.00000 | 0.01 | not rejected | no |
| mappo | rrt_iterations_per_robot_step | 112.8266 | -12.0788 | [-13.7179, -10.4396] | 0.00010 | 0.00060 | 0.01 | **rejected** | no |

### P4

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| madql | conflicts_per_episode | 1.8083 | -0.2722 | [-1.3019, 0.7575] | 0.60864 | 1.00000 | 0.005 | not rejected | no |
| madql | top_priority_common_flow_time | 64.7384 | 0.2538 | [-0.6269, 1.1344] | 0.55954 | 1.00000 | 0.005 | not rejected | no |
| madql | top_priority_undelivered | — | not evaluable | — | — | 1 | 0.005 | not rejected | — |
| ippo | conflicts_per_episode | 1.9611 | -0.1639 | [-1.4222, 1.0944] | 0.62654 | 1.00000 | 0.005 | not rejected | no |
| ippo | top_priority_common_flow_time | 45.6952 | 0.1177 | [-0.3853, 0.6207] | 0.63544 | 1.00000 | 0.005 | not rejected | no |
| ippo | top_priority_undelivered | — | not evaluable | — | — | 1 | 0.005 | not rejected | — |
| mappo | conflicts_per_episode | 1.9944 | -0.5250 | [-2.9567, 1.9067] | 0.42716 | 1.00000 | 0.005 | not rejected | no |
| mappo | top_priority_common_flow_time | 44.6069 | 0.3614 | [-0.0979, 0.8207] | 0.11849 | 1.00000 | 0.005 | not rejected | no |
| mappo | top_priority_undelivered | — | not evaluable | — | — | 1 | 0.005 | not rejected | — |

### Decision

```
{
 "no_harm": false,
 "harm": [],
 "lower_bound_ok": {
  "madql": true,
  "ippo": false,
  "mappo": false
 },
 "vetoes": [],
 "benefit": [
  "madql|rrt_iterations_per_robot_step",
  "ippo|rrt_iterations_per_robot_step",
  "mappo|rrt_iterations_per_robot_step"
 ],
 "verdict": "inconclusive"
}
```

### System (descriptive)

| Metric | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| Completion | 0.9914 | 1.0000 | 0.9993 | 0.9933 | 0.9999 | 0.9990 |
| Delivered | 6909 | 6969 | 6964 | 6922 | 6968 | 6962 |
| Undelivered / ep. | 0.167 | 0.000 | 0.014 | 0.131 | 0.003 | 0.019 |
| Successful ep. | 315 | 360 | 355 | 321 | 359 | 353 |
| Conflicts / ep. | 1.81 | 1.96 | 1.99 | 1.54 | 1.80 | 1.47 |
| Conflicts / delivered | 0.094 | 0.101 | 0.103 | 0.080 | 0.093 | 0.076 |
| Conflicts / 100 robot-steps | 0.084 | 0.147 | 0.148 | 0.072 | 0.135 | 0.108 |
| Flow time (delivered) | 100.68 | 61.36 | 59.96 | 100.98 | 61.57 | 60.75 |
| High-priority flow time | 64.74 | 45.70 | 44.61 | 64.99 | 45.81 | 44.97 |
| Wait ratio | 0.1546 | 0.0150 | 0.0108 | 0.1551 | 0.0149 | 0.0132 |
| Steps / ep. | 269.9 | 166.6 | 168.7 | 266.8 | 166.6 | 169.5 |
| Makespan (mean) | 251.3 | 165.2 | 164.0 | 250.6 | 164.8 | 163.5 |
| Stuck robots | 60 | 0 | 5 | 47 | 1 | 7 |
| Supervisor changes / ep. | 15.18 | 0.73 | 0.46 | 14.85 | 0.66 | 0.60 |
| Calls / 100 robot-steps | 18.98 | 12.12 | 11.34 | 19.09 | 12.07 | 11.31 |
| Deviations / 100 robot-steps | 15.40 | 7.00 | 6.19 | 15.46 | 6.95 | 6.16 |

### RRT\* and runtime (descriptive)

| Metric | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| RRT\* calls / ep. | 409.2 | 161.2 | 152.3 | 407.1 | 160.4 | 152.6 |
| Iterations / call | 1000.0 | 1000.0 | 1000.0 | 832.7 | 892.6 | 895.0 |
| Share of solved calls stopped early | 0.000 | 0.000 | 0.000 | 0.515 | 0.354 | 0.349 |
| Budget-exhausted calls | 3759 | 1787 | 2075 | 3790 | 1769 | 2228 |
| Mean first-solution iteration | 103.6 | 141.1 | 142.0 | 104.2 | 140.4 | 142.4 |
| Moves / successful plan | 12.29 | 17.67 | 17.59 | 12.47 | 17.67 | 17.72 |
| ms / call (mean) | 102.3 | 100.7 | 99.7 | 78.1 | 85.2 | 84.6 |
| ms / call (median of ep. medians) | 104.1 | 102.6 | 102.3 | 81.4 | 91.1 | 90.4 |
| RRT\* s / ep. (median) | 37.32 | 14.69 | 13.55 | 28.06 | 12.10 | 11.50 |
| Wall s / ep. (median) | 38.18 | 15.08 | 14.08 | 28.86 | 12.47 | 11.95 |
| Planning share of step time | 0.980 | 0.967 | 0.965 | 0.973 | 0.961 | 0.959 |
| Tree nodes / call | 694.6 | 691.7 | 690.8 | 564.2 | 607.7 | 608.3 |
| Max tree nodes | 841 | 827 | 827 | 832 | 827 | 821 |
| Segment checks / call | 7833 | 7468 | 7430 | 5561 | 6020 | 5989 |
| RRT\* s total | 15069 | 5842 | 5464 | 11452 | 4921 | 4646 |

### Conflicts by type (total)

| Category | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| INV | 0 | 0 | 0 | 0 | 0 | 0 |
| ROs | 0 | 0 | 0 | 0 | 0 | 0 |
| ROv | 264 | 539 | 488 | 232 | 577 | 479 |
| RRs | 2 | 41 | 23 | 1 | 41 | 21 |
| RRv | 385 | 126 | 207 | 320 | 29 | 29 |

### Planner calls by trigger (share)

| Category | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| T1 | 0.1039 | 0.2634 | 0.2783 | 0.1047 | 0.2653 | 0.2779 |
| T2 | 0.0492 | 0.1158 | 0.1200 | 0.0502 | 0.1162 | 0.1199 |
| T3 | 0.8167 | 0.5803 | 0.5505 | 0.8157 | 0.5790 | 0.5489 |
| T4 | 0.0003 | 0.0002 | 0.0003 | 0.0002 | 0.0001 | 0.0001 |
| T5 | 0.0251 | 0.0304 | 0.0369 | 0.0255 | 0.0301 | 0.0395 |
| T6 | 0.0003 | 0.0006 | 0.0011 | 0.0002 | 0.0003 | 0.0027 |
| T7 | 0.0013 | 0.0036 | 0.0030 | 0.0012 | 0.0034 | 0.0031 |
| T8 | 0.0032 | 0.0055 | 0.0099 | 0.0024 | 0.0057 | 0.0079 |

### RRT\* call status (share)

| Category | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| budget_exhausted | 0.0255 | 0.0308 | 0.0378 | 0.0259 | 0.0306 | 0.0405 |
| ok | 0.9739 | 0.9682 | 0.9609 | 0.9737 | 0.9686 | 0.9564 |
| ok_degraded | 0.0005 | 0.0010 | 0.0013 | 0.0004 | 0.0007 | 0.0030 |

### Stuck robots by category (per ep.)

| Category | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| blocked_by_idle | 0.0028 | 0.0000 | 0.0028 | 0.0000 | 0.0000 | 0.0000 |
| conflict_loop | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| deadlock_cycle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| frozen | 0.0083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0028 |
| mostly_stationary | 0.0028 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oscillating | 0.0056 | 0.0000 | 0.0000 | 0.0056 | 0.0000 | 0.0000 |
| other_moving | 0.1472 | 0.0000 | 0.0111 | 0.1250 | 0.0028 | 0.0167 |

### Safety

| Metric | madql full | ippo full | mappo full | madql p2 | ippo p2 | mappo p2 |
|---|---|---|---|---|---|---|
| Physical collisions | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid moves (INV) | 0 | 0 | 0 | 0 | 0 | 0 |
| Invalid reference paths | 0 | 0 | 0 | 0 | 0 | 0 |
| Reference paths checked | 143710 | 56367 | 53015 | 142864 | 56104 | 52956 |
| Errors | 0 | 0 | 0 | 0 | 0 | 0 |

### P1 interval sensitivity (not pre-registered; does not change the verdict)

| Family | Δ̂ | bootstrap-t 95 % CI | percentile 95 % CI | −δ₁ | percentile lower > −δ₁ |
|---|---|---|---|---|---|
| madql | 0.0019 | [-0.0005, 0.0042] | [-0.0004, 0.0041] | -0.0517 | True |
| ippo | -0.0001 | [-inf, inf] | [-0.0004, 0.0000] | -0.0517 | True |
| mappo | -0.0003 | [-inf, inf] | [-0.0007, 0.0000] | -0.0517 | True |

### Supportive (nominal)

| Outcome | Δ̂ | nominal 95 % CI | full θ | p2 θ |
|---|---|---|---|---|
| madql|undelivered_per_episode | -0.0361 | [-0.0820, 0.0098] | 0.1667 | 0.1306 |
| madql|stuck_total_per_episode | -0.0361 | [-0.0820, 0.0098] | 0.1667 | 0.1306 |
| madql|steps_per_episode | -3.0833 | [-7.3734, 1.2068] | 269.8750 | 266.7917 |
| madql|rrt_iterations_per_call | -167.2732 | [-172.4490, -162.0973] | 1000.0000 | 832.7268 |
| madql|rrt_time_s_per_episode | -10.0469 | [-10.7452, -9.3486] | 41.8570 | 31.8101 |
| madql|moves_per_successful_plan | 0.1790 | [-0.0224, 0.3804] | 12.2880 | 12.4670 |
| madql|conflicts_per_100_robot_steps | -0.0118 | [-0.0591, 0.0356] | 0.0838 | 0.0720 |
| madql|mean_flow_time_delivered | 0.2950 | [-0.6428, 1.2328] | 100.6830 | 100.9780 |
| madql|wait_ratio | 0.0005 | [-0.0039, 0.0049] | 0.1546 | 0.1551 |
| ippo|undelivered_per_episode | 0.0028 | [-inf, inf] | 0.0000 | 0.0028 |
| ippo|stuck_total_per_episode | 0.0028 | [-inf, inf] | 0.0000 | 0.0028 |
| ippo|steps_per_episode | 0.0056 | [-2.0039, 2.0150] | 166.5778 | 166.5833 |
| ippo|rrt_iterations_per_call | -107.4473 | [-110.8786, -104.0161] | 1000.0000 | 892.5527 |
| ippo|rrt_time_s_per_episode | -2.5600 | [-2.7868, -2.3332] | 16.2282 | 13.6682 |
| ippo|moves_per_successful_plan | 0.0009 | [-0.1373, 0.1391] | 17.6707 | 17.6716 |
| ippo|conflicts_per_100_robot_steps | -0.0123 | [-0.1033, 0.0787] | 0.1472 | 0.1349 |
| ippo|mean_flow_time_delivered | 0.2145 | [-0.1678, 0.5967] | 61.3570 | 61.5715 |
| ippo|wait_ratio | -0.0001 | [-0.0025, 0.0023] | 0.0150 | 0.0149 |
| mappo|undelivered_per_episode | 0.0056 | [-inf, inf] | 0.0139 | 0.0194 |
| mappo|stuck_total_per_episode | 0.0056 | [-inf, inf] | 0.0139 | 0.0194 |
| mappo|steps_per_episode | 0.7528 | [-2.7279, 4.2334] | 168.7417 | 169.4944 |
| mappo|rrt_iterations_per_call | -104.9661 | [-109.7316, -100.2006] | 1000.0000 | 895.0339 |
| mappo|rrt_time_s_per_episode | -2.2719 | [-2.4733, -2.0705] | 15.1778 | 12.9059 |
| mappo|moves_per_successful_plan | 0.1331 | [-0.0380, 0.3041] | 17.5871 | 17.7201 |
| mappo|conflicts_per_100_robot_steps | -0.0394 | [-0.2185, 0.1398] | 0.1477 | 0.1084 |
| mappo|mean_flow_time_delivered | 0.7907 | [0.3909, 1.1905] | 59.9632 | 60.7540 |
| mappo|wait_ratio | 0.0023 | [-0.0015, 0.0062] | 0.0108 | 0.0132 |

### Per level, completion (nominal)

| Family / level | full | p2 | Δ̂ | CI |
|---|---|---|---|---|
| madql|static | 0.9964 | 0.9969 | 0.0005 | [-0.0016, 0.0027] |
| madql|obstacles | 0.9953 | 0.9958 | 0.0005 | [-0.0016, 0.0027] |
| madql|full | 0.9859 | 0.9895 | 0.0035 | [-0.0012, 0.0082] |
| ippo|static | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| ippo|obstacles | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| ippo|full | 1.0000 | 0.9997 | -0.0003 | [-inf, inf] |
| mappo|static | 1.0000 | 0.9995 | -0.0005 | [-inf, inf] |
| mappo|obstacles | 0.9990 | 0.9984 | -0.0005 | [-inf, inf] |
| mappo|full | 0.9990 | 0.9990 | 0.0000 | [0.0000, 0.0000] |

### Per policy × level (descriptive): full → p2

| Policy / level | Completion | Undelivered | Conflicts / ep. | Flow time | Steps / ep. | Iterations / call | RRT\* calls / ep. | RRT\* s / ep. (median) | Wall s / ep. (median) |
|---|---|---|---|---|---|---|---|---|---|
| madql_seed1|static | 0.9953 → 0.9953 | 3 → 3 | 2.90 → 1.90 | 92.5 → 93.1 | 219.8 → 220.6 | 1000 → 818 | 346.8 → 339.1 | 35.64 → 24.93 | 36.24 → 25.45 |
| madql_seed1|obstacles | 0.9953 → 0.9969 | 3 → 2 | 6.67 → 5.15 | 92.4 → 92.3 | 221.8 → 212.8 | 1000 → 839 | 360.2 → 349.7 | 34.70 → 26.31 | 35.29 → 26.98 |
| madql_seed1|full | 0.9952 → 0.9962 | 5 → 4 | 1.75 → 3.17 | 99.8 → 99.4 | 320.0 → 313.6 | 1000 → 836 | 583.1 → 587.4 | 58.66 → 44.97 | 60.12 → 46.31 |
| madql_seed2|static | 0.9953 → 0.9969 | 3 → 2 | 0.10 → 0.05 | 109.3 → 110.5 | 278.9 → 278.0 | 1000 → 819 | 307.8 → 306.8 | 31.93 → 23.22 | 32.58 → 23.94 |
| madql_seed2|obstacles | 0.9938 → 0.9953 | 4 → 3 | 0.85 → 0.82 | 104.4 → 105.0 | 275.9 → 274.3 | 1000 → 847 | 347.0 → 341.9 | 34.94 → 27.30 | 35.90 → 28.00 |
| madql_seed2|full | 0.9674 → 0.9779 | 34 → 23 | 1.02 → 0.95 | 109.9 → 110.6 | 360.5 → 360.9 | 1000 → 843 | 559.2 → 563.9 | 55.93 → 44.33 | 57.51 → 45.80 |
| madql_seed3|static | 0.9984 → 0.9984 | 1 → 1 | 0.00 → 0.07 | 95.0 → 93.4 | 215.7 → 204.4 | 1000 → 810 | 302.5 → 287.8 | 32.11 → 21.41 | 32.72 → 21.87 |
| madql_seed3|obstacles | 0.9969 → 0.9953 | 2 → 3 | 1.18 → 0.60 | 95.1 → 95.0 | 225.8 → 224.4 | 1000 → 836 | 328.4 → 335.1 | 33.35 → 26.19 | 34.03 → 26.87 |
| madql_seed3|full | 0.9952 → 0.9942 | 5 → 6 | 1.80 → 1.10 | 102.1 → 103.1 | 310.5 → 312.0 | 1000 → 832 | 548.4 → 552.3 | 56.52 → 44.05 | 57.90 → 45.35 |
| ippo_seed1|static | 1.0000 → 1.0000 | 0 → 0 | 0.05 → 0.10 | 60.2 → 61.0 | 135.1 → 132.7 | 1000 → 886 | 112.0 → 117.1 | 10.96 → 9.83 | 11.33 → 10.30 |
| ippo_seed1|obstacles | 1.0000 → 1.0000 | 0 → 0 | 1.20 → 1.35 | 64.6 → 64.2 | 142.8 → 137.8 | 1000 → 893 | 145.3 → 141.5 | 14.90 → 11.67 | 15.32 → 12.14 |
| ippo_seed1|full | 1.0000 → 1.0000 | 0 → 0 | 5.20 → 2.70 | 63.6 → 63.7 | 239.1 → 243.7 | 1000 → 897 | 235.7 → 230.1 | 23.39 → 19.37 | 24.41 → 20.26 |
| ippo_seed2|static | 1.0000 → 1.0000 | 0 → 0 | 0.10 → 0.07 | 56.2 → 57.3 | 117.2 → 122.5 | 1000 → 890 | 98.6 → 101.3 | 9.95 → 8.55 | 10.30 → 8.85 |
| ippo_seed2|obstacles | 1.0000 → 1.0000 | 0 → 0 | 2.75 → 2.80 | 62.0 → 62.1 | 135.2 → 131.8 | 1000 → 899 | 134.6 → 132.7 | 13.65 → 11.24 | 13.99 → 11.69 |
| ippo_seed2|full | 1.0000 → 1.0000 | 0 → 0 | 4.17 → 4.97 | 59.9 → 59.9 | 227.4 → 228.3 | 1000 → 896 | 215.2 → 212.4 | 20.80 → 17.75 | 21.74 → 18.65 |
| ippo_seed3|static | 1.0000 → 1.0000 | 0 → 0 | 0.05 → 0.20 | 59.0 → 59.2 | 128.3 → 125.1 | 1000 → 882 | 118.2 → 116.7 | 11.63 → 9.53 | 11.99 → 9.89 |
| ippo_seed3|obstacles | 1.0000 → 1.0000 | 0 → 0 | 1.62 → 1.48 | 63.6 → 63.5 | 139.8 → 138.4 | 1000 → 892 | 152.7 → 148.3 | 14.76 → 12.18 | 15.15 → 12.59 |
| ippo_seed3|full | 1.0000 → 0.9990 | 0 → 1 | 2.50 → 2.50 | 62.0 → 62.4 | 234.4 → 239.0 | 1000 → 892 | 238.6 → 243.7 | 23.20 → 19.48 | 24.12 → 20.43 |
| mappo_seed1|static | 1.0000 → 0.9984 | 0 → 1 | 0.40 → 0.00 | 57.3 → 56.8 | 128.7 → 126.9 | 1000 → 886 | 115.8 → 113.3 | 11.26 → 9.30 | 11.59 → 9.63 |
| mappo_seed1|obstacles | 0.9984 → 0.9984 | 1 → 1 | 2.42 → 2.35 | 61.6 → 62.6 | 140.3 → 145.7 | 1000 → 894 | 146.5 → 149.0 | 14.60 → 12.52 | 14.98 → 12.92 |
| mappo_seed1|full | 0.9990 → 1.0000 | 1 → 0 | 3.40 → 3.15 | 60.1 → 61.4 | 230.8 → 235.8 | 1000 → 894 | 236.9 → 234.3 | 22.93 → 19.10 | 23.83 → 19.98 |
| mappo_seed2|static | 1.0000 → 1.0000 | 0 → 0 | 0.17 → 0.07 | 56.7 → 57.5 | 125.8 → 127.3 | 1000 → 892 | 101.0 → 101.2 | 10.04 → 8.49 | 10.38 → 8.83 |
| mappo_seed2|obstacles | 0.9984 → 0.9984 | 1 → 1 | 1.32 → 1.60 | 61.7 → 63.1 | 144.3 → 143.6 | 1000 → 899 | 140.8 → 141.3 | 13.16 → 11.73 | 13.56 → 12.24 |
| mappo_seed2|full | 0.9990 → 0.9981 | 1 → 2 | 2.20 → 2.00 | 60.6 → 61.4 | 241.2 → 239.4 | 1000 → 900 | 223.4 → 223.1 | 20.57 → 17.85 | 21.55 → 18.80 |
| mappo_seed3|static | 1.0000 → 1.0000 | 0 → 0 | 0.17 → 0.10 | 57.2 → 57.8 | 127.8 → 125.9 | 1000 → 884 | 89.9 → 89.7 | 8.66 → 7.53 | 8.98 → 7.84 |
| mappo_seed3|obstacles | 1.0000 → 0.9984 | 0 → 1 | 1.62 → 1.62 | 63.0 → 62.9 | 138.9 → 142.3 | 1000 → 900 | 120.8 → 121.0 | 11.88 → 10.29 | 12.28 → 10.70 |
| mappo_seed3|full | 0.9990 → 0.9990 | 1 → 1 | 6.22 → 2.33 | 60.6 → 61.8 | 240.9 → 238.5 | 1000 → 898 | 195.7 → 200.8 | 18.55 → 17.08 | 19.57 → 18.14 |

### Seeds (completion: better / worse / equal)

| Policy | better | worse | equal |
|---|---|---|---|
| madql_seed1 | 3 | 3 | 34 |
| madql_seed2 | 14 | 7 | 19 |
| madql_seed3 | 2 | 4 | 34 |
| ippo_seed1 | 0 | 0 | 40 |
| ippo_seed2 | 0 | 0 | 40 |
| ippo_seed3 | 0 | 1 | 39 |
| mappo_seed1 | 1 | 1 | 38 |
| mappo_seed2 | 0 | 1 | 39 |
| mappo_seed3 | 1 | 1 | 38 |

## B: baseline controller, RRT\* (resolver on, no supervisor)

### G0

```
{
 "episodes": 240,
 "errors": [],
 "physical_collisions": 0,
 "inv_total": 0,
 "invalid_reference_paths": 0,
 "reference_paths_checked": 16330,
 "p2_arm_rule_ok": true,
 "full_arm_rule_absent": true,
 "full_arm_never_stopped_early": true,
 "repair_off": true,
 "instances_identical_across_arms": true,
 "no_supervisor": true,
 "resolver_on": true,
 "pass": true
}
```

δ₁: {"baseline": 0.0517}

### P1

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| baseline | completion | 1.0000 | 0.0000 | [0.0000, 0.0000] | 1.00000 | 1.00000 | 0.03 | not rejected | no |

### E

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| baseline | planner_calls_per_100_robot_steps | 5.4700 | -0.0000 | [-0.1245, 0.1245] | 0.99990 | 0.99990 | 0.01 | not rejected | no |
| baseline | rrt_iterations_per_robot_step | 54.6487 | -5.3140 | [-6.4195, -4.2084] | 0.00010 | 0.00020 | 0.01 | **rejected** | no |

### P4

| Unit | Outcome | OFF θ | Δ̂ | nominal 95 % CI | p | Holm p | α | Decision | MC-sens. |
|---|---|---|---|---|---|---|---|---|---|
| baseline | conflicts_per_episode | 1.1250 | 0.2083 | [-0.0208, 0.4374] | 0.06889 | 0.20668 | 0.005 | not rejected | no |
| baseline | top_priority_common_flow_time | 44.2317 | 0.7683 | [-0.4030, 1.9395] | 0.18478 | 0.36956 | 0.005 | not rejected | no |
| baseline | top_priority_undelivered | — | not evaluable | — | — | 1 | 0.005 | not rejected | — |

### Decision

```
{
 "no_harm": true,
 "harm": [],
 "lower_bound_ok": {
  "baseline": true
 },
 "vetoes": [],
 "benefit": [
  "baseline|rrt_iterations_per_robot_step"
 ],
 "verdict": "adopt"
}
```

### System (descriptive)

| Metric | baseline full | baseline p2 |
|---|---|---|
| Completion | 1.0000 | 1.0000 |
| Delivered | 2323 | 2323 |
| Undelivered / ep. | 0.000 | 0.000 |
| Successful ep. | 120 | 120 |
| Conflicts / ep. | 1.12 | 1.33 |
| Conflicts / delivered | 0.058 | 0.069 |
| Conflicts / 100 robot-steps | 0.087 | 0.103 |
| Flow time (delivered) | 60.01 | 60.66 |
| High-priority flow time | 44.23 | 45.00 |
| Wait ratio | 0.0457 | 0.0428 |
| Steps / ep. | 161.2 | 161.5 |
| Makespan (mean) | 159.7 | 160.1 |
| Stuck robots | 0 | 0 |
| Supervisor changes / ep. | 0.00 | 0.00 |
| Calls / 100 robot-steps | 5.47 | 5.47 |
| Deviations / 100 robot-steps | 0.00 | 0.00 |

### RRT\* and runtime (descriptive)

| Metric | baseline full | baseline p2 |
|---|---|---|
| RRT\* calls / ep. | 70.5 | 70.6 |
| Iterations / call | 1000.0 | 903.2 |
| Share of solved calls stopped early | 0.000 | 0.321 |
| Budget-exhausted calls | 353 | 261 |
| Mean first-solution iteration | 150.2 | 148.6 |
| Moves / successful plan | 18.72 | 18.96 |
| ms / call (mean) | 98.4 | 84.9 |
| ms / call (median of ep. medians) | 101.5 | 91.3 |
| RRT\* s / ep. (median) | 6.25 | 5.07 |
| Wall s / ep. (median) | 6.52 | 5.35 |
| Planning share of step time | 0.954 | 0.947 |
| Tree nodes / call | 685.6 | 612.0 |
| Max tree nodes | 819 | 812 |
| Segment checks / call | 7333 | 6084 |
| RRT\* s total | 832 | 719 |

### Conflicts by type (total)

| Category | baseline full | baseline p2 |
|---|---|---|
| INV | 0 | 0 |
| ROs | 0 | 0 |
| ROv | 135 | 160 |
| RRs | 0 | 0 |
| RRv | 0 | 0 |

### Planner calls by trigger (share)

| Category | baseline full | baseline p2 |
|---|---|---|
| T1 | 0.6034 | 0.6038 |
| T2 | 0.2738 | 0.2845 |
| T4 | 0.0150 | 0.0129 |
| T5 | 0.0411 | 0.0305 |
| T6 | 0.0018 | 0.0019 |
| T7 | 0.0608 | 0.0624 |
| T8 | 0.0040 | 0.0040 |

### RRT\* call status (share)

| Category | baseline full | baseline p2 |
|---|---|---|
| budget_exhausted | 0.0417 | 0.0308 |
| ok | 0.9565 | 0.9673 |
| ok_degraded | 0.0018 | 0.0019 |

### Stuck robots by category (per ep.)

| Category | baseline full | baseline p2 |
|---|---|---|
| blocked_by_idle | 0.0000 | 0.0000 |
| conflict_loop | 0.0000 | 0.0000 |
| deadlock_cycle | 0.0000 | 0.0000 |
| frozen | 0.0000 | 0.0000 |
| mostly_stationary | 0.0000 | 0.0000 |
| oscillating | 0.0000 | 0.0000 |
| other_moving | 0.0000 | 0.0000 |

### Safety

| Metric | baseline full | baseline p2 |
|---|---|---|
| Physical collisions | 0 | 0 |
| Invalid moves (INV) | 0 | 0 |
| Invalid reference paths | 0 | 0 |
| Reference paths checked | 8112 | 8218 |
| Errors | 0 | 0 |

### P1 interval sensitivity (not pre-registered; does not change the verdict)

| Family | Δ̂ | bootstrap-t 95 % CI | percentile 95 % CI | −δ₁ | percentile lower > −δ₁ |
|---|---|---|---|---|---|
| baseline | 0.0000 | [0.0000, 0.0000] | [0.0000, 0.0000] | -0.0517 | True |

### Supportive (nominal)

| Outcome | Δ̂ | nominal 95 % CI | full θ | p2 θ |
|---|---|---|---|---|
| baseline|undelivered_per_episode | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| baseline|stuck_total_per_episode | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| baseline|steps_per_episode | 0.2667 | [-4.3589, 4.8922] | 161.2000 | 161.4667 |
| baseline|rrt_iterations_per_call | -96.8129 | [-102.3952, -91.2306] | 1000.0000 | 903.1871 |
| baseline|rrt_time_s_per_episode | -0.9477 | [-1.1288, -0.7665] | 6.9357 | 5.9881 |
| baseline|moves_per_successful_plan | 0.2352 | [0.0151, 0.4554] | 18.7236 | 18.9588 |
| baseline|conflicts_per_100_robot_steps | 0.0160 | [-0.0014, 0.0334] | 0.0872 | 0.1032 |
| baseline|mean_flow_time_delivered | 0.6517 | [0.0407, 1.2627] | 60.0121 | 60.6638 |
| baseline|wait_ratio | -0.0029 | [-0.0091, 0.0033] | 0.0457 | 0.0428 |

### Per level, completion (nominal)

| Family / level | full | p2 | Δ̂ | CI |
|---|---|---|---|---|
| baseline|static | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| baseline|obstacles | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| baseline|full | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |

### Per policy × level (descriptive): full → p2

| Policy / level | Completion | Undelivered | Conflicts / ep. | Flow time | Steps / ep. | Iterations / call | RRT\* calls / ep. | RRT\* s / ep. (median) | Wall s / ep. (median) |
|---|---|---|---|---|---|---|---|---|---|
| baseline|static | 1.0000 → 1.0000 | 0 → 0 | 0.00 → 0.00 | 57.0 → 57.4 | 124.0 → 119.8 | 1000 → 894 | 41.2 → 39.5 | 3.98 → 3.44 | 4.17 → 3.62 |
| baseline|obstacles | 1.0000 → 1.0000 | 0 → 0 | 1.35 → 1.48 | 62.4 → 62.6 | 130.4 → 133.4 | 1000 → 899 | 64.4 → 64.8 | 6.30 → 5.21 | 6.57 → 5.43 |
| baseline|full | 1.0000 → 1.0000 | 0 → 0 | 2.02 → 2.52 | 60.4 → 61.5 | 229.2 → 231.3 | 1000 → 909 | 105.9 → 107.4 | 10.26 → 8.87 | 10.91 → 9.36 |

### Seeds (completion: better / worse / equal)

| Policy | better | worse | equal |
|---|---|---|---|
| baseline | 0 | 0 | 40 |

