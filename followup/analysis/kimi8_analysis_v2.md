# Prospective KDA follow-up results

Status: **complete**. Model: `/models/models--mlx-community--Kimi-Linear-48B-A3B-Instruct-8bit/snapshots/553a7a5673491df2b2d38f7380188a9d8e9f9a66`.

Evidence class: `model_evaluation`.

Manifest SHA-256: `2003aaddfece115196f8f2a6d457d90004acc26484e6645e6664807df488eda5`.

Completed 80/80 preselected contexts: 40 evaluation and 40 calibration. Qualification: 40 passed, 0 failed. All completed evaluation records contribute to the primary table, including qualification failures.

Estimates include paired 95% whole-record bootstrap intervals. Log-probability and NLL differences are condition minus never-stored; positive target log-probability difference indicates residual lift.

| Condition | Target log-probability difference | Retained log-probability difference | Continuation NLL difference |
|---|---:|---:|---:|
| present | 3.2399 [3.0625, 3.4113] | -0.0002 [-0.0005, -0.0000] | 0.0055 [-0.0083, 0.0192] |
| masked | 0.4858 [0.4257, 0.5463] | -0.0003 [-0.0005, -0.0001] | -0.0025 [-0.0161, 0.0114] |
| static | 3.1783 [3.0102, 3.3414] | 0.0001 [-0.0001, 0.0003] | -0.0133 [-0.0315, 0.0038] |
| ledger | 3.1884 [3.0179, 3.3528] | -0.0001 [-0.0002, 0.0001] | -0.0041 [-0.0193, 0.0098] |
| transport | 3.1878 [3.0180, 3.3514] | -0.0001 [-0.0002, 0.0001] | -0.0057 [-0.0194, 0.0078] |
| mask_transport | 0.0178 [0.0108, 0.0245] | -0.0001 [-0.0002, 0.0000] | 0.0049 [-0.0050, 0.0151] |
| replay | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| never_stored | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| never_stored_repeat | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| prompt_forget | 3.1205 [2.9432, 3.2908] | 0.0001 [-0.0002, 0.0004] | 0.0124 [-0.0081, 0.0339] |
| prompt_neutral | 3.2588 [3.0801, 3.4306] | -0.0029 [-0.0037, -0.0022] | -0.0356 [-0.0591, -0.0118] |

The qualification rate has a Wilson 95% interval [0.9124, 1.0000]; a zero observed failure count does not imply a zero population failure rate.

The matched token-budget contrast (prompt forgetting minus neutral instruction) is -0.1383 [-0.1550, -0.1228] for target log probability, 0.0030 [0.0024, 0.0037] for retained log probability, and 0.0480 [0.0231, 0.0733] for continuation NLL.

The forgetting and neutral instructions each add 31.3500 [31.2000, 31.5000] tokens per context (mean and whole-record interval). Their token counts match within every record.

| Condition | Target contains | Target exact | Retained contains | Retained exact |
|---|---:|---:|---:|---:|
| present | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| masked | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| static | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| ledger | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| transport | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| mask_transport | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| replay | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| never_stored | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| never_stored_repeat | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| prompt_forget | 0.8250 [0.7000, 0.9250] | 0.8250 [0.7000, 0.9250] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| prompt_neutral | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |

## Fixed finite-query attack

Three fixed candidate-scoring queries; nominal calibration FPR 5%. Threshold: -2.560179816351997. Evaluation FPR: 0.0750 [0.0000, 0.1750]. TPR intervals below include calibration-threshold uncertainty.

| Condition | AUC | TPR | TPR − FPR |
|---|---:|---:|---:|
| present | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9250 [0.7750, 1.0000] |
| masked | 0.7400 [0.6950, 0.8094] | 0.3000 [0.0500, 0.5000] | 0.2250 [0.0250, 0.3750] |
| static | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9250 [0.7750, 1.0000] |
| ledger | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9250 [0.7750, 1.0000] |
| transport | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9250 [0.7750, 1.0000] |
| mask_transport | 0.5188 [0.5113, 0.5431] | 0.0750 [0.0000, 0.2250] | 0.0000 [0.0000, 0.0750] |
| replay | 0.5000 [0.5000, 0.5000] | 0.0750 [0.0000, 0.2250] | 0.0000 [0.0000, 0.0000] |
| never_stored | 0.5000 [0.5000, 0.5000] | 0.0750 [0.0000, 0.2250] | 0.0000 [0.0000, 0.0000] |
| never_stored_repeat | 0.5000 [0.5000, 0.5000] | 0.0750 [0.0000, 0.2250] | 0.0000 [0.0000, 0.0000] |
| prompt_forget | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9250 [0.7750, 1.0000] |
| prompt_neutral | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9250 [0.7750, 1.0000] |

## Complete-state reference audits

- all_completed_records, replay_vs_never: 80/80 equal; array count range [94, 94], offset count range [7, 7]; maximum array/logit differences 0.0/0.0; equality-rate Wilson 95% interval [0.9541818704644728, 0.9999999999999999].
- all_completed_records, repeat_never_vs_never: 80/80 equal; array count range [94, 94], offset count range [7, 7]; maximum array/logit differences 0.0/0.0; equality-rate Wilson 95% interval [0.9541818704644728, 0.9999999999999999].
- evaluation_records, replay_vs_never: 40/40 equal; array count range [94, 94], offset count range [7, 7]; maximum array/logit differences 0.0/0.0; equality-rate Wilson 95% interval [0.9123783988027135, 1.0].
- evaluation_records, repeat_never_vs_never: 40/40 equal; array count range [94, 94], offset count range [7, 7]; maximum array/logit differences 0.0/0.0; equality-rate Wilson 95% interval [0.9123783988027135, 1.0].

## Scope and uncertainty

- Intervals describe variation across this frozen synthetic-context design; they do not establish deployment-population guarantees.
- Qualification failures remain in the primary estimand; qualified-only estimates are secondary and selection-conditioned.
- Percentile bootstrap intervals are marginal descriptive intervals without multiplicity adjustment.
- The token-matched neutral instruction uses fixed prose repeated/truncated to length and can end in a fragment; the prompt contrast is conditional on these specific strings.
- AUC intervals resample whole matched contexts, keeping positive and never-stored scores together.
- Zero observed event rates can produce degenerate percentile intervals; inspect the Wilson binomial interval for finite-sample rate resolution.
