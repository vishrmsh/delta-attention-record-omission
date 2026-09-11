# Kimi matched precision control

Replay matched its own never-stored state and audit logits in 8/8 released community 8-bit contexts and 8/8 original-BF16 contexts. Transport residuals and their observed arithmetic floors are reported separately for each precision below.

This is a secondary comparison of separately pinned released artifacts. The community 8-bit card names the same base repository but does not pin the original master revision. Differences cannot be attributed causally to quantization alone.

The original BF16 master is [Moonshot AI Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/e1df551a447157d4658b573f9a695d57658590e9), revision `e1df551a447157d4658b573f9a695d57658590e9`; all 20 original weight-shard SHA256 values were verified. Its 98,248,224,120 bytes of original weight files contain BF16 weights plus FP32 parameters. This is not a dequantized 8-bit model.

Eight contexts were selected before reading their outcomes: evaluation indices 000, 001, 002, 003, 006, 007, 008 and 009. Four have 128-token suffixes and four have 1,024-token suffixes. Context segments, probes, targets, instructions and stop-token IDs match exactly across the paired tokenizers; observed run token plans also agree. All records remain included regardless of qualification. Each variant is compared with its own never-stored reference, never with another precision's reference.

| State/readout check | Released community 8-bit | Original BF16 |
|---|---:|---:|
| Qualified contexts | 8/8 | 8/8 |
| Replay complete-state and audit-logit equality | 8/8 | 8/8 |
| Repeated never-stored complete-state and logit equality | 8/8 | 8/8 |
| Maximum replay state absolute difference | 0 | 0 |
| Maximum replay logit absolute difference | 0 | 0 |

State checks cover every active KDA recurrent/convolution array, logical MLA keys/values and offsets, plus the full next-token audit logit vector. Equality here means matching shapes/offsets and zero float32 maximum absolute differences, not a bytewise certificate. With eight contexts, even 8/8 successes have a Wilson 95% interval of approximately 67.6–100%; observed zero differences do not guarantee population-wide equivalence.

## Paired target readout

Values are target mean log-probability differences against the same precision's own never-stored reference, averaged across three fixed target formulations (nats per target token). Brackets show paired whole-context bootstrap 95% intervals. The final column subtracts the 8-bit own-reference difference from the BF16 own-reference difference.

| Condition | Released community 8-bit minus own reference | BF16 minus own reference | BF16-minus-8-bit difference of differences |
|---|---:|---:|---:|
| present | 3.251 [2.772, 3.746] | 3.255 [2.773, 3.756] | 0.004 [-0.013, 0.025] |
| masked | 0.409 [0.258, 0.557] | 0.403 [0.227, 0.583] | -0.006 [-0.056, 0.042] |
| static | 3.193 [2.731, 3.688] | 3.206 [2.738, 3.692] | 0.013 [-0.014, 0.041] |
| ledger | 3.211 [2.748, 3.699] | 3.212 [2.741, 3.702] | 0.001 [-0.020, 0.023] |
| transport | 3.205 [2.745, 3.693] | 3.211 [2.739, 3.700] | 0.006 [-0.013, 0.029] |
| mask_transport | 0.021 [0.013, 0.030] | -0.015 [-0.040, 0.011] | -0.036 [-0.062, -0.012] |
| replay | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| never_stored | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| never_stored_repeat | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| prompt_forget | 3.143 [2.661, 3.635] | 3.140 [2.653, 3.632] | -0.002 [-0.028, 0.027] |
| prompt_neutral | 3.279 [2.793, 3.785] | 3.286 [2.798, 3.789] | 0.007 [-0.018, 0.037] |

## Retained and continuation utility

The table gives paired mean log-probability changes for the three retained records and paired held-out continuation NLL changes, each versus the same precision's own reference. A positive retained-score change is higher probability; a positive NLL change is worse continuation scoring. All intervals resample the eight contexts, keeping their probes attached.

| Condition | Retained score Δ: Released community 8-bit | Retained score Δ: BF16 | Continuation NLL Δ: Released community 8-bit | Continuation NLL Δ: BF16 |
|---|---:|---:|---:|---:|
| present | -0.000354 [-0.000842, 0.000160] | -0.000634 [-0.001061, -0.000226] | 0.020 [-0.018, 0.065] | 0.019 [-0.003, 0.042] |
| masked | -0.000205 [-0.000506, 0.000159] | -0.000575 [-0.000890, -0.000227] | 0.012 [-0.019, 0.050] | 0.022 [0.007, 0.039] |
| transport | -0.000298 [-0.000726, 0.000084] | -0.000396 [-0.000716, -0.000063] | 0.014 [-0.029, 0.053] | 0.010 [-0.012, 0.036] |
| mask_transport | -0.000129 [-0.000381, 0.000113] | -0.000159 [-0.000462, 0.000189] | 0.009 [-0.023, 0.047] | 0.007 [-0.021, 0.040] |
| replay | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| prompt_forget | 0.000086 [-0.000386, 0.000642] | 0.000364 [-0.000126, 0.000996] | 0.028 [0.004, 0.053] | 0.029 [-0.009, 0.073] |
| prompt_neutral | -0.004670 [-0.007544, -0.002630] | -0.005882 [-0.010510, -0.003093] | 0.012 [-0.031, 0.052] | 0.014 [-0.033, 0.060] |

Continuation utility uses two fixed synthetic passages per context, selected from a shared bank of six. These passages exclude the deleted record's information. This is a narrow synthetic utility check, not a broad language benchmark. Exact-answer rates in the JSON compare the complete decoded answer after outer-whitespace stripping and casefolding; containment rates separately count a target appearing anywhere in the answer. Decoding is greedy with the frozen 24-token cap.

## Transport arithmetic diagnostics

Independently reconstruct native present/omitted recurrent states from each branch's captured k,v,g,beta in serial fp32. Also propagate two initial states through identical present inputs/writes and compare their difference to separately transported receipt. The reported empirical floor is max(sum of native-versus-serial endpoint reconstruction norms, frozen-input identity-error norm). It is an observed numerical diagnostic, not a rigorous bound or security guarantee.

| Context / suffix | Transport residual / original state difference: Released community 8-bit | BF16 | Residual / own empirical floor: Released community 8-bit | BF16 |
|---|---:|---:|---:|---:|
| 000 / 128 | 0.649 | 0.645 | 7.56e+04 | 7.56e+04 |
| 001 / 128 | 0.679 | 0.674 | 7.96e+04 | 7.79e+04 |
| 002 / 128 | 0.689 | 0.711 | 7.45e+04 | 8.04e+04 |
| 003 / 128 | 0.671 | 0.715 | 6.96e+04 | 7.7e+04 |
| 006 / 1024 | 0.712 | 0.692 | 2.56e+04 | 2.47e+04 |
| 007 / 1024 | 0.792 | 0.784 | 2.71e+04 | 2.65e+04 |
| 008 / 1024 | 0.792 | 0.817 | 2.33e+04 | 2.54e+04 |
| 009 / 1024 | 0.699 | 0.680 | 2.19e+04 | 2.1e+04 |

Released community 8-bit: maximum global present/omitted reconstruction relative errors were 1.09e-06/1.08e-06. At layer level, 3/160 layer-context residuals were at or below their own empirical floor; global ratios do not imply that every layer exceeds the floor.

Original BF16: maximum global present/omitted reconstruction relative errors were 1.08e-06/1.07e-06. At layer level, 3/160 layer-context residuals were at or below their own empirical floor; global ratios do not imply that every layer exceeds the floor.

The numerical diagnostics distinguish a transport residual from the observed arithmetic discrepancies in these runs. They do not establish a rigorous error bound or a general quantization claim. All intervals are descriptive marginal percentile intervals from 2,000 paired whole-context resamples; they exclude training variation, alternative quantizers, model families and domain shift. This eight-context control complements the larger cohort and does not establish universal deletion or utility guarantees.

Machine-readable comparison: [precision_cached8_comparison_v1.json](precision_cached8_comparison_v1.json).
