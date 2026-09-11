# Kimi matched precision control

Replay matched its own never-stored state and audit logits in 8/8 derived 8-bit contexts and 8/8 original-BF16 contexts. Transport residuals and their observed arithmetic floors are reported separately for each precision below.

The primary comparison starts from one SHA-verified original BF16 checkpoint. Its paired 8-bit variant was quantized in memory with MLX affine weight quantization, 8 bits, group size 64, and the Kimi model's quantization predicate through the supported-module/divisibility wrapper. Quantization preceded attention-mask installation; no quantized weight files were saved. The same main evaluator and token plans were used. The intervention is this specified quantization recipe and its runtime kernels, not an abstract precision-only change.

The original BF16 master is [Moonshot AI Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/e1df551a447157d4658b573f9a695d57658590e9), revision `e1df551a447157d4658b573f9a695d57658590e9`; all 20 original weight-shard SHA256 values were verified. Its 98,248,224,120 bytes of original weight files contain BF16 weights plus FP32 parameters. This is not a dequantized 8-bit model.

Eight contexts were selected before reading their outcomes: evaluation indices 000, 001, 002, 003, 006, 007, 008 and 009. Four have 128-token suffixes and four have 1,024-token suffixes. Context segments, probes, targets, instructions and stop-token IDs match exactly across the paired tokenizers; observed run token plans also agree. All records remain included regardless of qualification. Each variant is compared with its own never-stored reference, never with another precision's reference.

| State/readout check | Derived 8-bit | Original BF16 |
|---|---:|---:|
| Qualified contexts | 8/8 | 8/8 |
| Replay complete-state and audit-logit equality | 8/8 | 8/8 |
| Repeated never-stored complete-state and logit equality | 8/8 | 8/8 |
| Maximum replay state absolute difference | 0 | 0 |
| Maximum replay logit absolute difference | 0 | 0 |

State checks cover every active KDA recurrent/convolution array, logical MLA keys/values and offsets, plus the full next-token audit logit vector. Equality here means matching shapes/offsets and zero float32 maximum absolute differences, not a bytewise certificate. With eight contexts, even 8/8 successes have a Wilson 95% interval of approximately 67.6–100%; observed zero differences do not guarantee population-wide equivalence.

## Paired target readout

Values are target mean log-probability differences against the same precision's own never-stored reference, averaged across three fixed target formulations (nats per target token). Brackets show paired whole-context bootstrap 95% intervals. The final column subtracts the 8-bit own-reference difference from the BF16 own-reference difference.

| Condition | Derived 8-bit minus own reference | BF16 minus own reference | BF16-minus-8-bit difference of differences |
|---|---:|---:|---:|
| present | 3.287 [2.793, 3.810] | 3.255 [2.773, 3.756] | -0.032 [-0.076, 0.003] |
| masked | 0.426 [0.268, 0.580] | 0.403 [0.227, 0.583] | -0.023 [-0.068, 0.017] |
| static | 3.226 [2.754, 3.741] | 3.206 [2.738, 3.692] | -0.020 [-0.065, 0.018] |
| ledger | 3.241 [2.769, 3.753] | 3.212 [2.741, 3.702] | -0.029 [-0.071, 0.007] |
| transport | 3.239 [2.766, 3.753] | 3.211 [2.739, 3.700] | -0.028 [-0.073, 0.010] |
| mask_transport | 0.016 [-0.001, 0.034] | -0.015 [-0.040, 0.011] | -0.031 [-0.054, -0.009] |
| replay | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| never_stored | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| never_stored_repeat | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| prompt_forget | 3.159 [2.668, 3.680] | 3.140 [2.653, 3.632] | -0.019 [-0.073, 0.028] |
| prompt_neutral | 3.313 [2.815, 3.836] | 3.286 [2.798, 3.789] | -0.027 [-0.072, 0.015] |

## Retained and continuation utility

The table gives paired mean log-probability changes for the three retained records and paired held-out continuation NLL changes, each versus the same precision's own reference. A positive retained-score change is higher probability; a positive NLL change is worse continuation scoring. All intervals resample the eight contexts, keeping their probes attached.

| Condition | Retained score Δ: Derived 8-bit | Retained score Δ: BF16 | Continuation NLL Δ: Derived 8-bit | Continuation NLL Δ: BF16 |
|---|---:|---:|---:|---:|
| present | -0.000239 [-0.000586, 0.000090] | -0.000634 [-0.001061, -0.000226] | 0.011 [-0.011, 0.035] | 0.019 [-0.003, 0.042] |
| masked | -0.000215 [-0.000505, 0.000079] | -0.000575 [-0.000890, -0.000227] | 0.030 [0.004, 0.057] | 0.022 [0.007, 0.039] |
| transport | -0.000049 [-0.000396, 0.000334] | -0.000396 [-0.000716, -0.000063] | 0.021 [-0.002, 0.044] | 0.010 [-0.012, 0.036] |
| mask_transport | 0.000048 [-0.000246, 0.000294] | -0.000159 [-0.000462, 0.000189] | 0.029 [0.007, 0.051] | 0.007 [-0.021, 0.040] |
| replay | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| prompt_forget | 0.000165 [-0.000422, 0.000876] | 0.000364 [-0.000126, 0.000996] | 0.033 [0.019, 0.047] | 0.029 [-0.009, 0.073] |
| prompt_neutral | -0.003766 [-0.006055, -0.001916] | -0.005882 [-0.010510, -0.003093] | 0.023 [-0.021, 0.069] | 0.014 [-0.033, 0.060] |

Continuation utility uses two fixed synthetic passages per context, selected from a shared bank of six. These passages exclude the deleted record's information. This is a narrow synthetic utility check, not a broad language benchmark. Exact-answer rates in the JSON compare the complete decoded answer after outer-whitespace stripping and casefolding; containment rates separately count a target appearing anywhere in the answer. Decoding is greedy with the frozen 24-token cap.

## Transport arithmetic diagnostics

Independently reconstruct native present/omitted recurrent states from each branch's captured k,v,g,beta in serial fp32. Also propagate two initial states through identical present inputs/writes and compare their difference to separately transported receipt. The reported empirical floor is max(sum of native-versus-serial endpoint reconstruction norms, frozen-input identity-error norm). It is an observed numerical diagnostic, not a rigorous bound or security guarantee.

| Context / suffix | Transport residual / original state difference: Derived 8-bit | BF16 | Residual / own empirical floor: Derived 8-bit | BF16 |
|---|---:|---:|---:|---:|
| 000 / 128 | 0.665 | 0.645 | 7.98e+04 | 7.56e+04 |
| 001 / 128 | 0.702 | 0.674 | 8.47e+04 | 7.79e+04 |
| 002 / 128 | 0.683 | 0.711 | 7.31e+04 | 8.04e+04 |
| 003 / 128 | 0.680 | 0.715 | 6.92e+04 | 7.7e+04 |
| 006 / 1024 | 0.704 | 0.692 | 2.51e+04 | 2.47e+04 |
| 007 / 1024 | 0.780 | 0.784 | 2.61e+04 | 2.65e+04 |
| 008 / 1024 | 0.779 | 0.817 | 2.26e+04 | 2.54e+04 |
| 009 / 1024 | 0.696 | 0.680 | 2.19e+04 | 2.1e+04 |

Derived 8-bit: maximum global present/omitted reconstruction relative errors were 1.09e-06/1.08e-06. At layer level, 3/160 layer-context residuals were at or below their own empirical floor; global ratios do not imply that every layer exceeds the floor.

Original BF16: maximum global present/omitted reconstruction relative errors were 1.08e-06/1.07e-06. At layer level, 3/160 layer-context residuals were at or below their own empirical floor; global ratios do not imply that every layer exceeds the floor.

The numerical diagnostics distinguish a transport residual from the observed arithmetic discrepancies in these runs. They do not establish a rigorous error bound or a general quantization claim. All intervals are descriptive marginal percentile intervals from 2,000 paired whole-context resamples; they exclude training variation, alternative quantizers, model families and domain shift. This eight-context control complements the larger cohort and does not establish universal deletion or utility guarantees.

Machine-readable comparison: [precision_comparison_v1.json](precision_comparison_v1.json).

The separately pinned community-model comparison is retained as a [secondary artifact comparison](precision_cached8_comparison_v1.md); its exact master-checkpoint relationship is unverified and its differences are not labelled causal quantization effects.
