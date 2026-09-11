# Qwen independent-cohort confirmation — 11 September 2026

Completed 32/32 preselected records, with 64 short/long suffix evaluations. 32/32 records passed every frozen check. Complete-state checkpoint replay had exact numerical array equality and matching bookkeeping flags against a fresh never-stored reference in 64/64 suffix evaluations; the independent same-schedule repeat was exact in 64/64. Every replay comparison includes 65 arrays/outputs and 82 bookkeeping flags, covering all recurrent and attention layers, final logits and logical cache position. Arrays were compared with `torch.equal` and zero maximum absolute difference; this establishes exact numerical equality, not byte-level identity of dtype representations or signed zeros.

This is a new synthetic cohort, separate from the three staged scenarios in the prior Qwen audit. It uses the same pretrained Qwen3.5-4B-Base model revision and bf16 execution. It expands the number and range of contexts; it does not establish results on natural user records, other precisions, other models, arbitrary chunk schedules, or generated-answer deletion.

Manifest frozen at 2026-09-11T21:24:30.171645+00:00; evaluation started at 2026-09-11T21:35:51.648353+00:00 and ended at 2026-09-11T21:41:57.294964+00:00. Runtime after loading: 360.5 seconds. Model revision: `57370f0ea82c3cca33558a95212e032c344e5fd5`. All weights were loaded from the existing local snapshot with network access disabled in the loader.

## Fixed cohort and evaluation

The 32 generated target phrases are unique. Prefixes vary over eight domains and contain two to five retained entries. Prefix lengths range from 91 to 191 tokens; deleted entries contain 37–41 tokens. Short suffixes contain 90–94 tokens and long suffixes contain 527–539. Raw concatenated text and segment tokenizations agree at every boundary. There was no qualification filter, target replacement or outcome-dependent stopping.

Each suffix evaluation advances record-present, checkpoint-restored replay, fresh never-stored and an independent fresh never-stored repeat. The omitted and replay trajectories have identical token and segment schedules. Frozen-input transport and native forcing are analyzed at linear-attention layers 0, 16 and 30; complete-state replay comparisons cover every active layer.

## Whole-record success intervals

Each binary success requires both suffixes to pass. Denominators include every preselected record, including any execution failure. The bootstrap resamples whole records, keeping suffixes and layers together. With all records succeeding, its interval is necessarily degenerate; the Wilson binomial interval is supplied to avoid presenting [100%, 100%] as a population guarantee.

| Frozen check | Records passing | Wilson 95% CI | Paired-record bootstrap 95% CI |
|---|---:|---:|---:|
| finite | 32/32 | 89.3%–100.0% | 100.0%–100.0% |
| forcing decomposition closes | 32/32 | 89.3%–100.0% | 100.0%–100.0% |
| frozen transport closes | 32/32 | 89.3%–100.0% | 100.0%–100.0% |
| live recurrence conforms | 32/32 | 89.3%–100.0% | 100.0%–100.0% |
| native transport mismatch above floor | 32/32 | 89.3%–100.0% | 100.0%–100.0% |
| repeat complete state exact | 32/32 | 89.3%–100.0% | 100.0%–100.0% |
| replay complete state exact | 32/32 | 89.3%–100.0% | 100.0%–100.0% |

## Transport and numerical controls

All recorded states and controls were finite: True. The maximum frozen-transport error relative to the omitted-state norm was 1.047e-06; the maximum live recurrence conformance error was 1.19e-06; the maximum relative forcing-decomposition error was 5.935e-06. The frozen acceptance threshold for each is 0.005.

The native mismatch check requires at least one selected layer to exceed five times its numerical floor, with a nonzero native record effect. The floor is the maximum of the absolute frozen identity error and handwritten-recurrence/live-kernel mismatch, bounded below by 1e-12. Passing this check indicates that the transported boundary receipt does not reproduce the native omitted trajectory for the tested layer and suffix.

| Suffix | Layer | Native residual / native effect, median [min, max] | Native mismatch / numerical floor, median | Records above 5× floor |
|---|---:|---:|---:|---:|
| short | 0 | 0.01511 [0.01275, 0.01807] | 2723 | 32/32 |
| short | 16 | 0.8658 [0.7991, 0.904] | 4.119e+05 | 32/32 |
| short | 30 | 0.6808 [0.6113, 0.7953] | 2.535e+05 | 32/32 |
| long | 0 | 0.00845 [0.005714, 0.01212] | 432.7 | 32/32 |
| long | 16 | 0.8761 [0.847, 0.9155] | 1.484e+05 | 32/32 |
| long | 30 | 0.849 [0.7849, 0.9137] | 1.415e+05 | 32/32 |

Whole-record bootstrap means average the three selected layers within each record before resampling. The paired long-minus-short contrast uses the same sampled records for both suffix conditions.

| Metric | Short mean [95% CI] | Long mean [95% CI] | Paired long − short [95% CI] |
|---|---:|---:|---:|
| native transport relative residual | 0.5222 [0.5151, 0.5296] | 0.5803 [0.576, 0.5847] | 0.0581 [0.0506, 0.06545] |
| native transport state scaled error | 0.09457 [0.0914, 0.09779] | 0.08785 [0.08557, 0.09027] | -0.006725 [-0.01039, -0.003195] |
| frozen static state scaled error | 0.2884 [0.285, 0.2917] | 0.3059 [0.3022, 0.3093] | 0.01744 [0.01606, 0.01875] |
| frozen ledger state scaled error | 0.1032 [0.1019, 0.1045] | 0.0892 [0.08814, 0.09026] | -0.01402 [-0.01484, -0.01316] |
| frozen transport state scaled error | 4.138e-07 [4.107e-07, 4.171e-07] | 8.275e-07 [8.202e-07, 8.351e-07] | 4.137e-07 [4.062e-07, 4.215e-07] |
| live recurrence state scaled error | 4.685e-07 [4.558e-07, 4.813e-07] | 8.918e-07 [8.799e-07, 9.038e-07] | 4.233e-07 [4.052e-07, 4.418e-07] |

Long suffixes can drive the frozen record effect near the floating-point floor. The results retain the corresponding effect-relative errors; the predeclared frozen-transport decision uses omitted-state scaling, and the native decision compares absolute mismatch to the live numerical floor. Individual layer rows, all complete-state comparisons, all check outcomes and failures remain in the raw result files.

## Verification and files

The report generator checked exact manifest/source hashes, model configuration/index hashes, ordered coverage of all manifest records, both suffixes, identity of the append-only JSONL and structured results, and exact reproducibility of the summary from raw records. Three tests cover transport/forcing algebra against changed native inputs and perturbed live states, whole-record failure accounting and pairing, and complete-cache replay through the actual tiny Qwen architecture with an 80-token suffix. Tiny weights were used only for tests; the reported experiment uses the cached pretrained 4B weights.

- Manifest SHA-256: `fcba8ea06e8d7106b3e6b09d9f0cc12f2123552f50a418a80e8eb53a14bb39c2`
- Runner SHA-256: `ff66b0bdf1d00ce727875dea362ed4b66e09ed1b6a7a8522ed08dcef4c19a8c5`
- Base audit SHA-256: `fe2990438a0b80dae5f0d45746f132cb0471b024b3b4ced6666a1af0236d8f80`
- Manifest: `followup/manifests/qwen_confirmation_20260911.json`
- Raw append-only records: `followup/raw/qwen_confirmation/records.jsonl`
- Full results: `followup/raw/qwen_confirmation/results.json`
- Record-level intervals: `followup/analysis/qwen_summary.json`
- Reproduce verification/report: `.venv311/bin/python -m recurrent_families.qwen_confirmation_report`
