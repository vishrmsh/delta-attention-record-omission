# Synthetic KDA cohort protocol

## Frozen inputs

`manifests/main_v1.json` fixes 40 calibration and 40 evaluation contexts before model evaluation. Its canonical payload SHA-256, excluding the `manifest_sha256` field, is `2003aaddfece115196f8f2a6d457d90004acc26484e6645e6664807df488eda5`. Canonicalization uses sorted keys, compact JSON separators and UTF-8. The loader rejects changed content; results record this hash, model identifier, record identifier, tiny-model flag and completion status. Run configurations record runtime and runner hashes. Preserve the manifest and use a fresh output directory.

Each context is initialized independently on one fixed model. Six synthetic administrative domains supply a preamble, zero to two neutral prefix entries, three retained records, one target record and a neutral suffix of 128 or 1,024 model tokens. Retained records precede the target. Suffix text is truncated without consulting model outcomes. Identifiers and codes are unique across both splits. These units are context constructions, not independent models or people; short/long groups contain different contexts.

## Eleven paired conditions

Record-present; native attention masking; static receipt subtraction; diagonal decay ledger; receipt transport; masking plus transport; checkpoint replay; never-stored; independently rebuilt never-stored; prompt forgetting; neutral instruction.

Static, ledger and transport interventions replace recurrent matrices only. Other state components remain record-present. Replay restores the checkpoint before the target and processes the surviving tokens with the reference's segment schedule. Each probe starts from the same saved condition state.

The fixed forgetting instruction names the target key without repeating its code. The fixed neutral instruction is tokenized, repeated if necessary and truncated to the same extra token count. Save both token sequences. All conditions share probes and decoding. Forgetting minus neutral is a paired contrast at a matched token budget; contrasts with unextended conditions include the extra context length. The neutral string may end in a sentence fragment, so this contrast concerns the specified strings.

## Qualification and scoring

A context qualifies when the direct target probe after its selected suffix has record-present lift of at least 0.05 nats/token over never-stored and first-token rank at most ten. Qualification never removes a context from primary analysis. All 40 evaluation contexts form the primary denominator; qualified-only results are secondary. Keep failures visible and do not replace them. The suffix sweep measures admission at a different cut, so its rates are separate.

Three fixed target formulations receive teacher-forced scores, averaged as three mean target-token log probabilities. The first also receives greedy generation. Each of three retained records receives teacher-forced scoring and greedy generation. Decoding is greedy with a 24-token cap. Report containment and whole-answer equality separately after outer-whitespace stripping and casefolding; punctuation remains significant. The runner's `greedy_exact` field denotes containment. The analyzer recomputes both metrics from saved text without modifying raw results.

Two held-out continuation tasks per context come from six generic passages. Neither prompt nor target contains the deleted identifier or code, and the passages are not ingested beforehand. Sum negative target log probabilities and divide by total target tokens within each context, then give contexts equal weight. Perplexity exponentiates the mean context-level NLL; paired perplexity ratios exponentiate paired NLL differences. This measures the fixed passage bank.

## Fixed three-query attack

The attacker knows the candidate record identifier and code and receives only that candidate span's teacher-forced log probabilities under the three fixed target prompts. The budget is three scoring calls per challenged condition/context. State arrays, evaluation never-stored references, retained probes, generations and continuation scores are unavailable to the attacker. Average the three per-probe mean target-token scores, with larger values indicating membership. Fix prompts, weights, direction and budget before evaluation.

Only never-stored calibration scores determine the shared threshold. With nominal alpha = 0.05 and m calibration contexts, select the `ceil((m+1)*(1-alpha))`th score in ascending order; classify strict exceedance as positive. If the selected order exceeds m, use positive infinity. With m = 40 the order is 39, so at most one calibration score strictly exceeds it. Report the measured evaluation FPR separately from the nominal level; the realized calibration sample does not guarantee an exact deployment FPR.

Each condition supplies positive scores and matched never-stored contexts supply negative scores. AUC gives half credit to ties; never flip its direction after observing results. Report AUC, TPR, measured FPR and paired TPR minus FPR. Replay and independent rebuilding are null controls: identical scores imply measured AUC 0.5 and TPR equal to FPR for this attack. Use Wilson 95% intervals for event rates, qualification and complete-state equality, including all-success or all-failure observations.

The separate exploratory AUC sensitivity excludes comparisons from the same source context, including duplicate copies of that source in a bootstrap sample. It does not replace the primary AUC, score, threshold or interval rule.

## Paired uncertainty and state audit

Use 2,000 bootstrap replicates, the manifest's frozen seed and percentile 95% intervals. Resample whole contexts with all conditions and probes attached. Report means, paired condition-minus-never differences and paired forgetting-minus-neutral differences. Probes, tokens, layers and the matched positive/negative classes are not independent observations. Short/long and qualified-only groups are secondary; intervals are marginal and have no multiplicity correction.

AUC intervals resample paired evaluation contexts. Detection intervals include both a calculation conditional on the deployed threshold and a two-stage calculation that independently resamples calibration contexts, recalibrates the threshold and resamples paired evaluation contexts. This includes calibration uncertainty while keeping evaluation outcomes out of threshold selection.

Compare replay and independent rebuilding against never-stored on all 80 recurrent/convolution arrays, 14 logical MLA key/value arrays, seven offsets and one full next-token audit logit vector. Check shapes and float32 maximum absolute differences; compare offsets directly. Exclude unused attention-buffer capacity. Report all completed contexts and the evaluation subset separately. Normalize recurrent correction residuals by the original present-minus-never difference. Equality is an empirical numerical check, not a byte-level comparison.

## Completion and analysis

The primary analyzer rejects missing or failed result files. `--allow-partial` produces only an explicitly incomplete report listing missing and failed identifiers. Investigate or resume without changing the cohort. Execution errors and qualification failures are separate. Never pool different model identifiers, quantizations or tiny/real runs; tiny models test plumbing only.

From the artifact root, after installing `requirements-analysis.txt`:

```bash
PYTHONPATH=code python3 -m kda_followup.analyze --manifest followup/manifests/main_v1.json --results PATH_TO_ONE_MODEL_RESULTS --output PATH_TO_SUMMARY_JSON --markdown PATH_TO_SUMMARY_MD
```

The raw records preserve scores, token plans, generated answers, state-difference summaries and per-record errors needed to audit these rules.
