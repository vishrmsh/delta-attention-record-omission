# Prospective synthetic KDA follow-up protocol

The manifest is generated and hashed before model evaluation. The default file
contains 40 calibration contexts and 40 evaluation contexts, all newly generated
for this study. None is selected from the original pilot outputs. These are
independent context constructions evaluated on one fixed model, not independent
models or people. The scope of any interval is this synthetic-context design.

The frozen default SHA-256 is
`2003aaddfece115196f8f2a6d457d90004acc26484e6645e6664807df488eda5`.
Every result carries this hash, a model identifier, a tiny-model flag, its record
identifier, and its completion status. Run configuration records the runtime and
runner source hash. The manifest loader rejects changed content. An output file
is never overwritten. Results from different model identifiers or tiny/real runs
must be analyzed separately.

## Context construction and paired conditions

The six domains are shipping, archives, maintenance, laboratories, venues, and
warehouses. Each context has a domain-specific preamble, zero to two neutral
prefix entries, three retained records with distinct assignment codes, one target
record, and a neutral suffix. The retained records precede the target. Suffixes
are 128 or 1,024 model tokens, truncated from frozen text without using model
outcomes. Generated codes and identifiers are unique across both splits. Every
context is initialized independently; records do not accumulate across cases.

The eleven paired conditions are record-present, native attention masking,
static receipt subtraction, diagonal decay ledger, receipt transport, masking
plus transport, checkpoint replay, never-stored, independently rebuilt
never-stored, prompt forgetting, and a neutral instruction. Correction conditions
use the original paper's recurrence and attention-mask implementation. Only the
recurrent state is replaced for static, ledger, and transport interventions;
other surviving state is deliberately retained and that limitation must remain
explicit in the report.

Prompt forgetting appends the fixed manifest instruction identifying the record
by its key, without repeating its secret. The neutral instruction is tokenized,
repeated if necessary, and truncated to exactly the same extra token count.
The actual token sequences are retained in each result. All conditions share the
same probes and decoding. The forgetting-versus-neutral paired contrast measures
the instruction effect at a matched extra-token budget. Comparisons with the
unextended conditions include the instruction's context-length cost.
Repetition and truncation can leave the neutral control ending in a sentence
fragment. The paired contrast is specific to these frozen instruction strings;
it does not isolate semantics independently of wording or establish robustness
across alternative forgetting and neutral prompts.

## Qualification and behavioral utility

The direct target probe qualifies a context when record-present has at least
0.05 nats/token lift over never-stored and first-token rank at most ten, measured
after the selected suffix. **Qualification never controls whether a case is
evaluated or included in the primary analysis.** All 40 evaluation contexts are
the primary denominator; qualified-only estimates are a secondary analysis with
explicit selection conditioning. Failures remain visible and are not replaced.
This follow-up admission measurement is at the final suffix, which differs from
the original sweep's smallest probed cut; do not silently pool admission rates.

Three frozen target formulations receive teacher-forced scores. The first also
receives greedy generation. Each of the three retained records receives its own
teacher-forced probe and greedy generation. Greedy decoding has a fixed maximum
of 24 generated tokens. Report both target-string containment and normalized
whole-answer equality, each after stripping whitespace and casefolding. The raw
runner inherited the label `greedy_exact` for containment; analysis recomputes
the distinctly named metrics from saved generation text without modifying raw
results.

Two frozen prose continuation tasks per context assess held-out language
continuation. Their target sentences and prompts contain neither the deleted
record's identifier nor its assignment code. They are not ingested beforehand.
They come from a small shared bank of generic operational passages; therefore
the result is a narrow continuation-utility check rather than a broad language
benchmark. Sum their negative target log probabilities and divide by total
target tokens within each context. Give contexts equal weight. Aggregate
perplexity is the exponential of this mean context-level NLL; paired perplexity
ratios exponentiate the paired NLL difference.

## Fixed finite-query distinguisher

The attacker knows a candidate record identifier and candidate assignment code.
Access is limited to teacher-forced log probabilities of that candidate under
the three fixed target probes: a budget of three candidate-scoring calls per
challenged condition/context. This is a scoring-API threat model, not a
generation-only API or an arbitrary adaptive attacker. State arrays, retained
probes, greedy generations, and held-out continuations are not attacker inputs.
One call returns the candidate span's score; the budget does not mean three
generated tokens. The attacker score is the arithmetic mean of the three
per-probe mean target-token log probabilities. Larger scores mean membership.
No score weights, prompts, directions, or query counts are selected using
evaluation results.

Only calibration never-stored scores determine a shared threshold. For nominal
false-positive level alpha = 0.05 and m calibration records, choose the
`ceil((m+1)*(1-alpha))`th ordered score and classify a strict exceedance as
positive. If that order exceeds m, the threshold is positive infinity and makes
no positive calls. Ties are conservative. The conformal order-statistic rule
controls a marginal tail under exchangeable sampling; this balanced synthetic
design does not establish a universal deployment false-positive guarantee.
The finite-sample rank correction follows the split-conformal construction
described by [Angelopoulos and Bates, Section 1.1](https://arxiv.org/html/2107.07511v5#S1.SS1);
its marginal interpretation does not guarantee an exact conditional FPR for
the one realized calibration sample.
With 40 calibration records, one calibration score can strictly exceed the
selected threshold, so the empirical calibration rate is at most 2.5%. Report
the actual evaluation FPR instead of calling it guaranteed 5%.

Each tested condition supplies positive-class candidate scores. Matched
never-stored contexts supply negative-class scores. Report AUC with half credit
for ties, TPR, measured FPR, and TPR minus FPR. Do not flip an AUC below 0.5 after
seeing the evaluation results. Replay and independent never-stored rebuilding
are measured null controls. If their scores exactly equal never-stored, their
AUC is 0.5 and TPR equals FPR; that is a null finding for this fixed attack, not
a general indistinguishability theorem.

The 40-record evaluation has 2.5 percentage-point resolution for event rates.
Confidence intervals near 5% will be wide. A zero observed rate is not evidence
of a zero population rate: report the supplementary Wilson 95% binomial interval
alongside bootstrap intervals, whose endpoints can degenerate at zero or one.
Qualification and measured complete-state equality rates receive the same
Wilson interval diagnostic, so even an all-passing cohort does not imply
population-wide success.

## Intervals, complete-state audits, and missingness

Use 2,000 bootstrap replicates, the seed frozen in the manifest, and 95%
percentile intervals. Resample whole contexts with all condition and probe
measurements attached. Never resample probes, tokens, layers, or positive and
negative classes as independent observations. Report condition means, paired
condition-minus-never differences, and paired prompt-forget-minus-neutral
differences. Short/long suffix and qualified-only groups are secondary. These
are marginal descriptive intervals without multiplicity correction.

Attack AUC uses the paired evaluation-context bootstrap. Detection rates have
two intervals: one conditional on the selected threshold and a two-stage
interval that independently resamples calibration contexts, recalibrates the
threshold, and resamples evaluation contexts with paired condition scores.
This includes finite calibration uncertainty without leaking evaluation data
into the threshold. Bootstrap uncertainty does not cover model-training
variation, domain shift, or a search over stronger attacks.

The runner compares replay and independent rebuilding against never-stored
using every active recurrent/convolution array, every logical attention key and
value array, all attention offsets, and a next-token logit vector. Physical
unused attention buffer capacity is excluded. Report equality and maximum
absolute differences for all completed contexts and the evaluation subset
separately. State-correction residuals are measured directly against the actual
never-stored state, normalized by the original present-minus-never difference.
Numerical equality is an empirical check at this runtime and dtype.

An execution error is different from a qualification failure. The primary
analysis refuses missing or failed result files. `--allow-partial` is for an
explicitly labelled incomplete progress report; it lists every missing and
failed identifier and does not claim that a completed subset is the prospective
full cohort. Resume or investigate failed records without replacing them.
Tiny random-model runs are plumbing checks, not paper evidence. Mixed models or
quantizations are never pooled.

## Commands

```sh
.venv311/bin/python -m kda_followup.manifest --output followup/manifests/main_v1.json
.venv311/bin/python -m kda_followup.analyze --manifest followup/manifests/main_v1.json --results PATH_TO_ONE_MODEL_RESULTS --output PATH_TO_SUMMARY_JSON --markdown PATH_TO_SUMMARY_MD
```

The analysis command requires the complete cohort by default. Detailed raw
scores, state arrays' difference summaries, token costs, per-record errors, and
the frozen inputs remain available for audit even when a result does not support
the paper's hoped-for conclusion.

## Execution history

The initial 8-bit run in `outputs/kda_followup/kimi8_v1` stalled reading weights
from an external disk and was terminated before it produced any record result.
Its configuration is preserved. The restart in `kimi8_v2` uses the same pinned
model revision (`553a7a5673491df2b2d38f7380188a9d8e9f9a66`) already cached on the
internal SSD, the same frozen manifest, and the unchanged runner source. The
interrupted no-result attempt is an infrastructure failure, not a qualification
failure and not an extra observation in the cohort. The completed-record and
qualification denominators are determined only from the v2 result files.

After the complete primary analysis, `auc_sensitivity.py` was added as an
explicitly exploratory check. Ordinary empirical AUC includes the matched
positive/negative pair for each source context. The sensitivity calculation
instead averages comparisons over distinct source contexts and excludes
same-source pairs even when a bootstrap sample duplicates a context. It
quantifies the finite-cohort contribution of matched pairs; it does not replace
the frozen primary AUC, attack score, threshold, or intervals.
