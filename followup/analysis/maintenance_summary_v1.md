# Kimi maintenance follow-up results

All 12 preselected amendment contexts and all 36 sequential-maintenance transitions completed. Corrected/replayed states and probe logits had zero observed maximum absolute difference from their appropriate fresh references.

Model: `/models/models--mlx-community--Kimi-Linear-48B-A3B-Instruct-8bit/snapshots/553a7a5673491df2b2d38f7380188a9d8e9f9a66`. Frozen maintenance manifest: `50edd37c5f7d060daeb976e539abffbc421cbf7ff2ef10a9f4ecd7bed01b45e7`. All 12 old-target qualification lifts exceeded one nat versus never-stored; no records were excluded. Extra neutral suffixes were 0 tokens in six contexts, 128 in three and 512 in three. Target positions varied across the four-record contexts.

| Amendment outcome | Exact answer: contexts (Wilson 95% CI) | Target containment: contexts (Wilson 95% CI) |
|---|---|---|
| Old answer before amendment | 12/12 (100.0%; 95% CI 75.8–100.0%) | 12/12 (100.0%; 95% CI 75.8–100.0%) |
| New answer before amendment | 0/12 (0.0%; 95% CI 0.0–24.2%) | 0/12 (0.0%; 95% CI 0.0–24.2%) |
| Old answer after amendment | 0/12 (0.0%; 95% CI 0.0–24.2%) | 0/12 (0.0%; 95% CI 0.0–24.2%) |
| New answer after amendment | 12/12 (100.0%; 95% CI 75.8–100.0%) | 12/12 (100.0%; 95% CI 75.8–100.0%) |
| New answer in corrected-from-start reference | 12/12 (100.0%; 95% CI 75.8–100.0%) | 12/12 (100.0%; 95% CI 75.8–100.0%) |
| All three retained answers after amendment | 11/12 (91.7%; 95% CI 64.6–98.5%) | 12/12 (100.0%; 95% CI 75.8–100.0%) |

**Scoring.** Exact means full-answer equality after stripping outer whitespace and applying Unicode casefolding; punctuation is preserved. Containment means the target appears anywhere in that normalized answer. Both are computed from the saved generated text. Decoding was greedy with a 16-token cap. The three retained queries form one context for uncertainty estimates: 35/36 answers exactly match and 36/36 contain the target; the intervals use the 12 context outcomes shown above.

| Paired change in mean target log probability (nats/token) | Estimate | Record-bootstrap 95% CI |
|---|---:|---:|
| old mean logprob after minus as heard | -6.827667 | [-7.172543, -6.446844] |
| old mean logprob after minus corrected from start | 0.000000 | [0.000000, 0.000000] |
| new mean logprob after minus as heard | 7.102466 | [6.456614, 7.827157] |
| new mean logprob after minus corrected from start | 0.000000 | [0.000000, 0.000000] |
| retained mean logprob after minus as heard | -0.001859 | [-0.008686, 0.002288] |
| retained mean logprob after minus corrected from start | 0.000000 | [0.000000, 0.000000] |

Paired intervals use 2,000 resamples of whole amendment contexts; all before/after/reference measurements remain paired, and retained scores average the three queries within each context. Zero-width intervals at zero describe deterministic agreement observed here and are not guarantees for future records.

**Sequential deletion and admission.** Six disjoint record pairs were each tested in both deletion orders. Every path deletes one target, admits a new record, then deletes the other target: 12 paths and 36 transitions. Every transition was checked against a newly built surviving-record reference; both final orders had identical surviving-record keys and saved probe scores in 6/6 groups. All-transition state/logit agreement: 6/6 (100.0%; 95% CI 61.0–100.0%). Both deleted answers absent in both final orders: 6/6 (100.0%; 95% CI 61.0–100.0%). All retained and admitted answers exact in both final orders: 5/6 (83.3%; 95% CI 43.6–97.0%).

All retained and admitted answers contained their targets in both orders in 6/6 (100.0%; 95% CI 61.0–100.0%). The independent sequential unit is the record pair (n=6), not a path or transition. These pairs reuse the amendment targets and do not enlarge the independent 12-target cohort.

**State checks and scope.** Each comparison covers all 80 KDA recurrent/convolution arrays, all 14 active MLA key/value arrays, seven MLA offsets, and one full next-token logit vector. All 48 intervention-to-reference checks and 48 repeated-reference checks agreed numerically. Array dtypes and shapes were checked; float32 maximum absolute state and logit differences were 0.0. These are numeric comparisons, not bytewise certificates. The repeated-reference control measures deterministic repeatability; transport arithmetic is assessed separately.

The evidence concerns this small, frozen synthetic cohort. Wilson intervals assume exchangeability of the stated context or pair units. Timings overlapped other experiments and a weight download, so they support no controlled latency claim.

Raw results: [maintenance_kimi8_v1.json](../raw/maintenance_kimi8_v1.json).
