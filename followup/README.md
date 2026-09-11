# Completed follow-up evidence

This release adds the frozen synthetic follow-up experiments completed on 11 September 2026. Start with [the results and limits](RESULTS.md), [the frozen protocol](protocol.md), and [the main analysis](analysis/kimi8_analysis_v2.md). Historical evidence elsewhere in the artifact is retained.

| Evidence | Independent units | Files |
|---|---:|---|
| Main Kimi community 8-bit cohort | 40 calibration + 40 evaluation contexts; all 11 conditions | [Manifest](manifests/main_v1.json), `raw/kimi8_v2/`, [analysis](analysis/kimi8_analysis_v2.json) |
| Qwen3.5-4B-Base BF16 confirmation | 32 contexts, each with short and long suffixes | [Manifest](manifests/qwen_confirmation_20260911.json), `raw/qwen_confirmation/`, [report](analysis/qwen_report.md), [summary](analysis/qwen_summary.json) |
| Same-master precision control | 8 frozen contexts in original BF16 and an in-memory derived 8-bit model | [Subset](manifests/precision_subset_v1.json), `raw/kimi_bf16_v1/`, `raw/kimi_derived8_v1/`, [comparison](analysis/precision_comparison_v1.md) |
| Own arithmetic diagnostics | The same 8 contexts per variant, including the secondary community 8-bit variant | Three `raw/transport_floor_*/` directories |
| Amendments and sequential deletion | 12 amendment contexts; six disjoint pairs with both deletion orders and an intervening admission | [Raw measurements and embedded manifest](raw/maintenance_kimi8_v1.json), [report](analysis/maintenance_summary_v1.md) |

Sequential paths reuse the amendment targets: 12 paths and 36 transitions are not independent samples. The larger attention-free delta-rule model remains deferred. Timings overlapped other workloads and are diagnostic, not controlled performance evidence.

The main cohort retains all qualification outcomes and all raw synthetic targets, query text, token plans and generated answers. Raw `greedy_exact` is the inherited runner's substring-containment flag. The analyzer recomputes strict equality by stripping outer whitespace and casefolding the complete decoded answer and target; it reports containment separately. Retained utility covers three synthetic retained records and two continuations from a fixed six-passage bank, not a broad language benchmark.

## Offline verification and analysis

From the artifact root:

```bash
python3 -B verify_offline.py
python3 -B -m unittest test_verify_offline
```

These checks need only the standard library. They verify the complete inventory, strict JSON and source syntax, original/released provenance, frozen record coverage, recorded source fingerprints, cross-file identities and legacy summary regeneration. They do not rerun models or independently establish weight provenance.

For exact CPU-only regeneration of every new statistical summary, including bootstrap intervals and numerical tables:

```bash
python3 -m venv /tmp/kda-artifact-analysis
/tmp/kda-artifact-analysis/bin/python -m pip install -r requirements-analysis.txt
/tmp/kda-artifact-analysis/bin/python -B verify_followup_analysis.py
PYTHONPATH=code /tmp/kda-artifact-analysis/bin/python -B -m kda_followup.analyze \
  --manifest followup/manifests/main_v1.json --results followup/raw/kimi8_v2 \
  --output /tmp/kda-main-regenerated.json --markdown /tmp/kda-main-regenerated.md
```

Installing dependencies needs network access unless cached; verification itself is offline. `verify_followup_analysis.py` compares all main, Qwen, precision, arithmetic-floor and maintenance aggregates plus the exploratory AUC sensitivity and TeX table. It explicitly substitutes recorded external snapshot metadata for the precision summarizer's live model preflight. This avoids loading or accessing weights; it is not a new weight verification. The standalone analyzer reports hashes of the released raw JSON, which differ from the recorded original-file hashes only because paths were anonymized.

## Model reproduction

Use Python 3.11, the appropriate pinned requirements in the artifact root, and fresh output directories. Measurement implementations in `code/kda_followup/`, `code/kimi_sv/`, and `code/recurrent_families/qwen_confirmation.py` are the original run implementations. `PYTHONPATH=code` exposes the packages. The serialized `/models/...` strings are neutral provenance labels; supply your actual local snapshot directory to each loader.

The main snapshot is `mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit` at revision `553a7a5673491df2b2d38f7380188a9d8e9f9a66`. Original BF16 is `moonshotai/Kimi-Linear-48B-A3B-Instruct` at revision `e1df551a447157d4658b573f9a695d57658590e9`. The 20 original BF16 shards total 98,248,224,120 bytes; their recorded SHA checks are in [the original-weight audit](provenance/kimi_bf16_provenance.json). No weights, caches or credentials are distributed. Kimi experiments used an Apple M3 Ultra with 512 GiB unified memory; hardware and package versions are recorded in each run configuration.

```bash
PYTHONPATH=code python3 -m kda_followup.run \
  --manifest followup/manifests/main_v1.json --out-dir /tmp/kda-new-main \
  --model /path/to/community-8bit-snapshot
PYTHONPATH=code python3 -m kda_followup.precision inspect \
  --model /path/to/original-bf16-snapshot --expected bf16 --verify-official \
  --out /tmp/kda-new-bf16-audit.json
PYTHONPATH=code python3 -m kda_followup.run --help
PYTHONPATH=code python3 -m kda_followup.derived_precision --help
PYTHONPATH=code python3 -m kda_followup.transport_floor --help
PYTHONPATH=code python3 -m kda_followup.amendments --help
```

The eight IDs in the frozen precision subset must be reused. The derived runner uses the original verified master with `mlx_lm.utils.quantize_model`, affine 8-bit weights, group size 64, and the model predicate after supported-module/divisibility checks. It records actual packed tensors and saves no quantized weight files. Both variants use their own never-stored reference and own observed arithmetic floor. The separate [community-artifact comparison](analysis/precision_cached8_comparison_v1.md) does not establish exact original-master correspondence and cannot isolate a causal quantization effect.

Qwen uses `Qwen/Qwen3.5-4B-Base` revision `57370f0ea82c3cca33558a95212e032c344e5fd5` and the Torch requirements. Copy its frozen manifest to a fresh working file and replace only `local_model_directory` with your local snapshot path before a new run. Do not overwrite the release manifest; a new local manifest naturally has a different file SHA. Preserve all 32 text/token records and acceptance/analysis settings.

```bash
PYTHONPATH=code python3 -m recurrent_families.qwen_confirmation run \
  --manifest /tmp/qwen-local-manifest.json --outdir /tmp/qwen-new-results --device mps
```

For analysis regression tests install pytest as needed and run `PYTHONPATH=code python3 -B -m pytest code/tests/test_kda_followup_analysis.py`. Other bundled tests include tiny MLX/Torch model checks and require their respective runtimes. Random tiny models are plumbing tests, never pretrained-model evidence.

## Anonymization and provenance

[PROVENANCE.json](PROVENANCE.json) maps every added source file to its released path, original SHA-256, released SHA-256 and byte counts. JSON entries additionally carry original and released canonical hashes and every changed string pointer. Only local filesystem path metadata changed in JSON; all other values, including synthetic secrets and readbacks, are preserved. A masked-content digest checks the content outside those enumerated path leaves. Original files are not shipped, so this is a recorded transformation audit rather than independently available originals.

Embedded run/source/raw-result hashes remain the recorded pre-anonymization hashes. The verifier resolves them through the explicit map; it never asserts that redacted bytes have an original hash. Measurement source files and the main manifest are byte-identical. The precision and Qwen manifest path fields were anonymized while their original frozen hashes remain in the recorded results. TeX headers explicitly identify the original analysis hash. Historical provenance in the root `provenance.json` retains its earlier scope.

The model URLs in provenance identify public upstream models only. The author review ZIP, private working context, creator names/addresses, local cache paths, model weights and nonpublic repository URLs are excluded. The source-paper archive can reuse the aggregate files and main manifest under the same `followup/` paths.
