# Cohort reproduction

## Saved measurements

| Experiment | Units | Inputs and results |
|---|---|---|
| Kimi 8-bit | 40 calibration + 40 evaluation contexts, 11 conditions | `manifests/main_v1.json`, `raw/kimi8_v2/`, `analysis/kimi8_analysis_v2.json` |
| Qwen bf16 | 32 contexts, each with two suffixes | `manifests/qwen_confirmation_20260911.json`, `raw/qwen_confirmation/`, `analysis/qwen_summary.json` |
| Same-master precision | Eight fixed contexts per variant | `manifests/precision_subset_v1.json`, `raw/kimi_bf16_v1/`, `raw/kimi_derived8_v1/`, `analysis/precision_comparison_v1.json` |
| Arithmetic diagnostics | The same eight contexts per variant | `raw/transport_floor_*/` |
| Maintenance | 12 amendment contexts; six reused target pairs | `raw/maintenance_kimi8_v1.json.gz`, `analysis/maintenance_summary_v1.json` |

[Protocol](protocol.md) specifies qualification, scoring, calibration, paired intervals and state checks. Retain every frozen context and all qualification outcomes. Sequential paths reuse amendment targets; 36 transitions are not 36 independent samples. The raw `greedy_exact` flag means containment; analysis recomputes strict whole-answer equality from saved text. Run times overlap other workloads and are not controlled latency measurements.

## Verify and regenerate statistics

From the artifact root:

```bash
python3 -B verify_offline.py
python3 -m venv /tmp/kda-analysis
/tmp/kda-analysis/bin/python -m pip install -r requirements-analysis.txt
/tmp/kda-analysis/bin/python -B verify_followup_analysis.py
```

The first command uses the standard library; the second verifier regenerates main, Qwen, precision, arithmetic and maintenance aggregates with NumPy, including the exploratory AUC sensitivity. It uses recorded precision snapshot metadata, without reading weights. `PROVENANCE.json` maps original/released hashes and anonymized local-path fields; embedded original hashes need not match anonymized bytes. `/models/...` paths are provenance labels, so supply actual local snapshot paths for inference.

## Models and setup

Use Python 3.11, `requirements-kimi.txt` for MLX and `requirements-torch.txt` for Qwen. Kimi used MLX 0.32.0 and mlx-lm 0.31.3; tokenization/loading used Transformers 5.12.1. Runs used an Apple M3 Ultra with 512 GiB unified memory. Obtain each snapshot with its tokenizer/configuration:

| Variable | Model | Revision |
|---|---|---|
| `KDA_8BIT` | `mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit` | `553a7a5673491df2b2d38f7380188a9d8e9f9a66` |
| `KDA_BF16` | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | `e1df551a447157d4658b573f9a695d57658590e9` |
| `KDA_QWEN` | `Qwen/Qwen3.5-4B-Base` | `57370f0ea82c3cca33558a95212e032c344e5fd5` |

The original Kimi weights occupy 98,248,224,120 bytes across 20 shards. The inspection command below verifies their SHA-256 values. The paired 8-bit variant is quantized in memory from this master with affine 8-bit weights, group size 64 and the model's supported-module predicate. Each variant uses its own never-stored reference and arithmetic diagnostic. The separate community-artifact comparison in `analysis/precision_cached8_comparison_v1.json` has unverified exact master correspondence.

Run from the artifact root, using fresh output paths:

```bash
export PYTHONPATH=code
KDA_8BIT=/path/to/community-8bit-snapshot
KDA_BF16=/path/to/original-bf16-snapshot
KDA_QWEN=/path/to/qwen-snapshot
KDA_RUN=/tmp/kda-cohort-rerun
KDA_MANIFEST=followup/manifests/main_v1.json
KDA_PLAN=followup/manifests/precision_subset_v1.json
mkdir -p "$KDA_RUN"
KDA_IDS=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["record_ids"]))' "$KDA_PLAN")

python3 -m kda_followup.run --manifest "$KDA_MANIFEST" \
  --model "$KDA_8BIT" --out-dir "$KDA_RUN/main"
python3 -m kda_followup.precision inspect --model "$KDA_BF16" \
  --expected bf16 --verify-official --out "$KDA_RUN/bf16-audit.json"
python3 -m kda_followup.run --manifest "$KDA_MANIFEST" --record-ids "$KDA_IDS" \
  --model "$KDA_BF16" --out-dir "$KDA_RUN/bf16"
python3 -m kda_followup.transport_floor --manifest "$KDA_MANIFEST" --plan "$KDA_PLAN" \
  --model "$KDA_BF16" --precision bf16 --provenance "$KDA_RUN/bf16-audit.json" --out-dir "$KDA_RUN/floor-bf16"
python3 -m kda_followup.derived_precision --manifest "$KDA_MANIFEST" --plan "$KDA_PLAN" \
  --model "$KDA_BF16" --source-provenance "$KDA_RUN/bf16-audit.json" \
  --out-dir "$KDA_RUN/derived8" --out-provenance "$KDA_RUN/derived8-audit.json" --floor-out-dir "$KDA_RUN/floor-derived8"
python3 -m kda_followup.precision verify-tokenizers --plan "$KDA_PLAN" \
  --model-8bit "$KDA_BF16" --model-bf16 "$KDA_BF16" --out "$KDA_RUN/paired-tokens.json"
python3 -m kda_followup.precision summarize-pair --plan "$KDA_PLAN" \
  --results-8bit "$KDA_RUN/derived8" --results-bf16 "$KDA_RUN/bf16" \
  --token-check "$KDA_RUN/paired-tokens.json" --audit-8bit "$KDA_RUN/derived8-audit.json" \
  --audit-bf16 "$KDA_RUN/bf16-audit.json" --out "$KDA_RUN/precision.json"
python3 -m kda_followup.amendments --model "$KDA_8BIT" --targets 12 \
  --seed 202609117 --mode both --greedy-tokens 16 --out "$KDA_RUN/maintenance.json"
```

Both same-master variants use the original snapshot tokenizer; the paired summary also checks observed token plans. For the community arithmetic control, reuse the same eight-context plan:

```bash
python3 -m kda_followup.precision inspect --model "$KDA_8BIT" --expected 8bit --out "$KDA_RUN/community-audit.json"
python3 -m kda_followup.transport_floor --manifest "$KDA_MANIFEST" --plan "$KDA_PLAN" --model "$KDA_8BIT" --precision 8bit --provenance "$KDA_RUN/community-audit.json" --out-dir "$KDA_RUN/floor-community8"
```

Qwen loads its model path from the manifest. Make a working copy changing only `local_model_directory`; preserve every text, token sequence and acceptance/analysis setting:

```bash
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["local_model_directory"]=sys.argv[2]; json.dump(d,open(sys.argv[3],"w"),indent=2)' \
  followup/manifests/qwen_confirmation_20260911.json "$KDA_QWEN" "$KDA_RUN/qwen-manifest.json"
python3 -m recurrent_families.qwen_confirmation run \
  --manifest "$KDA_RUN/qwen-manifest.json" --outdir "$KDA_RUN/qwen" --device mps
python3 -m recurrent_families.qwen_confirmation summarize --manifest "$KDA_RUN/qwen-manifest.json" --outdir "$KDA_RUN/qwen"
python3 -m kda_followup.analyze --manifest "$KDA_MANIFEST" --results "$KDA_RUN/main" \
  --output "$KDA_RUN/main-analysis.json" --markdown "$KDA_RUN/main-analysis.md"
```

Use the frozen manifests rather than generating new cohorts. A local Qwen path change produces a different manifest-file hash. Numerical state equality applies to the declared active arrays and execution schedule; it is not byte-level identity or a guarantee for other contexts. No model weights or credentials are included; obtain dependencies under their own licenses.
