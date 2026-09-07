# Anonymous KDA submission artifact

This supplement accompanies *Exact Record Omission in Delta Attention: A
Transport Criterion, Its Cost, and a Replay Certificate*, prepared for SaTML
2027. It contains the numerical evidence and selected reproduction code. The
manuscript and its figure sources are in the separate paper source package.

## Start here: offline verification

Run from this directory with Python 3.11 or later:

```bash
python3 -B verify_offline.py
```

This needs only the Python standard library. It verifies the file inventory,
SHA-256 checksums, Python syntax, JSON validity, the recorded Qwen manifest and
implementation fingerprints, and regeneration of all three sweep/sampling/family
summary JSONs from the 116 shipped configuration reports. It also regenerates
the exact numerical TeX macros supplied to the paper. It performs no inference,
network access, package installation, credential access, or clinical-data read.
These checks validate the packaged records and aggregation; they do not repeat
the original measurements or establish their correctness independently.

## Contents and evidence boundaries

| Location | Contents |
| --- | --- |
| `benchmarks/` | Eleven benchmark/protocol JSONs, including family predictions and the hybrid-channel diagnostic |
| `artifacts/kimi_sv/` | Three reviewed aggregate replay reports and the full source-free transport report |
| `reports/kimi_forcing_sweep/` | 40 reports: 24 synthetic and 16 TOFU configurations |
| `reports/kimi_masked_sampling/` | 40 reports: 24 synthetic and 16 TOFU configurations |
| `reports/recurrent_families/` | 36 reports: 12 each for Mamba-2, Falcon-H1 and RWKV-7 |
| `code/` | Kimi, recurrent-family and Qwen audit code; shared clinical loaders and disclosure checks; minimal MLX gate dependency; existing focused tests |
| `expected/` | Reference numerical TeX macros for offline comparison |
| `provenance.json` | Original and released hashes and the treatment of each copied file |
| `MANIFEST.json` | Complete file inventory, sizes and SHA-256 hashes, excluding the manifest itself |

The 13 original release-evidence records, full transport report and hybrid
diagnostic are copied verbatim. Every numeric value, boolean, null value and
measurement in the 116 configuration reports is unchanged. Exactly 3,760
`greedy` string fields were removed from the 40 forcing reports to create a
source-free export. Those fields contained synthetic/TOFU generated readbacks
and are not consumed by the summary code; `greedy_exact` boolean measurements
are preserved. The other 76 reports are copied verbatim.

The Qwen protocol contains explicitly invented audit dialogue. The amendment
example contains invented medical statements, and the original fixed-position
record retains reviewed synthetic/TOFU readbacks. Those are the only intended
text-bearing evidence exceptions. No real patient text, clinical source
identifiers, user conversations, credentials, model weights, caches, or
checkpoints are included. The `patient-*` labels in synthetic reports are
generated example keys, not clinical identifiers. Clinical replay records contain
only reviewed aggregates and anonymous positions.

Historical report paths recorded within evidence identify provenance; a path
such as `outputs/kimi_sv/decay_contrast_8bit.json` is not a bundled file. The
hybrid diagnostic is included, but that historical full report and the full
Qwen live report are not. Their recorded hashes are preserved.

Selected measurement implementation modules are unchanged. The quantitative
figure renderer has release-only label, accessibility-description and comment
corrections: norm-ratio terminology, a linear reference, and shorter labels
that fit within the unchanged axes. Its computations and plotted values are
unchanged. `gemma_sv/__init__.py` and
`svattn/__init__.py` are minimal packaging initializers so this subset does not
import unrelated application code. The exact Qwen audit implementation hash
still matches its measured result. No model, data, or dependency licenses are
replaced by this review bundle; obtain third-party components under their own
terms. A public software license is not assigned by this packaging step.

## Optional figure regeneration

Install `requirements-figures.txt`, then run from this directory:

```bash
PYTHONPATH=code python -B -m kimi_sv.make_forcing_sweep_figures \
  --sweep benchmarks/kda_forcing_sweep_summary_v1.json \
  --families benchmarks/recurrent_families_summary_v1.json \
  --out-dir regenerated_figures
```

This recreates the three quantitative figures without models or data. The other
five paper figures are authored SVGs in the paper source package; this script
does not recreate them.

## Optional model reruns

These are separate from the verified offline path. No model reruns were
performed for submission packaging. Use a separate environment with Python
3.11. Kimi requires Apple silicon with MLX and substantial unified memory for
the 48B 8-bit checkpoint plus activations and caches. The shipped sweep code
sets a default process limit of 80 GiB; this is a limit, not a verified minimum
memory requirement. Original reports record macOS 26.5/arm64, but do not record
a complete machine specification or dependency lock for every run. Timing
reproduction therefore depends on hardware, OS, kernels and workload.

Install `requirements-kimi.txt` for Kimi and `requirements-torch.txt` for Qwen
or the other recurrent families. The family harness currently imports the
synthetic case generator from `kimi_sv.protocol`, so its unmodified import
path also requires the Kimi dependencies. The recorded family and Qwen device
is MPS. The pins reflect the available packaging environment; only Qwen's
Transformers 5.12.1 requirement is explicitly bound by its protocol. They are
not a reconstructed lock for all historical runs. Model and tokenizer loads
may download external files; Kimi and RWKV loaders enable repository code.

The Kimi transport evidence records revision
`553a7a5673491df2b2d38f7380188a9d8e9f9a66` of
`mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit`. The general Kimi runner
defaults to a model ID rather than a revision; to target this recorded revision,
obtain that snapshot separately and pass its directory through `--model`.
Other Kimi evidence and the three family reports do not establish full model
revision locks. Qwen's model revision is frozen in `benchmarks/qwen35_replay_v3.json`.

Run commands from this directory with `PYTHONPATH=code`. These examples write
new results under `rerun/`, preserving submitted evidence:

```bash
mkdir -p rerun
PYTHONPATH=code python -B -m kimi_sv.eval_separability \
  --source synthetic --out rerun/kda_separability.json
PYTHONPATH=code python -B -m kimi_sv.eval_oracle \
  --source synthetic --records 16 --victims 1 8 14 \
  --mechanism native --memory-limit-gb 75 --out rerun/oracle_synthetic.json
PYTHONPATH=code python -B -m kimi_sv.amendment_demo \
  --mechanism native --out rerun/amendment.json
PYTHONPATH=code python -B -m kimi_sv.eval_forcing_sweep \
  --source synthetic --n-cases 192 --victims 0,1,2,3,4,5 \
  --prefix-depths 0,4,16,64 --no-greedy --out-dir rerun/forcing
PYTHONPATH=code python -B -m kimi_sv.eval_masked_sampling \
  --source synthetic --n-cases 192 --victims 0,1,2,3,4,5 \
  --prefix-depths 0,4,16,64 --cuts 0,4096 --samples 100 \
  --temperature 1 --top-p 1 --seed 0 --out-dir rerun/sampling
PYTHONPATH=code python -B -m recurrent_families.audit \
  --family mamba2 --device mps --n-cases 192 --victims 0,1,2,3 \
  --prefix-depths 0,4,16 --cuts 0,16,64,256,1024 --out-dir rerun/families
PYTHONPATH=code python -B -m gemma_sv.qwen_replay_audit \
  --manifest benchmarks/qwen35_replay_v3.json --device mps \
  --out rerun/qwen_audit_v3.json
PYTHONPATH=code python -B -m gemma_sv.publish_qwen_replay \
  --manifest benchmarks/qwen35_replay_v3.json --report rerun/qwen_audit_v3.json \
  --out rerun/qwen_result_v3.json
```

For TOFU sweep/sampling use `--source tofu --victims 0,1,2,3,4,5,6,7
--prefix-depths 0,16`; `locuslab/TOFU`, `forget10`, train split is downloaded
by the loader. For the other family runs substitute `--family falcon_h1` or
`--family rwkv7`. Inspect each shipped report's protocol for the measured
configuration. `--no-greedy` omits decoding from the example sweep command,
so it will not regenerate `greedy_exact` statistics; remove that flag if those
statistics are required, and keep resulting text-bearing reports local.

After dependencies are available, existing implementation tests may be run
from `code/` with `python -m pytest tests -q`. Several instantiate tiny random
models and exercise accelerator kernels. Only the five data-free tests in
`test_publish_qwen_replay.py` were executed during packaging; all five passed.
The model tests were included but were not executed. They do not rerun
pretrained results.

## Controlled-data reproduction

MIMIC-IV-Ext-CDS 1.0.2 and MIMIC-IV-Note require independently authorized local
access. They are not obtainable from this bundle. The `mimic/` loaders preserve
the original selection, qualification and disclosure policy. Configure
`MIMIC_EXT_CDS_DIR` or `MIMIC_NOTE_DIR` locally and use the appropriate
`--source cds` or `--source notes`. The replay aggregates record their own
selection and record counts. The fixed-position CDS run used 16 records with
victims 1, 8, 14; the note run used 8 records with victims 1, 4, 6. Always use
`--mechanism native` to reproduce the reported masking control. Keep raw
clinical files and text-bearing reports outside this submission folder.

The offline artifact can be fully checked without those datasets, but the
clinical measurements cannot be independently rerun from the supplied files
alone. Missing full historical dependency and model-revision locks also limit
claims of bitwise reproduction outside the declared execution schedule.
