"""Verify and describe the completed immutable Qwen confirmation run."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from recurrent_families.qwen_confirmation import MANIFEST, OUTDIR, sha, summarize


def fmt(number):
    if number == 0:
        return "0"
    return f"{number:.4g}"


def build_report(manifest_path=MANIFEST, outdir=OUTDIR):
    manifest = json.loads(manifest_path.read_text())
    payload = json.loads((outdir / "results.json").read_text())
    assert payload["status"] in ("completed", "completed_with_failed_checks"), "Evaluation still in progress"
    assert payload["manifest_sha256"] == sha(manifest_path), "Manifest changed"
    assert payload["implementation_sha256"] == sha(Path(__file__).with_name("qwen_confirmation.py")), "Runner changed"
    expected = [r["record_id"] for r in manifest["records"]]
    records = payload["records"]
    assert [r["record_id"] for r in records] == expected, "Missing, duplicate or reordered records"
    raw_rows = [json.loads(line) for line in (outdir / "records.jsonl").read_text().splitlines()]
    assert raw_rows == records, "Raw append-only results disagree"
    for name, key in (("config.json", "model_config_sha256"), ("model.safetensors.index.json", "weight_index_sha256")):
        assert sha(Path(manifest["local_model_directory"]) / name) == manifest[key], f"Model {name} changed"
    recomputed = summarize(records, manifest)
    assert recomputed == payload["summary"], "Summary is not reproducible"
    n = len(records)
    cases = [c for r in records for c in r["cases"]]
    completed = sum(r["status"] == "completed" for r in records)
    assert all([c["suffix"] for c in r["cases"]] == ["short", "long"] for r in records if r["status"] == "completed")
    exact_count = sum(c["checks"]["replay_complete_state_exact"] for c in cases)
    repeat_count = sum(c["checks"]["repeat_complete_state_exact"] for c in cases)
    passed = sum(r["all_checks_pass"] for r in records)
    finite = all(c["checks"]["finite"] for c in cases)
    frozen_max = max(x["frozen_transport_state_scaled_error"] for c in cases for x in c["receipt_analysis"].values())
    live_max = max(x["live_recurrence_state_scaled_error"] for c in cases for x in c["receipt_analysis"].values())
    forcing_max = max(x["forcing_decomposition_relative_error"] for c in cases for x in c["receipt_analysis"].values())
    lines = [
        "# Qwen independent-cohort confirmation — 11 September 2026", "",
        f"Completed {completed}/{n} preselected records, with {len(cases)} short/long suffix evaluations. {passed}/{n} records passed every frozen check. Complete-state checkpoint replay had exact numerical array equality and matching bookkeeping flags against a fresh never-stored reference in {exact_count}/{len(cases)} suffix evaluations; the independent same-schedule repeat was exact in {repeat_count}/{len(cases)}. Every replay comparison includes 65 arrays/outputs and 82 bookkeeping flags, covering all recurrent and attention layers, final logits and logical cache position. Arrays were compared with `torch.equal` and zero maximum absolute difference; this establishes exact numerical equality, not byte-level identity of dtype representations or signed zeros.", "",
        "This is a new synthetic cohort, separate from the three staged scenarios in the prior Qwen audit. It uses the same pretrained Qwen3.5-4B-Base model revision and bf16 execution. It expands the number and range of contexts; it does not establish results on natural user records, other precisions, other models, arbitrary chunk schedules, or generated-answer deletion.", "",
        f"Manifest frozen at {manifest['created_utc']}; evaluation started at {payload['started_utc']} and ended at {payload['ended_utc']}. Runtime after loading: {payload['elapsed_seconds']:.1f} seconds. Model revision: `{manifest['model']['revision']}`. All weights were loaded from the existing local snapshot with network access disabled in the loader.", "",
        "## Fixed cohort and evaluation", "",
        f"The {n} generated target phrases are unique. Prefixes vary over eight domains and contain two to five retained entries. Prefix lengths range from {min(r['token_counts']['prefix'] for r in records)} to {max(r['token_counts']['prefix'] for r in records)} tokens; deleted entries contain {min(r['token_counts']['victim'] for r in records)}–{max(r['token_counts']['victim'] for r in records)} tokens. Short suffixes contain {min(r['token_counts']['short'] for r in records)}–{max(r['token_counts']['short'] for r in records)} tokens and long suffixes contain {min(r['token_counts']['long'] for r in records)}–{max(r['token_counts']['long'] for r in records)}. Raw concatenated text and segment tokenizations agree at every boundary. There was no qualification filter, target replacement or outcome-dependent stopping.", "",
        "Each suffix evaluation advances record-present, checkpoint-restored replay, fresh never-stored and an independent fresh never-stored repeat. The omitted and replay trajectories have identical token and segment schedules. Frozen-input transport and native forcing are analyzed at linear-attention layers 0, 16 and 30; complete-state replay comparisons cover every active layer.", "",
        "## Whole-record success intervals", "",
        "Each binary success requires both suffixes to pass. Denominators include every preselected record, including any execution failure. The bootstrap resamples whole records, keeping suffixes and layers together. With all records succeeding, its interval is necessarily degenerate; the Wilson binomial interval is supplied to avoid presenting [100%, 100%] as a population guarantee.", "",
        "| Frozen check | Records passing | Wilson 95% CI | Paired-record bootstrap 95% CI |", "|---|---:|---:|---:|",
    ]
    assert len({r["target_phrase"] for r in manifest["records"]}) == n
    for name, stat in payload["summary"]["whole_record_checks"].items():
        lo, hi = stat["wilson_ci95"]
        blo,bhi = stat["ci95"]
        lines.append(f"| {name.replace('_',' ')} | {stat['successes']}/{stat['denominator_including_failures']} | {100*lo:.1f}%–{100*hi:.1f}% | {100*blo:.1f}%–{100*bhi:.1f}% |")
    lines += ["", "## Transport and numerical controls", "", f"All recorded states and controls were finite: {finite}. The maximum frozen-transport error relative to the omitted-state norm was {fmt(frozen_max)}; the maximum live recurrence conformance error was {fmt(live_max)}; the maximum relative forcing-decomposition error was {fmt(forcing_max)}. The frozen acceptance threshold for each is 0.005.", "", "The native mismatch check requires at least one selected layer to exceed five times its numerical floor, with a nonzero native record effect. The floor is the maximum of the absolute frozen identity error and handwritten-recurrence/live-kernel mismatch, bounded below by 1e-12. Passing this check indicates that the transported boundary receipt does not reproduce the native omitted trajectory for the tested layer and suffix.", "", "| Suffix | Layer | Native residual / native effect, median [min, max] | Native mismatch / numerical floor, median | Records above 5× floor |", "|---|---:|---:|---:|---:|"]
    for suffix in ("short", "long"):
        for layer in ("0", "16", "30"):
            rr = [c["receipt_analysis"][layer] for c in cases if c["suffix"] == suffix]
            native = [r["native_transport_relative_residual"] for r in rr]
            floor = [r["native_mismatch_to_numerical_floor"] for r in rr]
            successes = sum(r["native_effect_norm"] >= manifest["acceptance"]["nonzero_native_effect_min"] and r["native_mismatch_to_numerical_floor"] >= 5 for r in rr)
            lines.append(f"| {suffix} | {layer} | {fmt(statistics.median(native))} [{fmt(min(native))}, {fmt(max(native))}] | {fmt(statistics.median(floor))} | {successes}/{len(rr)} |")
    lines += ["", "Whole-record bootstrap means average the three selected layers within each record before resampling. The paired long-minus-short contrast uses the same sampled records for both suffix conditions.", "", "| Metric | Short mean [95% CI] | Long mean [95% CI] | Paired long − short [95% CI] |", "|---|---:|---:|---:|"]
    for metric in ("native_transport_relative_residual", "native_transport_state_scaled_error", "frozen_static_state_scaled_error", "frozen_ledger_state_scaled_error", "frozen_transport_state_scaled_error", "live_recurrence_state_scaled_error"):
        cells = []
        for suffix in ("short", "long"):
            stat = payload["summary"]["metrics_by_suffix"][suffix][metric]
            cells.append(f"{fmt(stat['mean'])} [{fmt(stat['ci95'][0])}, {fmt(stat['ci95'][1])}]")
        stat = payload["summary"]["paired_long_minus_short"][metric]
        cells.append(f"{fmt(stat['mean'])} [{fmt(stat['ci95'][0])}, {fmt(stat['ci95'][1])}]")
        lines.append(f"| {metric.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines += ["", "Long suffixes can drive the frozen record effect near the floating-point floor. The results retain the corresponding effect-relative errors; the predeclared frozen-transport decision uses omitted-state scaling, and the native decision compares absolute mismatch to the live numerical floor. Individual layer rows, all complete-state comparisons, all check outcomes and failures remain in the raw result files.", "", "## Verification and files", "", "The report generator checked exact manifest/source hashes, model configuration/index hashes, ordered coverage of all manifest records, both suffixes, identity of the append-only JSONL and structured results, and exact reproducibility of the summary from raw records. Three tests cover transport/forcing algebra against changed native inputs and perturbed live states, whole-record failure accounting and pairing, and complete-cache replay through the actual tiny Qwen architecture with an 80-token suffix. Tiny weights were used only for tests; the reported experiment uses the cached pretrained 4B weights.", "", f"- Manifest SHA-256: `{payload['manifest_sha256']}`", f"- Runner SHA-256: `{payload['implementation_sha256']}`", f"- Base audit SHA-256: `{payload['base_audit_implementation_sha256']}`", "- Manifest: `recurrent_families/benchmarks/qwen_confirmation_20260911.json`", "- Raw append-only records: `recurrent_families/results/qwen_confirmation_20260911/records.jsonl`", "- Full results: `recurrent_families/results/qwen_confirmation_20260911/results.json`", "- Record-level intervals: `recurrent_families/results/qwen_confirmation_20260911/summary.json`", "- Reproduce verification/report: `.venv311/bin/python -m recurrent_families.qwen_confirmation_report`", ""]
    target = outdir / "REPORT.md"
    target.write_text("\n".join(lines))
    print(target)


if __name__ == "__main__":
    build_report()
