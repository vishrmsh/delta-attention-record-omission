"""Human-readable matched precision evidence with each variant's own floor."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def attach_floors(comparison: dict[str, Any], floor_8bit: Path, floor_bf16: Path) -> dict[str, Any]:
    ids = comparison["record_ids"]
    attached = {}
    for precision, directory in (("8bit", floor_8bit), ("bf16", floor_bf16)):
        config = json.loads((directory / "run_config.json").read_text())
        if config["tiny_plumbing_only"] or config["plan_sha256"] != comparison["plan_sha256"]:
            raise ValueError("arithmetic floor run is tiny or differs from the frozen precision subset")
        if config["precision"] != precision or set(config["record_ids"]) != set(ids):
            raise ValueError("arithmetic floor precision or record inventory differs")
        if precision == "8bit" and comparison["8bit_variant"] == "derived_in_memory_quantization" and config.get("model_variant") != "derived_in_memory_quantization":
            raise ValueError("derived 8-bit comparison requires its own derived 8-bit arithmetic floor")
        floor_summary = json.loads((directory / "summary.json").read_text())
        if floor_summary["status"] != "complete" or floor_summary["n_records"] != len(ids):
            raise ValueError("arithmetic floor cohort is incomplete")
        records = []
        for record_id, core_ratio in zip(ids, comparison["by_precision"][precision]["transport_residual_relative_to_present"]):
            row = json.loads((directory / f"{record_id}.json").read_text())
            if row["status"] != "complete" or row["record_id"] != record_id:
                raise ValueError("arithmetic floor record is missing or incomplete")
            diagnostic_ratio = row["global"]["native_residual_relative_to_present_difference"]
            if not math.isclose(core_ratio, diagnostic_ratio, abs_tol=1e-5, rel_tol=1e-5):
                raise ValueError(f"core and companion transport residuals disagree for {precision}/{record_id}: {core_ratio} vs {diagnostic_ratio}")
            ratios = [layer["residual_to_max_empirical_floor_ratio"] for layer in row["per_layer"]]
            records.append({"record_id": record_id, "suffix_class": row["suffix_class"], "suffix_tokens": row["suffix_tokens"],
                            "global": row["global"], "max_abs_errors": row["max_abs_errors"],
                            "layer_count": len(ratios), "layer_residuals_at_or_below_observed_floor": sum(r is not None and r <= 1 for r in ratios),
                            "layer_residuals_above_ten_times_observed_floor": sum(r is not None and r > 10 for r in ratios)})
        attached[precision] = {"directory": str(directory.absolute()), "summary": floor_summary, "records": records,
                               "core_and_diagnostic_transport_residuals_agree": True}
    comparison["arithmetic_diagnostics"] = attached
    comparison["arithmetic_diagnostics_definition"] = (
        "Independently reconstruct native present/omitted recurrent states from each branch's captured k,v,g,beta in serial fp32. "
        "Also propagate two initial states through identical present inputs/writes and compare their difference to separately transported receipt. "
        "The reported empirical floor is max(sum of native-versus-serial endpoint reconstruction norms, frozen-input identity-error norm). "
        "It is an observed numerical diagnostic, not a rigorous bound or security guarantee.")
    comparison["status"] = "complete_with_own_arithmetic_diagnostics"
    return comparison


def _interval(item: dict[str, Any], digits: int = 3) -> str:
    return f"{item['estimate']:.{digits}f} [{item['ci_low']:.{digits}f}, {item['ci_high']:.{digits}f}]"


def _interval_range(values: list[float], digits: int = 3) -> str:
    return f"{min(values):.{digits}g}–{max(values):.{digits}g}"


def markdown(result: dict[str, Any], summary_path: Path, secondary: Path | None = None) -> str:
    derived = result["8bit_variant"] == "derived_in_memory_quantization"
    variant = "Derived 8-bit" if derived else "Released community 8-bit"
    n = result["n_records"]
    p = result["by_precision"]
    floor = result["arithmetic_diagnostics"]
    equal8 = p["8bit"]["replay_complete_state_and_logits_equal"]
    equalbf = p["bf16"]["replay_complete_state_and_logits_equal"]
    lines = ["# Kimi matched precision control", "",
             f"Replay matched its own never-stored state and audit logits in {equal8}/{n} {variant.lower()} contexts and {equalbf}/{n} original-BF16 contexts. Transport residuals and their observed arithmetic floors are reported separately for each precision below.", "",
             ("The primary comparison starts from one SHA-verified original BF16 checkpoint. Its paired 8-bit variant was quantized in memory with MLX affine weight quantization, 8 bits, group size 64, and the Kimi model's quantization predicate through the supported-module/divisibility wrapper. Quantization preceded attention-mask installation; no quantized weight files were saved. The same main evaluator and token plans were used. The intervention is this specified quantization recipe and its runtime kernels, not an abstract precision-only change."
              if derived else "This is a secondary comparison of separately pinned released artifacts. The community 8-bit card names the same base repository but does not pin the original master revision. Differences cannot be attributed causally to quantization alone."), "",
             "The original BF16 master is [Moonshot AI Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/tree/e1df551a447157d4658b573f9a695d57658590e9), revision `e1df551a447157d4658b573f9a695d57658590e9`; all 20 original weight-shard SHA256 values were verified. Its 98,248,224,120 bytes of original weight files contain BF16 weights plus FP32 parameters. This is not a dequantized 8-bit model.", "",
             f"Eight contexts were selected before reading their outcomes: evaluation indices 000, 001, 002, 003, 006, 007, 008 and 009. Four have 128-token suffixes and four have 1,024-token suffixes. Context segments, probes, targets, instructions and stop-token IDs match exactly across the paired tokenizers; observed run token plans also agree. All records remain included regardless of qualification. Each variant is compared with its own never-stored reference, never with another precision's reference.", "",
             f"| State/readout check | {variant} | Original BF16 |", "|---|---:|---:|"]
    for label, key in (("Qualified contexts", "qualified_records"), ("Replay complete-state and audit-logit equality", "replay_complete_state_and_logits_equal"),
                       ("Repeated never-stored complete-state and logit equality", "repeat_numerical_floor_complete_state_and_logits_equal")):
        lines.append(f"| {label} | {p['8bit'][key]}/{n} | {p['bf16'][key]}/{n} |")
    for label, key in (("Maximum replay state absolute difference", "replay_state_max_abs_diff"), ("Maximum replay logit absolute difference", "replay_logits_max_abs_diff")):
        lines.append(f"| {label} | {p['8bit'][key]:.3g} | {p['bf16'][key]:.3g} |")
    lines += ["", "State checks cover every active KDA recurrent/convolution array, logical MLA keys/values and offsets, plus the full next-token audit logit vector. Equality here means matching shapes/offsets and zero float32 maximum absolute differences, not a bytewise certificate. With eight contexts, even 8/8 successes have a Wilson 95% interval of approximately 67.6–100%; observed zero differences do not guarantee population-wide equivalence.", "",
              "## Paired target readout", "",
              "Values are target mean log-probability differences against the same precision's own never-stored reference, averaged across three fixed target formulations (nats per target token). Brackets show paired whole-context bootstrap 95% intervals. The final column subtracts the 8-bit own-reference difference from the BF16 own-reference difference.", "",
              f"| Condition | {variant} minus own reference | BF16 minus own reference | BF16-minus-8-bit difference of differences |",
              "|---|---:|---:|---:|"]
    conditions = p["8bit"]["conditions_paired_difference_vs_own_never"]
    for condition in conditions:
        metric = "target_mean_log_probability"
        lines.append(f"| {condition} | {_interval(p['8bit']['conditions_paired_difference_vs_own_never'][condition][metric])} | {_interval(p['bf16']['conditions_paired_difference_vs_own_never'][condition][metric])} | {_interval(result['difference_of_own_reference_deltas_bf16_minus_8bit'][condition][metric])} |")
    lines += ["", "## Retained and continuation utility", "",
              "The table gives paired mean log-probability changes for the three retained records and paired held-out continuation NLL changes, each versus the same precision's own reference. A positive retained-score change is higher probability; a positive NLL change is worse continuation scoring. All intervals resample the eight contexts, keeping their probes attached.", "",
              f"| Condition | Retained score Δ: {variant} | Retained score Δ: BF16 | Continuation NLL Δ: {variant} | Continuation NLL Δ: BF16 |",
              "|---|---:|---:|---:|---:|"]
    for condition in ("present", "masked", "transport", "mask_transport", "replay", "prompt_forget", "prompt_neutral"):
        entries = [p[precision]["conditions_paired_difference_vs_own_never"][condition][metric]
                   for metric in ("retained_mean_log_probability", "continuation_nll") for precision in ("8bit", "bf16")]
        lines.append(f"| {condition} | " + " | ".join(_interval(value, digits=6 if index < 2 else 3) for index, value in enumerate(entries)) + " |")
    lines += ["", "Continuation utility uses two fixed synthetic passages per context, selected from a shared bank of six. These passages exclude the deleted record's information. This is a narrow synthetic utility check, not a broad language benchmark. Exact-answer rates in the JSON compare the complete decoded answer after outer-whitespace stripping and casefolding; containment rates separately count a target appearing anywhere in the answer. Decoding is greedy with the frozen 24-token cap.", "",
              "## Transport arithmetic diagnostics", "",
              result["arithmetic_diagnostics_definition"], "",
              f"| Context / suffix | Transport residual / original state difference: {variant} | BF16 | Residual / own empirical floor: {variant} | BF16 |",
              "|---|---:|---:|---:|---:|"]
    for a, b in zip(floor["8bit"]["records"], floor["bf16"]["records"]):
        ga, gb = a["global"], b["global"]
        lines.append(f"| {a['record_id'].rsplit('-',1)[-1]} / {a['suffix_tokens']} | {ga['native_residual_relative_to_present_difference']:.3f} | {gb['native_residual_relative_to_present_difference']:.3f} | {ga['residual_to_max_empirical_floor_ratio']:.3g} | {gb['residual_to_max_empirical_floor_ratio']:.3g} |")
    for precision, label in (("8bit", variant), ("bf16", "Original BF16")):
        records = floor[precision]["records"]
        below = sum(r["layer_residuals_at_or_below_observed_floor"] for r in records)
        count = sum(r["layer_count"] for r in records)
        eplus = max(r["global"]["present_reconstruction_relative_error"] for r in records)
        eminus = max(r["global"]["omitted_reconstruction_relative_error"] for r in records)
        lines += ["", f"{label}: maximum global present/omitted reconstruction relative errors were {eplus:.3g}/{eminus:.3g}. At layer level, {below}/{count} layer-context residuals were at or below their own empirical floor; global ratios do not imply that every layer exceeds the floor."]
    lines += ["", "The numerical diagnostics distinguish a transport residual from the observed arithmetic discrepancies in these runs. They do not establish a rigorous error bound or a general quantization claim. All intervals are descriptive marginal percentile intervals from 2,000 paired whole-context resamples; they exclude training variation, alternative quantizers, model families and domain shift. This eight-context control complements the larger cohort and does not establish universal deletion or utility guarantees.", "",
              f"Machine-readable comparison: [{summary_path.name}]({summary_path.absolute()})."]
    if secondary is not None:
        lines += ["", f"The separately pinned community-model comparison is retained as a [secondary artifact comparison]({secondary.absolute()}); its exact master-checkpoint relationship is unverified and its differences are not labelled causal quantization effects."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--floor-8bit", type=Path, required=True)
    parser.add_argument("--floor-bf16", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--secondary-markdown", type=Path)
    args = parser.parse_args(argv)
    result = attach_floors(json.loads(args.comparison.read_text()), args.floor_8bit, args.floor_bf16)
    args.comparison.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown(result, args.comparison, args.secondary_markdown))
    print(json.dumps({"status": result["status"], "basis": result["comparison_basis"], "records": result["n_records"], "markdown": str(args.markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
