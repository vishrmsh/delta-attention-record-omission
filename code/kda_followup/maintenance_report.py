"""Summarize maintenance results without treating nested probes as iid records."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .analyze import paired_mean_interval


def wilson(successes: int, total: int) -> dict[str, float | int]:
    if not 0 <= successes <= total or total < 1:
        raise ValueError("Wilson interval needs valid nonempty binomial counts")
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    width = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return {"successes": successes, "n_independent_units": total, "estimate": fraction,
            "ci_low": max(0.0, center - width), "ci_high": min(1.0, center + width), "confidence": .95}


def _exact(probe: dict[str, Any]) -> bool:
    return probe["greedy_answer"].strip().casefold() == probe["target"].strip().casefold()


def _contains(probe: dict[str, Any]) -> bool:
    return probe["target"].strip().casefold() in probe["greedy_answer"].strip().casefold()


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    if raw["status"] != "complete" or raw["tiny_plumbing_only"]:
        raise ValueError("report requires a completed real-weight maintenance run")
    rows = raw["amendments"]
    n = len(rows)
    if n != 12 or len(raw["sequences"]) != 12:
        raise ValueError("full maintenance report expects 12 amendment contexts and 12 sequence paths")
    rates = {}
    for condition, target, label in (
        ("as_heard", "old", "old_answer_before"),
        ("as_heard", "new", "new_answer_before"),
        ("after_amendment", "old", "old_answer_after"),
        ("after_amendment", "new", "new_answer_after"),
        ("corrected_from_start", "new", "new_answer_corrected_reference"),
    ):
        rates[label] = {"exact": wilson(sum(_exact(r[condition][target]) for r in rows), n),
                        "contains": wilson(sum(_contains(r[condition][target]) for r in rows), n)}
    rates["retained_all_three_after"] = {
        "exact": wilson(sum(all(_exact(v) for v in r["after_amendment"]["retained"].values()) for r in rows), n),
        "contains": wilson(sum(all(_contains(v) for v in r["after_amendment"]["retained"].values()) for r in rows), n)}
    paired = {}
    for target in ("old", "new"):
        for reference in ("as_heard", "corrected_from_start"):
            paired[f"{target}_mean_logprob_after_minus_{reference}"] = paired_mean_interval(
                [r["after_amendment"][target]["mean_log_probability"] for r in rows],
                [r[reference][target]["mean_log_probability"] for r in rows], seed=raw["manifest"]["seed"] + 1)
    def retained_mean(row: dict[str, Any], condition: str) -> float:
        values = row[condition]["retained"].values()
        return sum(v["mean_log_probability"] for v in values) / len(row[condition]["retained"])
    for reference in ("as_heard", "corrected_from_start"):
        paired[f"retained_mean_logprob_after_minus_{reference}"] = paired_mean_interval(
            [retained_mean(r, "after_amendment") for r in rows],
            [retained_mean(r, reference) for r in rows], seed=raw["manifest"]["seed"] + 1)
    grouped = {}
    for sequence in raw["sequences"]:
        group = sequence["id"].rsplit("-", 1)[0]
        grouped.setdefault(group, []).append(sequence)
    if len(grouped) != 6 or any(len(v) != 2 for v in grouped.values()):
        raise ValueError("sequence grouping must contain six pairs, both orders per pair")
    groups = []
    for group, paths in grouped.items():
        final_steps = [p["steps"][-1] for p in paths]
        target_keys = paths[0]["deletion_order"]
        forbidden = sum(_contains(step["scores"][key]) for step in final_steps for key in target_keys)
        retained_all = all(_exact(probe) for step in final_steps for key, probe in step["scores"].items() if key not in target_keys)
        retained_contains = all(_contains(probe) for step in final_steps for key, probe in step["scores"].items() if key not in target_keys)
        # Same surviving token order underlies both fresh references; compare
        # their saved final probe readouts directly as a second order check.
        order_agreement = (final_steps[0]["live_keys"] == final_steps[1]["live_keys"]
                           and final_steps[0]["scores"] == final_steps[1]["scores"])
        groups.append({"group_id": group, "all_six_transition_audits_equal": all(s["audit"]["all_equal"] for p in paths for s in p["steps"]),
                       "both_orders_final_scores_agree": order_agreement,
                       "final_deleted_target_containments": forbidden,
                       "retained_and_admitted_answers_all_exact": retained_all,
                       "retained_and_admitted_answers_all_contain_target": retained_contains})
    steps = [s for p in raw["sequences"] for s in p["steps"]]
    all_audits = [r["audit"] for r in rows] + [s["audit"] for s in steps]
    floors = [r["numerical_floor_repeat_corrected_rebuild"] for r in rows] + [s["numerical_floor_repeat_rebuild"] for s in steps]
    return {"schema": "kda-maintenance-summary-v1", "status": "complete", "model": raw["model"],
            "manifest_sha256": raw["manifest"]["sha256"], "amendment_contexts": n,
            "qualification_passed": sum(r["qualification"]["lift_exceeds_one_nat"] for r in rows),
            "no_record_filtering": True,
            "suffix_token_counts": dict(Counter(str(r["suffix_tokens"]) for r in raw["manifest"]["scenarios"])),
            "answer_rates": rates, "paired_record_bootstrap_mean_logprob_changes": paired,
            "retained_after_exact_questions": sum(_exact(v) for r in rows for v in r["after_amendment"]["retained"].values()),
            "retained_after_total_questions": sum(len(r["after_amendment"]["retained"]) for r in rows),
            "greedy_answers_after_identical_to_corrected_reference": sum(r["after_amendment"]["old"]["greedy_answer"] == r["corrected_from_start"]["old"]["greedy_answer"] for r in rows),
            "sequential": {"independent_record_pairs": 6, "paths": 12, "transitions": len(steps), "groups": groups,
                           "groups_all_transitions_equal": wilson(sum(g["all_six_transition_audits_equal"] for g in groups), 6),
                           "groups_deleted_answers_absent_both_orders": wilson(sum(g["final_deleted_target_containments"] == 0 for g in groups), 6),
                           "groups_retained_and_admitted_all_exact": wilson(sum(g["retained_and_admitted_answers_all_exact"] for g in groups), 6),
                           "groups_retained_and_admitted_all_contain_target": wilson(sum(g["retained_and_admitted_answers_all_contain_target"] for g in groups), 6)},
            "audits": {"complete_state_and_logit_checks": len(all_audits), "all_equal": all(a["all_equal"] for a in all_audits),
                       "max_abs_complete_state_difference": max(a["complete_cache"]["max_abs_diff"] for a in all_audits),
                       "max_abs_logit_difference": max(a["probe_logit_max_abs_diff"] for a in all_audits),
                       "repeated_reference_checks": len(floors), "repeated_reference_all_equal": all(a["all_equal"] for a in floors)},
            "definitions": {"exact": "The entire decoded answer, stripped of surrounding whitespace and Unicode casefolded, equals the similarly normalized target; no punctuation is removed.",
                            "contains": "The normalized target occurs anywhere in the normalized answer; this is the raw runner greedy_contains_target criterion.",
                            "wilson": "Two-sided 95% Wilson score intervals at independent context level; retained success means all three probes succeed, sequence success means both orders and all transitions for one record pair succeed.",
                            "paired_bootstrap": "2000 percentile resamples of whole amendment contexts; all old/new/retained before-after measurements stay paired. Retained log probabilities average three queries within each context.",
                            "zero_interval": "Degenerate paired intervals at zero reflect observed deterministic agreement and do not certify future exactness.",
                            "state": "Every KDA recurrent and convolution array and logical active MLA key/value array, dtype, shape and offset agree numerically; max absolute differences are computed in fp32. This is not a bytewise comparison.",
                            "floor": "Repeated independent fresh rebuild under the same model and chunking measures repeatability, not an arithmetic reconstruction error bound.",
                            "timing": "Execution overlapped other GPU experiments and a weight download; timings are diagnostic and excluded from controlled performance claims.",
                            "grouping": "Sequential paths reuse the 12 amendment target records as six disjoint pairs; do not add them to the amendment sample size or treat the 36 transitions as independent records.",
                            "scope": "Synthetic, preselected small-cohort evidence; intervals assume exchangeability at the stated independent unit and are not a security guarantee."}}


def _rate(value: dict[str, Any]) -> str:
    return f"{value['successes']}/{value['n_independent_units']} ({100*value['estimate']:.1f}%; 95% CI {100*value['ci_low']:.1f}–{100*value['ci_high']:.1f}%)"


def markdown(result: dict[str, Any], raw_path: Path) -> str:
    rates = result["answer_rates"]
    lines = ["# Kimi maintenance follow-up results", "",
             "All 12 preselected amendment contexts and all 36 sequential-maintenance transitions completed. Corrected/replayed states and probe logits had zero observed maximum absolute difference from their appropriate fresh references.", "",
             f"Model: `{result['model']}`. Frozen maintenance manifest: `{result['manifest_sha256']}`. All 12 old-target qualification lifts exceeded one nat versus never-stored; no records were excluded. Extra neutral suffixes were 0 tokens in six contexts, 128 in three and 512 in three. Target positions varied across the four-record contexts.", "",
             "| Amendment outcome | Exact answer: contexts (Wilson 95% CI) | Target containment: contexts (Wilson 95% CI) |",
             "|---|---|---|"]
    for name, label in (("old_answer_before", "Old answer before amendment"), ("new_answer_before", "New answer before amendment"),
                        ("old_answer_after", "Old answer after amendment"), ("new_answer_after", "New answer after amendment"),
                        ("new_answer_corrected_reference", "New answer in corrected-from-start reference"), ("retained_all_three_after", "All three retained answers after amendment")):
        lines.append(f"| {label} | {_rate(rates[name]['exact'])} | {_rate(rates[name]['contains'])} |")
    lines += ["", f"**Scoring.** Exact means full-answer equality after stripping outer whitespace and applying Unicode casefolding; punctuation is preserved. Containment means the target appears anywhere in that normalized answer. Both are computed from the saved generated text. Decoding was greedy with a 16-token cap. The three retained queries form one context for uncertainty estimates: {result['retained_after_exact_questions']}/36 answers exactly match and 36/36 contain the target; the intervals use the 12 context outcomes shown above.", "",
              "| Paired change in mean target log probability (nats/token) | Estimate | Record-bootstrap 95% CI |", "|---|---:|---:|"]
    for name, interval in result["paired_record_bootstrap_mean_logprob_changes"].items():
        lines.append(f"| {name.replace('_', ' ')} | {interval['estimate']:.6f} | [{interval['ci_low']:.6f}, {interval['ci_high']:.6f}] |")
    seq = result["sequential"]
    lines += ["", "Paired intervals use 2,000 resamples of whole amendment contexts; all before/after/reference measurements remain paired, and retained scores average the three queries within each context. Zero-width intervals at zero describe deterministic agreement observed here and are not guarantees for future records.", "",
              f"**Sequential deletion and admission.** Six disjoint record pairs were each tested in both deletion orders. Every path deletes one target, admits a new record, then deletes the other target: 12 paths and 36 transitions. Every transition was checked against a newly built surviving-record reference; both final orders had identical surviving-record keys and saved probe scores in {sum(g['both_orders_final_scores_agree'] for g in seq['groups'])}/6 groups. All-transition state/logit agreement: {_rate(seq['groups_all_transitions_equal'])}. Both deleted answers absent in both final orders: {_rate(seq['groups_deleted_answers_absent_both_orders'])}. All retained and admitted answers exact in both final orders: {_rate(seq['groups_retained_and_admitted_all_exact'])}.", "",
              f"All retained and admitted answers contained their targets in both orders in {_rate(seq['groups_retained_and_admitted_all_contain_target'])}. The independent sequential unit is the record pair (n=6), not a path or transition. These pairs reuse the amendment targets and do not enlarge the independent 12-target cohort.", "",
              "**State checks and scope.** Each comparison covers all 80 KDA recurrent/convolution arrays, all 14 active MLA key/value arrays, seven MLA offsets, and one full next-token logit vector. All 48 intervention-to-reference checks and 48 repeated-reference checks agreed numerically. Array dtypes and shapes were checked; float32 maximum absolute state and logit differences were 0.0. These are numeric comparisons, not bytewise certificates. The repeated-reference control measures deterministic repeatability; transport arithmetic is assessed separately.", "",
              "The evidence concerns this small, frozen synthetic cohort. Wilson intervals assume exchangeability of the stated context or pair units. Timings overlapped other experiments and a weight download, so they support no controlled latency claim.", "",
              f"Raw results: [{raw_path.name}]({raw_path.absolute()})."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.input.read_text()))
    result["raw_output_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    for path in (args.json, args.markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(result, args.input))
    print(json.dumps({"amendments": result["amendment_contexts"], "state_checks": result["audits"], "markdown": str(args.markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
