"""Paired record-level inference for the frozen prospective KDA cohort.

Qualification is a reported property, never an admission filter. The primary
analysis rejects an incomplete cohort unless explicitly labelled exploratory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .manifest import canonical_bytes, load_manifest, validate_manifest


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}")
    return number


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Mann-Whitney AUC with half credit for ties; no label-trained direction."""
    positive = np.asarray(positive, dtype=float)
    negative = np.sort(np.asarray(negative, dtype=float))
    if not len(positive) or not len(negative):
        raise ValueError("AUC needs observations from both classes")
    below = np.searchsorted(negative, positive, side="left")
    at_or_below = np.searchsorted(negative, positive, side="right")
    return float(np.mean((below + at_or_below) / (2.0 * len(negative))))


def calibrate_threshold(negative: Sequence[float], target_fpr: float) -> float:
    """Conservative finite-sample order statistic; classify using score > t.

    If calibration is too small for the requested tail, an infinite threshold
    makes no positive calls. Infinity is represented as null in JSON reports.
    """
    negative = np.asarray(negative, dtype=float)
    if not len(negative) or not np.all(np.isfinite(negative)):
        raise ValueError("calibration requires finite negative scores")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie strictly between zero and one")
    rank = math.ceil((len(negative) + 1) * (1.0 - target_fpr))
    return float(np.sort(negative)[rank - 1]) if rank <= len(negative) else math.inf


def _interval(estimate: float, bootstrap: np.ndarray, n: int, confidence: float) -> dict[str, Any]:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return {"estimate": float(estimate), "ci_low": float(lower), "ci_high": float(upper),
            "confidence": confidence, "n_records": n}


def paired_mean_interval(left: Sequence[float], right: Sequence[float], *, seed: int = 0,
                         replicates: int = 2000, confidence: float = 0.95) -> dict[str, Any]:
    """Public helper: resample pairs, never condition rows independently."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired arrays must be equally sized nonempty vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired arrays must be finite")
    indices = np.random.default_rng(seed).integers(0, len(left), (replicates, len(left)))
    difference = left - right
    return _interval(float(np.mean(difference)), difference[indices].mean(axis=1), len(left), confidence)


def _probes(condition: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    probes = condition[name]
    if not isinstance(probes, list):
        raise ValueError(f"{name} must be a list")
    result = {probe["probe_id"]: probe for probe in probes}
    if len(result) != len(probes):
        raise ValueError(f"duplicate probe ids in {name}")
    return result


def _metrics(condition: dict[str, Any]) -> dict[str, float]:
    target = _probes(condition, "target_probes")
    retained = _probes(condition, "retained_probes")
    continuations = _probes(condition, "continuations")
    continuation_tokens = sum(int(probe["n_target_tokens"]) for probe in continuations.values())
    metrics = {
        "target_mean_log_probability": float(np.mean([probe["mean_log_probability"] for probe in target.values()])),
        "target_direct_log_probability": target["target-direct"]["mean_log_probability"],
        "target_direct_first_token_rank": target["target-direct"]["first_token_rank"],
        "retained_mean_log_probability": float(np.mean([probe["mean_log_probability"] for probe in retained.values()])),
        "continuation_nll": -sum(probe["total_log_probability"] for probe in continuations.values()) / continuation_tokens,
    }
    if "greedy_exact" in target["target-direct"]:
        metrics["target_direct_exact"] = float(target["target-direct"]["greedy_exact"])
    if all("greedy_exact" in probe for probe in retained.values()):
        metrics["retained_exact"] = float(np.mean([probe["greedy_exact"] for probe in retained.values()]))
    if "greedy_contains_target" in target["target-direct"]:
        metrics["target_direct_contains"] = float(target["target-direct"]["greedy_contains_target"])
    if all("greedy_contains_target" in probe for probe in retained.values()):
        metrics["retained_contains"] = float(np.mean([probe["greedy_contains_target"] for probe in retained.values()]))
    if "extra_instruction_tokens" in condition:
        metrics["extra_instruction_tokens"] = float(condition["extra_instruction_tokens"])
    return {name: _finite(value, name) for name, value in metrics.items()}


def _qualified(result: dict[str, Any], manifest: dict[str, Any]) -> bool:
    direct = manifest["protocol"]["qualification"]["probe_id"]
    present = _probes(result["conditions"]["present"], "target_probes")[direct]
    never = _probes(result["conditions"]["never_stored"], "target_probes")[direct]
    rule = manifest["protocol"]["qualification"]
    return (present["mean_log_probability"] - never["mean_log_probability"] >= rule["minimum_lift_nats"]
            and present["first_token_rank"] <= rule["maximum_first_token_rank"])


def _normalize_greedy(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Derive two distinctly named readback metrics from immutable raw text.

    The inherited runner called substring containment ``greedy_exact``. Keep
    the raw files unchanged and compute actual equality for the analysis.
    """
    result = copy.deepcopy(result)
    for condition in result["conditions"].values():
        for group in ("target_probes", "retained_probes"):
            targets = {probe["probe_id"]: probe["target"].strip().casefold() for probe in record[group]}
            for probe in condition[group]:
                if "greedy_text" in probe:
                    text = probe["greedy_text"].strip().casefold()
                    target = targets[probe["probe_id"]]
                    probe["greedy_contains_target"] = target in text
                    probe["greedy_exact"] = text == target
    return result


def _validate_result(result: dict[str, Any], record: dict[str, Any], manifest: dict[str, Any]) -> None:
    if result.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError(f"result has wrong manifest hash: {record['record_id']}")
    if result.get("schema_version", 1) != 1:
        raise ValueError("unsupported result schema")
    expected_conditions = set(manifest["protocol"]["conditions"])
    if set(result["conditions"]) != expected_conditions:
        raise ValueError(f"condition set differs from frozen protocol: {record['record_id']}")
    for name, condition in result["conditions"].items():
        for probe_kind in ("target_probes", "retained_probes", "continuations"):
            probes = _probes(condition, probe_kind)
            expected_ids = {probe["probe_id"] for probe in record[probe_kind]}
            if set(probes) != expected_ids:
                raise ValueError(f"probe set mismatch for {name}/{probe_kind}")
            for probe in probes.values():
                mean = _finite(probe["mean_log_probability"], "mean log probability")
                total = _finite(probe["total_log_probability"], "total log probability")
                count = int(probe["n_target_tokens"])
                if count <= 0 or mean > 1e-5 or total > 1e-5:
                    raise ValueError("invalid probe probability or target-token count")
                if not math.isclose(total / count, mean, abs_tol=1e-5, rel_tol=1e-5):
                    raise ValueError("inconsistent mean/total target log probability")
                rank = _finite(probe["first_token_rank"], "first-token rank")
                if rank < 1 or rank != int(rank):
                    raise ValueError("invalid first-token rank")
        _metrics(condition)
    if "qualification" in result and "qualified" in result["qualification"]:
        if bool(result["qualification"]["qualified"]) != _qualified(result, manifest):
            raise ValueError(f"qualification disagrees with frozen rule: {record['record_id']}")
    forget = result["conditions"]["prompt_forget"].get("extra_instruction_tokens")
    neutral = result["conditions"]["prompt_neutral"].get("extra_instruction_tokens")
    if forget is not None and neutral != forget:
        raise ValueError("prompt-forget and neutral controls have unmatched extra token counts")


def _mean_metrics(records: list[dict[str, Any]], conditions: list[str], *, rng: np.random.Generator,
                  replicates: int, confidence: float) -> dict[str, Any]:
    if not records:
        return {"n_records": 0, "conditions": {}, "qualification": "no records in this group"}
    n = len(records)
    indices = rng.integers(0, n, (replicates, n))
    extracted = {name: [_metrics(result["conditions"][name]) for result in records] for name in conditions}
    common = set.intersection(*(set(metric) for values in extracted.values() for metric in values))
    all_names = set.union(*(set(metric) for values in extracted.values() for metric in values))
    output: dict[str, Any] = {
        "n_records": n, "conditions": {}, "paired_prompt_forget_minus_neutral": {},
        "metric_record_counts": {name: {metric: sum(metric in item for item in extracted[name])
                                         for metric in sorted(all_names)} for name in conditions},
        "metrics_not_reported_due_to_incomplete_pairs": sorted(all_names - common),
    }
    for metric in sorted(common):
        forgetting = np.array([item[metric] for item in extracted["prompt_forget"]], dtype=float)
        neutral = np.array([item[metric] for item in extracted["prompt_neutral"]], dtype=float)
        difference = forgetting - neutral
        output["paired_prompt_forget_minus_neutral"][metric] = _interval(
            float(difference.mean()), difference[indices].mean(axis=1), n, confidence)
    for name in conditions:
        output["conditions"][name] = {}
        for metric in sorted(common):
            values = np.array([item[metric] for item in extracted[name]], dtype=float)
            reference = np.array([item[metric] for item in extracted["never_stored"]], dtype=float)
            difference = values - reference
            output["conditions"][name][metric] = {
                "absolute": _interval(float(values.mean()), values[indices].mean(axis=1), n, confidence),
                "paired_difference_vs_never_stored": _interval(
                    float(difference.mean()), difference[indices].mean(axis=1), n, confidence),
            }
            if metric == "continuation_nll":
                # PPL is exp(mean per-record token-weighted NLL), not arithmetic
                # mean of per-record perplexities. Bootstrap uses the same unit.
                output["conditions"][name]["continuation_perplexity"] = {
                    "absolute": _interval(float(np.exp(values.mean())), np.exp(values[indices].mean(axis=1)), n, confidence),
                    "paired_ratio_vs_never_stored": _interval(
                        float(np.exp(difference.mean())), np.exp(difference[indices].mean(axis=1)), n, confidence),
                }
    return output


def _attack_analysis(calibration: list[dict[str, Any]], evaluation: list[dict[str, Any]],
                     conditions: list[str], protocol: dict[str, Any], *, rng: np.random.Generator,
                     replicates: int, confidence: float) -> dict[str, Any]:
    if not calibration or not evaluation:
        return {"status": "unavailable", "reason": "both calibration and evaluation records are required"}
    score: Callable[[dict[str, Any], str], float] = lambda row, condition: _metrics(
        row["conditions"][condition])["target_mean_log_probability"]
    calibration_negative = np.array([score(row, "never_stored") for row in calibration])
    negative = np.array([score(row, "never_stored") for row in evaluation])
    threshold = calibrate_threshold(calibration_negative, protocol["target_fpr"])
    cal_idx = rng.integers(0, len(calibration), (replicates, len(calibration)))
    eval_idx = rng.integers(0, len(evaluation), (replicates, len(evaluation)))
    reestimated = np.array([calibrate_threshold(calibration_negative[idx], protocol["target_fpr"])
                            for idx in cal_idx])
    boot_negative = negative[eval_idx]
    fpr = float(np.mean(negative > threshold))
    fpr_fixed = (boot_negative > threshold).mean(axis=1)
    fpr_joint = (boot_negative > reestimated[:, None]).mean(axis=1)
    report: dict[str, Any] = {
        "status": "complete", "access": protocol["access"], "query_budget": protocol["query_budget"],
        "score": protocol["score"], "higher_is_member": True,
        "target_fpr": protocol["target_fpr"], "threshold": threshold if math.isfinite(threshold) else None,
        "threshold_is_positive_infinity": not math.isfinite(threshold),
        "threshold_comparison": "strictly greater than", "calibration_records": len(calibration),
        "evaluation_records": len(evaluation),
        "calibration_empirical_fpr": float(np.mean(calibration_negative > threshold)),
        "evaluation_fpr": _interval(fpr, fpr_fixed, len(evaluation), confidence),
        "evaluation_fpr_with_calibration_uncertainty": _interval(fpr, fpr_joint, len(evaluation), confidence),
        "conditions": {},
        "interpretation": ("This is one fixed candidate-scoring membership distinguisher, not an optimal attack. "
                           "The requested 5% level is a calibration target; evaluation FPR is measured. "
                           "Low-FPR resolution is limited by the number of independent records."),
    }
    for name in conditions:
        positive = np.array([score(row, name) for row in evaluation])
        boot_positive = positive[eval_idx]
        tpr = float(np.mean(positive > threshold))
        tpr_fixed = (boot_positive > threshold).mean(axis=1)
        tpr_joint = (boot_positive > reestimated[:, None]).mean(axis=1)
        auc_samples = np.array([auc(positive[idx], negative[idx]) for idx in eval_idx])
        report["conditions"][name] = {
            "auc": _interval(auc(positive, negative), auc_samples, len(evaluation), confidence),
            "tpr": _interval(tpr, tpr_fixed, len(evaluation), confidence),
            "tpr_with_calibration_uncertainty": _interval(tpr, tpr_joint, len(evaluation), confidence),
            "tpr_minus_fpr": _interval(tpr - fpr, tpr_fixed - fpr_fixed, len(evaluation), confidence),
            "tpr_minus_fpr_with_calibration_uncertainty": _interval(
                tpr - fpr, tpr_joint - fpr_joint, len(evaluation), confidence),
            "is_null_control": name in protocol["null_controls"] or name == "never_stored",
        }
    return report


def _state_summary(records: list[dict[str, Any]], *, rng: np.random.Generator,
                   replicates: int, confidence: float) -> dict[str, Any]:
    report: dict[str, Any] = {"n_records": len(records), "identity_audits": {}, "corrections": {}}
    for audit in ("replay_vs_never", "repeat_never_vs_never"):
        audited = [(row["record_id"], row["state_checks"][audit]) for row in records
                   if audit in row.get("state_checks", {})]
        report["identity_audits"][audit] = {
            "audited_records": len(audited), "missing_records": len(records) - len(audited),
            "equal_records": sum(bool(item["all_equal"]) for _, item in audited),
            "equal_rate_wilson_95": (wilson_interval(sum(bool(item["all_equal"]) for _, item in audited),
                                                     len(audited)) if audited else None),
            "logit_audited_records": sum("logits" in item for _, item in audited),
            "unequal_record_ids": [identifier for identifier, item in audited if not item["all_equal"]],
            "all_measured_arrays_offsets_and_logits_equal": bool(audited) and len(audited) == len(records)
                and all(item["all_equal"] and item.get("logits", {}).get("equal", False) for _, item in audited),
            "n_arrays_range": ([min(item["n_arrays"] for _, item in audited),
                                max(item["n_arrays"] for _, item in audited)] if audited else None),
            "n_offsets_range": ([min(len(item["offsets"]) for _, item in audited),
                                 max(len(item["offsets"]) for _, item in audited)] if audited else None),
            "max_abs_array_diff": max((item["max_abs_diff"] for _, item in audited), default=None),
            "max_abs_logit_diff": max((item["logits"]["max_abs_diff"] for _, item in audited
                                       if item.get("logits", {}).get("max_abs_diff") is not None), default=None),
        }
    for name in ("present", "static", "ledger", "transport"):
        values = np.array([row["state_checks"]["corrections"][name]["global_relative_to_present_difference"]
                           for row in records if name in row.get("state_checks", {}).get("corrections", {})], dtype=float)
        if len(values):
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite state residual")
            indices = rng.integers(0, len(values), (replicates, len(values)))
            report["corrections"][name] = {
                "global_relative_to_present_difference": _interval(float(values.mean()), values[indices].mean(axis=1),
                                                                    len(values), confidence),
                "minimum": float(values.min()), "maximum": float(values.max()),
                "missing_records": len(records) - len(values),
            }
    return report


def analyze(manifest: dict[str, Any], results: Sequence[dict[str, Any]], *,
            allow_partial: bool = False, replicates: int | None = None) -> dict[str, Any]:
    validate_manifest(manifest)
    protocol = manifest["protocol"]
    bootstrap = protocol["bootstrap"]
    b = int(replicates if replicates is not None else bootstrap["replicates"])
    if b < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    confidence = bootstrap["confidence"]
    rng = np.random.default_rng(bootstrap["seed"])
    manifest_records = {record["record_id"]: record for record in manifest["records"]}
    observed: dict[str, dict[str, Any]] = {}
    for result in results:
        identifier = result["record_id"]
        if identifier not in manifest_records:
            raise ValueError(f"result does not belong to manifest: {identifier}")
        if identifier in observed:
            raise ValueError(f"duplicate result for {identifier}")
        observed[identifier] = result
    missing = [identifier for identifier in manifest_records if identifier not in observed]
    failed = [identifier for identifier, result in observed.items() if result.get("status") != "complete"]
    if (missing or failed) and not allow_partial:
        raise ValueError(f"incomplete cohort: {len(missing)} missing and {len(failed)} failed; "
                         "use allow_partial only for explicitly incomplete progress analysis")
    complete = [observed[identifier] for identifier in manifest_records
                if identifier in observed and observed[identifier].get("status") == "complete"]
    model_ids = {row.get("model_id") for row in complete}
    if len(model_ids) > 1:
        raise ValueError("cannot pool results from different models or quantizations")
    for result in complete:
        _validate_result(result, manifest_records[result["record_id"]], manifest)
    tiny_flags = {row.get("tiny", False) for row in complete}
    if len(tiny_flags) > 1:
        raise ValueError("cannot pool tiny plumbing checks with real model results")
    complete = [_normalize_greedy(row, manifest_records[row["record_id"]]) for row in complete]
    evaluation = [row for row in complete if manifest_records[row["record_id"]]["split"] == "evaluation"]
    calibration = [row for row in complete if manifest_records[row["record_id"]]["split"] == "calibration"]
    qualified = [row for row in evaluation if _qualified(row, manifest)]
    groups: dict[str, list[dict[str, Any]]] = {"all_evaluation_records": evaluation,
                                             "qualified_evaluation_records_secondary": qualified}
    for suffix in ("short", "long"):
        groups[f"suffix_{suffix}"] = [row for row in evaluation
                                     if manifest_records[row["record_id"]]["suffix_class"] == suffix]
    qualification_flags = np.array([float(_qualified(row, manifest)) for row in evaluation])
    q_interval = None
    if len(evaluation):
        indices = rng.integers(0, len(evaluation), (b, len(evaluation)))
        q_interval = _interval(float(qualification_flags.mean()), qualification_flags[indices].mean(axis=1),
                               len(evaluation), confidence)
        q_interval["wilson_95"] = wilson_interval(len(qualified), len(evaluation))
    report = {
        "schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
        "analysis_status": "incomplete_exploratory" if missing or failed else "complete",
        "model_id": next(iter(model_ids)) if model_ids else None,
        "evidence_class": "tiny_random_model_plumbing_only" if True in tiny_flags else "model_evaluation",
        "preselected_records": len(manifest_records), "completed_records": len(complete),
        "missing_record_ids": missing, "failed_record_ids": failed,
        "evaluation_records": len(evaluation), "calibration_records": len(calibration),
        "qualification": {"qualified": len(qualified), "failures": len(evaluation) - len(qualified),
                          "rate": q_interval, "primary_analysis_includes_failures": True},
        "bootstrap": {**bootstrap, "replicates": b, "replicates_overridden": b != bootstrap["replicates"]},
        "analysis_provenance": {
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "raw_result_canonical_sha256": {
                identifier: hashlib.sha256(canonical_bytes(observed[identifier])).hexdigest()
                for identifier in sorted(observed)
            },
            "hash_scope": "raw result dictionaries before deriving exact-versus-containment scoring",
        },
        "groups": {name: _mean_metrics(rows, protocol["conditions"], rng=rng,
                                       replicates=b, confidence=confidence) for name, rows in groups.items()},
        "attack": _attack_analysis(calibration, evaluation, protocol["conditions"], protocol["attack"],
                                    rng=rng, replicates=b, confidence=confidence),
        "state_checks": {
            "all_completed_records": _state_summary(complete, rng=rng, replicates=b, confidence=confidence),
            "evaluation_records": _state_summary(evaluation, rng=rng, replicates=b, confidence=confidence),
        },
        "greedy_scoring": {
            "exact": "generated text stripped and casefolded equals stripped and casefolded target",
            "contains": "stripped and casefolded target is a substring of stripped and casefolded generated text",
            "raw_runner_greedy_exact": "legacy name for containment; recomputed from raw greedy_text for analysis",
        },
        "limitations": [
            "Intervals describe variation across this frozen synthetic-context design; they do not establish deployment-population guarantees.",
            "Qualification failures remain in the primary estimand; qualified-only estimates are secondary and selection-conditioned.",
            "Percentile bootstrap intervals are marginal descriptive intervals without multiplicity adjustment.",
            "The token-matched neutral instruction uses fixed prose repeated/truncated to length and can end in a fragment; the prompt contrast is conditional on these specific strings.",
            "AUC intervals resample whole matched contexts, keeping positive and never-stored scores together.",
            "Zero observed event rates can produce degenerate percentile intervals; inspect the Wilson binomial interval for finite-sample rate resolution.",
        ],
    }
    # Bootstrap intervals at 0/1 hide event-count resolution. Wilson intervals
    # accompany attack rates as a transparent small-cohort diagnostic.
    if report["attack"].get("status") == "complete":
        attack = report["attack"]
        attack["evaluation_fpr"]["wilson_95"] = wilson_interval(
            round(attack["evaluation_fpr"]["estimate"] * len(evaluation)), len(evaluation))
        for values in attack["conditions"].values():
            values["tpr"]["wilson_95"] = wilson_interval(
                round(values["tpr"]["estimate"] * len(evaluation)), len(evaluation))
    return report


def wilson_interval(successes: int, n: int) -> list[float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_results(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path_ in paths:
        path = Path(path_)
        if path.is_dir():
            results.extend(load_results(sorted(path.glob("*.json"))))
            continue
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            value = json.loads(path.read_text())
            values = value if isinstance(value, list) else value.get("results", [value])
        for value in values:
            if "record_id" in value:
                results.append(value)
    return results


def markdown_report(report: dict[str, Any]) -> str:
    def show(interval: dict[str, Any]) -> str:
        return f"{interval['estimate']:.4f} [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"

    lines = ["# Prospective KDA follow-up results", "",
             f"Status: **{report['analysis_status']}**. Model: `{report['model_id']}`.", "",
             f"Evidence class: `{report['evidence_class']}`.", "",
             f"Manifest SHA-256: `{report['manifest_sha256']}`.", "",
             f"Completed {report['completed_records']}/{report['preselected_records']} preselected contexts: "
             f"{report['evaluation_records']} evaluation and {report['calibration_records']} calibration. "
             f"Qualification: {report['qualification']['qualified']} passed, {report['qualification']['failures']} failed. "
             "All completed evaluation records contribute to the primary table, including qualification failures.", "",
             "Estimates include paired 95% whole-record bootstrap intervals. Log-probability and NLL "
             "differences are condition minus never-stored; positive target log-probability difference indicates residual lift.", "",
             "| Condition | Target log-probability difference | Retained log-probability difference | Continuation NLL difference |",
             "|---|---:|---:|---:|"]
    conditions = report["groups"]["all_evaluation_records"]["conditions"]
    for name, metrics in conditions.items():
        fields = [show(metrics[key]["paired_difference_vs_never_stored"])
                  for key in ("target_mean_log_probability", "retained_mean_log_probability", "continuation_nll")]
        lines.append(f"| {name} | {' | '.join(fields)} |")
    if report["qualification"]["rate"] is not None:
        lower, upper = report["qualification"]["rate"]["wilson_95"]
        lines.extend(["", f"The qualification rate has a Wilson 95% interval [{lower:.4f}, {upper:.4f}]; "
                      "a zero observed failure count does not imply a zero population failure rate."])
    contrasts = report["groups"]["all_evaluation_records"].get("paired_prompt_forget_minus_neutral", {})
    if contrasts:
        lines.extend(["", "The matched token-budget contrast (prompt forgetting minus neutral instruction) is "
                      f"{show(contrasts['target_mean_log_probability'])} for target log probability, "
                      f"{show(contrasts['retained_mean_log_probability'])} for retained log probability, and "
                      f"{show(contrasts['continuation_nll'])} for continuation NLL."])
    if conditions and "extra_instruction_tokens" in conditions.get("prompt_forget", {}):
        forgetting_tokens = conditions["prompt_forget"]["extra_instruction_tokens"]["absolute"]
        lines.extend(["", f"The forgetting and neutral instructions each add {show(forgetting_tokens)} "
                      "tokens per context (mean and whole-record interval). Their token counts match within every record."])
    if conditions and all("target_direct_contains" in values for values in conditions.values()):
        lines.extend(["", "| Condition | Target contains | Target exact | Retained contains | Retained exact |",
                      "|---|---:|---:|---:|---:|"])
        for name, metrics in conditions.items():
            cells = [show(metrics[key]["absolute"]) for key in ("target_direct_contains", "target_direct_exact",
                                                               "retained_contains", "retained_exact")]
            lines.append(f"| {name} | {' | '.join(cells)} |")
    attack = report["attack"]
    if attack.get("status") == "complete":
        lines.extend(["", "## Fixed finite-query attack", "",
                      f"Three fixed candidate-scoring queries; nominal calibration FPR {attack['target_fpr']:.0%}. "
                      f"Threshold: {attack['threshold']}. Evaluation FPR: {show(attack['evaluation_fpr'])}. "
                      "TPR intervals below include calibration-threshold uncertainty.", "",
                      "| Condition | AUC | TPR | TPR − FPR |", "|---|---:|---:|---:|"])
        for name, values in attack["conditions"].items():
            lines.append(f"| {name} | {show(values['auc'])} | "
                         f"{show(values['tpr_with_calibration_uncertainty'])} | "
                         f"{show(values['tpr_minus_fpr_with_calibration_uncertainty'])} |")
    lines.extend(["", "## Complete-state reference audits", ""])
    for group, summary in report["state_checks"].items():
        for audit, values in summary["identity_audits"].items():
            lines.append(f"- {group}, {audit}: {values['equal_records']}/{values['audited_records']} equal; "
                         f"array count range {values['n_arrays_range']}, offset count range {values['n_offsets_range']}; "
                         f"maximum array/logit differences {values['max_abs_array_diff']}/{values['max_abs_logit_diff']}; "
                         f"equality-rate Wilson 95% interval {values['equal_rate_wilson_95']}.")
    lines.extend(["", "## Scope and uncertainty", "", *[f"- {item}" for item in report["limitations"]], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    report = analyze(load_manifest(args.manifest), load_results(args.results), allow_partial=args.allow_partial,
                     replicates=args.bootstrap_replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report))
    print(json.dumps({"status": report["analysis_status"], "completed_records": report["completed_records"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
