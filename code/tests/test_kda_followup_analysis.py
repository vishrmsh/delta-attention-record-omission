"""Scientific-unit tests for prospective record selection and paired inference."""
from __future__ import annotations

import copy
import math

import pytest

from kda_followup.analyze import analyze, auc, calibrate_threshold, paired_mean_interval, wilson_interval
from kda_followup.manifest import build_manifest, load_manifest, write_manifest
from kda_followup.auc_sensitivity import offdiagonal_auc


def _results(manifest):
    rows = []
    for index, record in enumerate(manifest["records"]):
        row = {"schema_version": 1, "manifest_sha256": manifest["manifest_sha256"],
               "record_id": record["record_id"], "model_id": "test-only-score-fixture", "status": "complete",
               "conditions": {}}
        for condition in manifest["protocol"]["conditions"]:
            delta = (1.0 if index % 2 == 0 else .2) if condition == "present" else 0.0
            mean = -6.0 + (index % 40) / 100.0 + delta
            entry = {"extra_instruction_tokens": 20 if condition.startswith("prompt_") else 0}
            for group in ("target_probes", "retained_probes", "continuations"):
                entry[group] = [{"probe_id": probe["probe_id"], "mean_log_probability": mean,
                                 "total_log_probability": mean * 3, "n_target_tokens": 3,
                                 "first_token_rank": 1 if index % 2 == 0 else 20,
                                 "greedy_exact": condition == "present" and index % 2 == 0}
                                for probe in record[group]]
            row["conditions"][condition] = entry
        rows.append(row)
    return rows


def test_manifest_is_new_balanced_disjoint_and_target_free():
    manifest = build_manifest()
    assert manifest == build_manifest()
    records = manifest["records"]
    assert sum(row["split"] == "evaluation" for row in records) == 40
    assert sum(row["split"] == "calibration" for row in records) == 40
    assert {row["suffix_tokens"] for row in records} == {128, 1024}
    assert len({row["domain"] for row in records}) == 6
    identifiers = {row["target"]["identifier"] for row in records}
    assert len(identifiers) == 80
    for row in records:
        assert len(row["retained_records"]) == 3
        assert row["target"]["secret"].strip() not in row["forget_instruction"]
        for passage in row["continuations"]:
            assert row["target"]["secret"].strip() not in passage["target"]
            assert row["target"]["identifier"] not in passage["prompt"]


def test_frozen_manifest_detects_edit_and_refuses_replacement(tmp_path):
    path = tmp_path / "frozen.json"
    first = write_manifest(path)
    assert write_manifest(path) == first
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_manifest(path, seed=first["seed"] + 1)
    path.write_text(path.read_text().replace("700000", "700009"))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(path)


def test_record_pairing_keeps_shared_variation_out_of_difference_interval():
    reference = [-1000.0, 0.0, 1000.0, 100000.0]
    result = paired_mean_interval([x + 2 for x in reference], reference, replicates=100)
    assert result["estimate"] == 2.0
    assert result["ci_low"] == result["ci_high"] == 2.0


def test_auc_ties_and_fixed_direction():
    assert auc([1, 2, 3], [1, 2, 3]) == .5
    assert auc([2, 3], [0, 1]) == 1.0
    assert auc([0, 1], [2, 3]) == 0.0
    assert auc([1, 1], [1, 1]) == .5


def test_offdiagonal_sensitivity_removes_matched_pair_finite_sample_contribution():
    # The ordinary ROC AUC includes two guaranteed own-context wins. Across
    # distinct contexts, one ordering is correct and one is incorrect.
    assert auc([.1, 1.1], [0, 1]) == .75
    assert offdiagonal_auc([.1, 1.1], [0, 1]) == .5
    assert offdiagonal_auc([0, 1, 2], [0, 1, 2]) == .5


def test_conservative_calibration_order_statistic():
    threshold = calibrate_threshold(list(range(40)), .05)
    assert threshold == 38
    assert sum(x > threshold for x in range(40)) / 40 == .025
    assert math.isinf(calibrate_threshold(list(range(10)), .05))
    assert calibrate_threshold([0.0] * 40, .05) == 0.0
    assert wilson_interval(0, 40)[1] > .08


def test_primary_keeps_qualification_failures_and_replay_null_is_exact():
    manifest = build_manifest()
    report = analyze(manifest, _results(manifest), replicates=100)
    assert report["analysis_status"] == "complete"
    assert report["qualification"]["qualified"] == 20
    assert report["qualification"]["failures"] == 20
    low, high = report["qualification"]["rate"]["wilson_95"]
    assert low < .5 < high
    all_metrics = report["groups"]["all_evaluation_records"]["conditions"]
    selected = report["groups"]["qualified_evaluation_records_secondary"]["conditions"]
    assert all_metrics["present"]["target_mean_log_probability"]["paired_difference_vs_never_stored"]["estimate"] == pytest.approx(.6)
    assert selected["present"]["target_mean_log_probability"]["paired_difference_vs_never_stored"]["estimate"] == pytest.approx(1.0)
    replay = report["attack"]["conditions"]["replay"]
    assert replay["auc"]["estimate"] == .5
    assert replay["auc"]["ci_low"] == replay["auc"]["ci_high"] == .5
    assert replay["tpr_minus_fpr"]["estimate"] == 0.0
    assert replay["tpr_minus_fpr"]["ci_low"] == replay["tpr_minus_fpr"]["ci_high"] == 0.0
    assert all_metrics["replay"]["continuation_perplexity"]["paired_ratio_vs_never_stored"]["estimate"] == 1.0


def test_calibration_threshold_ignores_evaluation_scores():
    manifest = build_manifest()
    rows = _results(manifest)
    first = analyze(manifest, rows, replicates=100)
    for row in rows[40:]:
        for condition in row["conditions"].values():
            for group in ("target_probes", "retained_probes", "continuations"):
                for probe in condition[group]:
                    probe["mean_log_probability"] -= 100
                    probe["total_log_probability"] -= 300
    second = analyze(manifest, rows, replicates=100)
    assert first["attack"]["threshold"] == second["attack"]["threshold"]
    assert second["attack"]["evaluation_fpr"]["estimate"] == 0


def test_missing_failed_duplicate_and_wrong_hash_cannot_silently_drop_records():
    manifest = build_manifest()
    rows = _results(manifest)
    with pytest.raises(ValueError, match="incomplete cohort"):
        analyze(manifest, rows[:-1], replicates=100)
    partial = analyze(manifest, rows[:-1], allow_partial=True, replicates=100)
    assert partial["analysis_status"] == "incomplete_exploratory"
    assert partial["missing_record_ids"] == [rows[-1]["record_id"]]
    with pytest.raises(ValueError, match="duplicate result"):
        analyze(manifest, rows + [rows[0]], replicates=100)
    altered = copy.deepcopy(rows)
    altered[0]["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="wrong manifest hash"):
        analyze(manifest, altered, replicates=100)
    altered[0] = {**rows[0], "status": "error"}
    with pytest.raises(ValueError, match="1 failed"):
        analyze(manifest, altered, replicates=100)


def test_probe_or_token_control_mismatch_is_rejected():
    manifest = build_manifest()
    rows = _results(manifest)
    rows[0]["conditions"]["replay"]["target_probes"].pop()
    with pytest.raises(ValueError, match="probe set mismatch"):
        analyze(manifest, rows, replicates=100)
    rows = _results(manifest)
    rows[0]["conditions"]["prompt_neutral"]["extra_instruction_tokens"] = 21
    with pytest.raises(ValueError, match="unmatched extra token"):
        analyze(manifest, rows, replicates=100)


def test_legacy_contains_is_distinguished_from_actual_exact_and_prompt_contrast_is_paired():
    manifest = build_manifest()
    rows = _results(manifest)
    for row, record in zip(rows, manifest["records"]):
        for name, condition in row["conditions"].items():
            for group in ("target_probes", "retained_probes"):
                targets = {probe["probe_id"]: probe["target"] for probe in record[group]}
                for probe in condition[group]:
                    probe["greedy_text"] = "The code is" + targets[probe["probe_id"]] + "."
                    probe["greedy_exact"] = True  # Legacy runner's containment flag.
            if name == "prompt_forget":
                for group in ("target_probes", "retained_probes", "continuations"):
                    for probe in condition[group]:
                        probe["mean_log_probability"] += .2
                        probe["total_log_probability"] += .6
    original = copy.deepcopy(rows)
    report = analyze(manifest, rows, replicates=100)
    assert rows == original  # Never mutate the raw measurements.
    group = report["groups"]["all_evaluation_records"]
    present = group["conditions"]["present"]
    assert present["target_direct_contains"]["absolute"]["estimate"] == 1.0
    assert present["target_direct_exact"]["absolute"]["estimate"] == 0.0
    assert present["retained_contains"]["absolute"]["estimate"] == 1.0
    assert present["retained_exact"]["absolute"]["estimate"] == 0.0
    contrast = group["paired_prompt_forget_minus_neutral"]["target_mean_log_probability"]
    assert contrast["estimate"] == pytest.approx(.2)
    assert contrast["ci_low"] == pytest.approx(.2)
    assert contrast["ci_high"] == pytest.approx(.2)
