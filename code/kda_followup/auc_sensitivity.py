"""Exploratory off-diagonal AUC sensitivity for paired prospective contexts.

This diagnostic was added after the primary study completed. It does not
replace its frozen score, threshold, or ordinary empirical ROC AUC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze import _metrics
from .manifest import canonical_bytes, load_manifest


def comparison_matrix(positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if positive.shape != negative.shape or positive.ndim != 1 or len(positive) < 2:
        raise ValueError("paired off-diagonal AUC requires at least two paired observations")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("scores must be finite")
    return (positive[:, None] > negative[None, :]).astype(float) + .5 * (positive[:, None] == negative[None, :])


def offdiagonal_auc(positive, negative):
    matrix = comparison_matrix(positive, negative)
    n = len(matrix)
    return float((matrix.sum() - np.trace(matrix)) / (n * (n - 1)))


def run(manifest_path: Path, result_dir: Path, analysis_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    analysis = json.loads(analysis_path.read_text())
    if analysis["analysis_status"] != "complete" or analysis["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("sensitivity requires the completed matching primary analysis")
    contexts = [row for row in manifest["records"] if row["split"] == "evaluation"]
    rows = [json.loads((result_dir / (context["record_id"] + ".json")).read_text()) for context in contexts]
    for row in rows:
        expected = analysis["analysis_provenance"]["raw_result_canonical_sha256"][row["record_id"]]
        if hashlib.sha256(canonical_bytes(row)).hexdigest() != expected:
            raise ValueError("raw result changed since primary analysis")
    n = len(rows)
    seed = manifest["protocol"]["bootstrap"]["seed"] + 501
    replicates = manifest["protocol"]["bootstrap"]["replicates"]
    negative = np.array([_metrics(row["conditions"]["never_stored"])["target_mean_log_probability"] for row in rows])
    indices = np.random.default_rng(seed).integers(0, n, (replicates, n))
    weights = np.array([np.bincount(index, minlength=n) for index in indices])
    denominator = n * n - (weights * weights).sum(axis=1)
    usable = denominator > 0
    weights, denominator = weights[usable], denominator[usable]
    conditions = {}
    for name in manifest["protocol"]["conditions"]:
        positive = np.array([_metrics(row["conditions"][name])["target_mean_log_probability"] for row in rows])
        matrix = comparison_matrix(positive, negative)
        diagonal = np.diag(matrix).copy()
        ordinary = float(matrix.mean())
        np.fill_diagonal(matrix, 0)
        point = float(matrix.sum() / (n * (n - 1)))
        bootstrap = np.einsum("bi,ij,bj->b", weights, matrix, weights) / denominator
        lower, upper = np.quantile(bootstrap, [.025, .975])
        conditions[name] = {
            "primary_empirical_auc": ordinary, "matched_diagonal_mean_win": float(diagonal.mean()),
            "independent_pair_offdiagonal_auc": point,
            "offdiagonal_record_bootstrap_95": [float(lower), float(upper)],
            "primary_minus_offdiagonal_auc": ordinary - point,
        }
    return {
        "schema": "kda-auc-pairing-sensitivity-v1", "status": "complete_exploratory_sensitivity",
        "manifest_sha256": manifest["manifest_sha256"], "n_evaluation_records": n,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_seed": seed, "bootstrap_replicates": replicates, "usable_replicates": int(usable.sum()),
        "selection": "Added after the primary analysis to inspect near-null AUC deviations; does not replace its frozen score, threshold, or AUC.",
        "estimator": "Average positive-condition versus never-stored comparison over distinct source records i != j. Bootstrap resamples whole records; comparisons between copies of the same original source record are excluded.",
        "reason": "Ordinary empirical AUC includes n matched pairs in n^2 comparisons. These within-context wins can produce O(1/n) differences from the AUC for independent positive/negative contexts.",
        "conditions": conditions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.manifest, args.results, args.analysis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
