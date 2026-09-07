"""Compare the pre-registered family predictions with the measured audits.

For each family, every configuration is reduced to a verdict on the four
predicted properties, judged on the state-relative errors:

* frozen_static_exact / frozen_ledger_exact / frozen_full_exact: the receipt
  class transports the boundary receipt to the frozen-input counterfactual with
  error below ``exact_max_relative_error`` of the omitted state norm in every
  layer where the frozen difference is at least ``min_frozen_diff_rel_omitted``
  of the state (layers whose decay has driven the difference below that floor
  carry no information about transport);
* native_forcing_nonzero: the forcing share of the native difference exceeds
  ``native_forcing_nonzero_min_fraction_of_direct`` in the median layer.

The summary also carries the behavioral probes per condition so that the
single-channel families (Mamba-2, RWKV-7) can be contrasted with the hybrids.

Usage:
    python -m recurrent_families.summarize --root outputs/recurrent_families_v1 \
        --out gemma_sv/benchmarks/recurrent_families_summary_v1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PREDICTIONS = Path(__file__).resolve().parent / "benchmarks" / "family_predictions_v1.json"


def q(values) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {}
    return {"n": len(vals), "min": vals[0], "median": statistics.median(vals), "max": vals[-1]}


def exactness(report: Dict[str, Any], key: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Max state-relative error over informative layers, over all cuts > 0."""
    floor = thresholds["min_frozen_diff_rel_omitted"]
    errs: List[float] = []
    informative = 0
    for c, s in report["state"].items():
        if c == "0" or key not in s:
            continue
        for err, frac in zip(s[key], s["frozen_diff_rel_omitted"]):
            if frac >= floor:
                errs.append(err)
                informative += 1
    if not errs:
        return {"informative_layers": 0, "max_error_rel_omitted": None, "exact": None}
    mx = max(errs)
    return {
        "informative_layers": informative,
        "max_error_rel_omitted": mx,
        "exact": mx <= thresholds["exact_max_relative_error"],
        "clearly_inexact": mx >= thresholds["inexact_min_relative_error"],
    }


def summarize_family(reports: List[Dict[str, Any]], thresholds: Dict[str, float]) -> Dict[str, Any]:
    admitted = [r for r in reports if r["admission"]["qualified"]]
    verdicts: Dict[str, Any] = {}
    for name, key in (
        ("frozen_static_exact", "frozen_static_error_rel_omitted"),
        ("frozen_ledger_exact", "frozen_ledger_transport_error_rel_omitted"),
        ("frozen_full_exact", "frozen_full_transport_error_rel_omitted"),
    ):
        per = [exactness(r, key, thresholds) for r in admitted]
        maxes = [p["max_error_rel_omitted"] for p in per if p["max_error_rel_omitted"] is not None]
        verdicts[name] = {
            "max_error_rel_omitted": q(maxes),
            "exact_in_all_configurations": all(p["exact"] for p in per if p["exact"] is not None) if maxes else None,
            "clearly_inexact_in_all_configurations": all(p.get("clearly_inexact") for p in per if p["exact"] is not None) if maxes else None,
        }
    forcing_medians = []
    imprint_last = []
    recurrence_err = []
    last_cut = None
    for r in admitted:
        cuts = [int(c) for c in r["state"] if c != "0"]
        if not cuts:
            continue
        last_cut = max(cuts)
        s = r["state"][str(last_cut)]
        forcing_medians.append(statistics.median(s["residual_full_rel_direct"]))
        imprint_last.append(statistics.median(s["imprint_fraction"]))
        recurrence_err.append(max(s["recurrence_vs_model_error"]))
    verdicts["native_forcing_nonzero"] = {
        "median_forcing_share_at_last_cut": q(forcing_medians),
        "nonzero_in_all_configurations": all(f >= thresholds["native_forcing_nonzero_min_fraction_of_direct"] for f in forcing_medians) if forcing_medians else None,
    }
    behavior: Dict[str, Any] = {}
    for r in admitted:
        for c, conds in r["behavior"].items():
            for name, e in conds.items():
                label = name.split(":")[0] if ":" in name else name
                b = behavior.setdefault(c, {}).setdefault(label, defaultdict(list))
                b["lift"].append(e["lift_vs_omitted_nats"])
                b["rank"].append(e["victim"]["first_token_rank"])
                if "state_residual_vs_omitted" in e:
                    b["residual_mean"].append(e["state_residual_vs_omitted"]["mean"])
    behavior_summary = {
        c: {name: {"lift_nats": q(v["lift"]), "first_token_rank": q(v["rank"]), "state_residual_mean": q(v["residual_mean"])} for name, v in conds.items()}
        for c, conds in behavior.items()
    }
    return {
        "configurations": {"attempted": len(reports), "admitted": len(admitted)},
        "last_cut": last_cut,
        "recurrence_vs_model_error_max": max(recurrence_err) if recurrence_err else None,
        "replay_max_abs_diff": max((r["replay_check"]["max_abs_state_diff_vs_omitted"] for r in admitted), default=None),
        "imprint_fraction_median_at_last_cut": q(imprint_last),
        "verdicts": verdicts,
        "behavior": behavior_summary,
        "storage": admitted[0]["storage"] if admitted else {},
        "timing": admitted[0]["timing"] if admitted else {},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/recurrent_families_v1")
    ap.add_argument("--out", default="gemma_sv/benchmarks/recurrent_families_summary_v1.json")
    ap.add_argument("--min-frozen-diff-rel-omitted", type=float, default=1e-3)
    args = ap.parse_args(argv)
    predictions = json.loads(PREDICTIONS.read_text())
    thresholds = dict(predictions["thresholds"])
    thresholds["min_frozen_diff_rel_omitted"] = args.min_frozen_diff_rel_omitted
    families: Dict[str, Any] = {}
    for fam_dir in sorted(Path(args.root).iterdir()):
        if not fam_dir.is_dir():
            continue
        reports = [json.loads(p.read_text()) for p in sorted(fam_dir.glob("victim*-prefix*.json"))]
        reports = [r for r in reports if r.get("schema") == "recurrent-families-audit-v1"]
        if not reports:
            continue
        summary = summarize_family(reports, thresholds)
        pred = predictions["families"].get(fam_dir.name, {})
        agreement = {}
        for prop in ("frozen_static_exact", "frozen_ledger_exact", "frozen_full_exact"):
            measured = summary["verdicts"][prop]["exact_in_all_configurations"]
            agreement[prop] = {"predicted": pred.get(prop), "measured": measured, "agrees": (measured == pred.get(prop)) if measured is not None else None}
        measured_nf = summary["verdicts"]["native_forcing_nonzero"]["nonzero_in_all_configurations"]
        agreement["native_forcing_nonzero"] = {"predicted": pred.get("native_forcing_nonzero"), "measured": measured_nf, "agrees": (measured_nf == pred.get("native_forcing_nonzero")) if measured_nf is not None else None}
        summary["prediction_agreement"] = agreement
        summary["model"] = reports[0]["model"]
        summary["transition_class"] = reports[0]["transition_class"]
        families[fam_dir.name] = summary
        v = summary["verdicts"]
        print(
            f"{fam_dir.name:10s} ({summary['transition_class']}): {summary['configurations']['admitted']}/{summary['configurations']['attempted']} admitted; "
            f"static {v['frozen_static_exact']['max_error_rel_omitted'].get('max', float('nan')):.2e} "
            f"ledger {v['frozen_ledger_exact']['max_error_rel_omitted'].get('max', float('nan')):.2e} "
            f"full {v['frozen_full_exact']['max_error_rel_omitted'].get('max', float('nan')):.2e} (max err/state); "
            f"forcing share median {v['native_forcing_nonzero']['median_forcing_share_at_last_cut'].get('median', float('nan')):.3f}; "
            f"agreement {[k for k, a in agreement.items() if a['agrees'] is False] or 'all predictions hold'}"
        )
    out = {
        "schema": "recurrent-families-summary-v1",
        "predictions_file": str(PREDICTIONS.relative_to(Path(__file__).resolve().parents[1])),
        "thresholds": thresholds,
        "families": families,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
