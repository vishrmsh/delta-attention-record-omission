"""Aggregate the forcing-sweep configurations into one source-free summary.

Reads every ``victim*-prefix*.json`` written by ``eval_forcing_sweep`` and
reports, per suffix length, the distribution over configurations and layers of
the record's imprint, the residual left by each receipt class, the forcing share
of the difference, the truncated-recompute curve, and the behavioral probes per
condition. The summary keeps only norms, ratios, ranks, probabilities, counts,
and timings; it never copies text.

Usage:
    python -m kimi_sv.summarize_forcing_sweep --root outputs/kimi_sv/forcing_sweep_v1 \
        --out gemma_sv/benchmarks/kda_forcing_sweep_summary_v1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def q(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return {}
    n = len(vals)
    return {
        "n": n,
        "min": vals[0],
        "p10": vals[max(0, int(0.10 * (n - 1)))],
        "median": statistics.median(vals),
        "p90": vals[min(n - 1, int(0.90 * (n - 1)))],
        "max": vals[-1],
    }


def load_reports(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*/victim*-prefix*.json")):
        report = json.loads(path.read_text())
        if report.get("schema") != "kimi-sv-forcing-sweep-v1":
            continue
        report["_path"] = str(path)
        by_source[report["source"]].append(report)
    return by_source


STATE_KEYS = (
    "imprint_fraction",
    "residual_static_rel_omitted",
    "residual_ledger_rel_omitted",
    "residual_full_rel_omitted",
    "residual_static_rel_direct",
    "residual_ledger_rel_direct",
    "residual_full_rel_direct",
    "decomposition_error",
)


def summarize_source(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    admitted = [r for r in reports if r.get("admission", {}).get("qualified")]
    cuts = sorted({int(c) for r in admitted for c in r["state"]})

    # --- state curves: pool layers x configurations per cut ---------------------
    state: Dict[str, Any] = {}
    for c in cuts:
        pooled: Dict[str, List[float]] = defaultdict(list)
        per_config_median: Dict[str, List[float]] = defaultdict(list)
        for r in admitted:
            s = r["state"].get(str(c))
            if not s:
                continue
            for key in STATE_KEYS:
                if key in s:
                    pooled[key].extend(s[key])
                    per_config_median[key].append(statistics.median(s[key]))
        state[str(c)] = {
            key: {"layers_pooled": q(pooled[key]), "config_medians": q(per_config_median[key])}
            for key in pooled
        }

    # --- truncated recompute: residual vs fraction of suffix recomputed --------
    trunc: Dict[str, Dict[str, Any]] = {}
    pooled_by_frac: Dict[str, List[float]] = defaultdict(list)
    for r in admitted:
        for k, per_cut in r.get("truncated", {}).items():
            for c, rec in per_cut.items():
                frac = round(int(k) / int(c), 4)
                trunc.setdefault(str(c), {}).setdefault(k, []).extend(rec["residual_rel_direct"])
                pooled_by_frac[str(frac)].extend(rec["residual_rel_direct"])
    truncated = {
        "by_cut_and_k": {c: {k: q(v) for k, v in ks.items()} for c, ks in trunc.items()},
        "by_fraction_recomputed": {f: q(v) for f, v in sorted(pooled_by_frac.items(), key=lambda kv: float(kv[0]))},
    }

    # --- behavior: per cut, per condition ---------------------------------------
    behavior: Dict[str, Any] = {}
    scatter: List[Dict[str, float]] = []
    for c in cuts:
        conds: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        for r in admitted:
            b = r["behavior"].get(str(c))
            if not b:
                continue
            for name, e in b.items():
                # normalize truncated names to their fraction of the cut
                label = name
                if name.startswith("trunc:") or name.startswith("mask_trunc:"):
                    prefix, k = name.split(":")
                    label = f"{prefix}@{round(int(k) / c, 3) if c else 0}"
                conds[label]["lift"].append(e["lift_vs_omitted_nats"])
                conds[label]["rank"].append(e["victim"]["first_token_rank"])
                conds[label]["kl"].append(e["first_token_kl_omitted_to_condition"])
                conds[label]["state_residual_max"].append(e["state_residual_vs_omitted"]["max"])
                conds[label]["state_residual_mean"].append(e["state_residual_vs_omitted"]["mean"])
                if "greedy_exact" in e:
                    conds[label]["greedy_exact"].append(1.0 if e["greedy_exact"] else 0.0)
                if "retained_lift_vs_omitted_nats" in e:
                    conds[label]["retained_lift"].append(e["retained_lift_vs_omitted_nats"])
                scatter.append(
                    {
                        "cut": c,
                        "condition": label,
                        "state_residual_mean": e["state_residual_vs_omitted"]["mean"],
                        "lift_nats": e["lift_vs_omitted_nats"],
                        "first_token_rank": e["victim"]["first_token_rank"],
                        "kl": e["first_token_kl_omitted_to_condition"],
                    }
                )
        behavior[str(c)] = {
            name: {
                "n": len(v["lift"]),
                "lift_nats": q(v["lift"]),
                "first_token_rank": q(v["rank"]),
                "rank_le_10_fraction": sum(1 for x in v["rank"] if x <= 10) / max(len(v["rank"]), 1),
                "first_token_kl": q(v["kl"]),
                "state_residual_mean": q(v["state_residual_mean"]),
                "greedy_exact_rate": (sum(v["greedy_exact"]) / len(v["greedy_exact"])) if v["greedy_exact"] else None,
                "retained_lift_nats": q(v["retained_lift"]) if v["retained_lift"] else None,
            }
            for name, v in conds.items()
        }

    # Reference-rebuild adversary: an observer who can re-run the model on the
    # visible transcript holds the exact never-stored state, so any nonzero
    # first-token KL between a deployed state and that reference reveals that
    # the record was ingested. Replay reproduces the reference exactly (KL 0).
    rebuild_adversary: Dict[str, Any] = {}
    for cond in ("mask", "mask_full", "full", "static", "ledger", "present"):
        kls = [
            e["first_token_kl_omitted_to_condition"]
            for r in admitted
            for c, b in r["behavior"].items()
            if c != "0"
            for name, e in b.items()
            if name == cond
        ]
        if kls:
            rebuild_adversary[cond] = {
                "n": len(kls),
                "kl_min": min(kls),
                "kl_median": statistics.median(kls),
                "kl_max": max(kls),
                "fraction_detectable_above_1e-6": sum(1 for k in kls if k > 1e-6) / len(kls),
                "fraction_detectable_above_1e-3": sum(1 for k in kls if k > 1e-3) / len(kls),
            }
    timing = {
        "prefill_seconds_per_token": q(r["timing"]["prefill_seconds_per_token"] for r in admitted),
        "transport_seconds_per_token_all_layers": q(
            r["timing"]["transport_seconds_per_token_all_layers"] for r in admitted if r["timing"].get("transport_seconds_per_token_all_layers")
        ),
        "seconds_per_configuration": q(r["seconds"] for r in admitted),
    }
    storage = admitted[0]["storage"] if admitted else {}
    return {
        "configurations": {
            "attempted": len(reports),
            "admitted": len(admitted),
            "victims": sorted({r["victim_index"] for r in reports}),
            "prefix_depths": sorted({r["prefix_depth"] for r in reports}),
            "files": [Path(r["_path"]).name for r in reports],
        },
        "cuts": cuts,
        "replay_check_max_abs_diff": max((r["replay_check"].get("max_abs_state_diff_vs_omitted", 0.0) for r in admitted), default=None),
        "decomposition_error_max": max(
            (max(r["state"][c]["decomposition_error"]) for r in admitted for c in r["state"] if "decomposition_error" in r["state"][c]),
            default=None,
        ),
        "state": state,
        "truncated": truncated,
        "behavior": behavior,
        "rebuild_adversary": rebuild_adversary,
        "scatter_residual_vs_lift": scatter,
        "timing": timing,
        "storage": storage,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/kimi_sv/forcing_sweep_v1")
    ap.add_argument("--out", default="gemma_sv/benchmarks/kda_forcing_sweep_summary_v1.json")
    args = ap.parse_args(argv)
    by_source = load_reports(Path(args.root))
    if not by_source:
        raise SystemExit(f"no sweep reports under {args.root}")
    summary = {
        "schema": "kda-forcing-sweep-summary-v1",
        "model": next(iter(by_source.values()))[0]["model"],
        "contains_source_text": False,
        "sources": {src: summarize_source(reps) for src, reps in sorted(by_source.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    for src, s in summary["sources"].items():
        cuts = s["cuts"]
        if not cuts:
            continue
        last = str(cuts[-1])
        imp = s["state"][last]["imprint_fraction"]["layers_pooled"]
        full = s["state"][last].get("residual_full_rel_direct", {}).get("layers_pooled", {})
        b = s["behavior"][last]
        print(
            f"{src}: {s['configurations']['admitted']}/{s['configurations']['attempted']} admitted; at L={last}: "
            f"imprint median {imp.get('median', float('nan')):.3e} (range {imp.get('min', 0):.1e}-{imp.get('max', 0):.1e}); "
            f"forcing/direct median {full.get('median', float('nan')):.3f}; "
            + "; ".join(
                f"{name} lift med {v['lift_nats'].get('median', float('nan')):+.2f} rank med {v['first_token_rank'].get('median', float('nan')):.0f}"
                for name, v in b.items()
                if name in ("present", "mask", "full", "mask_full", "static")
            )
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
