"""Aggregate the sampling attack on masked and receipt-corrected Kimi states.

For each source and suffix cut, reports per condition the number of
configurations, the fraction of configurations with at least one sampled hit
(Leak@k at the configuration level, the endpoint the Gemma attack suite uses),
the pooled per-sample hit rate, and the distribution of per-configuration hit
rates. No sampled text is read or stored.

Usage:
    python -m kimi_sv.summarize_masked_sampling --root outputs/kimi_sv/masked_sampling_v1 \
        --out gemma_sv/benchmarks/kda_masked_sampling_summary_v1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def q(values) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}
    return {"n": len(vals), "min": vals[0], "median": statistics.median(vals), "max": vals[-1]}


def summarize_source(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_cut: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    k = None
    for r in reports:
        k = r["protocol"]["samples_per_condition"]
        for c, conds in r["sampling"].items():
            for name, e in conds.items():
                per_cut[c][name]["hits"].append(e["hits"])
                per_cut[c][name]["any"].append(1.0 if e["any_hit"] else 0.0)
                per_cut[c][name]["rate"].append(e["hit_rate"])
    out: Dict[str, Any] = {}
    for c, conds in per_cut.items():
        out[c] = {}
        for name, v in conds.items():
            n_cfg = len(v["hits"])
            out[c][name] = {
                "configurations": n_cfg,
                "total_draws": n_cfg * k,
                "total_hits": int(sum(v["hits"])),
                "configurations_with_any_hit": int(sum(v["any"])),
                "leak_at_k_configuration_fraction": sum(v["any"]) / n_cfg,
                "pooled_hit_rate": sum(v["hits"]) / (n_cfg * k),
                "per_configuration_hit_rate": q(v["rate"]),
            }
    return {
        "configurations": {"attempted": len(reports), "files": [f"victim{r['victim_index']:03d}-prefix{r['prefix_depth']:03d}" for r in reports]},
        "samples_per_condition": k,
        "protocol": reports[0]["protocol"] if reports else {},
        "by_cut": out,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/kimi_sv/masked_sampling_v1")
    ap.add_argument("--out", default="gemma_sv/benchmarks/kda_masked_sampling_summary_v1.json")
    args = ap.parse_args(argv)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(args.root).glob("*/victim*-prefix*.json")):
        r = json.loads(path.read_text())
        if r.get("schema") == "kimi-sv-masked-sampling-v1":
            by_source[r["source"]].append(r)
    if not by_source:
        raise SystemExit(f"no sampling reports under {args.root}")
    summary = {
        "schema": "kda-masked-sampling-summary-v1",
        "model": next(iter(by_source.values()))[0]["model"],
        "contains_source_text": False,
        "sources": {src: summarize_source(reps) for src, reps in sorted(by_source.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    for src, s in summary["sources"].items():
        for c, conds in sorted(s["by_cut"].items(), key=lambda kv: int(kv[0])):
            print(
                f"{src} L={c}: "
                + "; ".join(
                    f"{name} any-hit {v['configurations_with_any_hit']}/{v['configurations']} pooled {100 * v['pooled_hit_rate']:.1f}%"
                    for name, v in conds.items()
                )
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
