"""Render the sweep and family summaries as LaTeX macros for the KDA paper.

Every number the manuscript quotes from the suffix-length sweep
(``kda_forcing_sweep_summary_v1.json``) and the cross-family audit
(``recurrent_families_summary_v1.json``) is emitted here as a
``\\providecommand``, so the prose never carries a hand-typed value. The output
is a generated file and must not be edited.

Usage:
    python -m kimi_sv.build_kda_sweep_macros \
        --sweep gemma_sv/benchmarks/kda_forcing_sweep_summary_v1.json \
        --families gemma_sv/benchmarks/recurrent_families_summary_v1.json \
        --out kda_paper/paper/kda_sweep_macros.tex
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def sci(x: float, digits: int = 1) -> str:
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 1000:
        # Plain decimals read better than 3.0\times10^{0} in running text.
        return f"{x:.{max(digits, 2 - int(math.floor(math.log10(abs(x)))))}f}" if abs(x) < 1 else f"{x:.{digits}f}"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    if round(mant, digits) >= 10:
        mant /= 10
        exp += 1
    return f"{mant:.{digits}f}\\times10^{{{exp}}}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def integer(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def macros_for_sweep(summary: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for source, tag in (("synthetic", "Syn"), ("tofu", "Tofu")):
        s = summary["sources"].get(source)
        if not s:
            continue
        cfg = s["configurations"]
        out[f"Sweep{tag}Admitted"] = str(cfg["admitted"])
        out[f"Sweep{tag}Attempted"] = str(cfg["attempted"])
        out[f"Sweep{tag}Victims"] = str(len(cfg["victims"]))
        out[f"Sweep{tag}PrefixDepths"] = ", ".join(str(p) for p in cfg["prefix_depths"])
        cuts = s["cuts"]
        last = str(cuts[-1])
        first_nonzero = str([c for c in cuts if c > 0][0])
        out[f"Sweep{tag}MaxCut"] = integer(cuts[-1])
        out[f"Sweep{tag}NumCuts"] = str(len(cuts))
        st = s["state"]
        pooled = lambda cut, key: st[cut][key]["layers_pooled"]
        out[f"Sweep{tag}ImprintMedianZero"] = pct(pooled("0", "imprint_fraction")["median"])
        out[f"Sweep{tag}ImprintMedianLast"] = pct(pooled(last, "imprint_fraction")["median"])
        out[f"Sweep{tag}ImprintMinLast"] = pct(pooled(last, "imprint_fraction")["min"])
        out[f"Sweep{tag}ImprintMaxLast"] = pct(pooled(last, "imprint_fraction")["max"])
        out[f"Sweep{tag}StaticResidualMedianLast"] = pct(pooled(last, "residual_static_rel_omitted")["median"])
        out[f"Sweep{tag}LedgerResidualMedianLast"] = pct(pooled(last, "residual_ledger_rel_omitted")["median"])
        out[f"Sweep{tag}FullResidualMedianLast"] = pct(pooled(last, "residual_full_rel_omitted")["median"])
        out[f"Sweep{tag}ForcingShareMedianFirst"] = num(pooled(first_nonzero, "residual_full_rel_direct")["median"])
        out[f"Sweep{tag}ForcingShareMedianLast"] = num(pooled(last, "residual_full_rel_direct")["median"])
        out[f"Sweep{tag}ForcingShareMinLast"] = num(pooled(last, "residual_full_rel_direct")["min"], 3)
        out[f"Sweep{tag}ForcingShareMaxLast"] = num(pooled(last, "residual_full_rel_direct")["max"], 3)
        out[f"Sweep{tag}DecompErrMax"] = sci(s["decomposition_error_max"])
        out[f"Sweep{tag}ReplayMaxDiff"] = sci(s["replay_check_max_abs_diff"]) if s["replay_check_max_abs_diff"] else "0"
        frac = s["truncated"]["by_fraction_recomputed"]
        for key, name in (("0.5", "Half"), ("0.25", "Quarter"), ("0.125", "Eighth")):
            if key in frac:
                out[f"Sweep{tag}Trunc{name}Median"] = num(frac[key]["median"])
        b = s["behavior"][last]
        for cond, name in (("present", "Present"), ("full", "Full"), ("static", "Static"), ("ledger", "Ledger"), ("mask", "Mask"), ("mask_full", "MaskFull")):
            if cond in b:
                out[f"Sweep{tag}Lift{name}MedianLast"] = num(b[cond]["lift_nats"]["median"])
                out[f"Sweep{tag}Rank{name}MedianLast"] = integer(b[cond]["first_token_rank"]["median"])
                out[f"Sweep{tag}RankLeTen{name}Last"] = pct(b[cond]["rank_le_10_fraction"], 0)
                out[f"Sweep{tag}KL{name}MedianLast"] = sci(b[cond]["first_token_kl"]["median"], 1)
                out[f"Sweep{tag}Residual{name}MedianLast"] = pct(b[cond]["state_residual_mean"]["median"])
                if b[cond].get("greedy_exact_rate") is not None:
                    out[f"Sweep{tag}Greedy{name}RateLast"] = pct(b[cond]["greedy_exact_rate"], 0)
        b0 = s["behavior"]["0"]
        out[f"Sweep{tag}LiftPresentMedianZero"] = num(b0["present"]["lift_nats"]["median"])
        for cond, name in (("mask", "Mask"), ("mask_full", "MaskFull"), ("full", "Full")):
            ra = s.get("rebuild_adversary", {}).get(cond)
            if ra:
                out[f"Sweep{tag}RebuildKL{name}Min"] = sci(ra["kl_min"], 1)
                out[f"Sweep{tag}RebuildKL{name}Median"] = sci(ra["kl_median"], 1)
                out[f"Sweep{tag}RebuildDetectable{name}Pct"] = pct(ra["fraction_detectable_above_1e-6"], 0)
                out[f"Sweep{tag}RebuildDetectableMilli{name}Pct"] = pct(ra["fraction_detectable_above_1e-3"], 0)
                out[f"Sweep{tag}RebuildN{name}"] = str(ra["n"])
        t = s["timing"]
        out[f"Sweep{tag}PrefillMsPerToken"] = num(1000 * t["prefill_seconds_per_token"]["median"], 2)
        if t["transport_seconds_per_token_all_layers"]:
            out[f"Sweep{tag}TransportMsPerToken"] = num(1000 * t["transport_seconds_per_token_all_layers"]["median"], 1)
        out[f"Sweep{tag}MinutesPerConfiguration"] = num(t["seconds_per_configuration"]["median"] / 60, 1)
        sto = s["storage"]
        if sto:
            out[f"Sweep{tag}CheckpointMiB"] = num(sto["checkpoint_bytes"] / 2**20, 1)
            lb = sto["transition_log_bytes"]
            out[f"Sweep{tag}LogKiBPerToken"] = num(lb["per_token_all_layers_native_dtype"] / 1024, 0)
            out[f"Sweep{tag}LogMiBPerThousandTokens"] = num(1000 * lb["per_token_all_layers_native_dtype"] / 2**20, 0)
            out[f"Sweep{tag}LogTokensPerCheckpoint"] = integer(sto["checkpoint_bytes"] / lb["per_token_all_layers_native_dtype"])
    return out


def macros_for_sampling(summary: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for source, tag in (("synthetic", "Syn"), ("tofu", "Tofu")):
        s = summary["sources"].get(source)
        if not s:
            continue
        out[f"Samp{tag}K"] = str(s["samples_per_condition"])
        out[f"Samp{tag}Configs"] = str(s["configurations"]["attempted"])
        for c, conds in s["by_cut"].items():
            ctag = "Zero" if c == "0" else "Last"
            for cond, name in (("present", "Present"), ("omitted", "Omitted"), ("mask", "Mask"), ("static", "Static"), ("mask_static", "MaskStatic")):
                v = conds.get(cond)
                if not v:
                    continue
                out[f"Samp{tag}{name}Any{ctag}"] = str(v["configurations_with_any_hit"])
                out[f"Samp{tag}{name}Hits{ctag}"] = str(v["total_hits"])
                out[f"Samp{tag}Draws{ctag}"] = f"{v['total_draws']:,}".replace(",", "{,}")
                out[f"Samp{tag}{name}Configs{ctag}"] = str(v["configurations"])
                out[f"Samp{tag}{name}Pooled{ctag}"] = pct(v["pooled_hit_rate"], 1)
                out[f"Samp{tag}{name}MaxRate{ctag}"] = pct(v["per_configuration_hit_rate"]["max"], 0)
    return out


def macros_for_mla(report: Dict[str, Any]) -> Dict[str, str]:
    sources = report.get("sources") or {report["source"]: report}
    configs = sum(s["n_configurations"] for s in sources.values())
    diffs = [
        max(v.get("max_abs_diff_keys", float("inf")), v.get("max_abs_diff_values", float("inf")))
        for s in sources.values()
        for c in s["configurations"]
        for v in c["mla_layers"].values()
    ]
    names = {"synthetic": "synthetic", "tofu": "TOFU"}
    return {
        "MlaConfigs": str(configs),
        "MlaSources": " and ".join(names.get(k, k) for k in sorted(sources)),
        "MlaLayers": str(report["n_mla_layers"]),
        "MlaMaxAbsDiff": sci(max(diffs)) if diffs else "n/a",
        "MlaAllEqual": "yes" if report["all_equal"] else "no",
    }


def macros_for_families(summary: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    names = {"mamba2": "Mamba", "falcon_h1": "Falcon", "rwkv7": "Rwkv"}
    for fam, tag in names.items():
        f = summary["families"].get(fam)
        if not f:
            continue
        v = f["verdicts"]
        out[f"Fam{tag}Admitted"] = str(f["configurations"]["admitted"])
        out[f"Fam{tag}Attempted"] = str(f["configurations"]["attempted"])
        out[f"Fam{tag}LastCut"] = integer(f["last_cut"] or 0)
        for prop, name in (("frozen_static_exact", "Static"), ("frozen_ledger_exact", "Ledger"), ("frozen_full_exact", "Full")):
            m = v[prop]["max_error_rel_omitted"]
            if m:
                out[f"Fam{tag}{name}ErrMax"] = sci(m["max"])
        nf = v["native_forcing_nonzero"]["median_forcing_share_at_last_cut"]
        if nf:
            out[f"Fam{tag}ForcingMedian"] = num(nf["median"])
            out[f"Fam{tag}ForcingMin"] = num(nf["min"])
            out[f"Fam{tag}ForcingMax"] = num(nf["max"])
        if f.get("recurrence_vs_model_error_max") is not None:
            out[f"Fam{tag}RecurrenceErrMax"] = sci(f["recurrence_vs_model_error_max"]) if f["recurrence_vs_model_error_max"] else "0"
        out[f"Fam{tag}ReplayMaxDiff"] = sci(f["replay_max_abs_diff"]) if f["replay_max_abs_diff"] else "0"
        imp = f["imprint_fraction_median_at_last_cut"]
        if imp:
            out[f"Fam{tag}ImprintMedianLast"] = pct(imp["median"])
        last = str(f["last_cut"])
        beh = f["behavior"].get(last, {})
        for cond, name in (("present", "Present"), ("full", "Full"), ("ledger", "Ledger"), ("static", "Static")):
            if cond in beh and beh[cond]["lift_nats"]:
                out[f"Fam{tag}Lift{name}MedianLast"] = num(beh[cond]["lift_nats"]["median"])
                out[f"Fam{tag}Rank{name}MedianLast"] = integer(beh[cond]["first_token_rank"]["median"])
        agree = f["prediction_agreement"]
        out[f"Fam{tag}PredictionsHold"] = "yes" if all(a["agrees"] for a in agree.values() if a["agrees"] is not None) else "no"
    thr = summary["thresholds"]
    out["FamExactThreshold"] = sci(thr["exact_max_relative_error"], 0)
    return out


EXPECTED_FAMILY_MACROS = tuple(
    f"Fam{tag}{name}"
    for tag in ("Mamba", "Falcon", "Rwkv")
    for name in (
        "Admitted", "Attempted", "LastCut", "StaticErrMax", "LedgerErrMax", "FullErrMax",
        "ForcingMedian", "ForcingMin", "ForcingMax", "RecurrenceErrMax", "ReplayMaxDiff",
        "ImprintMedianLast", "LiftPresentMedianLast", "LiftFullMedianLast",
        "RankPresentMedianLast", "RankFullMedianLast", "PredictionsHold",
    )
)


def fill_pending(macros: Dict[str, str], *, allow_pending: bool) -> None:
    missing = [name for name in EXPECTED_FAMILY_MACROS if name not in macros]
    if missing and not allow_pending:
        raise SystemExit(f"{len(missing)} family macros have no data yet: {missing[:6]}...")
    for name in missing:
        macros[name] = "PENDING"


def render(macros: Dict[str, str]) -> str:
    lines = ["% GENERATED FILE -- DO NOT EDIT. Produced by kimi_sv.build_kda_sweep_macros from the sweep and family summaries."]
    for name in sorted(macros):
        lines.append(f"\\providecommand{{\\{name}}}{{{macros[name]}}}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="gemma_sv/benchmarks/kda_forcing_sweep_summary_v1.json")
    ap.add_argument("--families", default="gemma_sv/benchmarks/recurrent_families_summary_v1.json")
    ap.add_argument("--sampling", default="gemma_sv/benchmarks/kda_masked_sampling_summary_v1.json")
    ap.add_argument("--mla", default="gemma_sv/benchmarks/kda_mla_logical_equality_v1.json")
    ap.add_argument("--out", default="kda_paper/paper/kda_sweep_macros.tex")
    ap.add_argument("--allow-pending", action="store_true", help="emit PENDING for families without data (never for a release)")
    args = ap.parse_args(argv)
    macros: Dict[str, str] = {}
    if Path(args.sweep).exists():
        macros.update(macros_for_sweep(json.loads(Path(args.sweep).read_text())))
    if Path(args.families).exists():
        macros.update(macros_for_families(json.loads(Path(args.families).read_text())))
    if Path(args.sampling).exists():
        macros.update(macros_for_sampling(json.loads(Path(args.sampling).read_text())))
    if Path(args.mla).exists():
        macros.update(macros_for_mla(json.loads(Path(args.mla).read_text())))
    fill_pending(macros, allow_pending=args.allow_pending)
    Path(args.out).write_text(render(macros))
    print(f"wrote {len(macros)} macros to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
