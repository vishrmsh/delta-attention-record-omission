"""Generate manuscript-ready LaTeX tables directly from a complete analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LABELS = {
    "present": "Record present", "masked": "Attention masking", "static": "Static receipt",
    "ledger": "Diagonal ledger", "transport": "Receipt transport", "mask_transport": "Masking + transport",
    "replay": "Checkpoint replay", "never_stored": "Never stored", "never_stored_repeat": "Independent never stored",
    "prompt_forget": "Prompt forgetting", "prompt_neutral": "Neutral instruction",
}


def render_tables(report: dict, source_name: str, source_sha256: str) -> str:
    if report["analysis_status"] != "complete" or report["evidence_class"] != "model_evaluation":
        raise ValueError("manuscript tables require the complete real-model cohort")
    evaluation = report["groups"]["all_evaluation_records"]
    conditions = evaluation["conditions"]
    n, calibration = report["evaluation_records"], report["calibration_records"]
    attack = report["attack"]
    if attack["status"] != "complete" or set(conditions) != set(LABELS):
        raise ValueError("complete attack and all eleven frozen conditions required")

    def number(value: float, digits: int) -> str:
        return f"{0.0 if abs(value) < .5 * 10**(-digits) else value:.{digits}f}"

    def interval(values: dict, digits: int = 4, scale: float = 1.0) -> str:
        vals = [number(values[key] * scale, digits) for key in ("estimate", "ci_low", "ci_high")]
        return rf"${vals[0]}\;[{vals[1]},\,{vals[2]}]$"

    lines = [
        "% Automatically generated; do not hand-edit numerical values.",
        f"% Source: {source_name}", f"% Source SHA-256: {source_sha256}",
        f"% Manifest SHA-256: {report['manifest_sha256']}",
        "% This fragment needs only standard LaTeX; table* spans two IEEE columns.",
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Prospective synthetic Kimi Linear 48B-A3B 8-bit cohort: "
        + f"{n} evaluation contexts and {calibration} disjoint calibration contexts. "
        + r"Entries are condition-minus-never-stored differences with paired 95\% whole-context bootstrap intervals "
        + r"(2,000 replicates). Target scores average three fixed probes; retained scores average three retained records. "
        + r"Continuation NLL is token-weighted within each context and then averaged across contexts. "
        + r"Positive target differences indicate residual lift; positive NLL differences indicate worse continuation prediction. "
        + r"Unscaled differences are in nats/token. "
        + r"The retained column is multiplied by $10^3$ for legibility.}",
        r"\label{tab:kda-prospective-cohort}",
        r"\begin{tabular}{lccc}", r"\hline",
        r"Condition & Target log-probability $\Delta$ & Retained log-probability $\Delta\times10^3$ & Continuation NLL $\Delta$ \\",
        r"\hline",
    ]
    for name in LABELS:
        metrics = conditions[name]
        values = [interval(metrics[key]["paired_difference_vs_never_stored"], digits, scale)
                  for key, digits, scale in (("target_mean_log_probability", 4, 1),
                                             ("retained_mean_log_probability", 3, 1000),
                                             ("continuation_nll", 4, 1))]
        lines.append(LABELS[name] + " & " + " & ".join(values) + r" \\")
    qualified = report["qualification"]["qualified"]
    qlo, qhi = report["qualification"]["rate"]["wilson_95"]
    contrast = evaluation["paired_prompt_forget_minus_neutral"]
    token_cost = conditions["prompt_forget"]["extra_instruction_tokens"]["absolute"]["estimate"]
    lines += [r"\hline", r"\end{tabular}", r"\par\vspace{3pt}",
              r"\begin{minipage}{0.98\textwidth}\footnotesize",
              f"All {n} preselected evaluation contexts were retained; {qualified}/{n} qualified "
              + rf"(Wilson 95\% interval ${100*qlo:.1f}\%$--${100*qhi:.1f}\%$). "
              + "The forgetting and neutral instructions have exactly matched per-context token counts "
              + rf"(mean {token_cost:.2f} extra tokens). Forgetting minus neutral: target log probability "
              + interval(contrast["target_mean_log_probability"]) + ", retained log probability "
              + interval(contrast["retained_mean_log_probability"]) + ", and continuation NLL "
              + interval(contrast["continuation_nll"]) + " (all three unscaled, in nats/token). "
              + "The control is one fixed neutral string repeated/truncated to length; the contrast is wording-specific.",
              r"\end{minipage}", r"\end{table*}", ""]
    fpr = attack["evaluation_fpr"]
    flo, fhi = fpr["wilson_95"]
    lines += [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
              r"\caption{Fixed three-query candidate-scoring distinguisher on the same "
              + f"{n} evaluation contexts. "
              + rf"Nominal calibration FPR is {100*attack['target_fpr']:.1f}\%; empirical calibration FPR is "
              + rf"{100*attack['calibration_empirical_fpr']:.1f}\%. "
              + rf"The measured evaluation FPR is {100*fpr['estimate']:.1f}\% "
              + rf"(Wilson 95\% interval {100*flo:.1f}\%--{100*fhi:.1f}\%). "
              + r"AUC intervals resample paired whole contexts. The TPR bootstrap interval independently resamples "
              + r"calibration and evaluation contexts and recalibrates the threshold; the supplementary TPR Wilson "
              + r"interval is conditional on the deployed threshold.}",
              r"\label{tab:kda-prospective-attack}",
              r"\begin{tabular}{lccc}", r"\hline",
              r"Condition & AUC [95\% CI] & TPR, \% [two-stage 95\% CI] & TPR Wilson 95\% interval, \% \\",
              r"\hline"]
    for name in LABELS:
        data = attack["conditions"][name]
        lo, hi = data["tpr"]["wilson_95"]
        lines.append(LABELS[name] + " & " + interval(data["auc"])
                     + " & " + interval(data["tpr_with_calibration_uncertainty"], 1, 100)
                     + rf" & $[{100*lo:.1f},\,{100*hi:.1f}]$ \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\vspace{3pt}",
              r"\begin{minipage}{0.98\textwidth}\footnotesize",
              rf"The threshold is ${attack['threshold']:.6f}$ nats/token and is calibrated using only the "
              + f"{calibration} never-stored calibration scores. "
              + r"The attacker sees candidate-continuation log probabilities, not model state or evaluation "
              + r"never-stored references. Replay and independent rebuilding are measured null controls: their "
              + r"AUC is $0.5$ and their TPR equals FPR. Degenerate bootstrap intervals at zero or one describe "
              + r"the observed cohort; the Wilson intervals expose finite-sample rate uncertainty. "
              + r"This is one fixed attack on a synthetic cohort, not a general indistinguishability guarantee.",
              r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.analysis.read_bytes()
    tex = render_tables(json.loads(data), str(args.analysis), hashlib.sha256(data).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex)
    print(args.output)


if __name__ == "__main__":
    main()
