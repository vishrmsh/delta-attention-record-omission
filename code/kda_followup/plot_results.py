"""Publication-size forest plot from completed paired cohort estimates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(report_path: Path, output_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    import numpy as np

    report = json.loads(report_path.read_text())
    if report["analysis_status"] != "complete":
        raise ValueError("Refusing to plot a partial cohort as final evidence")
    conditions = report["groups"]["all_evaluation_records"]["conditions"]
    order = ["present", "prompt_forget", "prompt_neutral", "masked", "static",
             "ledger", "transport", "mask_transport", "replay", "never_stored_repeat"]
    labels = ["Record present", "Prompt: forget", "Prompt: neutral (matched tokens)",
              "Mask MLA record rows", "Static receipt", "Diagonal ledger",
              "Transported receipt", "Mask + transport", "Checkpoint replay", "Repeat never stored"]
    panels = [("target_mean_log_probability", "Deleted target readout", "Log-probability difference (nats/token)"),
              ("retained_mean_log_probability", "Three retained records", "Log-probability difference (nats/token)"),
              ("continuation_nll", "Held-out continuation", "NLL difference (nats/token)")]
    colors = ["#5c6972", "#95562a", "#b08b69", "#38748e", "#728393", "#728393",
              "#728393", "#38748e", "#217055", "#217055"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.8), sharey=True)
    y = np.arange(len(order))
    for ax, (metric, title, xlabel) in zip(axes, panels):
        for i, name in enumerate(order):
            values = conditions[name][metric]["paired_difference_vs_never_stored"]
            point, lo, hi = values["estimate"], values["ci_low"], values["ci_high"]
            ax.plot([lo, hi], [i, i], color=colors[i], lw=1.5, solid_capstyle="round")
            ax.scatter([point], [i], color=colors[i], s=30, zorder=3)
        ax.axvline(0, color="#273c47", lw=.8, ls="--", zorder=0)
        ax.grid(axis="y", color="#dce2e4", linewidth=.5)
        ax.set_title(title, loc="left", weight="bold", pad=16)
        ax.set_xlabel(xlabel, labelpad=10, fontsize=9)
        ax.tick_params(axis="y", length=0, pad=9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.margins(x=.14)
    axes[0].invert_yaxis()
    axes[2].xaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[2].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.suptitle("Independent Kimi cohort: each condition compared with never stored",
                 x=.035, ha="left", y=.98, fontsize=14, weight="bold")
    fig.text(.035, .035,
             f"n = {report['evaluation_records']} frozen evaluation contexts; 95% paired whole-record bootstrap intervals. "
             "Qualification failures included.\n"
             "Positive target differences indicate residual readout. Retained and continuation differences measure drift; "
             "continuations use a six-passage synthetic probe bank.", fontsize=8.5, color="#43545e")
    fig.subplots_adjust(left=.255, right=.98, top=.85, bottom=.18, wspace=.32)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output_stem.with_suffix("." + suffix), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    plot(args.report, args.output_stem)


if __name__ == "__main__":
    main()
