"""Render the suffix-length sweep and the cross-family audit as paper figures.

Inputs are the source-free summaries written by ``summarize_forcing_sweep`` and
``recurrent_families.summarize``, plus the saved replay timing report. SVG, PDF,
and PNG are written with stripped metadata. Run from the artifact root with
``PYTHONPATH=code``; use a fresh output directory outside the artifact.

Usage:
    python -m kimi_sv.make_forcing_sweep_figures \
        --sweep benchmarks/kda_forcing_sweep_summary_v1.json \
        --families benchmarks/recurrent_families_summary_v1.json \
        --out-dir /tmp/kda-figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11.3,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11.3,
        "xtick.labelsize": 11.3,
        "ytick.labelsize": 11.3,
        "legend.fontsize": 11.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    }
)

# Semantic grammar (Section 6 of the style guide).
TARGET = "#B44A1B"      # record present / the victim's imprint
RETAINED = "#006DAA"    # retained behavior
BEHAVIORAL = "#007A5E"  # never stored / omitted output (behavioral reference)
NUMERICAL = "#6F4C9B"   # replay / refit (numerical reference)
OPERATION = "#8A5A00"   # transport, mask, recompute
EXCLUDED = "#5F6368"

CONDITION_STYLE = {
    "present": dict(color=TARGET, marker="D", ls="-", label="record present"),
    "omitted": dict(color=BEHAVIORAL, marker="^", ls="--", label="never stored"),
    # Separate receipt classes by hue as well as their existing markers/styles.
    "static": dict(color="#0072B2", marker="x", ls=":", label="static receipt"),
    "ledger": dict(color="#CC79A7", marker="v", ls="-.", label="diagonal ledger"),
    "full": dict(color=OPERATION, marker="s", ls="-", label="full-matrix receipt"),
    "mask": dict(color=EXCLUDED, marker=">", ls="-", label="attention rows masked"),
    "mask_full": dict(color=NUMERICAL, marker="P", ls="-", label="masked + full-matrix"),
}


def _clean_svg(path: Path, description: str) -> None:
    """Make the SVG self-contained and deterministic: no DOCTYPE, no RDF
    metadata block, no xlink namespace (plain ``href`` is valid SVG 2), and a
    single <desc> carrying the accessibility description."""
    import re

    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!DOCTYPE[^>]*>\s*", "", text)
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.S)
    text = text.replace(' xmlns:xlink="http://www.w3.org/1999/xlink"', "").replace("xlink:href", "href")
    if "<desc>" not in text:
        text = re.sub(r"(<svg[^>]*>)", rf"\1\n <desc>{description}</desc>", text, count=1)
    path.write_text(text, encoding="utf-8")


def save(fig: plt.Figure, out_dir: Path, stem: str, description: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{stem}.svg", metadata={"Date": None, "Description": description}, **common)
    _clean_svg(out_dir / f"{stem}.svg", description)
    fig.savefig(out_dir / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}, **common)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, metadata={"Software": None}, **common)
    plt.close(fig)


def _series(state: Dict[str, Any], key: str, stat: str = "median") -> tuple[List[int], List[float], List[float], List[float]]:
    cuts = sorted(int(c) for c in state if key in state[c])
    med, lo, hi = [], [], []
    for c in cuts:
        pooled = state[str(c)][key]["layers_pooled"]
        med.append(pooled["median"])
        lo.append(pooled.get("p10", pooled["min"]))
        hi.append(pooled.get("p90", pooled["max"]))
    return cuts, med, lo, hi


def figure_sweep(summary: Dict[str, Any], out_dir: Path) -> None:
    src = summary["sources"].get("synthetic") or next(iter(summary["sources"].values()))
    n_cfg = src["configurations"]["admitted"]
    n_att = src["configurations"]["attempted"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 4.1))

    # (a) imprint and receipt residuals vs suffix length
    ax = axes[0]
    cuts, med, lo, hi = _series(src["state"], "imprint_fraction")
    x = [max(c, 1) for c in cuts]
    ax.fill_between(x, lo, hi, color=TARGET, alpha=0.15, lw=0)
    ax.plot(x, med, color=TARGET, marker="D", ms=3.5, lw=1.4, label="record present")
    for key, cond in (("residual_static_rel_omitted", "static"), ("residual_ledger_rel_omitted", "ledger"), ("residual_full_rel_omitted", "full")):
        c2, m2, _, _ = _series(src["state"], key)
        st = CONDITION_STYLE[cond]
        ax.plot([max(c, 1) for c in c2], m2, color=st["color"], marker=st["marker"], ls=st["ls"], ms=3.5, lw=1.2, label=st["label"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("suffix tokens")
    ax.set_ylabel("residual / state norm")
    ax.set_title("(a) State residual")

    # Denominators (admitted/attempted configurations) are stated in the caption.

    # (b) forcing-to-difference norm ratio vs suffix length
    ax = axes[1]
    cuts, med, lo, hi = _series(src["state"], "residual_full_rel_direct")
    x = [max(c, 1) for c in cuts]
    ax.fill_between(x, lo, hi, color=OPERATION, alpha=0.15, lw=0)
    ax.plot(x, med, color=OPERATION, marker="s", ms=3.5, lw=1.4, label="forcing norm / difference norm")
    ax.axhline(1.0, color=EXCLUDED, lw=0.8, ls=":")
    ax.text(x[0], 1.03, "equal norms", fontsize=11.3, color=EXCLUDED, va="bottom")
    ax.axhline(0.0, color=BEHAVIORAL, lw=0.8, ls="--")
    ax.text(x[0], 0.03, "exact transport", fontsize=11.3, color=BEHAVIORAL, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, max(1.15, max(hi) * 1.05))
    ax.set_xlabel("suffix tokens")
    ax.set_ylabel("forcing norm ratio")
    ax.set_title("(b) Forcing norm ratio")

    # (c) truncated recompute
    ax = axes[2]
    frac = src["truncated"]["by_fraction_recomputed"]
    xs = sorted(float(f) for f in frac)
    ys = [frac[str(f) if str(f) in frac else f"{f}"]["median"] if (str(f) in frac) else None for f in xs]
    # keys are strings of rounded floats; rebuild robustly
    pts = sorted((float(k), v["median"], v.get("p10", v["min"]), v.get("p90", v["max"])) for k, v in frac.items())
    if pts:
        fx = [0.0] + [p[0] for p in pts] + [1.0]
        fm = [src["state"][str(src["cuts"][-1])]["residual_full_rel_direct"]["layers_pooled"]["median"]] + [p[1] for p in pts] + [0.0]
        ax.plot(fx, fm, color=OPERATION, marker="D", ms=3.5, lw=1.4, label="recompute + transport")
        ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts], color=OPERATION, alpha=0.15, lw=0)
        ax.plot([0, 1], [fm[0], 0], color=EXCLUDED, lw=0.8, ls=":", label="linear reference")
    ax.axhline(0.0, color=NUMERICAL, lw=0.8, ls="-.")
    ax.text(0.02, 0.03, "0 = full replay", fontsize=11.3, color=NUMERICAL, va="bottom")
    ax.set_xlabel("fraction recomputed")
    ax.set_ylabel("remaining forcing ratio")
    ax.set_title("(c) Partial recomputation")
    ax.text(0.10, 0.97, "recompute +\ntransport", transform=ax.transAxes, va="top", fontsize=11.3)
    ax.text(0.35, 0.20, "linear", color=EXCLUDED, fontsize=11.3)
    axes[0].set_xticks([1, 16, 256, 4096], ["0", "16", "256", "4096"])
    axes[0].text(4, 0, "//", transform=axes[0].get_xaxis_transform(), ha="center", va="center", fontsize=11.3, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    axes[1].set_xticks([16, 256, 4096], ["16", "256", "4096"])
    axes[2].set_xticks([0, 0.5, 1])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=11.3)
    fig.tight_layout(w_pad=1.1, rect=(0, 0, 1, 0.81))
    save(
        fig,
        out_dir,
        "kda_suffix_sweep",
        "In the audited Kimi Delta Attention configurations, recurrent-state residuals persist across 4,096 suffix tokens. The forcing-to-difference norm ratio compares residual and direct-difference magnitudes. Partial-recomputation residuals are plotted with a linear reference, without implying uniformly distributed forcing.",
    )


def figure_behavior(summary: Dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.1))
    groups = [
        (("present",), TARGET, "D", "record present"),
        (("static", "ledger", "full"), OPERATION, "s", "receipt-corrected, any class"),
        (("mask",), EXCLUDED, ">", "attention masked"),
        (("mask_full",), NUMERICAL, "P", "masked + full-matrix"),
    ]
    for ax, source in zip(axes, ("synthetic", "tofu")):
        src = summary["sources"].get(source)
        if not src:
            ax.set_visible(False)
            continue
        pts = src["scatter_residual_vs_lift"]
        selected = [p for p in pts if p["condition"] in CONDITION_STYLE]
        positive = [p["state_residual_mean"] for p in selected if p["state_residual_mean"] > 0]
        # A separated left-edge location denotes actual zero; positive values
        # keep their original coordinates on the log axis. No averaging.
        zero_x = min(positive) / 4
        for conditions, color, marker, label in groups:
            rows = [p for p in pts if p["condition"] in conditions]
            ax.scatter([p["state_residual_mean"] if p["state_residual_mean"] > 0 else zero_x for p in rows],
                       [p["lift_nats"] for p in rows], s=17, color=color,
                       marker=marker, label=label, alpha=0.8,
                       linewidths=0.35, edgecolors="white")
        omitted = [p for p in pts if p["condition"] == "omitted"]
        ax.scatter([zero_x] * len(omitted), [p["lift_nats"] for p in omitted],
                   s=28, color=BEHAVIORAL, marker="^", zorder=4, label="never-stored state (x = 0)")
        ax.axhline(0, color=BEHAVIORAL, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(zero_x / 1.3, max(positive) * 1.2)
        ax.set_xticks([zero_x, 0.1, 1], ["0", "0.1", "1"])
        ax.minorticks_off()
        # Mark the visual gap separating zero from the positive log scale.
        gap_x = (zero_x * min(positive)) ** 0.5
        ax.text(gap_x, 0, "//", transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=11.3,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        ax.set_xlabel("recurrent residual / state norm")
        ax.set_ylabel("Target lift (nats/token)")
        ax.set_title(f"({'a' if source == 'synthetic' else 'b'}) {source.capitalize() if source == 'synthetic' else 'TOFU'}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=11.3, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(w_pad=1.1, rect=(0, 0, 1, 0.75))
    save(fig, out_dir, "kda_residual_vs_leakage",
         "Target-secret lift versus recurrent residual. All static, ledger, and full-matrix receipt points are retained in one visual series. The left-edge marker denotes actual zero outside the positive log scale. Masking lowers lift but includes positive scores across configurations and cuts.")


def figure_families(fam: Dict[str, Any], out_dir: Path) -> None:
    families = fam["families"]
    order = [f for f in ("mamba2", "falcon_h1", "rwkv7") if f in families]
    if not order:
        return
    thr = fam["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    classes = [("frozen_static_exact", "static", "x"), ("frozen_ledger_exact", "ledger", "v"), ("frozen_full_exact", "full matrix", "s")]
    width = 0.25
    for j, (key, label, marker) in enumerate(classes):
        xs, ys = [], []
        for i, f in enumerate(order):
            v = families[f]["verdicts"][key]["max_error_rel_omitted"]
            if v:
                xs.append(i + (j - 1) * width)
                ys.append(max(v["max"], 1e-9))
        ax.scatter(xs, ys, marker=marker, s=40, color=OPERATION, label=label, zorder=3)
    ax.axhline(thr["exact_max_relative_error"], color=NUMERICAL, ls="-.", lw=0.9)
    ax.text(0.01, thr["exact_max_relative_error"] * 1.6, "tolerance: 1e−4", fontsize=11.3, color=NUMERICAL, transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(3e-10, 3e2)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([{"mamba2": "Mamba-2", "falcon_h1": "Falcon-H1", "rwkv7": "RWKV-7"}[f] for f in order])
    ax.set_ylabel("max error / state norm")
    ax.set_title("(a) Frozen-input check")


    ax = axes[1]
    for i, f in enumerate(order):
        v = families[f]["verdicts"]["native_forcing_nonzero"]["median_forcing_share_at_last_cut"]
        if v:
            ax.errorbar(i, v["median"], yerr=[[v["median"] - v["min"]], [v["max"] - v["median"]]], fmt="s", color=OPERATION, ms=5, capsize=3, lw=1)
    ax.axhline(0.0, color=BEHAVIORAL, ls="--", lw=0.8)
    ax.text(0.02, 0.02, "exact transport", fontsize=11.3, color=BEHAVIORAL, transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([{"mamba2": "Mamba-2", "falcon_h1": "Falcon-H1", "rwkv7": "RWKV-7"}[f] for f in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.1, None)
    ax.set_ylabel("forcing norm ratio")
    ax.set_title("(b) Native residual")
    axes[1].tick_params(axis="x", labelrotation=25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=11.3, handletextpad=0.3)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.90))
    save(
        fig,
        out_dir,
        "kda_families",
        "Across Mamba-2, Falcon-H1, and RWKV-7 the predicted receipt class meets the numerical tolerance under frozen inputs. Native forcing is nonzero in the audited cases. Frozen-input tolerance checks are distinct from zero-residual replay.",
    )


def figure_replay(report: Dict[str, Any], out_dir: Path) -> None:
    timing = sorted(report["per_position"], key=lambda row: int(row["replayed_tokens"]))
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    xs = [row["replayed_tokens"] for row in timing]
    ys = [row["replay_seconds"] for row in timing]
    rebuild = [row["rebuild_seconds"] for row in timing]
    ax.plot(xs, ys, color=RETAINED, marker="s", ms=4, lw=1.4)
    ax.axhline(sum(rebuild)/len(rebuild), color=EXCLUDED, ls="--", lw=0.9)
    ax.text(max(xs), sum(rebuild)/len(rebuild)+0.12, "full rebuild: 6.77 s",
            ha="right", va="bottom", fontsize=11.3)
    ax.set_xlabel("surviving suffix tokens")
    ax.set_ylabel("replay time (s)")
    ax.set_ylim(-0.1, 7.6)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, out_dir, "kda_replay_iclr",
         "Nine single replay timings in one 128-record context, from zero seconds for the newest record to 6.70 seconds for the oldest, against a 6.77-second full rebuild. Cohort exactness results appear in the paper tables.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="benchmarks/kda_forcing_sweep_summary_v1.json")
    ap.add_argument("--families", default="benchmarks/recurrent_families_summary_v1.json")
    ap.add_argument("--out-dir", default="/tmp/kda-figures")
    ap.add_argument("--replay", default="artifacts/kimi_sv/mimic_deletion_8bit_n128_v2.json")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if Path(args.sweep).exists():
        s = json.loads(Path(args.sweep).read_text())
        figure_sweep(s, out)
        figure_behavior(s, out)
        print("wrote kda_suffix_sweep, kda_residual_vs_leakage")
    if Path(args.families).exists():
        figure_families(json.loads(Path(args.families).read_text()), out)
        print("wrote kda_families")
    replay = Path(args.replay)
    if not replay.exists():
        replay = Path("artifacts/kimi_sv/mimic_deletion_8bit_n128_v2.json")
    if replay.exists():
        figure_replay(json.loads(replay.read_text()), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
