"""Render the suffix-length sweep and the cross-family audit as paper figures.

Inputs are the source-free summaries written by ``summarize_forcing_sweep`` and
``recurrent_families.summarize``. Outputs follow the project figure grammar
(GEMMASV_FIGURE_STYLE_GUIDE.md): DejaVu Sans, one takeaway per figure, every
reference direct-labeled, denominators printed in the panel, measured points
filled, and the six semantic roles kept to their assigned colors, markers, and
line styles. SVG, PDF, and PNG are written with stripped metadata.

Usage:
    python -m kimi_sv.make_forcing_sweep_figures \
        --sweep gemma_sv/benchmarks/kda_forcing_sweep_summary_v1.json \
        --families gemma_sv/benchmarks/recurrent_families_summary_v1.json \
        --out-dir kda_paper/paper/figs
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
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
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
    "static": dict(color=OPERATION, marker="x", ls=":", label="static receipt subtracted"),
    "ledger": dict(color=OPERATION, marker="v", ls="-.", label="diagonal-ledger receipt"),
    "full": dict(color=OPERATION, marker="s", ls="-", label="full-transport receipt"),
    "mask": dict(color=EXCLUDED, marker=">", ls="-", label="attention rows masked"),
    "mask_full": dict(color=NUMERICAL, marker="P", ls="-", label="masked + full transport"),
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
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.75))

    # (a) imprint and receipt residuals vs suffix length
    ax = axes[0]
    cuts, med, lo, hi = _series(src["state"], "imprint_fraction")
    x = [max(c, 1) for c in cuts]
    ax.fill_between(x, lo, hi, color=TARGET, alpha=0.15, lw=0)
    ax.plot(x, med, color=TARGET, marker="D", ms=3.5, lw=1.4, label="no correction (imprint)")
    for key, cond in (("residual_static_rel_omitted", "static"), ("residual_ledger_rel_omitted", "ledger"), ("residual_full_rel_omitted", "full")):
        c2, m2, _, _ = _series(src["state"], key)
        st = CONDITION_STYLE[cond]
        ax.plot([max(c, 1) for c in c2], m2, color=st["color"], marker=st["marker"], ls=st["ls"], ms=3.5, lw=1.2, label=st["label"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("suffix tokens after record")
    ax.set_ylabel("residual / never-stored norm\n(0 = exact)")
    ax.set_title("(a) residual in recurrent state")
    ax.legend(loc="lower left", frameon=False, handlelength=2.2, fontsize=7)
    # Denominators (admitted/attempted configurations) are stated in the caption.

    # (b) forcing-to-difference norm ratio vs suffix length
    ax = axes[1]
    cuts, med, lo, hi = _series(src["state"], "residual_full_rel_direct")
    x = [max(c, 1) for c in cuts]
    ax.fill_between(x, lo, hi, color=OPERATION, alpha=0.15, lw=0)
    ax.plot(x, med, color=OPERATION, marker="s", ms=3.5, lw=1.4, label="forcing norm / direct-difference norm")
    ax.axhline(1.0, color=EXCLUDED, lw=0.8, ls=":")
    ax.text(x[0], 1.03, "1 = equal norm magnitudes", fontsize=7, color=EXCLUDED, va="bottom")
    ax.axhline(0.0, color=BEHAVIORAL, lw=0.8, ls="--")
    ax.text(x[0], 0.03, "0 = receipt transports exactly", fontsize=7, color=BEHAVIORAL, va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, max(1.15, max(hi) * 1.05))
    ax.set_xlabel("suffix tokens after record")
    ax.set_ylabel("forcing norm / difference norm\n(0 = exact)")
    ax.set_title("(b) forcing norm ratio")

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
        ax.plot(fx, fm, color=OPERATION, marker="D", ms=3.5, lw=1.4, label="recompute k; transport rest")
        ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts], color=OPERATION, alpha=0.15, lw=0)
        ax.plot([0, 1], [fm[0], 0], color=EXCLUDED, lw=0.8, ls=":", label="linear reference")
    ax.axhline(0.0, color=NUMERICAL, lw=0.8, ls="-.")
    ax.text(0.02, 0.03, "0 = full replay", fontsize=7, color=NUMERICAL, va="bottom")
    ax.set_xlabel("fraction recomputed")
    ax.set_ylabel("remaining forcing norm ratio\n(0 = exact)")
    ax.set_title("(c) partial recomputation")
    ax.legend(loc="upper right", bbox_to_anchor=(1.02, 0.93), frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(
        fig,
        out_dir,
        "kda_suffix_sweep",
        "In the audited Kimi Delta Attention configurations, recurrent-state residuals persist across 4,096 suffix tokens. The forcing-to-difference norm ratio compares residual and direct-difference magnitudes. Partial-recomputation residuals are plotted with a linear reference, without implying uniformly distributed forcing.",
    )


def figure_behavior(summary: Dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7))
    for ax, source in zip(axes, ("synthetic", "tofu")):
        src = summary["sources"].get(source)
        if not src:
            ax.set_visible(False)
            continue
        pts = src["scatter_residual_vs_lift"]
        for cond, st in CONDITION_STYLE.items():
            xs = [p["state_residual_mean"] for p in pts if p["condition"] == cond]
            ys = [p["lift_nats"] for p in pts if p["condition"] == cond]
            if not xs:
                continue
            ax.scatter([max(v, 1e-6) for v in xs], ys, s=16, color=st["color"], marker=st["marker"], label=st["label"], alpha=0.85, linewidths=0.6, edgecolors="white" if st["marker"] not in ("x", ">") else st["color"])
        ax.axhline(0.0, color=BEHAVIORAL, lw=0.8, ls="--")
        ax.text(0.98, 0.04, "0 = never-stored behavior", fontsize=7, color=BEHAVIORAL, ha="right", transform=ax.transAxes)
        ax.set_xscale("log")
        ax.set_xlabel("recurrent-state residual vs never stored\n(mean over layers; 0 = exact)")
        ax.set_ylabel("probe lift of the secret (nats)")
        n = src["configurations"]["admitted"]
        ax.set_title(f"({'a' if source == 'synthetic' else 'b'}) {source} records, {n} configurations × {len(src['cuts'])} cuts")
    axes[0].legend(loc="upper left", frameon=False, ncol=1, fontsize=7, markerscale=1.1)
    fig.tight_layout(w_pad=1.2)
    save(
        fig,
        out_dir,
        "kda_residual_vs_leakage",
        "In the Kimi hybrid, every recurrent-state correction leaves the secret fully recoverable while masking the record's attention rows brings recovery to the never-stored floor despite a large recurrent residual.",
    )


def figure_families(fam: Dict[str, Any], out_dir: Path) -> None:
    families = fam["families"]
    order = [f for f in ("mamba2", "falcon_h1", "rwkv7") if f in families]
    if not order:
        return
    thr = fam["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    classes = [("frozen_static_exact", "static", "x"), ("frozen_ledger_exact", "diagonal ledger", "v"), ("frozen_full_exact", "full matrix", "s")]
    width = 0.25
    for j, (key, label, marker) in enumerate(classes):
        xs, ys = [], []
        for i, f in enumerate(order):
            v = families[f]["verdicts"][key]["max_error_rel_omitted"]
            if v:
                xs.append(i + (j - 1) * width)
                ys.append(max(v["max"], 1e-9))
        ax.scatter(xs, ys, marker=marker, s=40, color=OPERATION, label=f"{label} receipt", zorder=3)
        for x_, y_ in zip(xs, ys):
            ax.vlines(x_, 1e-9, y_, color=OPERATION, lw=0.8, alpha=0.5)
    ax.axhline(thr["exact_max_relative_error"], color=NUMERICAL, ls="-.", lw=0.9)
    ax.text(0.01, thr["exact_max_relative_error"] * 1.6, "exact (≤ 1e−4 of state)", fontsize=7, color=NUMERICAL, transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(3e-10, 3e2)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{f}\n({families[f]['transition_class']})" for f in order])
    ax.set_ylabel("max frozen-input error /\nstate norm (lower is better)")
    ax.set_title("(a) which receipt class transports exactly")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=7, columnspacing=1.0, handletextpad=0.3)

    ax = axes[1]
    for i, f in enumerate(order):
        v = families[f]["verdicts"]["native_forcing_nonzero"]["median_forcing_share_at_last_cut"]
        if v:
            ax.errorbar(i, v["median"], yerr=[[v["median"] - v["min"]], [v["max"] - v["median"]]], fmt="s", color=OPERATION, ms=5, capsize=3, lw=1)
    ax.axhline(0.0, color=BEHAVIORAL, ls="--", lw=0.8)
    ax.text(0.02, 0.02, "0 = receipt sufficient", fontsize=7, color=BEHAVIORAL, transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.1, None)
    ax.set_ylabel("forcing norm ratio at last cut\n(median layer; range over runs)")
    ax.set_title("(b) native forcing in audited cases")
    fig.tight_layout(w_pad=1.2)
    save(
        fig,
        out_dir,
        "kda_families",
        "Across the audited Mamba-2, Falcon-H1, and RWKV-7 configurations, the predicted receipt class meets the frozen-input transport tolerance. Native forcing-to-difference norm ratios are nonzero in all three audited families.",
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="gemma_sv/benchmarks/kda_forcing_sweep_summary_v1.json")
    ap.add_argument("--families", default="gemma_sv/benchmarks/recurrent_families_summary_v1.json")
    ap.add_argument("--out-dir", default="kda_paper/paper/figs")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
