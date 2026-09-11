"""Render distinct KDA-only ICLR figures from source-free evidence.

The layouts are new for the split paper. They preserve the underlying measured
artifacts and write matching assets both to the anonymous paper and to
gemmasv_arxiv_v3/updated_figure_format for design review.
"""

from __future__ import annotations

import binascii
from html import escape
import json
from pathlib import Path
import struct
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "kda-iclr-split-v1",
    }
)
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGS = ROOT / "kda_paper/paper/figs"
UPDATED = ROOT / "gemmasv_arxiv_v3/updated_figure_format"
REPORTS = {
    "short": ROOT / "outputs/kimi_sv/mimic_deletion_8bit.json",
    "long": ROOT / "outputs/kimi_sv/mimic_deletion_8bit_n128_v2.json",
    "notes": ROOT / "outputs/kimi_sv/notes_deletion_8bit_n16.json",
}

INK = "#1E2933"
MUTED = "#66717D"
GRID = "#D8DEE5"
BLUE = "#0072B2"
BLUE_LIGHT = "#EAF4FA"
GREEN = "#00876C"
GREEN_LIGHT = "#E8F5F1"
ORANGE = "#C9460A"
ORANGE_LIGHT = "#FCEDE5"
GRAY = "#A7B0BA"
GRAY_LIGHT = "#F2F4F6"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOR_CHUNKS = frozenset({b"sRGB", b"iCCP", b"gAMA", b"cHRM"})


def add_svg_accessibility(
    path: Path,
    *,
    figure_id: str,
    title: str,
    description: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "<svg "
    start = text.index(marker)
    end = text.index(">", start) + 1
    title_id = f"{figure_id}-title"
    description_id = f"{figure_id}-desc"
    labelled = text[start:end].replace(
        marker,
        (
            f'<svg role="img" aria-labelledby="{title_id}" '
            f'aria-describedby="{description_id}" '
        ),
        1,
    )
    metadata = (
        f'\n <title id="{title_id}">{escape(title)}</title>'
        f'\n <desc id="{description_id}">{escape(description)}</desc>'
    )
    output = text[:start] + labelled + metadata + text[end:]
    path.write_text(
        "\n".join(line.rstrip() for line in output.splitlines()) + "\n",
        encoding="utf-8",
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum)
    )


def normalize_png_srgb(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG: {path}")
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload_end = offset + 8 + length
        payload = data[offset + 8 : payload_end]
        expected = struct.unpack(">I", data[payload_end : payload_end + 4])[0]
        actual = binascii.crc32(kind + payload) & 0xFFFFFFFF
        if expected != actual:
            raise ValueError(f"invalid PNG CRC for {kind!r}: {path}")
        if kind not in _COLOR_CHUNKS:
            chunks.append((kind, payload))
        offset = payload_end + 4
        if kind == b"IEND":
            break
    if not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        raise ValueError(f"invalid PNG chunk order: {path}")
    chunks.insert(1, (b"sRGB", b"\x00"))
    path.write_bytes(
        PNG_SIGNATURE
        + b"".join(_png_chunk(kind, payload) for kind, payload in chunks)
    )


def load_reports(
    paths: Mapping[str, Path] = REPORTS,
) -> dict[str, dict]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }


def validate_reports(reports: Mapping[str, Mapping]) -> None:
    expected = {
        "short": (8, 8),
        "long": (9, 9),
        "notes": (5, 4),
    }
    for name, (attempted, qualified) in expected.items():
        report = reports[name]
        admission = report["admission"]
        exactness = report["exactness"]
        if (
            admission["attempted"] != attempted
            or admission["admitted"] != qualified
            or exactness["admitted_with_zero_residual"] != qualified
            or exactness["logit_residual"]["max"] != 0.0
            or exactness["state_residual"]["max"] != 0.0
        ):
            raise ValueError(f"{name} replay evidence differs")
    if len(reports["long"]["per_position"]) != 9:
        raise ValueError("long-context timing count differs")


def _rounded(
    ax,
    xy,
    width,
    height,
    *,
    facecolor,
    edgecolor,
    linewidth=1.1,
    radius=0.02,
):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    return box


def build_channels_figure():
    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "(a) A record enters both channels of the Kimi hybrid",
        fontsize=11.4,
        weight="bold",
        color=INK,
        va="top",
    )
    ax.text(0.05, 0.82, "record", color=ORANGE, weight="bold", ha="center")
    _rounded(
        ax,
        (0.01, 0.71),
        0.08,
        0.10,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
    )
    ax.annotate(
        "",
        xy=(0.15, 0.78),
        xytext=(0.10, 0.76),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.4},
    )
    ax.annotate(
        "",
        xy=(0.15, 0.64),
        xytext=(0.10, 0.73),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.4},
    )

    _rounded(
        ax,
        (0.16, 0.69),
        0.36,
        0.14,
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
    )
    ax.text(0.18, 0.79, "7 MLA layers", color=BLUE, weight="bold")
    ax.text(
        0.18,
        0.735,
        "contextualized token-level K/V rows",
        color=INK,
    )
    for index in range(10):
        ax.add_patch(
            patches.Rectangle(
                (0.18 + 0.029 * index, 0.705),
                0.020,
                0.025,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=0.8,
            )
        )

    _rounded(
        ax,
        (0.16, 0.54),
        0.54,
        0.11,
        facecolor=GRAY_LIGHT,
        edgecolor=MUTED,
    )
    ax.text(0.18, 0.605, "20 KDA layers", color=INK, weight="bold")
    ax.text(
        0.33,
        0.605,
        "one evolving recurrent matrix plus convolution state per layer",
        color=MUTED,
        fontsize=8.7,
    )
    ax.text(
        0.74,
        0.72,
        "masking either\nchannel leaves\nthe other active",
        color=ORANGE,
        weight="bold",
        linespacing=1.2,
    )
    ax.text(
        0.02,
        0.48,
        "(b) Exact omission restores, skips, and replays the surviving suffix",
        fontsize=11.4,
        weight="bold",
        color=INK,
    )

    labels = ["unchanged prefix"] * 5 + ["victim"] + ["surviving suffix"] * 5
    colors = [GRAY_LIGHT] * 5 + [ORANGE_LIGHT] + [BLUE_LIGHT] * 5
    edges = [GRAY] * 5 + [ORANGE] + [BLUE] * 5
    start_x = 0.03
    for index, (label, color, edge) in enumerate(zip(labels, colors, edges)):
        x = start_x + index * 0.055
        ax.add_patch(
            patches.Rectangle(
                (x, 0.27),
                0.045,
                0.105,
                facecolor=color,
                edgecolor=edge,
                linewidth=1.0,
                linestyle="--" if label == "victim" else "-",
            )
        )
    ax.text(0.03, 0.395, "PREFIX — RESTORE", color=MUTED, weight="bold")
    ax.text(0.327, 0.395, "SKIP", color=ORANGE, weight="bold", ha="center")
    ax.text(0.49, 0.395, "REPLAY EVERY SURVIVING TOKEN", color=BLUE, weight="bold")

    steps = [
        (0.70, "1", "restore", GREEN),
        (0.80, "2", "skip", ORANGE),
        (0.90, "3", "replay", BLUE),
    ]
    for x, number, text, color in steps:
        ax.scatter([x], [0.32], s=180, color=color, zorder=3)
        ax.text(x, 0.32, number, color="white", weight="bold", ha="center", va="center")
        ax.text(x, 0.245, text, color=color, weight="bold", ha="center")
    for left, right in ((0.73, 0.77), (0.83, 0.87)):
        ax.annotate(
            "",
            xy=(right, 0.32),
            xytext=(left, 0.32),
            arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.2},
        )

    _rounded(
        ax,
        (0.64, 0.055),
        0.34,
        0.105,
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
    )
    ax.text(0.66, 0.137, "DECLARED ACTIVE SURFACE", color=GREEN, weight="bold")
    ax.text(0.66, 0.073, "final logits · 80 KDA arrays", color=INK)
    ax.text(
        0.02,
        0.08,
        "MLA offsets restored; MLA tensors are not byte-hashed",
        color=MUTED,
        fontsize=8.4,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.98, bottom=0.02)
    return fig


def build_replay_figure(reports: Mapping[str, Mapping] | None = None):
    reports = dict(reports or load_reports())
    validate_reports(reports)
    fig = plt.figure(figsize=(7.25, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.92, 1.38), wspace=0.28)
    audit = fig.add_subplot(grid[0])
    cost = fig.add_subplot(grid[1])

    audit.set_xlim(0, 1)
    audit.set_ylim(0, 1)
    audit.axis("off")
    audit.set_title(
        "(a) Replay certificate",
        loc="left",
        fontsize=10.8,
        weight="bold",
        color=INK,
        pad=10,
    )
    cohorts = [
        ("short tables", reports["short"]),
        ("128-record context", reports["long"]),
        ("long notes", reports["notes"]),
    ]
    audit.text(0.02, 0.86, "COHORT", color=MUTED, weight="bold", fontsize=8.2)
    audit.text(0.74, 0.86, "QUALIFIED", color=MUTED, weight="bold", fontsize=8.2)
    for index, (label, report) in enumerate(cohorts):
        y = 0.76 - index * 0.12
        admission = report["admission"]
        audit.text(0.02, y, label, color=INK)
        audit.text(
            0.78,
            y,
            f"{admission['admitted']}/{admission['attempted']}",
            color=GREEN,
            weight="bold",
        )
        audit.scatter([0.95], [y + 0.005], marker="D", s=28, color=GREEN)
        audit.text(0.95, y - 0.05, "0", color=GREEN, fontsize=7.5, ha="center")

    _rounded(
        audit,
        (0.02, 0.26),
        0.96,
        0.18,
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
    )
    audit.text(0.06, 0.375, "21/21 replay checks = 0", color=GREEN, weight="bold")
    audit.text(0.06, 0.315, "final logits and every KDA array", color=INK)

    _rounded(
        audit,
        (0.02, 0.05),
        0.43,
        0.13,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
    )
    audit.text(
        0.235,
        0.135,
        "ATTENTION MASK",
        color=ORANGE,
        weight="bold",
        fontsize=7.4,
        ha="center",
    )
    audit.text(
        0.235,
        0.085,
        "nonzero: all victims",
        color=INK,
        fontsize=7.3,
        ha="center",
    )
    _rounded(
        audit,
        (0.55, 0.05),
        0.43,
        0.13,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
    )
    audit.text(
        0.765,
        0.135,
        "NATIVE TRANSPORT",
        color=ORANGE,
        weight="bold",
        fontsize=7.4,
        ha="center",
    )
    audit.text(
        0.765,
        0.085,
        "nonzero residual",
        color=INK,
        fontsize=7.3,
        ha="center",
    )

    timing = sorted(
        reports["long"]["per_position"],
        key=lambda row: int(row["replayed_tokens"]),
    )
    tokens = np.asarray([row["replayed_tokens"] for row in timing], dtype=float)
    replay = np.asarray([row["replay_seconds"] for row in timing], dtype=float)
    rebuild = np.asarray([row["rebuild_seconds"] for row in timing], dtype=float)
    slope, intercept = np.polyfit(tokens, replay, 1)
    fit = slope * tokens + intercept

    cost.scatter(
        tokens,
        replay,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.6,
        s=38,
        label="suffix replay",
        zorder=3,
    )
    cost.plot(tokens, fit, color=BLUE, alpha=0.55, lw=1.5)
    cost.scatter(
        tokens,
        rebuild,
        marker="s",
        color=GRAY,
        s=24,
        label="full rebuild",
        zorder=3,
    )
    cost.axhline(rebuild.mean(), color=GRAY, ls="--", lw=1.0)
    cost.set_title(
        "(b) Work grows with surviving suffix",
        loc="left",
        fontsize=10.8,
        weight="bold",
        color=INK,
        pad=10,
    )
    cost.set_xlabel("surviving suffix tokens reprocessed")
    cost.set_ylabel("wall-clock seconds")
    cost.set_xlim(-120, tokens.max() * 1.035)
    cost.set_ylim(-0.2, max(rebuild.max(), replay.max()) * 1.14)
    cost.grid(axis="both", color=GRID, linewidth=0.6)
    cost.set_axisbelow(True)
    cost.spines[["top", "right"]].set_visible(False)
    cost.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.0)
    cost.text(
        0.98,
        0.03,
        "9 single timings · one 128-record context",
        transform=cost.transAxes,
        color=MUTED,
        fontsize=7.7,
        ha="right",
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.89, bottom=0.17)
    return fig


def write_figure(
    figure,
    output_base: Path,
    *,
    figure_id: str,
    title: str,
    description: str,
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(svg, metadata={"Date": None, "Description": description}, **common)
    add_svg_accessibility(
        svg,
        figure_id=figure_id,
        title=title,
        description=description,
    )
    figure.savefig(
        pdf,
        metadata={
            "Title": title,
            "Subject": description,
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    figure.savefig(
        png,
        dpi=300,
        metadata={"Software": "KDA split renderer", "Title": title},
        **common,
    )
    normalize_png_srgb(png)
    return svg, pdf, png


def main() -> int:
    reports = load_reports()
    validate_reports(reports)
    channels = build_channels_figure()
    replay = build_replay_figure(reports)
    destinations = (
        (
            channels,
            PAPER_FIGS / "kda_channels_iclr",
            "kda-channels-iclr",
            "KDA hybrid channels and replay operation",
            "A record enters token-level MLA rows and evolving KDA state; exact omission restores a checkpoint, skips the record, and replays the surviving suffix.",
        ),
        (
            channels,
            UPDATED / "kimi_channels",
            "kda-channels-updated",
            "KDA hybrid channels and replay operation",
            "Updated split-paper prototype with token-level attention rows and the declared active audit surface.",
        ),
        (
            replay,
            PAPER_FIGS / "kda_replay_iclr",
            "kda-replay-iclr",
            "Replay certificate and suffix-length cost",
            "All 21 qualified replay checks have zero residual; attention masking and native transport are nonzero controls, and replay work grows with surviving suffix length.",
        ),
        (
            replay,
            UPDATED / "kimi_replay",
            "kda-replay-updated",
            "Replay certificate and suffix-length cost",
            "Updated split-paper prototype with qualified denominators, negative controls, and non-overlapping panel titles.",
        ),
    )
    for figure, base, figure_id, title, description in destinations:
        written = write_figure(
            figure,
            base,
            figure_id=figure_id,
            title=title,
            description=description,
        )
        for path in written:
            print(f"saved -> {path}")
    plt.close(channels)
    plt.close(replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
