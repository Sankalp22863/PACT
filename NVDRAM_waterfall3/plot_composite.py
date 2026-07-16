"""
Composite STCO waterfall — xBM vs all-DRAM  (paper Fig. 12 style)
===================================================================
GPU compute-die peak per STCO stage for both experiments:
  * all-DRAM (../NVDRAM_baseline3)  — dark-red box edges
  * xBM    (this folder)          — dark-blue box edges
drawn as the imec staircase: rounded boxes coloured by temperature, curved
"hop" arrows with a teal intervention label at each step, a thick arrow
x-axis, and the paper's 2.5D reference (69.1 °C) with the red "3D thermal
penalty" ramp up to the 3D baseline. The lowest-DRAM-tier peak is the small
muted number under each box. Output: composite_waterfall.png / .pdf
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTE = "#2b2b2b", "#8a8a8a"
AD_EDGE, PH_EDGE = "#7f1d1d", "#1f4e79"    # box edges: all-DRAM / xBM
TEAL = "#2e9aa6"                           # intervention label boxes (paper style)
PURPLE = "#5b2a86"                         # "3D thermal penalty" label
GREEN_25D = "#b5c98e"                      # 2.5D reference box (paper style)
AXIS_BLUE = "#9dc6e0"                      # thick arrow axes

STAGE_X = [0, 1, 2, 3, 4, 5]
INTERVENTIONS = ["HBM base die removal", "HBM stack merging", "Top die thinning",
                 "GPU frequency scaling", "Thermal silicon optimization"]
T_25D = 69.1                               # paper 2.5D reference (Fig. 12)

CMAP = plt.get_cmap("RdYlGn_r")            # hot = red/pink, cool = green (paper feel)
NORM = Normalize(vmin=62.0, vmax=150.0)


def load(path):
    gpu, dram = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            gpu.append(float(r["peak_gpu_C"]))
            dram.append(float(r["bottom_dram_C"]))
    return gpu, dram


def _text_color(fc):
    """Dark text on light box fills, white on saturated ones."""
    r, g, b = fc[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.62 else INK


def series_stack(ax, x, y_anchor, title, gpu_t, dram_t, edge, dy):
    """One experiment at one stage: the big titled box shows the bottom (lowest)
    DRAM-tier peak; the small box stacked beneath shows the peak GPU temperature
    (both coloured by their own temperature), hung `dy` points below the anchor.
    Positioned by GPU peak (y_anchor) so the staircase shape is unchanged."""
    fc_d, fc_g = CMAP(NORM(dram_t)), CMAP(NORM(gpu_t))
    ax.annotate(f"{title}\n{dram_t:.1f}°C", (x, y_anchor), xytext=(0, dy),
                textcoords="offset points", ha="center", va="center",
                fontsize=10, fontweight="bold", color=_text_color(fc_d), zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc=fc_d, ec=edge, lw=1.8))
    ax.annotate(f"GPU {gpu_t:.1f}°C", (x, y_anchor), xytext=(0, dy - 27),
                textcoords="offset points", ha="center", va="center",
                fontsize=8, fontweight="bold", color=_text_color(fc_g), zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc=fc_g, ec=edge, lw=1.4))


HOP_COLOR = "#155e6e"                      # dark teal-blue optimization arrows


def small_hop(ax, x0, y0, x1, y1, rad=-0.35):
    """Small unlabeled hop arrows tracing the xBM staircase."""
    ax.add_patch(FancyArrowPatch((x0 + 0.15, y0), (x1 - 0.15, y1 + 1.2),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="Simple,head_width=7,head_length=8,tail_width=2.4",
                                 fc=HOP_COLOR, ec="#0e4552", lw=0.4, alpha=0.7, zorder=3))


def curly_brace(ax, x, y0, y1, w=0.12, lw=1.4, color=INK, zorder=6):
    """Thin vertical '{' spanning y0..y1: arms end at x, spine tip at x-w."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    ym, q = 0.5 * (y0 + y1), 0.25 * (y1 - y0)
    eps, xs = 0.02 * (y1 - y0), x - 0.5 * w
    verts = [(x, y0),
             (xs, y0), (xs, y0 + q),
             (xs, ym - eps),
             (xs, ym), (x - w, ym),
             (xs, ym), (xs, ym + eps),
             (xs, y1 - q),
             (xs, y1), (x, y1)]
    codes = [Path.MOVETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=color, lw=lw, zorder=zorder))


def hop(ax, x0, y0, x1, y1, rad=-0.45):
    """One broad curved hop arrow per optimization step (paper style)."""
    ax.add_patch(FancyArrowPatch((x0 + 0.16, y0), (x1 - 0.16, y1 + 2.0),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="Simple,head_width=13,head_length=13,tail_width=5.5",
                                 fc=HOP_COLOR, ec="#0e4552", lw=0.6, alpha=0.95, zorder=4))


def main():
    ad_gpu, ad_dram = load(os.path.join(HERE, "..", "NVDRAM_baseline3", "waterfall_results.csv"))
    ph_gpu, ph_dram = load(os.path.join(HERE, "waterfall_results.csv"))
    n = len(ad_gpu)

    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    ax.set_xlim(-1.75, n - 0.45)
    ax.set_ylim(52, 156)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.99, bottom=0.03)
    # convert the fixed 72-pt pair offset into data units (for arrows to the xBM boxes)
    pos = ax.get_position()
    ax_h_pts = fig.get_size_inches()[1] * 72.0 * pos.height
    pair_gap = 72.0 * (156 - 52) / ax_h_pts

    # ---- per stage: stacked [GPU + D1] pair per experiment, a clear vertical
    # gap between the 3D (all-DRAM) pair and the xBM pair, ONE hop per step ----
    for i in range(n):
        if i:
            hop(ax, STAGE_X[i-1], ad_gpu[i-1] - 1.5, STAGE_X[i], ad_gpu[i])
            small_hop(ax, STAGE_X[i-1], ad_gpu[i-1] - pair_gap - 1.2,
                      STAGE_X[i], ad_gpu[i] - pair_gap)
        series_stack(ax, STAGE_X[i], ad_gpu[i], "3D", ad_gpu[i], ad_dram[i], AD_EDGE, dy=0)
        series_stack(ax, STAGE_X[i], ad_gpu[i], "xBM", ph_gpu[i], ph_dram[i], PH_EDGE, dy=-72)

    # ---- thermal headroom: 3D vs xBM at the baseline stage ----
    # curly brace on the LEFT of the stage-0 pair, spanning both boxes, with the
    # label arrowed in from the free space above-left
    xh = STAGE_X[0] - 0.52                 # brace arms out to the left of the ramp
    ybm = ad_gpu[0] - pair_gap / 2.0
    curly_brace(ax, xh, ad_gpu[0] - pair_gap - 2.0, ad_gpu[0] + 2.0,
                w=0.10, lw=1.4, color=INK, zorder=6)
    ax.annotate("thermal\nheadroom", (xh - 0.14, ybm), ha="right", va="center",
                fontsize=10.5, fontstyle="italic", fontweight="bold",
                color=INK, zorder=6)

    # ---- GPU frequency-scaling factors, riding ON the freq-step arrows ----
    if n >= 5:
        xm = (STAGE_X[3] + STAGE_X[4]) / 2.0 + 0.08
        ym = (ad_gpu[3] + ad_gpu[4]) / 2.0 + 1.5
        ax.annotate("3D: 0.5×", (xm + 0.30, ym + 5.5), ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=AD_EDGE, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=AD_EDGE, lw=1.2))
        ax.annotate("xBM: 0.8×", (xm - 0.38, ym - pair_gap - 6.5), ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=PH_EDGE, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=PH_EDGE, lw=1.2))

    # ---- teal intervention labels hugging their own arrows (paper style):
    # anchored at the arc shoulder just right of the source box, so each label
    # unambiguously belongs to the arrow beneath it ----
    for i, label in enumerate(INTERVENTIONS):
        last = (i == len(INTERVENTIONS) - 1)
        if last:   # rightmost label would clip the frame; centre it over its arc
            xy, ha = ((STAGE_X[i] + STAGE_X[i+1]) / 2.0, max(ad_gpu[i], ad_gpu[i+1]) + 4.2), "center"
        else:
            xy, ha = (STAGE_X[i] + 0.42, ad_gpu[i] + 3.8), "left"
        ax.annotate(label, xy, ha=ha, va="bottom", fontsize=8.8,
                    fontweight="bold", color="white", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.28", fc=TEAL, ec="none"))

    # ---- 2.5D reference + red "3D thermal penalty" ramp (paper Fig. 12) ----
    x25 = -1.25
    y25 = T_25D - 6.0                      # sit the reference box a tad lower
    ax.annotate(f"2.5D\n{T_25D:.1f}°C", (x25, y25), ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.32", fc=GREEN_25D, ec="#8aa065", lw=1.5))
    ax.annotate(" (paper ref.)", (x25, y25), xytext=(0, -21), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, color=MUTE, zorder=5)
    ax.add_patch(FancyArrowPatch((x25 + 0.10, y25 + 3), (STAGE_X[0] - 0.34, ad_gpu[0] + 4.5),
                                 connectionstyle="arc3,rad=0.12",
                                 arrowstyle="Simple,head_width=16,head_length=14,tail_width=7",
                                 fc="#d7301f", ec="#a32316", alpha=0.95, zorder=3))
    ax.annotate("3D thermal penalty", (x25 + 0.42, (T_25D + ad_gpu[0]) / 2.0),
                ha="center", va="center", fontsize=9.3, fontweight="bold", color="white",
                rotation=0, zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc=PURPLE, ec="none"))

    # ---- thick arrow axes (paper style), no spines/ticks ----
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # x-axis: thick horizontal arrow
    ax.add_patch(FancyArrowPatch((-0.02, 0.015), (1.005, 0.015), transform=ax.transAxes,
                                 arrowstyle="Simple,head_width=14,head_length=12,tail_width=7",
                                 fc=AXIS_BLUE, ec="#6f9ec2", zorder=2, clip_on=False))
    # y-axis: thinner vertical arrow
    ax.add_patch(FancyArrowPatch((-0.005, 0.0), (-0.005, 1.0), transform=ax.transAxes,
                                 arrowstyle="Simple,head_width=9,head_length=10,tail_width=3.5",
                                 fc=AXIS_BLUE, ec="#6f9ec2", zorder=2, clip_on=False))
    ax.text(-0.035, 0.5, "Package Peak Temperature (°C)", transform=ax.transAxes,
            rotation=90, ha="center", va="center", fontsize=11, color=INK)

    # "STCO approach" caption + note (paper style)
    ax.annotate("STCO approach", (0.60, 0.045), xycoords="axes fraction", ha="center",
                fontsize=11, fontweight="bold", color=INK,
                bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=INK, lw=1.2))
    ax.annotate("Note: identical thermal boundaries for all cases.", (0.24, 0.045),
                xycoords="axes fraction", ha="center", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=MUTE, lw=0.8))

    # legend: series by box edge + box-content meaning
    handles = [
        Rectangle((0, 0), 1, 1, fc=CMAP(NORM(120)), ec=AD_EDGE, lw=1.8,
                  label="3D (all-DRAM): DRAM tier T"),
        Rectangle((0, 0), 1, 1, fc=CMAP(NORM(100)), ec=PH_EDGE, lw=1.8,
                  label="xBM: DRAM tier T"),
        Line2D([0], [0], lw=0, marker="s", ms=11, mfc=MUTE, mec="none",
               label="small box = peak GPU T"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
              borderpad=0.6, prop={"size": 12, "weight": "bold"})

    # ── frequency / performance callout (in the empty lower-centre) ─────────
    iax = ax.inset_axes([0.135, 0.105, 0.515, 0.205])
    iax.axis("off")
    iax.text(0.5, 1.06, "GPU frequency to stay thermally viable",
             ha="center", va="bottom", fontsize=13.5, fontweight="bold", color=INK,
             transform=iax.transAxes)
    # GPT-175B perf model. First number = reference HBM bandwidth, ( ) = with the
    # 4x bandwidth expected from 3D stacking. 3D all-DRAM @ 0.5f: 72%(87%) /
    # 122%(146%); xBM @ 0.8f: 89%(115%) / 150%(193%). Density gains include the
    # 3510->2080 mm^2 (1.688x) 3D package-area reduction.
    trows = [["3D all-DRAM", "0.5×", "72% (87%)", "122% (146%)"],
             ["xBM", "0.8×", "89% (115%)", "150% (193%)"]]
    tbl = iax.table(cellText=trows,
                    colLabels=["Config", "GPU\nfreq", "Through-\nput", "Thr.\ndensity"],
                    colWidths=[0.27, 0.15, 0.28, 0.30], cellLoc="center", loc="center",
                    bbox=[0, 0.12, 1, 0.80])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.9)
        cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.35)   # room for 2-line headers
        else:
            is_phi = trows[r - 1][0].startswith("xBM")
            if is_phi:                                   # highlight the xBM row
                cell.set_facecolor("#fff3cd")            # yellow band (matches config_table)
                cell.set_edgecolor("#e0a800")
                cell.set_linewidth(1.5)
            cell.set_text_props(color=INK,
                                fontweight="bold" if (c in (1, 3) or is_phi) else "normal")
            if c == 1:                                   # emphasise the GPU-freq column
                cell.set_text_props(fontweight="bold",
                                    color=("#1e8f5a" if is_phi else AD_EDGE))
            if c == 3:                                   # emphasise throughput density
                cell.set_text_props(fontweight="bold", color="#1f6f8b")
        cell.PAD = 0.04
    iax.text(0.5, 0.02, "first = ref HBM bandwidth   ·   ( ) = expected with 4× bandwidth (3D)",
             ha="center", va="top", fontsize=9.8, fontstyle="italic", color="black",
             transform=iax.transAxes)

    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"composite_waterfall.{ext}")
        fig.savefig(out, dpi=200)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
