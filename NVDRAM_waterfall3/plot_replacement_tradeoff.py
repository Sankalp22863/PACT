"""
FeRAM-replacement tradeoff — why not all-FeRAM, and why N = 4
============================================================
Sweeps N = 0..12 FeRAM dies replacing DRAM in the 12-high STCO-optimized stack
(414 W, no frequency scaling) and plots, on one x-axis (N FeRAM dies):

  * LEFT  y  — Peak DRAM temperature (°C), MEASURED from the PACT grids
               (replace_sweep/sweep_peaks.csv). DRAM is the thermally-constrained
               resource: its retention limit is 85 °C (tight) / 90 °C (relaxed).
               FeRAM is non-volatile (needs no refresh) so its own peak — always
               the hottest memory die, nearest the GPU — does not constrain the
               design AT THE 85 C DRAM LIMIT, and is shown only faintly here.
               It is NOT unconstrained though: see plot_reliability.py, where
               FeRAM retention at 87.6-94.7 C works out to 7.9-4.3 yr against a
               10-yr service life. Endurance, by contrast, passes by ~700x.
  * RIGHT y  — Decode throughput (%), a TIMING-AWARE BANDWIDTH model for
               weight-heavy LLM inference (see throughput() below).

Story the figure tells:
  - Replacing DRAM with FeRAM cools the DRAM fast (each FeRAM die both removes a
    hot DRAM die and pushes the survivors away from the GPU): DRAM crosses 85 °C
    at N = 4.
  - Throughput falls monotonically with N (FeRAM streams weights slower; past the
    capacity knee the read-write KV-cache can no longer fit in the shrinking DRAM).
  - => N = 4 is the MINIMUM FeRAM count that satisfies the 85 °C DRAM limit while
    giving up the least throughput. Going further buys thermal margin DRAM does
    not need; all-FeRAM (N = 12) has no DRAM for mutable state -> throughput
    collapses. That is "why not all-FeRAM" and "why N = 4".

Output: nvdram_replacement_tradeoff.png / .pdf
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTE = "#2b2b2b", "#8a8a8a"
T_DRAM = "#c0392b"        # DRAM temperature curve (red)
T_FERAM = "#e08214"       # FeRAM reference curve (orange, faint)
T_THRU = "#1f6f8b"        # throughput curve (teal)
XBM_N = 4                 # the chosen operating point

# DRAM retention limits (°C) — same values as the freq-comparison figure
DRAM_LIMIT_TIGHT, DRAM_LIMIT_RELAX = 85.0, 90.0

# ── timing-aware bandwidth throughput model (weight-heavy LLM inference) ───────
# Decode throughput is memory-bandwidth-bound: tokens/s ∝ sustained aggregate
# read bandwidth for streaming the (read-only) weights every step. Two effects
# lower it as DRAM dies are swapped for FeRAM:
#   1. Bandwidth timing:  a FeRAM die sustains only RHO of a DRAM die's read
#      bandwidth (longer array cycle tRC + destructive-read write-back). With N
#      FeRAM + (12-N) DRAM dies streaming in parallel,
#          BW(N)/BW(0) = [ (12-N)·1 + N·RHO ] / 12  = 1 - N(1-RHO)/12.
#   2. Capacity:  the read-write KV-cache / activations MUST live in DRAM. The
#      binding reason is WRITE ENERGY, not endurance: at 90 fJ/bit (plus the
#      destructive-read write-back) KV-rate traffic is prohibitive, whereas
#      plot_reliability.py shows write endurance clears KV-rate writes by ~700x
#      and cannot carry this argument. The mutable working set of a
#      served LLM occupies M_DRAM of the 12 dies; once (12-N) < M_DRAM the batch
#      size (hence throughput) shrinks ∝ (12-N)/M_DRAM, reaching 0 at N = 12
#      (no DRAM -> mutable state cannot reside -> inference cannot run).
# The knee is at N = 12 - M_DRAM: below it throughput is only the gentle BW droop;
# above it the KV-cache no longer fits and throughput falls steeply. Weight-heavy
# inference has a small mutable set (M_DRAM = 2), so the knee sits near N = 10 and
# N = 4 is justified by the MEASURED thermal argument (DRAM reaches its 85 C limit
# at N = 4), with throughput still ~87% there. Both knobs are exposed and reported.
RHO = 0.60                # FeRAM sustained read BW as a fraction of DRAM's
M_DRAM = 2                # DRAM dies the mutable KV-cache/working set needs -> knee at N = 12-2 = 10


def throughput(n, rho=RHO, m_dram=M_DRAM):
    bw = (12 - n * (1.0 - rho)) / 12.0          # timing-aware bandwidth factor
    cap = min(1.0, (12 - n) / float(m_dram))    # KV-cache capacity factor
    return 100.0 * bw * cap


def load_peaks():
    path = os.path.join(HERE, "replace_sweep", "sweep_peaks.csv")
    ns, dram, feram, gpu = [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ns.append(int(r["n_feram"]))
            dram.append(float(r["dram_peak_C"]) if r["dram_peak_C"] else np.nan)
            feram.append(float(r["feram_peak_C"]) if r["feram_peak_C"] else np.nan)
            gpu.append(float(r["gpu_C"]))
    return (np.array(ns), np.array(dram), np.array(feram), np.array(gpu))


def main():
    ns, dram, feram, gpu = load_peaks()
    thru = np.array([throughput(n) for n in ns])

    fig, axL = plt.subplots(figsize=(11.4, 7.0), dpi=200)
    axR = axL.twinx()

    # shade the thermally-constrained region (DRAM over the 85 °C limit) — N < 4
    xcross = np.interp(DRAM_LIMIT_TIGHT, dram[~np.isnan(dram)][::-1], ns[~np.isnan(dram)][::-1])
    axL.axvspan(-0.5, xcross, color="#f4c7c3", alpha=0.45, zorder=0)
    axL.annotate("DRAM thermally\nconstrained (> 85 °C)", (1.95, 81.5),
                 ha="center", va="center", fontsize=12, fontweight="bold",
                 color="#8a1f14", zorder=6)

    # ── LEFT: measured peak DRAM temperature ─────────────────────────────────
    md = ~np.isnan(dram)
    axL.plot(ns[md], dram[md], "-o", color=T_DRAM, lw=2.6, ms=8, mec="white", mew=1.2,
             zorder=6, label="Peak DRAM temp (measured)")
    # faint FeRAM reference (hottest memory die, but unconstrained)
    mf = ~np.isnan(feram)
    axL.plot(ns[mf], feram[mf], "--", color=T_FERAM, lw=1.6, alpha=0.7, zorder=4,
             label="Peak FeRAM temp (no refresh; retention 4–8 yr)")
    for temp, ls, lab in ((DRAM_LIMIT_TIGHT, ":", "85 °C DRAM limit"),
                          (DRAM_LIMIT_RELAX, ":", "90 °C DRAM limit")):
        axL.axhline(temp, ls=ls, lw=1.8, color=T_DRAM, alpha=0.55, zorder=2)
        axL.annotate(lab, (12.35, temp), ha="right", va="bottom", fontsize=11,
                     fontweight="bold", color=T_DRAM, alpha=0.85, zorder=6)

    # ── RIGHT: timing-aware throughput ───────────────────────────────────────
    axR.plot(ns, thru, "-s", color=T_THRU, lw=2.6, ms=7, mec="white", mew=1.2,
             zorder=6, label="Decode throughput (timing-aware BW model)")

    # ── the chosen operating point N = 4 ─────────────────────────────────────
    axL.axvline(XBM_N, color=INK, lw=1.6, ls="-", alpha=0.8, zorder=5)
    d4, t4 = dram[XBM_N], throughput(XBM_N)
    axL.annotate(f"XBM operating point  (N = {XBM_N})\nDRAM {d4:.1f} °C — at the 85 °C limit\n"
                 f"throughput {t4:.0f} %",
                 (XBM_N, 95.5), xytext=(XBM_N + 0.35, 98.5), ha="left", va="top",
                 fontsize=11.5, fontweight="bold", color=INK, zorder=8,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#e0a800", lw=1.6))

    # all-FeRAM callout
    axR.annotate("all-FeRAM: no DRAM for the\nmutable KV-cache → throughput\ncollapses",
                 (12, throughput(12)), xytext=(9.7, 34), ha="center", va="center",
                 fontsize=10.5, fontweight="bold", color=T_THRU, zorder=8,
                 arrowprops=dict(arrowstyle="-|>", color=T_THRU, lw=1.6,
                                 connectionstyle="arc3,rad=-0.25"))

    # axes cosmetics
    axL.set_xlabel("Number of FeRAM dies replacing DRAM  (of 12-high stack)",
                   fontsize=14, fontweight="bold", color=INK)
    axL.set_ylabel("Peak DRAM temperature (°C)", fontsize=14, fontweight="bold", color=T_DRAM)
    axR.set_ylabel("Decode throughput (% of all-DRAM)", fontsize=14, fontweight="bold", color=T_THRU)
    axL.set_xlim(-0.5, 12.5)
    axL.set_xticks(range(0, 13))
    axL.set_xticklabels([f"{n}\n{100*n//12}%" if n in (0, 4, 8, 12) else str(n) for n in range(13)],
                        fontsize=12, fontweight="bold", color=INK)
    axL.set_ylim(60, 106)
    axR.set_ylim(0, 106)
    axL.tick_params(axis="y", colors=T_DRAM, labelsize=12)
    axR.tick_params(axis="y", colors=T_THRU, labelsize=12)
    for lbl in axL.get_yticklabels():
        lbl.set_fontweight("bold")
    for lbl in axR.get_yticklabels():
        lbl.set_fontweight("bold")
    axL.grid(True, color=MUTE, alpha=0.25, lw=0.7, ls=":")
    axL.set_axisbelow(True)
    for ax in (axL, axR):
        for s in ("top",):
            ax.spines[s].set_visible(False)
    axL.spines["left"].set_color(T_DRAM)
    axR.spines["right"].set_color(T_THRU)

    # combined legend
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    leg = axL.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True, framealpha=0.95,
                     prop={"size": 11, "weight": "bold"}, bbox_to_anchor=(0.005, 0.015))
    leg.set_zorder(10)

    # model-parameter footnote (two lines so it never runs off the frame)
    fig.text(0.5, 0.038,
             f"Timing-aware BW model: FeRAM read BW = {RHO:.2f}× DRAM (tRC + destructive-read "
             f"write-back); mutable KV-cache needs ≥ {M_DRAM} DRAM dies → knee at N = {12 - M_DRAM}.",
             ha="center", va="bottom", fontsize=9, fontstyle="italic", color="#555555")
    fig.text(0.5, 0.010, "Temperatures measured from PACT; throughput modelled.",
             ha="center", va="bottom", fontsize=9, fontstyle="italic", color="#555555")

    fig.tight_layout(rect=(0, 0.065, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"nvdram_replacement_tradeoff.{ext}")
        fig.savefig(out, dpi=200)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
