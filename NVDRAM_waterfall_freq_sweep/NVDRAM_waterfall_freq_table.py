"""
GPU frequency-scaling table — 4 NVDRAM + 8 DRAM system (full STCO optimizations)
================================================================================
Renders NVDRAM_waterfall_freq_results.csv as a styled table (teal header,
alternating rows) in the style of the reference figure. Each row is one PACT run
of the full STCO endpoint (base-die removal + stack merging + top-die thinning +
thermal silicon, 4 NVDRAM + 8 DRAM, active read/write) with only the GPU
frequency (=> GPU power) changed. Columns: normalized operating frequency, GPU
power, GPU peak temperature, and the peak of the lowest DRAM layer (die 5 — the
first DRAM tier above the 4 bottom NVDRAM tiers).
Output: NVDRAM_waterfall_freq_table.png / .pdf
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER_BG = "#5f8ba3"       # teal header
ROW_A, ROW_B = "#e8f1f6", "#d3e4ee"   # alternating light-blue rows
INK = "#1f3540"

COLS = ["Operating\nFrequency\n(Normalized)", "GPU\nPower (W)",
        "GPU\nPeak T (°C)", "Lowest DRAM\nPeak T (°C)"]


def load():
    rows = []
    with open(os.path.join(HERE, "NVDRAM_waterfall_freq_results.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append([f'{float(r["freq_normalized"]):.1f}', r["gpu_power_W"],
                         f'{float(r["gpu_peak_C"]):.1f}', f'{float(r["lowest_dram_die5_peak_C"]):.1f}'])
    return rows


def main():
    data = load()
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * (len(data) + 1) + 0.6))
    ax.axis("off")

    tbl = ax.table(cellText=data, colLabels=COLS, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.6)

    ncols = len(COLS)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if r == 0:                                   # header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.35)
        else:
            cell.set_facecolor(ROW_A if r % 2 else ROW_B)
            cell.set_text_props(color=INK, fontweight="bold")

    ax.set_title("GPU frequency scaling — 4 NVDRAM + 8 DRAM (full STCO optimizations)",
                 fontsize=12.5, color=INK, pad=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"NVDRAM_waterfall_freq_table.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
