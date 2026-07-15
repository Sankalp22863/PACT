"""
Configuration summary table — thermal results
==============================================
Builds the slide table

  Configuration | Peak DRAM T | Lowest DRAM T | Peak GPU T | Thermally constrained

from the waterfall-model runs. Temperatures are read from the PACT grids:
  * Peak GPU T   = global peak of the GPU FEOL layer
  * Peak DRAM T  = hottest DRAM die (under-stack mask); "—" for all-NVDRAM
  * Lowest DRAM T= coolest DRAM die (top of stack, nearest lid); "—" for all-NVDRAM
  * Thermally constrained = True if the hottest DRAM die exceeds T_DRAM_LIMIT
    (90 °C, matching the "T > 90 °C" criterion in the freq-comparison figure);
    False if all DRAM dies are under it; N/A when the stack has no DRAM dies
    (all-NVDRAM — NVDRAM is non-volatile / not refresh-limited).

Rows (all thermally-optimized rows = STCO endpoint: base-die removal + merging
+ top-die thinning + thermal silicon; no frequency scaling unless noted):
  3D all-DRAM     ../NVDRAM_baseline_no_freq/0_baseline   (un-optimized)
  3D optimized    ../NVDRAM_baseline_no_freq/4_thermal_si (0 NV, all STCO steps)
  2 NVDRAM layers extra_nv2
  4 NVDRAM layers extra_nv4    (full 414 W; φ-HBM optimized is 4 NV + 0.8× freq)
  φ-HBM optimized 5_thermal_si (4 NV + 8 DRAM, 0.8× freq = 368 W)
  All-NVDRAM      extra_nv12

Outputs: config_table.md, config_table.csv, config_table.png (dark slide style).
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "hybrid.grid.steady"
GPU_X = 0.030
ROWS_G, COLS_G = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005
T_DRAM_LIMIT = 90.0        # DRAM thermal limit (°C) for the "thermally constrained" flag

# (row label, grid dir)
CONFIGS = [
    ("3D all-DRAM",     os.path.join(HERE, "..", "NVDRAM_baseline_no_freq", "0_baseline")),
    ("2 NVDRAM layers", os.path.join(HERE, "extra_nv2")),
    ("4 NVDRAM layers", os.path.join(HERE, "extra_nv4")),
    ("3D optimized",    os.path.join(HERE, "..", "NVDRAM_baseline_no_freq", "4_thermal_si")),
    ("φ-HBM optimized", os.path.join(HERE, "5_thermal_si")),
    ("All-NVDRAM",      os.path.join(HERE, "extra_nv12")),
]


def stack_mask():
    gl = GPU_X / COLS_G
    cw = (GPU_X - 2 * EDGE_INSERT - CENTRAL_VOID) / 2.0
    xs = [(EDGE_INSERT, EDGE_INSERT + cw),
          (EDGE_INSERT + cw + CENTRAL_VOID, EDGE_INSERT + cw + CENTRAL_VOID + cw)]
    m = np.zeros((ROWS_G, COLS_G), dtype=bool)
    for x0, x1 in xs:
        m[:, round(x0 / gl):round(x1 / gl)] = True
    return m


MASK = stack_mask()


def read_run(grid_dir):
    """(peak_dram_C, lowest_dram_C, peak_gpu_C) — DRAM temps None if no DRAM dies."""
    lcf = os.path.join(grid_dir, "hybrid_lcf.csv")
    gpu_layer, dram_layers = None, []
    with open(lcf) as fh:
        next(fh)
        for line in fh:
            idx, flp = [c.strip() for c in line.split(",")[:2]]
            if flp == "gpu_feol_flp.csv" and gpu_layer is None:
                gpu_layer = int(idx)
            elif flp == "dram_tier_flp.csv":
                dram_layers.append(int(idx))

    def grid(layer):
        return (np.loadtxt(os.path.join(grid_dir, f"{PREFIX}.layer{layer}"))
                - 273.15).reshape(ROWS_G, COLS_G)

    gpu_t = float(grid(gpu_layer).max())
    if dram_layers:
        peaks = [float(grid(l)[MASK].max()) for l in dram_layers]
        return max(peaks), min(peaks), gpu_t
    return None, None, gpu_t


def fmt_t(v):
    return f"{v:.1f} °C" if v is not None else "—"


def constrained(dram_hi):
    if dram_hi is None:
        return "N/A"
    return "True" if dram_hi > T_DRAM_LIMIT else "False"


def main():
    header = ["Configuration", "Peak DRAM T", "Lowest DRAM T", "Peak GPU T",
              "Thermally constrained"]
    rows = []
    for label, d in CONFIGS:
        dram_hi, dram_lo, gpu_t = read_run(d)
        rows.append([label, fmt_t(dram_hi), fmt_t(dram_lo), fmt_t(gpu_t),
                     constrained(dram_hi)])

    # markdown
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md_text = "\n".join(md) + "\n"
    open(os.path.join(HERE, "config_table.md"), "w").write(md_text)
    print(md_text)

    # csv
    with open(os.path.join(HERE, "config_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    # white-background PNG, black text (constrained flag stays colour-coded)
    CONSTR_COLOR = {"True": "#c0392b", "False": "#1e8f5a", "N/A": "#777777"}
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * (len(rows) + 1) + 0.3))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    disp_header = ["Configuration", "Peak DRAM\ntemp", "Lowest DRAM\ntemp",
                   "Peak GPU\ntemp", "Thermally\nconstrained"]
    tbl = ax.table(cellText=rows, colLabels=disp_header,
                   colWidths=[0.27, 0.17, 0.18, 0.16, 0.22],
                   cellLoc="right", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl[0, 0].set_height(tbl[0, 0].get_height() * 1.4)   # taller header for 2 lines
    for c in range(len(header)):
        tbl[0, c].set_height(tbl[0, c].get_height() * 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.8)
        color = "black"
        weight = "normal"
        if r == 0:
            weight = "bold"
            ha = "left" if c == 0 else "center"        # centre headers so they don't collide
        else:
            ha = "left" if c == 0 else "right"
            if c == 0 and rows[r - 1][0].startswith("φ-HBM"):
                weight = "bold"                        # highlight the φ-HBM row
            if c == len(header) - 1:                   # thermally-constrained column
                color = CONSTR_COLOR.get(rows[r - 1][c], "black")
                weight = "bold"
        cell.set_text_props(color=color, fontweight=weight, ha=ha)
        cell.PAD = 0.04
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"config_table.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
