"""
Configuration summary table — thermal results
==============================================
Builds the slide table

  Configuration | Peak DRAM T | Lowest DRAM T | Peak GPU T | Thermally constrained

from the waterfall-model runs. Temperatures are read from the PACT grids:
  * Peak GPU T   = global peak of the GPU FEOL layer
  * Peak DRAM T  = hottest DRAM die (under-stack mask); "—" for all-Fe-RAM
  * Lowest DRAM T= coolest DRAM die (top of stack, nearest lid); "—" for all-Fe-RAM
  * Thermally constrained = True if the hottest DRAM die exceeds T_DRAM_LIMIT
    (90 °C, matching the "T > 90 °C" criterion in the freq-comparison figure);
    False if all DRAM dies are under it; N/A when the stack has no DRAM dies
    (all-Fe-RAM — Fe-RAM is non-volatile / not refresh-limited).

Rows (all thermally-optimized rows = STCO endpoint: base-die removal + merging
+ top-die thinning + thermal silicon; no frequency scaling unless noted):
  3D all-DRAM     ../NVDRAM_baseline_no_freq/0_baseline   (un-optimized)
  3D optimized    ../NVDRAM_baseline_no_freq/4_thermal_si (0 NV, all STCO steps)
  2 Fe-RAM layers extra_nv2
  4 Fe-RAM layers extra_nv4    (full 414 W; XBM optimized is 4 Fe-RAM + 0.8× freq)
  XBM optimized   5_thermal_si (4 NV + 8 DRAM, 0.8× freq = 368 W)
  All-Fe-RAM      extra_nv12

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
    ("2 Fe-RAM layers", os.path.join(HERE, "extra_nv2")),
    ("4 Fe-RAM layers", os.path.join(HERE, "extra_nv4")),
    ("3D optimized",    os.path.join(HERE, "..", "NVDRAM_baseline_no_freq", "4_thermal_si")),
    ("XBM optimized",   os.path.join(HERE, "5_thermal_si")),
    ("All-Fe-RAM",      os.path.join(HERE, "extra_nv12")),
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
    disp_header = ["Configuration", "Peak DRAM\ntemp", "Lowest DRAM\ntemp",
                   "Peak GPU\ntemp", "Thermally\nconstrained"]
    # display copy: mark the "optimized" rows with * (STCO-only footnote below)
    disp_rows = [[(r[0] + "*" if r[0].endswith("optimized") else r[0])] + r[1:]
                 for r in rows]

    # size each column to its own content (widest of header lines / data) + padding,
    # so no column is wider than it needs to be
    ncols = len(header)
    def col_chars(c):
        hdr = max(len(s) for s in disp_header[c].split("\n"))
        dat = max((len(str(disp_rows[r][c])) for r in range(len(disp_rows))), default=0)
        return max(hdr, dat) + 2
    raw = [col_chars(c) for c in range(ncols)]
    total = sum(raw)
    colw = [w / total for w in raw]

    fig = plt.figure(figsize=(total * 0.165, 0.46 * (len(rows) + 1) + 0.35))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])        # axes fills the figure (no subplot margins)
    ax.axis("off")
    tbl = ax.table(cellText=disp_rows, colLabels=disp_header, colWidths=colw,
                   cellLoc="center", loc="center", bbox=[0, 0.11, 1, 0.89])  # bottom 11% = footnote
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15.5)
    for c in range(ncols):
        tbl[0, c].set_height(tbl[0, c].get_height() * 1.4)   # taller header for 2 lines
    phi_row = next((i for i, r_ in enumerate(rows) if r_[0].startswith("XBM")), None)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.8)
        color = "black"
        if r > 0:
            if c == len(header) - 1:                   # thermally-constrained column
                color = CONSTR_COLOR.get(rows[r - 1][c], "black")
            if r - 1 == phi_row:                       # highlight the XBM optimized row
                cell.set_facecolor("#fff3cd")          # yellow band (matches composite callout)
                cell.set_edgecolor("#e0a800")
                cell.set_linewidth(1.6)
        cell.set_text_props(color=color, fontweight="bold", ha="center")
        cell.PAD = 0.04
    fig.text(0.002, 0.05,
             "*STCO structural optimizations only (base-die removal, stack merging, "
             "top-die thinning, thermal silicon) — no GPU frequency scaling.",
             ha="left", va="center", fontsize=11, fontstyle="italic", color="#444444")
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"config_table.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02,
                    facecolor=fig.get_facecolor())
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
