# NVDRAM_Stacked — Run Commands

Quick command reference for the NVDRAM-on-GPU 3D stack experiment.
See [README.md](README.md) for the full architecture explanation.

Run everything from inside `NVDRAM_Stacked/`.

---

## 0. One-time per shell: put Xyce on PATH
```bash
source ../setenv.sh
```

## 1. Generate the stack inputs (floorplans, ptraces, LCF)
```bash
# Default: 4 NVDRAM tiers, 75 W GPU, 17 mm die, 8.5 mm memory macro
python3 generate_stacked.py --n 4 --gpu-tdp 75

# Other examples:
python3 generate_stacked.py --n 1                      # quick 1-tier sanity stack
python3 generate_stacked.py --n 8 --gpu-tdp 150        # taller stack, hotter GPU
python3 generate_stacked.py --n 4 --gpu-side 0.0212 --mem-side 0.0085   # bigger GPU die
```

## 2. Run the PACT thermal simulation
```bash
python3 ../src/PACT.py stacked_n4_lcf.csv experiment.config modelParams.config \
    --gridSteadyFile stacked_n4.grid.steady
```
- Transient + steady-state via Xyce (~1–2 min, serial). Adjust the LCF/prefix to match `--n`.
- For a **faster steady-state-only** run (no Xyce): in `modelParams.config`, comment out the
  `SPICE_transient` solver lines and uncomment the `SuperLU` lines.

## 3. Plot the per-layer heatmaps
```bash
python3 plot_heatmap.py                         # all device layers (-> stacked_n4.grid.steady.heatmaps.png)
python3 plot_heatmap.py --layer 1               # single layer (GPU FEOL)
python3 plot_heatmap.py --include-package        # also heat spreader + sink
python3 plot_heatmap.py --prefix stacked_n4.grid.steady --out figs/stack.png
```

---

## End-to-end (default N=4)
```bash
source ../setenv.sh
python3 generate_stacked.py --n 4 --gpu-tdp 75
python3 ../src/PACT.py stacked_n4_lcf.csv experiment.config modelParams.config \
    --gridSteadyFile stacked_n4.grid.steady
python3 plot_heatmap.py
```

## Quick inspect (per-layer min/mean/max, °C)
```bash
python3 - <<'EOF'
import numpy as np, glob, re
for p in sorted(glob.glob("stacked_n4.grid.steady.layer*"),
                key=lambda s:int(re.search(r"layer(\d+)$",s).group(1))):
    g = np.loadtxt(p) - 273.15
    lid = re.search(r"layer(\d+)$", p).group(1)
    print(f"layer{lid:>2}: min {g.min():6.2f}  mean {g.mean():6.2f}  max {g.max():6.2f}")
EOF
```
