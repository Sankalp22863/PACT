# Running the hybrid NVDRAM/DRAM experiment

All commands are run from this directory (`NVDRAM_HBM_Hybrid/`).

## 1. Generate the stack inputs

```bash
python3 generate_hybrid.py                       # defaults: GPU 414 W, 4 NVDRAM + 8 DRAM tiers
# variants:
python3 generate_hybrid.py --nv-tiers 2 --dram-tiers 12
python3 generate_hybrid.py --gpu-power 300 --dram-per-stack 60
```

Writes the floorplan/ptrace CSVs and `hybrid_lcf.csv`.

## 2. Run PACT (steady-state, SuperLU — pure Python, fast)

```bash
source ../setenv.sh        # only needed if you switch to the Xyce SPICE solver
python3 ../src/PACT.py hybrid_lcf.csv experiment.config modelParams.config \
    --gridSteadyFile hybrid.grid.steady
```

Produces one grid file per layer: `hybrid.grid.steady.layer{N}` (temperatures in
Kelvin, 48×48 row-major).

## 3. Plots

```bash
# per-layer heatmaps (all device layers, shared colour scale)
python3 plot_heatmap.py

# lowest-layer flat color plane (paper image (d) style: color = temperature, peak annotated)
python3 plot_heatmap.py --surface3d
python3 plot_heatmap.py --surface3d --layer 1     # e.g. the GPU FEOL hot layer

# cross-section schematic + layer table + future-integration diagram
python3 plot_stack.py
```

`plot_stack.py` is generic — point it at the other experiments too:

```bash
python3 plot_stack.py --lcf ../DRAM_3D_Stacked/dram3d_3d_lcf.csv \
                      --config ../DRAM_3D_Stacked/experiment.config --out-prefix dram3d
```

## 4. Quick per-layer min/mean/max check

```bash
python3 - <<'EOF'
import numpy as np, glob, re
for p in sorted(glob.glob("hybrid.grid.steady.layer*"),
                key=lambda q:int(re.search(r"layer(\d+)$",q).group(1))):
    g = np.loadtxt(p) - 273.15
    lid = re.search(r"layer(\d+)$", p).group(1)
    print(f"layer{lid:>2}: min {g.min():6.2f}  mean {g.mean():6.2f}  max {g.max():6.2f}")
EOF
```

## Peak-per-tier: hybrid vs pure DRAM

`plot_peak_per_tier.py` plots peak temperature up the memory stack for two
options, both at 414 W GPU:

- the **NVDRAM + DRAM hybrid** stack from this run — one continuous curve, NVDRAM
  tiers at the bottom flowing into the DRAM tiers continuing on top, and
- the **pure DRAM** stack (`../DRAM_3D_Stacked`) as the reference.

```bash
python3 plot_peak_per_tier.py        # -> peak_per_tier_hybrid_vs_dram.png
```

Requires that both this hybrid run and `../DRAM_3D_Stacked` have been simulated.
