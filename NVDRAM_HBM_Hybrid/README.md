# Hybrid NVDRAM-bottom / DRAM-top 3D Integration (PACT)

A "future 3D integration" thermal experiment in which **both** memory
technologies are stacked directly on top of a GPU compute die and cooled by a
top-side liquid cold-plate lid:

- the **non-volatile NVDRAM** tiers sit at the **bottom** of the memory stack
  (nearest the GPU), and
- the normal **(HBM) DRAM** tiers sit on **top** (nearest the lid).

This is the architecture in the IEDM paper's "Future 3D Integration HBM-on-GPU"
figure, modified per request so the NVDRAM stack is below the DRAM stack.

## Stack (bottom → top)

```
        ↑ Liquid cold-plate lid (NoPackage, HTC 30,000 W/m²K)   [TOP COOLING]
  DRAM tiers   ×N_dram   (DRAM_Si die, k=100; DAF bond k=0.8 between dies)
  NVDRAM tiers ×N_nv     (NV_FEOL die, k=14;  DAF bond k=0.8 between dies)
  uBump / hybrid-bond interface (k=20)
  GPU BEOL (k=12)
  GPU FEOL (active Si, k=110)                                   [HEAT SOURCE]
  GPU substrate / TSV (k=130)                                   [PACKAGE SIDE — adiabatic]
```

PACT cools only from the top lid; the bottom of layer 0 is adiabatic, so heat is
forced **up** through the entire memory stack to the lid (the paper's
lid-dominated "thermal bottleneck" regime).

## Lateral layout — memory on both sides, thermal silicon in the middle

Each memory tier reuses the `DRAM_3D_Stacked` geometry: four memory stacks in a
**2×2** arrangement on the GPU die. The cross-shaped gap between them and the
outer frame are filled with high-conductivity **thermal silicon** (k = 140
W/m·K). A vertical cross-section through the die centre therefore shows
`memory | thermal-silicon | memory` — memory on both sides with a central
thermal-silicon escape column (see `stack_cross_section.png`). The low-k DAF
between stacked dies is the dominant vertical resistance (the "bottleneck").

## Files

| File | Purpose |
|---|---|
| `generate_hybrid.py` | Builds floorplans, ptraces and the LCF for the hybrid stack. |
| `experiment.config` | Material thermal properties + top-side lid (merged NVDRAM + DRAM material sets). |
| `modelParams.config` | Steady-state SuperLU solver, 48×48 grid, NoPackage lid, material→Solid mappings. |
| `plot_heatmap.py` | Per-layer heatmaps + lowest-layer flat color plane (`--surface3d`, color = temperature). |
| `plot_stack.py` | Cross-section schematic, layer-properties table, future-integration diagram. |
| `plot_peak_per_tier.py` | Peak temperature up the stack: NVDRAM+DRAM hybrid (continuous) vs pure DRAM. |
| `RUN_README.md` | Exact command sequence to regenerate everything. |

## Materials

| Label | k (W/m·K) | Role |
|---|---|---|
| GPU_Si | 130 | GPU substrate / TSV |
| GPU_FEOL | 110 | GPU active Si (heat source) |
| GPU_BEOL | 12 | GPU Cu/low-k interconnect |
| UBUMP | 20 | GPU↔memory hybrid-bond interface |
| NV_FEOL | 14 | NVDRAM ferroelectric die |
| DRAM_Si | 100 | thinned HBM DRAM die |
| THERMAL_SI | 140 | thermal-silicon escape paths (cross + frame) |
| DAF | 0.8 | die-attach film between stacked dies (bottleneck) |

## Representative result (defaults: GPU 414 W, 4 NVDRAM + 8 DRAM tiers)

GPU FEOL (layer 1) is hottest (~127 °C); peak temperature falls monotonically up
the stack toward the lid (~63 °C at the top die). The DAF films impose the
per-tier steps; the thermal-silicon column keeps the surround cool.
