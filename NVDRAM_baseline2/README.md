# NVDRAM_baseline2 — all-DRAM STCO waterfall, paper-faithful layout (30 × 22 mm)

A faithful replication of the reference 3D HBM-on-GPU thermal experiment
(Chen et al., IEDM 2025), all-DRAM. Reproduces the reference STCO staircase
**across all tiers to within ~2 °C**.

## Geometry (reference Fig. 3)

- GPU die **30 × 22 mm** (44 × 60 grid, 0.5 mm cells).
- **Two HBM stack columns along the short edges** (each column = 2 stacks stacked
  vertically), reaching the die edges — no perimeter mold frame.
- A **central void** between the columns: at baseline it is mold (k = 3, the
  reference value), the hotspot; stack merging replaces it with continuous silicon
  (k = 140). Thin thermal-silicon inserts at the outer package edges.

This layout is what makes **stack merging a real lever** (the central mold void is
the baseline hotspot that merging fixes) — unlike the earlier square 2×2 model in
`../NVDRAM_baseline`, where the hotspot was the perimeter frame and merging did
almost nothing.

## Result vs. reference (all-DRAM)

Peak GPU temperature = global max on the GPU-FEOL layer; bottom-DRAM = max under
the stacks. Frequency stage = 0.5× → **300 W** (reference: only compute dynamic
power scales, L2/interconnect fixed — not linear 0.5×414).

| Stage | GPU (ours) | GPU (ref.) | Δ | Bottom-DRAM |
|---|---|---|---|---|
| Baseline (3D stacking) | **142.8** | 141.7 | +1.1 | 120.8 |
| + HBM base-die removal | 139.4 | 138.0 | +1.4 | 121.0 |
| + HBM stack merging | 122.7 | 120.4 | +2.3 | 111.1 |
| + Top-die thinning | 120.9 | 120.0 | +0.9 | 109.8 |
| + 0.5× GPU frequency | 98.7 | 99.2 | −0.5 | 90.9 |
| + Thermal silicon | 86.3 | 87.4 | −1.1 | 83.5 |

Merge step = **−16.8 °C** (reference −17.6). Figure: `waterfall.png/.pdf`.

## Calibration (3 frozen knobs)

A uniform-power compact model (PACT) cannot reproduce the reference's Ansys-Icepak
3D solve + non-uniform 0.5 mm power maps (which set where hotspots form and how
much each intervention helps). Three knobs are frozen so the whole staircase
matches:

| Knob | Value | Role | Where |
|---|---|---|---|
| GPU-Si substrate | 1500 µm | lateral heat spreading | `run_waterfall.py` `--gpu-si-um` |
| merge-void width | 5000 µm | sets the merge step to ~−17 °C (effective mold-gap width; the reference's central region is mostly thermal silicon with mold in "remaining gaps") | `run_waterfall.py` `--void-um` |
| lid HTC | 35 000 W/m²K | anchors the baseline to ~142 °C (near the physical 30 000) | `experiment.config` |

## Reproduce

```bash
python3 run_waterfall.py --nv-tiers 0 --dram-tiers 12   # -> waterfall_results.csv (GPU + bottom-DRAM)
python3 plot_waterfall.py                               # -> waterfall.png/.pdf
python3 plot_heatmaps.py                                # -> heatmaps_thermal_si.png/.pdf
```

## Heatmaps

`heatmaps_thermal_si.png/.pdf` — GPU compute die + bottom-most DRAM tier after
thermal-silicon optimization. Two hot stack columns with the cold central silicon
void between them. No NVDRAM layer (all-DRAM stack; third panel notes this).

## Known fidelity gaps (absorbed by calibration)

- **Uniform GPU power** (414 W) instead of the reference's non-uniform 0.5 mm power
  maps — the single biggest gap; the calibration knobs stand in for it.
- **Adiabatic package side** — the reference air-cools the laminate at HTC = 200
  (weak; matters only for the double-sided-cooling step, not included here).
- **Compact grid solver** vs. full 3D FEM.
