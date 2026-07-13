# NVDRAM_baseline_no_freq — all-DRAM STCO waterfall **without GPU frequency scaling**

Identical to [`../NVDRAM_baseline3`](../NVDRAM_baseline3/) (all-DRAM, 30 × 22 mm,
active read/write, 414 W GPU) **except the GPU frequency-scaling optimization is
removed**: the GPU stays at full 414 W all the way through thermal silicon. This
isolates how much of baseline3's relief came from frequency scaling.

## Stages (freq stage dropped)

| Stage | GPU (°C) | step | Bottom-DRAM (°C) |
|---|---|---|---|
| Baseline (3D stacking) | **143.1** | — | 121.2 |
| + HBM base-die removal | 139.8 | −3.4 | 121.3 |
| + HBM stack merging | 123.0 | −16.8 | 111.4 |
| + Top-die thinning | 121.2 | −1.8 | 110.1 |
| + Thermal silicon | **104.2** | −17.0 | 100.2 |

Final GPU **104.2 °C** vs baseline3's **86.6 °C** (which included 0.5× frequency
scaling): frequency scaling was worth ≈ **17.6 °C** of the total STCO relief.
Everything else (geometry, calibration, active-R/W power model) is unchanged.

## Figures (same as baseline3)

- `waterfall.png/.pdf` — GPU peak + bottom-most DRAM tier at every stage.
- `heatmaps_thermal_si.png/.pdf`, `surfaces_thermal_si.png/.pdf` — GPU + bottom
  DRAM layer maps after thermal-silicon optimization (stage `4_thermal_si`).

## Reproduce

```bash
python3 run_waterfall.py     # all-DRAM, 414 W throughout (no --gpu-power freq stage), --active
python3 plot_waterfall.py
python3 plot_heatmaps.py
python3 plot_surfaces.py
```
