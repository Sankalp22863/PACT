# EXP_NVDRAM_12_STCO — STCO endpoint, 12 NVDRAM + 0 DRAM tiers (no frequency scaling)

Built on the **NVDRAM_baseline_no_freq / waterfall model** (NOT the
Experiments-framework model used by EXP_DRAM_B / EXP_NVDRAM_*): rectangular
30x22 mm die, two stack columns on the short edges, 5 mm effective merge void
(`--void-um 5000`, calibrated so stack merging is worth ~-17 C as in the paper),
1 mm on-die edge inserts, 1.5 mm GPU-Si spreader (`--gpu-si-um 1500`,
calibration), uniform GPU power map, 44x60 grid, no package ring/anisotropy.

All four STCO structural steps are applied (= stage `4_thermal_si` of
`../../NVDRAM_baseline_no_freq/`): base-die removal, stack merging (void ->
MERGE_SI k=140), top-die thinning (169 -> 50 um), edge inserts -> THERMAL_SI.
**No GPU frequency scaling — the GPU runs the full 414 W.**

Memory power = standby + ACTIVE traffic (waterfall model, `--active`):
alpha = 0.2421 of 4.4 TB/s per stack; DRAM 70 fJ/bit -> 3.383 W/die,
NVDRAM 90 fJ/bit -> 1.147 W/die.

Run from `../scripts/`:  `python3 run_experiment.py ../EXP_NVDRAM_12_STCO`
Key figure: `Results/layerwise_temperature.png` (GPU / NVDRAM / DRAM layer peaks).
