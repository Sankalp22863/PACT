# EXP_NVDRAM_2_STCO — 2 NVDRAM + 10 DRAM tiers under the full STCO stack (no frequency scaling)

Same model as the [`EXP_NVDRAM_*`](../EXP_NVDRAM_B/) baseline-conditions family
(30x22 mm die, corner stacks, cluster power map p2a=2.19, 6 mm PKG_MOLD package
ring, anisotropic uBump/BSPDN), with the imec STCO structural interventions
applied cumulatively — but **no GPU frequency scaling**: the GPU runs the full
**414 W** workload.

STCO steps applied (see `NVDRAM_waterfall*` for the step-by-step study):
1. **Base-die removal** — the 2 HBM base-die sublayers are dropped.
2. **Stack merging** — no inter-stack mold gap (`--merged`).
3. **Top-die thinning** — top memory die Si 169 um -> 50 um.
4. **Edge thermal-Si** — 1 mm thermal-silicon inserts in the package ring
   adjacent to each short die edge, on the memory sublayers (paper Fig. 10).

Memory power: canonical standby model (DRAM = 40 W/stack budget; NVDRAM =
1.083 W/die leakage, access zeroed via `--bw-util 0`).

Run from `../scripts/`:  `python3 run_experiment.py ../EXP_NVDRAM_2_STCO`
Key figure: `Results/layerwise_temperature.png` (GPU / NVDRAM / DRAM layerwise peaks).
