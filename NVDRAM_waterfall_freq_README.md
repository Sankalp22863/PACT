# NVDRAM_waterfall_freq_* — GPU frequency sweep, 4 NVDRAM + 8 DRAM (full STCO)

One experiment per normalized GPU frequency. Each `NVDRAM_waterfall_freq_<f>`
folder is a single PACT run of the **full STCO endpoint** (base-die removal +
stack merging + top-die thinning + thermal silicon) on the **4 NVDRAM + 8 DRAM**
stack (30 × 22 mm, active read/write, baseline_no_freq model) — **only the GPU
frequency (⇒ GPU power) changes** between folders:

| Freq | Folder | GPU Power (W) | GPU Peak T (°C) | Lowest-DRAM Peak T (°C) |
|---|---|---|---|---|
| 1.0 | `NVDRAM_waterfall_freq_1p0` | 414 | 98.8 | 85.2 |
| 0.9 | `NVDRAM_waterfall_freq_0p9` | 391 | 95.3 | 82.4 |
| 0.8 | `NVDRAM_waterfall_freq_0p8` | 368 | 91.7 | 79.7 |
| 0.7 | `NVDRAM_waterfall_freq_0p7` | 346 | 88.3 | 77.0 |
| 0.6 | `NVDRAM_waterfall_freq_0p6` | 323 | 84.8 | 74.2 |
| 0.5 | `NVDRAM_waterfall_freq_0p5` | 300 | 81.2 | 71.4 |

- **GPU Peak T** = global max of the GPU-FEOL layer (L2).
- **Lowest-DRAM Peak T** = peak under the stacks of the lowest DRAM tier (L20 =
  the **5th memory die**, the first DRAM above the 4 bottom NVDRAM tiers).
- The GPU-power column (414 → 300 W) is the reference frequency→power mapping
  (only compute-dynamic power scales; non-linear).

## Outputs
- `NVDRAM_waterfall_freq_results.csv` — the raw table.
- `NVDRAM_waterfall_freq_table.png/.pdf` — styled table figure.

## Reproduce
```bash
# per frequency f with power P:
cd NVDRAM_waterfall_freq_<f>
python3 waterfall_generate.py --nv-tiers 4 --dram-tiers 8 --gpu-si-um 1500 \
    --void-um 5000 --active --no-base-die --merge-stacks --thin-top-die \
    --optimized --gpu-power <P>
python3 ../src/PACT.py hybrid_lcf.csv experiment.config modelParams.config \
    --gridSteadyFile hybrid.grid.steady
# then, from PACT/:
python3 NVDRAM_waterfall_freq_table.py
```
