# EXP_NVDRAM_B — hybrid NVDRAM-bottom / DRAM-top, paper BASELINE ("3D thermal penalty")

The **un-optimized** counterpart of [`EXP_NVDRAM_S`](../EXP_NVDRAM_S/): the same
hybrid stack (4 NVDRAM tiers at the bottom, 8 DRAM tiers on top, on a GPU compute
die, top-side liquid lid), but under the IEDM paper's "3D thermal penalty"
baseline conditions instead of the merged/optimized ones.

The two changes from EXP_NVDRAM_S mirror [`EXP_DRAM_B`](../EXP_DRAM_B/):

| | EXP_NVDRAM_S (merged/optimized) | EXP_NVDRAM_B (this, baseline) |
|---|---|---|
| Inter-stack gap | none — the two stacks on each short edge are merged | **mold gap** between the two stacks on each edge (paper: ~0.1 mm of k=3 mold; modelled 1.0 mm wide at k_eff = 30 W/m·K to preserve the lateral resistance — see `[MOLD]`) |
| GPU power map | uniform 414 W (idealized reference) | **structured cluster map** (`scripts/stack_common.py`: correlated random cluster utilisation, asymmetric tilt, powered central L2/NoC column, IO ring; tile peak-to-average 2.19, calibrated on EXP_DRAM_B against the paper's 141.7 °C baseline) |

The **central 8 mm column stays `THERMAL_SI`** in both — it is part of the
paper's baseline already. Both variants are embedded in a 6 mm `PKG_MOLD`
package ring with the copper lid extended over the whole 42×34 mm package, and
model the paper's anisotropic uBump/BSPDN in-plane conductivities.
`--merged --power-map uniform` reproduces EXP_NVDRAM_S; `--map-p2a` /
`--map-seed` tune the map.

## Power model — standby only (leakage + refresh), replacement semantics

Unchanged from EXP_NVDRAM_S. With the defaults (12-Hi = 4 NVDRAM + 8 DRAM,
`--dram-per-stack 40`, `--dram-refresh-frac 0.35`, `--nv-leakage-factor 0.5`):

| Per die | Value |
|---|---|
| DRAM (40/12) | 3.33 W  (leakage 2.17 + refresh 1.17) |
| NVDRAM | 1.08 W  (0.5 × DRAM leakage, no refresh) |

GPU 414 W (hotspot map) + DRAM 106.7 W + NVDRAM 17.3 W = memory **124 W**.

## Run it

From `../scripts/` (run **EXP_DRAM_B first** — the comparison plot needs it):
```bash
python3 run_experiment.py ../EXP_DRAM_B
python3 run_experiment.py ../EXP_NVDRAM_B
```
`peak_per_tier_hybrid_vs_dram.png` then compares this baseline hybrid against the
baseline pure-DRAM run (`EXP_DRAM_B`) — an apples-to-apples "thermal penalty"
comparison, the mirror of the optimized `EXP_NVDRAM_S` vs `EXP_DRAM_R` pair.
