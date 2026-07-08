# EXP_NVDRAM_B — hybrid NVDRAM-bottom / DRAM-top, paper BASELINE ("3D thermal penalty")

The **un-optimized** counterpart of [`EXP_NVDRAM_S`](../EXP_NVDRAM_S/): the same
hybrid stack (4 NVDRAM tiers at the bottom, 8 DRAM tiers on top, on a GPU compute
die, top-side liquid lid), but reproducing the IEDM paper's "3D thermal penalty"
regime instead of the thermal-silicon-optimized one.

The **only** change from EXP_NVDRAM_S is the cross/frame filler between and
around the four memory stacks:

| | Cross/frame filler | k (W/m·K) | Meaning |
|---|---|---|---|
| **EXP_NVDRAM_S** (optimized) | `THERMAL_SI` | 140 | high-k dummy-silicon escape path — *thermal-silicon optimization* applied |
| **EXP_NVDRAM_B** (this, baseline) | `MOLD` | 0.8 | epoxy mold compound; **no escape path**, heat forced up through the low-k memory BEOL / hybrid-bond bottleneck |

See [`EXP_DRAM_B`](../EXP_DRAM_B/) for the full rationale — "thermal silicon
optimization" is a *later* STCO step, so the mold-filled variant is the correct
un-optimized baseline. Everything else (power model, geometry, materials, top-only
cooling) is unchanged from EXP_NVDRAM_S.

> **Mold conductivity.** `MOLD` uses the industry-standard epoxy-mold-compound
> value **k = 0.8 W/m·K** (`thermalresistivity = 1.25`); the paper does not
> publish the exact baseline gap-fill value. Run `generate.py --optimized` to
> fall back to `THERMAL_SI` (= EXP_NVDRAM_S).

## Power model — standby only (leakage + refresh), replacement semantics

Unchanged from EXP_NVDRAM_S. With the defaults (12-Hi = 4 NVDRAM + 8 DRAM,
`--dram-per-stack 40`, `--dram-refresh-frac 0.35`, `--nv-leakage-factor 0.5`):

| Per die | Value |
|---|---|
| DRAM (40/12) | 3.33 W  (leakage 2.17 + refresh 1.17) |
| NVDRAM | 1.08 W  (0.5 × DRAM leakage, no refresh) |

GPU 414 W + DRAM 106.7 W + NVDRAM 17.3 W = memory **124 W**.

## Run it

From `../scripts/` (run **EXP_DRAM_B first** — the comparison plot needs it):
```bash
python3 run_experiment.py ../EXP_DRAM_B
python3 run_experiment.py ../EXP_NVDRAM_B
```
`peak_per_tier_hybrid_vs_dram.png` then compares this baseline hybrid against the
baseline pure-DRAM run (`EXP_DRAM_B`) — an apples-to-apples "thermal penalty"
comparison, the mirror of the optimized `EXP_NVDRAM_S` vs `EXP_DRAM_R` pair.
