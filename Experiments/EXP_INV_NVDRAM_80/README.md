# EXP_INV_NVDRAM_80 — inverted stack (GPU on top), 80% GPU / 0.8x freq (368 W)

**Inverted orientation**: the device stack is reversed vs the normal HBM-on-GPU
model, so the **GPU (BSPDN backside) sits directly under the TIM/cold-plate lid**
(top cooling) and the **hybrid 4 NVDRAM + 8 DRAM stack is buried below**, nearest the
adiabatic package side. Built on EXP_NVDRAM_B (baseline conditions: corner stacks,
cluster GPU power map p2a=2.19, 6 mm PKG_MOLD ring, anisotropic uBump/BSPDN)
with the new `--invert` flag; GPU at 80% GPU / 0.8x freq (368 W). Memory = standby power (NVDRAM leakage only, --bw-util 0).

The upright counterpart peaks at 141.7 °C (EXP_DRAM_B, 414 W); this run shows
what putting the GPU next to the cooling instead of the memory buys.

Run:  `python3 ../scripts/run_experiment.py ../EXP_INV_NVDRAM_80`
