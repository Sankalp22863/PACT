# NVDRAM vs 3D-Bonded DRAM — Thermal Comparison Experiment

This experiment uses PACT to compare the steady-state and transient thermal behaviour of two 32 Gb memory chip designs that share the same die footprint (8.5 × 8.5 mm) and functional floorplan but differ fundamentally in how the memory array is integrated above the CMOS logic layer.

---

## Motivation

NVDRAM uses **monolithic 3D integration**: ultra-thin (2 µm) poly-Si memory layers are deposited directly on the CMOS wafer without bonding. Conventional 3D DRAM uses **wafer-bonding**: a separately fabricated, thinned bulk-Si DRAM die (50 µm) is bonded to the CMOS die via a SiO₂ fusion bond interface. The two approaches create very different thermal paths despite having similar chip dimensions and power levels, making them interesting to compare directly in a thermal simulation.

---

## Chip Configurations

### NVDRAM (Monolithic 3D)

```
  ┌──────────────────────────────────────────┐
  │  1T1C Level 2  — PolySi1T1C, 2 µm       │  ← Layer 2
  ├──────────────────────────────────────────┤
  │  1T1C Level 1  — PolySi1T1C, 2 µm       │  ← Layer 1
  ├──────────────────────────────────────────┤
  │  CMOS Under Array — Si, 300 µm           │  ← Layer 0
  └──────────────────────────────────────────┘
              ↓ heat sink / package
```

| Layer | Material | Thickness | k (W/m·K) | R_th (mK/W)* |
|-------|----------|-----------|-----------|--------------|
| CMOS  | Bulk Si  | 300 µm    | 130       | 31.9         |
| 1T1C L1 | PolySi1T1C | 2 µm | 14      | 1.97         |
| 1T1C L2 | PolySi1T1C | 2 µm | 14      | 1.97         |

*R_th = thickness / (k × 72.25 mm²)

### 3D-Bonded DRAM (Wafer Bonded)

```
  ┌──────────────────────────────────────────┐
  │  DRAM Die      — Bulk Si, 50 µm          │  ← Layer 2
  ├──────────────────────────────────────────┤
  │  Fusion Bond   — SiO₂, 2 µm             │  ← Layer 1  ← thermal bottleneck
  ├──────────────────────────────────────────┤
  │  CMOS Base     — Si, 300 µm              │  ← Layer 0
  └──────────────────────────────────────────┘
              ↓ heat sink / package
```

| Layer | Material | Thickness | k (W/m·K) | R_th (mK/W)* |
|-------|----------|-----------|-----------|--------------|
| CMOS  | Bulk Si  | 300 µm    | 130       | 31.9         |
| Fusion Bond | SiO₂ | 2 µm  | 1.4       | 19.8         |
| DRAM Die | Bulk Si | 50 µm  | 130       | 5.32         |

> **Key insight:** Although the fusion bond and NVDRAM 1T1C layers are the same thickness (2 µm), the fusion bond has 10× lower thermal conductivity (1.4 vs 14 W/m·K), giving it **10× higher thermal resistance**. This bonding interface is a significant thermal bottleneck absent in monolithic NVDRAM.

---

## Power Traces

Both designs are characterised across three power scenarios, matching the NVDRAM experiment convention:

| Column | Scenario | Description |
|--------|----------|-------------|
| Power  | Idle     | Standby + background refresh (DRAM only) |
| Power1 | Active Read | Burst read at full bandwidth |
| Power2 | Active Write | Burst write at full bandwidth |

### NVDRAM CMOS Layer (`nvdram_cmos_ptrace.csv`)

| Block | Idle (W) | Read (W) | Write (W) |
|-------|----------|----------|-----------|
| MemArray | 0.5 | 2.0 | 3.5 |
| RowDecoder_L | 0.3 | 1.0 | 1.5 |
| RowDecoder_R | 0.3 | 1.0 | 1.5 |
| SenseAmp_B | 0.2 | 0.75 | 1.2 |
| SenseAmp_T | 0.2 | 0.75 | 1.2 |
| **Total** | **1.5 W** | **5.5 W** | **8.9 W** |

### NVDRAM 1T1C Memory Layers (`nvdram_mem_ptrace.csv`) — same for L1 and L2

| Block | Idle (W) | Read (W) | Write (W) |
|-------|----------|----------|-----------|
| MemArray | 0.05 | 0.30 | 0.80 |
| RowDecoder_L | 0.02 | 0.10 | 0.20 |
| RowDecoder_R | 0.02 | 0.10 | 0.20 |
| SenseAmp_B | 0.01 | 0.07 | 0.15 |
| SenseAmp_T | 0.01 | 0.07 | 0.15 |
| **Total/layer** | **0.11 W** | **0.64 W** | **1.50 W** |

### DRAM CMOS Layer (`dram_cmos_ptrace.csv`)

DRAM adds a refresh controller (~20% idle overhead) but saves write energy (no ferroelectric polarisation switching, no write-verify cycles → ~20% lower write power).

| Block | Idle (W) | Read (W) | Write (W) |
|-------|----------|----------|-----------|
| MemArray | 0.60 | 2.0 | 2.80 |
| RowDecoder_L | 0.36 | 1.0 | 1.20 |
| RowDecoder_R | 0.36 | 1.0 | 1.20 |
| SenseAmp_B | 0.24 | 0.75 | 0.96 |
| SenseAmp_T | 0.24 | 0.75 | 0.96 |
| **Total** | **1.80 W** | **5.5 W** | **7.12 W** |

### DRAM Fusion Bond (`dram_bond_ptrace.csv`)

Passive layer — no active power dissipation (values set to 10 µW to satisfy PACT's power trace requirement).

### DRAM Die (`dram_die_ptrace.csv`)

DRAM cells have higher leakage than ferroelectric NVDRAM (capacitor discharge requires periodic refresh), and lower write energy (no FE switching).

| Block | Idle (W) | Read (W) | Write (W) |
|-------|----------|----------|-----------|
| MemArray | 0.10 | 0.20 | 0.40 |
| RowDecoder_L | 0.025 | 0.08 | 0.15 |
| RowDecoder_R | 0.025 | 0.08 | 0.15 |
| SenseAmp_B | 0.015 | 0.05 | 0.10 |
| SenseAmp_T | 0.015 | 0.05 | 0.10 |
| **Total** | **0.18 W** | **0.46 W** | **0.90 W** |

---

## File Structure

```
NVDRAM_vs_DRAM_Experiment/
├── experiment.config          — material properties: Si, PolySi1T1C, FusionBond, HeatSink
├── modelParams.config         — PACT solver settings (SPICE_transient, 40×40 grid)
│
├── cmos_flp.csv               — shared CMOS floorplan (Si label, used by both chips)
├── nvdram_mem_flp.csv         — NVDRAM memory layer floorplan (PolySi1T1C label)
├── dram_bond_flp.csv          — DRAM fusion bond layer floorplan (FusionBond label)
├── dram_die_flp.csv           — DRAM die floorplan (Si label)
│
├── nvdram_lcf.csv             — NVDRAM layer configuration (3 layers)
├── nvdram_cmos_ptrace.csv     — NVDRAM CMOS power trace
├── nvdram_mem_ptrace.csv      — NVDRAM 1T1C memory power trace
│
├── dram_lcf.csv               — DRAM layer configuration (3 layers)
├── dram_cmos_ptrace.csv       — DRAM CMOS power trace (with refresh overhead)
├── dram_bond_ptrace.csv       — Bonding layer power trace (near-zero)
├── dram_die_ptrace.csv        — DRAM die power trace
│
└── run_comparison.py          — run both sims + generate comparison figures
```

---

## Running the Experiment

```bash
cd NVDRAM_vs_DRAM_Experiment/

# 1. Source Xyce (only needed once per shell session)
source ../setenv.sh

# 2. Run both simulations and generate all comparison figures
python run_comparison.py

# Or skip re-running PACT and just regenerate figures from existing outputs:
python run_comparison.py --skip-sim

# Custom output directory:
python run_comparison.py --out-dir ../diagrams/comparison/
```

---

## Output Files

### PACT simulation outputs

| File | Contents |
|------|----------|
| `nvdram.grid.steady.layer{N}` | NVDRAM steady-state 40×40 grid temperatures (K), layers 0–4 |
| `nvdram.block.transient.csv` | NVDRAM block-level temperatures vs time |
| `nvdram.cir.csv` | NVDRAM full transient grid trace (Xyce output) |
| `dram.grid.steady.layer{N}` | DRAM steady-state 40×40 grid temperatures (K), layers 0–4 |
| `dram.block.transient.csv` | DRAM block-level temperatures vs time |
| `dram.cir.csv` | DRAM full transient grid trace (Xyce output) |

### Comparison figures (saved to `figures/`)

| File | Description |
|------|-------------|
| `heatmap_comparison.png` | Side-by-side 40×40 thermal maps, all active layers |
| `peak_temperature_comparison.png` | Peak / mean / min temperature per layer, both chips |
| `block_transient_comparison.png` | Per-block temperature evolution, both chips |
| `thermal_resistance_breakdown.png` | Analytic R_th per layer — quantifies bonding penalty |
| `power_budget_comparison.png` | Stacked power bar chart per scenario, both chips |

---

## Expected Findings

1. **DRAM fusion bond creates a thermal barrier**: the 2 µm SiO₂ bond (k = 1.4 W/m·K) has 10× higher R_th than the NVDRAM 1T1C layers at the same thickness, trapping heat in the DRAM die above it.

2. **NVDRAM idle runs hotter at the CMOS layer**: despite lower idle power than the DRAM CMOS (1.5 W vs 1.8 W), the PolySi layers above reduce the upward heat dissipation path slightly — though the effect is small given the very low R_th of 2 µm PolySi.

3. **DRAM write is cooler than NVDRAM write**: no ferroelectric switching energy in DRAM → lower peak write temperatures despite the fusion bond penalty.

4. **Transient response**: both chips reach thermal steady state in < 400 ms due to the small die size and common heat sink, but the rate differs due to the different specific heat capacities of PolySi1T1C vs Si.

---

## Modifying the Experiment

**Change bonding type**: In `experiment.config`, edit `[FusionBond]` thermal resistivity:
- SiO₂ fusion bond: 0.714 (m·K)/W (default)
- Hybrid Cu/oxide bond: ~0.208 (m·K)/W (k ≈ 4.8 W/m·K)
- BCB polymer bond: ~3.45 (m·K)/W (k ≈ 0.29 W/m·K)

**Change DRAM die thickness**: Edit `dram_lcf.csv` Layer 2 thickness (currently 0.00005 m = 50 µm). Typical range: 10–100 µm.

**Steady-state only (no Xyce)**: In `modelParams.config`, change `[Solver]` to:
```ini
name = SuperLU
wrapper = SuperLUSolver.py
```
