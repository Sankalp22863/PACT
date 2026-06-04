# NVDRAM Thermal Simulation Experiment

PACT thermal simulation of a 32 Gb NVDRAM chip with a two-level monolithically integrated 1T1C (one-transistor one-capacitor) ferroelectric memory stack fabricated over a CMOS under-array.

---

## Chip Architecture

The NVDRAM die uses a monolithic 3D integration scheme where memory layers are deposited directly on top of the CMOS logic layer — unlike wafer-bonded 3D stacks, these memory layers are extremely thin:

```
  ┌─────────────────────────────────────────────┐
  │   BEOL Metallization + MIM Cap  (passive)   │  ← top
  ├─────────────────────────────────────────────┤
  │   1T1C Array — Level 2  (2 µm, active)      │  ← Layer 2
  ├─────────────────────────────────────────────┤
  │   1T1C Array — Level 1  (2 µm, active)      │  ← Layer 1
  ├─────────────────────────────────────────────┤
  │   CMOS Under Array      (300 µm, active)    │  ← Layer 0 (closest to package)
  └─────────────────────────────────────────────┘
              ↓ heat sink / package
```

Each 1T1C cell consists of a poly-Si access transistor (Lg = 80 nm, WEFF = 40 nm) and an HZO ferroelectric capacitor.

**Die dimensions:** 8.5 mm × 8.5 mm (~72.25 mm²), derived from reported bit density of 0.45 Gb/mm² at 32 Gb capacity.

---

## Floorplan

Both the CMOS and 1T1C layers share the same physical footprint, divided into five functional blocks:

```
  ┌──────────────────────────────────────────────┐
  │              SenseAmp_T  (8.5 × 1.0 mm)      │
  ├──────┬──────────────────────────┬─────────────┤
  │  RD  │                          │     RD      │
  │  _L  │      MemArray            │     _R      │
  │ 1.0  │   (6.5 × 6.5 mm)        │    1.0      │
  │  ×   │                          │     ×       │
  │ 6.5  │                          │    6.5      │
  ├──────┴──────────────────────────┴─────────────┤
  │              SenseAmp_B  (8.5 × 1.0 mm)      │
  └──────────────────────────────────────────────┘
```

| Block | Description | Area |
|---|---|---|
| MemArray | Central memory array (bitcells) | 42.25 mm² |
| RowDecoder_L | Left row decoder | 6.5 mm² |
| RowDecoder_R | Right row decoder | 6.5 mm² |
| SenseAmp_B | Bottom sense amplifiers + column decoder | 8.5 mm² |
| SenseAmp_T | Top sense amplifiers + column decoder | 8.5 mm² |

---

## Input Files

### `nvdram.config`
Material property definitions for each label used in the floorplans.

| Section | Material | Thermal Resistivity (m·K/W) | Specific Heat (J/m³·K) |
|---|---|---|---|
| `[Si]` | Bulk silicon — CMOS substrate | 0.0077 (k = 130 W/m·K) | 1,750,000 |
| `[PolySi1T1C]` | Poly-Si/HZO/oxide 1T1C layer | 0.07 (k ≈ 14 W/m·K) | 1,500,000 |
| `[HeatSink]` | Medium-cost heat sink (HotSpot model) | — | — |
| `[Init]` | Ambient = 318.15 K, Initial = 273.15 K | — | — |

The `PolySi1T1C` thermal conductivity (≈14 W/m·K) is an effective value for LPCVD poly-Si
with interspersed HZO (k ≈ 1.5 W/m·K) and gate oxide (k ≈ 1.4 W/m·K).

### `nvdram_cmos_flp.csv`
Floorplan for **Layer 0 (CMOS Under Array)**. All blocks carry `Label = Si`.

### `nvdram_1t1c_flp.csv`
Floorplan for **Layers 1 and 2 (1T1C memory arrays)**. Same geometry as the CMOS floorplan; blocks carry `Label = PolySi1T1C`.

### `nvdram_lcf.csv`
Layer configuration file defining the vertical die stack.

| Layer | Floorplan | Thickness | Power Trace | Physical layer |
|---|---|---|---|---|
| 0 | nvdram_cmos_flp.csv | 300 µm | nvdram_cmos_ptrace.csv | CMOS Under Array |
| 1 | nvdram_1t1c_flp.csv | 2 µm | nvdram_mem_ptrace.csv | 1T1C Level 1 |
| 2 | nvdram_1t1c_flp.csv | 2 µm | nvdram_mem_ptrace.csv | 1T1C Level 2 |

### `nvdram_cmos_ptrace.csv`
Power trace for the CMOS layer across three operating scenarios (columns = time steps):

| Block | Idle (W) | Active Read (W) | Active Write (W) |
|---|---|---|---|
| MemArray | 0.5 | 2.0 | 3.5 |
| RowDecoder_L | 0.3 | 1.0 | 1.5 |
| RowDecoder_R | 0.3 | 1.0 | 1.5 |
| SenseAmp_B | 0.2 | 0.75 | 1.2 |
| SenseAmp_T | 0.2 | 0.75 | 1.2 |
| **Total** | **1.5 W** | **5.5 W** | **8.9 W** |

### `nvdram_mem_ptrace.csv`
Power trace for each 1T1C memory layer. These layers dissipate significantly less power than the CMOS layer since they contain only the passive ferroelectric bitcells and local word/digit lines (no sense amp or logic).

| Block | Idle (W) | Active Read (W) | Active Write (W) |
|---|---|---|---|
| MemArray | 0.05 | 0.3 | 0.8 |
| RowDecoder_L | 0.02 | 0.1 | 0.2 |
| RowDecoder_R | 0.02 | 0.1 | 0.2 |
| SenseAmp_B | 0.01 | 0.07 | 0.15 |
| SenseAmp_T | 0.01 | 0.07 | 0.15 |
| **Total/layer** | **0.11 W** | **0.64 W** | **1.5 W** |

### `nvdram_modelParams.config`
Simulation parameters. Key settings:

| Parameter | Value | Notes |
|---|---|---|
| Solver | SPICE_transient | Full transient using Xyce backend |
| Grid | 40 × 40 | Uniform grid, 212.5 µm cell size |
| number_of_core | 1 | Serial Xyce build only |
| step_size | 3.33 ms | Xyce time step |
| ptrace_step_size | 333 ms | Each power trace column duration |
| total_simulation_time | 1332 ms | 4 ptrace cycles (~1.3 s) |

To switch to steady-state only (faster), comment out the `SPICE_transient` solver lines and uncomment `SuperLU` (no Xyce required).

---

## Running the Simulation

From inside `NVDRAM_Experiment/`:

```bash
# 1. Load Xyce into PATH (only needed once per shell session)
source ../setenv.sh

# 2. Run transient + steady-state simulation
python3 ../src/PACT.py nvdram_lcf.csv nvdram.config nvdram_modelParams.config \
    --gridSteadyFile nvdram.grid.steady
```

---

## Output Files

| File | Contents |
|---|---|
| `nvdram.grid.steady.layer{N}` | 40×40 grid of steady-state temperatures (°C) for each layer N |
| `nvdram.block.transient.csv` | Block-level temperature vs. time for each layer |
| `nvdram.cir.csv` | Full transient grid temperature trace from Xyce |
| `nvdram.cir.ic` | Final temperature state (use as initial condition for follow-on transient runs) |
| `nvdram.log` | Xyce solver log |

Layer numbering in output files matches the LCF: layer0 = CMOS, layer1 = 1T1C Level 1, layer2 = 1T1C Level 2. Layers 3 and 4 are the heat spreader and heat sink added automatically by PACT.

---

## Adjusting the Simulation

**Change power scenario:** Edit `nvdram_cmos_ptrace.csv` and `nvdram_mem_ptrace.csv`. Column order is idle → read → write.

**Use steady-state only (no Xyce):** In `nvdram_modelParams.config`, change solver to:
```ini
name = SuperLU
wrapper = SuperLUSolver.py
```
and in `nvdram.config`, comment out `[HeatSink]` and add `[NoPackage]` instead (SuperLU does not support the HeatSink package model).

**Adjust grid resolution:** Change `rows` and `cols` in `[Grid]`. Values must be multiples of 2 or 5, and must evenly divide the chip dimensions (8.5 mm). Recommended: 20, 40, or 85.

**Enable initial condition from steady state:** Set `init_file = True` in `[Simulation]` and pass `--init nvdram.cir.ic` to use the steady-state result as the transient starting point.
