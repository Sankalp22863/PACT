"""
Shared geometry / power-map helpers for the 3D HBM-on-GPU experiment generators.
================================================================================
Provides the two model features added on top of the bare die stack:

  * add_package_ring() — embeds the 30x22 mm die in a molded package surround
    (default 6 mm ring of PKG_MOLD, k=3) so heat can spread laterally through
    the mold and the extended copper lid, as in the paper's 65x65 mm package.
    The lid layer's ring is LID (the cold-plate lid spans the package).
    NOTE: the laminate (bottom) boundary stays adiabatic — PACT exposes a
    single convective boundary and the paper's 200 W/m2K air-cooled laminate
    removes only a small share of the 574 W.

  * cluster_power_tiles() — a structured, asymmetric stand-in for the paper's
    commercial 0.5 mm GPU power maps (not public): a lattice of compute
    clusters with spatially-correlated, seeded-random utilisation, an
    asymmetric large-scale activity tilt, a lower-power (but powered) central
    L2/NoC column, and a low-power IO ring at the die edge. Tile contrast is
    scaled so the tile peak-to-average power ratio equals `p2a` — the single
    knob calibrated against the paper's 141.7 C baseline GPU peak.

Generators import this via:  sys.path.insert(0, "../scripts")  (relative to the
experiment directory; run_experiment.py runs them with cwd = experiment dir).
"""

import numpy as np

PKG_MARGIN = 0.006      # m — mold ring width around the die
RING_LABEL = "PKG_MOLD"

POWER_TILE = 0.0005     # m — power-map tiles at the paper's 0.5 mm resolution
CLUSTER_X = 0.0025      # m — compute-cluster pitch in x
CLUSTER_Y = 0.0022      # m — compute-cluster pitch in y
L2_HALF_WIDTH = 0.004   # m — central L2/NoC column half-width
L2_FACTOR = 0.70        # L2/NoC power density vs compute average
IO_RING = 0.001         # m — low-power IO ring at the die edge
IO_FACTOR = 0.40
MAP_SEED = 7            # deterministic map; change for a different workload


def add_package_ring(blocks, die_len, die_wid, margin=PKG_MARGIN,
                     ring_label=RING_LABEL):
    """Offset die-local blocks by (margin, margin) and wrap them in a package
    ring so the layer tiles the full (die_len+2m) x (die_wid+2m) footprint."""
    if margin <= 0:
        return list(blocks)
    m = margin
    pl, pw = die_len + 2 * m, die_wid + 2 * m
    out = [(n, x + m, y + m, l, w, lab) for (n, x, y, l, w, lab) in blocks]
    out += [
        ("Ring_B", 0.0, 0.0,     pl, m, ring_label),
        ("Ring_T", 0.0, m + die_wid, pl, m, ring_label),
        ("Ring_L", 0.0, m, m, die_wid, ring_label),
        ("Ring_R", m + die_len, m, m, die_wid, ring_label),
    ]
    return out


def full_package_block(die_len, die_wid, margin, label):
    """Single block covering the whole package footprint (e.g. the lid)."""
    return [(label, 0.0, 0.0, die_len + 2 * margin, die_wid + 2 * margin, label)]


def _cluster_field(die_len, die_wid, tile, seed):
    """Raw per-tile power-density weights (unnormalised), die-local."""
    nx, ny = int(round(die_len / tile)), int(round(die_wid / tile))
    xs = (np.arange(nx) + 0.5) * tile
    ys = (np.arange(ny) + 0.5) * tile
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # compute-cluster lattice with seeded random utilisation ...
    rng = np.random.RandomState(seed)
    ncx = max(1, int(round(die_len / CLUSTER_X)))
    ncy = max(1, int(round(die_wid / CLUSTER_Y)))
    util = rng.uniform(0.55, 1.15, size=(ncx, ncy))
    # ... spatially correlated (3x3 box smoothing, edge-padded)
    p = np.pad(util, 1, mode="edge")
    util = sum(p[i:i + ncx, j:j + ncy] for i in range(3) for j in range(3)) / 9.0
    ci = np.minimum((X / die_len * ncx).astype(int), ncx - 1)
    cj = np.minimum((Y / die_wid * ncy).astype(int), ncy - 1)
    w = util[ci, cj]

    # asymmetric large-scale activity tilt (one die half busier)
    w *= 1.0 + 0.12 * (X / die_len - 0.5) * 2.0
    w *= 1.0 - 0.06 * (Y / die_wid - 0.5) * 2.0

    # central L2 / NoC column: cooler than compute but far from power-free
    w = np.where(np.abs(X - die_len / 2.0) < L2_HALF_WIDTH, L2_FACTOR * w, w)

    # low-power IO ring at the die edge
    edge = ((X < IO_RING) | (X > die_len - IO_RING) |
            (Y < IO_RING) | (Y > die_wid - IO_RING))
    w = np.where(edge, IO_FACTOR * w, w)
    return w


def cluster_power_tiles(die_len, die_wid, total_power, p2a,
                        tile=POWER_TILE, seed=MAP_SEED):
    """Structured asymmetric GPU power map. Returns (blocks, {name: power_W});
    blocks are die-local (caller offsets them via add_package_ring)."""
    w = _cluster_field(die_len, die_wid, tile, seed)
    # scale contrast about the mean so max/mean == p2a, clamp, renormalise
    mean = w.mean()
    raw = w.max() / mean
    if raw <= 1.0:
        raise SystemExit("cluster map: degenerate (flat) raw field")
    s = (p2a - 1.0) / (raw - 1.0)
    w = np.maximum(mean * 0.05, mean + s * (w - mean))
    p = total_power * w / w.sum()
    nx, ny = w.shape
    blocks, power = [], {}
    for i in range(nx):
        for j in range(ny):
            name = f"GPU_{i:03d}_{j:03d}"
            blocks.append((name, i * tile, j * tile, tile, tile, "GPU_FEOL"))
            power[name] = float(p[i, j])
    return blocks, power
