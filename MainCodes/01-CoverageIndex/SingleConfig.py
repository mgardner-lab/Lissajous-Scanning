"""Coverage Index for a single Lissajous configuration.

Computes CI = C * U where
    C = fraction of grid cells visited by the scan       (coverage)
    U = normalized Shannon entropy of visit distribution (uniformity)

A 14x14 grid is overlaid on the 28x28 image (one cell = 2x2 pixels).
The scan uses a fixed start (offset_t0 = 0) — matching the dataset builder
in 02_ControlExperiments — so CI is deterministic for a given (N, fx, fy, fs).

Edit the constants below and run:  python SingleConfig.py
"""

import math
import numpy as np

# --- Configuration ---
N = 200           # samples per scan
FX = 37
FY = 23
FS = 800

GRID_SIZE = 14
IMAGE_SIZE = 28


def lissajous_coords(N, fx, fy, fs):
    """Pixel (x, y) coordinates of the scan trajectory (0-based), fixed start."""
    t = np.arange(N) / fs
    x = np.cos(2.0 * np.pi * fx * t)
    y = np.cos(2.0 * np.pi * fy * t + np.pi / 2.0)     # symmetric phase
    xim = (x + 1.0) * (IMAGE_SIZE - 1) / 2.0 + 1.0
    yim = (y + 1.0) * (IMAGE_SIZE - 1) / 2.0 + 1.0
    xc = np.clip(np.floor(xim - 1.0).astype(np.int32), 0, IMAGE_SIZE - 1)
    yc = np.clip(np.floor(yim - 1.0).astype(np.int32), 0, IMAGE_SIZE - 1)
    return xc, yc


def coverage_index(N, fx, fy, fs):
    """Return (C, U, CI) for the fixed-start scan."""
    x, y = lissajous_coords(N, fx, fy, fs)
    cell = IMAGE_SIZE / GRID_SIZE
    xg = np.minimum((x // cell).astype(np.int32), GRID_SIZE - 1)
    yg = np.minimum((y // cell).astype(np.int32), GRID_SIZE - 1)
    counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    np.add.at(counts, (yg, xg), 1)

    occupied = counts[counts > 0].astype(np.float64)
    k = len(occupied)
    C = k / counts.size
    if k <= 1:
        U = 1.0
    else:
        p = occupied / occupied.sum()
        U = float(-np.sum(p * np.log(p)) / np.log(k))
    return C, U, C * U


if __name__ == "__main__":
    if math.gcd(FX, FY) != 1:
        raise ValueError(f"fx={FX}, fy={FY} must be coprime")
    if FS < 2 * max(FX, FY):
        raise ValueError(f"fs={FS} must be >= 2*max(fx,fy) (Nyquist)")

    C, U, CI = coverage_index(N, FX, FY, FS)
    print(f"Lissajous config:  N={N}  fx={FX}  fy={FY}  fs={FS}  (fixed start)")
    print(f"  Coverage      C  = {C:.4f}")
    print(f"  Uniformity    U  = {U:.4f}")
    print(f"  Coverage Idx  CI = {CI:.4f}")
