"""Coverage Index sweep over (fx, fy, fs) at multiple sample sizes N.

For each N in SAMPLE_SIZES, evaluates every coprime (fx, fy) pair from
FX_VALUES x FY_VALUES against every fs in FS_VALUES that satisfies the
Nyquist condition (fs >= 2*max(fx, fy)). Writes one ranked CSV per N plus
a summary CSV listing the best CI configuration at each N.

Uses a fixed scan start (offset_t0 = 0) to match the dataset builder in
02_ControlExperiments. See SingleConfig.py for the CI = C * U formula.
Edit constants below and run:  python FullPipeline.py
"""

import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuration ---
SAMPLE_SIZES = (50, 100, 200, 500)
FX_VALUES = tuple(range(11, 48, 2))          # 11, 13, ..., 47
FY_VALUES = tuple(range(11, 48, 2))
FS_VALUES = (200, 250, 300, 350, 400, 450, 500, 560, 620,
             700, 750, 800, 900, 1000, 1100, 1200)

GRID_SIZE = 14
IMAGE_SIZE = 28
OUT_DIR = Path(__file__).resolve().parent / "results"


def lissajous_coords(N, fx, fy, fs):
    t = np.arange(N) / fs
    x = np.cos(2.0 * np.pi * fx * t)
    y = np.cos(2.0 * np.pi * fy * t + np.pi / 2.0)
    xim = (x + 1.0) * (IMAGE_SIZE - 1) / 2.0 + 1.0
    yim = (y + 1.0) * (IMAGE_SIZE - 1) / 2.0 + 1.0
    xc = np.clip(np.floor(xim - 1.0).astype(np.int32), 0, IMAGE_SIZE - 1)
    yc = np.clip(np.floor(yim - 1.0).astype(np.int32), 0, IMAGE_SIZE - 1)
    return xc, yc


def coverage_index(N, fx, fy, fs):
    """Return (C, U, CI) for the fixed-start scan."""
    cell = IMAGE_SIZE / GRID_SIZE
    x, y = lissajous_coords(N, fx, fy, fs)
    xg = np.minimum((x // cell).astype(np.int32), GRID_SIZE - 1)
    yg = np.minimum((y // cell).astype(np.int32), GRID_SIZE - 1)
    counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    np.add.at(counts, (yg, xg), 1)
    occ = counts[counts > 0].astype(np.float64)
    k = len(occ)
    C = k / counts.size
    if k <= 1:
        U = 1.0
    else:
        p = occ / occ.sum()
        U = float(-np.sum(p * np.log(p)) / np.log(k))
    return float(C), float(U), float(C * U)


def is_valid(fx, fy, fs):
    """Coprime frequencies + Nyquist sampling."""
    return math.gcd(fx, fy) == 1 and fs >= 2 * max(fx, fy)


def run_sweep(N):
    rows = []
    for fx, fy, fs in product(FX_VALUES, FY_VALUES, FS_VALUES):
        if not is_valid(fx, fy, fs):
            continue
        C, U, CI = coverage_index(N, fx, fy, fs)
        rows.append({
            "N": N, "fx": fx, "fy": fy, "fs": fs,
            "C": round(C, 4),
            "U": round(U, 4),
            "CI": round(CI, 4),
        })
    df = pd.DataFrame(rows).sort_values("CI", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for N in SAMPLE_SIZES:
        print(f"Sweeping N={N} ...")
        df = run_sweep(N)
        path = OUT_DIR / f"CI_sweep_N{N}.csv"
        df.to_csv(path, index=False)
        best = df.iloc[0]
        print(f"  {len(df)} configs  |  best: fx={int(best.fx)} fy={int(best.fy)} "
              f"fs={int(best.fs)}  CI={best.CI:.4f}  ->  {path.name}")
        summary.append({"N": N, "best_fx": int(best.fx), "best_fy": int(best.fy),
                        "best_fs": int(best.fs), "best_CI": float(best.CI)})
    summary_path = OUT_DIR / "CI_sweep_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
