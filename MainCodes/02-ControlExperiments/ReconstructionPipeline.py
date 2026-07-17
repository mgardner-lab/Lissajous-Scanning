"""Reconstruct 28x28 images from Lissajous-scanned signals.

Linear (MATLAB-faithful) reconstruction:
  - Delaunay triangulation of the scan-point coordinates.
  - For each pixel in the 28x28 query grid, linear (barycentric) interpolation
    from the triangle that contains it.
  - Pixels outside the convex hull -> 0 (matches MATLAB `uint8(NaN) = 0`).
  - Signal divided by 4 to compensate for our bilinear scan weights so the
    intensity matches MATLAB's [0, 255] range, then cast to uint8.

Because the scan coordinates are identical for every image (fixed start),
the Delaunay triangulation is built ONCE and reused for all images. Per-image
cost is then just a barycentric weighted sum (microseconds).

Run:    python ReconstructionPipeline.py
Input:  data/N{N}_fx{FX}_fy{FY}_fs{FS}/{train,test}.pkl     (from DatasetBuilder)
Output: data/N{N}_fx{FX}_fy{FY}_fs{FS}/{train,test}_recon.npy   uint8 (n, 28, 28)
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

# --- Configuration ---
N = 200
FX = 37
FY = 23
FS = 800
DATA_DIR = (Path(__file__).resolve().parent / "data"
            / f"N{N}_fx{FX}_fy{FY}_fs{FS}")

IMAGE_SIZE = 28


def query_grid():
    """28x28 query grid in MATLAB 1-based coordinates."""
    xq, yq = np.meshgrid(np.arange(1, IMAGE_SIZE + 1, dtype=np.float64),
                         np.arange(1, IMAGE_SIZE + 1, dtype=np.float64))
    return np.stack([xq.ravel(), yq.ravel()], axis=-1)


def reconstruct_split(split_name):
    in_pkl = DATA_DIR / f"{split_name}.pkl"
    out_npy = DATA_DIR / f"{split_name}_recon.npy"

    with open(in_pkl, "rb") as f:
        data = pickle.load(f)
    signals = data["signals"]
    coords  = np.stack([data["coords_x"], data["coords_y"]], axis=-1)
    grid    = query_grid()

    # Build the triangulation once (re-used for every image because fixed start).
    interp = LinearNDInterpolator(coords, (signals[0] / 4.0).astype(np.float64),
                                  fill_value=0.0)

    n = len(signals)
    out = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i in range(n):
        interp.values = (signals[i] / 4.0).astype(np.float64).reshape(-1, 1)
        img = interp(grid).reshape(IMAGE_SIZE, IMAGE_SIZE)
        out[i] = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
        if (i + 1) % 5000 == 0:
            print(f"  {split_name}: {i + 1}/{n}", flush=True)

    np.save(out_npy, out)
    print(f"  saved {out_npy}  shape={out.shape}  dtype={out.dtype}")


if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found — run DatasetBuilder.py first.")
    print(f"Reconstructing N={N}, fx={FX}, fy={FY}, fs={FS} (linear, MATLAB-faithful)")
    reconstruct_split("train")
    reconstruct_split("test")
