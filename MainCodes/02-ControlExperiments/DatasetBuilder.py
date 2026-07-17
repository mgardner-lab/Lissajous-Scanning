"""Build a Lissajous-scanned dataset from MNIST.

For each MNIST image, scans the image along the Lissajous trajectory
    x(t) = cos(2*pi*fx*t)
    y(t) = cos(2*pi*fy*t + pi/2)
at N sample points (fixed scan start) and records the bilinearly-interpolated
intensity at each sample. The scan coordinates are identical for every image
(fixed start), which means downstream reconstruction can amortize the
triangulation across all images.

Writes a single pickle per split containing:
    signals    (n, N)        float32      bilinearly-sampled intensity
    coords_x   (N,)           float64      1-based fractional x of each sample
    coords_y   (N,)           float64      1-based fractional y of each sample
    labels     (n,)           int64        MNIST digit labels

Run:    python DatasetBuilder.py
Output: data/N{N}_fx{FX}_fy{FY}_fs{FS}/{train,test}.pkl
"""

import pickle
from pathlib import Path

import numpy as np
import torchvision

# --- Configuration ---
N = 200
FX = 37
FY = 23
FS = 800
MNIST_ROOT = Path(__file__).resolve().parent / "data" / "mnist_raw"
OUT_DIR = (Path(__file__).resolve().parent / "data"
           / f"N{N}_fx{FX}_fy{FY}_fs{FS}")

IMAGE_SIZE = 28


def scan_coords(N, fx, fy, fs):
    """Return (coords_x, coords_y) — the 1-based fractional pixel coordinates
    visited by the Lissajous scan. Shared across every image (fixed start)."""
    t = np.arange(N) / fs + 1.0 / (2.0 * fs)
    x = np.cos(2.0 * np.pi * fx * t)
    y = np.cos(2.0 * np.pi * fy * t + np.pi / 2.0)
    xim = (x + 1.0) * (IMAGE_SIZE - 1) / 2.0 + 1.0
    yim = (y + 1.0) * (IMAGE_SIZE - 1) / 2.0 + 1.0
    return xim, yim


def bilinear_sample(img, xim, yim):
    """Bilinearly sample one image at the given 1-based fractional coords."""
    sig = np.zeros(len(xim), dtype=np.float32)
    h, w = img.shape
    for i in range(len(xim)):
        x1 = max(0, min(w - 1, int(np.floor(xim[i])) - 1))
        x2 = max(0, min(w - 1, int(np.ceil(xim[i])) - 1))
        y1 = max(0, min(h - 1, int(np.floor(yim[i])) - 1))
        y2 = max(0, min(h - 1, int(np.ceil(yim[i])) - 1))
        a, b, c, d = img[y1, x1], img[y2, x1], img[y1, x2], img[y2, x2]
        wx2, wx1 = xim[i] - x1, x2 + 1 - xim[i]
        wy2, wy1 = yim[i] - y1, y2 + 1 - yim[i]
        sig[i] = a * wx1 * wy1 + b * wx1 * wy2 + c * wx2 * wy1 + d * wx2 * wy2
    return sig


def build_split(split_name, mnist_split, coords_x, coords_y):
    n = len(mnist_split)
    signals = np.zeros((n, N), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        img, lbl = mnist_split[i]
        signals[i] = bilinear_sample(np.array(img, dtype=np.float32),
                                     coords_x, coords_y)
        labels[i] = lbl
        if (i + 1) % 5000 == 0:
            print(f"  {split_name}: {i + 1}/{n}", flush=True)
    out_path = OUT_DIR / f"{split_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"signals": signals, "labels": labels,
                     "coords_x": coords_x, "coords_y": coords_y}, f)
    print(f"  saved {out_path}  signals={signals.shape}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building dataset N={N}, fx={FX}, fy={FY}, fs={FS}")
    print(f"  -> {OUT_DIR}")
    coords_x, coords_y = scan_coords(N, FX, FY, FS)
    mnist_train = torchvision.datasets.MNIST(str(MNIST_ROOT), train=True, download=True)
    mnist_test  = torchvision.datasets.MNIST(str(MNIST_ROOT), train=False, download=True)
    build_split("train", mnist_train, coords_x, coords_y)
    build_split("test",  mnist_test,  coords_x, coords_y)
