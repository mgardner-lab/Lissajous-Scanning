"""Latency benchmark with SEQUENTIAL reconstruction.

  1D pipeline:  raw Lissajous signal -> 1D CNN forward
  2D pipeline:  raw Lissajous signal -> linear reconstruction -> 2D CNN forward

Both pipelines start from the same raw signal, so the only difference is the
reconstruction step that the 2D pipeline pays and the 1D one skips.

Sequential reconstruction
-------------------------
Reconstruction is done one image at a time via scipy.interpolate.griddata,
which rebuilds the Delaunay triangulation on every call. This is the worst
case: every frame pays the full triangulation cost. It matches what a random-
scan-start scanner would actually pay -- each image has different scan
coordinates, so no triangulation reuse is possible.

The companion file LatencyMeasurementAmortized.py shows the fair / realistic
implementation that builds the triangulation once (valid under fixed scan
start) and reuses it for every image.

Methodology
-----------
- Warm up to remove framework cold-start, then time many runs and take the
  median per-image time. Median is multiplied by the test-set size to report
  a 10k-image total.
- On GPU we call torch.cuda.synchronize() before and after each timing window
  so we measure GPU work, not async kernel-launch returns.

Run:  python LatencyMeasurementSequential.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import griddata

# --- Configuration ---
N = 200
FX = 37
FY = 23
FS = 800
FILTER_SIZE = 41
NUM_FILTERS = 32
BATCH = 64
N_IMAGES = 10000

DATA_PKL = (Path(__file__).resolve().parent / "data"
            / f"N{N}_fx{FX}_fy{FY}_fs{FS}" / "test.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === 1D CNN (matches the model trained in the full sweep) ===
class CNN1D(nn.Module):
    def __init__(self, num_filters=32, filter_size=41, num_classes=10, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_filters, filter_size, padding="same")
        self.norm1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, 2 * num_filters, filter_size, padding="same")
        self.norm2 = nn.BatchNorm1d(2 * num_filters)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(2 * num_filters, num_classes)

    def forward(self, x):
        x = self.norm1(torch.relu(self.conv1(x)))
        x = self.norm2(torch.relu(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(self.drop(x))


# === 2D CNN (LeNet-style, the 2D baseline) ===
class CNN2D(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc2(torch.relu(self.fc1(x)))


def median_time(fn, n_warmup=10, n_runs=50):
    """Warm up, then time many calls of fn() and return the median seconds."""
    for _ in range(n_warmup):
        fn()
    runs = []
    for _ in range(n_runs):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs))


def time_1d_pipeline():
    model = CNN1D(NUM_FILTERS, FILTER_SIZE).to(DEVICE).eval()
    batch = torch.randn(BATCH, 1, N, device=DEVICE)
    with torch.no_grad():
        per_batch = median_time(lambda: model(batch))
    return per_batch / BATCH                      # seconds per image


def time_2d_pipeline():
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)
    coords = np.stack([data["coords_x"], data["coords_y"]], axis=-1)
    signals = data["signals"]

    grid_x, grid_y = np.meshgrid(np.arange(1, 29), np.arange(1, 29))
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

    # Sequential: triangulation is rebuilt inside every griddata call.
    def reconstruct(signal):
        vals = (signal / 4.0).astype(np.float64)
        img = griddata(coords, vals, grid, method="linear", fill_value=0.0)
        return np.clip(img.reshape(28, 28), 0, 255).round().astype(np.float32)

    # Fewer runs because each call rebuilds the triangulation (slow).
    recon_per_image = median_time(lambda: reconstruct(signals[0]),
                                  n_warmup=3, n_runs=15)

    model = CNN2D().to(DEVICE).eval()
    batch = torch.randn(BATCH, 1, 28, 28, device=DEVICE)
    with torch.no_grad():
        fwd_per_batch = median_time(lambda: model(batch))
    fwd_per_image = fwd_per_batch / BATCH

    return recon_per_image, fwd_per_image


if __name__ == "__main__":
    print(f"Device: {DEVICE} | N={N} | scaling to {N_IMAGES} images\n")
    t_1d = time_1d_pipeline()
    recon, fwd_2d = time_2d_pipeline()
    t_2d = recon + fwd_2d

    print("1D pipeline  (signal -> 1D CNN):")
    print(f"  {t_1d*1e3:.4f} ms/image  ->  {t_1d*N_IMAGES:.3f} s for {N_IMAGES} images\n")

    print("2D pipeline  (signal -> sequential reconstruction -> 2D CNN):")
    print(f"  reconstruction: {recon*1e3:.4f} ms/image  (triangulation rebuilt per image)")
    print(f"  2D CNN forward: {fwd_2d*1e3:.4f} ms/image")
    print(f"  total:          {t_2d*1e3:.4f} ms/image  ->  {t_2d*N_IMAGES:.3f} s\n")

    print(f"Speedup (2D / 1D): {t_2d / t_1d:.2f}x")
