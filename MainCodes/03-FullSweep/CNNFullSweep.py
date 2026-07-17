"""1D CNN full hyperparameter sweep on Lissajous-scanned MNIST.

Sweeps every combination of (filter_size, num_filters, dropout, lr) at each
sample size N, using the per-N Lissajous winner (fx, fy, fs). Builds the
Lissajous datasets on first use (one pickle per N), then trains every config,
records accuracy + precision + recall + F1 + 10k inference time + train time
+ parameter count, and writes a summary CSV at the end.

Resume: any config with an existing results/<name>/final_metrics.json is
skipped, so the script is safe to stop and restart.

Run:    python CNNFullSweep.py
Output: results/<name>/{metrics.csv, final_metrics.json}     per config
        results/summary.csv                                  aggregated table
"""

import copy
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# === Sweep grid (edit here) ===
FILTER_SIZES = (11, 21, 31, 41, 71, 81, 91)
NUM_FILTERS = (32, 64)
DROPOUTS = (0.0, 0.2)
LEARNING_RATES = (0.001, 0.01)
SAMPLE_SIZES = (50, 100, 200, 500)
SEEDS = (42,)

# Per-N Lissajous winner (fx, fy, fs) at fixed scan start.
LISSAJOUS_PER_N = {
    50:  (43, 37, 350),
    100: (17, 39, 450),
    200: (37, 23, 800),
    500: (39, 41, 1200),
}

# Training recipe.
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64
NUM_CLASSES = 10

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MNIST_ROOT = DATA_DIR / "mnist_raw"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 28


# === Model ===
class CNN1D(nn.Module):
    """Two-layer 1D conv with global pooling and FC classifier."""
    def __init__(self, num_classes=10, num_filters=32, filter_size=21, dropout=0.0):
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


# === Lissajous scan + dataset build (one pickle per N) ===
def lissajous_scan(img, fx, fy, fs, numsam):
    """Sample one image with bilinear interpolation along the Lissajous trajectory."""
    h, w = img.shape
    t = np.arange(numsam) / fs + 1.0 / (2.0 * fs)
    x = np.cos(2.0 * np.pi * fx * t)
    y = np.cos(2.0 * np.pi * fy * t + np.pi / 2.0)
    xim = (x + 1.0) * (w - 1) / 2.0 + 1.0
    yim = (y + 1.0) * (h - 1) / 2.0 + 1.0
    out = np.zeros(numsam, dtype=np.float32)
    for i in range(numsam):
        x1 = max(0, min(w - 1, int(np.floor(xim[i])) - 1))
        x2 = max(0, min(w - 1, int(np.ceil(xim[i])) - 1))
        y1 = max(0, min(h - 1, int(np.floor(yim[i])) - 1))
        y2 = max(0, min(h - 1, int(np.ceil(yim[i])) - 1))
        a, b, c, d = img[y1, x1], img[y2, x1], img[y1, x2], img[y2, x2]
        wx2, wx1 = xim[i] - x1, x2 + 1 - xim[i]
        wy2, wy1 = yim[i] - y1, y2 + 1 - yim[i]
        out[i] = a * wx1 * wy1 + b * wx1 * wy2 + c * wx2 * wy1 + d * wx2 * wy2
    return out


def ensure_dataset(N, fx, fy, fs):
    """Build {train,test}.pkl for one (N, fx, fy, fs) if not already present."""
    sub = DATA_DIR / f"N{N}_fx{fx}_fy{fy}_fs{fs}"
    train_pkl, test_pkl = sub / "train.pkl", sub / "test.pkl"
    if train_pkl.exists() and test_pkl.exists():
        return sub
    sub.mkdir(parents=True, exist_ok=True)
    mnist_train = torchvision.datasets.MNIST(str(MNIST_ROOT), train=True, download=True)
    mnist_test  = torchvision.datasets.MNIST(str(MNIST_ROOT), train=False, download=True)
    for split, mnist in (("train", mnist_train), ("test", mnist_test)):
        path = sub / f"{split}.pkl"
        if path.exists():
            continue
        n = len(mnist)
        signals = np.zeros((n, N), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)
        print(f"  [BUILD] {path.relative_to(ROOT)}  ({n} images)")
        for i in range(n):
            img, lbl = mnist[i]
            signals[i] = lissajous_scan(np.array(img, dtype=np.float32), fx, fy, fs, N)
            labels[i] = lbl
        with open(path, "wb") as f:
            pickle.dump({"signals": signals, "labels": labels}, f)
    return sub


class SignalDataset(Dataset):
    def __init__(self, pkl_path, mean=None, std=None):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        self.signals = d["signals"]; self.labels = d["labels"]
        self.mean = float(self.signals.mean()) if mean is None else mean
        self.std = float(self.signals.std()) if std is None else std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        s = (self.signals[i] - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(s).unsqueeze(0).float(), int(self.labels[i])


# === Training / metrics / latency ===
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            preds.append(model(x).argmax(1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def measure_inference_10k(model, N, n_warmup=3, n_runs=3, test_size=10000):
    """End-to-end forward latency over a full test_size pass, batched at BATCH_SIZE.

    Allocates a real (test_size, 1, N) tensor on the device and runs inference
    in BATCH_SIZE chunks, mirroring how the actual test set is consumed. The
    full pass is timed n_runs times (no scaling); returns the median full-pass
    duration in seconds.
    """
    model.eval()
    x = torch.randn(test_size, 1, N, device=DEVICE)
    runs = []
    with torch.no_grad():
        for _ in range(n_warmup):
            for i in range(0, test_size, BATCH_SIZE):
                _ = model(x[i:i + BATCH_SIZE])
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        for _ in range(n_runs):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(0, test_size, BATCH_SIZE):
                _ = model(x[i:i + BATCH_SIZE])
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            runs.append(time.perf_counter() - t0)
    return float(np.median(runs))


def train_one(N, fx, fy, fs, fsize, nfilt, drop, lr, seed, run_dir):
    """Train a single config, write metrics.csv + final_metrics.json. Return best val_acc."""
    torch.manual_seed(seed); np.random.seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)

    sub = ensure_dataset(N, fx, fy, fs)
    train_ds = SignalDataset(sub / "train.pkl")
    test_ds = SignalDataset(sub / "test.pkl", mean=train_ds.mean, std=train_ds.std)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = CNN1D(NUM_CLASSES, nfilt, fsize, drop).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    run_dir.mkdir(parents=True, exist_ok=True)
    best_state, best_acc, best_epoch, no_improve = None, 0.0, 0, 0
    t0 = time.perf_counter()

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(EPOCHS):
            model.train()
            tl = tc = tt = 0
            for x, y in train_ld:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                out = model(x); loss = crit(out, y); loss.backward(); opt.step()
                tl += loss.item() * x.size(0); tc += (out.argmax(1) == y).sum().item(); tt += x.size(0)
            sched.step()

            model.eval()
            vl = vc = vt = 0
            with torch.no_grad():
                for x, y in test_ld:
                    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                    out = model(x)
                    vl += crit(out, y).item() * x.size(0); vc += (out.argmax(1) == y).sum().item(); vt += x.size(0)

            tr_acc, val_acc = 100 * tc / tt, 100 * vc / vt
            w.writerow([epoch + 1, tl / tt, tr_acc, vl / vt, val_acc])
            f.flush()

            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

    train_time = time.perf_counter() - t0

    # Restore best weights for final metrics.
    if best_state is not None:
        model.load_state_dict(best_state)
    preds, labels = evaluate(model, test_ld)
    accuracy = float((preds == labels).mean())
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    inf_s = measure_inference_10k(model, N)

    final = {
        "N": N, "fx": fx, "fy": fy, "fs": fs,
        "filter_size": fsize, "num_filters": nfilt, "dropout": drop, "lr": lr, "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "accuracy": accuracy * 100,
        "precision_macro": float(p) * 100,
        "recall_macro": float(r) * 100,
        "f1_macro": float(f1) * 100,
        "inference_10k_s": inf_s,
        "inference_device": DEVICE,
        "train_time_s": train_time,
        "n_params": int(sum(pp.numel() for pp in model.parameters())),
    }
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    return best_acc, best_epoch


# === Sweep orchestration ===
def exp_name(N, fx, fy, fs, fsize, nfilt, drop, lr, seed):
    return (f"N{N}_fx{fx}_fy{fy}_fs{fs}_fsize{fsize}_nf{nfilt}"
            f"_drop{drop}_lr{lr}_seed{seed}").replace(".", "p")


def write_summary():
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []:
        path = d / "final_metrics.json"
        if path.is_file():
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote summary: {RESULTS_DIR/'summary.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(N, *LISSAJOUS_PER_N[N], fsize, nfilt, drop, lr, seed)
            for N in SAMPLE_SIZES for fsize in FILTER_SIZES
            for nfilt in NUM_FILTERS for drop in DROPOUTS
            for lr in LEARNING_RATES for seed in SEEDS]
    print(f"Device: {DEVICE}  |  jobs: {len(jobs)}")
    t_sweep = time.perf_counter()
    ok = skipped = failed = 0
    for i, job in enumerate(jobs, 1):
        name = exp_name(*job)
        run_dir = RESULTS_DIR / name
        if (run_dir / "final_metrics.json").exists():
            print(f"[{i:>4}/{len(jobs)}] {name}  SKIP"); skipped += 1; continue
        t0 = time.perf_counter()
        try:
            acc, ep = train_one(*job, run_dir=run_dir)
            print(f"[{i:>4}/{len(jobs)}] {name}  OK  acc={acc:.2f}% ep={ep} "
                  f"({time.perf_counter()-t0:.0f}s)")
            ok += 1
        except Exception as e:
            print(f"[{i:>4}/{len(jobs)}] {name}  FAIL  {e}"); failed += 1
    print(f"\nDone in {(time.perf_counter()-t_sweep)/60:.1f} min  |  "
          f"{ok} ok, {skipped} skipped, {failed} failed")
    write_summary()
