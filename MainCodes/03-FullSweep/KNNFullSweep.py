"""1D KNN full hyperparameter sweep on Lissajous-scanned MNIST.

Sweeps every combination of (n_neighbors, metric) at each sample size N,
using the per-N Lissajous winner (fx, fy, fs). Each Lissajous signal is fed
to scikit-learn's KNeighborsClassifier as a flat feature vector of length N.

Builds the Lissajous datasets on first use (one pickle per N), fits the
classifier on the standardised training signals, predicts on the 10k test
signals, and records accuracy + precision + recall + F1 + inference time
(timed on the actual test set since KNN has no batched-forward concept).

Resume: any config with an existing results/<name>/final_metrics.json is
skipped, so the script is safe to stop and restart.

Run:    python KNNFullSweep.py
Output: results/<name>/final_metrics.json     per config
        results/summary.csv                   aggregated table
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torchvision
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

# === Sweep grid (edit here) ===
N_NEIGHBORS = (3, 5, 7, 15)
METRICS = ("euclidean", "manhattan")
WEIGHTS = ("uniform", "distance")        # uniform votes = majority; distance = inverse-distance weighted
SAMPLE_SIZES = (50, 100, 200, 500)
SEEDS = (42,)                       # KNN is deterministic; seed only affects subsetting (none here)

# Per-N Lissajous winner (fx, fy, fs) at fixed scan start.
LISSAJOUS_PER_N = {
    50:  (43, 37, 350),
    100: (17, 39, 450),
    200: (37, 23, 800),
    500: (39, 41, 1200),
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results_knn"
MNIST_ROOT = DATA_DIR / "mnist_raw"

TEST_SIZE = 10000


# === Lissajous scan + dataset build (one pickle per N) ===
def lissajous_scan(img, fx, fy, fs, numsam):
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


def load_split(pkl_path, mean=None, std=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    sig, lbl = d["signals"], d["labels"]
    if mean is None:
        mean, std = float(sig.mean()), float(sig.std())
    X = ((sig - mean) / (std + 1e-8)).astype(np.float32)
    return X, lbl, mean, std


# === Sweep ===
def fit_and_eval(N, fx, fy, fs, k, metric, weights, seed):
    sub = ensure_dataset(N, fx, fy, fs)
    X_train, y_train, mean, std = load_split(sub / "train.pkl")
    X_test, y_test, _, _ = load_split(sub / "test.pkl", mean=mean, std=std)

    t_fit_start = time.perf_counter()
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t_fit_start

    # Inference time on the full 10k test set (KNN cost = neighbour search,
    # so measuring on the real test set is the right comparison).
    t_pred_start = time.perf_counter()
    preds = clf.predict(X_test)
    pred_time = time.perf_counter() - t_pred_start

    accuracy = float((preds == y_test).mean())
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)

    return {
        "N": N, "fx": fx, "fy": fy, "fs": fs,
        "n_neighbors": k, "metric": metric, "weights": weights, "seed": seed,
        "accuracy": accuracy * 100,
        "precision_macro": float(p) * 100,
        "recall_macro": float(r) * 100,
        "f1_macro": float(f1) * 100,
        "inference_10k_s": pred_time,
        "fit_time_s": fit_time,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(N),
    }


def exp_name(N, fx, fy, fs, k, metric, weights, seed):
    return f"N{N}_fx{fx}_fy{fy}_fs{fs}_k{k}_{metric}_{weights}_seed{seed}"


def write_summary():
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()) if RESULTS_DIR.exists() else []:
        path = d / "final_metrics.json"
        if path.is_file():
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        return
    import csv
    cols = list(rows[0].keys())
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote summary: {RESULTS_DIR/'summary.csv'}  ({len(rows)} rows)")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(N, *LISSAJOUS_PER_N[N], k, metric, weights, seed)
            for N in SAMPLE_SIZES for k in N_NEIGHBORS
            for metric in METRICS for weights in WEIGHTS for seed in SEEDS]
    print(f"Jobs: {len(jobs)}")
    t_sweep = time.perf_counter()
    ok = skipped = failed = 0
    for i, job in enumerate(jobs, 1):
        name = exp_name(*job)
        run_dir = RESULTS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / "final_metrics.json"
        if out_json.exists():
            print(f"[{i:>3}/{len(jobs)}] {name}  SKIP"); skipped += 1; continue
        t0 = time.perf_counter()
        try:
            final = fit_and_eval(*job)
            with open(out_json, "w") as f:
                json.dump(final, f, indent=2)
            print(f"[{i:>3}/{len(jobs)}] {name}  OK  acc={final['accuracy']:.2f}%  "
                  f"infer={final['inference_10k_s']:.2f}s  ({time.perf_counter()-t0:.0f}s total)")
            ok += 1
        except Exception as e:
            print(f"[{i:>3}/{len(jobs)}] {name}  FAIL  {e}"); failed += 1
    print(f"\nDone in {(time.perf_counter()-t_sweep)/60:.1f} min  |  "
          f"{ok} ok, {skipped} skipped, {failed} failed")
    write_summary()
