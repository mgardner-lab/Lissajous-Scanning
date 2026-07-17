"""Realistic Bound (RB) -- train a 2D CNN on Lissajous-reconstructed images.

This is the competing 2D method that the paper's 1D CNN is being compared
against: take the sparse Lissajous scan, reconstruct it into a 28x28 image,
then classify the image with a standard 2D CNN.

Train and test data are the linear-reconstructed images produced by
ReconstructionPipeline.py. Architectures are the same as UpperBound.py.

Run:    python RealisticBound.py
Input:  data/N{N}_fx{FX}_fy{FY}_fs{FS}/{train,test}_recon.npy
        data/N{N}_fx{FX}_fy{FY}_fs{FS}/{train,test}.pkl     (for labels)
Output: checkpoints/RB_{MODEL_NAME}_N{N}_seed{SEED}.pth
        metrics/RB_{MODEL_NAME}_N{N}_seed{SEED}.csv
"""

import copy
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# --- Configuration ---
N = 200
FX = 37
FY = 23
FS = 800
MODEL_NAME = "small_cnn"            # small_cnn | alexnet | resnet18
SEED = 42
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64
LR = 0.001

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / f"N{N}_fx{FX}_fy{FY}_fs{FS}"
CKPT_DIR = ROOT / "checkpoints"
METRIC_DIR = ROOT / "metrics"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    """LeNet-style native 28x28 baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def build_model(name):
    if name == "small_cnn":
        return SmallCNN(num_classes=10), (1, 28, 28)
    if name == "alexnet":
        return torchvision.models.alexnet(num_classes=10, weights=None), (3, 224, 224)
    if name == "resnet18":
        return torchvision.models.resnet18(num_classes=10, weights=None), (3, 224, 224)
    raise ValueError(f"unknown model {name!r}")


class ReconDataset(torch.utils.data.Dataset):
    """Loads reconstructed 28x28 uint8 images + MNIST labels (from the source pickle)."""
    def __init__(self, recon_npy, labels_pkl, input_shape):
        self.images = np.load(recon_npy)                            # (n, 28, 28) uint8
        with open(labels_pkl, "rb") as f:
            self.labels = pickle.load(f)["labels"]                  # (n,) int64
        # Pre-normalise once for speed (MNIST per-pixel mean/std).
        x = self.images.astype(np.float32) / 255.0
        self.x = (x - 0.1307) / 0.3081
        self.input_shape = input_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i]).unsqueeze(0)                # (1, 28, 28)
        c, h, w = self.input_shape
        if (c, h, w) != (1, 28, 28):
            x = F.interpolate(x.unsqueeze(0), size=h, mode="bilinear",
                              align_corners=False).squeeze(0)
            x = x.repeat(c, 1, 1)
        return x, int(self.labels[i])


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not (DATA_DIR / "train_recon.npy").exists():
        raise FileNotFoundError(
            f"{DATA_DIR/'train_recon.npy'} missing -- run ReconstructionPipeline.py first.")

    model, in_shape = build_model(MODEL_NAME)
    model = model.to(DEVICE)
    print(f"Model: {MODEL_NAME}  |  input: {in_shape}  |  N={N}  |  "
          f"params: {sum(p.numel() for p in model.parameters()):,}  |  device: {DEVICE}")

    train_ds = ReconDataset(DATA_DIR / "train_recon.npy", DATA_DIR / "train.pkl", in_shape)
    test_ds  = ReconDataset(DATA_DIR / "test_recon.npy",  DATA_DIR / "test.pkl",  in_shape)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_ld  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state, no_improve = 0.0, None, 0
    metrics = []
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        tl = tc = tt = 0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0); tc += (out.argmax(1) == y).sum().item(); tt += x.size(0)

        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                vl += crit(out, y).item() * x.size(0); vc += (out.argmax(1) == y).sum().item(); vt += x.size(0)

        tr_acc, val_acc = 100 * tc / tt, 100 * vc / vt
        metrics.append({"epoch": epoch + 1, "train_loss": tl / tt, "train_acc": tr_acc,
                        "val_loss": vl / vt, "val_acc": val_acc})
        print(f"  epoch {epoch+1:>2}/{EPOCHS}  train={tr_acc:.2f}%  val={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc, best_state, no_improve = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch+1}"); break

    train_time = time.perf_counter() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"RB_{MODEL_NAME}_N{N}_seed{SEED}.pth"
    torch.save({"model_name": MODEL_NAME, "state_dict": best_state,
                "best_val_acc": best_acc, "input_shape": in_shape,
                "N": N, "fx": FX, "fy": FY, "fs": FS}, ckpt)
    with open(METRIC_DIR / f"RB_{MODEL_NAME}_N{N}_seed{SEED}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writeheader(); w.writerows(metrics)

    print(f"\nBest val_acc: {best_acc:.2f}%  |  train time: {train_time/60:.1f} min")
    print(f"Saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    train()
