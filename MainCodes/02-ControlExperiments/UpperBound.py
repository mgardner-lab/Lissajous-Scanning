"""Upper Bound (UB) -- train a 2D CNN on clean MNIST.

Establishes the architecture ceiling: what accuracy each 2D model can reach
on the easiest possible input (the original MNIST image, no Lissajous scan).
This is the architecture-sanity check used by the Q4b control experiment.

Supports three architectures via MODEL_NAME:
  small_cnn   -- LeNet-style 2D CNN (28x28 input)
  alexnet     -- torchvision AlexNet, random init (224x224x3, resize+replicate)
  resnet18    -- torchvision ResNet18, random init (224x224x3, resize+replicate)

Run:    python UpperBound.py
Output: checkpoints/UB_{MODEL_NAME}_seed{SEED}.pth
        metrics/UB_{MODEL_NAME}_seed{SEED}.csv
"""

import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

# --- Configuration ---
MODEL_NAME = "small_cnn"            # small_cnn | alexnet | resnet18
SEED = 42
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64
LR = 0.001

ROOT = Path(__file__).resolve().parent
MNIST_ROOT = ROOT / "data" / "mnist_raw"
CKPT_DIR = ROOT / "checkpoints"
METRIC_DIR = ROOT / "metrics"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Architectures ===
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


# === Data wrapper: MNIST -> tensor, resized + 3-channel for ImageNet archs ===
class MNISTWrap(torch.utils.data.Dataset):
    def __init__(self, ds, input_shape):
        self.ds = ds
        self.input_shape = input_shape
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, lbl = self.ds[i]
        x = self.to_tensor(img)                                     # (1, 28, 28)
        x = (x - 0.1307) / 0.3081                                   # MNIST norm
        c, h, w = self.input_shape
        if (c, h, w) != (1, 28, 28):
            x = F.interpolate(x.unsqueeze(0), size=h, mode="bilinear",
                              align_corners=False).squeeze(0)
            x = x.repeat(c, 1, 1)
        return x, int(lbl)


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model, in_shape = build_model(MODEL_NAME)
    model = model.to(DEVICE)
    print(f"Model: {MODEL_NAME}  |  input: {in_shape}  |  "
          f"params: {sum(p.numel() for p in model.parameters()):,}  |  device: {DEVICE}")

    train_ds = torchvision.datasets.MNIST(str(MNIST_ROOT), train=True, download=True)
    test_ds  = torchvision.datasets.MNIST(str(MNIST_ROOT), train=False, download=True)
    train_ld = torch.utils.data.DataLoader(MNISTWrap(train_ds, in_shape),
                                           batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_ld  = torch.utils.data.DataLoader(MNISTWrap(test_ds, in_shape),
                                           batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

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
    ckpt = CKPT_DIR / f"UB_{MODEL_NAME}_seed{SEED}.pth"
    torch.save({"model_name": MODEL_NAME, "state_dict": best_state,
                "best_val_acc": best_acc, "input_shape": in_shape}, ckpt)
    with open(METRIC_DIR / f"UB_{MODEL_NAME}_seed{SEED}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writeheader(); w.writerows(metrics)

    print(f"\nBest val_acc: {best_acc:.2f}%  |  train time: {train_time/60:.1f} min")
    print(f"Saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    train()
