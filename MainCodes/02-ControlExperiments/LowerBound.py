"""Lower Bound -- evaluate a UB checkpoint on reconstructed images.

The Lower Bound takes a model trained on clean MNIST (the UB checkpoint) and
runs it on the Lissajous-reconstructed test set. The accuracy drop measures
how far the reconstruction has shifted the input distribution away from clean
MNIST. A large drop is expected at sparse N -- it confirms that reconstruction
creates out-of-distribution input that a model not exposed to reconstruction
artifacts cannot handle.

No training. Just loads the UB checkpoint, runs inference on the
reconstructed test set, reports accuracy.

Run:    python LowerBound.py
Input:  checkpoints/UB_{MODEL_NAME}_seed{SEED}.pth      (from UpperBound.py)
        data/N{N}_fx{FX}_fy{FY}_fs{FS}/test_recon.npy   (from ReconstructionPipeline.py)
        data/N{N}_fx{FX}_fy{FY}_fs{FS}/test.pkl         (for labels)
Output: metrics/LowerBound_{MODEL_NAME}_N{N}_seed{SEED}.txt
"""

import pickle
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
MODEL_NAME = "small_cnn"            # must match an existing UB checkpoint
SEED = 42
BATCH_SIZE = 256

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / f"N{N}_fx{FX}_fy{FY}_fs{FS}"
CKPT_DIR = ROOT / "checkpoints"
METRIC_DIR = ROOT / "metrics"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
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


def load_recon_test(input_shape):
    """Reconstructed test images, normalised the same way the training data was."""
    images = np.load(DATA_DIR / "test_recon.npy")                   # (n, 28, 28) uint8
    with open(DATA_DIR / "test.pkl", "rb") as f:
        labels = pickle.load(f)["labels"]
    x = images.astype(np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    x = torch.from_numpy(x).unsqueeze(1)                            # (n, 1, 28, 28)
    c, h, w = input_shape
    if (c, h, w) != (1, 28, 28):
        x = F.interpolate(x, size=h, mode="bilinear", align_corners=False)
        x = x.repeat(1, c, 1, 1)
    return x, torch.from_numpy(labels).long()


def main():
    ckpt_path = CKPT_DIR / f"UB_{MODEL_NAME}_seed{SEED}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} -- run UpperBound.py first.")
    if not (DATA_DIR / "test_recon.npy").exists():
        raise FileNotFoundError(
            f"{DATA_DIR/'test_recon.npy'} -- run ReconstructionPipeline.py first.")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model, in_shape = build_model(MODEL_NAME)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE).eval()
    print(f"Loaded UB checkpoint: {ckpt_path.name}  (best_val_acc on clean MNIST: "
          f"{ckpt['best_val_acc']:.2f}%)")

    x, y = load_recon_test(in_shape)
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            xb = x[i:i + BATCH_SIZE].to(DEVICE)
            yb = y[i:i + BATCH_SIZE].to(DEVICE)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    accuracy = 100 * correct / total

    print(f"\nLower Bound accuracy (UB checkpoint on reconstructed N={N} test set): {accuracy:.2f}%")
    print(f"  drop from clean MNIST: {ckpt['best_val_acc'] - accuracy:+.2f} pp")

    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    out = METRIC_DIR / f"LowerBound_{MODEL_NAME}_N{N}_seed{SEED}.txt"
    with open(out, "w") as f:
        f.write(f"model={MODEL_NAME}  N={N}  fx={FX} fy={FY} fs={FS}  seed={SEED}\n")
        f.write(f"UB_val_acc_clean_MNIST           = {ckpt['best_val_acc']:.4f}\n")
        f.write(f"LowerBound_acc_recon_test_set    = {accuracy:.4f}\n")
        f.write(f"drop_pp                          = {ckpt['best_val_acc'] - accuracy:.4f}\n")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
