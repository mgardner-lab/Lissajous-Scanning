# Lissajous-MNIST Code and Results Submission

Reproduction package for the paper *"Rapid Reconstruction-Free Object Recognition from Lissajous Sparse Sampling Using Deep Learning"*.

The project studies a reconstruction-free object-recognition pipeline in which a Lissajous trajectory sparsely scans an MNIST image into a 1D temporal signal. The sampled signal is classified directly using 1D learning models, avoiding the conventional intermediate step of reconstructing the full 2D image before classification.

The submission is organized into two main parts:

1. `MainCodes/` — all source code required to reproduce the experiments.
2. `Results/` — exported result files, trained weights, and confusion matrices for the selected top-performing models.

---

## Repository Structure

```text
Submission/
├── README.md
├── requirements.txt
├── MainCodes/
│   ├── 01-CoverageIndex/          # Coverage-Index pre-screening of Lissajous parameters
│   │   ├── SingleConfig.py
│   │   └── FullPipeline.py
│   │
│   ├── 02-ControlExperiments/     # Reconstruction-control experiments
│   │   ├── DatasetBuilder.py
│   │   ├── ReconstructionPipeline.py
│   │   ├── LatencyMeasurement.py
│   │   ├── UpperBound.py          # Clean MNIST training
│   │   ├── RealisticBound.py      # Reconstructed-image training
│   │   └── LowerBound.py          # UB model evaluated on reconstructed test set
│   │
│   └── 03-FullSweep/              # Full hyperparameter sweeps for 1D classifiers
│       ├── CNNFullSweep.py
│       ├── LSTMFullSweep.py
│       └── KNNFullSweep.py
│
└── Results/
    ├── 01-CNN/
    │   ├── ConfusionMatrices/     # 8 confusion matrices for the selected CNN models
    │   ├── Weights/               # Saved weights for the selected CNN models
    │   └── CNNFullSweep.xlsx      # Full CNN sweep results
    │
    ├── 02-LSTM/
    │   ├── ConfusionMatrices/     # Confusion matrices for the selected LSTM models
    │   ├── Weights/               # Saved weights for the selected LSTM models
    │   └── LSTMFullSweep.xlsx     # Full LSTM sweep results
    │
    └── 03-KNN/
        ├── ConfusionMatrices/     # Confusion matrices for the selected KNN models
        ├── Weights/               # Saved model artifacts for the selected KNN models
        └── KNNFullSweep.xlsx      # Full KNN sweep results
```

---

## Requirements

- Python 3.11+ tested on Python 3.13.13
- A CUDA-capable GPU is strongly recommended for the CNN and LSTM sweeps
- PyTorch tested with CUDA 12.4
- Approximately 3 GB of free disk space for cached Lissajous datasets

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch CUDA build that matches your local driver and CUDA runtime. The MNIST dataset is downloaded automatically the first time `DatasetBuilder.py` or any sweep script is executed through `torchvision.datasets.MNIST`.

---

## Code Organization

All reproducibility scripts are located inside `MainCodes/`.

Each script is designed to be self-contained:

- The Lissajous scan function is included directly in the relevant scripts.
- Dataset construction, model definition, training loop, and evaluation code are kept inside each script.
- No cross-imports between scripts are required.
- Configurations are controlled through editable constants near the top of each file.
- Finished outputs are skipped automatically when possible, so scripts are safe to interrupt and rerun.

---

## Section 1 — Coverage Index

The Coverage Index is used as a low-cost pre-screening metric for Lissajous scan parameters before expensive model training.

The metric is defined as:

```text
CI = C × U
```

where:

- `C` is the spatial coverage of the sampled trajectory.
- `U` is the Shannon-entropy-based uniformity of the scan distribution.

Run:

```bash
cd MainCodes/01_CoverageIndex
python SingleConfig.py
python FullPipeline.py
```

Outputs from `FullPipeline.py`:

```text
results/CI_sweep_N{N}.csv
results/CI_sweep_summary.csv
```

The first file reports ranked configurations for each sample size `N`, while the summary file stores the best Lissajous configuration per `N`.

---

## Section 2 — Reconstruction Control Experiments

This section evaluates the conventional reconstruction-based pipeline and compares it against the proposed reconstruction-free approach.

The control experiments include three conditions:

| Code | Train on | Evaluate on | Purpose |
| :--: | -------- | ----------- | ------- |
| UB | Clean MNIST | Clean MNIST | Measures the upper-bound architecture performance |
| RB | Reconstructed images | Reconstructed images | Represents the realistic 2D reconstruction-based baseline |
| LB | Clean MNIST | Reconstructed images | Measures the distribution shift caused by reconstruction |

Three 2D architectures are supported:

```text
small_cnn (a simple model with two CNN layers)
alexnet
resnet18
```

AlexNet and ResNet-18 use resized 224×224 inputs with channel replication to adapt MNIST images to ImageNet-style architectures.

### Execution Order

1. Build the Lissajous signal dataset.

   Edit `N`, `FX`, `FY`, and `FS` in `DatasetBuilder.py`, then run:

   ```bash
   cd MainCodes/02_ControlExperiments
   python DatasetBuilder.py
   ```

   This writes:

   ```text
   data/N{N}_fx{FX}_fy{FY}_fs{FS}/train.pkl
   data/N{N}_fx{FX}_fy{FY}_fs{FS}/test.pkl
   ```

2. Reconstruct the 2D images.

   Use the same `N`, `FX`, `FY`, and `FS` values:

   ```bash
   python ReconstructionPipeline.py
   ```

   This writes:

   ```text
   data/N{N}_fx{FX}_fy{FY}_fs{FS}/train_recon.npy
   data/N{N}_fx{FX}_fy{FY}_fs{FS}/test_recon.npy
   ```

3. Train and evaluate the control models.

   Edit `MODEL_NAME` and run the desired condition:

   ```bash
   python UpperBound.py
   python RealisticBound.py
   python LowerBound.py
   ```

Typical outputs include:

```text
checkpoints/UB_{model}_seed{seed}.pth
checkpoints/RB_{model}_N{N}_seed{seed}.pth
metrics/LowerBound_{model}_N{N}_seed{seed}.txt
```

### Latency Measurement

To measure the latency of the reconstruction-based pipeline:

```bash
python LatencyMeasurement.py
```

This measures the time required for reconstruction plus classification over a 10,000-image test workload.

---

## Section 3 — Full Hyperparameter Sweeps

The full sweep scripts train and evaluate 1D classifiers directly on Lissajous-sampled signals.

Run:

```bash
cd MainCodes/03_FullSweep
python CNNFullSweep.py
python LSTMFullSweep.py
python KNNFullSweep.py
```

The sweep scripts use the best Lissajous scan configuration for each sample size `N`.

### Sweep Outputs

For every trained configuration, the sweep scripts write a run directory containing:

```text
results/<run_name>/metrics.csv
results/<run_name>/final_metrics.json
```

where:

- `metrics.csv` stores epoch-level training and validation history.
- `final_metrics.json` stores final accuracy, macro precision, macro recall, macro F1, inference latency, training time, and parameter count.

At the end of each sweep, the scripts aggregate all completed runs into:

```text
results/summary.csv
```

The exported spreadsheet versions of the full sweeps are included in the top-level `Results/` directory:

```text
Results/01-CNN/CNNFullSweep.xlsx
Results/02-LSTM/LSTMFullSweep.xlsx
Results/03-KNN/KNNFullSweep.xlsx
```

---

## Selected Top Models

In addition to the full sweep outputs, the submission includes curated results for selected top-performing models.

For the CNN, eight selected models are included:

- Four top models by classification accuracy.
- Four top models by inference speed.

The selected CNN configurations are:

| Group | N | fx | fy | fs | filter_size | num_filters | lr |
| ----- | --: | --: | --: | --: | ----------: | ----------: | --: |
| Top accuracy | 500 | 39 | 41 | 1200 | 91 | 64 | 0.001 |
| Top accuracy | 200 | 37 | 23 | 800 | 91 | 64 | 0.001 |
| Top accuracy | 100 | 17 | 39 | 450 | 81 | 64 | 0.010 |
| Top accuracy | 50 | 43 | 37 | 350 | 31 | 64 | 0.001 |
| Top speed | 500 | 39 | 41 | 1200 | 11 | 32 | 0.010 |
| Top speed | 200 | 37 | 23 | 800 | 11 | 32 | 0.010 |
| Top speed | 100 | 17 | 39 | 450 | 11 | 32 | 0.001 |
| Top speed | 50 | 43 | 37 | 350 | 11 | 32 | 0.010 |

The associated CNN outputs are stored in:

```text
Results/01-CNN/ConfusionMatrices/
Results/01-CNN/Weights/
```

The same organization is used for LSTM and KNN:

```text
Results/02-LSTM/ConfusionMatrices/
Results/02-LSTM/Weights/

Results/03-KNN/ConfusionMatrices/
Results/03-KNN/Weights/
```


---

## Results Directory

The `Results/` directory is intended for direct inspection without rerunning the full experiments.

It contains:

1. Full sweep spreadsheets for CNN, LSTM, and KNN.
2. Confusion matrices for the selected top models.
3. Saved weights or model artifacts for the selected top models.

The naming convention is:

```text
01-CNN/
02-LSTM/
03-KNN/
```

This numbering matches the model-family order used in the paper and keeps the result folders easy to compare.

---

## Inference Latency

The sweep outputs report `inference_10k_s`, defined as the wall-clock time for one full inference pass over a 10,000-image test workload.

For CNN and LSTM:

- Latency is measured using tensors batched at 64.
- Warmup passes are performed before timing.
- The reported value is the median of three full inference passes.
- Timing is performed on the active device, either CPU or CUDA.

For KNN:

- Latency is measured using the actual `KNeighborsClassifier.predict()` call on the test set.
- The measurement reflects the neighbor-search cost directly.

---

## Hardware and Runtime Estimates

The approximate runtimes below are based on an RTX 3070 8 GB GPU with CUDA 12.4.

| Section | Approximate runtime |
| ------- | ------------------- |
| `01_CoverageIndex` full sweep | ~5 min on CPU |
| `02_ControlExperiments` dataset build | ~10 min on CPU |
| `02_ControlExperiments` reconstruction | ~3 min on CPU |
| `02_ControlExperiments` Upper/RealisticBound | ~10 min per run |
| `03_FullSweep` CNN | ~6–10 h on GPU |
| `03_FullSweep` LSTM | ~12–15 h on GPU |
| `03_FullSweep` KNN | ~30 min on CPU |

Runtime can vary depending on hardware, CUDA availability, disk speed, and whether cached datasets already exist.

---

## Reproducibility Notes

- Random seeds are fixed inside the scripts where applicable.
- Cached datasets are reused when available.
- Completed runs are skipped automatically based on existing output files.
- Full sweep outputs are provided as `.xlsx` files in the `Results/` directory.
- Selected top models are provided with their confusion matrices and saved weights/model artifacts.

---
