# MMD-CAN-IDS

Lightweight cross-vehicle intrusion detection for automotive CAN traffic.

MMD-CAN-IDS is a research tool for detecting anomalous CAN bus windows with a Temporal Convolutional Network Autoencoder (TCN-AE), then adapting the encoder to a new vehicle domain using Maximum Mean Discrepancy (MMD) alignment. The goal is to make cross-vehicle CAN IDS behavior visible and repeatable: train on one vehicle profile, score another vehicle, observe the false-positive shift, then adapt with unlabeled benign target traffic.

## Description

### What It Does

- Parses CAN-style CSV or SocketCAN logs into normalized temporal windows.
- Trains a compact TCN autoencoder on benign source-domain CAN traffic.
- Scores windows using reconstruction error and threshold-based anomaly flags.
- Fine-tunes the encoder with reconstruction loss plus MMD latent alignment.
- Demonstrates cross-vehicle behavior with an offline synthetic demo:
  - source vehicle: normal traffic baseline
  - target vehicle before adaptation: shifted benign traffic causes false positives
  - target vehicle after adaptation: cleaner benign/attack separation

### Key Features

- TCN-AE anomaly detector for temporal CAN behavior.
- MMD domain adaptation for unlabeled target-vehicle benign traffic.
- Frozen-decoder adaptation mode to reduce destructive retuning.
- CLI commands for demo, training, scoring, and adaptation.
- Dataset-safe layout: no bundled CAN-v1.5, ROAD, or MIRGU raw files.

### License

Apache License 2.0. See [LICENSE](LICENSE).



## Installation

### Requirements

- Python 3.10, 3.11, or 3.12 recommended.
- Windows 10/11, Linux, or macOS.
- CPU works for the demo and small experiments.
- NVIDIA GPU with CUDA is optional and useful for larger training runs.

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you need a CUDA build of PyTorch, install the correct wheel from the official PyTorch instructions before running large experiments.

## Usage

### 1. Run The Offline Demo

This is the recommended demo flow. It generates synthetic CAN-like logs locally, trains a small source model, scores a shifted target vehicle before adaptation, adapts on unlabeled target benign windows, and writes a before/after plot.

```bash
python scripts/arsenal_demo.py --output artifacts/demo_before_after.png
```

Expected terminal output:

```text
Source benign threshold: ...
Before adaptation: target benign FPR=...
Before adaptation: target attack AUC=...
After adaptation: target benign FPR=...
After adaptation: target attack AUC=...
Plot saved to artifacts/demo_before_after.png
```

### 2. Train A Base Model

Use your own legally obtained benign CAN log. CSV input should contain at least:

- `timestamp`
- `arbitration_id` or `can_id`
- `data_field` or `data_hex`

```bash
python -m can_mmd_ids.cli train \
  --input data/source_benign.csv \
  --model artifacts/source_tcn.pt \
  --epochs 20
```

### 3. Score A Log

```bash
python -m can_mmd_ids.cli score \
  --input data/target_log.csv \
  --model artifacts/source_tcn.pt \
  --threshold 0.02 \
  --output artifacts/target_scores.csv
```

Sample output CSV:

```csv
window_index,reconstruction_error,is_anomaly
0,0.0104,0
1,0.0112,0
2,0.0871,1
```

### 4. Adapt To A Target Vehicle

Adaptation uses source benign data and unlabeled target benign data. No target attack labels are required.

```bash
python -m can_mmd_ids.cli adapt \
  --source data/source_benign.csv \
  --target data/target_benign.csv \
  --base-model artifacts/source_tcn.pt \
  --adapted-model artifacts/target_adapted_tcn.pt \
  --epochs 10 \
  --mmd-lambda 0.01
```

Then score target traffic again:

```bash
python -m can_mmd_ids.cli score \
  --input data/target_log.csv \
  --model artifacts/target_adapted_tcn.pt \
  --threshold 0.02 \
  --output artifacts/target_scores_after.csv
```

## Dataset Policy

Raw CAN-v1.5, ROAD, and MIRGU files are not included. Download them from their official sources and follow their citation/license terms. See [docs/DATASETS.md](docs/DATASETS.md).

Recommended local layout after downloading datasets:

```text
data/
  can-v1.5/   # ignored by git
  road/       # ignored by git
  mirgu/      # ignored by git
```

