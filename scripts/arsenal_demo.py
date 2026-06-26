from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from can_mmd_ids.features import dataframe_to_features, make_windows
from can_mmd_ids.training import adapt_encoder, device, score_windows, train_autoencoder


def make_synthetic_can(seed: int, domain_shift: float, attack: bool, n: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.array([0x120, 0x188, 0x245, 0x310, 0x411])
    probs = np.array([0.28, 0.22, 0.18, 0.17, 0.15])
    timestamps = np.cumsum(rng.normal(0.004 + domain_shift, 0.00035, n).clip(0.001, None))
    can_ids = rng.choice(ids, p=probs, size=n)
    payload = rng.normal(90 + domain_shift * 5000, 18, size=(n, 8)).clip(0, 255).astype(int)
    labels = np.zeros(n, dtype=int)

    if attack:
        start = int(n * 0.55)
        end = int(n * 0.78)
        labels[start:end] = 1
        can_ids[start:end] = rng.choice([0x001, 0x7DF, 0x700], size=end - start)
        payload[start:end] = rng.integers(0, 255, size=(end - start, 8))
        timestamps[start:end] = timestamps[start:end] - np.linspace(0, 0.45, end - start)
        timestamps = np.maximum.accumulate(timestamps)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "can_id": can_ids,
            "data_hex": ["".join(f"{b:02X}" for b in row) for row in payload],
            "attack": labels,
        }
    )


def windows_from_df(df: pd.DataFrame):
    labels = df["attack"].to_numpy(dtype=np.int64)
    return make_windows(dataframe_to_features(df), labels, stride=5)


def summarize(name: str, benign_scores: np.ndarray, attack_scores: np.ndarray, attack_labels: np.ndarray, threshold: float) -> tuple[float, float]:
    fpr = float(np.mean(benign_scores > threshold))
    auc = float(roc_auc_score(attack_labels, attack_scores))
    print(f"{name}: target benign FPR={fpr:.3f}, target attack AUC={auc:.3f}")
    return fpr, auc


def main() -> None:
    parser = argparse.ArgumentParser(description="CAN IDS")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "demo_before_after.png"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--adapt-epochs", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dev = device(args.device)
    source_benign = windows_from_df(make_synthetic_can(1, domain_shift=0.0, attack=False))
    target_benign = windows_from_df(make_synthetic_can(2, domain_shift=0.0018, attack=False))
    target_attack = windows_from_df(make_synthetic_can(3, domain_shift=0.0018, attack=True))

    model = train_autoencoder(source_benign.windows, epochs=args.epochs, lr=1e-3, batch_size=128, dev=dev)
    source_scores = score_windows(model, source_benign.windows, 256, dev)
    threshold = float(np.percentile(source_scores, 95))
    print(f"Source benign threshold: {threshold:.6f}")

    before_benign = score_windows(model, target_benign.windows, 256, dev)
    before_attack = score_windows(model, target_attack.windows, 256, dev)
    before = summarize("Before adaptation", before_benign, before_attack, target_attack.labels, threshold)

    model = adapt_encoder(
        model,
        source_benign.windows,
        target_benign.windows,
        epochs=args.adapt_epochs,
        lr=1e-4,
        batch_size=128,
        mmd_lambda=0.05,
        dev=dev,
    )
    after_benign = score_windows(model, target_benign.windows, 256, dev)
    after_attack = score_windows(model, target_attack.windows, 256, dev)
    target_threshold = float(np.percentile(after_benign, 95))
    after = summarize("After adaptation + target P95 calibration", after_benign, after_attack, target_attack.labels, target_threshold)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, title, benign, attack, line in [
        (axes[0], "Before MMD Adaptation", before_benign, before_attack, threshold),
        (axes[1], "After Adaptation + Target Calibration", after_benign, after_attack, target_threshold),
    ]:
        ax.hist(benign, bins=40, alpha=0.75, label="target benign")
        ax.hist(attack[target_attack.labels == 1], bins=40, alpha=0.75, label="target attack")
        ax.axvline(line, color="black", linestyle="--", label="active threshold")
        ax.set_title(title)
        ax.set_xlabel("reconstruction error")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("window count")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Cross-vehicle CAN IDS demo: FPR {before[0]:.2f}->{after[0]:.2f}, AUC {before[1]:.2f}->{after[1]:.2f}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
