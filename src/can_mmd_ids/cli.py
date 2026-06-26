from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import load_windows
from .training import adapt_encoder, device, load_model, save_model, score_windows, train_autoencoder


def cmd_train(args: argparse.Namespace) -> None:
    dev = device(args.device)
    bundle = load_windows(args.input, stride=args.stride)
    model = train_autoencoder(bundle.windows, args.epochs, args.lr, args.batch_size, dev)
    save_model(model, args.model)
    scores = score_windows(model, bundle.windows, args.batch_size, dev)
    threshold = float(np.percentile(scores, args.threshold_percentile))
    print(f"saved_model={args.model}")
    print(f"suggested_threshold_p{args.threshold_percentile:g}={threshold:.6f}")


def cmd_score(args: argparse.Namespace) -> None:
    dev = device(args.device)
    bundle = load_windows(args.input, stride=args.stride)
    model = load_model(args.model, dev)
    scores = score_windows(model, bundle.windows, args.batch_size, dev)
    threshold = args.threshold if args.threshold is not None else float(np.percentile(scores, 95))
    out = pd.DataFrame(
        {
            "window_index": np.arange(len(scores)),
            "reconstruction_error": scores,
            "is_anomaly": (scores > threshold).astype(int),
        }
    )
    if bundle.labels is not None:
        out["label"] = bundle.labels
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"threshold={threshold:.6f}")
    print(f"anomaly_rate={out['is_anomaly'].mean():.4f}")
    print(f"wrote={args.output}")


def cmd_adapt(args: argparse.Namespace) -> None:
    dev = device(args.device)
    source = load_windows(args.source, stride=args.stride)
    target = load_windows(args.target, stride=args.stride)
    model = load_model(args.base_model, dev)
    model = adapt_encoder(
        model,
        source.windows,
        target.windows,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mmd_lambda=args.mmd_lambda,
        dev=dev,
    )
    save_model(model, args.adapted_model)
    print(f"saved_adapted_model={args.adapted_model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMD-CAN-IDS CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a TCN autoencoder on benign CAN windows")
    train.add_argument("--input", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--stride", type=int, default=5)
    train.add_argument("--threshold-percentile", type=float, default=95)
    train.add_argument("--device", default=None)
    train.set_defaults(func=cmd_train)

    score = sub.add_parser("score", help="score CAN windows with a trained model")
    score.add_argument("--input", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--threshold", type=float, default=None)
    score.add_argument("--output", required=True)
    score.add_argument("--batch-size", type=int, default=256)
    score.add_argument("--stride", type=int, default=5)
    score.add_argument("--device", default=None)
    score.set_defaults(func=cmd_score)

    adapt = sub.add_parser("adapt", help="MMD-adapt source model to target benign traffic")
    adapt.add_argument("--source", required=True)
    adapt.add_argument("--target", required=True)
    adapt.add_argument("--base-model", required=True)
    adapt.add_argument("--adapted-model", required=True)
    adapt.add_argument("--epochs", type=int, default=10)
    adapt.add_argument("--lr", type=float, default=1e-4)
    adapt.add_argument("--batch-size", type=int, default=128)
    adapt.add_argument("--mmd-lambda", type=float, default=0.01)
    adapt.add_argument("--stride", type=int, default=5)
    adapt.add_argument("--device", default=None)
    adapt.set_defaults(func=cmd_adapt)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

