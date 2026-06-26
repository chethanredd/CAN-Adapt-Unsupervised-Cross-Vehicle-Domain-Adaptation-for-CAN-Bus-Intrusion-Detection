from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .mmd import rbf_mmd
from .model import TcnAutoencoder, reconstruction_error


def device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(windows: np.ndarray, epochs: int, lr: float, batch_size: int, dev: torch.device) -> TcnAutoencoder:
    model = TcnAutoencoder(feature_dim=windows.shape[-1]).to(dev)
    loader = DataLoader(TensorDataset(torch.tensor(windows)), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(dev)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = model(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            opt.step()
    return model


def adapt_encoder(
    model: TcnAutoencoder,
    source_windows: np.ndarray,
    target_windows: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    mmd_lambda: float,
    dev: torch.device,
) -> TcnAutoencoder:
    model = model.to(dev)
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.from_latent.parameters():
        p.requires_grad = False

    src_loader = DataLoader(TensorDataset(torch.tensor(source_windows)), batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(TensorDataset(torch.tensor(target_windows)), batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (xs,), (xt,) in zip(src_loader, tgt_loader):
            xs, xt = xs.to(dev), xt.to(dev)
            opt.zero_grad(set_to_none=True)
            xt_hat, zt = model(xt)
            zs = model.encode(xs)
            loss = loss_fn(xt_hat, xt) + mmd_lambda * rbf_mmd(zs, zt)
            loss.backward()
            opt.step()

    for p in model.parameters():
        p.requires_grad = True
    return model


@torch.no_grad()
def score_windows(model: TcnAutoencoder, windows: np.ndarray, batch_size: int, dev: torch.device) -> np.ndarray:
    model = model.to(dev).eval()
    loader = DataLoader(TensorDataset(torch.tensor(windows)), batch_size=batch_size)
    scores: list[np.ndarray] = []
    for (xb,) in loader:
        err = reconstruction_error(model, xb.to(dev)).detach().cpu().numpy()
        scores.append(err)
    return np.concatenate(scores)


def save_model(model: TcnAutoencoder, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def load_model(path: str | Path, dev: torch.device) -> TcnAutoencoder:
    model = TcnAutoencoder().to(dev)
    ckpt = torch.load(path, map_location=dev)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    return model


def auc_or_nan(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))

