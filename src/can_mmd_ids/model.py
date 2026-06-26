from __future__ import annotations

import torch
from torch import nn


class TcnAutoencoder(nn.Module):
    def __init__(self, feature_dim: int = 15, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(64, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x.transpose(1, 2))
        h = self.pool(h).squeeze(-1)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor, window_size: int) -> torch.Tensor:
        h = self.from_latent(z).unsqueeze(-1).repeat(1, 1, window_size)
        return self.decoder(h).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z, x.shape[1]), z


def reconstruction_error(model: TcnAutoencoder, x: torch.Tensor) -> torch.Tensor:
    x_hat, _ = model(x)
    return torch.mean((x - x_hat) ** 2, dim=(1, 2))

