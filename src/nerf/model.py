"""
NeRF model definition (PyTorch).

This module implements a minimal but modular NeRF architecture similar to
the original paper, with:
- Positional encoding for 3D positions and view directions.
- A small MLP that predicts RGB and density (sigma).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard NeRF-style positional encoding.

    For each scalar input x, produces:
        [x, sin(2^0 x), cos(2^0 x), ..., sin(2^{L-1} x), cos(2^{L-1} x)]
    if include_input=True, or just the sin/cos terms otherwise.
    """

    def __init__(
        self,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            (..., C) tensor of input coordinates.
        """
        out = []
        if self.include_input:
            out.append(x)

        # Ensure freq_bands on the correct device
        freq_bands = self.freq_bands.to(x.device)

        for freq in freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)


@dataclass
class NeRFConfig:
    """Configuration for the NeRF model."""

    pos_freqs: int = 10
    dir_freqs: int = 4
    hidden_dim: int = 256
    num_hidden_layers: int = 8
    skips: Tuple[int, ...] = (4,)


class NeRF(nn.Module):
    """
    Minimal NeRF MLP.

    Follows the standard design:
    - Position encoding → trunk MLP for density + intermediate features.
    - View direction encoding + features → RGB head.
    """

    def __init__(self, config: NeRFConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = NeRFConfig()
        self.config = config

        self.pos_encoder = PositionalEncoding(num_freqs=config.pos_freqs, include_input=True)
        self.dir_encoder = PositionalEncoding(num_freqs=config.dir_freqs, include_input=True)

        pos_dim = (1 + 2 * config.pos_freqs) * 3
        dir_dim = (1 + 2 * config.dir_freqs) * 3

        # Trunk MLP for position → features + sigma
        layers = []
        in_dim = pos_dim
        for i in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            in_dim = config.hidden_dim
            if i in config.skips:
                in_dim += pos_dim  # for skip connection

        self.pts_linears = nn.ModuleList(layers)
        self.sigma_linear = nn.Linear(config.hidden_dim, 1)
        self.feature_linear = nn.Linear(config.hidden_dim, config.hidden_dim)

        # View-dependent color head
        self.dir_linear = nn.Linear(config.hidden_dim + dir_dim, config.hidden_dim // 2)
        self.rgb_linear = nn.Linear(config.hidden_dim // 2, 3)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Sampled 3D points, shape [N, 3].
        d:
            Corresponding view directions (unit vectors), shape [N, 3].

        Returns
        -------
        rgb:
            [N, 3] in [0, 1].
        sigma:
            [N, 1] density.
        """
        # Positional encoding
        x_enc = self.pos_encoder(x)
        h = x_enc
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h, inplace=True)
            if i in self.config.skips:
                h = torch.cat([x_enc, h], dim=-1)

        sigma = self.sigma_linear(h)
        feats = self.feature_linear(h)

        # Direction encoding for color
        d_enc = self.dir_encoder(d)
        h_color = torch.cat([feats, d_enc], dim=-1)
        h_color = F.relu(self.dir_linear(h_color), inplace=True)
        rgb = torch.sigmoid(self.rgb_linear(h_color))

        return rgb, sigma

