"""
NeRF model definition (PyTorch).

Implements the NeRF MLP architecture with positional encoding
as described in the original paper:
"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.
    
    Maps input coordinates x to a higher-dimensional space:
        gamma(x) = [x, sin(2^0 * pi * x), cos(2^0 * pi * x), ...,
                       sin(2^{L-1} * pi * x), cos(2^{L-1} * pi * x)]
    
    This helps the network learn high-frequency variations.
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
            # 2^0, 2^1, ..., 2^{L-1}
            freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)
        
        # Register as buffer (not a parameter, but moves with the model)
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim(self) -> int:
        """Output dimension per input coordinate."""
        dim = 2 * self.num_freqs  # sin and cos for each frequency
        if self.include_input:
            dim += 1
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (..., C).
            
        Returns
        -------
        torch.Tensor
            Encoded coordinates of shape (..., C * output_dim).
        """
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    """
    Neural Radiance Field MLP.
    
    Architecture follows the original paper:
    - 8-layer MLP for density and features
    - Skip connection at layer 5 (index 4)
    - View-dependent color head
    
    Parameters
    ----------
    config : ModelConfig
        Configuration for the model architecture.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # Positional encodings
        self.pos_encoder = PositionalEncoding(
            num_freqs=config.pos_freqs,
            include_input=True,
        )
        self.dir_encoder = PositionalEncoding(
            num_freqs=config.dir_freqs,
            include_input=True,
        )

        # Input dimensions after encoding
        pos_dim = 3 * self.pos_encoder.output_dim  # 3D position
        dir_dim = 3 * self.dir_encoder.output_dim  # 3D direction

        # Build the position MLP (trunk)
        self.pts_linears = nn.ModuleList()
        in_dim = pos_dim
        for i in range(config.num_hidden_layers):
            self.pts_linears.append(nn.Linear(in_dim, config.hidden_dim))
            in_dim = config.hidden_dim
            # Prepare for skip connection input at specified layers
            if i in config.skips:
                in_dim += pos_dim

        # Density output (view-independent)
        self.sigma_linear = nn.Linear(config.hidden_dim, 1)
        
        # Feature output for view-dependent color
        self.feature_linear = nn.Linear(config.hidden_dim, config.hidden_dim)

        # View-dependent color head
        if config.use_view_dirs:
            self.dir_linear = nn.Linear(
                config.hidden_dim + dir_dim, 
                config.hidden_dim // 2
            )
        else:
            self.dir_linear = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        
        self.rgb_linear = nn.Linear(config.hidden_dim // 2, 3)

    def forward(
        self, 
        x: torch.Tensor, 
        d: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the NeRF MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            3D positions, shape (N, 3).
        d : torch.Tensor, optional
            View directions (unit vectors), shape (N, 3).
            If None and use_view_dirs is True, uses zeros.
            
        Returns
        -------
        rgb : torch.Tensor
            RGB color values in [0, 1], shape (N, 3).
        sigma : torch.Tensor
            Volume density (non-negative), shape (N, 1).
        """
        # Encode positions
        x_enc = self.pos_encoder(x)
        h = x_enc

        # Pass through trunk MLP with skip connections
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            # Skip connection: concatenate encoded input
            if i in self.config.skips:
                h = torch.cat([x_enc, h], dim=-1)

        # Density output (using ReLU to ensure non-negative)
        sigma = F.relu(self.sigma_linear(h))
        
        # Feature for color prediction
        feats = self.feature_linear(h)

        # Color prediction (view-dependent)
        if self.config.use_view_dirs and d is not None:
            d_enc = self.dir_encoder(d)
            h_color = torch.cat([feats, d_enc], dim=-1)
        else:
            h_color = feats
            
        h_color = F.relu(self.dir_linear(h_color))
        rgb = torch.sigmoid(self.rgb_linear(h_color))

        return rgb, sigma


def create_nerf(config: ModelConfig | None = None) -> Tuple[NeRF, NeRF | None]:
    """
    Create NeRF models (coarse and optionally fine).
    
    Parameters
    ----------
    config : ModelConfig, optional
        Model configuration.
        
    Returns
    -------
    model_coarse : NeRF
        Coarse network.
    model_fine : NeRF or None
        Fine network (same architecture as coarse).
    """
    if config is None:
        config = ModelConfig()
    
    model_coarse = NeRF(config)
    model_fine = NeRF(config)  # Same architecture, different weights
    
    return model_coarse, model_fine





