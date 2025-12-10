"""
Utility functions for NeRF training and evaluation.
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.
    max_val : float
        Maximum possible value (1.0 for normalized images).
        
    Returns
    -------
    torch.Tensor
        PSNR value in dB.
    """
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (simplified version).
    
    For full SSIM, consider using torchmetrics or scikit-image.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted image, shape (H, W, 3) or (B, H, W, 3).
    target : torch.Tensor
        Ground truth image.
    window_size : int
        Window size for local statistics.
        
    Returns
    -------
    torch.Tensor
        SSIM value (higher is better, max 1.0).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Simple mean-based SSIM approximation
    mu_pred = pred.mean()
    mu_target = target.mean()
    
    sigma_pred = ((pred - mu_pred) ** 2).mean()
    sigma_target = ((target - mu_target) ** 2).mean()
    sigma_both = ((pred - mu_pred) * (target - mu_target)).mean()
    
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_both + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    
    return ssim


def depth_to_colormap(
    depth: torch.Tensor,
    near: Optional[float] = None,
    far: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert depth map to a colormap visualization.
    
    Parameters
    ----------
    depth : torch.Tensor
        Depth values, shape (H, W).
    near : float, optional
        Near plane for normalization.
    far : float, optional
        Far plane for normalization.
        
    Returns
    -------
    torch.Tensor
        RGB colormap, shape (H, W, 3).
    """
    if near is None:
        near = depth.min().item()
    if far is None:
        far = depth.max().item()
    
    # Normalize to [0, 1]
    depth_norm = (depth - near) / (far - near + 1e-8)
    depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
    
    # Simple viridis-like colormap
    # [0, 0.25]: dark purple to blue
    # [0.25, 0.5]: blue to green
    # [0.5, 0.75]: green to yellow
    # [0.75, 1.0]: yellow to bright
    
    r = torch.clamp(1.5 - torch.abs(depth_norm - 0.75) * 4, 0, 1)
    g = torch.clamp(1.5 - torch.abs(depth_norm - 0.5) * 4, 0, 1)
    b = torch.clamp(1.5 - torch.abs(depth_norm - 0.25) * 4, 0, 1)
    
    return torch.stack([r, g, b], dim=-1)


def create_spiral_poses(
    center: torch.Tensor,
    radius: float,
    height_range: tuple,
    num_frames: int,
    num_rotations: float = 2.0,
    focal_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create camera poses along a spiral path for novel view synthesis.
    
    Parameters
    ----------
    center : torch.Tensor
        Center point of the spiral, shape (3,).
    radius : float
        Radius of the spiral.
    height_range : tuple
        (min_height, max_height) range.
    num_frames : int
        Number of frames to generate.
    num_rotations : float
        Number of full rotations.
    focal_point : torch.Tensor, optional
        Point to look at. Defaults to center.
        
    Returns
    -------
    torch.Tensor
        Camera-to-world matrices, shape (num_frames, 4, 4).
    """
    if focal_point is None:
        focal_point = center.clone()
    
    poses = []
    
    for i in range(num_frames):
        t = i / num_frames
        
        # Spiral position
        theta = 2.0 * np.pi * num_rotations * t
        height = height_range[0] + (height_range[1] - height_range[0]) * t
        
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = height
        
        position = torch.tensor([x, y, z], dtype=center.dtype)
        
        # Look-at matrix
        forward = focal_point - position
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0.0, 0.0, 1.0], dtype=center.dtype)
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        
        up = torch.cross(right, forward)
        
        # Construct c2w matrix
        c2w = torch.eye(4, dtype=center.dtype)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = position
        
        poses.append(c2w)
    
    return torch.stack(poses, dim=0)


class AverageMeter:
    """Track running averages for metrics."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop









