"""
Metrics computation for NeRF evaluation.

Includes PSNR, SSIM, and LPIPS (if available).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional
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
        Predicted image.
    target : torch.Tensor
        Ground truth image.
    max_val : float
        Maximum possible value.
        
    Returns
    -------
    torch.Tensor
        PSNR in dB.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error."""
    return torch.mean((pred - target) ** 2)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted image, shape (H, W, 3) or (H, W).
    target : torch.Tensor
        Ground truth image.
    window_size : int
        Size of the Gaussian window.
    C1, C2 : float
        Stability constants.
        
    Returns
    -------
    torch.Tensor
        SSIM value (higher is better, max 1.0).
    """
    # Ensure we're working with float
    pred = pred.float()
    target = target.float()
    
    # Handle different input shapes
    if pred.dim() == 3:  # (H, W, C)
        pred = pred.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 2:  # (H, W)
        pred = pred.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        target = target.unsqueeze(0).unsqueeze(0)
    
    # Create Gaussian window
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g)
    
    window = gaussian_window(window_size).to(pred.device)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    C = pred.shape[1]  # Number of channels
    window = window.expand(C, 1, window_size, window_size)
    
    # Compute means
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=C)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    # Compute variances
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu_pred_target
    
    # Compute SSIM
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return ssim_map.mean()


class LPIPSMetric:
    """
    LPIPS perceptual metric wrapper.
    
    Requires lpips package: pip install lpips
    """
    
    def __init__(self, net: str = "vgg", device: str = "cuda"):
        self.device = device
        self.lpips_fn = None
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device)
                self.lpips_fn.eval()
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute LPIPS distance.
        
        Parameters
        ----------
        pred, target : torch.Tensor
            Images of shape (H, W, 3) in [0, 1].
            
        Returns
        -------
        torch.Tensor or None
            LPIPS distance (lower is better), or None if lpips not available.
        """
        if not self.available:
            return None
        
        # Convert to LPIPS format: (1, 3, H, W) in [-1, 1]
        pred = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        target = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        with torch.no_grad():
            return self.lpips_fn(pred, target).squeeze()


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_metric: Optional[LPIPSMetric] = None,
) -> Dict[str, float]:
    """
    Compute all image quality metrics.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted image, shape (H, W, 3).
    target : torch.Tensor
        Ground truth image, shape (H, W, 3).
    lpips_metric : LPIPSMetric, optional
        LPIPS metric instance.
        
    Returns
    -------
    dict
        Dictionary of metric values.
    """
    metrics = {
        "mse": compute_mse(pred, target).item(),
        "psnr": compute_psnr(pred, target).item(),
        "ssim": compute_ssim(pred, target).item(),
    }
    
    if lpips_metric is not None:
        lpips_val = lpips_metric(pred, target)
        if lpips_val is not None:
            metrics["lpips"] = lpips_val.item()
    
    return metrics







