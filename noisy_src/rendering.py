"""
Volume rendering for NeRF.

Implements the volume rendering equation to composite colors
and densities along rays into final pixel colors.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from .model import NeRF
from .rays import sample_along_rays, sample_hierarchical
from .config import RenderConfig


def raw2outputs(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_background: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert raw network outputs to rendered colors via volume rendering.
    
    Implements the volume rendering equation:
        C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
        
    where T_i = exp(-sum_{j<i} sigma_j * delta_j) is the transmittance.
    
    Parameters
    ----------
    rgb : torch.Tensor
        RGB color values from network, shape (..., N_samples, 3).
    sigma : torch.Tensor
        Density values from network, shape (..., N_samples, 1).
    z_vals : torch.Tensor
        Depth values along rays, shape (..., N_samples).
    rays_d : torch.Tensor
        Ray directions, shape (..., 3).
    raw_noise_std : float
        Noise to add to density during training.
    white_background : bool
        If True, composite on white background.
        
    Returns
    -------
    dict containing:
        rgb_map : torch.Tensor
            Rendered colors, shape (..., 3).
        depth_map : torch.Tensor
            Rendered depth, shape (...,).
        acc_map : torch.Tensor
            Accumulated opacity, shape (...,).
        weights : torch.Tensor
            Weights for each sample, shape (..., N_samples).
    """
    # Squeeze sigma if needed
    sigma = sigma.squeeze(-1)  # (..., N_samples)
    
    # Compute distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # Add infinite distance at the end
    dists = torch.cat([
        dists,
        torch.full_like(dists[..., :1], 1e10)
    ], dim=-1)
    
    # Scale by ray direction magnitude (for non-unit directions)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # Add noise to density during training
    if raw_noise_std > 0.0:
        noise = torch.randn_like(sigma) * raw_noise_std
        sigma = sigma + noise
    
    # Compute alpha (opacity) for each sample
    alpha = 1.0 - torch.exp(-torch.relu(sigma) * dists)
    
    # Compute transmittance (accumulated transparency)
    # T_i = prod_{j<i} (1 - alpha_j)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha + 1e-10
        ], dim=-1),
        dim=-1
    )[..., :-1]
    
    # Compute weights
    weights = alpha * transmittance
    
    # Render RGB
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    
    # Render depth as expected distance
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1)
    
    # White background compositing
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    
    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "weights": weights,
    }


def render_rays(
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    config: RenderConfig,
    is_train: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Render a batch of rays using NeRF models.
    
    Parameters
    ----------
    model_coarse : NeRF
        Coarse network.
    model_fine : NeRF, optional
        Fine network for hierarchical sampling.
    rays_o : torch.Tensor
        Ray origins, shape (N_rays, 3).
    rays_d : torch.Tensor
        Ray directions, shape (N_rays, 3).
    config : RenderConfig
        Rendering configuration.
    is_train : bool
        Whether in training mode (affects perturbation and noise).
        
    Returns
    -------
    dict containing:
        rgb_coarse : torch.Tensor
            Coarse RGB output.
        rgb_fine : torch.Tensor
            Fine RGB output (if hierarchical).
        depth_coarse : torch.Tensor
            Coarse depth.
        depth_fine : torch.Tensor
            Fine depth (if hierarchical).
        acc_coarse : torch.Tensor
            Coarse opacity.
        acc_fine : torch.Tensor
            Fine opacity (if hierarchical).
    """
    perturb = config.perturb if is_train else False
    raw_noise_std = config.raw_noise_std if is_train else 0.0
    
    # Normalize directions for view-dependent effects
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # === Coarse sampling ===
    pts_coarse, z_vals_coarse = sample_along_rays(
        rays_o=rays_o,
        rays_d=rays_d,
        near=config.near,
        far=config.far,
        num_samples=config.num_samples,
        perturb=perturb,
    )
    
    # Flatten for network input
    N_rays = rays_o.shape[0]
    N_coarse = config.num_samples
    
    pts_flat = pts_coarse.reshape(-1, 3)
    viewdirs_flat = viewdirs[:, None, :].expand(-1, N_coarse, -1).reshape(-1, 3)
    
    # Run coarse network
    rgb_coarse, sigma_coarse = model_coarse(pts_flat, viewdirs_flat)
    rgb_coarse = rgb_coarse.reshape(N_rays, N_coarse, 3)
    sigma_coarse = sigma_coarse.reshape(N_rays, N_coarse, 1)
    
    # Volume rendering for coarse
    outputs_coarse = raw2outputs(
        rgb=rgb_coarse,
        sigma=sigma_coarse,
        z_vals=z_vals_coarse,
        rays_d=rays_d,
        raw_noise_std=raw_noise_std,
        white_background=config.white_background,
    )
    
    results = {
        "rgb_coarse": outputs_coarse["rgb_map"],
        "depth_coarse": outputs_coarse["depth_map"],
        "acc_coarse": outputs_coarse["acc_map"],
    }
    
    # === Hierarchical (fine) sampling ===
    if config.use_hierarchical and model_fine is not None:
        pts_fine, z_vals_fine = sample_hierarchical(
            rays_o=rays_o,
            rays_d=rays_d,
            z_vals=z_vals_coarse,
            weights=outputs_coarse["weights"],
            num_samples_fine=config.num_samples_fine,
            det=not is_train,
        )
        
        N_fine = z_vals_fine.shape[-1]
        
        pts_flat = pts_fine.reshape(-1, 3)
        viewdirs_flat = viewdirs[:, None, :].expand(-1, N_fine, -1).reshape(-1, 3)
        
        # Run fine network
        rgb_fine, sigma_fine = model_fine(pts_flat, viewdirs_flat)
        rgb_fine = rgb_fine.reshape(N_rays, N_fine, 3)
        sigma_fine = sigma_fine.reshape(N_rays, N_fine, 1)
        
        # Volume rendering for fine
        outputs_fine = raw2outputs(
            rgb=rgb_fine,
            sigma=sigma_fine,
            z_vals=z_vals_fine,
            rays_d=rays_d,
            raw_noise_std=raw_noise_std,
            white_background=config.white_background,
        )
        
        results["rgb_fine"] = outputs_fine["rgb_map"]
        results["depth_fine"] = outputs_fine["depth_map"]
        results["acc_fine"] = outputs_fine["acc_map"]
    
    return results


class NeRFRenderer(nn.Module):
    """
    Wrapper module for rendering with NeRF models.
    
    Handles both coarse and fine networks with proper batching.
    """
    
    def __init__(
        self,
        model_coarse: NeRF,
        model_fine: Optional[NeRF],
        config: RenderConfig,
    ):
        super().__init__()
        self.model_coarse = model_coarse
        self.model_fine = model_fine
        self.config = config
    
    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        chunk_size: int = 1024 * 32,
        is_train: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Render rays in chunks to avoid OOM.
        
        Parameters
        ----------
        rays_o : torch.Tensor
            Ray origins, shape (N_rays, 3).
        rays_d : torch.Tensor
            Ray directions, shape (N_rays, 3).
        chunk_size : int
            Maximum rays per forward pass.
        is_train : bool
            Training mode flag.
            
        Returns
        -------
        dict
            Rendered outputs.
        """
        N_rays = rays_o.shape[0]
        
        if N_rays <= chunk_size:
            return render_rays(
                model_coarse=self.model_coarse,
                model_fine=self.model_fine,
                rays_o=rays_o,
                rays_d=rays_d,
                config=self.config,
                is_train=is_train,
            )
        
        # Process in chunks
        all_results = {}
        for i in range(0, N_rays, chunk_size):
            chunk_o = rays_o[i:i + chunk_size]
            chunk_d = rays_d[i:i + chunk_size]
            
            chunk_results = render_rays(
                model_coarse=self.model_coarse,
                model_fine=self.model_fine,
                rays_o=chunk_o,
                rays_d=chunk_d,
                config=self.config,
                is_train=is_train,
            )
            
            for key, value in chunk_results.items():
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append(value)
        
        # Concatenate chunks
        for key in all_results:
            all_results[key] = torch.cat(all_results[key], dim=0)
        
        return all_results





