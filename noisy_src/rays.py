"""
Ray generation and sampling utilities for NeRF.

This module handles:
- Generating rays from camera parameters
- Stratified sampling along rays
- Hierarchical sampling for fine network
"""

from __future__ import annotations

from typing import Tuple

import torch


def get_ray_directions(
    H: int,
    W: int,
    focal: float,
    center: Tuple[float, float] | None = None,
) -> torch.Tensor:
    """
    Generate ray directions for all pixels in camera coordinate frame.
    
    The camera looks down the -Z axis (OpenGL convention).
    
    Parameters
    ----------
    H : int
        Image height.
    W : int
        Image width.
    focal : float
        Focal length in pixels.
    center : tuple of float, optional
        Principal point (cx, cy). Defaults to image center.
        
    Returns
    -------
    directions : torch.Tensor
        Ray directions of shape (H, W, 3).
    """
    if center is None:
        cx, cy = W / 2.0, H / 2.0
    else:
        cx, cy = center

    # Create pixel grid
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    
    # Compute direction for each pixel
    # Camera looks down -Z, X is right, Y is down
    directions = torch.stack([
        (i - cx) / focal,
        -(j - cy) / focal,  # Negative because Y points down in image coords
        -torch.ones_like(i),  # Looking down -Z
    ], dim=-1)

    return directions


def get_rays(
    directions: torch.Tensor,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform ray directions from camera to world coordinates.
    
    Parameters
    ----------
    directions : torch.Tensor
        Ray directions in camera frame, shape (..., 3).
    c2w : torch.Tensor
        Camera-to-world transformation matrix, shape (4, 4).
        
    Returns
    -------
    rays_o : torch.Tensor
        Ray origins in world frame, shape (..., 3).
    rays_d : torch.Tensor
        Ray directions in world frame (normalized), shape (..., 3).
    """
    # Rotate directions from camera to world (use rotation part of c2w)
    rays_d = torch.sum(
        directions[..., None, :] * c2w[:3, :3], 
        dim=-1
    )
    # Normalize directions
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Origin is the camera position (translation part of c2w)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    
    return rays_o, rays_d


def get_rays_batch(
    H: int,
    W: int,
    focal: float,
    c2w_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for a batch of camera poses.
    
    Parameters
    ----------
    H, W : int
        Image dimensions.
    focal : float
        Focal length in pixels.
    c2w_batch : torch.Tensor
        Batch of camera-to-world matrices, shape (N, 4, 4).
        
    Returns
    -------
    rays_o : torch.Tensor
        Ray origins, shape (N, H, W, 3).
    rays_d : torch.Tensor
        Ray directions, shape (N, H, W, 3).
    """
    device = c2w_batch.device
    directions = get_ray_directions(H, W, focal).to(device)
    
    N = c2w_batch.shape[0]
    rays_o_list = []
    rays_d_list = []
    
    for i in range(N):
        rays_o, rays_d = get_rays(directions, c2w_batch[i])
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)
    
    rays_o = torch.stack(rays_o_list, dim=0)
    rays_d = torch.stack(rays_d_list, dim=0)
    
    return rays_o, rays_d


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
    lindisp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays using stratified sampling.
    
    Parameters
    ----------
    rays_o : torch.Tensor
        Ray origins, shape (..., 3).
    rays_d : torch.Tensor
        Ray directions, shape (..., 3).
    near : float
        Near plane distance.
    far : float
        Far plane distance.
    num_samples : int
        Number of samples per ray.
    perturb : bool
        Whether to add random jitter to sample positions.
    lindisp : bool
        If True, sample linearly in disparity (1/depth) rather than depth.
        
    Returns
    -------
    pts : torch.Tensor
        Sample points, shape (..., num_samples, 3).
    z_vals : torch.Tensor
        Depth values, shape (..., num_samples).
    """
    device = rays_o.device
    batch_shape = rays_o.shape[:-1]
    
    # Create evenly spaced samples
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    
    if lindisp:
        # Sample linearly in disparity
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        # Sample linearly in depth
        z_vals = near * (1.0 - t_vals) + far * t_vals
    
    # Expand to batch dimensions
    z_vals = z_vals.expand(*batch_shape, num_samples)
    
    if perturb:
        # Get bin widths for stratified sampling
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        
        # Random samples within each bin
        t_rand = torch.rand(*batch_shape, num_samples, device=device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Compute 3D sample points: r(t) = o + t*d
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    return pts, z_vals


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    det: bool = False,
) -> torch.Tensor:
    """
    Sample from a piecewise-constant PDF (inverse transform sampling).
    
    Used for hierarchical sampling in the fine network.
    
    Parameters
    ----------
    bins : torch.Tensor
        Bin edges (z values), shape (..., N).
    weights : torch.Tensor
        Weights for each bin, shape (..., N-1).
    num_samples : int
        Number of samples to draw.
    det : bool
        If True, use deterministic sampling (for evaluation).
        
    Returns
    -------
    samples : torch.Tensor
        Sampled z values, shape (..., num_samples).
    """
    device = weights.device
    
    # Add small epsilon to avoid NaN
    weights = weights + 1e-5
    
    # Compute PDF and CDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    
    # Sample uniformly
    if det:
        u = torch.linspace(0.0, 1.0, num_samples, device=device)
        u = u.expand(*cdf.shape[:-1], num_samples)
    else:
        u = torch.rand(*cdf.shape[:-1], num_samples, device=device)
    
    # Invert CDF via binary search
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    
    # Clamp indices
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)
    
    # Gather CDF and bin values
    cdf_g = torch.gather(cdf, -1, inds_g.reshape(*cdf.shape[:-1], -1))
    cdf_g = cdf_g.reshape(*inds_g.shape)
    
    bins_g = torch.gather(bins, -1, inds_g.reshape(*bins.shape[:-1], -1))
    bins_g = bins_g.reshape(*inds_g.shape)
    
    # Linear interpolation
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples


def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    num_samples_fine: int,
    det: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform hierarchical sampling for fine network.
    
    Parameters
    ----------
    rays_o : torch.Tensor
        Ray origins, shape (..., 3).
    rays_d : torch.Tensor
        Ray directions, shape (..., 3).
    z_vals : torch.Tensor
        Coarse sample depths, shape (..., N_coarse).
    weights : torch.Tensor
        Weights from coarse network, shape (..., N_coarse).
    num_samples_fine : int
        Number of additional fine samples.
    det : bool
        Deterministic sampling for evaluation.
        
    Returns
    -------
    pts_fine : torch.Tensor
        All sample points (coarse + fine, sorted), shape (..., N_total, 3).
    z_vals_fine : torch.Tensor
        All depth values (sorted), shape (..., N_total).
    """
    # Get bin edges
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    
    # Sample from the weight distribution
    z_samples = sample_pdf(
        z_vals_mid,
        weights[..., 1:-1],  # Middle weights (not including boundaries)
        num_samples_fine,
        det=det,
    )
    z_samples = z_samples.detach()
    
    # Combine coarse and fine samples, then sort
    z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    
    # Compute points
    pts_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
    
    return pts_fine, z_vals_fine







