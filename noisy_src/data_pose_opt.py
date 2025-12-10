"""
Data loading for joint pose optimization.

This module provides data loaders that store pixel coordinates instead of
precomputed rays, allowing rays to be regenerated from updated camera poses
during training.
"""

from __future__ import annotations

from typing import Iterator, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import BlenderData
from .rays import get_ray_directions, get_rays


@dataclass
class PixelBatch:
    """Batch of pixel coordinates with associated data."""
    image_indices: torch.Tensor  # (batch_size,) - which image each pixel belongs to
    pixel_coords: torch.Tensor   # (batch_size, 2) - (u, v) pixel coordinates
    target_rgb: torch.Tensor     # (batch_size, 3) - target colors


class PixelDataset:
    """
    Dataset that stores pixel coordinates instead of precomputed rays.
    
    This allows rays to be regenerated from updated camera poses during
    joint optimization.
    """
    
    def __init__(self, data: BlenderData):
        """
        Initialize pixel dataset.
        
        Parameters
        ----------
        data : BlenderData
            Loaded scene data
        """
        self.H = data.H
        self.W = data.W
        self.focal = data.focal
        self.device = data.images.device
        
        N_images = data.images.shape[0]
        N_pixels_per_image = self.H * self.W
        N_total_pixels = N_images * N_pixels_per_image
        
        # Create pixel coordinate grids
        v, u = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten: (H, W) -> (H*W, 2)
        pixel_coords_single = torch.stack([u.flatten(), v.flatten()], dim=-1)
        
        # Replicate for all images and add image indices
        self.image_indices = torch.repeat_interleave(
            torch.arange(N_images, dtype=torch.long),
            N_pixels_per_image
        ).to(self.device)
        
        self.pixel_coords = pixel_coords_single.unsqueeze(0).expand(
            N_images, -1, -1
        ).reshape(-1, 2).to(self.device)
        
        # Flatten colors: (N, H, W, 3) -> (N*H*W, 3)
        self.target_rgb = data.images.reshape(-1, 3)
        
        self.n_pixels = N_total_pixels
        
        # Precompute ray directions (shared across all images)
        self.ray_directions = get_ray_directions(self.H, self.W, self.focal).to(self.device)
    
    def get_rays_from_pixels(
        self,
        pixel_batch: PixelBatch,
        poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays from pixel coordinates and camera poses.
        
        Parameters
        ----------
        pixel_batch : PixelBatch
            Batch of pixels
        poses : torch.Tensor
            Camera poses for the images in the batch, shape (N_unique_images, 4, 4)
            
        Returns
        -------
        rays_o : torch.Tensor
            Ray origins, shape (batch_size, 3)
        rays_d : torch.Tensor
            Ray directions, shape (batch_size, 3)
        """
        batch_size = pixel_batch.image_indices.shape[0]
        unique_img_indices = torch.unique(pixel_batch.image_indices)
        
        # Build mapping from image index to pose
        img_idx_to_pose_idx = {idx.item(): i for i, idx in enumerate(unique_img_indices)}
        
        rays_o_list = []
        rays_d_list = []
        
        # Process each unique image
        for img_idx in unique_img_indices:
            # Find pixels belonging to this image
            mask = pixel_batch.image_indices == img_idx
            pixel_coords = pixel_batch.pixel_coords[mask]  # (n_pixels, 2)
            
            # Get pose for this image
            pose_idx = img_idx_to_pose_idx[img_idx.item()]
            pose = poses[pose_idx]
            
            # Get ray directions for these pixels
            # pixel_coords are (u, v) in [0, W) x [0, H)
            u = pixel_coords[:, 0].long()
            v = pixel_coords[:, 1].long()
            directions = self.ray_directions[v, u]  # (n_pixels, 3)
            
            # Transform to world space using pose
            rays_o_img, rays_d_img = get_rays(directions, pose)
            
            rays_o_list.append(rays_o_img)
            rays_d_list.append(rays_d_img)
        
        # Concatenate in the correct order (matching pixel_batch order)
        rays_o = torch.zeros(batch_size, 3, device=self.device)
        rays_d = torch.zeros(batch_size, 3, device=self.device)
        
        idx = 0
        for img_idx in unique_img_indices:
            mask = pixel_batch.image_indices == img_idx
            n_pixels = mask.sum()
            rays_o[mask] = rays_o_list[idx]
            rays_d[mask] = rays_d_list[idx]
            idx += 1
        
        return rays_o, rays_d


class PixelSampler:
    """
    Random pixel sampler for training with pose optimization.
    
    Samples random pixels and regenerates rays from current poses.
    """
    
    def __init__(
        self,
        dataset: PixelDataset,
        batch_size: int = 1024,
    ):
        """
        Initialize pixel sampler.
        
        Parameters
        ----------
        dataset : PixelDataset
            Pixel dataset
        batch_size : int
            Number of pixels per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = dataset.device
        self.n_pixels = dataset.n_pixels
    
    def sample_batch(self) -> PixelBatch:
        """
        Sample a random batch of pixels.
        
        Returns
        -------
        PixelBatch
            Batch of sampled pixels
        """
        # Random sampling with replacement
        indices = torch.randint(
            0, self.n_pixels,
            (self.batch_size,),
            device=self.device
        )
        
        return PixelBatch(
            image_indices=self.dataset.image_indices[indices],
            pixel_coords=self.dataset.pixel_coords[indices],
            target_rgb=self.dataset.target_rgb[indices],
        )
    
    def get_rays_for_batch(
        self,
        pixel_batch: PixelBatch,
        poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for a pixel batch using given poses.
        
        Parameters
        ----------
        pixel_batch : PixelBatch
            Batch of pixels
        poses : torch.Tensor
            Camera poses, shape (N_images, 4, 4)
            
        Returns
        -------
        rays_o, rays_d : torch.Tensor
            Ray origins and directions
        """
        unique_img_indices = torch.unique(pixel_batch.image_indices)
        selected_poses = poses[unique_img_indices]
        
        return self.dataset.get_rays_from_pixels(pixel_batch, selected_poses)


def create_pixel_dataset(data: BlenderData) -> Tuple[PixelDataset, PixelSampler]:
    """
    Create pixel dataset and sampler for pose optimization training.
    
    Parameters
    ----------
    data : BlenderData
        Loaded scene data
        
    Returns
    -------
    dataset : PixelDataset
        Pixel dataset
    sampler : PixelSampler
        Pixel sampler
    """
    dataset = PixelDataset(data)
    sampler = PixelSampler(dataset, batch_size=1024)
    return dataset, sampler



