"""
Data loading utilities for NeRF training.

Handles loading and preprocessing of the NeRF Synthetic (Blender) dataset.
Supports training with noisy camera poses for robustness experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .rays import get_ray_directions, get_rays
from .config import DataConfig
from .noise import NoiseConfig, add_noise_to_pose, set_noise_seed


@dataclass
class BlenderData:
    """
    Container for loaded Blender scene data.
    
    Attributes
    ----------
    images : torch.Tensor
        Images of shape (N, H, W, 3) in [0, 1].
    poses : torch.Tensor
        Camera-to-world matrices of shape (N, 4, 4).
    H : int
        Image height.
    W : int
        Image width.
    focal : float
        Focal length in pixels.
    """
    images: torch.Tensor
    poses: torch.Tensor
    H: int
    W: int
    focal: float


def load_blender_data(
    data_root: Path,
    scene_name: str,
    split: str = "train",
    img_scale: float = 0.5,
    device: str = "cpu",
) -> BlenderData:
    """
    Load a Blender synthetic scene.
    
    Parameters
    ----------
    data_root : Path
        Root directory containing the scene folders.
    scene_name : str
        Name of the scene (e.g., 'lego', 'chair').
    split : str
        One of 'train', 'val', or 'test'.
    img_scale : float
        Scale factor for images (0.5 = half resolution).
    device : str
        Device to load tensors to.
        
    Returns
    -------
    BlenderData
        Loaded scene data.
    """
    # Find scene directory
    scene_dir = None
    for candidate in [
        data_root / scene_name,
        data_root / "nerf_synthetic" / scene_name,
    ]:
        if candidate.exists():
            scene_dir = candidate
            break
    
    if scene_dir is None:
        raise FileNotFoundError(
            f"Could not find scene '{scene_name}' in {data_root}"
        )
    
    # Load transforms file
    transforms_path = scene_dir / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms file: {transforms_path}")
    
    with open(transforms_path, "r") as f:
        meta = json.load(f)
    
    camera_angle_x = float(meta["camera_angle_x"])
    
    images = []
    poses = []
    
    for frame in meta["frames"]:
        # Load image
        file_path = frame["file_path"]
        img_path = scene_dir / f"{file_path}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        
        img = Image.open(img_path)
        
        # Handle RGBA (composite on white for synthetic scenes)
        if img.mode == "RGBA":
            # White background compositing
            img_array = np.array(img, dtype=np.float32) / 255.0
            rgb = img_array[..., :3]
            alpha = img_array[..., 3:4]
            rgb = rgb * alpha + (1.0 - alpha)  # White background
            img = Image.fromarray((rgb * 255).astype(np.uint8))
        else:
            img = img.convert("RGB")
        
        W_orig, H_orig = img.size
        
        # Resize if needed
        if img_scale != 1.0:
            W_new = int(W_orig * img_scale)
            H_new = int(H_orig * img_scale)
            img = img.resize((W_new, H_new), Image.LANCZOS)
        
        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        )
        images.append(img_tensor)
        
        # Load pose
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        poses.append(torch.from_numpy(pose))
    
    # Stack into tensors
    images = torch.stack(images, dim=0)
    poses = torch.stack(poses, dim=0)
    
    H, W = images.shape[1:3]
    
    # Compute focal length from field of view
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    
    return BlenderData(
        images=images.to(device),
        poses=poses.to(device),
        H=H,
        W=W,
        focal=float(focal),
    )


class RayDataset(Dataset):
    """
    Dataset of rays for NeRF training.
    
    Precomputes all rays for efficient random sampling during training.
    Supports optional noise injection to camera poses.
    """
    
    def __init__(
        self,
        data: BlenderData,
        batch_size: int = 1024,
        noise_config: Optional[NoiseConfig] = None,
    ):
        """
        Parameters
        ----------
        data : BlenderData
            Loaded scene data.
        batch_size : int
            Number of rays per batch (not used directly, for compatibility).
        noise_config : NoiseConfig, optional
            If provided, adds noise to camera poses during ray generation.
            This simulates training with imperfect camera calibration.
        """
        self.H = data.H
        self.W = data.W
        self.focal = data.focal
        self.noise_config = noise_config
        
        # Store original poses for reference
        self.original_poses = data.poses.clone()
        
        # Precompute ray directions (shared across all images)
        directions = get_ray_directions(data.H, data.W, data.focal)
        self.directions = directions.to(data.images.device)
        
        # Set noise seed if specified
        if noise_config is not None and noise_config.seed is not None:
            set_noise_seed(noise_config.seed)
        
        # Generate all rays for all images
        all_rays_o = []
        all_rays_d = []
        all_colors = []
        all_noise_info = []
        
        N_images = data.images.shape[0]
        
        for i in range(N_images):
            pose = data.poses[i]
            
            # Apply noise to pose if configured
            if noise_config is not None and noise_config.has_noise:
                # Compute camera distance for percentage-based translation noise
                camera_pos = pose[:3, 3]
                camera_distance = torch.norm(camera_pos).item()
                trans_std = noise_config.get_translation_std(camera_distance)
                
                pose, noise_info = add_noise_to_pose(
                    pose,
                    rotation_noise_deg=noise_config.rotation_noise_deg,
                    translation_noise=trans_std,
                )
                all_noise_info.append(noise_info)
            
            rays_o, rays_d = get_rays(self.directions, pose)
            # Flatten spatial dimensions
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            colors = data.images[i].reshape(-1, 3)
            
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
            all_colors.append(colors)
        
        # Concatenate all rays
        self.rays_o = torch.cat(all_rays_o, dim=0)
        self.rays_d = torch.cat(all_rays_d, dim=0)
        self.colors = torch.cat(all_colors, dim=0)
        
        self.n_rays = self.rays_o.shape[0]
        self.noise_info = all_noise_info if all_noise_info else None
        
        # Log noise statistics
        if self.noise_info:
            rot_errors = [n.get("actual_rotation_deg", 0) for n in self.noise_info]
            trans_errors = [n.get("actual_translation_norm", 0) for n in self.noise_info]
            print(f"  Applied noise to {N_images} training poses:")
            print(f"    Rotation error: {np.mean(rot_errors):.3f} ± {np.std(rot_errors):.3f}°")
            print(f"    Translation error: {np.mean(trans_errors):.4f} ± {np.std(trans_errors):.4f}")
    
    def __len__(self) -> int:
        return self.n_rays
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "rays_o": self.rays_o[idx],
            "rays_d": self.rays_d[idx],
            "target_rgb": self.colors[idx],
        }


class RaySampler:
    """
    Efficient random ray sampler for training.
    
    Samples random batches of rays without DataLoader overhead.
    """
    
    def __init__(
        self,
        dataset: RayDataset,
        batch_size: int = 1024,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = dataset.rays_o.device
        
        self.n_rays = dataset.n_rays
        self._reset_indices()
    
    def _reset_indices(self):
        """Reset or shuffle indices."""
        if self.shuffle:
            self.indices = torch.randperm(self.n_rays, device=self.device)
        else:
            self.indices = torch.arange(self.n_rays, device=self.device)
        self.current_idx = 0
    
    def __iter__(self) -> Iterator[dict]:
        self._reset_indices()
        return self
    
    def __next__(self) -> dict:
        if self.current_idx >= self.n_rays:
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, self.n_rays)
        batch_indices = self.indices[self.current_idx:end_idx]
        self.current_idx = end_idx
        
        return {
            "rays_o": self.dataset.rays_o[batch_indices],
            "rays_d": self.dataset.rays_d[batch_indices],
            "target_rgb": self.dataset.colors[batch_indices],
        }
    
    def __len__(self) -> int:
        return (self.n_rays + self.batch_size - 1) // self.batch_size
    
    def sample_batch(self) -> dict:
        """Sample a random batch of rays."""
        indices = torch.randint(0, self.n_rays, (self.batch_size,), device=self.device)
        return {
            "rays_o": self.dataset.rays_o[indices],
            "rays_d": self.dataset.rays_d[indices],
            "target_rgb": self.dataset.colors[indices],
        }


def create_data_loaders(
    config: DataConfig,
    device: str = "cpu",
    noise_config: Optional[NoiseConfig] = None,
) -> Tuple[RaySampler, BlenderData, BlenderData]:
    """
    Create data loaders for training and validation.
    
    Parameters
    ----------
    config : DataConfig
        Data configuration.
    device : str
        Device for data tensors.
    noise_config : NoiseConfig, optional
        Noise to apply to training poses.
        
    Returns
    -------
    train_sampler : RaySampler
        Training ray sampler.
    train_data : BlenderData
        Training data (for reference).
    val_data : BlenderData
        Validation data.
    """
    data_root = config.data_root
    if data_root is None:
        # Default to project data directory
        data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
    
    # Load training data
    train_data = load_blender_data(
        data_root=data_root,
        scene_name=config.scene_name,
        split="train",
        img_scale=config.img_scale,
        device=device,
    )
    
    # Load validation data (always clean, no noise)
    val_data = load_blender_data(
        data_root=data_root,
        scene_name=config.scene_name,
        split="val",
        img_scale=config.img_scale,
        device=device,
    )
    
    # Create training dataset and sampler (with optional noise)
    train_dataset = RayDataset(
        train_data, 
        batch_size=config.batch_size,
        noise_config=noise_config,
    )
    train_sampler = RaySampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )
    
    return train_sampler, train_data, val_data
