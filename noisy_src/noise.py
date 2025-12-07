"""
Noise injection utilities for camera extrinsics.

Provides functions to add Gaussian noise to:
- Rotation (SO3 perturbations)
- Translation
- Combined pose noise
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Configuration for camera pose noise."""
    
    rotation_noise_deg: float = 0.0      # Rotation noise std in degrees
    translation_noise: float = 0.0       # Translation noise std (scene units)
    translation_noise_pct: float = 0.0   # Translation noise std (percentage of camera distance)
    seed: Optional[int] = None           # Random seed for reproducibility
    
    def __str__(self) -> str:
        parts = []
        if self.rotation_noise_deg > 0:
            parts.append(f"rot{self.rotation_noise_deg:.1f}deg")
        if self.translation_noise_pct > 0:
            parts.append(f"trans{self.translation_noise_pct:.1f}pct")
        elif self.translation_noise > 0:
            parts.append(f"trans{self.translation_noise:.3f}")
        if not parts:
            return "clean"
        return "_".join(parts)
    
    @property
    def has_noise(self) -> bool:
        return self.rotation_noise_deg > 0 or self.translation_noise > 0 or self.translation_noise_pct > 0
    
    def get_translation_std(self, camera_distance: float) -> float:
        """
        Get the translation noise standard deviation in scene units.
        
        If translation_noise_pct is set, converts percentage to absolute units
        based on camera distance. Otherwise returns translation_noise directly.
        
        Parameters
        ----------
        camera_distance : float
            Distance from camera to scene center (for percentage conversion).
            
        Returns
        -------
        float
            Translation noise standard deviation in scene units.
        """
        if self.translation_noise_pct > 0:
            return camera_distance * (self.translation_noise_pct / 100.0)
        return self.translation_noise


def set_noise_seed(seed: int):
    """Set random seed for reproducible noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def random_rotation_matrix(std_deg: float, device: str = "cpu") -> torch.Tensor:
    """
    Generate a random rotation matrix with Gaussian-distributed angles.
    
    Uses axis-angle representation: sample random axis and angle.
    
    Parameters
    ----------
    std_deg : float
        Standard deviation of rotation angle in degrees.
    device : str
        Device for the tensor.
        
    Returns
    -------
    torch.Tensor
        Random 3x3 rotation matrix.
    """
    if std_deg == 0:
        return torch.eye(3, device=device)
    
    # Convert to radians
    std_rad = std_deg * np.pi / 180.0
    
    # Sample rotation angle from Gaussian
    angle = torch.randn(1, device=device) * std_rad
    
    # Sample random axis (uniform on sphere)
    axis = torch.randn(3, device=device)
    axis = axis / torch.norm(axis)
    
    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=device)
    
    R = torch.eye(3, device=device) + \
        torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * (K @ K)
    
    return R


def random_translation(std: float, device: str = "cpu") -> torch.Tensor:
    """
    Generate a random translation vector with Gaussian noise.
    
    Parameters
    ----------
    std : float
        Standard deviation of translation noise.
    device : str
        Device for the tensor.
        
    Returns
    -------
    torch.Tensor
        Random 3D translation vector.
    """
    if std == 0:
        return torch.zeros(3, device=device)
    
    return torch.randn(3, device=device) * std


def add_noise_to_pose(
    pose: torch.Tensor,
    rotation_noise_deg: float = 0.0,
    translation_noise: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Add Gaussian noise to a camera-to-world pose matrix.
    
    Parameters
    ----------
    pose : torch.Tensor
        Original 4x4 camera-to-world matrix.
    rotation_noise_deg : float
        Std of rotation noise in degrees.
    translation_noise : float
        Std of translation noise.
        
    Returns
    -------
    noisy_pose : torch.Tensor
        Noisy 4x4 camera-to-world matrix.
    noise_info : dict
        Information about the applied noise.
    """
    device = pose.device
    noisy_pose = pose.clone()
    
    noise_info = {
        "rotation_noise_deg": rotation_noise_deg,
        "translation_noise": translation_noise,
    }
    
    # Add rotation noise
    if rotation_noise_deg > 0:
        R_noise = random_rotation_matrix(rotation_noise_deg, device)
        # Apply noise to rotation part: R_noisy = R_noise @ R_original
        noisy_pose[:3, :3] = R_noise @ pose[:3, :3]
        
        # Compute actual rotation angle applied
        trace = torch.trace(R_noise)
        angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        noise_info["actual_rotation_deg"] = float(angle_rad * 180 / np.pi)
    
    # Add translation noise
    if translation_noise > 0:
        t_noise = random_translation(translation_noise, device)
        noisy_pose[:3, 3] = pose[:3, 3] + t_noise
        noise_info["actual_translation_norm"] = float(torch.norm(t_noise))
    
    return noisy_pose, noise_info


def add_noise_to_poses(
    poses: torch.Tensor,
    noise_config: NoiseConfig,
) -> Tuple[torch.Tensor, list]:
    """
    Add noise to a batch of poses.
    
    Parameters
    ----------
    poses : torch.Tensor
        Original poses of shape (N, 4, 4).
    noise_config : NoiseConfig
        Noise configuration.
        
    Returns
    -------
    noisy_poses : torch.Tensor
        Noisy poses of shape (N, 4, 4).
    noise_info_list : list
        List of noise info dicts for each pose.
    """
    if noise_config.seed is not None:
        set_noise_seed(noise_config.seed)
    
    n_poses = poses.shape[0]
    noisy_poses = []
    noise_info_list = []
    
    for i in range(n_poses):
        # Compute camera distance from origin for percentage-based noise
        camera_pos = poses[i][:3, 3]
        camera_distance = torch.norm(camera_pos).item()
        
        # Get translation std (handles both absolute and percentage-based)
        trans_std = noise_config.get_translation_std(camera_distance)
        
        noisy_pose, noise_info = add_noise_to_pose(
            poses[i],
            rotation_noise_deg=noise_config.rotation_noise_deg,
            translation_noise=trans_std,
        )
        noisy_poses.append(noisy_pose)
        noise_info_list.append(noise_info)
    
    return torch.stack(noisy_poses, dim=0), noise_info_list


def compute_pose_error(
    pose_gt: torch.Tensor,
    pose_noisy: torch.Tensor,
) -> dict:
    """
    Compute error between ground truth and noisy pose.
    
    Returns
    -------
    dict with:
        rotation_error_deg : float
            Rotation error in degrees.
        translation_error : float
            Translation error (Euclidean distance).
    """
    # Rotation error
    R_gt = pose_gt[:3, :3]
    R_noisy = pose_noisy[:3, :3]
    R_diff = R_gt.T @ R_noisy
    trace = torch.trace(R_diff)
    angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    rotation_error_deg = float(angle_rad * 180 / np.pi)
    
    # Translation error
    t_gt = pose_gt[:3, 3]
    t_noisy = pose_noisy[:3, 3]
    translation_error = float(torch.norm(t_gt - t_noisy))
    
    return {
        "rotation_error_deg": rotation_error_deg,
        "translation_error": translation_error,
    }




