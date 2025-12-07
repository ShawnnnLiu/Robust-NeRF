"""
Robust-NeRF: Clean baseline implementation.

This package provides a modular, clean NeRF implementation
that can be extended to handle noisy camera parameters.
"""

__version__ = "0.1.0"

from .config import NeRFConfig, ModelConfig, RenderConfig, DataConfig, TrainConfig
from .model import NeRF, PositionalEncoding, create_nerf
from .rendering import NeRFRenderer, render_rays, raw2outputs
from .rays import (
    get_ray_directions,
    get_rays,
    sample_along_rays,
    sample_hierarchical,
)
from .data import load_blender_data, RayDataset, RaySampler, create_data_loaders
from .train import train
from .metrics import compute_psnr, compute_ssim, compute_mse, compute_all_metrics
from .logger import ExperimentLogger, TrainingMetrics, ValidationMetrics
from .noise import NoiseConfig, add_noise_to_pose, add_noise_to_poses, compute_pose_error

__all__ = [
    # Config
    "NeRFConfig",
    "ModelConfig", 
    "RenderConfig",
    "DataConfig",
    "TrainConfig",
    # Model
    "NeRF",
    "PositionalEncoding",
    "create_nerf",
    # Rendering
    "NeRFRenderer",
    "render_rays",
    "raw2outputs",
    # Rays
    "get_ray_directions",
    "get_rays",
    "sample_along_rays",
    "sample_hierarchical",
    # Data
    "load_blender_data",
    "RayDataset",
    "RaySampler",
    "create_data_loaders",
    # Training
    "train",
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_mse",
    "compute_all_metrics",
    # Logging
    "ExperimentLogger",
    "TrainingMetrics",
    "ValidationMetrics",
    # Noise
    "NoiseConfig",
    "add_noise_to_pose",
    "add_noise_to_poses",
    "compute_pose_error",
]

