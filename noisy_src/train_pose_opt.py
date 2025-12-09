"""
Joint NeRF and Camera Pose Optimization.

This is the CORE module for camera pose refinement with NeRF training.
It implements joint optimization of scene representation (NeRF) and camera
extrinsics (rotation + translation) to learn from noisy camera poses.

Key Features:
- Joint optimization of NeRF and camera poses
- SE(3) parameterization with axis-angle rotation
- Configurable initialization (clean or noisy poses)
- Separate learning rates for NeRF and poses
- Comprehensive tracking of pose refinement
- Support for staged optimization (freeze/unfreeze poses)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .config import NeRFConfig, ModelConfig, RenderConfig, DataConfig, TrainConfig
from .model import NeRF, create_nerf
from .rendering import NeRFRenderer, render_rays
from .data import create_data_loaders, load_blender_data, BlenderData
from .data_pose_opt import PixelDataset, PixelSampler, PixelBatch, create_pixel_dataset
from .rays import get_ray_directions, get_rays
from .metrics import compute_psnr, compute_ssim, compute_mse, compute_all_metrics, LPIPSMetric
from .logger import ExperimentLogger, TrainingMetrics, ValidationMetrics
from .noise import NoiseConfig, add_noise_to_poses, compute_pose_error


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CameraPoseParameters(nn.Module):
    """
    Learnable camera pose parameters using SE(3) representation.
    
    Uses axis-angle representation for rotations and 3D vectors for translations.
    This avoids gimbal lock and provides a minimal parameterization.
    
    Parameters are stored as deltas from initial poses:
        R_opt = exp(omega) @ R_init
        t_opt = t_init + delta_t
    
    Attributes
    ----------
    rotation_deltas : nn.Parameter
        Axis-angle rotation deltas, shape (N, 3)
    translation_deltas : nn.Parameter
        Translation deltas, shape (N, 3)
    initial_poses : torch.Tensor
        Initial camera poses (fixed), shape (N, 4, 4)
    """
    
    def __init__(
        self,
        initial_poses: torch.Tensor,
        learn_rotation: bool = True,
        learn_translation: bool = True,
    ):
        """
        Initialize camera pose parameters.
        
        Parameters
        ----------
        initial_poses : torch.Tensor
            Initial camera-to-world poses, shape (N, 4, 4)
        learn_rotation : bool
            Whether to optimize rotations
        learn_translation : bool
            Whether to optimize translations
        """
        super().__init__()
        
        self.n_poses = initial_poses.shape[0]
        self.learn_rotation = learn_rotation
        self.learn_translation = learn_translation
        
        # Store initial poses (not trainable)
        self.register_buffer("initial_poses", initial_poses.clone())
        
        # Initialize deltas to zero (start from initial poses)
        if learn_rotation:
            self.rotation_deltas = nn.Parameter(
                torch.zeros(self.n_poses, 3, device=initial_poses.device)
            )
        else:
            self.register_buffer(
                "rotation_deltas",
                torch.zeros(self.n_poses, 3, device=initial_poses.device)
            )
        
        if learn_translation:
            self.translation_deltas = nn.Parameter(
                torch.zeros(self.n_poses, 3, device=initial_poses.device)
            )
        else:
            self.register_buffer(
                "translation_deltas",
                torch.zeros(self.n_poses, 3, device=initial_poses.device)
            )
    
    def axis_angle_to_rotation_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle representation to rotation matrix using Rodrigues' formula.
        
        Parameters
        ----------
        axis_angle : torch.Tensor
            Axis-angle vectors, shape (..., 3)
            
        Returns
        -------
        torch.Tensor
            Rotation matrices, shape (..., 3, 3)
        """
        batch_shape = axis_angle.shape[:-1]
        axis_angle = axis_angle.reshape(-1, 3)
        
        # Compute rotation angle
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)
        
        # Handle zero rotation (add epsilon for numerical stability)
        small_angle = angle < 1e-6
        angle = torch.where(small_angle, torch.ones_like(angle), angle)
        
        # Normalize to get rotation axis
        axis = axis_angle / angle
        
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        K = self._skew_symmetric(axis)
        K2 = torch.bmm(K, K)
        
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(axis.shape[0], 3, 3)
        
        sin_angle = torch.sin(angle).unsqueeze(-1)
        cos_angle = torch.cos(angle).unsqueeze(-1)
        
        R = I + sin_angle * K + (1 - cos_angle) * K2
        
        # For very small angles, use identity matrix
        R = torch.where(small_angle.reshape(-1, 1, 1), I, R)
        
        return R.reshape(*batch_shape, 3, 3)
    
    def _skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """
        Create skew-symmetric matrix from vectors.
        
        Parameters
        ----------
        v : torch.Tensor
            Vectors of shape (N, 3)
            
        Returns
        -------
        torch.Tensor
            Skew-symmetric matrices of shape (N, 3, 3)
        """
        zeros = torch.zeros(v.shape[0], device=v.device)
        return torch.stack([
            torch.stack([zeros, -v[:, 2], v[:, 1]], dim=-1),
            torch.stack([v[:, 2], zeros, -v[:, 0]], dim=-1),
            torch.stack([-v[:, 1], v[:, 0], zeros], dim=-1),
        ], dim=1)
    
    def get_poses(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get current camera poses (initial + learned deltas).
        
        Parameters
        ----------
        indices : torch.Tensor, optional
            Indices of poses to retrieve. If None, returns all poses.
            
        Returns
        -------
        torch.Tensor
            Camera-to-world poses, shape (N, 4, 4) or (len(indices), 4, 4)
        """
        if indices is None:
            indices = torch.arange(self.n_poses, device=self.initial_poses.device)
        
        # Get initial poses
        poses_init = self.initial_poses[indices]
        
        # Apply rotation deltas: R_new = exp(omega) @ R_init
        if self.learn_rotation:
            R_delta = self.axis_angle_to_rotation_matrix(self.rotation_deltas[indices])
            R_init = poses_init[:, :3, :3]
            R_new = torch.bmm(R_delta, R_init)
        else:
            R_new = poses_init[:, :3, :3]
        
        # Apply translation deltas: t_new = t_init + delta_t
        if self.learn_translation:
            t_new = poses_init[:, :3, 3] + self.translation_deltas[indices]
        else:
            t_new = poses_init[:, :3, 3]
        
        # Construct new poses
        poses = torch.zeros_like(poses_init)
        poses[:, :3, :3] = R_new
        poses[:, :3, 3] = t_new
        poses[:, 3, 3] = 1.0
        
        return poses
    
    def get_all_poses(self) -> torch.Tensor:
        """Get all current camera poses."""
        return self.get_poses()
    
    def compute_pose_errors(
        self,
        ground_truth_poses: torch.Tensor,
        indices: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute errors between current poses and ground truth.
        
        Parameters
        ----------
        ground_truth_poses : torch.Tensor
            Ground truth poses, shape (N, 4, 4)
        indices : torch.Tensor, optional
            Which poses to evaluate
            
        Returns
        -------
        dict
            Statistics on rotation and translation errors
        """
        current_poses = self.get_poses(indices)
        if indices is not None:
            ground_truth_poses = ground_truth_poses[indices]
        
        rot_errors = []
        trans_errors = []
        
        for i in range(current_poses.shape[0]):
            error = compute_pose_error(ground_truth_poses[i], current_poses[i])
            rot_errors.append(error["rotation_error_deg"])
            trans_errors.append(error["translation_error"])
        
        return {
            "rotation_error_mean": float(np.mean(rot_errors)),
            "rotation_error_std": float(np.std(rot_errors)),
            "rotation_error_max": float(np.max(rot_errors)),
            "translation_error_mean": float(np.mean(trans_errors)),
            "translation_error_std": float(np.std(trans_errors)),
            "translation_error_max": float(np.max(trans_errors)),
        }


def generate_experiment_name(
    scene: str,
    noise_config: Optional[NoiseConfig],
    init_mode: str = "noisy",
) -> str:
    """Generate informative experiment name for pose optimization."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if noise_config is not None and noise_config.has_noise:
        noise_desc = str(noise_config)
    else:
        noise_desc = "clean"
    
    return f"{scene}_poseopt_{init_mode}init_{noise_desc}_{timestamp}"


def train_step_with_poses(
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    camera_params: CameraPoseParameters,
    pixel_sampler: PixelSampler,
    optimizer_nerf: torch.optim.Optimizer,
    optimizer_poses: Optional[torch.optim.Optimizer],
    pixel_batch: PixelBatch,
    render_config: RenderConfig,
    optimize_poses: bool = True,
    rotation_reg_weight: float = 0.0,
    translation_reg_weight: float = 0.0,
) -> Dict[str, float]:
    """
    Training step with joint NeRF and camera pose optimization.
    
    Parameters
    ----------
    model_coarse, model_fine : NeRF
        NeRF models
    camera_params : CameraPoseParameters
        Learnable camera parameters
    pixel_sampler : PixelSampler
        Pixel sampler to generate rays from poses
    optimizer_nerf : torch.optim.Optimizer
        Optimizer for NeRF parameters
    optimizer_poses : torch.optim.Optimizer, optional
        Optimizer for camera poses
    pixel_batch : PixelBatch
        Batch of pixels with coordinates and colors
    render_config : RenderConfig
        Rendering configuration
    optimize_poses : bool
        Whether to optimize poses this step
    rotation_reg_weight : float
        L2 regularization weight for rotation deltas
    translation_reg_weight : float
        L2 regularization weight for translation deltas
        
    Returns
    -------
    dict
        Training metrics
    """
    optimizer_nerf.zero_grad()
    if optimizer_poses is not None and optimize_poses:
        optimizer_poses.zero_grad()
    
    # Generate rays from current (optimized) camera poses
    # This is the KEY step - rays are computed from learnable poses
    all_poses = camera_params.get_all_poses()
    rays_o, rays_d = pixel_sampler.get_rays_for_batch(pixel_batch, all_poses)
    target_rgb = pixel_batch.target_rgb
    
    # Render rays
    outputs = render_rays(
        model_coarse=model_coarse,
        model_fine=model_fine,
        rays_o=rays_o,
        rays_d=rays_d,
        config=render_config,
        is_train=True,
    )
    
    # Compute losses
    rgb_coarse = outputs["rgb_coarse"]
    loss_coarse = torch.mean((rgb_coarse - target_rgb) ** 2)
    
    metrics = {
        "loss_coarse": loss_coarse.item(),
        "psnr_coarse": compute_psnr(rgb_coarse, target_rgb).item(),
    }
    
    if "rgb_fine" in outputs:
        rgb_fine = outputs["rgb_fine"]
        loss_fine = torch.mean((rgb_fine - target_rgb) ** 2)
        loss = loss_coarse + loss_fine
        
        metrics["loss_fine"] = loss_fine.item()
        metrics["psnr_fine"] = compute_psnr(rgb_fine, target_rgb).item()
        metrics["psnr"] = metrics["psnr_fine"]
    else:
        loss = loss_coarse
        metrics["loss_fine"] = None
        metrics["psnr"] = metrics["psnr_coarse"]
    
    # Add pose regularization (penalize deviation from initial poses)
    pose_reg_loss = 0.0
    if optimize_poses and (rotation_reg_weight > 0 or translation_reg_weight > 0):
        if rotation_reg_weight > 0 and camera_params.learn_rotation:
            rotation_reg = torch.mean(camera_params.rotation_deltas ** 2)
            pose_reg_loss += rotation_reg_weight * rotation_reg
            metrics["rotation_reg"] = rotation_reg.item()
        
        if translation_reg_weight > 0 and camera_params.learn_translation:
            translation_reg = torch.mean(camera_params.translation_deltas ** 2)
            pose_reg_loss += translation_reg_weight * translation_reg
            metrics["translation_reg"] = translation_reg.item()
        
        loss = loss + pose_reg_loss
        metrics["pose_reg_loss"] = pose_reg_loss.item() if isinstance(pose_reg_loss, torch.Tensor) else pose_reg_loss
    
    metrics["loss"] = loss.item()
    
    # Backprop
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model_coarse.parameters(), max_norm=1.0)
    if model_fine is not None:
        torch.nn.utils.clip_grad_norm_(model_fine.parameters(), max_norm=1.0)
    
    if optimize_poses and optimizer_poses is not None:
        # Clip pose gradients more conservatively
        torch.nn.utils.clip_grad_norm_(camera_params.parameters(), max_norm=0.1)
    
    # Update parameters
    optimizer_nerf.step()
    if optimizer_poses is not None and optimize_poses:
        optimizer_poses.step()
    
    return metrics


@torch.no_grad()
def render_image_with_pose(
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    pose: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    render_config: RenderConfig,
    chunk_size: int = 1024 * 4,
) -> Dict[str, torch.Tensor]:
    """Render a full image from a camera pose."""
    device = pose.device
    
    # Generate rays
    directions = get_ray_directions(H, W, focal).to(device)
    rays_o, rays_d = get_rays(directions, pose)
    
    # Flatten
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # Render in chunks
    all_rgb = []
    all_depth = []
    all_acc = []
    
    for i in range(0, rays_o.shape[0], chunk_size):
        chunk_o = rays_o[i:i + chunk_size]
        chunk_d = rays_d[i:i + chunk_size]
        
        outputs = render_rays(
            model_coarse=model_coarse,
            model_fine=model_fine,
            rays_o=chunk_o,
            rays_d=chunk_d,
            config=render_config,
            is_train=False,
        )
        
        if "rgb_fine" in outputs:
            all_rgb.append(outputs["rgb_fine"])
            all_depth.append(outputs["depth_fine"])
            all_acc.append(outputs["acc_fine"])
        else:
            all_rgb.append(outputs["rgb_coarse"])
            all_depth.append(outputs["depth_coarse"])
            all_acc.append(outputs["acc_coarse"])
    
    # Concatenate and reshape
    result = {
        "rgb": torch.cat(all_rgb, dim=0).reshape(H, W, 3),
        "depth": torch.cat(all_depth, dim=0).reshape(H, W),
        "acc": torch.cat(all_acc, dim=0).reshape(H, W),
    }
    
    return result


@torch.no_grad()
def evaluate_with_poses(
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    camera_params: CameraPoseParameters,
    val_data: BlenderData,
    val_indices: torch.Tensor,
    render_config: RenderConfig,
    logger: ExperimentLogger,
    iteration: int,
    num_images: int = 5,
    lpips_metric: Optional[LPIPSMetric] = None,
) -> ValidationMetrics:
    """Evaluation using ground truth validation poses."""
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_mse = []
    
    eval_indices = val_indices[:min(num_images, len(val_indices))]
    
    for i, idx in enumerate(eval_indices):
        # Use ground truth validation pose (not optimized training pose!)
        pose = val_data.poses[idx]
        target = val_data.images[idx]
        
        output = render_image_with_pose(
            model_coarse=model_coarse,
            model_fine=model_fine,
            pose=pose,
            H=val_data.H,
            W=val_data.W,
            focal=val_data.focal,
            render_config=render_config,
        )
        
        pred = output["rgb"]
        
        # Compute metrics
        mse = compute_mse(pred, target).item()
        psnr = compute_psnr(pred, target).item()
        ssim = compute_ssim(pred, target).item()
        
        all_mse.append(mse)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        
        if lpips_metric is not None:
            lpips_val = lpips_metric(pred, target)
            if lpips_val is not None:
                all_lpips.append(lpips_val.item())
        
        # Log images for first few
        if i < 3:
            logger.log_images(
                tag=f"val_{idx}",
                pred=pred,
                gt=target,
                iteration=iteration,
                depth=output["depth"],
            )
    
    metrics = ValidationMetrics(
        iteration=iteration,
        psnr=float(np.mean(all_psnr)),
        ssim=float(np.mean(all_ssim)),
        mse=float(np.mean(all_mse)),
        lpips=float(np.mean(all_lpips)) if all_lpips else None,
        per_image_psnr=all_psnr,
        per_image_ssim=all_ssim,
    )
    
    return metrics


def save_checkpoint_with_poses(
    output_dir: Path,
    iteration: int,
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    camera_params: CameraPoseParameters,
    optimizer_nerf: torch.optim.Optimizer,
    optimizer_poses: Optional[torch.optim.Optimizer],
    config: NeRFConfig,
    noise_config: Optional[NoiseConfig] = None,
    metrics: Optional[Dict] = None,
    pose_errors: Optional[Dict] = None,
    is_best: bool = False,
):
    """Save checkpoint including camera pose parameters."""
    checkpoint = {
        "iteration": iteration,
        "model_coarse": model_coarse.state_dict(),
        "camera_params": camera_params.state_dict(),
        "optimizer_nerf": optimizer_nerf.state_dict(),
        "initial_poses": camera_params.initial_poses.cpu(),
        "config": {
            "model": config.model.__dict__,
            "render": config.render.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v 
                     for k, v in config.data.__dict__.items()},
            "train": {k: str(v) if isinstance(v, Path) else v 
                     for k, v in config.train.__dict__.items()},
        },
    }
    
    if model_fine is not None:
        checkpoint["model_fine"] = model_fine.state_dict()
    
    if optimizer_poses is not None:
        checkpoint["optimizer_poses"] = optimizer_poses.state_dict()
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if pose_errors is not None:
        checkpoint["pose_errors"] = pose_errors
    
    if noise_config is not None:
        checkpoint["noise_config"] = {
            "rotation_noise_deg": noise_config.rotation_noise_deg,
            "translation_noise": noise_config.translation_noise,
            "translation_noise_pct": noise_config.translation_noise_pct,
            "seed": noise_config.seed,
        }
    
    checkpoint_path = output_dir / f"checkpoint_{iteration:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"✓ Saved BEST checkpoint to {best_path}")
    else:
        print(f"✓ Saved checkpoint to {checkpoint_path}")


def train_with_pose_optimization(
    config: NeRFConfig,
    noise_config: Optional[NoiseConfig] = None,
    init_mode: str = "noisy",
    pose_lr: float = 1e-4,
    pose_opt_delay: int = 1000,
    learn_rotation: bool = True,
    learn_translation: bool = True,
    rotation_reg_weight: float = 0.01,
    translation_reg_weight: float = 0.001,
):
    """
    Main training function with joint pose optimization.
    
    Parameters
    ----------
    config : NeRFConfig
        Training configuration
    noise_config : NoiseConfig, optional
        Noise configuration (for generating noisy initialization)
    init_mode : str
        Initialization mode: 'clean' or 'noisy'
        - 'clean': Initialize with ground truth poses
        - 'noisy': Initialize with noisy poses
    pose_lr : float
        Learning rate for camera pose parameters
    pose_opt_delay : int
        Start optimizing poses after this many iterations
    learn_rotation : bool
        Whether to optimize camera rotations
    learn_translation : bool
        Whether to optimize camera translations
    rotation_reg_weight : float
        L2 regularization weight for rotation deltas (prevents pose drift)
    translation_reg_weight : float
        L2 regularization weight for translation deltas (prevents pose drift)
    """
    # Setup
    set_seed(config.train.seed)
    device = config.train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Generate experiment name
    exp_name = generate_experiment_name(
        config.data.scene_name,
        noise_config,
        init_mode,
    )
    
    output_dir = config.train.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(
        output_dir=output_dir,
        experiment_name=exp_name,
        use_tensorboard=True,
    )
    
    logger.log_config(config)
    
    print(f"\n{'=' * 70}")
    print(f"  JOINT NERF AND CAMERA POSE OPTIMIZATION")
    print(f"{'=' * 70}")
    print(f"Scene:        {config.data.scene_name}")
    print(f"Experiment:   {exp_name}")
    print(f"Output:       {output_dir}")
    print(f"Device:       {device}")
    print(f"\nPose Optimization Settings:")
    print(f"  Initialization:  {init_mode}")
    print(f"  Learn rotation:  {learn_rotation}")
    print(f"  Learn translation: {learn_translation}")
    print(f"  Pose LR:         {pose_lr:.1e}")
    print(f"  Pose opt delay:  {pose_opt_delay} iterations")
    print(f"  Rotation reg:    {rotation_reg_weight:.1e}")
    print(f"  Translation reg: {translation_reg_weight:.1e}")
    
    if noise_config and noise_config.has_noise:
        print(f"\nNoise Configuration:")
        print(f"  Rotation:    {noise_config.rotation_noise_deg}° (std)")
        if noise_config.translation_noise_pct > 0:
            print(f"  Translation: {noise_config.translation_noise_pct}% (std)")
        else:
            print(f"  Translation: {noise_config.translation_noise} (std)")
    
    # Load data (without applying noise to training data)
    print(f"\n{'─' * 70}")
    print(f"Loading data...")
    
    data_root = config.data.data_root
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
    
    # Load clean training and validation data
    train_data = load_blender_data(
        data_root=data_root,
        scene_name=config.data.scene_name,
        split="train",
        img_scale=config.data.img_scale,
        device=device,
    )
    
    val_data = load_blender_data(
        data_root=data_root,
        scene_name=config.data.scene_name,
        split="val",
        img_scale=config.data.img_scale,
        device=device,
    )
    
    print(f"✓ Training images:   {train_data.images.shape[0]}")
    print(f"✓ Validation images: {val_data.images.shape[0]}")
    print(f"✓ Resolution:        {train_data.H} x {train_data.W}")
    print(f"✓ Focal length:      {train_data.focal:.2f}")
    
    # Store ground truth poses for error tracking
    gt_train_poses = train_data.poses.clone()
    
    # Initialize camera poses based on init_mode
    if init_mode == "noisy" and noise_config and noise_config.has_noise:
        print(f"\n{'─' * 70}")
        print(f"Applying noise to initial poses...")
        noisy_poses, noise_info_list = add_noise_to_poses(
            train_data.poses,
            noise_config,
        )
        
        # Compute initial errors
        init_errors = []
        for i in range(len(train_data.poses)):
            err = compute_pose_error(gt_train_poses[i], noisy_poses[i])
            init_errors.append(err)
        
        rot_errs = [e["rotation_error_deg"] for e in init_errors]
        trans_errs = [e["translation_error"] for e in init_errors]
        
        print(f"✓ Initial pose errors:")
        print(f"  Rotation:    {np.mean(rot_errs):.3f} ± {np.std(rot_errs):.3f}° (max: {np.max(rot_errs):.3f}°)")
        print(f"  Translation: {np.mean(trans_errs):.4f} ± {np.std(trans_errs):.4f} (max: {np.max(trans_errs):.4f})")
        
        initial_poses = noisy_poses
    else:
        print(f"\nInitializing with CLEAN poses (ground truth)")
        initial_poses = gt_train_poses.clone()
    
    # Create learnable camera parameters
    camera_params = CameraPoseParameters(
        initial_poses=initial_poses,
        learn_rotation=learn_rotation,
        learn_translation=learn_translation,
    ).to(device)
    
    print(f"✓ Created learnable camera parameters ({camera_params.n_poses} poses)")
    
    # Create NeRF models
    print(f"\n{'─' * 70}")
    print(f"Creating models...")
    model_coarse, model_fine = create_nerf(config.model)
    model_coarse = model_coarse.to(device)
    if config.render.use_hierarchical:
        model_fine = model_fine.to(device)
    else:
        model_fine = None
    
    logger.log_model_info(model_coarse, "model_coarse")
    if model_fine:
        logger.log_model_info(model_fine, "model_fine")
    
    # Create optimizers
    nerf_params = list(model_coarse.parameters())
    if model_fine is not None:
        nerf_params += list(model_fine.parameters())
    
    optimizer_nerf = Adam(nerf_params, lr=config.train.lr)
    optimizer_poses = Adam(camera_params.parameters(), lr=pose_lr)
    
    # Learning rate schedulers
    decay_rate = 0.1
    decay_steps = config.train.lr_decay * 1000
    
    def lr_lambda(step):
        return decay_rate ** (step / decay_steps)
    
    scheduler_nerf = torch.optim.lr_scheduler.LambdaLR(optimizer_nerf, lr_lambda)
    scheduler_poses = torch.optim.lr_scheduler.LambdaLR(optimizer_poses, lr_lambda)
    
    # Initialize LPIPS
    lpips_metric = LPIPSMetric(device=device) if device == "cuda" else None
    if lpips_metric and not lpips_metric.available:
        lpips_metric = None
    
    # Create pixel dataset and sampler
    print(f"\n{'─' * 70}")
    print(f"Creating pixel dataset...")
    
    pixel_dataset, pixel_sampler = create_pixel_dataset(train_data)
    pixel_sampler.batch_size = config.data.batch_size
    
    print(f"✓ Total pixels: {pixel_dataset.n_pixels:,}")
    print(f"✓ Batch size: {pixel_sampler.batch_size}")
    
    # Save experiment configuration
    exp_config = {
        "scene": config.data.scene_name,
        "experiment_name": exp_name,
        "init_mode": init_mode,
        "pose_optimization": {
            "learn_rotation": learn_rotation,
            "learn_translation": learn_translation,
            "pose_lr": pose_lr,
            "pose_opt_delay": pose_opt_delay,
            "rotation_reg_weight": rotation_reg_weight,
            "translation_reg_weight": translation_reg_weight,
        },
        "noise_config": {
            "rotation_noise_deg": noise_config.rotation_noise_deg if noise_config else 0,
            "translation_noise": noise_config.translation_noise if noise_config else 0,
            "translation_noise_pct": noise_config.translation_noise_pct if noise_config else 0,
            "seed": noise_config.seed if noise_config else None,
            "has_noise": noise_config.has_noise if noise_config else False,
        } if noise_config else None,
        "num_iterations": config.train.num_iterations,
        "batch_size": config.data.batch_size,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Training loop
    print(f"\n{'=' * 70}")
    print(f"Starting training for {config.train.num_iterations:,} iterations")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    best_psnr = 0.0
    
    val_img_indices = torch.arange(val_data.images.shape[0], device=device)
    
    for iteration in range(config.train.num_iterations):
        # Sample random batch of pixels
        pixel_batch = pixel_sampler.sample_batch()
        
        # Determine if we should optimize poses this iteration
        optimize_poses_now = iteration >= pose_opt_delay
        
        # Training step
        batch_start = time.time()
        metrics = train_step_with_poses(
            model_coarse=model_coarse,
            model_fine=model_fine,
            camera_params=camera_params,
            pixel_sampler=pixel_sampler,
            optimizer_nerf=optimizer_nerf,
            optimizer_poses=optimizer_poses if optimize_poses_now else None,
            pixel_batch=pixel_batch,
            render_config=config.render,
            optimize_poses=optimize_poses_now,
            rotation_reg_weight=rotation_reg_weight,
            translation_reg_weight=translation_reg_weight,
        )
        
        # Update learning rates
        scheduler_nerf.step()
        if optimize_poses_now:
            scheduler_poses.step()
        
        batch_time = time.time() - batch_start
        
        # Logging
        if iteration % config.train.log_every == 0:
            elapsed = time.time() - start_time
            
            pose_status = "✓ optimizing" if optimize_poses_now else "✗ frozen"
            
            print(
                f"[{iteration:7d}/{config.train.num_iterations}] "
                f"loss: {metrics['loss']:.5f} | "
                f"psnr: {metrics['psnr']:.2f} | "
                f"lr: {optimizer_nerf.param_groups[0]['lr']:.2e} | "
                f"poses: {pose_status} | "
                f"time: {elapsed/60:.1f}min"
            )
        
        # Validation and pose error tracking
        if iteration % config.train.val_every == 0 and iteration > 0:
            print(f"\n{'─' * 70}")
            print(f"Validation at iteration {iteration}")
            
            # Compute pose errors
            pose_errors = camera_params.compute_pose_errors(gt_train_poses)
            
            print(f"\nPose Refinement:")
            print(f"  Rotation error:    {pose_errors['rotation_error_mean']:.3f}° ± {pose_errors['rotation_error_std']:.3f}° "
                  f"(max: {pose_errors['rotation_error_max']:.3f}°)")
            print(f"  Translation error: {pose_errors['translation_error_mean']:.4f} ± {pose_errors['translation_error_std']:.4f} "
                  f"(max: {pose_errors['translation_error_max']:.4f})")
            
            # Validation metrics
            val_metrics = evaluate_with_poses(
                model_coarse=model_coarse,
                model_fine=model_fine,
                camera_params=camera_params,
                val_data=val_data,
                val_indices=val_img_indices,
                render_config=config.render,
                logger=logger,
                iteration=iteration,
                num_images=5,
                lpips_metric=lpips_metric,
            )
            
            logger.log_validation(val_metrics)
            
            print(f"\nRendering Quality:")
            print(f"  PSNR:  {val_metrics.psnr:.2f} dB")
            print(f"  SSIM:  {val_metrics.ssim:.4f}")
            if val_metrics.lpips is not None:
                print(f"  LPIPS: {val_metrics.lpips:.4f}")
            
            # Save checkpoint
            is_best = val_metrics.psnr > best_psnr
            if is_best:
                best_psnr = val_metrics.psnr
                print(f"\n  ★ NEW BEST PSNR!")
            
            save_checkpoint_with_poses(
                output_dir=output_dir,
                iteration=iteration,
                model_coarse=model_coarse,
                model_fine=model_fine,
                camera_params=camera_params,
                optimizer_nerf=optimizer_nerf,
                optimizer_poses=optimizer_poses,
                config=config,
                noise_config=noise_config,
                metrics={"psnr": val_metrics.psnr, "ssim": val_metrics.ssim},
                pose_errors=pose_errors,
                is_best=is_best,
            )
            
            print(f"{'─' * 70}\n")
        
        elif iteration % config.train.save_every == 0 and iteration > 0:
            pose_errors = camera_params.compute_pose_errors(gt_train_poses)
            save_checkpoint_with_poses(
                output_dir=output_dir,
                iteration=iteration,
                model_coarse=model_coarse,
                model_fine=model_fine,
                camera_params=camera_params,
                optimizer_nerf=optimizer_nerf,
                optimizer_poses=optimizer_poses,
                config=config,
                noise_config=noise_config,
                pose_errors=pose_errors,
            )
    
    # Final evaluation
    print(f"\n{'=' * 70}")
    print(f"FINAL EVALUATION")
    print(f"{'=' * 70}")
    
    final_pose_errors = camera_params.compute_pose_errors(gt_train_poses)
    
    print(f"\nFinal Pose Refinement:")
    print(f"  Rotation error:    {final_pose_errors['rotation_error_mean']:.3f}° ± {final_pose_errors['rotation_error_std']:.3f}° "
          f"(max: {final_pose_errors['rotation_error_max']:.3f}°)")
    print(f"  Translation error: {final_pose_errors['translation_error_mean']:.4f} ± {final_pose_errors['translation_error_std']:.4f} "
          f"(max: {final_pose_errors['translation_error_max']:.4f})")
    
    final_metrics = evaluate_with_poses(
        model_coarse=model_coarse,
        model_fine=model_fine,
        camera_params=camera_params,
        val_data=val_data,
        val_indices=val_img_indices,
        render_config=config.render,
        logger=logger,
        iteration=config.train.num_iterations,
        num_images=val_data.images.shape[0],
        lpips_metric=lpips_metric,
    )
    
    print(f"\nFinal Rendering Quality:")
    print(f"  PSNR:  {final_metrics.psnr:.2f} dB")
    print(f"  SSIM:  {final_metrics.ssim:.4f}")
    if final_metrics.lpips is not None:
        print(f"  LPIPS: {final_metrics.lpips:.4f}")
    
    # Save final checkpoint
    save_checkpoint_with_poses(
        output_dir=output_dir,
        iteration=config.train.num_iterations,
        model_coarse=model_coarse,
        model_fine=model_fine,
        camera_params=camera_params,
        optimizer_nerf=optimizer_nerf,
        optimizer_poses=optimizer_poses,
        config=config,
        noise_config=noise_config,
        metrics={"psnr": final_metrics.psnr, "ssim": final_metrics.ssim},
        pose_errors=final_pose_errors,
    )
    
    # Save final poses
    final_poses = camera_params.get_all_poses()
    torch.save({
        "initial_poses": camera_params.initial_poses.cpu(),
        "optimized_poses": final_poses.cpu(),
        "ground_truth_poses": gt_train_poses.cpu(),
        "pose_errors": final_pose_errors,
    }, output_dir / "final_poses.pt")
    
    logger.save_summary()
    logger.close()
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"✓ Training complete!")
    print(f"  Total time: {total_time / 3600:.2f} hours")
    print(f"  Best PSNR:  {best_psnr:.2f} dB")
    print(f"  Results:    {output_dir}")
    print(f"{'=' * 70}\n")


def main():
    """CLI entry point for pose optimization training."""
    parser = argparse.ArgumentParser(
        description="Train NeRF with joint camera pose optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with noisy initialization, optimize both rotation and translation
  python -m noisy_src.train_pose_opt --scene lego --init_mode noisy \\
         --rotation_noise 2.0 --translation_noise_pct 1.0
  
  # Train with clean initialization (should maintain performance)
  python -m noisy_src.train_pose_opt --scene lego --init_mode clean
  
  # Optimize only rotations
  python -m noisy_src.train_pose_opt --scene lego --init_mode noisy \\
         --rotation_noise 2.0 --no_learn_translation
  
  # Optimize only translations
  python -m noisy_src.train_pose_opt --scene lego --init_mode noisy \\
         --translation_noise_pct 1.0 --no_learn_rotation
        """
    )
    
    # Data args
    parser.add_argument("--scene", type=str, default="lego",
                        help="Scene name")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data directory")
    parser.add_argument("--img_scale", type=float, default=0.5,
                        help="Image scale factor")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size (rays)")
    parser.add_argument("--num_iters", type=int, default=50000,
                        help="Number of iterations")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="NeRF learning rate")
    
    # Pose optimization args
    pose_group = parser.add_argument_group("Pose Optimization")
    pose_group.add_argument("--init_mode", type=str, default="noisy",
                            choices=["clean", "noisy"],
                            help="Pose initialization mode")
    pose_group.add_argument("--pose_lr", type=float, default=1e-4,
                            help="Learning rate for poses")
    pose_group.add_argument("--pose_opt_delay", type=int, default=1000,
                            help="Start pose optimization after N iterations")
    pose_group.add_argument("--no_learn_rotation", action="store_true",
                            help="Don't optimize rotations")
    pose_group.add_argument("--no_learn_translation", action="store_true",
                            help="Don't optimize translations")
    pose_group.add_argument("--rotation_reg_weight", type=float, default=0.01,
                            help="L2 regularization weight for rotation deltas (prevents drift)")
    pose_group.add_argument("--translation_reg_weight", type=float, default=0.001,
                            help="L2 regularization weight for translation deltas (prevents drift)")
    
    # Noise args (for initialization)
    noise_group = parser.add_argument_group("Noise Configuration")
    noise_group.add_argument("--rotation_noise", type=float, default=0.0,
                             help="Rotation noise std in degrees")
    noise_group.add_argument("--translation_noise", type=float, default=0.0,
                             help="Translation noise std")
    noise_group.add_argument("--translation_noise_pct", type=float, default=0.0,
                             help="Translation noise std as %%")
    noise_group.add_argument("--noise_seed", type=int, default=None,
                             help="Noise random seed")
    
    # Model args
    parser.add_argument("--no_hierarchical", action="store_true",
                        help="Disable hierarchical sampling")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Coarse samples")
    parser.add_argument("--num_samples_fine", type=int, default=128,
                        help="Fine samples")
    
    # Logging args
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=2500)
    parser.add_argument("--save_every", type=int, default=10000)
    
    # Other args
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Build noise config
    noise_config = None
    if args.rotation_noise > 0 or args.translation_noise > 0 or args.translation_noise_pct > 0:
        noise_config = NoiseConfig(
            rotation_noise_deg=args.rotation_noise,
            translation_noise=args.translation_noise,
            translation_noise_pct=args.translation_noise_pct,
            seed=args.noise_seed,
        )
    
    # Build config
    config = NeRFConfig(
        model=ModelConfig(),
        render=RenderConfig(
            use_hierarchical=not args.no_hierarchical,
            num_samples=args.num_samples,
            num_samples_fine=args.num_samples_fine,
        ),
        data=DataConfig(
            scene_name=args.scene,
            data_root=Path(args.data_root) if args.data_root else None,
            img_scale=args.img_scale,
            batch_size=args.batch_size,
        ),
        train=TrainConfig(
            lr=args.lr,
            num_iterations=args.num_iters,
            output_dir=Path(args.output_dir),
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
        ),
    )
    
    train_with_pose_optimization(
        config=config,
        noise_config=noise_config,
        init_mode=args.init_mode,
        pose_lr=args.pose_lr,
        pose_opt_delay=args.pose_opt_delay,
        learn_rotation=not args.no_learn_rotation,
        learn_translation=not args.no_learn_translation,
        rotation_reg_weight=args.rotation_reg_weight,
        translation_reg_weight=args.translation_reg_weight,
    )


if __name__ == "__main__":
    main()

