"""
Training script for NeRF with comprehensive logging.

Supports:
- Clean baseline training
- Training with noisy camera poses (for robustness research)
- Comprehensive metrics and logging
- Automatic output folder naming with noise info
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .config import NeRFConfig, ModelConfig, RenderConfig, DataConfig, TrainConfig
from .model import NeRF, create_nerf
from .rendering import NeRFRenderer
from .data import create_data_loaders, load_blender_data, RayDataset, RaySampler
from .rays import get_ray_directions, get_rays
from .metrics import compute_psnr, compute_ssim, compute_mse, compute_all_metrics, LPIPSMetric
from .logger import ExperimentLogger, TrainingMetrics, ValidationMetrics
from .noise import NoiseConfig


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_experiment_name(
    scene: str,
    noise_config: Optional[NoiseConfig],
    base_name: str = "",
) -> str:
    """
    Generate informative experiment name including noise info and timestamp.
    
    Format: {scene}_{noise_desc}_{timestamp} or {scene}_clean_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if noise_config is not None and noise_config.has_noise:
        noise_desc = str(noise_config)  # "rot1.0deg" or "rot1.0deg_trans0.010"
    else:
        noise_desc = "clean"
    
    if base_name:
        return f"{scene}_{base_name}_{noise_desc}_{timestamp}"
    else:
        return f"{scene}_{noise_desc}_{timestamp}"


def train_step(
    renderer: NeRFRenderer,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Perform a single training step.
    
    Returns detailed metrics including separate coarse/fine losses.
    """
    optimizer.zero_grad()
    
    rays_o = batch["rays_o"]
    rays_d = batch["rays_d"]
    target_rgb = batch["target_rgb"]
    
    # Render rays
    outputs = renderer(rays_o, rays_d, is_train=True)
    
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
    
    metrics["loss"] = loss.item()
    
    # Backprop
    loss.backward()
    
    # Gradient clipping (optional but helps stability)
    torch.nn.utils.clip_grad_norm_(renderer.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return metrics


@torch.no_grad()
def render_image(
    renderer: NeRFRenderer,
    pose: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    chunk_size: int = 1024 * 4,
) -> Dict[str, torch.Tensor]:
    """
    Render a full image from a camera pose.
    
    Returns RGB, depth, and opacity maps.
    """
    device = pose.device
    
    # Generate rays
    directions = get_ray_directions(H, W, focal).to(device)
    rays_o, rays_d = get_rays(directions, pose)
    
    # Flatten
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # Render
    outputs = renderer(rays_o, rays_d, chunk_size=chunk_size, is_train=False)
    
    # Reshape to image
    result = {}
    if "rgb_fine" in outputs:
        result["rgb"] = outputs["rgb_fine"].reshape(H, W, 3)
        result["depth"] = outputs["depth_fine"].reshape(H, W)
        result["acc"] = outputs["acc_fine"].reshape(H, W)
    else:
        result["rgb"] = outputs["rgb_coarse"].reshape(H, W, 3)
        result["depth"] = outputs["depth_coarse"].reshape(H, W)
        result["acc"] = outputs["acc_coarse"].reshape(H, W)
    
    return result


@torch.no_grad()
def evaluate(
    renderer: NeRFRenderer,
    val_data,
    logger: ExperimentLogger,
    iteration: int,
    num_images: int = 5,
    lpips_metric: Optional[LPIPSMetric] = None,
) -> ValidationMetrics:
    """
    Comprehensive evaluation on validation set.
    
    Computes PSNR, SSIM, and optionally LPIPS for each image.
    """
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_mse = []
    
    indices = list(range(min(num_images, val_data.images.shape[0])))
    
    for i, idx in enumerate(indices):
        pose = val_data.poses[idx]
        target = val_data.images[idx]
        
        output = render_image(
            renderer=renderer,
            pose=pose,
            H=val_data.H,
            W=val_data.W,
            focal=val_data.focal,
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
    
    # Aggregate metrics
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


def save_checkpoint(
    output_dir: Path,
    iteration: int,
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    optimizer: torch.optim.Optimizer,
    config: NeRFConfig,
    noise_config: Optional[NoiseConfig] = None,
    metrics: Optional[Dict] = None,
    is_best: bool = False,
):
    """Save a training checkpoint with optional metrics."""
    checkpoint = {
        "iteration": iteration,
        "model_coarse": model_coarse.state_dict(),
        "optimizer": optimizer.state_dict(),
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
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if noise_config is not None:
        checkpoint["noise_config"] = {
            "rotation_noise_deg": noise_config.rotation_noise_deg,
            "translation_noise": noise_config.translation_noise,
            "translation_noise_pct": noise_config.translation_noise_pct,
            "seed": noise_config.seed,
        }
    
    checkpoint_path = output_dir / f"checkpoint_{iteration:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save best model separately
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved BEST checkpoint to {best_path}")
    else:
        print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model_coarse: NeRF,
    model_fine: Optional[NeRF],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load a training checkpoint. Returns the iteration number."""
    checkpoint = torch.load(checkpoint_path)
    
    model_coarse.load_state_dict(checkpoint["model_coarse"])
    if model_fine is not None and "model_fine" in checkpoint:
        model_fine.load_state_dict(checkpoint["model_fine"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint.get("iteration", 0)


def train(config: NeRFConfig, noise_config: Optional[NoiseConfig] = None):
    """
    Main training function with comprehensive logging.
    
    Parameters
    ----------
    config : NeRFConfig
        Training configuration.
    noise_config : NoiseConfig, optional
        Noise to apply to training poses.
    """
    # Setup
    set_seed(config.train.seed)
    device = config.train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Generate experiment name with noise info
    exp_name = config.train.experiment_name
    if exp_name == "auto" or exp_name == "":
        exp_name = generate_experiment_name(
            config.data.scene_name,
            noise_config,
        )
    
    # Create output directory
    output_dir = config.train.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = ExperimentLogger(
        output_dir=output_dir,
        experiment_name=exp_name,
        use_tensorboard=True,
    )
    
    # Log configuration
    logger.log_config(config)
    
    print(f"{'=' * 60}")
    print(f"NeRF Training")
    print(f"{'=' * 60}")
    print(f"Scene: {config.data.scene_name}")
    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    if noise_config is not None and noise_config.has_noise:
        print(f"\n*** TRAINING WITH NOISY POSES ***")
        print(f"  Rotation noise:    {noise_config.rotation_noise_deg}° (std)")
        if noise_config.translation_noise_pct > 0:
            print(f"  Translation noise: {noise_config.translation_noise_pct}% of camera distance (std)")
        elif noise_config.translation_noise > 0:
            print(f"  Translation noise: {noise_config.translation_noise} (std)")
        if noise_config.seed is not None:
            print(f"  Noise seed:        {noise_config.seed}")
    else:
        print(f"\nTraining mode: CLEAN (no pose noise)")
    
    # Load data with optional noise
    print(f"\nLoading scene: {config.data.scene_name}")
    train_sampler, train_data, val_data = create_data_loaders(
        config=config.data,
        device=device,
        noise_config=noise_config,
    )
    print(f"Training rays: {train_sampler.n_rays:,}")
    print(f"Training images: {train_data.images.shape[0]}")
    print(f"Validation images: {val_data.images.shape[0]}")
    print(f"Image resolution: {train_data.H} x {train_data.W}")
    print(f"Focal length: {train_data.focal:.2f}")
    
    # Create models
    print("\nCreating models...")
    model_coarse, model_fine = create_nerf(config.model)
    model_coarse = model_coarse.to(device)
    if config.render.use_hierarchical:
        model_fine = model_fine.to(device)
    else:
        model_fine = None
    
    # Log model info
    logger.log_model_info(model_coarse, "model_coarse")
    if model_fine:
        logger.log_model_info(model_fine, "model_fine")
    
    # Create renderer
    renderer = NeRFRenderer(model_coarse, model_fine, config.render)
    
    # Create optimizer
    params = list(model_coarse.parameters())
    if model_fine is not None:
        params += list(model_fine.parameters())
    
    optimizer = Adam(params, lr=config.train.lr)
    
    # Learning rate decay
    decay_rate = 0.1
    decay_steps = config.train.lr_decay * 1000
    
    def lr_lambda(step):
        return decay_rate ** (step / decay_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize LPIPS metric (optional)
    lpips_metric = LPIPSMetric(device=device) if device == "cuda" else None
    if lpips_metric and not lpips_metric.available:
        print("LPIPS not available. Install with: pip install lpips")
        lpips_metric = None
    
    # Save experiment config including noise
    config_path = output_dir / "experiment_config.json"
    exp_config = {
        "scene": config.data.scene_name,
        "experiment_name": exp_name,
        "noise_config": {
            "rotation_noise_deg": noise_config.rotation_noise_deg if noise_config else 0,
            "translation_noise": noise_config.translation_noise if noise_config else 0,
            "translation_noise_pct": noise_config.translation_noise_pct if noise_config else 0,
            "seed": noise_config.seed if noise_config else None,
            "has_noise": noise_config.has_noise if noise_config else False,
        },
        "num_iterations": config.train.num_iterations,
        "batch_size": config.data.batch_size,
        "img_scale": config.data.img_scale,
        "timestamp": datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Starting training for {config.train.num_iterations:,} iterations")
    print(f"{'=' * 60}\n")
    
    start_time = time.time()
    iteration = 0
    iter_time = time.time()
    
    best_psnr = 0.0
    
    while iteration < config.train.num_iterations:
        for batch in train_sampler:
            if iteration >= config.train.num_iterations:
                break
            
            batch_start = time.time()
            
            # Training step
            step_metrics = train_step(renderer, optimizer, batch)
            
            # Update learning rate
            scheduler.step()
            
            # Compute timing
            batch_time = time.time() - batch_start
            rays_per_sec = config.data.batch_size / batch_time
            
            # Create training metrics
            train_metrics = TrainingMetrics(
                iteration=iteration,
                loss=step_metrics["loss"],
                loss_coarse=step_metrics["loss_coarse"],
                loss_fine=step_metrics.get("loss_fine"),
                psnr=step_metrics["psnr"],
                learning_rate=optimizer.param_groups[0]["lr"],
                time_per_iter=batch_time,
                rays_per_sec=rays_per_sec,
            )
            
            # Log training metrics
            logger.log_training(train_metrics)
            
            # Console logging
            if iteration % config.train.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{iteration:7d}/{config.train.num_iterations}] "
                    f"loss: {step_metrics['loss']:.5f} | "
                    f"psnr: {step_metrics['psnr']:.2f} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"rays/s: {rays_per_sec:.0f} | "
                    f"time: {elapsed/60:.1f}min"
                )
            
            # Validation
            if iteration % config.train.val_every == 0 and iteration > 0:
                print(f"\n{'─' * 40}")
                print(f"Validation at iteration {iteration}")
                
                val_metrics = evaluate(
                    renderer, val_data, logger, iteration,
                    num_images=5, lpips_metric=lpips_metric
                )
                
                logger.log_validation(val_metrics)
                
                print(f"  PSNR:  {val_metrics.psnr:.2f} dB")
                print(f"  SSIM:  {val_metrics.ssim:.4f}")
                if val_metrics.lpips is not None:
                    print(f"  LPIPS: {val_metrics.lpips:.4f}")
                
                # Track best model
                is_best = val_metrics.psnr > best_psnr
                if is_best:
                    best_psnr = val_metrics.psnr
                    print(f"  *** New best PSNR! ***")
                
                save_checkpoint(
                    output_dir, iteration,
                    model_coarse, model_fine,
                    optimizer, config, noise_config,
                    metrics={"psnr": val_metrics.psnr, "ssim": val_metrics.ssim},
                    is_best=is_best,
                )
                
                print(f"{'─' * 40}\n")
            
            # Regular checkpointing
            elif iteration % config.train.save_every == 0 and iteration > 0:
                save_checkpoint(
                    output_dir, iteration,
                    model_coarse, model_fine,
                    optimizer, config, noise_config,
                )
            
            iteration += 1
    
    # Final checkpoint
    save_checkpoint(
        output_dir, iteration,
        model_coarse, model_fine,
        optimizer, config, noise_config,
    )
    
    # Final comprehensive evaluation
    print(f"\n{'=' * 60}")
    print("Final Evaluation")
    print(f"{'=' * 60}")
    
    final_metrics = evaluate(
        renderer, val_data, logger, iteration,
        num_images=val_data.images.shape[0],  # All validation images
        lpips_metric=lpips_metric
    )
    
    logger.log_validation(final_metrics)
    
    print(f"\nFinal Results:")
    print(f"  PSNR:  {final_metrics.psnr:.2f} dB")
    print(f"  SSIM:  {final_metrics.ssim:.4f}")
    if final_metrics.lpips is not None:
        print(f"  LPIPS: {final_metrics.lpips:.4f}")
    print(f"\n  Per-image PSNR: {final_metrics.per_image_psnr}")
    
    # Save experiment summary
    logger.save_summary()
    logger.close()
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    print(f"Results saved to: {output_dir}")
    if noise_config and noise_config.has_noise:
        trans_str = f"{noise_config.translation_noise_pct}%" if noise_config.translation_noise_pct > 0 else str(noise_config.translation_noise)
        print(f"Training noise: rot={noise_config.rotation_noise_deg}°, trans={trans_str}")
    print(f"{'=' * 60}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train NeRF with optional pose noise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean baseline training
  python -m noisy_src.train --scene lego --num_iters 50000

  # Training with rotation noise (simulates bad camera calibration)
  python -m noisy_src.train --scene lego --rotation_noise 2.0 --num_iters 50000

  # Training with percentage-based translation noise
  python -m noisy_src.train --scene lego --translation_noise_pct 1.0 --num_iters 50000

  # Training with both rotation and translation noise
  python -m noisy_src.train --scene lego --rotation_noise 2.0 --translation_noise_pct 2.0

  # Reproducible noisy training
  python -m noisy_src.train --scene lego --rotation_noise 2.0 --translation_noise_pct 1.0 --noise_seed 42
        """
    )
    
    # Data args
    parser.add_argument("--scene", type=str, default="lego",
                        help="Scene name (e.g., lego, chair, drums)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data directory")
    parser.add_argument("--img_scale", type=float, default=0.5,
                        help="Image scale factor (0.5 = half resolution)")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size (number of rays)")
    parser.add_argument("--num_iters", type=int, default=200000,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    
    # Model args
    parser.add_argument("--no_hierarchical", action="store_true",
                        help="Disable hierarchical sampling (no fine network)")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="Number of coarse samples")
    parser.add_argument("--num_samples_fine", type=int, default=128,
                        help="Number of fine samples")
    
    # Noise args (NEW!)
    noise_group = parser.add_argument_group("Pose Noise Options")
    noise_group.add_argument("--rotation_noise", type=float, default=0.0,
                             help="Rotation noise std in degrees (default: 0 = clean)")
    noise_group.add_argument("--translation_noise", type=float, default=0.0,
                             help="Translation noise std in scene units (default: 0 = clean)")
    noise_group.add_argument("--translation_noise_pct", type=float, default=0.0,
                             help="Translation noise std as %% of camera distance (default: 0 = clean)")
    noise_group.add_argument("--noise_seed", type=int, default=None,
                             help="Random seed for noise (for reproducibility)")
    
    # Logging args
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log training metrics every N iterations")
    parser.add_argument("--val_every", type=int, default=5000,
                        help="Run validation every N iterations")
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Save checkpoint every N iterations")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--exp_name", type=str, default="auto",
                        help="Experiment name (default: auto-generated with noise info)")
    
    # Other args
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
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
            experiment_name=args.exp_name,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
        ),
    )
    
    train(config, noise_config)


if __name__ == "__main__":
    main()
