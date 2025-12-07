"""
Inference script for trained NeRF models.

Load a trained checkpoint and:
- Render test set images (with optional camera noise)
- Render novel views (spiral path)
- Compute metrics on test set
- Run experiments with noise
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from .config import NeRFConfig, ModelConfig, RenderConfig, DataConfig, TrainConfig
from .model import NeRF, create_nerf
from .rendering import NeRFRenderer
from .data import load_blender_data
from .rays import get_ray_directions, get_rays
from .metrics import compute_psnr, compute_ssim, compute_mse, LPIPSMetric, compute_all_metrics
from .noise import NoiseConfig, add_noise_to_pose, add_noise_to_poses, compute_pose_error, set_noise_seed


def load_checkpoint(
    checkpoint_path: Path,
    device: str = "cuda",
) -> tuple:
    """
    Load a trained NeRF checkpoint.
    
    Returns
    -------
    renderer : NeRFRenderer
        The renderer with loaded weights.
    config : dict
        The training configuration.
    iteration : int
        The iteration at which the checkpoint was saved.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    cfg = checkpoint.get("config", {})
    model_cfg = ModelConfig(**cfg.get("model", {}))
    render_cfg = RenderConfig(**cfg.get("render", {}))
    
    # Create models
    model_coarse = NeRF(model_cfg).to(device)
    model_coarse.load_state_dict(checkpoint["model_coarse"])
    model_coarse.eval()
    
    model_fine = None
    if "model_fine" in checkpoint:
        model_fine = NeRF(model_cfg).to(device)
        model_fine.load_state_dict(checkpoint["model_fine"])
        model_fine.eval()
    
    # Create renderer
    renderer = NeRFRenderer(model_coarse, model_fine, render_cfg)
    
    iteration = checkpoint.get("iteration", 0)
    
    return renderer, cfg, iteration


@torch.no_grad()
def render_image(
    renderer: NeRFRenderer,
    pose: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    chunk_size: int = 1024 * 4,  # 4K rays default
) -> Dict[str, torch.Tensor]:
    """Render a single image from a camera pose."""
    device = pose.device
    
    directions = get_ray_directions(H, W, focal).to(device)
    rays_o, rays_d = get_rays(directions, pose)
    
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    outputs = renderer(rays_o, rays_d, chunk_size=chunk_size, is_train=False)
    
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


def save_image(img: torch.Tensor, path: Path):
    """Save tensor image to disk."""
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


def depth_to_colormap(depth: torch.Tensor) -> torch.Tensor:
    """Convert depth map to colormap visualization."""
    depth = depth.cpu()
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    
    # Turbo-like colormap
    r = torch.clamp(4 * depth_norm - 1.5, 0, 1)
    g = torch.clamp(2 - 4 * torch.abs(depth_norm - 0.5), 0, 1)
    b = torch.clamp(1.5 - 4 * depth_norm, 0, 1)
    
    return torch.stack([r, g, b], dim=-1)


def generate_output_folder_name(
    mode: str,
    noise_config: NoiseConfig,
    scene: str,
) -> str:
    """
    Generate informative output folder name.
    
    Format: {mode}_{scene}_{noise_desc}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_desc = str(noise_config)  # "clean" or "rot1.0deg_trans0.010"
    
    return f"{mode}_{scene}_{noise_desc}_{timestamp}"


@torch.no_grad()
def evaluate_test_set(
    renderer: NeRFRenderer,
    test_data,
    output_dir: Path,
    noise_config: NoiseConfig,
    device: str = "cuda",
    chunk_size: int = 1024 * 4,
) -> Dict[str, float]:
    """
    Evaluate on test set with optional camera noise.
    
    Saves rendered images and comprehensive metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set noise seed for reproducibility
    if noise_config.seed is not None:
        set_noise_seed(noise_config.seed)
    
    all_psnr = []
    all_ssim = []
    all_mse = []
    
    # Try to initialize LPIPS
    lpips_metric = LPIPSMetric(device=device)
    all_lpips = [] if lpips_metric.available else None
    
    n_images = test_data.images.shape[0]
    print(f"Evaluating {n_images} test images...")
    
    if noise_config.has_noise:
        print(f"  Rotation noise: {noise_config.rotation_noise_deg}°")
        print(f"  Translation noise: {noise_config.translation_noise}")
        if noise_config.seed is not None:
            print(f"  Noise seed: {noise_config.seed}")
    else:
        print("  No noise (clean evaluation)")
    
    per_image_metrics = []
    all_noise_info = []
    
    for i in range(n_images):
        original_pose = test_data.poses[i]
        target = test_data.images[i]
        
        # Apply noise to pose if configured
        if noise_config.has_noise:
            pose, noise_info = add_noise_to_pose(
                original_pose,
                rotation_noise_deg=noise_config.rotation_noise_deg,
                translation_noise=noise_config.translation_noise,
            )
            pose_error = compute_pose_error(original_pose, pose)
            noise_info.update(pose_error)
            all_noise_info.append(noise_info)
        else:
            pose = original_pose
            noise_info = {}
        
        # Render
        start = time.time()
        output = render_image(
            renderer, pose,
            test_data.H, test_data.W, test_data.focal,
            chunk_size=chunk_size
        )
        render_time = time.time() - start
        
        pred = output["rgb"]
        
        # Compute metrics
        mse = compute_mse(pred, target).item()
        psnr = compute_psnr(pred, target).item()
        ssim = compute_ssim(pred, target).item()
        
        img_metrics = {
            "image": i,
            "psnr": psnr,
            "ssim": ssim,
            "mse": mse,
            "render_time": render_time,
        }
        
        # Add noise info to metrics
        if noise_info:
            img_metrics.update({f"noise_{k}": v for k, v in noise_info.items()})
        
        all_mse.append(mse)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        
        if all_lpips is not None:
            lpips_val = lpips_metric(pred, target)
            if lpips_val is not None:
                all_lpips.append(lpips_val.item())
                img_metrics["lpips"] = lpips_val.item()
        
        per_image_metrics.append(img_metrics)
        
        # Save images
        save_image(pred, output_dir / f"pred_{i:03d}.png")
        save_image(target, output_dir / f"gt_{i:03d}.png")
        
        # Side-by-side comparison
        comparison = torch.cat([target, pred], dim=1)
        save_image(comparison, output_dir / f"comparison_{i:03d}.png")
        
        # Depth map
        depth_vis = depth_to_colormap(output["depth"])
        save_image(depth_vis, output_dir / f"depth_{i:03d}.png")
        
        # Progress output
        noise_str = ""
        if noise_config.has_noise:
            noise_str = f", rot_err={noise_info.get('rotation_error_deg', 0):.2f}°"
        print(f"  Image {i+1}/{n_images}: PSNR={psnr:.2f}, SSIM={ssim:.4f}{noise_str}, time={render_time:.2f}s")
    
    # Aggregate metrics
    avg_metrics = {
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "n_images": n_images,
    }
    
    if all_lpips:
        avg_metrics["lpips_mean"] = float(np.mean(all_lpips))
        avg_metrics["lpips_std"] = float(np.std(all_lpips))
    
    # Add noise statistics
    if noise_config.has_noise and all_noise_info:
        rot_errors = [n.get("rotation_error_deg", 0) for n in all_noise_info]
        trans_errors = [n.get("translation_error", 0) for n in all_noise_info]
        
        avg_metrics["noise_config"] = {
            "rotation_noise_deg": noise_config.rotation_noise_deg,
            "translation_noise": noise_config.translation_noise,
            "seed": noise_config.seed,
        }
        avg_metrics["actual_noise"] = {
            "rotation_error_mean_deg": float(np.mean(rot_errors)),
            "rotation_error_std_deg": float(np.std(rot_errors)),
            "translation_error_mean": float(np.mean(trans_errors)),
            "translation_error_std": float(np.std(trans_errors)),
        }
    
    # Save per-image metrics
    metrics_path = output_dir / "per_image_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(per_image_metrics, f, indent=2)
    
    # Save summary
    summary_path = output_dir / "test_metrics.json"
    with open(summary_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Save experiment config
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "mode": "test",
            "noise_config": {
                "rotation_noise_deg": noise_config.rotation_noise_deg,
                "translation_noise": noise_config.translation_noise,
                "seed": noise_config.seed,
            },
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir),
        }, f, indent=2)
    
    return avg_metrics


def create_spiral_poses(
    n_frames: int = 120,
    radius: float = 0.5,
    height: float = 0.0,
    n_rotations: float = 2.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create camera poses along a spiral path for video rendering.
    """
    poses = []
    
    for i in range(n_frames):
        t = i / n_frames
        theta = 2 * np.pi * n_rotations * t
        
        # Camera position on a circle
        base_dist = 4.0
        position = np.array([
            base_dist * np.cos(theta),
            base_dist * np.sin(theta),
            height,
        ])
        
        # Look at origin
        forward = -position / np.linalg.norm(position)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Construct camera-to-world matrix
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = position
        
        poses.append(torch.from_numpy(c2w))
    
    return torch.stack(poses, dim=0).to(device)


@torch.no_grad()
def render_video(
    renderer: NeRFRenderer,
    poses: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    output_dir: Path,
    noise_config: NoiseConfig,
    fps: int = 30,
    chunk_size: int = 1024 * 4,
) -> Path:
    """
    Render a video from a sequence of poses with optional noise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Set noise seed
    if noise_config.seed is not None:
        set_noise_seed(noise_config.seed)
    
    n_frames = poses.shape[0]
    print(f"Rendering {n_frames} frames...")
    
    if noise_config.has_noise:
        print(f"  With noise: rot={noise_config.rotation_noise_deg}°, trans={noise_config.translation_noise}")
    
    for i in range(n_frames):
        pose = poses[i]
        
        # Apply noise if configured
        if noise_config.has_noise:
            pose, _ = add_noise_to_pose(
                pose,
                rotation_noise_deg=noise_config.rotation_noise_deg,
                translation_noise=noise_config.translation_noise,
            )
        
        output = render_image(renderer, pose, H, W, focal, chunk_size=chunk_size)
        save_image(output["rgb"], frames_dir / f"frame_{i:04d}.png")
        
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i+1}/{n_frames} frames")
    
    # Save config
    config_path = output_dir / "video_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "n_frames": n_frames,
            "fps": fps,
            "noise_config": {
                "rotation_noise_deg": noise_config.rotation_noise_deg,
                "translation_noise": noise_config.translation_noise,
                "seed": noise_config.seed,
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    # Try to create video with ffmpeg
    video_path = output_dir / "video.mp4"
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Could not create video (ffmpeg required): {e}")
        print(f"Frames saved to {frames_dir}")
        video_path = frames_dir
    
    return video_path


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(
        description="NeRF Inference with optional camera noise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean evaluation on test set
  python -m noisy_src.inference checkpoint.pt --mode test

  # Evaluation with rotation noise (1 degree std)
  python -m noisy_src.inference checkpoint.pt --mode test --rotation_noise 1.0

  # Evaluation with both rotation and translation noise
  python -m noisy_src.inference checkpoint.pt --mode test --rotation_noise 2.0 --translation_noise 0.05

  # Reproducible noise experiment
  python -m noisy_src.inference checkpoint.pt --mode test --rotation_noise 1.0 --noise_seed 42

  # Render video with noise
  python -m noisy_src.inference checkpoint.pt --mode video --rotation_noise 0.5
        """
    )
    
    parser.add_argument("checkpoint", type=Path,
                        help="Path to checkpoint file")
    parser.add_argument("--mode", type=str, default="test",
                        choices=["test", "video", "single"],
                        help="Inference mode")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (auto-detected from config if not provided)")
    parser.add_argument("--data_root", type=Path, default=None,
                        help="Data root directory")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (auto-generated if not provided)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    
    # Noise options
    noise_group = parser.add_argument_group("Noise Options")
    noise_group.add_argument("--rotation_noise", type=float, default=0.0,
                             help="Rotation noise std in degrees (default: 0 = no noise)")
    noise_group.add_argument("--translation_noise", type=float, default=0.0,
                             help="Translation noise std in scene units (default: 0 = no noise)")
    noise_group.add_argument("--noise_seed", type=int, default=None,
                             help="Random seed for reproducible noise")
    
    # Video options
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument("--n_frames", type=int, default=120,
                             help="Number of frames for video")
    video_group.add_argument("--fps", type=int, default=30,
                             help="Video FPS")
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument("--chunk_size", type=int, default=1024*4,
                            help="Rays per chunk (default: 4096)")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Create noise config
    noise_config = NoiseConfig(
        rotation_noise_deg=args.rotation_noise,
        translation_noise=args.translation_noise,
        seed=args.noise_seed,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    renderer, config, iteration = load_checkpoint(args.checkpoint, device)
    print(f"Loaded model from iteration {iteration}")
    
    # Get scene name
    scene = args.scene
    if scene is None:
        scene = config.get("data", {}).get("scene_name", "lego")
    print(f"Scene: {scene}")
    
    # Setup output directory with informative name
    if args.output_dir is None:
        folder_name = generate_output_folder_name(args.mode, noise_config, scene)
        output_dir = args.checkpoint.parent / folder_name
    else:
        output_dir = args.output_dir
    
    print(f"Output directory: {output_dir}")
    
    # Load data
    data_root = args.data_root
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
    
    img_scale = config.get("data", {}).get("img_scale", 0.5)
    
    if args.mode == "test":
        # Evaluate on test set
        print("\nLoading test data...")
        test_data = load_blender_data(
            data_root=data_root,
            scene_name=scene,
            split="test",
            img_scale=img_scale,
            device=device,
        )
        
        print(f"\n{'=' * 60}")
        print(f"Test Set Evaluation")
        if noise_config.has_noise:
            print(f"  Rotation noise:    {noise_config.rotation_noise_deg}° (std)")
            print(f"  Translation noise: {noise_config.translation_noise} (std)")
            if noise_config.seed is not None:
                print(f"  Noise seed:        {noise_config.seed}")
        else:
            print("  Mode: Clean (no noise)")
        print(f"{'=' * 60}\n")
        
        metrics = evaluate_test_set(renderer, test_data, output_dir, noise_config, device, args.chunk_size)
        
        print(f"\n{'=' * 60}")
        print("Test Set Results:")
        print(f"  PSNR:  {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"  SSIM:  {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        if 'lpips_mean' in metrics:
            print(f"  LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
        
        if noise_config.has_noise and 'actual_noise' in metrics:
            actual = metrics['actual_noise']
            print(f"\nActual Noise Applied:")
            print(f"  Rotation error:    {actual['rotation_error_mean_deg']:.3f} ± {actual['rotation_error_std_deg']:.3f}°")
            print(f"  Translation error: {actual['translation_error_mean']:.4f} ± {actual['translation_error_std']:.4f}")
        
        print(f"{'=' * 60}")
        print(f"Results saved to: {output_dir}")
    
    elif args.mode == "video":
        # Render spiral video
        print("\nLoading data for camera parameters...")
        data = load_blender_data(
            data_root=data_root,
            scene_name=scene,
            split="train",
            img_scale=img_scale,
            device=device,
        )
        
        print("\nGenerating spiral poses...")
        poses = create_spiral_poses(
            n_frames=args.n_frames,
            device=device,
        )
        
        print("\nRendering video...")
        video_path = render_video(
            renderer, poses,
            data.H, data.W, data.focal,
            output_dir, noise_config, args.fps,
            chunk_size=args.chunk_size
        )
        print(f"Video saved to: {video_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
