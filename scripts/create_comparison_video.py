"""
Create side-by-side comparison video for NeRF experiments.

Renders three videos:
1. Clean baseline
2. Fixed noisy poses (5° + 5%)
3. Optimized poses (5° + 5%)

Then stitches them together horizontally for comparison.
"""

import argparse
import subprocess
from pathlib import Path
import sys
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noisy_src.inference import load_checkpoint, create_spiral_poses, render_image
from noisy_src.data import load_blender_data
from noisy_src.noise import NoiseConfig, add_noise_to_pose, set_noise_seed
from PIL import Image
import numpy as np


def save_image(img: torch.Tensor, path: Path):
    """Save tensor image to disk."""
    img_np = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(path)


@torch.no_grad()
def render_video_frames(
    checkpoint_path: Path,
    output_dir: Path,
    noise_config: NoiseConfig,
    n_frames: int = 120,
    device: str = "cuda",
    scene: str = "lego",
    data_root: Path = None,
    img_scale: float = 0.5,
):
    """
    Render video frames from a checkpoint with optional noise.
    
    Returns frames directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRendering from: {checkpoint_path.name}")
    
    # Load checkpoint
    renderer, config, iteration = load_checkpoint(checkpoint_path, device)
    print(f"  Loaded iteration: {iteration}")
    
    # Load data for camera parameters
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
    
    data = load_blender_data(
        data_root=data_root,
        scene_name=scene,
        split="train",
        img_scale=img_scale,
        device=device,
    )
    
    # Generate spiral poses
    print(f"  Generating {n_frames} spiral poses...")
    poses = create_spiral_poses(n_frames=n_frames, device=device)
    
    # Set noise seed for reproducibility
    if noise_config.seed is not None:
        set_noise_seed(noise_config.seed)
    
    print(f"  Rendering frames...")
    if noise_config.has_noise:
        print(f"    With noise: rot={noise_config.rotation_noise_deg}°, trans={noise_config.translation_noise_pct}%")
    
    for i in range(n_frames):
        pose = poses[i]
        
        # Apply noise if configured
        if noise_config.has_noise:
            # Get camera distance for percentage-based translation noise
            camera_pos = pose[:3, 3]
            camera_distance = torch.norm(camera_pos).item()
            trans_std = noise_config.get_translation_std(camera_distance)
            
            pose, _ = add_noise_to_pose(
                pose,
                rotation_noise_deg=noise_config.rotation_noise_deg,
                translation_noise=trans_std,
            )
        
        # Render
        output = render_image(renderer, pose, data.H, data.W, data.focal, chunk_size=1024*4)
        save_image(output["rgb"], output_dir / f"frame_{i:04d}.png")
        
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_frames} frames rendered")
    
    print(f"  ✓ Frames saved to: {output_dir}")
    return output_dir


def stitch_videos_horizontal(
    frames_dirs: list,
    labels: list,
    output_path: Path,
    fps: int = 30,
    add_labels: bool = True,
):
    """
    Stitch multiple video frame directories into a horizontal comparison video.
    
    Uses ffmpeg to create side-by-side comparison.
    """
    print(f"\n{'='*70}")
    print("Creating side-by-side comparison video...")
    print(f"{'='*70}")
    
    n_videos = len(frames_dirs)
    
    # Check all frame directories exist
    for i, frames_dir in enumerate(frames_dirs):
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        print(f"  Video {i+1}: {labels[i]}")
        print(f"    Frames: {frames_dir}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg filter for horizontal stacking
    # For 3 videos: [0:v][1:v][2:v]hstack=inputs=3
    input_args = []
    for frames_dir in frames_dirs:
        input_args.extend([
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
        ])
    
    # Create horizontal stack filter
    filter_complex = "".join([f"[{i}:v]" for i in range(n_videos)])
    filter_complex += f"hstack=inputs={n_videos}"
    
    # Add text labels if requested
    if add_labels:
        # Add text overlay for each video
        text_filters = []
        x_positions = [10, 10, 10]  # Will be adjusted based on video width
        
        # First, we need to know the width - let's assume 400x400 per video at 0.5 scale
        video_width = 400
        
        for i, label in enumerate(labels):
            x_pos = i * video_width + 10
            text_filter = f"drawtext=text='{label}':fontsize=24:fontcolor=white:x={x_pos}:y=10:box=1:boxcolor=black@0.5:boxborderw=5"
            text_filters.append(text_filter)
        
        filter_complex += "," + ",".join(text_filters)
    
    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "23",
        str(output_path)
    ]
    
    print(f"\n  Running ffmpeg...")
    print(f"  Output: {output_path}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"\n  ✓ Video created successfully!")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Error creating video:")
        print(f"  {e.stderr}")
        raise
    except FileNotFoundError:
        print("\n  ✗ ffmpeg not found. Please install ffmpeg:")
        print("     Windows: choco install ffmpeg")
        print("     Mac: brew install ffmpeg")
        print("     Linux: apt-get install ffmpeg")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side comparison video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (5° rotation only):
  python scripts/create_comparison_video.py \\
    --noisy outputs/lego_rot5.0deg_*/checkpoint_best.pt \\
    --optimized outputs/lego_poseopt_noisyinit_rot5.0deg_*/checkpoint_best.pt \\
    --output comparison_videos/lego_5deg_rotation.mp4
        """
    )
    
    # Checkpoint paths
    parser.add_argument("--noisy", type=Path, required=True,
                        help="Fixed noisy poses checkpoint")
    parser.add_argument("--optimized", type=Path, required=True,
                        help="Optimized poses checkpoint")
    
    # Output options
    parser.add_argument("--output", type=Path, required=True,
                        help="Output video path")
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="Directory to save intermediate frames (default: auto)")
    
    # Video options
    parser.add_argument("--n-frames", type=int, default=120,
                        help="Number of frames (default: 120)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    
    # Noise configuration (for the noisy checkpoint)
    parser.add_argument("--rotation-noise", type=float, default=5.0,
                        help="Rotation noise in degrees (default: 5.0)")
    parser.add_argument("--translation-noise-pct", type=float, default=0.0,
                        help="Translation noise percentage (default: 0.0 for rotation-only)")
    parser.add_argument("--noise-seed", type=int, default=42,
                        help="Noise seed for reproducibility (default: 42)")
    
    # Scene options
    parser.add_argument("--scene", type=str, default="lego",
                        help="Scene name (default: lego)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Data root directory")
    parser.add_argument("--img-scale", type=float, default=0.5,
                        help="Image scale (default: 0.5)")
    
    # Other options
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Don't add text labels to video")
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = "cpu"
    
    # Setup frames directory
    if args.frames_dir is None:
        frames_base = args.output.parent / f"{args.output.stem}_frames"
    else:
        frames_base = args.frames_dir
    
    frames_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"NeRF Comparison Video Creator")
    print(f"{'='*70}")
    print(f"Scene: {args.scene}")
    print(f"Frames: {args.n_frames} @ {args.fps} fps")
    print(f"Device: {device}")
    print(f"Output: {args.output}")
    print(f"{'='*70}")
    
    # Noise configuration for fixed noisy video
    noise_config = NoiseConfig(
        rotation_noise_deg=args.rotation_noise,
        translation_noise_pct=args.translation_noise_pct,
        seed=args.noise_seed,
    )
    
    # Clean noise config (no noise)
    clean_config = NoiseConfig(
        rotation_noise_deg=0.0,
        translation_noise_pct=0.0,
        seed=None,
    )
    
    # Render two sets of frames (rotation noise only)
    frames_dirs = []
    labels = [f"Fixed Noisy ({args.rotation_noise}°)", f"Joint Optimization ({args.rotation_noise}°)"]
    
    # 1. Fixed noisy (rotation only)
    print(f"\n{'='*70}")
    print(f"1/2: Rendering FIXED NOISY poses ({args.rotation_noise}° rotation)")
    print(f"{'='*70}")
    noisy_frames = render_video_frames(
        checkpoint_path=args.noisy,
        output_dir=frames_base / "noisy",
        noise_config=noise_config,
        n_frames=args.n_frames,
        device=device,
        scene=args.scene,
        data_root=args.data_root,
        img_scale=args.img_scale,
    )
    frames_dirs.append(noisy_frames)
    
    # 2. Optimized
    print(f"\n{'='*70}")
    print(f"2/2: Rendering OPTIMIZED poses ({args.rotation_noise}° rotation)")
    print(f"{'='*70}")
    optimized_frames = render_video_frames(
        checkpoint_path=args.optimized,
        output_dir=frames_base / "optimized",
        noise_config=clean_config,  # No additional noise - poses are already learned
        n_frames=args.n_frames,
        device=device,
        scene=args.scene,
        data_root=args.data_root,
        img_scale=args.img_scale,
    )
    frames_dirs.append(optimized_frames)
    
    # Stitch videos together
    video_path = stitch_videos_horizontal(
        frames_dirs=frames_dirs,
        labels=labels,
        output_path=args.output,
        fps=args.fps,
        add_labels=not args.no_labels,
    )
    
    print(f"\n{'='*70}")
    print(f"✓ Comparison video created!")
    print(f"{'='*70}")
    print(f"Output: {video_path}")
    print(f"Frames: {frames_base}")
    print(f"\nYou can now:")
    print(f"  1. Play the video: {video_path}")
    print(f"  2. Delete frames to save space: rm -rf {frames_base}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
