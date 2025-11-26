"""Baseline NeRF training script (lego, no camera noise).

This script is intentionally minimal but modular so it can be reused for
noisy-parameter and joint-optimization experiments.

Usage (from repo root):

    python -m src.training.train_baseline --scene lego

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.blender import load_blender_split
from src.nerf.model import NeRF, NeRFConfig
from src.nerf.rendering import get_rays, render_rays


def _find_scene_root(project_root: Path, scene_name: str) -> Path:
    candidates = [
        project_root / "data" / "raw" / scene_name,
        project_root / "data" / "raw" / "nerf_synthetic" / scene_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find scene '{scene_name}'. Tried:\n" + "\n".join(str(c) for c in candidates)
    )


def _prepare_training_rays(
    images: np.ndarray,
    poses: np.ndarray,
    hwf: Tuple[int, int, float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute ray origins/directions and target RGB for all pixels.

    To keep memory usage reasonable, you may want to use a downscaled
    version of the images (controlled by img_scale in load_blender_split).
    """

    H, W, focal = hwf
    images_t = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)

    all_rays_o = []
    all_rays_d = []
    all_rgbs = []

    for i in range(images_t.shape[0]):
        c2w = poses_t[i]
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        all_rays_o.append(rays_o.reshape(-1, 3))
        all_rays_d.append(rays_d.reshape(-1, 3))
        all_rgbs.append(images_t[i].reshape(-1, 3))

    rays_o = torch.cat(all_rays_o, dim=0)
    rays_d = torch.cat(all_rays_d, dim=0)
    rgbs = torch.cat(all_rgbs, dim=0)
    return rays_o, rays_d, rgbs


def train_baseline(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[2]
    scene_root = _find_scene_root(project_root, args.scene)
    print(f"Using scene root: {scene_root}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Device:", device)

    split = load_blender_split(
        split="train",
        scene_name=args.scene,
        root=project_root,
        img_scale=args.img_scale,
    )

    H, W, focal = split.hwf
    print(f"Loaded {split.images.shape[0]} training images at {H}x{W}, focal={focal:.2f}")

    rays_o, rays_d, target_rgb = _prepare_training_rays(split.images, split.poses, split.hwf, device)
    num_rays = rays_o.shape[0]
    print(f"Total training rays: {num_rays}")

    model = NeRF(NeRFConfig()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    near, far = args.near, args.far
    num_samples = args.num_samples

    for step in range(1, args.num_iters + 1):
        idx = torch.randint(0, num_rays, (args.batch_size,), device=device)
        rays_o_batch = rays_o[idx]
        rays_d_batch = rays_d[idx]
        rgb_target = target_rgb[idx]

        rgb_pred, _, _ = render_rays(
            model=model,
            rays_o=rays_o_batch,
            rays_d=rays_d_batch,
            near=near,
            far=far,
            num_samples=num_samples,
            perturb=True,
        )

        loss = F.mse_loss(rgb_pred, rgb_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            psnr = -10.0 * torch.log10(loss)
            print(f"Step {step:06d} | loss={loss.item():.6f} | psnr={psnr.item():.2f} dB")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline NeRF training (Blender / lego)")
    parser.add_argument("--scene", type=str, default="lego", help="Blender scene name (e.g. lego, chair)")
    parser.add_argument(
        "--img-scale",
        dest="img_scale",
        type=float,
        default=0.5,
        help="Uniform image downscale factor (e.g. 0.5)",
    )
    parser.add_argument(
        "--num-iters",
        dest="num_iters",
        type=int,
        default=20000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=1024,
        help="Rays per batch",
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        type=int,
        default=64,
        help="Samples per ray",
    )
    parser.add_argument("--near", type=float, default=2.0, help="Near plane distance")
    parser.add_argument("--far", type=float, default=6.0, help="Far plane distance")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--log-every",
        dest="log_every",
        type=int,
        default=100,
        help="Logging frequency (iterations)",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU even if CUDA is available")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()


