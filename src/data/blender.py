"""
NeRF Synthetic Blender (e.g. lego) data loading utilities.

This module provides a minimal, dependency-light loader for the Blender
dataset used in the original NeRF paper, structured to be reusable for
multiple scenes and experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class BlenderSplit:
    """
    Container for a single Blender split (e.g. train / val / test).

    Attributes
    ----------
    images:
        N x H x W x 3 float32 array in [0, 1].
    poses:
        N x 4 x 4 float32 camera-to-world matrices.
    hwf:
        Tuple (H, W, focal) describing image height, width, and focal length
        in pixels for this split.
    """

    images: np.ndarray
    poses: np.ndarray
    hwf: Tuple[int, int, float]


def _resolve_scene_root(root: Path | None = None, scene_name: str = "lego") -> Path:
    """
    Try to resolve a scene directory for a given Blender scene name.

    Prefers:
    - data/raw/lego
    and falls back to:
    - data/raw/nerf_synthetic/lego
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "data" / "raw" / scene_name,
        root / "data" / "raw" / "nerf_synthetic" / scene_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find scene '{scene_name}'. Tried:\n"
        + "\n".join(str(c) for c in candidates)
    )


def load_blender_split(
    split: str = "train",
    scene_name: str = "lego",
    root: Path | None = None,
    img_scale: float = 0.5,
) -> BlenderSplit:
    """
    Load a Blender split (train/val/test) for a given scene.

    Parameters
    ----------
    split:
        One of 'train', 'val', or 'test'.
    scene_name:
        Name of the Blender scene, e.g. 'lego', 'chair', etc.
    root:
        Project root, used to resolve data paths. If None, inferred from
        this file's location.
    img_scale:
        Uniform scaling factor for images (e.g., 0.5 for half-res). This
        also scales the focal length accordingly.
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split!r}")

    scene_root = _resolve_scene_root(root=root, scene_name=scene_name)
    meta_path = scene_root / f"transforms_{split}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find transforms file: {meta_path}")

    with meta_path.open("r") as f:
        meta = json.load(f)

    camera_angle_x = float(meta["camera_angle_x"])

    images = []
    poses = []

    for frame in meta["frames"]:
        file_path = frame["file_path"]  # e.g. './train/r_0'
        img_path = scene_root / f"{file_path}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing frame image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if img_scale != 1.0:
            new_w = int(w * img_scale)
            new_h = int(h * img_scale)
            img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            w, h = new_w, new_h

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        images.append(img_np)

        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        poses.append(pose)

    images_np = np.stack(images, axis=0)
    poses_np = np.stack(poses, axis=0)

    # Compute focal length from horizontal FOV.
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    if img_scale != 1.0:
        focal *= img_scale

    return BlenderSplit(images=images_np, poses=poses_np, hwf=(h, w, float(focal)))



