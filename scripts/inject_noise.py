"""
Simple Gaussian noise utilities for camera intrinsics and extrinsics.

These helpers are intended for quick experiments and prototyping. More
structured noise models can later live in ``src/camera/noise_models.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def add_gaussian_noise_to_matrix(
    mat: np.ndarray,
    mean: float = 0.0,
    std: float = 1e-3,
) -> np.ndarray:
    """
    Add element-wise Gaussian noise to a matrix.

    Parameters
    ----------
    mat:
        Input matrix (e.g., intrinsics or extrinsics).
    mean:
        Mean of the Gaussian noise.
    std:
        Standard deviation of the Gaussian noise.
    """
    noise = np.random.normal(loc=mean, scale=std, size=mat.shape)
    return mat + noise


def perturb_intrinsics(
    K: np.ndarray,
    std_pixels: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian noise to the camera intrinsics matrix.

    This is a simple placeholder; you can refine it later to only perturb
    specific parameters (e.g., focal length or principal point).
    """
    return add_gaussian_noise_to_matrix(K, mean=0.0, std=std_pixels)


def perturb_extrinsics(
    pose: np.ndarray,
    translation_std: float = 1e-3,
    rotation_std_deg: float = 0.5,
) -> np.ndarray:
    """
    Apply small Gaussian noise to a camera-to-world extrinsic matrix.

    - Translation noise is added directly to the translation vector.
    - Rotation noise is applied via a small random axis-angle vector, using
      a first-order approximation of the exponential map.
    """
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got shape {pose.shape}")

    noisy = pose.astype(np.float64).copy()

    # Translation noise
    noisy[:3, 3] += np.random.normal(scale=translation_std, size=3)

    # Small rotation noise (axis-angle, first-order exp map approximation)
    angle_rad = np.deg2rad(rotation_std_deg)
    omega = np.random.normal(scale=angle_rad, size=3)
    skew = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ],
        dtype=np.float64,
    )

    R = noisy[:3, :3]
    R_noisy = R + skew @ R  # first-order approximation of exp(skew) * R
    noisy[:3, :3] = R_noisy

    return noisy


def perturb_transforms_dict(
    transforms: dict,
    translation_std: float = 1e-3,
    rotation_std_deg: float = 0.5,
) -> dict:
    """
    Apply pose noise to every frame in a NeRF-style transforms JSON dict.

    Returns a new dict; the input is not modified in-place.
    """
    noisy = dict(transforms)
    noisy_frames = []

    for frame in transforms.get("frames", []):
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        noisy_mat = perturb_extrinsics(
            mat,
            translation_std=translation_std,
            rotation_std_deg=rotation_std_deg,
        )
        new_frame = dict(frame)
        new_frame["transform_matrix"] = noisy_mat.tolist()
        noisy_frames.append(new_frame)

    noisy["frames"] = noisy_frames
    return noisy


if __name__ == "__main__":
    # Minimal smoke test
    K_test = np.eye(3)
    pose_test = np.eye(4)

    print("Noisy intrinsics:\n", perturb_intrinsics(K_test))
    print("Noisy pose:\n", perturb_extrinsics(pose_test))


