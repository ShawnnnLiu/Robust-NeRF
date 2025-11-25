"""
Placeholder interfaces for camera noise models.

Concrete implementations can be adapted from ``scripts/inject_noise.py`` once
the NeRF training pipeline is in place.
"""

from __future__ import annotations

import numpy as np


def gaussian_noise_intrinsics(
    intrinsics: np.ndarray,
    std_pixels: float,
) -> np.ndarray:
    """
    Placeholder for adding Gaussian noise to intrinsics.

    This function is intentionally left unimplemented; use the utilities in
    ``scripts/inject_noise.py`` for experimentation until the full pipeline
    is ready.
    """
    raise NotImplementedError(
        "gaussian_noise_intrinsics will be implemented once the NeRF pipeline is ready."
    )


def gaussian_noise_extrinsics(
    extrinsics: np.ndarray,
    translation_std: float,
    rotation_std_deg: float,
) -> np.ndarray:
    """
    Placeholder for adding Gaussian noise to extrinsics.

    See ``scripts/inject_noise.py`` for a simple working version.
    """
    raise NotImplementedError(
        "gaussian_noise_extrinsics will be implemented once the NeRF pipeline is ready."
    )


