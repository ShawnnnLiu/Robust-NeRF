"""
Placeholder definitions for camera parameter containers.

The goal is to eventually have a clean interface to hold and manipulate
camera intrinsics and extrinsics (and possibly distortion parameters) so
they can be optimized jointly with NeRF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraParameters:
    """
    Lightweight container for per-camera parameters.

    This class is intentionally minimal for now; methods and behavior will be
    added once the NeRF model and training loop are implemented.
    """

    intrinsics: np.ndarray
    extrinsics: np.ndarray
    camera_id: Optional[str] = None


