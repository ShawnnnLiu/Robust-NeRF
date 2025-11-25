"""
Entry point for joint optimization of NeRF and camera parameters.

This experiment will:
- Start from noisy camera intrinsics / extrinsics.
- Treat (some of) these camera parameters as learnable.
- Jointly optimize NeRF and camera parameters using reconstruction loss.
"""

from __future__ import annotations


def main() -> None:
    """
    Placeholder training loop for the joint optimization experiment.

    Planned responsibilities:
    - Initialize camera parameter objects (e.g., src.camera.camera_parameters).
    - Hook camera parameters into the optimizer alongside NeRF weights.
    - Track how pose / intrinsics estimates evolve during training.
    """
    raise NotImplementedError("Joint NeRF + camera optimization is not implemented yet.")


if __name__ == "__main__":
    main()


