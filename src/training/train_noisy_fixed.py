"""
Entry point for training NeRF with fixed noisy camera parameters.

The idea is to:
- Inject synthetic noise into intrinsics / extrinsics.
- Keep these noisy parameters fixed during training.
- Study how reconstruction quality degrades under different noise levels.
"""

from __future__ import annotations


def main() -> None:
    """
    Placeholder training loop for the fixed-noisy camera experiment.

    Planned responsibilities:
    - Load the lego dataset and ground-truth camera parameters.
    - Apply noise to intrinsics and/or extrinsics.
    - Train a NeRF model using the noisy parameters without further updates.
    """
    raise NotImplementedError("Fixed-noisy NeRF training is not implemented yet.")


if __name__ == "__main__":
    main()


