"""
NeRF model stub.

The actual network architecture (MLP, positional encoding, etc.) will be
implemented later. For now, this file only defines a placeholder class and
interface.
"""

from __future__ import annotations

from typing import Any


class NeRFModel:
    """
    Placeholder NeRF model.

    Once implemented, this class will likely wrap a neural network (e.g.,
    a PyTorch ``nn.Module``) that maps 3D positions and view directions to
    color and density.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # TODO: define fields such as network depth, width, encoding sizes, etc.
        _ = args, kwargs

    def forward(self, rays_o, rays_d):
        """
        Placeholder forward method.

        Parameters
        ----------
        rays_o:
            Ray origins.
        rays_d:
            Ray directions.
        """
        raise NotImplementedError("NeRF forward pass is not implemented yet.")


