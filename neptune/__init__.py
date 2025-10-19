"""
Neptune: A transformer-based point cloud processing model for neutrino event reconstruction.

This package provides a clean, pip-installable implementation of the Neptune model
that can be easily integrated into any PyTorch workflow.

Example usage:
    >>> from neptune import NeptuneModel
    >>> model = NeptuneModel(
    ...     in_channels=6,
    ...     num_patches=128,
    ...     token_dim=768,
    ...     output_dim=3
    ... )
    >>> coords = torch.randn(1024, 4)        # [x, y, z, t]
    >>> features = torch.randn(1024, 6)
    >>> batch_ids = torch.zeros(1024, dtype=torch.long)
    >>> output = model(coords, features, batch_ids)
"""

from .model import NeptuneModel

__version__ = "0.1.0"
__all__ = ["NeptuneModel"]
