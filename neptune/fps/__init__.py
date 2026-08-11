"""Vendored farthest-point-sampling kernels (formerly the torch-fps package).

Triton on CUDA, JIT-compiled C++ on CPU, pure-torch reference everywhere
else — no compiled code at install time. See api.py for the contract.
"""
from .api import farthest_point_sampling, farthest_point_sampling_with_knn

__all__ = ["farthest_point_sampling", "farthest_point_sampling_with_knn"]
