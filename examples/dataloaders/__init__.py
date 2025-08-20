"""
Dataloaders package for Neptune training examples.

Contains dataset implementations for different neutrino detection experiments:
- Prometheus: Simulated data with summary statistics
- IceCube: Real detector data with pulse-level information
"""

from .data_utils import (
    ParquetFileSampler,
    IrregularDataCollator,
    batched_coordinates
)
from .prometheus import create_prometheus_dataloaders
from .icecube import create_icecube_dataloaders

__all__ = [
    'ParquetFileSampler',
    'IrregularDataCollator', 
    'batched_coordinates',
    'create_prometheus_dataloaders',
    'create_icecube_dataloaders'
]