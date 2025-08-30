"""
Unified data package for Neptune training scripts.

Provides clean interfaces for all supported dataset types:
- Memory-mapped datasets (Prometheus, IceCube) via unified format
- Legacy Kaggle IceCube format

All datasets share common interfaces and can be created via the unified
create_dataloaders() factory function.
"""

from .mmap_datasets import MmapDataset
from .kaggle_datasets import KaggleDataset, load_sensor_geometry, get_icecube_file_names
from .utils import IrregularDataCollator, batched_coordinates, RandomChunkSampler
from .base import create_dataloaders

__all__ = [
    'MmapDataset',
    'KaggleDataset',
    'load_sensor_geometry',
    'get_icecube_file_names',
    'IrregularDataCollator',
    'batched_coordinates',
    'RandomChunkSampler',
    'create_dataloaders'
]