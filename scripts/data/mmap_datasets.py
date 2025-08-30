"""
Unified memory-mapped dataset implementation for Neptune training.

Handles both Prometheus and IceCube neutrino data in the unified mmap format
created by nt-mmap-converter. Auto-detects data type and provides consistent
interface for both dataset types.
"""

import torch
import numpy as np
from typing import Union, List, Tuple
try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False

from .utils import load_ntmmap


class MmapDataset(torch.utils.data.Dataset):
    """
    Unified memory-mapped dataset class for both Prometheus and IceCube data.
    
    Automatically detects dataset type from file headers and provides consistent
    interface for both formats. Supports multiple mmap file sets for large datasets.
    """
    
    def __init__(self, 
                 mmap_paths: Union[str, List[str]], 
                 use_summary_stats: bool = True,
                 split: str = "full",
                 val_split: float = 0.2,
                 split_seed: int = 42):
        """
        Initialize unified mmap dataset.
        
        Args:
            mmap_paths: Path(s) to mmap files (without extension)
            use_summary_stats: Whether to use nt-summary-stats processing
            split: Dataset split - "train", "val", or "full"
            val_split: Fraction of data to use for validation (when split != "full")
            split_seed: Random seed for deterministic train/val splitting
        """
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS
        
        # Handle both single path and multiple paths
        if isinstance(mmap_paths, str):
            mmap_paths = [mmap_paths]
        
        self.datasets = []
        self.cumulative_lengths = []
        total_length = 0
        self.dataset_type = None
        
        # Load all memory-mapped file sets
        for path in mmap_paths:
            events, photons_array, photon_dtype = load_ntmmap(path)
            
            # Auto-detect dataset type from first file
            if self.dataset_type is None:
                field_names = set(events.dtype.names)
                # Prometheus has these fields that IceCube doesn't
                if 'bjorken_x' in field_names or 'column_depth' in field_names:
                    self.dataset_type = 'prometheus'
                else:
                    self.dataset_type = 'icecube'
            
            self.datasets.append((events, photons_array))
            total_length += len(events)
            self.cumulative_lengths.append(total_length)
        
        self.total_events = total_length
        self.photon_dtype = photon_dtype
        
        # Handle train/val splitting
        if split in ["train", "val"]:
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(self.total_events)
            
            val_size = int(self.total_events * val_split)
            if split == "val":
                self.indices = indices[:val_size]
            else:  # train
                self.indices = indices[val_size:]
        else:
            self.indices = None  # Use full dataset
        
        # Update length for split
        if self.indices is not None:
            split_length = len(self.indices)
            print(f"Loaded {self.dataset_type} dataset: {split_length:,} events ({split} split from {self.total_events:,} total)")
        else:
            print(f"Loaded {self.dataset_type} dataset with {self.total_events:,} events")
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.total_events
    
    def __getitem__(self, idx):
        """Get a single event - returns (coords, features, labels)"""
        # Handle split indices - map from split index to global index
        if self.indices is not None:
            global_idx = self.indices[idx]
        else:
            global_idx = idx
        
        # Find which dataset contains this global index
        dataset_idx = np.searchsorted(self.cumulative_lengths, global_idx + 1)
        
        # Calculate local index within the dataset
        if dataset_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_lengths[dataset_idx - 1]
        
        # Get data from the appropriate dataset
        events, photons_array = self.datasets[dataset_idx]
        event_record = events[local_idx]
        
        # Extract photons for this event (inline from original)
        start_idx = int(event_record['photon_start_idx'])
        end_idx = int(event_record['photon_end_idx'])
        
        if start_idx >= end_idx:
            raise ValueError("Invalid photon indices")
        else:
            photons = photons_array[start_idx:end_idx]
        
        # Process photons based on dataset type and settings
        if self.use_summary_stats and len(photons) > 0:
            # Convert structured photon array to dictionary format
            photons_dict = {
                'sensor_pos_x': photons['x'],
                'sensor_pos_y': photons['y'],
                'sensor_pos_z': photons['z'],
                't': photons['t'],
                'charge': photons['charge'], 
                'string_id': photons['string_id'],
                'sensor_id': photons['sensor_id']
            }
            
            # Add dataset-specific fields
            if self.dataset_type == 'prometheus' and 'id_idx' in photons.dtype.names:
                photons_dict['id_idx'] = photons['id_idx']
            
            # Process with nt-summary-stats
            sensor_positions, sensor_stats = nt_summary_stats.process_prometheus_event(photons_dict)

            # Create 4D coordinates: x, y, z, t (using first hit time)
            pos = np.column_stack([
                sensor_positions[:, 0],  # x
                sensor_positions[:, 1],  # y
                sensor_positions[:, 2],  # z
                sensor_stats[:, 3]       # first hit time
            ]).astype(np.float32)
            pos = pos / 1000.  # convert to km / microseconds
            
            # Use all summary statistics as features
            feats = sensor_stats.astype(np.float32)
            feats = np.log(feats + 1)
        else:
            # pulse-level
            pos = np.column_stack([photons['x'], photons['y'], photons['z'], photons['t']])
            pos = pos / 1000.
            feats = np.log(photons['charge'] + 1).reshape(-1, 1)

        # Extract labels
        initial_zenith = event_record['initial_zenith']
        initial_azimuth = event_record['initial_azimuth']
        initial_energy = event_record['initial_energy']
        log_energy = np.log10(max(initial_energy, 1e-6))
        
        # Convert spherical to Cartesian direction
        dir_x = np.sin(initial_zenith) * np.cos(initial_azimuth)
        dir_y = np.sin(initial_zenith) * np.sin(initial_azimuth)
        dir_z = np.cos(initial_zenith)
        
        if self.dataset_type == 'prometheus':
            # Get the particle id from final state
            pid = event_record['initial_type']
            labels = np.array([log_energy, dir_x, dir_y, dir_z, pid], dtype=np.float32)
        else:
            interaction_type = event_record['interaction']
            labels = np.array([log_energy, dir_x, dir_y, dir_z, interaction_type], dtype=np.float32)

        return pos, feats, labels