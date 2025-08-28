"""
Prometheus dataset implementation for Neptune training.

Handles Prometheus neutrino simulation data with memory-mapped files
for fast data loading and processes it using summary statistics for efficient training.
"""

import os
import pickle
import struct
import torch
import numpy as np
import nt_summary_stats
from torch.utils.data import DataLoader, RandomSampler
from typing import Tuple

from .data_utils import IrregularDataCollator


def load_ntmmap(input_path: str) -> Tuple[np.memmap, np.memmap, np.dtype]:
    """
    Load memory-mapped files with automatic dtype detection (format with headers).
    
    Args:
        input_path: Base path for input files (without extension)
        
    Returns:
        Tuple of (index_mmap, data_mmap, photon_dtype)
    """
    idx_path = f"{input_path}.idx"
    dat_path = f"{input_path}.dat"
    
    if not os.path.exists(idx_path) or not os.path.exists(dat_path):
        raise FileNotFoundError(f"Memory-mapped files not found: {input_path}")
    
    # Load index file with header
    with open(idx_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        event_dtype = pickle.loads(f.read(dtype_size))
        data_start = f.tell()
    
    index_mmap = np.memmap(idx_path, dtype=event_dtype, mode='r', offset=data_start)
    
    # Load data file with header
    with open(dat_path, 'rb') as f:
        dtype_size = struct.unpack('<I', f.read(4))[0]
        photon_dtype = pickle.loads(f.read(dtype_size))
        data_start = f.tell()
    
    # Memory map photons as structured array (not raw bytes)
    photons_array = np.memmap(dat_path, dtype=photon_dtype, mode='r', offset=data_start)
    
    return index_mmap, photons_array, photon_dtype


def get_event_photons(photons_array: np.memmap, event_record: np.ndarray) -> np.ndarray:
    """
    Extract photons for a specific event using photon indices.
    
    Args:
        photons_array: Memory-mapped photon array (structured)
        event_record: Single EventRecord
        
    Returns:
        Array of PhotonHit records
    """
    start_idx = int(event_record['photon_start_idx'])
    end_idx = int(event_record['photon_end_idx'])
    
    if start_idx >= end_idx:
        # Return empty array with correct dtype
        return np.array([], dtype=photons_array.dtype)
    
    # Direct array slicing
    return photons_array[start_idx:end_idx]


class PrometheusDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset class for Prometheus data."""
    
    def __init__(self, mmap_path, use_summary_stats=True):
        self.mmap_path = mmap_path
        self.use_summary_stats = use_summary_stats
        
        # Load memory-mapped files
        self.events, self.photons_array, self.photon_dtype = load_ntmmap(mmap_path)
        self.total_events = len(self.events)
        
    def __len__(self):
        return self.total_events
    
    def __getitem__(self, idx):
        # Get event record and photons directly from memory-mapped arrays
        event_record = self.events[idx]
        photons = get_event_photons(self.photons_array, event_record)
        
        if self.use_summary_stats:
            # Convert structured photon array to dictionary format for nt-summary-stats
            photons_dict = {
                'sensor_pos_x': photons['x'],
                'sensor_pos_y': photons['y'],
                'sensor_pos_z': photons['z'],
                't': photons['t'],
                'charge': photons['charge'], 
                'string_id': photons['string_id'],
                'sensor_id': photons['sensor_id'],
                'id_idx': photons['id_idx']
            }
            
            # Summary stats mode: Use nt-summary-stats to process the event and get sensor-level features
            sensor_positions, sensor_stats = nt_summary_stats.process_prometheus_event(photons_dict)

            # Create 4D coordinates: x, y, z, t (using time_mean for t)
            pos = np.column_stack([
                sensor_positions[:, 0],  # x
                sensor_positions[:, 1],  # y
                sensor_positions[:, 2],  # z
                sensor_stats[:, 3]       # first hit time
            ]).astype(np.float32)
            pos = pos / 1000. # convert to km / microseconds
            
            # Use all 9 summary statistics as features
            feats = sensor_stats.astype(np.float32)
            feats = np.log(feats + 1)
        else:
            pos = np.column_stack([photons['x'], photons['y'], photons['z'], photons['t']])
            pos = np.trunc(pos)
            pos, feats = np.unique(pos, return_counts=True, axis=0)
            
            pos = pos / 1000.
            feats = np.log(feats + 1).reshape(-1, 1)
        
        # Extract labels from event record (keep raw zenith/azimuth; log-transform energy)
        initial_zenith = event_record['initial_zenith']
        initial_azimuth = event_record['initial_azimuth']
        initial_energy = event_record['initial_energy']
        log_energy = np.log10(max(initial_energy, 1e-6))
        
        # Convert spherical to Cartesian direction using raw angles
        dir_x = np.sin(initial_zenith) * np.cos(initial_azimuth)
        dir_y = np.sin(initial_zenith) * np.sin(initial_azimuth)
        dir_z = np.cos(initial_zenith)
        
        # Labels: [log_energy, dir_x, dir_y, dir_z]
        labels = np.array([log_energy, dir_x, dir_y, dir_z], dtype=np.float32)
        
        return pos, feats, labels


def create_prometheus_dataloaders(cfg):
    """Create train and validation dataloaders for Prometheus data from config."""
    
    # Create datasets with memory-mapped files
    train_dataset = PrometheusDataset(
        cfg['data_options']['train_data_path'],
        cfg['data_options'].get('use_summary_stats', True)
    )
    
    valid_dataset = PrometheusDataset(
        cfg['data_options']['valid_data_path'],
        cfg['data_options'].get('use_summary_stats', True)
    )
    
    # Create samplers - use standard RandomSampler since memory-mapped data doesn't need file-aware batching
    train_sampler = RandomSampler(train_dataset)
    val_sampler = None  # Use sequential sampling for validation
    
    # Create dataloaders
    collate_fn = IrregularDataCollator()
    
    # Handle persistent_workers setting based on num_workers
    num_workers = cfg['training_options']['num_workers']
    persistent_workers = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training_options']['batch_size'], 
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg['training_options']['batch_size'], 
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
        num_workers=num_workers
    )
    
    return train_loader, val_loader