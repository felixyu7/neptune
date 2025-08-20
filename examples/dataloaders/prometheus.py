"""
Prometheus dataset implementation for Neptune training.

Handles Prometheus neutrino simulation data with nested photon information
and processes it using summary statistics for efficient training.
"""

import os
import torch
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
import nt_summary_stats
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from .data_utils import get_file_names, ParquetFileSampler, IrregularDataCollator


class PrometheusDataset(torch.utils.data.Dataset):
    """Dataset class for Prometheus data."""
    
    def __init__(self, files, use_latent_representation=False, geo_dict_path=None):
        self.files = files
        self.use_latent_representation = use_latent_representation
        
        # Count number of events in each file
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        
        # Create cumulative lengths for sampling
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(num_events)])
        self.total_events = sum(num_events)
        
        # Cache for the currently loaded file
        self._cached_file_idx = None
        self._cached_table = None
        
    def __len__(self):
        return self.total_events
    
    def __getitem__(self, idx):
        # Determine which file this index belongs to
        file_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # Load data from the appropriate file (with caching)
        if self._cached_file_idx != file_idx:
            data = pq.ParquetFile(self.files[file_idx])
            row_group = data.read_row_group(0)  # Assume single row group for simplicity
            self._cached_table = row_group.to_pandas()
            self._cached_file_idx = file_idx
        
        table = self._cached_table
        
        # Extract event data
        event_data = table.iloc[local_idx]
        
        # Extract photon hit data from the nested structure
        photons = event_data['photons']
        mc_truth = event_data['mc_truth']
        
        # Use nt-summary-stats to process the event and get sensor-level features
        sensor_positions, sensor_stats = nt_summary_stats.process_prometheus_event(photons)

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
        
        # Extract labels from mc_truth (keep raw zenith/azimuth; log-transform energy)
        initial_zenith = mc_truth['initial_state_zenith']
        initial_azimuth = mc_truth['initial_state_azimuth']
        initial_energy = mc_truth.get('initial_state_energy', 0.0)
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
    
    # Create datasets
    train_files = get_file_names(
        cfg['data_options']['train_data_files'], 
        cfg['data_options']['train_data_file_ranges'],
        cfg['data_options']['shuffle_files']
    )
    train_dataset = PrometheusDataset(
        train_files,
        cfg['data_options'].get('use_latent_representation', False),
        cfg['data_options'].get('geo_dict_path', None)
    )
    
    valid_files = get_file_names(
        cfg['data_options']['valid_data_files'], 
        cfg['data_options']['valid_data_file_ranges'],
        cfg['data_options']['shuffle_files']
    )
    valid_dataset = PrometheusDataset(
        valid_files,
        cfg['data_options'].get('use_latent_representation', False),
        cfg['data_options'].get('geo_dict_path', None)
    )
    
    # Create samplers
    train_sampler = ParquetFileSampler(
        train_dataset, 
        train_dataset.cumulative_lengths, 
        cfg['training_options']['batch_size']
    )
    val_sampler = ParquetFileSampler(
        valid_dataset, 
        valid_dataset.cumulative_lengths, 
        cfg['training_options']['batch_size']
    )
    
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