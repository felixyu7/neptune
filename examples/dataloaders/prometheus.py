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

from .data_utils import get_file_names, ParquetFileSampler, IrregularDataCollator, group_hits_by_window


class PrometheusDataset(torch.utils.data.Dataset):
    """Dataset class for Prometheus data."""
    
    def __init__(self, files, use_summary_stats=True):
        self.files = files
        self.use_summary_stats = use_summary_stats
        
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
        
        if self.use_summary_stats:
            # Summary stats mode: Use nt-summary-stats to process the event and get sensor-level features
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
        else:
            # Pulse mode: Group hits by window and create pulses
            pulse_positions = []
            pulse_features = []
            
            # Extract photon data arrays
            sensor_ids = photons['sensor_id']
            times = photons['t']
            sensor_pos_x = photons['sensor_pos_x']
            sensor_pos_y = photons['sensor_pos_y'] 
            sensor_pos_z = photons['sensor_pos_z']
            
            # Group photons by sensor
            sensors = {}
            for i in range(len(sensor_ids)):
                sensor_id = sensor_ids[i]
                if sensor_id not in sensors:
                    sensors[sensor_id] = {
                        'times': [], 
                        'charges': [],  # Use constant charge of 1.0 for each hit
                        'position': [sensor_pos_x[i], sensor_pos_y[i], sensor_pos_z[i]]
                    }
                sensors[sensor_id]['times'].append(times[i])
                sensors[sensor_id]['charges'].append(1.0)  # Each photon hit has charge 1.0
            
            # Process each sensor to create pulses
            for sensor_id, sensor_data in sensors.items():
                hit_times = np.array(sensor_data['times'])
                hit_charges = np.array(sensor_data['charges'])

                # Group hits by 3ns window
                pulse_times, pulse_charges = group_hits_by_window(hit_times, hit_charges, window_ns=3.0)
                
                # Create position for each pulse (x, y, z, pulse_time)
                sensor_pos = sensor_data['position']
                for p_time, p_charge in zip(pulse_times, pulse_charges):
                    pulse_positions.append([sensor_pos[0], sensor_pos[1], sensor_pos[2], p_time])
                    pulse_features.append([p_time, p_charge])
            
            # Convert to numpy arrays
            pos = np.array(pulse_positions, dtype=np.float32)
            pos = pos / 1000. # convert to km / microseconds
            
            # Features are just [time, charge] for each pulse
            feats = np.array(pulse_features, dtype=np.float32)
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
        cfg['data_options'].get('use_summary_stats', True)
    )
    
    valid_files = get_file_names(
        cfg['data_options']['valid_data_files'], 
        cfg['data_options']['valid_data_file_ranges'],
        cfg['data_options']['shuffle_files']
    )
    valid_dataset = PrometheusDataset(
        valid_files,
        cfg['data_options'].get('use_summary_stats', True)
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