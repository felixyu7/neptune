"""
IceCube dataset implementation for Neptune training.

Handles IceCube neutrino detection data with either pulse-level information
or sensor-level summary statistics computed using nt_summary_stats.
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import random
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from collections import OrderedDict
from bisect import bisect_right

try:
    import nt_summary_stats
except ImportError:
    nt_summary_stats = None

from .data_utils import IrregularDataCollator, RandomChunkSampler


def load_sensor_geometry(geometry_path: str) -> np.ndarray:
    """Load IceCube sensor geometry from CSV file."""
    geometry_df = pd.read_csv(geometry_path)
    # Return as numpy array: [x, y, z] for each sensor_id
    return geometry_df[['x', 'y', 'z']].values.astype(np.float32)


def get_icecube_file_names(batch_dir: str, 
                          ranges: List[int], 
                          shuffle_files: bool = False) -> List[str]:
    """Get batch file names from IceCube directory within specified ranges."""
    all_files = sorted(glob.glob(os.path.join(batch_dir, 'batch_*.parquet')))
    if shuffle_files:
        random.shuffle(all_files)
    
    # Apply range filtering
    if len(ranges) == 2:
        filtered_files = all_files[ranges[0]:ranges[1]]
    else:
        filtered_files = all_files
    
    return sorted(filtered_files)


class IceCubeDataset(torch.utils.data.Dataset):
    """
    Dataset class for IceCube neutrino detection data with chunk-based caching.
    
    Based on the notebook approach: uses OrderedDict for efficient caching of
    batch files and works with RandomChunkSampler for optimal performance.
    
    Supports two modes:
    - Pulse-level (use_summary_stats=False): Uses raw pulse data as in original implementation
    - Summary stats (use_summary_stats=True): Groups pulses by sensor and computes summary statistics
    """
    
    def __init__(self, 
                 meta_dir: str,
                 batch_files: List[str], 
                 sensor_geometry: np.ndarray,
                 cache_size: int = 2,
                 use_summary_stats: bool = False):
        self.batch_files = batch_files
        self.sensor_geometry = sensor_geometry
        self.cache_size = cache_size
        self.use_summary_stats = use_summary_stats
        
        # Check if nt_summary_stats is available when needed
        if use_summary_stats and nt_summary_stats is None:
            raise ImportError("nt_summary_stats is required when use_summary_stats=True. "
                            "Install with: pip install nt_summary_stats")
        
        # Initialize caches using OrderedDict for LRU behavior
        self.batch_cache = None  # Will be OrderedDict[str, pl.DataFrame]
        self.meta_cache = None   # Will be OrderedDict[str, pd.DataFrame]
        
        # Process batch files to determine chunks (events per file)
        print(f"Analyzing {len(batch_files)} batch files...")
        self.batch_file_names = []
        self.chunks = []  # Number of events in each batch file
        
        # Load metadata files to get event counts per batch
        for batch_file in batch_files:
            batch_num = int(Path(batch_file).stem.split('_')[1])
            self.batch_file_names.append(Path(batch_file).name)
            
            # Load corresponding metadata batch file
            meta_batch_file = os.path.join(meta_dir, f"train_meta_{batch_num}.parquet")
            if os.path.exists(meta_batch_file):
                meta_table = pq.read_table(meta_batch_file)
                batch_meta_df = meta_table.to_pandas()
                self.chunks.append(len(batch_meta_df))
            else:
                raise FileNotFoundError(f"Metadata file not found: {meta_batch_file}")
        
        # Compute cumulative chunk sizes for indexing
        self.chunk_cumsum = np.cumsum(self.chunks)
        self.total_events = self.chunk_cumsum[-1]
        
        print(f"Dataset: {self.total_events} events across {len(self.chunks)} batch files")
        print(f"Chunk sizes: {self.chunks}")
        
        # Store metadata directory path for lazy loading
        self.meta_dir = meta_dir
        
        # Create mapping from batch file name to full path  
        self.batch_name_to_path = {}
        for batch_file in batch_files:
            self.batch_name_to_path[Path(batch_file).name] = batch_file
    
    def __len__(self):
        return self.total_events
    
    def _load_batch_data(self, batch_filename: str) -> pl.DataFrame:
        """Load batch data with OrderedDict caching using Polars."""
        if self.batch_cache is None:
            self.batch_cache = OrderedDict()
        
        if batch_filename not in self.batch_cache:
            # Load the batch file with Polars
            batch_path = self.batch_name_to_path[batch_filename]
            batch_data = pl.read_parquet(batch_path)
            
            # Group by event_id and aggregate like the notebook
            batch_data = batch_data.group_by("event_id").agg([
                pl.len().alias("count"),
                pl.col("sensor_id"),
                pl.col("time"),
                pl.col("charge"),
                pl.col("auxiliary"),
            ]).sort('event_id')
            
            # Store in cache
            self.batch_cache[batch_filename] = batch_data
            
            # Remove oldest entry if cache is full
            if len(self.batch_cache) > self.cache_size:
                oldest_key = next(iter(self.batch_cache))
                del self.batch_cache[oldest_key]
        
        return self.batch_cache[batch_filename]
    
    def _load_meta_data(self, batch_filename: str) -> pd.DataFrame:
        """Load metadata with OrderedDict caching."""
        if self.meta_cache is None:
            self.meta_cache = OrderedDict()
        
        if batch_filename not in self.meta_cache:
            # Extract batch number from filename
            batch_num = int(batch_filename.split('.')[0].split('_')[1])
            
            # Load corresponding metadata batch file directly
            meta_batch_file = os.path.join(self.meta_dir, f"train_meta_{batch_num}.parquet")
            if not os.path.exists(meta_batch_file):
                raise FileNotFoundError(f"Metadata file not found: {meta_batch_file}")
            
            meta_table = pq.read_table(meta_batch_file)
            batch_meta = meta_table.to_pandas()
            batch_meta = batch_meta.sort_values('event_id').reset_index(drop=True)
            
            # Store in cache
            self.meta_cache[batch_filename] = batch_meta
            
            # Remove oldest entry if cache is full
            if len(self.meta_cache) > self.cache_size:
                oldest_key = next(iter(self.meta_cache))
                del self.meta_cache[oldest_key]
        
        return self.meta_cache[batch_filename]
    
    def __getitem__(self, idx):
        """
        Get item using chunk-aware indexing.
        
        The idx is a global index across all batch files. We use bisect_right
        to find which batch file it belongs to, then get the local index within
        that batch file.
        """
        # Find which batch file this index belongs to
        batch_file_idx = bisect_right(self.chunk_cumsum, idx)
        batch_filename = self.batch_file_names[batch_file_idx]
        
        # Calculate local index within the batch file
        local_idx = idx - (self.chunk_cumsum[batch_file_idx - 1] if batch_file_idx > 0 else 0)
        
        # Load batch data and metadata
        batch_data = self._load_batch_data(batch_filename)
        batch_meta = self._load_meta_data(batch_filename)
        
        # Get event metadata
        event_meta = batch_meta.iloc[local_idx]
        event_id = event_meta['event_id']
        azimuth = event_meta['azimuth']
        zenith = event_meta['zenith']
        
        # Extract pulses for this event (using Polars row access like notebook)
        event_row = batch_data[int(local_idx)]
        sensor_ids = event_row['sensor_id'][0].to_numpy()
        times = event_row['time'][0].to_numpy()
        charges = event_row['charge'][0].to_numpy()
        auxiliary = event_row['auxiliary'][0].to_numpy()

        if self.use_summary_stats:
            # Summary stats mode: Group pulses by unique sensors and compute summary statistics
            unique_sensors = np.unique(sensor_ids)
            sensor_positions_list = []
            sensor_stats_list = []
            
            for sensor_id in unique_sensors:
                # Get all pulses for this sensor
                sensor_mask = sensor_ids == sensor_id
                sensor_times = times[sensor_mask]
                sensor_charges = charges[sensor_mask]
                
                # Compute summary stats using nt_summary_stats
                stats = nt_summary_stats.compute_summary_stats(sensor_times, sensor_charges)
                sensor_stats_list.append(stats)
                
                # Get sensor position
                sensor_pos = self.sensor_geometry[sensor_id]
                sensor_positions_list.append(sensor_pos)
            
            # Convert to numpy arrays
            sensor_positions = np.array(sensor_positions_list, dtype=np.float32)
            sensor_stats = np.array(sensor_stats_list, dtype=np.float32)
            
            # Create 4D coordinates: x, y, z, t (using first_pulse_time for t)
            pos = np.column_stack([
                sensor_positions[:, 0] / 500.0,  # x (normalized like original)
                sensor_positions[:, 1] / 500.0,  # y
                sensor_positions[:, 2] / 500.0,  # z
                (sensor_stats[:, 3] - 1e4) / 3e4  # first_pulse_time, normalized like original
            ]).astype(np.float32)
            
            # Use all 9 summary statistics as features (log transform like Prometheus)
            feats = np.log(sensor_stats + 1).astype(np.float32)
        else:
            # Pulse-level mode: Use original pulse-level processing
            # Get sensor positions
            sensor_positions = self.sensor_geometry[sensor_ids]
            
            # Apply preprocessing exactly like the notebook
            times_norm = (times - 1e4) / 3e4
            charges_norm = np.log10(charges) / 3.0
            auxiliary_norm = auxiliary.astype(np.float32) - 0.5
            
            # Build 4D space-time coordinates: [x, y, z, t]
            pos = np.column_stack([
                sensor_positions[:, 0] / 1000.0,  # x (notebook uses /500)
                sensor_positions[:, 1] / 1000.0,  # y
                sensor_positions[:, 2] / 1000.0,  # z
                times / 1000.0,                   # normalized time
            ]).astype(np.float32)
            
            # Build 3D features: [time, charge, auxiliary]
            feats = np.column_stack([
                times_norm,                      # normalized time
                charges_norm,                    # normalized charge
                auxiliary_norm                   # normalized auxiliary
            ]).astype(np.float32)
        
        # Build labels: convert spherical to Cartesian direction
        log_energy = 0.0  # Dummy energy value (IceCube doesn't provide energy)
        
        # Convert spherical to Cartesian direction
        dir_x = np.sin(zenith) * np.cos(azimuth)
        dir_y = np.sin(zenith) * np.sin(azimuth)
        dir_z = np.cos(zenith)
        
        # Labels: [log_energy, dir_x, dir_y, dir_z]
        labels = np.array([log_energy, dir_x, dir_y, dir_z], dtype=np.float32)

        return pos, feats, labels


def create_icecube_dataloaders(cfg):
    """Create train and validation dataloaders for IceCube data from config."""
    
    # Load sensor geometry (shared between train and validation)
    sensor_geometry = load_sensor_geometry(cfg['data_options']['sensor_geometry'])
    
    # Create train dataset
    train_batch_files = get_icecube_file_names(
        cfg['data_options']['train_batch_dir'], 
        cfg['data_options']['train_batch_ranges'],
        cfg['data_options']['shuffle_files']
    )
    train_dataset = IceCubeDataset(
        cfg['data_options']['train_meta_dir'],
        train_batch_files,
        sensor_geometry,
        cache_size=2,  # Cache up to 2 batch files per worker
        use_summary_stats=cfg['data_options'].get('use_summary_stats', False)
    )

    # Create validation dataset
    valid_batch_files = get_icecube_file_names(
        cfg['data_options']['valid_batch_dir'], 
        cfg['data_options']['valid_batch_ranges'],
        cfg['data_options']['shuffle_files']
    )
    valid_dataset = IceCubeDataset(
        cfg['data_options']['valid_meta_dir'],
        valid_batch_files,
        sensor_geometry,
        cache_size=2,  # Cache up to 2 batch files per worker
        use_summary_stats=cfg['data_options'].get('use_summary_stats', False)
    )
    
    # Create chunk-aware samplers for efficient caching
    train_sampler = RandomChunkSampler(
        train_dataset, 
        chunks=train_dataset.chunks
    )
    
    # For validation, use sequential sampling (no shuffling needed)
    val_sampler = None  # Will use default sequential sampling
    
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
        shuffle=False,  # Sequential validation
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
