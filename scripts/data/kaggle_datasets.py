"""
Legacy IceCube Kaggle dataset implementation for Neptune training.

Handles the original IceCube Kaggle competition format with separate batch files (pulse data)
and meta files (truth labels). Maintains exact compatibility with the original implementation.
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
from typing import List, Optional, Dict
from pathlib import Path
from collections import OrderedDict
from bisect import bisect_right

try:
    import nt_summary_stats
    HAS_SUMMARY_STATS = True
except ImportError:
    nt_summary_stats = None
    HAS_SUMMARY_STATS = False


def load_sensor_geometry(geometry_path: str) -> np.ndarray:
    """Load IceCube sensor geometry from CSV file."""
    geometry_df = pd.read_csv(geometry_path)
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


class KaggleDataset(torch.utils.data.Dataset):
    """
    Dataset class for legacy IceCube Kaggle format with separate batch and meta files.
    
    This matches the original IceCube implementation exactly:
    - Batch files contain pulse data (sensor_id, time, charge, auxiliary)
    - Meta files contain truth labels (event_id, zenith, azimuth)
    - Uses OrderedDict caching for efficient file loading
    - Supports RandomChunkSampler for optimal I/O performance
    """
    
    def __init__(self, 
                 meta_dir: str,
                 batch_files: List[str],
                 sensor_geometry: np.ndarray,
                 cache_size: int = 2,
                 use_summary_stats: bool = True):
        """
        Initialize Kaggle IceCube dataset exactly like the original.
        
        Args:
            meta_dir: Directory containing metadata parquet files
            batch_files: List of batch file paths  
            sensor_geometry: Sensor geometry array [n_sensors, 3]
            cache_size: Number of batch files to keep in memory
            use_summary_stats: Whether to use nt-summary-stats processing
        """
        self.meta_dir = meta_dir
        self.batch_files = batch_files
        self.batch_file_names = [Path(f).name for f in batch_files]  
        self.sensor_geometry = sensor_geometry
        self.cache_size = cache_size
        self.use_summary_stats = use_summary_stats and HAS_SUMMARY_STATS
        
        # Initialize cache
        self.batch_cache = None
        self.meta_cache = None
        
        # Calculate dataset size by reading metadata files
        self.chunks = []
        for batch_file in batch_files:
            # Extract batch number from filename  
            batch_num = int(Path(batch_file).name.split('.')[0].split('_')[1])
            
            # Find corresponding metadata batch file
            meta_batch_file = os.path.join(meta_dir, f"train_meta_{batch_num}.parquet")
            if os.path.exists(meta_batch_file):
                # Read just to get length
                meta_table = pq.read_table(meta_batch_file, columns=['event_id'])
                batch_meta_df = meta_table.to_pandas()
                self.chunks.append(len(batch_meta_df))
            else:
                raise FileNotFoundError(f"Metadata file not found: {meta_batch_file}")
        
        # Compute cumulative chunk sizes for indexing
        self.chunk_cumsum = np.cumsum(self.chunks)
        self.total_events = self.chunk_cumsum[-1]
        
        print(f"Dataset: {self.total_events} events across {len(self.chunks)} batch files")
        print(f"Chunk sizes: {self.chunks}")
        
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
            
            # Group by event_id and aggregate like the original
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
        Get item using chunk-aware indexing exactly like the original.
        
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
        
        # Extract pulses for this event (using Polars row access like original)
        event_row = batch_data[int(local_idx)]
        sensor_ids = event_row['sensor_id'][0].to_numpy()
        times = event_row['time'][0].to_numpy()
        charges = event_row['charge'][0].to_numpy()
        auxiliary = event_row['auxiliary'][0].to_numpy()

        if self.use_summary_stats:
            # Summary stats mode: Group pulses by unique sensors and compute summary statistics per sensor
            # This matches the original IceCube implementation
            unique_sensors = np.unique(sensor_ids)
            sensor_positions_list = []
            sensor_stats_list = []
            
            for sensor_id in unique_sensors:
                # Get all pulses for this sensor
                sensor_mask = sensor_ids == sensor_id
                sensor_times = times[sensor_mask]
                sensor_charges = charges[sensor_mask]
                
                # Compute summary stats using nt_summary_stats per sensor
                stats = nt_summary_stats.compute_summary_stats(sensor_times, sensor_charges)
                sensor_stats_list.append(stats)
                
                # Get sensor position
                sensor_pos = self.sensor_geometry[sensor_id]
                sensor_positions_list.append(sensor_pos)
            
            # Convert to numpy arrays
            sensor_positions = np.array(sensor_positions_list, dtype=np.float32)
            sensor_stats = np.array(sensor_stats_list, dtype=np.float32)
            
            # Create 4D coordinates: x, y, z, t (matching original normalization)
            pos = np.column_stack([
                sensor_positions[:, 0] / 500.0,  # x (normalized like original)
                sensor_positions[:, 1] / 500.0,  # y
                sensor_positions[:, 2] / 500.0,  # z
                (sensor_stats[:, 3] - 1e4) / 3e4  # first_pulse_time, normalized like original
            ]).astype(np.float32)
            
            # Use all 9 summary statistics as features (log transform like original)
            feats = np.log(sensor_stats + 1).astype(np.float32)
            
        else:
            # Pulse-level mode: Use original pulse-level processing exactly like the original
            # Get sensor positions
            sensor_positions = self.sensor_geometry[sensor_ids]
            
            # Apply preprocessing exactly like the original
            times_norm = (times - 1e4) / 3e4
            charges_norm = np.log10(charges) / 3.0
            auxiliary_norm = auxiliary.astype(np.float32) - 0.5
            
            # Build 4D space-time coordinates: [x, y, z, t]
            pos = np.column_stack([
                sensor_positions[:, 0] / 1000.0,  # x (original uses /500, but /1000 was in code)
                sensor_positions[:, 1] / 1000.0,  # y
                sensor_positions[:, 2] / 1000.0,  # z
                times / 1000.0,                   # normalized time
            ]).astype(np.float32)
            
            # Build 3D features: [time, charge, auxiliary] exactly like original
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
        
        # Labels: [log_energy, dir_x, dir_y, dir_z] - exactly 4 elements like original
        labels = np.array([log_energy, dir_x, dir_y, dir_z], dtype=np.float32)

        return pos, feats, labels