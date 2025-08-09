import os
import glob
import torch
import numpy as np
import awkward as ak
import pyarrow.parquet as pq
import random
import nt_summary_stats
from torch import Tensor
from torch.utils.data import Sampler, Dataset, DataLoader
from typing import List, Tuple, Optional, Union


def get_file_names(data_dirs: List[str], 
                   ranges: List[List[int]], 
                   shuffle_files: bool = False) -> List[str]:
    """Get file names from directories within specified ranges."""
    filtered_files = []
    for i, directory in enumerate(data_dirs):
        all_files = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        if shuffle_files:
            random.shuffle(all_files)
        file_range = ranges[i]
        filtered_files.extend(
            all_files[file_range[0]:file_range[1]]
        )
    return sorted(filtered_files)


class ParquetFileSampler(Sampler):
    """Custom sampler for parquet files that respects file boundaries during batching."""
    
    def __init__(self, 
                 data_source: Dataset, 
                 cumulative_lengths: np.ndarray, 
                 batch_size: int):
       super().__init__(data_source)
       self.data_source = data_source
       self.cumulative_lengths = cumulative_lengths
       self.batch_size = batch_size

    def __iter__(self):
        n_files = len(self.cumulative_lengths) - 1
        file_order = np.random.permutation(n_files)
        
        # Collect all indices with their file association
        all_indices_with_files = []
        for file_index in file_order:
            start_idx = self.cumulative_lengths[file_index]
            end_idx = self.cumulative_lengths[file_index + 1]
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            all_indices_with_files.extend(indices.tolist())
        
        # Shuffle all indices across files for maximum diversity
        np.random.shuffle(all_indices_with_files)
        
        # Yield in batch sizes
        for i in range(0, len(all_indices_with_files), self.batch_size):
            batch = all_indices_with_files[i:i+self.batch_size]
            yield from batch

    def __len__(self) -> int:
       return len(self.data_source)


def batched_coordinates(list_of_coords: List[Tensor]) -> Tensor:
    """
    Convert list of coordinate tensors to a single batched tensor.
    Adds batch index as first column.
    """
    batched_coords = []
    for batch_idx, coords in enumerate(list_of_coords):
        # Ensure coords is a tensor
        if not torch.is_tensor(coords):
            coords = torch.as_tensor(coords, dtype=torch.float32)
        
        # Add batch index as first column
        batch_indices = torch.full((coords.shape[0], 1), batch_idx, dtype=coords.dtype, device=coords.device)
        batched_coord = torch.cat([batch_indices, coords], dim=1)
        batched_coords.append(batched_coord)
    
    return torch.cat(batched_coords, dim=0)


class IrregularDataCollator(object):
    """Collator for irregular point cloud data."""
    
    def __call__(self, batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        pos, feats, labels = list(zip(*batch))
        
        # Convert to tensors if needed
        pos_tensors = [torch.as_tensor(p, dtype=torch.float32) if not torch.is_tensor(p) else p for p in pos]
        
        # Create batched coordinates
        bcoords = batched_coordinates(pos_tensors)
        
        # Concatenate features and labels
        if isinstance(feats[0], torch.Tensor):
            feats_batch = torch.cat(feats, dim=0).float()
        else:
            feats_batch = torch.from_numpy(np.concatenate(feats, axis=0)).float()

        if isinstance(labels[0], torch.Tensor):
            labels_batch = torch.stack(labels, dim=0).float()
        else:
            labels_batch = torch.from_numpy(np.stack(labels, axis=0)).float()

        return bcoords, feats_batch, labels_batch


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
        
        # sensor_positions: (N_sensors, 3) - x, y, z coordinates
        # sensor_stats: (N_sensors, 9) - [time_mean, time_std, time_min, time_max, n_hits, charge_sum, charge_mean, charge_std, duration]
        
        # Create 4D coordinates: x, y, z, t (using time_mean for t)
        pos = np.column_stack([
            sensor_positions[:, 0],  # x
            sensor_positions[:, 1],  # y
            sensor_positions[:, 2],  # z
            sensor_stats[:, 0]       # time_mean as t coordinate
        ]).astype(np.float32)
        
        # Use all 9 summary statistics as features
        feats = sensor_stats.astype(np.float32)  # [time_mean, time_std, time_min, time_max, n_hits, charge_sum, charge_mean, charge_std, duration]
        
        # Extract labels from mc_truth
        initial_zenith = mc_truth['initial_state_zenith']
        initial_azimuth = mc_truth['initial_state_azimuth']
        initial_energy = mc_truth.get('initial_state_energy', 0.0)  # Default to 0 if not available
        
        # Convert spherical to Cartesian direction
        dir_x = np.sin(initial_zenith) * np.cos(initial_azimuth)
        dir_y = np.sin(initial_zenith) * np.sin(initial_azimuth)
        dir_z = np.cos(initial_zenith)
        
        # Format labels as [energy, dir_x, dir_y, dir_z] to match Lightning version
        labels = np.array([initial_energy, dir_x, dir_y, dir_z], dtype=np.float32)
        
        return pos, feats, labels


def create_dataloaders(cfg):
    """Create train and validation dataloaders from config."""
    
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training_options']['batch_size'], 
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg['training_options']['num_workers']
    )
    
    val_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg['training_options']['batch_size'], 
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=cfg['training_options']['num_workers']
    )
    
    return train_loader, val_loader