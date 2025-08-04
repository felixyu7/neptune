"""
Prometheus Data Module for Neptune Examples.
Extracted and simplified from original implementation.
"""

import os
import glob
import torch
import numpy as np
import lightning.pytorch as pl
import awkward as ak
import pyarrow.parquet as pq
import random
from torch import Tensor
from torch.utils.data import Sampler, Dataset
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
        
        for file_index in file_order:
            start_idx = self.cumulative_lengths[file_index]
            end_idx = self.cumulative_lengths[file_index + 1]
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            for i in range(0, len(indices), self.batch_size):
                yield from indices[i:i+self.batch_size].tolist()

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
            labels_batch = torch.cat(labels, dim=0).float()
        else:
            labels_batch = torch.from_numpy(np.concatenate(labels, axis=0)).float()

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
        
    def __len__(self):
        return self.total_events
    
    def __getitem__(self, idx):
        # Determine which file this index belongs to
        file_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # Load data from the appropriate file
        data = pq.ParquetFile(self.files[file_idx])
        row_group = data.read_row_group(0)  # Assume single row group for simplicity
        
        # Convert to awkward array
        table = row_group.to_pandas()
        
        # Extract event data (simplified - adjust based on your data format)
        event_data = table.iloc[local_idx]
        
        # Extract coordinates (x, y, z, t)
        # Adjust these field names based on your actual Prometheus data format
        pos = np.column_stack([
            event_data.get('x', np.array([])),
            event_data.get('y', np.array([])), 
            event_data.get('z', np.array([])),
            event_data.get('t', np.array([]))
        ]).astype(np.float32)
        
        # Extract features (charge, auxiliary features)
        feats = np.column_stack([
            event_data.get('charge', np.array([])),
            # Add other feature columns as needed
        ]).astype(np.float32)
        
        # Extract labels (energy, direction, etc.)
        labels = np.array([
            event_data.get('energy', 0.0),      # Energy
            event_data.get('dir_x', 0.0),       # Direction x
            event_data.get('dir_y', 0.0),       # Direction y  
            event_data.get('dir_z', 1.0),       # Direction z
            # Add other labels as needed
        ]).astype(np.float32)
        
        return pos, feats, labels


class PrometheusDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for Prometheus dataset."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        """Set up train and validation datasets."""
        if self.cfg['training']:
            train_files = get_file_names(
                self.cfg['data_options']['train_data_files'], 
                self.cfg['data_options']['train_data_file_ranges'],
                self.cfg['data_options']['shuffle_files']
            )
            self.train_dataset = PrometheusDataset(
                train_files,
                self.cfg['data_options'].get('use_latent_representation', False),
                self.cfg['data_options'].get('geo_dict_path', None)
            )
            
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'], 
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files']
        )
        self.valid_dataset = PrometheusDataset(
            valid_files,
            self.cfg['data_options'].get('use_latent_representation', False),
            self.cfg['data_options'].get('geo_dict_path', None)
        )
            
    def train_dataloader(self):
        """Returns the training dataloader."""
        sampler = ParquetFileSampler(
            self.train_dataset, 
            self.train_dataset.cumulative_lengths, 
            self.cfg['training_options']['batch_size']
        )
        collate_fn = IrregularDataCollator()
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.cfg['training_options']['batch_size'], 
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.cfg['training_options']['num_workers']
        )
    
    def val_dataloader(self):
        """Returns the validation dataloader."""
        sampler = ParquetFileSampler(
            self.valid_dataset, 
            self.valid_dataset.cumulative_lengths, 
            self.cfg['training_options']['batch_size']
        )
        collate_fn = IrregularDataCollator()
        return torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_size=self.cfg['training_options']['batch_size'], 
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            num_workers=self.cfg['training_options']['num_workers']
        )

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        return self.val_dataloader()