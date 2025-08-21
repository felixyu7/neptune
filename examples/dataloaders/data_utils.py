"""
Shared data utilities for all dataloaders.

Contains common components used by different dataset implementations:
- Sampling strategies for parquet files
- Collation functions for irregular data
- Coordinate batching utilities
"""

import os
import glob
import torch
import numpy as np
import random
from torch import Tensor
from torch.utils.data import Sampler, Dataset, DataLoader
from typing import List, Tuple, Optional, Union, Iterator, Sized


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
        """
        Iterate through files in random order, but keep all events from a file
        together (randomly permuted within the file) before moving to the next.
        This preserves file-level locality and avoids thrashing the parquet
        reader cache while still providing per-epoch randomness.
        """
        n_files = len(self.cumulative_lengths) - 1
        file_order = np.random.permutation(n_files)
        for file_index in file_order:
            start_idx = self.cumulative_lengths[file_index]
            end_idx = self.cumulative_lengths[file_index + 1]
            # Random permutation of indices within this file
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            # Yield indices sequentially (DataLoader will batch them)
            for idx in indices:
                yield idx

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


class RandomChunkSampler(Sampler[int]):
    """
    Random sampler that processes chunks (batch files) completely before moving to next.
    
    This sampler ensures that:
    1. Chunks are processed in random order (for good training randomness)  
    2. All events within a chunk are processed before moving to next chunk
    3. Events within each chunk are shuffled randomly
    
    This maximizes cache efficiency since the dataloader can load one batch file
    and process all its events before needing to load the next file.
    """

    def __init__(self, 
                 data_source: Sized, 
                 chunks: list,
                 num_samples: Optional[int] = None,
                 generator=None) -> None:
        """
        Initialize the RandomChunkSampler.
        
        Args:
            data_source: The dataset
            chunks: List of chunk sizes (number of events in each batch file)
            num_samples: Number of samples to draw (defaults to dataset size)
            generator: Random number generator
        """
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.chunks = chunks

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer, got {self.num_samples}")

    @property
    def num_samples(self) -> int:
        """Return number of samples to draw."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices in chunk-aware order.
        
        Strategy:
        1. Randomize order of chunks/batch files
        2. For each chunk, shuffle events within that chunk  
        3. Yield all events from chunk before moving to next
        """
        cumsum = np.cumsum(self.chunks)
        
        # Setup generator
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # Randomize chunk order
        chunk_list = torch.randperm(len(self.chunks), generator=generator).tolist()
        
        # Process chunks in random order, events within chunks shuffled
        for chunk_idx in chunk_list:
            chunk_len = self.chunks[chunk_idx]
            offset = cumsum[chunk_idx - 1] if chunk_idx > 0 else 0
            
            # Generate shuffled indices for this chunk
            chunk_indices = offset + torch.randperm(chunk_len, generator=generator)
            
            # Yield all indices from this chunk
            yield from chunk_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples
    
def group_hits_by_window(times: np.ndarray, charges: np.ndarray, 
                         window_ns: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress arrays by grouping values that lie within `window_ns` of the first element of each group.
    
    This matches the optimized grouping logic used in Mercury and the training dataloader.
    
    Parameters
    ----------
    times : array_like
        Input hit times (any order, any numeric dtype).
    charges : array_like  
        Input hit charges corresponding to times.
    window_ns : float
        Width of the inclusion window (inclusive on the right).
        
    Returns
    -------
    grouped_times : ndarray
        Representative values (the first element in each group).
    grouped_charges : ndarray
        Sum of charges in each group.
    """
    if len(times) == 0:
        return np.array([]), np.array([])
    
    # Sort both arrays by time
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_charges = charges[sort_idx]
    
    # endpoints[i] == first index where value > sorted_times[i] + window_ns
    endpoints = np.searchsorted(sorted_times,
                               sorted_times + window_ns,
                               side='right')

    reps, charge_sums = [], []
    i = 0
    n = sorted_times.size
    while i < n:
        reps.append(sorted_times[i])
        j = endpoints[i]       # jump to the start of the next group
        charge_sums.append(np.sum(sorted_charges[i:j]))
        i = j

    return np.asarray(reps), np.asarray(charge_sums)