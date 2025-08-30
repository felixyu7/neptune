"""
Utilities for memory-mapped dataset loading and data processing.
Integrates the unified mmap format from nt-mmap-converter.
"""

import os
import pickle
import struct
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.sampler import Sampler
from typing import Tuple, List, Dict, Any, Optional, Iterator, Sized, Union


def load_ntmmap(input_path: str) -> Tuple[np.memmap, np.memmap, np.dtype]:
    """
    Load memory-mapped files with automatic dtype detection (new format with headers).
    Auto-detects source type (Prometheus vs IceCube) from stored event dtype.
    
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


class IrregularDataCollator:
    """
    Collates variable-length point cloud data into batches.
    Handles different event sizes by creating batch-indexed coordinate arrays.
    """
    
    def __init__(self, max_points_per_event: int = 5000):
        self.max_points_per_event = max_points_per_event
    
    def __call__(self, batch: List[Union[Tuple[Any, Any, Any], Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate a batch of variable-length events into concatenated tensors suitable for point-cloud models.
        
        Input sample format (supported):
        - Tuple: (coords [N,4], features [N,F], labels [L])
        - Dict: keys 'coords', 'features', and one of 'labels' | 'targets' | 'target'
        
        Output:
        - coords_b: [sum_i N_i, 1+4] where column 0 is batch index, then x,y,z,t
        - features_b: [sum_i N_i, F]
        - labels_b: [B, L]
        """
        batch_coords: List[np.ndarray] = []
        batch_features: List[np.ndarray] = []
        batch_labels: List[np.ndarray] = []
        
        for batch_idx, event in enumerate(batch):
            if isinstance(event, dict):
                coords = event.get('coords')
                features = event.get('features')
                labels = event.get('labels', None)
                if labels is None:
                    labels = event.get('targets', None)
                if labels is None:
                    labels = event.get('target', None)
                if coords is None or features is None or labels is None:
                    raise KeyError("Each dict sample must include 'coords', 'features', and 'labels'/'targets'/'target'")
            else:
                try:
                    coords, features, labels = event
                except Exception as e:
                    raise TypeError(f"Each sample must be a tuple (coords, features, labels) or a dict. Got: {type(event)}") from e
            
            coords = np.asarray(coords, dtype=np.float32)
            features = np.asarray(features, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            
            # Subsample if too many points
            if coords.shape[0] > self.max_points_per_event:
                indices = np.random.choice(coords.shape[0], self.max_points_per_event, replace=False)
                coords = coords[indices]
                features = features[indices]
            
            # Add batch index as first column to coords
            batch_indices = np.full((coords.shape[0], 1), batch_idx, dtype=np.float32)
            coords_with_batch = np.concatenate([batch_indices, coords], axis=1)  # [N, 1+4]
            
            batch_coords.append(coords_with_batch)
            batch_features.append(features)
            batch_labels.append(labels)
        
        # Concatenate along point dimension; stack labels per event
        if not batch_coords:
            # Should not happen with DataLoader, but keep safe defaults
            coords_b = torch.empty((0, 5), dtype=torch.float32)
            features_b = torch.empty((0, 0), dtype=torch.float32)
            labels_b = torch.empty((0,), dtype=torch.float32)
        else:
            coords_b = torch.from_numpy(np.vstack(batch_coords)).float()
            features_b = torch.from_numpy(np.vstack(batch_features)).float()
            # Ensure labels shape [B, L]
            try:
                labels_np = np.stack(batch_labels, axis=0).astype(np.float32)
            except ValueError:
                # Fallback if labels already arrays with consistent shapes but np.stack failed due to object dtype
                labels_np = np.array(batch_labels, dtype=np.float32)
            labels_b = torch.from_numpy(labels_np).float()
        
        # Return tuple to match trainer expectations
        return coords_b, features_b, labels_b


def batched_coordinates(coords: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Add batch indices to coordinate arrays.
    
    Args:
        coords: [N, D] coordinate array
        batch_size: Size of batch
        
    Returns:
        [N, D+1] coordinates with batch indices in first column
    """
    batch_indices = np.arange(len(coords)) // batch_size
    return np.column_stack([batch_indices, coords])



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