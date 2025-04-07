import torch
import numpy as np
import glob
import os
import random
from torch import Tensor
from torch.utils.data import Sampler, Dataset
from typing import List, Tuple, Optional, Union, Any

def get_file_names(data_dirs: List[str], ranges: List[List[int]]) -> List[str]:
    """
    Get file names from directories within specified ranges.
    
    Args:
        data_dirs: List of directories to search for files
        ranges: List of [start, end] ranges for each directory
        
    Returns:
        List of file paths
    """
    filtered_files = []
    for i, directory in enumerate(data_dirs):
        all_files = sorted(glob.glob(os.path.join(directory, '*.parquet')))
        file_range = ranges[i]
        filtered_files.extend(
            all_files[file_range[0]:file_range[1]]
        )
    return sorted(filtered_files)

class ParquetFileSampler(Sampler):
    """
    Custom sampler for parquet files that respects file boundaries during batching.
    
    This sampler first selects files in random order, then for each file,
    it shuffles the indices and yields batches from that file before moving
    to the next file.
    """
    def __init__(self, data_source: Dataset, cumulative_lengths: np.ndarray, batch_size: int):
       super().__init__(data_source) # Call Sampler's __init__
       self.data_source = data_source
       self.cumulative_lengths = cumulative_lengths  # expects array starting with 0, then cumulative sums
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

class IrregularDataCollator(object):
    """
    Collator for irregular point cloud data.
    
    Handles batching of variable-sized point clouds by adding batch index
    to coordinates.
    """
    def __init__(self):
        pass
        
    def __call__(self, batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        pos, feats, labels = list(zip(*batch))
        
        # Ensure pos is a list of Tensors before passing to batched_coordinates
        pos_tensors = [torch.as_tensor(p) if not torch.is_tensor(p) else p for p in pos]
        
        bcoords = batched_coordinates(pos_tensors)
        
        # Ensure feats and labels are concatenated correctly (assuming they are numpy arrays or tensors)
        if isinstance(feats[0], torch.Tensor):
            feats_batch = torch.cat(feats, dim=0).float()
        else:
            feats_batch = torch.from_numpy(np.concatenate(feats, axis=0)).float()

        if isinstance(labels[0], torch.Tensor):
            labels_batch = torch.cat(labels, dim=0).float()
        else:
            labels_batch = torch.from_numpy(np.concatenate(labels, axis=0)).float()

        return bcoords, feats_batch, labels_batch
        
def batched_coordinates(list_of_coords: List[Tensor], 
                      dtype: Optional[torch.dtype] = None, 
                      device: Optional[Union[torch.device, str]] = None, 
                      requires_grad: bool = False) -> Tensor:
    """
    Convert a list of coordinate tensors to a batched tensor with batch indices.
    
    Args:
        list_of_coords: List of tensors, each of shape [Mi, D]
        dtype: Output data type
        device: Output device
        requires_grad: Whether output requires gradients
        
    Returns:
        Batched coordinates tensor of shape [sum(Mi), D+1] with batch indices prepended
    """
    # Infer the backend (NumPy or PyTorch) from the first element
    first = list_of_coords[0]
    # is_torch = torch.is_tensor(first) # No longer needed, assume torch tensors

    # if is_torch: # Assume torch tensor input
    # If no dtype provided, use the dtype of the first tensor
    if dtype is None:
        dtype = first.dtype
    # If no device provided, use the device of the first tensor
    if device is None:
        device = first.device

    # Collect the list of (batch_index, coordinates) pairs
    cat_list = []
    for b, coords in enumerate(list_of_coords):
        # Convert coords to desired dtype/device if needed
        coords = coords.to(device=device, dtype=dtype)

        # Create a column of batch indices
        b_idx = torch.full((coords.shape[0], 1), b, dtype=dtype, device=device,
                           requires_grad=requires_grad)

        # Concatenate [batch_index_column, coords]
        out_coords = torch.cat((b_idx, coords), dim=1)
        cat_list.append(out_coords)

    # Final concatenation
    out = torch.cat(cat_list, dim=0)
    # If the user wants gradient tracking on the final tensor:
    if requires_grad and not out.requires_grad:
        out.requires_grad_(True)

    return out

    # else: # Remove NumPy logic as we expect tensors now
    #     # NumPy logic:
    #     # If no dtype provided, use the dtype of the first array
    #     if dtype is None:
    #         dtype = first.dtype

    #     cat_list = []
    #     for b, coords in enumerate(list_of_coords):
    #         coords = coords.astype(dtype, copy=False)

    #         # Create a column of batch indices
    #         b_idx = np.full((coords.shape[0], 1), b, dtype=dtype)

    #         # Concatenate [batch_index_column, coords]
    #         out_coords = np.concatenate([b_idx, coords], axis=1)
    #         cat_list.append(out_coords)

    #     # Final concatenation
    #     out = np.concatenate(cat_list, axis=0)

    #     return out 