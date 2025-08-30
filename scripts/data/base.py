"""
Unified dataloader factory for Neptune training.

Provides a single interface to create dataloaders for all supported dataset types:
- Memory-mapped datasets (Prometheus, IceCube) 
- Legacy Kaggle IceCube format

Auto-detects format and creates appropriate datasets with consistent interfaces.
"""

import torch
from torch.utils.data import DataLoader, RandomSampler
from typing import Dict, Any, Tuple

from .mmap_datasets import MmapDataset
from .kaggle_datasets import KaggleDataset  
from .utils import IrregularDataCollator, RandomChunkSampler


def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders based on config.
    
    Args:
        cfg: Configuration dictionary with data_options
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_options = cfg['data_options']
    dataloader_type = cfg.get('dataloader', 'prometheus')  # default to prometheus for compatibility
    
    # Check if using new single-path format with runtime splitting
    if 'data_path' in data_options:
        # New format: single path with runtime train/val splitting
        val_split = data_options.get('val_split', 0.2)
        split_seed = data_options.get('split_seed', 42)
        
        if dataloader_type == 'kaggle':
            raise ValueError("Runtime splitting not yet supported for Kaggle datasets. Use separate train_data_path/valid_data_path.")
        
        # Unified mmap format with splitting
        train_dataset = MmapDataset(
            mmap_paths=data_options['data_path'],
            use_summary_stats=data_options.get('use_summary_stats', True),
            split="train",
            val_split=val_split,
            split_seed=split_seed
        )
        
        valid_dataset = MmapDataset(
            mmap_paths=data_options['data_path'],
            use_summary_stats=data_options.get('use_summary_stats', True),
            split="val",
            val_split=val_split,
            split_seed=split_seed
        )
        
    else:
        # Legacy format: separate train/valid paths
        if dataloader_type == 'kaggle':
            # Legacy Kaggle IceCube format - load geometry and get batch files
            from .kaggle_datasets import load_sensor_geometry, get_icecube_file_names
            
            # Load sensor geometry (shared between train and validation)
            sensor_geometry = load_sensor_geometry(data_options['geometry_path'])
            
            # Get batch file lists
            train_batch_files = get_icecube_file_names(
                data_options['train_batch_dir'], 
                data_options['train_batch_ranges'],
                data_options.get('shuffle_files', False)
            )
            valid_batch_files = get_icecube_file_names(
                data_options['valid_batch_dir'], 
                data_options['valid_batch_ranges'],
                data_options.get('shuffle_files', False)
            )
            
            train_dataset = KaggleDataset(
                meta_dir=data_options['train_meta_dir'],
                batch_files=train_batch_files,
                sensor_geometry=sensor_geometry,
                use_summary_stats=data_options.get('use_summary_stats', True)
            )
            
            valid_dataset = KaggleDataset(
                meta_dir=data_options['valid_meta_dir'],
                batch_files=valid_batch_files, 
                sensor_geometry=sensor_geometry,
                use_summary_stats=data_options.get('use_summary_stats', True)
            )
            
        else:
            # Unified mmap format (prometheus or icecube) - legacy paths
            train_dataset = MmapDataset(
                mmap_paths=data_options['train_data_path'],
                use_summary_stats=data_options.get('use_summary_stats', True)
            )
            
            valid_dataset = MmapDataset(
                mmap_paths=data_options['valid_data_path'],
                use_summary_stats=data_options.get('use_summary_stats', True)
            )
    
    # Create samplers - use RandomChunkSampler for Kaggle datasets for performance
    if dataloader_type == 'kaggle':
        train_sampler = RandomChunkSampler(train_dataset, train_dataset.chunks)
    else:
        train_sampler = RandomSampler(train_dataset)
    val_sampler = None  # Sequential for validation
    
    # Create collator
    collate_fn = IrregularDataCollator()
    
    # Get batch size from data_options or training_options for backward compatibility
    batch_size = data_options.get('batch_size') or cfg['training_options']['batch_size']
    num_workers = data_options.get('num_workers') or cfg['training_options'].get('num_workers', 0)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    print(f"Created dataloaders: train={len(train_dataset):,}, val={len(valid_dataset):,}")
    
    return train_loader, val_loader


# Backward compatibility functions
def create_prometheus_dataloaders(cfg):
    """Backward compatibility wrapper for Prometheus dataloaders."""
    return create_dataloaders(cfg)


def create_icecube_dataloaders(cfg):
    """Backward compatibility wrapper for IceCube dataloaders.""" 
    return create_dataloaders(cfg)