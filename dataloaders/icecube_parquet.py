import os
import glob
import torch
import numpy as np
import lightning.pytorch as pl
import awkward as ak
import pyarrow.parquet as pq

from neptune.dataloaders.data_utils import IrregularDataCollator, get_file_names, ParquetFileSampler

class ICParquetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for IceCube parquet datasets.
    """
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field
        
    def prepare_data(self):
        """Called only once and on 1 GPU."""
        pass
    
    def setup(self, stage=None):
        """
        Called on each GPU separately - shards data to all GPUs.
        
        Sets up train and validation datasets.
        """
        if self.cfg['training']:
            train_files = get_file_names(self.cfg['data_options']['train_data_files'], self.cfg['data_options']['train_data_file_ranges'])
            self.train_dataset = IceCube_Parquet_Dataset(train_files,
                                                         self.cfg['data_options']['use_latent_representation'],
                                                         self.cfg['data_options']['geo_inverse_dict_path'])
            
        valid_files = get_file_names(self.cfg['data_options']['valid_data_files'], self.cfg['data_options']['valid_data_file_ranges'])
        self.valid_dataset = IceCube_Parquet_Dataset(valid_files,
                                                     self.cfg['data_options']['use_latent_representation'],
                                                     self.cfg['data_options']['geo_inverse_dict_path'])
            
    def train_dataloader(self):
        """Returns the training dataloader."""
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        collate_fn = IrregularDataCollator()
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                 batch_size = self.cfg['training_options']['batch_size'], 
                                                 sampler=sampler,
                                                 collate_fn=collate_fn,
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 num_workers=self.cfg['training_options']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        """Returns the validation dataloader."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        collate_fn = IrregularDataCollator()
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           batch_size = self.cfg['training_options']['batch_size'], 
                                           sampler=sampler,
                                           collate_fn=collate_fn,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           num_workers=self.cfg['training_options']['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        collate_fn = IrregularDataCollator()
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           batch_size = self.cfg['training_options']['batch_size'], 
                                           sampler=sampler,
                                           collate_fn=collate_fn,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           num_workers=self.cfg['training_options']['num_workers'])
        
class IceCube_Parquet_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for IceCube parquet data.
    
    Handles loading data from parquet files and preprocessing it for the model.
    """
    def __init__(self, files, use_latent_representation, geo_inverse_dict_path):
        self.files = files
        self.use_latent_representation = use_latent_representation
        
        # Count number of events in each file
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        num_events = np.array(num_events)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(num_events)))
        self.dataset_size = self.cumulative_lengths[-1]

        # Default path for geometry dictionary - can be overridden
        if geo_inverse_dict_path is None:
            raise ValueError("geo_inverse_dict_path must be provided for IceCube_Parquet_Dataset")
        
        self.pos_to_str_sensor_dict = np.load(geo_inverse_dict_path, allow_pickle=True).item()
        
        self.current_file = ''
        self.current_data = None
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, i):
        """
        Get an item from the dataset.
        
        Args:
            i: Index of the item
            
        Returns:
            pos: Position and time data
            feats: Feature data
            label: Label data
        """
        if i < 0 or i >= self.dataset_size:
            raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_lengths, i+1) - 1
        true_idx = i - self.cumulative_lengths[file_index]
                
        # Load file if it's not already loaded
        if self.current_file != self.files[file_index]:
            self.current_file = self.files[file_index]
            self.current_data = ak.from_parquet(self.files[file_index])
        
        event = self.current_data[true_idx]
        
        # Extract MC truth information
        direction = event.mc_truth.mu_direction.to_numpy()
        energy = event.mc_truth.mu_energy
        label = [np.log10(energy), direction[0], direction[1], direction[2]]
        
        # Extract position information
        pos = np.array([event.pulses.sensor_pos_x.to_numpy(),
                        event.pulses.sensor_pos_y.to_numpy(),
                        event.pulses.sensor_pos_z.to_numpy()]).T
        string_sensor_pos = [self.pos_to_str_sensor_dict[tuple(coord)] for coord in pos]
        string_sensor_pos = np.array(string_sensor_pos) - 1
        
        # Extract features
        if self.use_latent_representation:
            feats = event.pulses.latents.to_numpy()
        else:
            feats = event.pulses.summary_stats.to_numpy()
            feats = np.log(feats + 1)
            
        # Get first hit time on each sensor
        first_hit_time = []
        for times in event.pulses.pulse_times:
            first_hit_time.append(min(times))
        first_hit_time = np.array(first_hit_time)
        
        # Combine position and time
        pos_t = np.column_stack([pos, first_hit_time])
        pos_t /= 100.  # Scale down
             
        pos_t = torch.from_numpy(pos_t).float()
        feats = torch.from_numpy(feats).float()
        label = torch.from_numpy(np.array([label])).float()
        
        return pos_t, feats, label 