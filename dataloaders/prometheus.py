import os
import glob
import torch
import numpy as np
import lightning.pytorch as pl
import awkward as ak
import pyarrow.parquet as pq

from neptune.dataloaders.data_utils import IrregularDataCollator, get_file_names, ParquetFileSampler

class PrometheusDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for Prometheus dataset.
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
            train_files = get_file_names(
                self.cfg['data_options']['train_data_files'], 
                self.cfg['data_options']['train_data_file_ranges'],
                self.cfg['data_options']['shuffle_files']
            )
            self.train_dataset = PrometheusDataset(train_files,
                                                   self.cfg['data_options']['use_om2vec'])
            
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'], 
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files']
        )
        self.valid_dataset = PrometheusDataset(valid_files,
                                               self.cfg['data_options']['use_om2vec'])

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
        
class PrometheusDataset(torch.utils.data.Dataset):
    """
    Dataset class for Prometheus data.
    
    Handles loading data from parquet files and preprocessing it for the model.
    """
    def __init__(self, files, use_om2vec):
        self.files = files
        self.use_om2vec = use_om2vec

        # Count number of events in each file
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        num_events = np.array(num_events)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(num_events)))
        self.dataset_size = self.cumulative_lengths[-1]
        
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
        zenith = event.mc_truth.initial_state_zenith
        azimuth = event.mc_truth.initial_state_azimuth

        dir_x = np.sin(zenith) * np.cos(azimuth)
        dir_y = np.sin(zenith) * np.sin(azimuth)
        dir_z = np.cos(zenith)
        
        log_energy = np.log10(event.mc_truth.initial_state_energy)
        
        label = [log_energy, 
                 dir_x,
                 dir_y,
                 dir_z]
        
        if self.use_om2vec:
            pos = np.array([event.om2vec.sensor_pos_x.to_numpy(),
                            event.om2vec.sensor_pos_y.to_numpy(),
                            event.om2vec.sensor_pos_z.to_numpy()]).T
            # first hit time from summary stats
            ts = event.om2vec.summary_stats.to_numpy()[:, 3]
            pos_t = np.concatenate((pos, ts[:, np.newaxis]), axis=1)
            feats = event.om2vec.latents.to_numpy()
        else: # using summary stats
            pos = np.array([event.om2vec.sensor_pos_x.to_numpy(),
                            event.om2vec.sensor_pos_y.to_numpy(),
                            event.om2vec.sensor_pos_z.to_numpy()]).T
            # first hit time from summary stats
            ts = event.om2vec.summary_stats.to_numpy()[:, 3]
            pos_t = np.concatenate((pos, ts[:, np.newaxis]), axis=1)
            feats = event.om2vec.summary_stats.to_numpy()
            
        return torch.from_numpy(pos_t).float(), torch.from_numpy(feats).float(), torch.from_numpy(np.array([label])).float() 