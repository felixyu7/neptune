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
                                                   self.cfg['data_options']['use_latent_representation'],
                                                   self.cfg['data_options']['geo_dict_path'])
            
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'], 
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files']
        )
        self.valid_dataset = PrometheusDataset(valid_files,
                                               self.cfg['data_options']['use_latent_representation'],
                                               self.cfg['data_options']['geo_dict_path'])
            
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
    def __init__(self, files, use_latent_representation, geo_dict_path):
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
        if geo_dict_path is None:
            raise ValueError("geo_dict_path must be provided for PrometheusDataset")
        
        self.geo_dict = np.load(geo_dict_path, allow_pickle=True).item()
        
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
        
        if self.use_latent_representation:
            # Process latent representation data
            latents = event.latents.to_numpy()
            hs = latents[:, 0].astype(int) - 1
            ws = latents[:, 1].astype(int) - 1
            
            hits = np.array([event.photons.sensor_pos_x.to_numpy(),
                            event.photons.sensor_pos_y.to_numpy(),
                            event.photons.sensor_pos_z.to_numpy(),
                            event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
            spos_t = hits[np.argsort(hits[:,-1])]
            _, indices, counts = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
            first_hits = spos_t[indices]
            times = first_hits[:,-1]
            unique_coords = first_hits[:,:3]
            
            pos = []
            first_hit_time = []
            for i in range(hs.shape[0]):
                position = self.geo_dict[(hs[i]+1, ws[i]+1)]
                # find position in unique_coords
                idx = np.where((unique_coords == position).all(axis=1))[0][0]
                first_hit_time.append(times[idx])
                pos.append(position + np.array([0., 0., 2000.]))
            pos = np.array(pos) / 100.
            
            first_hit_time = np.array(first_hit_time) / 100.
            pos_t = np.column_stack([pos, first_hit_time])
            pos = pos_t
            
            feats = latents[:, 2:]
            
            pos = torch.from_numpy(pos).float()
            feats = torch.from_numpy(feats).float()
            label = torch.from_numpy(np.array([label]))
            return pos, feats, label
        else:
            # Process raw data
            pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                              event.photons.sensor_pos_y.to_numpy(),
                              event.photons.sensor_pos_z.to_numpy(),
                              event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
            
            # Bin times and aggregate spatial information
            time_bins = np.floor(pos_t[:,3] / 3) * 3
            unique_bins, inverse, feats = np.unique(time_bins, return_inverse=True, return_counts=True)
            binned_spatial = np.array([
                pos_t[inverse == i, :3].mean(axis=0)  # mean over spatial dimensions for bin i
                for i in range(len(unique_bins))
            ])
            pos_t = np.hstack([binned_spatial, unique_bins.reshape(-1, 1)])
            
            feats = np.log(feats + 1).reshape(-1, 1)
            pos_t = pos_t / 100.
            
            return torch.from_numpy(pos_t).float(), torch.from_numpy(feats).float(), torch.from_numpy(np.array([label])).float() 