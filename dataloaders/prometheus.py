import os
import glob
import torch
import numpy as np
import lightning.pytorch as pl
import awkward as ak
import pyarrow.parquet as pq

from dataloaders.data_utils import IrregularDataCollator, ZippedDataCollator, get_file_names, ParquetFileSampler
 
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
                                                   self.cfg['data_options'].get('use_om2vec', False),
                                                   self.cfg['data_options'].get('use_summary_stats', False),
                                                   self.cfg['data_options'].get('add_noise', False),
                                                   self.cfg['data_options'].get('time_variance', 1.0),
                                                   self.cfg['data_options'].get('dropout_fraction', 0.1))
            if self.cfg['regularization_options']['coral_regularization']:
                real_files = get_file_names(
                    self.cfg['data_options']['real_data_files'],
                    self.cfg['data_options']['real_data_file_ranges'],
                    self.cfg['data_options']['shuffle_files']
                )
                self.real_dataset = PrometheusDataset(real_files,
                                                    self.cfg['data_options'].get('use_om2vec', False),
                                                    self.cfg['data_options'].get('use_summary_stats', False),
                                                    self.cfg['data_options'].get('add_noise', False),
                                                    self.cfg['data_options'].get('time_variance', 1.0),
                                                    self.cfg['data_options'].get('dropout_fraction', 0.1),
                                                    is_real_data=True)
             
        valid_files = get_file_names(
            self.cfg['data_options']['valid_data_files'],
            self.cfg['data_options']['valid_data_file_ranges'],
            self.cfg['data_options']['shuffle_files']
        )
        self.valid_dataset = PrometheusDataset(valid_files,
                                               self.cfg['data_options'].get('use_om2vec', False),
                                               self.cfg['data_options'].get('use_summary_stats', False),
                                               self.cfg['data_options'].get('add_noise', False),
                                               self.cfg['data_options'].get('time_variance', 1.0),
                                               self.cfg['data_options'].get('dropout_fraction', 0.1))

    def train_dataloader(self):
        """Returns the training dataloader."""
        train_sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_sampler=train_sampler,
                                                    collate_fn=IrregularDataCollator(),
                                                    pin_memory=True,
                                                    persistent_workers=True,
                                                    num_workers=self.cfg['training_options']['num_workers'])
        if self.cfg['regularization_options']['coral_regularization']:
            real_sampler = ParquetFileSampler(self.real_dataset, self.real_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
            real_dataloader = torch.utils.data.DataLoader(self.real_dataset,
                                                        batch_sampler=real_sampler,
                                                        collate_fn=IrregularDataCollator(has_labels=False),
                                                        pin_memory=True,
                                                        persistent_workers=True,
                                                        num_workers=self.cfg['training_options']['num_workers'])
            
            # Create a zipped dataloader
            return ZippedDataCollator([train_dataloader, real_dataloader])
        else:
            return train_dataloader
    
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
    def __init__(self, files, use_om2vec, use_summary_stats=False, add_noise=False, time_variance=1.0, dropout_fraction=0.1, is_real_data=False):
        self.files = files
        self.use_om2vec = use_om2vec
        self.use_summary_stats = use_summary_stats
        self.is_real_data = is_real_data
        self.add_noise = add_noise
        self.time_variance = time_variance
        self.dropout_fraction = dropout_fraction

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
            pos, feats, label
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
        
        if not self.is_real_data:
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
            ts = event.om2vec.summary_stats.to_numpy()[:, 3] / 1000. # convert to microseconds
            pos_t = np.concatenate((pos, ts[:, np.newaxis]), axis=1)
            feats = event.om2vec.latents.to_numpy()
        elif self.use_summary_stats: # using summary stats
            pos = np.array([event.om2vec.sensor_pos_x.to_numpy(),
                            event.om2vec.sensor_pos_y.to_numpy(),
                            event.om2vec.sensor_pos_z.to_numpy()]).T
            # first hit time from summary stats
            ts = event.om2vec.summary_stats.to_numpy()[:, 3] / 1000. # convert to microseconds
            pos_t = np.concatenate((pos, ts[:, np.newaxis]), axis=1)
            feats = event.om2vec.summary_stats.to_numpy()
            # log normalize
            feats = np.log(feats + 1)
        else: # using full pulse series
            pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                              event.photons.sensor_pos_y.to_numpy(),
                              event.photons.sensor_pos_z.to_numpy(),
                              event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
            
            if self.add_noise:
                # 1. Apply position-dependent photon efficiency noise (dropout)
                n_photons = pos_t.shape[0]
                # Normalize z to [0, 1] for linear dropout scaling
                z = pos_t[:, 2]
                z_min, z_max = z.min(), z.max()
                z_norm = (z - z_min) / (z_max - z_min + 1e-8)
                # Dropout fraction varies linearly from 0 at min(z) to self.dropout_fraction at max(z)
                dropout_probs = z_norm * self.dropout_fraction
                dropout_mask = np.random.random(n_photons) > dropout_probs
                pos_t = pos_t[dropout_mask]
            
            # Bin times (3 ns)
            time_bins = np.floor(pos_t[:, 3] / 3.0) * 3.0
            rows = np.column_stack((pos_t[:, :3], time_bins))    # shape (N_hits, 4)
            unique_rows, counts = np.unique(rows, axis=0, return_counts=True)
            pos_t = unique_rows.astype(np.float32)
            
            if self.add_noise:
                # 2. Add random timing noise to the binned time
                time_noise = np.random.normal(0, self.time_variance, size=pos_t.shape[0])
                pos_t[:, 3] += time_noise

            pos_t /= 1000.0
            feats = np.log(counts + 1).astype(np.float32)[:, None]   # (N_bins, 1)

        if self.is_real_data:
            return torch.from_numpy(pos_t).float(), torch.from_numpy(feats).float()
        else:
            return torch.from_numpy(pos_t).float(), torch.from_numpy(feats).float(), torch.from_numpy(np.array([label])).float()