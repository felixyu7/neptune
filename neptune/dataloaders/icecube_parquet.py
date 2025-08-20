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
    def __init__(self, 
                 cfg, 
                 field='mc_truth',
                 columnar=False):
        super().__init__()
        self.cfg = cfg
        self.field = field
        self.columnar = columnar
        
        self.data_options = self.cfg['data_options']
        self.training_options = self.cfg['training_options']
        
        self.use_pulse_series = self.data_options['use_pulse_series']
        self.use_latent_representation = self.data_options['use_latent_representation']
        
        if 'geo_inverse_dict_path' in self.data_options.keys():
            self.geo_inverse_dict_path = self.data_options['geo_inverse_dict_path']
        else:
            self.geo_inverse_dict_path = None
        
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
                self.data_options['train_data_files'], 
                self.data_options['train_data_file_ranges'],
                self.data_options['shuffle_files']
            )
            self.train_dataset = IceCube_Parquet_Dataset(train_files,
                                                         self.use_pulse_series,
                                                         self.use_latent_representation,
                                                         self.geo_inverse_dict_path)
            
        valid_files = get_file_names(
            self.data_options['valid_data_files'], 
            self.data_options['valid_data_file_ranges'],
            self.data_options['shuffle_files']
        )
        self.valid_dataset = IceCube_Parquet_Dataset(valid_files,
                                                     self.use_pulse_series,
                                                     self.use_latent_representation,
                                                     self.geo_inverse_dict_path)
            
    def train_dataloader(self):
        """Returns the training dataloader."""
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.training_options['batch_size'])
        collate_fn = IrregularDataCollator()
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                 batch_size = self.training_options['batch_size'], 
                                                 sampler=sampler,
                                                 collate_fn=collate_fn,
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 prefetch_factor=4,
                                                #  shuffle=True,
                                                 num_workers=self.training_options['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        """Returns the validation dataloader."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.training_options['batch_size'])
        collate_fn = IrregularDataCollator()
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           batch_size = self.training_options['batch_size'], 
                                           sampler=sampler,
                                           collate_fn=collate_fn,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=4,
                                           num_workers=self.training_options['num_workers'])

    def test_dataloader(self):
        """Returns the test dataloader (same as validation for now)."""
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths,  self.training_options['batch_size'])
        collate_fn = IrregularDataCollator()
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           batch_size = self.training_options['batch_size'], 
                                           sampler=sampler,
                                           collate_fn=collate_fn,
                                           pin_memory=True,
                                           persistent_workers=True,
                                           prefetch_factor=4,
                                           num_workers=self.training_options['num_workers'])
        
class IceCube_Parquet_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for IceCube parquet data.
    
    Handles loading data from parquet files and preprocessing it for the model.
    """
    def __init__(self, 
                 files,
                 use_pulse_series=False, 
                 use_latent_representation=False, 
                 geo_inverse_dict_path=None):
        self.files = files
        self.use_pulse_series = use_pulse_series
        self.use_latent_representation = use_latent_representation
        
        self.columns = [
            "primary_direction", "primary_energy", "morphology", "bundleness",
            "background", "visible_energy",
            "pulses.sensor_pos_x", "pulses.sensor_pos_y", "pulses.sensor_pos_z",
            "pulses.summary_stats", "pulses.pulse_times", "pulses.pulse_charges",
            # choose ONE of these branches depending on your mode:
            # "pulses.pulse_times", "pulses.pulse_charges", "pulses.aux",
            # or "pulses.latents",
            # or "pulses.summary_stats",
        ]

        # Count number of events in each file
        num_events = []
        self.rg_cumlens = []
        for file in self.files:
            pf = pq.ParquetFile(file)
            num_events.append(pf.metadata.num_rows)

            sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
            self.rg_cumlens.append(np.concatenate(([0], np.cumsum(np.array(sizes, dtype=np.int64)))))
        
        self._cur_file_idx = None
        self._cur_rg_idx = None
        self._cur_rg_data = None

        num_events = np.array(num_events)
        self.cumulative_lengths = np.concatenate(([0], np.cumsum(num_events)))
        self.dataset_size = self.cumulative_lengths[-1]

        # Default path for geometry dictionary - can be overridden
        # if geo_inverse_dict_path is None:
        #     raise ValueError("geo_inverse_dict_path must be provided for IceCube_Parquet_Dataset")
        # self.pos_to_str_sensor_dict = np.load(geo_inverse_dict_path, allow_pickle=True).item()
        
        self.current_file = ''
        self.current_data = None

    def _load_row_group(self, file_idx, rg_idx):
        if self._cur_file_idx == file_idx and self._cur_rg_idx == rg_idx:
            return
        path = self.files[file_idx]
        self._cur_rg_data = ak.from_parquet(path, columns=self.columns, row_groups={rg_idx})
        self._cur_file_idx, self._cur_rg_idx = file_idx, rg_idx
        
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
        """
        true_idx = i - self.cumulative_lengths[file_index]
        # Load file if it's not already loaded
        if self.current_file != self.files[file_index]:
            self.current_file = self.files[file_index]
            self.current_data = ak.from_parquet(self.files[file_index])
        
        event = self.current_data[true_idx]
        """
        idx_in_file = i - self.cumulative_lengths[file_index]
        rg_cum = self.rg_cumlens[file_index]
        rg_idx = np.searchsorted(rg_cum, idx_in_file+1) - 1
        idx_in_rg = idx_in_file - rg_cum[rg_idx]

        self._load_row_group(file_index, rg_idx)
        event = self._cur_rg_data[idx_in_rg]

        # Extract MC truth information
        direction = event.primary_direction.to_numpy()
        energy = event.primary_energy
        morphology = event.morphology
        bundleness = event.bundleness
        background = event.background
        visible_energy = event.visible_energy
        if visible_energy is None: visible_energy = 1e-9
        
        label = [np.log10(energy),
                 direction[0],
                 direction[1],
                 direction[2],
                 morphology,
                 bundleness,
                 background,
                 np.log10(visible_energy+1.0)]
        
        # Extract position information
        pos = np.array([event.pulses.sensor_pos_x.to_numpy(),
                        event.pulses.sensor_pos_y.to_numpy(),
                        event.pulses.sensor_pos_z.to_numpy()]).T
        #string_sensor_pos = [self.pos_to_str_sensor_dict[tuple(coord)] for coord in pos]
        #string_sensor_pos = np.array(string_sensor_pos) - 1
        
        if self.use_pulse_series:
            # Extract pulse series
            pulse_ts = event.pulses.pulse_times        # List of N lists of times
            pulse_qs = event.pulses.pulse_charges      # List of N lists of charges
            pulse_aux = event.pulses.aux
            sensor_x = event.pulses.sensor_pos_x       # 1‑D list/array of length N
            sensor_y = event.pulses.sensor_pos_y
            sensor_z = event.pulses.sensor_pos_z
            
            counts = np.array([len(ts) for ts in pulse_ts])

            flat_t = ak.flatten(pulse_ts).to_numpy()
            flat_q = ak.flatten(pulse_qs).to_numpy()
            flat_aux = ak.flatten(pulse_aux).to_numpy()

            flat_x = np.repeat(sensor_x.to_numpy(), counts)
            flat_y = np.repeat(sensor_y.to_numpy(), counts)
            flat_z = np.repeat(sensor_z.to_numpy(), counts)

            pos_t = np.asarray(np.column_stack((flat_x, flat_y, flat_z, flat_t))) / 100.0  # Scale down
            feats = np.column_stack((np.log(flat_q + 1), flat_aux))
        else:
            # Extract features
            if self.use_latent_representation:
                feats = event.pulses.latents.to_numpy()
            else:
                feats = event.pulses.summary_stats.to_numpy()
                feats = np.log(feats + 1)
                
            # Get first hit time on each sensor
            """
            first_hit_time = []
            for times in event.pulses.pulse_times:
                first_hit_time.append(min(times))
            first_hit_time = np.array(first_hit_time)
            """
            first_hit_time = ak.to_numpy(ak.min(event.pulses.pulse_times, axis=1))
        
            # Combine position and time
            pos_t = np.asarray(np.column_stack([pos, first_hit_time])) / 100.0  # Scale down
             
        pos_t = torch.from_numpy(pos_t).float()
        feats = torch.from_numpy(feats).float()
        label = torch.from_numpy(np.array([label])).float()
        
        return pos_t, feats, label
    
if __name__ == "__main__":
    test_fn = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MC/NuGen_22852/labels/000000-000999/NuGen_22852_labels_000999.parquet"
    ds = IceCube_Parquet_Dataset([test_fn])
    pos_t, feats, labels = ds[0]
    print("pos_t: ", pos_t)
    print("feats: ", feats)
    print("labels: ", labels)