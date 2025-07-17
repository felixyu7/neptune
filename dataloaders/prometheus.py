import os
import glob
import torch
import numpy as np
import lightning.pytorch as pl
import awkward as ak
import pyarrow.parquet as pq

from dataloaders.data_utils import IrregularDataCollator, get_file_names, ParquetFileSampler

import mercury
import nt_summary_stats as nss

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
        self.valid_dataset.__getitem__(0)

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
    def __init__(self, files, use_om2vec, use_summary_stats=False, add_noise=False, time_variance=1.0, dropout_fraction=0.1):
        self.files = files
        self.use_om2vec = use_om2vec
        self.use_summary_stats = use_summary_stats
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
        
        if self.use_om2vec:
            self.mercury_model = mercury.Mercury(precision='bfloat16', sensor_batch_size=256)
            # compile the model if possible
            try:
                self.mercury_model = torch.compile(self.mercury_model, mode='reduce-overhead')
            except Exception as e:
                print(f"Warning: failed to compile Mercury model: {e}")
        
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
        
        # Extract MC truth information
        zenith = event.mc_truth.initial_state_zenith
        azimuth = event.mc_truth.initial_state_azimuth

        dir_x = np.sin(zenith) * np.cos(azimuth)
        dir_y = np.sin(zenith) * np.sin(azimuth)
        dir_z = np.cos(zenith)
        
        log_energy = np.log10(event.mc_truth.initial_state_energy)
        
        # class labels for single vs. double cascade
        pdg_code = event.mc_truth.final_state_type[0]
        if pdg_code == 15: # tau minus
            class_label = 1
        else:
            class_label = 0
            
        label = [log_energy,
                 dir_x,
                 dir_y,
                 dir_z,
                 class_label]
        
        if self.use_om2vec:
            pos = np.array([event.mercury.sensor_pos_x.to_numpy(),
                            event.mercury.sensor_pos_y.to_numpy(),
                            event.mercury.sensor_pos_z.to_numpy()]).T
            # first hit time from summary stats
            ts = event.mercury.summary_stats.to_numpy()[:, 3] / 1000. # convert to microseconds
            pos_t = np.concatenate((pos, ts[:, np.newaxis]), axis=1)
            feats = event.mercury.latents.to_numpy()
        elif self.use_summary_stats: # using summary stats
            pos, summary_stats = nss.process_prometheus_event(event, grouping_window_ns=2.0)
            pos_t = np.array([pos[:, 0], pos[:, 1], pos[:, 2], summary_stats[:, 3]]).T
            pos_t /= 1000.0  # convert to metres
            feats = np.log(summary_stats + 1)
        else: # using full pulse series
            pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                              event.photons.sensor_pos_y.to_numpy(),
                              event.photons.sensor_pos_z.to_numpy(),
                              event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
            
            if self.add_noise:
                # 1. Apply position-dependent photon efficiency noise (dropout)
                n_photons = pos_t.shape[0]
                dropout_mask = np.random.random(n_photons) > self.dropout_fraction
                pos_t = pos_t[dropout_mask]

                # 2. Add random timing noise to the binned time
                time_noise = np.random.normal(0, self.time_variance, size=pos_t.shape[0])
                pos_t[:, 3] += time_noise
                
                # 3. dark noise hits
                geo = load_dom_xyz('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/prometheus/resources/geofiles/icecube.geo')
                dark_rate = 100.0
                noise_pos_t = generate_noise_hits(geo, dark_rate, (pos_t[:,3].min(), pos_t[:,3].max() + 1000.))
                
                # Combine noise hits with the existing hits
                pos_t = np.concatenate((pos_t, noise_pos_t), axis=0)
                  
            # Bin times (3 ns)
            time_bins = np.floor(pos_t[:, 3] / 3.0) * 3.0
            rows = np.column_stack((pos_t[:, :3], time_bins))    # shape (N_hits, 4)
            unique_rows, counts = np.unique(rows, axis=0, return_counts=True)
            pos_t = unique_rows.astype(np.float32)

            pos_t /= 1000.0
            feats = np.log(counts + 1).astype(np.float32)[:, None]   # (N_bins, 1)

        return torch.from_numpy(pos_t).float(), torch.from_numpy(feats).float(), torch.from_numpy(np.array([label])).float()
    
def generate_noise_hits(dom_xyz, rate_Hz, t_window, seed=None):
    """
    Parameters
    ----------
    dom_xyz : (5160, 3) float32/64
        DOM positions in metres.
    rate_Hz : float
        Dark rate per DOM (∼500 Hz for IceCube).
    t_window : tuple(float, float)
        (t_start, t_end) in nanoseconds.
    seed : int or None
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    delta_t = t_window[1] - t_window[0]

    # 1) number of noise hits for every DOM in one go
    n_by_dom = rng.poisson(rate_Hz / 1e9 * delta_t, size=dom_xyz.shape[0])
    tot_n = n_by_dom.sum()
    if tot_n == 0:
        return np.empty((0, 4), dtype=float)

    # 2) build (x,y,z) part by repeating rows
    xyz_noise = np.repeat(dom_xyz, n_by_dom, axis=0)

    # 3) build time part
    t_offsets = rng.uniform(0, delta_t, size=tot_n)
    t_noise = t_offsets + t_window[0]

    # 4) glue
    noise_hits = np.column_stack((xyz_noise, t_noise))
    return noise_hits

from pathlib import Path
def load_dom_xyz(path: str | Path, *, dtype=np.float64) -> np.ndarray:
    """
    Parse an IceCube-style geometry text file and return an (N, 3) array
    with DOM coordinates in metres.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the geometry file.
    dtype : NumPy dtype, optional
        Floating type for the output array.  Default is float64.

    Returns
    -------
    xyz : ndarray of shape (N, 3)
        Each row is (x, y, z) in metres.

    Raises
    ------
    ValueError
        If the '### Modules ###' section cannot be located.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    coords = []
    with path.open() as fh:
        in_modules = False
        for ln in fh:
            ln = ln.strip()
            if not in_modules:
                # Look for the marker that starts the useful block
                if ln.startswith("### Modules"):
                    in_modules = True
                continue

            if not ln or ln.startswith("#"):
                # Skip blank lines or stray comments
                continue

            # We expect at least 3 floats; anything beyond is ignored
            parts = ln.split()
            try:
                x, y, z = map(float, parts[:3])
                coords.append((x, y, z))
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse line after '### Modules ###':\n{ln}"
                ) from exc

    if not in_modules:
        raise ValueError(
            "The file does not contain a '### Modules ###' section; "
            "cannot locate DOM coordinates."
        )

    return np.asarray(coords, dtype=dtype)