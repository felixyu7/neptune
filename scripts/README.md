# Neptune Training

Minimal training scripts for Neptune neutrino event reconstruction. Pure PyTorch, unified dataloaders, and built‑in summary stats.

## Install

```bash
pip install -e .
pip install pyyaml tqdm scipy  # core script deps

# Optional
pip install wandb              # logging
pip install pandas polars pyarrow  # Kaggle (legacy) loader
```

## Quickstart

```bash
# Prometheus memory-mapped data
python run.py -c configs/prometheus_angular_reco.cfg

# IceCube memory-mapped data
python run.py -c configs/i3_angular_reco.cfg

# Legacy Kaggle format (parquet)
python run.py -c configs/icecube_kaggle_angular_reco.cfg --no-wandb
```

## Data

- Memory-mapped datasets: See nt-mmap-converter for format and conversion: https://github.com/felixyu7/nt-mmap-converter
- Kaggle legacy format: parquet batch files + metadata, with chunk-aware sampling.
- Summary statistics are computed by the bundled `nt_summary_stats` module.

## Config (minimal)

```yaml
dataloader: "prometheus"  # or "icecube"
data_options:
  train_data_path: "/path/to/train"   # or list of paths
  valid_data_path: "/path/to/valid"
  use_summary_stats: true              # per-sensor features (9 stats)
training_options:
  batch_size: 16
  num_workers: 4
```

## Model

```python
from neptune import NeptuneModel

model = NeptuneModel(in_channels=9, output_dim=3)  # 9 = summary stats per sensor
coords, feats, labels = next(iter(train_loader))   # coords: [N,5], feats: [N,9]
out = model(coords, feats)
```

## Features

- Unified dataloader (Prometheus/IceCube/Kaggle)
- Built‑in summary stats (9 classic features)
- Mixed precision, checkpointing, CSV/WandB logging
