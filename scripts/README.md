# Neptune Training Scripts  

Training and inference scripts for Neptune neutrino event reconstruction.

## Install

```bash
pip install -e .
pip install pyyaml awkward pyarrow scipy nt-summary-stats tqdm

# Optional for WandB logging
pip install wandb
```

## Usage

Clean, efficient vanilla PyTorch training with minimal dependencies and unified dataloader interface.

```bash
# Prometheus memory-mapped data
python run.py -c configs/prometheus_angular_reco.cfg

# IceCube memory-mapped data  
python run.py -c configs/icecube_angular_reco.cfg

# Legacy Kaggle IceCube format
python run.py -c configs/kaggle_icecube.cfg --no-wandb
```

## Supported Dataset Types

The unified dataloader automatically detects and handles:

1. **Prometheus mmap** - High-energy neutrino simulation data (new format)
2. **IceCube mmap** - IceCube detector data (new unified format) 
3. **Kaggle IceCube** - Legacy parquet-based IceCube competition data

All formats share the same interface and support summary statistics processing via `nt-summary-stats`.

## Features

- **Unified Interface**: Single dataloader for all dataset types with auto-detection
- **Optional WandB**: Automatic fallback to CSV logging if unavailable
- **Mixed precision**: GPU training with automatic CPU fallback  
- **Checkpointing**: Automatic saving and resuming
- **Memory-mapped I/O**: Efficient large dataset handling
- **Summary Statistics**: Sensor-level feature extraction via nt-summary-stats
- **Minimal Dependencies**: ~500 lines total, no framework overhead

## File Structure

```
scripts/
├── run.py                 # Main training script
├── trainer.py             # Training logic and utilities  
├── loss_functions.py      # Task-specific loss functions
├── configs/               # Training configurations
└── data/                  # Unified dataloader package
    ├── __init__.py        # Public interface
    ├── base.py            # Dataloader factory
    ├── mmap_datasets.py   # Prometheus + IceCube mmap datasets
    ├── kaggle_datasets.py # Legacy Kaggle IceCube format
    └── utils.py           # Common utilities and mmap loader
```

## Configuration

### Memory-mapped Datasets (Recommended)

```yaml  
dataloader: "prometheus"  # or "icecube" 
data_options:
  train_data_path: "/path/to/data/train"     # Single path or list of paths
  valid_data_path: "/path/to/data/valid"     # Single path or list of paths  
  use_summary_stats: true                    # Use nt-summary-stats processing
```

### Legacy Kaggle Format

```yaml
dataloader: "kaggle"
data_options:
  train_data_path: "/path/to/batch_files/"
  valid_data_path: "/path/to/batch_files/"
  geometry_path: "/path/to/sensor_geometry.csv"
  train_ranges: [0, 800]     # File range for training
  valid_ranges: [800, 1000]  # File range for validation
```

## Memory-Mapped Format

The new unified format created by [nt-mmap-converter](https://github.com/yourname/nt-mmap-converter):

- **Instant access**: O(1) event indexing, no file parsing
- **Unified interface**: Same code handles Prometheus and IceCube  
- **Efficient storage**: Memory-mapped files with header-based auto-detection
- **Scalable**: Handle datasets with millions of events seamlessly

## Model Interface

Uses the clean Neptune package:

```python  
from neptune import NeptuneModel

model = NeptuneModel(in_channels=9, output_dim=3)  # 9 features from summary stats
output = model(coords, features)  # coords: [N,5], features: [N,9]
```