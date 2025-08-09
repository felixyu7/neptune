# Neptune Examples

Examples showing how to use Neptune for neutrino event reconstruction.

## Install

```bash
pip install -e .
pip install pyyaml awkward pyarrow scipy nt-summary-stats tqdm

# Optional for WandB logging
pip install wandb
```

## Usage

Clean, efficient vanilla PyTorch training with minimal dependencies.

```bash
# With WandB logging (default)
python run.py -c configs/prometheus_angular_reco.cfg

# With CSV logging only (no WandB dependency)  
python run.py -c configs/prometheus_angular_reco.cfg --no-wandb

# Energy reconstruction
python run.py -c configs/prometheus_energy_reco.cfg
```

## Features

- **Optional WandB**: Automatic fallback to CSV logging if unavailable
- **Mixed precision**: GPU training with automatic CPU fallback
- **Checkpointing**: Automatic saving and resuming  
- **Real-time metrics**: Progress tracking and validation during training
- **Minimal**: ~400 lines total, no framework overhead

## Files

- `run.py` - Main training script
- `trainer.py` - Training logic and utilities
- `prometheus_data.py` - Prometheus dataset and data loading
- `loss_functions.py` - Task-specific loss functions
- `configs/` - Training configurations

## Config

Update data paths in config files:

```yaml
data_options:
  train_data_files:
    - "/path/to/your/prometheus/data"
```

## Model

Uses the clean Neptune package:

```python
from neptune import NeptuneModel

model = NeptuneModel(in_channels=6, output_dim=3)
output = model(coords, features)
```

That's it.