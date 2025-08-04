## Neptune

Transformer for point cloud-based neutrino telescope event reconstruction.

```bash
pip install neptune-reco
```

## Usage

```python
import torch
from neptune import NeptuneModel

model = NeptuneModel(
    in_channels = 6,        # point features
    num_patches = 128,      # tokens after FPS sampling  
    token_dim = 768,        # transformer dim
    num_layers = 12,        # transformer layers
    output_dim = 3          # task output (3D direction, energy, etc.)
)

# coordinates: [N, 5] -> [batch_idx, x, y, z, t]
# features: [N, 6] -> point features
coords = torch.randn(1000, 5)
features = torch.randn(1000, 6)

out = model(coords, features) # [batch_size, 3]
```

## How it works

1. **FPS sampling** - select representative points from irregular point cloud
2. **Neighborhood aggregation** - k-NN around each sample point → tokens  
3. **Transformer** - standard attention + 4D position embeddings (x,y,z,t)
4. **Global pooling** - average pool tokens → single representation
5. **Head** - linear layer for task (direction, energy, classification)

## Parameters

- `in_channels`: input features per point (default: 6)
- `num_patches`: max tokens after sampling (default: 128) 
- `token_dim`: transformer hidden dim (default: 768)
- `num_layers`: transformer depth (default: 12)
- `num_heads`: attention heads (default: 12)
- `output_dim`: task output dim (default: 3)
- `k_neighbors`: neighbors for aggregation (default: 16)
- `pool_method`: 'max' or 'mean' pooling (default: 'max')

## Training examples

See `examples/` for PyTorch Lightning integration with neutrino datasets

## Requirements

- torch >= 1.12.0
- fpsample >= 0.2.0
