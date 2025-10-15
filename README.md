# Neptune: An Efficient Point Transformer for Ultrarelativistic Neutrino Events

Neptune (a**N** **E**fficient **P**oint **T**ransformer for **U**ltrarelativistic **N**eutrino **E**vents) is a transformer-based point cloud processing model for neutrino event reconstruction. (WIP, not yet on PyPI)

```bash
pip install -e .
```

## Usage

```python
import torch
from neptune import NeptuneModel

model = NeptuneModel(
    in_channels = 6,                   # point features
    num_patches = 128,                 # max tokens after tokenization
    token_dim = 768,                   # transformer dim
    num_layers = 12,                   # transformer layers
    output_dim = 3,                    # task output (3D direction, energy, etc.)
    tokenizer_type = "fps"             # or "learned_importance"
)

# coordinates: [N, 4] -> [x, y, z, t]
# features: [N, 6] -> point features
coords = torch.randn(1000, 4)
features = torch.randn(1000, 6)
batch_ids = torch.zeros(1000, dtype=torch.long)  # per-point batch indices

out = model(coords, features, batch_ids) # [batch_size, 3]

# train with angular distance loss for 3D directions
import torch.nn.functional as F

def angular_distance_loss(pred, truth):
    pred_norm = F.normalize(pred, dim=1)
    truth_norm = F.normalize(truth, dim=1) 
    cos_sim = F.cosine_similarity(pred_norm, truth_norm)
    return torch.acos(torch.clamp(cos_sim, -1+1e-7, 1-1e-7)).mean()

# training loop
directions = torch.randn(batch_size, 3)  # true directions
loss = angular_distance_loss(out, directions)
loss.backward()
```

For full training runs, the CLI entry point `scripts/run.py` uses shared tooling from [ml-common](https://github.com/felixyu7/ml-common) for dataloaders, losses, and the trainer:

```bash
python scripts/run.py -c scripts/configs/prometheus_angular_reco.cfg
```

## How it works

1. **Tokenization** – choose `fps` (farthest-point sampling + k-NN aggregation) or `learned_importance` (learned top-k selection per batch).
2. **Transformer encoder** – RoPE-enabled self-attention over centroid-aware tokens.
3. **Pooling** – masked mean pool to obtain a global representation.
4. **Prediction head** – MLP for the downstream task.

## Parameters

- `in_channels`: input features per point (default: 6)
- `num_patches`: max tokens after sampling (default: 128) 
- `token_dim`: transformer hidden dim (default: 768)
- `num_layers`: transformer depth (default: 12)
- `num_heads`: attention heads (default: 12)
- `output_dim`: task output dim (default: 3)
- `tokenizer_type`: `"learned_importance"` (default) or `"fps"`
- `k_neighbors`: only used when `tokenizer_type="fps"` (default: 16)
- `tokenizer_kwargs`: optional dict forwarded to the tokenizer implementation

## Requirements

- torch >= 2.0
