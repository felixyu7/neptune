# Neptune: An Efficient Point Transformer for Ultrarelativistic Neutrino Events

Neptune (a**N** **E**fficient **P**oint **T**ransformer for **U**ltrarelativistic **N**eutrino **E**vents) is a transformer-based point cloud processing model for neutrino event reconstruction, as described in our [paper](https://arxiv.org/abs/2510.01733).

## Installation

```bash
pip install -e .
```

Neptune is pure Python at install time. The FPS/kNN kernels are vendored in `neptune/fps/`: on CUDA they are Triton kernels JIT-compiled for whatever GPU you run on (no compute-capability coupling at install — safe across heterogeneous clusters), and on CPU a small C++ extension compiles automatically on first use (cached; requires a C++ compiler, otherwise a slower pure-PyTorch fallback is used with a warning).

## Usage

```python
import torch
from neptune import NeptuneModel

model = NeptuneModel(
    in_channels = 6,                   # point features
    num_patches = 128,                 # max tokens after tokenization
    token_dim = 768,                   # transformer dim
    num_layers = 12,                   # transformer layers
    output_dim = 3                     # task output (3D direction, energy, etc.)
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

Neptune assumes `coords` is an `[N, 4]` tensor with time in the fourth column.

For full training runs, the CLI entry point `scripts/run.py` uses shared tooling from [ml-common](https://github.com/felixyu7/ml-common) for dataloaders, losses, and the trainer. Run `git submodule update --init --recursive` to pull the [ml-common](https://github.com/felixyu7/ml-common) submodule. Then, run by providing a config file:

```bash
python scripts/run.py -c scripts/configs/what-1_angular_reco.cfg
```

Configs for the WhaT-1 and Prometheus tasks live under `scripts/configs/`. The four `what-1_*.cfg` files are the v1.2 release configs (direction, energy, morphology, neutrino-vs-background) and reproduce the released v1.2 checkpoints.

Run the test suite with

```bash
pytest tests/
```

## How it works

1. **Tokenization** – farthest-point sampling down to `num_patches` centroids, then per-token pooling by nearest-centroid (Voronoi) assignment (or k-NN gather with `assign_mode="knn"`), with optional charge-weighted Lloyd refinement of the centroids.
2. **Transformer encoder** – 4D RoPE-enabled (based on this [paper](https://arxiv.org/abs/2504.06308)) self-attention over tokens, plus a Fourier absolute position encoding. Compiled with `torch.compile` by default; sparse batches use a packed block-diagonal `flex_attention` path on GPU.
3. **Pooling** – masked mean (or attention) pool to obtain a global representation.
4. **Prediction head** – MLP for the downstream task.

## Parameters

- `in_channels`: input features per point (default: 6)
- `num_patches`: max tokens after sampling (default: 128) 
- `token_dim`: transformer hidden dim (default: 768)
- `num_layers`: transformer depth (default: 12)
- `num_heads`: attention heads (default: 12)
- `output_dim`: task output dim (default: 3)
- `pool_type`: `"mean"` or `"attention"` (default: `"mean"`)
- `attn_impl`: `"auto"`, `"padded"`, or `"packed"` attention path (default: `"auto"`)
- `compile_encoder`: compile the encoder with `torch.compile` (default: `True`)
- `tokenizer_kwargs`: optional dict forwarded to the tokenizer (e.g. `assign_mode`, `lloyd_iters`, `knn_pool`, `k_neighbors`)

## Requirements

- torch >= 2.0
