"""Variable-length packing helpers for the Neptune encoder.

The tokenizer emits a fixed ``[B, S]`` padded layout (S = num_patches), but at
S=128 the encoder is linear-layer bound, so the padded slots — masked out of
attention yet still pushed through every Linear — waste up to ~4x of the dominant
compute. These helpers pack the valid tokens of a batch into a single
``[1, N_total, D]`` sequence and drive a block-diagonal ``flex_attention`` so each
event attends only within itself, eliminating that waste on GPU. CPU keeps the
padded path (B=1 there, so nothing to gain).

The packed sequence is the *exact* total valid-token count (no bucketing): the
inner encoder is compiled with ``dynamic=True``, so a varying sequence length
reuses one graph instead of recompiling. Because there's no padding tail, every
packed token belongs to a real event — no fully-masked rows, hence no NaN to
guard against. ``pack``/``unpack`` use data-dependent shapes (``nonzero``) and run
eagerly, outside the compiled inner encoder.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_AVAILABLE = True
except Exception:  # pragma: no cover - older torch without flex_attention
    create_block_mask = None
    flex_attention = None
    FLEX_AVAILABLE = False

# Pack only when the batch is at most this full. Packing trades padded-token
# Linear work for a less efficient attention kernel (block-diagonal flex vs the
# padded path's batched flash): it wins below ~80% occupancy and loses a few %
# above it, so above this fraction the caller uses the padded path. Internal
# constant, not a user knob — it reflects the flex/flash crossover, not a dataset.
_PACK_OCCUPANCY_MAX = 0.8


def pack(
    tokens: Tensor,      # [B, S, D]
    centroids: Tensor,   # [B, S, 4]
    masks: Tensor,       # [B, S] bool (True = valid)
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Pack valid tokens into one ``[1, N_total, D]`` sequence.

    Returns ``(packed_tokens[1, N, D], packed_centroids[1, N, 4], doc_id[N],
    pack_idx[N])`` or ``None`` when packing wouldn't pay off — an empty batch, or
    occupancy at/above ``_PACK_OCCUPANCY_MAX`` (little padding to skip). On
    ``None`` the caller runs the padded path; no token is ever dropped.
    """
    B, S, D = tokens.shape

    pack_idx = masks.reshape(B * S).nonzero(as_tuple=True)[0]   # [N]; one host sync
    n = int(pack_idx.numel())
    if n == 0 or n >= _PACK_OCCUPANCY_MAX * B * S:
        return None

    packed_tokens = tokens.reshape(B * S, D).index_select(0, pack_idx).unsqueeze(0)
    packed_centroids = centroids.reshape(B * S, 4).index_select(0, pack_idx).unsqueeze(0)
    doc_id = pack_idx // S   # event index per token (valid tokens are row-major)

    # Tell torch.compile the packed length varies, so it builds one dynamic graph
    # instead of recompiling per distinct N.
    torch._dynamo.mark_dynamic(packed_tokens, 1)
    torch._dynamo.mark_dynamic(packed_centroids, 1)
    return packed_tokens, packed_centroids, doc_id, pack_idx


def build_block_mask(doc_id: Tensor, n: int):
    """Block-diagonal flex BlockMask: attention is allowed only within an event."""
    def mask_mod(b, h, q_idx, kv_idx):
        return doc_id[q_idx] == doc_id[kv_idx]

    # _compile=True fuses the mask construction (~50x faster than eager, which
    # otherwise dominates per-step packing overhead).
    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n,
        device=doc_id.device, _compile=True,
    )


def unpack(out: Tensor, pack_idx: Tensor, B: int, S: int) -> Tensor:
    """Scatter packed encoder output ``[1, N, D]`` back to padded ``[B, S, D]``,
    the layout the existing pooling expects."""
    D = out.shape[-1]
    result = out.new_zeros(B * S, D)
    result.index_copy_(0, pack_idx, out[0])
    return result.reshape(B, S, D)
