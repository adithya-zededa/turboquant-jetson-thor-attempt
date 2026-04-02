"""
TurboQuant score module — attention computation over compressed + exact segments.

Handles the read path:
  - Compute attention scores over compressed historical KV (via Triton or PyTorch fallback)
  - Compute attention scores over exact recent buffer (via standard matmul / SDPA)
  - Merge logits and weighted values from both segments

Design rule: compressed path is only invoked when history is large enough
to justify it (>= 16 tokens).
"""

from __future__ import annotations

import math
import logging
import torch
import torch.nn.functional as F

from turboquant.store import FlatCache, CompressedKVStore
from turboquant.kv_cache import dequantize_values
from turboquant.quantizer import TurboQuantProd

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    num_query_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 for decode
        store: compressed KV store with historical tokens
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads (for GQA expansion)
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale
        )

    # Both segments present — merge via log-sum-exp trick
    return _attend_hybrid(
        query, flat, store.quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale,
    )


def _attend_compressed_only(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over compressed history only (PyTorch path)."""
    k_dequant = quantizer.dequantize(flat.prod_q)  # (H_kv, N, D)
    v_dequant = dequantize_values(flat.value_q, 32)

    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over exact recent buffer only."""
    return _matmul_attend(
        query, recent_k.transpose(0, 1), recent_v.transpose(0, 1),
        gqa_ratio, num_kv_heads, scale,
    )


def _attend_hybrid(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """Merge compressed history + exact recent via concatenated attention."""
    k_hist = quantizer.dequantize(flat.prod_q)  # (H_kv, N_hist, D)
    v_hist = dequantize_values(flat.value_q, 32)

    k_recent = recent_k.transpose(0, 1)   # (H_kv, N_recent, D)
    v_recent = recent_v.transpose(0, 1)

    k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
    v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)

    return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(
            f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}"
        )

    # Avoid repeat_interleave(Q/H) on KV tensors to keep memory bounded at long context.
    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)   # (H_kv, 1, N, D) broadcast over G
    v = kv_values.float().unsqueeze(1) # (H_kv, 1, N, D) broadcast over G

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
