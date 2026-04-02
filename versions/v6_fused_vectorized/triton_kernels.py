"""
TurboQuant fused Triton kernels for decode attention.

The main bottleneck during decode is computing attention scores from the
packed TurboQuant representation. Without fusion, the PyTorch path is:

  1. Unpack MSE indices (bit-shift)
  2. Lookup centroids (gather)
  3. Rotate back (d×d matmul)
  4. Scale by norms
  5. Dot with query (another matmul)
  ──
  6. Sketch query through S (d×d matmul)
  7. Unpack QJL signs (bit-shift)
  8. Dot sketched query with signs
  9. Scale by residual norms

With fusion, we avoid materializing the full d-dim dequantized vectors.
Instead we compute the score directly from packed data.

Kernel 1: turboquant_mse_score
  For each (query, quantized_key) pair, compute <q, dequant(key)>
  by fusing steps 1-5 into a single kernel.

Kernel 2: turboquant_qjl_score
  For each (query, qjl_signs) pair, compute <S^T q, signs> * scale
  by fusing steps 6-9 (query sketch is precomputed once per query).

Kernel 3: turboquant_fused_decode_attention
  Full fused kernel: computes softmax(scores/sqrt(d)) @ V in one pass
  using online softmax (flash-attention style) over TQ-compressed KV.
"""

import math
import torch
import triton
import triton.language as tl


# ─── Kernel 1: MSE score computation ──────────────────────────────────
#
# Given:
#   query:       (B*H, 1, D)           float16/float32
#   mse_packed:  (B*H, N, packed_d)    uint8 (bit-packed MSE indices)
#   norms:       (B*H, N)              float16/float32 (original vector norms)
#   centroids:   (2^mse_bits,)         float32 (codebook centroids)
#   Pi:          (D, D)                float32 (rotation matrix)
#
# Computes: scores[b,n] = sum_j query_rot[j] * centroid[idx[n,j]] * norms[n]
#
# Key insight: instead of rotating key back (y@Pi), we rotate query forward (q@Pi^T)
# Then score = norms * sum_j q_rot[j] * centroid[idx[j]]
# This avoids materializing the D-dim dequantized key vectors entirely.

@triton.jit
def _turboquant_mse_score_kernel(
    # Pointers
    Q_ptr,          # (BH, D) query vectors (already rotated: q @ Pi^T)
    MSE_ptr,        # (BH, N, packed_d) bit-packed indices
    NORMS_ptr,      # (BH, N) original norms
    CENTROIDS_ptr,  # (n_clusters,) centroid values
    OUT_ptr,        # (BH, N) output scores
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    # Dimensions
    BH: tl.constexpr,
    N,   # number of KV tokens (variable)
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    # Quantization params
    BITS: tl.constexpr,        # MSE bits (1, 2, or 4 after rounding)
    VALS_PER_BYTE: tl.constexpr,  # how many indices per packed byte
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    """Compute MSE attention scores for a block of KV tokens."""
    pid_bh = tl.program_id(0)   # batch*head index
    pid_n = tl.program_id(1)    # KV token block index

    # Bounds
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Precompute query base pointer once (avoids pid_bh*stride_q_bh MUL inside the hot loop)
    q_base = Q_ptr + pid_bh * stride_q_bh

    # Accumulate score for each token in the block
    scores = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Bit mask for extracting indices
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Process packed bytes — each byte contains VALS_PER_BYTE indices
    for byte_idx in range(PACKED_D):
        # Load packed bytes for this block of tokens: (BLOCK_N,)
        packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        # Extract each index from the packed byte
        for sub in range(VALS_PER_BYTE):
            coord_idx = byte_idx * VALS_PER_BYTE + sub
            if coord_idx < D:
                # Extract index for this coordinate
                idx = (packed >> (sub * BITS)) & BIT_MASK

                # Lookup centroid value
                centroid_val = tl.load(CENTROIDS_ptr + idx)

                # Load query element using precomputed base (one fewer multiply per iter)
                q_val = tl.load(q_base + coord_idx * stride_q_d).to(tl.float32)
                scores += q_val * centroid_val

    # Multiply by norms
    norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                     mask=n_mask, other=0.0).to(tl.float32)
    scores = scores * norms

    # Store
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
             scores, mask=n_mask)


# ─── Kernel 2: QJL score computation ──────────────────────────────────
#
# Given:
#   q_sketched:     (BH, D)         float32 — precomputed q @ S^T
#   qjl_signs:      (BH, N, D//8)   uint8 — packed sign bits
#   residual_norms: (BH, N)         float32
#   qjl_scale:      scalar          float32 — sqrt(pi/2) / D
#
# Computes: scores[b,n] = qjl_scale * res_norms[n] * sum_j q_sketched[j] * sign[n,j]

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr,    # (BH, D) pre-sketched query
    SIGNS_ptr,       # (BH, N, packed_d) packed sign bits
    RES_NORMS_ptr,   # (BH, N) residual norms
    OUT_ptr,         # (BH, N) output QJL scores (added to existing)
    # Strides
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_o_bh, stride_o_n,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,  # D // 8
    QJL_SCALE,  # sqrt(pi/2) / D
    # Block sizes
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Precompute sketch base pointer once
    qs_base = Q_SKETCH_ptr + pid_bh * stride_qs_bh

    # Accumulate dot product of q_sketched with sign vectors
    dot = tl.zeros([BLOCK_N], dtype=tl.float32)

    for byte_idx in range(PACKED_D_SIGNS):
        # Load packed sign byte for this block: (BLOCK_N,)
        packed = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
            mask=n_mask, other=0
        ).to(tl.int32)

        # Extract 8 sign bits per byte
        for bit in range(8):
            coord_idx = byte_idx * 8 + bit
            if coord_idx < D:
                sign_bit = (packed >> bit) & 1
                # Convert {0,1} -> {-1, +1}
                sign_val = tl.where(sign_bit == 1, 1.0, -1.0)

                # Load using precomputed base
                q_val = tl.load(qs_base + coord_idx * stride_qs_d).to(tl.float32)
                dot += q_val * sign_val

    # Scale by residual norms and QJL constant
    res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                         mask=n_mask, other=0.0).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    # Add to existing MSE scores (or store fresh)
    existing = tl.load(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
                        mask=n_mask, other=0.0)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
             existing + qjl_scores, mask=n_mask)


# ─── Kernel 3: Fused decode attention (online softmax over TQ keys + values) ──
#
# For decode, query has n_q=1. We iterate over KV tokens in blocks,
# computing scores from TQ-compressed keys and accumulating the
# weighted value sum using online softmax (flash-attention style).
#
# This is the big payoff: we read compressed KV (~3 bits/element),
# never materialize the full FP16 KV, and produce the final output
# in a single pass.

@triton.jit
def _turboquant_fused_decode_kernel(
    # Query (already rotated for MSE, and sketched for QJL)
    Q_ROT_ptr,       # (BH, D) q @ Pi^T
    Q_SKETCH_ptr,    # (BH, D) q @ S^T
    # Quantized keys
    MSE_ptr,         # (BH, N, packed_d_mse) packed MSE indices
    SIGNS_ptr,       # (BH, N, packed_d_signs) packed QJL signs
    NORMS_ptr,       # (BH, N) key norms
    RES_NORMS_ptr,   # (BH, N) residual norms
    CENTROIDS_ptr,   # (n_clusters,) codebook
    # Values (group-quantized)
    V_DATA_ptr,      # (BH, N, D) uint8 quantized values
    V_SCALES_ptr,    # (BH, N, N_GROUPS) value scales
    V_ZEROS_ptr,     # (BH, N, N_GROUPS) value zeros
    # Output
    OUT_ptr,         # (BH, D) output
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_o_bh, stride_o_d,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Quant params
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,  # 1/sqrt(d)
    # Block
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # Precompute both query base pointers once (avoids pid_bh multiply inside each block loop)
    q_rot_base    = Q_ROT_ptr    + pid_bh * stride_q_bh
    q_sketch_base = Q_SKETCH_ptr + pid_bh * stride_q_bh

    # Online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([1], dtype=tl.float32)                   # running sum of exp
    acc = tl.zeros([D], dtype=tl.float32)                    # running weighted sum

    num_blocks = tl.cdiv(N, BLOCK_N)

    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── Compute TQ attention score for this block ──

        # Part 1: MSE score
        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_MSE):
            packed = tl.load(
                MSE_ptr + pid_bh * stride_m_bh + n_offs * stride_m_n + byte_idx * stride_m_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for sub in range(VALS_PER_BYTE):
                coord_idx = byte_idx * VALS_PER_BYTE + sub
                if coord_idx < D:
                    idx = (packed >> (sub * BITS)) & BIT_MASK
                    centroid_val = tl.load(CENTROIDS_ptr + idx)
                    q_rot_val = tl.load(q_rot_base + coord_idx * stride_q_d).to(tl.float32)
                    mse_scores += q_rot_val * centroid_val

        key_norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # Part 2: QJL score
        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for byte_idx in range(PACKED_D_SIGNS):
            packed = tl.load(
                SIGNS_ptr + pid_bh * stride_s_bh + n_offs * stride_s_n + byte_idx * stride_s_d,
                mask=n_mask, other=0
            ).to(tl.int32)
            for bit in range(8):
                coord_idx = byte_idx * 8 + bit
                if coord_idx < D:
                    sign_bit = (packed >> bit) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_sketch_val = tl.load(q_sketch_base + coord_idx * stride_q_d).to(tl.float32)
                    qjl_dot += q_sketch_val * sign_val

        res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        # Combined score
        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # New exponentials
        p = tl.exp(scores - m_new)

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, 0)
        # Update accumulator: rescale old, add new
        acc = acc * alpha

        # ── Dequantize values for this block and accumulate ──
        # Load full value tile: (BLOCK_N, D)
        d_offs = tl.arange(0, D)
        # Value data
        v_quant = tl.load(
            V_DATA_ptr + pid_bh * stride_v_bh
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0
        ).to(tl.float32)
        # Value scales: group index = d_offs // GROUP_SIZE
        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr + pid_bh * stride_vs_bh
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_bh * stride_vz_bh
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)
        # Dequantize: (BLOCK_N, D)
        v_dequant = v_quant * v_scale + v_zero
        # Weighted sum: p (BLOCK_N,) @ v_dequant (BLOCK_N, D) -> (D,)
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    # Final normalization
    acc = acc / l_i

    # Store output
    d_offs = tl.arange(0, D)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


# ─── Python wrappers ──────────────────────────────────────────────────

def _get_packing_params(bits: int):
    """Get packing parameters matching _pack_indices logic."""
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2  # 3-bit rounds up to 4-bit packing
    else:
        return 8, 1


def turboquant_mse_score(
    query_rot: torch.Tensor,     # (BH, D) or (BH, 1, D) — q @ Pi^T
    mse_packed: torch.Tensor,    # (BH, N, packed_d) uint8
    norms: torch.Tensor,         # (BH, N) float
    centroids: torch.Tensor,     # (n_clusters,) float32
    mse_bits: int,
) -> torch.Tensor:
    """
    Compute MSE attention scores using Triton kernel.

    Returns: (BH, N) attention logits (before scaling by 1/sqrt(d)).
    """
    if query_rot.dim() == 3:
        query_rot = query_rot.squeeze(1)  # (BH, D)

    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, N, device=query_rot.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))

    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        BH=BH, N=N, D=D, PACKED_D=packed_d,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        BLOCK_N=BLOCK_N,
    )

    return out


def turboquant_qjl_score(
    q_sketched: torch.Tensor,       # (BH, D) — q @ S^T
    qjl_signs: torch.Tensor,        # (BH, N, D//8) uint8 packed signs
    residual_norms: torch.Tensor,   # (BH, N)
    qjl_scale: float,               # sqrt(pi/2) / D
    out: torch.Tensor = None,       # (BH, N) — will be ADDED to if provided
) -> torch.Tensor:
    """
    Compute QJL attention score contribution.

    If `out` is provided, the QJL scores are added to it in-place.
    Returns: (BH, N) combined scores.
    """
    if q_sketched.dim() == 3:
        q_sketched = q_sketched.squeeze(1)

    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    packed_d_signs = qjl_signs.shape[2]

    if out is None:
        out = torch.zeros(BH, N, device=q_sketched.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D, PACKED_D_SIGNS=packed_d_signs,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N,
    )

    return out


def turboquant_attention_score(
    query: torch.Tensor,               # (B, H, 1, D) or (BH, 1, D)
    quantized_key,                      # ProdQuantized namedtuple
    Pi: torch.Tensor,                   # (D, D) rotation matrix
    S: torch.Tensor,                    # (D, D) QJL matrix
    centroids: torch.Tensor,           # (n_clusters,) codebook
    mse_bits: int,
    qjl_scale: float,
) -> torch.Tensor:
    """
    High-level: compute TurboQuant attention scores using Triton kernels.

    Precomputes q_rot = q @ Pi^T and q_sketch = q @ S^T,
    then calls the two Triton kernels.

    Returns: (BH, N) raw logits (caller applies /sqrt(d) and softmax).
    """
    # Flatten batch/head dims
    if query.dim() == 4:
        B, H, Q, D = query.shape
        query_flat = query.reshape(B * H, Q, D)
    else:
        query_flat = query
        D = query.shape[-1]

    # Precompute rotated and sketched queries (one-time per decode step)
    q_rot = torch.matmul(query_flat.squeeze(1).float(), Pi.T)      # (BH, D)
    q_sketch = torch.matmul(query_flat.squeeze(1).float(), S.T)    # (BH, D)

    # Flatten quantized key batch dims
    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() == 4:
        BH_shape = mse_packed.shape[:2]
        BH = BH_shape[0] * BH_shape[1]
        mse_packed = mse_packed.reshape(BH, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH, *qjl_signs.shape[2:])
        norms = norms.reshape(BH, -1)
        res_norms = res_norms.reshape(BH, -1)

    # MSE scores
    scores = turboquant_mse_score(q_rot, mse_packed, norms, centroids, mse_bits)

    # Add QJL scores
    scores = turboquant_qjl_score(q_sketch, qjl_signs, res_norms, qjl_scale, out=scores)

    return scores


def turboquant_fused_decode(
    query: torch.Tensor,               # (BH, 1, D) or (BH, D)
    quantized_key,                      # ProdQuantized
    value_quantized,                    # ValueQuantized
    Pi: torch.Tensor,                   # (D, D)
    S: torch.Tensor,                    # (D, D)
    centroids: torch.Tensor,           # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Fully fused decode attention: scores + softmax + value aggregation.
    Single pass over compressed KV, flash-attention style online softmax.

    Returns: (BH, D) attention output.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_shape = mse_packed.shape[:2]
        BH_actual = BH_shape[0] * BH_shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    v_data = value_quantized.data
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros

    # Unpack bit-packed values if needed (2-bit: 4 vals/byte, 4-bit: 2 vals/byte)
    v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
    if v_bits == 2 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
        # v_data is now (..., N, D) uint8
    elif v_bits == 4 and v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)

    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, N, -1)
        v_scales = v_scales.reshape(BH, N, -1)
        v_zeros = v_zeros.reshape(BH, N, -1)

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.zeros(BH, D, device=query.device, dtype=torch.float32)

    BLOCK_N = min(64, triton.next_power_of_2(N))

    grid = (BH,)

    _turboquant_fused_decode_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        out,
        # Q strides
        q_rot.stride(0), q_rot.stride(1),
        # MSE strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        # Signs strides
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        # Res norms strides
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Out strides
        out.stride(0), out.stride(1),
        # Dims
        N=N, D=D, PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        # Quant params
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        # Block
        # 8 warps on Blackwell (SM 11.0) improves SM utilisation with 20 SMs
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return out.to(query.dtype)


# ─── Kernel 4: Fused hybrid decode (compressed + buffer in one pass) ──
#
# Extends kernel 3 to handle both TQ-compressed and full-precision buffer
# tokens in a single kernel launch with online softmax.  Eliminates ~15
# extra PyTorch kernel launches that the _fused_hybrid path requires.

@triton.jit
def _turboquant_fused_hybrid_kernel(
    # Queries
    Q_ROT_ptr,       # (BH, D) q @ Pi^T
    Q_SKETCH_ptr,    # (BH, D) q @ S^T
    Q_ORIG_ptr,      # (BH, D) original query (for buffer dot products)
    # Compressed KV (same as kernel 3)
    MSE_ptr, SIGNS_ptr, NORMS_ptr, RES_NORMS_ptr, CENTROIDS_ptr,
    V_DATA_ptr, V_SCALES_ptr, V_ZEROS_ptr,
    # Buffer KV (full precision)
    K_BUF_ptr,       # (BH, N_B, D)
    V_BUF_ptr,       # (BH, N_B, D)
    # Output
    OUT_ptr,         # (BH, D)
    # Strides — compressed (same layout as kernel 3)
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    # Strides — buffer
    stride_kb_bh, stride_kb_n, stride_kb_d,
    stride_vb_bh, stride_vb_n, stride_vb_d,
    # Strides — output
    stride_o_bh, stride_o_d,
    # Dims
    N_C,  # compressed token count
    N_B,  # buffer token count
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Quant params
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    # Block
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    q_rot_base    = Q_ROT_ptr    + pid_bh * stride_q_bh
    q_sketch_base = Q_SKETCH_ptr + pid_bh * stride_q_bh

    # Online softmax state (shared across both phases)
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    # ════ Phase 1: Compressed KV — VECTORIZED tile approach ════
    #
    # Load entire (BLOCK_N, PACKED_D) tiles and process with vectorized ops.
    # MSE: 4 vectorized iterations (one per sub-value in each packed byte)
    # QJL: 8 vectorized iterations (one per sign bit)
    # Total: 12 tile-parallel iterations instead of 256 serial scalar ops.
    #
    byte_range_mse = tl.arange(0, PACKED_D_MSE)
    byte_range_signs = tl.arange(0, PACKED_D_SIGNS)
    d_offs = tl.arange(0, D)

    num_c_blocks = tl.cdiv(N_C, BLOCK_N)
    for block_idx in range(num_c_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N_C

        # ── Vectorized MSE score ──
        all_packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh
            + n_offs[:, None] * stride_m_n + byte_range_mse[None, :] * stride_m_d,
            mask=n_mask[:, None], other=0
        ).to(tl.int32)

        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for sub in range(VALS_PER_BYTE):
            idx = (all_packed >> (sub * BITS)) & BIT_MASK
            centroid_vals = tl.load(CENTROIDS_ptr + idx)
            coord_offsets = byte_range_mse * VALS_PER_BYTE + sub
            q_vals = tl.load(
                q_rot_base + coord_offsets * stride_q_d,
                mask=coord_offsets < D, other=0.0
            ).to(tl.float32)
            mse_scores += tl.sum(centroid_vals * q_vals[None, :], axis=1)

        key_norms = tl.load(NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── Vectorized QJL score ──
        all_signs = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh
            + n_offs[:, None] * stride_s_n + byte_range_signs[None, :] * stride_s_d,
            mask=n_mask[:, None], other=0
        ).to(tl.int32)

        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for bit in range(8):
            sign_bits = (all_signs >> bit) & 1
            sign_vals = tl.where(sign_bits == 1, 1.0, -1.0)
            coord_offsets = byte_range_signs * 8 + bit
            q_sketch_vals = tl.load(
                q_sketch_base + coord_offsets * stride_q_d,
                mask=coord_offsets < D, other=0.0
            ).to(tl.float32)
            qjl_dot += tl.sum(sign_vals * q_sketch_vals[None, :], axis=1)

        res_norms = tl.load(RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
                            mask=n_mask, other=0.0).to(tl.float32)
        scores = (mse_scores + qjl_dot * res_norms * QJL_SCALE) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax + value dequant + accumulate ──
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        g_offs = d_offs // GROUP_SIZE
        v_quant = tl.load(
            V_DATA_ptr + pid_bh * stride_v_bh
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0
        ).to(tl.float32)
        v_scale = tl.load(
            V_SCALES_ptr + pid_bh * stride_vs_bh
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr + pid_bh * stride_vz_bh
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        acc += tl.sum(p[:, None] * v_dequant, 0)
        m_i = m_new

    # ════ Phase 2: Buffer KV (full precision, vectorized) ════
    d_offs = tl.arange(0, D)
    q_orig = tl.load(Q_ORIG_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d).to(tl.float32)

    num_b_blocks = tl.cdiv(N_B, BLOCK_N)
    for block_idx in range(num_b_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N_B

        # Load buffer keys tile: (BLOCK_N, D)
        k_buf = tl.load(
            K_BUF_ptr + pid_bh * stride_kb_bh
            + n_offs[:, None] * stride_kb_n + d_offs[None, :] * stride_kb_d,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)

        # Dot product scores: (BLOCK_N,)
        scores = tl.sum(k_buf * q_orig[None, :], axis=1) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # Online softmax update (continues from Phase 1)
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # Load buffer values tile: (BLOCK_N, D) — already fp16, no dequant needed
        v_buf = tl.load(
            V_BUF_ptr + pid_bh * stride_vb_bh
            + n_offs[:, None] * stride_vb_n + d_offs[None, :] * stride_vb_d,
            mask=n_mask[:, None], other=0.0
        ).to(tl.float32)

        acc += tl.sum(p[:, None] * v_buf, 0)
        m_i = m_new

    # Normalize and store
    acc = acc / l_i
    d_offs = tl.arange(0, D)
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


def turboquant_fused_hybrid(
    query: torch.Tensor,               # (BH, D) original query
    quantized_key,                      # ProdQuantized
    value_quantized,                    # ValueQuantized
    key_buffer: torch.Tensor,          # (BH, N_B, D) buffer keys
    value_buffer: torch.Tensor,        # (BH, N_B, D) buffer values
    Pi_S_T: torch.Tensor,              # (D, 2D) precomputed [Pi^T | S^T]
    centroids: torch.Tensor,           # (n_clusters,)
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
    value_data_unpacked: torch.Tensor = None,  # pre-unpacked uint8 values
) -> torch.Tensor:
    """
    Fused hybrid decode: compressed + buffer in ONE kernel launch.

    Optimizations:
      - Combined Pi_S_T matmul: 1 launch vs float() + 2 separate matmuls
      - Pre-unpacked values: skip unpack_values() entirely
      - Total: ~2 kernel launches (matmul + Triton) vs ~7 before
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    # Combined query transform: one matmul → [q_rot | q_sketch]
    q_combined = torch.matmul(query.float(), Pi_S_T)  # (BH, 2D)
    q_rot = q_combined[:, :D]       # view, no copy
    q_sketch = q_combined[:, D:]    # view, no copy

    # Flatten compressed key tensors
    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N_C = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    # Use pre-unpacked values if available (avoids 2 kernel launches)
    if value_data_unpacked is not None:
        v_data = value_data_unpacked
    else:
        v_data = value_quantized.data
        v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
        if v_bits in (2, 4) and v_data.shape[-1] != D:
            from turboquant.kv_cache import unpack_values
            v_data = unpack_values(value_quantized)

    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros
    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, N_C, -1)
        v_scales = v_scales.reshape(BH, N_C, -1)
        v_zeros = v_zeros.reshape(BH, N_C, -1)

    # Flatten buffer
    k_buf = key_buffer.reshape(BH, -1, D) if key_buffer.dim() > 3 else key_buffer
    v_buf = value_buffer.reshape(BH, -1, D) if value_buffer.dim() > 3 else value_buffer
    N_B = k_buf.shape[1]

    N_GROUPS = D // group_size
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)

    out = torch.empty(BH, D, device=query.device, dtype=torch.float32)
    BLOCK_N = min(64, triton.next_power_of_2(max(N_C, N_B)))

    _turboquant_fused_hybrid_kernel[(BH,)](
        q_rot, q_sketch, query.float(),
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        k_buf, v_buf,
        out,
        # Q strides
        q_rot.stride(0), q_rot.stride(1),
        # MSE strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        # Signs strides
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        # Res norms strides
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Buffer strides
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        v_buf.stride(0), v_buf.stride(1), v_buf.stride(2),
        # Out strides
        out.stride(0), out.stride(1),
        # Dims
        N_C=N_C, N_B=N_B, D=D,
        PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_signs,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return out.to(query.dtype)
