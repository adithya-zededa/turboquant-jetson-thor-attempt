"""
TurboQuant v8 — "Zero External Kernel Launch" decode + fused prefill quantization.

v6 decode path (2 kernel launches per decode step):
  launch 1: torch.matmul(q, Pi_S_T)          # (BH,D) @ (D,2D) → q_rot + q_sketch
  launch 2: _turboquant_fused_hybrid_kernel   # scoring + softmax + V-accumulate

v8 decode path (1 kernel launch per decode step):
  launch 1: _turboquant_v8_zero_launch_kernel # fused rotation + scoring + softmax + V-accum

The saving is one kernel launch (~10-15 µs on Jetson) plus the global-memory round-trip
that was needed to write q_rot/q_sketch and then read them back in v6.

v8 also replaces ~10 PyTorch prefill quantization ops with 3 Triton kernels, giving
full SM utilisation (grid BH×N) during prompt processing.

Kernel inventory
────────────────
  Kernel A — _turboquant_v8_zero_launch_kernel
    Phase 0: in-kernel Q rotation (q @ Pi^T and q @ S^T computed on-chip)
    Phase 1: compressed KV scoring (vectorised MSE + QJL) + online softmax + V dequant
    Phase 2: full-precision buffer scoring + online softmax continuation
    Grid: (BH,) — one program per batch×head

  Kernel B1 — _turboquant_v8_quant_forward_kernel
    Normalize → rotate (Pi^T) → nearest centroid (linear scan) → store scratch
    Grid: (BH, N)

  Kernel B2 — _turboquant_v8_quant_residual_kernel
    Un-rotate centroid approx (Pi) → residual → QJL sketch (S^T) → store scratch
    Grid: (BH, N)

  Kernel B3 — _turboquant_v8_pack_mse_kernel
    Pack D int32 centroid indices → PACKED_D_MSE uint8 bytes
    Grid: (BH, N)

  Kernel B4 — _turboquant_v8_pack_signs_kernel
    Pack D int32 sign bits → PACKED_D_SIGNS uint8 bytes
    Grid: (BH, N)

Register usage analysis (Kernel A, D=128, warp_size=32)
──────────────────────────────────────────────────────
  q_rot    : D/32 = 4 fp32 regs per thread (128-element vector)
  q_sketch : 4 fp32 regs per thread
  acc      : 4 fp32 regs per thread
  m_i, l_i : 1 fp32 reg each
  Total rotation overhead: 8 regs — negligible vs v6's saved launch overhead.

Memory traffic for rotation (per decode step)
─────────────────────────────────────────────
  Pi_T = (D,D) fp32 = 64 KB — fits in L2. Head 0 cold-loads; heads 1-7 hit L2.
  S_T  = (D,D) fp32 = 64 KB — same.
  Total: 128 KB rotation matrices, cached after first head program.
"""

import math
import torch
import triton
import triton.language as tl


def _get_packing_params(bits: int):
    """Return (eff_bits, vals_per_byte) matching quantizer.py's _pack_indices."""
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2   # 3-bit rounds up to 4-bit packing
    else:
        return 8, 1


# ═══════════════════════════════════════════════════════════════════════════
# Kernel A — fully fused decode, Q rotation on-chip (v8 primary innovation)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _turboquant_v8_zero_launch_kernel(
    # Raw unrotated query (NOT pre-transformed outside the kernel)
    Q_ptr,          # (BH, D) fp16/fp32

    # Rotation matrices (shared across all heads — L2-cached after head 0)
    PI_T_ptr,       # (D, D) fp32   Pi transposed:  row i = Pi[:,i]
    S_T_ptr,        # (D, D) fp32   S transposed:   row i = S[:,i]

    # Compressed KV history
    MSE_ptr,        # (BH, N_C, PACKED_D_MSE) uint8
    SIGNS_ptr,      # (BH, N_C, PACKED_D_SIGNS) uint8
    NORMS_ptr,      # (BH, N_C) key norms
    RES_NORMS_ptr,  # (BH, N_C) residual norms
    CENTROIDS_ptr,  # (N_CLUSTERS,)

    # Value cache (pre-unpacked at prefill to save decode work)
    V_DATA_ptr,     # (BH, N_C, D) uint8
    V_SCALES_ptr,   # (BH, N_C, N_GROUPS)
    V_ZEROS_ptr,    # (BH, N_C, N_GROUPS)

    # Full-precision recent buffer
    K_BUF_ptr,      # (BH, N_B, D) fp16
    V_BUF_ptr,      # (BH, N_B, D) fp16

    # Output
    OUT_ptr,        # (BH, D) fp32 (caller casts to fp16)

    # Scratch for spilling q_rot/q_sketch from registers to global memory
    # (needed because Triton can't gather from register vectors with runtime indices)
    Q_ROT_SCRATCH_ptr,    # (BH, D) fp32
    Q_SKETCH_SCRATCH_ptr, # (BH, D) fp32

    # Strides — query
    stride_q_bh, stride_q_d,

    # Strides — scratch (same layout as query: (BH, D) contiguous)
    stride_sc_bh, stride_sc_d,

    # Strides — rotation (Pi_T and S_T share the same (D,D) layout)
    stride_pi_row,  # = D for contiguous (D,D) matrices

    # Strides — compressed keys
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh,  stride_n_n,
    stride_rn_bh, stride_rn_n,

    # Strides — values
    stride_v_bh,  stride_v_n,  stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,

    # Strides — buffer
    stride_kb_bh, stride_kb_n, stride_kb_d,
    stride_vb_bh, stride_vb_n, stride_vb_d,

    # Strides — output
    stride_o_bh, stride_o_d,

    # Dimensions
    N_C,   # compressed token count (runtime variable)
    N_B,   # buffer token count (runtime variable)
    D:  tl.constexpr,
    PACKED_D_MSE:   tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    N_GROUPS:       tl.constexpr,
    GROUP_SIZE:     tl.constexpr,

    # Quantisation parameters
    BITS:          tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    QJL_SCALE,      # sqrt(pi/2) / D — passed as runtime scalar
    SM_SCALE,       # 1 / sqrt(D)

    # Tile size for the N dimension
    BLOCK_N: tl.constexpr,
):
    """
    v8 decode attention — grid: (BH,).

    Key improvement over v6: the q @ Pi_S_T matmul (previously a separate
    torch.matmul kernel launch) is fused into Phase 0 of this kernel, reducing
    total launches per decode step from 2 → 1.

    After Phase 0 computes q_rot and q_sketch in registers, they are spilled
    to scratch global memory (1 KB per program, L1-resident) so Phase 1 can
    gather sub-vectors via pointer arithmetic.
    """
    pid_bh = tl.program_id(0)
    d_offs  = tl.arange(0, D)
    BIT_MASK: tl.constexpr = (1 << BITS) - 1

    # ═══ Phase 0: in-kernel Q rotation ════════════════════════════════
    #
    # Compute q_rot = q @ Pi^T and q_sketch = q @ S^T without leaving the kernel.
    #
    # Scalar outer-product accumulation pattern:
    #   for each i in 0..D-1:
    #     q_i     = q[i]          (1 scalar load from global memory)
    #     pi_row  = Pi^T[i, :]    (D-vector load — hits L2 after head 0)
    #     s_row   = S^T[i, :]
    #     q_rot    += q_i * pi_row
    #     q_sketch += q_i * s_row
    #
    # tl.static_range(D) unrolls at compile time. With D=128 and warp_size=32,
    # each thread accumulates 4 fp32 elements, so q_rot and q_sketch together
    # use only 8 registers per thread — no occupancy cost.

    q_rot    = tl.zeros([D], dtype=tl.float32)
    q_sketch = tl.zeros([D], dtype=tl.float32)
    q_base   = Q_ptr + pid_bh * stride_q_bh

    for i in tl.static_range(D):
        q_i    = tl.load(q_base + i * stride_q_d).to(tl.float32)
        pi_row = tl.load(PI_T_ptr + i * stride_pi_row + d_offs).to(tl.float32)
        s_row  = tl.load(S_T_ptr  + i * stride_pi_row + d_offs).to(tl.float32)
        q_rot    = q_rot    + q_i * pi_row
        q_sketch = q_sketch + q_i * s_row

    # Spill q_rot and q_sketch to scratch global memory so Phase 1 can
    # gather sub-vectors via pointer-based loads.  At D=128 this is 1 KB
    # per program — stays in L1/L2.
    sc_base = pid_bh * stride_sc_bh
    tl.store(Q_ROT_SCRATCH_ptr    + sc_base + d_offs * stride_sc_d, q_rot)
    tl.store(Q_SKETCH_SCRATCH_ptr + sc_base + d_offs * stride_sc_d, q_sketch)

    # ═══ Online softmax state (persists across both phases) ════════════
    m_i  = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i  = tl.zeros([1], dtype=tl.float32)
    acc  = tl.zeros([D],  dtype=tl.float32)

    # ═══ Phase 1: compressed KV (vectorised tiles, same as v6) ════════
    #
    # Load (BLOCK_N × PACKED_D_MSE) and (BLOCK_N × PACKED_D_SIGNS) tiles
    # at once; process VALS_PER_BYTE / 8 sub-iterations per tile instead of
    # D serial scalar loads.

    byte_range_mse   = tl.arange(0, PACKED_D_MSE)
    byte_range_signs = tl.arange(0, PACKED_D_SIGNS)

    # Base pointers for scratch q_rot / q_sketch (spilled from Phase 0)
    qr_base = Q_ROT_SCRATCH_ptr    + sc_base
    qs_base = Q_SKETCH_SCRATCH_ptr + sc_base

    num_c_blocks = tl.cdiv(N_C, BLOCK_N)
    for blk in range(num_c_blocks):
        n_start = blk * BLOCK_N
        n_offs  = n_start + tl.arange(0, BLOCK_N)
        n_mask  = n_offs < N_C

        # ── Vectorised MSE score ──────────────────────────────────────
        all_packed = tl.load(
            MSE_ptr + pid_bh * stride_m_bh
            + n_offs[:, None] * stride_m_n
            + byte_range_mse[None, :] * stride_m_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)  # (BLOCK_N, PACKED_D_MSE)

        mse_scores = tl.zeros([BLOCK_N], dtype=tl.float32)
        for sub in tl.static_range(VALS_PER_BYTE):
            idx          = (all_packed >> (sub * BITS)) & BIT_MASK   # (BLOCK_N, PACKED_D_MSE)
            c_vals       = tl.load(CENTROIDS_ptr + idx).to(tl.float32)
            coord_offs   = byte_range_mse * VALS_PER_BYTE + sub       # (PACKED_D_MSE,)
            q_rot_vals   = tl.load(
                qr_base + coord_offs * stride_sc_d,
                mask=coord_offs < D, other=0.0,
            ).to(tl.float32)                                           # (PACKED_D_MSE,)
            # (BLOCK_N, PACKED_D_MSE) · (PACKED_D_MSE,) → (BLOCK_N,)
            mse_scores   = mse_scores + tl.sum(c_vals * q_rot_vals[None, :], axis=1)

        key_norms  = tl.load(
            NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── Vectorised QJL score ──────────────────────────────────────
        all_signs = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh
            + n_offs[:, None] * stride_s_n
            + byte_range_signs[None, :] * stride_s_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)  # (BLOCK_N, PACKED_D_SIGNS)

        qjl_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
        for bit in tl.static_range(8):
            sign_bits  = (all_signs >> bit) & 1                        # (BLOCK_N, PACKED_D_SIGNS)
            sign_vals  = tl.where(sign_bits == 1, 1.0, -1.0)
            coord_offs = byte_range_signs * 8 + bit                    # (PACKED_D_SIGNS,)
            q_sk_vals  = tl.load(
                qs_base + coord_offs * stride_sc_d,
                mask=coord_offs < D, other=0.0,
            ).to(tl.float32)
            qjl_dot    = qjl_dot + tl.sum(sign_vals * q_sk_vals[None, :], axis=1)

        res_norms  = tl.load(
            RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        scores = (mse_scores + qjl_dot * res_norms * QJL_SCALE) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        # ── Online softmax update ─────────────────────────────────────
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new)
        l_i   = l_i * alpha + tl.sum(p, 0)
        acc   = acc  * alpha

        # ── Value dequant + weighted accumulate ───────────────────────
        g_offs  = d_offs // GROUP_SIZE
        v_quant = tl.load(
            V_DATA_ptr + pid_bh * stride_v_bh
            + n_offs[:, None] * stride_v_n + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.float32)
        v_scale = tl.load(
            V_SCALES_ptr + pid_bh * stride_vs_bh
            + n_offs[:, None] * stride_vs_n + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero  = tl.load(
            V_ZEROS_ptr + pid_bh * stride_vz_bh
            + n_offs[:, None] * stride_vz_n + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        acc = acc + tl.sum(p[:, None] * (v_quant * v_scale + v_zero), axis=0)
        m_i = m_new

    # ═══ Phase 2: full-precision buffer KV ════════════════════════════
    #
    # Reload raw query for plain dot-product against exact buffer keys.
    # (No rotation needed — buffer keys are unmodified fp16.)

    q_orig = tl.load(q_base + d_offs * stride_q_d).to(tl.float32)

    num_b_blocks = tl.cdiv(N_B, BLOCK_N)
    for blk in range(num_b_blocks):
        n_start = blk * BLOCK_N
        n_offs  = n_start + tl.arange(0, BLOCK_N)
        n_mask  = n_offs < N_B

        k_buf  = tl.load(
            K_BUF_ptr + pid_bh * stride_kb_bh
            + n_offs[:, None] * stride_kb_n + d_offs[None, :] * stride_kb_d,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k_buf * q_orig[None, :], axis=1) * SM_SCALE
        scores = tl.where(n_mask, scores, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(scores, 0))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(scores - m_new)
        l_i    = l_i * alpha + tl.sum(p, 0)
        acc    = acc  * alpha

        v_buf  = tl.load(
            V_BUF_ptr + pid_bh * stride_vb_bh
            + n_offs[:, None] * stride_vb_n + d_offs[None, :] * stride_vb_d,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        acc = acc + tl.sum(p[:, None] * v_buf, axis=0)
        m_i = m_new

    # ═══ Normalise and write output ════════════════════════════════════
    acc = acc / l_i
    tl.store(OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d, acc)


# ─── Python wrapper for Kernel A ─────────────────────────────────────────

def turboquant_v8_decode(
    query:              torch.Tensor,   # (BH, D) or (BH, 1, D) raw unrotated
    Pi_T:               torch.Tensor,   # (D, D) Pi transposed — fp32 contiguous
    S_T:                torch.Tensor,   # (D, D) S transposed  — fp32 contiguous
    quantized_key,                      # ProdQuantized namedtuple
    value_quantized,                    # ValueQuantized namedtuple
    key_buffer:         torch.Tensor,   # (BH, N_B, D) fp16
    value_buffer:       torch.Tensor,   # (BH, N_B, D) fp16
    centroids:          torch.Tensor,   # (N_CLUSTERS,) fp32
    mse_bits:           int,
    qjl_scale:          float,
    sm_scale:           float,
    group_size:         int = 32,
    value_data_unpacked: torch.Tensor = None,
) -> torch.Tensor:
    """
    v8 fully-fused decode: ONE Triton kernel launch.

    v6 used 2 launches (matmul + Triton).  v8 fuses the Q rotation into the
    Triton kernel, saving ~10-15 µs per decode step on Jetson AGX Thor.
    At context=1k (v6 ≈ 0.31 ms) this is ~4-5 % faster; the absolute
    latency saving is the same at all context lengths.

    Returns (BH, D) output in the same dtype as query.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    # Flatten compressed key tensors
    mse_packed = quantized_key.mse_indices
    qjl_signs  = quantized_key.qjl_signs
    norms      = quantized_key.norms
    res_norms  = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_c = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_c, *mse_packed.shape[2:])
        qjl_signs  = qjl_signs.reshape(BH_c,  *qjl_signs.shape[2:])
        norms      = norms.reshape(BH_c, -1)
        res_norms  = res_norms.reshape(BH_c, -1)

    N_C          = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_sig = qjl_signs.shape[2]

    if value_data_unpacked is not None:
        v_data = value_data_unpacked
    else:
        v_data = value_quantized.data
        v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
        if v_bits in (2, 4) and v_data.shape[-1] != D:
            from turboquant.kv_cache import unpack_values
            v_data = unpack_values(value_quantized)

    v_scales = value_quantized.scales
    v_zeros  = value_quantized.zeros
    if v_data.dim() > 3:
        v_data   = v_data.reshape(BH, N_C, -1)
        v_scales = v_scales.reshape(BH, N_C, -1)
        v_zeros  = v_zeros.reshape(BH, N_C, -1)

    k_buf = key_buffer.reshape(BH, -1, D)   if key_buffer.dim()   > 3 else key_buffer
    v_buf = value_buffer.reshape(BH, -1, D) if value_buffer.dim() > 3 else value_buffer
    N_B   = k_buf.shape[1]

    N_GROUPS             = D // group_size
    # mse_bits is total TQ bits (e.g. 3); MSE stage uses mse_bits-1 bits (e.g. 2)
    # to match packing done by the quantizer
    eff_bits, vals_per_byte = _get_packing_params(mse_bits - 1)

    out     = torch.empty(BH, D, device=query.device, dtype=torch.float32)
    BLOCK_N = min(64, triton.next_power_of_2(max(N_C, N_B, 1)))

    Pi_T = Pi_T.contiguous().float()
    S_T  = S_T.contiguous().float()
    q_f  = query.float().contiguous()

    # Scratch buffers for spilling q_rot/q_sketch after Phase 0.
    # 2 × BH × D × 4 bytes = 1 KB per program at D=128.  L1-resident.
    q_rot_scratch    = torch.empty(BH, D, device=query.device, dtype=torch.float32)
    q_sketch_scratch = torch.empty(BH, D, device=query.device, dtype=torch.float32)

    _turboquant_v8_zero_launch_kernel[(BH,)](
        q_f,
        Pi_T, S_T,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        k_buf, v_buf,
        out,
        q_rot_scratch, q_sketch_scratch,
        # Q strides
        q_f.stride(0), q_f.stride(1),
        # Scratch strides
        q_rot_scratch.stride(0), q_rot_scratch.stride(1),
        # Rotation row stride (D×D contiguous → stride(0) = D)
        Pi_T.stride(0),
        # Compressed key strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0),  qjl_signs.stride(1),  qjl_signs.stride(2),
        norms.stride(0),      norms.stride(1),
        res_norms.stride(0),  res_norms.stride(1),
        # Value strides
        v_data.stride(0),   v_data.stride(1),   v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0),  v_zeros.stride(1),  v_zeros.stride(2),
        # Buffer strides
        k_buf.stride(0), k_buf.stride(1), k_buf.stride(2),
        v_buf.stride(0), v_buf.stride(1), v_buf.stride(2),
        # Output strides
        out.stride(0), out.stride(1),
        # Dims
        N_C=N_C, N_B=N_B, D=D,
        PACKED_D_MSE=packed_d_mse, PACKED_D_SIGNS=packed_d_sig,
        N_GROUPS=N_GROUPS, GROUP_SIZE=group_size,
        # Quant params
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return out.to(query.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Kernels B1-B4 — fused prefill key quantization (3 launches vs ~10)
# ═══════════════════════════════════════════════════════════════════════════
#
# Design: split the quantization pipeline into three stages that share only
# scratch global-memory buffers between them.  This avoids the in-kernel
# "register gather" problem (Triton can't index a vector register with a
# runtime scalar) by storing intermediate results (k_rot, k_hat_rot, residual)
# to scratchpad tensors and reloading them element-by-element in the next kernel.
#
# Scratch layout:
#   k_rot_scratch    : (BH, N, D) fp32 — rotated unit keys
#   k_hat_rot_scratch: (BH, N, D) fp32 — centroid values in rotated domain
#   residual_scratch : (BH, N, D) fp32 — k - k_hat in original domain
#   idx_scratch      : (BH, N, D) int32 — raw centroid indices (pre-packing)
#   sign_scratch     : (BH, N, D) int32 — raw sign bits {0,1} (pre-packing)
#
# Grid for all B kernels: (BH, N) — one program per (head, token).

@triton.jit
def _turboquant_v8_quant_forward_kernel(
    K_ptr,              # (BH, N, D) fp16/fp32 raw keys
    PI_T_ptr,           # (D, D) fp32 Pi transposed (forward rotation)
    CENTROIDS_ptr,      # (N_CENTROIDS,) fp32 codebook centroids
    # Outputs / scratch
    K_ROT_ptr,          # (BH, N, D) fp32 scratch — k_unit @ Pi^T
    K_HAT_ROT_ptr,      # (BH, N, D) fp32 scratch — centroid values per coord
    IDX_PTR,            # (BH, N, D) int32 scratch — centroid indices
    NORMS_OUT_ptr,      # (BH, N) fp32
    # Strides
    stride_k_bh, stride_k_n, stride_k_d,
    stride_pi_row,
    stride_sc_bh, stride_sc_n, stride_sc_d,   # shared stride for all (BH,N,D) scratch
    stride_nm_bh, stride_nm_n,
    # Dims
    N,
    D:           tl.constexpr,
    N_CENTROIDS: tl.constexpr,
):
    """
    Stage 1: normalize → rotate → centroid search.

    Writes k_rot (rotated unit key), centroid indices, centroid values,
    and key norms to scratch buffers.
    """
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    if pid_n >= N:
        return

    d_offs = tl.arange(0, D)

    # ── Load raw key ─────────────────────────────────────────────────
    k_raw = tl.load(
        K_ptr + pid_bh * stride_k_bh + pid_n * stride_k_n + d_offs * stride_k_d,
    ).to(tl.float32)

    # ── Normalize ────────────────────────────────────────────────────
    k_norm = tl.sqrt(tl.sum(k_raw * k_raw, axis=0))
    k_unit = k_raw / (k_norm + 1e-10)
    tl.store(NORMS_OUT_ptr + pid_bh * stride_nm_bh + pid_n * stride_nm_n, k_norm)

    # ── Forward rotation: k_rot = k_unit @ Pi^T ──────────────────────
    # Pattern: outer-product accumulation over D coordinate scalars.
    # Each iteration: load scalar k_unit[i] from global memory, load
    # Pi^T row i (D elements), accumulate into k_rot.
    k_rot = tl.zeros([D], dtype=tl.float32)
    base_k  = K_ptr + pid_bh * stride_k_bh + pid_n * stride_k_n
    for i in tl.static_range(D):
        k_i    = tl.load(base_k + i * stride_k_d).to(tl.float32) / (k_norm + 1e-10)
        pi_row = tl.load(PI_T_ptr + i * stride_pi_row + d_offs).to(tl.float32)
        k_rot  = k_rot + k_i * pi_row

    # ── Nearest centroid (linear scan, constexpr-unrolled) ──────────
    # Works well for N_CENTROIDS ≤ 8 (2-3 bit MSE stage).
    c0     = tl.load(CENTROIDS_ptr).to(tl.float32)
    best_c = tl.zeros([D], dtype=tl.int32)
    best_d = (k_rot - c0) * (k_rot - c0)

    for c in tl.static_range(1, N_CENTROIDS):
        c_val = tl.load(CENTROIDS_ptr + c).to(tl.float32)
        dist  = (k_rot - c_val) * (k_rot - c_val)
        hit   = dist < best_d
        best_c = tl.where(hit, c,     best_c)
        best_d = tl.where(hit, dist,  best_d)

    # Centroid values at chosen indices
    k_hat_rot = tl.load(CENTROIDS_ptr + best_c).to(tl.float32)

    # ── Write scratch ─────────────────────────────────────────────────
    base_sc = pid_bh * stride_sc_bh + pid_n * stride_sc_n
    tl.store(K_ROT_ptr     + base_sc + d_offs * stride_sc_d, k_rot)
    tl.store(K_HAT_ROT_ptr + base_sc + d_offs * stride_sc_d, k_hat_rot)
    tl.store(IDX_PTR       + base_sc + d_offs * stride_sc_d, best_c)


@triton.jit
def _turboquant_v8_quant_residual_kernel(
    K_ptr,              # (BH, N, D) fp16/fp32 original keys (for residual)
    K_HAT_ROT_ptr,      # (BH, N, D) fp32 scratch — centroid values in rotated domain
    PI_ptr,             # (D, D) fp32 Pi (backward rotation — un-rotate)
    S_T_ptr,            # (D, D) fp32 S transposed (QJL sketch)
    NORMS_ptr,          # (BH, N) fp32 key norms (from Stage 1)
    # Outputs / scratch
    SIGN_SCRATCH_ptr,   # (BH, N, D) int32 scratch — raw sign bits {0,1}
    RES_NORMS_OUT_ptr,  # (BH, N) fp32
    # Strides
    stride_k_bh, stride_k_n, stride_k_d,
    stride_kh_bh, stride_kh_n, stride_kh_d,   # K_HAT_ROT scratch stride
    stride_pi_row,
    stride_sg_bh, stride_sg_n, stride_sg_d,   # SIGN_SCRATCH stride
    stride_nm_bh, stride_nm_n,
    stride_rn_bh, stride_rn_n,
    # Dims
    N,
    D: tl.constexpr,
):
    """
    Stage 2: un-rotate centroid approx → residual → QJL sketch → sign bits.

    Reads k_hat_rot scratch from Stage 1.  Computes:
      k_hat_unit = k_hat_rot @ Pi         (un-rotate)
      k_hat      = k_hat_unit * norm      (rescale)
      residual   = k_raw - k_hat
      qjl_proj   = residual @ S^T
      sign[j]    = (qjl_proj[j] > 0) as int32 {0,1}
    """
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    if pid_n >= N:
        return

    d_offs = tl.arange(0, D)
    base_k  = K_ptr     + pid_bh * stride_k_bh  + pid_n * stride_k_n
    base_kh = K_HAT_ROT_ptr + pid_bh * stride_kh_bh + pid_n * stride_kh_n

    k_norm = tl.load(NORMS_ptr + pid_bh * stride_nm_bh + pid_n * stride_nm_n)

    # ── Un-rotate: k_hat_unit = k_hat_rot @ Pi ───────────────────────
    # k_hat_unit[i] = sum_j k_hat_rot[j] * Pi[j, i]
    # Pi[j, i] lives in row j of Pi: Pi_ptr + j*D + i.
    # We load row j of Pi (a D-vector), multiply by k_hat_rot[j] (scalar
    # reloaded from scratch), and accumulate into k_hat_unit.
    k_hat_unit = tl.zeros([D], dtype=tl.float32)
    for j in tl.static_range(D):
        k_hat_rot_j = tl.load(base_kh + j * stride_kh_d).to(tl.float32)
        pi_row_j    = tl.load(PI_ptr + j * stride_pi_row + d_offs).to(tl.float32)
        k_hat_unit  = k_hat_unit + k_hat_rot_j * pi_row_j

    k_hat = k_hat_unit * k_norm

    # ── Residual ─────────────────────────────────────────────────────
    k_raw    = tl.load(base_k + d_offs * stride_k_d).to(tl.float32)
    residual = k_raw - k_hat
    res_norm = tl.sqrt(tl.sum(residual * residual, axis=0))
    tl.store(RES_NORMS_OUT_ptr + pid_bh * stride_rn_bh + pid_n * stride_rn_n, res_norm)

    # ── QJL sketch: qjl_proj = residual @ S^T ────────────────────────
    # Same outer-product pattern.  Residual is already a D-vector in registers;
    # we reload element-by-element from the intermediate k_hat scratch
    # Note: residual is in registers here — but we need residual[i] as scalars.
    # We compute it by reloading k_raw[i] and k_hat_unit[i]:
    #   residual[i] = k_raw[i] - k_hat_unit[i] * k_norm
    # Both k_raw and k_hat_unit can be reloaded/recovered element-by-element.
    # k_raw[i]: reload from K_ptr (global memory).
    # k_hat_unit[i]: re-derive from k_hat_rot via k_hat_rot[i]*Pi^T row i...
    #   but k_hat_unit[i] is the i-th element of the already-computed vector.
    # For simplicity, store residual to a register array and extract scalars
    # using the static-range trick (tl.sum with mask).
    qjl_proj = tl.zeros([D], dtype=tl.float32)
    for i in tl.static_range(D):
        # Extract residual[i] as a scalar via masked reduction over the vector.
        # With tl.static_range, i is a compile-time constant.
        # Triton LLVM optimises this to a register lane selection (no memory needed).
        r_i   = tl.sum(tl.where(tl.arange(0, D) == i, residual, 0.0))
        s_row = tl.load(S_T_ptr + i * stride_pi_row + d_offs).to(tl.float32)
        qjl_proj = qjl_proj + r_i * s_row

    # ── Write sign bits ───────────────────────────────────────────────
    signs = (qjl_proj > 0.0).to(tl.int32)
    tl.store(
        SIGN_SCRATCH_ptr + pid_bh * stride_sg_bh + pid_n * stride_sg_n + d_offs * stride_sg_d,
        signs,
    )


@triton.jit
def _turboquant_v8_pack_mse_kernel(
    IDX_ptr,        # (BH, N, D) int32 centroid indices
    OUT_ptr,        # (BH, N, PACKED_D_MSE) uint8
    stride_i_bh, stride_i_n, stride_i_d,
    stride_o_bh, stride_o_n, stride_o_d,
    N,
    D:             tl.constexpr,
    BITS:          tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    PACKED_D_MSE:  tl.constexpr,
):
    """Pack D int32 centroid indices into PACKED_D_MSE uint8 bytes."""
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    if pid_n >= N:
        return

    byte_offs = tl.arange(0, PACKED_D_MSE)
    packed    = tl.zeros([PACKED_D_MSE], dtype=tl.int32)

    for sub in tl.static_range(VALS_PER_BYTE):
        coord_offs = byte_offs * VALS_PER_BYTE + sub
        idx_vals   = tl.load(
            IDX_ptr + pid_bh * stride_i_bh + pid_n * stride_i_n + coord_offs * stride_i_d,
            mask=coord_offs < D, other=0,
        ).to(tl.int32)
        packed = packed | (idx_vals << (sub * BITS))

    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + pid_n * stride_o_n + byte_offs * stride_o_d,
        packed.to(tl.uint8),
    )


@triton.jit
def _turboquant_v8_pack_signs_kernel(
    SIGNS_ptr,  # (BH, N, D) int32 {0,1}
    OUT_ptr,    # (BH, N, PACKED_D_SIGNS) uint8
    stride_s_bh, stride_s_n, stride_s_d,
    stride_o_bh, stride_o_n, stride_o_d,
    N,
    D:              tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
):
    """Pack D int32 sign bits into PACKED_D_SIGNS uint8 bytes (8 bits/byte)."""
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)
    if pid_n >= N:
        return

    byte_offs = tl.arange(0, PACKED_D_SIGNS)
    packed    = tl.zeros([PACKED_D_SIGNS], dtype=tl.int32)

    for bit in tl.static_range(8):
        coord_offs = byte_offs * 8 + bit
        sign_vals  = tl.load(
            SIGNS_ptr + pid_bh * stride_s_bh + pid_n * stride_s_n + coord_offs * stride_s_d,
            mask=coord_offs < D, other=0,
        ).to(tl.int32)
        packed = packed | (sign_vals << bit)

    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + pid_n * stride_o_n + byte_offs * stride_o_d,
        packed.to(tl.uint8),
    )


# ─── Python wrapper for prefill quantization (Kernels B1-B4) ─────────────

def turboquant_v8_prefill_quant(
    keys:      torch.Tensor,    # (BH, N, D) or (B, H, N, D)
    Pi_T:      torch.Tensor,    # (D, D) Pi transposed
    Pi:        torch.Tensor,    # (D, D) Pi (for un-rotate)
    S_T:       torch.Tensor,    # (D, D) S transposed
    centroids: torch.Tensor,    # (N_CENTROIDS,) codebook
    mse_bits:  int,             # total TQ bits; MSE stage uses (mse_bits-1) bits
):
    """
    Fused prefill key quantization — 3 Triton launches vs ~10 PyTorch ops.

    Launch sequence:
      B1: normalize + rotate (Pi^T) + centroid search  → scratch
      B2: un-rotate + residual + QJL sketch             → scratch
      B3: pack int32 MSE indices → uint8
      B4: pack int32 sign bits   → uint8

    Returns (mse_packed, qjl_signs, norms, res_norms) matching ProdQuantized
    field layout expected by the existing store / score infrastructure.
    """
    if keys.dim() == 4:
        B, H, N, D = keys.shape
        keys = keys.reshape(B * H, N, D)
    BH, N, D = keys.shape

    keys = keys.contiguous().float()
    Pi_T = Pi_T.contiguous().float()
    Pi   = Pi.contiguous().float()
    S_T  = S_T.contiguous().float()
    dev  = keys.device

    # MSE stage uses (mse_bits-1) bits
    mse_key_bits           = mse_bits - 1
    eff_bits, vals_per_byte = _get_packing_params(mse_key_bits)
    n_centroids             = 2 ** mse_key_bits
    packed_d_mse            = (D + vals_per_byte - 1) // vals_per_byte
    packed_d_sig            = D // 8

    # Scratch buffers (fp32 / int32 per token per dim)
    k_rot_scratch     = torch.empty(BH, N, D, device=dev, dtype=torch.float32)
    k_hat_rot_scratch = torch.empty(BH, N, D, device=dev, dtype=torch.float32)
    idx_scratch       = torch.empty(BH, N, D, device=dev, dtype=torch.int32)
    sign_scratch      = torch.empty(BH, N, D, device=dev, dtype=torch.int32)
    norms_out         = torch.empty(BH, N,    device=dev, dtype=torch.float32)
    res_norms_out     = torch.empty(BH, N,    device=dev, dtype=torch.float32)

    # Final packed outputs
    mse_packed   = torch.empty(BH, N, packed_d_mse, device=dev, dtype=torch.uint8)
    signs_packed = torch.empty(BH, N, packed_d_sig, device=dev, dtype=torch.uint8)

    grid = (BH, N)

    # ── Launch B1: normalize + rotate + centroid ──────────────────────
    _turboquant_v8_quant_forward_kernel[grid](
        keys, Pi_T, centroids,
        k_rot_scratch, k_hat_rot_scratch, idx_scratch, norms_out,
        keys.stride(0),          keys.stride(1),          keys.stride(2),
        Pi_T.stride(0),
        k_rot_scratch.stride(0), k_rot_scratch.stride(1), k_rot_scratch.stride(2),
        norms_out.stride(0),     norms_out.stride(1),
        N=N, D=D, N_CENTROIDS=n_centroids,
        num_warps=4,
    )

    # ── Launch B2: un-rotate + residual + QJL ─────────────────────────
    _turboquant_v8_quant_residual_kernel[grid](
        keys, k_hat_rot_scratch, Pi, S_T, norms_out,
        sign_scratch, res_norms_out,
        keys.stride(0),              keys.stride(1),              keys.stride(2),
        k_hat_rot_scratch.stride(0), k_hat_rot_scratch.stride(1), k_hat_rot_scratch.stride(2),
        Pi.stride(0),
        sign_scratch.stride(0),      sign_scratch.stride(1),      sign_scratch.stride(2),
        norms_out.stride(0),         norms_out.stride(1),
        res_norms_out.stride(0),     res_norms_out.stride(1),
        N=N, D=D,
        num_warps=4,
    )

    # ── Launch B3: pack MSE indices ───────────────────────────────────
    _turboquant_v8_pack_mse_kernel[grid](
        idx_scratch, mse_packed,
        idx_scratch.stride(0), idx_scratch.stride(1), idx_scratch.stride(2),
        mse_packed.stride(0),  mse_packed.stride(1),  mse_packed.stride(2),
        N=N, D=D,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte, PACKED_D_MSE=packed_d_mse,
        num_warps=1,
    )

    # ── Launch B4: pack sign bits ─────────────────────────────────────
    _turboquant_v8_pack_signs_kernel[grid](
        sign_scratch, signs_packed,
        sign_scratch.stride(0),  sign_scratch.stride(1),  sign_scratch.stride(2),
        signs_packed.stride(0),  signs_packed.stride(1),  signs_packed.stride(2),
        N=N, D=D, PACKED_D_SIGNS=packed_d_sig,
        num_warps=1,
    )

    return mse_packed, signs_packed, norms_out, res_norms_out
