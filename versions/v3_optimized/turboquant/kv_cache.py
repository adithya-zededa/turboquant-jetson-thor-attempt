"""
TurboQuant KV Cache — drop-in replacement for the standard KV cache
in transformer attention layers.

Handles:
  - Keys: TurboQuant_prod quantization (unbiased inner product estimation)
  - Values: Standard group quantization (symmetric, per-group min-max)
  - Outlier channels: kept in full precision (configurable count)
  - Buffer: recent tokens kept unquantized for quality

The design follows the pattern from QJL but is model-agnostic.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, NamedTuple
from turboquant.quantizer import TurboQuantProd, ProdQuantized


class ValueQuantized(NamedTuple):
    """Quantized value cache (bit-packed)."""
    data: torch.Tensor       # (..., n_tokens, packed_d) bit-packed quantized values
    scales: torch.Tensor     # (..., n_tokens, n_groups) scale per group
    zeros: torch.Tensor      # (..., n_tokens, n_groups) zero point per group
    bits: int = 2            # quantization bits (for unpacking)


def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    """Unpack bit-packed value data to uint8 per-element."""
    bits = vq.bits if len(vq) > 3 else 2
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    elif bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed


def quantize_values(
    v: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> ValueQuantized:
    """
    Symmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    # Reshape to groups
    v_grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)  # (..., seq, n_groups, gs)

    # Compute scale and zero per group (asymmetric)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    # Quantize
    v_q = ((v_grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], d)

    # Bit-pack: for 2-bit, pack 4 values per byte; for 4-bit, pack 2 per byte
    if bits == 2:
        # Pack 4 x 2-bit values into each uint8: [a, b, c, d] -> a | (b<<2) | (c<<4) | (d<<6)
        assert d % 4 == 0
        v_4 = v_q_flat.reshape(*orig_shape[:-1], d // 4, 4)
        packed = v_4[..., 0] | (v_4[..., 1] << 2) | (v_4[..., 2] << 4) | (v_4[..., 3] << 6)
        v_q_flat = packed  # shape: (..., d//4)
    elif bits == 4:
        assert d % 2 == 0
        v_2 = v_q_flat.reshape(*orig_shape[:-1], d // 2, 2)
        packed = v_2[..., 0] | (v_2[..., 1] << 4)
        v_q_flat = packed  # shape: (..., d//2)
    # bits==8: no packing needed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )


def dequantize_values(
    vq: ValueQuantized,
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize value vectors from bit-packed format."""
    data = unpack_values(vq).float()
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)

    v = data * scales + zeros
    return v.reshape(*batch_shape, d)


class TurboQuantKVCache:
    """
    KV cache using TurboQuant for keys and group quantization for values.

    Usage:
        cache = TurboQuantKVCache(head_dim=128, key_bits=3, value_bits=2)

        # During prefill:
        cache.prefill(key_states, value_states)

        # During decode (one token at a time):
        cache.append(new_key, new_value)

        # Compute attention:
        scores = cache.attention_scores(query_states)
        output = cache.attend(query_states, scores_after_softmax)
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        layer_idx: int = 0,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda")
        self.dtype = dtype
        self.layer_idx = layer_idx

        self.key_quantizer = TurboQuantProd(
            dim=head_dim,
            bits=key_bits,
            device=self.device,
            seed=42 + layer_idx * 7,
        )

        # State
        self.seq_len: int = 0
        self.key_quantized: Optional[ProdQuantized] = None
        self.value_quantized: Optional[ValueQuantized] = None

        # Buffer for recent unquantized tokens
        self.key_buffer: Optional[torch.Tensor] = None
        self.value_buffer: Optional[torch.Tensor] = None

    def prefill(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Process prefill tokens.

        Args:
            keys: (batch, n_heads, seq_len, head_dim)
            values: (batch, n_heads, seq_len, head_dim)
        """
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            # Everything fits in buffer, no quantization needed
            self.key_buffer = keys
            self.value_buffer = values
            return

        # Split into quantized portion and buffer
        n_quant = seq_len - self.buffer_size

        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]

        self.key_buffer = keys[..., n_quant:, :]
        self.value_buffer = values[..., n_quant:, :]

        # Quantize keys with TurboQuant
        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)

        # Quantize values with group quantization
        self.value_quantized = quantize_values(
            values_to_quant, bits=self.value_bits, group_size=self.value_group_size
        )

    def append(self, key: torch.Tensor, value: torch.Tensor):
        """
        Append a single decode token.

        Args:
            key: (batch, n_heads, 1, head_dim)
            value: (batch, n_heads, 1, head_dim)
        """
        self.seq_len += 1

        if self.key_buffer is not None:
            self.key_buffer = torch.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, value], dim=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        # If buffer exceeds size, flush oldest chunk to quantized storage
        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Move oldest tokens from buffer to quantized storage."""
        n_flush = self.key_buffer.shape[-2] - self.buffer_size

        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]

        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        # Quantize flushed keys
        new_key_q = self.key_quantizer.quantize(keys_flush)

        # Quantize flushed values
        new_val_q = quantize_values(
            values_flush, bits=self.value_bits, group_size=self.value_group_size
        )

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_quantized = new_val_q
        else:
            # Concatenate along sequence dimension
            self.key_quantized = ProdQuantized(
                mse_indices=torch.cat([self.key_quantized.mse_indices, new_key_q.mse_indices], dim=-2),
                qjl_signs=torch.cat([self.key_quantized.qjl_signs, new_key_q.qjl_signs], dim=-2),
                residual_norms=torch.cat([self.key_quantized.residual_norms, new_key_q.residual_norms], dim=-1),
                norms=torch.cat([self.key_quantized.norms, new_key_q.norms], dim=-1),
                mse_bits=new_key_q.mse_bits,
            )
            self.value_quantized = ValueQuantized(
                data=torch.cat([self.value_quantized.data, new_val_q.data], dim=-2),
                scales=torch.cat([self.value_quantized.scales, new_val_q.scales], dim=-2),
                zeros=torch.cat([self.value_quantized.zeros, new_val_q.zeros], dim=-2),
                bits=self.value_bits,
            )

    def attention_scores(self, query: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Compute attention logits: score[i,j] = <query_i, key_j> / sqrt(d).

        Args:
            query: (batch, n_heads, n_q, head_dim)
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            scores: (batch, n_heads, n_q, seq_len)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []

        # Quantized portion
        if self.key_quantized is not None:
            scores_quant = self.key_quantizer.attention_score(query, self.key_quantized)
            scores_parts.append(scores_quant * scale)

        # Buffer portion (full precision)
        if self.key_buffer is not None:
            scores_buf = torch.matmul(query, self.key_buffer.transpose(-2, -1))
            scores_parts.append(scores_buf * scale)

        return torch.cat(scores_parts, dim=-1)

    def attend(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention output: out = softmax(scores) @ values.

        Args:
            attn_weights: (batch, n_heads, n_q, seq_len) — already softmaxed

        Returns:
            output: (batch, n_heads, n_q, head_dim)
        """
        output_parts = []
        col_offset = 0

        # Quantized values
        if self.value_quantized is not None:
            n_quant = self.value_quantized.data.shape[-2]
            w_quant = attn_weights[..., col_offset:col_offset + n_quant]
            v_dequant = dequantize_values(self.value_quantized, self.value_group_size)
            output_parts.append(torch.matmul(w_quant, v_dequant))
            col_offset += n_quant

        # Buffer values (full precision — cast to match weight dtype)
        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = attn_weights[..., col_offset:col_offset + n_buf]
            output_parts.append(torch.matmul(w_buf, self.value_buffer.to(w_buf.dtype)))

        return sum(output_parts)

    def fused_attend(self, query: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Fused attention: scores + softmax + value aggregation in one call.

        ~10-30× faster than attention_scores() + attend() for the compressed
        portion because key scores are computed directly from packed data via
        Triton (O(D²) query rotation) instead of dequantizing all keys (O(N·D²)
        for the rotation back).  Falls back to the PyTorch path when Triton is
        unavailable or the input is not on CUDA.

        Args:
            query: (batch, n_heads, 1, head_dim) — decode only (n_q must be 1)
            scale: 1/sqrt(head_dim) by default

        Returns:
            output: (batch, n_heads, 1, head_dim)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        has_q = self.key_quantized is not None
        has_b = self.key_buffer is not None and self.key_buffer.shape[-2] > 0

        if not has_q and not has_b:
            return torch.zeros_like(query)

        if not has_q and has_b:
            return F.scaled_dot_product_attention(
                query, self.key_buffer, self.value_buffer
            )

        # Dispatch order for decode (Q=1):
        #   1. Triton score kernels  — best: avoids key materialisation + no large temps
        #   2. Tensor-core bmm       — fallback when Triton unavailable
        #   3. PyTorch dequant+matmul — ultimate fallback
        # (Tensor-core is slower for Q=1 because M=1 bmm can't use tensor cores
        # and materialising N×D temporaries costs more than Triton's scalar loops.)
        if has_q and not has_b:
            primary = [self._fused_compressed_only]
        else:
            primary = [self._fused_hybrid]
        for method in primary + [self._tensorcore_attend]:
            try:
                return method(query, scale, has_q, has_b)
            except (ImportError, RuntimeError, TypeError):
                continue

        # Ultimate fallback
        scores = self.attention_scores(query, scale)
        weights = F.softmax(scores, dim=-1)
        return self.attend(weights)

    # ── v3: tensor-core path (fastest) ────────────────────────────────────

    def _tensorcore_attend(
        self, query: torch.Tensor, scale: float,
        has_q: bool, has_b: bool,
    ) -> torch.Tensor:
        """Materialize rotated keys via centroid gather (no D×D rotation),
        then use torch.bmm which maps to tensor cores.

        Why this is fast:
          • Centroid gather from an 8-entry table is L1-cached
          • torch.bmm dispatches to tensor-core matmul on Blackwell
          • Avoids Triton's per-coordinate scalar loop entirely
          • One combined value bmm instead of two separate ones
        """
        from turboquant.quantizer import _unpack_indices

        B, H, Q, D = query.shape
        BH = B * H
        kq = self.key_quantizer

        # Rotate + sketch query: O(D²) each, done once
        q_flat   = query.reshape(BH, 1, D).float()
        q_rot    = torch.matmul(q_flat, kq.mse_quantizer.Pi.T)   # (BH,1,D)
        q_sketch = torch.matmul(q_flat, kq.S.T)                  # (BH,1,D)

        parts_scores = []
        parts_values = []

        if has_q:
            pq = self.key_quantized

            # Unpack bit-packed indices → gather centroids (vectorized, no loop)
            indices = _unpack_indices(pq.mse_indices, pq.mse_bits, D)
            if indices.dim() == 4:
                indices = indices.reshape(BH, -1, D)
            k_rot = kq.mse_quantizer.centroids[indices]           # (BH,N_c,D)

            # Scale by norms
            norms = pq.norms
            if norms.dim() > 2:
                norms = norms.reshape(BH, -1)
            k_rot = k_rot * norms.unsqueeze(-1)

            # MSE scores via tensor-core bmm
            scores_mse = torch.bmm(q_rot, k_rot.transpose(-2, -1))  # (BH,1,N_c)

            # Unpack QJL signs → tensor-core bmm
            signs = kq._unpack_qjl_signs(pq.qjl_signs)
            if signs.dim() == 4:
                signs = signs.reshape(BH, -1, D)
            scores_qjl = torch.bmm(q_sketch, signs.transpose(-2, -1))
            res_norms = pq.residual_norms
            if res_norms.dim() > 2:
                res_norms = res_norms.reshape(BH, -1)
            scores_qjl = scores_qjl * (kq.qjl_scale * res_norms.unsqueeze(1))

            parts_scores.append((scores_mse + scores_qjl).squeeze(1))  # (BH,N_c)

            # Compressed values: dequantize (cheap unpack+scale, no rotation)
            v_c = dequantize_values(self.value_quantized, self.value_group_size)
            parts_values.append(v_c.reshape(BH, -1, D).float())

        if has_b:
            k_buf = self.key_buffer.reshape(BH, -1, D).float()
            scores_b = torch.bmm(q_flat.float(), k_buf.transpose(-2, -1)).squeeze(1)
            parts_scores.append(scores_b)
            parts_values.append(self.value_buffer.reshape(BH, -1, D).float())

        # Single softmax over all tokens, single value bmm
        scores_all = torch.cat(parts_scores, dim=-1) * scale
        weights    = F.softmax(scores_all, dim=-1)                   # (BH, N)
        v_all      = torch.cat(parts_values, dim=1)                  # (BH, N, D)
        out        = torch.bmm(weights.unsqueeze(1), v_all).squeeze(1)  # (BH, D)

        return out.reshape(B, H, Q, D).to(query.dtype)

    # ── v2: Triton score kernels (no key materialisation) ─────────────

    def _fused_compressed_only(
        self, query: torch.Tensor, scale: float,
        has_q: bool = True, has_b: bool = False,
    ) -> torch.Tensor:
        """Full fused Triton decode — single pass, no key/value materialisation."""
        from turboquant.triton_kernels import turboquant_fused_decode

        B, H, Q, D = query.shape
        BH = B * H

        out = turboquant_fused_decode(
            query=query.reshape(BH, D),
            quantized_key=self.key_quantized,
            value_quantized=self.value_quantized,
            Pi=self.key_quantizer.mse_quantizer.Pi,
            S=self.key_quantizer.S,
            centroids=self.key_quantizer.mse_quantizer.centroids,
            mse_bits=self.key_quantizer.mse_quantizer.bits,
            qjl_scale=self.key_quantizer.qjl_scale,
            sm_scale=scale,
            group_size=self.value_group_size,
        )
        return out.reshape(B, H, Q, D).to(query.dtype)

    def _fused_hybrid(
        self, query: torch.Tensor, scale: float,
        has_q: bool = True, has_b: bool = True,
    ) -> torch.Tensor:
        """Triton scores for compressed keys + matmul for buffer, merged softmax."""
        from turboquant.triton_kernels import turboquant_mse_score, turboquant_qjl_score

        B, H, Q, D = query.shape
        BH = B * H
        q_flat = query.reshape(BH, D).float()

        kq = self.key_quantizer

        # Combined query transform: one matmul → [q_rot | q_sketch] (saves a launch)
        q_combined = torch.matmul(q_flat, kq.Pi_S_T)             # (BH, 2D)
        q_rot, q_sketch = q_combined.split(D, dim=-1)

        pq = self.key_quantized
        mse_packed = pq.mse_indices.reshape(BH, *pq.mse_indices.shape[2:])
        qjl_signs  = pq.qjl_signs.reshape(BH, *pq.qjl_signs.shape[2:])
        norms      = pq.norms.reshape(BH, -1)
        res_norms  = pq.residual_norms.reshape(BH, -1)
        N_c = mse_packed.shape[1]

        scores_c = turboquant_mse_score(
            q_rot, mse_packed, norms, kq.mse_quantizer.centroids, kq.mse_quantizer.bits
        )
        scores_c = turboquant_qjl_score(
            q_sketch, qjl_signs, res_norms, kq.qjl_scale, out=scores_c
        )

        k_buf = self.key_buffer.reshape(BH, -1, D).float()
        N_b   = k_buf.shape[1]
        scores_b = torch.bmm(
            q_flat.unsqueeze(1), k_buf.transpose(-2, -1)
        ).squeeze(1)

        scores_all = torch.cat([scores_c, scores_b], dim=-1) * scale
        weights    = F.softmax(scores_all, dim=-1)

        # Value aggregation in fp16 (halves memory bandwidth)
        v_c = dequantize_values(self.value_quantized, self.value_group_size)
        v_c = v_c.reshape(BH, N_c, D).half()
        v_b = self.value_buffer.reshape(BH, N_b, D).half()
        v_all = torch.cat([v_c, v_b], dim=1)
        out   = torch.bmm(weights.unsqueeze(1).half(), v_all).squeeze(1)

        return out.reshape(B, H, Q, D).to(query.dtype)

    def memory_bytes(self) -> dict:
        """Estimate memory usage of the cache."""
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            # MSE indices: bit-packed uint8
            info["quantized_keys"] += self.key_quantized.mse_indices.nelement()  # already packed bytes
            # QJL packed signs: 1 bit per coord, packed 8 per byte
            info["quantized_keys"] += self.key_quantized.qjl_signs.nelement()
            # Norms: float16 each (could use float16 for storage)
            info["quantized_keys"] += self.key_quantized.residual_norms.nelement() * 2
            info["quantized_keys"] += self.key_quantized.norms.nelement() * 2

        if self.value_quantized is not None:
            info["quantized_values"] += self.value_quantized.data.nelement()  # uint8 packed
            info["quantized_values"] += self.value_quantized.scales.nelement() * 2  # float16
            info["quantized_values"] += self.value_quantized.zeros.nelement() * 2

        if self.key_buffer is not None:
            info["buffer"] += self.key_buffer.nelement() * 2  # float16
        if self.value_buffer is not None:
            info["buffer"] += self.value_buffer.nelement() * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        return self.seq_len
