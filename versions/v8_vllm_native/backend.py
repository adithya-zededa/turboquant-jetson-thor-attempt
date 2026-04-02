"""
TurboQuant v8 vLLM integration — composition over monkey-patching.

Current approach (v6/integration/vllm.py)
──────────────────────────────────────────
  Uses types.MethodType to replace methods on vLLM impl *instances* at runtime:
    impl.forward       = types.MethodType(patched_fn, impl)
    impl.do_kv_cache_update = types.MethodType(patched_fn, impl)

  Problems with this approach:
  • Fragile: any vLLM update that touches impl method dispatch breaks it.
  • Hard to type-check or introspect.
  • install_hooks() iterates the static_forward_context and patches in place,
    which means the patch cannot be easily undone or composed.

v8 approach — TurboQuantImpl wrapper (composition)
───────────────────────────────────────────────────
  TurboQuantImpl wraps the original impl via __getattr__ delegation:
    attn_module.impl = TurboQuantImpl(original_impl, tq_state)

  Benefits:
  • No types.MethodType. No function-scope closures over mutable state.
  • __getattr__ transparently forwards all attribute accesses vLLM needs
    (num_heads, num_kv_heads, head_size, scale, ...) to the wrapped impl.
  • install_v8() replaces the impl object itself — a single clean assignment
    instead of two method patches per layer.
  • Uninstall / swap is trivially: attn_module.impl = original_impl.

v8 decode path comparison
──────────────────────────
  v6: compute_hybrid_attention() in score.py
        → q @ Pi_S_T  (torch.matmul, 1 kernel launch)
        → turboquant_fused_hybrid() (1 Triton kernel)
      Total: 2 kernel launches

  v8: TurboQuantImpl.forward() for decode
        → turboquant_v8_decode()  (1 Triton kernel — rotation fused inside)
      Total: 1 kernel launch
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("turboquant.v8.backend")

# Import shared infrastructure from the main turboquant package.
# turboquant_v8_decode lives in this same versions/v8_vllm_native directory.
from turboquant.capture import KVCaptureEngine
from turboquant.store import CompressedKVStore

from .triton_kernels import turboquant_v8_decode, turboquant_v8_prefill_quant

# ─── Mode constants (kept identical to integration/vllm.py for compatibility) ─

MODE_OFF          = "off"
MODE_CAPTURE_ONLY = "capture_only"
MODE_HYBRID       = "hybrid"
MODE_FULL_TQ      = "full_tq"
_VALID_MODES      = (MODE_OFF, MODE_CAPTURE_ONLY, MODE_HYBRID, MODE_FULL_TQ)

_GLOBAL_MODE: str = MODE_CAPTURE_ONLY


def set_mode(mode: str) -> None:
    global _GLOBAL_MODE
    assert mode in _VALID_MODES, f"Invalid mode {mode!r}. Valid: {_VALID_MODES}"
    _GLOBAL_MODE = mode
    logger.info("[TurboQuant v8] Mode → %s", mode)


def get_mode() -> str:
    return _GLOBAL_MODE


# ─── Per-layer configuration & state ──────────────────────────────────────

@dataclass
class V8LayerConfig:
    head_dim:       int
    num_kv_heads:   int
    num_query_heads: int
    key_bits:       int   = 3
    value_bits:     int   = 2
    value_group_size: int = 32
    ring_capacity:  int   = 128
    layer_idx:      int   = 0
    device: torch.device  = field(default_factory=lambda: torch.device("cuda"))


@dataclass
class V8LayerState:
    config:  V8LayerConfig
    store:   CompressedKVStore
    engine:  KVCaptureEngine

    @property
    def supports_hybrid(self) -> bool:
        return True   # v8 only wraps flash-attention layers

    def reset(self) -> None:
        self.engine.reset()


def _create_v8_state(cfg: V8LayerConfig) -> V8LayerState:
    store = CompressedKVStore(
        head_dim=cfg.head_dim,
        num_kv_heads=cfg.num_kv_heads,
        key_bits=cfg.key_bits,
        value_bits=cfg.value_bits,
        value_group_size=cfg.value_group_size,
        device=cfg.device,
        layer_idx=cfg.layer_idx,
    )
    engine = KVCaptureEngine(
        store=store,
        ring_capacity=cfg.ring_capacity,
        device=cfg.device,
    )
    return V8LayerState(config=cfg, store=store, engine=engine)


# ─── TurboQuantImpl — wraps original impl via composition ─────────────────

class TurboQuantImpl:
    """
    Composition wrapper around a vLLM attention impl.

    Replaces the original impl on attn_module.impl.  All attribute accesses
    that vLLM queries (num_heads, head_size, scale, …) are forwarded to the
    wrapped impl via __getattr__, so vLLM's introspection continues to work
    without any modifications to vLLM itself.

    Decode attention is handled by turboquant_v8_decode (single Triton launch).
    Prefill uses the original impl (SDPA / FlashAttention) unchanged.
    KV capture happens in forward() and do_kv_cache_update() as before.
    """

    def __init__(
        self,
        wrapped_impl,
        state:   V8LayerState,
        no_alloc: bool = False,
    ) -> None:
        # Store in object dict directly to avoid triggering __setattr__ → __getattr__
        object.__setattr__(self, "_impl",     wrapped_impl)
        object.__setattr__(self, "_state",    state)
        object.__setattr__(self, "_no_alloc", no_alloc)

    # Forward unknown attribute access to the wrapped impl so vLLM can read
    # num_heads, head_size, scale, etc. without the wrapper being transparent.
    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_impl"), name)

    # ── KV capture (write path) ────────────────────────────────────────

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        impl     = object.__getattribute__(self, "_impl")
        state    = object.__getattribute__(self, "_state")
        no_alloc = object.__getattribute__(self, "_no_alloc")

        if not no_alloc:
            impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

        if _GLOBAL_MODE == MODE_OFF:
            return

        num_tokens = slot_mapping.shape[0]
        if num_tokens <= 1:
            state.engine.ingest_decode(key, value, num_tokens)
        else:
            state.engine.ingest_prefill(key, value, num_tokens)

    # ── Attention forward (read path) ──────────────────────────────────

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        impl     = object.__getattribute__(self, "_impl")
        state    = object.__getattribute__(self, "_state")
        no_alloc = object.__getattribute__(self, "_no_alloc")

        # ── Capture K/V if the backend has no separate do_kv_cache_update ─
        if (
            not hasattr(impl, "do_kv_cache_update")
            and _GLOBAL_MODE != MODE_OFF
            and attn_metadata is not None
        ):
            num_tokens = getattr(attn_metadata, "num_actual_tokens", key.shape[0])
            if num_tokens <= 1:
                state.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
            else:
                state.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)

        # ── Passthrough modes ──────────────────────────────────────────
        if _GLOBAL_MODE in (MODE_OFF, MODE_CAPTURE_ONLY):
            return impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # ── Profiling / dummy pass ─────────────────────────────────────
        if attn_metadata is None:
            return impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # ── Prefill: use original FlashAttention ───────────────────────
        is_prefill = attn_metadata.max_query_len > 1
        if is_prefill:
            if no_alloc:
                return _sdpa_prefill(state, impl, query, key, value, attn_metadata, output)
            return impl.forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # ── Decode: v8 single-launch path ─────────────────────────────
        if _GLOBAL_MODE == MODE_HYBRID:
            result = _v8_hybrid_decode(state, impl, query, attn_metadata)
            if result is not None:
                return _write_output(result, query, state, output)

        # Fallback to original impl (e.g. insufficient history)
        if no_alloc:
            num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
            if query.dim() == 3:
                return torch.zeros_like(query[:num_actual])
            return torch.zeros(
                num_actual,
                state.config.num_query_heads * state.config.head_dim,
                dtype=query.dtype, device=query.device,
            )
        return impl.forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )


# ─── Decode helper ────────────────────────────────────────────────────────

def _v8_hybrid_decode(
    state:        V8LayerState,
    impl,
    query:        torch.Tensor,
    attn_metadata,
) -> Optional[torch.Tensor]:
    """
    Run TurboQuant v8 decode: ONE Triton kernel launch.

    Returns (num_tokens, num_query_heads, head_dim) or None if history is
    too small to justify TQ overhead.
    """
    MIN_HISTORY = 16

    flat = state.store.get_flat_cache()
    if flat is None or flat.num_tokens < MIN_HISTORY:
        return None

    cfg        = state.config
    num_actual = getattr(attn_metadata, "num_actual_tokens", query.shape[0])
    q = query[:num_actual]
    if q.dim() == 2:
        q = q.view(num_actual, cfg.num_query_heads, cfg.head_dim)

    # Flatten (num_tokens, Q_heads, D) → (BH, D) for BH = Q_heads (single token decode)
    # GQA: expand KV to match Q head count inside turboquant_v8_decode
    gqa_ratio    = cfg.num_query_heads // cfg.num_kv_heads
    num_kv_heads = cfg.num_kv_heads
    head_dim     = cfg.head_dim

    # Build BH-indexed query: shape (BH, D) where BH = num_kv_heads * num_tokens
    # For single-token decode (num_actual=1): BH = num_query_heads
    q_bh = q.reshape(num_actual * cfg.num_query_heads, head_dim)  # (BH, D)

    # Retrieve rotation matrices from the quantizer
    quantizer = state.store.quantizer
    Pi_T = quantizer.mse_quantizer.Pi.T.contiguous()   # (D, D) Pi transposed
    S_T  = quantizer.S.T.contiguous()                  # (D, D) S transposed

    # Retrieve compressed KV and exact buffer
    prod_q = flat.prod_q
    value_q = flat.value_q

    # GQA: repeat compressed KV to match Q head count
    if gqa_ratio > 1:
        # mse_indices shape: (num_kv_heads, N, ...)
        # repeat to (num_query_heads, N, ...)
        from turboquant.store import FlatCache
        from turboquant.quantizer import ProdQuantized
        mi = prod_q.mse_indices.repeat_interleave(gqa_ratio, dim=0)
        qs = prod_q.qjl_signs.repeat_interleave(gqa_ratio, dim=0)
        nm = prod_q.norms.repeat_interleave(gqa_ratio, dim=0)
        rn = prod_q.residual_norms.repeat_interleave(gqa_ratio, dim=0)
        prod_q_bh = ProdQuantized(
            mse_indices=mi, qjl_signs=qs, residual_norms=rn,
            norms=nm, mse_bits=prod_q.mse_bits,
        )
        vd = value_q.data.repeat_interleave(gqa_ratio, dim=0)
        vs = value_q.scales.repeat_interleave(gqa_ratio, dim=0)
        vz = value_q.zeros.repeat_interleave(gqa_ratio, dim=0)
        from turboquant.kv_cache import ValueQuantized
        value_q_bh = ValueQuantized(data=vd, scales=vs, zeros=vz, bits=value_q.bits)
    else:
        prod_q_bh  = prod_q
        value_q_bh = value_q

    # Exact recent buffer
    recent = state.engine.ring.peek()
    if recent is not None:
        recent_k, recent_v = recent
        # recent_k: (N_recent, num_kv_heads, D) → transpose → (num_kv_heads, N_recent, D)
        k_buf = recent_k.transpose(0, 1)
        v_buf = recent_v.transpose(0, 1)
        if gqa_ratio > 1:
            k_buf = k_buf.repeat_interleave(gqa_ratio, dim=0)
            v_buf = v_buf.repeat_interleave(gqa_ratio, dim=0)
    else:
        # Empty buffer — create zero-length placeholders
        k_buf = torch.empty(cfg.num_query_heads, 0, head_dim, device=q.device, dtype=q.dtype)
        v_buf = torch.empty(cfg.num_query_heads, 0, head_dim, device=q.device, dtype=q.dtype)

    scale = getattr(impl, "scale", 1.0 / math.sqrt(head_dim))

    # ── One Triton kernel launch ──────────────────────────────────────
    out_bh = turboquant_v8_decode(
        query=q_bh,
        Pi_T=Pi_T,
        S_T=S_T,
        quantized_key=prod_q_bh,
        value_quantized=value_q_bh,
        key_buffer=k_buf.float(),
        value_buffer=v_buf.float(),
        centroids=quantizer.mse_quantizer.centroids,
        mse_bits=quantizer.mse_quantizer.bits + 1,  # mse_bits = total bits
        qjl_scale=quantizer.qjl_scale,
        sm_scale=scale,
        group_size=cfg.value_group_size,
    )  # → (BH, D) where BH = num_query_heads * num_actual

    # Reshape back to (num_actual, num_query_heads, D)
    return out_bh.reshape(num_actual, cfg.num_query_heads, head_dim)


def _sdpa_prefill(state, impl, query, key, value, attn_metadata, output):
    """SDPA fallback for no_alloc prefill (no paged KV cache)."""
    num_actual = attn_metadata.num_actual_tokens
    q = query[:num_actual]
    k = key[:num_actual]
    v = value[:num_actual]

    if q.dim() == 2:
        q = q.view(num_actual, state.config.num_query_heads, state.config.head_dim)
    if k.dim() == 2:
        k = k.view(num_actual, state.config.num_kv_heads, state.config.head_dim)
        v = v.view(num_actual, state.config.num_kv_heads, state.config.head_dim)

    if state.config.num_query_heads != state.config.num_kv_heads:
        rep = state.config.num_query_heads // state.config.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    scale = getattr(impl, "scale", 1.0 / math.sqrt(state.config.head_dim))
    out = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
        is_causal=True, scale=scale,
    ).squeeze(0).transpose(0, 1)  # (num_actual, Q_heads, D)

    return _write_output(out, query, state, output)


def _write_output(result, query, state, output):
    """Write attention result into vLLM's output buffer (or return directly)."""
    cfg        = state.config
    num_actual = result.shape[0]
    result_flat = result.reshape(
        num_actual, cfg.num_query_heads * cfg.head_dim
    ).to(query.dtype)

    if output is not None:
        out_slice = output[:num_actual]
        if out_slice.dim() == 3:
            out_slice.copy_(result.to(out_slice.dtype))
        else:
            out_slice.copy_(result_flat)
        return output

    if query.dim() == 3:
        return result.to(query.dtype)
    return result_flat


# ─── Public install API ───────────────────────────────────────────────────

def install_v8(
    model_runner,
    key_bits:             int  = 3,
    value_bits:           int  = 2,
    value_group_size:     int  = 32,
    ring_capacity:        int  = 128,
    initial_layers_count: int  = 4,
    initial_layers_key_bits: Optional[int] = None,
    mode:     str  = MODE_CAPTURE_ONLY,
    no_alloc: bool = False,
) -> dict:
    """
    Install TurboQuant v8 on all flash-attention layers in a vLLM model runner.

    Difference from install_hooks() in integration/vllm.py:
      • Replaces attn_module.impl with a TurboQuantImpl wrapper object.
      • No types.MethodType — no instance-level method patching.
      • Single assignment per layer: attn_module.impl = TurboQuantImpl(orig, state)

    Returns dict mapping layer_name → V8LayerState.
    """
    global _GLOBAL_MODE
    _GLOBAL_MODE = mode

    if initial_layers_key_bits is None:
        initial_layers_key_bits = min(key_bits + 1, 4)

    static_ctx = model_runner.compilation_config.static_forward_context
    device     = model_runner.device

    layer_states: dict[str, V8LayerState] = {}
    layer_idx = 0

    for layer_name, attn_module in static_ctx.items():
        if not hasattr(attn_module, "impl"):
            continue

        impl = attn_module.impl

        # Skip already-wrapped layers
        if isinstance(impl, TurboQuantImpl):
            logger.warning("[TurboQuant v8] Layer %s already wrapped, skipping", layer_name)
            continue

        num_kv_heads = getattr(impl, "num_kv_heads", None)
        if num_kv_heads is None:
            continue

        head_dim = getattr(impl, "head_size", None)
        if head_dim is None:
            continue

        # MLA / GDN layers are unsupported in v8 (same as v6)
        if _is_mla(impl):
            layer_idx += 1
            continue

        num_query_heads = _infer_q_heads(attn_module, impl)
        bits = initial_layers_key_bits if layer_idx < initial_layers_count else key_bits

        cfg = V8LayerConfig(
            head_dim=int(head_dim),
            num_kv_heads=int(num_kv_heads),
            num_query_heads=num_query_heads,
            key_bits=bits,
            value_bits=value_bits,
            value_group_size=min(value_group_size, int(head_dim)),
            ring_capacity=ring_capacity,
            layer_idx=layer_idx,
            device=device,
        )

        state = _create_v8_state(cfg)
        layer_states[layer_name] = state

        # ── Replace impl with composition wrapper (the v8 difference) ──
        attn_module.impl = TurboQuantImpl(impl, state, no_alloc=no_alloc)

        # Back-reference for diagnostics / free_kv_cache
        attn_module.impl._tq_layer_state = state

        layer_idx += 1

    model_runner._tq_v8_layer_states = layer_states
    model_runner._tq_no_alloc = no_alloc
    logger.info(
        "[TurboQuant v8] Wrapped %d layers (mode=%s, no_alloc=%s)",
        len(layer_states), mode, no_alloc,
    )
    return layer_states


def uninstall_v8(model_runner) -> int:
    """
    Unwrap TurboQuantImpl from all attention modules, restoring the original impls.

    Returns the number of layers unwrapped.
    """
    static_ctx   = model_runner.compilation_config.static_forward_context
    layer_states = getattr(model_runner, "_tq_v8_layer_states", {})
    unwrapped    = 0

    for layer_name in list(layer_states.keys()):
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        impl = attn_module.impl
        if isinstance(impl, TurboQuantImpl):
            # Restore original impl via __getattr__ delegation
            attn_module.impl = object.__getattribute__(impl, "_impl")
            unwrapped += 1

    model_runner._tq_v8_layer_states = {}
    logger.info("[TurboQuant v8] Unwrapped %d layers", unwrapped)
    return unwrapped


def free_kv_cache_v8(model_runner) -> int:
    """Free paged KV cache tensors for TQ-wrapped layers. Returns bytes freed."""
    import torch

    static_ctx   = model_runner.compilation_config.static_forward_context
    layer_states = getattr(model_runner, "_tq_v8_layer_states", {})
    device       = model_runner.device
    freed        = 0
    tiny         = torch.zeros(1, dtype=torch.int8, device=device)

    ptrs_to_free = set()
    for layer_name, state in layer_states.items():
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0 and hasattr(kv_list[0], "data_ptr"):
            ptrs_to_free.add(kv_list[0].data_ptr())

    for layer_name, state in layer_states.items():
        attn_module = static_ctx.get(layer_name)
        if attn_module is None:
            continue
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            old = kv_list[0]
            freed += old.nelement() * old.element_size()
            kv_list[0] = tiny

    for i in range(len(model_runner.kv_caches)):
        entry = model_runner.kv_caches[i]
        if isinstance(entry, list):
            for j in range(len(entry)):
                if hasattr(entry[j], "data_ptr") and entry[j].data_ptr() in ptrs_to_free:
                    entry[j] = tiny
        elif hasattr(entry, "data_ptr") and entry.data_ptr() in ptrs_to_free:
            model_runner.kv_caches[i] = tiny

    torch.cuda.empty_cache()
    logger.info("[TurboQuant v8] Freed %.0f MB KV cache", freed / 1e6)
    return freed


def get_stats_v8(model_runner) -> dict:
    """Return per-layer and aggregate statistics for v8 layer states."""
    layer_states = getattr(model_runner, "_tq_v8_layer_states", {})
    if not layer_states:
        return {}

    total_compressed = 0
    total_buffered   = 0
    total_memory     = 0

    for state in layer_states.values():
        total_compressed += state.store.num_tokens
        total_buffered   += state.engine.ring.size
        total_memory     += state.store.memory_bytes()

    n = max(len(layer_states), 1)
    return {
        "num_layers":             len(layer_states),
        "avg_compressed_tokens":  total_compressed // n,
        "avg_buffered_tokens":    total_buffered   // n,
        "total_memory_bytes":     total_memory,
        "mode":                   _GLOBAL_MODE,
        "version":                "v8",
    }


# ─── Internal helpers ─────────────────────────────────────────────────────

def _is_mla(impl) -> bool:
    return (
        hasattr(impl, "forward_mqa")
        and hasattr(impl, "do_kv_cache_update")
        and not hasattr(impl, "forward")
    )


def _infer_q_heads(attn_module, impl) -> int:
    for attr in ("num_heads", "num_attention_heads"):
        val = getattr(attn_module, attr, None) or getattr(impl, attr, None)
        if val:
            return int(val)
    return int(impl.num_kv_heads)
