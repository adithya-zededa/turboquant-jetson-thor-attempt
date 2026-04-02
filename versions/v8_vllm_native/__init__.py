"""
TurboQuant v8 — Zero External Kernel Launch (vLLM-native).

Two primary contributions over v6:

1. Fused Q rotation in the Triton decode kernel
   ─────────────────────────────────────────────
   v6: torch.matmul(q, Pi_S_T)  [kernel launch 1]
       + _turboquant_fused_hybrid_kernel [kernel launch 2]

   v8: _turboquant_v8_zero_launch_kernel [kernel launch 1, rotation fused inside]

   At D=128, Pi_T and S_T (each 64 KB) fit in L2 cache.  The rotation is done
   as a scalar outer-product accumulation (D scalar loads + D vector FMAs) with
   negligible register cost (8 fp32 regs/thread extra vs v6).

   Expected improvement: ~10-15 µs saved per decode step regardless of context
   length.  At short contexts (v6 ≈ 0.31 ms) this is ~4-5 %.  At long contexts
   the proportion shrinks but the absolute saving remains.

2. Composition-based vLLM integration  (no monkey-patching)
   ──────────────────────────────────────────────────────────
   v6: types.MethodType replaces two methods on each vLLM impl instance.
   v8: attn_module.impl = TurboQuantImpl(original_impl, state)
       __getattr__ delegation preserves full vLLM API compatibility.

   Benefits: cleaner install/uninstall, no closure-over-mutable-state bugs,
   easy introspection, one-line removal (attn_module.impl = original).

3. Fused prefill key quantization (bonus)
   ───────────────────────────────────────
   ~10 PyTorch ops → 3 Triton kernel launches (B1: normalize+rotate+centroid,
   B2: un-rotate+residual+QJL, B3/B4: bit-pack).
   Grid (BH × N) gives full SM utilisation during prompt processing.

Performance comparison (Jetson AGX Thor, D=128, 8 KV heads)
──────────────────────────────────────────────────────────────
   Version | ctx=1k  | ctx=16k  | Launches/decode | Key innovation
   --------+---------+----------+-----------------+---------------------------
   SDPA    | 0.04 ms | 0.37 ms  | 1               | baseline
   v6      | 0.31 ms | 1.51 ms  | 2               | fused hybrid + vectorised
   v8 est. | 0.28 ms | 1.48 ms  | 1               | Q rotation fused in kernel
"""

from .triton_kernels import turboquant_v8_decode, turboquant_v8_prefill_quant
from .backend import (
    TurboQuantImpl,
    V8LayerConfig,
    V8LayerState,
    install_v8,
    uninstall_v8,
    free_kv_cache_v8,
    get_stats_v8,
    set_mode,
    get_mode,
    MODE_OFF,
    MODE_CAPTURE_ONLY,
    MODE_HYBRID,
    MODE_FULL_TQ,
)

__all__ = [
    "turboquant_v8_decode",
    "turboquant_v8_prefill_quant",
    "TurboQuantImpl",
    "V8LayerConfig",
    "V8LayerState",
    "install_v8",
    "uninstall_v8",
    "free_kv_cache_v8",
    "get_stats_v8",
    "set_mode",
    "get_mode",
    "MODE_OFF",
    "MODE_CAPTURE_ONLY",
    "MODE_HYBRID",
    "MODE_FULL_TQ",
]
