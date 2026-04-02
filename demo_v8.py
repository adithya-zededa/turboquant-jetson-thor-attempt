#!/usr/bin/env python3
"""
TurboQuant v8 — Zero External Kernel Launch Benchmark
======================================================

Validates and benchmarks the v8 "vLLM-native" implementation:
  1. Hardware info
  2. v8 decode kernel correctness (vs PyTorch reference)
  3. v8 decode kernel throughput (vs v6 fused hybrid)
  4. v8 prefill quantization kernel correctness + throughput
  5. End-to-end attention benchmark: v8 vs v6 vs SDPA fp16
  6. vLLM integration: baseline vs TurboQuant v8

Usage:
    # Inside the Docker container:
    python3 /workspace/demo_v8.py

    # Via run_jetson.sh:
    ./run_jetson.sh v8

    # Skip vLLM section (kernel-only benchmarks):
    SKIP_VLLM=1 python3 /workspace/demo_v8.py
"""

import sys, os, math, time, json

# ── Path setup ────────────────────────────────────────────────────────────
for _p in [
    "/workspace/turboquant",
    os.path.join(os.path.dirname(__file__), "turboquant"),
]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Add project root so `versions.v8_vllm_native` is importable
for _p in ["/workspace", os.path.dirname(os.path.abspath(__file__))]:
    if os.path.isdir(os.path.join(_p, "versions", "v8_vllm_native")) and _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float16
SKIP_VLLM = os.environ.get("SKIP_VLLM", "0") == "1"
MODEL     = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU_MEM   = float(os.environ.get("GPU_MEM", "0.70"))
MAX_LEN   = int(os.environ.get("MAX_MODEL_LEN", "4096"))
KEY_BITS  = int(os.environ.get("KEY_BITS", "3"))
VAL_BITS  = int(os.environ.get("VAL_BITS", "2"))
RING_CAP  = int(os.environ.get("RING_CAPACITY", "128"))
HEAD_DIM  = 128
N_KV_HEADS = 8
TRITON_OK  = False

def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

def ok(msg):   print(f"  [OK]  {msg}")
def info(msg): print(f"        {msg}")
def warn(msg): print(f"  [!!]  {msg}")

def _bench(fn, warmup=5, reps=20):
    """Benchmark a callable, return ms/call."""
    if not torch.cuda.is_available():
        fn()
        return float("nan")
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


# ══════════════════════════════════════════════════════════════════════════
# 1. Hardware Info
# ══════════════════════════════════════════════════════════════════════════

section("1 - Hardware Info")

if not torch.cuda.is_available():
    warn("CUDA not available — kernel tests will be skipped.")
else:
    props = torch.cuda.get_device_properties(0)
    info(f"GPU           : {props.name}")
    info(f"Arch          : SM {props.major}.{props.minor}")
    info(f"Memory        : {props.total_memory / 1024**3:.1f} GB")
    info(f"SMs           : {props.multi_processor_count}")
    info(f"L2 cache      : {props.L2_cache_size / 1024**2:.0f} MB")
    info(f"PyTorch       : {torch.__version__}")
    try:
        import triton
        TRITON_OK = True
        info(f"Triton        : {triton.__version__}")
    except ImportError:
        warn("Triton not found — kernel tests will be skipped.")


# ══════════════════════════════════════════════════════════════════════════
# 2. v8 Decode Kernel — Correctness
# ══════════════════════════════════════════════════════════════════════════

section("2 - v8 Decode Kernel — Correctness")

if not TRITON_OK or not torch.cuda.is_available():
    warn("Skipped (Triton or CUDA not available)")
else:
    from turboquant.quantizer import TurboQuantProd
    from turboquant.kv_cache import quantize_values, unpack_values, ValueQuantized, dequantize_values
    from versions.v8_vllm_native.triton_kernels import turboquant_v8_decode

    BH       = 8
    N_HIST   = 1024
    N_BUF    = 64
    SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

    torch.manual_seed(42)

    # Build quantizer and compress keys
    prod_quantizer = TurboQuantProd(dim=HEAD_DIM, bits=3, device=DEVICE, dtype=torch.float32)
    keys_raw   = torch.randn(BH, N_HIST, HEAD_DIM, device=DEVICE)
    values_raw = torch.randn(BH, N_HIST, HEAD_DIM, device=DEVICE)
    query_raw  = torch.randn(BH, HEAD_DIM, device=DEVICE)

    # Buffer (exact recent tokens)
    k_buf = torch.randn(BH, N_BUF, HEAD_DIM, device=DEVICE)
    v_buf = torch.randn(BH, N_BUF, HEAD_DIM, device=DEVICE)

    q_keys = prod_quantizer.quantize(keys_raw.float())
    val_q  = quantize_values(values_raw.float(), bits=2, group_size=32)
    val_unpacked = unpack_values(val_q)
    val_for_kernel = ValueQuantized(
        data=val_unpacked, scales=val_q.scales, zeros=val_q.zeros, bits=2
    )

    # Rotation matrices
    Pi_T = prod_quantizer.mse_quantizer.Pi.T.contiguous().float()
    S_T  = prod_quantizer.S.T.contiguous().float()

    # ── PyTorch reference ─────────────────────────────────────────────
    # Use attention_score() which computes MSE + QJL (same as v8 kernel)
    v_dequant_ref = dequantize_values(val_q, 32).float()
    q_f = query_raw.float().unsqueeze(1)  # (BH, 1, D)

    # Compressed scores via TQ estimator (MSE + QJL)
    scores_hist = prod_quantizer.attention_score(
        q_f, q_keys
    ).squeeze(1) * SM_SCALE  # (BH, N_HIST)

    # Buffer scores via plain dot product (unrotated query vs exact keys)
    scores_buf  = torch.bmm(q_f, k_buf.float().transpose(-2, -1)).squeeze(1) * SM_SCALE
    scores_all  = torch.cat([scores_hist, scores_buf], dim=-1)
    weights_all = F.softmax(scores_all, dim=-1)
    v_all       = torch.cat([v_dequant_ref, v_buf.float()], dim=1)
    out_ref     = torch.bmm(weights_all.unsqueeze(1), v_all).squeeze(1)

    # ── v8 Triton kernel ──────────────────────────────────────────────
    out_v8 = turboquant_v8_decode(
        query=query_raw,
        Pi_T=Pi_T, S_T=S_T,
        quantized_key=q_keys,
        value_quantized=val_for_kernel,
        key_buffer=k_buf.float(),
        value_buffer=v_buf.float(),
        centroids=prod_quantizer.mse_quantizer.centroids,
        mse_bits=prod_quantizer.mse_quantizer.bits + 1,
        qjl_scale=prod_quantizer.qjl_scale,
        sm_scale=SM_SCALE,
        group_size=32,
    )

    max_diff  = (out_v8.float() - out_ref).abs().max().item()
    mean_diff = (out_v8.float() - out_ref).abs().mean().item()
    info(f"v8 decode vs PyTorch reference (BH={BH}, N_hist={N_HIST}, N_buf={N_BUF}):")
    info(f"  Max  |diff| = {max_diff:.5f}")
    info(f"  Mean |diff| = {mean_diff:.5f}")
    assert max_diff < 0.15, f"v8 decode disagrees too much: {max_diff:.4f}"
    ok("v8 decode kernel output matches PyTorch reference")


# ══════════════════════════════════════════════════════════════════════════
# 3. v8 vs v6 Decode Kernel — Throughput
# ══════════════════════════════════════════════════════════════════════════

section("3 - v8 vs v6 Decode Kernel — Throughput")

if not TRITON_OK or not torch.cuda.is_available():
    warn("Skipped (Triton or CUDA not available)")
else:
    from turboquant.triton_kernels import turboquant_fused_decode
    from versions.v8_vllm_native.triton_kernels import turboquant_v8_decode

    WARMUP, REPS = 20, 100
    kernel_results = []

    info(f"  {'N_hist':>8}  {'v6 (ms)':>10}  {'v8 (ms)':>10}  {'speedup':>8}  {'saved':>8}")
    info(f"  {'-' * 52}")

    for n_hist in [256, 1024, 4096, 16384]:
        torch.manual_seed(42)
        prod_q = TurboQuantProd(dim=HEAD_DIM, bits=3, device=DEVICE, dtype=torch.float32)
        k = torch.randn(BH, n_hist, HEAD_DIM, device=DEVICE)
        v = torch.randn(BH, n_hist, HEAD_DIM, device=DEVICE)
        q = torch.randn(BH, HEAD_DIM, device=DEVICE)
        k_b = torch.randn(BH, N_BUF, HEAD_DIM, device=DEVICE)
        v_b = torch.randn(BH, N_BUF, HEAD_DIM, device=DEVICE)

        qk = prod_q.quantize(k.float())
        vq = quantize_values(v.float(), bits=2, group_size=32)
        vq2 = ValueQuantized(data=unpack_values(vq), scales=vq.scales, zeros=vq.zeros, bits=2)

        Pi_T_l = prod_q.mse_quantizer.Pi.T.contiguous().float()
        S_T_l  = prod_q.S.T.contiguous().float()
        Pi_S_T = torch.cat([Pi_T_l, S_T_l], dim=-1)  # (D, 2D) for v6

        # ── v6: matmul + Triton kernel (2 launches) ──
        def run_v6():
            q_combined = torch.matmul(q.float(), Pi_S_T)
            q_rot = q_combined[:, :HEAD_DIM]
            q_sketch = q_combined[:, HEAD_DIM:]
            return turboquant_fused_decode(
                query=q, quantized_key=qk, value_quantized=vq2,
                Pi=prod_q.mse_quantizer.Pi, S=prod_q.S,
                centroids=prod_q.mse_quantizer.centroids,
                mse_bits=2, qjl_scale=prod_q.qjl_scale,
                sm_scale=SM_SCALE, group_size=32,
            )

        # ── v8: single Triton kernel (1 launch) ──
        def run_v8():
            return turboquant_v8_decode(
                query=q, Pi_T=Pi_T_l, S_T=S_T_l,
                quantized_key=qk, value_quantized=vq2,
                key_buffer=k_b.float(), value_buffer=v_b.float(),
                centroids=prod_q.mse_quantizer.centroids,
                mse_bits=prod_q.mse_quantizer.bits + 1,
                qjl_scale=prod_q.qjl_scale,
                sm_scale=SM_SCALE, group_size=32,
            )

        v6_ms = _bench(run_v6, WARMUP, REPS)
        v8_ms = _bench(run_v8, WARMUP, REPS)
        speedup = v6_ms / v8_ms if v8_ms > 0 else float('nan')
        saved_us = (v6_ms - v8_ms) * 1000

        kernel_results.append({
            "n_hist": n_hist, "v6_ms": v6_ms, "v8_ms": v8_ms,
            "speedup": speedup, "saved_us": saved_us,
        })
        info(f"  {n_hist:>8,}  {v6_ms:>8.3f}ms  {v8_ms:>8.3f}ms  "
             f"{speedup:>6.2f}x  {saved_us:>+6.0f}us")

        del k, v, q, k_b, v_b, qk, vq, vq2
        torch.cuda.empty_cache()

    ok("v8 vs v6 throughput comparison complete")


# ══════════════════════════════════════════════════════════════════════════
# 4. v8 Prefill Quantization — Correctness + Throughput
# ══════════════════════════════════════════════════════════════════════════

section("4 - v8 Prefill Quantization Kernels")

if not TRITON_OK or not torch.cuda.is_available():
    warn("Skipped (Triton or CUDA not available)")
else:
    from versions.v8_vllm_native.triton_kernels import turboquant_v8_prefill_quant

    BH_PQ   = 8
    N_PQ    = 512
    BITS_PQ = 3

    torch.manual_seed(7)
    prod_q_pq = TurboQuantProd(dim=HEAD_DIM, bits=BITS_PQ, device=DEVICE, dtype=torch.float32)
    keys_pq   = torch.randn(BH_PQ, N_PQ, HEAD_DIM, device=DEVICE)

    Pi_T_pq = prod_q_pq.mse_quantizer.Pi.T.contiguous().float()
    Pi_pq   = prod_q_pq.mse_quantizer.Pi.contiguous().float()
    S_T_pq  = prod_q_pq.S.T.contiguous().float()

    # ── Reference: PyTorch quantization ───────────────────────────────
    ref_quant = prod_q_pq.quantize(keys_pq.float())

    # ── v8 Triton prefill quant ───────────────────────────────────────
    mse_packed, signs_packed, norms, res_norms = turboquant_v8_prefill_quant(
        keys=keys_pq,
        Pi_T=Pi_T_pq, Pi=Pi_pq, S_T=S_T_pq,
        centroids=prod_q_pq.mse_quantizer.centroids,
        mse_bits=BITS_PQ,
    )

    info(f"Prefill quant (BH={BH_PQ}, N={N_PQ}, D={HEAD_DIM}):")
    info(f"  MSE packed shape : {mse_packed.shape} (ref: {ref_quant.mse_indices.shape})")
    info(f"  Signs shape      : {signs_packed.shape} (ref: {ref_quant.qjl_signs.shape})")
    info(f"  Norms shape      : {norms.shape}")

    # Compare norms (should match closely)
    norm_diff = (norms - ref_quant.norms).abs().max().item()
    info(f"  Norm max |diff|  : {norm_diff:.6f}")

    # Compare MSE packed bytes
    mse_match = (mse_packed == ref_quant.mse_indices).float().mean().item()
    info(f"  MSE pack match   : {mse_match*100:.1f}%")

    # Compare sign packed bytes
    sign_match = (signs_packed == ref_quant.qjl_signs).float().mean().item()
    info(f"  Sign pack match  : {sign_match*100:.1f}%")

    if mse_match > 0.95 and sign_match > 0.90:
        ok("v8 prefill quant matches PyTorch reference")
    else:
        warn(f"Partial match — MSE: {mse_match*100:.1f}%, Signs: {sign_match*100:.1f}%")
        info("  (Small differences expected from floating-point order-of-operations)")

    # ── Throughput: v8 Triton vs PyTorch ──────────────────────────────
    def run_pytorch_quant():
        return prod_q_pq.quantize(keys_pq.float())

    def run_v8_quant():
        return turboquant_v8_prefill_quant(
            keys=keys_pq, Pi_T=Pi_T_pq, Pi=Pi_pq, S_T=S_T_pq,
            centroids=prod_q_pq.mse_quantizer.centroids, mse_bits=BITS_PQ,
        )

    pytorch_ms = _bench(run_pytorch_quant, 10, 50)
    v8_ms      = _bench(run_v8_quant, 10, 50)
    speedup_pq = pytorch_ms / v8_ms if v8_ms > 0 else float('nan')

    info(f"  PyTorch quant    : {pytorch_ms:.3f} ms")
    info(f"  v8 Triton quant  : {v8_ms:.3f} ms")
    info(f"  Speedup          : {speedup_pq:.2f}x")
    ok("Prefill quantization benchmark complete")


# ══════════════════════════════════════════════════════════════════════════
# 5. End-to-End Attention — v8 vs v6 vs SDPA fp16
# ══════════════════════════════════════════════════════════════════════════

section("5 - End-to-End Attention — v8 vs v6 vs SDPA fp16")

if not TRITON_OK or not torch.cuda.is_available():
    warn("Skipped")
else:
    from turboquant.kv_cache import TurboQuantKVCache

    N_HEADS_E2E = 8
    WARMUP_E2E  = 5
    REPS_E2E    = 20

    info(f"  Config: n_heads={N_HEADS_E2E}, head_dim={HEAD_DIM}")
    info(f"  TurboQuant: key_bits={KEY_BITS}, value_bits={VAL_BITS}, buffer={RING_CAP}")
    info(f"  {'ctx':>7}  {'SDPA fp16':>10}  {'TQ v6':>10}  {'TQ v8':>10}  "
         f"{'v8 vs SDPA':>10}  {'v8 vs v6':>10}  {'mem ratio':>9}")
    info(f"  {'-' * 75}")

    e2e_rows = []
    for ctx in [1_024, 4_096, 16_384, 65_536, 131_072]:
        torch.manual_seed(7)
        keys_ctx   = torch.randn(1, N_HEADS_E2E, ctx, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        values_ctx = torch.randn(1, N_HEADS_E2E, ctx, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        query_1tok = torch.randn(1, N_HEADS_E2E, 1, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        scale_val  = 1.0 / math.sqrt(HEAD_DIM)

        # ── A. FP16 baseline: SDPA ───────────────────────────────────
        fp16_ms = _bench(
            lambda: F.scaled_dot_product_attention(query_1tok, keys_ctx, values_ctx),
            WARMUP_E2E, REPS_E2E,
        )

        # ── B. TurboQuant v6 fused hybrid ────────────────────────────
        cache = TurboQuantKVCache(
            head_dim=HEAD_DIM, key_bits=KEY_BITS, value_bits=VAL_BITS,
            buffer_size=RING_CAP, device=DEVICE, dtype=DTYPE,
        )
        cache.prefill(keys_ctx, values_ctx)

        try:
            tq_v6_ms = _bench(
                lambda: cache._fused_hybrid(query_1tok, scale_val, True, True),
                WARMUP_E2E, REPS_E2E,
            )
        except (ImportError, RuntimeError):
            tq_v6_ms = float('nan')

        # ── C. TurboQuant v8 zero-launch ─────────────────────────────
        # Use the same compressed data from the cache, but route through v8 kernel
        try:
            kq = cache.key_quantizer
            prod_q_e2e = cache.key_quantized
            val_q_e2e  = cache.value_quantized
            Pi_T_e2e = kq.mse_quantizer.Pi.T.contiguous().float()
            S_T_e2e  = kq.S.T.contiguous().float()

            # Flatten for BH-indexed kernel: (1, H, 1, D) -> (H, D)
            BH_e2e = N_HEADS_E2E
            q_bh = query_1tok.squeeze(0).squeeze(-2).float()  # (H, D)

            # Buffer: (1, H, N_buf, D) -> (H, N_buf, D)
            if cache.key_buffer is not None and cache.key_buffer.shape[-2] > 0:
                k_buf_e2e = cache.key_buffer.squeeze(0).float()  # (H, N_buf, D)
                v_buf_e2e = cache.value_buffer.squeeze(0).float()
            else:
                k_buf_e2e = torch.empty(BH_e2e, 0, HEAD_DIM, device=DEVICE, dtype=torch.float32)
                v_buf_e2e = torch.empty(BH_e2e, 0, HEAD_DIM, device=DEVICE, dtype=torch.float32)

            # Pre-unpack values
            from turboquant.kv_cache import unpack_values
            val_unpacked_e2e = unpack_values(val_q_e2e)
            val_for_v8 = ValueQuantized(
                data=val_unpacked_e2e, scales=val_q_e2e.scales,
                zeros=val_q_e2e.zeros, bits=2,
            )

            def run_v8_e2e():
                return turboquant_v8_decode(
                    query=q_bh, Pi_T=Pi_T_e2e, S_T=S_T_e2e,
                    quantized_key=prod_q_e2e, value_quantized=val_for_v8,
                    key_buffer=k_buf_e2e, value_buffer=v_buf_e2e,
                    centroids=kq.mse_quantizer.centroids,
                    mse_bits=kq.mse_quantizer.bits + 1,
                    qjl_scale=kq.qjl_scale,
                    sm_scale=scale_val, group_size=32,
                )

            tq_v8_ms = _bench(run_v8_e2e, WARMUP_E2E, REPS_E2E)
        except Exception as e:
            info(f"    v8 failed at ctx={ctx}: {e}")
            import traceback; traceback.print_exc()
            tq_v8_ms = float('nan')

        # Memory comparison
        def kv_fp16_bytes(seq_len, n_kv_heads, head_dim):
            return 2 * n_kv_heads * seq_len * head_dim * 2
        fp16_mb = kv_fp16_bytes(ctx, N_KV_HEADS, HEAD_DIM) / 1024**2
        tq_mb   = cache.memory_bytes()["total"] / 1024**2
        ratio   = fp16_mb / tq_mb if tq_mb > 0 else 0.0

        overhead_v8 = tq_v8_ms / fp16_ms if fp16_ms > 0 else float('nan')
        v8_vs_v6    = tq_v6_ms / tq_v8_ms if tq_v8_ms > 0 else float('nan')

        e2e_rows.append(dict(
            ctx=ctx, fp16_ms=fp16_ms, tq_v6_ms=tq_v6_ms, tq_v8_ms=tq_v8_ms,
            overhead_v8=overhead_v8, v8_vs_v6=v8_vs_v6,
            fp16_mb=fp16_mb, tq_mb=tq_mb, ratio=ratio,
        ))

        info(f"  {ctx:>7,}  {fp16_ms:>8.3f}ms  {tq_v6_ms:>8.3f}ms  {tq_v8_ms:>8.3f}ms  "
             f"{overhead_v8:>8.1f}x  {v8_vs_v6:>8.2f}x  {ratio:>7.1f}x")

        del keys_ctx, values_ctx, cache
        torch.cuda.empty_cache()

    ok("End-to-end benchmark complete")


# ══════════════════════════════════════════════════════════════════════════
# 6. vLLM Integration — Baseline vs TurboQuant v8
# ══════════════════════════════════════════════════════════════════════════

if not SKIP_VLLM:
    section("6 - vLLM Integration — Baseline vs TurboQuant v8")

    # Force full GPU memory cleanup before vLLM profiling — vLLM's memory
    # profiler asserts free_memory_now <= free_memory_at_init, so we must
    # release all memory from sections 2-5 before starting.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        warn("vLLM not installed — skipping integration test")
        SKIP_VLLM = True

if not SKIP_VLLM:
    PROMPTS = [
        "What is KV cache compression? Explain in two sentences.",
        "Write a Python function to find the longest common subsequence of two strings. Include comments explaining each step.",
        ("You are a helpful assistant. " * 50) + "\nGiven the above context, what was the first instruction you received?",
    ]

    def get_gpu_memory_mb():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0

    info(f"Model     : {MODEL}")
    info(f"TQ config : {KEY_BITS}-bit keys, {VAL_BITS}-bit values, ring={RING_CAP}")

    # ── Phase 1: Baseline vLLM ────────────────────────────────────────
    info("")
    info("Phase 1 — Baseline vLLM (fp16 KV cache)")
    info("Loading model...")

    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    effective_util = min(GPU_MEM, (free_gb - 4.0) / total_gb)
    effective_util = max(0.15, min(effective_util, 0.50))
    info(f"Free VRAM: {free_gb:.1f} GB / {total_gb:.1f} GB -> gpu_mem_util={effective_util:.2f}")

    llm = LLM(
        model=MODEL, dtype="auto",
        gpu_memory_utilization=effective_util,
        max_model_len=MAX_LEN, tensor_parallel_size=1,
        trust_remote_code=True, max_num_seqs=1,
        enforce_eager=True,
    )

    vram_after_load = get_gpu_memory_mb()
    info(f"VRAM after load: {vram_after_load:.0f} MB")

    params = SamplingParams(temperature=0.0, max_tokens=128)

    info("Generating (baseline)...")
    t0 = time.perf_counter()
    baseline_outputs = llm.generate(PROMPTS, params)
    baseline_time = time.perf_counter() - t0

    vram_after_gen = get_gpu_memory_mb()
    info(f"VRAM after gen:  {vram_after_gen:.0f} MB")
    info(f"Total time:      {baseline_time:.2f}s")

    for i, out in enumerate(baseline_outputs):
        text = out.outputs[0].text.strip()[:120]
        toks = len(out.outputs[0].token_ids)
        info(f"Prompt {i+1}: ({toks} tokens) {text}...")

    # ── Phase 2: TurboQuant v8 ────────────────────────────────────────
    info("")
    info("Phase 2 — TurboQuant v8 (composition-based, 1-launch decode)")

    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    info("Installing TurboQuant v8 hooks...")
    from versions.v8_vllm_native.backend import (
        install_v8, free_kv_cache_v8, get_stats_v8, MODE_HYBRID,
    )

    def _install_v8(worker):
        return len(install_v8(
            worker.model_runner,
            key_bits=KEY_BITS, value_bits=VAL_BITS,
            ring_capacity=RING_CAP, mode=MODE_HYBRID,
        ))

    try:
        num_layers = executor.collective_rpc(_install_v8)
        n_hooked = num_layers[0] if isinstance(num_layers, list) else num_layers
    except Exception:
        try:
            n_hooked = _install_v8(executor)
        except Exception:
            runner = executor.model_runner if hasattr(executor, 'model_runner') else executor
            n_hooked = len(install_v8(
                runner, key_bits=KEY_BITS, value_bits=VAL_BITS,
                ring_capacity=RING_CAP, mode=MODE_HYBRID,
            ))

    info(f"Hooked {n_hooked} attention layers (v8 composition wrapper)")

    info("Generating (TurboQuant v8)...")
    t0 = time.perf_counter()
    tq_outputs = llm.generate(PROMPTS, params)
    tq_time = time.perf_counter() - t0

    vram_after_tq = get_gpu_memory_mb()
    info(f"VRAM after gen:  {vram_after_tq:.0f} MB")
    info(f"Total time:      {tq_time:.2f}s")

    for i, out in enumerate(tq_outputs):
        text = out.outputs[0].text.strip()[:120]
        toks = len(out.outputs[0].token_ids)
        info(f"Prompt {i+1}: ({toks} tokens) {text}...")

    # ── Get stats ─────────────────────────────────────────────────────
    try:
        def _stats(worker):
            return get_stats_v8(worker.model_runner)
        stats = executor.collective_rpc(_stats)
        tq_stats = stats[0] if isinstance(stats, list) else stats
    except Exception:
        tq_stats = None

    # ── Free KV cache ─────────────────────────────────────────────────
    info("Freeing paged KV cache...")
    try:
        def _free(worker):
            return free_kv_cache_v8(worker.model_runner)
        freed_list = executor.collective_rpc(_free)
        freed = freed_list[0] if isinstance(freed_list, list) else freed_list
    except Exception:
        freed = 0

    vram_after_free = get_gpu_memory_mb()
    info(f"VRAM after free: {vram_after_free:.0f} MB")
    info(f"KV cache freed:  {freed / 1e6:.0f} MB")

    # ── Summary ───────────────────────────────────────────────────────
    info("")
    info(f"  {'':>20}  {'Baseline':>10}  {'TQ v8':>10}")
    info(f"  {'-' * 44}")
    info(f"  {'Gen time':>20}  {baseline_time:>8.2f}s  {tq_time:>8.2f}s")
    info(f"  {'VRAM after gen':>20}  {vram_after_gen:>7.0f}MB  {vram_after_tq:>7.0f}MB")
    info(f"  {'VRAM after free':>20}  {'n/a':>10}  {vram_after_free:>7.0f}MB")
    info(f"  {'KV cache freed':>20}  {'n/a':>10}  {freed/1e6:>7.0f}MB")

    if tq_stats:
        info("")
        info("TurboQuant v8 stats:")
        for k, v in tq_stats.items():
            info(f"  {k}: {v}")

    info("")
    info("Output comparison (prompt 1):")
    bl_text = baseline_outputs[0].outputs[0].text.strip()[:200]
    tq_text = tq_outputs[0].outputs[0].text.strip()[:200]
    info(f"  Baseline: {bl_text}")
    info(f"  TQ v8:    {tq_text}")

    ok("vLLM integration test complete")


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════

section("Summary")

summary = {
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    "triton_available": TRITON_OK,
    "version": "v8",
}

if TRITON_OK and torch.cuda.is_available():
    summary["kernel_throughput"] = kernel_results if 'kernel_results' in dir() else []
    summary["e2e_benchmark"] = [
        {
            "ctx": r["ctx"],
            "fp16_ms": round(r["fp16_ms"], 3),
            "tq_v6_ms": round(r["tq_v6_ms"], 3),
            "tq_v8_ms": round(r["tq_v8_ms"], 3),
            "v8_overhead_vs_sdpa": round(r["overhead_v8"], 1),
            "v8_speedup_vs_v6": round(r["v8_vs_v6"], 2),
            "mem_ratio": round(r["ratio"], 2),
        }
        for r in e2e_rows
    ] if 'e2e_rows' in dir() else []

print()
print(json.dumps(summary, indent=2))

print(f"""
  TurboQuant v8 — Zero External Kernel Launch

  Key innovations:
    1. Fused Q rotation in Triton decode kernel (2 launches -> 1)
    2. Composition-based vLLM integration (no monkey-patching)
    3. Fused prefill quantization (~10 PyTorch ops -> 3 Triton launches)

  Expected: ~10-15 us saved per decode step (kernel launch overhead)
""")
