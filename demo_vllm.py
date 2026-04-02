#!/usr/bin/env python3
"""
TurboQuant + vLLM — Real LLM inference on Jetson AGX Thor
==========================================================

Runs an actual language model with TurboQuant KV-cache compression.
Shows side-by-side: baseline vLLM vs TurboQuant (memory, speed, output).

Usage:
    # Inside the Docker container:
    python3 /workspace/demo_vllm.py

    # With a specific model:
    MODEL=Qwen/Qwen2.5-7B-Instruct python3 /workspace/demo_vllm.py

    # Via run_jetson.sh:
    ./run_jetson.sh vllm
"""
import os, sys, time, json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# ── Config ──────────────────────────────────────────────────────────────
MODEL       = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU_MEM     = float(os.environ.get("GPU_MEM", "0.70"))
MAX_LEN     = int(os.environ.get("MAX_MODEL_LEN", "4096"))
KEY_BITS    = int(os.environ.get("KEY_BITS", "3"))
VAL_BITS    = int(os.environ.get("VAL_BITS", "2"))
RING_CAP    = int(os.environ.get("RING_CAPACITY", "128"))

# Prompts that exercise different context lengths
PROMPTS = [
    # Short — should generate quickly
    "What is KV cache compression? Explain in two sentences.",
    # Medium — reasoning
    "Write a Python function to find the longest common subsequence of two strings. Include comments explaining each step.",
    # Longer prompt — tests attending over more context
    ("You are a helpful assistant. " * 50) + "\nGiven the above context, what was the first instruction you received?",
]


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    import torch
    from vllm import LLM, SamplingParams

    section("TurboQuant + vLLM — Real LLM Inference")

    print(f"  Model     : {MODEL}")
    print(f"  GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  TQ config : {KEY_BITS}-bit keys, {VAL_BITS}-bit values, ring={RING_CAP}")
    print(f"  Max len   : {MAX_LEN:,}")

    # ── Phase 1: Baseline vLLM ──────────────────────────────────────────
    section("Phase 1 — Baseline vLLM (fp16 KV cache)")

    print("  Loading model...", flush=True)
    # Detect available memory and adjust automatically
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    effective_util = min(GPU_MEM, (free_gb - 4.0) / total_gb)  # leave 4GB headroom
    effective_util = max(0.15, min(effective_util, 0.50))
    print(f"  Free VRAM: {free_gb:.1f} GB / {total_gb:.1f} GB "
          f"→ gpu_mem_util={effective_util:.2f}", flush=True)

    llm = LLM(
        model=MODEL,
        dtype="auto",
        gpu_memory_utilization=effective_util,
        max_model_len=MAX_LEN,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_num_seqs=1,
        enforce_eager=True,  # skip torch.compile to save memory + startup time
    )

    vram_after_load = get_gpu_memory_mb()
    print(f"  VRAM after load: {vram_after_load:.0f} MB")

    params = SamplingParams(temperature=0.0, max_tokens=128)

    print("  Generating (baseline)...", flush=True)
    t0 = time.perf_counter()
    baseline_outputs = llm.generate(PROMPTS, params)
    baseline_time = time.perf_counter() - t0

    vram_after_gen = get_gpu_memory_mb()
    print(f"  VRAM after gen:  {vram_after_gen:.0f} MB")
    print(f"  Total time:      {baseline_time:.2f}s")

    for i, out in enumerate(baseline_outputs):
        text = out.outputs[0].text.strip()[:120]
        toks = len(out.outputs[0].token_ids)
        print(f"  Prompt {i+1}: ({toks} tokens) {text}...")

    # ── Phase 2: TurboQuant ─────────────────────────────────────────────
    section("Phase 2 — TurboQuant (compressed KV cache)")

    # Access the model runner to install hooks
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    print("  Installing TurboQuant hooks...", flush=True)
    from turboquant.vllm_attn_backend import (
        install_turboquant_hooks, free_kv_cache, MODE_ACTIVE,
    )
    from turboquant.integration.vllm import get_stats

    # Install on all workers (handles TP)
    def _install(worker):
        return len(install_turboquant_hooks(
            worker.model_runner,
            key_bits=KEY_BITS,
            value_bits=VAL_BITS,
            buffer_size=RING_CAP,
            mode=MODE_ACTIVE,
        ))

    try:
        num_layers = executor.collective_rpc(_install)
        n_hooked = num_layers[0] if isinstance(num_layers, list) else num_layers
    except Exception as e:
        # Fallback: try direct access (single-GPU)
        try:
            n_hooked = _install(executor)
        except Exception:
            n_hooked = install_turboquant_hooks(
                executor.model_runner if hasattr(executor, 'model_runner') else executor,
                key_bits=KEY_BITS,
                value_bits=VAL_BITS,
                buffer_size=RING_CAP,
                mode=MODE_ACTIVE,
            )
            n_hooked = len(n_hooked) if isinstance(n_hooked, list) else n_hooked

    print(f"  Hooked {n_hooked} attention layers")

    print("  Generating (TurboQuant)...", flush=True)
    t0 = time.perf_counter()
    tq_outputs = llm.generate(PROMPTS, params)
    tq_time = time.perf_counter() - t0

    vram_after_tq = get_gpu_memory_mb()
    print(f"  VRAM after gen:  {vram_after_tq:.0f} MB")
    print(f"  Total time:      {tq_time:.2f}s")

    for i, out in enumerate(tq_outputs):
        text = out.outputs[0].text.strip()[:120]
        toks = len(out.outputs[0].token_ids)
        print(f"  Prompt {i+1}: ({toks} tokens) {text}...")

    # ── Get TQ stats ────────────────────────────────────────────────────
    try:
        def _stats(worker):
            return get_stats(worker.model_runner)
        stats = executor.collective_rpc(_stats)
        tq_stats = stats[0] if isinstance(stats, list) else stats
    except Exception:
        tq_stats = None

    # ── Free KV cache ───────────────────────────────────────────────────
    print("  Freeing paged KV cache...", flush=True)
    try:
        def _free(worker):
            return free_kv_cache(worker.model_runner)
        freed_list = executor.collective_rpc(_free)
        freed = freed_list[0] if isinstance(freed_list, list) else freed_list
    except Exception:
        freed = 0

    vram_after_free = get_gpu_memory_mb()
    print(f"  VRAM after free: {vram_after_free:.0f} MB")
    print(f"  KV cache freed:  {freed / 1e6:.0f} MB")

    # ── Summary ─────────────────────────────────────────────────────────
    section("Summary — Baseline vs TurboQuant")

    print(f"  Model: {MODEL}")
    print(f"  TurboQuant: {KEY_BITS}-bit keys, {VAL_BITS}-bit values")
    print(f"  Layers hooked: {n_hooked}")
    print()
    print(f"  {'':>20}  {'Baseline':>10}  {'TurboQuant':>10}")
    print(f"  {'-' * 44}")
    print(f"  {'Gen time':>20}  {baseline_time:>8.2f}s  {tq_time:>8.2f}s")
    print(f"  {'VRAM after gen':>20}  {vram_after_gen:>7.0f}MB  {vram_after_tq:>7.0f}MB")
    print(f"  {'VRAM after free':>20}  {'n/a':>10}  {vram_after_free:>7.0f}MB")
    print(f"  {'KV cache freed':>20}  {'n/a':>10}  {freed/1e6:>7.0f}MB")

    if tq_stats:
        print()
        print(f"  TurboQuant stats:")
        for k, v in tq_stats.items():
            print(f"    {k}: {v}")

    print()
    print(f"  Output comparison (prompt 1):")
    bl_text = baseline_outputs[0].outputs[0].text.strip()[:200]
    tq_text = tq_outputs[0].outputs[0].text.strip()[:200]
    print(f"    Baseline:   {bl_text}")
    print(f"    TurboQuant: {tq_text}")

    # JSON output for programmatic use
    result = {
        "model": MODEL,
        "key_bits": KEY_BITS,
        "value_bits": VAL_BITS,
        "layers_hooked": n_hooked,
        "baseline_time_s": round(baseline_time, 2),
        "tq_time_s": round(tq_time, 2),
        "vram_after_gen_mb": round(vram_after_gen),
        "vram_after_tq_mb": round(vram_after_tq),
        "kv_freed_mb": round(freed / 1e6),
    }
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
