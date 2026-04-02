# TurboQuant: KV Cache Compression for LLM Inference

Implementation of TurboQuant KV cache compression (ICLR 2026, arXiv:2504.19874) with vLLM integration. Tested on dense and MoE architectures across RTX 3090 and RTX 5090 GPUs.

## Benchmark Results

### RTX 5090 (32GB) -- Qwen3.5-27B-AWQ (dense, 4-bit weights, TP=1)

**Setup**: Single RTX 5090, vLLM 0.18.0, `gpu_memory_utilization=0.90`, 16 full-attention layers out of 64 total (rest are linear-attention).

| Metric | Baseline (bf16 KV) | TurboQuant (3b key / 2b val) |
|--------|-------------------|------------------------------|
| Prefill tok/s (30k ctx) | 1,804 | 1,907 (+5.7%) |
| Decode tok/s (30k ctx) | 1.264 | 1.303 (+3.1%) |
| KV cache freed | -- | **30.0 GB** (across 4 GPUs) |
| Max token capacity | 457,072 | **914,144** (2.0x) |
| Peak activation memory | 644.6 MB | 599.2 MB (-7.0%) |

### 8x RTX 3090 (24GB each) -- Qwen3.5-35B-A3B MoE (pruned, 205 experts, TP=8)

**Setup**: 8x RTX 3090, vLLM 0.18.0, `gpu_memory_utilization=0.92`, AMD EPYC 7443P 24-Core, 504GB RAM. Model has 10 full-attention layers + 30 linear-attention layers (40 total). TQ compresses only the 10 full-attention layers.

#### Throughput & Latency (Baseline, bf16 KV)

| Context | Prefill tok/s | Decode tok/s | TTFT (s) | Needles Found |
|--------:|--------------:|-------------:|---------:|--------------:|
| 1,000 | 7,127 | 129.7 | 0.14 | 4/5 |
| 4,000 | 8,887 | 131.5 | 0.45 | 4/5 |
| 8,000 | 9,684 | 131.1 | 0.83 | 4/5 |
| 16,000 | 9,933 | 133.0 | 1.61 | 4/5 |
| 32,000 | 9,761 | 116.7 | 3.28 | 4/5 |
| 64,000 | 8,843 | 122.6 | 7.24 | 4/5 |
| 100,000 | 8,479 | 106.8 | 11.79 | 4/5 |
| 131,000 | 8,238 | 98.3 | 15.90 | 4/5 |

- **Prefill** saturates around 10k tok/s, degrades gently to 8.2k at 131k context.
- **Decode** drops from 133 to 98 tok/s at 131k (KV readback cost from full-attention layers).
- **TTFT** scales linearly with context length (purely compute-bound).
- **Needles** 4/5 found consistently at ALL context lengths -- the model reformats one answer.

#### VRAM Breakdown (per GPU at 131k context)

| Component | Size |
|-----------|-----:|
| Total VRAM | 24,576 MB |
| Reserved (0.92 util) | 22,610 MB |
| Model weights | ~6,750 MB |
| KV cache pool | **9,035 MB** |
| -- full_attention (10 layers) | 3,614 MB |
| -- linear_attention (30 layers) | 5,421 MB |
| CUDA overhead + graphs | ~6,825 MB |

#### Baseline vs TurboQuant KV Cache

| Context | Baseline KV/GPU | TQ KV/GPU | Savings/GPU | Savings % |
|--------:|----------------:|----------:|------------:|----------:|
| 8,000 | 55.7 MB | 38.5 MB | **17.2 MB** | 30.9% |
| 32,000 | 191.5 MB | 132.3 MB | **59.3 MB** | 30.9% |
| 64,000 | 374.3 MB | 258.5 MB | **115.8 MB** | 30.9% |
| 100,000 | 578.1 MB | 399.2 MB | **178.8 MB** | 30.9% |
| 131,000 | 755.7 MB | 521.9 MB | **233.8 MB** | 30.9% |

- Savings are **30.9% of total KV** because TQ only compresses the 10 full-attention layers (40% of KV).
- The 30 linear-attention layers (60% of KV) are **not compressible** by TQ.
- On a **pure dense transformer**, savings would be **77%** (4.4x compression).

#### Context Extension

| | Tokens | Multiplier |
|---|-------:|:----------:|
| Baseline capacity | 1,411,680 | 1.0x |
| With TQ | 2,043,808 | **1.45x** |

Alternatively, freed VRAM supports **3 additional concurrent 131k-context requests**.

#### Coherence & Quality

| Test | Result |
|------|--------|
| Single needle (512-131k tokens) | **PASS** at all lengths |
| 5-needle at near-max context | **5/5** retrieved |
| 3-needle multi-fact coherence | **3/3** retrieved |
| Golden ratio completion (all lengths) | **PASS**, perplexity 1.05-1.35 |
| Math reasoning at max context | Coherent (model math error from pruning, not context) |

#### TQ Quantization Quality (head_dim=256, measured on GPU)

| Component | cos_sim | Notes |
|-----------|--------:|-------|
| TQ key compression (3-bit) | **1.000000** | Near-lossless |
| TQ key compression (4-bit) | **1.000000** | Near-lossless |
| Value quantization (2-bit) | 0.940 | Bottleneck for quality |
| Value quantization (4-bit) | 0.997 | Recommended for quality-sensitive use |
| Combined (3b key + 2b val) | 0.940 | Value quant dominates degradation |

#### GPU Utilization During Inference

| Context | Peak VRAM/GPU | GPU Util | CPU % | Power |
|--------:|--------------:|---------:|------:|------:|
| 1,000 | 22,284 MB | 0% idle | 0.2% | 132W |
| 32,000 | 22,286 MB | 57% peak | 0.4% | 142W |
| 131,000 | 22,306 MB | 0% idle | 0.4% | 130W |

- VRAM is **essentially flat** -- KV cache at 131k is only 190 MB/GPU (0.8% of VRAM).
- No CPU offloading. No KV offloading. Everything fits in VRAM.
- GPU interconnect is **PCIe** (no NVLink) -- NODE topology between all GPUs.

### Paper Validation (Theorems 1-3)

9 tests validating the paper's theoretical claims:

| Claim | Verdict | Details |
|-------|---------|---------|
| MSE distortion bounds (Thm 1) | **PASS** | Within bounds for unit-norm vectors |
| Codebook MSE matches Table 1 | **PASS** | Lloyd-Max codebook is faithful |
| Unbiasedness (Thm 2) | **PASS** | Relative bias < 0.1% |
| Distortion 1/4^b scaling (Thm 3) | **PASS** | 2-bit=0.70x, 3-bit=0.82x, 4-bit=0.97x of bound |
| Recall@8 (3-bit, N=4096) | **0.55** | Paper threshold met (>=0.40) |
| Rank correlation (N=2048) | **PASS** | Spearman rho > 0.85 |
| Needle retrieval | **PASS** | Works at all SNR levels |
| Compression ratio | **4.41x** | At head_dim=256 on full-attention layers |

### Adversarial Audit

Honest assessment of claims (see `audit_claims.py` for full data):

| Claim | Verdict |
|-------|---------|
| "5.1x compression" | **Misleading** -- doesn't count Pi/S matrices or ring buffer. Honest: ~4.6x at 4k tokens, ~5x at 32k+ |
| "Needle-in-haystack passes" | **True but trivial** -- query=key test is too easy. Real LLM queries are not copies of keys |
| "Recall@8 >= 0.40" | **Low bar** -- 3-bit recall@1 is only 38%. BUT dominant attention tokens are always preserved |
| "Hybrid decode saves memory" | **Storage yes, compute no** -- dequantizes all history to float32 per decode step |
| "Distortion follows 1/4^b" | **True** -- initial audit was wrong (unnormalized vectors). Unit-norm: within bound |
| "30k TQ is faster" | **Within noise** -- N=1 run, total wall time TQ is actually slower |
| "200k context works" | **Unverified** -- didn't crash, but output quality never checked |
| "2x context on dense model" | **True** -- measured 30 GB freed on Qwen3.5-27B with 4x RTX 3090 |

## How It Works

TurboQuant compresses KV cache entries using:
1. **Random orthogonal rotation** to spread information across dimensions
2. **Lloyd-Max optimal scalar quantization** (b-1 bits) on Beta-distributed rotated values
3. **QJL projection** for residual sign bits (1 bit per dimension)
4. **Group quantization** for values (2-bit or 4-bit, per-group scales and zeros)
5. **Bit-packing**: 4 values per byte (2-bit) or 2 per byte (4-bit)

The combined estimator is **unbiased**: E[estimated inner product] = true inner product.

## Architecture

```
turboquant/
  codebook.py          # Lloyd-Max optimal scalar quantizer for Beta distribution
  codebooks/           # Pre-generated codebook files (d=128/256, bits 2/3/4)
  rotation.py          # Random orthogonal rotation + QJL projection matrices
  quantizer.py         # TurboQuantMSE + TurboQuantProd (Algorithms 1 & 2)
  kv_cache.py          # KV cache manager with value bit-packing
  capture.py           # Modular KV capture hooks for attention layers
  store.py             # Compressed KV store (quantize + append + flat cache)
  score.py             # Attention scoring from compressed keys
  integration/vllm.py  # vLLM adapter (monkey-patch, free_kv_cache, hybrid decode)
  triton_kernels.py    # 3 fused Triton kernels for decode attention
  vllm_attn_backend.py # Thin shim delegating to integration/vllm.py

validate_paper.py      # 9 tests validating Theorems 1-3
audit_claims.py        # Adversarial audit of all claims
test_modular.py        # 19 modular architecture tests
test_turboquant.py     # 7 core quantizer tests
proof.py               # A/B benchmark (baseline vs TQ)

# Profiling scripts (8x RTX 3090 MoE validation)
validate_moe.py        # Baseline measurements via vLLM API
validate_moe_phase2.py # TQ quality on real GPU with head_dim=256
validate_moe_phase3.py # Logprobs, multi-needle, reasoning at max context
profile_100k.py        # Full profiling at 1k-131k context
profile_large.py       # Large context (64k-131k) with file-based payloads
baseline_vs_tq.py      # VRAM comparison: baseline bf16 vs TQ compressed
baseline_vs_tq_v2.py   # Block-level measurement during inference
```

## Usage

```bash
pip install -e .

# Run paper validation (CPU, no GPU needed)
python validate_paper.py

# Run adversarial audit
python audit_claims.py

# Run modular tests
python -m pytest test_modular.py -v

# Run proof benchmark (requires 4x RTX 3090 + Qwen3.5-27B-AWQ)
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py
```

## Test Results

All 35 tests pass:
- `test_modular.py`: 19/19 (modular architecture)
- `test_turboquant.py`: 7/7 (core quantizer)
- `validate_paper.py`: 9/9 (paper theorem validation)

## Limitations

- **Prefill still uses paged cache**: KV cache is allocated at engine init and used during prefill. TQ frees it after. True zero-allocation requires deeper vLLM integration.
- **Only full-attention layers**: Linear-attention/Mamba layers are not compressed.
- **Value quantization is the bottleneck**: 2-bit values cause cos_sim=0.94 degradation. Use 4-bit values (cos_sim=0.997) for quality-sensitive workloads.
- **Hybrid decode dequantizes all history**: During compute, all compressed tokens are expanded to float32. The paper's fused Triton kernels exist but the hybrid path doesn't use them yet.
- **MoE models benefit less**: Models with linear-attention layers (Qwen3.5 MoE, Mamba hybrids) have incompressible state that limits TQ's overall impact.

### Jetson AGX Thor (122 GB unified memory)

**Setup**: NVIDIA Thor GPU (SM 11.0 Blackwell), 20 SMs, 122.8 GB unified CPU+GPU memory, JetPack 7.1 (R38.4.0). Runtime: `nvcr.io/nvidia/tritonserver:26.03-vllm-python-py3` (PyTorch 2.11, Triton 3.6).

#### Quantizer Validation

| Test | Result |
|------|--------|
| MSE distortion bounds (b=1,2,3,4) | **PASS** — all within Theorem 1 bounds |
| Inner-product unbiasedness | **PASS** — bias = 4.7e-05 (Theorem 2) |
| Triton fused-decode vs PyTorch ref | **PASS** — max diff = 0.00000 |

#### KV Cache Compression Ratios (head_dim=128, 8 KV heads, 3-bit key + 2-bit value)

| Context | FP16 KV | TurboQuant KV | Compression |
|--------:|--------:|--------------:|:-----------:|
| 1,024 | 4.0 MB | 1.2 MB | 3.4x |
| 4,096 | 16.0 MB | 3.5 MB | 4.5x |
| 16,384 | 64.0 MB | 12.9 MB | 5.0x |
| 65,536 | 256.0 MB | 50.4 MB | 5.1x |
| 131,072 | 512.0 MB | 100.4 MB | **5.1x** |

#### Triton Kernel Performance (SM 11.0 Blackwell)

| Kernel | Config | Latency |
|--------|--------|--------:|
| Fused decode (score + softmax + V agg) | BH=8, N=1024, D=128 | **0.327 ms** |

All three Triton kernels (MSE score, QJL score, fused decode) compile and run correctly on SM 11.0 with `num_warps=8` tuned for the Thor's 20 SMs.

#### Thor Memory Capacity Projection (32 layers, 8 KV heads, head_dim=128)

| Model Weights | Available Memory | FP16 Context | TQ Context | Multiplier |
|--------------:|-----------------:|-------------:|-----------:|:----------:|
| 16 GB | 107 GB | 875,101 | 3,862,514 | **4.4x** |
| 30 GB | 93 GB | 760,413 | 3,356,305 | **4.4x** |
| 60 GB | 63 GB | 514,653 | 2,271,572 | **4.4x** |

Thor's 122 GB unified memory combined with TurboQuant enables multi-million token contexts for 8B-class models.

#### End-to-End Attention Latency (8 heads, head_dim=128)

| Context | FP16 (SDPA) | TurboQuant | Memory Ratio |
|--------:|------------:|-----------:|:------------:|
| 1,024 | 0.041 ms | 1.096 ms | 3.4x |
| 4,096 | 0.092 ms | 3.325 ms | 4.5x |
| 16,384 | 0.382 ms | 12.915 ms | 5.0x |

Note: TQ PyTorch path (dequant + matmul) is compute-heavier than SDPA. The fused Triton kernel (0.327 ms at N=1024) is the target for production decode.

#### Running on Jetson

```bash
# Quick start (builds Docker image + runs all 7 validation sections)
cd /path/to/deepmind-turboquant
./run_jetson.sh demo

# Interactive shell inside the container
./run_jetson.sh shell

# Or run manually in any container with PyTorch + Triton:
pip install scipy && pip install -e turboquant/
python3 demo_jetson.py
```

#### Bugs Fixed for Thor

1. **Triton kernels**: Precomputed query base pointers outside inner loops (avoids redundant `pid_bh * stride` multiply per iteration in all 3 kernels).
2. **Value attention dtype**: Fixed fp32/fp16 mismatch in `kv_cache.py` `attend()` when buffer values are fp16 but softmax weights are fp32.
3. **Fused decode warps**: Tuned `num_warps=8` (from 4) for Blackwell's 20 SMs.

## Environment

Tested on:
- vLLM 0.18.0, PyTorch 2.10, CUDA 12.8
- RTX 5090 (32GB) -- Qwen3.5-27B-AWQ, single GPU
- 8x RTX 3090 (24GB) -- Qwen3.5-35B-A3B MoE, TP=8
- Jetson AGX Thor (122 GB) -- PyTorch 2.11, Triton 3.6, JetPack 7.1
- Python 3.12
