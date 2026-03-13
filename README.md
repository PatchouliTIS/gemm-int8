# Fused Blockwise INT8 GEMM Kernels

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

High-performance **fused blockwise-dequantization INT8 GEMM** kernels for W8A8 quantized inference, built on CUTLASS SM80 Tensor Core primitives. All dequantization happens **in-register** inside the mainloop — zero intermediate HBM traffic, zero extra kernel launches.

## Key Optimizations

The core innovation lives in [`csrc/kernels/gemm_blockwise_fused.cu`](csrc/kernels/gemm_blockwise_fused.cu). Multiple kernel variants are provided, each targeting a different trade-off between dequantization overhead and quantization granularity:

### 1. Fused Mainloop Dequantization (bq128 / bq256)

- **Tile shape**: `128×64×128` (M×N×K), 3-stage async pipeline
- CUTLASS `cp.async` multi-stage mainloop drives GMEM→SMEM→RF data flow
- After each K-tile's IMMA (`m16n8k32`), INT32 partial products are **immediately dequantized** to FP32 using per-block scales:
  ```
  fp32_accum[i] += float(int32_accum[i]) * scale_A[m_qb, kt] * scale_B[n_qb, kt]
  ```
- INT32 accumulator is cleared per K-tile, preventing overflow at large K
- bq256 variant accumulates 2 K-tiles before dequant, halving scale loads

### 2. Bias-Offset Fast Dequant (bq256 fast_dequant)

- Replaces the standard `I2F` (INT32→FP32 conversion) with a **bias-offset magic number trick**:
  ```c
  uint32_t u = (uint32_t)(val + kBias) | 0x4B000000u;  // kBias = 2^22
  float result = __int_as_float(u);  // == 2^23 + val + bias
  ```
- 3 instructions per element (IADD3 + LOP3 + FMUL), **0 XU (conversion unit) usage**
- Deferred bias correction: `fp32_accum[i] -= kBiasFloat * sum(scales)` applied once before epilogue

### 3. Hybrid IMUL + Magic Dequant (⭐ Recommended)

The **highest-performance** variant. Two-level scale quantization eliminates all FP32 work from the K-loop:

- **Host-side**: 2nd-level scale quantization splits each FP32 scale `S_k` into `Q_k` (INT32) × `F_g` (FP32 per super-group)
- **K-loop (pure INT32)**: `int32_weighted[i] += Q_combined * int32_accum[i]` — only IMAD instructions
- **Epilogue (single I2F)**: `fp32[i] = float(int32_weighted[i]) * (F_A * F_B)`
- Supports optional **fused bias addition** in the epilogue (row-broadcast, zero-overhead)
- **Threadblock swizzle** for L2 cache locality

Three CTA tile configurations:

| Config | Tile (M×N×K) | Warp Shape | Best For |
|--------|-------------|------------|----------|
| `hybrid` (default) | 128×64×128 | 64×32×128 | General workloads |
| `hybrid_small` | 64×64×128 | 32×32×128 | Small batch (better wave utilization) |
| `hybrid_large` | 128×128×64 | 64×32×64 | Large batch (higher compute density) |

### 4. bq512 Two-Accumulator

- `block_size=512`: dequant every 4 K-tiles (K/512 times vs K/128)
- Scale interface: `Q_k` (INT32 per quant block) × `F_g` (FP32 per super-group)
- Lowest dequant frequency, minimal register pressure

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- CUDA 11.8+
- NVIDIA GPU with Compute Capability ≥ 70 (Volta and above)
- Linux x86_64 (primary platform)

## Quick Start

### Build from Source

```bash
git clone --recursive https://github.com/IST-DASLab/gemm-int8.git
cd gemm-int8

# Ensure CUDA_HOME is set
echo $CUDA_HOME  # e.g., /usr/local/cuda

pip install cmake ninja
./build.sh
pip install -e .
```

### Basic Usage — Hybrid Kernel (Recommended)

```python
import torch
import math
import gemm_int8

# Problem size (typical LLM linear layer)
M, K, N = 32768, 2560, 2048
block_size = 256
device = torch.device('cuda')

# INT8 quantized inputs (A: activations, B: weights transposed)
A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
B_int8_t = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)

# Block-wise scales
K_blocks = math.ceil(K / block_size)
M_blocks = math.ceil(M / 128)   # tile_M = 128
N_blocks = math.ceil(N / 64)    # tile_N = 64

scale_A = torch.rand(M_blocks, K_blocks, dtype=torch.float32, device=device) * 0.1
scale_B = torch.rand(N_blocks, K_blocks, dtype=torch.float32, device=device) * 0.1

# 2nd-level scale quantization for hybrid kernel
Q_A, Q_B, F_A, F_B = gemm_int8.quantize_scales_for_hybrid(
    scale_A, scale_B,
    quant_block_size=block_size,
    super_group_size=K_blocks,
    Q_max=64
)

# Run fused kernel: quant(A) + INT8 GEMM + dequant in one shot
out = gemm_int8.blockwise_fused_matmul_hybrid(
    A_int8, B_int8_t, Q_A, Q_B, F_A, F_B,
    quant_block_size=block_size,
    super_group_size=K_blocks
)
# out: [M, N] bfloat16
```

### Hybrid Kernel with Fused Bias

```python
bias = torch.randn(N, dtype=torch.float32, device=device)

out = gemm_int8.blockwise_fused_matmul_hybrid_bias(
    A_int8, B_int8_t, Q_A, Q_B, F_A, F_B, bias,
    quant_block_size=block_size,
    super_group_size=K_blocks
)
```

### End-to-End: Online Quantization + GEMM

A full pipeline including activation quantization (Triton kernel) and GEMM:

```python
import torch
import torch.nn.functional as F
import math
import gemm_int8
from int8_utils import blockwise_quant_int8

M, N, K = 65536, 5120, 2048
block_size = 256
device = torch.device('cuda')

# BF16 inputs
A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

# --- Offline: pre-quantize weights ---
K_blocks = math.ceil(K / block_size)

def blockwise_quant_offline(x, block_m, block_k):
    M_dim, K_dim = x.shape
    m_blks = math.ceil(M_dim / block_m)
    k_blks = math.ceil(K_dim / block_k)
    x_q = torch.zeros(M_dim, k_blks * block_k, dtype=torch.int8, device=x.device)
    x_s = torch.zeros(m_blks, k_blks, dtype=torch.float32, device=x.device)
    for i in range(m_blks):
        for j in range(k_blks):
            r0, r1 = i * block_m, min((i + 1) * block_m, M_dim)
            c0, c1 = j * block_k, min((j + 1) * block_k, K_dim)
            blk = x[r0:r1, c0:c1].float()
            absmax = blk.abs().max().clamp(min=1e-10)
            scale = absmax / 127.0
            x_s[i, j] = scale
            x_q[r0:r1, c0:c1] = (blk / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, x_s

def quantize_scales_vectorized(scale, super_group_size, Q_max):
    num_rows, K_blocks = scale.shape
    G = math.ceil(K_blocks / super_group_size)
    K_pad = G * super_group_size
    if K_pad != K_blocks:
        scale = F.pad(scale, (0, K_pad - K_blocks), value=0.0)
    s = scale.view(num_rows, G, super_group_size)
    F_g = s.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / Q_max
    Q = torch.round(s / F_g).to(torch.int32).view(num_rows, K_pad)[:, :K_blocks].contiguous()
    return Q, F_g.squeeze(2)

# Weight quantization (offline, run once)
W_q, W_s = blockwise_quant_offline(W_bf16, 128, block_size)
Q_B, F_B = quantize_scales_vectorized(W_s, super_group_size=K_blocks, Q_max=64)

# --- Online: quantize activations + GEMM ---
def e2e_hybrid(A_bf16):
    # Step 1: Triton blockwise quantization of activations
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size)
    # Step 2: 2nd-level scale quantization
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks, Q_max=64)
    # Step 3: Fused INT8 GEMM
    return gemm_int8.blockwise_fused_matmul_hybrid(
        A_q, W_q, Q_A, Q_B, F_A, F_B,
        quant_block_size=block_size,
        super_group_size=K_blocks
    )

out = e2e_hybrid(A_bf16)  # [M, N] bfloat16
```

## Benchmarking

### Kernel-Only Profiling (NCU)

Profile a single kernel launch with NVIDIA Nsight Compute:

```bash
ncu --set full -o profile_hybrid python ncu_profile_hybrid.py
```

See [`ncu_profile_hybrid.py`](ncu_profile_hybrid.py) for the profiling script.

### End-to-End Benchmark

Compare all kernel variants including online activation quantization:

```bash
python e2e_benchmark.py
```

This benchmarks **quant(A) + GEMM** end-to-end for all variants against a BF16 baseline (`F.linear`), reporting latency, TFLOPS, and speedup ratios. See [`e2e_benchmark.py`](e2e_benchmark.py) for the full benchmark script.

## API Reference

### Core Functions

| Function | Tile | Block Size | Description |
|----------|------|------------|-------------|
| `blockwise_fused_matmul(x, y, x_s, y_s)` | 128×64 | 128 | Basic fused dequant |
| `blockwise_fused_matmul_128x128(x, y, x_s, y_s)` | 128×128 | 128 | Larger tile for big M |
| `blockwise_fused_matmul_kk256(x, y, x_s, y_s)` | 128×64 | 256 | bq256 fused dequant |
| `blockwise_fused_matmul_fast_dequant(x, y, x_s, y_s)` | 128×64 | 256 | Bias-offset magic I2F |
| `blockwise_fused_matmul_hybrid(x, y, Q_A, Q_B, F_A, F_B, ...)` | 128×64 | 256/512 | ⭐ Hybrid IMUL+magic |
| `blockwise_fused_matmul_hybrid_bias(x, y, Q_A, Q_B, F_A, F_B, bias, ...)` | 128×64 | 256/512 | Hybrid + fused bias |
| `blockwise_fused_matmul_hybrid_small(...)` | 64×64 | 256/512 | Small-batch hybrid |
| `blockwise_fused_matmul_hybrid_small_bias(...)` | 64×64 | 256/512 | Small-batch + bias |
| `blockwise_fused_matmul_hybrid_large(...)` | 128×128 | 256/512 | Large-batch hybrid |
| `blockwise_fused_matmul_hybrid_large_bias(...)` | 128×128 | 256/512 | Large-batch + bias |
| `blockwise_fused_matmul_bq512(...)` | 128×64 | 512 | bq512 two-accumulator |
| `blockwise_fused_matmul_bq512_fast_dequant(...)` | 128×64 | 512 | bq512 + bias-offset |

### Scale Quantization Helper

```python
Q_A, Q_B, F_A, F_B = gemm_int8.quantize_scales_for_hybrid(
    input_scale,   # [M_blocks, K_blocks] FP32
    weight_scale,  # [N_blocks, K_blocks] FP32
    quant_block_size=256,
    super_group_size=4,
    Q_max=64       # integer quantization range
)
# Returns:
#   Q_A: [M_blocks, K_blocks] INT32
#   Q_B: [N_blocks, K_blocks] INT32
#   F_A: [M_blocks, num_super_groups] FP32
#   F_B: [N_blocks, num_super_groups] FP32
```

### `torch.compile` Compatibility

All kernels are registered as custom ops and work within `torch.compile` scopes:

```python
@torch.compile(dynamic=True)
def compiled_forward(x_q, w_q, Q_A, Q_B, F_A, F_B):
    return gemm_int8.blockwise_fused_matmul_hybrid(
        x_q, w_q, Q_A, Q_B, F_A, F_B,
        quant_block_size=256, super_group_size=8)
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{gemm_int8,
  author = {Roberto L. Castro and Saleh Ashkboos and Soroush Tabesh},
  title = {Fused Blockwise INT8 GEMM Kernels},
  url = {https://github.com/IST-DASLab/gemm-int8},
  year = {2024},
}
```

```bibtex
@article{halo2025,
      title={HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs}, 
      author={Saleh Ashkboos and Mahdi Nikdan and Soroush Tabesh and Roberto L. Castro and Torsten Hoefler and Dan Alistarh},
      year={2025},
      eprint={2501.02625},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.02625}, 
}
```

## Acknowledgements

This project uses [CUTLASS](https://github.com/NVIDIA/cutlass) for optimized CUDA Tensor Core primitives.
