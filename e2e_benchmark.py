import torch
import torch.nn.functional as F
import math
import gemm_int8
from int8_utils import blockwise_quant_int8, w8a8_block_int8_matmul as triton_w8a8_matmul


def quantize_scales_vectorized(scale: torch.Tensor, super_group_size: int, Q_max: int):
    """Vectorized 2nd-level scale quantization (no Python loop over super-groups).

    Args:
        scale: [num_rows, K_blocks] FP32
        super_group_size: number of K-blocks per super-group
        Q_max: integer quantization range
    Returns:
        Q: [num_rows, K_blocks] INT32
        F: [num_rows, num_super_groups] FP32
    """
    num_rows, K_blocks = scale.shape
    G = math.ceil(K_blocks / super_group_size)
    # Pad K_blocks to multiple of super_group_size
    K_pad = G * super_group_size
    if K_pad != K_blocks:
        scale_pad = F.pad(scale, (0, K_pad - K_blocks), value=0.0)
    else:
        scale_pad = scale
    # [num_rows, G, super_group_size]
    s = scale_pad.view(num_rows, G, super_group_size)
    # F_g: [num_rows, G, 1]
    F_g = s.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / Q_max
    # Q: [num_rows, G, super_group_size] -> [num_rows, K_pad]
    Q_pad = torch.round(s / F_g).to(torch.int32).view(num_rows, K_pad)
    Q = Q_pad[:, :K_blocks].contiguous()
    F_out = F_g.squeeze(2)  # [num_rows, G]
    return Q, F_out

M, N, K = 65536, 5120, 2048
num_iterations = 100

device = torch.device('cuda')
print(f"Device: {torch.cuda.get_device_name(device)}")
print(f"Shape: M={M}, K={K}, N={N}")

# -----------------------------------------------------------------------
# Input tensor (bf16, online quantized each iteration)
# -----------------------------------------------------------------------
A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)

# -----------------------------------------------------------------------
# Offline: pre-quantize W for each kernel variant
# -----------------------------------------------------------------------
def blockwise_quant_offline(x, block_m, block_k):
    """CPU-side reference quant for weight (offline, not benchmarked)."""
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

block_size_128 = 128
block_size_256 = 256
block_size_512 = 512
K_blocks_128 = math.ceil(K / block_size_128)
K_blocks_256 = math.ceil(K / block_size_256)
K_blocks_512 = math.ceil(K / block_size_512)
N_blocks_128 = math.ceil(N / block_size_128)

# W quantized offline for each variant
# bq128 fused: W tile [64, 128]
W_q128, W_s128 = blockwise_quant_offline(W_bf16, 64, block_size_128)
# bq256 fused / fast_dequant: W tile [64, 256]
W_q256, W_s256 = blockwise_quant_offline(W_bf16, 64, block_size_256)
# hybrid bq256: W tile [64, 256]
W_q_hyb256, W_s_hyb256 = blockwise_quant_offline(W_bf16, 64, block_size_256)
Q_B_hyb256, F_B_hyb256 = quantize_scales_vectorized(W_s_hyb256, super_group_size=K_blocks_256, Q_max=64)
# hybrid bq512: W tile [64, 512]
W_q_hyb512, W_s_hyb512 = blockwise_quant_offline(W_bf16, 64, block_size_512)
Q_B_hyb512, F_B_hyb512 = quantize_scales_vectorized(W_s_hyb512, super_group_size=K_blocks_512, Q_max=64)
# hybrid_small bq256: W tile [64, 256] (CTA 64x64)
W_q_hybs256, W_s_hybs256 = W_q_hyb256, W_s_hyb256
Q_B_hybs256, F_B_hybs256 = Q_B_hyb256, F_B_hyb256
# hybrid_small bq512: W tile [64, 512] (CTA 64x64)
W_q_hybs512, W_s_hybs512 = W_q_hyb512, W_s_hyb512
Q_B_hybs512, F_B_hybs512 = Q_B_hyb512, F_B_hyb512
# hybrid_large bq256: W tile [128, 256] (CTA 128x128)
W_q_hybl256, W_s_hybl256 = blockwise_quant_offline(W_bf16, 128, block_size_256)
Q_B_hybl256, F_B_hybl256 = quantize_scales_vectorized(W_s_hybl256, super_group_size=K_blocks_256, Q_max=64)
# hybrid_large bq512: W tile [128, 512] (CTA 128x128)
W_q_hybl512, W_s_hybl512 = blockwise_quant_offline(W_bf16, 128, block_size_512)
Q_B_hybl512, F_B_hybl512 = quantize_scales_vectorized(W_s_hybl512, super_group_size=K_blocks_512, Q_max=64)
# bq512 2-accum / fast_dequant: W tile [64, 512]
W_q512, W_s512 = blockwise_quant_offline(W_bf16, 64, block_size_512)
Q_B_512, F_B_512 = quantize_scales_vectorized(W_s512, super_group_size=1, Q_max=8)
# triton bq128: W tile [128, 128]
W_q_tr128, W_s_tr128 = blockwise_quant_offline(W_bf16, block_size_128, block_size_128)
# triton bq256: W tile [128, 256]
W_q_tr256, W_s_tr256 = blockwise_quant_offline(W_bf16, block_size_128, block_size_256)

bias_fp32 = torch.randn(N, dtype=torch.float32, device=device)


def bench(fn, warmup=10, iters=num_iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# -----------------------------------------------------------------------
# E2E lambdas: quant(A) + GEMM
# Each lambda quantizes A_bf16 online, then runs GEMM with pre-quantized W.
# -----------------------------------------------------------------------

# bf16 baseline (no quant)
lat_bf16 = bench(lambda: F.linear(A_bf16, W_bf16))

# bq128 fused (128x64): A tile [128, 128]
def e2e_bq128():
    A_q, A_s = blockwise_quant_int8(A_bf16, block_size_128, block_size_128)
    return gemm_int8.blockwise_fused_matmul(A_q, W_q128, A_s, W_s128)
lat_e2e_bq128 = bench(e2e_bq128)

# bq128 fused (128x128): A tile [128, 128]
def e2e_bq128_128x128():
    A_q, A_s = blockwise_quant_int8(A_bf16, block_size_128, block_size_128)
    return gemm_int8.blockwise_fused_matmul_128x128(A_q, W_q128, A_s, W_s128)
lat_e2e_bq128_128x128 = bench(e2e_bq128_128x128)

# bq256 fused (128x64): A tile [256, 256]
def e2e_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, block_size_256, block_size_256)
    return gemm_int8.blockwise_fused_matmul_kk256(A_q, W_q256, A_s, W_s256)
lat_e2e_bq256 = bench(e2e_bq256)

# bq256 fast_dequant (128x64): A tile [256, 256]
def e2e_bq256_fast():
    A_q, A_s = blockwise_quant_int8(A_bf16, block_size_256, block_size_256)
    return gemm_int8.blockwise_fused_matmul_fast_dequant(A_q, W_q256, A_s, W_s256)
lat_e2e_bq256_fast = bench(e2e_bq256_fast)

# hybrid bq256 (128x64): A tile [128, 256], then vectorized 2nd-level scale quant
def e2e_hybrid_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_256)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_256, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid(
        A_q, W_q_hyb256, Q_A, Q_B_hyb256, F_A, F_B_hyb256,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
lat_e2e_hybrid_bq256 = bench(e2e_hybrid_bq256)

# hybrid bq512 (128x64): A tile [128, 512], then vectorized 2nd-level scale quant
def e2e_hybrid_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_512, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid(
        A_q, W_q_hyb512, Q_A, Q_B_hyb512, F_A, F_B_hyb512,
        quant_block_size=block_size_512, super_group_size=K_blocks_512)
lat_e2e_hybrid_bq512 = bench(e2e_hybrid_bq512)

# hybrid+bias bq256 (128x64)
def e2e_hybrid_bias_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_256)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_256, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_bias(
        A_q, W_q_hyb256, Q_A, Q_B_hyb256, F_A, F_B_hyb256, bias_fp32,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
lat_e2e_hybrid_bias_bq256 = bench(e2e_hybrid_bias_bq256)

# hybrid+bias bq512 (128x64)
def e2e_hybrid_bias_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_512, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_bias(
        A_q, W_q_hyb512, Q_A, Q_B_hyb512, F_A, F_B_hyb512, bias_fp32,
        quant_block_size=block_size_512, super_group_size=K_blocks_512)
lat_e2e_hybrid_bias_bq512 = bench(e2e_hybrid_bias_bq512)

# hybrid_small bq256 (64x64): A tile [64, 256]
def e2e_hybrid_small_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, 64, block_size_256)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_256, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_small(
        A_q, W_q_hybs256, Q_A, Q_B_hybs256, F_A, F_B_hybs256,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
lat_e2e_hybrid_small_bq256 = bench(e2e_hybrid_small_bq256)

# hybrid_small bq512 (64x64): A tile [64, 512]
def e2e_hybrid_small_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 64, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_512, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_small(
        A_q, W_q_hybs512, Q_A, Q_B_hybs512, F_A, F_B_hybs512,
        quant_block_size=block_size_512, super_group_size=K_blocks_512)
lat_e2e_hybrid_small_bq512 = bench(e2e_hybrid_small_bq512)

# hybrid_small+bias bq256 (64x64)
def e2e_hybrid_small_bias_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, 64, block_size_256)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_256, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_small_bias(
        A_q, W_q_hybs256, Q_A, Q_B_hybs256, F_A, F_B_hybs256, bias_fp32,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
lat_e2e_hybrid_small_bias_bq256 = bench(e2e_hybrid_small_bias_bq256)

# hybrid_small+bias bq512 (64x64)
def e2e_hybrid_small_bias_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 64, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_512, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_small_bias(
        A_q, W_q_hybs512, Q_A, Q_B_hybs512, F_A, F_B_hybs512, bias_fp32,
        quant_block_size=block_size_512, super_group_size=K_blocks_512)
lat_e2e_hybrid_small_bias_bq512 = bench(e2e_hybrid_small_bias_bq512)

# hybrid_large bq256 (128x128): A tile [128, 256]
def e2e_hybrid_large_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_256)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_256, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_large(
        A_q, W_q_hybl256, Q_A, Q_B_hybl256, F_A, F_B_hybl256,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
lat_e2e_hybrid_large_bq256 = bench(e2e_hybrid_large_bq256)

# hybrid_large bq512 (128x128): A tile [128, 512]
def e2e_hybrid_large_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_512, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_large(
        A_q, W_q_hybl512, Q_A, Q_B_hybl512, F_A, F_B_hybl512,
        quant_block_size=block_size_512, super_group_size=K_blocks_512)
lat_e2e_hybrid_large_bq512 = bench(e2e_hybrid_large_bq512)

# hybrid_large+bias bq256 (128x128)
def e2e_hybrid_large_bias_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_256)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_256, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_large_bias(
        A_q, W_q_hybl256, Q_A, Q_B_hybl256, F_A, F_B_hybl256, bias_fp32,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
lat_e2e_hybrid_large_bias_bq256 = bench(e2e_hybrid_large_bias_bq256)

# hybrid_large+bias bq512 (128x128)
def e2e_hybrid_large_bias_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=K_blocks_512, Q_max=64)
    return gemm_int8.blockwise_fused_matmul_hybrid_large_bias(
        A_q, W_q_hybl512, Q_A, Q_B_hybl512, F_A, F_B_hybl512, bias_fp32,
        quant_block_size=block_size_512, super_group_size=K_blocks_512)
lat_e2e_hybrid_large_bias_bq512 = bench(e2e_hybrid_large_bias_bq512)

# bq512 2-accum (128x64): A tile [128, 512], vectorized scale quant (Q_max=8, super_group=1)
def e2e_bq512():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=1, Q_max=8)
    return gemm_int8.blockwise_fused_matmul_bq512(
        A_q, W_q512, Q_A, Q_B_512, F_A, F_B_512,
        quant_block_size=block_size_512, super_group_size=1)
lat_e2e_bq512 = bench(e2e_bq512)

# bq512 fast_dequant (128x64)
def e2e_bq512_fast():
    A_q, A_s = blockwise_quant_int8(A_bf16, 128, block_size_512)
    Q_A, F_A = quantize_scales_vectorized(A_s, super_group_size=1, Q_max=8)
    return gemm_int8.blockwise_fused_matmul_bq512_fast_dequant(
        A_q, W_q512, Q_A, Q_B_512, F_A, F_B_512,
        quant_block_size=block_size_512, super_group_size=1)
lat_e2e_bq512_fast = bench(e2e_bq512_fast)

# triton bq128: A tile [128, 128]
def e2e_triton_bq128():
    A_q, A_s = blockwise_quant_int8(A_bf16, block_size_128, block_size_128)
    return triton_w8a8_matmul(
        A_q, W_q_tr128, A_s, W_s_tr128,
        block_size=[128, 128], output_dtype=torch.bfloat16, input_quant_mode="blockwise")
lat_e2e_triton_bq128 = bench(e2e_triton_bq128)

# triton bq256: A tile [128, 256]
def e2e_triton_bq256():
    A_q, A_s = blockwise_quant_int8(A_bf16, block_size_128, block_size_256)
    return triton_w8a8_matmul(
        A_q, W_q_tr256, A_s, W_s_tr256,
        block_size=[128, 256], output_dtype=torch.bfloat16, input_quant_mode="blockwise")
lat_e2e_triton_bq256 = bench(e2e_triton_bq256)

# -----------------------------------------------------------------------
# Also benchmark standalone quant kernel for reference
# -----------------------------------------------------------------------
lat_quant_128 = bench(lambda: blockwise_quant_int8(A_bf16, block_size_128, block_size_128))
lat_quant_256 = bench(lambda: blockwise_quant_int8(A_bf16, block_size_256, block_size_256))
lat_quant_128x256 = bench(lambda: blockwise_quant_int8(A_bf16, 128, block_size_256))
lat_quant_128x512 = bench(lambda: blockwise_quant_int8(A_bf16, 128, block_size_512))
lat_quant_64x256 = bench(lambda: blockwise_quant_int8(A_bf16, 64, block_size_256))
lat_quant_64x512 = bench(lambda: blockwise_quant_int8(A_bf16, 64, block_size_512))

# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
flops = 2.0 * M * N * K

e2e_results = [
    ("bf16 GEMM (F.linear)",                   lat_bf16),
    ("int8 bq128 fused (128x64)",               lat_e2e_bq128),
    ("int8 bq128 fused (128x128)",              lat_e2e_bq128_128x128),
    ("int8 bq256 fused (128x64)",               lat_e2e_bq256),
    ("int8 bq256 fast_dequant (128x64)",        lat_e2e_bq256_fast),
    ("int8 hybrid IMUL+magic bq256 (128x64)",   lat_e2e_hybrid_bq256),
    ("int8 hybrid IMUL+magic bq512 (128x64)",   lat_e2e_hybrid_bq512),
    ("int8 hybrid+bias bq256 (128x64)",          lat_e2e_hybrid_bias_bq256),
    ("int8 hybrid+bias bq512 (128x64)",          lat_e2e_hybrid_bias_bq512),
    ("int8 hybrid_small bq256 (64x64)",          lat_e2e_hybrid_small_bq256),
    ("int8 hybrid_small bq512 (64x64)",          lat_e2e_hybrid_small_bq512),
    ("int8 hybrid_small+bias bq256 (64x64)",     lat_e2e_hybrid_small_bias_bq256),
    ("int8 hybrid_small+bias bq512 (64x64)",     lat_e2e_hybrid_small_bias_bq512),
    ("int8 hybrid_large bq256 (128x128)",        lat_e2e_hybrid_large_bq256),
    ("int8 hybrid_large bq512 (128x128)",        lat_e2e_hybrid_large_bq512),
    ("int8 hybrid_large+bias bq256 (128x128)",   lat_e2e_hybrid_large_bias_bq256),
    ("int8 hybrid_large+bias bq512 (128x128)",   lat_e2e_hybrid_large_bias_bq512),
    ("int8 bq512 2-accum (128x64)",             lat_e2e_bq512),
    ("int8 bq512 fast_dequant (128x64)",        lat_e2e_bq512_fast),
    ("triton w8a8 bq128 (64x128)",              lat_e2e_triton_bq128),
    ("triton w8a8 bq256 (64x128)",              lat_e2e_triton_bq256),
]

print(f"\nFLOPs: {flops / 1e9:.2f} GFLOP, iters: {num_iterations}")
print(f"\n{'E2E: quant(A) + GEMM':<50} {'Latency(ms)':<14} {'TFLOPS':<10} {'vs bf16':<10}")
print("=" * 85)
for name, lat in e2e_results:
    tflops = flops / (lat * 1e-3) / 1e12
    ratio = f"{lat_bf16 / lat:.2f}x" if name != "bf16 GEMM (F.linear)" else "-"
    print(f"{name:<50} {lat:<14.4f} {tflops:<10.2f} {ratio:<10}")

print(f"\n{'Standalone quant kernel (A only)':<50} {'Latency(ms)':<14} {'BW(GB/s)':<12}")
print("=" * 76)
quant_bytes = M * K * 2  # read bf16
for qname, qlat, bm, bk in [
    ("blockwise_quant_int8 [128x128]",  lat_quant_128,     128, 128),
    ("blockwise_quant_int8 [256x256]",  lat_quant_256,     256, 256),
    ("blockwise_quant_int8 [128x256]",  lat_quant_128x256, 128, 256),
    ("blockwise_quant_int8 [128x512]",  lat_quant_128x512, 128, 512),
    ("blockwise_quant_int8 [64x256]",   lat_quant_64x256,  64,  256),
    ("blockwise_quant_int8 [64x512]",   lat_quant_64x512,  64,  512),
]:
    k_blks = math.ceil(K / bk)
    total_bytes = quant_bytes + M * k_blks * bk + math.ceil(M / bm) * k_blks * 4
    bw = total_bytes / (qlat * 1e-3) / 1e9
    print(f"{qname:<50} {qlat:<14.4f} {bw:<12.1f}")

quant_lat_map = {
    "int8 bq128 fused (128x64)":              lat_quant_128,
    "int8 bq128 fused (128x128)":             lat_quant_128,
    "int8 bq256 fused (128x64)":              lat_quant_256,
    "int8 bq256 fast_dequant (128x64)":       lat_quant_256,
    "int8 hybrid IMUL+magic bq256 (128x64)":  lat_quant_128x256,
    "int8 hybrid IMUL+magic bq512 (128x64)":  lat_quant_128x512,
    "int8 hybrid+bias bq256 (128x64)":         lat_quant_128x256,
    "int8 hybrid+bias bq512 (128x64)":         lat_quant_128x512,
    "int8 hybrid_small bq256 (64x64)":         lat_quant_64x256,
    "int8 hybrid_small bq512 (64x64)":         lat_quant_64x512,
    "int8 hybrid_small+bias bq256 (64x64)":    lat_quant_64x256,
    "int8 hybrid_small+bias bq512 (64x64)":    lat_quant_64x512,
    "int8 hybrid_large bq256 (128x128)":       lat_quant_128x256,
    "int8 hybrid_large bq512 (128x128)":       lat_quant_128x512,
    "int8 hybrid_large+bias bq256 (128x128)":  lat_quant_128x256,
    "int8 hybrid_large+bias bq512 (128x128)":  lat_quant_128x512,
    "int8 bq512 2-accum (128x64)":            lat_quant_128x512,
    "int8 bq512 fast_dequant (128x64)":       lat_quant_128x512,
    "triton w8a8 bq128 (64x128)":             lat_quant_128,
    "triton w8a8 bq256 (64x128)":             lat_quant_128x256,
}

print(f"\n{'Quant overhead (quant lat / e2e lat)':<48} {'Quant(ms)':<11} {'E2E(ms)':<11} {'Overhead'}")
print("=" * 76)
for name, e2e_lat in e2e_results:
    if name not in quant_lat_map:
        continue
    qlat = quant_lat_map[name]
    print(f"{name:<48} {qlat:<11.4f} {e2e_lat:<11.4f} {qlat / e2e_lat * 100:.1f}%")
