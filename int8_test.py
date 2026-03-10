import torch
import torch.nn.functional as F
import math
import gemm_int8
from int8_utils import w8a8_block_int8_matmul as triton_w8a8_matmul

M, K, N = 32768, 3420, 1280
block_size = 128
num_iterations = 100

device = torch.device('cuda')
print(f"Device: {torch.cuda.get_device_name(device)}")
print(f"Shape: M={M}, K={K}, N={N}")

A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)  # weight: [N, K]

A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)

# gemm_int8 computes x @ y.T, so y is (N, K)
B_int8_t = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)

# blockwise scales: bq128
K_blocks = math.ceil(K / block_size)
M_blocks = math.ceil(M / block_size)
N_blocks = math.ceil(N / block_size)
scale_a = torch.rand(M_blocks, K_blocks, dtype=torch.float32, device=device) * 0.1
scale_b = torch.rand(N_blocks, K_blocks, dtype=torch.float32, device=device) * 0.1

# blockwise scales: bq256
block_size_256 = 256
K_blocks_256 = math.ceil(K / block_size_256)
scale_a_256 = torch.rand(M_blocks, K_blocks_256, dtype=torch.float32, device=device) * 0.1
scale_b_256 = torch.rand(N_blocks, K_blocks_256, dtype=torch.float32, device=device) * 0.1

# hybrid scales: CTA tile 128x64
M_blocks_hyb = math.ceil(M / 128)
N_blocks_hyb = math.ceil(N / 64)
scale_a_hyb = torch.rand(M_blocks_hyb, K_blocks_256, dtype=torch.float32, device=device) * 0.1
scale_b_hyb = torch.rand(N_blocks_hyb, K_blocks_256, dtype=torch.float32, device=device) * 0.1
Q_A_hybrid, Q_B_hybrid, F_A_hybrid, F_B_hybrid = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb, scale_b_hyb,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)

# bq512 scales
block_size_512 = 512
K_blocks_512 = math.ceil(K / block_size_512)
scale_a_512 = torch.rand(M_blocks, K_blocks_512, dtype=torch.float32, device=device) * 0.1
scale_b_512 = torch.rand(N_blocks, K_blocks_512, dtype=torch.float32, device=device) * 0.1
Q_A_bq512, Q_B_bq512, F_A_bq512, F_B_bq512 = gemm_int8.quantize_scales_for_hybrid(
    scale_a_512, scale_b_512,
    quant_block_size=block_size_512, super_group_size=1, Q_max=8)

# hybrid_small scales: CTA tile 64x64
M_blocks_hyb_s = math.ceil(M / 64)
N_blocks_hyb_s = math.ceil(N / 64)
scale_a_hyb_s = torch.rand(M_blocks_hyb_s, K_blocks_256, dtype=torch.float32, device=device) * 0.1
scale_b_hyb_s = torch.rand(N_blocks_hyb_s, K_blocks_256, dtype=torch.float32, device=device) * 0.1
Q_A_hyb_s, Q_B_hyb_s, F_A_hyb_s, F_B_hyb_s = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb_s, scale_b_hyb_s,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)

scale_a_hyb_s512 = torch.rand(M_blocks_hyb_s, K_blocks_512, dtype=torch.float32, device=device) * 0.1
scale_b_hyb_s512 = torch.rand(N_blocks_hyb_s, K_blocks_512, dtype=torch.float32, device=device) * 0.1
Q_A_hyb_s512, Q_B_hyb_s512, F_A_hyb_s512, F_B_hyb_s512 = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb_s512, scale_b_hyb_s512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512, Q_max=64)

# hybrid_large scales: CTA tile 128x128
M_blocks_hyb_l = math.ceil(M / 128)
N_blocks_hyb_l = math.ceil(N / 128)
scale_a_hyb_l = torch.rand(M_blocks_hyb_l, K_blocks_256, dtype=torch.float32, device=device) * 0.1
scale_b_hyb_l = torch.rand(N_blocks_hyb_l, K_blocks_256, dtype=torch.float32, device=device) * 0.1
Q_A_hyb_l, Q_B_hyb_l, F_A_hyb_l, F_B_hyb_l = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb_l, scale_b_hyb_l,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)

scale_a_hyb_l512 = torch.rand(M_blocks_hyb_l, K_blocks_512, dtype=torch.float32, device=device) * 0.1
scale_b_hyb_l512 = torch.rand(N_blocks_hyb_l, K_blocks_512, dtype=torch.float32, device=device) * 0.1
Q_A_hyb_l512, Q_B_hyb_l512, F_A_hyb_l512, F_B_hyb_l512 = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb_l512, scale_b_hyb_l512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512, Q_max=64)

# hybrid bq512 scales
scale_a_hyb512 = torch.rand(M_blocks_hyb, K_blocks_512, dtype=torch.float32, device=device) * 0.1
scale_b_hyb512 = torch.rand(N_blocks_hyb, K_blocks_512, dtype=torch.float32, device=device) * 0.1
Q_A_hyb512, Q_B_hyb512, F_A_hyb512, F_B_hyb512 = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb512, scale_b_hyb512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512, Q_max=64)

# Triton w8a8 GEMM scales
triton_scale_a_128 = torch.rand(M_blocks, K_blocks, dtype=torch.float32, device=device) * 0.1
triton_scale_b_128 = torch.rand(N_blocks, K_blocks, dtype=torch.float32, device=device) * 0.1
triton_scale_a_256 = torch.rand(M_blocks, K_blocks_256, dtype=torch.float32, device=device) * 0.1
triton_scale_b_256 = torch.rand(N_blocks, K_blocks_256, dtype=torch.float32, device=device) * 0.1


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


lat_bf16          = bench(lambda: F.linear(A_bf16, W_bf16))
lat_gemm_int8     = bench(lambda: gemm_int8.matmul(A_int8, B_int8_t))
lat_blockwise_fused          = bench(lambda: gemm_int8.blockwise_fused_matmul(A_int8, B_int8_t, scale_a, scale_b))
lat_blockwise_fused_128x128  = bench(lambda: gemm_int8.blockwise_fused_matmul_128x128(A_int8, B_int8_t, scale_a, scale_b))
lat_blockwise_fused_bq256    = bench(lambda: gemm_int8.blockwise_fused_matmul_kk256(A_int8, B_int8_t, scale_a_256, scale_b_256))
lat_blockwise_fused_fast_dequant = bench(lambda: gemm_int8.blockwise_fused_matmul_fast_dequant(A_int8, B_int8_t, scale_a_256, scale_b_256))
lat_hybrid        = bench(lambda: gemm_int8.blockwise_fused_matmul_hybrid(
    A_int8, B_int8_t, Q_A_hybrid, Q_B_hybrid, F_A_hybrid, F_B_hybrid,
    quant_block_size=block_size_256, super_group_size=K_blocks_256))
lat_hybrid_bq512  = bench(lambda: gemm_int8.blockwise_fused_matmul_hybrid(
    A_int8, B_int8_t, Q_A_hyb512, Q_B_hyb512, F_A_hyb512, F_B_hyb512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512))
lat_hybrid_large  = bench(lambda: gemm_int8.blockwise_fused_matmul_hybrid_large(
    A_int8, B_int8_t, Q_A_hyb_l, Q_B_hyb_l, F_A_hyb_l, F_B_hyb_l,
    quant_block_size=block_size_256, super_group_size=K_blocks_256))
lat_hybrid_large_bq512 = bench(lambda: gemm_int8.blockwise_fused_matmul_hybrid_large(
    A_int8, B_int8_t, Q_A_hyb_l512, Q_B_hyb_l512, F_A_hyb_l512, F_B_hyb_l512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512))
lat_hybrid_small  = bench(lambda: gemm_int8.blockwise_fused_matmul_hybrid_small(
    A_int8, B_int8_t, Q_A_hyb_s, Q_B_hyb_s, F_A_hyb_s, F_B_hyb_s,
    quant_block_size=block_size_256, super_group_size=K_blocks_256))
lat_hybrid_small_bq512 = bench(lambda: gemm_int8.blockwise_fused_matmul_hybrid_small(
    A_int8, B_int8_t, Q_A_hyb_s512, Q_B_hyb_s512, F_A_hyb_s512, F_B_hyb_s512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512))
lat_bq512         = bench(lambda: gemm_int8.blockwise_fused_matmul_bq512(
    A_int8, B_int8_t, Q_A_bq512, Q_B_bq512, F_A_bq512, F_B_bq512,
    quant_block_size=block_size_512, super_group_size=1))
lat_bq512_fast_dequant = bench(lambda: gemm_int8.blockwise_fused_matmul_bq512_fast_dequant(
    A_int8, B_int8_t, Q_A_bq512, Q_B_bq512, F_A_bq512, F_B_bq512,
    quant_block_size=block_size_512, super_group_size=1))
lat_triton_gemm_bq128 = bench(lambda: triton_w8a8_matmul(
    A_int8, B_int8_t, triton_scale_a_128, triton_scale_b_128,
    block_size=[128, 128], output_dtype=torch.bfloat16, input_quant_mode="blockwise"))
lat_triton_gemm_bq256 = bench(lambda: triton_w8a8_matmul(
    A_int8, B_int8_t, triton_scale_a_256, triton_scale_b_256,
    block_size=[128, 256], output_dtype=torch.bfloat16, input_quant_mode="blockwise"))

flops = 2.0 * M * N * K

results = [
    ("bf16 GEMM (F.linear)",                   lat_bf16),
    ("int8 GEMM (gemm_int8.matmul)",            lat_gemm_int8),
    ("int8 blockwise fused bq128 (128x64)",     lat_blockwise_fused),
    ("int8 blockwise fused bq128 (128x128)",    lat_blockwise_fused_128x128),
    ("int8 blockwise fused bq256 (128x64)",     lat_blockwise_fused_bq256),
    ("int8 bq256 fast_dequant (128x64)",        lat_blockwise_fused_fast_dequant),
    ("int8 hybrid IMUL+magic bq256 (128x64)",   lat_hybrid),
    ("int8 hybrid IMUL+magic bq512 (128x64)",   lat_hybrid_bq512),
    ("int8 hybrid_large bq256 (128x128)",        lat_hybrid_large),
    ("int8 hybrid_large bq512 (128x128)",        lat_hybrid_large_bq512),
    ("int8 hybrid_small bq256 (64x64)",         lat_hybrid_small),
    ("int8 hybrid_small bq512 (64x64)",         lat_hybrid_small_bq512),
    ("int8 bq512 2-accum (128x64)",             lat_bq512),
    ("int8 bq512 fast_dequant (128x64)",        lat_bq512_fast_dequant),
    ("triton w8a8 GEMM bq128 (64x128)",         lat_triton_gemm_bq128),
    ("triton w8a8 GEMM bq256 (64x128)",         lat_triton_gemm_bq256),
]

print(f"\nFLOPs: {flops / 1e9:.2f} GFLOP, iters: {num_iterations}")
print(f"\n{'Kernel':<45} {'Latency(ms)':<14} {'TFLOPS':<10}")
print("=" * 70)
for name, lat in results:
    tflops = flops / (lat * 1e-3) / 1e12
    print(f"{name:<45} {lat:<14.4f} {tflops:<10.2f}")

print(f"\n{'Speedup vs bf16':<55} {'Ratio':<10}")
print("=" * 70)
for name, lat in results[1:]:
    print(f"{name:<55} {lat_bf16 / lat:<10.2f}x")

print(f"\n{'bq256 vs bq128 fused':<55} {'Ratio':<10}")
print("=" * 70)
print(f"{'bq256 / bq128':<55} {lat_blockwise_fused / lat_blockwise_fused_bq256:<10.2f}x")

print(f"\n{'fast_dequant vs bq256':<55} {'Ratio':<10}")
print("=" * 70)
print(f"{'fast_dequant / bq256':<55} {lat_blockwise_fused_bq256 / lat_blockwise_fused_fast_dequant:<10.2f}x")

print(f"\n{'hybrid vs bq256':<55} {'Ratio':<10}")
print("=" * 70)
print(f"{'hybrid bq256 / bq256':<55} {lat_blockwise_fused_bq256 / lat_hybrid:<10.2f}x")
print(f"{'hybrid bq512 / bq256':<55} {lat_blockwise_fused_bq256 / lat_hybrid_bq512:<10.2f}x")
print(f"{'hybrid bq512 / hybrid bq256':<55} {lat_hybrid / lat_hybrid_bq512:<10.2f}x")

print(f"\n{'bq512 vs bq256':<55} {'Ratio':<10}")
print("=" * 70)
print(f"{'bq512 / bq256':<55} {lat_blockwise_fused_bq256 / lat_bq512:<10.2f}x")
print(f"{'bq512_fast_dequant / bq256':<55} {lat_blockwise_fused_bq256 / lat_bq512_fast_dequant:<10.2f}x")
print(f"{'bq512_fast_dequant / bq512':<55} {lat_bq512 / lat_bq512_fast_dequant:<10.2f}x")

print(f"\n{'128x128 vs 128x64 fused':<55} {'Ratio':<10}")
print("=" * 70)
print(f"{'128x128 / 128x64':<55} {lat_blockwise_fused / lat_blockwise_fused_128x128:<10.2f}x")

# =====================================================================
# Accuracy vs BF16 reference
# =====================================================================

def blockwise_quant_int8_ref(x, block_m, block_k):
    M_dim, K_dim = x.shape
    m_blks = math.ceil(M_dim / block_m)
    k_blks = math.ceil(K_dim / block_k)
    x_q = torch.zeros_like(x, dtype=torch.int8)
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


def compute_metrics(out, ref):
    o, r = out.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    rel_l1 = (o - r).abs().sum().item() / r.abs().sum().clamp(min=1e-12).item()
    rmse = ((o - r) ** 2).mean().sqrt().item()
    mse = ((o - r) ** 2).mean().item()
    return cos, rel_l1, rmse, mse


A_src = torch.randn(M, K, dtype=torch.bfloat16, device=device)
W_src = torch.randn(N, K, dtype=torch.bfloat16, device=device)
out_bf16 = F.linear(A_src, W_src)

# --- bq128 ---
A_q128, A_s128 = blockwise_quant_int8_ref(A_src, block_size, block_size)
W_q128, W_s128 = blockwise_quant_int8_ref(W_src, block_size, block_size)
out_bq128 = gemm_int8.blockwise_fused_matmul(A_q128, W_q128, A_s128, W_s128)

# --- bq256 ---
A_q256, A_s256 = blockwise_quant_int8_ref(A_src, block_size_256, block_size_256)
W_q256, W_s256 = blockwise_quant_int8_ref(W_src, block_size_256, block_size_256)
out_bq256 = gemm_int8.blockwise_fused_matmul_kk256(A_q256, W_q256, A_s256, W_s256)

# --- bq256 fast_dequant ---
out_fast_dequant = gemm_int8.blockwise_fused_matmul_fast_dequant(A_q256, W_q256, A_s256, W_s256)

# --- hybrid bq256 ---
A_q_hyb, A_s_hyb = blockwise_quant_int8_ref(A_src, 128, block_size_256)
W_q_hyb, W_s_hyb = blockwise_quant_int8_ref(W_src, 64, block_size_256)
Q_A_hyb, Q_B_hyb, F_A_hyb, F_B_hyb = gemm_int8.quantize_scales_for_hybrid(
    A_s_hyb, W_s_hyb,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)
out_hybrid = gemm_int8.blockwise_fused_matmul_hybrid(
    A_q_hyb, W_q_hyb, Q_A_hyb, Q_B_hyb, F_A_hyb, F_B_hyb,
    quant_block_size=block_size_256, super_group_size=K_blocks_256)

# --- hybrid_small bq256 ---
A_q_hyb_s, A_s_hyb_s = blockwise_quant_int8_ref(A_src, 64, block_size_256)
W_q_hyb_s, W_s_hyb_s = blockwise_quant_int8_ref(W_src, 64, block_size_256)
Q_A_hs, Q_B_hs, F_A_hs, F_B_hs = gemm_int8.quantize_scales_for_hybrid(
    A_s_hyb_s, W_s_hyb_s,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)
out_hybrid_small = gemm_int8.blockwise_fused_matmul_hybrid_small(
    A_q_hyb_s, W_q_hyb_s, Q_A_hs, Q_B_hs, F_A_hs, F_B_hs,
    quant_block_size=block_size_256, super_group_size=K_blocks_256)

# --- hybrid_small bq512 ---
A_q_hyb_s512, A_s_hyb_s512 = blockwise_quant_int8_ref(A_src, 64, block_size_512)
W_q_hyb_s512, W_s_hyb_s512 = blockwise_quant_int8_ref(W_src, 64, block_size_512)
Q_A_hs512, Q_B_hs512, F_A_hs512, F_B_hs512 = gemm_int8.quantize_scales_for_hybrid(
    A_s_hyb_s512, W_s_hyb_s512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512, Q_max=64)
out_hybrid_small_bq512 = gemm_int8.blockwise_fused_matmul_hybrid_small(
    A_q_hyb_s512, W_q_hyb_s512, Q_A_hs512, Q_B_hs512, F_A_hs512, F_B_hs512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512)

# --- hybrid bq512 ---
A_q_hyb512, A_s_hyb512 = blockwise_quant_int8_ref(A_src, 128, block_size_512)
W_q_hyb512, W_s_hyb512 = blockwise_quant_int8_ref(W_src, 64, block_size_512)
Q_A_h512, Q_B_h512, F_A_h512, F_B_h512 = gemm_int8.quantize_scales_for_hybrid(
    A_s_hyb512, W_s_hyb512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512, Q_max=64)
out_hybrid_bq512 = gemm_int8.blockwise_fused_matmul_hybrid(
    A_q_hyb512, W_q_hyb512, Q_A_h512, Q_B_h512, F_A_h512, F_B_h512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512)

# --- hybrid_large bq256 ---
A_q_hyb_l, A_s_hyb_l = blockwise_quant_int8_ref(A_src, 128, block_size_256)
W_q_hyb_l, W_s_hyb_l = blockwise_quant_int8_ref(W_src, 128, block_size_256)
Q_A_hl, Q_B_hl, F_A_hl, F_B_hl = gemm_int8.quantize_scales_for_hybrid(
    A_s_hyb_l, W_s_hyb_l,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)
out_hybrid_large = gemm_int8.blockwise_fused_matmul_hybrid_large(
    A_q_hyb_l, W_q_hyb_l, Q_A_hl, Q_B_hl, F_A_hl, F_B_hl,
    quant_block_size=block_size_256, super_group_size=K_blocks_256)

# --- hybrid_large bq512 ---
A_q_hyb_l512, A_s_hyb_l512 = blockwise_quant_int8_ref(A_src, 128, block_size_512)
W_q_hyb_l512, W_s_hyb_l512 = blockwise_quant_int8_ref(W_src, 128, block_size_512)
Q_A_hl512, Q_B_hl512, F_A_hl512, F_B_hl512 = gemm_int8.quantize_scales_for_hybrid(
    A_s_hyb_l512, W_s_hyb_l512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512, Q_max=64)
out_hybrid_large_bq512 = gemm_int8.blockwise_fused_matmul_hybrid_large(
    A_q_hyb_l512, W_q_hyb_l512, Q_A_hl512, Q_B_hl512, F_A_hl512, F_B_hl512,
    quant_block_size=block_size_512, super_group_size=K_blocks_512)

# --- bq512 ---
A_q512, A_s512 = blockwise_quant_int8_ref(A_src, 128, block_size_512)
W_q512, W_s512 = blockwise_quant_int8_ref(W_src, 64, block_size_512)
Q_A_512, Q_B_512, F_A_512, F_B_512 = gemm_int8.quantize_scales_for_hybrid(
    A_s512, W_s512,
    quant_block_size=block_size_512, super_group_size=1, Q_max=8)
out_bq512 = gemm_int8.blockwise_fused_matmul_bq512(
    A_q512, W_q512, Q_A_512, Q_B_512, F_A_512, F_B_512,
    quant_block_size=block_size_512, super_group_size=1)

# --- bq512 fast_dequant ---
out_bq512_fast_dequant = gemm_int8.blockwise_fused_matmul_bq512_fast_dequant(
    A_q512, W_q512, Q_A_512, Q_B_512, F_A_512, F_B_512,
    quant_block_size=block_size_512, super_group_size=1)

# --- triton w8a8 bq128: block_size=[128,128], so A/W quantized with block_m=128, block_k=128 ---
A_q_tr128, A_s_tr128 = blockwise_quant_int8_ref(A_src, 128, 128)
W_q_tr128, W_s_tr128 = blockwise_quant_int8_ref(W_src, 128, 128)
out_triton_bq128 = triton_w8a8_matmul(
    A_q_tr128, W_q_tr128, A_s_tr128, W_s_tr128,
    block_size=[128, 128], output_dtype=torch.bfloat16, input_quant_mode="blockwise")

# --- triton w8a8 bq256: block_size=[128,256], so A/W quantized with block_m=128, block_k=256 ---
A_q_tr256, A_s_tr256 = blockwise_quant_int8_ref(A_src, 128, block_size_256)
W_q_tr256, W_s_tr256 = blockwise_quant_int8_ref(W_src, 128, block_size_256)
out_triton_bq256 = triton_w8a8_matmul(
    A_q_tr256, W_q_tr256, A_s_tr256, W_s_tr256,
    block_size=[128, 256], output_dtype=torch.bfloat16, input_quant_mode="blockwise")

print(f"\n{'='*80}")
print(f"  Accuracy vs BF16 reference  (M={M}, N={N}, K={K})")
print(f"{'='*80}")
print(f"{'Kernel':<22s} | {'Cos%':>8s} | {'RelL1':>8s} | {'RMSE':>10s} | {'MSE':>10s}")
print(f"{'-'*80}")

for tag, out in [
    ("bq128",              out_bq128),
    ("bq256",              out_bq256),
    ("bq256 fast_dequant", out_fast_dequant),
    ("hybrid bq256",       out_hybrid),
    ("hybrid bq512",       out_hybrid_bq512),
    ("hybrid_L bq256",     out_hybrid_large),
    ("hybrid_L bq512",     out_hybrid_large_bq512),
    ("hybrid_s bq256",     out_hybrid_small),
    ("hybrid_s bq512",     out_hybrid_small_bq512),
    ("bq512",              out_bq512),
    ("bq512 fast_dequant", out_bq512_fast_dequant),
    ("triton bq128",       out_triton_bq128),
    ("triton bq256",       out_triton_bq256),
]:
    cos, rel_l1, rmse, mse = compute_metrics(out, out_bf16)
    print(f"{tag:<22s} | {cos*100:8.4f} | {rel_l1:8.6f} | {rmse:10.6f} | {mse:10.6f}")
