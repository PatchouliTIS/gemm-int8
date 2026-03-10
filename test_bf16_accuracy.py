import torch
import math
import sys
sys.path.insert(0, "/deploy/image_caption/gemm-int8")
import gemm_int8

BLOCK = 128


def blockwise_quant_int8(x: torch.Tensor, block_m: int, block_k: int):
    M, K = x.shape
    m_blocks = math.ceil(M / block_m)
    k_blocks = math.ceil(K / block_k)
    x_q = torch.zeros_like(x, dtype=torch.int8)
    x_s = torch.zeros(m_blocks, k_blocks, dtype=torch.float32, device=x.device)
    for i in range(m_blocks):
        for j in range(k_blocks):
            r0, r1 = i * block_m, min((i + 1) * block_m, M)
            c0, c1 = j * block_k, min((j + 1) * block_k, K)
            block = x[r0:r1, c0:c1].float()
            absmax = block.abs().max().clamp(min=1e-10)
            scale = absmax / 127.0
            x_s[i, j] = scale
            x_q[r0:r1, c0:c1] = (block / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, x_s


def compute_metrics(out, ref):
    o, r = out.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    rel_l1 = (o - r).abs().sum().item() / r.abs().sum().clamp(min=1e-12).item()
    rmse = ((o - r) ** 2).mean().sqrt().item()
    mse = ((o - r) ** 2).mean().item()
    return cos, rel_l1, rmse, mse


def print_header(title):
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"{'Shape':>22s} | {'Cos%':>8s} | {'RelL1':>8s} | {'RMSE':>10s} | {'MSE':>10s}")
    print("-" * 80)


def print_row(M, N, K, cos, rel_l1, rmse, mse):
    shape = f"({M},{N},{K})"
    print(f"{shape:>22s} | {cos*100:8.4f} | {rel_l1:8.6f} | {rmse:10.6f} | {mse:10.6f}")


torch.manual_seed(42)
device = "cuda"

cases = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (8192, 2048, 2560),
    (8192, 2560, 2048),
]


# ---- BQ128: quant(128,128), kernel CTA 128x64 ----
print_header("BQ128 fused vs BF16 torch.mm")
for M, N, K in cases:
    a_fp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_fp = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bf16_ref = (a_fp @ w_fp.T).float()

    a_q, a_s = blockwise_quant_int8(a_fp, BLOCK, BLOCK)
    w_q, w_s = blockwise_quant_int8(w_fp, BLOCK, BLOCK)
    fused = gemm_int8.blockwise_fused_matmul(a_q, w_q, a_s, w_s).float()

    cos, rel_l1, rmse, mse = compute_metrics(fused, bf16_ref)
    print_row(M, N, K, cos, rel_l1, rmse, mse)
print()


# ---- BQ256: quant(256,256), kernel CTA 128x64 ----
print_header("BQ256 fused vs BF16 torch.mm")
for M, N, K in cases:
    if K % 256 != 0:
        continue
    a_fp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_fp = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bf16_ref = (a_fp @ w_fp.T).float()

    a_q, a_s = blockwise_quant_int8(a_fp, 256, 256)
    w_q, w_s = blockwise_quant_int8(w_fp, 256, 256)
    fused = gemm_int8.blockwise_fused_matmul_kk256(a_q, w_q, a_s, w_s).float()

    cos, rel_l1, rmse, mse = compute_metrics(fused, bf16_ref)
    print_row(M, N, K, cos, rel_l1, rmse, mse)
print()


# ---- BQ512: quant(128x512 for A, 64x512 for B) ----
print_header("BQ512 2-accum fused vs BF16 torch.mm")
for M, N, K in cases:
    if K % 512 != 0:
        continue
    a_fp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_fp = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bf16_ref = (a_fp @ w_fp.T).float()

    a_q, a_s = blockwise_quant_int8(a_fp, 128, 512)
    w_q, w_s = blockwise_quant_int8(w_fp, 64, 512)
    Q_A, Q_B, F_A, F_B = gemm_int8.quantize_scales_for_hybrid(
        a_s, w_s, quant_block_size=512, super_group_size=1, Q_max=8)
    fused = gemm_int8.blockwise_fused_matmul_bq512(
        a_q, w_q, Q_A, Q_B, F_A, F_B, quant_block_size=512, super_group_size=1).float()

    cos, rel_l1, rmse, mse = compute_metrics(fused, bf16_ref)
    print_row(M, N, K, cos, rel_l1, rmse, mse)
print()
