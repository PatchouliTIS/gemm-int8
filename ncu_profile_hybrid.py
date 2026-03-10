import torch
import math
import gemm_int8

M, K, N = 32768, 2560, 2048
block_size_256 = 256
device = torch.device('cuda')

A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
B_int8_t = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=device)

K_blocks_256 = math.ceil(K / block_size_256)
M_blocks_hyb = math.ceil(M / 128)
N_blocks_hyb = math.ceil(N / 64)
scale_a_hyb = torch.rand(M_blocks_hyb, K_blocks_256, dtype=torch.float32, device=device) * 0.1
scale_b_hyb = torch.rand(N_blocks_hyb, K_blocks_256, dtype=torch.float32, device=device) * 0.1
Q_A, Q_B, F_A, F_B = gemm_int8.quantize_scales_for_hybrid(
    scale_a_hyb, scale_b_hyb,
    quant_block_size=block_size_256, super_group_size=K_blocks_256, Q_max=64)

# Warmup
for _ in range(3):
    out = gemm_int8.blockwise_fused_matmul_hybrid(
        A_int8, B_int8_t, Q_A, Q_B, F_A, F_B,
        quant_block_size=block_size_256, super_group_size=K_blocks_256)
torch.cuda.synchronize()

# Single launch for ncu
out = gemm_int8.blockwise_fused_matmul_hybrid(
    A_int8, B_int8_t, Q_A, Q_B, F_A, F_B,
    quant_block_size=block_size_256, super_group_size=K_blocks_256)
torch.cuda.synchronize()
