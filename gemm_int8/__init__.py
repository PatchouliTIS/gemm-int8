import torch
import os
import glob

package_dir = os.path.dirname(os.path.abspath(__file__))

lib_pattern = os.path.join(package_dir, "gemm_int8_CUDA*.so")
lib_files = glob.glob(lib_pattern)
if not lib_files:
    raise ImportError(f"Could not find compiled CUDA extension in {package_dir}")

for lib_file in lib_files:
    torch.ops.load_library(lib_file)

BLOCK_QUANT_SIZE = 128


def _pad_k(x: torch.Tensor, align: int) -> torch.Tensor:
    K = x.shape[1]
    rem = K % align
    if rem == 0:
        return x
    pad_k = align - rem
    return torch.nn.functional.pad(x, (0, pad_k), value=0)


def _pad_k_scales(s: torch.Tensor, new_k_blocks: int) -> torch.Tensor:
    if s.shape[1] >= new_k_blocks:
        return s
    pad_cols = new_k_blocks - s.shape[1]
    return torch.nn.functional.pad(s, (0, pad_cols), value=0)


@torch.library.register_fake("gemm_int8_CUDA::int8_matmul")
def _(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    torch._check(y.device.type == "cuda", "y must be a CUDA tensor")
    torch._check(x.dtype == torch.int8, "x must be an int8 tensor")
    torch._check(y.dtype == torch.int8, "y must be an int8 tensor")
    torch._check(len(x.shape) == 2, "x must be a 2D tensor")
    torch._check(len(y.shape) == 2, "y must be a 2D tensor")
    torch._check(x.shape[1] == y.shape[1], "x.shape[1] must be equal to y.shape[1]")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul")
def _(x: torch.Tensor, y: torch.Tensor, x_scale: torch.Tensor, y_scale: torch.Tensor):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    torch._check(y.device.type == "cuda", "y must be a CUDA tensor")
    torch._check(x.dtype == torch.int8, "x must be an int8 tensor")
    torch._check(y.dtype == torch.int8, "y must be an int8 tensor")
    torch._check(x_scale.dtype == torch.float32, "x_scale must be float32")
    torch._check(y_scale.dtype == torch.float32, "y_scale must be float32")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_kk256")
def _(x: torch.Tensor, y: torch.Tensor, x_scale: torch.Tensor, y_scale: torch.Tensor):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    torch._check(y.device.type == "cuda", "y must be a CUDA tensor")
    torch._check(x.dtype == torch.int8, "x must be an int8 tensor")
    torch._check(y.dtype == torch.int8, "y must be an int8 tensor")
    torch._check(x_scale.dtype == torch.float32, "x_scale must be float32")
    torch._check(y_scale.dtype == torch.float32, "y_scale must be float32")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_fast_dequant")
def _(x: torch.Tensor, y: torch.Tensor, x_scale: torch.Tensor, y_scale: torch.Tensor):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    torch._check(y.device.type == "cuda", "y must be a CUDA tensor")
    torch._check(x.dtype == torch.int8, "x must be an int8 tensor")
    torch._check(y.dtype == torch.int8, "y must be an int8 tensor")
    torch._check(x_scale.dtype == torch.float32, "x_scale must be float32")
    torch._check(y_scale.dtype == torch.float32, "y_scale must be float32")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_hybrid")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_hybrid_large")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_hybrid_small")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_hybrid_bias")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor, bias: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_hybrid_large_bias")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor, bias: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_hybrid_small_bias")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor, bias: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_bq512")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_bq512_fast_dequant")
def _(x: torch.Tensor, y: torch.Tensor, Q_A: torch.Tensor, Q_B: torch.Tensor,
      F_A: torch.Tensor, F_B: torch.Tensor,
      quant_block_size: int, super_group_size: int):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


@torch.library.register_fake("gemm_int8_CUDA::int8_blockwise_fused_matmul_128x128")
def _(x: torch.Tensor, y: torch.Tensor, x_scale: torch.Tensor, y_scale: torch.Tensor):
    torch._check(x.device.type == "cuda", "x must be a CUDA tensor")
    torch._check(y.device.type == "cuda", "y must be a CUDA tensor")
    torch._check(x.dtype == torch.int8, "x must be an int8 tensor")
    torch._check(y.dtype == torch.int8, "y must be an int8 tensor")
    torch._check(x_scale.dtype == torch.float32, "x_scale must be float32")
    torch._check(y_scale.dtype == torch.float32, "y_scale must be float32")
    return torch.empty(x.shape[0], y.shape[0], device=x.device, dtype=torch.bfloat16)


def matmul(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Matrix-Matrix Multiplication for INT8 data type in the form of (x @ y.t())*alpha.
    The output is BF16 data type. todo: support arbitrary output dtype!
    Argumengs:
        x: torch.Tensor, shape (M, K)
        y: torch.Tensor, shape (K, N)
        alpha: float, which is multiplied by the output (default=1.0)
    """
    K_ALIGN = 128
    x = _pad_k(x, K_ALIGN)
    y = _pad_k(y, K_ALIGN)
    return torch.ops.gemm_int8_CUDA.int8_matmul(x, y, alpha)


def blockwise_fused_matmul(x: torch.Tensor, y: torch.Tensor,
                           x_scale: torch.Tensor, y_scale: torch.Tensor):
    """
    Fused mainloop blockwise-quantized INT8 matmul (Scheme A).

    Dequantization happens in-register after each K-block's MMA,
    eliminating all intermediate HBM traffic.

    x_q:     [M, K] INT8, pre-quantized
    y_q:     [N, K] INT8, pre-quantized
    x_scale: [ceil(M/128), ceil(K/128)] FP32
    y_scale: [ceil(N/128), ceil(K/128)] FP32

    Returns: [M, N] BF16
    """
    K_ALIGN = BLOCK_QUANT_SIZE
    K = x.shape[1]
    if K % K_ALIGN != 0:
        x = _pad_k(x, K_ALIGN)
        y = _pad_k(y, K_ALIGN)
        new_k_blocks = (x.shape[1] + K_ALIGN - 1) // K_ALIGN
        x_scale = _pad_k_scales(x_scale, new_k_blocks)
        y_scale = _pad_k_scales(y_scale, new_k_blocks)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul(x, y, x_scale, y_scale)


def blockwise_fused_matmul_kk256(x: torch.Tensor, y: torch.Tensor,
                                  x_scale: torch.Tensor, y_scale: torch.Tensor):
    K_ALIGN = 256
    K = x.shape[1]
    if K % K_ALIGN != 0:
        x = _pad_k(x, K_ALIGN)
        y = _pad_k(y, K_ALIGN)
        new_k_blocks = (x.shape[1] + K_ALIGN - 1) // K_ALIGN
        x_scale = _pad_k_scales(x_scale, new_k_blocks)
        y_scale = _pad_k_scales(y_scale, new_k_blocks)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_kk256(x, y, x_scale, y_scale)


def blockwise_fused_matmul_fast_dequant(x: torch.Tensor, y: torch.Tensor,
                                         x_scale: torch.Tensor, y_scale: torch.Tensor):
    K_ALIGN = 256
    K = x.shape[1]
    if K % K_ALIGN != 0:
        x = _pad_k(x, K_ALIGN)
        y = _pad_k(y, K_ALIGN)
        new_k_blocks = (x.shape[1] + K_ALIGN - 1) // K_ALIGN
        x_scale = _pad_k_scales(x_scale, new_k_blocks)
        y_scale = _pad_k_scales(y_scale, new_k_blocks)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_fast_dequant(x, y, x_scale, y_scale)


def blockwise_fused_matmul_128x128(x: torch.Tensor, y: torch.Tensor,
                                    x_scale: torch.Tensor, y_scale: torch.Tensor):
    K_ALIGN = BLOCK_QUANT_SIZE
    K = x.shape[1]
    if K % K_ALIGN != 0:
        x = _pad_k(x, K_ALIGN)
        y = _pad_k(y, K_ALIGN)
        new_k_blocks = (x.shape[1] + K_ALIGN - 1) // K_ALIGN
        x_scale = _pad_k_scales(x_scale, new_k_blocks)
        y_scale = _pad_k_scales(y_scale, new_k_blocks)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_128x128(x, y, x_scale, y_scale)


def quantize_scales_for_hybrid(input_scale: torch.Tensor,
                                weight_scale: torch.Tensor,
                                quant_block_size: int = 256,
                                super_group_size: int = 4,
                                Q_max: int = 8):
    """
    2nd-level scale quantization for hybrid kernel.
    Independently quantizes A and B scales per super-group.

    Input:
        input_scale:  [M_blocks, K_blocks] FP32
        weight_scale: [N_blocks, K_blocks] FP32
    Returns:
        Q_A: [M_blocks, K_blocks] INT32
        Q_B: [N_blocks, K_blocks] INT32
        F_A: [M_blocks, num_super_groups] FP32
        F_B: [N_blocks, num_super_groups] FP32
    """
    K_blocks = input_scale.shape[1]
    G = (K_blocks + super_group_size - 1) // super_group_size

    def _quantize_scale_matrix(scale, Q_max):
        num_rows = scale.shape[0]
        Q = torch.zeros_like(scale, dtype=torch.int32)
        F = torch.zeros(num_rows, G, dtype=torch.float32, device=scale.device)
        for g in range(G):
            start = g * super_group_size
            end = min(start + super_group_size, K_blocks)
            group = scale[:, start:end]
            S_max = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            F_g = S_max / Q_max
            F[:, g:g+1] = F_g
            Q[:, start:end] = torch.round(group / F_g).to(torch.int32)
        return Q, F

    Q_A, F_A = _quantize_scale_matrix(input_scale, Q_max)
    Q_B, F_B = _quantize_scale_matrix(weight_scale, Q_max)
    return Q_A, Q_B, F_A, F_B


def _pad_hybrid_k(x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size):
    K = x.shape[1]
    K_ALIGN = max(quant_block_size, 128)
    if K % K_ALIGN == 0:
        return x, y, Q_A, Q_B, F_A, F_B
    x = _pad_k(x, K_ALIGN)
    y = _pad_k(y, K_ALIGN)
    new_K = x.shape[1]
    new_k_blocks = (new_K + quant_block_size - 1) // quant_block_size
    Q_A = _pad_k_scales(Q_A, new_k_blocks)
    Q_B = _pad_k_scales(Q_B, new_k_blocks)
    new_super_groups = (new_k_blocks + super_group_size - 1) // super_group_size
    F_A = _pad_k_scales(F_A, new_super_groups)
    F_B = _pad_k_scales(F_B, new_super_groups)
    return x, y, Q_A, Q_B, F_A, F_B


def blockwise_fused_matmul_hybrid(x: torch.Tensor, y: torch.Tensor,
                                  Q_A: torch.Tensor, Q_B: torch.Tensor,
                                  F_A: torch.Tensor, F_B: torch.Tensor,
                                  quant_block_size: int = 256,
                                  super_group_size: int = 4):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_hybrid(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)


def blockwise_fused_matmul_hybrid_large(x: torch.Tensor, y: torch.Tensor,
                                        Q_A: torch.Tensor, Q_B: torch.Tensor,
                                        F_A: torch.Tensor, F_B: torch.Tensor,
                                        quant_block_size: int = 256,
                                        super_group_size: int = 4):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_hybrid_large(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)


def blockwise_fused_matmul_hybrid_small(x: torch.Tensor, y: torch.Tensor,
                                        Q_A: torch.Tensor, Q_B: torch.Tensor,
                                        F_A: torch.Tensor, F_B: torch.Tensor,
                                        quant_block_size: int = 256,
                                        super_group_size: int = 4):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_hybrid_small(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)


def blockwise_fused_matmul_hybrid_bias(x: torch.Tensor, y: torch.Tensor,
                                       Q_A: torch.Tensor, Q_B: torch.Tensor,
                                       F_A: torch.Tensor, F_B: torch.Tensor,
                                       bias: torch.Tensor,
                                       quant_block_size: int = 256,
                                       super_group_size: int = 4):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_hybrid_bias(
        x, y, Q_A, Q_B, F_A, F_B, bias, quant_block_size, super_group_size)


def blockwise_fused_matmul_hybrid_large_bias(x: torch.Tensor, y: torch.Tensor,
                                             Q_A: torch.Tensor, Q_B: torch.Tensor,
                                             F_A: torch.Tensor, F_B: torch.Tensor,
                                             bias: torch.Tensor,
                                             quant_block_size: int = 256,
                                             super_group_size: int = 4):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_hybrid_large_bias(
        x, y, Q_A, Q_B, F_A, F_B, bias, quant_block_size, super_group_size)


def blockwise_fused_matmul_hybrid_small_bias(x: torch.Tensor, y: torch.Tensor,
                                             Q_A: torch.Tensor, Q_B: torch.Tensor,
                                             F_A: torch.Tensor, F_B: torch.Tensor,
                                             bias: torch.Tensor,
                                             quant_block_size: int = 256,
                                             super_group_size: int = 4):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_hybrid_small_bias(
        x, y, Q_A, Q_B, F_A, F_B, bias, quant_block_size, super_group_size)


def blockwise_fused_matmul_bq512(x: torch.Tensor, y: torch.Tensor,
                                 Q_A: torch.Tensor, Q_B: torch.Tensor,
                                 F_A: torch.Tensor, F_B: torch.Tensor,
                                 quant_block_size: int = 512,
                                 super_group_size: int = 1):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_bq512(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)


def blockwise_fused_matmul_bq512_fast_dequant(x: torch.Tensor, y: torch.Tensor,
                                              Q_A: torch.Tensor, Q_B: torch.Tensor,
                                              F_A: torch.Tensor, F_B: torch.Tensor,
                                              quant_block_size: int = 512,
                                              super_group_size: int = 1):
    x, y, Q_A, Q_B, F_A, F_B = _pad_hybrid_k(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)
    return torch.ops.gemm_int8_CUDA.int8_blockwise_fused_matmul_bq512_fast_dequant(
        x, y, Q_A, Q_B, F_A, F_B, quant_block_size, super_group_size)


__all__ = ["matmul", "blockwise_fused_matmul",
           "blockwise_fused_matmul_kk256", "blockwise_fused_matmul_fast_dequant",
           "blockwise_fused_matmul_128x128", "blockwise_fused_matmul_hybrid",
           "blockwise_fused_matmul_hybrid_large",
           "blockwise_fused_matmul_hybrid_small",
           "blockwise_fused_matmul_hybrid_bias",
           "blockwise_fused_matmul_hybrid_large_bias",
           "blockwise_fused_matmul_hybrid_small_bias",
           "blockwise_fused_matmul_bq512", "blockwise_fused_matmul_bq512_fast_dequant",
           "quantize_scales_for_hybrid", "BLOCK_QUANT_SIZE"]
