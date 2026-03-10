# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/blob/4cb53ecd0cffceb6dee5c011a58f65997a86f151/python/sglang/srt/layers/quantization/int8_kernel.py
import functools
import json
import logging
import os
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)


def cutlass_block_int8_supported() -> bool:
    # return False
    """Check if CUTLASS INT8 blockwise GEMM is available (SM80+)."""
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    if capability_tuple is None:
        return False
    capability = capability_tuple.to_int()
    # SM80 (Ampere) and above support INT8 tensor cores
    return capability >= 80


CUTLASS_BLOCK_INT8_SUPPORTED = cutlass_block_int8_supported()


class W8A8BlockInt8LinearOp:
    """
    Block-wise INT8 W8A8 Linear operator.

    Encapsulates the CUTLASS int8_blockwise_scaled_mm kernel (primary)
    and the Triton w8a8_block_int8_matmul kernel (fallback).

    The CUTLASS kernel expects:
      - A: [M, K] int8, RowMajor, contiguous
      - B: [K, N] int8, ColumnMajor (stride(0)==1)
      - scale_a: [M, ceil(K/group_size)] float32 (per-token-group)
                 or [ceil(M/128), ceil(K/128)] float32 (blockwise)
      - scale_b: [ceil(N/128), ceil(K/128)] float32
      - output: [M, N] bfloat16

    The Triton kernel expects:
      - A: [M, K] int8, RowMajor
      - B: [N, K] int8, RowMajor (standard weight layout)
      - scale_a: [M, ceil(K/group_size)] float32 (per_token_group) or
                 [ceil(M/block_m), ceil(K/block_k)] float32 (blockwise)
      - scale_b: [ceil(N/block_n), ceil(K/block_k)] float32

    input_quant_mode controls how activations are quantized:
      - "per_token_group" (default): each token independently, scale shape
        [M, ceil(K/group_size)]. Higher precision.
      - "blockwise": 2D block quantization, scale shape
        [ceil(M/128), ceil(K/128)]. Lower precision but potentially faster.
    """

    def __init__(
        self,
        weight_block_size: list[int],
        use_cutlass: bool = CUTLASS_BLOCK_INT8_SUPPORTED,
        input_quant_mode: str = "per_token_group",
    ):
        assert input_quant_mode in ("per_token_group", "blockwise"), (
            f"input_quant_mode must be 'per_token_group' or 'blockwise', "
            f"got '{input_quant_mode}'"
        )
        self.weight_block_size = weight_block_size
        self.use_cutlass = use_cutlass
        self.input_quant_mode = input_quant_mode

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform block-wise INT8 W8A8 linear.

        Args:
            input: [..., K] input tensor (bf16/fp16)
            weight: [K, N] int8 ColumnMajor (stride(0)==1, stride(1)==K)
            weight_scale: [ceil(N/block_n), ceil(K/block_k)] float32
            input_scale: unused, must be None
            bias: optional [N] bias tensor
        """
        assert input_scale is None
        input_2d = input.contiguous().view(-1, input.shape[-1])
        # Weight is stored as [K, N] ColumnMajor (stride(0)==1),
        # so shape[0]==K, shape[1]==N.
        N = weight.shape[1]
        output_shape = [*input.shape[:-1], N]

        if self.use_cutlass:
            output = self._run_cutlass(input_2d, weight, weight_scale, bias)
        else:
            output = self._run_triton(input_2d, weight, weight_scale)
            if bias is not None:
                output = output + bias

        return output.to(dtype=input.dtype).view(*output_shape)

    def _run_cutlass(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        CUTLASS path: quantize input, then call int8_blockwise_scaled_mm.

        weight must be [K, N] ColumnMajor with stride(0)==1.
        When input K is not divisible by group_size, per_token_group_quant_int8
        will pad q_input to [M, padded_K]. We also pad weight's K dimension
        to match. The CUTLASS C++ kernel has its own internal padding logic
        for M/N/K alignment.
        """
        group_size = self.weight_block_size[1]
        use_blockwise_input = (self.input_quant_mode == "blockwise")

        if use_blockwise_input:
            block_m = self.weight_block_size[0]
            block_k = self.weight_block_size[1]
            q_input, input_scale = blockwise_quant_int8(
                input_2d, block_m, block_k
            )
        else:
            # q_input may be padded to [M, padded_K] if K % group_size != 0
            q_input, input_scale = per_token_group_quant_int8(
                input_2d, group_size
            )

        padded_k = q_input.shape[-1]
        orig_k = weight.shape[0]
        if padded_k != orig_k:
            # Weight is [K, N] ColumnMajor (stride(0)==1), which is
            # a transposed view of an [N, K] contiguous (RowMajor) tensor.
            # To pad K: transpose to [N, K] contiguous, pad last dim,
            # then transpose back to get [padded_K, N] ColumnMajor.
            weight_nk = weight.t().contiguous()  # [N, K] RowMajor
            weight_nk = torch.nn.functional.pad(
                weight_nk, (0, padded_k - orig_k), value=0
            )  # [N, padded_K] RowMajor
            weight = weight_nk.t()  # [padded_K, N] ColumnMajor (stride(0)==1)
        # logger.debug(
        #     "CUTLASS int8_blockwise_scaled_mm inputs: "
        #     "q_input=%s (stride=%s), weight=%s (stride=%s), "
        #     "input_scale=%s, weight_scale=%s, group_size=%d, "
        #     "per_token_group=%s",
        #     q_input.shape, q_input.stride(),
        #     weight.shape, weight.stride(),
        #     input_scale.shape, weight_scale.shape,
        #     group_size, not use_blockwise_input,
        # )
        output = ops.int8_blockwise_scaled_mm(
            a=q_input,
            b=weight,
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
            bias=bias,
            per_token_group=not use_blockwise_input,
            input_group_size=group_size,
        )
        return output

    def _run_triton(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triton fallback path.

        weight is [K, N] ColumnMajor (stride(0)==1).
        Triton kernel expects [N, K] RowMajor, so we transpose
        and make contiguous.
        When input K is padded, weight K is also padded to match.
        """
        block_m, group_size = self.weight_block_size
        if self.input_quant_mode == "blockwise":
            q_input, input_scale = blockwise_quant_int8(
                input_2d, block_m, group_size
            )
        else:
            # q_input may be padded to [M, padded_K] if K % group_size != 0
            q_input, input_scale = per_token_group_quant_int8(
                input_2d, group_size
            )

        # [K, N] ColMajor -> .t() -> [N, K] -> .contiguous() -> [N, K] RowMajor
        weight_rowmajor = weight.t().contiguous()
        padded_k = q_input.shape[-1]
        orig_k = weight_rowmajor.shape[-1]
        if padded_k != orig_k:
            # Pad weight from [N, K] to [N, padded_K] RowMajor
            weight_rowmajor = torch.nn.functional.pad(
                weight_rowmajor, (0, padded_k - orig_k), value=0
            )
        return w8a8_block_int8_matmul(
            q_input,
            weight_rowmajor,
            input_scale,
            weight_scale,
            self.weight_block_size,
            output_dtype=input_2d.dtype,
            input_quant_mode=self.input_quant_mode,
        )


def apply_w8a8_block_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: list[int],
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for quantization
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    # q_input may be padded to [M, padded_K] if K % group_size != 0
    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    padded_k = q_input.shape[-1]
    orig_k = weight.shape[-1]
    if padded_k != orig_k:
        # Pad weight from [N, K] to [N, padded_K] (RowMajor for Triton)
        weight = torch.nn.functional.pad(
            weight, (0, padded_k - orig_k), value=0
        )
    output = w8a8_block_int8_matmul(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input.dtype
    )

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)


def input_to_int8(
    x: torch.Tensor, dtype: torch.dtype = torch.int8
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to int8 values with
    tensor-wise quantization."""
    iinfo = torch.iinfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    int8_min, int8_max = iinfo.min, iinfo.max
    scale = int8_max / amax
    x_scl_sat = (x * scale).clamp(min=int8_min, max=int8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """This function conducts block-wise dequantization.
    The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are dequantized tensor.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ] *= x_s[j][i]

    return x_dq_block


if current_platform.is_rocm():

    @triton.jit
    def round_int8(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int8)

else:

    @triton.jit
    def round_int8(x):
        return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = round_int8(x_q)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, original_shape[-1])
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    x = x.contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    x_q = x_q.view(*original_shape)
    scales = scales.view(*original_shape[:-1], 1)
    return x_q, scales


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid to divide zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    When the last dimension of `x` is not divisible by `group_size`, the input
    is automatically padded with zeros to the next multiple of `group_size`.
    The returned quantized tensor retains the padded dimension so that it can
    be directly used in subsequent GEMM kernels.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8`
            is supported for now.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor (possibly
            padded) and the scaling factor for quantization.
    """
    assert x.is_contiguous(), "`x` is not contiguous"

    orig_k = x.shape[-1]
    needs_pad = (orig_k % group_size != 0)
    if needs_pad:
        padded_k = ((orig_k + group_size - 1) // group_size) * group_size
        x = torch.nn.functional.pad(
            x, (0, padded_k - orig_k), value=0.0
        ).contiguous()

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    # prefer CUDA kernel if available
    if current_platform.is_cuda():
        torch.ops._C.per_token_group_quant_int8(
            x, x_q, x_s, group_size, eps, float(int8_min), float(int8_max)
        )
        return x_q, x_s

    M = x.numel() // group_size
    N = group_size

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


@triton.jit
def _blockwise_quant_int8(
    x_ptr,
    xq_ptr,
    xs_ptr,
    stride_xm,
    stride_xk,
    stride_xqm,
    stride_xqk,
    stride_xsm,
    stride_xsk,
    M,
    K,
    K_PADDED,
    eps,
    int8_min,
    int8_max,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    in_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    in_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    x_block = tl.load(in_ptrs, mask=in_mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.max(tl.abs(x_block), axis=1), axis=0)
    absmax = tl.maximum(absmax, eps)
    scale = absmax / int8_max

    q_block = round_int8(tl.clamp(x_block / scale, int8_min, int8_max))

    out_ptrs = xq_ptr + offs_m[:, None] * stride_xqm + offs_k[None, :] * stride_xqk
    out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_PADDED)
    tl.store(out_ptrs, q_block, mask=out_mask)

    scale_ptr = xs_ptr + pid_m * stride_xsm + pid_k * stride_xsk
    tl.store(scale_ptr, scale)


def blockwise_quant_int8(
    x: torch.Tensor,
    block_m: int,
    block_k: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform 2D block-wise quantization on an input tensor `x`.

    Unlike per_token_group_quant_int8 which quantizes each token independently
    (block_m=1), this function groups tokens in blocks of `block_m` along the
    M dimension and `block_k` along the K dimension. Tokens within the same
    M-block share scales along each K-block.

    Args:
        x: [M, K] input tensor (bf16/fp16/fp32), must be contiguous.
        block_m: Block size along the M (token) dimension.
        block_k: Block size along the K (hidden) dimension.
        eps: Minimum value to avoid dividing by zero.
        dtype: Output dtype (torch.int8).

    Returns:
        x_q: [M, K_padded] int8 quantized tensor, where
             K_padded = ceil(K/block_k)*block_k.
        x_s: [ceil(M/block_m), ceil(K/block_k)] float32 scales.
    """
    assert x.ndim == 2 and x.is_contiguous(), (
        "`x` must be a 2D contiguous tensor"
    )
    assert dtype == torch.int8, "Only torch.int8 is supported"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max

    M, K = x.shape
    m_blocks = (M + block_m - 1) // block_m
    k_blocks = (K + block_k - 1) // block_k
    K_padded = k_blocks * block_k

    x_q = torch.empty((M, K_padded), device=x.device, dtype=dtype)
    x_s = torch.empty((m_blocks, k_blocks), device=x.device, dtype=torch.float32)

    if x.is_cuda:
        num_warps = 4 if block_m * block_k <= 4096 else 8
        _blockwise_quant_int8[(m_blocks, k_blocks)](
            x,
            x_q,
            x_s,
            x.stride(0),
            x.stride(1),
            x_q.stride(0),
            x_q.stride(1),
            x_s.stride(0),
            x_s.stride(1),
            M,
            K,
            K_padded,
            eps,
            float(iinfo.min),
            float(int8_max),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
        return x_q, x_s

    M_padded = m_blocks * block_m

    # CPU fallback
    needs_pad = (M_padded != M) or (K_padded != K)
    if needs_pad:
        x_padded = torch.zeros(
            M_padded, K_padded, device=x.device, dtype=x.dtype
        )
        x_padded[:M, :K] = x
    else:
        x_padded = x

    x_reshaped = x_padded.view(m_blocks, block_m, k_blocks, block_k)
    absmax = x_reshaped.float().abs().amax(dim=(1, 3)).clamp(min=eps)
    x_s = absmax / int8_max
    scale_expanded = x_s[:, None, :, None]
    x_q_full = (x_reshaped.float() / scale_expanded).round().clamp(
        iinfo.min, int8_max
    ).to(dtype)
    x_q = x_q_full.view(M_padded, K_padded)[:M, :].contiguous()

    return x_q, x_s


@triton.jit
def _w8a8_block_int8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    input_block_m,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_SCALE_BLOCKWISE: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if A_SCALE_BLOCKWISE:
        offs_asm = offs_am // input_block_m
    else:
        offs_asm = offs_am
    As_ptrs = As + offs_asm * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b).to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def _w8a8_block_int8_matmul_no_scales(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    input_block_m,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_SCALE_BLOCKWISE: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if A_SCALE_BLOCKWISE:
        offs_asm = offs_am // input_block_m
    else:
        offs_asm = offs_am
    As_ptrs = As + offs_asm * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b).to(tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _int8_gemm_no_scales(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Pure INT8 GEMM kernel: only tl.dot, no scale loading or multiplication."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b).to(tl.float32)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _int8_gemm_raw_acc(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Pure INT8 GEMM kernel with raw int32 accumulator, no type conversion."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@functools.lru_cache
@torch._dynamo.disable
def get_w8a8_block_int8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=int8_w8a8,block_shape=[{block_n},{block_k}].json"  # noqa: E501

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block INT8 kernel.",
                config_file_path,
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        (
            "Using default W8A8 Block INT8 kernel config. Performance might "
            "be sub-optimal! Config file not found at %s"
        ),
        config_file_path,
    )
    return None


def w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
    input_quant_mode: str = "per_token_group",
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: Quantization scale for `A`.
            - per_token_group: [M, ceil(K/block_k)]
            - blockwise: [ceil(M/block_m), ceil(K/block_k)]
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be
            2-dim, e.g., [128, 128].
        output_dtype: The dtype of the returned tensor.
        input_quant_mode: "per_token_group" or "blockwise".

    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert input_quant_mode in ("per_token_group", "blockwise")
    use_blockwise_input = (input_quant_mode == "blockwise")

    assert A.shape[-1] == B.shape[-1]
    assert A.is_contiguous()
    M = A.numel() // A.shape[-1]
    if use_blockwise_input:
        assert As.ndim == 2
        assert As.shape[0] == triton.cdiv(M, block_n)
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[1]
    else:
        assert A.shape[:-1] == As.shape[:-1]
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    if torch.compiler.is_compiling():
        configs = None
    else:
        configs = get_w8a8_block_int8_configs(N, K, block_size[0], block_size[1])
    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_K must be divisible by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_int8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        block_n,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        A_SCALE_BLOCKWISE=use_blockwise_input,
        **config,
    )

    return C

