#include <torch/extension.h>
#include <gemm.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <iostream>
#include <utility>

torch::Tensor int8_matmul(const torch::Tensor &A,
                          const torch::Tensor &B,
                          double alpha)
{
    float alpha_f = static_cast<float>(alpha);
    torch::checkAllContiguous("int8_matmul", {{A, "A", 0},
                                              {B, "B", 1}});
    torch::checkDeviceType("int8_matmul", {A, B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_matmul", {{A, "A", 0},
                                           {B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    return int8_matmul_host(A, B, C, alpha_f);
}

torch::Tensor int8_blockwise_fused_matmul(const torch::Tensor &A,
                                          const torch::Tensor &B,
                                          const torch::Tensor &A_scale,
                                          const torch::Tensor &B_scale)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul",
                              {{A, "A", 0}, {B, "B", 1},
                               {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    torch::checkDeviceType("int8_blockwise_fused_matmul", {A, B, A_scale, B_scale}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul",
                           {{A, "A", 0}, {B, "B", 1},
                            {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    return int8_blockwise_fused_matmul_host(A, B, A_scale, B_scale);
}

torch::Tensor int8_blockwise_fused_matmul_kk256(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &A_scale,
                                                const torch::Tensor &B_scale)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_kk256",
                              {{A, "A", 0}, {B, "B", 1},
                               {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_kk256", {A, B, A_scale, B_scale}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_kk256",
                           {{A, "A", 0}, {B, "B", 1},
                            {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    return int8_blockwise_fused_matmul_kk256_host(A, B, A_scale, B_scale);
}

torch::Tensor int8_blockwise_fused_matmul_fast_dequant(const torch::Tensor &A,
                                                      const torch::Tensor &B,
                                                      const torch::Tensor &A_scale,
                                                      const torch::Tensor &B_scale)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_fast_dequant",
                              {{A, "A", 0}, {B, "B", 1},
                               {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_fast_dequant", {A, B, A_scale, B_scale}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_fast_dequant",
                           {{A, "A", 0}, {B, "B", 1},
                            {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    return int8_blockwise_fused_matmul_fast_dequant_host(A, B, A_scale, B_scale);
}

torch::Tensor int8_blockwise_fused_matmul_hybrid(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &Q_A,
                                                const torch::Tensor &Q_B,
                                                const torch::Tensor &F_A,
                                                const torch::Tensor &F_B,
                                                int64_t quant_block_size,
                                                int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_hybrid",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_hybrid",
                           {A, B, Q_A, Q_B, F_A, F_B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_hybrid",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    return int8_blockwise_fused_matmul_hybrid_host(A, B, Q_A, Q_B, F_A, F_B,
                                                   quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_large(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &Q_A,
                                                const torch::Tensor &Q_B,
                                                const torch::Tensor &F_A,
                                                const torch::Tensor &F_B,
                                                int64_t quant_block_size,
                                                int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_hybrid_large",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_hybrid_large",
                           {A, B, Q_A, Q_B, F_A, F_B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_hybrid_large",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    return int8_blockwise_fused_matmul_hybrid_large_host(A, B, Q_A, Q_B, F_A, F_B,
                                                        quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_small(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &Q_A,
                                                const torch::Tensor &Q_B,
                                                const torch::Tensor &F_A,
                                                const torch::Tensor &F_B,
                                                int64_t quant_block_size,
                                                int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_hybrid_small",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_hybrid_small",
                           {A, B, Q_A, Q_B, F_A, F_B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_hybrid_small",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    return int8_blockwise_fused_matmul_hybrid_small_host(A, B, Q_A, Q_B, F_A, F_B,
                                                        quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_bias(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &Q_A,
                                                const torch::Tensor &Q_B,
                                                const torch::Tensor &F_A,
                                                const torch::Tensor &F_B,
                                                const torch::Tensor &bias,
                                                int64_t quant_block_size,
                                                int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_hybrid_bias",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5},
                               {bias, "bias", 6}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_hybrid_bias",
                           {A, B, Q_A, Q_B, F_A, F_B, bias}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_hybrid_bias",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5},
                            {bias, "bias", 6}});
    return int8_blockwise_fused_matmul_hybrid_bias_host(A, B, Q_A, Q_B, F_A, F_B, bias,
                                                        quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_large_bias(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &Q_A,
                                                const torch::Tensor &Q_B,
                                                const torch::Tensor &F_A,
                                                const torch::Tensor &F_B,
                                                const torch::Tensor &bias,
                                                int64_t quant_block_size,
                                                int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_hybrid_large_bias",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5},
                               {bias, "bias", 6}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_hybrid_large_bias",
                           {A, B, Q_A, Q_B, F_A, F_B, bias}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_hybrid_large_bias",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5},
                            {bias, "bias", 6}});
    return int8_blockwise_fused_matmul_hybrid_large_bias_host(A, B, Q_A, Q_B, F_A, F_B, bias,
                                                              quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_small_bias(const torch::Tensor &A,
                                                const torch::Tensor &B,
                                                const torch::Tensor &Q_A,
                                                const torch::Tensor &Q_B,
                                                const torch::Tensor &F_A,
                                                const torch::Tensor &F_B,
                                                const torch::Tensor &bias,
                                                int64_t quant_block_size,
                                                int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_hybrid_small_bias",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5},
                               {bias, "bias", 6}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_hybrid_small_bias",
                           {A, B, Q_A, Q_B, F_A, F_B, bias}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_hybrid_small_bias",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5},
                            {bias, "bias", 6}});
    return int8_blockwise_fused_matmul_hybrid_small_bias_host(A, B, Q_A, Q_B, F_A, F_B, bias,
                                                              quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_bq512(const torch::Tensor &A,
                                               const torch::Tensor &B,
                                               const torch::Tensor &Q_A,
                                               const torch::Tensor &Q_B,
                                               const torch::Tensor &F_A,
                                               const torch::Tensor &F_B,
                                               int64_t quant_block_size,
                                               int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_bq512",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_bq512",
                           {A, B, Q_A, Q_B, F_A, F_B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_bq512",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    return int8_blockwise_fused_matmul_bq512_host(A, B, Q_A, Q_B, F_A, F_B,
                                                  quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_bq512_fast_dequant(const torch::Tensor &A,
                                               const torch::Tensor &B,
                                               const torch::Tensor &Q_A,
                                               const torch::Tensor &Q_B,
                                               const torch::Tensor &F_A,
                                               const torch::Tensor &F_B,
                                               int64_t quant_block_size,
                                               int64_t super_group_size)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_bq512_fast_dequant",
                              {{A, "A", 0}, {B, "B", 1},
                               {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                               {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_bq512_fast_dequant",
                           {A, B, Q_A, Q_B, F_A, F_B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_bq512_fast_dequant",
                           {{A, "A", 0}, {B, "B", 1},
                            {Q_A, "Q_A", 2}, {Q_B, "Q_B", 3},
                            {F_A, "F_A", 4}, {F_B, "F_B", 5}});
    return int8_blockwise_fused_matmul_bq512_fast_dequant_host(A, B, Q_A, Q_B, F_A, F_B,
                                                              quant_block_size, super_group_size);
}

torch::Tensor int8_blockwise_fused_matmul_128x128(const torch::Tensor &A,
                                                  const torch::Tensor &B,
                                                  const torch::Tensor &A_scale,
                                                  const torch::Tensor &B_scale)
{
    torch::checkAllContiguous("int8_blockwise_fused_matmul_128x128",
                              {{A, "A", 0}, {B, "B", 1},
                               {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    torch::checkDeviceType("int8_blockwise_fused_matmul_128x128", {A, B, A_scale, B_scale}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("int8_blockwise_fused_matmul_128x128",
                           {{A, "A", 0}, {B, "B", 1},
                            {A_scale, "A_scale", 2}, {B_scale, "B_scale", 3}});
    return int8_blockwise_fused_matmul_128x128_host(A, B, A_scale, B_scale);
}

//====== pybind ======

TORCH_LIBRARY(gemm_int8_CUDA, m)
{
    m.def("int8_matmul(Tensor A, Tensor B, float alpha) -> Tensor");
    m.def("int8_blockwise_fused_matmul(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) -> Tensor");
    m.def("int8_blockwise_fused_matmul_kk256(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) -> Tensor");
    m.def("int8_blockwise_fused_matmul_fast_dequant(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) -> Tensor");
    m.def("int8_blockwise_fused_matmul_hybrid(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_hybrid_large(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_hybrid_small(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_hybrid_bias(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, Tensor bias, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_hybrid_large_bias(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, Tensor bias, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_hybrid_small_bias(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, Tensor bias, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_bq512(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_bq512_fast_dequant(Tensor A, Tensor B, Tensor Q_A, Tensor Q_B, Tensor F_A, Tensor F_B, int quant_block_size, int super_group_size) -> Tensor");
    m.def("int8_blockwise_fused_matmul_128x128(Tensor A, Tensor B, Tensor A_scale, Tensor B_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(gemm_int8_CUDA, CUDA, m)
{
    m.impl("int8_matmul", &int8_matmul);
    m.impl("int8_blockwise_fused_matmul", &int8_blockwise_fused_matmul);
    m.impl("int8_blockwise_fused_matmul_kk256", &int8_blockwise_fused_matmul_kk256);
    m.impl("int8_blockwise_fused_matmul_fast_dequant", &int8_blockwise_fused_matmul_fast_dequant);
    m.impl("int8_blockwise_fused_matmul_hybrid", &int8_blockwise_fused_matmul_hybrid);
    m.impl("int8_blockwise_fused_matmul_hybrid_large", &int8_blockwise_fused_matmul_hybrid_large);
    m.impl("int8_blockwise_fused_matmul_hybrid_small", &int8_blockwise_fused_matmul_hybrid_small);
    m.impl("int8_blockwise_fused_matmul_hybrid_bias", &int8_blockwise_fused_matmul_hybrid_bias);
    m.impl("int8_blockwise_fused_matmul_hybrid_large_bias", &int8_blockwise_fused_matmul_hybrid_large_bias);
    m.impl("int8_blockwise_fused_matmul_hybrid_small_bias", &int8_blockwise_fused_matmul_hybrid_small_bias);
    m.impl("int8_blockwise_fused_matmul_bq512", &int8_blockwise_fused_matmul_bq512);
    m.impl("int8_blockwise_fused_matmul_bq512_fast_dequant", &int8_blockwise_fused_matmul_bq512_fast_dequant);
    m.impl("int8_blockwise_fused_matmul_128x128", &int8_blockwise_fused_matmul_128x128);
}
