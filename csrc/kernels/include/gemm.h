#pragma once
#include <common.h>
#include <torch/types.h>


torch::Tensor int8_matmul_host(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor out,   // BF16
                                  float alpha          // FP32
);

torch::Tensor int8_matmul_blockwise_scaled_host(
    torch::Tensor input,      // INT8 [M, K]
    torch::Tensor weight,     // INT8 [N, K]
    torch::Tensor scale_a,    // FP32 [M/128, K/128]
    torch::Tensor scale_b,    // FP32 [N/128, K/128]
    torch::Tensor out         // BF16 [M, N]
);
