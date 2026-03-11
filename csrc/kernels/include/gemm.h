#pragma once
#include <common.h>
#include <torch/types.h>


torch::Tensor int8_matmul_host(torch::Tensor input,  // INT8
                                  torch::Tensor weight, // INT8
                                  torch::Tensor out,   // BF16
                                  float alpha          // FP32
);

torch::Tensor int8_blockwise_fused_matmul_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor input_scale,   // [M_blocks, K_blocks] FP32
    torch::Tensor weight_scale   // [N_blocks, K_blocks] FP32
);

torch::Tensor int8_blockwise_fused_matmul_kk256_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor input_scale,   // [M_blocks, K_blocks] FP32
    torch::Tensor weight_scale   // [N_blocks, K_blocks] FP32
);

torch::Tensor int8_blockwise_fused_matmul_fast_dequant_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor input_scale,   // [M_blocks, K_blocks] FP32
    torch::Tensor weight_scale   // [N_blocks, K_blocks] FP32
);

torch::Tensor int8_blockwise_fused_matmul_hybrid_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor Q_A,           // [M_blocks, num_quant_blocks] INT32
    torch::Tensor Q_B,           // [N_blocks, num_quant_blocks] INT32
    torch::Tensor F_A,           // [M_blocks, num_super_groups] FP32
    torch::Tensor F_B,           // [N_blocks, num_super_groups] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_hybrid_bias_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    torch::Tensor bias,          // [N] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_hybrid_large_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor Q_A,           // [M_blocks, num_quant_blocks] INT32
    torch::Tensor Q_B,           // [N_blocks, num_quant_blocks] INT32
    torch::Tensor F_A,           // [M_blocks, num_super_groups] FP32
    torch::Tensor F_B,           // [N_blocks, num_super_groups] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_hybrid_large_bias_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    torch::Tensor bias,          // [N] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_hybrid_small_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor Q_A,           // [M_blocks, num_quant_blocks] INT32
    torch::Tensor Q_B,           // [N_blocks, num_quant_blocks] INT32
    torch::Tensor F_A,           // [M_blocks, num_super_groups] FP32
    torch::Tensor F_B,           // [N_blocks, num_super_groups] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_hybrid_small_bias_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    torch::Tensor bias,          // [N] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_bq512_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor Q_A,           // [M_blocks, num_quant_blocks] INT32
    torch::Tensor Q_B,           // [N_blocks, num_quant_blocks] INT32
    torch::Tensor F_A,           // [M_blocks, num_super_groups] FP32
    torch::Tensor F_B,           // [N_blocks, num_super_groups] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_bq512_fast_dequant_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor Q_A,           // [M_blocks, num_quant_blocks] INT32
    torch::Tensor Q_B,           // [N_blocks, num_quant_blocks] INT32
    torch::Tensor F_A,           // [M_blocks, num_super_groups] FP32
    torch::Tensor F_B,           // [N_blocks, num_super_groups] FP32
    int64_t quant_block_size,
    int64_t super_group_size
);

torch::Tensor int8_blockwise_fused_matmul_128x128_host(
    torch::Tensor input_q,       // [M, K] INT8
    torch::Tensor weight_q,      // [N, K] INT8
    torch::Tensor input_scale,   // [M_blocks, K_blocks] FP32
    torch::Tensor weight_scale   // [N_blocks, K_blocks] FP32
);

