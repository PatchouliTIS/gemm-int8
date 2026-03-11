#include <gemm.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h>
#include <cutlass/epilogue/threadblock/shared_load_iterator.h>
#include <cutlass/epilogue/threadblock/shared_load_iterator_mixed.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include "custom_mma_multistage.h"

static constexpr int kBlockQuantSize = 128;

// ============================================================
// Config 1: kK=128, stg=3 (current best)
// ============================================================

using ElementA          = int8_t;
using ElementB          = int8_t;
using ElementAccum      = int32_t;
using ElementOutput     = cutlass::bfloat16_t;
using LayoutA           = cutlass::layout::RowMajor;
using LayoutB           = cutlass::layout::ColumnMajor;
using LayoutOutput      = cutlass::layout::RowMajor;

using TileShape         = cutlass::gemm::GemmShape<128, 64, 128>;
using WarpShape         = cutlass::gemm::GemmShape<64, 32, 128>;
using InstructionShape  = cutlass::gemm::GemmShape<16, 8, 32>;
static constexpr int kStages = 3;
static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

static_assert(TileShape::kK == kBlockQuantSize,
    "kK must equal kBlockQuantSize for direct alignment");
static constexpr int kTilesPerQuantBlock = 1;

using DefaultMma = cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccum, LayoutOutput,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TileShape, WarpShape, InstructionShape,
    kStages,
    cutlass::arch::OpMultiplyAddSaturate,
    false,
    cutlass::gemm::SharedMemoryClearOption::kNone>;

using _OrigMma    = typename DefaultMma::ThreadblockMma;
using IteratorA   = typename DefaultMma::IteratorA;
using IteratorB   = typename DefaultMma::IteratorB;

// Use custom MMA fork to avoid modifying CUTLASS source
using Mma = custom_mma::MmaMultistage<
    typename DefaultMma::MmaCore::Shape,
    IteratorA, typename DefaultMma::MmaCore::SmemIteratorA,
    DefaultMma::MmaCore::kCacheOpA,
    IteratorB, typename DefaultMma::MmaCore::SmemIteratorB,
    DefaultMma::MmaCore::kCacheOpB,
    ElementAccum, LayoutOutput,
    typename DefaultMma::MmaCore::MmaPolicy, kStages>;
using FragmentC   = typename Mma::FragmentC;

using WarpMmaOperator = typename DefaultMma::MmaCore::MmaPolicy::Operator;
using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
using OperatorShape = typename ArchMmaOperator::Shape;
using OperatorFragmentC = typename ArchMmaOperator::FragmentC;

static constexpr int kEpilogueElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    kEpilogueElementsPerAccess,
    float,
    float>;

using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    TileShape, WarpShape, 1, ElementOutput, kEpilogueElementsPerAccess
>::Type;

using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap, ElementOutput>;

using AccumulatorFragmentIterator = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShape,
    OperatorShape,
    float,
    cutlass::Array<float, OperatorFragmentC::kElements>,
    cutlass::layout::RowMajor>;

using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape, OperatorShape, float, 32, 16, 8, 8>;

using SharedLoadIterator = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    typename OutputTileThreadMap::CompactedThreadMap, float, 32, 16, 8, 8>;

using Padding = typename WarpTileIterator::Padding;
static constexpr int kFragmentsPerIteration = WarpShape::kN / OperatorShape::kN;

struct FakeWarpMma {
    using Shape = WarpShape;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementC = float;
    struct FakeOperator {
        using Shape = OperatorShape;
        using ElementC = float;
        using FragmentC = cutlass::Array<float, OperatorFragmentC::kElements>;
    };
    struct FakePolicy { using Operator = FakeOperator; };
    using Policy = FakePolicy;
};

using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    TileShape,
    FakeWarpMma,
    1,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    EpilogueOutputOp,
    Padding,
    kFragmentsPerIteration>;

using FP32AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

struct SharedStorage {
    union {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
};

struct KernelParams {
    cutlass::gemm::GemmCoord problem_size;
    typename IteratorA::Params params_A;
    const ElementA* ptr_A;
    typename IteratorB::Params params_B;
    const ElementB* ptr_B;
    typename OutputTileIterator::Params params_D;
    ElementOutput* ptr_D;
    const float* ptr_scale_A;
    const float* ptr_scale_B;
    int scale_stride_A;
    int scale_stride_B;
    int K_blocks;
};

__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel(KernelParams params) {
    extern __shared__ char smem_buf[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int cta_m = blockIdx.x;
    int cta_n = blockIdx.y;
    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape::kN};

    IteratorA iterator_A(
        params.params_A,
        const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K),
        thread_idx,
        tb_offset_A);

    IteratorB iterator_B(
        params.params_B,
        const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N),
        thread_idx,
        tb_offset_B);

    int gemm_k_iterations = (K + TileShape::kK - 1) / TileShape::kK;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentC int32_accum;
    int32_accum.clear();

    FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] = 0.0f;
    }

    typename Mma::PipeState pipe_state;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape::kK - 1) / TileShape::kK;
    int m_qb = (cta_m * TileShape::kM) / kBlockQuantSize;
    int n_qb = (cta_n * TileShape::kN) / kBlockQuantSize;

    static constexpr int kWarpGemmIter = Mma::Base::kWarpGemmIterations;  // 4

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
            // LDSM: load next warp-tile fragments
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter - 1) {
                int group_start_A = warp_mma_k * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter) {
                int group_start_A = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        // Dequant: kK=128=block_size, every k_tile is a quant block
        float sa = params.ptr_scale_A[m_qb * params.scale_stride_A + kt];
        float sb = params.ptr_scale_B[n_qb * params.scale_stride_B + kt];
        float combined_scale = sa * sb;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentC::kElements; ++i) {
            fp32_accum[i] += static_cast<float>(int32_accum[i]) * combined_scale;
        }
        int32_accum.clear();
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape::kM, cta_n * TileShape::kN};

    OutputTileIterator iterator_D(
        params.params_D,
        params.ptr_D,
        cutlass::MatrixCoord(M, N),
        thread_idx,
        threadblock_offset);

    OutputTileIterator iterator_C = iterator_D;
    EpilogueOutputOp output_op({1.0f, 0.0f});
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

// ============================================================
// Config 2: kK=128, stg=3, dequant every 2 k_tiles (block_size=256)
// Reuses the same tile config as Config 1. Only dequant frequency changes.
// This tests the effect of halving the FP32/IMMA ratio (from 1:2 to ~1:1).
// ============================================================

static constexpr int kBlockQuantSize256 = 256;

__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel_bq256(KernelParams params) {
    extern __shared__ char smem_buf[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int cta_m = blockIdx.x;
    int cta_n = blockIdx.y;
    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape::kN};

    IteratorA iterator_A(
        params.params_A,
        const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K),
        thread_idx, tb_offset_A);

    IteratorB iterator_B(
        params.params_B,
        const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N),
        thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShape::kK - 1) / TileShape::kK;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentC int32_accum;
    int32_accum.clear();

    FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] = 0.0f;
    }

    typename Mma::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape::kK - 1) / TileShape::kK;
    int m_qb = (cta_m * TileShape::kM) / kBlockQuantSize256;
    int n_qb = (cta_n * TileShape::kN) / kBlockQuantSize256;
    int k_tiles_per_qb = kBlockQuantSize256 / TileShape::kK;  // 2

    static constexpr int kWarpGemmIter = Mma::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter - 1) {
                int group_start_A = warp_mma_k * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter) {
                int group_start_A = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        // Dequant every k_tiles_per_qb k_tiles (= every 2 for block_size=256)
        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int kb = kt / k_tiles_per_qb;
            float sa = params.ptr_scale_A[m_qb * params.scale_stride_A + kb];
            float sb = params.ptr_scale_B[n_qb * params.scale_stride_B + kb];
            float combined_scale = sa * sb;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; ++i) {
                fp32_accum[i] += static_cast<float>(int32_accum[i]) * combined_scale;
            }
            int32_accum.clear();
        }
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape::kM, cta_n * TileShape::kN};

    OutputTileIterator iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutputTileIterator iterator_C = iterator_D;
    EpilogueOutputOp output_op({1.0f, 0.0f});
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}


// ============================================================
// Config 2b: kK=128, stg=3, dequant every 2 k_tiles (block_size=256)
// Fast dequant variant: bias-offset method (3 instr/element, 0 XU)
// ============================================================

namespace fast_dequant {

// Bias-offset INT32→FP32: adds a constant bias to shift signed values into
// the non-negative range [0, 2^23), then uses the IEEE754 magic-number trick.
// SASS: IADD3 + LOP3 + FFMA = 3 instructions/element (2 ALU + 1 FMA, 0 XU).
// The accumulated bias error is corrected once before epilogue.
static constexpr uint32_t kMagic = 0x4B000000u;   // FP32 bit pattern of 2^23
static constexpr int32_t  kBias  = (1 << 22);      // 4194304, shifts range to [0, ~6.3M]
static constexpr float    kBiasFloat = 8388608.0f + static_cast<float>(kBias);  // 2^23 + bias

__device__ __forceinline__ float biased_i32_as_f32(int32_t val) {
    uint32_t u = static_cast<uint32_t>(val + kBias) | kMagic;
    return __int_as_float(u);  // == 2^23 + val + bias == kBiasFloat + val
}

}  // namespace fast_dequant

__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel_fast_dequant(KernelParams params) {
    extern __shared__ char smem_buf[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int cta_m = blockIdx.x;
    int cta_n = blockIdx.y;
    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape::kN};

    IteratorA iterator_A(
        params.params_A,
        const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K),
        thread_idx, tb_offset_A);

    IteratorB iterator_B(
        params.params_B,
        const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N),
        thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShape::kK - 1) / TileShape::kK;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentC int32_accum;
    int32_accum.clear();

    FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] = 0.0f;
    }

    float bias_accum = 0.0f;

    typename Mma::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape::kK - 1) / TileShape::kK;
    int m_qb = (cta_m * TileShape::kM) / kBlockQuantSize256;
    int n_qb = (cta_n * TileShape::kN) / kBlockQuantSize256;
    int k_tiles_per_qb = kBlockQuantSize256 / TileShape::kK;  // 2

    static constexpr int kWarpGemmIter = Mma::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter - 1) {
                int group_start_A = warp_mma_k * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter) {
                int group_start_A = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int kb = kt / k_tiles_per_qb;
            float sa = params.ptr_scale_A[m_qb * params.scale_stride_A + kb];
            float sb = params.ptr_scale_B[n_qb * params.scale_stride_B + kb];
            float combined_scale = sa * sb;

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; ++i) {
                fp32_accum[i] += fast_dequant::biased_i32_as_f32(int32_accum[i]) * combined_scale;
            }
            bias_accum += combined_scale;
            int32_accum.clear();
        }
    }

    // Deferred bias correction: subtract accumulated (kBiasFloat * scale) from every element
    float bias_correction = fast_dequant::kBiasFloat * bias_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] -= bias_correction;
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape::kM, cta_n * TileShape::kN};

    OutputTileIterator iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutputTileIterator iterator_C = iterator_D;
    EpilogueOutputOp output_op({1.0f, 0.0f});
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

// ============================================================
// Config 5: bq512 — block_size=512, dequant every 4 kTiles
// Two accumulators only (int32_accum + fp32_accum), no register spill.
// Scale interface: Q_k (INT32 per qb) * F_g (FP32 per super-group).
// At each qb boundary: fp32_accum += float(int32_accum) * float(Q_combined) * F_g
// where Q_combined = Q_A[k] * Q_B[k] (scalar I2F, 1 instruction not 64).
// Dequant frequency: K/512 times (half of bq256).
// ============================================================

static constexpr int kBlockQuantSize512 = 512;

struct Bq512KernelParams {
    cutlass::gemm::GemmCoord problem_size;
    typename IteratorA::Params params_A;
    const ElementA* ptr_A;
    typename IteratorB::Params params_B;
    const ElementB* ptr_B;
    typename OutputTileIterator::Params params_D;
    ElementOutput* ptr_D;
    const int32_t* ptr_Q_A;    // [M_blocks, num_quant_blocks] INT32
    const int32_t* ptr_Q_B;    // [N_blocks, num_quant_blocks] INT32
    const float* ptr_F_A;      // [M_blocks, num_super_groups] FP32
    const float* ptr_F_B;      // [N_blocks, num_super_groups] FP32
    int k_tiles_per_qb;         // 512/128 = 4
    int num_quant_blocks;
    int super_group_size;
    int num_super_groups;
    int q_stride;
    int f_stride;
};

__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel_bq512(Bq512KernelParams params) {
    extern __shared__ char smem_buf[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int cta_m = blockIdx.x;
    int cta_n = blockIdx.y;
    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape::kN};

    IteratorA iterator_A(
        params.params_A, const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K), thread_idx, tb_offset_A);

    IteratorB iterator_B(
        params.params_B, const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N), thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShape::kK - 1) / TileShape::kK;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    // Only 2 accumulators — no register spill
    FragmentC int32_accum;
    int32_accum.clear();

    FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] = 0.0f;
    }

    typename Mma::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape::kK - 1) / TileShape::kK;
    int k_tiles_per_qb = params.k_tiles_per_qb;
    int m_qb = cta_m;
    int n_qb = cta_n;

    static constexpr int kWarpGemmIter = Mma::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter - 1) {
                int group_start_A = warp_mma_k * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter) {
                int group_start_A = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        // Dequant every k_tiles_per_qb kTiles (=4 for block_size=512)
        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int qb_idx = kt / k_tiles_per_qb;
            int group_idx = qb_idx / params.super_group_size;

            // Scalar scale reconstruction: combined = float(Q_A * Q_B) * (F_A * F_B)
            int q_a = params.ptr_Q_A[m_qb * params.q_stride + qb_idx];
            int q_b = params.ptr_Q_B[n_qb * params.q_stride + qb_idx];
            float f_a = params.ptr_F_A[m_qb * params.f_stride + group_idx];
            float f_b = params.ptr_F_B[n_qb * params.f_stride + group_idx];
            float combined_scale = static_cast<float>(q_a * q_b) * (f_a * f_b);

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; ++i) {
                fp32_accum[i] += static_cast<float>(int32_accum[i]) * combined_scale;
            }
            int32_accum.clear();
        }
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape::kM, cta_n * TileShape::kN};

    OutputTileIterator iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutputTileIterator iterator_C = iterator_D;
    EpilogueOutputOp output_op({1.0f, 0.0f});
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

// ============================================================
// Config 5b: bq512 + bias-offset fast dequant (3 instr/element, 0 XU)
// Same as bq512 but replaces I2F with bias-offset magic number trick.
// ============================================================

__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel_bq512_fast_dequant(Bq512KernelParams params) {
    extern __shared__ char smem_buf[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int cta_m = blockIdx.x;
    int cta_n = blockIdx.y;
    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape::kN};

    IteratorA iterator_A(
        params.params_A, const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K), thread_idx, tb_offset_A);

    IteratorB iterator_B(
        params.params_B, const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N), thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShape::kK - 1) / TileShape::kK;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentC int32_accum;
    int32_accum.clear();

    FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] = 0.0f;
    }

    float bias_accum = 0.0f;

    typename Mma::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape::kK - 1) / TileShape::kK;
    int k_tiles_per_qb = params.k_tiles_per_qb;
    int m_qb = cta_m;
    int n_qb = cta_n;

    static constexpr int kWarpGemmIter = Mma::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter - 1) {
                int group_start_A = warp_mma_k * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter) {
                int group_start_A = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int qb_idx = kt / k_tiles_per_qb;
            int group_idx = qb_idx / params.super_group_size;

            int q_a = params.ptr_Q_A[m_qb * params.q_stride + qb_idx];
            int q_b = params.ptr_Q_B[n_qb * params.q_stride + qb_idx];
            float f_a = params.ptr_F_A[m_qb * params.f_stride + group_idx];
            float f_b = params.ptr_F_B[n_qb * params.f_stride + group_idx];
            float combined_scale = static_cast<float>(q_a * q_b) * (f_a * f_b);

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; ++i) {
                fp32_accum[i] += fast_dequant::biased_i32_as_f32(int32_accum[i]) * combined_scale;
            }
            bias_accum += combined_scale;
            int32_accum.clear();
        }
    }

    float bias_correction = fast_dequant::kBiasFloat * bias_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] -= bias_correction;
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape::kM, cta_n * TileShape::kN};

    OutputTileIterator iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutputTileIterator iterator_C = iterator_D;
    EpilogueOutputOp output_op({1.0f, 0.0f});
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

// ============================================================
// Config 4: Hybrid IMUL + Magic Dequant
// Host-side 2nd-level scale quantization: S_k = F_g * Q_k (Q_k integer)
// Mainloop: int32_accum += Q_k * imma_partial  (pure INT32 IMAD)
// Group boundary: fp32_accum += magic_i2f(int32_accum) * F_g (0 XU)
// ============================================================

struct HybridKernelParams {
    cutlass::gemm::GemmCoord problem_size;
    typename IteratorA::Params params_A;
    const ElementA* ptr_A;
    typename IteratorB::Params params_B;
    const ElementB* ptr_B;
    typename OutputTileIterator::Params params_D;
    ElementOutput* ptr_D;
    const int32_t* ptr_Q_A;    // [M_blocks, num_quant_blocks] INT32 (host-quantized)
    const int32_t* ptr_Q_B;    // [N_blocks, num_quant_blocks] INT32 (host-quantized)
    const float* ptr_F_A;      // [M_blocks, num_super_groups] FP32
    const float* ptr_F_B;      // [N_blocks, num_super_groups] FP32
    int num_quant_blocks;
    int super_group_size;       // L: quant blocks per super-group
    int num_super_groups;       // G = ceil(num_quant_blocks / L)
    int k_tiles_per_qb;         // kTiles per quant block
    int q_stride;               // stride for Q arrays (= num_quant_blocks)
    int f_stride;               // stride for F arrays (= num_super_groups)
    int log_swizzle;            // log2 of swizzle tile size (0=disabled)
    int grid_tiled_n;           // original tiled_n before swizzle remapping
};

// Kernel: full-INT32 K-loop, zero I2F until epilogue.
// 2 accumulators only: int32_accum (per-qb IMMA) + int32_weighted (full-K IMUL).
// At each qb boundary: int32_weighted[i] += Q_combined * int32_accum[i] (pure IMUL).
// Epilogue: fp32 = float(int32_weighted) * (F_A * F_B).
// F_A/F_B are per-row scalars (single super-group covering entire K).
// Register cost: 64 + 64 = 128 INT32 regs for accumulators (no FP32 accum in mainloop).

__global__ void __launch_bounds__(Mma::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel_hybrid(HybridKernelParams params) {
    extern __shared__ char smem_buf[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int log_tile = params.log_swizzle;
    int cta_m = blockIdx.x >> log_tile;
    int cta_n = (blockIdx.y << log_tile) + (blockIdx.x & ((1 << log_tile) - 1));
    if (cta_n >= params.grid_tiled_n) return;

    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape::kN};

    IteratorA iterator_A(
        params.params_A,
        const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K),
        thread_idx, tb_offset_A);

    IteratorB iterator_B(
        params.params_B,
        const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N),
        thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShape::kK - 1) / TileShape::kK;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentC int32_accum;       // per-quant-block IMMA accumulator
    int32_accum.clear();

    FragmentC int32_weighted;    // full-K weighted accumulator (pure INT32)
    int32_weighted.clear();

    typename Mma::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape::kK - 1) / TileShape::kK;
    int k_tiles_per_qb = params.k_tiles_per_qb;

    int m_qb = cta_m;
    int n_qb = cta_n;

    static constexpr int kWarpGemmIter = Mma::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter - 1) {
                int group_start_A = warp_mma_k * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter) {
                int group_start_A = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        // Quant block boundary: pure IMUL, zero I2F
        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int qb_idx = kt / k_tiles_per_qb;
            int q_combined = params.ptr_Q_A[m_qb * params.q_stride + qb_idx]
                           * params.ptr_Q_B[n_qb * params.q_stride + qb_idx];

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; ++i) {
                int32_weighted[i] += q_combined * int32_accum[i];
            }
            int32_accum.clear();
        }
    }

    // Epilogue: single I2F + FP32 scale at the very end
    float fa = params.ptr_F_A[m_qb * params.f_stride];
    float fb = params.ptr_F_B[n_qb * params.f_stride];
    float F_final = fa * fb;

    FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumulatorTile::kElements; ++i) {
        fp32_accum[i] = static_cast<float>(int32_weighted[i]) * F_final;
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape::kM, cta_n * TileShape::kN};

    OutputTileIterator iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutputTileIterator iterator_C = iterator_D;
    EpilogueOutputOp output_op({1.0f, 0.0f});
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

// ============================================================
// Config 3: TileShape 128x128x128, stages=2 (optimized for large-M small-K)
// Doubles tile_N from 64→128, halving grid_N and B-matrix loads.
// stages=2 keeps smem at 64KB to allow 2 CTAs/SM.
// ============================================================

namespace config3 {

using TileShape3       = cutlass::gemm::GemmShape<128, 128, 128>;
using WarpShape3       = cutlass::gemm::GemmShape<64, 64, 128>;
static constexpr int kStages3 = 3;
using DefaultMma3 = cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccum, LayoutOutput,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TileShape3, WarpShape3, InstructionShape,
    kStages3,
    cutlass::arch::OpMultiplyAddSaturate,
    false,
    cutlass::gemm::SharedMemoryClearOption::kNone>;

using IteratorA3 = typename DefaultMma3::IteratorA;
using IteratorB3 = typename DefaultMma3::IteratorB;

// Use custom MMA fork to avoid modifying CUTLASS source
using Mma3 = custom_mma::MmaMultistage<
    typename DefaultMma3::MmaCore::Shape,
    IteratorA3, typename DefaultMma3::MmaCore::SmemIteratorA,
    DefaultMma3::MmaCore::kCacheOpA,
    IteratorB3, typename DefaultMma3::MmaCore::SmemIteratorB,
    DefaultMma3::MmaCore::kCacheOpB,
    ElementAccum, LayoutOutput,
    typename DefaultMma3::MmaCore::MmaPolicy, kStages3>;
using FragmentC3 = typename Mma3::FragmentC;

using WarpMmaOp3    = typename DefaultMma3::MmaCore::MmaPolicy::Operator;
using ArchMmaOp3    = typename WarpMmaOp3::ArchMmaOperator;
using OpShape3      = typename ArchMmaOp3::Shape;
using OpFragC3      = typename ArchMmaOp3::FragmentC;

static constexpr int kEpiAccess3 = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using EpilogueOp3 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, kEpiAccess3, float, float>;

using OutThreadMap3 = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    TileShape3, WarpShape3, 1, ElementOutput, kEpiAccess3>::Type;

using OutTileIter3 = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutThreadMap3, ElementOutput>;

using AccumFragIter3 = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShape3, OpShape3, float,
    cutlass::Array<float, OpFragC3::kElements>,
    cutlass::layout::RowMajor>;

using WarpTileIter3 = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShape3, OpShape3, float, 32, 16, 8, 8>;

using SharedLoadIter3 = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    typename OutThreadMap3::CompactedThreadMap, float, 32, 16, 8, 8>;

using Padding3 = typename WarpTileIter3::Padding;
static constexpr int kFragsPerIter3 = WarpShape3::kN / OpShape3::kN;

struct FakeWarpMma3 {
    using Shape = WarpShape3;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementC = float;
    struct FakeOperator {
        using Shape = OpShape3;
        using ElementC = float;
        using FragmentC = cutlass::Array<float, OpFragC3::kElements>;
    };
    struct FakePolicy { using Operator = FakeOperator; };
    using Policy = FakePolicy;
};

using Epilogue3 = cutlass::epilogue::threadblock::Epilogue<
    TileShape3, FakeWarpMma3, 1,
    OutTileIter3, AccumFragIter3, WarpTileIter3,
    SharedLoadIter3, EpilogueOp3, Padding3, kFragsPerIter3>;

using FP32Accum3 = typename AccumFragIter3::AccumulatorTile;

struct SharedStorage3 {
    union {
        typename Mma3::SharedStorage main_loop;
        typename Epilogue3::SharedStorage epilogue;
    };
};

struct KernelParams3 {
    cutlass::gemm::GemmCoord problem_size;
    typename IteratorA3::Params params_A;
    const ElementA* ptr_A;
    typename IteratorB3::Params params_B;
    const ElementB* ptr_B;
    typename OutTileIter3::Params params_D;
    ElementOutput* ptr_D;
    const float* ptr_scale_A;
    const float* ptr_scale_B;
    int scale_stride_A;
    int scale_stride_B;
    int K_blocks;
};

__global__ void __launch_bounds__(Mma3::WarpCount::kCount * 32, 2)
blockwise_fused_gemm_kernel_128x128(KernelParams3 params) {
    extern __shared__ char smem_buf[];
    SharedStorage3& shared_storage = *reinterpret_cast<SharedStorage3*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int cta_m = blockIdx.x;
    int cta_n = blockIdx.y;
    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShape3::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShape3::kN};

    IteratorA3 iterator_A(
        params.params_A, const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K), thread_idx, tb_offset_A);

    IteratorB3 iterator_B(
        params.params_B, const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N), thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShape3::kK - 1) / TileShape3::kK;

    Mma3 mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentC3 int32_accum;
    int32_accum.clear();

    FP32Accum3 fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32Accum3::kElements; ++i) {
        fp32_accum[i] = 0.0f;
    }

    typename Mma3::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShape3::kK - 1) / TileShape3::kK;
    int m_qb = (cta_m * TileShape3::kM) / kBlockQuantSize;
    int n_qb = (cta_n * TileShape3::kN) / kBlockQuantSize;

    static constexpr int kWarpGemmIter3 = Mma3::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter3; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter3);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIter3);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIter3 - 1) {
                int group_start_A = warp_mma_k * Mma3::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * Mma3::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIter3) {
                int group_start_A = (warp_mma_k + 1) * Mma3::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * Mma3::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIter3) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        float sa = params.ptr_scale_A[m_qb * params.scale_stride_A + kt];
        float sb = params.ptr_scale_B[n_qb * params.scale_stride_B + kt];
        float combined_scale = sa * sb;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentC3::kElements; ++i) {
            fp32_accum[i] += static_cast<float>(int32_accum[i]) * combined_scale;
        }
        int32_accum.clear();
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShape3::kM, cta_n * TileShape3::kN};

    OutTileIter3 iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutTileIter3 iterator_C = iterator_D;
    EpilogueOp3 output_op({1.0f, 0.0f});
    Epilogue3 epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

}  // namespace config3

// ============================================================
// Config small: TileShape 64x64x128, WarpShape 32x32x128
// Optimized for small-M: doubles grid_M vs 128x64, better wave utilization.
// ============================================================

namespace config_small {

using TileShapeS       = cutlass::gemm::GemmShape<64, 64, 128>;
using WarpShapeS       = cutlass::gemm::GemmShape<32, 32, 128>;
static constexpr int kStagesS = 3;

using DefaultMmaS = cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccum, LayoutOutput,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TileShapeS, WarpShapeS, InstructionShape,
    kStagesS,
    cutlass::arch::OpMultiplyAddSaturate,
    false,
    cutlass::gemm::SharedMemoryClearOption::kNone>;

using IteratorAS  = typename DefaultMmaS::IteratorA;
using IteratorBS  = typename DefaultMmaS::IteratorB;

// Use custom MMA fork to avoid modifying CUTLASS source
using MmaS = custom_mma::MmaMultistage<
    typename DefaultMmaS::MmaCore::Shape,
    IteratorAS, typename DefaultMmaS::MmaCore::SmemIteratorA,
    DefaultMmaS::MmaCore::kCacheOpA,
    IteratorBS, typename DefaultMmaS::MmaCore::SmemIteratorB,
    DefaultMmaS::MmaCore::kCacheOpB,
    ElementAccum, LayoutOutput,
    typename DefaultMmaS::MmaCore::MmaPolicy, kStagesS>;
using FragmentCS  = typename MmaS::FragmentC;

using WarpMmaOpS    = typename DefaultMmaS::MmaCore::MmaPolicy::Operator;
using ArchMmaOpS    = typename WarpMmaOpS::ArchMmaOperator;
using OpShapeS      = typename ArchMmaOpS::Shape;
using OpFragCS      = typename ArchMmaOpS::FragmentC;

static constexpr int kEpiAccessS = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using EpilogueOpS = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, kEpiAccessS, float, float>;

using OutThreadMapS = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    TileShapeS, WarpShapeS, 1, ElementOutput, kEpiAccessS>::Type;

using OutTileIterS = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutThreadMapS, ElementOutput>;

using AccumFragIterS = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShapeS, OpShapeS, float,
    cutlass::Array<float, OpFragCS::kElements>,
    cutlass::layout::RowMajor>;

using WarpTileIterS = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShapeS, OpShapeS, float, 32, 16, 8, 8>;

using SharedLoadIterS = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    typename OutThreadMapS::CompactedThreadMap, float, 32, 16, 8, 8>;

using PaddingS = typename WarpTileIterS::Padding;
static constexpr int kFragsPerIterS = WarpShapeS::kN / OpShapeS::kN;

struct FakeWarpMmaS {
    using Shape = WarpShapeS;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementC = float;
    struct FakeOperator {
        using Shape = OpShapeS;
        using ElementC = float;
        using FragmentC = cutlass::Array<float, OpFragCS::kElements>;
    };
    struct FakePolicy { using Operator = FakeOperator; };
    using Policy = FakePolicy;
};

using EpilogueS = cutlass::epilogue::threadblock::Epilogue<
    TileShapeS, FakeWarpMmaS, 1,
    OutTileIterS, AccumFragIterS, WarpTileIterS,
    SharedLoadIterS, EpilogueOpS, PaddingS, kFragsPerIterS>;

using FP32AccumS = typename AccumFragIterS::AccumulatorTile;

struct SharedStorageS {
    union {
        typename MmaS::SharedStorage main_loop;
        typename EpilogueS::SharedStorage epilogue;
    };
};

struct HybridKernelParamsS {
    cutlass::gemm::GemmCoord problem_size;
    typename IteratorAS::Params params_A;
    const ElementA* ptr_A;
    typename IteratorBS::Params params_B;
    const ElementB* ptr_B;
    typename OutTileIterS::Params params_D;
    ElementOutput* ptr_D;
    const int32_t* ptr_Q_A;
    const int32_t* ptr_Q_B;
    const float* ptr_F_A;
    const float* ptr_F_B;
    int num_quant_blocks;
    int super_group_size;
    int num_super_groups;
    int k_tiles_per_qb;
    int q_stride;
    int f_stride;
    int log_swizzle;
    int grid_tiled_n;
};

__global__ void __launch_bounds__(MmaS::WarpCount::kCount * 32, 2)
blockwise_fused_gemm_kernel_hybrid_small(HybridKernelParamsS params) {
    extern __shared__ char smem_buf[];
    SharedStorageS& shared_storage = *reinterpret_cast<SharedStorageS*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    int log_tile = params.log_swizzle;
    int cta_m = blockIdx.x >> log_tile;
    int cta_n = (blockIdx.y << log_tile) + (blockIdx.x & ((1 << log_tile) - 1));
    if (cta_n >= params.grid_tiled_n) return;

    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShapeS::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShapeS::kN};

    IteratorAS iterator_A(
        params.params_A,
        const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K),
        thread_idx, tb_offset_A);

    IteratorBS iterator_B(
        params.params_B,
        const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N),
        thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShapeS::kK - 1) / TileShapeS::kK;

    MmaS mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentCS int32_accum;
    int32_accum.clear();

    FragmentCS int32_weighted;
    int32_weighted.clear();

    typename MmaS::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShapeS::kK - 1) / TileShapeS::kK;
    int k_tiles_per_qb = params.k_tiles_per_qb;
    int m_qb = cta_m;
    int n_qb = cta_n;

    static constexpr int kWarpGemmIterS = MmaS::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterS; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterS);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterS);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIterS - 1) {
                int group_start_A = warp_mma_k * MmaS::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * MmaS::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIterS) {
                int group_start_A = (warp_mma_k + 1) * MmaS::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * MmaS::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIterS) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int qb_idx = kt / k_tiles_per_qb;
            int q_combined = params.ptr_Q_A[m_qb * params.q_stride + qb_idx]
                           * params.ptr_Q_B[n_qb * params.q_stride + qb_idx];

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentCS::kElements; ++i) {
                int32_weighted[i] += q_combined * int32_accum[i];
            }
            int32_accum.clear();
        }
    }

    float fa = params.ptr_F_A[m_qb * params.f_stride];
    float fb = params.ptr_F_B[n_qb * params.f_stride];
    float F_final = fa * fb;

    FP32AccumS fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumS::kElements; ++i) {
        fp32_accum[i] = static_cast<float>(int32_weighted[i]) * F_final;
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShapeS::kM, cta_n * TileShapeS::kN};

    OutTileIterS iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutTileIterS iterator_C = iterator_D;
    EpilogueOpS output_op({1.0f, 0.0f});
    EpilogueS epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

}  // namespace config_small

// ============================================================
// Config large: TileShape 256x128x64, WarpShape 64x64x64
// Optimized for large-M: matches int8_matmul tile shape for maximum
// compute density and L2 reuse. Includes threadblock swizzle.
// ============================================================

namespace config_large {

using TileShapeL       = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShapeL       = cutlass::gemm::GemmShape<64, 32, 64>;
static constexpr int kStagesL = 3;

using DefaultMmaL = cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA,
    ElementB, LayoutB, kAlignmentB,
    ElementAccum, LayoutOutput,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    TileShapeL, WarpShapeL, InstructionShape,
    kStagesL,
    cutlass::arch::OpMultiplyAddSaturate,
    false,
    cutlass::gemm::SharedMemoryClearOption::kNone>;

using IteratorAL = typename DefaultMmaL::IteratorA;
using IteratorBL = typename DefaultMmaL::IteratorB;

// Use custom MMA fork to avoid modifying CUTLASS source
using MmaL = custom_mma::MmaMultistage<
    typename DefaultMmaL::MmaCore::Shape,
    IteratorAL, typename DefaultMmaL::MmaCore::SmemIteratorA,
    DefaultMmaL::MmaCore::kCacheOpA,
    IteratorBL, typename DefaultMmaL::MmaCore::SmemIteratorB,
    DefaultMmaL::MmaCore::kCacheOpB,
    ElementAccum, LayoutOutput,
    typename DefaultMmaL::MmaCore::MmaPolicy, kStagesL>;
using FragmentCL = typename MmaL::FragmentC;

using WarpMmaOpL    = typename DefaultMmaL::MmaCore::MmaPolicy::Operator;
using ArchMmaOpL    = typename WarpMmaOpL::ArchMmaOperator;
using OpShapeL      = typename ArchMmaOpL::Shape;
using OpFragCL      = typename ArchMmaOpL::FragmentC;

static constexpr int kEpiAccessL = 128 / cutlass::sizeof_bits<ElementOutput>::value;

using EpilogueOpL = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, kEpiAccessL, float, float>;

using OutThreadMapL = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    TileShapeL, WarpShapeL, 1, ElementOutput, kEpiAccessL>::Type;

using OutTileIterL = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutThreadMapL, ElementOutput>;

using AccumFragIterL = cutlass::epilogue::warp::FragmentIteratorTensorOp<
    WarpShapeL, OpShapeL, float,
    cutlass::Array<float, OpFragCL::kElements>,
    cutlass::layout::RowMajor>;

using WarpTileIterL = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
    WarpShapeL, OpShapeL, float, 32, 16, 8, 8>;

using SharedLoadIterL = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
    typename OutThreadMapL::CompactedThreadMap, float, 32, 16, 8, 8>;

using PaddingL = typename WarpTileIterL::Padding;
static constexpr int kFragsPerIterL = WarpShapeL::kN / OpShapeL::kN;

struct FakeWarpMmaL {
    using Shape = WarpShapeL;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementC = float;
    struct FakeOperator {
        using Shape = OpShapeL;
        using ElementC = float;
        using FragmentC = cutlass::Array<float, OpFragCL::kElements>;
    };
    struct FakePolicy { using Operator = FakeOperator; };
    using Policy = FakePolicy;
};

using EpilogueL = cutlass::epilogue::threadblock::Epilogue<
    TileShapeL, FakeWarpMmaL, 1,
    OutTileIterL, AccumFragIterL, WarpTileIterL,
    SharedLoadIterL, EpilogueOpL, PaddingL, kFragsPerIterL>;

using FP32AccumL = typename AccumFragIterL::AccumulatorTile;

struct SharedStorageL {
    union {
        typename MmaL::SharedStorage main_loop;
        typename EpilogueL::SharedStorage epilogue;
    };
};

struct HybridKernelParamsL {
    cutlass::gemm::GemmCoord problem_size;
    typename IteratorAL::Params params_A;
    const ElementA* ptr_A;
    typename IteratorBL::Params params_B;
    const ElementB* ptr_B;
    typename OutTileIterL::Params params_D;
    ElementOutput* ptr_D;
    const int32_t* ptr_Q_A;
    const int32_t* ptr_Q_B;
    const float* ptr_F_A;
    const float* ptr_F_B;
    int num_quant_blocks;
    int super_group_size;
    int num_super_groups;
    int k_tiles_per_qb;
    int q_stride;
    int f_stride;
    int log_swizzle;
    int grid_tiled_n;
};

__global__ void __launch_bounds__(MmaL::WarpCount::kCount * 32, 1)
blockwise_fused_gemm_kernel_hybrid_large(HybridKernelParamsL params) {
    extern __shared__ char smem_buf[];
    SharedStorageL& shared_storage = *reinterpret_cast<SharedStorageL*>(smem_buf);

    int thread_idx = threadIdx.x;
    int warp_idx   = cutlass::canonical_warp_idx_sync();
    int lane_idx   = threadIdx.x % 32;

    // Threadblock swizzle: remap blockIdx to improve L2 locality
    int log_tile = params.log_swizzle;
    int cta_m = blockIdx.x >> log_tile;
    int cta_n = (blockIdx.y << log_tile) + (blockIdx.x & ((1 << log_tile) - 1));

    int M = params.problem_size.m();
    int N = params.problem_size.n();
    int K = params.problem_size.k();

    if (cta_n >= params.grid_tiled_n) return;

    cutlass::MatrixCoord tb_offset_A{cta_m * TileShapeL::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * TileShapeL::kN};

    IteratorAL iterator_A(
        params.params_A,
        const_cast<ElementA*>(params.ptr_A),
        cutlass::MatrixCoord(M, K),
        thread_idx, tb_offset_A);

    IteratorBL iterator_B(
        params.params_B,
        const_cast<ElementB*>(params.ptr_B),
        cutlass::MatrixCoord(K, N),
        thread_idx, tb_offset_B);

    int gemm_k_iterations = (K + TileShapeL::kK - 1) / TileShapeL::kK;

    MmaL mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    FragmentCL int32_accum;
    int32_accum.clear();

    FragmentCL int32_weighted;
    int32_weighted.clear();

    typename MmaL::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + TileShapeL::kK - 1) / TileShapeL::kK;
    int k_tiles_per_qb = params.k_tiles_per_qb;
    int m_qb = cta_m;
    int n_qb = cta_n;

    static constexpr int kWarpGemmIterL = MmaL::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {

        CUTLASS_PRAGMA_UNROLL
        for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterL; ++warp_mma_k) {
            mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterL);
            mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_A_;

            mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterL);
            mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            ++mma.warp_tile_iterator_B_;

            if (warp_mma_k > 0) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                    pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
            }

            mma.warp_mma_(
                int32_accum,
                pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                int32_accum);

            if (warp_mma_k < kWarpGemmIterL - 1) {
                int group_start_A = warp_mma_k * MmaL::Detail::kAccessesPerGroupA;
                int group_start_B = warp_mma_k * MmaL::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);
            }

            if (warp_mma_k + 2 == kWarpGemmIterL) {
                int group_start_A = (warp_mma_k + 1) * MmaL::Detail::kAccessesPerGroupA;
                int group_start_B = (warp_mma_k + 1) * MmaL::Detail::kAccessesPerGroupB;
                mma.copy_tiles_and_advance(
                    iterator_A, iterator_B, group_start_A, group_start_B);

                cutlass::arch::cp_async_fence();
                mma.gmem_wait();
                mma.advance_smem_write_stage(iterator_A, iterator_B);
                mma.advance_smem_read_stage();

                --gemm_k_iterations;
                iterator_A.clear_mask(gemm_k_iterations == 0);
                iterator_B.clear_mask(gemm_k_iterations == 0);
            }

            if (warp_mma_k + 1 == kWarpGemmIterL) {
                mma.warp_mma_.transform(
                    pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
                    pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
            }
        }

        if ((kt + 1) % k_tiles_per_qb == 0 || kt == total_k_tiles - 1) {
            int qb_idx = kt / k_tiles_per_qb;
            int q_combined = params.ptr_Q_A[m_qb * params.q_stride + qb_idx]
                           * params.ptr_Q_B[n_qb * params.q_stride + qb_idx];

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentCL::kElements; ++i) {
                int32_weighted[i] += q_combined * int32_accum[i];
            }
            int32_accum.clear();
        }
    }

    float fa = params.ptr_F_A[m_qb * params.f_stride];
    float fb = params.ptr_F_B[n_qb * params.f_stride];
    float F_final = fa * fb;

    FP32AccumL fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FP32AccumL::kElements; ++i) {
        fp32_accum[i] = static_cast<float>(int32_weighted[i]) * F_final;
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * TileShapeL::kM, cta_n * TileShapeL::kN};

    OutTileIterL iterator_D(
        params.params_D, params.ptr_D,
        cutlass::MatrixCoord(M, N), thread_idx, threadblock_offset);

    OutTileIterL iterator_C = iterator_D;
    EpilogueOpL output_op({1.0f, 0.0f});
    EpilogueL epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);
}

}  // namespace config_large

torch::Tensor int8_blockwise_fused_matmul_hybrid_large_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    int64_t quant_block_size,
    int64_t super_group_size
) {
    using namespace config_large;
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int num_quant_blocks = (K + static_cast<int>(quant_block_size) - 1) / static_cast<int>(quant_block_size);
    int k_tiles_per_qb = static_cast<int>(quant_block_size) / TileShapeL::kK;
    int num_super_groups = (num_quant_blocks + static_cast<int>(super_group_size) - 1) / static_cast<int>(super_group_size);

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    int tiled_m = (M + TileShapeL::kM - 1) / TileShapeL::kM;
    int tiled_n = (N + TileShapeL::kN - 1) / TileShapeL::kN;

    // Compute swizzle factor (same logic as GemmIdentityThreadblockSwizzle<1>)
    int log_swizzle = 0;
    if (tiled_n >= 6) log_swizzle = 3;
    else if (tiled_n >= 3) log_swizzle = 2;
    else if (tiled_n >= 2) log_swizzle = 1;

    int swizzle_tile = 1 << log_swizzle;

    HybridKernelParamsL params;
    params.problem_size     = {M, N, K};
    params.ptr_Q_A          = static_cast<const int32_t*>(Q_A.data_ptr());
    params.ptr_Q_B          = static_cast<const int32_t*>(Q_B.data_ptr());
    params.ptr_F_A          = static_cast<const float*>(F_A.data_ptr());
    params.ptr_F_B          = static_cast<const float*>(F_B.data_ptr());
    params.num_quant_blocks = num_quant_blocks;
    params.super_group_size = static_cast<int>(super_group_size);
    params.num_super_groups = num_super_groups;
    params.k_tiles_per_qb   = k_tiles_per_qb;
    params.q_stride          = num_quant_blocks;
    params.f_stride          = num_super_groups;
    params.log_swizzle       = log_swizzle;
    params.grid_tiled_n      = tiled_n;
    params.ptr_D             = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A          = typename IteratorAL::Params(LayoutA::packed({M, K}));
    params.ptr_A             = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B          = typename IteratorBL::Params(LayoutB::packed({K, N}));
    params.ptr_B             = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D          = typename OutTileIterL::Params(LayoutOutput::packed({M, N}));

    dim3 grid(tiled_m * swizzle_tile, (tiled_n + swizzle_tile - 1) / swizzle_tile, 1);
    dim3 block(MmaL::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorageL));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_hybrid_large,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_hybrid_large<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_small_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    int64_t quant_block_size,
    int64_t super_group_size
) {
    using namespace config_small;
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int num_quant_blocks = (K + static_cast<int>(quant_block_size) - 1) / static_cast<int>(quant_block_size);
    int k_tiles_per_qb = static_cast<int>(quant_block_size) / TileShapeS::kK;
    int num_super_groups = (num_quant_blocks + static_cast<int>(super_group_size) - 1) / static_cast<int>(super_group_size);

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    HybridKernelParamsS params;
    params.problem_size     = {M, N, K};
    params.ptr_Q_A          = static_cast<const int32_t*>(Q_A.data_ptr());
    params.ptr_Q_B          = static_cast<const int32_t*>(Q_B.data_ptr());
    params.ptr_F_A          = static_cast<const float*>(F_A.data_ptr());
    params.ptr_F_B          = static_cast<const float*>(F_B.data_ptr());
    params.num_quant_blocks = num_quant_blocks;
    params.super_group_size = static_cast<int>(super_group_size);
    params.num_super_groups = num_super_groups;
    params.k_tiles_per_qb   = k_tiles_per_qb;
    params.q_stride          = num_quant_blocks;
    params.f_stride          = num_super_groups;
    params.ptr_D             = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A          = typename IteratorAS::Params(LayoutA::packed({M, K}));
    params.ptr_A             = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B          = typename IteratorBS::Params(LayoutB::packed({K, N}));
    params.ptr_B             = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D          = typename OutTileIterS::Params(LayoutOutput::packed({M, N}));

    int tiled_m = (M + TileShapeS::kM - 1) / TileShapeS::kM;
    int tiled_n = (N + TileShapeS::kN - 1) / TileShapeS::kN;

    int log_swizzle = 0;
    for (int s = 3; s >= 1; --s) {
        if (tiled_n >= (1 << s)) { log_swizzle = s; break; }
    }
    int swizzle_tile = 1 << log_swizzle;

    params.log_swizzle  = log_swizzle;
    params.grid_tiled_n = tiled_n;

    dim3 grid(tiled_m * swizzle_tile, (tiled_n + swizzle_tile - 1) / swizzle_tile, 1);
    dim3 block(MmaS::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorageS));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_hybrid_small,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_hybrid_small<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_hybrid_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,            // [M_blocks, num_quant_blocks] INT16
    torch::Tensor Q_B,            // [N_blocks, num_quant_blocks] INT16
    torch::Tensor F_A,            // [M_blocks, num_super_groups] FP32
    torch::Tensor F_B,            // [N_blocks, num_super_groups] FP32
    int64_t quant_block_size,
    int64_t super_group_size
) {
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int num_quant_blocks = (K + static_cast<int>(quant_block_size) - 1) / static_cast<int>(quant_block_size);
    int k_tiles_per_qb = static_cast<int>(quant_block_size) / TileShape::kK;
    int num_super_groups = (num_quant_blocks + static_cast<int>(super_group_size) - 1) / static_cast<int>(super_group_size);

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    HybridKernelParams params;
    params.problem_size     = {M, N, K};
    params.ptr_Q_A          = static_cast<const int32_t*>(Q_A.data_ptr());
    params.ptr_Q_B          = static_cast<const int32_t*>(Q_B.data_ptr());
    params.ptr_F_A          = static_cast<const float*>(F_A.data_ptr());
    params.ptr_F_B          = static_cast<const float*>(F_B.data_ptr());
    params.num_quant_blocks = num_quant_blocks;
    params.super_group_size = static_cast<int>(super_group_size);
    params.num_super_groups = num_super_groups;
    params.k_tiles_per_qb   = k_tiles_per_qb;
    params.q_stride          = num_quant_blocks;
    params.f_stride          = num_super_groups;
    params.ptr_D             = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A          = typename IteratorA::Params(LayoutA::packed({M, K}));
    params.ptr_A             = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B          = typename IteratorB::Params(LayoutB::packed({K, N}));
    params.ptr_B             = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D          = typename OutputTileIterator::Params(LayoutOutput::packed({M, N}));

    int tiled_m = (M + TileShape::kM - 1) / TileShape::kM;
    int tiled_n = (N + TileShape::kN - 1) / TileShape::kN;

    int log_swizzle = 0;
    for (int s = 3; s >= 1; --s) {
        if (tiled_n >= (1 << s)) { log_swizzle = s; break; }
    }
    int swizzle_tile = 1 << log_swizzle;

    params.log_swizzle  = log_swizzle;
    params.grid_tiled_n = tiled_n;

    dim3 grid(tiled_m * swizzle_tile, (tiled_n + swizzle_tile - 1) / swizzle_tile, 1);
    dim3 block(Mma::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_hybrid,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_hybrid<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_bq512_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    int64_t quant_block_size,
    int64_t super_group_size
) {
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int num_quant_blocks = (K + static_cast<int>(quant_block_size) - 1) / static_cast<int>(quant_block_size);
    int k_tiles_per_qb = static_cast<int>(quant_block_size) / TileShape::kK;
    int num_super_groups = (num_quant_blocks + static_cast<int>(super_group_size) - 1) / static_cast<int>(super_group_size);

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    Bq512KernelParams params;
    params.problem_size     = {M, N, K};
    params.ptr_Q_A          = static_cast<const int32_t*>(Q_A.data_ptr());
    params.ptr_Q_B          = static_cast<const int32_t*>(Q_B.data_ptr());
    params.ptr_F_A          = static_cast<const float*>(F_A.data_ptr());
    params.ptr_F_B          = static_cast<const float*>(F_B.data_ptr());
    params.num_quant_blocks = num_quant_blocks;
    params.super_group_size = static_cast<int>(super_group_size);
    params.num_super_groups = num_super_groups;
    params.k_tiles_per_qb   = k_tiles_per_qb;
    params.q_stride          = num_quant_blocks;
    params.f_stride          = num_super_groups;
    params.ptr_D             = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A          = typename IteratorA::Params(LayoutA::packed({M, K}));
    params.ptr_A             = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B          = typename IteratorB::Params(LayoutB::packed({K, N}));
    params.ptr_B             = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D          = typename OutputTileIterator::Params(LayoutOutput::packed({M, N}));

    dim3 grid(
        (M + TileShape::kM - 1) / TileShape::kM,
        (N + TileShape::kN - 1) / TileShape::kN,
        1);
    dim3 block(Mma::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_bq512,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_bq512<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_bq512_fast_dequant_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor Q_A,
    torch::Tensor Q_B,
    torch::Tensor F_A,
    torch::Tensor F_B,
    int64_t quant_block_size,
    int64_t super_group_size
) {
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int num_quant_blocks = (K + static_cast<int>(quant_block_size) - 1) / static_cast<int>(quant_block_size);
    int k_tiles_per_qb = static_cast<int>(quant_block_size) / TileShape::kK;
    int num_super_groups = (num_quant_blocks + static_cast<int>(super_group_size) - 1) / static_cast<int>(super_group_size);

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    Bq512KernelParams params;
    params.problem_size     = {M, N, K};
    params.ptr_Q_A          = static_cast<const int32_t*>(Q_A.data_ptr());
    params.ptr_Q_B          = static_cast<const int32_t*>(Q_B.data_ptr());
    params.ptr_F_A          = static_cast<const float*>(F_A.data_ptr());
    params.ptr_F_B          = static_cast<const float*>(F_B.data_ptr());
    params.num_quant_blocks = num_quant_blocks;
    params.super_group_size = static_cast<int>(super_group_size);
    params.num_super_groups = num_super_groups;
    params.k_tiles_per_qb   = k_tiles_per_qb;
    params.q_stride          = num_quant_blocks;
    params.f_stride          = num_super_groups;
    params.ptr_D             = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A          = typename IteratorA::Params(LayoutA::packed({M, K}));
    params.ptr_A             = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B          = typename IteratorB::Params(LayoutB::packed({K, N}));
    params.ptr_B             = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D          = typename OutputTileIterator::Params(LayoutOutput::packed({M, N}));

    dim3 grid(
        (M + TileShape::kM - 1) / TileShape::kM,
        (N + TileShape::kN - 1) / TileShape::kN,
        1);
    dim3 block(Mma::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_bq512_fast_dequant,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_bq512_fast_dequant<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_fast_dequant_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor input_scale,
    torch::Tensor weight_scale
) {
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int K_blocks = (K + kBlockQuantSize256 - 1) / kBlockQuantSize256;

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    KernelParams params;
    params.problem_size   = {M, N, K};
    params.K_blocks       = K_blocks;
    params.scale_stride_A = K_blocks;
    params.scale_stride_B = K_blocks;
    params.ptr_scale_A    = static_cast<const float*>(input_scale.data_ptr());
    params.ptr_scale_B    = static_cast<const float*>(weight_scale.data_ptr());
    params.ptr_D          = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A       = typename IteratorA::Params(LayoutA::packed({M, K}));
    params.ptr_A          = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B       = typename IteratorB::Params(LayoutB::packed({K, N}));
    params.ptr_B          = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D       = typename OutputTileIterator::Params(LayoutOutput::packed({M, N}));

    dim3 grid(
        (M + TileShape::kM - 1) / TileShape::kM,
        (N + TileShape::kN - 1) / TileShape::kN,
        1);
    dim3 block(Mma::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_fast_dequant,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_fast_dequant<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_128x128_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor input_scale,
    torch::Tensor weight_scale
) {
    using namespace config3;
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int K_blocks = (K + kBlockQuantSize - 1) / kBlockQuantSize;

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    KernelParams3 params;
    params.problem_size   = {M, N, K};
    params.K_blocks       = K_blocks;
    params.scale_stride_A = K_blocks;
    params.scale_stride_B = K_blocks;
    params.ptr_scale_A    = static_cast<const float*>(input_scale.data_ptr());
    params.ptr_scale_B    = static_cast<const float*>(weight_scale.data_ptr());
    params.ptr_D          = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A       = typename IteratorA3::Params(LayoutA::packed({M, K}));
    params.ptr_A          = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B       = typename IteratorB3::Params(LayoutB::packed({K, N}));
    params.ptr_B          = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D       = typename OutTileIter3::Params(LayoutOutput::packed({M, N}));

    dim3 grid(
        (M + TileShape3::kM - 1) / TileShape3::kM,
        (N + TileShape3::kN - 1) / TileShape3::kN,
        1);
    dim3 block(Mma3::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage3));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_128x128,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_128x128<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_kk256_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor input_scale,
    torch::Tensor weight_scale
) {
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int K_blocks = (K + kBlockQuantSize256 - 1) / kBlockQuantSize256;

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    KernelParams params;
    params.problem_size   = {M, N, K};
    params.K_blocks       = K_blocks;
    params.scale_stride_A = K_blocks;
    params.scale_stride_B = K_blocks;
    params.ptr_scale_A    = static_cast<const float*>(input_scale.data_ptr());
    params.ptr_scale_B    = static_cast<const float*>(weight_scale.data_ptr());
    params.ptr_D          = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A       = typename IteratorA::Params(LayoutA::packed({M, K}));
    params.ptr_A          = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B       = typename IteratorB::Params(LayoutB::packed({K, N}));
    params.ptr_B          = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D       = typename OutputTileIterator::Params(LayoutOutput::packed({M, N}));

    dim3 grid(
        (M + TileShape::kM - 1) / TileShape::kM,
        (N + TileShape::kN - 1) / TileShape::kN,
        1);
    dim3 block(Mma::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel_bq256,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel_bq256<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor int8_blockwise_fused_matmul_host(
    torch::Tensor input_q,
    torch::Tensor weight_q,
    torch::Tensor input_scale,
    torch::Tensor weight_scale
) {
    int M = input_q.size(0);
    int K = input_q.size(1);
    int N = weight_q.size(0);
    int K_blocks = (K + kBlockQuantSize - 1) / kBlockQuantSize;

    auto out = torch::empty({M, N},
        torch::dtype(torch::kBFloat16).device(input_q.device()));

    KernelParams params;
    params.problem_size   = {M, N, K};
    params.K_blocks       = K_blocks;
    params.scale_stride_A = K_blocks;
    params.scale_stride_B = K_blocks;
    params.ptr_scale_A    = static_cast<const float*>(input_scale.data_ptr());
    params.ptr_scale_B    = static_cast<const float*>(weight_scale.data_ptr());
    params.ptr_D          = static_cast<ElementOutput*>(out.data_ptr());
    params.params_A       = typename IteratorA::Params(LayoutA::packed({M, K}));
    params.ptr_A          = static_cast<const ElementA*>(input_q.data_ptr());
    params.params_B       = typename IteratorB::Params(LayoutB::packed({K, N}));
    params.ptr_B          = static_cast<const ElementB*>(weight_q.data_ptr());
    params.params_D       = typename OutputTileIterator::Params(LayoutOutput::packed({M, N}));

    dim3 grid(
        (M + TileShape::kM - 1) / TileShape::kM,
        (N + TileShape::kN - 1) / TileShape::kN,
        1);
    dim3 block(Mma::WarpCount::kCount * 32);

    int smem_size = static_cast<int>(sizeof(SharedStorage));
    auto stream = at::cuda::getCurrentCUDAStream();

    if (smem_size > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            blockwise_fused_gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    blockwise_fused_gemm_kernel<<<grid, block, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
