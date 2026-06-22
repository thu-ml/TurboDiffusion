/*
 * Copyright (c) 2025 by TurboDiffusion team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Citation (please cite if you use this code):
 *
 * @article{zhang2025turbodiffusion,
 *   title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
 *   author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
 *   journal={arXiv preprint arXiv:2512.16093},
 *   year={2025}
 * }
 *
 * rocWMMA-based GEMM kernel for AMD RDNA3 GPUs.
 * This kernel performs int8 GEMM with per-block quantization scaling.
 * 
 * Based on rocWMMA from https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma
 */

#pragma once

#include <hip/hip_runtime.h>

// Undefine the no-half-conversion macros that PyTorch sets
// rocWMMA needs these conversions to work properly
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <rocwmma/rocwmma.hpp>

#include "common/platform.hpp"

using namespace rocwmma;

// Helper for float to output type conversion
template <typename T>
TURBO_DEVICE TURBO_INLINE T float_to_output(float val);

template <>
TURBO_DEVICE TURBO_INLINE __half float_to_output<__half>(float val) {
    return __float2half(val);
}

template <>
TURBO_DEVICE TURBO_INLINE hip_bfloat16 float_to_output<hip_bfloat16>(float val) {
    return hip_bfloat16(val);
}

template <>
TURBO_DEVICE TURBO_INLINE float float_to_output<float>(float val) {
    return val;
}

template <>
TURBO_DEVICE TURBO_INLINE int32_t float_to_output<int32_t>(float val) {
    return static_cast<int32_t>(val);
}

// RDNA3 (gfx11) specific parameters
// Wave size: 32, Block sizes: 16x16x16
namespace rdna3 {
    constexpr uint32_t ROCWMMA_M = 16u;
    constexpr uint32_t ROCWMMA_N = 16u; 
    constexpr uint32_t ROCWMMA_K = 16u;
    constexpr uint32_t WAVE_SIZE = 32u;
    constexpr uint32_t QUANT_BLOCK = 128u;  // Quantization block size
}

template <
    class OutputDtype_,
    bool IsEvenM,
    bool IsEvenN
>
struct GemmKernelRocWMMA {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using OutputDtype = OutputDtype_;
    using AccumulatorDtype = int32_t;
    using ComputeDtype = int32_t;  // MMA accumulator type
    
    // Tile sizes
    static constexpr int TileM = rdna3::ROCWMMA_M;
    static constexpr int TileN = rdna3::ROCWMMA_N;
    static constexpr int TileK = rdna3::ROCWMMA_K;
    static constexpr int BlockSize = rdna3::QUANT_BLOCK;
    static constexpr int WaveSize = rdna3::WAVE_SIZE;
    
    // Warp tile: how many MMA tiles each wave computes
    static constexpr int WarpTileM = 2;  // 2 tiles in M direction = 32
    static constexpr int WarpTileN = 2;  // 2 tiles in N direction = 32
    
    // Thread block configuration
    static constexpr int TBlockX = 128;  // 4 waves
    static constexpr int TBlockY = 1;
    static constexpr int NumWarps = TBlockX / WaveSize;  // 4 waves
    
    // Macro tile computed by entire thread block
    static constexpr int MacroTileM = NumWarps * WarpTileM * TileM;  // 4 * 2 * 16 = 128
    static constexpr int MacroTileN = TBlockY * WarpTileN * TileN;   // 1 * 2 * 16 = 32
    
    // Fragment types - using row_major for A and col_major for B (NT layout)
    using FragA = fragment<matrix_a, TileM, TileN, TileK, ElementA, row_major>;
    using FragB = fragment<matrix_b, TileM, TileN, TileK, ElementB, col_major>;
    using FragAcc = fragment<accumulator, TileM, TileN, TileK, ComputeDtype>;
    
    struct Params {
        void const* Aptr;
        void const* ASptr;
        void const* Bptr;
        void const* BSptr;
        void* Dptr;
        int64_t const m;
        int64_t const n;
        int64_t const k;
        int const swizzle_dir;
        int const swizzle_size;
    };

    using Arguments = Params;
    
    static constexpr int ThreadNum = TBlockX * TBlockY;
    // Shared memory for storing fragments: each wave needs TileM*TileN*sizeof(float) per tile
    // With WarpTileM=2, WarpTileN=2, and NumWarps=4 waves:
    // Size = NumWarps * WarpTileM * WarpTileN * TileM * TileN * sizeof(float)
    //      = 4 * 2 * 2 * 16 * 16 * 4 = 16384 bytes
    static constexpr int ShmSize = NumWarps * WarpTileM * WarpTileN * TileM * TileN * sizeof(float);
    static constexpr int MaxThreadsPerBlock = ThreadNum;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    
    static bool can_implement(int64_t m, int64_t n, int64_t k) {
        if (k % BlockSize != 0) return false;
        if ((n * sizeof(OutputDtype)) % 16 != 0) return false;
        return true;
    }
    
    static Params to_underlying_arguments(Arguments const& args) {
        return args;
    }
    
    TURBO_HOST_DEVICE
    static int64_t cdiv(int64_t a, int64_t b) {
        return (a + b - 1) / b;
    }
    
    static dim3 get_grid_size(int64_t m, int64_t n) {
        int64_t grid_m = cdiv(m, MacroTileM);
        int64_t grid_n = cdiv(n, MacroTileN);
        return dim3(grid_m * grid_n);
    }
    
    TURBO_DEVICE
    void operator()(Params const& params, char* smem_data) {
        int64_t const m = params.m;
        int64_t const n = params.n;
        int64_t const k = params.k;
        
        // Wave and lane indices
        int waveId = threadIdx.x / WaveSize;
        int laneId = threadIdx.x % WaveSize;
        
        // Grid dimensions
        int64_t grid_m = cdiv(m, MacroTileM);
        int64_t grid_n = cdiv(n, MacroTileN);
        
        // Block coordinates (linear to 2D)
        int64_t block_m = blockIdx.x % grid_m;
        int64_t block_n = blockIdx.x / grid_m;
        
        // Base coordinates for this wave's output tiles
        int64_t wave_m_base = block_m * MacroTileM + waveId * WarpTileM * TileM;
        int64_t wave_n_base = block_n * MacroTileN;
        
        // Pointers
        ElementA const* A = reinterpret_cast<ElementA const*>(params.Aptr);
        ElementB const* B = reinterpret_cast<ElementB const*>(params.Bptr);
        float const* AS = reinterpret_cast<float const*>(params.ASptr);
        float const* BS = reinterpret_cast<float const*>(params.BSptr);
        
        // Number of quantization blocks in K dimension
        int64_t num_quant_blocks_k = k / BlockSize;
        
        // Float accumulators for dequantized results
        float floatAcc[WarpTileM][WarpTileN][FragAcc::num_elements];
        
        // Initialize accumulators
        TURBO_PRAGMA_UNROLL
        for (int wm = 0; wm < WarpTileM; ++wm) {
            TURBO_PRAGMA_UNROLL
            for (int wn = 0; wn < WarpTileN; ++wn) {
                TURBO_PRAGMA_UNROLL
                for (int i = 0; i < FragAcc::num_elements; ++i) {
                    floatAcc[wm][wn][i] = 0.0f;
                }
            }
        }
        
        // Process each quantization block
        for (int64_t qb = 0; qb < num_quant_blocks_k; ++qb) {
            int64_t k_start = qb * BlockSize;
            int64_t k_end = k_start + BlockSize;
            
            // Integer accumulators for this quant block
            FragAcc fragAcc[WarpTileM][WarpTileN];
            TURBO_PRAGMA_UNROLL
            for (int wm = 0; wm < WarpTileM; ++wm) {
                TURBO_PRAGMA_UNROLL
                for (int wn = 0; wn < WarpTileN; ++wn) {
                    fill_fragment(fragAcc[wm][wn], static_cast<ComputeDtype>(0));
                }
            }
            
            // K-loop within quantization block
            for (int64_t kk = k_start; kk < k_end; kk += TileK) {
                // Load and compute for each tile in warp tile
                TURBO_PRAGMA_UNROLL
                for (int wm = 0; wm < WarpTileM; ++wm) {
                    int64_t tile_m = wave_m_base + wm * TileM;
                    
                    FragA fragA;
                    if (tile_m < m) {
                        // A is row-major: A[m, k]
                        load_matrix_sync(fragA, A + tile_m * k + kk, k);
                    } else {
                        fill_fragment(fragA, static_cast<ElementA>(0));
                    }
                    
                    TURBO_PRAGMA_UNROLL
                    for (int wn = 0; wn < WarpTileN; ++wn) {
                        int64_t tile_n = wave_n_base + wn * TileN;
                        
                        FragB fragB;
                        if (tile_n < n) {
                            // B is stored as [N, K] in col-major (K changes fastest when reading B[n, :])
                            load_matrix_sync(fragB, B + tile_n * k + kk, k);
                        } else {
                            fill_fragment(fragB, static_cast<ElementB>(0));
                        }
                        
                        // Matrix multiply-accumulate
                        mma_sync(fragAcc[wm][wn], fragA, fragB, fragAcc[wm][wn]);
                    }
                }
            }
            
            // Dequantize this block's contribution
            TURBO_PRAGMA_UNROLL
            for (int wm = 0; wm < WarpTileM; ++wm) {
                int64_t tile_m = wave_m_base + wm * TileM;
                int64_t qblock_m = tile_m / BlockSize;
                
                // Get scale for A
                float scale_a = 1.0f;
                if (qblock_m < cdiv(m, BlockSize) && qb < num_quant_blocks_k) {
                    scale_a = AS[qblock_m * num_quant_blocks_k + qb];
                }
                
                TURBO_PRAGMA_UNROLL
                for (int wn = 0; wn < WarpTileN; ++wn) {
                    int64_t tile_n = wave_n_base + wn * TileN;
                    int64_t qblock_n = tile_n / BlockSize;
                    
                    // Get scale for B
                    float scale_b = 1.0f;
                    if (qblock_n < cdiv(n, BlockSize) && qb < num_quant_blocks_k) {
                        scale_b = BS[qblock_n * num_quant_blocks_k + qb];
                    }
                    
                    float scale = scale_a * scale_b;
                    
                    // Accumulate dequantized values
                    TURBO_PRAGMA_UNROLL
                    for (int i = 0; i < FragAcc::num_elements; ++i) {
                        floatAcc[wm][wn][i] += static_cast<float>(fragAcc[wm][wn].x[i]) * scale;
                    }
                }
            }
        }
        
        // Store final results using store_matrix_sync to shared memory temp buffer
        // This ensures correct fragment layout interpretation
        OutputDtype* D = reinterpret_cast<OutputDtype*>(params.Dptr);
        
        // Each wave gets its own section of shared memory
        float* smem_temp = reinterpret_cast<float*>(smem_data);
        float* wave_smem = smem_temp + waveId * WarpTileM * WarpTileN * TileM * TileN;
        
        TURBO_PRAGMA_UNROLL
        for (int wm = 0; wm < WarpTileM; ++wm) {
            int64_t tile_m = wave_m_base + wm * TileM;
            
            TURBO_PRAGMA_UNROLL
            for (int wn = 0; wn < WarpTileN; ++wn) {
                int64_t tile_n = wave_n_base + wn * TileN;
                
                // Create a float fragment from the accumulated values
                fragment<accumulator, TileM, TileN, TileK, float> fragFloat;
                TURBO_PRAGMA_UNROLL
                for (int i = 0; i < FragAcc::num_elements; ++i) {
                    fragFloat.x[i] = floatAcc[wm][wn][i];
                }
                
                // Store to wave's temp buffer using rocWMMA (row-major layout)
                float* tile_buf = wave_smem + (wm * WarpTileN + wn) * TileM * TileN;
                store_matrix_sync(tile_buf, fragFloat, TileN, mem_row_major);
                
                __syncthreads();
                
                // Now read from tile_buf with linear indexing
                for (int e = laneId; e < TileM * TileN; e += WaveSize) {
                    int local_row = e / TileN;
                    int local_col = e % TileN;
                    
                    int64_t global_row = tile_m + local_row;
                    int64_t global_col = tile_n + local_col;
                    
                    if (global_row < m && global_col < n) {
                        D[global_row * n + global_col] = float_to_output<OutputDtype>(tile_buf[e]);
                    }
                }
                
                __syncthreads();
            }
        }
    }
};
