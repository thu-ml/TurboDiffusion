/*
 * Copyright (c) 2025 by TurboDiffusion team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Quantization kernel for AMD GPUs using HIP.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "common/platform.hpp"

// Helper for input type to float conversion (handles half types properly)
template <typename T>
TURBO_DEVICE TURBO_INLINE float input_to_float(T val);

template <>
TURBO_DEVICE TURBO_INLINE float input_to_float<__half>(__half val) {
    return __half2float(val);
}

template <>
TURBO_DEVICE TURBO_INLINE float input_to_float<hip_bfloat16>(hip_bfloat16 val) {
    return static_cast<float>(val);
}

template <>
TURBO_DEVICE TURBO_INLINE float input_to_float<float>(float val) {
    return val;
}

TURBO_HOST_DEVICE int64_t cdiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BOOL_SWITCH(COND, CONST_NAME, ...)    \
[&] {                                         \
  if (COND) {                                 \
    static constexpr bool CONST_NAME = true;  \
    return (__VA_ARGS__)();                   \
  } else {                                    \
    static constexpr bool CONST_NAME = false; \
    return (__VA_ARGS__)();                   \
  }                                           \
}()

template <
    class InputDtype_,
    int NumThrPerCta_,
    bool IsEvenM,
    bool IsEvenN
>
class QuantizationHIP {
public:
    using InputDtype = InputDtype_;
    using OutputDtype = int8_t;

    static constexpr int BlockSize = 128;
    static constexpr int NumThrPerCta = NumThrPerCta_;
    static constexpr int NumElementPerThread = BlockSize * BlockSize / NumThrPerCta;
    static constexpr int NumThrPerRow = BlockSize / NumElementPerThread;

    static_assert(BlockSize * BlockSize % NumThrPerCta == 0);
    static_assert(NumThrPerCta % BlockSize == 0);

    static constexpr size_t ShmSize = 32;
    static constexpr float int8_max = 128.f;

    struct Params {
        void const* Iptr;
        void* Optr;
        void* OSptr;
        int64_t const m;
        int64_t const n;
    };

    using Arguments = Params;

    static Params to_underlying_arguments(Arguments const& args) {
        return args;
    }

    static dim3 get_grid_size(int64_t m, int64_t n) {
        return dim3(
            cdiv(n, BlockSize),
            cdiv(m, BlockSize)
        );
    }

    static dim3 get_cta_size(int64_t m, int64_t n) {
        return dim3(NumThrPerCta, 1, 1);
    }

    TURBO_DEVICE
    void operator()(Params const& params, char* shared_data) {
        int blk_m = blockIdx.y;
        int blk_n = blockIdx.x;
        int tidx = threadIdx.x;

        float float_reg[NumElementPerThread];

        // Load input data
        load_input(params.Iptr, float_reg, params.m, params.n, blk_m, blk_n, tidx);
        
        // Quantize
        quantize(float_reg, params.Optr, params.OSptr, params.m, params.n, blk_m, blk_n, tidx, shared_data);
    }

private:
    TURBO_DEVICE
    void load_input(void const* input_ptr, float* thr_output_reg, 
                    int64_t m, int64_t n, int blk_m, int blk_n, int tid) {
        int thr_m_offset = tid / NumThrPerRow;
        int thr_n_offset = (tid % NumThrPerRow) * NumElementPerThread;
        
        int64_t global_m = blk_m * BlockSize + thr_m_offset;
        int64_t global_n = blk_n * BlockSize + thr_n_offset;
        
        InputDtype const* input = reinterpret_cast<InputDtype const*>(input_ptr);
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEvenM && IsEvenN) {
                thr_output_reg[i] = input_to_float<InputDtype>(input[global_m * n + global_n + i]);
            } else {
                if (global_m < m && (global_n + i) < n) {
                    thr_output_reg[i] = input_to_float<InputDtype>(input[global_m * n + global_n + i]);
                } else {
                    thr_output_reg[i] = 0.0f;
                }
            }
        }
    }

    TURBO_DEVICE
    void quantize(float* float_reg, void* Optr, void* OSptr,
                  int64_t m, int64_t n, int blk_m, int blk_n, int tidx, char* shared_data) {
        OutputDtype output_reg[NumElementPerThread];
        
        float amax = reduce_amax(float_reg, (float*)shared_data);
        
        float scale = int8_max / amax;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            float val = float_reg[i] * scale;
            val = fmaxf(-128.0f, fminf(127.0f, rintf(val)));
            output_reg[i] = static_cast<OutputDtype>(val);
        }
        
        float scale_inv = amax / int8_max;
        
        // Store output
        store_output(Optr, OSptr, output_reg, scale_inv, m, n, blk_m, blk_n, tidx);
        
        __syncthreads();
    }

    TURBO_DEVICE
    float reduce_amax(float* reg, float* smem_ptr) {
        float amax = 1e-8f;
        
        // Thread reduction
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            amax = fmaxf(amax, fabsf(reg[i]));
        }

        __syncwarp();

        // Warp reduction
        TURBO_PRAGMA_UNROLL
        for (int i = 16; i >= 1; i /= 2) {
            amax = fmaxf(__shfl_xor(amax, i, 32), amax);
        }

        // CTA reduction
        if (threadIdx.x == 0) {
            *smem_ptr = 0;
        }
        __syncthreads();

        atomicMax((unsigned int*)smem_ptr, __float_as_uint(amax));

        __syncthreads();

        amax = __uint_as_float(*(unsigned int*)smem_ptr);
        return amax;
    }

    TURBO_DEVICE
    void store_output(void* Optr, void* OSptr, OutputDtype* reg, float scale_inv,
                      int64_t m, int64_t n, int blk_m, int blk_n, int tid) {
        int thr_m_offset = tid / NumThrPerRow;
        int thr_n_offset = (tid % NumThrPerRow) * NumElementPerThread;
        
        int64_t global_m = blk_m * BlockSize + thr_m_offset;
        int64_t padded_n = cdiv(n, BlockSize) * BlockSize;
        int64_t global_n = blk_n * BlockSize + thr_n_offset;
        
        OutputDtype* output = reinterpret_cast<OutputDtype*>(Optr);
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEvenM && IsEvenN) {
                output[global_m * padded_n + global_n + i] = reg[i];
            } else {
                if (global_m < m && (global_n + i) < n) {
                    output[global_m * padded_n + global_n + i] = reg[i];
                }
            }
        }

        if (tid == 0) {
            float* scale_ptr = reinterpret_cast<float*>(OSptr);
            scale_ptr[blk_m * cdiv(n, BlockSize) + blk_n] = scale_inv;
        }
    }
};

template <class Kernel>
__global__ void quant_kernel_hip(typename Kernel::Params const params) {
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

template <
    class InputDtype,
    int BlockSize,
    int NumThrPerCta
>
bool quantization_hip(
    void const* Iptr, void* Optr, void* OSptr,
    int64_t m, int64_t n,
    hipStream_t stream = nullptr
) {
    BOOL_SWITCH(m % BlockSize == 0, IsEvenM, [&] {
        BOOL_SWITCH(n % BlockSize == 0, IsEvenN, [&] {
            using Kernel = QuantizationHIP<InputDtype, NumThrPerCta, IsEvenM, IsEvenN>;
            using Arguments = typename Kernel::Arguments;
            
            Arguments args = {Iptr, Optr, OSptr, m, n};
            auto params = Kernel::to_underlying_arguments(args);
            auto grid_shape = Kernel::get_grid_size(m, n);
            auto cta_shape = Kernel::get_cta_size(m, n);
            static constexpr size_t ShmSize = Kernel::ShmSize;
            
            hipLaunchKernelGGL(quant_kernel_hip<Kernel>, grid_shape, cta_shape, ShmSize, stream, params);
        });
    });

    return true;
}

