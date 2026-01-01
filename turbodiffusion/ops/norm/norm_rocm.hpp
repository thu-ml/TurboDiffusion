/*
 * Copyright (c) 2025 by TurboDiffusion team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Normalization kernels for AMD GPUs using HIP.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "common/platform.hpp"

TURBO_HOST_DEVICE inline int64_t cdiv_norm(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

#define MIN_NORM(a, b) ((a) > (b) ? (b) : (a))
#define MAX_NORM(a, b) ((a) > (b) ? (a) : (b))

#define BOOL_SWITCH_NORM(COND, CONST_NAME, ...)    \
[&] {                                         \
  if (COND) {                                 \
    static constexpr bool CONST_NAME = true;  \
    return (__VA_ARGS__)();                   \
  } else {                                    \
    static constexpr bool CONST_NAME = false; \
    return (__VA_ARGS__)();                   \
  }                                           \
}()

// RMSNorm Kernel
template <
    class InputDtype_,
    class OutputDtype_,
    class WeightDtype_,
    int MaxHiddenSize_,
    int NumThrPerCta_,
    bool IsEven
>
class RMSNormHIP {
public:
    using InputDtype = InputDtype_;
    using OutputDtype = OutputDtype_;
    using WeightDtype = WeightDtype_;
    static constexpr int NumThrPerCta = NumThrPerCta_;
    static constexpr int MaxHiddenSize = MaxHiddenSize_;
    static constexpr size_t ShmSize = 32;
    static constexpr int NumElementPerThread = MaxHiddenSize / NumThrPerCta;

    static_assert(MaxHiddenSize % NumThrPerCta == 0);

    struct Params {
        void const* Iptr;
        void const* Wptr;
        void* Optr;
        float eps;
        int64_t m;
        int64_t n;
    };

    using Arguments = Params;

    static Params to_underlying_arguments(Arguments const& args) {
        return args;
    }

    static dim3 get_grid_size(int64_t m, int64_t n) {
        return dim3(m);
    }

    static dim3 get_cta_size(int64_t m, int64_t n) {
        return dim3(NumThrPerCta, 1, 1);
    }

    TURBO_DEVICE
    void operator()(Params const& params, char* shared_data) {
        int blk_m = blockIdx.x;
        int tidx = threadIdx.x;
        float x[NumElementPerThread];
        float w[NumElementPerThread];

        // Load input
        load_input(params.Iptr, x, params.m, params.n, blk_m, tidx);

        // RMS reduction
        float rms = sqrtf(reduce_square(x, shared_data) / params.n + params.eps);

        // Load weight
        load_weight(params.Wptr, w, params.n, tidx);

        // Normalize
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            x[i] = w[i] * x[i] / rms;
        }

        // Store output
        store_output(params.Optr, x, params.m, params.n, blk_m, tidx);
    }

private:
    TURBO_DEVICE
    void load_input(void const* input_ptr, float* reg, int64_t m, int64_t n, int blk_m, int tidx) {
        InputDtype const* input = reinterpret_cast<InputDtype const*>(input_ptr);
        int64_t offset = blk_m * n + tidx * NumElementPerThread;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEven || (tidx * NumElementPerThread + i) < n) {
                reg[i] = static_cast<float>(input[offset + i]);
            } else {
                reg[i] = 0.0f;
            }
        }
    }

    TURBO_DEVICE
    void load_weight(void const* weight_ptr, float* reg, int64_t n, int tidx) {
        if (weight_ptr == nullptr) {
            TURBO_PRAGMA_UNROLL
            for (int i = 0; i < NumElementPerThread; ++i) {
                reg[i] = 1.0f;
            }
            return;
        }
        
        WeightDtype const* weight = reinterpret_cast<WeightDtype const*>(weight_ptr);
        int64_t offset = tidx * NumElementPerThread;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEven || (tidx * NumElementPerThread + i) < n) {
                reg[i] = static_cast<float>(weight[offset + i]);
            } else {
                reg[i] = 1.0f;
            }
        }
    }

    TURBO_DEVICE
    void store_output(void* output_ptr, float* reg, int64_t m, int64_t n, int blk_m, int tidx) {
        OutputDtype* output = reinterpret_cast<OutputDtype*>(output_ptr);
        int64_t offset = blk_m * n + tidx * NumElementPerThread;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEven || (tidx * NumElementPerThread + i) < n) {
                output[offset + i] = static_cast<OutputDtype>(reg[i]);
            }
        }
    }

    TURBO_DEVICE
    float reduce_square(float* reg, char* shared_data) {
        float sum_square = 0;
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            sum_square += reg[i] * reg[i];
        }

        TURBO_PRAGMA_UNROLL
        for (int i = 16; i >= 1; i >>= 1) {
            sum_square += __shfl_down(sum_square, i, 32);
        }
        
        if (threadIdx.x == 0) {
            *(float*)shared_data = 0;
        }
        __syncthreads();

        if (threadIdx.x % 32 == 0) {
            atomicAdd((float*)shared_data, sum_square);
        }

        __syncthreads();
        sum_square = *(float*)shared_data;
        return sum_square;
    }
};

// LayerNorm Kernel
template <
    class InputDtype_,
    class OutputDtype_,
    class WeightDtype_,
    bool Affine,
    bool Bias,
    int MaxHiddenSize_,
    int NumThrPerCta_,
    bool IsEven
>
class LayerNormHIP {
public:
    using InputDtype = InputDtype_;
    using OutputDtype = OutputDtype_;
    using WeightDtype = WeightDtype_;
    static constexpr int NumThrPerCta = NumThrPerCta_;
    static constexpr int MaxHiddenSize = MaxHiddenSize_;
    static constexpr size_t ShmSize = 64;
    static constexpr int NumElementPerThread = MaxHiddenSize / NumThrPerCta;

    static_assert(MaxHiddenSize % NumThrPerCta == 0);

    struct Params {
        void const* Iptr;
        void const* Wptr;
        void const* Bptr;
        void* Optr;
        float eps;
        int64_t m;
        int64_t n;
    };

    using Arguments = Params;

    static Params to_underlying_arguments(Arguments const& args) {
        return args;
    }

    static dim3 get_grid_size(int64_t m, int64_t n) {
        return dim3(m);
    }

    static dim3 get_cta_size(int64_t m, int64_t n) {
        return dim3(NumThrPerCta, 1, 1);
    }

    TURBO_DEVICE
    void operator()(Params const& params, char* shared_data) {
        int blk_m = blockIdx.x;
        int tidx = threadIdx.x;
        float x[NumElementPerThread];
        float w[NumElementPerThread];
        float b[NumElementPerThread];

        // Load input
        load_input(params.Iptr, x, params.m, params.n, blk_m, tidx);

        // Compute mean and variance
        float mean, var;
        reduce_mean_var(x, shared_data, params.n, mean, var);
        float rstd = rsqrtf(var + params.eps);

        // Load weight and bias
        if constexpr (Affine) {
            load_weight(params.Wptr, w, params.n, tidx);
        }
        if constexpr (Bias) {
            load_weight(params.Bptr, b, params.n, tidx);
        }

        // Normalize
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            float val = (x[i] - mean) * rstd;
            if constexpr (Affine) {
                val *= w[i];
            }
            if constexpr (Bias) {
                val += b[i];
            }
            x[i] = val;
        }

        // Store output
        store_output(params.Optr, x, params.m, params.n, blk_m, tidx);
    }

private:
    TURBO_DEVICE
    void load_input(void const* input_ptr, float* reg, int64_t m, int64_t n, int blk_m, int tidx) {
        InputDtype const* input = reinterpret_cast<InputDtype const*>(input_ptr);
        int64_t offset = blk_m * n + tidx * NumElementPerThread;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEven || (tidx * NumElementPerThread + i) < n) {
                reg[i] = static_cast<float>(input[offset + i]);
            } else {
                reg[i] = 0.0f;
            }
        }
    }

    TURBO_DEVICE
    void load_weight(void const* weight_ptr, float* reg, int64_t n, int tidx) {
        if (weight_ptr == nullptr) {
            TURBO_PRAGMA_UNROLL
            for (int i = 0; i < NumElementPerThread; ++i) {
                reg[i] = 1.0f;
            }
            return;
        }
        
        WeightDtype const* weight = reinterpret_cast<WeightDtype const*>(weight_ptr);
        int64_t offset = tidx * NumElementPerThread;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEven || (tidx * NumElementPerThread + i) < n) {
                reg[i] = static_cast<float>(weight[offset + i]);
            } else {
                reg[i] = 0.0f;
            }
        }
    }

    TURBO_DEVICE
    void store_output(void* output_ptr, float* reg, int64_t m, int64_t n, int blk_m, int tidx) {
        OutputDtype* output = reinterpret_cast<OutputDtype*>(output_ptr);
        int64_t offset = blk_m * n + tidx * NumElementPerThread;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            if (IsEven || (tidx * NumElementPerThread + i) < n) {
                output[offset + i] = static_cast<OutputDtype>(reg[i]);
            }
        }
    }

    TURBO_DEVICE
    void reduce_mean_var(float* reg, char* shared_data, int64_t n, float& mean, float& var) {
        float sum = 0;
        float sum_sq = 0;
        
        TURBO_PRAGMA_UNROLL
        for (int i = 0; i < NumElementPerThread; ++i) {
            sum += reg[i];
            sum_sq += reg[i] * reg[i];
        }

        // Warp reduction
        TURBO_PRAGMA_UNROLL
        for (int i = 16; i >= 1; i >>= 1) {
            sum += __shfl_down(sum, i, 32);
            sum_sq += __shfl_down(sum_sq, i, 32);
        }
        
        float* smem = (float*)shared_data;
        if (threadIdx.x == 0) {
            smem[0] = 0;
            smem[1] = 0;
        }
        __syncthreads();

        if (threadIdx.x % 32 == 0) {
            atomicAdd(&smem[0], sum);
            atomicAdd(&smem[1], sum_sq);
        }

        __syncthreads();
        
        sum = smem[0];
        sum_sq = smem[1];
        
        mean = sum / n;
        var = sum_sq / n - mean * mean;
    }
};

// Kernel launchers
template <class Kernel>
__global__ void norm_kernel_hip(typename Kernel::Params const params) {
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

template <
    class InputDtype,
    class OutputDtype,
    class WeightDtype,
    int MaxHiddenSize,
    int NumThrPerCta
>
bool rmsnorm_hip(
    void const* Iptr, void const* Wptr,
    void* Optr, float eps,
    int64_t m, int64_t n,
    hipStream_t stream = nullptr
) {
    BOOL_SWITCH_NORM(n % MaxHiddenSize == 0, IsEven, [&] {
        using Kernel = RMSNormHIP<InputDtype, OutputDtype, WeightDtype, MaxHiddenSize, NumThrPerCta, IsEven>;
        using Arguments = typename Kernel::Arguments;
        
        Arguments args = {Iptr, Wptr, Optr, eps, m, n};
        auto params = Kernel::to_underlying_arguments(args);
        auto grid_shape = Kernel::get_grid_size(m, n);
        auto cta_shape = Kernel::get_cta_size(m, n);
        static constexpr size_t ShmSize = Kernel::ShmSize;
        
        hipLaunchKernelGGL(norm_kernel_hip<Kernel>, grid_shape, cta_shape, ShmSize, stream, params);
    });
    return true;
}

template <
    class InputDtype,
    class OutputDtype,
    class WeightDtype,
    bool Affine,
    bool Bias,
    int MaxHiddenSize,
    int NumThrPerCta
>
bool layernorm_hip(
    void const* Iptr, void const* Wptr, void const* Bptr,
    void* Optr, float eps,
    int64_t m, int64_t n,
    hipStream_t stream = nullptr
) {
    BOOL_SWITCH_NORM(n % MaxHiddenSize == 0, IsEven, [&] {
        using Kernel = LayerNormHIP<InputDtype, OutputDtype, WeightDtype, Affine, Bias, MaxHiddenSize, NumThrPerCta, IsEven>;
        using Arguments = typename Kernel::Arguments;
        
        Arguments args = {Iptr, Wptr, Bptr, Optr, eps, m, n};
        auto params = Kernel::to_underlying_arguments(args);
        auto grid_shape = Kernel::get_grid_size(m, n);
        auto cta_shape = Kernel::get_cta_size(m, n);
        static constexpr size_t ShmSize = Kernel::ShmSize;
        
        hipLaunchKernelGGL(norm_kernel_hip<Kernel>, grid_shape, cta_shape, ShmSize, stream, params);
    });
    return true;
}

