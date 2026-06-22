/*
 * Copyright (c) 2025 by TurboDiffusion team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * rocWMMA GEMM kernel launch wrapper for AMD GPUs.
 */

#pragma once

#include <hip/hip_runtime.h>
#include "common/platform.hpp"
#include "gemm/kernel_rocwmma.hpp"

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

template <class Kernel>
__global__ void rocwmma_gemm_kernel(
    typename Kernel::Params const params
) {
    extern __shared__ char smem[];
    Kernel op;
    op(params, smem);
}

template <class OutputDtype>
bool int8_gemm_rocwmma(
    int8_t const* Aptr, float const* ASptr,
    int8_t const* Bptr, float const* BSptr,
    OutputDtype* Dptr, int64_t m, int64_t n, int64_t k,
    int swizzle_dir = 1, int swizzle_size_log = 0,
    hipStream_t stream = nullptr
) {
    BOOL_SWITCH(m % 128 == 0, IsEvenM, [&] {
        BOOL_SWITCH(n % 128 == 0, IsEvenN, [&] {
            using Kernel = GemmKernelRocWMMA<OutputDtype, IsEvenM, IsEvenN>;
            
            if (!Kernel::can_implement(m, n, k)) {
                return false;
            }
            
            using Args = typename Kernel::Arguments;
            Args args{
                (void*)Aptr, (void*)ASptr,
                (void*)Bptr, (void*)BSptr, (void*)Dptr,
                m, n, k, swizzle_dir,
                swizzle_size_log
            };

            auto params = Kernel::to_underlying_arguments(args);

            static constexpr size_t ShmSize = Kernel::ShmSize;
            dim3 grid_shape = Kernel::get_grid_size(m, n);
            dim3 block_shape = dim3(Kernel::ThreadNum);
            
            auto func = rocwmma_gemm_kernel<Kernel>;
            if (ShmSize >= 48 * 1024) {
                hipFuncSetAttribute(
                    func,
                    hipFuncAttributeMaxDynamicSharedMemorySize,
                    ShmSize
                );
            }
            
            hipLaunchKernelGGL(func, grid_shape, block_shape, ShmSize, stream, params);
            return true;
        });
    });
    return true;
}

